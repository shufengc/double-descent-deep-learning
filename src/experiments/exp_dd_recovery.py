"""
Double-descent recovery experiment for May 4 presentation.

Self-contained: defines its own fractional-k ResNet, its own data pipeline
(which returns corrupted indices for memorization-fraction diagnostic), and
its own training loop.

Modes:
  smoke  -- single run, k=1, n=1000, 100 epochs
  probe  -- k-sweep at n=4000, 400 epochs, 1 seed -- maps the transition
  main   -- headline DD curve at n=4000, 2000 epochs, 2 seeds
  nslice -- matched sweep at n=8000, 1500 epochs, 1 seed (peak shifts)

Writes per-run config.json and results.json; resumable.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T


# -------------------- Architecture --------------------
class _Block(nn.Module):
    def __init__(self, cin, cout, stride, norm):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = norm(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = norm(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), norm(cout))

    def forward(self, x):
        y = F.relu(self.n1(self.conv1(x)))
        y = self.n2(self.conv2(y))
        return F.relu(y + self.short(x))


class ResNetK(nn.Module):
    """3-stage ResNet with fractional width multiplier k.

    stage widths = max(1, round(16k, 32k, 64k)).
    Auto-switch BN -> GroupNorm if any width <= 2.
    """
    def __init__(self, k, num_classes=10):
        super().__init__()
        c1 = max(1, round(16 * k))
        c2 = max(1, round(32 * k))
        c3 = max(1, round(64 * k))
        self.widths = (c1, c2, c3)
        use_gn = min(self.widths) <= 2
        def norm(n):
            if use_gn:
                groups = 1 if n < 4 else min(8, n)
                while n % groups != 0:
                    groups -= 1
                return nn.GroupNorm(groups, n)
            return nn.BatchNorm2d(n)
        self.norm_kind = "gn" if use_gn else "bn"
        self.conv1 = nn.Conv2d(3, c1, 3, 1, 1, bias=False)
        self.n1 = norm(c1)
        self.layer1 = nn.Sequential(_Block(c1, c1, 1, norm), _Block(c1, c1, 1, norm))
        self.layer2 = nn.Sequential(_Block(c1, c2, 2, norm), _Block(c2, c2, 1, norm))
        self.layer3 = nn.Sequential(_Block(c2, c3, 2, norm), _Block(c3, c3, 1, norm))
        self.linear = nn.Linear(c3, num_classes)

    def forward(self, x, return_feats=False):
        y = F.relu(self.n1(self.conv1(x)))
        y = self.layer3(self.layer2(self.layer1(y)))
        y = F.adaptive_avg_pool2d(y, 1).flatten(1)
        out = self.linear(y)
        if return_feats:
            return out, y
        return out

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# -------------------- Data --------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def build_cifar_datasets(data_dir, augment=True):
    norm = T.Normalize(CIFAR_MEAN, CIFAR_STD)
    if augment:
        train_tf = T.Compose([T.RandomCrop(32, padding=4),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(), norm])
    else:
        train_tf = T.Compose([T.ToTensor(), norm])
    test_tf = T.Compose([T.ToTensor(), norm])
    train_full = torchvision.datasets.CIFAR10(data_dir, train=True,
                                              download=True, transform=train_tf)
    test_full = torchvision.datasets.CIFAR10(data_dir, train=False,
                                             download=True, transform=test_tf)
    return train_full, test_full


def corrupt_and_subsample(train_full, n_subset, noise_rate, seed, num_classes=10):
    """Return (Subset, corrupted_index_mask_over_subset).

    The noise mask is in subset-local coordinates so we can compute memorization
    fraction in O(batch) at train time.
    """
    rng = np.random.RandomState(seed)
    # Subsample from the full 50k training set first, then corrupt.
    all_idx = np.arange(len(train_full))
    rng.shuffle(all_idx)
    subset_idx = all_idx[:n_subset]

    targets = np.array(train_full.targets, dtype=np.int64)
    orig_labels = targets[subset_idx].copy()
    rng2 = np.random.RandomState(seed + 1)
    noise_mask = rng2.rand(n_subset) < noise_rate  # subset-local
    corrupted = np.zeros(n_subset, dtype=np.int64) - 1
    for i in range(n_subset):
        if noise_mask[i]:
            choices = [c for c in range(num_classes) if c != orig_labels[i]]
            corrupted[i] = rng2.choice(choices)
            targets[subset_idx[i]] = corrupted[i]
    train_full.targets = targets.tolist()
    subset = Subset(train_full, subset_idx.tolist())
    return subset, noise_mask, subset_idx, orig_labels


class IndexedSubset(torch.utils.data.Dataset):
    """Wraps a Subset to also return subset-local index, used for
    memorization-fraction tracking."""
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        x, y = self.subset[i]
        return x, y, i


# -------------------- Training --------------------
def train_one_run(k, n_train, noise_rate, epochs, seed, data_dir,
                  out_dir, augment=True, batch_size=256, lr=1e-4,
                  device="cuda", eval_every=50, log_every=10):
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(k=k, n_train=n_train, noise_rate=noise_rate, epochs=epochs,
               seed=seed, augment=augment, batch_size=batch_size, lr=lr)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_full, test_full = build_cifar_datasets(data_dir, augment=augment)
    subset, noise_mask, _, _ = corrupt_and_subsample(
        train_full, n_train, noise_rate, seed)
    indexed = IndexedSubset(subset)
    train_loader = DataLoader(indexed, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=False,
                              persistent_workers=True)
    test_loader = DataLoader(test_full, batch_size=512, shuffle=False,
                             num_workers=2, pin_memory=True,
                             persistent_workers=True)

    model = ResNetK(k).to(device)
    n_params = model.count_params()
    print(f"  widths={model.widths} norm={model.norm_kind} params={n_params:,}"
          f" p/n={n_params/n_train:.2f}")
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()
    noise_mask_t = torch.tensor(noise_mask, dtype=torch.bool, device=device)

    history = []
    best_test_acc = 0.0
    best_epoch = 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        corrupt_correct, corrupt_total = 0, 0
        clean_correct, clean_total = 0, 0
        for x, y, idx in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            idx = idx.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                pred = out.argmax(1)
                total += y.size(0)
                correct += (pred == y).sum().item()
                loss_sum += loss.item() * y.size(0)
                corr_mask = noise_mask_t[idx]
                cln = ~corr_mask
                if corr_mask.any():
                    corrupt_total += corr_mask.sum().item()
                    corrupt_correct += (pred[corr_mask] == y[corr_mask]).sum().item()
                if cln.any():
                    clean_total += cln.sum().item()
                    clean_correct += (pred[cln] == y[cln]).sum().item()

        train_loss = loss_sum / max(1, total)
        train_acc = 100.0 * correct / max(1, total)
        memo_frac = (corrupt_correct / corrupt_total) if corrupt_total > 0 else None
        clean_acc = (100.0 * clean_correct / clean_total) if clean_total > 0 else None

        record = dict(epoch=ep, train_loss=train_loss, train_acc=train_acc,
                      memorization_frac=memo_frac, train_acc_on_clean=clean_acc)

        if ep % eval_every == 0 or ep == epochs:
            model.eval()
            tloss, tcorr, ttot = 0.0, 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    out = model(x)
                    l = loss_fn(out, y)
                    tloss += l.item() * y.size(0)
                    tcorr += (out.argmax(1) == y).sum().item()
                    ttot += y.size(0)
            test_loss = tloss / ttot
            test_acc = 100.0 * tcorr / ttot
            with torch.no_grad():
                w_norm = math.sqrt(sum((p.float() ** 2).sum().item()
                                       for p in model.parameters()))
            record.update(test_loss=test_loss, test_acc=test_acc,
                          weight_l2=w_norm)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = ep

        if ep % log_every == 0 or ep == epochs or ep == 1:
            dt = time.time() - t0
            print(f"  ep {ep:5d}/{epochs} tr_acc={train_acc:5.2f}"
                  f" memo={memo_frac if memo_frac is None else f'{memo_frac:.3f}'}"
                  + (f" te_acc={record.get('test_acc', float('nan')):5.2f}"
                     if 'test_acc' in record else "")
                  + f" [{dt:.0f}s]")
        history.append(record)

    # end-of-training diagnostics
    model.eval()
    eff_rank = compute_effective_rank(model, test_loader, device)
    results = dict(config=cfg, params=n_params, widths=list(model.widths),
                   norm_kind=model.norm_kind,
                   best_test_acc=best_test_acc, best_epoch=best_epoch,
                   final_test_acc=history[-1].get("test_acc"),
                   final_train_acc=history[-1]["train_acc"],
                   effective_rank=eff_rank,
                   history=history,
                   wallclock_sec=time.time() - t0)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f)
    return results


@torch.no_grad()
def compute_effective_rank(model, test_loader, device, n_samples=1024):
    feats = []
    seen = 0
    for x, _ in test_loader:
        x = x.to(device)
        _, z = model(x, return_feats=True)
        feats.append(z.float().cpu())
        seen += x.size(0)
        if seen >= n_samples:
            break
    Z = torch.cat(feats, dim=0)[:n_samples]
    # stable rank = ||Z||_F^2 / ||Z||_op^2
    fro2 = (Z * Z).sum().item()
    s = torch.linalg.svdvals(Z - Z.mean(0, keepdim=True))
    op2 = (s[0].item()) ** 2 if len(s) else 1.0
    return fro2 / max(op2, 1e-12)



# -------------------- Mode dispatchers --------------------
SMOKE_CFG = dict(n_train=1000, epochs=100, noise_rate=0.15, seed=42,
                 ks=[1.0], eval_every=25, log_every=20)

PROBE_CFG = dict(n_train=4000, epochs=400, noise_rate=0.15, seed=42,
                 ks=[0.0625, 0.125, 0.25, 0.5, 1.0, 2.0],
                 eval_every=50, log_every=50)

MAIN_CFG = dict(n_train=4000, epochs=2000, noise_rate=0.15,
                ks=[0.0625, 0.125, 0.1875, 0.25, 0.375,
                    0.5, 0.75, 1.0, 2.0],
                seeds=[42, 7],
                eval_every=100, log_every=100)

NSLICE_CFG = dict(n_train=8000, epochs=1500, noise_rate=0.15, seed=42,
                  ks=[0.125, 0.25, 0.5, 1.0],
                  eval_every=100, log_every=100)


def run_sweep(mode, out_root, data_dir, device, seed_override=None, skip_done=True):
    if mode == "smoke":
        cfg = SMOKE_CFG
        seeds = [cfg["seed"]]
    elif mode == "probe":
        cfg = PROBE_CFG
        seeds = [cfg["seed"]]
    elif mode == "main":
        cfg = MAIN_CFG
        seeds = [seed_override] if seed_override is not None else cfg["seeds"]
    elif mode == "nslice":
        cfg = NSLICE_CFG
        seeds = [cfg["seed"]]
    else:
        raise ValueError(mode)

    mode_dir = Path(out_root) / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for seed in seeds:
        for k in cfg["ks"]:
            run_name = f"k{k:g}_n{cfg['n_train']}_ep{cfg['epochs']}_s{seed}"
            run_dir = mode_dir / run_name
            results_path = run_dir / "results.json"
            if skip_done and results_path.exists():
                print(f"[{mode}] SKIP (done): {run_name}")
                with open(results_path) as f:
                    summary.append(json.load(f))
                continue
            print(f"[{mode}] RUN: {run_name}")
            r = train_one_run(k=k, n_train=cfg["n_train"],
                              noise_rate=cfg["noise_rate"],
                              epochs=cfg["epochs"], seed=seed,
                              data_dir=data_dir, out_dir=str(run_dir),
                              device=device,
                              eval_every=cfg.get("eval_every", 50),
                              log_every=cfg.get("log_every", 50))
            summary.append(r)

    # Aggregate
    summary_path = mode_dir / "summary.json"
    slim = [dict(k=r["config"]["k"], n=r["config"]["n_train"],
                 seed=r["config"]["seed"], params=r["params"],
                 best_test_acc=r["best_test_acc"],
                 final_test_acc=r["final_test_acc"],
                 final_train_acc=r["final_train_acc"],
                 effective_rank=r["effective_rank"]) for r in summary]
    with open(summary_path, "w") as f:
        json.dump(slim, f, indent=2)
    print(f"[{mode}] wrote summary to {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "probe", "main", "nslice"],
                   required=True)
    p.add_argument("--out_root",
                   default="results/dd_recovery_5090_focused")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None,
                   help="Override seed (used for main mode to split 2 seeds)")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if results.json exists")
    args = p.parse_args()
    run_sweep(args.mode, args.out_root, args.data_dir, args.device,
              seed_override=args.seed, skip_done=not args.force)


if __name__ == "__main__":
    main()

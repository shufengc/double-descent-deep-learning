"""
Depth-axis ablation on the fractional-k ResNet (W5 / report §6.12).

Closes Lecture 12 Theme 1 ("approximation theory and the impact of depth")
within the fractional-k family. Holds k=0.5 fixed (over-parameterised, where
DD-Recovery is most stable), n=4000, 15% noise, 1500 epochs, varies the number
of ResNet stages ∈ {2, 4} (depth=3 already covered by §5.3 main sweep).

Usage on dd-5090:
  python -m src.experiments.exp_depth_ablation --device cuda

Output:
  results/depth_ablation/d{D}_k{K}_n{N}_ep{EP}_s{S}/results.json
  results/depth_ablation/summary.json
"""
import argparse
import json
import math
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.experiments.exp_dd_recovery import (
    _Block, build_cifar_datasets, corrupt_and_subsample, IndexedSubset,
    compute_effective_rank,
)


class ResNetKDepth(nn.Module):
    """Fractional-k ResNet with configurable stage count.

    n_stages=3 is identical to the original ResNetK (widths 16k, 32k, 64k).
    n_stages=2 drops the deepest stage (widths 16k, 32k).
    n_stages=4 adds an extra deeper stage (widths 16k, 32k, 64k, 128k).
    """
    def __init__(self, k, n_stages=3, num_classes=10):
        super().__init__()
        widths = [max(1, round(16 * k * (2 ** s))) for s in range(n_stages)]
        self.widths = tuple(widths)
        self.n_stages = n_stages
        use_gn = min(self.widths) <= 2

        def norm(n):
            if use_gn:
                groups = 1 if n < 4 else min(8, n)
                while n % groups != 0:
                    groups -= 1
                return nn.GroupNorm(groups, n)
            return nn.BatchNorm2d(n)

        self.norm_kind = "gn" if use_gn else "bn"
        c0 = widths[0]
        self.conv1 = nn.Conv2d(3, c0, 3, 1, 1, bias=False)
        self.n1 = norm(c0)

        layers = []
        prev_c = c0
        for s in range(n_stages):
            c = widths[s]
            stride = 1 if s == 0 else 2
            layers.append(nn.Sequential(_Block(prev_c, c, stride, norm),
                                        _Block(c, c, 1, norm)))
            prev_c = c
        self.stages = nn.ModuleList(layers)
        self.linear = nn.Linear(prev_c, num_classes)

    def forward(self, x, return_feats=False):
        y = F.relu(self.n1(self.conv1(x)))
        for stage in self.stages:
            y = stage(y)
        y = F.adaptive_avg_pool2d(y, 1).flatten(1)
        out = self.linear(y)
        if return_feats:
            return out, y
        return out

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_one(k, n_stages, n_train, noise_rate, epochs, seed, data_dir,
              out_dir, batch_size=256, lr=1e-4, device="cuda",
              eval_every=100, log_every=100):
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(k=k, n_stages=n_stages, n_train=n_train, noise_rate=noise_rate,
               epochs=epochs, seed=seed)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_full, test_full = build_cifar_datasets(data_dir, augment=True)
    subset, _, _, _ = corrupt_and_subsample(train_full, n_train, noise_rate, seed)
    train_loader = DataLoader(IndexedSubset(subset), batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True,
                              persistent_workers=True)
    test_loader = DataLoader(test_full, batch_size=512, shuffle=False,
                             num_workers=2, pin_memory=True,
                             persistent_workers=True)

    model = ResNetKDepth(k, n_stages=n_stages).to(device)
    n_params = model.count_params()
    print(f"  d{n_stages} k{k} widths={model.widths} norm={model.norm_kind}"
          f" params={n_params:,} p/n={n_params/n_train:.2f}", flush=True)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_test_acc = 0.0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total, correct = 0, 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                total += y.size(0)
                correct += (out.argmax(1) == y).sum().item()
        train_acc = 100.0 * correct / max(1, total)
        rec = dict(epoch=ep, train_acc=train_acc)

        if ep % eval_every == 0 or ep == epochs:
            model.eval()
            tcorr, ttot = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    pred = model(x).argmax(1)
                    tcorr += (pred == y).sum().item()
                    ttot += y.size(0)
            test_acc = 100.0 * tcorr / ttot
            rec["test_acc"] = test_acc
            if test_acc > best_test_acc:
                best_test_acc = test_acc

        if ep % log_every == 0 or ep == epochs or ep == 1:
            dt = time.time() - t0
            print(f"  d{n_stages} ep{ep:5d}/{epochs} train_acc={train_acc:5.2f}"
                  + (f" test_acc={rec.get('test_acc', float('nan')):5.2f}"
                     if 'test_acc' in rec else "")
                  + f" [{dt:.0f}s]", flush=True)
        history.append(rec)

    model.eval()
    eff_rank = compute_effective_rank(model, test_loader, device)
    final_test = history[-1].get("test_acc")
    final_train = history[-1]["train_acc"]
    results = dict(config=cfg, params=n_params, widths=list(model.widths),
                   norm_kind=model.norm_kind,
                   best_test_acc=best_test_acc,
                   final_test_acc=final_test,
                   final_train_acc=final_train,
                   effective_rank=eff_rank,
                   wallclock_sec=time.time() - t0,
                   history_last=history[-3:])  # save tail only, not all eps
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f)
    return results


def run(args):
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        depths = [2]
        seeds = [42]
        epochs = 50
    else:
        depths = [int(d) for d in args.depths.split(",")]
        seeds = [int(s) for s in args.seeds.split(",")]
        epochs = args.epochs

    grid = list(product(depths, seeds))
    print(f"[depth] plan: {len(grid)} runs at k={args.k}, n={args.n_train},"
          f" ep={epochs}, depths={depths}, seeds={seeds}", flush=True)

    summary = []
    t_total = time.time()
    for i, (d, s) in enumerate(grid, 1):
        run_name = f"d{d}_k{args.k:g}_n{args.n_train}_ep{epochs}_s{s}"
        run_dir = out_root / run_name
        results_path = run_dir / "results.json"
        if results_path.exists() and not args.force:
            print(f"\n[{i}/{len(grid)}] {run_name}  SKIP cached", flush=True)
            with open(results_path) as f:
                summary.append(json.load(f))
            continue
        print(f"\n[{i}/{len(grid)}] {run_name}", flush=True)
        t0 = time.time()
        r = train_one(k=args.k, n_stages=d, n_train=args.n_train,
                      noise_rate=args.noise_rate, epochs=epochs, seed=s,
                      data_dir=args.data_dir, out_dir=str(run_dir),
                      batch_size=args.batch_size, lr=args.lr,
                      device=args.device, eval_every=args.eval_every,
                      log_every=args.log_every)
        summary.append(r)
        slim = [dict(k=r["config"]["k"], n_stages=r["config"]["n_stages"],
                     n=r["config"]["n_train"], seed=r["config"]["seed"],
                     params=r["params"], widths=r["widths"],
                     best_test_acc=r["best_test_acc"],
                     final_test_acc=r["final_test_acc"],
                     final_train_acc=r["final_train_acc"],
                     effective_rank=r["effective_rank"]) for r in summary]
        with open(out_root / "summary.json", "w") as f:
            json.dump(slim, f, indent=2)
        print(f"  done in {(time.time()-t0)/60:.1f} min  best_test={r['best_test_acc']:.2f}",
              flush=True)

    print(f"\n[depth] complete. wall {(time.time()-t_total)/60:.1f} min", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depths", default="2,4")
    p.add_argument("--seeds", default="42,7")
    p.add_argument("--k", type=float, default=0.5)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--noise-rate", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-root", default="./results/depth_ablation")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

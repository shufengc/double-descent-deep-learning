"""
Literal ResNet18 controlled comparison (W2 / report §5.3.1).

Runs torchvision-style ResNet18 (4-stage BasicBlock-based) at three width
multipliers ∈ {0.5, 1.0, 2.0}, on the SAME hyperparameters as the fractional-k
DD-Recovery sweep: n=4000, 15% label noise, Adam lr=1e-4, 2000 epochs, 1 seed.

The point: confirm head-to-head that literal ResNet18 does not exhibit the
same DD trajectory as the fractional-k family. Closes a credibility gap in
§5.3 where the "ResNet18 fails" claim was based on an earlier flat-curve
diagnostic, not a controlled comparison.

Usage:
  python -m src.experiments.exp_resnet18_compare --device cuda
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.experiments.exp_dd_recovery import (
    build_cifar_datasets, corrupt_and_subsample, IndexedSubset,
    compute_effective_rank,
)


# Standard BasicBlock (torchvision-style)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * BasicBlock.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """Literal ResNet18 architecture for CIFAR-10 (4 stages, 2 blocks each).

    width_mult scales all 4 base widths (default 64, 128, 256, 512).
    width_mult=1.0 → 11.2M params (canonical ResNet18).
    width_mult=0.5 → ~2.8M params.
    width_mult=2.0 → ~44.8M params.
    """
    def __init__(self, width_mult=1.0, num_classes=10):
        super().__init__()
        c1, c2, c3, c4 = (max(1, round(64*width_mult)),
                          max(1, round(128*width_mult)),
                          max(1, round(256*width_mult)),
                          max(1, round(512*width_mult)))
        self.in_planes = c1
        self.widths = (c1, c2, c3, c4)
        self.conv1 = nn.Conv2d(3, c1, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.layer1 = self._make_layer(c1, 2, 1)
        self.layer2 = self._make_layer(c2, 2, 2)
        self.layer3 = self._make_layer(c3, 2, 2)
        self.layer4 = self._make_layer(c4, 2, 2)
        self.linear = nn.Linear(c4, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feats=False):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.layer4(self.layer3(self.layer2(self.layer1(y))))
        y = F.adaptive_avg_pool2d(y, 1).flatten(1)
        out = self.linear(y)
        if return_feats:
            return out, y
        return out

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_one(width_mult, n_train, noise_rate, epochs, seed, data_dir,
              out_dir, batch_size=256, lr=1e-4, device="cuda",
              eval_every=100, log_every=200):
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(width_mult=width_mult, n_train=n_train, noise_rate=noise_rate,
               epochs=epochs, seed=seed, arch="resnet18")
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

    model = ResNet18(width_mult=width_mult).to(device)
    n_params = model.count_params()
    print(f"  ResNet18×{width_mult} widths={model.widths} params={n_params:,}"
          f" p/n={n_params/n_train:.2f}", flush=True)
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
            print(f"  rn18×{width_mult} ep{ep:5d}/{epochs} tr={train_acc:5.2f}"
                  + (f" te={rec.get('test_acc', float('nan')):5.2f}"
                     if 'test_acc' in rec else "")
                  + f" [{dt:.0f}s]", flush=True)
        history.append(rec)

    model.eval()
    eff_rank = compute_effective_rank(model, test_loader, device)
    results = dict(config=cfg, params=n_params, widths=list(model.widths),
                   best_test_acc=best_test_acc,
                   final_test_acc=history[-1].get("test_acc"),
                   final_train_acc=history[-1]["train_acc"],
                   effective_rank=eff_rank,
                   wallclock_sec=time.time() - t0)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f)
    return results


def run(args):
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        mults = [1.0]
        epochs = 30
    else:
        mults = [float(x) for x in args.mults.split(",")]
        epochs = args.epochs

    print(f"[resnet18] plan: {len(mults)} runs, mults={mults}, n={args.n_train},"
          f" ep={epochs}, seed={args.seed}", flush=True)

    summary = []
    t_total = time.time()
    for i, m in enumerate(mults, 1):
        run_name = f"rn18m{m:g}_n{args.n_train}_ep{epochs}_s{args.seed}"
        run_dir = out_root / run_name
        rp = run_dir / "results.json"
        if rp.exists() and not args.force:
            print(f"\n[{i}/{len(mults)}] {run_name}  SKIP cached", flush=True)
            with open(rp) as f:
                summary.append(json.load(f))
            continue
        print(f"\n[{i}/{len(mults)}] {run_name}", flush=True)
        t0 = time.time()
        r = train_one(width_mult=m, n_train=args.n_train,
                      noise_rate=args.noise_rate, epochs=epochs, seed=args.seed,
                      data_dir=args.data_dir, out_dir=str(run_dir),
                      batch_size=args.batch_size, lr=args.lr,
                      device=args.device, eval_every=args.eval_every,
                      log_every=args.log_every)
        summary.append(r)
        slim = [dict(width_mult=r["config"]["width_mult"],
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

    print(f"\n[resnet18] complete. wall {(time.time()-t_total)/60:.1f} min", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mults", default="0.5,1.0,2.0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--noise-rate", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-root", default="./results/resnet18_compare")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

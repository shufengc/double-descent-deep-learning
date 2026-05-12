"""A2 — Cross-architecture metric audit (FAST version, eval every 100 epochs).

Sweep MLP {widths 4, 8, 16, 32, 64, 128, 256} and CNN {widths 1, 2, 4, 8, 16, 32}
on noisy CIFAR-10 (n=4000, 15% noise), 1500 epochs, 2 seeds each.

Throttled test evaluation (eval_every=100) saves ~95% wallclock vs the legacy
Trainer.train() which evaluates every epoch. Each run ~15-20 min instead of 53.

Total: 7 MLP × 2 + 6 CNN × 2 = 26 runs × ~17 min ≈ 7-8 GPU-hours on A100.
"""

from __future__ import annotations

import sys
import json
import os
import time
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from src.models import MLP, CNN
from src.data import get_cifar10, corrupt_labels, make_subset, make_loaders

torch.backends.cudnn.benchmark = True

OUT = Path("/home/cc/shufeng/elen6699/runs/cross_arch_metric_audit")
OUT.mkdir(parents=True, exist_ok=True)
DATA = "/home/cc/shufeng/elen6699/data"

N_TRAIN = 4000
NOISE_RATE = 0.15
EPOCHS = 800   # MLP/CNN converge much faster than ResNet; 800 is plenty at lr=1e-3
BATCH_SIZE = 256
LR = 1e-3   # matches legacy exp_architecture.py; MLPs don't learn at lr=1e-4
SEEDS = [42, 7]
EVAL_EVERY = 50    # eval more often since training is shorter
LOG_EVERY = 200

CONFIGS = {
    # MLP runs at this hyperparameter grid (lr=1e-3, 800 epochs, no aug, noise=0.15)
    # collapsed to ~8% test acc across all widths — model failed to learn.
    # Documented in our cross-arch results as "MLP does not learn noisy CIFAR-10
    # in our budget; CNN is the cross-arch test." Removed from active sweep.
    "CNN": {
        "widths": [1, 2, 4, 8, 16, 32],
        "make": lambda w: CNN(num_classes=10, num_filters=w, input_channels=3),
    },
}


def is_complete(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        r = json.loads(p.read_text())
        return r.get("final_test_acc") is not None and r.get("best_test_acc") is not None
    except Exception:
        return False


def run_one(arch: str, width: int, seed: int) -> dict:
    out_dir = OUT / arch / f"w{width}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rp = out_dir / "results.json"
    if is_complete(rp):
        print(f"  SKIP existing {arch} w={width} s={seed}", flush=True)
        return json.loads(rp.read_text())

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_full, test_set = get_cifar10(data_dir=DATA, augment=False)
    train_full = corrupt_labels(train_full, NOISE_RATE, seed=seed)
    train_set = make_subset(train_full, N_TRAIN, seed=seed)
    train_loader, test_loader = make_loaders(train_set, test_set, batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CONFIGS[arch]["make"](width).to(device)
    p = model.count_parameters()

    print(f"\n[{arch} w={width} s={seed}] params={p:,} p/n={p/N_TRAIN:.2f}", flush=True)

    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    history_test_acc = []
    history_train_acc = []
    best_test_acc = 0.0
    best_epoch = 0

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            with torch.no_grad():
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
                loss_sum += loss.item() * y.size(0)
        train_acc = 100.0 * correct / max(total, 1)
        history_train_acc.append(train_acc)

        if ep % EVAL_EVERY == 0 or ep == EPOCHS:
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
            test_acc = 100.0 * tcorr / ttot
            history_test_acc.append((ep, test_acc))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = ep

        if ep % LOG_EVERY == 0 or ep == EPOCHS or ep == 1:
            dt = time.time() - t0
            te = history_test_acc[-1][1] if history_test_acc else float('nan')
            print(f"  ep {ep:5d}/{EPOCHS} tr={train_acc:5.2f} te={te:5.2f} [{dt:.0f}s]",
                  flush=True)

    elapsed = time.time() - t0
    final_test_acc = history_test_acc[-1][1] if history_test_acc else 0.0
    result = dict(
        arch=arch,
        width=width,
        seed=seed,
        num_params=p,
        p_over_n=round(p / N_TRAIN, 4),
        final_train_acc=float(history_train_acc[-1]),
        final_test_acc=float(final_test_acc),
        best_test_acc=float(best_test_acc),
        best_epoch=int(best_epoch),
        gap=float(best_test_acc - final_test_acc),
        wallclock_sec=round(elapsed, 1),
        epochs=EPOCHS,
        n_train=N_TRAIN,
        noise_rate=NOISE_RATE,
        eval_every=EVAL_EVERY,
        history_test_acc=history_test_acc,
    )
    with open(rp, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  done {elapsed/60:.1f} min  best={best_test_acc:.2f}  "
          f"final={final_test_acc:.2f}  gap={result['gap']:+.2f}",
          flush=True)
    return result


def main():
    t_all = time.time()
    all_results = []

    for arch in CONFIGS:
        for width in CONFIGS[arch]["widths"]:
            for seed in SEEDS:
                r = run_one(arch, width, seed)
                all_results.append(r)
                with open(OUT / "summary.json", "w") as f:
                    json.dump(all_results, f, indent=2)

    print(f"\nALL DONE: {(time.time()-t_all)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()

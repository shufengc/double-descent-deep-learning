"""A3 — Weight-decay sweep at the DD valley (RFF↔NN bridge).

Hold k=0.5 (valley floor on n=4000), sweep weight_decay across 5 values × 2 seeds.
Hypothesis: weight decay smooths the NN valley analogously to how ridge
smooths the RFF peak (paper §3.2, ridge sweep). Direct theoretical bridge
between the paper's RFF and NN halves.

Citable prior art: Yilmaz & Heckel (2022), "Regularization-wise double descent"
(arXiv:2206.01378) — analytic linear models. We extend to trained NN.

Hyperparameters: k=0.5, n=4000, 15% noise, 2000 epochs, Adam lr=1e-4.
Five λ_wd values: {0, 1e-5, 1e-4, 1e-3, 1e-2}, 2 seeds each = 10 runs.
Estimated wallclock: ~10 GPU-hours on A100 (k=0.5 is medium-size; ~17 min/run
based on Yusheng-replicate benchmarks).
"""
from __future__ import annotations

import sys
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
from src.experiments.exp_dd_recovery import train_one_run

torch.backends.cudnn.benchmark = True

OUT = Path("/home/cc/shufeng/elen6699/runs/wd_sweep_valley")
OUT.mkdir(parents=True, exist_ok=True)
DATA = "/home/cc/shufeng/elen6699/data"

# Pin to the valley floor we identified at k=0.5 (Yusheng-replicate, n=4000)
K_VALLEY = 0.5
N_TRAIN = 4000
NOISE_RATE = 0.15
EPOCHS = 2000
LR = 1e-4

# Sweep grid: 5 weight-decay values × 2 seeds = 10 runs
WD_VALUES = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
SEEDS = [42, 7]


def is_complete(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        r = json.loads(p.read_text())
        return r.get("final_test_acc") is not None and bool(r.get("history"))
    except Exception:
        return False


t_all = time.time()
JOBS = [(wd, s) for wd in WD_VALUES for s in SEEDS]

for i, (wd, seed) in enumerate(JOBS, 1):
    name = "k{:g}_n{}_ep{}_wd{:g}_s{}".format(K_VALLEY, N_TRAIN, EPOCHS, wd, seed)
    d = OUT / name
    rp = d / "results.json"
    if is_complete(rp):
        print(f"[{i}/{len(JOBS)}] SKIP complete {name}", flush=True)
        continue

    print(f"\n[{i}/{len(JOBS)}] RUN {name}  (wd={wd})", flush=True)
    t0 = time.time()
    # exp_dd_recovery.train_one_run does NOT take a weight_decay kwarg by default.
    # We need to either monkey-patch or use a different call path.
    # Path: extract just the trainer setup since exp_dd_recovery uses Adam(weight_decay=0).
    # We will fall back to calling Adam directly in the same shape, copying the loop.
    # For simplicity, use the higher-level Trainer class with the WD setting.

    # Replicate the exp_dd_recovery training pipeline with weight_decay
    import numpy as np
    from src.experiments.exp_dd_recovery import (
        ResNetK, build_cifar_datasets, corrupt_and_subsample, IndexedSubset,
        compute_effective_rank,
    )
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.optim as optim
    import math

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_full, test_full = build_cifar_datasets(DATA, augment=True)
    subset, _, _, _ = corrupt_and_subsample(train_full, N_TRAIN, NOISE_RATE, seed)
    train_loader = DataLoader(IndexedSubset(subset), batch_size=256, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=False,
                              persistent_workers=True)
    test_loader = DataLoader(test_full, batch_size=512, shuffle=False,
                             num_workers=2, pin_memory=True, persistent_workers=True)

    model = ResNetK(K_VALLEY).to("cuda")
    n_params = model.count_params()
    cfg = dict(k=K_VALLEY, n_train=N_TRAIN, noise_rate=NOISE_RATE, epochs=EPOCHS,
               seed=seed, augment=True, batch_size=256, lr=LR, weight_decay=wd)

    d.mkdir(parents=True, exist_ok=True)
    with open(d / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best_test_acc = 0.0
    best_epoch = 0
    eval_every = 100
    log_every = 400

    print(f"  widths={model.widths} norm={model.norm_kind} params={n_params:,} wd={wd}",
          flush=True)
    t_start = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        total, correct = 0, 0
        loss_sum = 0.0
        memo_total, memo_correct = 0, 0
        for x, y, idx in train_loader:
            x = x.to("cuda", non_blocking=True)
            y = y.to("cuda", non_blocking=True)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            with torch.no_grad():
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                loss_sum += loss.item() * y.size(0)
                # Memorization on label-noise indices not tracked here for simplicity
        train_acc = 100.0 * correct / max(total, 1)
        train_loss = loss_sum / max(total, 1)
        memo_frac = None
        record = dict(epoch=ep, train_loss=train_loss, train_acc=train_acc,
                      memorization_frac=memo_frac)

        if ep % eval_every == 0 or ep == EPOCHS:
            model.eval()
            tloss, tcorr, ttot = 0.0, 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to("cuda", non_blocking=True)
                    y = y.to("cuda", non_blocking=True)
                    out = model(x)
                    l = loss_fn(out, y)
                    tloss += l.item() * y.size(0)
                    tcorr += (out.argmax(1) == y).sum().item()
                    ttot += y.size(0)
            test_acc = 100.0 * tcorr / ttot
            test_loss = tloss / ttot
            record["test_loss"] = test_loss
            record["test_acc"] = test_acc
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = ep
        history.append(record)

        if ep % log_every == 0 or ep == EPOCHS or ep == 1:
            dt = time.time() - t_start
            print(f"  ep {ep:5d}/{EPOCHS} tr_acc={train_acc:5.2f}"
                  + (f" te_acc={record.get('test_acc', float('nan')):5.2f}"
                     if 'test_acc' in record else "")
                  + f" [{dt:.0f}s]", flush=True)

    # End-of-training diagnostics
    model.eval()
    eff_rank = compute_effective_rank(model, test_loader, "cuda")
    results = dict(
        config=cfg, params=n_params, widths=list(model.widths),
        norm_kind=model.norm_kind,
        best_test_acc=best_test_acc, best_epoch=best_epoch,
        final_test_acc=history[-1].get("test_acc"),
        final_train_acc=history[-1]["train_acc"],
        effective_rank=eff_rank, history=history,
        wallclock_sec=time.time() - t_start,
    )
    with open(rp, "w") as f:
        json.dump(results, f)
    print(f"[done] {name}: {(time.time()-t0)/60:.1f} min  "
          f"best={best_test_acc:.2f}  final={history[-1].get('test_acc', float('nan')):.2f}",
          flush=True)

print(f"\nALL DONE: {(time.time()-t_all)/3600:.2f}h", flush=True)

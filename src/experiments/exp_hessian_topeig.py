"""
Hessian top-eigenvalue (sharpness) at the DD-recovery onset (N2 / report §6.13).

Usage:
  python -m src.experiments.exp_hessian_topeig --device cuda
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
    ResNetK, build_cifar_datasets, corrupt_and_subsample, IndexedSubset,
)


def power_iter_top_eig(model, x, y, params, n_iter=30, tol=1e-4):
    """Power iteration for top Hessian eigenvalue.

    Hessian is ∇²_θ L(θ) where L is mean cross-entropy on (x, y).
    """
    loss_fn = nn.CrossEntropyLoss()
    out = model(x)
    loss = loss_fn(out, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_flat = torch.cat([g.reshape(-1) for g in grads])

    # Random unit vector
    v = torch.randn_like(grad_flat)
    v = v / v.norm()

    eig_prev = 0.0
    for it in range(n_iter):
        # Hv = grad of (grad . v)
        gv = (grad_flat * v).sum()
        Hv = torch.autograd.grad(gv, params, retain_graph=True)
        Hv_flat = torch.cat([h.reshape(-1) for h in Hv])
        eig = (Hv_flat * v).sum().item()
        v = Hv_flat / Hv_flat.norm()
        if abs(eig - eig_prev) / max(abs(eig), 1e-8) < tol:
            return eig, it + 1
        eig_prev = eig
    return eig, n_iter


def train_then_hessian(k, n_train, noise_rate, epochs, seed,
                      data_dir, out_dir, n_samples_hessian=256,
                      power_iters=30, batch_size=256, lr=1e-4,
                      device="cuda", log_every=200):
    os.makedirs(out_dir, exist_ok=True)
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

    model = ResNetK(k).to(device)
    n_params = model.count_params()
    print(f"  k={k:g} widths={model.widths} params={n_params:,}", flush=True)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.time()
    last_train_acc = 0.0
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
        last_train_acc = 100.0 * correct / max(1, total)
        if ep % log_every == 0 or ep == epochs:
            print(f"    ep {ep}/{epochs} train_acc={last_train_acc:.2f}"
                  f" [{time.time()-t0:.0f}s]", flush=True)

    # Compute Hessian top eigenvalue
    model.eval()
    # Build a small batch for Hessian (random sample of training subset)
    x_h, y_h = [], []
    for batch_x, batch_y, _ in train_loader:
        x_h.append(batch_x)
        y_h.append(batch_y)
        if sum(b.size(0) for b in x_h) >= n_samples_hessian:
            break
    x_h = torch.cat(x_h, 0)[:n_samples_hessian].to(device)
    y_h = torch.cat(y_h, 0)[:n_samples_hessian].to(device)

    params = list(model.parameters())
    t_pi = time.time()
    top_eig, iters_used = power_iter_top_eig(
        model, x_h, y_h, params, n_iter=power_iters, tol=1e-4)
    pi_time = time.time() - t_pi
    print(f"    top Hessian eig = {top_eig:.4e}  (power iter: {iters_used} steps,"
          f" {pi_time:.1f}s)", flush=True)

    # Test acc for sanity
    model.eval()
    tcorr, ttot = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            tcorr += (model(x).argmax(1) == y).sum().item()
            ttot += y.size(0)
    test_acc = 100.0 * tcorr / ttot

    record = dict(
        k=k, n_train=n_train, seed=seed, epochs=epochs,
        params=n_params, widths=list(model.widths),
        train_acc=float(last_train_acc),
        test_acc=float(test_acc),
        top_hessian_eig=float(top_eig),
        power_iters_used=int(iters_used),
        n_samples_hessian=int(n_samples_hessian),
        pi_wallclock_sec=float(pi_time),
        train_wallclock_sec=float(time.time() - t0 - pi_time),
    )
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(record, f, indent=2)
    return record


def run(args):
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        ks = [0.5]
        epochs = 30
    else:
        ks = [float(x) for x in args.ks.split(",")]
        epochs = args.epochs

    print(f"[hessian] plan: {len(ks)} runs, ks={ks}, n={args.n_train}, ep={epochs}",
          flush=True)

    summary = []
    t_total = time.time()
    for i, k in enumerate(ks, 1):
        run_name = f"k{k:g}_n{args.n_train}_ep{epochs}_s{args.seed}"
        run_dir = out_root / run_name
        rp = run_dir / "results.json"
        if rp.exists() and not args.force:
            print(f"\n[{i}/{len(ks)}] {run_name}  SKIP cached", flush=True)
            with open(rp) as f:
                summary.append(json.load(f))
            continue
        print(f"\n[{i}/{len(ks)}] {run_name}", flush=True)
        t0 = time.time()
        r = train_then_hessian(
            k=k, n_train=args.n_train, noise_rate=args.noise_rate,
            epochs=epochs, seed=args.seed,
            data_dir=args.data_dir, out_dir=str(run_dir),
            n_samples_hessian=args.n_samples_hessian,
            power_iters=args.power_iters,
            batch_size=args.batch_size, lr=args.lr,
            device=args.device, log_every=args.log_every,
        )
        summary.append(r)
        with open(out_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  done in {(time.time()-t0)/60:.1f} min  test={r['test_acc']:.2f}"
              f"  top_eig={r['top_hessian_eig']:.3e}", flush=True)

    print(f"\n[hessian] complete. wall {(time.time()-t_total)/60:.1f} min",
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ks", default="0.0625,0.125,0.1875,0.5,2.0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--noise-rate", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-samples-hessian", type=int, default=256)
    p.add_argument("--power-iters", type=int, default=30)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-root", default="./results/hessian_topeig")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

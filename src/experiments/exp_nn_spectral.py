"""
NN spectral mechanism on fractional-k ResNet (R3 / report §6.10).

Usage:
  python -m src.experiments.exp_nn_spectral --device cuda --epochs 500
  python -m src.experiments.exp_nn_spectral --smoke --device cuda

Output:
  results/nn_spectral/k{K}_n{N}_ep{EP}_s{S}/
    config.json
    spectrum.json   (full singular spectrum + scalar diagnostics)
  results/nn_spectral/summary.json
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.experiments.exp_dd_recovery import (
    ResNetK, build_cifar_datasets, corrupt_and_subsample,
    IndexedSubset,
)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


DEFAULT_KS = [0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 2.0]


def train_for_spectrum(k, n_train, noise_rate, epochs, seed,
                       data_dir, batch_size=256, lr=1e-4, device="cuda",
                       log_every=100):
    """Lighter-weight training loop than exp_dd_recovery (no per-epoch test eval).

    Returns the trained model and the test_loader for spectrum computation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_full, test_full = build_cifar_datasets(data_dir, augment=True)
    subset, _, _, _ = corrupt_and_subsample(train_full, n_train, noise_rate, seed)
    indexed = IndexedSubset(subset)
    train_loader = DataLoader(indexed, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True,
                              persistent_workers=True)
    test_loader = DataLoader(test_full, batch_size=512, shuffle=False,
                             num_workers=2, pin_memory=True,
                             persistent_workers=True)

    model = ResNetK(k).to(device)
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
                pred = out.argmax(1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        last_train_acc = 100.0 * correct / max(1, total)
        if ep % log_every == 0 or ep == epochs:
            print(f"    [k={k:g}] ep {ep:4d}/{epochs} train_acc={last_train_acc:5.2f}"
                  f" [{time.time()-t0:.0f}s]", flush=True)

    return model, test_loader, last_train_acc, time.time() - t0


@torch.no_grad()
def collect_features(model, loader, n_samples, device):
    """Return Z ∈ R^{n×c3} of penultimate-layer features (test set)."""
    model.eval()
    feats = []
    seen = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _, z = model(x, return_feats=True)
        feats.append(z.float().cpu())
        seen += x.size(0)
        if seen >= n_samples:
            break
    Z = torch.cat(feats, dim=0)[:n_samples]
    return Z


@torch.no_grad()
def test_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def spectrum_diagnostics(Z):
    """Compute a battery of spectral diagnostics from feature matrix Z."""
    Zc = Z - Z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Zc).cpu().numpy().astype(float)  # descending
    s = s[s > 0]  # drop trivially-zero singular values

    fro2 = float((s ** 2).sum())
    op2 = float(s[0] ** 2)
    eff_rank_stable = fro2 / max(op2, 1e-12)        # ||Z||_F^2 / ||Z||_op^2

    # Renyi-2 entropy participation ratio: (Σλ_i)^2 / Σλ_i^2  where λ = s^2
    lam = s ** 2
    pr = float((lam.sum() ** 2) / max(float((lam ** 2).sum()), 1e-12))

    cond = float(s[0] / s[-1]) if len(s) > 1 else float("inf")

    # Spectral entropy on normalized λ
    p = lam / max(float(lam.sum()), 1e-12)
    spec_entropy = float(-(p * np.log(p + 1e-30)).sum())

    return {
        "n_samples": int(Z.shape[0]),
        "feat_dim": int(Z.shape[1]),
        "eff_rank_stable": eff_rank_stable,
        "participation_ratio": pr,
        "condition_number": cond,
        "spectral_entropy": spec_entropy,
        "top_singular_values": s[:50].tolist(),
        "full_singular_values": s.tolist(),
        "fro_norm": float(math.sqrt(fro2)),
        "op_norm": float(math.sqrt(op2)),
    }


def run(args):
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        ks = [0.5]
        epochs = 50
        seeds = [42]
        n_train = 1000
    else:
        ks = [float(x) for x in args.ks.split(",")]
        epochs = args.epochs
        seeds = [int(x) for x in args.seeds.split(",")]
        n_train = args.n_train

    grid = [(k, s) for s in seeds for k in ks]
    print(f"[spectral] plan: {len(grid)} runs, ks={ks}, seeds={seeds},"
          f" n={n_train}, ep={epochs}", flush=True)

    summary = []
    for i, (k, seed) in enumerate(grid, 1):
        run_name = f"k{k:g}_n{n_train}_ep{epochs}_s{seed}"
        run_dir = out_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        spec_path = run_dir / "spectrum.json"
        if spec_path.exists() and not args.force:
            print(f"\n[{i}/{len(grid)}] {run_name}  SKIP (cached)", flush=True)
            with open(spec_path) as f:
                summary.append(json.load(f))
            continue

        print(f"\n[{i}/{len(grid)}] {run_name}", flush=True)
        t0 = time.time()
        model, test_loader, train_acc, train_wall = train_for_spectrum(
            k=k, n_train=n_train, noise_rate=args.noise_rate,
            epochs=epochs, seed=seed,
            data_dir=args.data_dir,
            batch_size=args.batch_size, lr=args.lr,
            device=args.device, log_every=args.log_every,
        )

        # Spectrum on penultimate-layer features (test subset)
        Z = collect_features(model, test_loader, args.n_features_samples,
                             args.device)
        diag = spectrum_diagnostics(Z)
        ta = test_acc(model, test_loader, args.device)

        record = dict(
            k=k, n=n_train, seed=seed, epochs=epochs,
            params=model.count_params(),
            widths=list(model.widths),
            train_acc=train_acc,
            test_acc=ta,
            wallclock_sec=time.time() - t0,
            train_wallclock_sec=train_wall,
            **diag,
        )
        with open(spec_path, "w") as f:
            json.dump(record, f, indent=2)
        with open(run_dir / "config.json", "w") as f:
            json.dump(dict(k=k, n_train=n_train, seed=seed, epochs=epochs,
                           noise_rate=args.noise_rate, lr=args.lr,
                           batch_size=args.batch_size,
                           n_features_samples=args.n_features_samples), f,
                      indent=2)
        summary.append(record)

        # Free memory before next k
        del model, test_loader, Z
        torch.cuda.empty_cache()

        # Slim summary written after each run for partial-progress safety
        slim = [{kk: vv for kk, vv in r.items() if kk != "full_singular_values"}
                for r in summary]
        with open(out_root / "summary.json", "w") as f:
            json.dump(slim, f, indent=2)
        print(f"  done in {(time.time()-t0)/60:.1f} min  test_acc={ta:.2f}"
              f"  eff_rank={diag['eff_rank_stable']:.2f}"
              f"  cond={diag['condition_number']:.1f}", flush=True)

    print(f"\n[spectral] complete. summary at {out_root/'summary.json'}",
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ks", default=",".join(str(x) for x in DEFAULT_KS))
    p.add_argument("--seeds", default="42")
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--noise-rate", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--n-features-samples", type=int, default=2048)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-root", default="./results/nn_spectral")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

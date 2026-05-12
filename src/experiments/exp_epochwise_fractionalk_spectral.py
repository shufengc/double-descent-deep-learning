"""
Train-time penultimate-feature spectrum (§6.10-style) on the fractional-k epochwise protocol.

Outputs:
  results/fractionalk_epochwise_spectral/
    k{K}_n4000_ep2000_s{S}/results.json   (history + spectral_trace)
    summary.json
  figures/fractionalk_epochwise_spectral.png
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.experiments.exp_dd_recovery import ResNetK, build_cifar_datasets, corrupt_and_subsample
from src.experiments.exp_nn_spectral import collect_features, spectrum_diagnostics

FIXED_KS = [0.125, 0.1875, 0.5]
FIXED_SEEDS = [42, 7]
FIXED_EPOCHS = 2000
FIXED_EVAL_EVERY = 25
FIXED_N_TRAIN = 4000
FIXED_NOISE = 0.15


def _split_train_val(subset, val_frac, seed):
    n = len(subset)
    rng = np.random.RandomState(seed + 999)
    order = np.arange(n)
    rng.shuffle(order)
    n_val = max(1, int(n * val_frac))
    val_idx = order[:n_val].tolist()
    train_idx = order[n_val:].tolist()
    return Subset(subset, train_idx), Subset(subset, val_idx)


@torch.no_grad()
def _evaluate(model, loader, device, loss_fn):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss = loss_fn(out, y)
        total += y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        loss_sum += loss.item() * y.size(0)
    return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)


def _compact_spectral(diag: dict, c3: int) -> dict:
    """Scalars + moderate-sized singular lists for JSON (aligned with §6.10)."""
    out = {
        "eff_rank_stable": diag["eff_rank_stable"],
        "eff_rank_over_c3": diag["eff_rank_stable"] / max(c3, 1),
        "participation_ratio": diag["participation_ratio"],
        "pr_over_c3": diag["participation_ratio"] / max(c3, 1),
        "condition_number": diag["condition_number"],
        "spectral_entropy": diag["spectral_entropy"],
        "fro_norm": diag["fro_norm"],
        "op_norm": diag["op_norm"],
        "top_singular_values": diag["top_singular_values"],
    }
    # Full spectrum is small (min(N, c3) ≤ c3 for wide Z)
    if len(diag["full_singular_values"]) <= 64:
        out["full_singular_values"] = diag["full_singular_values"]
    return out


def _run_one(k, seed, args):
    run_name = f"k{k:g}_n{FIXED_N_TRAIN}_ep{FIXED_EPOCHS}_s{seed}"
    run_dir = Path(args.out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "results.json"

    if result_path.exists() and not args.force:
        with open(result_path, "r") as f:
            cached = json.load(f)
        print(f"[fractionalk-spectral] SKIP cached: {run_name}", flush=True)
        return cached

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_full, test_full = build_cifar_datasets(args.data_dir, augment=True)
    subset, _, _, _ = corrupt_and_subsample(train_full, FIXED_N_TRAIN, FIXED_NOISE, seed)
    train_subset, val_subset = _split_train_val(subset, args.val_frac, seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=512,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_full,
        batch_size=512,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    model = ResNetK(k).to(args.device)
    c3 = model.widths[2]
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    spectral_trace = []
    best_val_acc = -1.0
    best_epoch = -1
    best_test_acc = None
    t0 = time.time()

    for ep in range(1, FIXED_EPOCHS + 1):
        model.train()
        tr_total, tr_correct, tr_loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_total += y.size(0)
            tr_correct += (out.argmax(1) == y).sum().item()
            tr_loss_sum += loss.item() * y.size(0)

        record = {
            "epoch": ep,
            "train_loss": tr_loss_sum / max(tr_total, 1),
            "train_acc": 100.0 * tr_correct / max(tr_total, 1),
        }

        if ep % FIXED_EVAL_EVERY == 0 or ep == FIXED_EPOCHS:
            val_loss, val_acc = _evaluate(model, val_loader, args.device, loss_fn)
            test_loss, test_acc = _evaluate(model, test_loader, args.device, loss_fn)
            record.update({
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            })
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = ep
                best_test_acc = test_acc

        if ep % args.spectral_every == 0 or ep == FIXED_EPOCHS:
            Z = collect_features(model, test_loader, args.spectral_samples, args.device)
            diag = spectrum_diagnostics(Z)
            snap = {"epoch": ep, **_compact_spectral(diag, c3)}
            spectral_trace.append(snap)
            record["spectral"] = {
                "eff_rank_stable": snap["eff_rank_stable"],
                "eff_rank_over_c3": snap["eff_rank_over_c3"],
                "condition_number": snap["condition_number"],
            }

        if ep == 1 or ep % args.log_every == 0 or ep == FIXED_EPOCHS:
            dt = time.time() - t0
            extra = ""
            if "spectral" in record:
                extra = (
                    f" sr={record['spectral']['eff_rank_stable']:.2f}"
                    f" sr/c3={record['spectral']['eff_rank_over_c3']:.3f}"
                )
            print(
                f"  [{run_name}] ep {ep:4d}/{FIXED_EPOCHS} "
                f"tr_acc={record['train_acc']:.2f}% "
                + (
                    f"val_acc={record.get('val_acc', float('nan')):.2f}% "
                    f"te_acc={record.get('test_acc', float('nan')):.2f}% "
                    if "val_acc" in record
                    else ""
                )
                + extra
                + f" elapsed={dt/60:.1f}m",
                flush=True,
            )

        history.append(record)

    final_eval = next((r for r in reversed(history) if "test_acc" in r), None)
    final_test_acc = final_eval["test_acc"] if final_eval else None
    final_train_acc = history[-1]["train_acc"]
    gap = (best_test_acc - final_test_acc) if (best_test_acc is not None and final_test_acc is not None) else None

    last_spec = spectral_trace[-1] if spectral_trace else {}
    result = {
        "config": {
            "k": k,
            "seed": seed,
            "n_train": FIXED_N_TRAIN,
            "noise_rate": FIXED_NOISE,
            "epochs": FIXED_EPOCHS,
            "eval_every": FIXED_EVAL_EVERY,
            "spectral_every": args.spectral_every,
            "spectral_samples": args.spectral_samples,
            "c3": c3,
            "val_frac": args.val_frac,
            "lr": args.lr,
            "batch_size": args.batch_size,
        },
        "params": model.count_params(),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "final_test_acc": final_test_acc,
        "final_train_acc": final_train_acc,
        "gap": gap,
        "final_eff_rank_stable": last_spec.get("eff_rank_stable"),
        "final_eff_rank_over_c3": last_spec.get("eff_rank_over_c3"),
        "history": history,
        "spectral_trace": spectral_trace,
        "wallclock_sec": time.time() - t0,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[fractionalk-spectral] wrote {result_path}", flush=True)
    return result


def _plot(all_results, figure_path):
    by_k = {k: [] for k in FIXED_KS}
    for r in all_results:
        by_k[r["config"]["k"]].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {0.125: "tab:blue", 0.1875: "tab:orange", 0.5: "tab:green"}

    # Left: normalised stable rank vs epoch (spectral collapse / expansion trajectory)
    for k in FIXED_KS:
        runs = by_k[k]
        for r in runs:
            tr = r.get("spectral_trace") or []
            xs = [t["epoch"] for t in tr]
            ys = [t["eff_rank_over_c3"] for t in tr]
            axes[0].plot(xs, ys, alpha=0.55, color=colors[k], linewidth=1.6)
        if runs:
            axes[0].plot([], [], color=colors[k], label=f"k={k:g}")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(r"$\|Z_c\|_F^2 / (\|Z_c\|_{\mathrm{op}}^2 \cdot c_3)$")
    axes[0].set_title("Train-time penultimate spectrum (§6.10 normalisation)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Right: test accuracy vs epoch (same protocol as epochwise)
    for k in FIXED_KS:
        runs = by_k[k]
        for r in runs:
            pts = [(h["epoch"], h["test_acc"]) for h in r["history"] if "test_acc" in h]
            if pts:
                xs, ys = zip(*pts)
                axes[1].plot(xs, ys, alpha=0.55, color=colors[k], linewidth=1.6)
        if runs:
            axes[1].plot([], [], color=colors[k], label=f"k={k:g}")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test accuracy (%)")
    axes[1].set_title("Test accuracy (eval every 25 epochs)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(figure_path) or ".", exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fractionalk-spectral] wrote figure: {figure_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-root", default="./results/fractionalk_epochwise_spectral")
    p.add_argument("--figure-path", default="./figures/fractionalk_epochwise_spectral.png")
    p.add_argument("--device", default="cuda")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--spectral-every", type=int, default=100)
    p.add_argument("--spectral-samples", type=int, default=2048)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")

    os.makedirs(args.out_root, exist_ok=True)
    grid = [(k, s) for k in FIXED_KS for s in FIXED_SEEDS]
    all_results = []
    for idx, (k, seed) in enumerate(grid, 1):
        print(f"\n[{idx}/{len(grid)}] k={k:g}, seed={seed}", flush=True)
        all_results.append(_run_one(k, seed, args))

    summary = []
    for r in all_results:
        summary.append({
            "k": r["config"]["k"],
            "seed": r["config"]["seed"],
            "params": r["params"],
            "c3": r["config"]["c3"],
            "best_epoch": r["best_epoch"],
            "best_test_acc": r["best_test_acc"],
            "final_test_acc": r["final_test_acc"],
            "gap": r["gap"],
            "final_eff_rank_stable": r["final_eff_rank_stable"],
            "final_eff_rank_over_c3": r["final_eff_rank_over_c3"],
        })
    summary_path = Path(args.out_root) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[fractionalk-spectral] wrote summary: {summary_path}", flush=True)

    _plot(all_results, args.figure_path)


if __name__ == "__main__":
    main()

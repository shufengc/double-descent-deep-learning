"""
Epoch-wise early-stopping diagnostics on fractional-k ResNet.

Outputs:
  results/fractionalk_epochwise/
    k{K}_n4000_ep2000_s{S}/results.json
    summary.json
  figures/fractionalk_epochwise.png
"""

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


def _run_one(k, seed, args):
    run_name = f"k{k:g}_n{FIXED_N_TRAIN}_ep{FIXED_EPOCHS}_s{seed}"
    run_dir = Path(args.out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "results.json"

    if result_path.exists() and not args.force:
        with open(result_path, "r") as f:
            cached = json.load(f)
        print(f"[fractionalk-epochwise] SKIP cached: {run_name}", flush=True)
        return cached

    source_run = Path(args.reuse_existing_from) / run_name / "results.json"
    if args.reuse_existing_from and source_run.exists() and not args.force:
        with open(source_run, "r") as f:
            src = json.load(f)
        history = []
        best_val_acc = -1.0
        best_epoch = -1
        best_test_acc = None
        for h in src["history"]:
            rec = {
                "epoch": h["epoch"],
                "train_loss": h["train_loss"],
                "train_acc": h["train_acc"],
            }
            if "test_acc" in h:
                # Reuse DD-Recovery test checkpoints as validation proxy.
                rec["val_acc"] = h["test_acc"]
                rec["val_loss"] = h["test_loss"]
                rec["test_acc"] = h["test_acc"]
                rec["test_loss"] = h["test_loss"]
                if rec["val_acc"] > best_val_acc:
                    best_val_acc = rec["val_acc"]
                    best_epoch = h["epoch"]
                    best_test_acc = rec["test_acc"]
            history.append(rec)

        final_eval = next((r for r in reversed(history) if "test_acc" in r), None)
        final_test_acc = final_eval["test_acc"] if final_eval else None
        final_train_acc = history[-1]["train_acc"]
        gap = (best_test_acc - final_test_acc) if (best_test_acc is not None and final_test_acc is not None) else None

        result = {
            "config": {
                "k": k,
                "seed": seed,
                "n_train": FIXED_N_TRAIN,
                "noise_rate": FIXED_NOISE,
                "epochs": FIXED_EPOCHS,
                "eval_every": FIXED_EVAL_EVERY,
                "val_frac": args.val_frac,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "source": str(source_run),
            },
            "params": src["params"],
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "best_test_acc": best_test_acc,
            "final_test_acc": final_test_acc,
            "final_train_acc": final_train_acc,
            "gap": gap,
            "history": history,
            "wallclock_sec": 0.0,
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[fractionalk-epochwise] reused existing run: {run_name}", flush=True)
        return result

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
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    history = []
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

        if ep == 1 or ep % args.log_every == 0 or ep == FIXED_EPOCHS:
            dt = time.time() - t0
            print(
                f"  [{run_name}] ep {ep:4d}/{FIXED_EPOCHS} "
                f"tr_acc={record['train_acc']:.2f}% "
                + (f"val_acc={record.get('val_acc', float('nan')):.2f}% "
                   f"te_acc={record.get('test_acc', float('nan')):.2f}% "
                   if "val_acc" in record else "")
                + f"elapsed={dt/60:.1f}m",
                flush=True,
            )

        history.append(record)

    final_eval = next((r for r in reversed(history) if "test_acc" in r), None)
    final_test_acc = final_eval["test_acc"] if final_eval else None
    final_train_acc = history[-1]["train_acc"]
    gap = (best_test_acc - final_test_acc) if (best_test_acc is not None and final_test_acc is not None) else None

    result = {
        "config": {
            "k": k,
            "seed": seed,
            "n_train": FIXED_N_TRAIN,
            "noise_rate": FIXED_NOISE,
            "epochs": FIXED_EPOCHS,
            "eval_every": FIXED_EVAL_EVERY,
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
        "history": history,
        "wallclock_sec": time.time() - t0,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[fractionalk-epochwise] wrote {result_path}", flush=True)
    return result


def _plot(all_results, figure_path):
    by_k = {k: [] for k in FIXED_KS}
    for r in all_results:
        by_k[r["config"]["k"]].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {0.125: "tab:blue", 0.1875: "tab:orange", 0.5: "tab:green"}

    # Left: test acc vs epoch, one curve per run
    for k in FIXED_KS:
        runs = by_k[k]
        for r in runs:
            epochs = [h["epoch"] for h in r["history"] if "test_acc" in h]
            test_acc = [h["test_acc"] for h in r["history"] if "test_acc" in h]
            axes[0].plot(epochs, test_acc, alpha=0.55, color=colors[k], linewidth=1.6)
            axes[0].scatter([r["best_epoch"]], [r["best_test_acc"]], color=colors[k], s=26)
        if runs:
            label = f"k={k:g}"
            axes[0].plot([], [], color=colors[k], label=label)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_title("Fractional-k Epoch-wise Dynamics")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Right: best vs final per k (seed-avg)
    ks = FIXED_KS
    x = np.arange(len(ks))
    best_vals = []
    final_vals = []
    for k in ks:
        runs = by_k[k]
        best_vals.append(float(np.mean([r["best_test_acc"] for r in runs])))
        final_vals.append(float(np.mean([r["final_test_acc"] for r in runs])))
    width = 0.35
    axes[1].bar(x - width / 2, best_vals, width, label="Best test acc (early-stop)")
    axes[1].bar(x + width / 2, final_vals, width, label="Final test acc (epoch 2000)")
    axes[1].set_xticks(x, [f"k={k:g}" for k in ks])
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Early-stop vs Final (seed average)")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fractionalk-epochwise] wrote figure: {figure_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-root", default="./results/fractionalk_epochwise")
    p.add_argument("--figure-path", default="./figures/fractionalk_epochwise.png")
    p.add_argument("--device", default="cuda")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--reuse-existing-from", default="./results/dd_recovery_5090_focused/main")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available() and not args.reuse_existing_from:
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
            "best_epoch": r["best_epoch"],
            "best_test_acc": r["best_test_acc"],
            "final_test_acc": r["final_test_acc"],
            "gap": r["gap"],
        })
    summary_path = Path(args.out_root) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[fractionalk-epochwise] wrote summary: {summary_path}", flush=True)

    _plot(all_results, args.figure_path)


if __name__ == "__main__":
    main()

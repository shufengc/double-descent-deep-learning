"""
Corrected architecture comparison using SGD + weight decay.

This script keeps the original Yusheng Exp 5 width grid so that each
architecture crosses the interpolation threshold (p/n ~ 1), while using a
better-behaved optimizer setting than the Adam baseline.
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import corrupt_labels, get_cifar10, make_loaders, make_subset
from src.models import CNN, MLP, ResNet
from src.trainer import Trainer


FULL_CONFIGS = {
    "MLP": [1, 2, 5, 10, 20, 50, 100, 200],
    "CNN": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
    "ResNet": [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
}

SMOKE_CONFIGS = {
    "MLP": [20],
    "CNN": [8],
    "ResNet": [0.5],
}


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def make_model(arch, width):
    if arch == "MLP":
        return MLP(
            input_dim=3072,
            num_classes=10,
            hidden_width=int(width),
            num_hidden_layers=1,
        )
    if arch == "CNN":
        return CNN(num_classes=10, num_filters=int(width), input_channels=3)
    if arch == "ResNet":
        return ResNet(num_classes=10, k=float(width))
    raise ValueError(f"Unknown architecture: {arch}")


def width_id(width):
    return f"{float(width):g}"


def load_results(results_path, overwrite=False):
    if overwrite or not os.path.exists(results_path):
        return {}
    with open(results_path, "r") as f:
        return json.load(f)


def save_results(results, results_path):
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def plot_results(results, save_path, n_train, noise_rate, optimizer_label):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    colors = {"MLP": "tab:green", "CNN": "tab:blue", "ResNet": "tab:purple"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for arch, rows in results.items():
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: row["num_params"])
        params = [row["num_params"] for row in rows]
        final_err = [100 - row["test_acc"] for row in rows]
        best_err = [100 - row["best_test_acc"] for row in rows]
        train_err = [100 - row["train_acc"] for row in rows]

        axes[0].plot(
            params, final_err, "o-", color=colors[arch],
            label=f"{arch} (final)", markersize=5,
        )
        axes[0].plot(
            params, best_err, "s--", color=colors[arch],
            label=f"{arch} (best)", markersize=4, alpha=0.6,
        )
        axes[1].plot(
            params, train_err, "o-", color=colors[arch],
            label=arch, markersize=5,
        )

    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(
            n_train, color="gray", linestyle=":", alpha=0.5,
            label=f"p=n={n_train}",
        )
        ax.set_xlabel("Number of Parameters")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Error (%)")
    axes[0].set_title("Architecture Comparison: Test Error")
    axes[1].set_ylabel("Train Error (%)")
    axes[1].set_title("Architecture Comparison: Train Error")

    plt.suptitle(
        f"CIFAR-10 (n={n_train}, {noise_rate:.0%} noise, {optimizer_label})",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def select_configs(preset, architectures):
    base = SMOKE_CONFIGS if preset == "smoke" else FULL_CONFIGS
    return {arch: base[arch] for arch in architectures}


def run(args):
    device = get_device()
    print(f"Device: {device}", flush=True)
    print(
        f"Preset={args.preset} | epochs={args.epochs} | optimizer={args.optimizer} "
        f"| lr={args.lr} | weight_decay={args.weight_decay} | "
        f"scheduler={args.scheduler} | augment={args.augment}",
        flush=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")
    plot_path = os.path.join(args.output_dir, "dd_curves.png")

    architectures = [arch.strip() for arch in args.architectures.split(",") if arch.strip()]
    configs = select_configs(args.preset, architectures)
    all_results = load_results(results_path, overwrite=args.overwrite)

    train_full, test_set = get_cifar10(data_dir=args.data_dir, augment=args.augment)
    if args.noise > 0:
        train_full = corrupt_labels(train_full, args.noise, seed=args.seed)
    train_set = make_subset(train_full, args.n_train, seed=args.seed)
    train_loader, test_loader = make_loaders(
        train_set,
        test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    optimizer_label = (
        f"{args.optimizer.upper()}, lr={args.lr}, wd={args.weight_decay}, "
        f"scheduler={args.scheduler}"
    )

    for arch, widths in configs.items():
        print(f"\n{'=' * 60}\n  {arch}\n{'=' * 60}", flush=True)
        existing_rows = all_results.get(arch, [])
        completed = {row.get("width_id", width_id(row["width"])) for row in existing_rows}

        for width in widths:
            current_width_id = width_id(width)
            if current_width_id in completed:
                print(f"  Skipping {arch} width={width} (already complete)", flush=True)
                continue

            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

            model = make_model(arch, width)
            num_params = model.count_parameters()
            ratio = num_params / args.n_train

            trainer = Trainer(
                model,
                device=device,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                optimizer_type=args.optimizer,
                scheduler_type=None if args.scheduler == "none" else args.scheduler,
                scheduler_tmax=args.epochs,
            )

            t0 = time.time()
            history = trainer.train(
                train_loader,
                test_loader,
                epochs=args.epochs,
                log_interval=args.log_interval,
                verbose=True,
            )
            elapsed = time.time() - t0

            row = {
                "arch": arch,
                "width": width,
                "width_id": current_width_id,
                "num_params": num_params,
                "p_over_n": round(ratio, 4),
                "train_loss": history["train_loss"][-1],
                "train_acc": history["train_acc"][-1],
                "test_loss": history["test_loss"][-1],
                "test_acc": history["test_acc"][-1],
                "best_test_acc": max(history["test_acc"]),
                "elapsed_s": round(elapsed, 1),
                "optimizer": args.optimizer,
                "lr": args.lr,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "scheduler": args.scheduler,
                "augment": args.augment,
                "epochs": args.epochs,
            }
            existing_rows.append(row)
            all_results[arch] = existing_rows
            save_results(all_results, results_path)
            plot_results(
                all_results,
                plot_path,
                n_train=args.n_train,
                noise_rate=args.noise,
                optimizer_label=optimizer_label,
            )

            print(
                f"  {arch} w={width:<5} p={num_params:>8,} p/n={ratio:>7.2f} "
                f"train_err={100 - row['train_acc']:5.1f}% "
                f"test_err={100 - row['test_acc']:5.1f}% "
                f"best_test_err={100 - row['best_test_acc']:5.1f}% "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    print(f"\nResults saved to {args.output_dir}/", flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "full"], default="full")
    parser.add_argument("--architectures", default="MLP,CNN,ResNet")
    parser.add_argument(
        "--output-dir",
        default="./results/yusheng_exp5_architecture_sgd_wd",
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", choices=["none", "cosine", "step"], default="cosine")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

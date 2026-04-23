"""
Experiment: Architecture Comparison for Model-Wise Double Descent
Compares MLP, CNN, ResNet on CIFAR-10 (n=4000, 10% noise, Adam, 500 epochs).
"""

import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models import MLP, CNN, ResNet
from src.data import get_cifar10, corrupt_labels, make_subset, make_loaders
from src.trainer import Trainer


def run(output_dir="./results/exp_architecture", data_dir="./data",
        n_train=4000, noise_rate=0.1, epochs=500, seed=42):

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_full, test_set = get_cifar10(data_dir=data_dir, augment=False)
    train_full = corrupt_labels(train_full, noise_rate, seed=seed)
    train_set = make_subset(train_full, n_train, seed=seed)
    train_loader, test_loader = make_loaders(train_set, test_set, batch_size=256)

    configs = {
        "MLP": {
            "widths": [4, 8, 16, 32, 64, 128, 256],
            "make": lambda w: MLP(input_dim=3072, num_classes=10,
                                  hidden_width=w, num_hidden_layers=2),
        },
        "CNN": {
            "widths": [1, 2, 4, 6, 8, 12, 16, 24, 32],
            "make": lambda w: CNN(num_classes=10, num_filters=w, input_channels=3),
        },
        "ResNet": {
            "widths": [1, 2, 4, 8, 16],
            "make": lambda w: ResNet(num_classes=10, k=int(w)),
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for arch, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"  {arch}")
        print(f"{'='*60}")
        results = []
        for width in cfg["widths"]:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = cfg["make"](width)
            p = model.count_parameters()
            ratio = p / n_train

            trainer = Trainer(model, device=device, lr=1e-3,
                              weight_decay=0.0, optimizer_type="adam")
            t0 = time.time()
            history = trainer.train(train_loader, test_loader,
                                    epochs=epochs, log_interval=100,
                                    verbose=True)
            elapsed = time.time() - t0

            r = {
                "arch": arch,
                "width": width,
                "num_params": p,
                "p_over_n": round(ratio, 4),
                "train_loss": history["train_loss"][-1],
                "train_acc": history["train_acc"][-1],
                "test_loss": history["test_loss"][-1],
                "test_acc": history["test_acc"][-1],
                "best_test_acc": max(history["test_acc"]),
                "elapsed_s": round(elapsed, 1),
            }
            results.append(r)
            print(f"  {arch} w={width:<6} p={p:>8,} p/n={ratio:>7.2f} "
                  f"train_err={100-r['train_acc']:5.1f}% "
                  f"test_err={100-r['test_acc']:5.1f}% "
                  f"best_test_err={100-r['best_test_acc']:5.1f}% "
                  f"({elapsed:.0f}s)")
        all_results[arch] = results
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [saved incremental results for {arch}]")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    colors = {"MLP": "tab:green", "CNN": "tab:blue", "ResNet": "tab:purple"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for arch, results in all_results.items():
        results = sorted(results, key=lambda x: x["num_params"])
        params = [r["num_params"] for r in results]
        test_err = [100 - r["test_acc"] for r in results]
        best_err = [100 - r["best_test_acc"] for r in results]
        train_err = [100 - r["train_acc"] for r in results]

        axes[0].plot(params, test_err, "o-", color=colors[arch],
                     label=f"{arch} (final)", markersize=5)
        axes[0].plot(params, best_err, "s--", color=colors[arch],
                     label=f"{arch} (best)", markersize=4, alpha=0.6)

        axes[1].plot(params, train_err, "o-", color=colors[arch],
                     label=arch, markersize=5)

    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(n_train, color="gray", linestyle=":", alpha=0.5,
                   label=f"p=n={n_train}")
        ax.set_xlabel("Number of Parameters")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Test Error (%)")
    axes[0].set_title(f"Architecture Comparison: Test Error\n"
                      f"(CIFAR-10, n={n_train}, {noise_rate:.0%} noise, "
                      f"Adam, {epochs} epochs)")
    axes[1].set_ylabel("Train Error (%)")
    axes[1].set_title("Architecture Comparison: Train Error")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dd_curves.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./results/exp_architecture")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.output_dir, args.data_dir, args.n_train, args.noise,
        args.epochs, args.seed)

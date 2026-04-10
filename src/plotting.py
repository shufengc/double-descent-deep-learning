"""
Plotting utilities for double descent experiments.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "lines.linewidth": 2,
    "figure.dpi": 150,
})


def plot_model_wise_double_descent(results, save_path="results/model_wise_dd.png"):
    """
    Plot test error/loss vs model complexity (number of parameters).
    results: list of dicts with keys 'num_params', 'train_loss', 'test_loss',
             'train_acc', 'test_acc', and optionally 'label' for noise rates.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if isinstance(results, list):
        results = {0.0: results}
    elif isinstance(results, dict):
        first_val = next(iter(results.values()))
        if not isinstance(first_val, list):
            results = {0.0: results}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for noise_rate, data_list in sorted(results.items()):
        params = [r["num_params"] for r in data_list]
        test_err = [100.0 - r["test_acc"] for r in data_list]
        train_err = [100.0 - r["train_acc"] for r in data_list]

        label_suffix = f" (noise={noise_rate:.0%})" if noise_rate > 0 else " (clean)"
        axes[0].plot(params, test_err, "o-", label=f"Test{label_suffix}")
        axes[0].plot(params, train_err, "s--", alpha=0.5, label=f"Train{label_suffix}")

        test_loss = [r["test_loss"] for r in data_list]
        train_loss = [r["train_loss"] for r in data_list]
        axes[1].plot(params, test_loss, "o-", label=f"Test{label_suffix}")
        axes[1].plot(params, train_loss, "s--", alpha=0.5, label=f"Train{label_suffix}")

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Number of Parameters")
    axes[0].set_ylabel("Error (%)")
    axes[0].set_title("Model-Wise Double Descent: Error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Number of Parameters")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Model-Wise Double Descent: Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_epoch_wise_double_descent(histories, save_path="results/epoch_wise_dd.png"):
    """
    Plot test error vs epoch for different model sizes.
    histories: dict mapping model_label -> training history dict.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, hist in histories.items():
        epochs = hist["epoch"]
        test_err = [100.0 - a for a in hist["test_acc"]]
        train_err = [100.0 - a for a in hist["train_acc"]]

        axes[0].plot(epochs, test_err, label=f"{label} (test)")
        axes[0].plot(epochs, train_err, "--", alpha=0.4, label=f"{label} (train)")
        axes[1].plot(epochs, hist["test_loss"], label=f"{label} (test)")
        axes[1].plot(epochs, hist["train_loss"], "--", alpha=0.4, label=f"{label} (train)")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Error (%)")
    axes[0].set_title("Epoch-Wise Double Descent: Error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Epoch-Wise Double Descent: Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_sample_wise_double_descent(results, save_path="results/sample_wise_dd.png"):
    """
    Plot test error vs number of training samples.
    results: list of dicts with keys 'n_samples', 'test_acc', 'train_acc', 'test_loss', 'train_loss'.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_samples = [r["n_samples"] for r in results]
    test_err = [100.0 - r["test_acc"] for r in results]
    train_err = [100.0 - r["train_acc"] for r in results]

    axes[0].plot(n_samples, test_err, "o-", label="Test")
    axes[0].plot(n_samples, train_err, "s--", alpha=0.5, label="Train")
    axes[0].set_xlabel("Number of Training Samples")
    axes[0].set_ylabel("Error (%)")
    axes[0].set_title("Sample-Wise Double Descent: Error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    test_loss = [r["test_loss"] for r in results]
    train_loss = [r["train_loss"] for r in results]
    axes[1].plot(n_samples, test_loss, "o-", label="Test")
    axes[1].plot(n_samples, train_loss, "s--", alpha=0.5, label="Train")
    axes[1].set_xlabel("Number of Training Samples")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Sample-Wise Double Descent: Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_noise_comparison(results_by_noise, save_path="results/noise_comparison.png"):
    """
    Compare double descent curves across different noise rates.
    results_by_noise: dict mapping noise_rate -> list of dicts.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis
    noise_rates = sorted(results_by_noise.keys())
    colors = [cmap(i / max(len(noise_rates) - 1, 1)) for i in range(len(noise_rates))]

    for noise_rate, color in zip(noise_rates, colors):
        data_list = results_by_noise[noise_rate]
        params = [r["num_params"] for r in data_list]
        test_err = [100.0 - r["test_acc"] for r in data_list]
        ax.plot(params, test_err, "o-", color=color,
                label=f"Noise = {noise_rate:.0%}")

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Test Error (%)")
    ax.set_title("Effect of Label Noise on Model-Wise Double Descent")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

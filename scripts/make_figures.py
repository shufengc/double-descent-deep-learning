"""
Regenerate paper figures from saved experiment result JSON files.

This script does not rerun training. It reads the committed outputs under
results/**/results.json and redraws the figures referenced by report.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

COLORS = {
    "0.0": "#2b6cb0",
    "0.1": "#38a169",
    "0.2": "#dd6b20",
    "0.4": "#805ad5",
    "sgd": "#2b6cb0",
    "adam": "#c53030",
}


plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "axes.grid": True,
        "grid.alpha": 0.28,
    }
)


def load_json(result_dir: str):
    base = RESULTS / result_dir
    for name in ("summary.json", "results.json"):
        path = base / name
        if path.exists():
            return json.loads(path.read_text()), path
    raise FileNotFoundError(f"Expected summary.json or results.json under {base}")


def save(fig: plt.Figure, rel_path: str) -> None:
    path = ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {rel_path}")


def as_float_key(value: str) -> float:
    return float(value)


def sorted_noise_items(data: dict):
    return sorted(data.items(), key=lambda item: as_float_key(item[0]))


def ratio_series(records: list[dict], key: str):
    return [r["p_over_n"] for r in records], [r[key] for r in records]


def add_threshold(ax, x=1.0, label="interpolation"):
    ax.axvline(x, color="black", linestyle=":", linewidth=1.4, alpha=0.7)
    ymax = ax.get_ylim()[1]
    ax.text(x, ymax, f" {label}", va="top", ha="left", fontsize=8, alpha=0.75)


def fig1_model_wise_rff() -> None:
    data, _ = load_json("exp1_model_wise_rff")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for noise, records in sorted_noise_items(data):
        label = f"{int(float(noise) * 100)}% noise"
        x, test_mse = ratio_series(records, "test_mse")
        _, train_mse = ratio_series(records, "train_mse")
        _, test_acc = ratio_series(records, "test_acc")
        color = COLORS.get(noise)
        axes[0].plot(x, test_mse, "o-", color=color, label=f"test, {label}")
        axes[0].plot(x, train_mse, "--", color=color, alpha=0.55, label=f"train, {label}")
        axes[1].plot(x, test_acc, "o-", color=color, label=label)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Feature ratio p/n")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("RFF Model-wise Double Descent")
    add_threshold(axes[0])
    axes[0].legend(ncol=2)

    axes[1].set_xlabel("Feature ratio p/n")
    axes[1].set_ylabel("Test accuracy (%)")
    axes[1].set_title("Post-threshold Recovery")
    add_threshold(axes[1])
    axes[1].legend()
    save(fig, "figures/fig1_model_wise_rff.png")


def fig2_noise_effect() -> None:
    data, _ = load_json("exp1_model_wise_rff")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    peaks = []

    for noise, records in sorted_noise_items(data):
        label = f"{int(float(noise) * 100)}% noise"
        zoomed = [r for r in records if 0.7 <= r["p_over_n"] <= 1.3]
        x, test_mse = ratio_series(zoomed, "test_mse")
        color = COLORS.get(noise)
        axes[0].plot(x, test_mse, "o-", color=color, label=label)
        peak = max(records, key=lambda r: r["test_mse"])
        peaks.append((label, peak["test_mse"]))

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Feature ratio p/n")
    axes[0].set_ylabel("Test MSE")
    axes[0].set_title("Noise Amplifies the Interpolation Peak")
    add_threshold(axes[0])
    axes[0].legend()

    labels, values = zip(*peaks)
    bars = axes[1].bar(labels, values, color=[COLORS[k] for k, _ in sorted_noise_items(data)])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Peak test MSE")
    axes[1].set_title("Peak Height by Label Noise")
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    save(fig, "figures/fig2_noise_effect.png")


def table1_summary() -> None:
    exp1, _ = load_json("exp1_model_wise_rff")
    exp2, _ = load_json("exp2_sample_wise_rff")
    exp3, _ = load_json("exp3_nn_model_wise")
    exp4, _ = load_json("exp4_epoch_wise_nn")

    clean_peak = max(exp1["0.0"], key=lambda r: r["test_mse"])
    noisy_peak = max(exp1["0.2"], key=lambda r: r["test_mse"])
    sample_peak = max(exp2, key=lambda r: r["test_mse"])
    nn_noisy_final = max(exp3["0.2"], key=lambda r: r["p_over_n"])
    epoch_over = next(v for k, v in exp4.items() if "p/n=2.79" in k)

    rows = [
        ["RFF model-wise, clean", f"{clean_peak['test_mse']:.1f}", "p/n = 1.00", "92.9% acc at p/n=8"],
        ["RFF model-wise, 20% noise", f"{noisy_peak['test_mse']:.1f}", "p/n = 1.00", "83.0% acc at p/n=8"],
        ["RFF sample-wise", f"{sample_peak['test_mse']:.1f}", f"n = {sample_peak['n_samples']}", "More data can hurt near p=n"],
        ["CNN model-wise, 20% noise", f"{nn_noisy_final['test_loss']:.1f} loss", f"p/n = {nn_noisy_final['p_over_n']:.2f}", "Memorization without recovery"],
        ["CNN epoch-wise, wide", f"{epoch_over['test_loss'][-1]:.1f} loss", "1000 epochs", f"{epoch_over['train_acc'][-1]:.1f}% train acc"],
    ]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Experiment", "Peak / final metric", "Where", "Takeaway"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.30, 0.18, 0.18, 0.34],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#f8fafc" if row % 2 else "#ffffff")
    ax.set_title("Table 1: Summary of Main Empirical Findings", fontweight="bold", pad=12)
    save(fig, "figures/table1_summary.png")


def fig3_sample_wise_rff() -> None:
    data, _ = load_json("exp2_sample_wise_rff")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    n = [r["n_samples"] for r in data]
    test_mse = [r["test_mse"] for r in data]
    train_mse = [r["train_mse"] for r in data]
    test_acc = [r["test_acc"] for r in data]

    axes[0].plot(n, test_mse, "o-", label="test MSE", color="#2b6cb0")
    axes[0].plot(n, train_mse, "--", label="train MSE", color="#2b6cb0", alpha=0.55)
    axes[0].set_yscale("log")
    axes[0].axvline(500, color="black", linestyle=":", linewidth=1.4)
    axes[0].set_xlabel("Training samples n")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Sample-wise Double Descent, Fixed D=500")
    axes[0].legend()

    axes[1].plot(n, test_acc, "o-", color="#dd6b20")
    axes[1].axvline(500, color="black", linestyle=":", linewidth=1.4)
    axes[1].set_xlabel("Training samples n")
    axes[1].set_ylabel("Test accuracy (%)")
    axes[1].set_title("Adding Data Can Hurt Near p=n")
    save(fig, "figures/fig3_sample_wise_rff.png")


def fig4_nn_model_wise() -> None:
    data, _ = load_json("exp3_nn_model_wise")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for noise, records in sorted_noise_items(data):
        label = "clean" if noise == "0.0" else f"{int(float(noise) * 100)}% noise"
        x, test_acc = ratio_series(records, "test_acc")
        _, train_acc = ratio_series(records, "train_acc")
        _, test_loss = ratio_series(records, "test_loss")
        color = COLORS.get(noise)
        axes[0].plot(x, [100 - v for v in test_acc], "o-", color=color, label=f"test, {label}")
        axes[0].plot(x, [100 - v for v in train_acc], "--", color=color, alpha=0.55, label=f"train, {label}")
        axes[1].plot(x, test_loss, "o-", color=color, label=label)

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Parameter ratio p/n")
    axes[0].set_ylabel("Error (%)")
    axes[0].set_title("CNN Model-wise Double Descent")
    axes[0].legend(ncol=2)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Parameter ratio p/n")
    axes[1].set_ylabel("Test loss")
    axes[1].set_title("Noise Causes Memorization")
    axes[1].legend()
    save(fig, "figures/fig4_nn_model_wise.png")


def fig5_epoch_wise() -> None:
    data, _ = load_json("exp4_epoch_wise_nn")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for label, hist in data.items():
        epochs = hist["epoch"]
        axes[0].plot(epochs, [100 - v for v in hist["test_acc"]], label=label)
        axes[1].plot(epochs, hist["train_acc"], label=label)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test error (%)")
    axes[0].set_title("Epoch-wise Behavior Under Label Noise")
    axes[0].legend()
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Train accuracy (%)")
    axes[1].set_title("Memorization During Training")
    axes[1].legend()
    save(fig, "figures/fig5_epoch_wise.png")


def fig6_train_test_overview() -> None:
    exp1, _ = load_json("exp1_model_wise_rff")
    exp3, _ = load_json("exp3_nn_model_wise")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.ravel()

    for noise, records in sorted_noise_items(exp1):
        x, test_mse = ratio_series(records, "test_mse")
        _, train_mse = ratio_series(records, "train_mse")
        color = COLORS.get(noise)
        axes[0].plot(x, test_mse, "o-", color=color, label=f"{int(float(noise) * 100)}%")
        axes[1].plot(x, train_mse, "o-", color=color, label=f"{int(float(noise) * 100)}%")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_title("RFF test MSE")
    axes[1].set_title("RFF train MSE")

    for noise, records in sorted_noise_items(exp3):
        x, test_acc = ratio_series(records, "test_acc")
        _, train_acc = ratio_series(records, "train_acc")
        color = COLORS.get(noise)
        axes[2].plot(x, test_acc, "o-", color=color, label=f"{int(float(noise) * 100)}%")
        axes[3].plot(x, train_acc, "o-", color=color, label=f"{int(float(noise) * 100)}%")
    axes[2].set_xscale("log")
    axes[3].set_xscale("log")
    axes[2].set_title("CNN test accuracy")
    axes[3].set_title("CNN train accuracy")

    for ax in axes:
        ax.set_xlabel("p/n")
        ax.legend()
    save(fig, "figures/fig6_train_test_overview.png")


def result_noise_multiseed() -> None:
    data, _ = load_json("exp_noise_multiseed")
    aggregated = data["aggregated"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for noise, records in sorted_noise_items(aggregated):
        x = np.array([r["p_over_n"] for r in records])
        mse = np.array([r["test_mse_mean"] for r in records])
        mse_std = np.array([r["test_mse_std"] for r in records])
        acc = np.array([r["test_acc_mean"] for r in records])
        acc_std = np.array([r["test_acc_std"] for r in records])
        label = f"{int(float(noise) * 100)}% noise"
        color = COLORS.get(noise)
        axes[0].plot(x, mse, "o-", color=color, label=label)
        axes[0].fill_between(x, np.maximum(mse - mse_std, 1e-12), mse + mse_std, color=color, alpha=0.12)
        axes[1].plot(x, acc, "o-", color=color, label=label)
        axes[1].fill_between(x, acc - acc_std, acc + acc_std, color=color, alpha=0.12)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Feature ratio p/n")
    axes[0].set_ylabel("Mean test MSE")
    axes[0].set_title("Multi-seed Noise Robustness")
    add_threshold(axes[0])
    axes[0].legend()

    axes[1].set_xlabel("Feature ratio p/n")
    axes[1].set_ylabel("Mean test accuracy (%)")
    axes[1].set_title("Accuracy Across Seeds")
    add_threshold(axes[1])
    axes[1].legend()
    save(fig, "results/exp_noise_multiseed/dd_curves.png")


def result_bias_variance() -> None:
    data, _ = load_json("expB_bias_variance")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for noise, records in sorted_noise_items(data):
        x = [r["p_over_n"] for r in records]
        color = COLORS.get(noise)
        label = f"{int(float(noise) * 100)}% noise"
        axes[0].plot(x, [r["bias2"] for r in records], "--", color=color, label=f"bias^2, {label}")
        axes[0].plot(x, [r["variance"] for r in records], "o-", color=color, label=f"variance, {label}")
        axes[1].plot(x, [r["total_mse"] for r in records], "o-", color=color, label=label)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Feature ratio p/n")
    axes[0].set_ylabel("Bias/variance component")
    axes[0].set_title("Bias-Variance Decomposition")
    add_threshold(axes[0])
    axes[0].legend(ncol=2)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Feature ratio p/n")
    axes[1].set_ylabel("Total test MSE")
    axes[1].set_title("Peak Tracks Variance")
    add_threshold(axes[1])
    axes[1].legend()
    save(fig, "results/expB_bias_variance/bias_variance.png")

    fig2, ax = plt.subplots(figsize=(8, 5))
    for noise, records in sorted_noise_items(data):
        x = [r["p_over_n"] for r in records]
        ax.plot(x, [r["total_mse"] for r in records], "o-", color=COLORS.get(noise), label=f"{int(float(noise) * 100)}% noise")
    ax.set_yscale("log")
    ax.set_xlabel("Feature ratio p/n")
    ax.set_ylabel("Total test MSE")
    ax.set_title("Bias-Variance Experiment DD Curve")
    add_threshold(ax)
    ax.legend()
    save(fig2, "results/expB_bias_variance/dd_curves.png")


def result_epoch_sgd_resnet() -> None:
    data, _ = load_json("expC_epoch_sgd_resnet")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for row, optimizer in enumerate(["sgd", "adam"]):
        for k, record in sorted(data[optimizer].items(), key=lambda item: int(item[0])):
            hist = record["history"]
            label = f"k={k}, p/n={record['p_over_n']:.1f}"
            axes[row][0].plot(hist["epoch"], [100 - v for v in hist["test_acc"]], label=label)
            axes[row][1].plot(hist["epoch"], hist["train_acc"], label=label)
        axes[row][0].set_title(f"{optimizer.upper()}: test error")
        axes[row][1].set_title(f"{optimizer.upper()}: train accuracy")
        axes[row][0].set_ylabel("Error (%)")
        axes[row][1].set_ylabel("Accuracy (%)")
        for ax in axes[row]:
            ax.set_xlabel("Epoch")
            ax.legend()
    save(fig, "results/expC_epoch_sgd_resnet/epoch_wise_dd.png")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))
    for optimizer, records in data.items():
        ordered = [records[k] for k in sorted(records, key=lambda x: int(x))]
        x = [r["p_over_n"] for r in ordered]
        final_test_error = [100 - r["history"]["test_acc"][-1] for r in ordered]
        final_train_acc = [r["history"]["train_acc"][-1] for r in ordered]
        axes2[0].plot(x, final_test_error, "o-", color=COLORS.get(optimizer), label=optimizer.upper())
        axes2[1].plot(x, final_train_acc, "o-", color=COLORS.get(optimizer), label=optimizer.upper())
    axes2[0].set_xscale("log")
    axes2[1].set_xscale("log")
    axes2[0].set_xlabel("Parameter ratio p/n")
    axes2[1].set_xlabel("Parameter ratio p/n")
    axes2[0].set_ylabel("Final test error (%)")
    axes2[1].set_ylabel("Final train accuracy (%)")
    axes2[0].set_title("ResNet Epoch-wise Final Test Error")
    axes2[1].set_title("ResNet Memorization")
    axes2[0].legend()
    axes2[1].legend()
    save(fig2, "results/expC_epoch_sgd_resnet/dd_curves.png")


def result_emc() -> None:
    data, _ = load_json("expA_emc")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k, record in sorted(data.items(), key=lambda item: int(item[0])):
        emc = record["emc"]
        epochs = sorted((int(epoch) for epoch in emc.keys()))
        ax.plot(epochs, [emc[str(epoch)] for epoch in epochs], "o-", label=f"k={k}, params={record['num_params']:,}")
    ax.axhline(4000, color="black", linestyle=":", linewidth=1.4, label="n=4000")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Effective Model Complexity")
    ax.set_title("Effective Model Complexity Saturates Early")
    ax.legend()
    save(fig, "results/expA_emc/emc_curves.png")


def build_all() -> None:
    fig1_model_wise_rff()
    fig2_noise_effect()
    table1_summary()
    fig3_sample_wise_rff()
    fig4_nn_model_wise()
    fig5_epoch_wise()
    fig6_train_test_overview()
    result_noise_multiseed()
    result_bias_variance()
    result_epoch_sgd_resnet()
    result_emc()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-only", action="store_true", help="Regenerate only figures referenced by report.md.")
    args = parser.parse_args()
    build_all()


if __name__ == "__main__":
    main()

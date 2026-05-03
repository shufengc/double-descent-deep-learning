"""
Extension B — Label noise as a stress test for interpolation.

Extends the existing 0/10/20% RFF noise sweep to include 30% and 40%.
The figure overlays five test-MSE curves (one per noise rate) to make the
peak amplification visible. The accompanying summary table reports, for each
noise rate:

  - peak MSE (the maximum test_mse across the sweep)
  - peak location (p/n at the maximum)
  - peak-to-valley ratio (peak / min over p/n >= 1.5, i.e. recovery regime)
  - recovery test accuracy at p/n = 8

Reuses the same RFF pipeline as Extension A. Output:
results/personB_noise_sweep/{results.json, summary.json},
figures/personB_noise_amplification.png.
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.experiments.comprehensive_dd import (
    random_fourier_features,
    min_norm_solution,
    load_mnist_numpy,
)


NOISE_RATES = [0.0, 0.1, 0.2, 0.3, 0.4]
RATIOS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98,
          1.0, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0]


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.abspath(
        os.path.join(os.path.dirname(args.output_dir.rstrip("/")), "..", "figures"))
    os.makedirs(figures_dir, exist_ok=True)

    n = args.n_train
    seeds = list(range(args.n_seeds))
    all_results = {}

    for nr in NOISE_RATES:
        print(f"\n=== noise = {nr:.0%} ===")
        per_ratio = {r: [] for r in RATIOS}
        for seed in seeds:
            X_tr, Y_tr, y_tr, X_te, Y_te, y_te = load_mnist_numpy(
                args.data_dir, n_train=n, noise_rate=nr, seed=seed)
            for ratio in RATIOS:
                D = max(1, int(ratio * n))
                Phi_tr = random_fourier_features(X_tr, D, sigma=5.0, seed=seed)
                Phi_te = random_fourier_features(X_te, D, sigma=5.0, seed=seed)
                w = min_norm_solution(Phi_tr, Y_tr, lam=1e-10)
                pred_tr = Phi_tr @ w
                pred_te = Phi_te @ w
                per_ratio[ratio].append({
                    "seed": seed, "D": D, "p_over_n": ratio,
                    "train_mse": float(np.mean((Y_tr - pred_tr) ** 2)),
                    "test_mse": float(np.mean((Y_te - pred_te) ** 2)),
                    "train_acc": float(np.mean(np.argmax(pred_tr, 1) == y_tr) * 100),
                    "test_acc": float(np.mean(np.argmax(pred_te, 1) == y_te) * 100),
                })
        flat = []
        for ratio in RATIOS:
            tests = [r["test_mse"] for r in per_ratio[ratio]]
            flat.append({
                "p_over_n": ratio,
                "D": per_ratio[ratio][0]["D"],
                "test_mse_mean": float(np.mean(tests)),
                "test_mse_std": float(np.std(tests)),
                "test_acc_mean": float(np.mean([r["test_acc"] for r in per_ratio[ratio]])),
                "train_mse_mean": float(np.mean([r["train_mse"] for r in per_ratio[ratio]])),
                "per_seed": per_ratio[ratio],
            })
            print(f"  p/n={ratio:.2f}  D={flat[-1]['D']:5d}  "
                  f"test_mse={flat[-1]['test_mse_mean']:.4f}  "
                  f"test_acc={flat[-1]['test_acc_mean']:.1f}%")
        all_results[f"{nr:.2f}"] = flat

    # Summary table
    summary = []
    for nr in NOISE_RATES:
        rows = all_results[f"{nr:.2f}"]
        peak_row = max(rows, key=lambda r: r["test_mse_mean"])
        recovery_rows = [r for r in rows if r["p_over_n"] >= 1.5]
        valley = min(recovery_rows, key=lambda r: r["test_mse_mean"])
        recovery8 = next((r for r in rows if abs(r["p_over_n"] - 8.0) < 1e-9), None)
        summary.append({
            "noise_rate": nr,
            "peak_mse": peak_row["test_mse_mean"],
            "peak_location_p_over_n": peak_row["p_over_n"],
            "peak_to_valley_ratio": peak_row["test_mse_mean"] / max(valley["test_mse_mean"], 1e-12),
            "valley_mse": valley["test_mse_mean"],
            "valley_location_p_over_n": valley["p_over_n"],
            "test_acc_at_pn8": recovery8["test_acc_mean"] if recovery8 else None,
            "test_mse_at_pn8": recovery8["test_mse_mean"] if recovery8 else None,
        })

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({
            "config": {
                "n_train": n, "noise_rates": NOISE_RATES,
                "ratios": RATIOS, "n_seeds": args.n_seeds, "sigma": 5.0,
            },
            "results": all_results,
        }, f, indent=2)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary table ===")
    print(f"{'noise':>6}  {'peak_mse':>10}  {'peak_p/n':>9}  "
          f"{'pk/valley':>10}  {'acc@p/n=8':>10}")
    for s in summary:
        print(f"{s['noise_rate']:>6.0%}  {s['peak_mse']:>10.4f}  "
              f"{s['peak_location_p_over_n']:>9.2f}  "
              f"{s['peak_to_valley_ratio']:>10.2f}  "
              f"{s['test_acc_at_pn8']:>9.1f}%")

    # Figure
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("plasma")
    for i, nr in enumerate(NOISE_RATES):
        rows = sorted(all_results[f"{nr:.2f}"], key=lambda r: r["p_over_n"])
        x = [r["p_over_n"] for r in rows]
        y = [r["test_mse_mean"] for r in rows]
        s = [r["test_mse_std"] for r in rows]
        color = cmap(0.15 + 0.75 * i / max(1, len(NOISE_RATES) - 1))
        ax.plot(x, y, "o-", color=color, label=f"noise={nr:.0%}",
                markersize=4, linewidth=1.8)
        ax.fill_between(x, np.array(y) - np.array(s), np.array(y) + np.array(s),
                        color=color, alpha=0.12)
    ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.6, label="p/n=1")
    ax.set_xlabel("p / n")
    ax.set_ylabel("Test MSE")
    ax.set_yscale("log")
    ax.set_title(f"Label noise amplifies the double descent peak\n"
                 f"(RFF on MNIST, n={n}, {args.n_seeds} seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    out_png = os.path.join(figures_dir, "personB_noise_amplification.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nWrote figure {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./results/personB_noise_sweep")
    args = parser.parse_args()
    t0 = time.time()
    run(args)
    print(f"\nTotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

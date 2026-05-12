"""
Extension A — Ridge regularization smooths the RFF double descent peak.

Reuses random_fourier_features, min_norm_solution, load_mnist_numpy from
comprehensive_dd.py. 
Output: results/personA_ridge_sweep/results.json plus
figures/personA_ridge_smooths_peak.png.
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


LAMBDAS = [0.0, 1e-8, 1e-6, 1e-4, 1e-2]
RATIOS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98,
          1.0, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0]


def _solve_with_lam(Phi, Y, lam):
    n, p = Phi.shape
    eps = 1e-12 if lam == 0.0 else 0.0
    if p >= n:
        K = Phi @ Phi.T + (lam + eps) * np.eye(n)
        alpha = np.linalg.solve(K, Y)
        return Phi.T @ alpha
    G = Phi.T @ Phi + (lam + eps) * np.eye(p)
    return np.linalg.solve(G, Phi.T @ Y)


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(os.path.dirname(args.output_dir.rstrip("/")), "..", "figures")
    figures_dir = os.path.abspath(figures_dir)
    os.makedirs(figures_dir, exist_ok=True)

    n = args.n_train
    seeds = list(range(args.n_seeds))
    all_results = {}

    for lam in LAMBDAS:
        print(f"\n=== lambda = {lam:.0e} ===")
        per_ratio = {r: [] for r in RATIOS}
        for seed in seeds:
            X_tr, Y_tr, y_tr, X_te, Y_te, y_te = load_mnist_numpy(
                args.data_dir, n_train=n, noise_rate=args.noise_rate, seed=seed)
            for ratio in RATIOS:
                D = max(1, int(ratio * n))
                Phi_tr = random_fourier_features(X_tr, D, sigma=5.0, seed=seed)
                Phi_te = random_fourier_features(X_te, D, sigma=5.0, seed=seed)
                w = _solve_with_lam(Phi_tr, Y_tr, lam)
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
                  f"test_mse={flat[-1]['test_mse_mean']:.4f} "
                  f"+- {flat[-1]['test_mse_std']:.4f}")
        all_results[f"{lam:.0e}"] = flat

    out_json = os.path.join(args.output_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump({
            "config": {
                "n_train": n, "noise_rate": args.noise_rate,
                "lambdas": LAMBDAS, "ratios": RATIOS, "n_seeds": args.n_seeds,
                "sigma": 5.0,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nWrote {out_json}")

    # Figure
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("viridis")
    for i, lam in enumerate(LAMBDAS):
        key = f"{lam:.0e}"
        rows = sorted(all_results[key], key=lambda r: r["p_over_n"])
        x = [r["p_over_n"] for r in rows]
        y = [r["test_mse_mean"] for r in rows]
        s = [r["test_mse_std"] for r in rows]
        color = cmap(i / max(1, len(LAMBDAS) - 1))
        label = "λ=0 (ridgeless)" if lam == 0.0 else f"λ={lam:.0e}"
        ax.plot(x, y, "o-", color=color, label=label, markersize=4, linewidth=1.8)
        ax.fill_between(x, np.array(y) - np.array(s), np.array(y) + np.array(s),
                        color=color, alpha=0.12)
    ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.6, label="p/n=1")
    ax.set_xlabel("p / n")
    ax.set_ylabel("Test MSE")
    ax.set_yscale("log")
    ax.set_title(f"Ridge regularization smooths the double descent peak\n"
                 f"(RFF on MNIST, n={n}, noise={args.noise_rate:.0%}, "
                 f"{args.n_seeds} seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    out_png = os.path.join(figures_dir, "personA_ridge_smooths_peak.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Wrote {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--noise-rate", type=float, default=0.10)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./results/personA_ridge_sweep")
    args = parser.parse_args()
    t0 = time.time()
    run(args)
    print(f"\nTotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

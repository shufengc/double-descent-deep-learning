"""
Zhengda Exp8: Noise x Ridge Mechanism Study

This is a standalone follow-up experiment for Zhengda's Exp5 and the fixed Exp6.
Put this file under:
    src/experiments/zhengda_exp8_noise_lambda_mechanism.py

Run from the project root with:
    python -m src.experiments.zhengda_exp8_noise_lambda_mechanism

What this experiment studies:
1. Exp5 deep dive:
   Ridge regularization suppresses the double descent peak.
   Here we measure *why* by tracking solution norm, regularized condition number,
   and effective degrees of freedom.

2. Fixed Exp6 deep dive:
   Label noise amplifies the interpolation peak, but the original single-seed
   result is unstable. Here we use multiple seeds and report mean/std.

Outputs:
    results/zhengda_exp8_noise_lambda_mechanism/results.json
    results/zhengda_exp8_noise_lambda_mechanism/peak_heatmap.png
    results/zhengda_exp8_noise_lambda_mechanism/mechanism_curves.png
    results/zhengda_exp8_noise_lambda_mechanism/noise_peak_boxplot.png
    results/zhengda_exp8_noise_lambda_mechanism/peak_summary.csv
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make imports work when running as `python -m src.experiments...` from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Reuse Zhengda's original Exp5/RFF utilities from comprehensive_dd.py.
# comprehensive_dd.py contains:
#   random_fourier_features(...)
#   load_mnist_numpy(...)
#   min_norm_solution(...)
# We use random_fourier_features and load_mnist_numpy for consistency, but use an
# SVD-based solver below so lambda=0 is a stable true minimum-norm interpolator.
try:
    from src.experiments.comprehensive_dd import random_fourier_features, load_mnist_numpy
except Exception:
    # Fallback if this file is run directly from src/experiments.
    from comprehensive_dd import random_fourier_features, load_mnist_numpy


DEFAULT_RATIOS = [
    0.2, 0.5, 0.8, 0.9, 0.95, 0.98,
    1.0, 1.02, 1.05, 1.1, 1.2,
    1.5, 2.0, 3.0, 5.0, 8.0,
]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def svd_ridge_solution(
    U: np.ndarray,
    singular_values: np.ndarray,
    Vt: np.ndarray,
    Y: np.ndarray,
    lam: float,
    rcond: float = 1e-12,
) -> np.ndarray:
    """Return ridge/minimum-norm solution using precomputed compact SVD.

    Phi = U diag(s) Vt.

    Ridge solution:
        w_lam = V diag(s / (s^2 + lam)) U^T Y.

    For lam=0, this becomes the Moore-Penrose minimum-norm solution with
    small singular values truncated by rcond.
    """
    s = singular_values
    Uy = U.T @ Y

    if lam == 0.0:
        cutoff = rcond * max(s.shape[0], Vt.shape[1]) * s.max()
        factors = np.zeros_like(s)
        keep = s > cutoff
        factors[keep] = 1.0 / s[keep]
    else:
        factors = s / (s ** 2 + lam)

    return Vt.T @ (factors[:, None] * Uy)


def compute_spectral_metrics(singular_values: np.ndarray, lam: float) -> Dict[str, float]:
    """Compute mechanism metrics from singular values of Phi."""
    eigvals = singular_values ** 2
    eps = 1e-12
    lam_eff = lam if lam > 0 else eps

    reg_condition = float((eigvals.max() + lam_eff) / (eigvals.min() + lam_eff))
    effective_df = float(np.sum(eigvals / (eigvals + lam_eff)))
    min_singular = float(singular_values.min())
    max_singular = float(singular_values.max())

    return {
        "regularized_condition_number": reg_condition,
        "effective_degrees_of_freedom": effective_df,
        "min_singular_value": min_singular,
        "max_singular_value": max_singular,
    }


def evaluate_solution(
    Phi_train: np.ndarray,
    Phi_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    w: np.ndarray,
) -> Dict[str, float]:
    pred_train = Phi_train @ w
    pred_test = Phi_test @ w

    train_mse = float(np.mean((Y_train - pred_train) ** 2))
    test_mse = float(np.mean((Y_test - pred_test) ** 2))
    train_acc = float(np.mean(np.argmax(pred_train, axis=1) == y_train) * 100.0)
    test_acc = float(np.mean(np.argmax(pred_test, axis=1) == y_test) * 100.0)

    # Frobenius norm because w is D x 10 for one-vs-all multiclass regression.
    solution_norm = float(np.linalg.norm(w))

    return {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "solution_norm": solution_norm,
    }


def run_experiment(args: argparse.Namespace) -> Dict:
    os.makedirs(args.output_dir, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    noise_rates = parse_float_list(args.noise_rates)
    lambdas = parse_float_list(args.lambdas)
    ratios = parse_float_list(args.ratios) if args.ratios else DEFAULT_RATIOS

    all_rows: List[Dict] = []
    peak_rows: List[Dict] = []

    for seed in seeds:
        print(f"\n{'=' * 80}\nSeed = {seed}\n{'=' * 80}")

        for noise_rate in noise_rates:
            print(f"\n--- Loading MNIST: noise={noise_rate:.0%}, n_train={args.n_train} ---")
            X_tr, Y_tr, y_tr, X_te, Y_te, y_te = load_mnist_numpy(
                data_dir=args.data_dir,
                n_train=args.n_train,
                noise_rate=noise_rate,
                seed=seed,
            )

            # Store rows for peak selection for each lambda.
            rows_by_lambda: Dict[float, List[Dict]] = {lam: [] for lam in lambdas}

            for ratio in ratios:
                D = max(1, int(round(ratio * args.n_train)))
                print(f"\n  p/n={ratio:.2f}, D={D}")

                # Same RFF map for train and test, matching the original code style.
                Phi_tr = random_fourier_features(X_tr, D, sigma=args.sigma, seed=seed)
                Phi_te = random_fourier_features(X_te, D, sigma=args.sigma, seed=seed)

                # One SVD per (seed, noise, ratio), reused for all lambdas.
                U, s, Vt = np.linalg.svd(Phi_tr, full_matrices=False)

                for lam in lambdas:
                    w = svd_ridge_solution(U, s, Vt, Y_tr, lam=lam)
                    perf = evaluate_solution(Phi_tr, Phi_te, Y_tr, Y_te, y_tr, y_te, w)
                    spectral = compute_spectral_metrics(s, lam=lam)

                    row = {
                        "seed": seed,
                        "noise_rate": noise_rate,
                        "lambda": lam,
                        "p_over_n": ratio,
                        "D": D,
                        **perf,
                        **spectral,
                    }
                    all_rows.append(row)
                    rows_by_lambda[lam].append(row)

                    print(
                        f"    lambda={lam:.0e}: "
                        f"test_mse={perf['test_mse']:.4g}, "
                        f"test_acc={perf['test_acc']:.2f}%, "
                        f"||w||={perf['solution_norm']:.2e}, "
                        f"df={spectral['effective_degrees_of_freedom']:.1f}, "
                        f"cond_lam={spectral['regularized_condition_number']:.2e}"
                    )

            # Peak summary for each lambda under this seed/noise.
            for lam, lam_rows in rows_by_lambda.items():
                peak = max(lam_rows, key=lambda r: r["test_mse"])
                best_acc = max(lam_rows, key=lambda r: r["test_acc"])
                peak_rows.append({
                    "seed": seed,
                    "noise_rate": noise_rate,
                    "lambda": lam,
                    "peak_test_mse": peak["test_mse"],
                    "peak_p_over_n": peak["p_over_n"],
                    "peak_test_acc": peak["test_acc"],
                    "peak_solution_norm": peak["solution_norm"],
                    "peak_regularized_condition_number": peak["regularized_condition_number"],
                    "peak_effective_degrees_of_freedom": peak["effective_degrees_of_freedom"],
                    "best_test_acc": best_acc["test_acc"],
                    "best_acc_p_over_n": best_acc["p_over_n"],
                })

    aggregate_rows = aggregate_peaks(peak_rows, noise_rates, lambdas)

    payload = {
        "description": "Noise x ridge mechanism deep dive for Zhengda Exp5 and fixed Exp6.",
        "n_train": args.n_train,
        "sigma": args.sigma,
        "seeds": seeds,
        "noise_rates": noise_rates,
        "lambdas": lambdas,
        "ratios": ratios,
        "all_rows": all_rows,
        "peak_rows": peak_rows,
        "aggregate_peak_rows": aggregate_rows,
    }

    save_json(payload, os.path.join(args.output_dir, "results.json"))
    save_csv(all_rows, os.path.join(args.output_dir, "all_rows.csv"))
    save_csv(peak_rows, os.path.join(args.output_dir, "peak_summary.csv"))
    save_csv(aggregate_rows, os.path.join(args.output_dir, "aggregate_peak_summary.csv"))

    plot_peak_heatmap(aggregate_rows, noise_rates, lambdas, args.output_dir)
    plot_mechanism_curves(all_rows, args.representative_seed, args.representative_noise, args.output_dir)
    plot_noise_peak_boxplot(peak_rows, lambdas, args.output_dir)

    return payload


def aggregate_peaks(peak_rows: List[Dict], noise_rates: List[float], lambdas: List[float]) -> List[Dict]:
    aggregate_rows = []
    for noise_rate in noise_rates:
        for lam in lambdas:
            subset = [
                r for r in peak_rows
                if r["noise_rate"] == noise_rate and r["lambda"] == lam
            ]
            if not subset:
                continue

            def mean_std(key: str) -> Tuple[float, float]:
                values = np.array([r[key] for r in subset], dtype=float)
                return float(values.mean()), float(values.std(ddof=0))

            peak_mean, peak_std = mean_std("peak_test_mse")
            acc_mean, acc_std = mean_std("best_test_acc")
            norm_mean, norm_std = mean_std("peak_solution_norm")
            cond_mean, cond_std = mean_std("peak_regularized_condition_number")
            df_mean, df_std = mean_std("peak_effective_degrees_of_freedom")

            aggregate_rows.append({
                "noise_rate": noise_rate,
                "lambda": lam,
                "num_seeds": len(subset),
                "peak_test_mse_mean": peak_mean,
                "peak_test_mse_std": peak_std,
                "best_test_acc_mean": acc_mean,
                "best_test_acc_std": acc_std,
                "peak_solution_norm_mean": norm_mean,
                "peak_solution_norm_std": norm_std,
                "peak_regularized_condition_number_mean": cond_mean,
                "peak_regularized_condition_number_std": cond_std,
                "peak_effective_degrees_of_freedom_mean": df_mean,
                "peak_effective_degrees_of_freedom_std": df_std,
            })
    return aggregate_rows


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_peak_heatmap(
    aggregate_rows: List[Dict],
    noise_rates: List[float],
    lambdas: List[float],
    output_dir: str,
) -> None:
    matrix = np.full((len(noise_rates), len(lambdas)), np.nan)
    for i, noise_rate in enumerate(noise_rates):
        for j, lam in enumerate(lambdas):
            matches = [
                r for r in aggregate_rows
                if r["noise_rate"] == noise_rate and r["lambda"] == lam
            ]
            if matches:
                matrix[i, j] = matches[0]["peak_test_mse_mean"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(np.log10(matrix + 1e-12), aspect="auto")

    ax.set_xticks(np.arange(len(lambdas)))
    ax.set_xticklabels([f"{lam:.0e}" for lam in lambdas])
    ax.set_yticks(np.arange(len(noise_rates)))
    ax.set_yticklabels([f"{nr:.0%}" for nr in noise_rates])
    ax.set_xlabel("Ridge lambda")
    ax.set_ylabel("Label noise rate")
    ax.set_title("Noise × Ridge: log10(mean peak test MSE)")

    for i in range(len(noise_rates)):
        for j in range(len(lambdas)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2g}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label="log10(mean peak test MSE)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "peak_heatmap.png"), dpi=150)
    plt.close()


def plot_mechanism_curves(
    all_rows: List[Dict],
    representative_seed: int,
    representative_noise: float,
    output_dir: str,
) -> None:
    subset = [
        r for r in all_rows
        if r["seed"] == representative_seed and abs(r["noise_rate"] - representative_noise) < 1e-12
    ]
    if not subset:
        # Fallback to the first available seed/noise pair.
        first = all_rows[0]
        representative_seed = first["seed"]
        representative_noise = first["noise_rate"]
        subset = [
            r for r in all_rows
            if r["seed"] == representative_seed and abs(r["noise_rate"] - representative_noise) < 1e-12
        ]

    lambdas = sorted({r["lambda"] for r in subset})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    plot_specs = [
        ("test_mse", "Test MSE", True),
        ("solution_norm", "Solution norm ||W||", True),
        ("regularized_condition_number", "Regularized condition number", True),
        ("effective_degrees_of_freedom", "Effective degrees of freedom", False),
    ]

    for lam in lambdas:
        rows = sorted([r for r in subset if r["lambda"] == lam], key=lambda r: r["p_over_n"])
        x = [r["p_over_n"] for r in rows]
        label = f"lambda={lam:.0e}"
        for ax, (key, title, log_scale) in zip(axes, plot_specs):
            ax.plot(x, [r[key] for r in rows], "o-", markersize=3.5, label=label)
            ax.set_title(title)
            ax.set_xlabel("p/n")
            ax.axvline(1.0, color="gray", linestyle=":", alpha=0.7)
            ax.grid(True, alpha=0.3)
            if log_scale:
                ax.set_yscale("log")

    for ax in axes:
        ax.legend(fontsize=8)

    plt.suptitle(
        f"Ridge mechanism curves, seed={representative_seed}, noise={representative_noise:.0%}",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mechanism_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()


def plot_noise_peak_boxplot(peak_rows: List[Dict], lambdas: List[float], output_dir: str) -> None:
    # Use lambda=0 if present; otherwise use the smallest lambda.
    target_lambda = 0.0 if 0.0 in lambdas else min(lambdas)
    subset = [r for r in peak_rows if r["lambda"] == target_lambda]
    noise_rates = sorted({r["noise_rate"] for r in subset})
    data = [[r["peak_test_mse"] for r in subset if r["noise_rate"] == nr] for nr in noise_rates]

    if not data or all(len(x) == 0 for x in data):
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=[f"{nr:.0%}" for nr in noise_rates], showmeans=True)
    ax.set_yscale("log")
    ax.set_xlabel("Label noise rate")
    ax.set_ylabel("Peak test MSE")
    ax.set_title(f"Fixed Exp6: multi-seed peak distribution (lambda={target_lambda:.0e})")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "noise_peak_boxplot.png"), dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results/zhengda_exp8_noise_lambda_mechanism")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=5.0)

    # This is intentionally multi-seed, unlike the original buggy/single-seed Exp6.
    parser.add_argument("--seeds", type=str, default="42,123,456")

    # Jointly extend Exp5 and fixed Exp6.
    parser.add_argument("--noise-rates", type=str, default="0.0,0.1,0.2,0.4")
    parser.add_argument("--lambdas", type=str, default="0,1e-6,1e-4,1e-2,1e-1")
    parser.add_argument(
        "--ratios",
        type=str,
        default="",
        help="Comma-separated p/n ratios. Default uses a dense grid near 1.",
    )

    # Representative curve for mechanism_curves.png.
    parser.add_argument("--representative-seed", type=int, default=42)
    parser.add_argument("--representative-noise", type=float, default=0.1)

    args = parser.parse_args()
    run_experiment(args)
    print("\nDone.")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

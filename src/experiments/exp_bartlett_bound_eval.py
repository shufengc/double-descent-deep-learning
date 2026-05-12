"""
Bartlett 2020 effective-rank bound evaluation / diagnostic.

Purpose
-------
This script evaluates Bartlett-style effective-dimension quantities on the
trained fractional-k ResNet spectral summaries and compares them with observed
test risk.

Inputs
------
Default inputs are expected to already exist in the repository:

    results/nn_spectral/summary.json
    results/dd_recovery_5090_focused/main/summary.json

Outputs
-------
    results/bartlett_bound_eval/summary.json
    results/bartlett_bound_eval/summary.csv
    figures/bartlett_bound_eval_vs_observed.png

Usage
-----
    python -m src.experiments.exp_bartlett_bound_eval

Optional:
    python -m src.experiments.exp_bartlett_bound_eval \
        --spectral-summary results/nn_spectral/summary.json \
        --dd-summary results/dd_recovery_5090_focused/main/summary.json
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def mean_by_k(rows, key):
    buckets = {}
    for r in rows:
        k = float(r["k"])
        buckets.setdefault(k, []).append(float(r[key]))
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def get_dd_accuracy_by_k(dd_rows, metric_pref="final_test_acc"):
    """Aggregate per-k DD accuracy. Default prefers final_test_acc to avoid
    test-set selection bias from max-over-epochs (audit 2026-05-05).
    Pass metric_pref="best_test_acc" for legacy comparison."""
    out = {}
    for r in dd_rows:
        k = float(r["k"])
        # Honor caller's preference if available
        if metric_pref in r:
            acc = float(r[metric_pref])
        # Fallbacks
        elif "final_test_acc" in r:
            acc = float(r["final_test_acc"])
        elif "best_test_acc" in r:
            acc = float(r["best_test_acc"])
        elif "test_acc" in r:
            acc = float(r["test_acc"])
        else:
            continue
        out.setdefault(k, []).append(acc)
    return {k: sum(v) / len(v) for k, v in out.items()}


def spectral_record_to_metrics(r):
    """Extract Bartlett-style empirical effective dimensions from one record."""
    k = float(r["k"])
    n = int(r.get("n", 4000))
    params = int(r.get("params", 0))
    feat_dim = int(r["feat_dim"])
    test_acc = float(r.get("test_acc", float("nan")))

    # In exp_nn_spectral.py, eff_rank_stable = ||Z||_F^2 / ||Z||_op^2.
    # If Σ = Z_c^T Z_c (up to a constant), this is tr(Σ) / ||Σ||_op.
    bartlett_r = float(r["eff_rank_stable"])

    # Participation ratio is (Σλ_i)^2 / Σλ_i^2, a second effective dimension
    # analogous to Bartlett's R_k tail quantity, but without choosing a tail k.
    participation = float(r["participation_ratio"])

    # Normalized variants are useful because feat_dim grows with width k.
    r_frac = bartlett_r / max(1, feat_dim)
    pr_frac = participation / max(1, feat_dim)

    # Simple Bartlett-style complexity proxies. The unknown constants are not
    # identifiable from this experiment, so we compare shapes and calibrated
    # versions rather than claiming theorem-certified numbers.
    proxy_r = math.sqrt(max(bartlett_r, 0.0) / max(1, n))
    proxy_pr = math.sqrt(max(participation, 0.0) / max(1, n))
    proxy_r_frac = math.sqrt(max(r_frac, 0.0) / max(1, n))
    proxy_pr_frac = math.sqrt(max(pr_frac, 0.0) / max(1, n))

    return {
        "k": k,
        "n": n,
        "params": params,
        "feat_dim": feat_dim,
        "spectral_test_acc": test_acc,
        "bartlett_r": bartlett_r,
        "participation_ratio": participation,
        "r_frac": r_frac,
        "pr_frac": pr_frac,
        "proxy_sqrt_r_over_n": proxy_r,
        "proxy_sqrt_pr_over_n": proxy_pr,
        "proxy_sqrt_rfrac_over_n": proxy_r_frac,
        "proxy_sqrt_prfrac_over_n": proxy_pr_frac,
        "condition_number": float(r.get("condition_number", float("nan"))),
    }


def calibrate_proxy(rows, proxy_key, risk_key, reference_k=None):
    """Scale proxy by one constant so it can be compared to observed risk."""
    valid = [
        r for r in rows
        if math.isfinite(r[proxy_key]) and r[proxy_key] > 0
        and math.isfinite(r[risk_key]) and r[risk_key] > 0
    ]
    if not valid:
        return 1.0

    if reference_k is not None:
        candidates = [r for r in valid if abs(r["k"] - reference_k) < 1e-12]
        if candidates:
            r0 = candidates[0]
            return r0[risk_key] / max(r0[proxy_key], 1e-12)

    # Conservative shape comparison: choose least-squares scale.
    num = sum(r[proxy_key] * r[risk_key] for r in valid)
    den = sum(r[proxy_key] ** 2 for r in valid)
    return num / max(den, 1e-12)


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot(rows, output):
    rows = sorted(rows, key=lambda r: r["k"])
    ks = [r["k"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Panel A: observed risk vs calibrated proxies
    ax = axes[0, 0]
    ax.plot(ks, [r["observed_test_risk"] for r in rows], "o-", label="Observed test risk", linewidth=2)
    ax.plot(ks, [r["cal_bound_r"] for r in rows], "s--", label=r"Calibrated $\sqrt{r(\Sigma)/n}$ proxy")
    ax.plot(ks, [r["cal_bound_pr"] for r in rows], "^--", label=r"Calibrated $\sqrt{PR(\Sigma)/n}$ proxy")
    ax.set_xscale("log")
    ax.axvline(0.1875, linestyle=":", color="gray", alpha=0.8)
    ax.set_xlabel("Width multiplier k")
    ax.set_ylabel("Risk / calibrated proxy")
    ax.set_title("(a) Bartlett-style proxies vs observed risk")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # Panel B: effective dimensions
    ax = axes[0, 1]
    ax.plot(ks, [r["bartlett_r"] for r in rows], "o-", label=r"$r(\Sigma)$")
    ax.plot(ks, [r["participation_ratio"] for r in rows], "s-", label="Participation ratio")
    ax.set_xscale("log")
    ax.axvline(0.1875, linestyle=":", color="gray", alpha=0.8)
    ax.set_xlabel("Width multiplier k")
    ax.set_ylabel("Effective dimension")
    ax.set_title("(b) Bartlett effective dimensions")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # Panel C: normalized effective dimensions
    ax = axes[1, 0]
    ax.plot(ks, [r["r_frac"] for r in rows], "o-", label=r"$r(\Sigma)/c_3$")
    ax.plot(ks, [r["pr_frac"] for r in rows], "s-", label=r"PR$/c_3$")
    ax.set_xscale("log")
    ax.axvline(0.1875, linestyle=":", color="gray", alpha=0.8)
    ax.set_xlabel("Width multiplier k")
    ax.set_ylabel("Fraction of feature dimension")
    ax.set_title("(c) Normalized effective dimension")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # Panel D: vacuity ratio after calibration
    ax = axes[1, 1]
    ax.plot(ks, [r["vacuity_ratio_r"] for r in rows], "o-", label=r"proxy-r / observed")
    ax.plot(ks, [r["vacuity_ratio_pr"] for r in rows], "s-", label=r"proxy-PR / observed")
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1)
    ax.set_xscale("log")
    ax.axvline(0.1875, linestyle=":", color="gray", alpha=0.8)
    ax.set_xlabel("Width multiplier k")
    ax.set_ylabel("Calibrated proxy / observed risk")
    ax.set_title("(d) Informative if close to 1 and shape-tracking")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Bartlett 2020 Theorem-1-style effective-rank diagnostic\n"
        "Empirical evaluation on trained fractional-k ResNet spectra",
        y=1.02,
        fontsize=12,
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"wrote {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spectral-summary", default="results/nn_spectral/summary.json")
    p.add_argument("--dd-summary", default="results/dd_recovery_5090_focused/main/summary.json")
    p.add_argument("--out-dir", default="results/bartlett_bound_eval")
    p.add_argument("--figure", default="figures/bartlett_bound_eval_vs_observed.png")
    p.add_argument(
        "--calibrate-at-k",
        type=float,
        default=0.125,
        help="k used to scale proxy curves to observed risk. Use -1 for least-squares calibration.",
    )
    p.add_argument(
        "--metric",
        default="final_test_acc",
        choices=["final_test_acc", "best_test_acc"],
        help="Which DD metric to use as observed risk. Default: final_test_acc (post-audit 2026-05-05). Use best_test_acc for legacy.",
    )
    args = p.parse_args()

    spectral = load_json(args.spectral_summary)
    dd = load_json(args.dd_summary) if os.path.exists(args.dd_summary) else []

    rows = [spectral_record_to_metrics(r) for r in spectral]

    dd_acc = get_dd_accuracy_by_k(dd, metric_pref=args.metric)
    for r in rows:
        k = r["k"]
        # Prefer DD-Recovery best test acc if matching k exists; otherwise use spectral test acc.
        obs_acc = dd_acc.get(k, r["spectral_test_acc"])
        r["observed_test_acc"] = obs_acc
        r["observed_test_risk"] = 1.0 - obs_acc / 100.0

    reference_k = None if args.calibrate_at_k < 0 else args.calibrate_at_k
    scale_r = calibrate_proxy(rows, "proxy_sqrt_r_over_n", "observed_test_risk", reference_k)
    scale_pr = calibrate_proxy(rows, "proxy_sqrt_pr_over_n", "observed_test_risk", reference_k)

    for r in rows:
        r["cal_bound_r"] = scale_r * r["proxy_sqrt_r_over_n"]
        r["cal_bound_pr"] = scale_pr * r["proxy_sqrt_pr_over_n"]
        r["vacuity_ratio_r"] = r["cal_bound_r"] / max(r["observed_test_risk"], 1e-12)
        r["vacuity_ratio_pr"] = r["cal_bound_pr"] / max(r["observed_test_risk"], 1e-12)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "summary.json")
    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    write_csv(rows, csv_path)
    plot(rows, args.figure)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print("\nKey rows:")
    for r in sorted(rows, key=lambda x: x["k"]):
        print(
            f"k={r['k']:>7g}  risk={r['observed_test_risk']:.3f}  "
            f"r={r['bartlett_r']:.2f}  PR={r['participation_ratio']:.2f}  "
            f"cal_proxy_r={r['cal_bound_r']:.3f}  ratio={r['vacuity_ratio_r']:.2f}"
        )

    print(
        "\nInterpretation note: this is a Bartlett-style empirical diagnostic, "
        "not a theorem-certified neural-network bound. It tests whether the "
        "effective-rank quantities from Bartlett 2020 are informative on the "
        "trained DD-Recovery ResNet spectra."
    )


if __name__ == "__main__":
    main()

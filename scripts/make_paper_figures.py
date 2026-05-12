"""Generate paper-final figures from the A100 reproduction.

Color scheme:
  final = green   (the corrected metric)
  peek  = red     (legacy best_test_acc; "peek at test set")
  RFF   = blue
  ResNet = orange
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures" / "paper_final"
OUT.mkdir(parents=True, exist_ok=True)

GREEN = "#2ca02c"
RED = "#d62728"
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"


# ---------------------------------------------------------------------------
# Figure 1: Main DD valley with metric audit overlay (n=4000)
# ---------------------------------------------------------------------------
def fig_main_valley():
    """Best vs final test accuracy across the full fractional-k sweep at n=4000.
    Sources: yusheng_replicate_a100/{main,full/main}/k*_n4000_ep2000_s*/results.json
    """
    base = REPO / "results" / "yusheng_replicate_a100"
    rows = []
    for d in list((base / "main").glob("k*_n4000_ep2000_s*")) + list((base / "full" / "main").glob("k*_n4000_ep2000_s*")):
        try:
            r = json.load(open(d / "results.json"))
            cfg = r["config"]
            rows.append({
                "k": cfg["k"],
                "seed": cfg["seed"],
                "best": r["best_test_acc"],
                "final": r["final_test_acc"],
            })
        except Exception:
            pass

    by_k = defaultdict(lambda: {"best": [], "final": []})
    for r in rows:
        by_k[r["k"]]["best"].append(r["best"])
        by_k[r["k"]]["final"].append(r["final"])

    ks = sorted(by_k.keys())
    bests = [np.mean(by_k[k]["best"]) for k in ks]
    finals = [np.mean(by_k[k]["final"]) for k in ks]
    best_err = [np.std(by_k[k]["best"]) if len(by_k[k]["best"]) > 1 else 0 for k in ks]
    final_err = [np.std(by_k[k]["final"]) if len(by_k[k]["final"]) > 1 else 0 for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ks, finals, yerr=final_err, fmt="o-", color=GREEN, linewidth=2.2,
                markersize=8, label="final (corrected)", capsize=3)
    ax.errorbar(ks, bests, yerr=best_err, fmt="s--", color=RED, linewidth=1.8,
                markersize=6, alpha=0.85, label="best (legacy, max-over-epochs)", capsize=3)

    # Annotate the gap at valley
    if 0.5 in by_k or 0.6 in by_k:
        idx = next(i for i, k in enumerate(ks) if k in [0.5, 0.6])
        ax.annotate(f"gap = +{bests[idx]-finals[idx]:.1f}pp",
                    xy=(ks[idx], (bests[idx]+finals[idx])/2),
                    xytext=(ks[idx]*1.7, (bests[idx]+finals[idx])/2),
                    fontsize=10, color="black",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1))

    ax.set_xscale("log")
    ax.set_xlabel("Width multiplier k", fontsize=11)
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.set_title("Fractional-k ResNet on noisy CIFAR-10 (n=4000, 15% noise)\n"
                 "Metric audit: best-over-epochs hides the over-param valley",
                 fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_main_valley_metric_audit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'fig1_main_valley_metric_audit.png'}")


# ---------------------------------------------------------------------------
# Figure 2: Sample-wise second descent (n=8000 nslice)
# ---------------------------------------------------------------------------
def fig_sample_wise():
    """n=8000 second descent vs n=4000 baseline."""
    base = REPO / "results" / "yusheng_replicate_a100"

    def aggregate(glob_pat):
        rows = []
        for d in base.glob(glob_pat):
            try:
                r = json.load(open(d / "results.json"))
                cfg = r["config"]
                rows.append((cfg["k"], cfg["seed"], r["final_test_acc"]))
            except Exception:
                pass
        return rows

    n4_rows = aggregate("**/main/k*_n4000_ep2000_s*")
    n8_rows = aggregate("**/nslice/k*_n8000_ep1500_s*")

    def by_k(rows):
        d = defaultdict(list)
        for k, s, v in rows:
            d[k].append(v)
        return {k: (np.mean(vs), np.std(vs) if len(vs) > 1 else 0) for k, vs in d.items()}

    n4 = by_k(n4_rows)
    n8 = by_k(n8_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, dat, color in [("n = 4000", n4, ORANGE), ("n = 8000", n8, GREEN)]:
        ks = sorted(dat.keys())
        means = [dat[k][0] for k in ks]
        errs = [dat[k][1] for k in ks]
        ax.errorbar(ks, means, yerr=errs, fmt="o-", color=color, linewidth=2.2,
                    markersize=8, label=label, capsize=3)

    ax.set_xscale("log")
    ax.set_xlabel("Width multiplier k", fontsize=11)
    ax.set_ylabel("Final test accuracy (%)", fontsize=11)
    ax.set_title("Sample-wise second descent: n=8000 keeps climbing where n=4000 plateaus",
                 fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_sample_wise_second_descent.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'fig2_sample_wise_second_descent.png'}")


# ---------------------------------------------------------------------------
# Figure 3: Bartlett vacuity ratio (final vs best calibration)
# ---------------------------------------------------------------------------
def fig_bartlett_vacuity():
    base = REPO / "results"

    def load(p):
        rows = json.load(open(p))
        ks = [float(r["k"]) for r in rows]
        ratios = [float(r["vacuity_ratio_r"]) for r in rows]
        return ks, ratios

    ks_f, r_f = load(base / "bartlett_bound_eval_final" / "summary.json")
    ks_b, r_b = load(base / "bartlett_bound_eval_best" / "summary.json")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks_f, r_f, "o-", color=GREEN, linewidth=2.2, markersize=8,
            label="final_test_acc (post-audit)")
    ax.plot(ks_b, r_b, "s--", color=RED, linewidth=1.8, markersize=6, alpha=0.85,
            label="best_test_acc (legacy)")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(min(ks_f), 1.05, "tight bound (ratio=1)", fontsize=8, color="gray")

    ax.set_xscale("log")
    ax.set_xlabel("Width multiplier k", fontsize=11)
    ax.set_ylabel("Vacuity ratio (calibrated bound / observed risk)", fontsize=11)
    ax.set_title("Bartlett (2020) effective-rank diagnostic on trained DD-ResNet\n"
                 "Vacuity rises monotonically; metric choice shifts absolute scale ~10-15%",
                 fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_bartlett_vacuity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'fig3_bartlett_vacuity.png'}")


# ---------------------------------------------------------------------------
# Figure 4: Bias-variance decomposition at p/n=1 (RFF, MNIST)
# ---------------------------------------------------------------------------
def fig_bias_variance():
    p = REPO / "results" / "shufeng_exps_a100" / "expB_bias_variance" / "results.json"
    d = json.load(open(p))

    # Structure: per-noise -> list of records {D, p_over_n, bias_sq, variance, total_mse}
    fig, ax = plt.subplots(figsize=(8, 5))
    # The inner key structure varies; just plot the "total" curves and one bias/var pair near p/n=1
    # First flatten:
    series = []
    if isinstance(d, dict):
        for noise_key, recs in d.items():
            if not isinstance(recs, list):
                continue
            xs = [r["D"] / r["n"] if "D" in r and "n" in r else r.get("p_over_n") for r in recs]
            bias = [r.get("bias_sq", np.nan) for r in recs]
            var = [r.get("variance", np.nan) for r in recs]
            total = [r.get("total_mse", r.get("test_mse", np.nan)) for r in recs]
            series.append((str(noise_key), xs, bias, var, total))

    if not series:
        print("  [warn] bias-variance JSON unexpected shape; skipping")
        return

    # Plot total error for each noise level + bias and variance for one representative
    # Use first noise level as representative
    name, xs, bias, var, total = series[0]
    ax.plot(xs, var, "o-", color=RED, linewidth=2.2, label="variance")
    ax.plot(xs, bias, "s-", color=ORANGE, linewidth=2.0, label="bias²")
    ax.plot(xs, total, "^-", color=GREEN, linewidth=2.4, label="total MSE")
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.text(1.02, 0.5, "p/n = 1", fontsize=8, color="gray", transform=ax.get_xaxis_transform())

    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xscale("log")
    ax.set_xlabel("p/n", fontsize=11)
    ax.set_ylabel("MSE component", fontsize=11)
    ax.set_title(f"Bias-variance decomposition (RFF on MNIST, {name})\n"
                 "DD peak is pure variance: bias² flat, variance spikes at p/n=1",
                 fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_bias_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'fig4_bias_variance.png'}")


# ---------------------------------------------------------------------------
# Figure 5: Optimizer collapse at high p/n (Exp 7)
# ---------------------------------------------------------------------------
def fig_optimizer_collapse():
    p = REPO / "results" / "shufeng_exps_a100" / "expC_epoch_sgd_resnet" / "results.json"
    if not p.exists():
        print(f"  [warn] {p} not found, skipping")
        return
    d = json.load(open(p))

    # Structure: d[opt_name][k_str] = {"label", "num_params", "p_over_n", "history"}
    # history is a dict of column arrays: {"epoch": [...], "test_acc": [...], ...}
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for opt_name, opt_d in d.items():
        for k_str, run in opt_d.items():
            hist = run.get("history", {})
            if not hist or "epoch" not in hist:
                continue
            xs = hist["epoch"]
            test_accs = hist.get("test_acc", [])
            ys = [100 - a for a in test_accs]  # convert to test error
            color = ORANGE if opt_name.lower() == "sgd" else GREEN
            linestyle = "-" if k_str == "1" else "--" if k_str == "2" else ":"
            p_over_n = run.get("p_over_n")
            label = f"{opt_name.upper()} k={k_str} (p/n={p_over_n:.0f})" \
                    if isinstance(p_over_n, (int, float)) \
                    else f"{opt_name.upper()} k={k_str}"
            ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1.6, label=label)
            plotted = True

    if plotted:
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Test error (%)", fontsize=11)
        ax.set_title("Optimizer collapse at high p/n (n=4000, 20% noise, 4000 epochs)\n"
                     "Both SGD and Adam memorize and collapse — optimizer is secondary",
                     fontsize=11)
        ax.set_ylim(40, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right", ncol=2)
        fig.tight_layout()
        fig.savefig(OUT / "fig5_optimizer_collapse.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {OUT / 'fig5_optimizer_collapse.png'}")
    else:
        print("  [warn] expC JSON empty histories; skipping")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Generating paper-final figures in {OUT}")
    fig_main_valley()
    fig_sample_wise()
    fig_bartlett_vacuity()
    fig_bias_variance()
    fig_optimizer_collapse()
    print(f"\nDone. {len(list(OUT.glob('*.png')))} PNGs in {OUT}")


if __name__ == "__main__":
    main()

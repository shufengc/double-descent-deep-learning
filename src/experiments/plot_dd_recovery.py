"""
Plotting and analysis script for dd_recovery_5090_focused results.
Generates 3 figures:
  1. dd_curve_main.png       - model-wise DD: test error vs params (log-x), 2-seed band
  2. dd_curve_nslice.png     - n=4000 vs n=8000 comparison, peak shift
  3. dd_mechanism_panel.png  - memorization fraction + training curve vs k
"""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_BASE = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "dd_recovery_5090_focused"
)
FIGS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "dd_recovery_5090_focused", "figures"
)
os.makedirs(FIGS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_phase(phase):
    """Return list of result dicts from a phase directory."""
    pattern = os.path.join(RESULTS_BASE, phase, "*/results.json")
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            r = json.load(f)
        rows.append(r)
    return rows


def aggregate_seeds(rows, n_filter=None):
    """
    Given rows from the main phase (multiple seeds), aggregate by k:
    returns dict k -> {params, train_mean, train_std, best_test_mean, best_test_std,
                       final_test_mean, final_test_std, seeds}
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        k = r["config"]["k"]
        n = r["config"]["n_train"]
        if n_filter is not None and n != n_filter:
            continue
        buckets[k].append(r)
    agg = {}
    for k, rs in sorted(buckets.items()):
        params = rs[0]["params"]
        trains = [r["final_train_acc"] for r in rs]
        best_tests = [r["best_test_acc"] for r in rs]
        final_tests = [r["final_test_acc"] for r in rs]
        agg[k] = {
            "params": params,
            "train_mean": np.mean(trains),
            "train_std": np.std(trains),
            "best_test_mean": np.mean(best_tests),
            "best_test_std": np.std(best_tests),
            "final_test_mean": np.mean(final_tests),
            "final_test_std": np.std(final_tests),
            "seeds": [r["config"]["seed"] for r in rs],
            "rows": rs,
        }
    return agg


# ---------------------------------------------------------------------------
# Figure 1: Model-wise DD curve, 2-seed confidence band (main, n=4000)
# ---------------------------------------------------------------------------

def plot_dd_curve_main():
    rows = load_phase("main")
    agg = aggregate_seeds(rows, n_filter=4000)

    k_vals = sorted(agg.keys())
    params = [agg[k]["params"] for k in k_vals]
    train_mean = np.array([agg[k]["train_mean"] for k in k_vals])
    test_mean  = np.array([agg[k]["best_test_mean"] for k in k_vals])
    test_std   = np.array([agg[k]["best_test_std"] for k in k_vals])

    # Convert accuracy to error
    train_err = 100 - train_mean
    test_err  = 100 - test_mean
    test_err_lo = 100 - (test_mean + test_std)
    test_err_hi = 100 - (test_mean - test_std)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogx(params, train_err, "b--o", linewidth=1.8, markersize=6, label="Train error")
    ax.semilogx(params, test_err,  "r-o",  linewidth=2.0, markersize=6, label="Test error (best, mean of 2 seeds)")
    ax.fill_between(params, test_err_lo, test_err_hi, color="red", alpha=0.15, label="Test error ±1 std (2 seeds)")

    # Annotate interpolation threshold region
    # Find k where train_err first drops below ~10%  (train acc > 90%)
    thresh_idxs = [i for i, te in enumerate(100 - train_mean) if te < 30]
    if thresh_idxs:
        thresh_p = params[thresh_idxs[0]]
        ax.axvline(thresh_p, color="gray", linestyle=":", linewidth=1.5, label=f"Interpolation threshold (p≈{thresh_p:,})")

    ax.set_xlabel("Number of parameters (log scale)")
    ax.set_ylabel("Error rate (%)")
    ax.set_title("Model-Wise Double Descent — ResNet-k on CIFAR-10\n(n=4,000, 15% label noise, Adam lr=1e-4, 2,000 epochs)")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # k labels on top
    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(params)
    ax2.set_xticklabels([str(k) for k in k_vals], fontsize=8, rotation=45)
    ax2.set_xlabel("k (width multiplier)", fontsize=10)

    plt.tight_layout()
    out = os.path.join(FIGS_DIR, "dd_curve_main.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return agg


# ---------------------------------------------------------------------------
# Figure 2: N-slice comparison (n=4000 vs n=8000)
# ---------------------------------------------------------------------------

def plot_nslice_comparison(agg_main):
    rows_nslice = load_phase("nslice")
    agg_nslice  = aggregate_seeds(rows_nslice, n_filter=8000)

    # Shared k values
    shared_k = sorted(set(agg_main.keys()) & set(agg_nslice.keys()))

    params_4k = [agg_main[k]["params"]           for k in shared_k]
    test_4k   = [agg_main[k]["best_test_mean"]   for k in shared_k]
    params_8k = [agg_nslice[k]["params"]          for k in shared_k]
    test_8k   = [agg_nslice[k]["best_test_mean"]  for k in shared_k]

    # Also include full main range for n=4000
    k_all = sorted(agg_main.keys())
    params_all_4k = [agg_main[k]["params"] for k in k_all]
    test_all_4k   = [agg_main[k]["best_test_mean"] for k in k_all]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(params_all_4k, test_all_4k, "r-o", linewidth=2, markersize=6,
                label="n=4,000 (2-seed avg)")
    ax.semilogx(params_8k, test_8k, "b-s", linewidth=2, markersize=7,
                label="n=8,000 (seed=42)")

    ax.set_xlabel("Number of parameters (log scale)")
    ax.set_ylabel("Best test accuracy (%)")
    ax.set_title("N-Slice: Peak Shifts Right with More Training Data\nResNet-k, CIFAR-10, 15% label noise")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    out = os.path.join(FIGS_DIR, "dd_curve_nslice.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3: Mechanism panel — memorization fraction & training dynamics
# ---------------------------------------------------------------------------

def plot_mechanism_panel(agg_main):
    """
    For a subset of k values (0.125, 0.25, 0.5, 1.0, 2.0) from seed=42:
    - Left: train accuracy vs epoch
    - Middle: test accuracy vs epoch
    - Right: memorization fraction (final epoch) vs k
    """
    k_subset = [0.125, 0.25, 0.5, 1.0, 2.0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(k_subset)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Collect per-epoch histories for seed=42
    rows_main = load_phase("main")
    histories = {}
    for r in rows_main:
        k = r["config"]["k"]
        seed = r["config"]["seed"]
        if seed == 42 and k in k_subset:
            histories[k] = r["history"]

    for i, k in enumerate(k_subset):
        if k not in histories:
            continue
        h = histories[k]
        epochs = [e["epoch"] for e in h]
        train_acc = [e["train_acc"] for e in h]
        test_acc  = [e.get("test_acc") for e in h]
        valid_ep  = [ep for ep, ta in zip(epochs, test_acc) if ta is not None]
        valid_ta  = [ta for ta in test_acc if ta is not None]
        c = colors[i]
        label = f"k={k} ({agg_main[k]['params']:,} params)"
        axes[0].plot(epochs, train_acc, color=c, linewidth=1.2, label=label, alpha=0.85)
        if valid_ta:
            axes[1].plot(valid_ep, valid_ta, color=c, linewidth=1.2, label=label, alpha=0.85)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train accuracy (%)")
    axes[0].set_title("Training Accuracy Dynamics\n(seed=42, n=4,000)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test accuracy (%)")
    axes[1].set_title("Test Accuracy Dynamics\n(seed=42, n=4,000)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # Right panel: memorization fraction at final epoch vs k
    k_vals = sorted(agg_main.keys())
    mem_fracs = []
    for k in k_vals:
        frac_list = []
        for r in agg_main[k]["rows"]:
            h = r["history"]
            last = h[-1]
            mf = last.get("memorization_frac")
            if mf is not None:
                frac_list.append(mf)
        mem_fracs.append(np.mean(frac_list) if frac_list else np.nan)

    params_list = [agg_main[k]["params"] for k in k_vals]
    test_means  = [agg_main[k]["best_test_mean"] for k in k_vals]

    ax3 = axes[2]
    ax3b = ax3.twinx()
    p1, = ax3.semilogx(params_list, [m * 100 for m in mem_fracs], "g-o",
                        linewidth=2, markersize=6, label="Memorization fraction (%)")
    p2, = ax3b.semilogx(params_list, test_means, "r--^",
                         linewidth=2, markersize=6, label="Best test accuracy (%)")
    ax3.set_xlabel("Number of parameters")
    ax3.set_ylabel("Memorization fraction (%)", color="green")
    ax3b.set_ylabel("Best test accuracy (%)", color="red")
    ax3.set_title("Memorization vs Generalization\n(mean of 2 seeds)")
    ax3.tick_params(axis="y", colors="green")
    ax3b.tick_params(axis="y", colors="red")
    lines = [p1, p2]
    ax3.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="lower right")
    ax3.grid(alpha=0.3)
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    out = os.path.join(FIGS_DIR, "dd_mechanism_panel.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_summary(agg):
    print("\n=== Main Results (n=4000, 2-seed average) ===")
    print(f"{'k':>8} {'params':>8} {'train%':>8} {'best_te%':>9} {'fin_te%':>9} {'n_seeds':>8}")
    for k, d in sorted(agg.items()):
        print(f"{k:8.4f} {d['params']:8,d} {d['train_mean']:8.2f} "
              f"{d['best_test_mean']:9.2f}±{d['best_test_std']:.1f} "
              f"{d['final_test_mean']:9.2f} {len(d['seeds']):8d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating DD recovery figures...")
    agg_main = plot_dd_curve_main()
    print_summary(agg_main)
    plot_nslice_comparison(agg_main)
    plot_mechanism_panel(agg_main)
    print("\nAll figures saved to:", FIGS_DIR)

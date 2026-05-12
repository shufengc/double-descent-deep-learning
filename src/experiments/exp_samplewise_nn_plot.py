"""
Plot sample-wise NN double descent: test accuracy vs k (width multiplier)
for n ∈ {1000, 2000, 4000, 8000}, showing the peak shifts right as n grows.

Data sources:
  results/samplewise_nn/summary.json          -- n=1000, 2000
  results/dd_recovery_5090_focused/main/summary.json   -- n=4000, 2 seeds
  results/dd_recovery_5090_focused/nslice/summary.json -- n=8000, 1 seed

Writes:
  figures/samplewise_nn_dd.png
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCES = [
    ("results/samplewise_nn/summary.json",
     "results/samplewise_nn"),
    ("results/dd_recovery_5090_focused/main/summary.json",
     "results/dd_recovery_5090_focused/main"),
    ("results/dd_recovery_5090_focused/nslice/summary.json",
     "results/dd_recovery_5090_focused/nslice"),
]

N_COLORS = {
    1000: "#1f77b4",   # blue
    2000: "#ff7f0e",   # orange
    4000: "#2ca02c",   # green
    8000: "#d62728",   # red
}
N_LABELS = {
    1000: "n = 1 000",
    2000: "n = 2 000",
    4000: "n = 4 000",
    8000: "n = 8 000",
}


def load_all(base_dir, sources):
    records = []
    for rel_path, _ in sources:
        full = os.path.join(base_dir, rel_path)
        if os.path.exists(full):
            with open(full) as f:
                records.extend(json.load(f))
        else:
            print(f"  [warn] not found, skipping: {full}")
    return records


def aggregate(records, metric="best_test_acc"):
    """Group by (n, k), average across seeds, return sorted lists."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["n"], r["k"])].append(r[metric])
    result = {}
    for (n, k), vals in groups.items():
        result.setdefault(n, []).append((k, np.mean(vals), np.std(vals) if len(vals) > 1 else 0.0))
    for n in result:
        result[n].sort(key=lambda x: x[0])
    return result


def plot(agg, output):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ns_sorted = sorted(agg.keys())
    for n in ns_sorted:
        pts = agg[n]
        ks = [p[0] for p in pts]
        means = [p[1] for p in pts]
        stds = [p[2] for p in pts]
        color = N_COLORS.get(n, "gray")
        label = N_LABELS.get(n, f"n = {n}")
        ax.plot(ks, means, "-o", color=color, label=label,
                markersize=6, linewidth=2)
        ax.fill_between(ks,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.15)

    # Annotate the interpolation threshold for each n.
    # For each n, find the k that maximizes test acc — this is the DD peak.
    for n in ns_sorted:
        pts = agg[n]
        if len(pts) < 3:
            continue
        peak_idx = int(np.argmax([p[1] for p in pts]))
        peak_k = pts[peak_idx][0]
        peak_acc = pts[peak_idx][1]
        color = N_COLORS.get(n, "gray")
        ax.axvline(x=peak_k, color=color, linestyle=":", linewidth=0.8, alpha=0.6)
        ax.annotate(f"k*≈{peak_k:g}",
                    xy=(peak_k, peak_acc),
                    xytext=(peak_k + 0.03, peak_acc - 1.5),
                    fontsize=7.5, color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.7))

    ax.set_xlabel("Width multiplier k  (number of params ∝ k²)", fontsize=11)
    ax.set_ylabel("Best test accuracy (%) over training", fontsize=11)
    ax.set_title(
        "Sample-wise NN double descent:\npeak shifts right as n grows  "
        "(CIFAR-10, 15% label noise, fractional-k ResNet)",
        fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default=".")
    p.add_argument("--output", default="figures/samplewise_nn_dd.png")
    p.add_argument("--metric", default="best_test_acc",
                   choices=["best_test_acc", "final_test_acc"])
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    records = load_all(args.base_dir, SOURCES)
    print(f"Loaded {len(records)} run records from {len(SOURCES)} source files.")
    ns = sorted(set(r["n"] for r in records))
    print(f"  n values: {ns}")
    for n in ns:
        sub = [r for r in records if r["n"] == n]
        ks = sorted(set(r["k"] for r in sub))
        print(f"  n={n}: {len(sub)} records, k={ks}")

    agg = aggregate(records, metric=args.metric)
    plot(agg, args.output)


if __name__ == "__main__":
    main()

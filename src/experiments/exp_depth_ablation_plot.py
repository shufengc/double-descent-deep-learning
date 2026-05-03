"""
Plot depth-axis ablation for fractional-k ResNet at k=0.5 (N4 / report §6.12).

Reads:
  results/depth_ablation/summary.json     (depth ∈ {2, 4} × 2 seeds)
  results/dd_recovery_5090_focused/main/summary.json  (depth=3 baseline at k=0.5)

Writes figures/depth_ablation.png — bar/line plot showing best test acc vs depth.
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_baseline_k05(main_summary_path):
    """Pull k=0.5, n=4000 records from main summary as the depth=3 baseline."""
    with open(main_summary_path) as f:
        d = json.load(f)
    return [r for r in d if r["k"] == 0.5 and r["n"] == 4000]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth-summary", default="results/depth_ablation/summary.json")
    p.add_argument("--main-summary",
                   default="results/dd_recovery_5090_focused/main/summary.json")
    p.add_argument("--output", default="figures/depth_ablation.png")
    args = p.parse_args()

    rows = []
    if os.path.exists(args.depth_summary):
        with open(args.depth_summary) as f:
            depth_data = json.load(f)
        for r in depth_data:
            rows.append((int(r["n_stages"]), float(r["best_test_acc"]),
                         int(r["seed"]), int(r["params"])))
    # Add depth=3 baseline (k=0.5 from main)
    if os.path.exists(args.main_summary):
        for r in load_baseline_k05(args.main_summary):
            rows.append((3, float(r["best_test_acc"]), int(r["seed"]),
                         int(r["params"])))

    if not rows:
        raise SystemExit("no data — run exp_depth_ablation first")

    by_d = defaultdict(list)
    for d, acc, _, prm in rows:
        by_d[d].append((acc, prm))
    depths = sorted(by_d.keys())
    means = [np.mean([t[0] for t in by_d[d]]) for d in depths]
    stds = [np.std([t[0] for t in by_d[d]]) for d in depths]
    params = [int(np.mean([t[1] for t in by_d[d]])) for d in depths]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(depths, means, yerr=stds, capsize=8, width=0.5,
                  color="#1f77b4", edgecolor="black", linewidth=1.2,
                  alpha=0.85)
    for d, m, prm in zip(depths, means, params):
        ax.text(d, m + 1.2, f"{m:.1f}%\n({prm:,} p)",
                ha="center", fontsize=10)
    ax.set_xticks(depths)
    ax.set_xlabel("ResNet depth (number of stages)", fontsize=11)
    ax.set_ylabel("Best test accuracy (%) at $k=0.5$", fontsize=11)
    ax.set_title(
        "Depth-axis ablation — fractional-$k$ ResNet at $k=0.5$\n"
        "$n=4{,}000$, 15% noise, 1500 epochs (depth=3 from §5.3 baseline)",
        fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(min(means) - 5, max(means) + 8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {args.output}")
    print(f"data: {[(d, round(m, 2), round(s, 2)) for d, m, s in zip(depths, means, stds)]}")


if __name__ == "__main__":
    main()

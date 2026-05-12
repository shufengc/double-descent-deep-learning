"""
Plot tight full empirical-NTK diagnostics versus k for fractional-k ResNet
(N5 / report §6.10).

Reads results/full_empirical_ntk/summary.json (the tight 9 k × 800 ep × 32
NTK sample sweep).

Marks the DD-recovery onset at k=0.1875.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="results/full_empirical_ntk/summary.json")
    p.add_argument("--output", default="figures/full_empirical_ntk_tight.png")
    p.add_argument("--peak-k", type=float, default=0.1875)
    args = p.parse_args()

    with open(args.summary) as f:
        records = json.load(f)
    records.sort(key=lambda r: float(r["k"]))
    ks = [float(r["k"]) for r in records]
    cond = [r["condition_number"] for r in records]
    sr = [r["stable_rank"] for r in records]
    pr = [r["participation_ratio"] for r in records]
    test_acc = [r["test_acc"] for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))

    ax = axes[0]
    ax.plot(ks, cond, "o-", color="#d62728", markersize=7, linewidth=2)
    ax.axvline(args.peak_k, color="red", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Width multiplier $k$", fontsize=11)
    ax.set_ylabel("Condition number  $\\sigma_{\\max}/\\sigma_{\\min}$",
                  fontsize=11)
    ax.set_title("(a) Full empirical-NTK condition number", fontsize=11)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(ks, sr, "o-", color="#1f77b4", markersize=7, linewidth=2)
    ax.axvline(args.peak_k, color="red", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Width multiplier $k$", fontsize=11)
    ax.set_ylabel("Stable rank  $\\mathrm{tr}(K)/\\|K\\|_\\mathrm{op}$",
                  fontsize=11)
    ax.set_title("(b) Stable rank of NTK Gram", fontsize=11)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    ax.plot(ks, pr, "o-", color="#2ca02c", markersize=7, linewidth=2)
    ax.axvline(args.peak_k, color="red", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Width multiplier $k$", fontsize=11)
    ax.set_ylabel("Participation ratio", fontsize=11)
    ax.set_title("(c) Participation ratio of NTK Gram", fontsize=11)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Tight full empirical-NTK — fractional-$k$ ResNet, $n=2{,}000$, "
        "800 ep, 32 NTK samples (9 widths)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

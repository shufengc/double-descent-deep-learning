"""
Plot literal ResNet18 controlled comparison vs fractional-k ResNet (N3 / §5.3.1).

Reads:
  results/resnet18_compare/summary.json
  results/dd_recovery_5090_focused/main/summary.json

Writes figures/resnet18_vs_fractionalk.png.
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
    p.add_argument("--rn18-summary",
                   default="results/resnet18_compare/summary.json")
    p.add_argument("--main-summary",
                   default="results/dd_recovery_5090_focused/main/summary.json")
    p.add_argument("--output", default="figures/resnet18_vs_fractionalk.png")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Fractional-k from main summary at n=4000
    with open(args.main_summary) as f:
        main_d = json.load(f)
    fk = sorted([r for r in main_d if r["n"] == 4000], key=lambda r: r["params"])
    fk_p = [r["params"] for r in fk]
    fk_acc = [r["best_test_acc"] for r in fk]
    ax.plot(fk_p, fk_acc, "o-", color="#1f77b4", markersize=8, linewidth=2.2,
            label=f"fractional-$k$ ResNet (3-stage), $k\\!\\in\\![0.0625,2]$")

    # ResNet18 controlled
    if os.path.exists(args.rn18_summary):
        with open(args.rn18_summary) as f:
            rn18 = json.load(f)
        rn18 = sorted(rn18, key=lambda r: r["params"])
        rn18_p = [r["params"] for r in rn18]
        rn18_acc = [r["best_test_acc"] for r in rn18]
        rn18_w = [r["width_mult"] for r in rn18]
        ax.plot(rn18_p, rn18_acc, "s-", color="#d62728", markersize=10, linewidth=2.2,
                label="literal ResNet18 (4-stage), width $\\in\\{0.5,1,2\\}$")
        for p_, a, w in zip(rn18_p, rn18_acc, rn18_w):
            ax.annotate(f"$\\times{w:g}$\n{a:.1f}%", xy=(p_, a),
                        xytext=(8, -22), textcoords="offset points",
                        fontsize=9.5, color="#d62728")

    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters (log scale)", fontsize=11)
    ax.set_ylabel("Best test accuracy (%)", fontsize=11)
    ax.set_title(
        "Literal ResNet18 vs fractional-$k$ ResNet — controlled comparison\n"
        "$n=4{,}000$, 15% noise, Adam lr=$10^{-4}$, 2000 epochs (Person C settings)",
        fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

"""
Plot Hessian top eigenvalue vs k for fractional-k ResNet (N2 / §6.13).

Reads results/hessian_topeig/summary.json. Writes figures/hessian_topeig_vs_k.png.
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
    p.add_argument("--summary", default="results/hessian_topeig/summary.json")
    p.add_argument("--output", default="figures/hessian_topeig_vs_k.png")
    p.add_argument("--peak-k", type=float, default=0.1875)
    args = p.parse_args()

    with open(args.summary) as f:
        records = json.load(f)
    records.sort(key=lambda r: float(r["k"]))

    ks = [float(r["k"]) for r in records]
    eigs = [float(r["top_hessian_eig"]) for r in records]
    test_acc = [float(r["test_acc"]) for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    ax.plot(ks, eigs, "o-", color="#9467bd", markersize=8, linewidth=2.2)
    ax.axvline(args.peak_k, color="red", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Width multiplier $k$", fontsize=11)
    ax.set_ylabel(r"Top Hessian eigenvalue $\lambda_{\max}(\nabla^2 \mathcal{L})$",
                  fontsize=11)
    ax.set_title("(a) Sharpness vs $k$ (log-log)", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    for k_, e_ in zip(ks, eigs):
        ax.annotate(f"{e_:.2e}", xy=(k_, e_), xytext=(6, 4),
                    textcoords="offset points", fontsize=8)

    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(ks, test_acc, "o-", color="#2ca02c", markersize=8, linewidth=2,
            label="test acc (left)")
    ax2.plot(ks, eigs, "s--", color="#9467bd", markersize=7, linewidth=1.5,
             alpha=0.7, label="$\\lambda_{\\max}$ (right, log)")
    ax2.set_yscale("log")
    ax.axvline(args.peak_k, color="red", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Width multiplier $k$", fontsize=11)
    ax.set_ylabel("Test accuracy (%)", fontsize=11, color="#2ca02c")
    ax2.set_ylabel("Top Hessian eigenvalue", fontsize=11, color="#9467bd")
    ax.set_title("(b) Sharpness aligns with test-acc transition", fontsize=11)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Hessian top eigenvalue vs $k$ — fractional-$k$ ResNet "
        "(Origin: Yao et al. 2020 PyHessian; Foret et al. 2021 SAM)",
        fontsize=11, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

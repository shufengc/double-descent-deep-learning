"""
Plot NN spectral diagnostics versus k for fractional-k ResNet at n=4000.

Three panels (figure: nn_effective_rank_vs_k.png):
  (a) normalized stable rank  eff_rank / c_3       — fraction of feature dim spanned
  (b) condition number        sigma_max / sigma_min — log-scale
  (c) normalized participation ratio  PR / c_3      — Renyi-2 effective dimension

Reads results/nn_spectral/summary.json (penultimate-feature spectrum from the
spectral sweep). The penultimate-feature dimension c_3 = max(1, round(64k))
varies with k; normalising by c_3 puts diagnostics on a comparable scale and
exposes the local extremum near the DD-recovery onset.
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)


def aggregate(records, key):
    by_k = defaultdict(list)
    for r in records:
        if r.get(key) is not None:
            by_k[r["k"]].append(r[key])
    ks = sorted(by_k.keys())
    means = [float(np.mean(by_k[k])) for k in ks]
    return ks, means


def make_three_panel(spec, output, peak_k=0.1875):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))

    # Per-k feature dim (= c_3 = max(1, round(64k)))
    by_k = defaultdict(list)
    for r in spec:
        by_k[r["k"]].append((r["eff_rank_stable"], r["condition_number"],
                             r["participation_ratio"], r["feat_dim"],
                             r["test_acc"]))
    ks = sorted(by_k.keys())
    eff = [float(np.mean([t[0] for t in by_k[k]])) for k in ks]
    cond = [float(np.mean([t[1] for t in by_k[k]])) for k in ks]
    pr = [float(np.mean([t[2] for t in by_k[k]])) for k in ks]
    fdim = [int(np.mean([t[3] for t in by_k[k]])) for k in ks]
    ta = [float(np.mean([t[4] for t in by_k[k]])) for k in ks]

    frac_eff = [e / d for e, d in zip(eff, fdim)]
    frac_pr = [p / d for p, d in zip(pr, fdim)]

    # Panel (a): normalized stable rank
    ax = axes[0]
    ax.plot(ks, frac_eff, "o-", color="#1f77b4", markersize=7, linewidth=2,
            label="$\\mathrm{eff\\_rank}/c_3$")
    ax.axvline(peak_k, color="red", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(peak_k * 1.07, max(frac_eff) * 0.95,
            f"DD recovery\n$k\\!=\\!{peak_k:g}$",
            fontsize=8.5, color="red", va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Width multiplier $k$", fontsize=11)
    ax.set_ylabel("Fraction of feature dim spanned",
                  fontsize=11)
    ax.set_title("(a) Normalised stable rank", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="best")

    # Panel (b): condition number
    ax = axes[1]
    ax.plot(ks, cond, "o-", color="#d62728", markersize=7, linewidth=2,
            label="$\\sigma_{\\max}/\\sigma_{\\min}$")
    ax.axvline(peak_k, color="red", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Width multiplier $k$", fontsize=11)
    ax.set_ylabel("Condition number", fontsize=11)
    ax.set_title("(b) Condition number (log-log)", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="best")

    # Panel (c): normalized participation ratio
    ax = axes[2]
    ax.plot(ks, frac_pr, "o-", color="#2ca02c", markersize=7, linewidth=2,
            label="$\\mathrm{PR}/c_3$")
    ax.axvline(peak_k, color="red", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Width multiplier $k$", fontsize=11)
    ax.set_ylabel("Renyi-2 effective dim / $c_3$",
                  fontsize=11)
    ax.set_title("(c) Normalised participation ratio", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        "NN spectral signature — fractional-$k$ ResNet, $n=4{,}000$, "
        "penultimate features (test, $N=2{,}048$)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default=".")
    p.add_argument("--output",
                   default="figures/nn_effective_rank_vs_k.png")
    p.add_argument("--peak-k", type=float, default=0.1875)
    args = p.parse_args()

    spec_path = os.path.join(args.base_dir, "results/nn_spectral/summary.json")
    spec = load(spec_path)
    print(f"spectral records: {len(spec)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    make_three_panel(spec, args.output, peak_k=args.peak_k)


if __name__ == "__main__":
    main()

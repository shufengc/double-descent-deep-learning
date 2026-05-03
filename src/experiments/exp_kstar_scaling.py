"""
k*(n) EMC-scaling fit (N6 / report §6.9 addendum).

Origin: Nakkiran, Kaplun, Bansal, Yang, Barak, Sutskever (2021), "Deep Double
Descent" §3 — EMC definition. Belkin/Hastie kernel-DD theory predicts
p* ∝ √n in the proportional limit; for our fractional-k ResNet that translates
to k* ∝ √n (since params ~ k²).

Reads three summary files (n=1k+2k from samplewise_nn, n=4k from main, n=8k
from nslice). Computes k*(n) = arg-max of best_test_acc(k) at each n. Fits
log k* = α log n + β. Reports slope α (theory: 0.5).
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCES = [
    "results/samplewise_nn/summary.json",
    "results/dd_recovery_5090_focused/main/summary.json",
    "results/dd_recovery_5090_focused/nslice/summary.json",
]


def load_all(base_dir):
    records = []
    for s in SOURCES:
        p = Path(base_dir) / s
        if p.exists():
            with open(p) as f:
                records.extend(json.load(f))
    return records


def k_star_per_n(records):
    """k*(n) = the k that maximises mean best_test_acc at each n."""
    by_nk = defaultdict(list)
    for r in records:
        by_nk[(r["n"], r["k"])].append(r["best_test_acc"])
    by_n = defaultdict(list)
    for (n, k), vs in by_nk.items():
        by_n[n].append((k, float(np.mean(vs))))
    out = {}
    for n, lst in by_n.items():
        lst.sort(key=lambda x: x[0])
        # k* = argmax over k
        idx = int(np.argmax([t[1] for t in lst]))
        out[n] = {"k_star": lst[idx][0], "best": lst[idx][1],
                  "n_points": len(lst), "all": lst}
    return out


def fit_loglog(ns, ks):
    log_n = np.log(np.array(ns))
    log_k = np.log(np.array(ks))
    A = np.vstack([log_n, np.ones_like(log_n)]).T
    coef, _, _, _ = np.linalg.lstsq(A, log_k, rcond=None)
    return float(coef[0]), float(coef[1])


def plot(by_n, alpha, beta, output):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ns = sorted(by_n.keys())
    ks = [by_n[n]["k_star"] for n in ns]

    ax.plot(ns, ks, "o", markersize=12, color="#1f77b4",
            label="empirical $k^*(n)$")
    nn = np.linspace(min(ns)*0.85, max(ns)*1.15, 100)
    kk_fit = np.exp(alpha * np.log(nn) + beta)
    ax.plot(nn, kk_fit, "--", color="#1f77b4", alpha=0.7,
            label=f"fit: $\\log k^* = {alpha:.3f}\\log n + {beta:.3f}$")

    # Theory prediction k* ∝ n^0.5 (kernel proportional limit)
    # anchor at n=4000 to match the headline
    k_anchor = by_n[4000]["k_star"] if 4000 in by_n else ks[len(ks)//2]
    n_anchor = 4000 if 4000 in by_n else ns[len(ns)//2]
    kk_theory = k_anchor * np.sqrt(nn / n_anchor)
    ax.plot(nn, kk_theory, ":", color="#d62728", alpha=0.8,
            label=r"theory: $k^* \propto \sqrt{n}$ (slope 0.5)")

    for n in ns:
        ax.annotate(f"n={n}\nk*={by_n[n]['k_star']:g}",
                    xy=(n, by_n[n]["k_star"]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=9, alpha=0.85)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training set size $n$", fontsize=11)
    ax.set_ylabel("DD-recovery onset $k^*(n)$", fontsize=11)
    ax.set_title(r"$k^*(n)$ scaling fit on the fractional-$k$ ResNet"
                 + "\n" + r"(Origin: Nakkiran et al. 2021 §3 EMC; theory $k^* \propto \sqrt{n}$)",
                 fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default=".")
    p.add_argument("--output", default="figures/k_star_vs_n_scaling.png")
    p.add_argument("--summary-out", default="results/kstar_scaling/summary.json")
    args = p.parse_args()

    records = load_all(args.base_dir)
    print(f"loaded {len(records)} records from {len(SOURCES)} sources")

    by_n = k_star_per_n(records)
    ns = sorted(by_n.keys())
    print("k*(n) per n:")
    for n in ns:
        d = by_n[n]
        print(f"  n={n:>5d}  k*={d['k_star']:>7g}  best_acc={d['best']:5.2f}"
              f"  ({d['n_points']} k-points)")

    if len(ns) >= 2:
        alpha, beta = fit_loglog(ns, [by_n[n]["k_star"] for n in ns])
        print(f"fit: log k* = {alpha:.3f} log n + {beta:.3f}")
        print(f"empirical slope α = {alpha:.3f}  (theory 0.5)")
    else:
        alpha, beta = 0.5, 0.0

    os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
    with open(args.summary_out, "w") as f:
        json.dump({
            "k_star_per_n": {str(n): by_n[n] for n in ns},
            "fit_alpha": alpha, "fit_beta": beta,
            "theory_slope": 0.5,
        }, f, indent=2, default=float)
    plot(by_n, alpha, beta, args.output)


if __name__ == "__main__":
    main()

"""
Bartlett benign-overfitting effective rank rk(Σ) = tr(Σ) / ||Σ||_op
on penultimate-feature covariance of the trained fractional-k ResNet
(N1 / report §6.10).

Origin: Bartlett, Long, Lugosi, Tsigler (2020). "Benign overfitting in linear
regression." PNAS 117(48):30063-30070. Theorem 1 — rk(Σ) and R_k(Σ) are the
two effective-dimension quantities that control the generalisation bound for
minimum-norm interpolation. We compute the empirical analogue rk(Σ̂) on the
penultimate-feature covariance of the trained NN.

This is *exactly* the quantity Bartlett et al. use in their bound, computed
for the first time on a trained DD-recovery ResNet's last-layer features.
4th independent witness for the spectral phase transition at k=0.1875.

Reads:
  results/nn_spectral/k*/spectrum.json   (full singular values already saved)

Writes:
  figures/bartlett_eff_rank_vs_k.png
  results/bartlett_eff_rank/summary.json
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


def load_spectrum_data(spectral_dir):
    records = []
    p = Path(spectral_dir)
    for run_dir in sorted(p.iterdir()):
        if not run_dir.is_dir():
            continue
        spec_path = run_dir / "spectrum.json"
        if not spec_path.exists():
            continue
        with open(spec_path) as f:
            r = json.load(f)
        records.append(r)
    return records


def bartlett_rk(singular_values):
    """rk(Σ) = tr(Σ) / ||Σ||_op  where Σ is computed from these singular values
    (i.e. λ_i = σ_i^2 are the eigenvalues of Σ)."""
    s = np.array([v for v in singular_values if v > 0])
    lam = s ** 2
    if len(lam) == 0:
        return 0.0
    return float(lam.sum() / lam.max())


def Rk(singular_values, k_idx=1):
    """Bartlett's R_k(Σ) = (Σ_{i>k} λ_i)² / Σ_{i>k} λ_i² — second effective-
    dimension quantity. We use k_idx=1 (largest direction excluded)."""
    s = np.array([v for v in singular_values if v > 0])
    lam = s ** 2
    if len(lam) <= k_idx:
        return 0.0
    tail = lam[k_idx:]
    num = tail.sum() ** 2
    den = (tail ** 2).sum()
    return float(num / max(den, 1e-30))


def aggregate(records):
    by_k = defaultdict(list)
    for r in records:
        sv = r["full_singular_values"]
        by_k[r["k"]].append({
            "rk_sigma": bartlett_rk(sv),
            "rk_normalized": bartlett_rk(sv) / max(1, r["feat_dim"]),
            "Rk_sigma": Rk(sv, k_idx=1),
            "feat_dim": r["feat_dim"],
            "test_acc": r["test_acc"],
        })
    rows = []
    for k in sorted(by_k.keys()):
        ms = by_k[k]
        rows.append({
            "k": k,
            "feat_dim": ms[0]["feat_dim"],
            "rk_sigma": float(np.mean([m["rk_sigma"] for m in ms])),
            "rk_normalized": float(np.mean([m["rk_normalized"] for m in ms])),
            "Rk_sigma": float(np.mean([m["Rk_sigma"] for m in ms])),
            "test_acc": float(np.mean([m["test_acc"] for m in ms])),
        })
    return rows


def plot(rows, output, peak_k=0.1875):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ks = [r["k"] for r in rows]

    ax = axes[0]
    ax.plot(ks, [r["rk_sigma"] for r in rows], "o-",
            color="#1f77b4", markersize=7, linewidth=2,
            label=r"$r_k(\Sigma)$")
    ax.set_xscale("log")
    ax.axvline(peak_k, color="red", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Width multiplier $k$", fontsize=11)
    ax.set_ylabel(r"Bartlett effective rank $r_k(\Sigma)=\mathrm{tr}(\Sigma)/\|\Sigma\|_\mathrm{op}$",
                  fontsize=11)
    ax.set_title("(a) Bartlett 2020 effective rank", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(ks, [r["rk_normalized"] for r in rows], "s-",
            color="#9467bd", markersize=7, linewidth=2,
            label=r"$r_k(\Sigma) / c_3$")
    ax.set_xscale("log")
    ax.axvline(peak_k, color="red", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Width multiplier $k$", fontsize=11)
    ax.set_ylabel("Normalised effective rank", fontsize=11)
    ax.set_title("(b) Normalised by feature dim $c_3$", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    fig.suptitle(
        r"Bartlett benign-overfitting diagnostic on penultimate features"
        + "\n" + r"(Origin: Bartlett, Long, Lugosi, Tsigler 2020 — Thm 1)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spectral-dir", default="results/nn_spectral")
    p.add_argument("--output", default="figures/bartlett_eff_rank_vs_k.png")
    p.add_argument("--summary-out", default="results/bartlett_eff_rank/summary.json")
    p.add_argument("--peak-k", type=float, default=0.1875)
    args = p.parse_args()

    records = load_spectrum_data(args.spectral_dir)
    print(f"loaded {len(records)} spectrum records")
    rows = aggregate(records)
    print("Bartlett rk(Σ) / R_k(Σ) by k:")
    for r in rows:
        print(f"  k={r['k']:>7g}  feat_dim={r['feat_dim']:>3d}"
              f"  rk={r['rk_sigma']:7.3f}  rk/c3={r['rk_normalized']:6.3f}"
              f"  Rk={r['Rk_sigma']:7.3f}  test_acc={r['test_acc']:5.2f}")

    os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
    with open(args.summary_out, "w") as f:
        json.dump(rows, f, indent=2)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plot(rows, args.output, peak_k=args.peak_k)


if __name__ == "__main__":
    main()

"""A1 — Gap-as-Sixth-Witness analysis.

Computes gap(k) = best_test_acc(k) − final_test_acc(k) across all our
n=4000 reproductions, overlays it with the 5 existing spectral witnesses,
and reports Pearson correlations.

This is GPU-free; reads only existing JSONs in results/.

Outputs:
  results/gap_witness/correlations.csv
  results/gap_witness/per_k_table.csv
  figures/paper_final/fig6_gap_as_sixth_witness.png
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent

# Color scheme per team agreement
COLORS = {
    "gap": "#9467bd",          # purple — the new witness
    "bartlett_r": "#2ca02c",    # green
    "ntk_kappa": "#d62728",     # red
    "hessian_per_p": "#ff7f0e", # orange
    "stable_rank": "#1f77b4",   # blue
    "participation": "#8c564b", # brown
}

LABELS = {
    "gap": "Gap = best − final (NEW)",
    "bartlett_r": "Bartlett r_k(Σ)",
    "ntk_kappa": "NTK condition κ",
    "hessian_per_p": "Hessian λ_max / p",
    "stable_rank": "Penultimate stable rank",
    "participation": "Participation ratio",
}


# ---------------------------------------------------------------------------
# Step 1: gap(k) from yusheng_replicate_a100 + initial verify runs
# ---------------------------------------------------------------------------
def compute_gap_by_k() -> dict[float, dict]:
    """Return {k: {"best_mean", "final_mean", "gap_mean", "gap_std", "n_seeds"}}
    for n=4000 across all reproduction sources."""
    base = REPO / "results" / "yusheng_replicate_a100"
    rows = []
    for d in list(base.glob("main/k*_n4000_ep2000_s*")) + list(base.glob("full/main/k*_n4000_ep2000_s*")):
        try:
            r = json.load(open(d / "results.json"))
            cfg = r["config"]
            if cfg.get("n_train") != 4000:
                continue
            rows.append({
                "k": float(cfg["k"]),
                "seed": int(cfg["seed"]),
                "best": float(r["best_test_acc"]),
                "final": float(r["final_test_acc"]),
            })
        except Exception as e:
            print(f"  [warn] {d}: {e}")

    by_k = defaultdict(list)
    for r in rows:
        by_k[r["k"]].append(r)

    out = {}
    for k, recs in by_k.items():
        bests = np.array([r["best"] for r in recs])
        finals = np.array([r["final"] for r in recs])
        gaps = bests - finals
        out[k] = {
            "best_mean": float(bests.mean()),
            "final_mean": float(finals.mean()),
            "gap_mean": float(gaps.mean()),
            "gap_std": float(gaps.std(ddof=0)) if len(gaps) > 1 else 0.0,
            "n_seeds": len(recs),
        }
    return out


# ---------------------------------------------------------------------------
# Step 2: load 5 spectral witnesses, indexed by k
# ---------------------------------------------------------------------------
def load_witnesses() -> dict[str, dict[float, float]]:
    """Returns {witness_name: {k: value}} for each of the 5 existing witnesses.

    Witnesses come from existing pipelines:
      - bartlett_r       <- results/bartlett_bound_eval_final/summary.json (bartlett_r)
      - ntk_kappa        <- results/full_empirical_ntk_quick/summary.json (condition_number)
      - hessian_per_p    <- results/hessian_topeig/summary.json (top_hessian_eig / params)
      - stable_rank      <- results/nn_spectral/summary.json (eff_rank_stable)
      - participation    <- results/bartlett_bound_eval_final/summary.json (participation_ratio)
    """
    out = {w: {} for w in COLORS if w != "gap"}

    # bartlett_r + participation
    p = REPO / "results" / "bartlett_bound_eval_final" / "summary.json"
    if p.exists():
        for r in json.load(open(p)):
            k = float(r["k"])
            out["bartlett_r"][k] = float(r["bartlett_r"])
            out["participation"][k] = float(r["participation_ratio"])

    # ntk_kappa
    p = REPO / "results" / "full_empirical_ntk_quick" / "summary.json"
    if p.exists():
        for r in json.load(open(p)):
            k = float(r["k"])
            out["ntk_kappa"][k] = float(r["condition_number"])

    # hessian per param
    p = REPO / "results" / "hessian_topeig" / "summary.json"
    if p.exists():
        for r in json.load(open(p)):
            k = float(r["k"])
            params = float(r["params"])
            out["hessian_per_p"][k] = float(r["top_hessian_eig"]) / params if params > 0 else float("nan")

    # stable rank
    p = REPO / "results" / "nn_spectral" / "summary.json"
    if p.exists():
        for r in json.load(open(p)):
            k = float(r["k"])
            out["stable_rank"][k] = float(r["eff_rank_stable"])

    return out


# ---------------------------------------------------------------------------
# Step 3: correlation analysis
# ---------------------------------------------------------------------------
def pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r and a rough p-value approximation. Returns (r, p_approx)."""
    if len(x) < 3:
        return float("nan"), float("nan")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx = x.mean()
    my = y.mean()
    sx = x.std(ddof=0)
    sy = y.std(ddof=0)
    if sx == 0 or sy == 0:
        return 0.0, 1.0
    r = float(((x - mx) * (y - my)).mean() / (sx * sy))
    # rough p approximation (two-tailed) via Fisher transform
    n = len(x)
    if abs(r) >= 1:
        p = 0.0
    else:
        z = 0.5 * np.log((1 + r) / (1 - r))
        # standard error: 1/sqrt(n-3)
        se = 1.0 / np.sqrt(max(n - 3, 1))
        # 2-tail p
        from math import erf, sqrt
        zscore = abs(z) / se
        p = float(2 * (1 - 0.5 * (1 + erf(zscore / sqrt(2)))))
    return r, p


def correlations(gap_by_k: dict[float, dict],
                 witnesses: dict[str, dict[float, float]]) -> dict[str, dict]:
    """For each witness, compute Pearson(gap, witness) on the shared k support."""
    out = {}
    gap_dict = {k: v["gap_mean"] for k, v in gap_by_k.items()}
    for w_name, w_dict in witnesses.items():
        shared_k = sorted(set(gap_dict.keys()) & set(w_dict.keys()))
        if len(shared_k) < 3:
            out[w_name] = {"r": float("nan"), "p": float("nan"), "n": len(shared_k)}
            continue
        gaps = np.array([gap_dict[k] for k in shared_k])
        ws = np.array([w_dict[k] for k in shared_k])
        r, p = pearson(gaps, ws)
        out[w_name] = {"r": r, "p": p, "n": len(shared_k), "shared_k": shared_k}
    return out


# ---------------------------------------------------------------------------
# Step 4: write CSVs and plot
# ---------------------------------------------------------------------------
def write_per_k_table(gap_by_k, witnesses, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_ks = sorted(set(gap_by_k.keys()) | {k for w in witnesses.values() for k in w})
    fields = ["k", "best_mean", "final_mean", "gap_mean", "gap_std", "n_seeds",
              "bartlett_r", "ntk_kappa", "hessian_per_p", "stable_rank", "participation"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in all_ks:
            row = {"k": k}
            if k in gap_by_k:
                g = gap_by_k[k]
                row.update({
                    "best_mean": f"{g['best_mean']:.3f}",
                    "final_mean": f"{g['final_mean']:.3f}",
                    "gap_mean": f"{g['gap_mean']:.3f}",
                    "gap_std": f"{g['gap_std']:.3f}",
                    "n_seeds": g["n_seeds"],
                })
            for wn, wd in witnesses.items():
                if k in wd:
                    row[wn] = f"{wd[k]:.4f}"
            w.writerow(row)
    print(f"  wrote {out_path}")


def write_correlations_csv(corrs, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["witness", "pearson_r", "p_value_approx", "n_shared_k", "shared_k"])
        for wn, c in corrs.items():
            shared_k = c.get("shared_k", [])
            w.writerow([
                wn,
                f"{c['r']:.4f}" if c["r"] == c["r"] else "nan",
                f"{c['p']:.4f}" if c["p"] == c["p"] else "nan",
                c["n"],
                "|".join(f"{k:g}" for k in shared_k),
            ])
    print(f"  wrote {out_path}")


def plot_panel(gap_by_k, witnesses, corrs, out_path):
    """One panel: gap(k) overlaid with 5 witnesses, all rescaled to [0,1]."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def rescale(d: dict[float, float]) -> dict[float, float]:
        if not d:
            return d
        vs = np.array(list(d.values()), dtype=float)
        vmin, vmax = vs.min(), vs.max()
        if vmax - vmin < 1e-12:
            return {k: 0.5 for k in d}
        return {k: float((v - vmin) / (vmax - vmin)) for k, v in d.items()}

    gap_rescaled = rescale({k: v["gap_mean"] for k, v in gap_by_k.items()})
    rescaled = {w: rescale(d) for w, d in witnesses.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [2, 1]})

    # Left: rescaled overlay
    ks = sorted(gap_rescaled.keys())
    ax1.plot(ks, [gap_rescaled[k] for k in ks], "o-", color=COLORS["gap"],
             linewidth=3.5, markersize=11, label=LABELS["gap"], zorder=5)
    for w_name, d in rescaled.items():
        if not d:
            continue
        wks = sorted(d.keys())
        ax1.plot(wks, [d[k] for k in wks], "s--", color=COLORS[w_name],
                 linewidth=1.6, markersize=6, alpha=0.8,
                 label=LABELS[w_name])

    ax1.set_xscale("log")
    ax1.set_xlabel("Width multiplier k", fontsize=11)
    ax1.set_ylabel("Rescaled magnitude (each witness to [0,1])", fontsize=11)
    ax1.set_title("Gap-as-Sixth-Witness: gap(k) co-locates with the five existing\n"
                  "spectral witnesses around the recovery onset",
                  fontsize=11)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(fontsize=8, loc="upper center", ncol=2)

    # Right: bar chart of Pearson r
    names = ["bartlett_r", "ntk_kappa", "hessian_per_p", "stable_rank", "participation"]
    rs = [corrs[n]["r"] for n in names]
    colors = [COLORS[n] for n in names]
    bars = ax2.barh([LABELS[n] for n in names], rs, color=colors, edgecolor="black")
    for bar, r in zip(bars, rs):
        ax2.text(bar.get_width() + 0.02 * np.sign(bar.get_width()),
                 bar.get_y() + bar.get_height() / 2,
                 f"r={r:.2f}", va="center",
                 ha="left" if r >= 0 else "right",
                 fontsize=9)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_xlabel("Pearson r (with gap(k))", fontsize=11)
    ax2.set_title("Correlation: gap(k) vs each witness", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
def main():
    print("Computing gap(k) from reproductions...")
    gap_by_k = compute_gap_by_k()
    for k in sorted(gap_by_k.keys()):
        v = gap_by_k[k]
        print(f"  k={k:>7g}  best={v['best_mean']:6.2f}  final={v['final_mean']:6.2f}  "
              f"gap={v['gap_mean']:+6.2f}  n_seeds={v['n_seeds']}")

    print("\nLoading spectral witnesses...")
    witnesses = load_witnesses()
    for w_name, d in witnesses.items():
        print(f"  {w_name}: {len(d)} k-values")

    print("\nComputing correlations...")
    corrs = correlations(gap_by_k, witnesses)
    for wn, c in corrs.items():
        print(f"  gap vs {wn:<18}  r={c['r']:+.3f}  p={c['p']:.4f}  (n={c['n']})")

    print("\nWriting outputs...")
    out_dir = REPO / "results" / "gap_witness"
    write_per_k_table(gap_by_k, witnesses, out_dir / "per_k_table.csv")
    write_correlations_csv(corrs, out_dir / "correlations.csv")
    plot_panel(gap_by_k, witnesses, corrs,
               REPO / "figures" / "paper_final" / "fig6_gap_as_sixth_witness.png")

    print("\n=== Summary ===")
    print(f"Gap range across {len(gap_by_k)} k-values: "
          f"{min(v['gap_mean'] for v in gap_by_k.values()):+.2f} to "
          f"{max(v['gap_mean'] for v in gap_by_k.values()):+.2f}")
    print(f"Strongest correlation: ", end="")
    best = max(corrs.items(), key=lambda x: abs(x[1]["r"]) if x[1]["r"] == x[1]["r"] else 0)
    print(f"{best[0]} (r={best[1]['r']:+.3f})")


if __name__ == "__main__":
    main()

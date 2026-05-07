"""A3 — Weight-Decay Sweep at the Valley analysis.

Reads results/wd_sweep_valley/ JSONs and produces:
  - results/wd_sweep_valley/summary.csv  (aggregated table)
  - figures/paper_final/fig7_wd_sweep_valley.png  (test-acc vs wd, gap vs wd)
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
RESULTS_DIR = REPO / "results" / "wd_sweep_valley"
FIG_PATH = REPO / "figures" / "paper_final" / "fig7_wd_sweep_valley.png"


def load_records():
    rows = []
    for d in sorted(RESULTS_DIR.glob("k*_wd*_s*")):
        rp = d / "results.json"
        if not rp.exists():
            continue
        r = json.loads(rp.read_text())
        cfg = r["config"]
        rows.append({
            "wd": float(cfg["weight_decay"]),
            "seed": int(cfg["seed"]),
            "best": float(r["best_test_acc"]),
            "final": float(r["final_test_acc"]),
            "gap": float(r["best_test_acc"] - r["final_test_acc"]),
            "wallclock_sec": float(r["wallclock_sec"]),
        })
    return rows


def aggregate(rows):
    by_wd = defaultdict(list)
    for r in rows:
        by_wd[r["wd"]].append(r)
    out = {}
    for wd, recs in by_wd.items():
        bs = np.array([r["best"] for r in recs])
        fs = np.array([r["final"] for r in recs])
        gs = np.array([r["gap"] for r in recs])
        out[wd] = {
            "n_seeds": len(recs),
            "best_mean": float(bs.mean()),
            "best_std": float(bs.std(ddof=0)) if len(bs) > 1 else 0.0,
            "final_mean": float(fs.mean()),
            "final_std": float(fs.std(ddof=0)) if len(fs) > 1 else 0.0,
            "gap_mean": float(gs.mean()),
            "gap_std": float(gs.std(ddof=0)) if len(gs) > 1 else 0.0,
        }
    return out


def write_summary_csv(agg, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["wd", "n_seeds", "best_mean", "best_std",
              "final_mean", "final_std", "gap_mean", "gap_std"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for wd in sorted(agg.keys()):
            v = agg[wd]
            w.writerow({
                "wd": f"{wd:.0e}" if wd > 0 else "0",
                "n_seeds": v["n_seeds"],
                "best_mean": f"{v['best_mean']:.3f}",
                "best_std": f"{v['best_std']:.3f}",
                "final_mean": f"{v['final_mean']:.3f}",
                "final_std": f"{v['final_std']:.3f}",
                "gap_mean": f"{v['gap_mean']:.3f}",
                "gap_std": f"{v['gap_std']:.3f}",
            })
    print(f"  wrote {out_path}")


def plot_wd_sweep(agg, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wds = sorted(agg.keys())
    # x-axis: use small positive offset for log scale of wd=0 (use 1e-7 as proxy)
    wds_for_plot = [w if w > 0 else 1e-7 for w in wds]
    bests = [agg[w]["best_mean"] for w in wds]
    bests_err = [agg[w]["best_std"] for w in wds]
    finals = [agg[w]["final_mean"] for w in wds]
    finals_err = [agg[w]["final_std"] for w in wds]
    gaps = [agg[w]["gap_mean"] for w in wds]
    gaps_err = [agg[w]["gap_std"] for w in wds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: best vs final test acc as a function of wd
    ax1.errorbar(wds_for_plot, bests, yerr=bests_err, fmt="s--",
                 color="#d62728", linewidth=1.6, markersize=7,
                 label="best (max-over-epochs)", capsize=4, alpha=0.85)
    ax1.errorbar(wds_for_plot, finals, yerr=finals_err, fmt="o-",
                 color="#2ca02c", linewidth=2.2, markersize=8,
                 label="final (last epoch)", capsize=4)
    ax1.set_xscale("log")
    ax1.set_xlabel("Weight decay  (wd=0 plotted at 1e-7)", fontsize=11)
    ax1.set_ylabel("Test accuracy (%)", fontsize=11)
    ax1.set_title("WD sweep at k=0.5 (valley floor)\n"
                  "n=4000, 15% noise, 2000 epochs, 2 seeds each",
                  fontsize=11)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(fontsize=10, loc="lower right")

    # Right panel: gap vs wd
    ax2.errorbar(wds_for_plot, gaps, yerr=gaps_err, fmt="o-",
                 color="#9467bd", linewidth=2.2, markersize=8,
                 label="gap = best − final", capsize=4)
    ax2.set_xscale("log")
    ax2.set_xlabel("Weight decay  (wd=0 plotted at 1e-7)", fontsize=11)
    ax2.set_ylabel("Gap (pp)", fontsize=11)
    ax2.set_title("Gap as a function of wd at k=0.5\n"
                  "Modest smoothing at wd=1e-2; within seed variance",
                  fontsize=11)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax2.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    print("Loading WD sweep records...")
    rows = load_records()
    print(f"  {len(rows)} records loaded")

    print("\nAggregating by wd...")
    agg = aggregate(rows)
    print(f"  {len(agg)} wd values")

    print("\nWriting outputs...")
    write_summary_csv(agg, RESULTS_DIR / "summary.csv")
    plot_wd_sweep(agg, FIG_PATH)

    print("\n=== Aggregated table ===")
    print(f"{'wd':>10} {'n':>3} {'best':>10} {'final':>11} {'gap':>9} {'gap_std':>8}")
    print('-' * 60)
    for wd in sorted(agg.keys()):
        v = agg[wd]
        wd_str = f"{wd:.0e}" if wd > 0 else "0"
        print(f"{wd_str:>10} {v['n_seeds']:>3} {v['best_mean']:>10.2f} "
              f"{v['final_mean']:>11.2f} {v['gap_mean']:>+9.2f} {v['gap_std']:>8.2f}")

    print("\n=== Headline finding ===")
    final0 = agg[0.0]["final_mean"]
    final_max = agg[1e-2]["final_mean"]
    gap0 = agg[0.0]["gap_mean"]
    gap_max = agg[1e-2]["gap_mean"]
    print(f"final_test_acc improvement at wd=1e-2 vs wd=0: "
          f"{final_max:.2f} - {final0:.2f} = +{final_max-final0:.2f}pp")
    print(f"gap reduction at wd=1e-2 vs wd=0: "
          f"{gap0:.2f} - {gap_max:.2f} = {gap0-gap_max:+.2f}pp "
          f"({100*(gap0-gap_max)/gap0:.0f}% reduction)")
    print(f"BUT gap_std at wd=1e-2 is {agg[1e-2]['gap_std']:.2f}pp — "
          f"effect within seed variance on 2 seeds.")


if __name__ == "__main__":
    main()

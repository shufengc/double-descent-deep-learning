"""
This script re-aggregates ALL existing summary.json files in results/,
extracting both `best_test_acc` and `final_test_acc` side-by-side. It does
not retrain anything and does not modify any input file.

Outputs:
  results/metric_audit/comparison.csv      — full per-(scope, n, k, seed) table
  results/metric_audit/summary_by_nk.csv   — averaged across seeds
  figures/best_vs_final_comparison.png     — overlay plot, panel per n

Usage:
  python scripts/audit_metrics.py              # all scopes
  python scripts/audit_metrics.py --scope main # just main DD sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Map each summary file to a "scope" label for grouping output.
SCOPE_SOURCES: Dict[str, List[str]] = {
    "main": [
        "results/dd_recovery_5090_focused/main/summary.json",
    ],
    "nslice": [
        "results/dd_recovery_5090_focused/nslice/summary.json",
    ],
    "samplewise": [
        "results/samplewise_nn/summary.json",
    ],
    "depth": [
        "results/depth_ablation/summary.json",
    ],
}


def load_records(scope: str, sources: Iterable[str], base_dir: str) -> List[dict]:
    """Load all records from a list of summary.json paths, tagging with scope."""
    out: List[dict] = []
    for rel in sources:
        full = os.path.join(base_dir, rel)
        if not os.path.exists(full):
            print(f"  [warn] missing: {full}", file=sys.stderr)
            continue
        with open(full) as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"  [warn] non-list summary: {full}", file=sys.stderr)
            continue
        for rec in data:
            rec = dict(rec)
            rec["_scope"] = scope
            rec["_source"] = rel
            out.append(rec)
    return out


def has_both_metrics(rec: dict) -> bool:
    return "best_test_acc" in rec and "final_test_acc" in rec


def write_per_seed_csv(records: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "scope",
        "source",
        "n",
        "k",
        "seed",
        "params",
        "best_test_acc",
        "final_test_acc",
        "gap_best_minus_final",
        "final_train_acc",
    ]
    rows_written = 0
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in records:
            if not has_both_metrics(rec):
                continue
            best = float(rec["best_test_acc"])
            final = float(rec["final_test_acc"])
            w.writerow(
                {
                    "scope": rec["_scope"],
                    "source": rec["_source"],
                    "n": rec.get("n", ""),
                    "k": rec.get("k", ""),
                    "seed": rec.get("seed", ""),
                    "params": rec.get("params", ""),
                    "best_test_acc": f"{best:.4f}",
                    "final_test_acc": f"{final:.4f}",
                    "gap_best_minus_final": f"{best - final:.4f}",
                    "final_train_acc": rec.get("final_train_acc", ""),
                }
            )
            rows_written += 1
    print(f"  wrote {rows_written} rows -> {path}")


def aggregate_by_nk(
    records: List[dict],
) -> Dict[Tuple[str, int, float], Dict[str, float]]:
    """Group by (scope, n, k); average across seeds.

    Returns mapping (scope, n, k) -> dict with means, stds, gap, count."""
    buckets: Dict[Tuple[str, int, float], List[dict]] = defaultdict(list)
    for rec in records:
        if not has_both_metrics(rec):
            continue
        n = rec.get("n")
        k = rec.get("k")
        if n is None or k is None:
            continue
        buckets[(rec["_scope"], int(n), float(k))].append(rec)

    out: Dict[Tuple[str, int, float], Dict[str, float]] = {}
    for key, recs in buckets.items():
        bests = np.array([float(r["best_test_acc"]) for r in recs])
        finals = np.array([float(r["final_test_acc"]) for r in recs])
        gaps = bests - finals
        out[key] = {
            "best_mean": float(bests.mean()),
            "best_std": float(bests.std(ddof=0)) if len(bests) > 1 else 0.0,
            "final_mean": float(finals.mean()),
            "final_std": float(finals.std(ddof=0)) if len(finals) > 1 else 0.0,
            "gap_mean": float(gaps.mean()),
            "gap_std": float(gaps.std(ddof=0)) if len(gaps) > 1 else 0.0,
            "count": len(recs),
            "params": int(recs[0].get("params", 0)),
        }
    return out


def write_summary_csv(agg: Dict[Tuple[str, int, float], Dict[str, float]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "scope",
        "n",
        "k",
        "params",
        "n_seeds",
        "best_mean",
        "best_std",
        "final_mean",
        "final_std",
        "gap_mean",
        "gap_std",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (scope, n, k), v in sorted(agg.items()):
            w.writerow(
                {
                    "scope": scope,
                    "n": n,
                    "k": k,
                    "params": v["params"],
                    "n_seeds": v["count"],
                    "best_mean": f"{v['best_mean']:.3f}",
                    "best_std": f"{v['best_std']:.3f}",
                    "final_mean": f"{v['final_mean']:.3f}",
                    "final_std": f"{v['final_std']:.3f}",
                    "gap_mean": f"{v['gap_mean']:.3f}",
                    "gap_std": f"{v['gap_std']:.3f}",
                }
            )
    print(f"  wrote summary -> {path}")


def plot_comparison(
    agg: Dict[Tuple[str, int, float], Dict[str, float]],
    output: str,
) -> None:
    """One panel per n; both metrics overlaid as lines vs k."""
    by_n: Dict[int, Dict[float, Dict[str, float]]] = defaultdict(dict)
    for (scope, n, k), v in agg.items():
        # Prefer the "main" / "samplewise" / "nslice" scopes for the plot.
        # Skip "depth" because its k axis is fixed at 0.5.
        if scope == "depth":
            continue
        if k not in by_n[n] or v["count"] > by_n[n][k]["count"]:
            by_n[n][k] = v

    if not by_n:
        print("  [warn] no plottable data; skipping figure", file=sys.stderr)
        return

    ns_sorted = sorted(by_n.keys())
    n_panels = len(ns_sorted)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.5 * n_panels, 4.6), sharey=True
    )
    if n_panels == 1:
        axes = [axes]

    for ax, n in zip(axes, ns_sorted):
        items = sorted(by_n[n].items())
        ks = [k for k, _ in items]
        bests = [v["best_mean"] for _, v in items]
        finals = [v["final_mean"] for _, v in items]
        gaps = [v["gap_mean"] for _, v in items]

        ax.plot(ks, bests, "s--", color="#d62728", label="best (max over epochs)",
                markersize=6, linewidth=1.6, alpha=0.85)
        ax.plot(ks, finals, "o-", color="#1f77b4", label="final (last epoch)",
                markersize=6, linewidth=2.2)

        # Annotate the gap at each k.
        for k, b, fi, g in zip(ks, bests, finals, gaps):
            if abs(g) >= 1.0:  # only annotate non-trivial gaps
                ax.annotate(f"+{g:.1f}", xy=(k, b),
                            xytext=(0, 4), textcoords="offset points",
                            fontsize=7, color="#d62728", ha="center")

        ax.set_xscale("log")
        ax.set_xlabel("Width multiplier k", fontsize=10)
        ax.set_title(f"n = {n}", fontsize=11)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9, loc="lower right")

    axes[0].set_ylabel("Test accuracy (%)", fontsize=10)
    fig.suptitle(
        "Best-vs-final test accuracy across the fractional-k sweep "
        "— red dashed = legacy 'best' metric (test-set selection); "
        "blue solid = corrected 'final' metric",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote figure -> {output}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", default=".",
                   help="repo root (where results/ and figures/ live)")
    p.add_argument("--scope", choices=["all"] + list(SCOPE_SOURCES.keys()),
                   default="all", help="which sweep(s) to audit")
    p.add_argument("--out-dir", default="results/metric_audit")
    p.add_argument("--fig-out", default="figures/best_vs_final_comparison.png")
    args = p.parse_args(argv)

    if args.scope == "all":
        scopes = list(SCOPE_SOURCES.keys())
    else:
        scopes = [args.scope]

    print(f"audit_metrics.py — scopes: {scopes}")
    all_records: List[dict] = []
    for s in scopes:
        recs = load_records(s, SCOPE_SOURCES[s], args.base_dir)
        print(f"  scope={s}: loaded {len(recs)} records")
        all_records.extend(recs)

    if not all_records:
        print("no records loaded; nothing to do", file=sys.stderr)
        return 1

    out_dir = os.path.join(args.base_dir, args.out_dir)
    write_per_seed_csv(all_records,
                       os.path.join(out_dir, "comparison.csv"))
    agg = aggregate_by_nk(all_records)
    write_summary_csv(agg,
                      os.path.join(out_dir, "summary_by_nk.csv"))

    fig_path = os.path.join(args.base_dir, args.fig_out)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plot_comparison(agg, fig_path)

    # Also print a terse summary to stdout for quick eyeballing.
    print("\n=== Aggregated by (scope, n, k) ===")
    print(f"{'scope':12} {'n':>5} {'k':>8} {'best':>8} {'final':>8} {'gap':>8}  seeds")
    for (scope, n, k), v in sorted(agg.items()):
        print(f"{scope:12} {n:>5} {k:>8.4f} {v['best_mean']:>8.2f} "
              f"{v['final_mean']:>8.2f} {v['gap_mean']:>+8.2f}  {v['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

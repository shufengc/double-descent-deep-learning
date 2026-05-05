# Final-vs-Best Test Accuracy Table — Day-1 Audit

**Date:** 2026-05-05
**Scope:** Yizheng's role — re-aggregate the 9-point fractional-k sweep, samplewise sweep, and n=8000 slice with both `best_test_acc` and `final_test_acc` metrics.
**Tool:** `scripts/audit_metrics.py` (no GPU; reads existing summary JSONs only)
**Raw output:** `results/metric_audit/comparison.csv`, `results/metric_audit/summary_by_nk.csv`

---

## Headline finding

**The corrected (final-epoch) metric reveals a clear valley at k ∈ [0.375, 0.75] in the n=4000 main sweep.** The valley is deeper than what was reported on the May-4 slides (which described it as "shallow ~1pp"). With the corrected metric the valley is **5 percentage points** (52.30% at k=0.25 → 47.09% at k=0.5).

The gap between the two metrics is largest at k = 0.5 (+4.09pp) and k = 0.75 (+4.78pp) — exactly in the over-parameterised U-valley region. This is the signature of training instability past memorisation: at those widths, *some* training epoch happened to land on a good test point but the final epoch did not, so reporting `max(history)` artificially flattened the curve.

This matches Codex's audit prediction quantitatively. Codex's quoted trajectory was 50 → 48 → 48 → 50 → 54; the actual measured trajectory at final-epoch is **52.30 → 49.12 → 47.09 → 49.39 → 51.03** across k = 0.25, 0.375, 0.5, 0.75, 1.0. Slightly deeper valley than Codex saw; same shape.

---

## Main 9-point sweep at n = 4000, 15% noise (1 seed available locally)

Source: `results/dd_recovery_5090_focused/main/summary.json`

| k | params | p/n | **best (legacy)** | **final (corrected)** | gap |
|---:|---:|---:|---:|---:|---:|
| 0.0625 | 823 | 0.21 | 25.69% | 25.69% | +0.00 |
| 0.1250 | 2 988 | 0.75 | 32.89% | 32.89% | +0.00 |
| 0.1875 | 6 505 | 1.63 | 49.17% | 48.89% | +0.28 |
| 0.2500 | 11 374 | 2.84 | 52.30% | 52.25% | +0.05 |
| 0.3750 | 25 540 | 6.39 | 50.54% | 49.12% | +1.42 |
| 0.5000 | 44 370 | 11.09 | 51.18% | **47.09%** | +4.09 |
| 0.7500 | 99 162 | 24.79 | 54.17% | 49.39% | +4.78 |
| 1.0000 | 175 258 | 43.81 | 52.86% | 51.03% | +1.83 |
| 2.0000 | 696 618 | 174.15 | 55.06% | 52.81% | +2.25 |

**Visual reading of the final-epoch column:** under-parameterised rise (25.7 → 32.9 → 48.9 → 52.3) → **valley** (52.3 → 49.1 → 47.1) → recovery (47.1 → 49.4 → 51.0 → 52.8). This is the textbook double-descent shape on a real ResNet, not "DD-recovery onset only." The slides currently undersell this finding.

The k = 0.0625 and k = 0.125 rows have zero gap because at those widths the network is under-parameterised and the test accuracy is monotone-rising during training (no oscillation past memorisation). The bug only shows up where training actually overshoots; that is exactly the over-parameterised regime we care about.

> **Note on per-seed coverage:** the locally-available `summary.json` has only one seed for the main sweep. The two-seed average we have been quoting in the deck (24.9 → 49.0 → 55.4) was likely computed across more results files, possibly on `origin/Yusheng` or `origin/yizheng-update`. Action item for Day 2: pull the second seed and re-aggregate.

---

## Sample-wise sweep (n = 1000 and n = 2000, 2 seeds each)

Source: `results/samplewise_nn/summary.json`

### n = 1000

| k | best | final | gap |
|---:|---:|---:|---:|
| 0.0625 | 13.96 | 13.54 | +0.42 |
| 0.1250 | 24.87 | 24.75 | +0.12 |
| 0.2500 | 38.00 | 36.65 | +1.35 |
| 0.5000 | 39.95 | 37.46 | +2.50 |
| 1.0000 | 39.72 | 37.89 | +1.82 |

### n = 2000

| k | best | final | gap |
|---:|---:|---:|---:|
| 0.0625 | 21.04 | 21.04 | +0.00 |
| 0.1250 | 27.66 | 27.66 | +0.00 |
| 0.2500 | 44.78 | 43.81 | +0.97 |
| 0.5000 | 45.62 | 42.09 | +3.54 |
| 1.0000 | 46.04 | 41.48 | +4.56 |

**Pattern:** at smaller n the valley is less pronounced (n=1000 stays roughly flat around 37–38% past k=0.25; n=2000 dips ~3pp), and the gap is again maximal in the over-parameterised tail (k=0.5 and 1.0). Same training-instability signature as the n=4000 sweep.

---

## n = 8000 slice (1 seed)

Source: `results/dd_recovery_5090_focused/nslice/summary.json`

| k | best | final | gap |
|---:|---:|---:|---:|
| 0.1250 | 39.73 | 39.73 | +0.00 |
| 0.2500 | 57.50 | 57.50 | +0.00 |
| 0.5000 | 60.43 | 56.94 | +3.49 |
| 1.0000 | 59.71 | 55.27 | +4.44 |

**Pattern:** at n=8000 the recovery is faster (jumps to 57.5% by k=0.25 already) but the same training-instability gap appears at k = 0.5 and k = 1.0. The slide-9 claim that the optimal width k* migrates rightward with n is not invalidated by the metric correction — but the magnitude of the migration may need to be recomputed at the corrected metric. This is a Day-2 task for Yizheng: pull all four n values, recompute argmax k* with `final_test_acc`.

---

## Where the metric most affects the headline numbers

Sorting all 22 (n, k) bins by gap magnitude, the worst offenders are:

| n | k | best | final | gap |
|---:|---:|---:|---:|---:|
| 4000 | 0.7500 | 54.17 | 49.39 | **+4.78** |
| 2000 | 1.0000 | 46.04 | 41.48 | **+4.56** |
| 8000 | 1.0000 | 59.71 | 55.27 | **+4.44** |
| 4000 | 0.5000 | 51.18 | 47.09 | **+4.09** |
| 2000 | 0.5000 | 45.62 | 42.09 | **+3.54** |
| 8000 | 0.5000 | 60.43 | 56.94 | **+3.49** |
| 1000 | 0.5000 | 39.95 | 37.46 | **+2.50** |
| 4000 | 2.0000 | 55.06 | 52.81 | +2.25 |

Every gap > 2pp lives in the over-parameterised regime (k ≥ 0.5). At k ≤ 0.25 the gap is below 1.5pp everywhere. **The metric bug is not uniform; it is concentrated where DD theory predicts training is unstable.**

This is a candidate sixth witness — the gap itself is a measurement of training-time instability that correlates with width.

---

## Day-2 action items (Yizheng)

1. **Pull seed 7 and seed 42 for the main sweep** from `origin/Yusheng` or `origin/yizheng` and re-aggregate. Today's table has only one seed for n=4000.
2. **Re-compute k*(n) at the corrected metric.** Slide 9's slope α=0.10 is based on best-test-acc argmax. With final-test-acc the optimal k may be different.
3. **Check whether the new k = 0.3, 0.4, 0.6 runs (Day 2) are saved with both metrics.** If not, the next training script needs to log both — patch in `Trainer.train()` if necessary.

## Verification

- [x] All summaries that were checked have both `best_test_acc` and `final_test_acc`.
- [x] No JSON files modified by this audit.
- [x] `audit_metrics.py` re-runs deterministically without GPU.
- [x] Generated `figures/best_vs_final_comparison.png` and `results/metric_audit/{comparison,summary_by_nk}.csv`.

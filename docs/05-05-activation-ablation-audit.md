# Activation-Ablation Metric Audit (Day 1)

**Date:** 2026-05-05
**Scope:** Yusheng's role — verify whether the May-3 activation ablation (slide-13 bullet 5: ReLU 48.19 / GELU 49.88 / Tanh 46.15) was reported with the correct metric.
**Source:** `origin/Yusheng:results/activation_ablation/results.json` (six records — three activations × two seeds).

---

## TL;DR

**No issue. The reported activation numbers are already at final-epoch (or, more precisely, best=final at k=0.1875 because training is stable at that width).** The slide-13 bullet 5 claim "GELU 49.88 > ReLU 48.19 > Tanh 46.15" stands without modification.

This is a **positive finding** consistent with the broader audit: the best-vs-final gap is concentrated in the over-parameterised regime (k ≥ 0.5), not at the recovery onset (k = 0.1875).

---

## Per-record metric comparison

| Activation | Seed | Params | best_test_acc | final_test_acc | gap |
|---|---:|---:|---:|---:|---:|
| ReLU | 42 | 6 505 | 48.09 | 48.09 | 0.00 |
| ReLU | 7  | 6 505 | 48.48 | 48.28 | **0.20** |
| GELU | 42 | 6 505 | 48.82 | 48.82 | 0.00 |
| GELU | 7  | 6 505 | 50.94 | 50.94 | 0.00 |
| Tanh | 42 | 6 505 | 47.36 | 47.36 | 0.00 |
| Tanh | 7  | 6 505 | 44.94 | 44.94 | 0.00 |

**Means across seeds:**

| Activation | best_mean | final_mean | gap |
|---|---:|---:|---:|
| ReLU | 48.28 | **48.19** | +0.10 |
| GELU | 49.88 | **49.88** | 0.00 |
| Tanh | 46.15 | **46.15** | 0.00 |

The slide-13 numbers (48.19 / 49.88 / 46.15) are the **final_mean** column, not the best_mean column. So Yusheng's pipeline reported the right metric — at least the figure-rebuild script `gamma-slides-prep/figures/make_activation_ablation_fig.py` hardcodes the per-seed *final* values (48.09, 48.28, 48.82, 50.94, 47.36, 44.94) which exactly match the `final_test_acc` columns of the JSON.

The only non-zero gap is ReLU seed 7: best=48.48 vs final=48.28, a 0.20pp gap. The mean ReLU best (48.28) is 0.10pp higher than final (48.19). At k = 0.1875 this is well within seed noise.

---

## Why this confirms the broader audit's narrative

In [05-05-final-vs-best-table.md](05-05-final-vs-best-table.md) we showed that the gap between best and final test accuracy grows with k, peaking at k = 0.5–0.75 where it reaches +4–5pp. At k = 0.0625 and k = 0.125 the gap was identically zero. The activation ablation at k = 0.1875 sits at the boundary of the under-/over-parameterised regimes and therefore exhibits a near-zero gap. This is exactly the pattern a "training-instability past memorisation" interpretation would predict.

So the activation ablation:
1. Is not affected by the metric bug.
2. **Is itself evidence that the bug is regime-specific** — at the recovery onset width, training is stable enough that the two metrics agree. The bug only kicks in once you have spare capacity.

---

## What is *not* in the ablation that we should flag

The activation ablation only covers k = 0.1875. We have not run GELU/Tanh at k = 0.5 or k = 2.0. If we wanted to push the spectral-mechanism-is-activation-agnostic claim harder, we would need an activation × k sweep — listed as future work in the paper. It is not Day-2 scope (no GPU budget for a full grid this week).

---

## Day-2 action items (Yusheng)

1. **None for the activation ablation itself** — numbers are already correct.
2. **Cross-check the figure**: confirm `gamma-slides-prep/figures/activation_ablation.png` and `overleaf-bundle/figures/activation_ablation.png` use the per-seed final values listed above. (Visual check; no code change needed.)
3. **For the Day-2 densification training runs (k = 0.3, 0.4, 0.6)**: confirm the experiment script logs both `best_test_acc` and `final_test_acc`, and report **final** as the headline number. The existing `exp_dd_recovery.py` does both, so Yusheng's job is just to confirm `final_test_acc` is what gets quoted in the WeChat update.

---

## Verification

- [x] No source files modified.
- [x] All six per-record entries from `origin/Yusheng:results/activation_ablation/results.json` checked.
- [x] Slide-13 numbers match the final_mean column to two decimals.
- [x] Cross-reference to main-sweep audit established (the regime-specific gap pattern).

# Shufeng's Day-1 Contribution — Metric Audit Across the Pipeline

**To:** Zhengda, Yusheng, Yizheng
**From:** Shufeng
**Date:** 2026-05-05 (Tue), end of day
**Branch:** all changes pushed to `origin/shufeng`. Nothing merged to `main`. Please review before Day 2 work begins tomorrow.

---

## What I did today (one paragraph)

Codex flagged a methodological bug last night: our README's NN-DD trajectory uses `best_test_acc = max(history["test_acc"])`, which is an implicit test-set selection that flattens the DD valley. Today I (a) confirmed the bug from the source code, (b) built a re-aggregation tool that scans every `summary.json` in the repo and reports both `best_test_acc` and `final_test_acc` side-by-side, (c) ran it across the main sweep, the samplewise sweep, the n=8000 slice, and the depth ablation, (d) audited the Bartlett bound and NTK pipelines, (e) audited the activation ablation and the Hessian pipeline, and (f) wrote up everything I found, including a Day-2 scope decision and division of work for tomorrow. **Conclusion: Codex was right and the effect is bigger than they estimated. With the corrected metric, the n=4000 valley at k ∈ [0.375, 0.75] is 5 percentage points deep, not "shallow ~1pp" as our slide claimed.**

The good news: three of our five spectral witnesses (NTK quick, NTK tight, Hessian) are computed at final-epoch checkpoints and are not affected. The activation ablation is also clean (gap ≈ 0 at k = 0.1875). The only re-run we need is the Bartlett bound calibration.

---

## Files I added — please read these in this order

All files are committed to `origin/shufeng`. Pull and skim before tomorrow's stand-up.

1. **[docs/05-05-day1-synthesis-and-day2-scope.md](05-05-day1-synthesis-and-day2-scope.md)** — start here. This is the executive summary plus the Day-2 division of work (locked). Two pages.

2. **[docs/05-05-final-vs-best-table.md](05-05-final-vs-best-table.md)** — Yizheng's role. The full re-aggregation table, side-by-side, for all 22 (n, k) bins. Read this if you want to see the numbers behind the headline.

3. **[docs/05-05-bartlett-ntk-checkpoint-audit.md](05-05-bartlett-ntk-checkpoint-audit.md)** — Zhengda's role. Confirms the Bartlett vacuity table needs a re-run; NTK pipelines are clean. File:line references for everything.

4. **[docs/05-05-activation-ablation-audit.md](05-05-activation-ablation-audit.md)** — Yusheng's role. Activation ablation numbers stand. Includes a defensive checkpoint for the figure script.

5. **[docs/05-05-hessian-final-epoch-audit.md](05-05-hessian-final-epoch-audit.md)** — my own. Hessian is clean.

6. **[scripts/audit_metrics.py](../scripts/audit_metrics.py)** — the re-aggregation tool. ~250 LOC, pure stdlib + matplotlib. Re-runnable with `python scripts/audit_metrics.py`. Reads only; never modifies any input file.

7. **[figures/best_vs_final_comparison.png](../figures/best_vs_final_comparison.png)** — the overlay plot (one panel per n). The legacy "best" curve sits visibly above the corrected "final" curve in the over-parameterised tail.

8. **[results/metric_audit/comparison.csv](../results/metric_audit/comparison.csv)** and **[results/metric_audit/summary_by_nk.csv](../results/metric_audit/summary_by_nk.csv)** — the raw and aggregated CSVs. Use these if you want to plot anything else.

---

## Headline numbers you should remember tomorrow

The n=4000 main sweep, corrected to final-epoch metric:

| k | best (legacy) | **final (corrected)** | gap |
|---:|---:|---:|---:|
| 0.0625 | 25.69 | 25.69 | 0.00 |
| 0.125 | 32.89 | 32.89 | 0.00 |
| 0.1875 | 49.17 | **48.89** | +0.28 |
| 0.25 | 52.30 | **52.25** | +0.05 |
| 0.375 | 50.54 | **49.12** | +1.42 |
| 0.5 | 51.18 | **47.09** | **+4.09** |
| 0.75 | 54.17 | **49.39** | **+4.78** |
| 1.0 | 52.86 | **51.03** | +1.83 |
| 2.0 | 55.06 | **52.81** | +2.25 |

**Read the bold "final" column.** That is now our headline trajectory. The valley is 5pp deep, k = 0.5 sits at 47.1%, and recovery is to 52.8% at k = 2.0. This is a real DD shape on a real ResNet — not "DD-recovery onset only" as we claimed at the talk.

The gap column is itself interesting: it is concentrated in the over-parameterised regime, suggesting the bug is regime-specific (training is unstable past memorisation, so picking max-over-epochs introduces a heterogeneous bias). I think this is a candidate **sixth witness** — see the Day-2 plan in the synthesis doc.

---

## What is NOT changing in our story

- The five-witness narrative survives. Three witnesses are clean, one (Bartlett) needs recalibration but the qualitative claim (vacuity rises monotonically) almost certainly stays.
- The "literal ResNet-18 cannot show DD" finding (slide 8) is unchanged — those numbers (54.08 / 56.62 / 60.94) were always at final-epoch.
- The activation ablation is unchanged.
- The depth ablation framing is mostly unchanged (depth at k=0.5 has a +1.84pp gap, but the depth-ranking 52.14 / 51.18 / 47.77 holds at both metrics; need to spot-check on Day 2).
- All the spectral measurements (Bartlett r_k, NTK condition, λ_max) are unchanged. Only the *calibration target* of the Bartlett bound shifts.

---

## What IS changing — Day-2 priorities

Per the locked Day-2 division of work in [`05-05-day1-synthesis-and-day2-scope.md`](05-05-day1-synthesis-and-day2-scope.md) §"Day-2 division of work (locked)":

- **Yizheng:** pull the second seed for the n=4000 main sweep, re-compute k*(n) with corrected metric, update §6.9.
- **Yusheng:** train k = 0.3, 0.4, 0.6 at n=4000 (six runs), update the figure-rebuild script to read JSON.
- **Zhengda:** patch `exp_bartlett_bound_eval.py` to default to `final_test_acc`, re-run calibration, update §6.5.
- **Shufeng (me):** build the gap-vs-k figure (sixth-witness candidate), compute correlation with existing witnesses, draft the new §6.X metric-audit sub-section.

GPU sequencing: Yusheng first (smallest models), then anything that needs a checkpoint downstream.

WeChat stand-ups: 09:00 / 13:00 / 18:00 tomorrow. Single source of truth for new numbers: `docs/05-05-final-numbers-master.md` (Yizheng owns).

---

## Why this is a good thing for the paper

The audit converts a "modest-but-honest empirical paper" into a **paper with a methodological contribution**. We publicly identified a metric leak in our own pipeline between presentation and submission, fixed it, and showed that the corrected story is *stronger* than the original. Most papers do not do this. Per Lecture-12 page-54 §4 ("New Results"), this is exactly the kind of beat the paper rubric rewards.

Specifically, the new §4 sub-section will be:

> **Section 4.X — A metric audit and the case for final-epoch evaluation in DD.**
> During paper week, a re-audit of our own pipeline revealed that the headline NN-DD trajectory had been computed using `best_test_acc = max(history["test_acc"])`, which is an implicit form of model selection on the test set. We compare both metrics across the entire sweep, find that the gap between them is concentrated in the over-parameterised regime (consistent with training instability past memorisation), and report the corrected trajectory. The valley at k ∈ [0.375, 0.75] is 5 percentage points deep with the corrected metric — substantially more pronounced than what `best_test_acc` reporting suggested. We propose final-epoch evaluation as the default for double-descent claims and treat the per-k gap between the two metrics as a candidate sixth spectral witness.

That paragraph is roughly the structure I will draft tomorrow.

---

## A short note on what I did not do

- I did not patch any source file. The `best_test_acc` defaults in `exp_samplewise_nn_plot.py:126` and `exp_bartlett_bound_eval.py:80` are unchanged. Patching is Day-2 work after team alignment — and Zhengda owns the Bartlett one specifically.
- I did not re-run any training. Day 1 was no-training by design; everything fits into existing JSONs.
- I did not modify the README. The README's "Best Test%" tables are still wrong; updating them is Day-3 scope.
- I did not push anything to `main`. `origin/shufeng` only.

---

## How to use this work

If you want to dig into the data yourself:
```bash
git fetch origin
git checkout shufeng
git pull
python scripts/audit_metrics.py
# Outputs in results/metric_audit/ and figures/best_vs_final_comparison.png
```

If you only want to read the headline finding: read `docs/05-05-day1-synthesis-and-day2-scope.md` (10 minutes).

If you have questions before tomorrow's stand-up: WeChat me.

— Shufeng

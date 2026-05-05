# Day-1 Synthesis and Day-2 Scope Decision

**Date:** 2026-05-05 (Tue)
**Status:** Day-1 verification complete. **Codex's claim is confirmed and the bug is more impactful than initially estimated.** Day-2 scope locked: extend the sweep at the corrected metric and treat the audit itself as a paper contribution.

---

## Headline (one paragraph)

The metric bug Codex found in `exp_architecture.py:79` and `exp_samplewise_nn_plot.py:126` is real, present in two downstream pipelines (the main NN sweep and the Bartlett bound calibration), and produces a **5-percentage-point flattening of the over-parameterised valley** at n=4000. With the corrected `final_test_acc` metric the trajectory is **52.3 → 49.1 → 47.1 → 49.4 → 51.0 → 52.8** across k = 0.25 → 2.0 — a textbook double-descent shape with a real valley at k = 0.5. The "shallow ~1pp valley" framing on the May-4 slides was an artefact of the metric. NTK quick, NTK tight, Hessian, and the activation ablation are *not* affected and remain robust witnesses. Three out of five spectral diagnostics are clean; the fourth (Bartlett vacuity ratio) needs a calibration re-run.

---

## Per-role summary

### Yizheng — main NN sweep re-aggregation [LARGEST IMPACT]

Re-ran `audit_metrics.py` over all summary JSONs. The summaries already contained both metrics, so no retraining was needed. Output:
- `results/metric_audit/comparison.csv` (per-seed)
- `results/metric_audit/summary_by_nk.csv` (averaged)
- `figures/best_vs_final_comparison.png` (overlay plot, panel per n)
- `docs/05-05-final-vs-best-table.md` (the writeup)

**Key finding:** the gap between best and final is concentrated in the over-parameterised regime (k ≥ 0.5), where it reaches +4.78pp. At small k the gap is ≤ 0.5pp. The bug is regime-specific in exactly the way DD theory would predict (training is unstable past memorisation).

### Zhengda — Bartlett + NTK audit [SECOND LARGEST IMPACT]

Static read of `exp_bartlett_bound_eval.py` and the two NTK scripts.

- **Bartlett bound: AFFECTED.** The vacuity-ratio table on slide-13 bullet 5 was calibrated against `best_test_acc`-derived risk (`exp_bartlett_bound_eval.py:80,84,265`). The qualitative monotone-rise story survives but absolute numbers will shift by 5–15% at the wider k values. Day-2 work: re-run the bound calibration with `final_test_acc`.
- **NTK quick + tight: CLEAN.** Both NTK scripts compute on the final-epoch model. The slide-10 condition-number numbers (18.7 → 558.8 → 79.1 → 40.3 quick; 1700 → 407 → 121 tight) are not affected.

Writeup: `docs/05-05-bartlett-ntk-checkpoint-audit.md`.

### Yusheng — activation ablation audit [NO IMPACT]

Read `origin/Yusheng:results/activation_ablation/results.json`. All six per-seed records have `best ≈ final` (gaps ≤ 0.20pp). The slide-13 numbers (ReLU 48.19 / GELU 49.88 / Tanh 46.15) are already the final-epoch means.

This is itself a *positive piece of evidence* for the metric-audit narrative: at k = 0.1875 (recovery onset) the gap vanishes, confirming the bug is regime-specific.

Writeup: `docs/05-05-activation-ablation-audit.md`.

### Shufeng — Hessian audit [NO IMPACT]

Read `exp_hessian_topeig.py`. Hessian computation is "at end of training" (verbatim from the docstring). The trajectory 21 → 274 → 2394 → 9882 → 554 across k = 0.0625 → 2.0 is computed on final-epoch models. Per-parameter peak at k = 0.1875 stands.

One small pre-existing caveat: Hessian uses n = 2000 / 600 epochs, not the headline n = 4000 / 2000 epochs. This is a budget choice, not a metric issue. Worth flagging in §6.13 of the paper.

Writeup: `docs/05-05-hessian-final-epoch-audit.md`.

---

## What the corrected story looks like

Five-witness panel after the audit:

| Witness | Affected by metric bug? | Status |
|---|---|---|
| Penultimate stable rank | Probably no (not yet verified — Day-2 spot check) | Likely clean |
| Bartlett effective rank r_k(Σ) | Measurement clean; **bound calibration affected** | Re-run on Day 2 |
| NTK quick condition number | No | Clean |
| NTK tight condition number | No | Clean |
| Hessian per-parameter λ_max | No | Clean |

Headline test-accuracy claim:

| Quantity | Pre-audit (best) | Post-audit (final) |
|---|---|---|
| k = 0.0625 → 0.1875 → 2.0 | 24.9 → 49.0 → 55.4 | 25.7 → 48.9 → 52.8 (single seed) |
| Valley at k ∈ [0.375, 0.75] | ~1pp ("shallow") | **~5pp ("real DD valley")** |
| Slide-13 bullet 5 (activation) | 48.19 / 49.88 / 46.15 | unchanged |
| Slide-13 bullet 5 (Bartlett vacuity) | 1.0 → 3.4 | TBD Day 2 (likely 1.0 → ~3.0) |

---

## Day-2 scope decision

Per the three-outcome framing in `docs/05-05-three-day-code-finalization-plan.md` §1, Day-1 evidence puts us in the **"valley clear"** branch:

> Outcome A — Valley clear (≥ 3pp dip): "We observe a recovery valley at k ∈ [0.25, 0.5], distinct from the literature's emphasis on a sharp peak."

Specifically: the n=4000 trajectory now shows a **5pp** valley (52.3 → 47.1) — well past the 3pp threshold. So the paper claim upgrades from "DD-recovery onset, shallow valley" to **"genuine DD-recovery valley at k ∈ [0.375, 0.75]"** with the corrected metric.

The 5pp claim rests on a single seed at n=4000 in the locally-available data. Day 2 must:

1. **Pull the second seed** (seed 7 or seed 42, whichever is missing) for the n=4000 main sweep from `origin/Yusheng` or another branch, and re-aggregate. This is the highest-priority Day-2 action because it will tell us whether the 5pp valley is robust to seed.
2. **Densify the valley region with new training runs** at k = 0.3, 0.4, 0.6 (Yusheng's task per the original plan). With the corrected metric the densification becomes much more interesting — we are nailing down a real valley, not just confirming a shallow one.
3. **Re-run Bartlett with final_test_acc** (Zhengda).
4. **Re-compute k*(n) using `final_test_acc`** at all four sample sizes (Yizheng); update slide-9's α slope number in the paper.
5. **Build the gap-vs-k figure** as the candidate "sixth witness" (Shufeng).

---

## Day-2 division of work (locked)

| Owner | Task | Output | Time |
|---|---|---|---|
| **Yizheng** | (a) Pull second seed for n=4000 main sweep; re-aggregate. (b) Re-compute k*(n) with `final_test_acc`. (c) Update slide-9 / §6.9 narrative. | Updated `results/metric_audit/comparison.csv` with both seeds; revised α slope value | Half day |
| **Yusheng** | (a) Train k = 0.3, 0.4, 0.6 at n=4000, seeds 42 + 7. Six runs. Save both metrics. (b) Update `make_activation_ablation_fig.py` to draw from JSON rather than hardcoded list (defensive). | `results/dd_valley_densification/*.json`; updated figure script | Full day |
| **Zhengda** | (a) Patch `exp_bartlett_bound_eval.py:80–88` to default to `final_test_acc`. (b) Re-run calibration. (c) Diff old vs new vacuity table; update §6.5 paragraph. | Patched script + new `summary.csv` + diff doc | Half day |
| **Shufeng** | (a) Build `figures/gap_vs_k.png` (the sixth-witness candidate). (b) Compute correlation between gap_vs_k and the existing five witnesses. (c) Draft new §6.X "metric audit" sub-section (~600 words). (d) Optional: run Hessian on Yusheng's new k = 0.3/0.4/0.6 checkpoints if GPU is free at end of day. | Figure + correlation table + draft sub-section | Full day |

**Coordination:** WeChat stand-ups at 09:00 / 13:00 / 18:00. Yizheng owns `docs/05-05-final-numbers-master.md` (single source of truth — every new number lands here first).

**GPU sequencing for Day 2:** Yusheng's six densification runs first (small models, ~30 min each = ~3 hours total). Then Zhengda's Bartlett rerun (no GPU needed). Then Shufeng's optional Hessian extension.

**Hard stop for Day 2:** 23:59 Wed 2026-05-06. Anything not landed by then slips to Day 3 integration scope.

---

## Risks for the rest of the week

| Risk | Severity | Mitigation |
|---|---|---|
| Second-seed pull (Yizheng Day 2) reveals the 5pp valley is seed-dependent (e.g., one seed shows 5pp, the other shows 1pp) | High | Run a third seed if GPU budget allows; otherwise report both seeds and acknowledge the ~3pp uncertainty in the paper. |
| New k = 0.3/0.4/0.6 runs (Yusheng Day 2) don't show valley with `final_test_acc` | Medium | Pre-committed framing C: "capacity-threshold recovery, monotone past threshold." Still publishable, but weaker headline. |
| Bartlett re-calibration (Zhengda Day 2) destabilises §6.5 conclusion | Medium | The qualitative claim (vacuity rises monotonically) survives; only absolute numbers shift. We have already pre-disclosed the limitation. |
| Shufeng's exam (May 12) collides with last-minute paper edits | High | Code freeze Thu 2026-05-07 evening. Paper writing May 8–10 and May 13 only. May 11 exam prep. May 12 exam. May 13 light final polish. |

---

## Bottom line

We came into Day 1 expecting to either confirm or refute Codex's claim. We confirmed it, and the effect is **larger than Codex reported numerically** (5pp valley vs Codex's quoted ~2pp). The audit therefore upgrades a "modest-but-honest empirical paper" into a **paper with a methodological contribution**: we publicly identified and corrected a metric leak in our own pipeline, between presentation and submission. Most papers do not do this.

Day 2 is now the *experimental* day — we add seed coverage, densify the valley, re-run Bartlett, and ship a gap-vs-k figure as the candidate sixth witness. Day 3 is integration and freeze.

The presentation is done. The paper is the deliverable. We are in better shape after Day 1 than before.

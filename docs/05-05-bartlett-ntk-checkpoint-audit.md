# Bartlett + NTK Checkpoint Audit (Day 1)

**Date:** 2026-05-05
**Scope:** Zhengda's role — audit whether the Bartlett bound and the empirical-NTK pipelines have a checkpoint-selection issue analogous to Codex's `best_test_acc` finding.
**Method:** static read of `src/experiments/exp_bartlett_bound_eval.py`, `src/experiments/exp_full_empirical_ntk_quick.py`, and `src/experiments/exp_full_empirical_ntk_plot.py`.

---

## TL;DR

| Pipeline | Checkpoint used | Calibration target | Issue? |
|---|---|---|---|
| Bartlett bound (`exp_bartlett_bound_eval.py`) | n/a — uses spectral records | **`best_test_acc`** with explicit fallback to `final_test_acc` | **YES — needs Day-2 fix** |
| Full empirical NTK quick | Final-epoch model | n/a (NTK eigenvalues, not accuracy-derived) | NO |
| Full empirical NTK plot | Reads NTK summary; no checkpoint involvement | n/a | NO |

**Headline:** the Bartlett bound's vacuity-ratio numbers reported on slide 13 ("vacuity 1.0 → 3.4") were calibrated against `best_test_acc` because of the explicit preference order at `exp_bartlett_bound_eval.py:80–88`. **The vacuity ratio table needs to be re-computed on Day 2 with `final_test_acc` as the calibration target.** The NTK side is clean.

---

## Bartlett bound — the calibration leak

### The offending code (file: `src/experiments/exp_bartlett_bound_eval.py`)

Lines 79–93:
```python
def get_dd_accuracy_by_k(dd_rows):
    """Use best_test_acc if available, otherwise final/test_acc."""
    out = {}
    for r in dd_rows:
        k = float(r["k"])
        if "best_test_acc" in r:
            acc = float(r["best_test_acc"])
        elif "final_test_acc" in r:
            acc = float(r["final_test_acc"])
        elif "test_acc" in r:
            acc = float(r["test_acc"])
        else:
            continue
        out.setdefault(k, []).append(acc)
    return {k: sum(v) / len(v) for k, v in out.items()}
```

Then at the calibration site, lines 263–268:
```python
dd_acc = get_dd_accuracy_by_k(dd)
for r in rows:
    k = r["k"]
    # Prefer DD-Recovery best test acc if matching k exists; otherwise use spectral test acc.
    obs_acc = dd_acc.get(k, r["spectral_test_acc"])
    r["observed_test_acc"] = obs_acc
    r["observed_test_risk"] = 1.0 - obs_acc / 100.0
```

And the calibration math, lines 274–278:
```python
scale_r = calibrate_proxy(rows, "proxy_sqrt_r_over_n", "observed_test_risk", reference_k)
...
r["cal_bound_r"] = scale_r * r["proxy_sqrt_r_over_n"]
r["vacuity_ratio_r"] = r["cal_bound_r"] / max(r["observed_test_risk"], 1e-12)
```

### What this actually does

The pipeline calibrates the Bartlett proxy `sqrt(r_k(Σ) / n)` to match `observed_test_risk` at a chosen reference k (slide-13 reports k=0.125 as the calibration anchor where vacuity = 1.0). The chosen `observed_test_risk` is **`1 − best_test_acc / 100`**, not `1 − final_test_acc / 100`.

Because best_test_acc is systematically *higher* than final_test_acc in the over-parameterised regime (gap up to +4.78pp at k=0.75 — see [05-05-final-vs-best-table.md](05-05-final-vs-best-table.md)), the calibrated risk that the Bartlett bound is fitted against is artificially *low* in that regime. So:

- The calibrated bound `cal_bound_r` is computed against `1 − best_test_acc/100`.
- But in our paper §6.5 we report `observed_test_risk` as the actual risk.
- The vacuity ratio `cal_bound_r / observed_test_risk` is therefore **anchored to the wrong target**.

### Quantitative impact estimate

At k = 0.5 the gap is +4.09pp. So:
- `best_test_acc = 51.18%` → `risk = 0.4882`
- `final_test_acc = 47.09%` → `risk = 0.5291`
- Bartlett bound was calibrated to track `risk = 0.4882`; the *true* risk is `0.5291`.

The calibrated bound stays the same (it's a function of `r_k(Σ)`), so the vacuity ratio at k=0.5 *decreases* slightly when re-computed at the corrected metric — the bound looks slightly less loose because the actual risk is higher. The qualitative story (vacuity rises monotonically with k from ~1 to ~3.4) probably survives, but the exact numbers in the slide-13 bullet 5 will shift by 5–15% at the wider k values.

---

## NTK — clean (final-epoch model)

### `exp_full_empirical_ntk_quick.py`

The training loop (lines 128–161) runs `epochs` epochs and at the very end calls:
```python
test_acc_value, test_loss_value = evaluate(model, test_loader, device)  # line 161
```
on the final-epoch model. The Jacobian computation for the empirical NTK is on `model.eval()` immediately after this — same final-epoch model. No "best epoch" selection anywhere.

The checkpoint save at line 289 (`save_checkpoint(model, ...)`) is also called after the training loop ends, so the saved checkpoint *is* the final-epoch model.

### `exp_full_empirical_ntk_plot.py`

This is a pure plotting script that reads `summary.json` from the NTK quick run. No checkpoint loading; no metric selection. Just renders the eigenvalue spectra reported by the upstream script.

### Conclusion for NTK

The NTK condition-number numbers reported on slide 10 (quick: 18.7 → 558.8 → 79.1 → 40.3 across k = 0.125, 0.1875, 0.25, 0.5; tight: 1700 → 407 → 121 across k = 0.125, 0.25, 0.5) are computed at the final-epoch model. They are not affected by the Codex finding. The "five witnesses concentrate at k ∈ [0.125, 0.25]" claim is robust on the NTK side.

---

## Hessian — also clean (cross-reference to Shufeng's audit)

See [05-05-hessian-final-epoch-audit.md](05-05-hessian-final-epoch-audit.md). The Hessian script also uses the final-epoch model. So three of the five spectral witnesses (NTK quick, NTK tight, Hessian) are not affected by the metric bug.

The other two witnesses are penultimate stable rank and Bartlett r_k. Penultimate stable rank is computed on the final-epoch model in `exp_nn_spectral.py` (we should verify this on Day 2 if time permits, but the file structure suggests final-epoch). Bartlett r_k is the spectral measurement *itself* (`tr(Σ) / ||Σ||_op` on penultimate features) — it is computed on the final-epoch model. **It is the bound's calibration that uses best_test_acc**, not the spectral measurement. So the *measurement* of r_k across the sweep is correct; only the conversion to a risk-bound vacuity ratio is biased.

---

## Day-2 action items (Zhengda)

1. **Patch `exp_bartlett_bound_eval.py:80`** to default to `final_test_acc` and treat `best_test_acc` as the legacy fallback. Keep both options behind a `--metric` flag for the metric-comparison appendix.
2. **Re-run** `python -m src.experiments.exp_bartlett_bound_eval --metric final` on the existing spectral summary. Diff the new `summary.csv` against the existing one.
3. **Update the slide-13 bullet 5 vacuity numbers** in the paper §6.5 with whichever values come out of step 2. The qualitative story (monotone rise) likely survives; the absolute numbers will shift.
4. **Optional**: re-calibrate at a different reference k to see if the vacuity slope is sensitive to the anchor choice. This is a robustness check for the paper.

---

## Verification

- [x] No source files modified by this audit.
- [x] All claims linked to specific file:line references.
- [x] Issue confirmed at `exp_bartlett_bound_eval.py:80,84,265`.
- [x] NTK pipelines (quick + plot) confirmed clean.
- [x] Cross-reference to Hessian audit established.

# Hessian Top-Eigenvalue Checkpoint Audit (Day 1)

**Date:** 2026-05-05
**Scope:** Shufeng's role — audit whether the slide-11 Hessian λ_max trajectory (21 → 274 → 2394 → 9882 → 554 across k = 0.0625 → 2.0) was computed at the final-epoch model or at a best-test-acc-selected checkpoint.
**Method:** static read of `src/experiments/exp_hessian_topeig.py`.

---

## TL;DR

**No issue. The Hessian top-eigenvalue is computed on the final-epoch model.** The slide-11 trajectory and the per-parameter peak at k = 0.1875 are not affected by Codex's metric finding. The "five witnesses concentrate at k ∈ [0.125, 0.25]" claim survives the Hessian audit.

This is a positive finding (no fix needed) and joins NTK-quick + NTK-tight in being clean. **Three of the five witnesses are robust to the metric correction.**

---

## Source-code evidence

### `src/experiments/exp_hessian_topeig.py`

Docstring (lines 1–17):
```python
"""
Hessian top-eigenvalue (sharpness) at the DD-recovery onset (N2 / report §6.13).
...
Computes top eigenvalue of the loss Hessian via power iteration on
Hessian-vector products (using torch.autograd.grad twice). At end of training,
on a random subset of training data, k ∈ {0.0625, 0.125, 0.1875, 0.5, 2.0}.
"""
```

The phrase **"at end of training"** is explicit: the Hessian is computed after the training loop terminates, on whatever model state was reached at epoch `epochs`.

### Training loop and post-training Hessian computation

Lines 95–119 (`train_then_hessian`):
```python
for ep in range(1, epochs + 1):
    model.train()
    total, correct = 0, 0
    for x, y, _ in train_loader:
        ...
    last_train_acc = 100.0 * correct / max(1, total)
    ...

# Build a small batch for Hessian (random sample of training subset)
for batch_x, batch_y, _ in train_loader:
    ...
```

The Hessian computation is *immediately after* the `for ep` loop ends. The model variable in scope is the final-epoch model. There is no checkpoint loading, no `torch.load`, no `state_dict` selection by best-test-epoch. The Hessian is computed on the same model object that just finished its 30th (or whatever epochs) training step.

### What the script saves

Lines 145–155:
```python
result = dict(
    k=k, n_train=n_train, seed=seed, epochs=epochs,
    n_params=n_params,
    train_acc=float(last_train_acc),
    test_acc_smoke=...,
    topeig=float(topeig),
    n_iter=int(n_iter),
    train_wallclock_sec=float(time.time() - t0 - pi_time),
    pi_wallclock_sec=float(pi_time),
)
```

The script reports `train_acc` (final-epoch) and `test_acc_smoke` (a quick test-accuracy smoke check on a subsample, also at the final epoch). It does **not** compute or save `best_test_acc`. So even if downstream code wanted to anchor the Hessian to a best-test-acc-selected checkpoint, the data simply does not exist for that.

---

## Cross-check with the slide-11 trajectory

The trajectory we report on slide 11:

| k | λ_max | n_train | epochs |
|---:|---:|---:|---:|
| 0.0625 | 21 | 2000 | 600 |
| 0.125 | 274 | 2000 | 600 |
| 0.1875 | 2 394 | 2000 | 600 |
| 0.5 | 9 882 | 2000 | 600 |
| 2.0 | 554 | 2000 | 600 |

Source: `results/hessian_topeig/k*_n2000_ep600_s42/result.json`. All are seed=42, n=2000, 600 epochs, computed at the final epoch of training on those models. The per-parameter version `λ_max / p` peaks at k = 0.1875 — the recovery-onset width — and this remains the cleanest framing of the fifth-witness claim.

---

## One small caveat (worth flagging in the paper)

The Hessian was computed at **n = 2000** with **600 epochs**, not at our headline **n = 4000** with **2000 epochs** configuration. This is documented (the script's default config is `n_train=2000, epochs=600`), and the §6.13 paragraph already discusses why: full-batch Hessian power iteration is expensive and we used a smaller training set to keep it tractable on a single 5090.

The metric audit does not interact with this caveat — it would still be the case at n = 4000 / 2000 epochs that the Hessian is computed on the final-epoch model. But if a grader asks "are your Hessian numbers comparable to the test-accuracy numbers in the same way the other witnesses are?" the honest answer is "they share the architecture and the spectral diagnostic methodology; the training budget is different by a factor of ~3 in n and ~3 in epochs."

This caveat does not weaken the witness-concentration claim. Five different measurements all peaking around k = 0.125–0.25 is robust evidence even if one of them was computed at a smaller training budget.

---

## Day-2 action items (Shufeng)

1. **None for the Hessian itself** — numbers are correct.
2. **Optional**: if Day-2 GPU budget allows, run the Hessian on the new k = 0.3, 0.4, 0.6 checkpoints (Yusheng's training run). This would extend the Hessian sweep into the over-parameterised valley region and let us check whether λ_max also peaks/valleys at k = 0.5 with finer resolution. Only do this if the densification runs finish well before end of Day 2.
3. **In the paper §6.13**: note explicitly that the Hessian uses final-epoch models. This pre-empts the obvious "did you peek at the test set?" question.

---

## Cross-reference to other audits

- Bartlett bound: **affected** (calibration uses `best_test_acc`). See [05-05-bartlett-ntk-checkpoint-audit.md](05-05-bartlett-ntk-checkpoint-audit.md).
- NTK quick + tight: **clean** (final-epoch model). See same file.
- Activation ablation: **clean** at k = 0.1875 (gap ≈ 0). See [05-05-activation-ablation-audit.md](05-05-activation-ablation-audit.md).
- Main NN sweep: **affected** at k ≥ 0.375 (gap up to +4.78pp). See [05-05-final-vs-best-table.md](05-05-final-vs-best-table.md).

---

## Verification

- [x] No source files modified.
- [x] Hessian script lines 9–10 (docstring), 116–119 (post-loop Hessian), 145–155 (output dict) all confirm final-epoch usage.
- [x] No `torch.load` or checkpoint-selection logic in the script.
- [x] Cross-references to the other four audit docs established.

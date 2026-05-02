---
marp: true
theme: default
paginate: true
size: 16:9
math: katex
style: |
  section { font-size: 22px; }
  h1 { font-size: 34px; }
  h2 { font-size: 28px; }
  .small { font-size: 17px; }
  .citation { font-size: 15px; color: #888; }
  table { font-size: 17px; }
---

<!-- _class: lead -->

# Double descent isn't free — you have to design the architecture to see it

**Fractional-$k$ ResNet recovers Nakkiran model-wise DD where literal ResNet18 cannot — with four spectral diagnostics localizing the phase transition at $k=0.1875$**

Zhengda Li (zl3651) · Yusheng Li (yl6009) · Yizheng Lin (yl6079) · Shufeng Chen (sc5739)
EECS 6699 · Mathematics of Deep Learning · Spring 2026 · 12 min

<span class="small">Repo: `github.com/shufengc/double-descent-deep-learning` branch `shufeng`</span>

---

## 1. The question

> *"What did you learn about a neural network?"* — Lecture 12

A canonical generalisation theory predicts a **U-curve** in test error as you grow the model. Modern overparameterised networks routinely violate this — exhibiting **double descent** (DD).

**Open question for our project:** does the textbook DD picture (Belkin 2019, Nakkiran 2021) hold on real architectures, or is it fragile to architectural choices?

**One-sentence headline.** Width is a clean axis to span the interpolation threshold; *literal* ResNet18 at $n=4{,}000$ does not exhibit DD; a *fractional-$k$* ResNet does, with the recovery onset at $k=0.1875$, where four independent spectral diagnostics localize the phase transition.

---

## 2. Background — classical U-curve → DD

<div style="display:grid; grid-template-columns: 1fr 1fr; gap: 16px;">

<div>

**Classical (Vapnik 1971; bias-variance):**
$$\text{test err} \approx \text{bias}(p)^2 + \text{var}(p) + \sigma^2$$
Predicts: more parameters $\Rightarrow$ more variance $\Rightarrow$ worse test error past a point.

**Belkin (2019) / Nakkiran (2021):**
On real overparameterised models, test error rises near the interpolation threshold $p^\star(n)$, then **descends again**.

**EMC** (effective model complexity, Nakkiran §3): peak at **EMC ≈ n**, not $p \approx n$.

Three DD axes: model-wise, sample-wise, epoch-wise.

</div>

<div>

![w:480](../figures/fig1_model_wise_rff.png)

</div>

</div>

---

## 3. Setup

**Two model families, both run end-to-end:**

- **RFF + min-norm interpolation** on MNIST. Closed-form, no training. Sweep $p \in [10, 6000]$, $n=1000$, $\lambda \to 0^+$. → Exps 1–2, 5–8.
- **Fractional-$k$ ResNet** on CIFAR-10 with 15% label noise. 3-stage ResNet with widths $(c_1, c_2, c_3) = (\max(1, 16k), 32k, 64k)$. Sweep $k \in \{0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 2.0\}$. Adam lr $10^{-4}$, 2000 ep, 2 seeds, GPU-resident. → Exps 3–4, 6.5–6.13.

**Why fractional-$k$:** literal ResNet18 has the smallest stage at width 64 → ~11M params, already deep in the over-parameterised tail at $n=4{,}000$. To traverse the DD threshold, the architecture must continuously span it.

---

## 4. Reproduction — RFF model-wise DD

<div style="display:grid; grid-template-columns: 1fr 1fr; gap: 16px;">

<div>

**Setup.** RFF $\phi(x) = \sqrt{2/D}\cos(Wx + b)$, $D=p$ features, MNIST 10-class subset, $n=1{,}000$, ridge $\lambda \to 0$.

**Result.** Test error spikes near $p/n = 1$, descends as $p$ grows. The classical U-curve only holds restricted to $p < n$. The interpolation threshold is the source of the peak.

This is the textbook reproduction of Belkin (2019) Figure 1.

</div>

<div>

![w:520](../figures/fig1_model_wise_rff.png)

</div>

</div>

---

## 5. RFF mechanism — variance + ridge fix (Person A)

<div style="display:grid; grid-template-columns: 1fr 1fr; gap: 16px;">

<div>

**Why does test error spike at $p/n = 1$?**
The Gram matrix $\Phi^\top\Phi$ has condition number $\kappa \to \infty$ at the threshold; min-norm interpolant amplifies any noise direction.

**Test:** add ridge $\lambda > 0$ → directly shrinks small singular values → smooths peak.

**Result (§6.5).** $\lambda \in [10^{-6}, 10^{-2}]$ monotonically smooths the peak. By $\lambda = 10^{-2}$ the U-shape vanishes — confirming variance explosion at $p/n = 1$ is the mechanism.

</div>

<div>

![w:520](../figures/personA_ridge_smooths_peak.png)

</div>

</div>

---

## 6. Puzzle — literal ResNet18 fails (controlled, §5.3.1)

<div style="display:grid; grid-template-columns: 0.95fr 1fr; gap: 16px;">

<div>

**Test.** torchvision `resnet18` at width multipliers $\in \{0.5, 1.0, 2.0\}$ on the same hyperparameters as our fractional-$k$ sweep ($n=4{,}000$, 15% noise, Adam lr=$10^{-4}$, 2000 ep).

**Result.** All three sit at $p/n \in [700, 11200]$ — deep in the over-param tail. Test accuracy is **flat** across multipliers. No DD trajectory appears because BasicBlock widths cannot continuously span the threshold.

**Diagnosis.** The experiment requires an architecture whose width *continuously* spans the threshold. Hence fractional-$k$.

</div>

<div>

![w:520](../figures/resnet18_vs_fractionalk.png)

</div>

</div>

---

<!-- _class: lead -->

## 7. Headline — fractional-$k$ ResNet recovers DD

<div style="display:grid; grid-template-columns: 0.85fr 1.15fr; gap: 18px;">

<div>

**At $n=4{,}000$, sweeping $k \in \{0.0625, \ldots, 2.0\}$:**

| $k$ | best test acc |
|---:|---:|
| 0.0625 (under-fit) | **24.9%** |
| 0.1875 (recovery onset) | **49.0%** |
| 2.0 (over-param tail) | **55.4%** |

Classical rise → recovery onset → plateau, on real noisy CIFAR-10, no kernel approximation.

The single biggest jump is the $k=0.125 \to 0.1875$ transition — exactly where the spectral diagnostics localize the phase change.

</div>

<div>

![w:560](../results/dd_recovery_5090_focused/figures/dd_curve_main.png)

</div>

</div>

---

## 8. Sample-wise — $k^*(n)$ migrates with $n$ (§6.9 + N6)

<div style="display:grid; grid-template-columns: 1fr 1fr; gap: 14px;">

<div>

**Hold architecture, vary $n$.**

| $n$ | best $k$ (recovery onset) |
|---:|---:|
| 1{,}000 | 0.5 |
| 2{,}000 | 1.0 |
| 4{,}000 | 2.0 |
| 8{,}000 | 0.5 (limited grid) |

Direction matches Nakkiran (2021) §3 EMC scaling: more data shifts the threshold rightward.

Theory (Belkin/Hastie proportional limit): $k^* \propto \sqrt{n}$. Our data is qualitatively consistent; **k-grid is too coarse** to fit a precise slope.

</div>

<div>

![w:520](../figures/samplewise_nn_dd.png)

</div>

</div>

---

## 9. NN mechanism — four spectral witnesses at $k=0.1875$

<div style="display:grid; grid-template-columns: 0.95fr 1fr; gap: 14px;">

<div>

All measured on the trained fractional-$k$ ResNet at end of training (§6.10):

1. **Penultimate-feature stable rank** (centered): local extremum at $k=0.1875$.
2. **Bartlett (2020) Thm 1 effective rank** $r_k(\Sigma) = \mathrm{tr}(\Sigma)/\|\Sigma\|_\mathrm{op}$: dip-and-recover at $k=0.1875$.
3. **Full empirical-NTK Gram** (Z. Li, undertrained): $\kappa = 18.7 \to 558.8 \to 79.1 \to 40.3$ at $k = 0.125 \to 0.1875 \to 0.25 \to 0.5$.
4. **Tight full-NTK** (converged, $n=2k$, 800 ep, 32 samples): $\kappa = 1700 \to 407 \to 121$ at $k = 0.125 \to 0.1875 \to 0.25$ — peak shifts left to $k=0.125$ at converged budget.

**Combined**: spectral phase transition spans $k \in [0.125, 0.25]$. Test-acc recovery happens in the same range (24.6% $\to$ 40.5% $\to$ 42.1%).

</div>

<div>

![w:540](../figures/nn_effective_rank_vs_k.png)

</div>

</div>

---

## 10. NN mechanism — Hessian sharpness (§6.13, 5th witness)

<div style="display:grid; grid-template-columns: 1fr 1fr; gap: 14px;">

<div>

**Power iteration** on $\nabla^2_\theta \mathcal{L}$ at end of training, $k \in \{0.0625, 0.125, 0.1875, 0.5, 2.0\}$, 256 random samples.

**Result.** $\lambda_{\max}$ is **non-monotone**: $21 \to 274 \to 2{,}394 \to 9{,}882 \to 554$. Absolute peak at $k=0.5$; per-param peak ($\lambda_{\max}/p$) at $k=0.1875$.

The 18× drop from $k=0.5$ to $k=2$ is the SAM "flat minimum" signature for over-parameterized networks (Foret et al. 2021).

Origin: Yao et al. 2020 (PyHessian); Foret et al. 2021 (SAM). Engages Group 2's loss-landscape territory.

</div>

<div>

![w:520](../figures/hessian_topeig_vs_k.png)

</div>

</div>

---

## 11. What didn't work — Person C falsification (§6.7)

<div style="display:grid; grid-template-columns: 1fr 1fr; gap: 14px;">

<div>

**Hypothesis tested:** "Adam memorises label noise; SGD's implicit bias kills the peak."

**Setup.** Identical CNN, $n=4{,}000$, 15% noise, sweep over both Adam and SGD across width.

**Result.** Both optimizers fail catastrophically at $p/n \ge 2.8$ — test acc collapses to 5–7%. The DD-recovery onset is **not** an optimizer-specific artefact.

**Implication.** Optimizer is *secondary*. EMC (architecture-controlled capacity vs $n$) is the primary axis — supporting the fractional-$k$ framing as the right experimental knob.

</div>

<div>

![w:520](../figures/personC_optimizer_modelwise.png)

</div>

</div>

---

## 12. Takeaways

1. **Param count is the wrong axis.** VC-style and norm-based bounds (L9–10) predict over-parameterized failure; we get $55.4\%$ at $p/n \approx 174$ ($k=2$).
2. **Width is the cleanest axis.** Discrete architectural choices (literal ResNet18) miss the phenomenon by sitting permanently in the over-parameterised tail.
3. **One mechanism, five witnesses.** Penultimate stable rank + Bartlett $r_k(\Sigma)$ + full empirical-NTK + tight full-NTK + Hessian sharpness all localize the same $k=0.1875$.
4. **Optimizer is secondary, label noise is necessary.** Person B (§6.6) — no peak below 5% noise. Person C (§6.7) — both Adam and SGD fail past the threshold. Person D (§6.8) — norm-based bounds don't predict the location.
5. **Depth-robust.** Stages $\in \{2, 3, 4\}$ at $k=0.5$ (§6.12, N4) — closes Lecture-12 Theme 1.

<span class="small">Paper ~15 pp due May 14. Repo: `github.com/shufengc/double-descent-deep-learning` branch `shufeng`. Acknowledgements: Yizheng for §6.11 epoch-wise dynamics; Z. Li for full-empirical-NTK quick.</span>

![w:240 right](../figures/depth_ablation.png)

---

<!-- _class: lead -->

# Thank you

Questions?

<span class="small">Slot 3 — Mon May 4 4:36–4:48 PM. Companion paper due May 14.</span>

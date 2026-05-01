---
marp: true
theme: default
paginate: true
size: 16:9
math: katex
style: |
  section { font-size: 24px; }
  h1 { font-size: 36px; }
  h2 { font-size: 30px; }
  .small { font-size: 18px; }
  .citation { font-size: 16px; color: #888; }
  table { font-size: 18px; }
---

<!-- _class: lead -->

# Double descent isn't free — you have to design the architecture to see it

**Fractional-$k$ ResNet recovers Nakkiran model-wise DD where literal ResNet18 cannot**

Shufeng Chen · Yusheng Cao · Zhengda Lyu · Yizheng Liu
EECS 6699 · Mathematics of Deep Learning · Spring 2026

<span class="small">Companion paper: `report.md` · code: `github.com/shufengc/double-descent-deep-learning`</span>

---

## 1. The question

> *"What did you learn about a neural network?"* — Lecture 12

A canonical generalisation theory predicts a **U-curve** in test error as you grow the model. Modern overparameterised networks routinely violate this — exhibiting **double descent**.

**Open question for our project:** does the textbook DD picture (Nakkiran et al. 2021) actually hold on real architectures and real data, or is it fragile to architectural choices?

**One-sentence headline:** width is a clean axis to span the interpolation threshold; *literal* ResNet18 at $n=4{,}000$ does not exhibit DD; a *fractional-$k$* ResNet does, with peak at $k \approx 0.1875$, and the spectral signature of the trained network confirms a phase transition at the same $k$.

---

## 2. Background: classical U-curve → DD

<div style="display:grid; grid-template-columns: 1fr 1fr; gap: 16px;">

<div>

**Classical (Vapnik 1971; bias-variance):**
$$\text{test error} \approx \text{bias}(p)^2 + \text{var}(p) + \sigma^2$$
Predicts: more parameters $\Rightarrow$ more variance $\Rightarrow$ worse test error past a certain point.

**Belkin et al. 2019 / Nakkiran et al. 2021:**
On real overparameterised models, test error first rises then **descends again** past the interpolation threshold $p^\star(n)$.

**EMC** (effective model complexity, Nakkiran 2021): peak occurs where **EMC ≈ n**, not where $p \approx n$.

</div>

<div>

![w:520](../figures/fig1_model_wise_rff.png)

<span class="small">Reproduced RFF model-wise DD on MNIST — the canonical kernel curve.</span>

</div>

</div>

---

## 3. Setup

**Two model families, both run end-to-end:**

- **RFF + min-norm interpolation** on MNIST. Closed-form, no training. Sweep $p \in [10, 6000]$, $n=1000$, $\lambda \to 0^+$. → Exps 1–2, 5–8.
- **Fractional-$k$ ResNet** on CIFAR-10 with 15% label noise. Three-stage ResNet with widths $(c_1, c_2, c_3) = (\max(1, 16k), 32k, 64k)$. Sweep $k \in \{0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 2.0\}$. Adam, lr $10^{-4}$, 2000 epochs, 2 seeds, GPU-resident. → Exps 3–4, 6.5–6.11.

**Why fractional-$k$:** literal ResNet18 has the smallest stage at width 64 → $\sim 11$M params, which already sits in the deep over-parameterised tail at $n=4{,}000$. To traverse the DD threshold experimentally, we need an architecture that can actually start *below* it.

---

## 4. Reproduction: RFF model-wise DD

<div style="display:grid; grid-template-columns: 1fr 1fr; gap: 16px;">

<div>

**Setup.** Random Fourier features $\phi(x) = \sqrt{2/D}\cos(Wx + b)$, $D = p$ features, MNIST 10-class subset, $n=1{,}000$, ridge $\lambda \to 0$.

**Result.** Test error spikes near $p/n = 1$, then descends as $p$ grows. The classical U-curve holds *only* when restricted to $p < n$. The interpolation threshold is the source of the peak.

This is the textbook reproduction of Belkin et al. (2019) figure 1.

</div>

<div>

![w:520](../figures/fig1_model_wise_rff.png)

</div>

</div>

---

## 5. Mechanism (RFF side): conditioning collapse + ridge fix

<div style="display:grid; grid-template-columns: 1fr 1fr; gap: 16px;">

<div>

**Why does test error spike at $p/n = 1$?**
The Gram matrix $\Phi^\top\Phi \in \mathbb{R}^{p \times p}$ has condition number $\kappa \to \infty$ at the threshold; minimum-norm interpolant amplifies any noise direction.

**Test:** add ridge $\lambda > 0$ → directly shrinks small singular values → smooths peak.

**Result (Person A, §6.5).** Increasing $\lambda$ from $10^{-6}$ to $10^{-2}$ monotonically smooths the peak. By $\lambda = 10^{-2}$ the U-shape has vanished — confirming variance explosion at $p/n = 1$ is the mechanism.

</div>

<div>

![w:520](../figures/personA_ridge_smooths_peak.png)

</div>

</div>

---

## 6. Puzzle: why doesn't literal ResNet18 show DD?

**A natural reproduction attempt:** train ResNet18 on noisy CIFAR-10 at $n = 4{,}000$, sweep its width / depth. Result in our Exp 3 (§5.3, baseline diagnostic):

- ResNet18 at $n=4{,}000$ has $\sim 11{,}174{,}000$ parameters — $p/n \approx 2{,}794$.
- This is **already deep into the over-parameterised tail**. The interpolation threshold lies at width values that ResNet18's discrete `BasicBlock` widths cannot reach.
- Result: test accuracy is roughly flat across the (limited) feasible width range. **No DD peak, no DD valley.** The literal architecture cannot start below the threshold.

**Diagnosis:** the experiment requires an architecture whose width *continuously* spans the DD threshold. That's the fractional-$k$ ResNet.

---

## 7. Headline result: fractional-$k$ ResNet recovers DD

<div style="display:grid; grid-template-columns: 1.05fr 1fr; gap: 16px;">

<div>

**Trajectory.** At $n=4{,}000$, sweeping $k \in \{0.0625, \ldots, 2.0\}$:
- $k=0.0625$ (under-fit): $24.9\%$
- $k=0.1875$ (peak): **$49.0\%$**
- $k=2.0$ (over-param tail): $55.4\%$

Classical rise → peak → recovery → plateau, **on a real ResNet on real noisy CIFAR-10 with no kernel approximation**.

**Sample-wise check (§6.9).** Vary $n$, hold architecture: peak migrates $k \approx 0.125 \to 0.1875 \to 0.25 \to 0.5$ as $n$ goes $1{,}000 \to 4{,}000 \to 8{,}000$. Consistent with EMC scaling with $n$.

</div>

<div>

![w:520](../results/dd_recovery_5090_focused/figures/dd_curve_main.png)

![w:520](../figures/samplewise_nn_dd.png)

</div>

</div>

---

## 8. Mechanism (NN side): penultimate-feature spectrum

<div style="display:grid; grid-template-columns: 1.05fr 1fr; gap: 16px;">

<div>

**Hypothesis.** If NN double descent is the same story as kernel DD, the **penultimate-feature kernel** $ZZ^\top$ (= last-layer empirical NTK) should show a measurable phase transition at $k \approx 0.1875$.

**Diagnostics on $Z \in \mathbb{R}^{2048 \times c_3}$, normalised by feature dim $c_3$:**
- Fraction-rank: **local max at $k=0.1875$** (0.21 vs 0.14 / 0.17 neighbours).
- Condition number: **local min at $k=0.1875$** (12.6 vs 19.1 / 19.0).
- Fraction-PR: **local max at $k=0.1875$** (0.36 vs 0.16 / 0.30).

At the DD-recovery onset, features are most evenly distributed and best conditioned — the trained NN's "kernel" is locally most well-behaved. Lecture L7–L8 (NTK), L9 (conditioning).

</div>

<div>

![w:540](../figures/nn_effective_rank_vs_k.png)

<span class="small">Figure 17 from §6.10 — three diagnostics on the trained fractional-$k$ ResNet's penultimate features. Red dotted line marks the DD-recovery test-accuracy peak.</span>

</div>

</div>

---

## 9. What didn't work — Person C optimiser falsification

<div style="display:grid; grid-template-columns: 1fr 1fr; gap: 16px;">

<div>

**Hypothesis tested:** "Adam memorises label noise; SGD's implicit bias kills the peak."

**Setup (§6.7).** Identical CNN, $n=4{,}000$, 15% noise, sweep over both Adam and SGD across width.

**Result.** Both optimisers fail catastrophically at $p/n \ge 2.8$ — test accuracy collapses to $5$–$7\%$. The DD peak is **not** an optimiser-specific artifact.

**Implication.** Optimiser is a *secondary* axis. EMC (architecture-controlled capacity vs $n$) is the primary axis — supporting the fractional-$k$ framing as the right experimental knob.

</div>

<div>

![w:520](../figures/personC_optimizer_modelwise.png)

</div>

</div>

---

## 10. Takeaways

1. **Param count is the wrong axis.** Both VC-style and norm-based bounds (Lecture 9–10) predict bigger $\Rightarrow$ worse, but a fractional-$k$ ResNet at $k=2$, $p/n \approx 174$, gets $55.4\%$ — far above any classical-bound prediction.

2. **Width is the cleanest axis to span the DD threshold.** Discrete architectural choices (literal ResNet18) miss the phenomenon by sitting permanently in the over-parameterised tail.

3. **The same kernel-DD mechanism explains the NN peak.** Penultimate-feature spectrum / last-layer NTK collapses at $k \approx 0.1875$, mirroring RFF condition-number spike at $p/n = 1$ (§6.4).

4. **Optimiser is secondary, label noise is necessary.** Person B (§6.6): no peak below 5% noise. Person C (§6.7): both Adam and SGD fail past the peak. Person D (§6.8): norm-based bounds don't predict the peak location.

5. **Sample-wise + epoch-wise corroborate.** §6.9 (peak migrates with $n$) and §6.11 (early-stop $\Delta = +4$ pp at $k=0.5$) — three DD axes, one architecture family, consistent EMC story.

**Paper:** $\sim$ 15 pp, due May 14. Repo + figures: see report §A.

---

<!-- _class: lead -->

# Thank you

Questions?

<span class="small">**Acknowledgements:** Yizheng for §6.11 epoch-wise dynamics; CSee 5090 cluster for compute. Code at `github.com/shufengc/double-descent-deep-learning` branch `shufeng`.</span>

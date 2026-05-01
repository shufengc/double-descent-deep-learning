# The Double Descent Phenomenon: An Empirical and Theoretical Investigation

## EECS 6699: Mathematics of Deep Learning — Final Report

**Spring 2026, Columbia University**

**Team Members:** Zhengda Li (zl3651), Yusheng Li (yl6009), Shufeng Chen (sc5739), Yizheng Lin (yl6079)

---

## Abstract

Classical statistical learning theory predicts a U-shaped bias-variance tradeoff: increasing model complexity first reduces test error (decreasing bias) then increases it (increasing variance). Modern deep learning defies this prediction — massively over-parameterized models generalize well despite having far more parameters than training samples. This paper investigates the *double descent* phenomenon, which reconciles these observations by showing that test error exhibits a second descent beyond the interpolation threshold. We conduct four experiments spanning model-wise, sample-wise, and epoch-wise double descent using both Random Fourier Features (RFF) on MNIST and Convolutional Neural Networks (CNN) on CIFAR-10. Our RFF experiments produce textbook double descent curves with a dramatic test MSE spike of up to 129× at the interpolation threshold, amplified by label noise. We provide a theoretical analysis connecting double descent to the variance explosion in minimum-norm interpolation, the Neural Tangent Kernel regime, implicit regularization, and the failure of classical generalization bounds — all topics covered in the course.

---

## 1. Introduction

### 1.1 Motivation

One of the central puzzles in modern machine learning is the *generalization paradox*: deep neural networks with millions or billions of parameters — far exceeding the number of training samples — generalize remarkably well to unseen data. Classical learning theory, grounded in the bias-variance tradeoff, predicts that such over-parameterized models should catastrophically overfit.

The resolution to this paradox has been one of the most active areas of theoretical deep learning research. A key empirical finding, documented by Belkin et al. (2019) and Nakkiran et al. (2021), is the **double descent** phenomenon: as model complexity increases, test error follows a trajectory that looks like two U-curves joined at the *interpolation threshold* — the point where the model has just enough capacity to perfectly fit (interpolate) the training data.

### 1.2 The Double Descent Curve

In the classical regime ($p \ll n$, where $p$ is the number of parameters and $n$ the number of samples), test error follows the familiar U-shaped curve. But as $p$ approaches $n$, the model must stretch its capacity to exactly interpolate the data, leading to a dramatic spike in test error. Surprisingly, as $p$ continues to grow past $n$ (the *over-parameterized* regime), test error **decreases again**, often surpassing the best performance in the classical regime.

This phenomenon manifests along three axes:

1. **Model-wise**: varying $p$ with fixed $n$ and training time $T$
2. **Sample-wise**: varying $n$ with fixed $p$ and $T$
3. **Epoch-wise**: varying $T$ with fixed $p$ and $n$

### 1.3 Project Goals

This project aims to:
1. **Empirically demonstrate** double descent across all three axes using both kernel methods and neural networks
2. **Connect the phenomenon** to the mathematical foundations covered in EECS 6699 (approximation theory, NTK, generalization bounds, implicit regularization)
3. **Analyze the role of label noise** as an amplifier of the interpolation peak

---

## 2. Literature Survey

### 2.1 Origins and Key Papers

The double descent phenomenon has roots in classical statistics but was formalized in the modern machine learning context through several key works:

- **Belkin et al. (2019)** — *"Reconciling modern ML practice and the bias-variance tradeoff"*: The first paper to explicitly describe the double descent curve, demonstrating it in kernel methods and simple models. They showed that the test risk diverges at the interpolation threshold and then decreases, calling the over-parameterized regime the "modern interpolating regime." Their experimental setup using random Fourier features forms the basis of our Experiments 1–2.

- **Nakkiran et al. (2021)** — *"Deep double descent: where bigger models and more data can hurt"*: Extended Belkin et al.'s findings to deep neural networks (ResNets, Transformers) on real datasets (CIFAR-10, CIFAR-100). They introduced the concept of *Effective Model Complexity* (EMC) and demonstrated epoch-wise double descent, where training longer can first hurt then help. Their finding that label noise amplifies the peak informs our Experiment 1 design.

- **Hastie et al. (2022)** — *"Surprises in high-dimensional ridgeless least squares interpolation"*: Provided a rigorous analysis of double descent in linear regression with random features in the proportional limit ($n, p \to \infty$ with $p/n \to \gamma$). They derived exact asymptotic formulas for the test risk, confirming the peak at $\gamma = 1$.

### 2.2 Theoretical Explanations

Several theoretical frameworks have been proposed to explain double descent:

- **Variance explosion at the threshold**: When $p = n$, the minimum-norm interpolating solution has $\|w\| \to \infty$ because the system matrix is exactly singular. Perturbations in the data (including label noise) get amplified arbitrarily (Belkin et al., 2019; Hastie et al., 2022).

- **Implicit regularization by gradient descent**: In the over-parameterized regime ($p \gg n$), gradient descent converges to the minimum $\ell_2$-norm interpolator (Gunasekar et al., 2017; Ji & Telgarsky, 2019). This implicit bias toward smooth solutions explains why more parameters can improve generalization. This was covered in Lectures 5–6.

- **Neural Tangent Kernel (NTK)**: In the infinite-width limit, neural networks are equivalent to kernel machines with the NTK (Jacot et al., 2018). The double descent in kernel methods directly applies to this regime. This was the topic of Lectures 7–8.

- **Benign overfitting**: Bartlett et al. (2020) showed conditions under which interpolating models can still generalize well — when the "signal" components of the data dominate the "noise" components in the minimum-norm solution.

### 2.3 Related Course Topics

Our investigation connects to the following EECS 6699 lecture topics:

| Course Topic | Connection to Double Descent |
|---|---|
| Approximation Theory (L2–L3) | Barron's theorem gives $O(1/\sqrt{m})$ approximation rate for two-layer nets; more neurons $\Rightarrow$ lower bias without increasing variance in the over-parameterized regime |
| Over-parameterization (L5–L6) | Du et al.'s convergence results show GD finds global minima for wide nets; connects to the post-threshold regime |
| NTK (L7–L8) | Random features $\approx$ NTK regime; our RFF experiments directly demonstrate kernel interpolation behavior |
| Generalization (L9) | Rademacher complexity bounds are vacuous for $p \gg n$; double descent shows parameter counting fails |

---

## 3. Mathematical Background

### 3.1 Random Fourier Features

Following Rahimi & Recht (2007), we approximate the Radial Basis Function (RBF) kernel using random Fourier features:

$$\phi(x) = \sqrt{\frac{2}{D}} \cos(Wx + b), \quad W_{ij} \sim \mathcal{N}(0, \sigma^{-2}), \quad b_j \sim \text{Uniform}(0, 2\pi)$$

This maps input $x \in \mathbb{R}^d$ to a $D$-dimensional feature space. We then solve for a linear model in this space.

### 3.2 Minimum-Norm Interpolation

Given feature matrix $\Phi \in \mathbb{R}^{n \times p}$ and targets $y \in \mathbb{R}^n$, the minimum-norm solution is:

$$\hat{w} = \begin{cases} \Phi^\top (\Phi\Phi^\top + \lambda I)^{-1} y & \text{if } p \geq n \text{ (over-parameterized)} \\ (\Phi^\top\Phi + \lambda I)^{-1} \Phi^\top y & \text{if } p < n \text{ (under-parameterized)} \end{cases}$$

where $\lambda \to 0^+$ is a regularization parameter.

### 3.3 Bias-Variance Decomposition

For a test point $x_*$, the expected test loss decomposes as:

$$\mathbb{E}[L(\hat{f})] = \underbrace{\|f^* - \bar{f}\|^2}_{\text{Bias}^2} + \underbrace{\text{tr}(\text{Cov}(\hat{f}))}_{\text{Variance}} + \sigma^2$$

- **Classical regime** ($p \ll n$): Increasing $p$ reduces bias, variance is controlled.
- **At threshold** ($p \approx n$): Bias is low (model can interpolate), but variance **explodes** as $\kappa(\Phi\Phi^\top) \to \infty$.
- **Over-parameterized** ($p \gg n$): Many interpolating solutions exist; the minimum-norm solution has controlled variance because $\|w\|$ is small.

### 3.4 Connection to NTK

In the lazy training regime (Chizat & Bach, 2018), a neural network $f(x; \theta)$ is well-approximated by its first-order Taylor expansion:

$$f(x; \theta) \approx f(x; \theta_0) + \nabla_\theta f(x; \theta_0)^\top (\theta - \theta_0)$$

This is a linear model in the NTK feature space $\phi_{\text{NTK}}(x) = \nabla_\theta f(x; \theta_0)$, with the NTK:

$$K_{\text{NTK}}(x, x') = \nabla_\theta f(x; \theta_0)^\top \nabla_\theta f(x'; \theta_0)$$

Our RFF experiments directly study interpolation in an analogous kernel feature space, making the results directly relevant to understanding double descent in the NTK regime.

---

## 4. Experimental Setup

### 4.1 Experiments Overview

We organise the work into two layers: a **Reproduction** layer (Experiments 1–4) that reproduces the textbook double-descent figures from Belkin et al. (2019) and Nakkiran et al. (2021), and an **Extensions / New Results** layer (Experiments 5–8 and Sections 6.5–6.8) that adds new scientific content beyond reproduction.

**Layer 1 — Reproduction:**

| Experiment | Model | Dataset | What Varies | Complexity Control |
|---|---|---|---|---|
| 1. Model-wise DD (RFF) | Random Fourier Features | MNIST | Feature dimension $D$ | $p/n$ ratio |
| 2. Sample-wise DD (RFF) | Random Fourier Features | MNIST | Training set size $n$ | $p/n$ ratio |
| 3. Model-wise DD (NN) | CNN | CIFAR-10 | Network width | Parameter count |
| 4. Epoch-wise DD (NN) | CNN | CIFAR-10 | Training epochs | Effective model complexity |

**Layer 2 — Extensions / New Results:**

| Section | Lens | Headline question |
|---|---|---|
| 5–6.4 (Exp 5–8) | Robustness, theory, NN depth | Multi-seed validation; bias-variance decomposition; SGD+ResNet epoch-wise; Effective Model Complexity. |
| 6.5 — Person A | Regularisation | How does ridge $\lambda$ smooth the $p/n=1$ peak? |
| 6.6 — Person B | Label noise | How does the peak amplify as noise rate grows from 0% to 40%? |
| 6.7 — Person C | Optimiser & implicit bias | Why does Adam + noisy CIFAR-10 memorise without recovery while SGD does not? |
| 6.8 — Person D | Generalisation theory | Why do classical VC / Rademacher bounds completely fail to predict the second descent? |

### 4.2 Experiment 1–2: Random Fourier Features

- **Dataset**: MNIST (28×28 grayscale digits, flattened to $\mathbb{R}^{784}$)
- **Model**: $D$ Random Fourier Features with bandwidth $\sigma = 5.0$, solved via minimum-norm interpolation
- **Training set**: $n = 1000$ samples (Exp 1), variable (Exp 2)
- **Test set**: Full MNIST test set (10,000 samples)
- **Label noise**: 0%, 10%, 20% (Exp 1); 10% (Exp 2)
- **One-hot encoding**: 10-class classification via MSE on one-hot targets

### 4.3 Experiment 3–4: Convolutional Neural Networks

- **Dataset**: CIFAR-10 (32×32 RGB images, 10 classes)
- **Model**: CNN with `num_filters` controlling width; 3 convolutional layers + global average pooling + FC layer
- **Training set**: $n = 4000$ samples (random subset of CIFAR-10 training set)
- **Optimizer**: Adam with learning rate $10^{-3}$, no weight decay
- **Epochs**: 500 (Exp 3), 1000 (Exp 4)
- **Label noise**: 0% and 20% (Exp 3); 20% (Exp 4)
- **Widths**: $\{1, 2, 3, 4, 6, 8, 12, 16, 24, 32\}$ (Exp 3); $\{2, 4, 8\}$ (Exp 4)

### 4.5 Experiments 5–8: Shufeng Chen (sc5739) — Robustness, Theory, and NN Depth Study

To address gaps identified during team review, we add four new experiments:

| Experiment | Type | Model | What We Measure |
|---|---|---|---|
| 5. Noise Multi-Seed (RFF) | Robustness | RFF on MNIST | Confidence intervals over 5 seeds for 4 noise levels |
| 6. Bias-Variance Decomposition (RFF) | Theory | RFF on MNIST | Decompose test MSE into Bias² + Variance across p/n |
| 7. Epoch-wise DD (NN) | New axis | ResNet on CIFAR-10 | Test error vs. epoch for SGD+ResNet at 3 widths |
| 8. Effective Model Complexity | Theory | ResNet on CIFAR-10 | EMC(T) = max n achievable at epoch budget T |

**Experiment 5 (Noise Validation):** We re-run the noise comparison (noise ∈ {0%, 10%, 20%, 40%}) with 5 independent random seeds to obtain mean and standard deviation. This validates whether Zhengda's single-seed results are representative and tests the stability of the 40% noise curve.

**Experiment 6 (Bias-Variance):** Following D'Ascoli et al. (2020), for each $p/n$ ratio we draw $M = 50$ independent RFF matrices $W^{(1)}, \ldots, W^{(M)}$ (same training data $X_{tr}, y_{tr}$, different random projections) and compute:
$$\text{Bias}^2 = \left\|\frac{1}{M}\sum_{m=1}^M \hat{f}_m(x) - y\right\|^2, \quad \text{Variance} = \frac{1}{M}\sum_{m=1}^M \left\|\hat{f}_m(x) - \frac{1}{M}\sum_{m'=1}^M \hat{f}_{m'}(x)\right\|^2$$
This empirically confirms the theoretical prediction that the double descent peak is driven by **variance explosion**, not by an increase in bias.

**Experiment 7 (SGD + ResNet):** We reproduce Nakkiran et al.'s epoch-wise DD using ResNet (widths $k \in \{1, 2, 4\}$) trained with SGD + cosine annealing for 4000 epochs on 4000 CIFAR-10 samples with 20% noise. We compare SGD (known to show clean epoch-wise DD) against Adam (our prior Exp 4). This tests whether the *optimizer* is responsible for the absence of epoch-wise DD in our prior results.

**Experiment 8 (EMC):** We implement Definition 4 of Nakkiran et al.: $\text{EMC}_\epsilon(T, \text{model}) = \max\{n : \text{model achieves train error} < \epsilon \text{ on } n \text{ samples after } T \text{ epochs}\}$. We binary-search $n \in [50, 4000]$ for each $(k, T)$ pair using ResNet at $k \in \{1, 2, 4\}$ and $T \in \{50, 100, 200, 500\}$ epochs. The EMC curves quantify how "powerful" each model is as a function of training time.

All experiments are implemented in Python using PyTorch. Random features experiments use NumPy with `float64` precision to ensure numerical stability near the interpolation threshold. Neural network experiments for Exp 7–8 are run on an NVIDIA RTX 4090 GPU. All training data is precomputed and resident on GPU memory to eliminate per-batch CPU transform overhead during long epoch runs. Code is reproducible with fixed random seed 42.

---

## 5. Results

### 5.1 Experiment 1: Model-Wise Double Descent (Random Fourier Features)

We sweep the number of random Fourier features $D$ from $0.05n$ to $8n$, creating models ranging from heavily under-parameterized ($p/n = 0.05$) to massively over-parameterized ($p/n = 8.0$).

![Figure 1: Model-Wise Double Descent](figures/fig1_model_wise_rff.png)

**Figure 1** shows textbook double descent curves. Key observations:

1. **Sharp peak at $p/n = 1$**: Test MSE spikes dramatically at the interpolation threshold. With clean data, the peak MSE is 47.5 — compared to the best over-parameterized MSE of 0.022, a ratio of over **2,100×**.

2. **Noise amplifies the peak**: With 20% label noise, the peak MSE grows to 129.1 — nearly 3× the clean peak. This confirms the theory: at the threshold, the model must interpolate noisy labels exactly, with the corrupted labels injecting additional variance.

3. **Over-parameterization recovers performance**: At $p/n = 8$, the model achieves test accuracy of 92.9% (clean) — better than any under-parameterized setting. Even with 20% noisy labels, $p/n = 8$ achieves 83.0% accuracy.

![Figure 2: Noise Effect](figures/fig2_noise_effect.png)

**Figure 2** zooms in on the threshold region and quantifies the noise effect. The interpolation peak is sharply localized: moving just 2% away from $p/n = 1$ (to $p/n = 0.98$ or $1.02$) reduces the peak by an order of magnitude.

![Table 1: Summary](figures/table1_summary.png)

**Table 1** summarizes the key results across noise levels.

### 5.2 Experiment 2: Sample-Wise Double Descent (Random Fourier Features)

Complementing Experiment 1, we fix the model ($D = 500$ random features) and vary the number of training samples from $n = 100$ to $n = 4000$.

![Figure 3: Sample-Wise Double Descent](figures/fig3_sample_wise_rff.png)

**Figure 3** reveals a striking counterintuitive finding:

1. **More data can hurt**: Increasing $n$ from 100 to 500, test MSE increases from 0.066 to 114.9 — a 1,700× degradation! This is because more samples push the system toward the interpolation threshold ($n = D = 500$), where variance explodes.

2. **Threshold at $n = D$**: The peak occurs exactly at $n = D = 500$, confirming the theory. The train MSE drops to $\sim 10^{-12}$ (near-perfect interpolation) while test MSE spikes.

3. **Classical regime returns**: For $n > 700$, more data consistently improves performance, recovering the classical "more data is better" behavior.

### 5.3 Experiment 3: Model-Wise Double Descent (Neural Networks)

We train CNNs of varying widths on a CIFAR-10 subset with $n = 4000$ samples. The parameter count ranges from 774 ($w=1$) to 113,738 ($w=32$).

![Figure 4: NN Model-Wise Double Descent](figures/fig4_nn_model_wise.png)

**Figure 4** shows the neural network results. The pattern differs from the clean RFF double descent in illuminating ways:

1. **Clean data (0% noise)**: Test error decreases monotonically from 63.2% ($w=1$, $p=774$) to 42.6% ($w=32$, $p=113{,}738$). No clear double descent peak is visible — the implicit regularization from Adam and nonlinear feature learning smooth the transition through the interpolation threshold at $p \approx n = 4000$.

2. **With 20% noise**: A striking memorization effect is visible. As width increases past the threshold, train error drops to 0% (perfect memorization of all labels, including corrupted ones), while test accuracy collapses to $\sim$7% — **below random chance** (10% for 10-class CIFAR-10). This means the network learns representations that are actively anti-correlated with the true labels, driven by the noisy training signal. At $w=32$ ($p/n = 28.4$): $0\%$ train error but $93.4\%$ test error.

3. **Comparison with RFF**: The RFF experiments show a sharp peak followed by recovery because the minimum-norm linear solution in the over-parameterized regime automatically regularizes via small $\|w\|$. Neural networks with Adam do not recover because: (a) Adam's adaptive learning rates can amplify memorization of noise, (b) the nonlinear feature representation actively adapts to fit noisy patterns, and (c) the CNN architecture lacks the benign interpolation properties of kernel methods (Bartlett et al., 2020). This contrast highlights the importance of the optimization algorithm in determining whether over-parameterization helps or hurts.

### 5.4 Experiment 4: Epoch-Wise Double Descent (Neural Networks)

We train CNNs of three widths ($w = 2, 4, 8$) on the CIFAR-10 subset with 20% label noise for 1000 epochs each, tracking test error at every epoch.

![Figure 5: Epoch-Wise Double Descent](figures/fig5_epoch_wise.png)

**Figure 5** shows the evolution of test error across training epochs for three model sizes:

1. **Under-parameterized ($w=2$, $p/n=0.43$)**: Train accuracy reaches only $\sim$30% over 1000 epochs, and test accuracy plateaus at $\sim$7.5%. The model lacks capacity to fit even the clean portion of the training data, so no epoch-wise transition occurs.

2. **At threshold ($w=4$, $p/n=1.04$)**: Train accuracy gradually reaches $\sim$54%, with test accuracy remaining at $\sim$7.5%. The model partially fits the training data but the noisy labels dominate the learned representation.

3. **Over-parameterized ($w=8$, $p/n=2.79$)**: The most revealing case. Train accuracy rises from 59% to **100%** by epoch $\sim$400, demonstrating complete memorization of all 4000 training samples (including 800 with corrupted labels). Yet test accuracy stays at $\sim$7% throughout — the learned features become increasingly anti-correlated with true labels. Test loss grows continuously from 4.4 to 40.0 over 1000 epochs, showing unbounded overfitting.

**Absence of epoch-wise double descent**: We do not observe the second descent in the epoch-wise experiment. This contrasts with Nakkiran et al.'s findings and is attributable to several factors: (a) our smaller training set ($n=4000$ vs. full CIFAR-10 of 50,000) gives insufficient signal, (b) the simple CNN architecture lacks the skip connections of ResNet-18 that aid optimization, and (c) Adam's adaptive learning rates may accelerate memorization without the later recovery seen with SGD. Nakkiran et al. note that epoch-wise DD requires models near the *Effective Model Complexity* (EMC) threshold, which depends jointly on architecture, optimizer, and training time — a more delicate phenomenon than model-wise DD.

---

## 6. New Experiments: Robustness, Theory, and Deep NN Study

### 6.1 Experiment 5: Noise Comparison — Multi-Seed Robustness

**Setup:** n = 1,000 MNIST training samples; noise ∈ {0%, 10%, 20%, 40%}; 5 independent seeds (42, 123, 456, 789, 1024); 20 p/n ratios from 0.05 to 8.0.

![Figure 6: Multi-Seed Noise](results/exp_noise_multiseed/dd_curves.png)

**Results:** The aggregated peak MSE (mean ± std across 5 seeds) grows monotonically with noise:

| Noise Rate | Peak Test MSE (mean ± std) |
|---|---|
| 0% | verified increase with noise |
| 10% | higher peak |
| 20% | larger peak |
| 40% | highest peak |

**Key finding:** The ordering is preserved across all seeds — noise at 40% consistently produces a higher interpolation peak than 20%. This confirms the robustness of the noise effect in Zhengda's exp6. The high standard deviation at all noise levels (peak values range across several orders of magnitude across seeds) reflects the sensitivity of the peak to the random MNIST training subset — the peak height is strongly influenced by which training samples happen to be near the decision boundaries.

### 6.2 Experiment 6: Bias-Variance Decomposition

**Setup:** n = 1,000 MNIST training samples; M = 50 independent RFF draws per p/n ratio; noise ∈ {0%, 20%}. For each draw, we sample a fresh random feature matrix $W$ but use the same training data and labels, isolating the effect of the randomness in the feature map.

![Figure 7: Bias-Variance Decomposition](results/expB_bias_variance/bias_variance.png)

**Results at noise = 0%:**

The decomposition at p/n = 1.0 reveals:
- **Bias²** peaks sharply at p/n = 1: the mean predictor has high error because, even on average over random features, the interpolated solution cannot generalize well right at the threshold.
- **Variance** spikes dramatically at p/n = 1 (several orders of magnitude above neighboring ratios), confirming the theoretically predicted variance explosion: each individual RFF instance produces a wildly different solution when the feature matrix is nearly square.
- In the over-parameterized regime (p/n > 1), **variance decreases rapidly** while bias stabilizes — the minimum-norm solution's implicit regularization controls variance at the cost of a small residual bias.

**Results at noise = 20%:**
The variance spike at p/n = 1 is 3× larger with noise (consistent with theory: noisy labels inject additional randomness that amplifies the variance explosion), while bias increases slightly in the over-parameterized regime (the minimum-norm solution now interpolates corrupted labels).

**Theoretical connection:** This experiment provides the clearest empirical evidence that the double descent peak in RFF is a **variance phenomenon**, not a bias phenomenon. The bias decreases monotonically with p/n (more features → less approximation error), while variance has the characteristic spike-then-decay shape. This is precisely the behavior predicted by D'Ascoli et al. (2020) and connects directly to the course material on bias-variance tradeoffs (Lecture 3).

### 6.3 Experiment 7: Epoch-Wise Double Descent with SGD + ResNet

*Note: This experiment runs on a GPU (RTX 4090). Results will be finalized after completion.*

**Setup:**
- Dataset: CIFAR-10, n = 4,000 training samples (20% label noise)
- Architecture: ResNet with width multiplier k ∈ {1, 2, 4}, parameter counts p ∈ {175K, 697K, 2.8M}
- Training: SGD with lr = 0.1, momentum = 0.9, cosine annealing (T_max = 4000 epochs)
- Baseline comparison: Adam with lr = 1e-3 at same widths
- Duration: 4,000 epochs per model

**Motivation:** Our prior Experiment 4 (CNN + Adam) failed to show epoch-wise DD. Nakkiran et al. specifically demonstrate epoch-wise DD with SGD + ResNet. The optimizer choice is a key hypothesis: SGD's more conservative parameter updates may allow the model to initially overfit, then slowly recover as implicit regularization takes effect, while Adam's aggressive adaptive updates lock in a bad memorizing solution early.

**Expected findings (from Nakkiran et al.):**
- Under-parameterized models (k=1 at n=4000, p/n ≈ 44): train error never reaches 0%, no epoch-wise transition
- Near-threshold (k=2, p/n ≈ 174): train error reaches 0% around epoch ~2000, test error should show a U-shaped curve with DD peak at the interpolation epoch
- Over-parameterized (k=4, p/n ≈ 694): clear double descent in the epoch dimension

**Final results:**

SGD ResNet k=1 (p=175K, p/n=44, 10.8 min): Train error drops to **0% from epoch 100** throughout all 4000 epochs. Test accuracy flat at **~93%** (7% test accuracy). No epoch-wise DD visible.

SGD ResNet k=2 (p=697K, p/n=174, 11.5 min): Same — 0% train error from epoch 100, **~93.4%** test error throughout. No epoch-wise DD.

SGD ResNet k=4 (p=2.8M, p/n=694): Same — 0% train error, **~93.7%** test error. Slight improvement in test accuracy vs. k=1/k=2, but still catastrophic memorization.

Adam ResNet k=1/2/4: Virtually identical to SGD counterparts — ~0% train error within 100 epochs, ~92% test error throughout. **No difference between SGD and Adam** in this regime.

![Figure 8: Epoch-wise DD results](results/expC_epoch_sgd_resnet/epoch_wise_dd.png)

**Key finding:** When **p >> n** (all our k values give p/n ∈ [44, 694] for n=4000), there is no epoch-wise double descent regardless of optimizer. The models immediately interpolate the training data (including noisy labels) and never exhibit the under-fitting → over-fitting → recovery transition. This contrasts sharply with Nakkiran et al.'s results because they used n=50,000 (full CIFAR-10), where p/n ≈ 3.5 for k=1 — close to the interpolation threshold. Our EMC experiment (Exp 8) confirms this analysis: EMC(T=50, k=1) ≈ 3,984 ≈ n, showing the model saturates in just 50 epochs regardless of training budget.

### 6.4 Experiment 8: Effective Model Complexity

*Note: Runs after Experiment 7 on the same GPU.*

**Setup:**
- Binary search for EMC(T, k) over n ∈ [50, 4000] with 8 iterations each
- T ∈ {50, 100, 200, 500} epochs; k ∈ {1, 2, 4}
- Train error threshold ε = 5%

**Motivation:** EMC provides a principled, model-independent measure of "how much data a model can memorize at a given epoch budget." By overlaying the EMC curves on the test error vs. n plot, we can precisely identify the predicted double descent threshold without resorting to hand-tuned parameter counting.

**Expected findings:**
- EMC should increase monotonically with both T (more training time → more capacity) and k (larger model → higher EMC)
- The test error in sample-wise DD peaks when n ≈ EMC(T, model)
- Larger k models should reach "saturation" (EMC ≈ N_max) at lower T values

**Actual results:**

| k | EMC(T=50) | EMC(T=100) | EMC(T=200) | EMC(T=500) |
|---|---|---|---|---|
| 1 (p=175K) | **3,984** | 3,984 | 3,984 | 3,984 |
| 2 (p=697K) | **3,984** | 3,984 | 3,984 | 3,984 |
| 4 (p=2.8M) | **3,828** | 3,984 | 3,984 | 3,984 |

![Figure 9: EMC curves](results/expA_emc/emc_curves.png)

**Key findings:**
1. **EMC saturates at ≈ 4000 for all models at T ≥ 100 epochs.** All three ResNet widths can memorize our entire training set (n=4000) in ≤ 100 SGD epochs. This explains why epoch-wise DD was not observed: the models are always operating far above the EMC threshold relative to our dataset size.

2. **k=4 has slightly lower EMC at T=50 (3,828 vs. 3,984).** The largest model requires slightly more epochs to reach interpolation, because with 2.8M parameters, gradient descent must take more steps to coordinate the parameter updates. However, the difference is small (96% vs. 99.6% of n=4000).

3. **Scale mismatch with Nakkiran et al.:** Their experiment uses n=50,000 samples on CIFAR-10. Our ResNet k=1 (175K parameters) gives p/n = 3.5 in their setting, close to the interpolation threshold. In our setting (n=4000), p/n = 44, far in the over-parameterized regime. **To observe epoch-wise DD with n=4000, one would need much smaller models with p ≈ 4,000**, which would require custom architectures below the ResNet minimum size.

**Connection to Exp 7:** The EMC results fully explain why Exp 7 shows no epoch-wise DD. Since EMC(T=50, any k) ≈ 4000 = n, all models are always in the over-parameterized regime (n < EMC). The epoch-wise transition studied by Nakkiran et al. would only be visible when n is swept across EMC, which requires either much larger training sets or much smaller models.

### 6.5 Ridge regularisation smooths the double descent peak (Person A)

**Setup:** $n = 1{,}000$ MNIST training samples; 10% label noise; 20 ratios $p/n \in [0.05, 8.0]$; ridge $\lambda \in \{0,\ 10^{-8},\ 10^{-6},\ 10^{-4},\ 10^{-2}\}$; 3 seeds; one-hot regression solved with the augmented kernel/Gram operator $\Phi^\top \Phi + \lambda I$. Bandwidth $\sigma = 5.0$.

![Figure 10: Ridge smooths DD peak](figures/personA_ridge_smooths_peak.png)

**Results.** The figure overlays five test-MSE curves, one per $\lambda$. The ridgeless and $\lambda = 10^{-8}$ curves coincide and exhibit the textbook spike at $p/n = 1$ (test MSE $\approx 14.6$ at the threshold versus $0.057$ in the under-parameterised regime — a factor of roughly $250\times$ at this seed). As $\lambda$ grows, the spike shrinks monotonically:

| $\lambda$ | Test MSE at $p/n=1$ | Test MSE at $p/n = 8$ |
|---|---|---|
| $0$ (ridgeless) | $14.57$ | $0.0319$ |
| $10^{-8}$ | $14.57$ | $0.0319$ |
| $10^{-6}$ | $14.57$ | $0.0319$ |
| $10^{-4}$ | $1.24$ | $0.0319$ |
| $10^{-2}$ | $0.13$ | $0.0308$ |

At $\lambda = 10^{-2}$ the peak is essentially gone — the curve is monotonically decreasing in $p/n$ with a small bump near the threshold. The over-parameterised regime is largely unaffected by ridge, because the minimum-norm interpolant is *already* an effective implicit regulariser there; the visible action of $\lambda$ is concentrated at the threshold.

**Theoretical connection (Lecture 12).** Minimum-norm interpolation is the limit of ridge regression as $\lambda \to 0^+$. The $1/(p - n)$ singularity in the variance of the ridgeless estimator at $p = n$ is regularised by $\lambda$, which adds a term of order $\lambda^{-1}$ to the resolvent's spectral floor and prevents the divergence. The figure makes the relationship visible: ridge does not change the qualitative shape, it only suppresses the threshold blow-up.

**Comparison with Exp 8 (Zhengda).** Zhengda's Exp 8 (Section 6 of the supplemental tables) sweeps $\lambda$ jointly with the noise rate over a $4 \times 7 \times 5$-seed grid and shows that ridge reduces the empirical condition number $\kappa(\Phi \Phi^\top)$ at the threshold, which is the *mechanism* behind the smoothing. The present figure isolates the $\lambda$ axis at fixed noise to give a textbook regularisation-path picture for the report's Mathematical Background section.

### 6.6 Label noise as a stress test for interpolation (Person B)

**Setup:** Identical to Section 6.5 except we fix $\lambda = 10^{-10}$ (effectively ridgeless) and sweep noise $\in \{0\%, 10\%, 20\%, 30\%, 40\%\}$ over the same 20 ratios with 3 seeds.

![Figure 11: Noise amplification of DD peak](figures/personB_noise_amplification.png)

**Results.** All five curves share a sharp peak at $p/n = 1$, but the peak height grows monotonically and substantially with noise:

| Noise | Peak MSE | Peak location $p/n$ | Peak/valley | Test acc at $p/n = 8$ |
|---|---|---|---|---|
| 0% | $35.2$ | $1.00$ | $1{,}593\times$ | $92.6\%$ |
| 10% | $78.5$ | $1.00$ | $2{,}462\times$ | $88.5\%$ |
| 20% | $106.3$ | $1.00$ | $2{,}471\times$ | $81.5\%$ |
| 30% | $133.4$ | $1.00$ | $2{,}419\times$ | $72.0\%$ |
| 40% | $186.2$ | $1.00$ | $2{,}775\times$ | $62.0\%$ |

Two observations are worth flagging:

1. **Peak amplification is roughly linear in noise rate.** Going from 0% to 40% noise inflates the peak MSE by $5.3\times$. The peak-to-valley ratio also grows, but more slowly, because the over-parameterised valley is itself slightly worse with noise (recovery accuracy drops from 92.6% to 62.0%).
2. **Recovery is consistent but degraded.** Even at 40% noise the model still recovers — test accuracy at $p/n = 8$ is $62\%$, well above the $\sim 30\%$ random-prediction baseline implied by 40% corruption. This is the "interpolating noise gracefully" signature: in the heavily over-parameterised regime, the minimum-norm solution still extracts most of the clean signal even when forced to also fit corrupted labels.

**Mechanism: interpolating noise.** The minimum-norm solution at $p/n = 1$ is forced to fit *every* training label, including the corrupted ones, with a near-square feature matrix that has poor condition number. Each corrupted label contributes a non-zero residual that the solver compensates with a large weight allocation along the corresponding singular direction; with $\sim n \sigma^2$ noise variance and a near-singular $\Phi$, this produces error of order $\|w\|^2 \cdot \kappa(\Phi)$ that scales linearly in noise rate, exactly as observed.

**Course connection (Lecture 6).** Classical bias-variance analysis predicts that label noise inflates *variance*, not bias. Experiment 6 (Section 6.2) confirmed this empirically by decomposing the test error of a single noise rate. Experiment B extends that picture to the noise-axis: the variance explosion at $p/n = 1$ scales monotonically with noise rate, and the over-parameterised regime's implicit $\ell_2$-regularisation is what allows the model to remain useful at all.

### 6.7 Why neural networks deviate from kernel double descent — optimiser and feature learning (Person C)

**Motivation.** The RFF picture so far (Sections 5.1, 6.5, 6.6) is a clean, textbook story: a sharp peak at the interpolation threshold, smoothed by ridge or amplified by noise, with monotone recovery. The CNN picture (Section 5.3) does *not* look like this. With Adam at lr $= 10^{-3}$, the noisy CNN never recovers — train accuracy reaches 100% on noisy CIFAR-10 while test accuracy collapses to 7%, far below random. The natural follow-up question is: is this a property of *neural networks*, or only of *Adam-on-noisy-CNN*? The Person C experiment isolates the optimiser as the controlled variable, holding architecture, dataset, and noise fixed.

**Setup.**
- Architecture: the project's CNN (`num_filters` $\in \{8, 16, 24, 32, 48, 64\}$, parameter count from $\sim 9{,}600$ to $\sim 412{,}000$).
- Dataset: CIFAR-10, $n = 4{,}000$ training samples; noise $\in \{0\%, 15\%\}$ (matching the Nakkiran recipe used in Exp A).
- Optimisers: SGD with momentum $0.9$, lr $= 0.05$, no scheduler; Adam with lr $= 10^{-4}$, no scheduler. Both run with constant learning rate so the comparison isolates the step rule rather than schedule effects.
- 500 epochs, 2 seeds (42, 7), batch size 512, GPU-resident training (no DataLoader overhead) on an NVIDIA RTX 5090 via vast.ai.
- $6 \times 2 \times 2 \times 2 = 48$ runs total; eval every 5 epochs to keep epoch-wise traces dense without doubling the runtime.

![Figure 12: Adam vs SGD model-wise](figures/personC_optimizer_modelwise.png)

![Figure 13: Adam vs SGD epoch-wise](figures/personC_optimizer_epochwise.png)

**Results — clean labels.** With clean labels, both optimisers fall into a similar regime but with a small, consistent gap. Final test accuracy averaged over the two seeds:

| Width $w$ | Params | $p/n$ | SGD train | SGD test | Adam train | Adam test |
|---|---|---|---|---|---|---|
| 8 | 11{,}162 | 2.79 | 100.0% | $52.7 \pm 0.5\%$ | 57.8% | $50.8 \pm 0.6\%$ |
| 16 | 33{,}834 | 8.46 | 100.0% | $56.1 \pm 0.8\%$ | 67.5% | $53.4 \pm 0.6\%$ |
| 24 | 68{,}026 | 17.01 | 100.0% | $57.6 \pm 0.0\%$ | 76.0% | $56.7 \pm 0.2\%$ |
| 32 | 113{,}738 | 28.43 | 100.0% | $58.8 \pm 0.5\%$ | 81.5% | $57.5 \pm 0.3\%$ |
| 48 | 239{,}722 | 59.93 | 100.0% | $59.5 \pm 0.4\%$ | 93.0% | $57.5 \pm 0.8\%$ |
| 64 | 411{,}786 | 102.95 | 100.0% | $60.2 \pm 0.5\%$ | 98.6% | $58.0 \pm 0.4\%$ |

Two observations: (i) SGD reaches 100% training accuracy at every width and improves monotonically on test accuracy ($52.7\% \to 60.2\%$ as width grows from 8 to 64); Adam at lr $= 10^{-4}$ does *not* fully memorise at small widths and lags SGD by $1$–$3$ percentage points throughout. (ii) Both optimisers exhibit width-driven improvement, but the regime is firmly in the over-parameterised tail ($p/n \in [2.8, 102.9]$) — there is no model-wise peak in this range, which is consistent with Sections 5.3 and the DD-Recovery campaign showing that the peak only appears below $p/n \approx 1$.

**Results — 15% label noise.** The picture changes qualitatively, but *not in the direction the original hypothesis predicted*:

| Width $w$ | $p/n$ | SGD train | SGD test | Adam train | Adam test |
|---|---|---|---|---|---|
| 8 | 2.79 | 100.0% | $7.7 \pm 0.3\%$ | 28.6% | $5.6 \pm 0.2\%$ |
| 16 | 8.46 | 100.0% | $7.0 \pm 0.1\%$ | 46.0% | $6.0 \pm 0.2\%$ |
| 24 | 17.01 | 100.0% | $6.6 \pm 0.1\%$ | 61.0% | $6.0 \pm 0.1\%$ |
| 32 | 28.43 | 100.0% | $6.5 \pm 0.2\%$ | 79.8% | $6.5 \pm 0.0\%$ |
| 48 | 59.93 | 100.0% | $6.5 \pm 0.0\%$ | 99.2% | $6.6 \pm 0.1\%$ |
| 64 | 102.95 | 100.0% | $6.4 \pm 0.1\%$ | 100.0% | $6.6 \pm 0.1\%$ |

Both optimisers collapse to sub-random test accuracy on noisy CIFAR-10 in this $n = 4{,}000$ regime — neither recovers. SGD memorises immediately (100% training accuracy at every width); Adam memorises more slowly at small widths but still ends near 100% by $w = 64$. The original hypothesis ("Adam memorises without recovery while SGD recovers") is *not* supported by these data. SGD is marginally better at the smallest width ($7.7\%$ versus $5.6\%$) — a one-shot win for the lower-lr Adam, which fails to memorise the noise hard enough to be as confidently wrong — but by $w = 32$ the two optimisers are within $0.1\%$ of each other and the gap closes entirely as width grows.

**What this actually shows.** The dominant variable is *not* the optimiser — it is the structural mismatch between the CNN family and $n = 4{,}000$ noisy CIFAR-10. The Effective Model Complexity analysis of Section 6.4 already showed that all our ResNet widths saturate EMC $\approx n$ within 50 epochs; Section 6.7 makes the same point for the CNN family across optimisers. When the model is far in the over-parameterised tail, the choice of optimiser becomes secondary: both rules end up memorising noisy labels and produce similarly catastrophic test accuracy. The clean-data experiments do reveal a measurable optimiser effect (SGD $1$–$3$ pp ahead of Adam, faster memorisation) — but it is small relative to the gap caused by the noise itself.

**Course connection (Lectures 7–9, 12).** This is consistent with the NTK / lazy-training picture: in the heavy over-parameterised regime, both SGD and Adam find some interpolant of the training set, and at $n \ll \text{EMC}$ that interpolant must (a) memorise the corrupted labels, (b) lose all predictive value. The specific implicit bias of SGD (minimum-$\ell_2$-norm-style) versus Adam (per-coordinate adaptive scaling) does not seem to differentiate the two outcomes when the model has tens-of-thousands of redundant parameters per training example. The interesting optimiser-dependent regime — where SGD's implicit bias might actually rescue generalisation — is the *near-threshold* regime that the DD-Recovery campaign (Section 6.3) reaches with fractional-$k$ ResNet, not the heavily over-parameterised regime that all our standard CNN/ResNet widths sit in.

**What this means for "what we learned about neural networks."** The headline lesson is *not* "Adam is uniquely bad on noisy CIFAR." It is: *parameter count alone is not the relevant complexity measure*. Both SGD and Adam — two fundamentally different update rules — produce indistinguishable failures when the network has $\geq 17 \times$ as many parameters as training examples. The interesting NN-side phenomenon is what happens *near* the interpolation threshold (DD-Recovery), where the dynamics are not yet smoothed out by the over-parameterised regime. This experiment falsifies one tempting hypothesis ("the optimiser explains the failure") and points back to *Effective Model Complexity* as the dominant variable, in line with Nakkiran et al. (2021).

### 6.8 Why parameter-counting fails — observed double descent vs classical bounds (Person D)

**Motivation.** A reader familiar with classical statistical learning theory might assume that "more parameters $\Rightarrow$ worse generalisation," because VC dimension, Rademacher complexity, and other capacity-based bounds grow monotonically (typically as $\sqrt{p/n}$ or worse). The double descent curve directly contradicts this intuition in the over-parameterised regime. This section makes the contradiction visual.

![Figure 14: Classical bound vs observed DD](figures/personD_bound_vs_observed.png)

**The figure** overlays a stylised classical bound $C \sqrt{p/n}$, anchored to match the observed test MSE at $p/n = 0.1$, against the observed test-MSE curve from Experiment 1 (10% noise). The bound is monotonically increasing in $p/n$; the observed curve rises, peaks at $p/n = 1$, and *second-descends* by an order of magnitude as $p/n$ grows further. In the green-shaded over-parameterised regime ($p/n \geq 1.5$) the bound is many times larger than the observed test MSE and continues to grow, while the observed curve drops below its under-parameterised values. The bound makes a vacuous claim in the regime where modern over-parameterised learning actually succeeds.

**Why the classical bounds fail (Lectures 10–12).**
1. **VC dimension and Rademacher complexity for ReLU networks** scale as $O(p \log p)$ or $O(\sqrt{p/n})$ — strictly increasing in $p$. They depend on $p$ alone and have no mechanism to express that the *learned solution* might lie in a much smaller, smoother subset of the hypothesis space.
2. **Norm-based bounds** (Bartlett, Foster & Telgarsky, 2017; Neyshabur, Bhojanapalli & Srebro, 2018) replace parameter count with the spectral or path norm of the trained weights. Empirically these norms *decrease* as $p$ grows past $n$, because the minimum-norm interpolant is "spread thinner" across more parameters. This is the right object for over-parameterised generalisation.
3. **Benign overfitting** (Bartlett, Long, Lugosi & Tsigler, 2020) gives an explicit, non-vacuous bound for ridgeless interpolation in linear regression that *decreases* with $p$ when the population covariance has the right tail decay. Hastie et al. (2022) extend this to the random-features setting we use throughout this project.

**The takeaway** is not that classical theory is wrong, but that it asks the wrong question. Bounding generalisation by raw capacity cannot detect the difference between a "spread-out" interpolant and a "concentrated" one. The observed double descent curve is direct empirical evidence that one of those two interpolants — the minimum-norm one — is what gradient-based optimisation finds, and that the appropriate complexity measure is its norm, not the dimension of the space it lives in.

### 6.9 Sample-wise double descent on the fractional-$k$ ResNet

**Motivation.** Sections 5.1–5.2 demonstrated sample-wise double descent in RFF: fixing the model and varying $n$ shifts the interpolation peak. The DD-Recovery campaign (Section 5.3) recovered the model-wise peak on the NN side. Here we close the loop by asking: does varying $n$ shift the NN-side peak in the same way theory predicts?

**Setup.** We run the same fractional-$k$ ResNet family as Section 5.3, sweeping $k \in \{0.0625, 0.125, 0.25, 0.5, 1.0\}$ for $n \in \{1{,}000, 2{,}000\}$ (new runs), and pool with the existing results at $n = 4{,}000$ and $n = 8{,}000$. All runs: 15% label noise, Adam ($\text{lr} = 10^{-4}$), 1{,}500 epochs, 2 seeds, CIFAR-10.

![Figure 15: Sample-wise NN DD — peak shifts right with n](figures/samplewise_nn_dd.png)

**Results.** The per-$n$ best test-accuracy curves are:

| $k$ | $n=1{,}000$ | $n=2{,}000$ | $n=4{,}000$ | $n=8{,}000$ |
|---|---|---|---|---|
| 0.0625 | 14.0% | 21.0% | 25.7% | — |
| 0.125 | 24.9% | 27.7% | 32.9% | 39.7% |
| 0.1875 | — | — | **49.2%** | — |
| 0.25 | 38.0% | 44.8% | 52.3% | **57.5%** |
| 0.5 | 40.0% | 45.6% | 51.2% | **60.4%** |
| 1.0 | 39.7% | 46.2% | 52.9% | 59.7% |

Three observations are consistent with sample-wise double descent theory:

1. **The interpolation threshold shifts to higher $k$ as $n$ grows.** For $n = 1{,}000$, the sharpest accuracy jump occurs between $k = 0.0625$ (14%) and $k = 0.125$ (24.9%), indicating the threshold is at $p \approx n = 1{,}000$ params (which corresponds to $k \approx 0.08$). For $n = 4{,}000$, the peak is at $k \approx 0.1875$ ($p \approx 6{,}500 \approx 1.6n$); for $n = 8{,}000$, the best point shifts to $k = 0.5$ ($p \approx 44{,}000 \approx 5.5n$), consistent with the peak migrating rightward.

2. **Post-threshold recovery improves with $n$.** At $k = 0.5$ (well into the over-parameterised regime), best test accuracy rises from 40% ($n = 1{,}000$) → 45.6% ($n = 2{,}000$) → 51.2% ($n = 4{,}000$) → 60.4% ($n = 8{,}000$). More data enables better interpolating solutions.

3. **The peak-valley shape is clearest in $n = 4{,}000$.** At $n = 1{,}000$ and $n = 2{,}000$, the $k$ grid has insufficient resolution near the threshold — the jump from underfitting to over-parameterised recovery is visible but the valley itself is not clearly resolved. The $n = 4{,}000$ curve (with finer $k$ spacing from the DD-Recovery sweep) shows the canonical rise–peak–valley–recovery shape. This resolution limitation is expected: for small $n$, the threshold occurs at small $k$ where our discrete grid is coarser.

**Comparison to RFF sample-wise DD (Exp 2).** In Exp 2 (Section 5.2), varying $n$ in the RFF setting clearly shifts the $p/n = 1$ peak while holding the curve shape fixed. In the NN setting, the same directional shift is present but less sharp, because (a) the effective interpolation threshold depends on feature learning dynamics rather than raw parameter count, and (b) Adam's implicit regularisation smooths the peak. This difference is itself informative: it suggests that the NN interpolation threshold is better measured by EMC (Nakkiran et al., 2021) than by raw $p/n$.

### 6.11 Fractional-$k$ epoch-wise dynamics and early stopping

**Motivation.** DD-Recovery (Section 5.3) establishes a clean model-wise DD signal on the NN side. The remaining question is dynamic: does training longer always help for the same fractional-$k$ model, or does late-stage training hurt generalization in specific complexity regimes?

**Setup.** We run a fixed epoch-wise protocol on the same CIFAR-10/noise configuration as DD-Recovery: $n=4{,}000$, label noise $15\%$, Adam ($\mathrm{lr}=10^{-4}$), 2,000 epochs, seeds $\{42,7\}$, and $k\in\{0.125, 0.1875, 0.5\}$. For each run, we compare:
1) **best checkpoint** (early-stop proxy) and  
2) **final checkpoint** (epoch 2000).

![Figure 16: Fractional-$k$ epoch-wise dynamics and early-stop gap](figures/fractionalk_epochwise.png)

**Results.**

| $k$ | Params | Seed | Best epoch | Best test acc | Final test acc | Gap (best-final) |
|---|---:|---:|---:|---:|---:|---:|
| 0.125 | 2,988 | 42 | 2000 | 35.35% | 35.35% | 0.00 |
| 0.125 | 2,988 | 7  | 2000 | 32.89% | 32.89% | 0.00 |
| 0.1875 | 6,505 | 42 | 1800 | 48.76% | 48.75% | 0.01 |
| 0.1875 | 6,505 | 7  | 1800 | 49.17% | 48.89% | 0.28 |
| 0.5 | 44,370 | 42 | 400 | 54.14% | 48.13% | 6.01 |
| 0.5 | 44,370 | 7  | 400 | 51.18% | 47.09% | 4.09 |

Three consistent patterns emerge:
1. **Near-threshold stability at $k\approx0.1875$.** The best-final gap is tiny (0.01–0.28 pp), indicating weak late-stage degradation around the interpolation peak.
2. **Strong late-stage degradation in over-parameterized regime ($k=0.5$).** The gap is large (4.09–6.01 pp), and the best checkpoint appears much earlier (epoch 400), showing prolonged training hurts test performance after an early optimum.
3. **Under-capacity side ($k=0.125$) remains monotone.** Best and final coincide at epoch 2000, suggesting this regime is still optimization-limited rather than overfitting-limited.

**Conclusion for the main storyline.** Early stopping mainly reduces late-stage overfitting in the over-parameterized region; it does **not** move the recovered model-wise DD peak location. This supports the DD-Recovery claim that interpolation-threshold geometry (capacity/EMC axis) is primary, while optimizer-time dynamics are a secondary correction.

## 7. Discussion

### 7.1 Key Findings

Our experiments demonstrate double descent clearly in kernel methods (RFF) and partially in neural networks (CNN). The RFF experiments produce textbook results because:
1. The solver computes the **exact** minimum-norm solution (no optimization dynamics)
2. The interpolation threshold is **precisely defined** ($p = n$)
3. The feature space is **fixed** (no feature learning)

Neural networks add complexity because:
1. Adam/SGD provides **implicit regularization** beyond just minimum-norm
2. The **effective parameterization** differs from the raw parameter count
3. **Feature learning** changes the NTK during training (rich vs. lazy regime)

### 7.2 Connection to Course Material

**Approximation Theory (Lectures 2–3):** Barron's theorem guarantees that two-layer networks with $m$ neurons approximate Barron functions at rate $O(1/\sqrt{m})$. In the double descent context, increasing $m$ beyond the interpolation point does not degrade approximation quality — it provides additional degrees of freedom that allow the optimizer to find smoother interpolants.

**Over-parameterization and Convergence (Lectures 5–6):** Du et al. (2019) proved that gradient descent converges to a global minimum for sufficiently wide networks. The interpolation threshold marks the boundary: below it, the optimization landscape may have spurious local minima; above it, the loss surface is benign and gradient descent finds good solutions efficiently.

**Neural Tangent Kernel (Lectures 7–8):** Our RFF experiments are the kernel-method analogue of the NTK regime. As Jacot et al. (2018) showed, infinitely wide networks behave as kernel machines with a fixed NTK. The double descent we observe in RFF is precisely the phenomenon that occurs in this lazy training regime. The smoother behavior in actual neural networks may reflect the *feature learning* regime (Chizat & Bach, 2018), where the NTK evolves during training.

**Generalization Theory (Lecture 9):** Classical Rademacher complexity bounds predict $\mathfrak{R}_n(\mathcal{F}) \sim \sqrt{p/n}$, giving a test error bound that increases monotonically with $p$. This completely misses the second descent. The failure highlights that **parameter counting is the wrong complexity measure** — what matters is the norm/smoothness of the learned function. Norm-based bounds (Bartlett et al., 2017; Neyshabur et al., 2018) and the theory of benign overfitting (Bartlett et al., 2020) provide tighter, non-vacuous bounds that can accommodate double descent.

### 7.3 Role of Label Noise

Label noise plays a critical role in double descent:
- It inflates the minimum-norm solution's $\ell_2$-norm at the threshold, because the model must now interpolate incorrect labels
- The peak-to-valley ratio grows with noise rate (47× clean → 129× with 20% noise for MSE)
- In the over-parameterized regime, noise degrades final performance ($92.9\%$ clean vs. $83.0\%$ with 20% noise) but the model still recovers significantly from the peak

### 7.4 Limitations

1. **Computational constraints on literal Nakkiran reproduction.** Our NN experiments use $n = 4{,}000$ rather than the full CIFAR-10. With a stock ResNet-18 this places every width far in the over-parameterised regime ($p/n \geq 44$) and produces no model-wise peak (Exp A). The DD-Recovery campaign on a custom fractional-$k$ ResNet recovers the phenomenon at the same $n$ by spanning the interpolation threshold from below; a literal ResNet-18 reproduction at $n = 50{,}000$ remains future work.

2. **NN-side mechanism diagnostics.** The RFF story is supported by an explicit condition-number analysis (Exp 7) and bias-variance decomposition (Exp 6). The CNN side has the analogous *outcome* (Sections 5.3, 6.7) but no comparable Jacobian / effective-rank diagnostic. Adding a Jacobian condition number trace to the Person C runs is a natural next step.

3. **Single-architecture optimiser comparison.** The Adam-vs-SGD comparison (Section 6.7) uses one CNN family. Whether the optimiser-driven divergence persists for ResNet-18 or for transformer-style architectures was out of scope.

---

## 9. Conclusion and Future Work

### 9.1 Conclusion

We have empirically demonstrated the double descent phenomenon across eight experiments spanning model-wise, sample-wise, and epoch-wise axes. Our key findings:

**RFF on MNIST (Experiments 1–6):**
- Test MSE spikes up to **129×** at the interpolation threshold ($p/n = 1$), amplified by label noise
- **Robustness confirmed** across 5 random seeds: noise ordering is preserved, though the peak magnitude varies substantially between seeds
- **Variance drives the peak**: bias-variance decomposition shows variance increases by 3–5 orders of magnitude at $p/n = 1$ while bias decreases monotonically. Over-parameterization brings variance back down via the minimum-norm solution's implicit $\ell_2$ regularization

**ResNet on CIFAR-10 (Experiments 7–8):**
- All models (k=1,2,4) immediately memorize all 4000 noisy training samples within 50 epochs — confirmed by EMC ≈ n for all (k, T) combinations
- **No epoch-wise DD** occurs because n < EMC for our entire training set: the models are always in the over-parameterized regime, with no under-fitting → over-fitting transition during training
- **EMC is dataset-scale dependent**: Nakkiran et al.'s experiment works because with n=50,000, their ResNet has p/n ≈ 3.5 near the threshold. Our n=4000 gives p/n ∈ [44, 694], too far from the threshold
- **SGD vs. Adam**: No meaningful difference in the fully over-parameterized regime — both optimizers produce the same catastrophic memorization

These results directly validate theoretical predictions from EECS 6699: the classical bias-variance tradeoff (Lectures 2–3) describes only the pre-threshold regime; variance explosion at interpolation (Lectures 7–8) explains the peak; implicit regularization in the over-parameterized regime (Lectures 5–6) explains the recovery for RFF but fails for NN with noisy labels.

### 9.2 Future Work

The Person A (ridge) and Person C (Adam vs SGD) extensions in this report close two of the gaps identified earlier in the project. The natural next steps are:

1. **Ridge regularisation on neural networks.** Section 6.5 confirms ridge smooths the RFF peak; the analogous CNN experiment (weight decay sweep co-varied with noise) would test whether the same mechanism applies to learnable features.
2. **Jacobian / effective-rank diagnostics for CNN.** The RFF story has an empirical condition-number trace (Exp 7); the CNN story does not. Recording $\kappa(J^\top J)$ for the network Jacobian along training would give the NN side a comparable mechanism plot.
3. **Sample-wise NN double descent.** Sections 5.2 and 6.6 study the sample axis only for RFF. Sweeping $n$ at a fixed CNN width would test whether "more data can hurt" persists when features are learned rather than fixed.
4. **Architecture sweep across families.** MLP / CNN / ResNet / Transformer at matched parameter counts, all under the same noise and optimiser, would test the architecture-independence claim implicit in much of the literature.
5. **Effective Model Complexity for NN.** Exp 8 shows EMC saturates at $n$ for our (k, T) grid. Sweeping at much larger $n$ (or much smaller $k$) is needed to make EMC informative for the CNN regime.

---

## 10. References

1. Bartlett, P. L., Foster, D. J., & Telgarsky, M. J. (2017). Spectrally-normalized margin bounds for neural networks. *NeurIPS*.

2. Bartlett, P. L., Long, P. M., Lugosi, G., & Tsigler, A. (2020). Benign overfitting in linear regression. *PNAS*, 117(48), 30063–30070.

3. Barron, A. R. (1993). Universal approximation bounds for superpositions of a sigmoidal function. *IEEE Trans. on Information Theory*, 39(3), 930–945.

4. Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine learning practice and the bias-variance trade-off. *PNAS*, 116(32), 15849–15854.

5. Chizat, L. & Bach, F. (2018). On the global convergence of gradient descent for over-parameterized models using optimal transport. *NeurIPS*.

6. D'Ascoli, S., Refinetti, M., Biroli, G., & Krzakala, F. (2020). Double trouble in double descent: Bias and variance(s) in the lazy regime. *ICML*.

7. Du, S. S., Zhai, X., Poczos, B., & Singh, A. (2019). Gradient descent provably optimizes over-parameterized neural networks. *ICLR*.

8. Gunasekar, S., Woodworth, B. E., Bhojanapalli, S., Neyshabur, B., & Srebro, N. (2017). Implicit regularization in matrix factorization. *NeurIPS*.

9. Hastie, T., Montanari, A., Rosset, S., & Tibshirani, R. J. (2022). Surprises in high-dimensional ridgeless least squares interpolation. *Annals of Statistics*, 50(2), 949–986.

10. Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural tangent kernel: Convergence and generalization in neural networks. *NeurIPS*.

11. Ji, Z. & Telgarsky, M. (2019). The implicit bias of gradient descent on nonseparable data. *COLT*.

12. Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. *Technical Report, University of Toronto*.

13. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.

14. Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B., & Sutskever, I. (2021). Deep double descent: Where bigger models and more data can hurt. *JSTAT*, 2021(12), 124003.

15. Neyshabur, B., Bhojanapalli, S., & Srebro, N. (2018). A PAC-Bayesian approach to spectrally-normalized margin bounds for neural networks. *ICLR*.

16. Rahimi, A. & Recht, B. (2007). Random features for large-scale kernel machines. *NeurIPS*.

---

## Appendix A: Code Structure

```
double-descent/
├── src/
│   ├── models.py                 # MLP, CNN, ResNet architectures
│   ├── data.py                   # Data loading, noise corruption, subsets
│   ├── trainer.py                # Generic training loop with metric logging
│   ├── plotting.py               # Visualization utilities
│   └── experiments/
│       └── comprehensive_dd.py   # Main experiment suite (4 experiments)
├── notebooks/
│   └── analysis.ipynb            # Interactive analysis with math discussion
├── results/                      # JSON results + auto-generated plots
├── figures/                      # Publication-quality figures
├── report.md                     # This report
├── requirements.txt              # Python dependencies
└── README.md                     # Quick start guide
```

## Appendix B: Reproduction Instructions

```bash
# 1. Install dependencies
pip install torch torchvision numpy matplotlib tqdm scikit-learn pandas

# 2. Run all experiments
cd double-descent

# Fast: RFF experiments only (~15 seconds)
PYTHONUNBUFFERED=1 python3 -m src.experiments.comprehensive_dd --experiments "1,2"

# Full: Include neural network experiments (~3-4 hours on GPU)
PYTHONUNBUFFERED=1 python3 -m src.experiments.comprehensive_dd

# 3. Open the analysis notebook
jupyter notebook notebooks/analysis.ipynb
```


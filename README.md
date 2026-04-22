# Double Descent Phenomenon in Deep Learning

**EECS 6699: Mathematics of Deep Learning — Final Project (Spring 2026)**

**Team:** Zhengda Li (zl3651), Yusheng Li (yl6009), Shufeng Chen (sc5739), Yizheng Lin (yl6079)

---

## Overview

This project provides a comprehensive empirical investigation of the **double descent** phenomenon — a surprising behavior where test error first decreases, then increases (classical bias-variance tradeoff), and then decreases *again* as model complexity grows beyond the interpolation threshold. We go beyond simply observing double descent: we explain **why** it happens (spectral analysis), how to **control** it (ridge regularization), and what **hyperparameters** affect it (kernel bandwidth, optimal λ).

We study three manifestations of double descent:

1. **Model-wise double descent**: varying the number of parameters $p$ (Experiments 1, 3, 5)
2. **Sample-wise double descent**: varying the number of training samples $n$ (Experiment 2)
3. **Epoch-wise double descent**: varying training duration $T$ (Experiment 4)

And provide in-depth mechanistic analysis:

4. **Ridge regularization analysis**: how explicit regularization smooths the DD peak (Experiment 6)
5. **Spectral analysis**: condition number and solution norm explosion explain the peak (Experiment 7)
6. **Optimal regularization**: finding the best λ for each complexity level (Experiment 8)
7. **Kernel bandwidth sensitivity**: how feature quality affects the DD phenomenon (Experiment 9)

We use two complementary approaches:
- **Random Fourier Features (RFF)** on MNIST — kernel method providing clean, theoretically grounded results (Experiments 1, 2, 6, 7, 8, 9)
- **Neural Networks (MLP, CNN, ResNet)** on CIFAR-10 — real deep learning behavior with feature learning (Experiments 3, 4, 5)

---

## Repository Structure

```
double-descent-deep-learning/
├── src/                                 # Source code
│   ├── models.py                        # MLP, CNN, ResNet architectures
│   ├── data.py                          # Data loading, noise corruption, subsets
│   ├── trainer.py                       # Generic training loop with metric logging
│   ├── plotting.py                      # Visualization utilities
│   └── experiments/
│       └── comprehensive_dd.py          # Main experiment suite (9 experiments)
├── results/                             # Experiment outputs (JSON data + auto-generated plots)
│   ├── exp1_model_wise_rff/             # Model-wise DD with Random Fourier Features
│   ├── exp2_sample_wise_rff/            # Sample-wise DD with Random Fourier Features
│   ├── exp3_nn_model_wise/              # Model-wise DD with CNN
│   ├── exp4_epoch_wise_nn/              # Epoch-wise DD with CNN
│   ├── exp5_architecture_comparison/    # MLP vs CNN vs ResNet comparison
│   ├── exp6_rff_ridge/                  # Ridge regularization effect on DD
│   ├── exp7_spectral_analysis/          # Condition number & solution norm analysis
│   ├── exp8_optimal_lambda/             # Optimal ridge lambda analysis
│   └── exp9_sigma_sensitivity/          # RFF kernel bandwidth sensitivity
├── figures/                             # Publication-quality figures
├── notebooks/
│   └── analysis.ipynb                   # Interactive analysis with mathematical discussion
├── plot_exp5.py                         # Standalone plotting script for Exp5
├── report.md                            # Final report (survey + experimental analysis)
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## Quick Start

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib tqdm scikit-learn pandas
```

### Running Experiments

```bash
# Set environment variable to avoid OpenMP conflicts (Windows)
set KMP_DUPLICATE_LIB_OK=TRUE

# Run all RFF-based experiments (Exp 1, 2, 6, 7, 8, 9) — ~3 minutes total
python -u -m src.experiments.comprehensive_dd --experiments "1,2,6,7,8,9"

# Run a single experiment
python -u -m src.experiments.comprehensive_dd --experiments "6"

# Run neural network experiments (Exp 3, 4, 5) — requires GPU, ~2-4 hours
python -u -m src.experiments.comprehensive_dd --experiments "3,4,5"

# Run everything
python -u -m src.experiments.comprehensive_dd

# Custom parameters
python -u -m src.experiments.comprehensive_dd \
    --experiments "1,6" \
    --n-train 2000 \
    --seed 123 \
    --output-dir ./my_results
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiments` | `"1,2,3,4,5,6,7,8,9"` | Comma-separated list of experiments to run |
| `--n-train` | `1000` | Number of training samples for RFF experiments |
| `--n-train-nn` | `4000` | Number of training samples for NN experiments |
| `--epochs-nn` | `500` | Training epochs for NN model-wise experiments |
| `--epochs-epoch` | `1000` | Training epochs for epoch-wise experiment |
| `--data-dir` | `./data` | Directory for downloading/caching datasets |
| `--output-dir` | `./results` | Directory for saving results and plots |
| `--seed` | `42` | Random seed for reproducibility |

---

## Experiments in Detail

### Experiment 1: Model-Wise Double Descent (Random Fourier Features)

**Goal:** Demonstrate the classic double descent curve by sweeping the number of random features $D$ while keeping training set size $n$ fixed.

- **Dataset:** MNIST ($d = 784$), $n = 1{,}000$ training samples
- **Model:** $D$ Random Fourier Features (RBF kernel, $\sigma = 5.0$), solved via minimum-norm interpolation
- **Sweep:** $p/n$ ratio from 0.05 to 8.0 (20 values)
- **Noise levels:** 0%, 10%, 20% label corruption
- **Runtime:** ~15 seconds

**Key Findings:**
- Sharp peak at $p/n = 1.0$ with test MSE spike of **2,100×** (clean data)
- Label noise amplifies the peak: 47.5 (clean) → 129.1 (20% noise)
- Over-parameterized regime ($p/n = 8$) achieves **92.9%** accuracy — the best overall

---

### Experiment 2: Sample-Wise Double Descent (Random Fourier Features)

**Goal:** Show that *more training data can hurt* performance when it pushes the system toward the interpolation threshold.

- **Dataset:** MNIST, $D = 500$ fixed features
- **Sweep:** $n$ from 100 to 4{,}000
- **Noise:** 10% label corruption
- **Runtime:** ~10 seconds

**Key Findings:**
- Increasing $n$ from 100 to 500 makes test MSE **1,700× worse**
- Peak occurs exactly at $n = D = 500$ (the interpolation threshold)
- For $n > 700$, the classical "more data is better" behavior resumes

---

### Experiment 3: Model-Wise Double Descent (Neural Networks)

**Goal:** Investigate model-wise double descent in actual neural networks (CNNs) trained with gradient descent.

- **Dataset:** CIFAR-10, $n = 4{,}000$ samples
- **Model:** CNN with variable width (num_filters ∈ {1, 2, 3, 4, 6, 8, 12, 16, 24, 32})
- **Optimizer:** Adam (lr=0.001, no weight decay)
- **Epochs:** 500
- **Runtime:** ~1-2 hours (GPU), ~6-8 hours (CPU)

**Key Findings:**
- Clean data: monotonic improvement with width — Adam's implicit regularization smooths the DD peak
- 20% noise: catastrophic memorization with 0% train error but ~93% test error

---

### Experiment 4: Epoch-Wise Double Descent (Neural Networks)

**Goal:** Track test error evolution over training time to observe epoch-wise double descent.

- **Dataset:** CIFAR-10, $n = 4{,}000$, 20% noise
- **Model:** CNN with widths {2, 4, 8}
- **Epochs:** 1{,}000
- **Runtime:** ~1-3 hours (GPU)

**Key Findings:**
- Over-parameterized models achieve 100% train accuracy by epoch ~400
- Test error remains at random-chance level throughout → no epoch-wise recovery observed
- The absence is explained by small dataset size and simple architecture lacking skip connections

---

### Experiment 5: Architecture Comparison (MLP vs CNN vs ResNet)

**Goal:** Compare how different neural network architectures exhibit the double descent phenomenon under identical conditions.

- **Dataset:** CIFAR-10, $n = 4{,}000$, 10% label noise
- **Architectures:**
  - **MLP**: hidden widths {1, 2, 5, 10, 20, 50, 100, 200} → params from 3K to 617K
  - **CNN**: filter counts {1, 2, 3, 4, 6, 8, 12, 16, 24, 32} → params from 774 to 114K
  - **ResNet**: width multiplier $k$ ∈ {0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0} → params from 3K to 697K
- **Runtime:** ~2-4 hours (GPU)

**Key Findings:**
- All three architectures achieve 100% training accuracy when sufficiently wide
- All show severe over-fitting in the over-parameterized regime with noisy labels
- ResNet reaches interpolation threshold at fewer parameters than MLP due to parameter efficiency
- The results highlight that without explicit regularization, over-parameterized NNs on small noisy datasets cannot recover — unlike kernel methods (RFF) with min-norm solutions

---

### Experiment 6: Ridge Regularization Effect on DD Peak (RFF)

**Goal:** Show how explicit ridge regularization ($\lambda > 0$) smooths the double descent peak, connecting to Hastie et al.'s theory on optimal ridgeless interpolation.

- **Dataset:** MNIST, $n = 1{,}000$, 10% noise
- **Sweep:** $\lambda$ ∈ {0, 1e-8, 1e-6, 1e-4, 0.01, 0.1, 1.0} × $p/n$ ratio from 0.05 to 8.0
- **Runtime:** ~1 minute

**Key Findings:**
- **λ = 0 (ridgeless):** MSE at $p/n = 1$ spikes to **65.2** (error: 88%)
- **λ = 0.01:** Peak compressed to **0.13** — a **500× reduction**
- **λ = 0.1:** DD peak almost entirely eliminated, smooth U-shaped curve
- **λ = 1.0:** Complete monotonic decrease, no peak at all — but over-regularization prevents learning fine features
- **Conclusion:** A small amount of regularization eliminates the catastrophic interpolation peak while preserving the benefits of over-parameterization

---

### Experiment 7: Spectral Analysis — Why DD Happens (RFF)

**Goal:** Provide a mechanistic explanation for the interpolation peak by analyzing the kernel matrix's spectral properties and the solution norm.

- **Dataset:** MNIST, $n = 1{,}000$, 10% noise
- **Metrics:** Condition number $\kappa(\Phi\Phi^\top)$, solution norm $\|w\|_2$, test MSE
- **Runtime:** ~30 seconds

**Key Findings:**
- At $p/n = 1.0$: condition number explodes to **10⁹** (vs ~10⁴ at $p/n = 0.5$)
- Solution norm $\|w\|_2$ spikes from ~40 to **1,229** (30× increase)
- **Three quantities spike in sync:** condition number ↑ → solution norm ↑ → test MSE ↑
- **Mechanistic explanation:** at $p \approx n$, the kernel matrix $\Phi\Phi^\top$ becomes nearly singular. The minimum-norm solution is forced to take extremely large values to interpolate the training data (including noisy labels), which amplifies noise in predictions. This is a direct empirical verification of the variance explosion mechanism described by Hastie et al. (2022).

---

### Experiment 8: Optimal Ridge Lambda Analysis (RFF)

**Goal:** For each model complexity level ($p/n$ ratio), find the optimal regularization strength $\lambda^*$ that minimizes test error.

- **Dataset:** MNIST, $n = 1{,}000$, 10% noise
- **Sweep:** 40 $\lambda$ values (log-spaced from $10^{-10}$ to $10^2$) × 20 $p/n$ ratios
- **Runtime:** ~1 minute

**Key Findings:**
- **Under-parameterized ($p/n < 0.5$):** Optimal $\lambda^* \approx 2.9$ — strong regularization needed
- **Near threshold ($p/n \approx 1$):** Optimal $\lambda^* \approx 0.7$ — still significant regularization to suppress the peak
- **Over-parameterized ($p/n > 5$):** Optimal $\lambda^*$ drops to ~0.35 — over-parameterization provides implicit regularization, reducing the need for explicit regularization
- **With optimal λ, DD completely disappears:** the "best achievable MSE" curve decreases monotonically with $p/n$
- The right panel shows MSE vs λ for individual $p/n$ values — the $p/n = 1.0$ curve (gray) has the highest MSE across all λ values, but a well-chosen λ reduces it from 65 to 0.035

---

### Experiment 9: Kernel Bandwidth (σ) Sensitivity (RFF)

**Goal:** Study how the RFF kernel bandwidth $\sigma$ (which controls feature quality and expressiveness) affects the double descent phenomenon.

- **Dataset:** MNIST, $n = 1{,}000$, 10% noise
- **Sweep:** $\sigma$ ∈ {0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0} × 20 $p/n$ ratios
- **Runtime:** ~1 minute

**Key Findings:**
- **DD peak at $p/n = 1$ occurs for ALL $\sigma$ values** — confirming DD is a universal phenomenon, not an artifact of feature choice
- **σ = 0.5 (too narrow):** Features are nearly useless; MSE spikes to **5,000** at threshold; final accuracy only ~13%
- **σ = 5.0 (optimal):** Best overall performance; over-parameterized region achieves **89.3%** accuracy
- **σ = 50.0 (too wide):** Features become too smooth; accuracy degrades to ~73% but peak is relatively smaller
- **Sweet spot:** $\sigma = 5$–$10$ balances feature expressiveness with smoothness
- The experiment reveals a **bandwidth-complexity tradeoff**: poor kernel bandwidth makes double descent more severe (higher peak-to-valley ratio) because low-quality features waste model capacity

---

## Summary of Key Results

| Experiment | Type | Runtime | Core Finding |
|------------|------|---------|--------------|
| **Exp 1** | Model-wise DD (RFF) | ~15s | Test MSE spikes 2,100× at $p/n = 1$; noise amplifies peak |
| **Exp 2** | Sample-wise DD (RFF) | ~10s | More data can hurt: 1,700× worse MSE at $n = D$ |
| **Exp 3** | Model-wise DD (CNN) | ~1-2h | Adam prevents DD peak on clean data; noise causes catastrophic memorization |
| **Exp 4** | Epoch-wise DD (CNN) | ~1-3h | No epoch-wise recovery observed (dataset too small, architecture too simple) |
| **Exp 5** | Architecture comparison | ~2-4h | MLP, CNN, ResNet all overfit on small noisy datasets without regularization |
| **Exp 6** | Ridge λ vs DD peak | ~1min | **λ = 0.01 reduces peak 500×; λ = 1.0 eliminates DD entirely** |
| **Exp 7** | Spectral analysis | ~30s | **Condition number 10⁹ and ‖w‖ = 1,229 at threshold explain the peak** |
| **Exp 8** | Optimal λ* analysis | ~1min | **Optimal λ decreases with over-parameterization; DD vanishes with proper λ** |
| **Exp 9** | Kernel bandwidth σ | ~1min | **DD is universal across all σ; σ = 5 is optimal for MNIST** |

---

## Connection to Course Material

| Lecture Topic | Connection to Our Experiments |
|---|---|
| **Approximation Theory (L2–L3)** | Barron's theorem: more neurons reduce bias; over-parameterization provides additional degrees of freedom for smoother interpolants |
| **Over-parameterization (L5–L6)** | Du et al.'s convergence for wide nets; our Exp 3 shows the transition to benign loss landscape |
| **NTK (L7–L8)** | RFF ≈ kernel regime; Exp 1, 6–9 directly demonstrate kernel interpolation phenomena |
| **Generalization (L9)** | Rademacher bounds miss the second descent; Exp 7 shows parameter counting fails — norm matters |
| **Ridge Regression** | Exp 6 & 8 empirically validate Hastie et al.'s ridgeless interpolation theory |
| **Implicit Regularization** | Exp 3 vs Exp 1: Adam provides implicit regularization that kernel methods lack |

---

## Theoretical Story (Experiments 6–9)

Our RFF analysis experiments (6–9) form a complete narrative:

1. **Why does DD happen?** (Exp 7) → The kernel matrix becomes nearly singular at $p = n$, causing the condition number to explode to $10^9$. The minimum-norm solution is forced to take huge values ($\|w\| = 1{,}229$), amplifying noise.

2. **What does DD look like?** (Exp 1, 2) → A sharp spike in test error at the interpolation threshold, with magnitude depending on noise level.

3. **How to fix it?** (Exp 6) → Ridge regularization ($\lambda > 0$) caps the condition number, preventing the variance explosion. Even $\lambda = 0.01$ reduces the peak by 500×.

4. **What's the best λ?** (Exp 8) → The optimal regularization strength $\lambda^*$ depends on the complexity regime: higher λ needed for under-parameterized models, lower λ for over-parameterized ones.

5. **Is DD universal?** (Exp 9) → Yes — it occurs across all kernel bandwidth choices, but feature quality (σ) modulates the severity.

---

## References

See [`report.md`](report.md) for the complete reference list (16 citations). Key references:

1. Belkin, M. et al. (2019). *Reconciling modern ML practice and the bias-variance tradeoff.* PNAS.
2. Nakkiran, P. et al. (2021). *Deep double descent: Where bigger models and more data can hurt.* JSTAT.
3. Hastie, T. et al. (2022). *Surprises in high-dimensional ridgeless least squares interpolation.* Annals of Statistics.
4. Rahimi, A. & Recht, B. (2007). *Random features for large-scale kernel machines.* NeurIPS.
5. Jacot, A. et al. (2018). *Neural tangent kernel: Convergence and generalization in neural networks.* NeurIPS.
6. Bartlett, P. L. et al. (2020). *Benign overfitting in linear regression.* PNAS.

---

## License

This project is for academic use as part of EECS 6699 at Columbia University.

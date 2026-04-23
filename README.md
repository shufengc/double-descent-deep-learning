# Double Descent Phenomenon in Deep Learning

**EECS 6699: Mathematics of Deep Learning — Final Project (Spring 2026)**

**Team:** Zhengda Li (zl3651), Yusheng Li (yl6009), Shufeng Chen (sc5739), Yizheng Lin (yl6079)

---

## Overview

This project empirically investigates the **double descent** phenomenon — a surprising behavior where test error first decreases, then increases (classical bias-variance tradeoff), then decreases *again* as model complexity grows beyond the interpolation threshold. We study three manifestations:

1. **Model-wise double descent**: varying the number of parameters $p$ (Experiments 1, 3, Architecture)
2. **Sample-wise double descent**: varying the number of training samples $n$ (Experiment 2)
3. **Epoch-wise double descent**: varying training duration $T$ (Experiments 4, C)

We use two complementary approaches:
- **Random Fourier Features (RFF)** on MNIST — kernel method providing clean, theoretically grounded results
- **Neural Networks (MLP, CNN, ResNet)** on CIFAR-10 — real neural network behavior with feature learning

## Repository Structure

```
├── src/
│   ├── models.py                      # MLP, CNN, ResNet architectures
│   ├── data.py                        # Data loading, noise corruption, subsets
│   ├── trainer.py                     # Generic training loop with metric logging
│   ├── plotting.py                    # Visualization utilities
│   └── experiments/
│       ├── comprehensive_dd.py        # Original experiment suite (Exp 1–4)
│       ├── shufeng_experiments.py     # Extended experiments (Exp 5–8, A–C)
│       └── exp_architecture.py        # Architecture comparison (MLP/CNN/ResNet)
├── results/
│   ├── exp1_model_wise_rff/           # Exp 1: RFF model-wise DD
│   ├── exp2_sample_wise_rff/          # Exp 2: RFF sample-wise DD
│   ├── exp3_nn_model_wise/            # Exp 3: CNN model-wise DD
│   ├── exp4_epoch_wise_nn/            # Exp 4: CNN epoch-wise DD
│   ├── exp_noise_multiseed/           # Exp 5: 5-seed noise robustness
│   ├── expB_bias_variance/            # Exp 6: Bias-variance decomposition
│   ├── expC_epoch_sgd_resnet/         # Exp 7: ResNet SGD vs Adam epoch-wise
│   ├── expA_emc/                      # Exp 8: Effective Model Complexity
│   ├── exp_architecture/              # Architecture comparison (negative finding)
│   ├── zhengda_exp5_lambda/           # Zhengda: Ridge λ sweep
│   ├── zhengda_exp6_noise/            # Zhengda: Noise comparison
│   ├── zhengda_exp7_optimizer/        # Zhengda: Optimizer comparison
│   ├── yusheng_exp7_spectral/         # Yusheng: Spectral analysis
│   ├── yusheng_exp9_sigma/            # Yusheng: Kernel bandwidth sensitivity
│   ├── yusheng_exp5_architecture/     # Yusheng: Architecture comparison (ref)
│   ├── yusheng_exp8_optimal_lambda/   # Yusheng: Optimal λ per ratio
│   └── yizheng_multiseed/             # Yizheng: 3-seed RFF results (exp1–4)
├── figures/                           # Publication-quality figures
├── notebooks/
│   └── analysis.ipynb                 # Interactive analysis
├── report.md                          # Final report (survey + research)
├── requirements.txt                   # Python dependencies
└── README.md
```

## Quick Start

```bash
pip install torch torchvision numpy matplotlib tqdm scikit-learn pandas

# Run RFF experiments only (~15 seconds)
PYTHONUNBUFFERED=1 python3 -m src.experiments.comprehensive_dd --experiments "1,2"

# Run full experiment suite including neural networks (~3-4 hours)
PYTHONUNBUFFERED=1 python3 -m src.experiments.comprehensive_dd

# Run architecture comparison (requires GPU, ~6-10 hours)
python3 -m src.experiments.exp_architecture --epochs 500 --noise 0.1
```

## Experiments and Key Results

### Core Experiments (Shufeng)

| # | Experiment | Key Finding |
|---|---|---|
| 1 | RFF model-wise DD | Textbook DD curve. 2,100× peak at p/n=1. Over-parameterized (p/n=8): 92.9% accuracy |
| 2 | RFF sample-wise DD | "More data can hurt": adding samples near p=n increases MSE 1,700× |
| 3 | CNN model-wise DD | DD visible under 20% noise. Clean data: monotonic improvement with width |
| 4 | CNN epoch-wise DD | Under Adam + noise: catastrophic memorization, no epoch-wise DD |
| 5 | Multi-seed noise (5 seeds) | Zhengda's 40% noise anomaly is a seed artifact; peaks are monotonic in noise rate |
| 6 | Bias-variance decomposition | DD peak is a **pure variance** phenomenon — bias decreases monotonically |
| 7 | ResNet SGD vs Adam (4000 epochs) | No epoch-wise DD: models reach 100% train acc within 100 epochs, test acc plateaus |
| 8 | Effective Model Complexity | EMC saturates at n=4000 by epoch 50 — model stays over-parameterized throughout training |
| Arch | Architecture comparison (MLP/CNN/ResNet) | Negative finding: all architectures collapse to random chance under Adam + noise + no regularization |

### Teammate Contributions (cherry-picked into main)

| Contributor | Experiment | Key Finding |
|---|---|---|
| Zhengda | λ sweep (ridge regularization) | λ=0.01 reduces DD peak by >90%; λ=1.0 eliminates it entirely |
| Zhengda | Noise comparison | 0/10/20/40% noise; 40% anomaly identified (seed artifact) |
| Yusheng | Spectral analysis | Condition number explodes at p/n=1 (24 → 17,132), explaining the variance spike |
| Yusheng | σ sensitivity | Kernel bandwidth σ=5 optimal (89.3% acc); σ<2 gives random chance |
| Yizheng | Multi-seed framework (3 seeds) | Seed=42 results match ours exactly; ±σ uncertainty bands for all RFF curves |
| Yizheng | Theory connections | λ sweep → Hastie theorem; SGD+augment → Nakkiran's "hidden DD" |

## Connection to Course Material

| Lecture Topic | Our Experiment | Connection |
|---|---|---|
| Bias-Variance (L3) | Exp 6 (bias-variance decomposition) | DD peak is pure variance; bias decreases monotonically past interpolation |
| Approximation Theory (L2–L3) | Exp 1 (RFF) | Barron's theorem: more features reduce bias in over-parameterized regime |
| Over-parameterization (L5–L6) | Exp 7–8 (epoch-wise + EMC) | EMC saturates early → model never crosses interpolation threshold during training |
| NTK (L7–L8) | Spectral analysis (Yusheng) | Condition number explosion at p=n; kernel interpolation phenomenon |
| Generalization (L9) | Exp 5 (multi-seed) | Peak variance is heavy-tailed; Rademacher bounds miss the second descent |
| Regularization (L10) | λ sweep (Zhengda) + σ sensitivity (Yusheng) | Ridge regularization → Hastie et al. theorem: "cut the peak, keep the valley" |

## References

See [`report.md`](report.md) for the complete reference list (16 citations).

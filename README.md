# Double Descent Phenomenon in Deep Learning

**EECS 6699: Mathematics of Deep Learning — Final Project (Spring 2026)**

**Team:** Zhengda Li (zl3651), Yusheng Li (yl6009), Shufeng Chen (sc5739), Yizheng Lin (yl6079)

---

## Overview

This project empirically investigates the **double descent** phenomenon — a surprising behavior where test error first decreases, then increases (classical bias-variance tradeoff), then decreases *again* as model complexity grows beyond the interpolation threshold. We study six manifestations:

1. **Model-wise double descent**: varying the number of parameters $p$ (Experiments 1 & 3)
2. **Sample-wise double descent**: varying the number of training samples $n$ (Experiment 2)
3. **Epoch-wise double descent**: varying training duration $T$ (Experiment 4)
4. **Regularization-wise comparison**: varying ridge regularization $\lambda$ in RFF (Experiment 5)
5. **Noise-rate comparison**: varying label noise rates (0%, 10%, 20%, 40%) in RFF (Experiment 6)
6. **Optimizer comparison**: Adam vs SGD in CNN model-wise DD (Experiment 7)

We use two complementary approaches:
- **Random Fourier Features (RFF)** on MNIST — kernel method providing clean, theoretically grounded results
- **Convolutional Neural Networks (CNN)** on CIFAR-10 — real neural network behavior with feature learning

## Repository Structure

```
├── src/                          # Source code
│   ├── models.py                 # MLP, CNN, ResNet architectures
│   ├── data.py                   # Data loading, noise corruption, subsets
│   ├── trainer.py                # Generic training loop with metric logging
│   ├── plotting.py               # Visualization utilities
│   └── experiments/
│       └── comprehensive_dd.py   # Main experiment suite
├── notebooks/
│   └── analysis.ipynb            # Interactive analysis with mathematical discussion
├── results/                      # Experiment outputs (JSON + auto-generated plots)
├── figures/                      # Publication-quality figures
├── report.md                     # Final report (survey + research)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Quick Start

```bash
pip install torch torchvision numpy matplotlib tqdm scikit-learn pandas

# Run RFF experiments only (~15 seconds)
PYTHONUNBUFFERED=1 python3 -m src.experiments.comprehensive_dd --experiments "1,2"

# Run regularization comparison (RFF lambda sweep)
PYTHONUNBUFFERED=1 python3 -m src.experiments.comprehensive_dd --experiments "5"

# Run noise-rate comparison (RFF)
PYTHONUNBUFFERED=1 python3 -m src.experiments.comprehensive_dd --experiments "6"

# Run optimizer comparison (CNN, Adam vs SGD; Adam lr=1e-3, SGD lr=5e-2, wd=1e-4, 10% noise by default)
PYTHONUNBUFFERED=1 python3 -m src.experiments.comprehensive_dd --experiments "7"
# Tune exp 7: e.g. `--optimizer-lr-sgd 0.1 --optimizer-noise-rate 0.2 --epochs-optimizer 400`

# Run full experiment suite including neural networks (~3-4 hours)
PYTHONUNBUFFERED=1 python3 -m src.experiments.comprehensive_dd --experiments "1,2,3,4,5,6,7"
```

## Key Results

### Random Fourier Features (Belkin et al. setup)
- Clear double descent peak at $p/n = 1.0$
- Test MSE spikes **2,100×** at the interpolation threshold (clean data)
- Label noise amplifies the peak: **47.5** (clean) → **129.1** (20% noise) MSE at threshold
- Over-parameterized models ($p/n = 8$) achieve **92.9%** accuracy — best overall

### Sample-Wise: More Data Can Hurt
- Increasing samples from $n=100$ to $n=500$ makes test MSE **1,700× worse**
- Peak occurs exactly at $n = D$ (number of features)

### Neural Networks (CNN on CIFAR-10)
- Clean data: monotonic improvement with width (no DD peak — Adam provides implicit regularization)
- 20% noise: catastrophic memorization — 0% train error but ~93% test error (below random chance)

## Connection to Course Material

| Lecture Topic | Connection |
|---|---|
| Approximation Theory (L2–L3) | Barron's theorem: more neurons reduce bias in the over-parameterized regime |
| Over-parameterization (L5–L6) | Du et al.'s convergence for wide nets; benign loss landscape past threshold |
| NTK (L7–L8) | RFF ≈ kernel regime; double descent is a kernel interpolation phenomenon |
| Generalization (L9) | Rademacher bounds miss the second descent; parameter counting fails |

## References

See [`report.md`](report.md) for the complete reference list (16 citations).

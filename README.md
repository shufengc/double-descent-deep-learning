# Double Descent Phenomenon in Deep Learning

**EECS 6699: Mathematics of Deep Learning - Final Project (Spring 2026)**

**Team:** Zhengda Li (zl3651), Yusheng Li (yl6009), Shufeng Chen (sc5739), Yizheng Lin (yl6079)

This repository contains the code, saved experiment results, and generated figures for an empirical study of double descent in random-feature models and neural networks.

## Overview

The project studies three double-descent settings:

1. **Model-wise double descent**: varying model capacity at fixed training-set size.
2. **Sample-wise double descent**: varying the number of training samples at fixed capacity.
3. **Epoch-wise dynamics**: tracking training time, memorization, and evaluation protocol effects.

We use two complementary testbeds:

- **Random Fourier Features (RFF) on MNIST**, which gives a clean closed-form baseline where the feature dimension is the capacity axis.
- **Neural networks on CIFAR-10**, including MLP/CNN/ResNet baselines and a fractional-width ResNet that crosses the interpolation threshold.

> **Headline NN result.** Naive CNN and ResNet18 width sweeps do not show model-wise double descent because their capacity range misses the interpolation threshold. A fractional-$k$ ResNet fixes the capacity axis and recovers a double-descent profile: final-epoch accuracy follows roughly 26% -> 49% -> 53% under 15% label noise. Best-over-epochs reaches about 55%, but the metric audit treats that number as optimistic because it selects checkpoints using the test set.

## Repository Structure

```text
src/
  data.py                         # Data loading, noise corruption, subsets
  models.py                       # MLP, CNN, and ResNet architectures
  trainer.py                      # Generic training and evaluation loop
  plotting.py                     # Shared plotting helpers
  experiments/
    comprehensive_dd.py           # Core RFF and CNN experiments
    exp_dd_recovery.py            # Fractional-k ResNet DD recovery campaign
    plot_dd_recovery.py           # DD recovery figures
    exp_architecture.py           # Architecture comparison baseline
    exp_nakkiran_recipe.py        # Nakkiran-style recipe and augmentation ablation
    exp_samplewise_nn.py          # Neural-network sample-wise sweep
    exp_samplewise_nn_plot.py     # Sample-wise plotting
    exp_bartlett_bound_eval.py    # Bartlett-style effective-rank diagnostic
    supplemental_dd_extras.py     # OOD/ID, ordered sampling, early stopping extras

scripts/
  make_figures.py                 # Regenerate core figures from saved results
  make_paper_figures.py           # Regenerate paper-final figures from saved results
  audit_metrics.py                # Metric-audit helper
  run_*.py                        # Additional experiment launchers

results/                          # Saved JSON/CSV metrics and selected result figures
figures/                          # Publication-quality generated figures
notebooks/                        # Exploratory analysis notebook
requirements.txt                  # Python dependencies
Makefile                          # Figure-regeneration target
README.md
```

`results/` also contains diagnostic subdirectories such as depth ablations, Hessian top-eigenvalue runs, empirical NTK runs, epoch-wise fractional-k runs, metric-audit outputs, and archived reproduction runs. These files support the submitted report but are kept as code/data artifacts rather than as the report itself.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

A minimal CPU run that reproduces the RFF model-wise and sample-wise double-descent curves:

```bash
PYTHONUNBUFFERED=1 python3 -m src.experiments.comprehensive_dd --experiments "1,2"
```

Regenerate figures from saved results:

```bash
make figures
```

## Main Experiment Entry Points

### Core RFF and CNN Experiments

```bash
PYTHONUNBUFFERED=1 python3 -m src.experiments.comprehensive_dd
```

This runs the original core pipeline: RFF model-wise DD, RFF sample-wise DD, CNN model-wise sweep, and CNN epoch-wise dynamics. The full neural-network parts are GPU-oriented.

### Fractional-k DD Recovery

```bash
python3 src/experiments/exp_dd_recovery.py --mode smoke
python3 src/experiments/exp_dd_recovery.py --mode probe
python3 src/experiments/exp_dd_recovery.py --mode main
python3 src/experiments/exp_dd_recovery.py --mode nslice
python3 src/experiments/plot_dd_recovery.py
```

Modes:

- `smoke`: quick sanity check.
- `probe`: shorter one-seed sweep to map the transition.
- `main`: headline 2,000-epoch n=4,000 sweep.
- `nslice`: n=8,000 comparison showing threshold migration.

### Supplemental Modules

```bash
python3 -m src.experiments.exp_architecture --epochs 500 --noise 0.1
python3 -m src.experiments.exp_nakkiran_recipe --exp all
python3 -m src.experiments.exp_activation_ablation --device cuda
python3 -m src.experiments.exp_samplewise_nn \
    --ns 1000,2000 --ks 0.0625,0.125,0.25,0.5,1.0 --seeds 42,7 --epochs 1500
python3 -m src.experiments.exp_samplewise_nn_plot
python3 -m src.experiments.exp_bartlett_bound_eval
python3 -m src.experiments.supplemental_dd_extras --experiments S1,S2,S3
```

Additional scripts under `scripts/` reproduce metric audits, weight-decay sweeps, cross-architecture checks, and paper-final figure generation from saved results.

## Key Results

| Module | Key finding |
|---|---|
| RFF model-wise DD | Textbook double-descent curve with a sharp peak near p/n = 1 and strong over-parameterized recovery. |
| RFF sample-wise DD | More data can hurt near the interpolation threshold when capacity is fixed. |
| Ridge and noise sweeps | Ridge regularization smooths the RFF peak; label noise amplifies it. |
| Bias-variance decomposition | The RFF interpolation peak is dominated by variance. |
| Naive NN width sweeps | Small CNN and literal ResNet18 sweeps do not cross the relevant threshold and therefore do not show model-wise DD. |
| Fractional-k ResNet | A continuous width axis recovers model-wise DD on noisy CIFAR-10. |
| Spectral diagnostics | Penultimate-feature spectrum, Bartlett-style effective rank, empirical NTK, and Hessian diagnostics localize the same transition region. |
| Sample-wise NN sweep | Increasing n shifts the effective threshold toward larger widths. |
| Metric audit | Best-over-epochs reporting makes the over-parameterized valley look too shallow; final-epoch reporting gives a cleaner DD profile. |
| Ablations | Activation, depth, optimizer, weight decay, and bounds diagnostics clarify where the DD recovery is robust and where evidence remains limited. |

## Selected Figures

Core and diagnostic figures are tracked under `figures/` and selected `results/` subdirectories. Important examples include:

- `figures/fig1_model_wise_rff.png`
- `figures/fig3_sample_wise_rff.png`
- `figures/paper_final/fig1_main_valley_metric_audit.png`
- `figures/resnet18_vs_fractionalk.png`
- `figures/nn_effective_rank_vs_k.png`
- `figures/full_empirical_ntk_quick.png`
- `figures/hessian_topeig_vs_k.png`
- `figures/samplewise_nn_dd.png`
- `figures/paper_final/fig6_gap_as_sixth_witness.png`
- `figures/paper_final/fig7_wd_sweep_valley.png`

## Reproducibility Notes

- Dataset downloads are handled by torchvision and are ignored via `data/`.
- Heavy per-run outputs are ignored where summary files carry the aggregate metrics.
- GPU-heavy experiments may take hours; the repository includes saved results so figures can be regenerated without rerunning all training.

## References

Key references for the submitted report include:

- Belkin et al. (2019), *Reconciling Modern Machine Learning Practice and the Bias-Variance Trade-Off*.
- Nakkiran et al. (2021), *Deep Double Descent*.
- Hastie et al. (2022), *Surprises in High-Dimensional Ridgeless Least Squares Interpolation*.
- Bartlett et al. (2020), *Benign Overfitting in Linear Regression*.
- D'Ascoli et al. (2020), *Triple Descent and the Two Kinds of Overfitting*.

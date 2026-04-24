# Zhengda Exp8: Noise × Ridge Mechanism Analysis

## Overview

This experiment is a follow-up study built on top of Zhengda's original Exp5 and the corrected version of Exp6.

The original Exp5 showed that ridge regularization can suppress the double descent peak in Random Fourier Features (RFF) models. The original Exp6 studied the effect of label noise on the double descent curve, but its single-seed result contained an anomaly: the 40% noise setting did not always produce a larger peak than the 20% noise setting. A later multi-seed correction showed that this anomaly was caused by seed sensitivity near the interpolation threshold.

This new experiment combines the two directions and asks a deeper question:

> When label noise amplifies the interpolation peak, can ridge regularization consistently suppress that peak, and what mechanism explains the suppression?

In short, this experiment turns the original Exp5 and Exp6 from two separate observations into a single mechanism-level story:

```text
label noise increases instability
        ↓
ridgeless interpolation near p/n ≈ 1 becomes ill-conditioned
        ↓
solution norm and test MSE explode
        ↓
ridge regularization stabilizes the solution
        ↓
double descent peak is suppressed
```

---

## Relationship to Previous Experiments

### Original Exp5: Ridge Regularization Sweep

Original Exp5 swept ridge values under a fixed 10% label-noise setting.

It showed:

> Ridge regularization can dramatically reduce the double descent peak while preserving good performance in the overparameterized regime.

However, Exp5 only answered whether ridge works. It did not fully explain why ridge works, and it only focused on one noise level.

This new experiment extends Exp5 by:

1. Testing ridge regularization across multiple label-noise levels.
2. Running multiple seeds for robustness.
3. Measuring mechanism variables such as solution norm, condition number, and effective degrees of freedom.
4. Showing that ridge does not just reduce test error; it stabilizes the underlying linear system.

---

### Corrected Exp6: Multi-Seed Noise Sweep

Original Exp6 swept label noise levels in a ridgeless RFF model. The single-seed result had an anomaly: the 40% noise case did not always produce a larger peak than the 20% noise case.

The corrected multi-seed version showed:

> Label noise increases the average interpolation peak, but the peak is highly seed-sensitive.

This new experiment extends corrected Exp6 by:

1. Reconfirming the noise-amplification effect.
2. Showing that the ridgeless interpolation peak is heavy-tailed.
3. Testing whether ridge regularization can suppress this noise-amplified peak.
4. Explaining the role of ill-conditioning and solution norm explosion.

---

## Experiment Question

The main research question is:

> How do label noise and ridge regularization jointly affect the double descent peak?

More specifically, this experiment asks:

1. Does label noise amplify the ridgeless interpolation peak?
2. Is the interpolation peak robust across random seeds, or is it heavy-tailed?
3. Can ridge regularization suppress the noise-amplified peak?
4. Does ridge work by reducing solution norm, improving conditioning, and lowering effective degrees of freedom?

---

## Experimental Setup

### Dataset

The experiment uses MNIST.

- Training samples: `n = 1000`
- Test set: full MNIST test set
- Input: flattened 28×28 images
- Labels: one-hot encoded labels

### Model

The model uses Random Fourier Features (RFF):

\[
\Phi(x) = \sqrt{\frac{2}{D}} \cos(Wx + b)
\]

where `D` is the number of random features.

A linear predictor is fitted on top of the random features:

\[
\hat{Y} = \Phi W
\]

The solution is computed in closed form using either the ridgeless minimum-norm solution or ridge regression.

---

## Swept Parameters

### Label Noise Rates

The experiment uses four label-noise levels:

```text
0%, 10%, 20%, 40%
```

### Ridge Regularization Strengths

The experiment sweeps eight ridge values:

```text
λ = 0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0
```

These cover:

- ridgeless interpolation: `λ = 0`
- extremely weak ridge: `1e-10`, `1e-8`
- weak but visible ridge: `1e-6`, `1e-4`
- moderate ridge: `1e-2`, `1e-1`
- strong ridge: `1.0`

### Model Size Ratios

The experiment sweeps the full model-wise double descent range:

```text
p/n = 0.05, 0.1, 0.2, 0.3, 0.5, 0.7,
      0.8, 0.9, 0.95, 0.98,
      1.0,
      1.02, 1.05, 1.1, 1.2,
      1.5, 2.0, 3.0, 5.0, 8.0
```

This covers:

1. underparameterized regime: `p/n < 1`
2. interpolation threshold: `p/n ≈ 1`
3. overparameterized regime: `p/n > 1`

### Random Seeds

The experiment uses five random seeds:

```text
42, 123, 456, 789, 1024
```

This is important because the interpolation peak is highly seed-sensitive.

---

## Command Used

The full experiment was run using:

```bash
python -m src.experiments.zhengda_exp8_noise_lambda_mechanism \
  --seeds 42,123,456,789,1024 \
  --noise-rates 0.0,0.1,0.2,0.4 \
  --lambdas 0,1e-10,1e-8,1e-6,1e-4,1e-2,1e-1,1.0 \
  --ratios 0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.9,0.95,0.98,1.0,1.02,1.05,1.1,1.2,1.5,2.0,3.0,5.0,8.0 \
  --output-dir ./results/zhengda_exp8_noise_lambda_full
```

Total number of runs:

```text
5 seeds × 4 noise rates × 8 lambdas × 20 p/n ratios = 3200 runs
```

---

## Output Files

The experiment outputs the following files:

```text
results/zhengda_exp8_noise_lambda_full/
├── results.json
├── all_rows.csv
├── peak_summary.csv
├── aggregate_peak_summary.csv
├── peak_heatmap.png
├── noise_peak_boxplot.png
└── mechanism_curves.png
```

### `all_rows.csv`

Contains every individual run.

Each row corresponds to one combination of:

```text
seed, noise_rate, lambda, p/n ratio
```

### `peak_summary.csv`

For each combination of:

```text
seed, noise_rate, lambda
```

this file records the peak test MSE across all p/n ratios.

### `aggregate_peak_summary.csv`

Aggregates the peak statistics across seeds for each:

```text
noise_rate × lambda
```

This file is used to generate the heatmap.

### `peak_heatmap.png`

Shows:

```text
noise rate × ridge lambda → mean peak test MSE
```

This is the main summary figure.

### `noise_peak_boxplot.png`

Shows the multi-seed distribution of the ridgeless peak, with:

```text
λ = 0
```

This figure supports the corrected Exp6 result.

### `mechanism_curves.png`

Shows four mechanism curves for a representative setting:

1. Test MSE vs p/n
2. Solution norm vs p/n
3. Regularized condition number vs p/n
4. Effective degrees of freedom vs p/n

---

## Metrics Collected

For every configuration, the experiment records both performance metrics and mechanism metrics.

### Performance Metrics

```text
train_mse
test_mse
train_acc
test_acc
```

### Mechanism Metrics

```text
solution_norm
regularized_condition_number
effective_degrees_of_freedom
min_singular_value
max_singular_value
```

These mechanism metrics are used to explain why the peak appears and why ridge regularization suppresses it.

---

## Mechanism Metrics Explained

### 1. Solution Norm

The solution norm is:

\[
\|W\|
\]

A large solution norm indicates that the learned predictor is sensitive to label noise and small perturbations.

In the ridgeless setting, the solution norm spikes near the interpolation threshold. Ridge regularization reduces this norm.

---

### 2. Regularized Condition Number

The regularized condition number is:

\[
\kappa_\lambda = \frac{s_{\max}^2 + \lambda}{s_{\min}^2 + \lambda}
\]

where \(s_{\max}\) and \(s_{\min}\) are the largest and smallest singular values of the RFF feature matrix.

Near \(p/n = 1\), the feature matrix becomes ill-conditioned. The smallest singular value can become extremely small, making the ridgeless solution unstable.

Ridge regularization stabilizes the system by adding \(\lambda\), preventing small singular values from being amplified too much.

---

### 3. Effective Degrees of Freedom

The effective degrees of freedom are computed as:

\[
df(\lambda) = \sum_i \frac{s_i^2}{s_i^2 + \lambda}
\]

This measures the effective complexity of the ridge-regularized model.

When \(\lambda = 0\), the effective degrees of freedom are close to the rank of the feature matrix.

When \(\lambda\) increases, the effective degrees of freedom decrease, even though the raw feature dimension \(D\) is unchanged.

This shows that raw parameter count is not the only relevant complexity measure. Ridge regularization controls effective complexity.

---

## Main Results

## Result 1: Label noise amplifies the ridgeless interpolation peak

In the ridgeless setting (`λ = 0`), the mean interpolation peak increases with label noise.

Approximate mean peak MSE values:

| Noise rate | Mean peak test MSE |
|---|---:|
| 0% | 1.9e4 |
| 10% | 2.7e4 |
| 20% | 5.9e4 |
| 40% | 6.9e4 |

This supports the corrected Exp6 conclusion:

> Label noise amplifies the interpolation peak.

---

## Result 2: The ridgeless peak is heavy-tailed and seed-sensitive

The boxplot for `λ = 0` shows that the peak test MSE has very large variance across seeds.

The mean is much larger than the median because a few seeds produce extremely large peaks near \(p/n \approx 1\).

This explains why the original single-seed Exp6 could produce an anomaly.

Conclusion:

> The interpolation peak is not only high on average; it is also heavy-tailed and seed-sensitive.

---

## Result 3: Ridge regularization suppresses the noise-amplified peak

The heatmap shows that ridge regularization dramatically reduces the mean peak test MSE across all noise levels.

For example, at 10% label noise:

| λ | Mean peak test MSE |
|---|---:|
| 0 | 2.7e4 |
| 1e-10 | 5.1e3 |
| 1e-8 | 75 |
| 1e-6 | 13 |
| 1e-4 | 1.2 |
| 1e-2 | 0.13 |
| 1e-1 | 0.069 |
| 1.0 | 0.069 |

This shows that even a very small amount of ridge regularization can reduce the peak by several orders of magnitude.

Conclusion:

> Ridge regularization consistently suppresses the noise-amplified double descent peak across all noise levels.

---

## Result 4: Ridge turns sharp double descent into a smooth curve

The mechanism curves show that with very small ridge values, such as:

```text
λ = 0, 1e-10, 1e-8
```

there is a sharp test MSE spike near:

```text
p/n = 1
```

As λ increases, the spike becomes much smaller.

For moderate ridge values such as:

```text
λ = 1e-2, 1e-1, 1.0
```

the test MSE curve becomes smooth and the double descent spike is nearly eliminated.

Conclusion:

> Ridge regularization transforms a sharp double descent curve into a smoother generalization curve.

---

## Result 5: The peak coincides with solution norm explosion

The solution norm plot shows that the ridgeless and near-ridgeless solutions have a large norm spike near \(p/n = 1\).

This norm spike appears at the same location as the test MSE peak.

As λ increases, the solution norm is strongly reduced.

Conclusion:

> The double descent peak is associated with a large-norm interpolating solution.

This supports the interpretation that ridge improves generalization by controlling the norm of the learned solution.

---

## Result 6: The peak coincides with condition number explosion

The regularized condition number plot shows that the feature matrix becomes extremely ill-conditioned near \(p/n = 1\).

For small λ, the condition number spikes by several orders of magnitude.

As λ increases, the condition number is greatly reduced.

Conclusion:

> The interpolation peak is driven by numerical instability caused by ill-conditioning of the feature matrix.

Ridge regularization stabilizes the solution by improving the effective condition number.

---

## Result 7: Ridge reduces effective complexity

The effective degrees of freedom plot shows that ridge regularization reduces effective model complexity.

With small λ, the effective degrees of freedom quickly approach approximately:

```text
n = 1000
```

With stronger λ, the effective degrees of freedom are much lower.

For example:

- `λ = 0`: effective df is close to 1000 near and after interpolation.
- `λ = 1e-1`: effective df is substantially lower.
- `λ = 1.0`: effective df is much lower, around a few hundred.

Conclusion:

> Ridge regularization reduces effective degrees of freedom, showing that effective complexity, not raw feature dimension alone, controls generalization.

---

## Overall Interpretation

The main interpretation is:

> The double descent peak is not caused by parameter count alone. It arises when noisy interpolation interacts with an ill-conditioned feature matrix near \(p/n = 1\).

The mechanism is:

```text
label noise increases target instability
        ↓
near p/n = 1, the feature matrix becomes ill-conditioned
        ↓
the ridgeless interpolating solution has very large norm
        ↓
test MSE explodes
        ↓
double descent peak appears
```

Ridge regularization interrupts this chain:

```text
ridge λ increases
        ↓
small singular values are stabilized
        ↓
condition number decreases
        ↓
solution norm decreases
        ↓
effective degrees of freedom decrease
        ↓
double descent peak is suppressed
```

---

## Final Conclusion

This experiment provides a mechanism-level explanation for the results of Exp5 and the corrected Exp6.

The main conclusions are:

1. Label noise amplifies the double descent peak.
2. The ridgeless interpolation peak is heavy-tailed and seed-sensitive.
3. Ridge regularization suppresses the peak across all noise levels.
4. The peak coincides with explosions in solution norm and condition number.
5. Ridge works by reducing solution norm, improving conditioning, and lowering effective degrees of freedom.
6. Effective complexity is more informative than raw parameter count for understanding generalization.

In short:

> Double descent in RFF models is best understood as a noise-sensitive instability of the interpolating solution, and ridge regularization controls this instability by reducing effective complexity.

---

## Recommended Figure Usage

For the final report, the most useful figures are:

1. `peak_heatmap.png`  
   Main result: ridge suppresses noise-amplified peaks.

2. `noise_peak_boxplot.png`  
   Corrected Exp6 result: noise amplifies the ridgeless peak and the peak distribution is heavy-tailed.

3. `mechanism_curves.png`  
   Mechanism result: ridge reduces norm, condition number, and effective degrees of freedom.

The heatmap should be the main figure. The boxplot and mechanism curves can be used as supporting figures.

---

## Overlap Check with Existing Experiments

This experiment has some intentional connections to previous experiments, but it is not a duplicate.

### Not a duplicate of Zhengda Exp5

Exp5 sweeps ridge λ at one fixed noise level, 10% label noise.

This new experiment sweeps both noise and λ, uses multiple seeds, and records mechanism metrics. It explains why the ridge sweep works.

### Not a duplicate of corrected Exp6 / Shufeng multi-seed noise validation

Corrected Exp6 validates that label noise increases the ridgeless peak on average.

This new experiment asks whether ridge can suppress that noise-amplified peak and explains the mechanism using norm, conditioning, and effective degrees of freedom.

### Not a duplicate of Shufeng bias-variance decomposition

The bias-variance experiment explains that the double descent peak is driven by variance.

This new experiment does not decompose bias and variance. Instead, it studies how ridge changes solution norm, condition number, and effective degrees of freedom. It gives a complementary mechanism for how ridge controls the variance-driven peak.

### Not a duplicate of Yusheng spectral analysis

The spectral analysis studies condition number as a function of p/n.

This new experiment includes condition number, but adds two extra dimensions: label noise and ridge λ. It also connects condition number directly to peak MSE, solution norm, and effective degrees of freedom.

### Not a duplicate of Yusheng optimal λ experiment

The optimal λ experiment searches for the best λ at each p/n ratio.

This experiment is not primarily an optimizer over λ. It is a mechanism study showing how increasing λ stabilizes interpolation across different noise levels.

### Not a duplicate of Yizheng multi-seed framework

Yizheng's work adds multi-seed uncertainty bands and additional λ sweep validation.

This experiment uses multi-seed evaluation, but its new contribution is the joint noise × ridge design and the mechanism metrics.

Overall, this experiment is best described as:

> A mechanism-level extension of Zhengda Exp5 and corrected Exp6, connecting label noise, ridge regularization, ill-conditioning, solution norm, and effective complexity.

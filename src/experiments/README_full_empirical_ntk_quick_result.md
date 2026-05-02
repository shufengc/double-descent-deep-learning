# Zhengda Supplemental Experiment: Full Empirical-NTK Quick Validation

## Overview

This supplemental experiment extends the neural-network spectral analysis in Section 6.10 by computing a **full empirical Neural Tangent Kernel (empirical-NTK)** for fractional-width ResNet models.

The existing Section 6.10 analysis studies the spectrum of the **penultimate-layer feature representation**. That result shows that several spectral diagnostics have a local irregularity around \(k \approx 0.1875\), which aligns with the onset of the recovered neural-network double descent behavior.

This experiment asks a deeper follow-up question:

> Does a similar spectral irregularity also appear in the full network Jacobian, rather than only in the penultimate-layer features?

To answer this, we compute the empirical-NTK Gram matrix using Jacobians over **all trainable model parameters**.

---

## Motivation

The earlier RFF experiments showed that double descent peaks are closely connected to instability in the feature matrix:

- In RFF models, the interpolation threshold \(p/n \approx 1\) produces a condition-number spike.
- Ridge regularization suppresses the double descent peak by reducing solution norm, improving conditioning, and lowering effective degrees of freedom.

The neural-network spectral analysis by Shufeng showed a related phenomenon in fractional-width ResNets:

- Penultimate-feature spectral diagnostics show a local extremum near \(k \approx 0.1875\).
- This aligns with the onset of the NN double descent recovery experiment.

This experiment provides a lightweight check of whether the same signal appears at the level of the **full empirical-NTK**, which uses the full parameter Jacobian.

---

## Experiment Setup

### Model

The experiment uses the fractional-width ResNet model from the DD-Recovery experiments.

The tested widths are:

```text
k = 0.125, 0.1875, 0.25, 0.5
```

These values are chosen because \(k \approx 0.1875\) is the region where the DD-Recovery experiment and penultimate-feature spectral analysis show interesting behavior.

### Dataset

The experiment uses CIFAR-10 with the same overall setup as the fractional-k experiments:

```text
n_train = 1000
noise_rate = 15%
seed = 42
```

The models are trained locally in a quick setting:

```text
epochs = 200
```

This is intentionally much smaller than the full DD-Recovery run, which used longer training. Therefore, this experiment should be interpreted as a **sanity check / supplemental validation**, not as a replacement for the main NN DD-Recovery experiment.

### Empirical-NTK Samples

To keep the full Jacobian computation feasible on a local machine, the empirical-NTK is computed on:

```text
n_ntk_samples = 12
```

For each selected input \(x_i\), the experiment computes the Jacobian of all output logits with respect to all trainable model parameters.

---

## Empirical-NTK Definition

For a model \(f_\theta(x)\), the full empirical-NTK Gram matrix is computed as:

\[
K_{ij} = \sum_c 
\left\langle
\nabla_\theta f_c(x_i),
\nabla_\theta f_c(x_j)
\right\rangle
\]

where:

- \(i,j\) index samples,
- \(c\) indexes output classes,
- \(\theta\) includes all trainable model parameters.

This is different from the penultimate-feature spectrum because it uses the **full parameter Jacobian**, not only the last hidden representation.

---

## Command Used

The quick validation experiment was run with:

```bash
python -m src.experiments.exp_full_empirical_ntk_quick \
  --ks 0.125,0.1875,0.25,0.5 \
  --n-train 1000 \
  --epochs 200 \
  --n-ntk-samples 12 \
  --seed 42 \
  --device cuda
```

If CIFAR-10 already exists in another local project directory, the dataset path can be specified with:

```bash
--data-dir "PATH_TO_EXISTING_DATA_DIR"
```

---

## Output Files

The experiment outputs:

```text
results/full_empirical_ntk_quick/
├── summary.json
├── all_rows.csv
├── k0.125_seed42/
│   ├── config.json
│   ├── history.json
│   └── ntk_spectrum.json
├── k0.1875_seed42/
│   ├── config.json
│   ├── history.json
│   └── ntk_spectrum.json
├── k0.25_seed42/
│   ├── config.json
│   ├── history.json
│   └── ntk_spectrum.json
└── k0.5_seed42/
    ├── config.json
    ├── history.json
    └── ntk_spectrum.json
```

The main figure is saved as:

```text
figures/full_empirical_ntk_quick.png
```

---

## Main Results

The experiment produced the following summary:

| k | Parameters | Train Acc | Test Acc | NTK Condition Number | Stable Rank / N | Participation Ratio / N |
|---:|---:|---:|---:|---:|---:|---:|
| 0.125 | 2,988 | 15.1% | 14.8% | 18.7 | 0.221 | 0.440 |
| 0.1875 | 6,505 | 28.2% | 29.6% | **558.8** | **0.125** | **0.177** |
| 0.25 | 11,374 | 31.9% | 29.4% | 79.1 | 0.185 | 0.311 |
| 0.5 | 44,370 | 49.3% | 39.3% | 40.3 | 0.197 | 0.347 |

---

## Result 1: Full empirical-NTK condition number spikes near \(k = 0.1875\)

The most important result is that the full empirical-NTK condition number has a clear local spike at:

```text
k = 0.1875
```

The condition number changes as:

```text
18.7 → 558.8 → 79.1 → 40.3
```

This suggests that the full network Jacobian becomes most ill-conditioned near \(k \approx 0.1875\).

This aligns with the penultimate-feature spectral analysis, where spectral diagnostics also showed a local irregularity near the same \(k\).

---

## Result 2: Stable rank and participation ratio dip near \(k = 0.1875\)

The stable rank normalized by the number of NTK samples changes as:

```text
0.221 → 0.125 → 0.185 → 0.197
```

The participation ratio normalized by the number of NTK samples changes as:

```text
0.440 → 0.177 → 0.311 → 0.347
```

Both quantities reach their minimum at \(k=0.1875\).

This means that the empirical-NTK spectrum becomes more concentrated in fewer eigendirections near this width.

In other words:

> Near \(k \approx 0.1875\), the full empirical-NTK is less effectively full-rank and more spectrally imbalanced.

---

## Result 3: The result supports the Section 6.10 spectral story

Shufeng's Section 6.10 analysis shows that penultimate-feature spectral diagnostics have a local irregularity near the DD-Recovery onset.

This quick full empirical-NTK experiment supports that story from a deeper perspective:

- The earlier experiment looked at penultimate features.
- This experiment looks at the full parameter Jacobian.
- Both show an abnormal spectral pattern near \(k \approx 0.1875\).

Therefore, the spectral irregularity is not only a last-layer representation artifact. It also appears in the full empirical-NTK geometry.

---

## Interpretation

This experiment suggests that the neural-network DD-Recovery onset is associated with a spectral instability in the model's effective kernel geometry.

The connection to the RFF experiments is:

```text
RFF:
p/n ≈ 1
→ feature matrix becomes ill-conditioned
→ solution norm and test MSE spike

Fractional-k ResNet:
k ≈ 0.1875
→ full empirical-NTK becomes ill-conditioned
→ effective rank / participation ratio drop
→ aligns with DD-Recovery onset
```

This strengthens the broader project conclusion:

> Double descent is not controlled by raw parameter count alone. It is closely related to effective geometry, conditioning, and spectral structure.

---

## Limitations

This experiment is intentionally lightweight and should be treated as a supplemental sanity check.

Important limitations:

1. **Small NTK sample size**  
   The empirical-NTK is computed on only 12 samples, so the spectrum may be noisy.

2. **Single seed**  
   The experiment uses only seed 42. Multi-seed validation would be needed for a stronger claim.

3. **Shorter training**  
   The models are trained for 200 epochs, while the full DD-Recovery experiments use longer training. Therefore, this experiment does not replace the full DD-Recovery result.

4. **Limited k sweep**  
   Only four k values are used. A denser sweep would provide a more precise picture.

Because of these limitations, the result should be written as:

> A lightweight full empirical-NTK sanity check that qualitatively supports the penultimate-feature spectral analysis.

It should not be written as a standalone headline result.

---

## Suggested Report Wording

The following paragraph can be added as a supplemental note to Section 6.10:

> To check whether the spectral irregularity observed in the penultimate-feature analysis is also present in the full network Jacobian, we ran a lightweight full empirical-NTK experiment. We trained fractional-width ResNets locally for 200 epochs at \(k \in \{0.125, 0.1875, 0.25, 0.5\}\) and computed the empirical-NTK Gram matrix over 12 samples using Jacobians with respect to all trainable parameters. Although this experiment is only a small-scale sanity check, the full empirical-NTK condition number shows a clear local spike at \(k=0.1875\), while stable rank and participation ratio both dip at the same point. This matches the penultimate-feature spectral analysis and suggests that the irregularity near the DD-Recovery onset is also visible in the full parameter-Jacobian geometry.

---

## Final Conclusion

This experiment provides supplemental support for the neural-network spectral mechanism story.

The main conclusion is:

> In a quick local validation, the full empirical-NTK shows a condition-number spike and effective-rank drop at \(k \approx 0.1875\), matching the spectral irregularity observed in the penultimate-feature analysis. This suggests that the DD-Recovery onset is associated with a broader Jacobian-level spectral instability, not merely a last-layer feature artifact.

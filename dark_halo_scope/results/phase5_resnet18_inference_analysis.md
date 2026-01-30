# Phase 5 ResNet18 Inference Analysis

**Date**: 2026-01-30  
**Author**: AI Assistant (Claude)  
**Purpose**: Document the first complete training and inference run for the Dark Halo Scope lens finder

---

## Executive Summary

We completed training and inference of a ResNet18-based gravitational lens finder. While the model achieves high AUROC (0.9963), detailed analysis reveals that the **false positive rate (FPR) at operationally-useful completeness levels is significantly higher than expected**. This document analyzes the root causes and identifies critical gaps in our data preparation pipeline.

---

## 1. Data Collection Process

### 1.1 Target Galaxy Selection (Phase 3)

We selected Luminous Red Galaxies (LRGs) from the DESI Legacy Survey DR10 as lens candidates.

**Selection Criteria (v3_color_relaxed variant)**:
- Morphology: `TYPE != 'PSF'` (extended sources only)
- Positive flux in all bands: `flux_r > 0`, `flux_z > 0`, `flux_w1 > 0`
- Magnitude cut: `z < 20.4`
- Color cuts: `r - z > 0.4`, `z - W1 > 0.8`

**Output**: ~145,000 LRG targets across the survey footprint

### 1.2 Region Selection and Splitting

Targets were assigned to geographic regions based on brick boundaries, then split into train/val/test sets:

| Split | Percentage | Purpose |
|-------|------------|---------|
| train | 25.9% | Model training |
| val | 39.5% | Hyperparameter tuning, early stopping |
| test | 34.6% | Final evaluation (held out) |

Splits were determined by region to ensure **no spatial leakage** between sets.

---

## 2. Data Preparation Process

### 2.1 Injection Manifest Generation (Phase 4a)

We generated injection tasks specifying lens parameters for each target.

**Injection Grid (grid_small)**:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `theta_e_arcsec` | [0.3, 0.6, 1.0] | Einstein radius |
| `src_dmag` | [1.0, 2.0] | Source magnitude fainter than lens |
| `src_reff_arcsec` | [0.08, 0.15] | Source effective radius |
| `src_e` | [0.0, 0.3] | Source ellipticity |
| `shear` | [0.0, 0.03] | External shear |

**Control Sample Strategy**:
- 50% of samples designated as controls
- Controls use the **same LRG cutouts** but with no lens injection
- Control assignment is deterministic (hash-based) for reproducibility

### 2.2 Image Caching (Phase 4b)

Downloaded DR10 coadd images from NERSC for all bricks containing targets:
- Bands: g, r, z
- Cached locally on S3 for fast access during injection

### 2.3 Lens Injection (Phase 4c)

For each injection task:

1. **Load background**: 64×64 pixel cutout centered on LRG (16.8 arcsec)
2. **Render lensed source**: SIE lens model via lenstronomy with Sersic source profile
3. **Convolve with PSF**: Per-band Gaussian approximation from survey PSF measurements
4. **Add to background**: Flux-correct addition in nanomaggy units
5. **Compute metrics**: arc_snr, magnification, total_injected_flux

**Output Statistics**:
```
Total samples:     10,627,158
Controls:           5,299,324 (49.9%)
Injections:         5,327,834 (50.1%)
Cutout success:     100.0%
```

### 2.4 Stamp Normalization (Phase 5 Training)

During training, stamps are normalized using robust statistics:
```python
for each channel c in [g, r, z]:
    median = np.median(stamp[c])
    mad = np.median(np.abs(stamp[c] - median))
    stamp[c] = (stamp[c] - median) / (1.4826 * mad + epsilon)
    stamp[c] = np.clip(stamp[c], -10, 10)
```

### 2.5 Data Augmentation

- Random horizontal flip (50%)
- Random vertical flip (50%)
- Random 90° rotation (k ∈ {0, 1, 2, 3})

---

## 3. Model Architecture and Training

### 3.1 Architecture

**ResNet18 (modified for 64×64 input)**:
- First conv layer: 3×3 kernel, stride 1, padding 1 (instead of 7×7 stride 2)
- Removed max pooling after first conv
- Final FC layer: 512 → 1 (binary classification)

### 3.2 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Epochs | 10 |
| Loss | BCEWithLogitsLoss |
| Mixed precision | fp16 |

### 3.3 Training Results

| Epoch | Train Loss | Val Loss | Val AUROC |
|-------|-----------|----------|-----------|
| 1 | 0.2831 | 0.1824 | 0.9762 |
| 5 | 0.0892 | 0.0734 | 0.9941 |
| 10 | 0.0612 | 0.0589 | 0.9963 |

**Best model checkpoint**: epoch 10, Val AUROC = 0.9963

---

## 4. Inference Results

### 4.1 Validation Set Composition

```
Total validation samples: ~1.06M
Positives (injections):   ~530k
Negatives (controls):     ~530k
```

### 4.2 FPR vs Completeness (TPR) Analysis

| Completeness (TPR) | FPR | Threshold | log₁₀(FPR) |
|--------------------|-----|-----------|------------|
| 99.0% | 4.14e-01 | 0.0105 | -0.38 |
| 95.0% | 1.88e-01 | 0.0699 | -0.73 |
| 90.0% | 1.02e-01 | 0.1622 | -0.99 |
| **85.0%** | **6.17e-02** | 0.2540 | **-1.21** |
| 80.0% | 3.97e-02 | 0.3380 | -1.40 |
| 70.0% | 1.79e-02 | 0.4879 | -1.75 |
| 50.0% | 5.58e-03 | 0.6870 | -2.25 |

### 4.3 TPR at Fixed FPR Levels

| Target FPR | Actual FPR | Completeness (TPR) | Threshold |
|------------|------------|-------------------|-----------|
| 1e-2 | 1.02e-02 | 74.7% | 0.4295 |
| 1e-3 | 1.01e-03 | 35.2% | 0.7812 |
| 1e-4 | 9.87e-05 | 0.4% | 0.9823 |
| 1e-5 | N/A | ~0% | N/A |

### 4.4 Key Observation

At 85% completeness (a typical operational target), the FPR is **6.17%**. This means:
- For every 100 true lenses detected, we would also flag ~6 non-lenses as candidates
- In a survey with 100,000 targets, this yields ~6,000 false positives requiring human review

---

## 5. Critical Gaps Identified

### 5.1 Gap #1: Trivially-Easy Negative Samples

**Observation**: Controls are simply the same LRG cutouts without any injection.

**Problem**: The model learns to detect "is there extra flux added to this image?" rather than "does this image contain arc-like morphology?"

**Evidence**: The model achieves very high AUROC (0.9963) despite having never seen realistic lens mimics like:
- Ring galaxies
- Spiral galaxies with prominent arms
- Edge-on disk galaxies
- Merger systems with tidal features
- AGN with jets or extended structure

**Impact**: In real survey application, these mimics would cause many false positives that our validation does not measure.

### 5.2 Gap #2: Majority of Injections Are Unresolved

**Observation**: Our Einstein radius range (0.3-1.0 arcsec) overlaps heavily with the PSF size (~1.3 arcsec median).

**Statistics**:
```
Resolution distribution (theta_e / PSF):
- Unresolved (< 0.5):   59.6%
- Marginal (0.5-1.0):   40.3%
- Resolved (>= 1.0):     0.1%
```

**Problem**: For 60% of injections, the arc morphology is completely blurred out by the PSF. The model cannot learn arc-specific features for these samples - it can only learn "there's extra flux somewhere."

**Impact**: The model may fail to generalize to real lenses where arc morphology is the distinguishing feature.

### 5.3 Gap #3: Parametric Source Profiles

**Observation**: We use Sersic elliptical profiles for lensed sources.

**Problem**: Real lensed galaxies have clumpy, irregular morphology from star-forming regions, dust lanes, and structural asymmetries that Sersic profiles do not capture.

**Impact**: The model may not recognize realistic arc morphologies in real data.

### 5.4 Gap #4: No Iterative Hard Negative Mining

**Observation**: Training set is fixed - no feedback loop.

**Problem**: State-of-the-art lens finders typically use iterative training:
1. Train initial model
2. Run on survey data
3. Collect false positives
4. Add to training as hard negatives
5. Repeat

**Impact**: Without hard negative mining, we cannot systematically reduce FPR on challenging cases.

---

## 6. Resolution Analysis

### 6.1 Why Resolution Matters

Strong gravitational lensing produces characteristic arc morphology at the Einstein radius. If theta_e < PSF/2, the arc is unresolved and appears as a point-like flux enhancement rather than an extended arc.

### 6.2 Our Resolution Distribution

| theta_e (arcsec) | Count | Median PSF | theta_e/PSF |
|------------------|-------|------------|-------------|
| 0.3 | 1.78M | 1.31 | 0.23 (unresolved) |
| 0.6 | 1.76M | 1.32 | 0.45 (unresolved) |
| 1.0 | 1.78M | 1.31 | 0.76 (marginal) |

**Conclusion**: Our theta_e grid is too small relative to typical survey PSF. Most injections cannot be distinguished from point sources based on morphology alone.

---

## 7. Lessons Learned

### 7.1 Data Quality > Model Complexity

The ResNet18 model is more than capable of learning the classification task. The performance limitation is entirely in the training data:
- Too-easy negatives (no hard mimics)
- Unresolved positives (no learnable arc morphology)

### 7.2 AUROC Can Be Misleading

High AUROC (0.9963) suggests excellent discrimination, but this metric is dominated by the easy cases. The operationally-relevant metric is FPR at high completeness, which reveals the model's weakness.

### 7.3 Validation Must Match Deployment

Our validation set contains the same distribution as training - easy negatives. In deployment on real survey data, the model will encounter:
- Natural lens mimics (ring galaxies, spirals)
- Image artifacts
- Rare astrophysical phenomena

The gap between validation FPR and deployment FPR is likely substantial.

---

## 8. Recommendations

### 8.1 Immediate Actions

1. **Extend theta_e range**: Use [0.5, 2.5] arcsec to ensure most injections are resolved
2. **Filter training data**: Exclude samples where theta_e/PSF < 0.5
3. **Increase source brightness**: Use src_dmag in [0.5, 1.5] for higher SNR arcs

### 8.2 Medium-Term Actions

1. **Add hard negatives**: Query Galaxy Zoo for ring galaxies, edge-on disks
2. **Validate on external data**: Use known lens samples (e.g., from SLACS, BELLS)
3. **Stratified evaluation**: Report FPR separately by theta_e bin and resolution bin

### 8.3 Long-Term Actions

1. **Replace Sersic sources**: Use real COSMOS/HST galaxy cutouts
2. **Implement hard negative mining**: Iterative training with FP feedback
3. **Ensemble methods**: Combine multiple architectures for robust predictions

---

## 9. Files and Artifacts

### 9.1 Trained Model

```
Location: /lambda/nfs/darkhaloscope-training-dc/phase5/models/resnet18_v1/
Files:
- checkpoint_best.pt (best validation AUROC)
- checkpoint_last.pt (final epoch)
- training.log
```

### 9.2 Training Data

```
S3: s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/stamps/train_stamp64_bandsgrz_gridgrid_small/
Partitions: region_split={train,val,test}
Format: Parquet with stamp_npz binary column
```

### 9.3 Code

```
Training: dark_halo_scope/scripts/train_lambda.py
Inference: dark_halo_scope/model/phase5_infer_scores.py
Pipeline: dark_halo_scope/emr/spark_phase4_pipeline.py
```

---

## 10. Conclusion

The Phase 5 ResNet18 model demonstrates that our training pipeline is functional and the model can learn to distinguish injections from controls. However, the **high FPR at operational completeness levels** reveals fundamental limitations in our training data:

1. **Controls are too easy** - no realistic lens mimics
2. **Most injections are unresolved** - cannot learn arc morphology
3. **No hard negative mining** - cannot iteratively improve

These are **data problems, not model problems**. Improving model architecture or training longer will not address these gaps. The path forward requires regenerating training data with:
- Larger Einstein radii (resolved arcs)
- Hard negative samples (ring galaxies, spirals, etc.)
- Real galaxy morphologies for sources

This document serves as a record of our first complete training run and the lessons learned for future iterations.

---

*Document created: 2026-01-30*  
*Last updated: 2026-01-30*


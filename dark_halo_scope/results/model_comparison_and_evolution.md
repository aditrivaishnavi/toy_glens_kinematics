# Dark Halo Scope: Model Comparison and Evolution Document

**Project**: CNN-Based Strong Gravitational Lens Finder for DESI Legacy Survey DR10  
**Target Publication**: MNRAS / ApJ / AAS  
**Last Updated**: 2026-02-03

---

## Table of Contents

1. [Model Generation 1: ResNet18 on v3_color_relaxed](#generation-1-resnet18-on-v3_color_relaxed)
2. [Model Generation 2: ConvNeXt-Tiny on v4_sota (Gaussian PSF)](#generation-2-convnext-tiny-on-v4_sota-gaussian-psf)
3. [Model Generation 3: ConvNeXt-Tiny on v4_sota_moffat (Moffat PSF)](#generation-3-convnext-tiny-on-v4_sota_moffat-moffat-psf)
4. [Model Generation 4: ConvNeXt-Tiny with Hard Negative Mining](#generation-4-convnext-tiny-with-hard-negative-mining)
5. [Comparative Analysis](#comparative-analysis)
6. [Key Insights and Lessons Learned](#key-insights-and-lessons-learned)
7. [Sim-to-Real Gap Analysis](#sim-to-real-gap-analysis)
8. [Generation 5: COSMOS Source Integration (In Preparation)](#generation-5-cosmos-source-integration-in-preparation)

---

## Generation 1: ResNet18 on v3_color_relaxed

### Overview

| Property | Value |
|----------|-------|
| Model Name | ResNet18-v3 |
| Data Version | v3_color_relaxed |
| Training Date | 2026-01-28 to 2026-01-29 |
| Training Platform | Lambda Labs GH200 (96GB) |
| Status | ✅ Completed |

### Phase 3: Target Galaxy Selection

**Selection Criteria (v3_color_relaxed)**:
```python
# Galaxy morphology filter
TYPE != 'PSF'  # Extended sources only

# Flux requirements
flux_r > 0 AND flux_z > 0 AND flux_w1 > 0

# Magnitude cut
z < 20.4

# LRG color cuts
r - z > 0.4
z - W1 > 0.8
```

**Output**: ~145,000 LRG targets across the DESI Legacy Survey footprint

**Region Splits**:
| Split | Percentage |
|-------|------------|
| train | 25.9% |
| val | 39.5% |
| test | 34.6% |

### Phase 4a: Injection Manifest Generation

**Configuration** (`_stage_config.json`):
```json
{
  "variant": "v3_color_relaxed",
  "tiers": {
    "train": {
      "grid": "grid_small",
      "n_total_per_split": 200000,
      "control_frac": 0.5
    }
  }
}
```

**Injection Grid: `grid_small`**:
| Parameter | Values | Description |
|-----------|--------|-------------|
| theta_e_arcsec | [0.3, 0.6, 1.0] | Einstein radius |
| src_dmag | [1.0, 2.0] | Source magnitude fainter than lens |
| src_reff_arcsec | [0.08, 0.15] | Source effective radius |
| src_e | [0.0, 0.3] | Source ellipticity |
| shear | [0.0, 0.03] | External shear |

**Total configurations**: 3 × 2 × 2 × 2 × 2 = 48 injection configs

**Control Strategy**: 
- **PAIRED CONTROLS** ❌ (same galaxy, no injection)
- 50% of samples marked as controls with theta_e=0
- Hash-based deterministic assignment per galaxy

### Phase 4c: Lens Injection

**Injection Method**:
- Lens model: SIE (Singular Isothermal Ellipsoid) via lenstronomy
- Source profile: Sersic (n=1, exponential disk)
- PSF convolution: **Gaussian** approximation from survey psfsize_r
- Pixel scale: 0.262 arcsec/pixel

**Output Statistics**:
```
Total samples:     ~10.6M
Controls:          ~5.3M (49.9%)
Injections:        ~5.3M (50.1%)
Cutout success:    100%
Compression:       Snappy (parquet default)
```

### Phase 5: Model Training

**Architecture**: ResNet18 (modified for 64×64 input)
```python
# Modifications from standard ResNet18:
- conv1: 3×3 kernel, stride=1, padding=1 (not 7×7 stride 2)
- maxpool: removed (Identity)
- fc: 512 → 1 (binary classification)
```

**Training Configuration**:
| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Epochs | 10 |
| Loss | BCEWithLogitsLoss |
| Mixed precision | fp16 |
| Augmentation | Flip + Rot90 |

### Evaluation Results

**Final Model Performance**:
| Metric | Value |
|--------|-------|
| AUROC | 0.9963 |
| tpr@fpr1e-4 | **0.4%** |
| tpr@fpr1e-3 | 35.2% |
| tpr@fpr1e-2 | 74.7% |
| fpr@tpr0.85 | 6.2% |

### Issues Identified

1. **Paired Controls (Critical)**: Controls used the SAME galaxy as positives. Model learned to detect "extra flux added" rather than arc morphology.

2. **Unresolved Injections (Critical)**: With theta_e range [0.3, 0.6, 1.0] and median PSF ~1.3", ~60% of injections had theta_e/PSF < 0.5 (unresolved).

3. **Gaussian PSF Approximation**: Real DECaLS PSFs have extended wings; Gaussian underestimates this.

---

## Generation 2: ConvNeXt-Tiny on v4_sota (Gaussian PSF)

### Overview

| Property | Value |
|----------|-------|
| Model Name | ConvNeXt-v4sota |
| Data Version | v4_sota |
| Training Date | 2026-01-31 |
| Training Platform | Lambda Labs GH200 (96GB) |
| Status | ✅ Completed |
| Checkpoint Location | `/lambda/nfs/darkhaloscope-training-dc/runs/gen2_final/` |

### Key Changes from Gen1

| Aspect | Gen1 | Gen2 | Impact |
|--------|------|------|--------|
| Control Type | Paired ❌ | **Unpaired** ✅ | No shortcut learning |
| theta_e range | 0.3-1.0" | **0.5-2.5"** | Resolvable lenses |
| Injection Grid | 48 configs | **1,008 configs** | Better coverage |
| Architecture | ResNet18 | **ConvNeXt-Tiny** | More capacity |
| Loss | BCE | BCE | Same |

### Phase 4a: Injection Manifest Generation

**Injection Grid: `grid_sota`** (Extended for Resolved Lenses):
| Parameter | Values | Description |
|-----------|--------|-------------|
| theta_e_arcsec | [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5] | Einstein radius (extended!) |
| src_dmag | [0.5, 1.0, 1.5, 2.0] | Source brightness |
| src_reff_arcsec | [0.06, 0.10, 0.15, 0.20] | Source size |
| src_e | [0.0, 0.2, 0.4] | Source ellipticity |
| shear | [0.0, 0.02, 0.04] | External shear |

**Total configurations**: 7 × 4 × 4 × 3 × 3 = 1,008 injection configs

**Control Strategy**:
- **UNPAIRED CONTROLS** ✅ (different galaxies for controls vs positives)
- Hash-based deterministic assignment ensures disjoint galaxy sets
- Overlap verification: **0 galaxies** shared between controls and positives

### Phase 4c: Lens Injection

**Injection Method**:
- Lens model: SIE (Singular Isothermal Ellipsoid)
- Source profile: Sersic (n=1, exponential disk)
- PSF convolution: **Gaussian** (psfsize per-band: g, r, z)
- Pixel scale: 0.262 arcsec/pixel

**Commands Used**:
```bash
spark-submit \
  --deploy-mode cluster \
  --conf spark.sql.parquet.compression.codec=gzip \
  spark_phase4_pipeline.py \
  --stage 4c \
  --output-s3 s3://darkhaloscope/phase4_pipeline \
  --variant v4_sota \
  --experiment-id train_stamp64_bandsgrz_gridgrid_sota
```

### Phase 5: Model Training

**Training Configuration**:
| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-2 |
| Batch size | 512 |
| Epochs | 7 (early stopped) |
| Loss | BCEWithLogitsLoss |
| Mixed precision | bf16 |
| Augmentation | Flip + Rot90 |
| Early stopping | Patience 5 |

**Training Command**:
```bash
PYTHONUNBUFFERED=1 python3 -u phase5_train_fullscale_gh200_v2.py \
    --data /lambda/nfs/darkhaloscope-training-dc/phase4c_v4_sota \
    --out_dir /lambda/nfs/darkhaloscope-training-dc/runs/gen2_final \
    --arch convnext_tiny \
    --epochs 50 \
    --batch_size 512 \
    --lr 3e-4 \
    --use_bf16 \
    --augment \
    --early_stopping_patience 5
```

### Evaluation Results

**Training Progression**:
| Epoch | Train Loss | AUROC | tpr@fpr1e-4 | tpr@fpr1e-3 |
|-------|-----------|-------|-------------|-------------|
| 0 | 0.0828 | 0.9910 | 69.8% | 83.1% |
| 1 | 0.0234 | 0.9892 | 78.8% | 81.5% |
| 2 | 0.0107 | 0.9894 | 77.8% | 82.2% |
| 3 | 0.0071 | 0.9902 | 79.8% | 85.4% |
| **4** | 0.0049 | 0.9906 | **75.1%** ★ | 86.1% |

**Best Model (Epoch 4)**:
| Metric | Value |
|--------|-------|
| AUROC | 0.9906 |
| **tpr@fpr1e-4** | **75.1%** ★ Best across all generations |
| tpr@fpr1e-3 | 86.1% |
| tpr@fpr1e-2 | 93.3% |

### Known Issues

1. **Worker Sharding Bug**: DataLoader workers processed same fragments (8x duplication) - fixed in Gen3+
2. **No resolvability filter**: All theta_e values included, some still unresolved
3. **No metadata fusion**: PSF/depth conditioning not used

---

## Generation 3: ConvNeXt-Tiny on v4_sota_moffat (Moffat PSF)

### Overview

| Property | Value |
|----------|-------|
| Model Name | ConvNeXt-v4sota-moffat |
| Data Version | v4_sota_moffat |
| Training Date | 2026-02-01 |
| Training Platform | Lambda Labs GH200 (96GB) |
| Status | ✅ Completed |
| Checkpoint Location | `/lambda/nfs/darkhaloscope-training-dc/runs/gen3_moffat/` |

### Key Changes from Gen2

| Aspect | Gen2 | Gen3 | Rationale |
|--------|------|------|-----------|
| PSF Model | Gaussian | **Moffat (β=3.5)** | Extended wings, more realistic |
| Loss | BCE | **Focal Loss** | Focus on hard examples |
| Resolvability Filter | None | **θ/PSF ≥ 0.5** | Exclude unresolved |
| Worker Sharding | Bug | **Fixed** | No sample duplication |
| Metadata Guard | None | **Active** | Block label-leaking columns |
| Normalization | Full image | **Outer annulus** | Reduce injection leakage |

### Phase 4c: Lens Injection

**Key Difference**: Moffat PSF with β=3.5 instead of Gaussian

**Commands Used**:
```bash
spark-submit \
  --deploy-mode cluster \
  --conf spark.sql.parquet.compression.codec=gzip \
  spark_phase4_pipeline.py \
  --stage 4c \
  --output-s3 s3://darkhaloscope/phase4_pipeline \
  --variant v4_sota_moffat \
  --psf-model moffat \
  --moffat-beta 3.5 \
  --experiment-id train_stamp64_bandsgrz_gridgrid_sota
```

### Phase 5: Model Training

**Training Configuration**:
| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-2 |
| Batch size | 256 |
| Epochs | 8 (early stopped) |
| **Loss** | **Focal (α=0.25, γ=2.0)** |
| Mixed precision | bf16 |
| Augmentation | Flip + Rot90 |
| **min_theta_over_psf** | **0.5** |
| **norm_method** | **outer** |
| meta_cols | psfsize_r, psfdepth_r |

**Training Command**:
```bash
PYTHONUNBUFFERED=1 python3 -u phase5_train_fullscale_gh200_v2.py \
    --data /lambda/nfs/darkhaloscope-training-dc/phase4c_v4_sota_moffat \
    --out_dir /lambda/nfs/darkhaloscope-training-dc/runs/gen3_moffat \
    --arch convnext_tiny \
    --epochs 50 \
    --batch_size 256 \
    --lr 3e-4 \
    --use_bf16 \
    --augment \
    --loss focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --min_theta_over_psf 0.5 \
    --norm_method outer \
    --meta_cols psfsize_r,psfdepth_r \
    --early_stopping_patience 5
```

### Evaluation Results

**Training Progression**:
| Epoch | Train Loss | AUROC | tpr@fpr1e-4 | tpr@fpr1e-3 | binary_frac |
|-------|-----------|-------|-------------|-------------|-------------|
| 0 | - | 0.9636 | 30.1% | 46.9% | 0.0% |
| 1 | - | 0.9802 | 33.0% | 71.3% | 1.3% |
| 2 | - | 0.9818 | 29.2% | 74.4% | 21.9% |
| 3 | - | 0.9842 | 54.2% | 77.4% | 42.6% |
| **4** | - | 0.9829 | **66.8%** ★ | 78.1% | 62.2% |
| 5 | - | 0.9829 | 57.4% | 78.8% | 84.9% |
| 6 | - | 0.9808 | 57.7% | 78.7% | 93.4% |
| 7 | - | 0.9809 | 55.8% | 80.8% | 93.1% |

**Best Model (Epoch 4)**:
| Metric | Value |
|--------|-------|
| AUROC | 0.9829 |
| **tpr@fpr1e-4** | **66.8%** |
| tpr@fpr1e-3 | 78.1% |

### ⚠️ Issues Identified

1. **Calibration Collapse**: binary_score_frac increased from 0% to 93% over training
   - Model becoming overconfident
   - Scores clustering at 0 or 1, losing calibration

2. **Performance DECREASED from Gen2**: 
   - Gen2: 75.1% tpr@fpr1e-4
   - Gen3: 66.8% tpr@fpr1e-4
   - Moffat PSF actually hurt synthetic performance

3. **Possible explanation**: Moffat PSF makes arcs harder to detect (more spread out), but also makes negatives look more lens-like

---

## Generation 4: ConvNeXt-Tiny with Hard Negative Mining

### Overview

| Property | Value |
|----------|-------|
| Model Name | ConvNeXt-v4sota-moffat-hardneg |
| Data Version | v4_sota_moffat (same as Gen3) |
| Training Date | 2026-02-02 |
| Training Platform | Lambda Labs GH200 (96GB) |
| Status | ✅ Completed |
| Checkpoint Location | `/lambda/nfs/darkhaloscope-training-dc/models/gen4_hardneg/` |

### Key Changes from Gen3

| Aspect | Gen3 | Gen4 | Rationale |
|--------|------|------|-----------|
| Training Data | Base only | **Base + Hard Negatives** | Push FPR down |
| Hard Neg Source | N/A | **Gen2 + Gen3 FPs** | Controls scoring >0.9 |
| Hard Neg Weight | N/A | **5x upsampling** | Increase HN importance |
| Hard Neg Count | 0 | **~35k unique positions** | Substantial coverage |

### Hard Negative Mining Strategy

**Source**: False positives from Gen2 and Gen3 inference
- Gen2: ~20k controls with p_lens > 0.9
- Gen3: ~20k controls with p_lens > 0.9
- Total: ~40k candidates

**Processing**:
1. Merge Gen2 + Gen3 hard negatives
2. Deduplicate by (ra, dec) position → ~35k unique
3. Create lookup parquet with `ra_key`, `dec_key` (rounded to 6 decimals)
4. During training, match by position and upsample 5x

**Hard Negative Preparation Script**:
```python
# From run_gen4_training.sh
merged = pd.concat([gen2_hn, gen3_hn], ignore_index=True)
merged = merged.drop_duplicates(subset=["ra", "dec"])
merged["ra_key"] = np.round(merged["ra"], 6)
merged["dec_key"] = np.round(merged["dec"], 6)
lookup = merged[["ra_key", "dec_key", "brickname", "source_model"]]
lookup.to_parquet("hard_neg_lookup.parquet", index=False)
```

### Phase 5: Model Training

**Training Configuration**:
| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-2 |
| Batch size | 256 |
| Epochs | 13 (early stopped at 12) |
| Loss | Focal (α=0.25, γ=2.0) |
| Mixed precision | bf16 |
| Augmentation | Flip + Rot90 |
| min_theta_over_psf | 0.5 |
| norm_method | outer |
| meta_cols | psfsize_r, psfdepth_r |
| **hard_neg_path** | `hard_neg_lookup.parquet` |
| **hard_neg_weight** | **5** |

**Training Command**:
```bash
PYTHONUNBUFFERED=1 python3 -u phase5_train_gen4_hardneg.py \
    --data /lambda/nfs/darkhaloscope-training-dc/phase4c_v4_sota_moffat \
    --out_dir /lambda/nfs/darkhaloscope-training-dc/models/gen4_hardneg \
    --hard_neg_path /lambda/nfs/darkhaloscope-training-dc/hard_negatives/merged/hard_neg_lookup.parquet \
    --hard_neg_weight 5 \
    --arch convnext_tiny \
    --epochs 50 \
    --batch_size 256 \
    --lr 3e-4 \
    --weight_decay 1e-2 \
    --dropout 0.1 \
    --use_bf16 \
    --augment \
    --loss focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --min_theta_over_psf 0.5 \
    --norm_method outer \
    --meta_cols psfsize_r,psfdepth_r \
    --early_stopping_patience 5
```

### Evaluation Results

**Training Progression**:
| Epoch | Train Loss | AUROC | tpr@fpr1e-4 | tpr@fpr1e-3 | binary_frac | hard_neg_yield |
|-------|-----------|-------|-------------|-------------|-------------|----------------|
| 0 | 0.0109 | 0.9526 | 13.3% | 50.7% | 0.1% | ~12k |
| 1 | 0.0069 | 0.9746 | 32.5% | 64.4% | 0.0% | ~12k |
| 2 | 0.0054 | 0.9849 | 50.1% | 64.9% | 3.6% | ~13k |
| 3 | 0.0044 | 0.9912 | 56.0% | 74.1% | 0.0% | ~12k |
| 4 | 0.0038 | 0.9894 | 66.3% | 76.1% | 19.4% | ~13k |
| 5 | 0.0033 | 0.9873 | 63.4% | 74.9% | 33.4% | ~11k |
| 6 | 0.0028 | 0.9920 | 64.9% | 82.3% | 2.3% | ~12k |
| **7** | 0.0024 | 0.9904 | **74.5%** ★ | 80.1% | 8.1% | ~12k |
| 8 | 0.0021 | 0.9883 | 66.7% | 80.3% | 32.1% | ~14k |
| 9 | 0.0018 | 0.9872 | 70.3% | 76.3% | 38.6% | ~14k |
| 10 | 0.0015 | 0.9894 | 74.1% | 81.3% | 31.2% | ~12k |
| 11 | 0.0012 | 0.9877 | 67.2% | 83.2% | 2.9% | ~12k |
| 12 | 0.0011 | 0.9900 | 68.8% | 82.4% | 53.6% | ~13k |

**Best Model (Epoch 7)**:
| Metric | Value |
|--------|-------|
| AUROC | 0.9904 |
| **tpr@fpr1e-4** | **74.5%** |
| tpr@fpr1e-3 | 80.1% |
| tpr@fpr1e-2 | 93.0% |

**Training Statistics Per Epoch**:
- Train batches: 6,027
- Samples yielded: ~195k
- Hard negatives yielded: ~12-14k (with 5x weight)
- Total effective hard neg samples: ~60-70k per epoch

### Checkpoint Files

| File | Size | Description |
|------|------|-------------|
| `ckpt_best.pt` | 323 MB | Epoch 7 (best tpr@fpr1e-4) |
| `ckpt_last.pt` | 323 MB | Epoch 12 (final) |
| `ckpt_epoch_0.pt` | 323 MB | Initial checkpoint |
| `ckpt_epoch_5.pt` | 323 MB | Epoch 5 |
| `ckpt_epoch_10.pt` | 323 MB | Epoch 10 |

### ⚠️ Issues Identified

1. **Calibration Collapse (Final Epoch)**: binary_score_frac = 53.6%
   - Scores clustering at extremes
   - Loss of probability calibration

2. **Validation Set Imbalance**:
   - pos_eval: 5,064 (4%)
   - neg_eval: 122,936 (96%)
   - Very different from 50/50 training distribution

3. **Hard Negatives Did NOT Help Significantly**:
   - Gen2 (no HN): 75.1% tpr@fpr1e-4
   - Gen4 (with HN): 74.5% tpr@fpr1e-4
   - Synthetic hard negatives may not represent real contaminants

4. **Training Instability**: tpr@fpr1e-4 fluctuated between 63-75% across epochs

---

## Comparative Analysis

### Complete Results Summary

| Generation | Data | PSF | Controls | Hard Neg | Best tpr@fpr1e-4 | tpr@fpr1e-3 | AUROC |
|------------|------|-----|----------|----------|------------------|-------------|-------|
| **Gen1** | v3_color_relaxed | Gaussian | Paired ❌ | No | 0.4% | 35.2% | 0.9963 |
| **Gen2** | v4_sota | Gaussian | Unpaired ✅ | No | **75.1%** ★ | 86.1% | 0.9906 |
| **Gen3** | v4_sota_moffat | Moffat | Unpaired ✅ | No | 66.8% | 78.1% | 0.9829 |
| **Gen4** | v4_sota_moffat | Moffat | Unpaired ✅ | Yes (35k) | 74.5% | 80.1% | 0.9904 |

### Performance Progression Chart

```
tpr@fpr1e-4 Performance:

Gen1 (Paired)      ▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.4%
Gen2 (Gaussian)    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░  75.1% ★ BEST
Gen3 (Moffat)      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░  66.8%
Gen4 (Moffat+HN)   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░  74.5%
```

### Data Pipeline Comparison

| Aspect | Gen 1 | Gen 2 | Gen 3 | Gen 4 |
|--------|-------|-------|-------|-------|
| theta_e range | 0.3-1.0" | 0.5-2.5" | 0.5-2.5" | 0.5-2.5" |
| Control Type | Paired ❌ | Unpaired ✅ | Unpaired ✅ | Unpaired ✅ |
| PSF Model | Gaussian | Gaussian | **Moffat** | **Moffat** |
| Resolvability Filter | None | None | θ/PSF≥0.5 | θ/PSF≥0.5 |
| Hard Negatives | No | No | No | **Yes (35k)** |

### Training Configuration Comparison

| Aspect | Gen 1 | Gen 2 | Gen 3 | Gen 4 |
|--------|-------|-------|-------|-------|
| Architecture | ResNet18 | ConvNeXt-Tiny | ConvNeXt-Tiny | ConvNeXt-Tiny |
| Loss | BCE | BCE | **Focal** | **Focal** |
| Precision | fp16 | bf16 | bf16 | bf16 |
| Batch Size | 32 | 512 | 256 | 256 |
| Normalization | Full | Full | **Outer** | **Outer** |
| Worker Sharding | Bug | Bug | Fixed | Fixed |

---

## Key Insights and Lessons Learned

### Insight 1: Unpaired Controls >> Everything Else

The 188x improvement from Gen1→Gen2 came almost entirely from unpaired controls:
- Gen1 with paired controls: 0.4%
- Gen2 with unpaired controls: 75.1%

This is the single most important fix for synthetic lens training.

### Insight 2: Moffat PSF Did NOT Help (Unexpectedly)

Despite theoretical expectations:
- Gen2 (Gaussian): 75.1%
- Gen3 (Moffat): 66.8%

**Possible explanations**:
1. Moffat spreads flux into extended wings, making both lenses AND non-lenses look more similar
2. Gaussian PSF may be "unrealistically easy" but this helps synthetic metrics
3. The sim-to-real gap is NOT primarily about PSF model

### Insight 3: Synthetic Hard Negatives Have Limited Value

- Gen3 (no HN): 66.8%
- Gen4 (with 35k HN, 5x weight): 74.5%

The improvement exists but is modest. Synthetic hard negatives (high-scoring controls from previous models) may not represent **real** contaminants like:
- Ring galaxies
- Mergers
- Spirals with prominent arms
- Artifacts

### Insight 4: Calibration Collapse is a Warning Sign

Both Gen3 and Gen4 showed increasing `binary_score_frac`:
- Early epochs: 0-5%
- Late epochs: 30-93%

This indicates:
- Model is memorizing training data
- Scores are not calibrated probabilities
- May perform poorly on out-of-distribution data

### Insight 5: AUROC is Not a Reliable Metric

All generations achieve AUROC > 0.98, but operational performance varies enormously:
- Gen1: AUROC 0.9963, tpr@fpr1e-4 = 0.4%
- Gen2: AUROC 0.9906, tpr@fpr1e-4 = 75.1%

**Always report tpr@fpr and fpr@tpr at operationally relevant thresholds.**

---

## Sim-to-Real Gap Analysis

### What's Been Fixed

| Gap | Gen1 | Gen2 | Gen3 | Gen4 |
|-----|------|------|------|------|
| Paired controls | ❌ | ✅ | ✅ | ✅ |
| Unresolved lenses | ❌ | ⚠️ | ✅ | ✅ |
| Per-band PSF | ❌ | ✅ | ✅ | ✅ |
| Moffat PSF | ❌ | ❌ | ✅ | ✅ |
| Worker sharding bug | ❌ | ❌ | ✅ | ✅ |
| Outer normalization | ❌ | ❌ | ✅ | ✅ |
| Synthetic hard negatives | ❌ | ❌ | ❌ | ✅ |

### What's Still Missing

| Gap | Description | Priority |
|-----|-------------|----------|
| **Real source morphology** | Using Sersic n=1, not COSMOS galaxies | HIGH |
| **Real hard negatives** | Using synthetic HN, not GZ Rings/Mergers | HIGH |
| **Anchor baseline validation** | Never tested on SLACS/BELLS | CRITICAL |
| PSFEx models | DR10 doesn't provide, using psfsize | LOW |
| Spatially varying PSF | Using center-evaluated only | LOW |
| CCD artifacts | No cosmic rays, bad pixels | LOW |
| Color gradients | Source colors uniform | MEDIUM |

### Critical Next Step: COMPLETED ✅

**Anchor baseline evaluation completed on 2026-02-02**

---

## Stage 0: Anchor Baseline Results (CRITICAL)

### 🚨 CATASTROPHIC SIM-TO-REAL GAP CONFIRMED

**The model that achieved 75% tpr@fpr1e-4 on synthetic data can only detect 2.9% of real confirmed lenses!**

### Test Setup

- **Model**: Gen2 (best synthetic performance: 75.1% tpr@fpr1e-4)
- **Known Lenses**: 68 (48 SLACS + 20 BELLS)
- **Hard Negatives**: 14 in DR10 footprint (rings, mergers)
- **Cutout Service**: Legacy Survey DR10 via cutout API

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Recall @ 0.5** | **2.9%** (2/68) | >50% | ❌ FAILED |
| Recall @ 0.7 | 1.5% (1/68) | - | ❌ |
| Recall @ 0.9 | 0.0% (0/68) | - | ❌ |
| Contamination @ 0.5 | 7.1% (1/14) | <20% | ✅ OK |

### Score Distributions

| Set | Mean p_lens | Median p_lens | Std |
|-----|-------------|---------------|-----|
| **Known Lenses** | **0.232** | 0.209 | 0.126 |
| **Hard Negatives** | **0.323** | 0.290 | 0.223 |

⚠️ **The model scores hard negatives HIGHER than known lenses on average!**

### Top Detected Lenses

| Lens | p_lens | θ_e (arcsec) | Notes |
|------|--------|--------------|-------|
| SDSSJ0912+0029 | 0.749 | 1.63" | ✅ Only lens >0.7 |
| SDSSJ1205+4910 | 0.582 | 1.22" | ✅ Second detection |
| SDSSJ2300+0022 | 0.481 | 1.24" | ❌ Below threshold |
| BELLSJ1401+3845 | 0.440 | 1.35" | ❌ Below threshold |

### Completely Missed Lenses (Examples)

| Lens | p_lens | θ_e (arcsec) | Notes |
|------|--------|--------------|-------|
| SDSSJ1531-0105 | 0.042 | 1.71" | Large θ_e, still missed |
| SDSSJ1538+5817 | 0.069 | 1.0" | |
| SDSSJ1432+6317 | 0.081 | 1.25" | |
| SDSSJ1416+5136 | 0.082 | 1.37" | |

### False Positive (Hard Negative Scored High)

| Object | p_lens | Type |
|--------|--------|------|
| Merger002 | 0.991 | Merger galaxy |

### Root Cause Analysis

The synthetic training data differs from real lenses in several critical ways:

1. **Source Morphology**: Synthetic uses smooth Sersic n=1 profiles. Real lensed sources are clumpy, irregular galaxies with color gradients.

2. **Arc Appearance**: Synthetic arcs are smooth, symmetric. Real arcs have substructure, dust, star-forming regions.

3. **Lens Galaxy Properties**: SLACS/BELLS lenses are massive ellipticals at z~0.1-0.5. Our training focused on DECaLS LRGs which may differ.

4. **Training Shortcut**: Model learned to detect "smooth arc-like feature" which is NOT what real lensed arcs look like.

### Conclusion

**GATING RULE FAILED**: Further model architecture iteration is pointless until we fix the simulation realism.

**Priority Actions**:
1. **COSMOS source morphology** - Use real galaxy images as lensed sources
2. **Real hard negatives in training** - Include rings, mergers, spirals
3. **Validate after each change** - Re-run anchor baseline before proceeding

---

## Generation 5: COSMOS Source Integration (In Preparation)

### Overview

| Property | Value |
|----------|-------|
| Model Name | ConvNeXt-v5-COSMOS |
| Data Version | v5_cosmos_source |
| Training Date | TBD |
| Training Platform | Lambda Labs GH200 (96GB) |
| Status | 🔧 Pipeline Validated, Awaiting Production Run |

### Key Changes from Gen4

| Aspect | Gen4 | Gen5 | Rationale |
|--------|------|------|-----------|
| Source Profile | Sersic n=1 | **COSMOS Real Galaxies** | Clumpy, irregular morphology |
| Source Morphology | Smooth, symmetric | **Realistic substructure** | Bridge sim-to-real gap |
| COSMOS Bank | None | **~50k HST F814W stamps** | Real galaxy morphologies |
| Lens Model | SIE | SIE + **lenstronomy rendering** | More accurate magnification |

### Pipeline Validation (2026-02-03)

The Gen5 Spark pipeline was extensively tested and debugged. Key findings:

#### Bugs Fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| Module-level boto3 import | boto3=None on executors | Import boto3 inside functions |
| PSF kernel size > stamp | 46/50 stamps failed | Cap kernel radius to 31 (max 63×63) |
| `--bands grz` parsing | Treated as single band | Use `--bands g,r,z` (comma-separated) |
| s3:// vs s3a:// paths | Local Spark failed | Handle both prefixes |
| Inefficient smoke test | Full manifest scan even with `--test-limit` | Read single partition file directly |

#### arc_snr Variance Investigation

**Observation**: arc_snr ranged from 0.007 to 227 across test stamps.

**Investigation**: 4 stamps had arc_snr < 1 despite visible arc signal in the image.

**Root Cause**: These stamps fell on regions masked by DR10's MEDIUM star flag (bit 11):

```
Stamp at RA=121.389571, Dec=19.167777 (brick 1214p192):
  Good pixels: 1035/4096 (25.3%)
  Ring pixels: 68
  Ring pixels that are GOOD: 0    ← 100% of Einstein ring is masked!
  Ring pixels that are BAD: 68
```

**Conclusion**: arc_snr calculation is **correct**:
1. The arc IS rendered and added to the image (visible)
2. But DR10 maskbits flag the region as near a medium-bright star
3. arc_snr only considers unmasked pixels (correct behavior)
4. Low arc_snr correctly indicates the stamp is unreliable for detection

| arc_snr Range | Cause | Interpretation |
|---------------|-------|----------------|
| < 1 | Arc in masked region (MEDIUM star) | Correctly flagged as unreliable |
| 1-10 | Faint arcs or partial masking | Weak but detectable |
| 10-50 | Typical lensing signals | Good training samples |
| > 50 | Bright arcs, high magnification | Strong lensing events |

**Key Takeaway**: Wide arc_snr variance is expected and correctly reflects data quality. Stamps with low arc_snr will be filtered via `metrics_ok` or SNR thresholds.

#### Local Spark Smoke Test Results

```
Total stamps processed: 50
Successful (cutout_ok=1): 50/50 (100%)
arc_snr: mean=39.9, std=48.8, range=[0.007, 227]
Bricks processed: 26 unique bricks
```

### Production Run Prerequisites

1. **Coadd Cache**: All bricks in manifest must be in S3 coadd cache
2. **COSMOS Bank**: H5 file with ~50k galaxy stamps ready
3. **EMR Cluster**: 20+ m5.xlarge executors recommended

### Expected Improvements from Gen5

Based on anchor baseline failure analysis:

| Issue | Gen4 Status | Gen5 Solution |
|-------|-------------|---------------|
| Smooth Sersic sources | ❌ Unrealistic | ✅ COSMOS real galaxies |
| Missing source clumpiness | ❌ | ✅ HST F814W morphology |
| Arc substructure | ❌ None | ✅ Preserved from COSMOS |
| Color gradients | ❌ Uniform | ⚠️ Still monochromatic (future work) |

### S3 Data Locations

```
COSMOS Bank:     s3://darkhaloscope/cosmos/cosmos_source_bank_v1.h5 (TBD)
Gen5 Pipeline:   s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_source/
EMR Bootstrap:   s3://darkhaloscope/emr/gen5/emr_bootstrap_gen5.sh
```

---

## File Locations

### Models (Lambda NFS)
```
Gen 1: /lambda/nfs/darkhaloscope-training/phase5/models/resnet18_v1/
Gen 2: /lambda/nfs/darkhaloscope-training-dc/runs/gen2_final/
Gen 3: /lambda/nfs/darkhaloscope-training-dc/runs/gen3_moffat/
Gen 4: /lambda/nfs/darkhaloscope-training-dc/models/gen4_hardneg/
```

### Data (S3)
```
v3_color_relaxed: s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/
v4_sota:          s3://darkhaloscope/phase4_pipeline/phase4c/v4_sota/
v4_sota_moffat:   s3://darkhaloscope/phase4_pipeline/phase4c/v4_sota_moffat/
```

### Code
```
Pipeline:     dark_halo_scope/emr/spark_phase4_pipeline.py
Training:     dark_halo_scope/models/gen3_moffat/phase5_train_fullscale_gh200_v2.py
Gen4 Script:  dark_halo_scope/models/gen4_hardneg/phase5_train_gen4_hardneg.py
```

---

*Document created: 2026-01-31*  
*Last updated: 2026-02-03*  
*This is a living document - update as new models are trained.*

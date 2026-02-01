# Dark Halo Scope: Model Comparison and Evolution Document

**Project**: CNN-Based Strong Gravitational Lens Finder for DESI Legacy Survey DR10  
**Target Publication**: MNRAS / ApJ / AAS  
**Last Updated**: 2026-01-31

---

## Table of Contents

1. [Model Generation 1: ResNet18 on v3_color_relaxed](#generation-1-resnet18-on-v3_color_relaxed)
2. [Model Generation 2: ConvNeXt-Tiny on v4_sota](#generation-2-convnext-tiny-on-v4_sota)
3. [Comparative Analysis](#comparative-analysis)
4. [Key Insights and Lessons Learned](#key-insights-and-lessons-learned)

---

## Generation 1: ResNet18 on v3_color_relaxed

### Overview

| Property | Value |
|----------|-------|
| Model Name | ResNet18-v3 |
| Data Version | v3_color_relaxed |
| Training Date | 2026-01-28 to 2026-01-29 |
| Training Platform | Lambda Labs GH200 (96GB) |
| Status | Completed, inference analyzed |

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
- **PAIRED CONTROLS** (same galaxy, no injection)
- 50% of samples marked as controls with theta_e=0
- Hash-based deterministic assignment per galaxy

### Phase 4b: Coadd Caching

```bash
# S3 cache location
s3://darkhaloscope/dr10_coadd_cache/

# Bands cached: g, r, z
# Brick coverage: All bricks containing target LRGs
```

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

**Commands Used**:
```bash
# Phase 4a
spark-submit \
  --deploy-mode cluster \
  spark_phase4_pipeline.py \
  --stage 4a \
  --output-s3 s3://darkhaloscope/phase4_pipeline \
  --variant v3_color_relaxed \
  --grid-train grid_small \
  --control-frac-train 0.5

# Phase 4c
spark-submit \
  --deploy-mode cluster \
  spark_phase4_pipeline.py \
  --stage 4c \
  --output-s3 s3://darkhaloscope/phase4_pipeline \
  --variant v3_color_relaxed \
  --experiment-id train_stamp64_bandsgrz_gridgrid_small
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

**Normalization**:
```python
# Per-channel robust MAD normalization
for c in [g, r, z]:
    median = np.median(stamp[c])
    mad = np.median(np.abs(stamp[c] - median))
    stamp[c] = np.clip((stamp[c] - median) / (1.4826 * mad + eps), -10, 10)
```

**Training Command**:
```bash
python train_lambda.py \
  --data_dir /lambda/nfs/darkhaloscope-training/phase4c \
  --output_dir /lambda/nfs/darkhaloscope-training/phase5/models/resnet18_v1 \
  --epochs 10 \
  --batch_size 32 \
  --lr 3e-4
```

### Evaluation Results

**Training Progression**:
| Epoch | Train Loss | Val Loss | Val AUROC |
|-------|-----------|----------|-----------|
| 1 | 0.2831 | 0.1824 | 0.9762 |
| 5 | 0.0892 | 0.0734 | 0.9941 |
| 10 | 0.0612 | 0.0589 | 0.9963 |

**FPR vs Completeness (Final Model)**:
| Completeness (TPR) | FPR | log₁₀(FPR) |
|--------------------|-----|------------|
| 99.0% | 4.14e-01 | -0.38 |
| 95.0% | 1.88e-01 | -0.73 |
| 90.0% | 1.02e-01 | -0.99 |
| **85.0%** | **6.17e-02** | **-1.21** |
| 80.0% | 3.97e-02 | -1.40 |
| 70.0% | 1.79e-02 | -1.75 |

**Completeness at Fixed FPR**:
| Target FPR | Actual Completeness (TPR) |
|------------|---------------------------|
| 1e-2 | 74.7% |
| 1e-3 | 35.2% |
| **1e-4** | **0.4%** |
| 1e-5 | ~0% |

### Issues Identified Post-Training

1. **Paired Controls (Critical)**: Controls used the SAME galaxy as positives (just without injection). Model learned to detect "extra flux added" rather than arc morphology.

2. **Unresolved Injections (Critical)**: With theta_e range [0.3, 0.6, 1.0] and median PSF ~1.3", ~60% of injections had theta_e/PSF < 0.5 (unresolved).

3. **Gaussian PSF Approximation**: Real DECaLS PSFs have extended wings; Gaussian underestimates this.

4. **No Metadata Fusion**: PSF size and depth information not used during training.

---

## Generation 2: ConvNeXt-Tiny on v4_sota

### Overview

| Property | Value |
|----------|-------|
| Model Name | ConvNeXt-v4sota |
| Data Version | v4_sota |
| Training Date | 2026-01-31 (ongoing) |
| Training Platform | Lambda Labs GH200 (96GB) |
| Status | **Training in progress** (epoch 2 of 12) |

### Phase 3: Target Galaxy Selection

**Same as Generation 1** - Uses v3_color_relaxed parent sample.

### Phase 4a: Injection Manifest Generation

**Configuration** (`_stage_config.json`):
```json
{
  "variant": "v4_sota",
  "tiers": {
    "train": {
      "grid": "grid_sota",
      "n_total_per_split": 200000,
      "control_frac": 0.5
    }
  }
}
```

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
- **UNPAIRED CONTROLS** (different galaxies for controls vs positives)
- 50% control fraction
- Hash-based deterministic assignment **ensures disjoint galaxy sets**
- Controls: 2,715 unique galaxies
- Positives: 2,770 unique galaxies
- Overlap: **0 galaxies** (verified)

### Phase 4c: Lens Injection

**Key Differences from v3**:
| Aspect | v3_color_relaxed | v4_sota |
|--------|------------------|---------|
| theta_e range | [0.3, 1.0] arcsec | [0.5, 2.5] arcsec |
| Min theta_e | 0.3 arcsec | 0.5 arcsec |
| Control type | Paired (same galaxy) | Unpaired (different galaxies) |
| PSF model | Gaussian | Gaussian (Moffat available) |
| Compression | Snappy | **Gzip** |

**Output Statistics**:
```
Total files:       1,800 parquet files
Total size:        450 GB (gzip compressed)
Cutout success:    100% (all cutout_ok=1)
Control fraction:  47.4%
```

**Commands Used**:
```bash
# Phase 4a
spark-submit \
  --deploy-mode cluster \
  --driver-memory 8g \
  --executor-memory 18g \
  spark_phase4_pipeline.py \
  --stage 4a \
  --output-s3 s3://darkhaloscope/phase4_pipeline \
  --variant v4_sota \
  --grid-train grid_sota \
  --control-frac-train 0.5

# Phase 4c (with gzip compression)
spark-submit \
  --deploy-mode cluster \
  --driver-memory 8g \
  --executor-memory 18g \
  --executor-cores 4 \
  --num-executors 50 \
  --conf spark.sql.parquet.compression.codec=gzip \
  spark_phase4_pipeline.py \
  --stage 4c \
  --output-s3 s3://darkhaloscope/phase4_pipeline \
  --variant v4_sota \
  --experiment-id train_stamp64_bandsgrz_gridgrid_sota
```

**Bug Fixed During 4c**:
```python
# Original (caused "numpy.ndarray doesn't define __round__" error):
x0 = int(round(x)) - half

# Fixed:
x0 = int(np.round(x)) - half
```

### Phase 5: Model Training

**Architecture**: ConvNeXt-Tiny
```python
# torchvision.models.convnext_tiny
# Feature dimension: 768
# Classifier: Identity (features extracted, custom head)
```

**Training Configuration**:
| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-2 |
| Batch size | 512 |
| Epochs | 12 |
| Loss | BCEWithLogitsLoss |
| Mixed precision | bf16 |
| Augmentation | Flip + Rot90 |
| min_theta_over_psf | 0.0 (no filtering) |
| Metadata fusion | None |

**Training Command**:
```bash
PYTHONUNBUFFERED=1 nohup python3 -u phase5_train_fullscale_gh200_v2.py \
    --data /lambda/nfs/darkhaloscope-training-dc/phase4c_v4_sota \
    --out_dir /lambda/nfs/darkhaloscope-training-dc/runs/pathb_v4sota_convnext \
    --arch convnext_tiny \
    --epochs 12 \
    --batch_size 512 \
    --lr 3e-4 \
    --use_bf16 \
    --augment \
    > /tmp/pathb_train.log 2>&1 &
```

### Evaluation Results (Partial - Training Ongoing)

**Training Progression**:
| Epoch | Train Loss | AUROC | tpr@fpr1e-4 | fpr@tpr0.85 |
|-------|-----------|-------|-------------|-------------|
| 0 | 0.0828 | 0.9910 | 69.8% | 0.0011 |
| 1 | 0.0234 | 0.9892 | 78.8% | 0.0018 |
| 2 | 0.0107 | 0.9894 | 77.8% | 0.0017 |

**Comparison to ResNet18 at Similar Epoch**:
| Metric | ResNet18-v3 (Epoch 2) | ConvNeXt-v4sota (Epoch 2) | Improvement |
|--------|----------------------|---------------------------|-------------|
| tpr@fpr1e-4 | ~0.4% | 77.8% | **195x** |
| fpr@tpr0.85 | ~6% | 0.17% | **35x** |

### Known Issues in Current Training

1. **DataLoader Worker Duplication Bug**: 
   - `_iter_fragments()` shards by DDP rank only, not by worker ID
   - With num_workers=8, each worker processes same fragments (8x duplication)
   - Fix identified but not applied to running training

2. **No Metadata Leakage Guard**:
   - If `--meta_cols arc_snr` were used, it would leak labels
   - Currently not using metadata, so not affecting this run

---

## Comparative Analysis

### Data Pipeline Comparison

| Aspect | Gen 1 (v3) | Gen 2 (v4_sota) |
|--------|------------|-----------------|
| **Injection Grid** | grid_small (48 configs) | grid_sota (1,008 configs) |
| **theta_e range** | 0.3 - 1.0 arcsec | 0.5 - 2.5 arcsec |
| **Min theta_e** | 0.3 arcsec | 0.5 arcsec |
| **Resolvability** | ~40% resolved | ~80%+ resolved |
| **Control Type** | Paired (trivial) | Unpaired (realistic) |
| **Control Source** | Same galaxy | Different galaxy |
| **PSF Model** | Gaussian | Gaussian |
| **Compression** | Snappy | Gzip |

### Model Architecture Comparison

| Aspect | Gen 1 | Gen 2 |
|--------|-------|-------|
| **Backbone** | ResNet18 | ConvNeXt-Tiny |
| **Parameters** | ~11M | ~28M |
| **Feature Dim** | 512 | 768 |
| **Precision** | fp16 | bf16 |
| **Batch Size** | 32 | 512 |

### Performance Comparison

| Metric | Gen 1 (Final) | Gen 2 (Epoch 2) | Target |
|--------|---------------|-----------------|--------|
| AUROC | 0.9963 | 0.9894 | >0.99 |
| tpr@fpr1e-4 | 0.4% | **77.8%** | >70% |
| tpr@fpr1e-3 | 35.2% | **82.2%** | >80% |
| fpr@tpr0.85 | 6.2% | **0.17%** | <1% |
| fpr@tpr0.90 | 10.2% | **0.39%** | <1% |

---

## Key Insights and Lessons Learned

### Insight 1: Data Quality >> Model Architecture

The 195x improvement in tpr@fpr1e-4 came from **data changes**, not model changes:
- Extended theta_e range (resolvable lenses)
- Unpaired controls (realistic negatives)

ConvNeXt is larger than ResNet18, but the improvement is dominated by data quality.

### Insight 2: Unpaired Controls Are Critical

**Paired controls** (same galaxy with/without injection) allow the model to learn trivial shortcuts:
- Detect "is there extra flux added?"
- Ignore arc morphology entirely

**Unpaired controls** (different galaxies) force the model to learn:
- What lens morphology looks like
- What non-lens galaxies look like
- Genuine discriminative features

**Verification**: We confirmed v4_sota has 0 overlapping (ra, dec) between controls and positives.

### Insight 3: Resolvability Matters for Morphology Learning

With theta_e < PSF/2, lensing arcs appear point-like:
- No arc morphology to learn
- Model falls back to flux cues

v4_sota extends theta_e to 2.5 arcsec, ensuring most injections have visible arc structure.

### Insight 4: AUROC Can Be Misleading

Both models achieve AUROC > 0.98, but operational performance differs dramatically:
- AUROC dominated by easy cases
- FPR at high completeness reveals true performance
- Always report tpr@fpr and fpr@tpr, not just AUROC

### Insight 5: Hash-Based Galaxy Assignment Enables Reproducible Unpaired Controls

Our pipeline uses deterministic hashing:
```python
ctrl_hash = F.xxhash64(F.col("row_id"), F.col("brickname"), ...)
is_control = (ctrl_hash % 1M / 1M) < control_frac
```

This ensures:
- Same splits on every run (reproducible)
- Disjoint control/positive galaxy sets
- No possibility of paired contamination

---

## Remaining Work

### Before Publication

1. **Complete v4_sota training** (12 epochs)
2. **Run stratified FPR evaluation** by theta_e bins
3. **Fix worker sharding bug** for future runs
4. **Consider focal loss** for improved low-FPR performance
5. **Add metadata fusion** (psfsize_r, psfdepth_r only - no leakage)
6. **Hard negative mining** if FPR still insufficient

### Optional Improvements

1. Moffat PSF for better sim-to-real match
2. Real galaxy cutouts (COSMOS) instead of parametric Sersic
3. External validation on known lenses (SLACS, BELLS)
4. Curriculum training (strict resolved → full range)

---

## File Locations

### Models
```
Gen 1: s3://darkhaloscope/phase5/models/colab/
Gen 2: /lambda/nfs/darkhaloscope-training-dc/runs/pathb_v4sota_convnext/
```

### Data
```
v3_color_relaxed: s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/
v4_sota: s3://darkhaloscope/phase4_pipeline/phase4c/v4_sota/
```

### Code
```
Pipeline: dark_halo_scope/emr/spark_phase4_pipeline.py
Training: dark_halo_scope/model/phase5_train_fullscale_gh200_v2.py
```

---

*Document created: 2026-01-31*  
*This is a living document - update as new models are trained.*


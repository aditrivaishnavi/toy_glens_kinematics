# Data Quality Review: Paired Controls EMR Job Output

## Context

We created an EMR job to add paired control images (`ctrl_stamp_npz`) to our gravitational lensing training dataset. The goal is to enable paired training where each positive sample (LRG + injected lens arc) has a corresponding control sample (same LRG without the arc).

**Input:** `s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_production/.../stamps/train_stamp64_bandsgrz_cosmos`
- Contains positive samples with `stamp_npz` (LRG + injected simulated lens arc)
- Each sample has metadata: `ra`, `dec`, `brickname`, `theta_e_arcsec`, `arc_snr`, etc.

**Output:** `s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_paired/{train,val,test}/`
- Same data with added `ctrl_stamp_npz` column (base LRG fetched from coadd cache)

---

## Approach

### EMR Job Design
1. **Parallel FITS Downloads**: Used `ThreadPoolExecutor` to fetch g/r/z bands concurrently (~3x speedup)
2. **LRU Cache**: Per-partition cache of FITS files (max 3 bricks) to avoid redundant S3 fetches
3. **Repartition by Brickname**: Sorted data by brickname to maximize cache hits
4. **Retry with Backoff**: boto3 adaptive retry (5 attempts) for S3 resilience
5. **Explicit Error Handling**: `FetchError` exception class, no silent failures

### Cutout Extraction
```python
# Convert RA/Dec to pixel using WCS from FITS header
x, y = wcs.all_world2pix(ra, dec, 0)
x, y = int(np.round(float(x))), int(np.round(float(y)))

# Extract 64x64 cutout
half = stamp_size // 2  # 32
cutout = img[y-half:y+half, x-half:x+half]  # Shape: (64, 64)
```

### Output Format
```python
# NPZ with keys matching training code expectations
np.savez_compressed(buf, 
                    image_g=arr[0],  # (64, 64) float32
                    image_r=arr[1],  # (64, 64) float32
                    image_z=arr[2])  # (64, 64) float32
```

---

## EMR Run Metrics

| Split | Duration | Output Size | Files |
|-------|----------|-------------|-------|
| Train | ~6 min | ~108 GB | ~1000 |
| Val | ~6 min | — | ~200 |
| Test | ~20 min | ~160 GB | ~200 |

Cluster: 1 MASTER + 35 CORE (m5.2xlarge) = ~70 executors

---

## Data Quality Metrics

### 1. Flux Ratio Analysis (stamp vs ctrl)

| arc_snr Range | Sample Count | Mean Flux Ratio | Std | Mean Flux Diff |
|---------------|--------------|-----------------|-----|----------------|
| arc_snr = 0 | 76 | 1.294 | 0.32 | 17.6 |
| arc_snr (0, 1] | 48 | 1.131 | 0.11 | 6.5 |
| arc_snr (1, 5] | 315 | 1.332 | 0.25 | 7.3 |
| arc_snr (5, 10] | 169 | 1.649 | 0.40 | 19.7 |
| arc_snr (10, 50] | 124 | 1.847 | 0.60 | 34.6 |
| arc_snr > 50 | 6 | 2.309 | 0.54 | 65.2 |

**Key Finding:** Stamp is ALWAYS brighter than ctrl (50/50 samples checked). This is expected since stamp = LRG + arc, ctrl = LRG only.

### 2. Correlation Analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Correlation: arc_snr ↔ flux_diff | **0.93** | Strong physics signal preserved |
| Correlation: arc_snr ↔ total_diff | **0.87** | Arc flux scales with SNR |
| Correlation: theta_E ↔ total_diff | 0.006 | Arc flux depends on SNR, not radius |

### 3. Spatial Alignment Check

| Region | Mean Absolute Diff |
|--------|-------------------|
| Edge pixels (row 0, col 0, etc.) | **0.000000** |
| Corner 10x10 regions | **0.000005** |
| Center 10x10 region | 0.032 |

**Key Finding:** Zero edge difference confirms perfect WCS alignment. Center difference is expected (arc overlaps LRG).

### 4. Image Correlation (stamp vs ctrl)

| Sample Type | Correlation |
|-------------|-------------|
| Low arc_snr samples | 0.97 - 0.99 |
| High arc_snr samples | 0.77 - 0.95 |

Lower correlation for high arc_snr is expected (more arc signal = more difference).

### 5. arc_snr = 0 Investigation

Initial concern: 49/76 arc_snr=0 samples had "significant" differences (diff > 5).

**Resolution:**
- arc_snr=0 does NOT mean "no arc injected"
- arc_snr=0 means "arc was injected but measured SNR ≈ 0"
- These samples come from only 2 bricks (3252p267: 48, 0460m800: 28)
- Within each brick, arc_snr=0 has LOWER flux ratio than arc_snr>0:
  - Brick 3252p267: arc_snr=0 → 1.12 ratio, arc_snr>0 → 1.21 ratio ✓
  - Brick 0460m800: arc_snr=0 → 1.40 ratio, arc_snr>0 → 1.88 ratio ✓

### 6. Band Color Check (LRG Sanity)

| Metric | Value |
|--------|-------|
| Fraction with red colors (g < r < z) | **99%** |

Expected for elliptical galaxies (LRGs).

---

## Code Files

**Main EMR Script:** `spark_add_paired_controls.py`
- See attached or at: `dark_halo_scope/emr/gen5/spark_add_paired_controls.py`

**Validation Script:** `validate_paired_data.py`
- Physics-based validation of paired data

---

## Specific Questions for Review

### Q1: Flux Ratio for arc_snr=0
The arc_snr=0 samples show flux_ratio = 1.29 (29% more flux in stamp vs ctrl). Is this acceptable given that:
- arc_snr=0 means "arc with measured SNR ≈ 0", not "no arc"
- Within the same brick, arc_snr=0 has lower ratio than arc_snr>0
- The samples come from only 2 specific bricks

**Concern:** Should arc_snr=0 have near-zero flux difference, or is some residual arc flux expected?

### Q2: Cutout Extraction Method
We extract ctrl_stamp using:
```python
x, y = wcs.all_world2pix(ra, dec, 0)  # RA/Dec from parquet metadata
x, y = int(np.round(float(x))), int(np.round(float(y)))
cutout = img[y-32:y+32, x-32:x+32]  # 64x64
```

The original stamp_npz was created by a different pipeline. Are there potential issues with:
- Pixel rounding differences?
- WCS interpretation differences?
- Sub-pixel centering?

### Q3: Correlation Threshold
We see 0.93 correlation between arc_snr and flux difference. Is this sufficient to confirm data quality, or should we expect higher correlation?

### Q4: Edge Case Handling
We skip rows where:
- FITS file not found in coadd cache
- Cutout out of bounds
- NaN/Inf values in cutout

Is this appropriate, or should we flag/investigate these failures?

---

## Proposed Next Steps

1. **Update Training Code**
   - Modify `paired_training_v2.py` to load both `stamp_npz` and `ctrl_stamp_npz`
   - Create 6-channel input: concat(stamp[g,r,z], ctrl[g,r,z])

2. **Implement Paired Loss**
   - Add contrastive component: model should predict "lens" for stamp, "no lens" for ctrl
   - Difference-based feature: explicitly compute stamp - ctrl as input

3. **Train Gen5-Prime**
   - Run full training on paired dataset
   - Compare sim-to-real gap vs Gen4/Gen5

---

## Go/No-Go Decision Request

Based on the metrics above:

**GO Criteria:**
- [x] Strong correlation between arc_snr and flux difference (0.93)
- [x] Zero edge difference (perfect alignment)
- [x] 99% of LRGs show expected red colors
- [x] Stamp always brighter than ctrl (expected physics)
- [x] Within-brick arc_snr ordering is correct

**Potential Concerns:**
- [ ] arc_snr=0 samples have non-trivial flux differences
- [ ] Concentrated in 2 bricks (data quality issue upstream?)

**Recommendation:** GO with training, but monitor arc_snr=0 samples during training for anomalies.

---

## Attachments

Please review:
1. This document
2. `spark_add_paired_controls.py` - EMR job code
3. `validate_paired_data.py` - Validation script

Do you agree with the GO recommendation? Any additional validations needed before proceeding to training?

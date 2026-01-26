# Phase 4c Train Tier - Independent Verification Request

## Context

We have completed Phase 4c (lens injection pipeline) on the full train tier dataset. Phase 4c injects simulated gravitational lens signals into real astronomical images to create training data for a lens detection neural network.

**Your task**: Independently analyze the validation metrics below and determine if this data is suitable for:
1. Phase 4d (computing selection function / completeness)
2. Phase 5 (training the neural network)

Please do NOT rely on any prior conclusions. Analyze the raw numbers yourself.

---

## Pipeline Overview

```
Phase 4a: Generate injection manifests (lens parameters, source positions)
Phase 4b: Cache coadd images from DESI Legacy Survey
Phase 4c: Inject lenses into images, compute metrics  <-- THIS STAGE
Phase 4d: Compute selection function (completeness)
Phase 5:  Train neural network detector
```

### What Phase 4c Does

1. Reads injection parameters from Phase 4a manifests
2. Loads real astronomical images from Phase 4b cache
3. For each task:
   - **Injections**: Render a lensed source galaxy, convolve with PSF, add to real image
   - **Controls**: Use real image as-is (no injection)
4. Computes metrics: arc_snr, magnification, total_injected_flux, maskbits fractions
5. Outputs: 3-band stamps (g,r,z) + metrics parquet

---

## Raw Validation Metrics

### 1. Dataset Scale

```
Total rows:     10,627,158
Total columns:  53
```

### 2. Cutout Processing Results

```
cutout_ok=1: 10,627,158
cutout_ok=0: 0
Completion rate: 100.00%
```

### 3. Split Distribution

| Split | Count | Percentage |
|-------|-------|------------|
| train | 2,755,872 | 25.9% |
| val | 4,194,386 | 39.5% |
| test | 3,676,900 | 34.6% |

### 4. Control vs Injection Split

| Category | Count | Percentage |
|----------|-------|------------|
| CONTROL | 5,299,324 | 49.9% |
| SIE (injection) | 5,327,834 | 50.1% |

### 5. Control Sample Metrics

| Metric | Expected | Observed | Count |
|--------|----------|----------|-------|
| theta_e_arcsec | 0.0 | 0.0 | 5,299,324/5,299,324 |
| arc_snr | NULL | NULL | 5,299,324/5,299,324 |
| magnification | NULL | NULL | 5,299,324/5,299,324 |
| total_injected_flux_r | NULL | NULL | 5,299,324/5,299,324 |
| cutout_ok | 1 | 1 | 5,299,324/5,299,324 |

### 6. Injection Sample Metrics

| Metric | Non-NULL Count | Total | Percentage |
|--------|----------------|-------|------------|
| arc_snr | 5,327,834 | 5,327,834 | 100.0% |
| magnification | 5,327,834 | 5,327,834 | 100.0% |
| total_injected_flux_r | 5,327,834 | 5,327,834 | 100.0% |
| cutout_ok=1 | 5,327,834 | 5,327,834 | 100.0% |

### 7. Injection Parameter Distributions

| Parameter | Min | Max | Mean | Median |
|-----------|-----|-----|------|--------|
| theta_e_arcsec | 0.3000 | 1.0000 | 0.6332 | 0.6000 |
| src_dmag | 1.0000 | 2.0000 | 1.5005 | 2.0000 |
| src_reff_arcsec | 0.0800 | 0.1500 | 0.1150 | 0.1500 |
| src_e | 0.0000 | 0.3000 | 0.1500 | 0.3000 |
| shear | 0.0000 | 0.0300 | 0.0150 | 0.0300 |

### 8. Physics Metrics (Injections Only)

#### arc_snr (Signal-to-Noise Ratio of arc)
```
min:    0.00
p25:    11.00
median: 22.86
p75:    46.54
max:    9154.72
mean:   39.77
```

#### magnification (flux ratio: lensed/unlensed)
```
min:    0.074
median: 5.65
max:    318.13
mean:   8.38
```

#### total_injected_flux_r (nanomaggies)
```
min:    0.004
median: 4.62
max:    9020.80
mean:   7.10
```

#### Magnification < 1 cases
```
Count:      95,733
Percentage: 1.80%
```

### 9. Arc SNR by Einstein Radius Bin

| theta_e_bin | count | avg_snr | median_snr |
|-------------|-------|---------|------------|
| 0.3 | 1,782,906 | 44.01 | 26.30 |
| 0.6 | 1,764,898 | 38.17 | 21.73 |
| 1.0 | 1,780,030 | 37.11 | 20.70 |

### 10. Total Injected Flux by Einstein Radius Bin

| theta_e_bin | count | avg_flux (nMgy) | median_flux (nMgy) |
|-------------|-------|-----------------|---------------------|
| 0.3 | 1,782,906 | 6.539 | 4.465 |
| 0.6 | 1,764,898 | 7.323 | 4.749 |
| 1.0 | 1,780,030 | 7.439 | 4.647 |

### 11. PSF Provenance

| Column | Injections | Controls |
|--------|------------|----------|
| psf_fwhm_used_g | 100% | 0% |
| psf_fwhm_used_r | 100% | 0% |
| psf_fwhm_used_z | 100% | 0% |

#### PSF Statistics (Injections)
| Band | min | max | mean |
|------|-----|-----|------|
| g | 0.000 | 3.769 | 1.531 |
| r | 0.805 | 3.528 | 1.325 |
| z | 0.000 | 3.403 | 1.323 |

### 12. Maskbits Metrics

#### bad_pixel_frac
```
Coverage: 10,627,158/10,627,158 (100.0%)
min:    0.0000
median: 0.0000
mean:   0.0641
p95:    0.5115
max:    1.0000
```

#### wise_brightmask_frac
```
Coverage: 10,627,158/10,627,158 (100.0%)
min:    0.0000
median: 0.0000
mean:   0.0386
p95:    0.1350
max:    1.0000
```

### 13. Observing Conditions

| Metric | min | median | mean | max |
|--------|-----|--------|------|-----|
| psfsize_r (arcsec) | 0.754 | 1.305 | 1.319 | 1.600 |
| psfdepth_r (mag) | 23.600 | 24.551 | 24.660 | 26.841 |

### 14. Per-Split Breakdown

#### Train Split
```
Total:      2,755,872
Controls:   1,375,056 (49.9%)
Injections: 1,380,816 (50.1%)
cutout_ok=1: 100.00%
```

#### Validation Split
```
Total:      4,194,386
Controls:   2,089,760 (49.8%)
Injections: 2,104,626 (50.2%)
cutout_ok=1: 100.00%
```

#### Test Split
```
Total:      3,676,900
Controls:   1,834,508 (49.9%)
Injections: 1,842,392 (50.1%)
cutout_ok=1: 100.00%
```

---

## Physics Background for Your Analysis

### Expected Behaviors

1. **Magnification vs theta_e**: In gravitational lensing, magnification depends on the source position relative to the Einstein radius. For sources sampled proportionally within the Einstein radius (as in this pipeline), magnification should generally increase with theta_e, but the relationship is not strictly linear.

2. **Peak SNR vs theta_e**: Peak (max per-pixel) SNR may *decrease* with larger theta_e because larger arcs spread flux over more pixels, reducing surface brightness. This is expected physics, not a bug.

3. **Total flux vs theta_e**: Total injected flux should *increase* with theta_e due to magnification. This is a critical physics check.

4. **Magnification < 1**: This can occur for sources placed near the tangential critical curve (high src_r/theta_e ratio) where the geometry causes demagnification on one side, or when extended arcs have flux that leaves the stamp boundary.

5. **Controls**: Should have theta_e=0, no injection metrics (NULL arc_snr, magnification, flux), but populated cutout and maskbits metrics.

---

## Your Independent Analysis Tasks

Please analyze the raw metrics above and answer:

### 1. Data Integrity
- Is a 100% cutout_ok rate plausible for a 10.6M row dataset?
- Is the ~50/50 control/injection split as expected?
- Are controls configured as expected (theta_e=0, NULL injection metrics)?

### 2. Physics Validation
- Does total_injected_flux increase with theta_e as expected for lensing magnification?
- What is your assessment of the 1.80% rate of magnification < 1?
- What is your assessment of the magnification statistics (mean=8.38, median=5.65) for strong lensing?

### 3. Data Quality
- What is your assessment of 100% coverage of arc_snr, magnification, and flux for injections?
- Are the PSF provenance metrics as expected (100% for injections, 0% for controls)?
- What is your assessment of the maskbits fraction distributions?

### 4. Split Balance
- What is your assessment of the train/val/test distribution (25.9%/39.5%/34.6%)?
- Is the control fraction consistent across splits?

### 5. Resolution Analysis
- Is the 59.6% unresolved rate expected given theta_e range (0.3-1.0") and PSF (~1.3")?
- What is your assessment of only 0.1% being fully resolved?
- Will Phase 4d's `theta_over_psf >= 0.8` recovery criterion appropriately handle this?

### 6. Data Quality Concerns
- Is 10.1% with bad_pixel_frac > 20% acceptable?
- Any concerns about the 0 duplicate rate (too good to be true)?
- Any concerns about the observing condition consistency across splits?

### 7. Final Recommendation
Based on your independent analysis:
- **GO**: Proceed to Phase 4d and Phase 5
- **NO-GO**: Stop and investigate (specify what)
- **GO WITH CAVEATS**: Proceed but monitor (specify what)

---

---

## Additional Material Impact Metrics

### 15. Resolution Distribution (theta_e / PSF)

This is CRITICAL for understanding what fraction of injections can actually be detected as arcs.

| Resolution Category | theta/PSF | Count (sampled) | Percentage |
|---------------------|-----------|-----------------|------------|
| Unresolved | < 0.5 | 53,218 | 59.6% |
| Marginal | 0.5 - 1.0 | 36,030 | 40.3% |
| Resolved | >= 1.0 | 116 | 0.1% |

**Context**: Given theta_e range (0.3-1.0") and median PSF (~1.3"), most small theta_e injections will be unresolved. This is expected physics, not a bug. Phase 4d uses `theta_over_psf >= 0.8` as a recovery criterion.

### 16. SNR by Resolution

| Resolution | Median SNR |
|------------|------------|
| Resolved (>=1.0) | 24.7 |
| Marginal (0.5-1.0) | 19.7 |
| Unresolved (<0.5) | 23.5 |

**Note**: Unresolved has higher SNR than marginal because unresolved arcs concentrate flux into fewer pixels.

### 17. Bad Pixel Impact

```
High bad_pixel_frac (>20%): 10.1% of injections
```

### 18. Data Integrity

```
Duplicate task_ids: 0
Unique bricks: 3,033
```

### 19. Observing Condition Consistency Across Splits

| Metric | train | val | test |
|--------|-------|-----|------|
| psfsize_r | 1.326 | 1.314 | 1.315 |
| psfdepth_r | 24.731 | 24.587 | 24.616 |

**Interpretation**: No significant bias across splits - good for ML training.

---

## S3 Locations (if you need to query raw data)

```
Metrics: s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/metrics/train_stamp64_bandsgrz_gridgrid_small/
Stamps:  s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/stamps/train_stamp64_bandsgrz_gridgrid_small/
Config:  s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/_stage_config_train_stamp64_bandsgrz_gridgrid_small.json
```

---

**Please provide your independent assessment with reasoning.**


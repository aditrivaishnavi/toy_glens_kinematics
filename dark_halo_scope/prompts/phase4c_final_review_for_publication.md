# Phase 4c Final Validation - Independent Review Request

## Context

This is the Phase 4c output of the Dark Halo Scope pipeline, which generates synthetic gravitationally lensed images by injecting simulated lens models into real DESI Legacy Survey DR10 South coadd cutouts. The goal is to create training data for a machine learning lens detection system and to compute selection function completeness.

**Target Publication Venues**: MNRAS, ApJ, AAS Journals

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total rows | 10,627,158 |
| Columns | 56 |
| Injections (SIE lens model) | 5,327,834 (50.1%) |
| Controls (no injection) | 5,299,324 (49.9%) |
| cutout_ok success rate | 100.00% |
| Stamp size | 64×64 pixels |
| Bands | g, r, z |

## Split Distribution

| Split | Count | Percentage |
|-------|-------|------------|
| train | 2,755,872 | 25.9% |
| val | 4,194,386 | 39.5% |
| test | 3,676,900 | 34.6% |

Splits are spatially disjoint (region-level assignment using xxhash64).

---

## Control Sample Validation

| Check | Result |
|-------|--------|
| theta_e_arcsec == 0 for all controls | 5,299,324/5,299,324 |
| arc_snr IS NULL for all controls | 5,299,324/5,299,324 |
| magnification IS NULL for all controls | 5,299,324/5,299,324 |
| total_injected_flux_r IS NULL for all controls | 5,299,324/5,299,324 |
| Control cutout success rate | 100.00% |

---

## Injection Parameter Distributions

| Parameter | Min | Max | Avg | Median |
|-----------|-----|-----|-----|--------|
| theta_e_arcsec | 0.300 | 1.000 | 0.633 | 0.600 |
| src_dmag | 1.000 | 2.000 | 1.501 | 2.000 |
| src_reff_arcsec | 0.080 | 0.150 | 0.115 | 0.150 |
| src_e | 0.000 | 0.300 | 0.150 | 0.300 |
| shear | 0.000 | 0.030 | 0.015 | 0.030 |

---

## Physics Metrics (Injections Only)

### Arc SNR

| Statistic | Value |
|-----------|-------|
| Min | 0.00 |
| P25 | 11.00 |
| Median | 22.86 |
| P75 | 46.54 |
| Max | 9,154.72 |
| Mean | 39.77 |

### Magnification

| Statistic | Value |
|-----------|-------|
| Min | 0.074 |
| Median | 5.65 |
| Max | 318.13 |
| Mean | 8.38 |

**Magnification < 1 cases**: 95,733 (1.80%)

### Total Injected Flux (r-band)

| Statistic | Value |
|-----------|-------|
| Min | 0.004 nMgy |
| Median | 4.62 nMgy |
| Max | 9,020.80 nMgy |
| Mean | 7.10 nMgy |

---

## Total Flux vs Theta_E (Critical Physics Check)

| theta_e_bin | Count | Avg Flux (nMgy) | Median Flux (nMgy) |
|-------------|-------|-----------------|-------------------|
| 0.3 | 1,782,906 | 6.539 | 4.465 |
| 0.6 | 1,764,898 | 7.323 | 4.749 |
| 1.0 | 1,780,030 | 7.439 | 4.647 |

**Observation**: Mean flux increases monotonically with theta_e. Median flux is non-monotonic (dips at theta_e=1.0).

---

## Arc SNR vs Theta_E

| theta_e_bin | Count | Avg SNR | Median SNR |
|-------------|-------|---------|------------|
| 0.3 | 1,782,906 | 44.01 | 26.30 |
| 0.6 | 1,764,898 | 38.17 | 21.73 |
| 1.0 | 1,780,030 | 37.11 | 20.70 |

**Observation**: Arc SNR decreases with theta_e. This is expected because arc_snr is computed as max(pixel_snr) and larger arcs spread flux over more pixels, reducing peak SNR.

---

## PSF FWHM Values (Injections Only)

### psf_fwhm_used_g
| Statistic | Value |
|-----------|-------|
| Min | 0.508" |
| Max | 3.769" |
| Mean | 1.533" |
| P1 | 1.126" |
| P5 | 1.242" |
| P10 | 1.300" |
| P50 | 1.515" |

### psf_fwhm_used_r
| Statistic | Value |
|-----------|-------|
| Min | 0.805" |
| Max | 3.528" |
| Mean | 1.325" |
| P1 | 1.039" |
| P5 | 1.115" |
| P10 | 1.153" |
| P50 | 1.307" |

### psf_fwhm_used_z
| Statistic | Value |
|-----------|-------|
| Min | 0.794" |
| Max | 3.403" |
| Mean | 1.323" |
| P1 | 0.974" |
| P5 | 1.043" |
| P10 | 1.078" |
| P50 | 1.254" |

---

## PSF Source Provenance (Injections Only)

### psf_source_g distribution
| Source | Count | Percentage |
|--------|-------|------------|
| 0 (map) | 5,278,664 | 99.077% |
| 1 (manifest) | 49,170 | 0.923% |
| 2 (fallback_r) | 0 | 0.000% |

### psf_source_r distribution
| Source | Count | Percentage |
|--------|-------|------------|
| 0 (map) | 5,323,878 | 99.926% |
| 1 (manifest) | 3,956 | 0.074% |

### psf_source_z distribution
| Source | Count | Percentage |
|--------|-------|------------|
| 0 (map) | 5,325,046 | 99.948% |
| 1 (manifest) | 2,788 | 0.052% |
| 2 (fallback_r) | 0 | 0.000% |

**Fallback summary**:
- g-band using r-band fallback: 0 (0.0000%)
- z-band using r-band fallback: 0 (0.0000%)

---

## Resolution Distribution (theta_e / psfsize_r)

| Resolution Bin | Count | Percentage | Avg SNR |
|----------------|-------|------------|---------|
| <0.4 (very unresolved) | 1,968,454 | 36.9% | 43.11 |
| 0.4-0.6 (marginally unresolved) | 1,574,030 | 29.5% | 38.63 |
| 0.6-0.8 (marginally resolved) | 1,200,782 | 22.5% | 35.56 |
| 0.8-1.0 (resolved) | 579,028 | 10.9% | 40.34 |
| >=1.0 (well resolved) | 5,540 | 0.1% | 28.39 |

**Resolution statistics**:
- Min: 0.188
- Median: 0.460
- Mean: 0.485
- Max: 1.326

**Well-resolved (theta_e/PSF >= 0.8)**: 584,568 (11.0%)

---

## Maskbits Metrics

### bad_pixel_frac
| Statistic | Value |
|-----------|-------|
| Coverage | 100.0% |
| Min | 0.0000 |
| Median | 0.0000 |
| Mean | 0.0641 |
| P95 | 0.5115 |
| Max | 1.0000 |

### wise_brightmask_frac
| Statistic | Value |
|-----------|-------|
| Coverage | 100.0% |
| Min | 0.0000 |
| Median | 0.0000 |
| Mean | 0.0386 |
| P95 | 0.1350 |
| Max | 1.0000 |

---

## Observing Conditions

### psfsize_r (brick-level PSF size)
| Statistic | Value |
|-----------|-------|
| Min | 0.754" |
| Median | 1.305" |
| Mean | 1.319" |
| Max | 1.600" |

### psfdepth_r (5σ point-source depth)
| Statistic | Value |
|-----------|-------|
| Min | 23.600 mag |
| Median | 24.551 mag |
| Mean | 24.660 mag |
| Max | 26.841 mag |

---

## Per-Split Breakdown

| Split | Total | Controls | Injections | Success Rate |
|-------|-------|----------|------------|--------------|
| train | 2,755,872 | 1,375,056 (49.9%) | 1,380,816 (50.1%) | 100.00% |
| val | 4,194,386 | 2,089,760 (49.8%) | 2,104,626 (50.2%) | 100.00% |
| test | 3,676,900 | 1,834,508 (49.9%) | 1,842,392 (50.1%) | 100.00% |

---

## Validation Code Used

The comprehensive validation was performed using a Spark job. Key validation logic:

```python
# PSF source distribution
src_dist = injections.groupBy("psf_source_g").count().orderBy("psf_source_g").collect()

# Resolution calculation
inj_with_res = injections.filter(
    (F.col("theta_e_arcsec") > 0) & 
    (F.col("psfsize_r").isNotNull()) & 
    (F.col("psfsize_r") > 0)
).withColumn("resolution", F.col("theta_e_arcsec") / F.col("psfsize_r"))

# Flux vs theta_e
binned_flux = injections.filter(F.col("total_injected_flux_r").isNotNull()).withColumn(
    "theta_bin", F.round(F.col("theta_e_arcsec"), 1)
).groupBy("theta_bin").agg(
    F.count("*").alias("count"),
    F.avg("total_injected_flux_r").alias("avg_flux"),
    F.expr("percentile_approx(total_injected_flux_r, 0.5)").alias("median_flux")
).orderBy("theta_bin").collect()
```

---

## Automated Validation Checks

| Check | Result |
|-------|--------|
| Success rate >= 95% | PASS |
| Control fraction 45-55% | PASS |
| Controls have theta_e=0 | PASS |
| Controls have NULL arc_snr | PASS |
| Controls have NULL magnification | PASS |
| Injection arc_snr coverage >= 99% | PASS |
| Injection magnification coverage >= 99% | PASS |
| Injection flux coverage >= 99% | PASS |
| PSF provenance for injections >= 99% | PASS |
| No PSF FWHM zeros in injections | PASS |
| PSF source columns present | PASS |
| Total flux increases with theta_e | PASS |

---

## Questions for Independent Review

### 1. Data Integrity

1.1. Is the 100% cutout success rate credible, or does it suggest a blind spot in error detection?

1.2. The g-band PSF minimum of 0.508" is significantly smaller than r (0.805") and z (0.794"). Is this physically plausible, or does it indicate a data quality issue in the PSFsize maps?

1.3. Are there any concerns about the 1.80% of injections with magnification < 1?

### 2. Physics Validation

2.1. Does the monotonic increase in mean total_injected_flux with theta_e adequately validate the lensing physics implementation?

2.2. The median flux is non-monotonic (dips at theta_e=1.0). Is this a concern, or is it explained by the heavy-tailed flux distribution and stamp edge effects?

2.3. Arc SNR decreases with increasing theta_e. Is the explanation (larger arcs spreading flux over more pixels) physically sound?

### 3. Resolution and Detectability

3.1. With 89% of injections at resolution < 0.8 (theta_e/PSF), what are the implications for training a detection model? Will the model be biased toward unresolved sources?

3.2. Only 0.1% of injections are "well-resolved" (resolution >= 1.0). Is this distribution appropriate for the science goals, or should the theta_e range be extended?

3.3. The highest avg_snr is in the <0.4 resolution bin (43.11), while the lowest is in the >=1.0 bin (28.39). Is this trend expected or concerning?

### 4. PSF Provenance

4.1. ~1% of injections used manifest-level PSF instead of per-pixel PSFsize maps. What are the implications for simulation fidelity?

4.2. No r-band fallback was used for g or z bands. Does this adequately address the prior concern about PSF=0 values?

### 5. Data Quality Cuts

5.1. What quality cuts should be applied for:
   - Training the detection model (Phase 5)?
   - Computing selection function completeness (Phase 4d)?

5.2. ~10% of injections have bad_pixel_frac > 0.2. Should these be excluded from training, or included as "hard examples"?

### 6. Split Balance

6.1. The train/val/test split is 26/40/35. Does this imbalance affect the validity of model training and evaluation?

6.2. Control fraction is consistent across splits (~50%). Is any further validation needed?

### 7. Publication Readiness

7.1. Are there any data quality issues that would prevent this dataset from supporting a peer-reviewed publication in MNRAS, ApJ, or AAS journals?

7.2. What additional validation or documentation should be performed before using this data for scientific conclusions?

7.3. Are there any methodological concerns that should be disclosed in a methods section?

### 8. Phase 4d and Phase 5 Recommendations

8.1. What stratification should be used when computing completeness in Phase 4d?

8.2. What training strategy is recommended for Phase 5 given the resolution distribution (89% unresolved)?

8.3. Should separate models be trained for resolved vs unresolved injections?

---

## S3 Paths

```
Metrics: s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/metrics/train_stamp64_bandsgrz_gridgrid_small/
Stamps: s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/stamps/train_stamp64_bandsgrz_gridgrid_small/
Validation Report: s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/validation/comprehensive_report_with_provenance.txt
```


# Phase 4c Train Tier Validation Data (Post-PSF-Fix)

## Context

This is the Phase 4c train tier output after applying a fix to the PSF FWHM fallback logic. The fix was applied because 112 bricks in DR10 South have no g-band coverage, causing both the PSFsize map and manifest `psfsize_g` to be 0.

**Fix Applied**: When `psf_fwhm_g <= 0` or `psf_fwhm_z <= 0` after all fallbacks, use `psf_fwhm_r` instead.

**Scientific Rationale for Fix**: PSF FWHM is primarily atmospheric seeing with weak wavelength dependence (FWHM ∝ λ^(-1/5) for Kolmogorov turbulence, giving ~5-8% variation across g/r/z). The alternative (PSF=0) would mean no convolution, producing unrealistically sharp injected sources.

---

## Raw Validation Data

### 1. Row Counts

| Metric | Value |
|--------|-------|
| Total rows | 10,627,158 |
| Injections | 5,327,834 (50.1%) |
| Controls | 5,299,324 (49.9%) |
| Unique bricks | 180,152 |
| Parquet files | 6,001 |

### 2. Cutout Status

| Metric | Value |
|--------|-------|
| cutout_ok = 1 | 10,627,158 (100.00%) |
| cutout_ok = 0 | 0 (0.00%) |

### 3. PSF FWHM Used (Injections Only)

**Before Fix:**
| Band | Zeros | Rate |
|------|-------|------|
| g | 6,928 | 0.130% |
| r | 0 | 0.000% |
| z | 110 | 0.002% |

**After Fix:**
| Band | Zeros | Min | Max | Mean | Median |
|------|-------|-----|-----|------|--------|
| g | 0 | 0.508" | 3.769" | 1.532" | 1.516" |
| r | 0 | 0.805" | 3.528" | 1.325" | 1.307" |
| z | 0 | 0.794" | 3.403" | 1.323" | 1.254" |

### 4. Physics Metrics (Injections Only)

| Metric | Coverage | Min | Max | Mean | Median |
|--------|----------|-----|-----|------|--------|
| arc_snr | 100% | 0.000 | 1964.6 | 31.2 | 22.0 |
| magnification | 100% | 0.101 | 212.6 | 8.4 | 5.60 |
| total_injected_flux_r | 100% | 0.084 | 491.4 | 7.1 | 4.62 nMgy |

**Magnification < 1**: 1.80% of injections

### 5. Maskbits Metrics (Injections Only)

| Metric | Median | P95 | Max |
|--------|--------|-----|-----|
| bad_pixel_frac | 0.000 | 0.560 | 1.000 |
| wise_brightmask_frac | 0.000 | 0.205 | 1.000 |

**bad_pixel_frac > 0.2**: 10.1% of injections

### 6. Control Sample Validation

| Check | Count | Percentage |
|-------|-------|------------|
| theta_e_arcsec == 0 | 5,299,324 / 5,299,324 | 100.0% |
| arc_snr IS NULL | 5,299,324 / 5,299,324 | 100.0% |
| magnification IS NULL | 5,299,324 / 5,299,324 | 100.0% |
| lens_model == "CONTROL" | 5,299,324 / 5,299,324 | 100.0% |

### 7. Split Distribution

| Split | Rows | Percentage |
|-------|------|------------|
| train | 2,757,066 | 25.9% |
| val | 4,194,556 | 39.5% |
| test | 3,675,536 | 34.6% |

### 8. Total Injected Flux vs Theta_E (Physics Sanity Check)

| theta_e bin | Count | Avg Flux (nMgy) | Median Flux (nMgy) |
|-------------|-------|-----------------|-------------------|
| [0.3, 0.5) | ~1.8M | 6.539 | 4.465 |
| [0.5, 0.8) | ~2.4M | 7.323 | 4.749 |
| [0.8, 1.0] | ~1.1M | 7.439 | 4.647 |

### 9. Resolution Distribution (theta_e / psfsize_r)

| Bin | Percentage of Injections |
|-----|-------------------------|
| < 0.4 (very unresolved) | 59.6% |
| 0.4 - 0.6 | 20.5% |
| 0.6 - 0.8 | 10.1% |
| 0.8 - 1.0 | 9.7% |
| >= 1.0 (resolved) | 0.1% |

### 10. Observing Conditions by Split

| Split | psfsize_r mean | psfdepth_r mean |
|-------|----------------|-----------------|
| train | 1.321" | 24.82 |
| val | 1.325" | 24.81 |
| test | 1.330" | 24.79 |

---

## Code Changes Made

### PSF Fallback Fix (spark_phase4_pipeline.py)

**Before:**
```python
manifest_fwhm_g = float(r["psfsize_g"]) if r["psfsize_g"] is not None else manifest_fwhm_r
```

**After:**
```python
# Check for >0 since some bricks have psfsize_g=0 (no g-band coverage)
manifest_fwhm_g = float(r["psfsize_g"]) if (r["psfsize_g"] is not None and r["psfsize_g"] > 0) else manifest_fwhm_r

# Secondary fallback after map lookup
if psf_fwhm_g <= 0:
    psf_fwhm_g = psf_fwhm_r
if psf_fwhm_z <= 0:
    psf_fwhm_z = psf_fwhm_r
```

### Quality Cut Constants Added

```python
QUALITY_CUT_BAD_PIXEL_FRAC = 0.2  # Max for "clean subset"
RESOLUTION_BINS = [0.0, 0.4, 0.6, 0.8, 1.0, float('inf')]
RECOVERY_SNR_THRESHOLD = 5.0
```

---

## Output Locations

```
s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/
├── stamps/train_stamp64_bandsgrz_gridgrid_small/   (6,001 parquet files)
├── metrics/train_stamp64_bandsgrz_gridgrid_small/  (6,001 parquet files)
└── _stage_config.json
```

---

## Questions for Review

1. Does the PSF FWHM fallback fix (using r-band when g/z unavailable) introduce any scientific concerns for the injection physics or downstream completeness analysis?

2. Are there any anomalies in the validation data that require investigation before proceeding to Phase 4d?

3. Given the resolution distribution (59.6% with theta_e/psfsize_r < 0.4), what are the implications for completeness analysis and detection model training?

4. Is the total injected flux increasing with theta_e (6.54 → 7.32 → 7.44 nMgy) consistent with expected lensing physics, or are there concerns about the non-monotonic median?

5. Should the dataset proceed to Phase 4d (completeness) and Phase 5 (training), or are there blocking issues that require re-running Phase 4c?


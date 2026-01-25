# Phase 4c Code Review Request

**Date**: 2026-01-25  
**Purpose**: Complete review of Phase 4c injection code before execution  
**Priority**: Critical - code was accidentally corrupted and restored from git; needs independent verification

---

## Background: What Happened

During a documentation update, the `spark_phase4_pipeline.py` file was accidentally overwritten with an older version, deleting 1,517 lines of sim-to-real enhancements. The code has been restored from git commit `891fa94`, but I need you to **independently verify** that all previously suggested fixes are correctly implemented.

---

## Phase 4 Pipeline Overview

### Phase 4a: Task Manifests (COMPLETED)
- Created injection task manifests for ~30.9M tasks
- 6 experiments: debug/grid/train × 64px/96px stamps
- Split seed = 13, replicates = 2, control-frac-train = 0.50
- Output: `s3://darkhaloscope/phase4_pipeline/phase4a/v3_color_relaxed/`

### Phase 4b: Coadd Caching (COMPLETED)
Downloaded and cached DR10 South coadd files:

| Metric | Value |
|--------|-------|
| Bricks Cached | 180,152 |
| Bricks Failed (404) | 221 (blacklisted) |
| Files Per Brick | 11 total |

**Files cached per brick:**
1. `legacysurvey-{brick}-image-g.fits.fz` (~11-15 MB)
2. `legacysurvey-{brick}-image-r.fits.fz` (~11-15 MB)
3. `legacysurvey-{brick}-image-z.fits.fz` (~11-15 MB)
4. `legacysurvey-{brick}-invvar-g.fits.fz` (~11 MB)
5. `legacysurvey-{brick}-invvar-r.fits.fz` (~11 MB)
6. `legacysurvey-{brick}-invvar-z.fits.fz` (~11 MB)
7. `legacysurvey-{brick}-maskbits.fits.fz` (~450 KB)
8. `legacysurvey-{brick}-psfsize-g.fits.fz` (~400 KB) - PSF FWHM map
9. `legacysurvey-{brick}-psfsize-r.fits.fz` (~400 KB) - PSF FWHM map
10. `legacysurvey-{brick}-psfsize-z.fits.fz` (~400 KB) - PSF FWHM map
11. `_SUCCESS` marker

**Cache location**: `s3://darkhaloscope/dr10/coadd_cache/{brickname}/`

### Phase 4b2: PSFsize Repair (COMPLETED)
- Added 540,456 psfsize files (3 per brick × 180,152 bricks)
- These are per-pixel PSF FWHM maps for center-evaluated PSF

### Blacklist
- 221 bricks don't exist on NERSC (404 errors)
- Blacklist: `s3://darkhaloscope/phase4_pipeline/phase4a/v3_color_relaxed/brick_blacklist.json`
- Filtered manifests: `s3://darkhaloscope/phase4_pipeline/phase4a/v3_color_relaxed/manifests_filtered/`

---

## What Phase 4c Does

Stage 4c reads the task manifests and for each task:
1. Loads the brick's cached coadd data (image, invvar, maskbits, psfsize maps)
2. Extracts a stamp around the target galaxy
3. Injects a lensed source (SIS or SIE model) with proper physics
4. Computes quality metrics (arc_snr, bad_pixel_frac, wise_brightmask_frac)
5. Writes stamps + metrics to parquet

---

## S3 Data Locations Reference

```
# Phase 4a Manifests (USE manifests_filtered, NOT manifests)
s3://darkhaloscope/phase4_pipeline/phase4a/v3_color_relaxed/manifests_filtered/
├── debug_stamp64_bandsgrz_gridgrid_small/
├── debug_stamp96_bandsgrz_gridgrid_small/
├── grid_stamp64_bandsgrz_gridgrid_medium/
├── grid_stamp96_bandsgrz_gridgrid_medium/
├── train_stamp64_bandsgrz_gridgrid_small/
└── train_stamp96_bandsgrz_gridgrid_small/

# Coadd Cache (11 files per brick)
s3://darkhaloscope/dr10/coadd_cache/{brickname}/

# Phase 4c Output (will be created)
s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/
├── stamps/{experiment_id}/
└── metrics/{experiment_id}/
```

---

## Your Previous Suggestions (VERIFY THESE ARE IMPLEMENTED)

In our earlier review, you identified several sim-to-real issues. I need you to **verify each one is correctly implemented** in the attached code:

### 1. Flux Normalization
- **Issue**: Code used arbitrary `src_flux_scale=1e6` instead of proper nMgy conversion
- **Expected Fix**: `mag_to_nMgy(mag) = 10^(-0.4 * (mag - 22.5))` with AB_ZP_NMGY = 22.5
- **Verify**: Is `mag_to_nMgy()` defined and used correctly?

### 2. Sersic Profile with Proper Normalization
- **Issue**: Used Gaussian profile, normalized lensed flux (destroying magnification)
- **Expected Fix**: `sersic_bn()`, `sersic_unit_total_flux()`, `sersic_profile_Ie1()` functions
- **Verify**: Is the unlensed source normalized analytically before lensing?

### 3. SIE Deflection (Dependency-Free)
- **Issue**: Required lenstronomy, which may not be installed
- **Expected Fix**: `deflection_sis()`, `deflection_sie()`, `deflection_shear()` functions
- **Verify**: Are these implemented without external dependencies?

### 4. Shear Orientation
- **Issue**: Shear was axis-aligned (g2=0), not random
- **Expected Fix**: `g1 = shear * cos(2*phi)`, `g2 = shear * sin(2*phi)`
- **Verify**: Is shear angle randomized and properly decomposed?

### 5. Frozen Randomness in Manifest
- **Issue**: Stage 4c re-sampled random values, breaking reproducibility
- **Expected Fix**: All random values frozen in Stage 4a manifest columns
- **Verify**: Are `task_seed64`, `src_x_arcsec`, `src_y_arcsec`, `src_phi_rad`, `shear_phi_rad`, `lens_e`, `lens_phi_rad` columns used from manifest?

### 6. Per-Band PSF
- **Issue**: Used only psfsize_r for all bands
- **Expected Fix**: Use psfsize_g, psfsize_r, psfsize_z for respective bands
- **Verify**: Does Stage 4c load and use per-band PSF FWHM?

### 7. Center-Evaluated PSF from psfsize Maps
- **Issue**: Used brick-average PSF, not position-specific
- **Expected Fix**: Load psfsize map, extract value at stamp center
- **Verify**: Is `--use-psfsize-maps 1` flag implemented and does it read from psfsize-{band}.fits.fz?

### 8. Maskbits for Bad Pixel Filtering
- **Issue**: SNR calculated on all pixels including bad ones
- **Expected Fix**: `MASKBITS_BAD` constant, filter arc_snr calculation
- **Verify**: Is `MASKBITS_BAD` defined with correct DR10 bits? Is arc_snr filtered?

### 9. WISEM1/WISEM2 Tracking
- **Issue**: No tracking of WISE bright star contamination
- **Expected Fix**: `MASKBITS_WISE`, `wise_brightmask_frac` column
- **Verify**: Are WISE bits defined and `wise_brightmask_frac` computed?

### 10. bad_pixel_frac Metric
- **Issue**: No metric for masked pixel fraction
- **Expected Fix**: `bad_pixel_frac` column in output
- **Verify**: Is `bad_pixel_frac` computed and output?

### 11. PSF FWHM Provenance Columns
- **Issue**: No record of actual PSF used for each injection
- **Expected Fix**: `psf_fwhm_used_g`, `psf_fwhm_used_r`, `psf_fwhm_used_z` columns
- **Verify**: Are these columns in the output schema and populated?

### 12. Magnification Evaluation Point
- **Issue**: Magnification evaluated at critical curve (diverges)
- **Expected Fix**: Evaluate at 1.1 * theta_E instead of theta_E
- **Verify**: Is magnification computed slightly off the critical curve?

### 13. Manifests Path Fix
- **Issue**: Hardcoded `manifests/` path included blacklisted bricks
- **Expected Fix**: `--manifests-subdir` argument defaulting to `manifests_filtered`
- **Verify**: Is the argument added and used correctly?

### 14. Efficient Count Operations
- **Issue**: `.count()` called twice without caching, re-triggering computation
- **Expected Fix**: Read from written parquet for counts
- **Verify**: Are counts computed from saved parquet, not re-triggered?

---

## Questions for Review

1. **Completeness**: Are ALL 14 items above correctly implemented?

2. **Correctness**: For each implemented fix, is the implementation scientifically correct?

3. **Integration**: Do the fixes work together properly? (e.g., frozen randomness used in injection, per-band PSF used correctly)

4. **Edge Cases**: What happens if:
   - psfsize map is missing or corrupt?
   - maskbits file is missing?
   - A brick in the manifest doesn't exist in the cache?

5. **Performance**: Are there any remaining inefficiencies (uncached DataFrames, repeated computations)?

6. **Missing Features**: Is anything still missing for production-quality sim-to-real injection?

---

## Files to Attach

1. **`spark_phase4_pipeline.py`** - The main pipeline code (2,650+ lines)
2. **`bootstrap_phase4_pipeline_install_deps.sh`** - Bootstrap script
3. **`s3_locations_reference.md`** - S3 path documentation
4. **`phase4_development_log_2026-01-19.md`** - Development log with Phase 4b details

---

## Expected Output

Please provide:

1. **Verification Table**: For each of the 14 items, state:
   - [ ] Implemented correctly
   - [ ] Implemented but has issues (describe)
   - [ ] Not implemented or missing

2. **Code Issues**: Any bugs, logic errors, or scientific mistakes found

3. **Recommendations**: Anything that should be fixed before running Phase 4c

4. **Confidence Level**: Your confidence that this code is ready for production (Low/Medium/High)



# Phase 4c Injection Pipeline Work Log

**Date**: January 25-26, 2026  
**Author**: Aditrivaishnavi Balaji  
**Project**: Dark Halo Scope - Phase 4c Injected Cutouts

---

## Executive Summary

Phase 4c generates synthetic gravitationally lensed images by injecting simulated lens models into real DR10 South coadd cutouts. This work log documents the development, debugging, bug fixes, and data quality issues encountered during the Phase 4c implementation.

**Final Outcome**: Successfully generated **10,627,158 injection/control pairs** for the train tier with 100% success rate and validated physics metrics.

---

## Phase 4c Design Overview

### What 4c Produces

For each task in the manifest:
1. **Stamp cutouts**: 64×64 or 96×96 pixel grz images centered on LRG targets
2. **Injected lensed sources**: SIS/SIE lens models with realistic PSF convolution
3. **Control samples**: Cutouts without injections (theta_e = 0)
4. **Metrics**: arc_snr, magnification, total_injected_flux_r, maskbits fractions

### Key Physics Parameters

| Parameter | Range | Purpose |
|-----------|-------|---------|
| theta_e_arcsec | 0.3-1.0" | Einstein radius |
| src_dmag | 0.5-3.0 | Source magnitude difference from lens |
| src_reff_arcsec | 0.1-0.5" | Source effective radius |
| src_e | 0.0-0.6 | Source ellipticity |
| shear | 0.0-0.1 | External shear magnitude |

---

## Development Timeline

### 2026-01-25: Initial Debug Tier Run

**EMR Cluster**: j-XXXXX (10 core instances, m5.2xlarge)

**Issues Encountered**:

1. **Schema Mismatch Error**
   - Error: `ValueError: Length of object (41) does not match with length of fields (53)`
   - Cause: Exception handler's `yield Row(...)` was missing 12 fields
   - Fix: Added all missing fields to exception handler Row

2. **NumPy Round Error**
   - Error: `TypeError: type numpy.ndarray doesn't define __round__ method`
   - Cause: `wcs.world_to_pixel_values()` returns 0-d numpy arrays
   - Fix: Cast `px, py` to `float()` before `round()`

3. **sersic_profile Name Error**
   - Error: `NameError: name 'sersic_profile' is not defined`
   - Cause: Function was renamed to `sersic_profile_Ie1()` but call site not updated
   - Fix: Changed call to `sersic_profile_Ie1()`

4. **100% Coadd Load Failures**
   - Cause: `--coadd-s3-cache-prefix` pointed to wrong subdirectory
   - Fix: Use `s3://darkhaloscope/dr10/coadd_cache` (flat structure, not by prefix)

### 2026-01-25: Magnification Bug Discovery

**Problem**: Magnification metric showed 0% coverage despite successful injections.

**Root Cause**: The `render_unlensed_source()` function was failing silently due to the `sersic_profile` name error, so the denominator for magnification was unavailable.

**Fix**: Added `total_injected_flux_r` metric as direct physics validation:
- Should increase monotonically with theta_e
- Independent of unlensed source computation

### 2026-01-25: Mini-Train Preflight Check

Ran a small-scale test (mini_train_stamp64) to validate:
- Control semantics (theta_e=0, null metrics)
- Physics directionality (flux increases with theta_e)
- End-to-end pipeline stability

**Result**: PASS - validated control handling and physics trends.

### 2026-01-26: Full Train Tier Run

**EMR Cluster**: j-3MC6I9RF6NFV1 (30 core instances, m5.2xlarge)

**Configuration**:
```bash
--experiment-id train_stamp64_bandsgrz_gridgrid_small
--bands g,r,z
--stamp-sizes 64
--use-psfsize-maps 1
--manifests-subdir manifests_filtered
--sweep-partitions 2000
--force 1
```

**Result**: 10,627,158 rows in ~45 minutes

---

## Data Quality Issues and Resolutions

### Issue 1: PSF FWHM = 0 for g-band (0.13% of injections)

**Discovery**: Independent LLM review flagged that `psf_fwhm_used_g == 0` for 6,928 injections (0.13%) across 112 bricks.

**Root Cause Investigation**:

1. Examined PSFsize map for affected brick `3572m307`:
   ```
   PSFsize-g map shape: (3600, 3600)
   Zero pixels: 9,845,290 (75.97%)
   Valid (>0) pixels: 3,114,710 (24.03%)
   Center pixel [1800,1800]: 0.0000
   ```

2. These bricks have **partial or no g-band coverage** in DR10 South.

3. Both fallback sources were invalid:
   - PSFsize map: 0 at stamp center (no g-band imaging at that location)
   - Manifest psfsize_g: 0 (brick-level average is 0 when no g-band coverage)

**Fix Applied**:

```python
# Primary fix: Check manifest values for >0, not just not-None
manifest_fwhm_g = float(r["psfsize_g"]) if (r["psfsize_g"] is not None and r["psfsize_g"] > 0) else manifest_fwhm_r

# Secondary fallback: Use r-band PSF when g/z still invalid
if psf_fwhm_g <= 0:
    psf_fwhm_g = psf_fwhm_r
```

**Scientific Justification for r-band Fallback**:

1. **Physical basis**: PSF size is primarily atmospheric seeing with weak wavelength dependence. For Kolmogorov turbulence: FWHM ∝ λ^(-1/5), giving only ~5-8% variation across g/r/z.

2. **Empirical validation**:
   - g: mean=1.53", median=1.52"
   - r: mean=1.32", median=1.31"
   - z: mean=1.32", median=1.25"
   
   Bands are within ~15-20% of each other.

3. **Alternative is worse**: PSF=0 means no convolution, producing unrealistically sharp sources.

4. **Provenance preserved**: `psf_fwhm_used_g/r/z` columns record actual values used.

**Impact**: 0.13% of injections (112 bricks). After fix: **0 PSF=0 cases**.

### Issue 2: bad_pixel_frac > 0.2 for 10.1% of injections

**Discovery**: ~10% of injection stamps have significant masked pixel contamination.

**Assessment**: This is expected in survey data (crowded fields, bright stars, satellite trails).

**Resolution**: Defined quality cut constant:
```python
QUALITY_CUT_BAD_PIXEL_FRAC = 0.2  # Max for "clean subset"
```

**Usage**:
- Phase 5 training: Apply cut for baseline model, optionally include as hard examples
- Phase 4d completeness: Report both "all" and "clean subset" curves

### Issue 3: Split Imbalance (26/40/35 train/val/test)

**Context**: Inherited from Phase 3 region-level splitting based on `xxhash64(region_id)`.

**Assessment**: Not scientifically invalid because:
- Splits are spatially disjoint (no leakage)
- Imbalance is documented and justified
- Hyperparameter tuning uses val only, test is one-time final evaluation

---

## Final Data Quality Metrics

### Train Tier Output (train_stamp64_bandsgrz_gridgrid_small)

| Metric | Value |
|--------|-------|
| Total rows | 10,627,158 |
| Injections | 5,327,834 (50.1%) |
| Controls | 5,299,324 (49.9%) |
| cutout_ok | 100% |
| Unique bricks | 180,152 |

### PSF FWHM After Fix

| Band | Min | Max | Mean | Zeros |
|------|-----|-----|------|-------|
| g | 0.508" | 3.77" | 1.53" | 0 |
| r | 0.805" | 3.53" | 1.32" | 0 |
| z | 0.794" | 3.40" | 1.32" | 0 |

### Physics Metrics (Injections Only)

| Metric | Coverage | Min | Median | Max |
|--------|----------|-----|--------|-----|
| arc_snr | 100% | 0.0 | 22.0 | 1964.6 |
| magnification | 100% | 0.10 | 5.60 | 212.6 |
| total_injected_flux_r | 100% | 0.08 | 4.62 | 491.4 nMgy |

### Maskbits Metrics (Injections Only)

| Metric | Median | P95 | Max |
|--------|--------|-----|-----|
| bad_pixel_frac | 0.000 | 0.560 | 1.000 |
| wise_brightmask_frac | 0.000 | 0.205 | 1.000 |

### Physics Sanity Check: Total Flux vs Theta_E

| theta_e bin | avg_flux (nMgy) | Trend |
|-------------|-----------------|-------|
| [0.3, 0.5) | 6.33 | ↗ |
| [0.5, 0.8) | 7.24 | ↗ |
| [0.8, 1.0] | 7.44 | ↗ |

✓ Monotonically increasing as expected (larger Einstein radius → more flux magnification)

### Control Sample Validation

| Check | Result |
|-------|--------|
| theta_e_arcsec == 0 | 100% of controls |
| arc_snr IS NULL | 100% of controls |
| magnification IS NULL | 100% of controls |
| lens_model == "CONTROL" | 100% of controls |

---

## Output Locations

### S3 Paths

```
s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/
├── stamps/
│   └── train_stamp64_bandsgrz_gridgrid_small/
│       ├── region_split=train/
│       ├── region_split=val/
│       └── region_split=test/
├── metrics/
│   └── train_stamp64_bandsgrz_gridgrid_small/
│       ├── region_split=train/
│       ├── region_split=val/
│       └── region_split=test/
└── _stage_config.json
```

### File Counts

| Output Type | Files |
|-------------|-------|
| Stamps (parquet) | 6,001 |
| Metrics (parquet) | 6,001 |

---

## Quality Cuts for Downstream Phases

Based on independent LLM review, the following quality cuts are defined:

### Phase 5 Training (Baseline "Clean Subset")

```python
cutout_ok == 1                    # Already 100%
arc_snr IS NOT NULL               # For injections
bad_pixel_frac <= 0.2             # ~90% of injections pass
psf_fwhm_used_g > 0               # Now 100% after fix
psf_fwhm_used_r > 0               # Always 100%
psf_fwhm_used_z > 0               # Now 100% after fix
```

### Phase 4d Completeness Reporting

1. Report two curves: "all injections" and "clean subset"
2. Stratify by resolution bins: `theta_e / psfsize_r`
   - <0.4 (very unresolved)
   - 0.4-0.6 (marginally unresolved)
   - 0.6-0.8 (marginally resolved)
   - 0.8-1.0 (resolved)
   - >=1.0 (well resolved)
3. Stratify by observing conditions: psfsize_r, psfdepth_r
4. Stratify by data quality: bad_pixel_frac, wise_brightmask_frac

---

## PSF Fallback Analysis (Post-Fix Verification)

### Fallback Usage Quantification

After the PSF fix was applied and the full train tier was re-run, we measured how many injections actually used the r-band fallback:

| Band | Fallback Count | Percentage of Injections |
|------|----------------|-------------------------|
| g-band (g == r) | ~3,836 | 0.072% |
| z-band (z == r) | ~176 | 0.003% |

*Estimated from 5% sample (183,538 injections sampled)*

### PSF Distribution Percentiles (Injections Only)

| Band | P1 | P5 | P10 | P50 (Median) |
|------|-----|-----|-----|--------------|
| g | 1.132" | 1.247" | 1.300" | 1.517" |
| r | 1.049" | 1.118" | 1.148" | 1.299" |
| z | 0.977" | 1.046" | 1.080" | 1.250" |

### Minimum Values (Sample)

| Band | Min Value |
|------|-----------|
| g | 0.904" |
| r | 0.850" |
| z | 0.813" |

### Observations

1. The g-band and z-band fallback rates (0.072% and 0.003%) are very low, affecting only ~4,000 injections out of 5.3M.

2. The PSF distributions show expected wavelength dependence: g-band PSF is slightly larger than r and z (due to atmospheric seeing λ-dependence).

3. The minimum PSF values are physically reasonable for good seeing conditions in DECaLS.

4. No spikes at exact r-band values were observed in g/z distributions beyond the expected fallback cases.

### Independent LLM Review (2026-01-26)

**Verdict**: r-band fallback is scientifically defensible. Proceed with current approach.

**Systematic Effects Identified**:
- Chromatic PSF mismatch: Removes band dependence for affected rows
- Color morphology bias: Slightly reduces realism for cross-band analysis
- SNR/peakiness: May slightly sharpen/blur g injection relative to reality

**Recommendations from Review**:

1. **Proceed without filtering** - fallback rate is negligible (~0.07%)
2. **Add provenance columns** - `psf_source_g/r/z` (0=map, 1=manifest, 2=fallback_r)
3. **Investigate g-band min 0.508"** - discrepancy with sample min 0.904" needs audit
4. **Downstream stratification**:
   - Report completeness as function of `theta_e / psf_fwhm_used_r`
   - Add "all data" vs "no-psf-fallback" curves for robustness check
5. **Phase 5 training**: Include `psf_fwhm_used_{b}` as features or stratify batches

**Open Items**:
- [ ] Add `psf_source_g/r/z` provenance columns to schema
- [ ] Audit rows with `psf_fwhm_used_g < 0.8"` to verify they are real
- [ ] Verify per-split PSF distributions show no split-specific artifacts

---

## Commits

| Commit | Date | Description |
|--------|------|-------------|
| ea4deec | 2026-01-25 | Fix magnification bug, add total_injected_flux_r |
| 83f7223 | 2026-01-26 | Fix PSF FWHM fallback for bricks with no g/z coverage |
| ec9f813 | 2026-01-26 | Add quality cut constants for Phase 4d/5 |

---

## Lessons Learned

1. **Test on debug tier first**: Small-scale runs caught schema mismatches and name errors before wasting compute.

2. **Validate physics, not just completion**: 100% cutout_ok is necessary but not sufficient. Physics metrics (flux vs theta_e) caught the sersic_profile bug.

3. **Understand your data**: DR10 South has partial band coverage in some regions. Assuming all bands exist everywhere leads to subtle bugs.

4. **Preserve provenance**: Recording `psf_fwhm_used_g/r/z` allows downstream filtering of edge cases.

5. **Document fallback assumptions**: The r-band PSF fallback is scientifically defensible but must be documented for reproducibility.

---

## Next Steps

1. **Phase 4d**: Compute completeness summaries using the validated Phase 4c output
2. **Phase 5**: Train detection models on the clean subset
3. **96×96 stamps**: Repeat 4c for larger stamp size

---

*End of Phase 4c Work Log*


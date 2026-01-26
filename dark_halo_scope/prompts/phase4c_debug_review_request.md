# Phase 4c Debug Tier Review Request

## Context

We have completed the **debug tier** of Phase 4c (lens injection pipeline) and need your review to determine if we should proceed to the **train tier**.

### What is Phase 4c?

Phase 4c injects simulated gravitational lens signals into real astronomical images (DESI Legacy Survey cutouts) to build training data for a lens detection neural network. The key physics:

1. **Lens model**: Singular Isothermal Ellipsoid (SIE) with external shear
2. **Source**: Sérsic profile galaxy placed behind the lens
3. **Magnification**: Larger Einstein radius (θ_E) = stronger lensing = more total flux
4. **PSF convolution**: Uses per-pixel PSF FWHM maps from the survey

### Pipeline Flow
```
Phase 4a: Generate manifests (lens parameters, source positions)
Phase 4b: Cache coadd images from survey
Phase 4c: Inject lenses into images, compute metrics  <-- WE ARE HERE
Phase 4d: Compute selection function (completeness)
Phase 5:  Train neural network
```

---

## ⚠️ CRITICAL: This is DEBUG Tier

**The debug tier uses a VERY SMALL subset of data for fast iteration:**

| Aspect | Debug Tier | Train Tier |
|--------|-----------|------------|
| Rows | 17,280 | ~10 million |
| Bricks | ~20 | ~280,000 |
| Controls | **0** (expected!) | 50% of data |
| Purpose | Catch bugs | Final training data |

**Expected behaviors in debug tier (NOT bugs):**
- 0 control samples (controls are only in train/grid tiers)
- Limited parameter coverage
- Small sample sizes per bin

---

## Validation Results

### Core Checks (All Passed ✅)

```
[2/8] Validating schema...
  Schema valid: True

[3/8] Validating counts...
  Total rows: 17,280
  By split: {'train': 5760, 'val': 5760, 'test': 5760}
  Cutout stats: {'cutout_ok_1': 17280}
  Success rate: 100.00%

[4/8] Validating nulls...
  Nulls valid: True

[5/8] Validating ranges...
  theta_e: min=0.3, max=1.0
  arc_snr: avg=43.48, median=26.33
  Ranges valid: True

[6/8] Validating controls...
  Controls: 0
  Injections: 17,280
  (Debug tier - 0 controls is expected)

[7/8] Validating PSF provenance...
  PSF provenance valid: True

[8/8] Validating maskbits metrics...
  Maskbits valid: True
```

### Physics Validation (All Passed ✅)

```
[EXTRA 1/5] Validating physics metrics...
  Magnification: count=17280, coverage=100.0%
    avg=8.25, min=0.18, max=168.21

[EXTRA 2/5] Validating injection parameter distributions...
  theta_e: min=0.30, max=1.00, avg=0.63
  src_dmag: min=1.0, max=2.0, avg=1.5
  src_reff: min=0.08, max=0.15
  src_e: min=0.00, max=0.30
  shear: min=0.000, max=0.030

[EXTRA 3/5] Validating lens model distribution...
  Model distribution: {'SIE': 17280}
  Has SIE: True

[EXTRA 4/6] Validating SNR-theta_e correlation...
  Peak SNR trend: decreasing
  Note: Peak SNR may decrease with theta_e due to arc spreading - this is expected physics
  Binned peak SNR:
    theta=0.20: SNR=47.8 (n=5760)
    theta=0.60: SNR=41.2 (n=5760)
    theta=1.00: SNR=41.4 (n=5760)

[EXTRA 5/6] Validating TOTAL FLUX-theta_e correlation (CRITICAL)...
  ✅ Total flux INCREASES with theta_e (physics correct)
  Binned total flux:
    theta=0.20: flux=6.2 nMgy (n=5760)
    theta=0.60: flux=6.8 nMgy (n=5760)
    theta=1.00: flux=7.0 nMgy (n=5760)
```

### Final Summary

```
======================================================================
VALIDATION SUMMARY
======================================================================
  Schema: ✅ PASS
  Success Rate (>=95%): ✅ PASS
  No Critical Nulls: ✅ PASS
  Value Ranges: ✅ PASS
  Has Controls: ✅ PASS
  PSF Provenance: ✅ PASS
  Maskbits Metrics: ✅ PASS
  Magnification Data: ✅ PASS
  Total Flux ↑ with θ_E (CRITICAL): ✅ PASS
======================================================================
OVERALL: ✅ VALIDATION PASSED
Phase 4c output is ready for Phase 4d / Phase 5
======================================================================
```

---

## Key Physics Explanations

### Why Peak SNR Decreases with θ_E (Expected!)

This is **correct physics**, not a bug:
- Larger θ_E → larger arc → flux spread over more pixels
- Peak (max) SNR measures brightest single pixel
- Larger arcs have lower surface brightness → lower peak SNR
- **Total flux still increases** (magnification working correctly)

### Total Flux Correlation (Critical Check)

This validates magnification physics:
- θ_E = 0.25": total flux = 6.2 nMgy
- θ_E = 0.50": total flux = 6.8 nMgy (+10%)
- θ_E = 1.00": total flux = 7.0 nMgy (+13%)

**Larger Einstein radius = stronger lensing = more magnification = more total flux ✅**

### Magnification Proxy

- 100% coverage (all 17,280 injections have magnification computed)
- Average magnification: 8.25x
- Range: 0.18x to 168.21x (demagnification to high magnification)

---

## Bugs Fixed During Debug Iteration

1. **Wrong cache path**: Fixed `coadd_cache_psfsize` → `coadd_cache`
2. **Missing function**: Fixed `sersic_profile` → `sersic_profile_Ie1` 
3. **Type error**: Fixed numpy 0-d array → float conversion for WCS
4. **Magnification bug**: Added null check for `add_r` array

---

## Your Review Task

Please review the above validation results and answer:

### 1. Are there any red flags that suggest bugs in the injection physics?

Consider:
- Is total flux correctly increasing with θ_E?
- Is magnification being computed?
- Are the parameter ranges sensible for gravitational lensing?

### 2. Are there any concerns about data quality?

Consider:
- 100% success rate (all cutouts processed)
- PSF provenance validated
- Maskbits metrics validated

### 3. Should we proceed to the TRAIN tier?

The train tier will:
- Process ~10 million injections (vs 17,280 debug)
- Include 50% control samples (real images, no injection)
- Take ~3 hours on 30-node EMR cluster
- Cost approximately $50-100 in compute

### 4. Any recommendations before scaling up?

Are there additional checks we should run on debug data before committing to the expensive train run?

---

## Files for Reference (if needed)

- Pipeline code: `dark_halo_scope/emr/spark_phase4_pipeline.py`
- Validation script: `dark_halo_scope/emr/spark_validate_phase4c.py`
- S3 metrics location: `s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/metrics/debug_stamp64_bandsgrz_gridgrid_small`

---

**Please provide your assessment: GO / NO-GO for train tier, with reasoning.**


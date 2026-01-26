# Phase 4c PSF Fallback Analysis - Independent Review Request

## Context

This is Phase 4c of the Dark Halo Scope pipeline, which generates synthetic gravitationally lensed images by injecting simulated lens models into real DR10 South coadd cutouts.

A PSF FWHM fallback mechanism was implemented to handle bricks with missing or invalid PSF data in g-band and z-band.

## The Problem

112 bricks in DR10 South have no g-band coverage (or partial coverage). For these bricks:
- The PSFsize map has 0 at the stamp center location
- The manifest `psfsize_g` is also 0.0 (brick-level average is 0 when no g-band coverage)

Without intervention, injections in these bricks would have `psf_fwhm_used_g = 0`, meaning no convolution would be applied to the g-band injection.

## The Fix Applied

```python
# Primary fix: Check manifest values for >0, not just not-None
manifest_fwhm_g = float(r["psfsize_g"]) if (r["psfsize_g"] is not None and r["psfsize_g"] > 0) else manifest_fwhm_r

# Secondary fallback: Use r-band PSF when g/z still invalid after all lookups
if psf_fwhm_g <= 0:
    psf_fwhm_g = psf_fwhm_r
if psf_fwhm_z <= 0:
    psf_fwhm_z = psf_fwhm_r
```

## Scientific Rationale Provided

The rationale given for using r-band as fallback:
1. PSF size is primarily atmospheric seeing with weak wavelength dependence (FWHM ∝ λ^(-1/5) for Kolmogorov turbulence)
2. This gives ~5-8% variation across g/r/z bands (475nm/622nm/913nm)
3. The alternative (PSF=0 → no convolution) would produce unrealistically sharp injected sources

## Post-Fix Validation Data

### Dataset Summary (Train Tier)

| Metric | Value |
|--------|-------|
| Total rows | 10,627,158 |
| Injections | 5,327,834 |
| Controls | 5,299,324 |

### PSF Zero Counts (Before vs After Fix)

**Before Fix:**
| Band | Zeros | Rate |
|------|-------|------|
| g | 6,928 | 0.130% |
| r | 0 | 0.000% |
| z | 110 | 0.002% |

**After Fix:**
| Band | Zeros |
|------|-------|
| g | 0 |
| r | 0 |
| z | 0 |

### Fallback Usage Quantification (5% Sample: 183,538 injections)

| Metric | Count | Percentage |
|--------|-------|------------|
| g-band fallback (psf_fwhm_used_g == psf_fwhm_used_r) | 132 | 0.072% |
| z-band fallback (psf_fwhm_used_z == psf_fwhm_used_r) | 6 | 0.003% |

Extrapolated to full dataset: ~3,800 g-band fallbacks, ~160 z-band fallbacks

### PSF Distribution Percentiles (Injections Only, 5% Sample)

| Band | P1 | P5 | P10 | P50 (Median) |
|------|-----|-----|-----|--------------|
| g | 1.132" | 1.247" | 1.300" | 1.517" |
| r | 1.049" | 1.118" | 1.148" | 1.299" |
| z | 0.977" | 1.046" | 1.080" | 1.250" |

### PSF Minimum Values (5% Sample)

| Band | Min Value |
|------|-----------|
| g | 0.904" |
| r | 0.850" |
| z | 0.813" |

### Full Dataset PSF Statistics (from earlier full scan)

| Band | Min | Max | Mean | Median |
|------|-----|-----|------|--------|
| g | 0.508" | 3.769" | 1.532" | 1.516" |
| r | 0.805" | 3.528" | 1.325" | 1.307" |
| z | 0.794" | 3.403" | 1.323" | 1.254" |

## Provenance Tracking

The `psf_fwhm_used_g`, `psf_fwhm_used_r`, `psf_fwhm_used_z` columns record the actual PSF FWHM values used for each injection, regardless of source (map, manifest, or fallback).

There is currently no explicit `psf_source_g/r/z` column indicating whether the value came from the PSFsize map, manifest brick-average, or r-band fallback.

## Questions for Review

1. Is the r-band PSF fallback scientifically defensible for gravitational lens injection simulations? What systematic biases does it introduce?

2. Given that the fallback affects only ~0.07% of injections, is this rate low enough to proceed without concern, or should these injections be flagged/filtered in downstream analysis?

3. The full dataset shows g-band min PSF of 0.508", while the 5% sample shows 0.904". Is this discrepancy concerning? What could explain it?

4. Should a provenance column (`psf_source_g/r/z`) be added to explicitly track which injections used fallback values?

5. Are there any other data quality concerns in the PSF statistics that should be investigated before proceeding to Phase 4d (completeness) and Phase 5 (training)?

6. What stratification or filtering recommendations apply to downstream phases given this PSF fallback mechanism?


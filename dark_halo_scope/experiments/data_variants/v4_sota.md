# Data Variant: v4_sota

**Status**: Active (Gen2)  
**Created**: 2026-01-30  
**Used by**: Gen2

## Summary

Improved data variant with unpaired controls and extended theta_e range. Still uses Gaussian PSF.

## Key Improvements over v3

| Aspect | v3_color_relaxed | v4_sota |
|--------|-----------------|---------|
| Control Type | Paired | **Unpaired** |
| theta_e range | 0.3-1.0" | **0.5-2.5"** |
| Total configs | 48 | **1,008** |

## Parent Sample Selection (Phase 2/3)

Same as v3_color_relaxed: ~145,000 LRG targets

## Injection Grid (Phase 4a)

| Parameter | Values |
|-----------|--------|
| theta_e_arcsec | [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5] |
| src_dmag | [0.5, 1.0, 1.5, 2.0] |
| src_reff_arcsec | [0.06, 0.10, 0.15, 0.20] |
| src_e | [0.0, 0.2, 0.4] |
| shear | [0.0, 0.02, 0.04] |

**Total configs**: 1,008

## Phase 4c Parameters

| Parameter | Value |
|-----------|-------|
| PSF Model | Gaussian |
| Source Mode | Parametric (Sersic n=1) |
| Stamp Size | 64×64 |
| Bands | g, r, z |
| Control Type | UNPAIRED |
| Resolvability Filter | None |

## Verification: Unpaired Controls

```python
# Analysis of v4_sota data:
Total rows analyzed: 48,690
Unique control galaxy positions: 2,715
Unique positive galaxy positions: 2,770
Overlapping positions: 0

# CONFIRMED: Different galaxies for controls vs positives
```

## S3 Locations

```
Phase 4a manifests: s3://darkhaloscope/phase4_pipeline/phase4a/v4_sota/manifests/
Phase 4c stamps: s3://darkhaloscope/phase4_pipeline/phase4c/v4_sota/stamps/train_stamp64_bandsgrz_gridgrid_sota/
```

## Known Limitations

1. **Gaussian PSF**: Does not capture DECaLS PSF extended wings
2. **Parametric Sources**: Sersic n=1 is too smooth, missing clumpy morphology


# Data Variant: v3_color_relaxed

**Status**: Deprecated (Gen1 only)  
**Created**: 2026-01-28  
**Used by**: Gen1

## Summary

Initial data variant with relaxed color cuts. **Known to have critical issues.**

## Issues

1. **Paired Controls**: Controls used the SAME galaxy as positives (just without injection)
2. **Unresolved Injections**: theta_e = [0.3, 0.6, 1.0] with median PSF ~1.3" → 60% unresolved
3. **Gaussian PSF**: Does not match DECaLS PSF wings

## Parent Sample Selection (Phase 2/3)

```sql
TYPE != 'PSF'           -- Extended sources only
flux_r > 0 AND flux_z > 0 AND flux_w1 > 0
z < 20.4                -- Magnitude cut
r - z > 0.4             -- LRG color cuts
z - W1 > 0.8
```

**Output**: ~145,000 LRG targets

## Injection Grid (Phase 4a)

| Parameter | Values |
|-----------|--------|
| theta_e_arcsec | [0.3, 0.6, 1.0] |
| src_dmag | [1.0, 2.0] |
| src_reff_arcsec | [0.08, 0.15] |
| src_e | [0.0, 0.3] |
| shear | [0.0, 0.03] |

**Total configs**: 48

## Phase 4c Parameters

| Parameter | Value |
|-----------|-------|
| PSF Model | Gaussian |
| Source Mode | Parametric (Sersic n=1) |
| Stamp Size | 64×64 |
| Bands | g, r, z |
| Control Type | PAIRED |

## S3 Locations

```
Phase 4c stamps: s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/stamps/train_stamp64_bandsgrz_gridgrid_small/
```

## DO NOT USE

This variant should not be used for new experiments due to the paired control issue.


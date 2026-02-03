# Data Variant: v4_sota_moffat

**Status**: Active (Gen3, Gen4)  
**Created**: 2026-01-31  
**Used by**: Gen3, Gen4

## Summary

Current best data variant with Moffat PSF and resolvability filtering. Uses same grid as v4_sota but with improved PSF model.

## Key Improvements over v4_sota

| Aspect | v4_sota | v4_sota_moffat |
|--------|---------|----------------|
| PSF Model | Gaussian | **Moffat (β=3.5)** |
| PSF Evaluation | Brick average | **Center-evaluated** |
| Resolvability Filter | None | **θ/PSF ≥ 0.5** |

## Parent Sample Selection (Phase 2/3)

Same as v3/v4: ~145,000 LRG targets

## Injection Grid (Phase 4a)

Same as v4_sota:
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
| PSF Model | **Moffat** |
| Moffat Beta | **3.5** |
| Source Mode | Parametric (Sersic n=1) |
| Stamp Size | 64×64 |
| Bands | g, r, z |
| Control Type | UNPAIRED |
| Use PSFSize Maps | Yes (center-evaluated) |

## Phase 4c Command

```bash
spark-submit spark_phase4_pipeline.py \
  --stage 4c \
  --variant v4_sota_moffat \
  --psf-model moffat \
  --moffat-beta 3.5 \
  --use-psfsize-maps 1 \
  --experiment-id train_stamp64_bandsgrz_gridgrid_sota
```

## S3 Locations

```
Phase 4a manifests: s3://darkhaloscope/phase4_pipeline/phase4a/v4_sota/manifests/
Phase 4c stamps: s3://darkhaloscope/phase4_pipeline/phase4c/v4_sota_moffat/stamps/train_stamp64_bandsgrz_gridgrid_sota/
```

## Known Limitations

1. **Parametric Sources**: Sersic n=1 is still too smooth
2. **No Real Hard Negatives**: Training only on synthetic controls
3. **No COSMOS Sources**: Missing clumpy high-z galaxy morphology

## Training Results

| Model | tpr@fpr1e-4 | fpr@tpr0.85 | Notes |
|-------|-------------|-------------|-------|
| Gen3 | 84.5% | <0.1% | 50 epochs, focal loss |
| Gen4 | TBD | TBD | + hard negative mining |


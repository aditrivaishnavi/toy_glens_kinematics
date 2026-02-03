# Data Variant: v5_cosmos_source

**Status**: PLANNED  
**Target Date**: TBD  
**Used by**: Gen5 (planned)

## Summary

Next-generation data variant using real COSMOS galaxy morphologies as lensed sources instead of parametric Sersic profiles.

## Key Improvements over v4_sota_moffat

| Aspect | v4_sota_moffat | v5_cosmos_source |
|--------|----------------|------------------|
| Source Morphology | Sersic n=1 (parametric) | **Real COSMOS galaxies** |
| Source Clumpiness | None (smooth) | **Real HII clumps, dust** |
| Source Colors | Uniform | **Realistic gradients** |

## Parent Sample Selection (Phase 2/3)

Same as v3/v4: ~145,000 LRG targets

## COSMOS Source Library

| Property | Value |
|----------|-------|
| Source | GalSim COSMOS catalog |
| N galaxies | ~87,000 |
| Resolution | 0.03"/pixel (HST/ACS) |
| Bands | F814W (I-band) |
| Selection | Exclude point sources |

## Injection Grid (Phase 4a)

Same as v4_sota but with COSMOS source sampling:
| Parameter | Values | Notes |
|-----------|--------|-------|
| theta_e_arcsec | [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5] | |
| src_dmag | [0.5, 1.0, 1.5, 2.0] | Relative to host |
| source_id | Random from COSMOS | New parameter |
| clumpiness | Real value from COSMOS | Logged for analysis |

## Phase 4c Parameters

| Parameter | Value |
|-----------|-------|
| PSF Model | Moffat |
| Moffat Beta | 3.5 |
| **Source Mode** | **cosmos** |
| **COSMOS Library** | s3://darkhaloscope/cosmos_sources.h5 |
| Stamp Size | 64×64 |
| Bands | g, r, z |
| Control Type | UNPAIRED |

## Phase 4c Command (Planned)

```bash
spark-submit spark_phase4_pipeline.py \
  --stage 4c \
  --variant v5_cosmos_source \
  --psf-model moffat \
  --moffat-beta 3.5 \
  --source-mode cosmos \
  --cosmos-library s3://darkhaloscope/cosmos_sources.h5 \
  --experiment-id train_stamp64_bandsgrz_cosmos
```

## Implementation Requirements

1. **COSMOSLoader Update**: Modify to load real COSMOS HDF5
2. **spark_phase4_pipeline.py Update**: Add `--source-mode cosmos` flag
3. **Lensing Integration**: Resample COSMOS cutout, apply ray-tracing, convolve with PSF

## Expected Improvements

- More realistic arc morphology
- Better generalization to real lenses
- Reduced shortcut learning on smooth Sersic profiles


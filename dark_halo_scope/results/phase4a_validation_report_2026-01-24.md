# Phase 4a Validation Report

**Date**: 2026-01-24  
**Cluster ID**: j-2CD55XYNVLT7U  
**Status**: ✅ **PASSED**

---

## Executive Summary

Phase 4a manifest generation completed successfully. All validation checks passed. The output contains **11.7 million injection tasks** across debug, grid, and train tiers with proper holdout splits (train/val/test). The data is ready for Stage 4b (coadd caching) and Stage 4c (injection).

---

## Configuration Used

| Parameter | Value |
|-----------|-------|
| Variant | v3_color_relaxed |
| Stamp Size | 64 pixels |
| Bandset | grz |
| Replicates | 2 |
| Control Fraction (train) | 50% |
| Control Fraction (grid) | 10% |
| Control Fraction (debug) | 0% |
| Train n_total_per_split | 200,000 |
| Grid n_per_config | 40 |
| Debug n_per_config | 5 |

---

## Task Counts

### By Tier and Split

| Tier | Train Split | Val Split | Test Split | **Total** |
|------|-------------|-----------|------------|-----------|
| debug | 5,760 | 5,760 | 5,760 | **17,280** |
| grid | 330,480 | 345,600 | 344,880 | **1,020,960** |
| train | 2,768,516 | 4,198,484 | 3,681,570 | **10,648,570** |
| **TOTAL** | **3,104,756** | **4,549,844** | **4,032,210** | **11,686,810** |

### Observations

1. **Train tier dominates** (~91% of tasks) as expected for ML training
2. **Split imbalance**: Val > Test > Train for train tier
   - This reflects the underlying Phase 3 region split imbalance (test regions are large)
   - Not a bug - the splits are determined by region_split from Phase 3
3. **Grid tier**: ~1M tasks for systematic completeness analysis
4. **Debug tier**: ~17K tasks for quick end-to-end testing

---

## Control Sample Analysis

| Tier | Split | Observed | Expected | Status |
|------|-------|----------|----------|--------|
| debug | train | 0.0% | 0.0% | ✅ |
| debug | val | 0.0% | 0.0% | ✅ |
| debug | test | 0.0% | 0.0% | ✅ |
| grid | train | 8.9% | 10.0% | ✅ |
| grid | val | 11.9% | 10.0% | ✅ |
| grid | test | 10.9% | 10.0% | ✅ |
| train | train | 49.9% | 50.0% | ✅ |
| train | val | 49.8% | 50.0% | ✅ |
| train | test | 49.9% | 50.0% | ✅ |

**Analysis**: Control fractions match expected values within tolerance. The 50/50 split for train tier ensures balanced binary classification training.

---

## Schema Validation

### All 42 Required Columns Present ✅

```
task_id, experiment_id, selection_set_id, selection_strategy, ranking_mode,
region_id, region_split, brickname, ra, dec, zmag, rmag, w1mag, rz, zw1,
psfsize_r, psfdepth_r, ebv, psf_bin, depth_bin, config_id, theta_e_arcsec,
src_dmag, src_reff_arcsec, src_e, shear, stamp_size, bandset, replicate,
is_control, task_seed64, src_x_arcsec, src_y_arcsec, src_phi_rad, shear_phi_rad,
src_gr, src_rz, lens_model, lens_e, lens_phi_rad, ab_zp_nmgy, pipeline_version
```

### New Columns Added (vs previous version)

| Column | Description | Purpose |
|--------|-------------|---------|
| `lens_model` | "SIE" or "CONTROL" | Tracks lens model used |
| `lens_e` | Lens ellipticity [0.05, 0.5] | SIE parameter |
| `lens_phi_rad` | Lens orientation [0, π) | SIE parameter |
| `task_seed64` | xxhash64 of stable identifiers | Reproducibility |
| `src_x_arcsec`, `src_y_arcsec` | Source offset | Frozen randomness |
| `src_phi_rad` | Source orientation | Frozen randomness |
| `shear_phi_rad` | Shear orientation [0, π) | Proper shear physics |
| `src_gr`, `src_rz` | Source colors | Multi-band injection |
| `ab_zp_nmgy` | 22.5 (Legacy Survey) | Flux unit provenance |
| `pipeline_version` | Version string | Reproducibility |

---

## Parameter Range Validation

| Parameter | Min | Max | Valid Range | Status |
|-----------|-----|-----|-------------|--------|
| theta_e_arcsec | 0.0 | 1.2 | [0, 5] arcsec | ✅ |
| shear_phi_rad | 0.0 | ~π | [0, π] | ✅ |
| src_gr | -0.5 | 1.5 | [-1, 2] | ✅ |
| src_rz | -0.5 | 1.5 | [-1, 2] | ✅ |

---

## Data Quality Checks

| Check | Result | Notes |
|-------|--------|-------|
| Null key columns | ✅ PASSED | No nulls in brickname, ra, dec, experiment_id |
| Control theta_e = 0 | ✅ PASSED | All controls have theta_e = 0.0 |
| Injection theta_e > 0 | ✅ PASSED | All injections have theta_e > 0.0 |
| Frozen randomness | ✅ PASSED | src_x/y/phi, shear_phi non-null for injections |
| O(n²) explosion | ✅ PASSED | No sampling bug detected |
| Lens parameter consistency | ✅ PASSED | Controls have CONTROL model, injections have SIE |

---

## Bricks Manifest

- **Unique bricks**: 180,373
- **Coverage**: All bricks needed for Stage 4b coadd caching

---

## Performance Analysis

| Metric | This Run | Previous Run | Improvement |
|--------|----------|--------------|-------------|
| Runtime | 10 min 42 sec | 1 hr 11 min | **6.6x faster** |
| Tasks generated | 11.7M | Similar | - |

**Root cause of speedup**: Fixed O(n²) sampling bug for debug/grid tiers. Previously, the code sampled `n_per_config × n_cfg` galaxies then crossed with `n_cfg` configs, causing O(n_cfg²) explosion.

---

## Recommendations

### Ready for Next Steps

1. **Stage 4b**: Cache coadds for 180,373 bricks
   - Estimated: ~2-4 hours with 10 core nodes
   - Cost: ~$10-20

2. **Stage 4c**: Generate injected cutouts
   - Train tier: ~10.6M tasks (large, run with metrics-only first)
   - Grid tier: ~1M tasks (systematic completeness)
   - Debug tier: ~17K tasks (quick validation)

### Suggested Stage 4b Command

```bash
python3 emr/submit_phase4_pipeline_emr_cluster.py \
  --region us-east-2 \
  --stage 4b \
  --log-uri s3://darkhaloscope/emr-logs/phase4/ \
  --service-role EMR_DefaultRole \
  --jobflow-role EMR_EC2_DefaultRole \
  --subnet-id subnet-01ca3ae3325cec025 \
  --ec2-key-name root \
  --script-s3 s3://darkhaloscope/phase4/code/spark_phase4_pipeline.py \
  --bootstrap-s3 s3://darkhaloscope/phase4/code/bootstrap_phase4_pipeline_install_deps.sh \
  --core-instance-count 20 \
  --spark-args "\
--output-s3 s3://darkhaloscope/phase4_pipeline \
--variant v3_color_relaxed \
--coadd-s3-cache-prefix s3://darkhaloscope/dr10/coadd_cache/ \
--coadd-base-url https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/coadd \
--bands g,r,z \
--skip-if-exists 1 \
"
```

---

## Files Generated

| File | Location |
|------|----------|
| Validation log | `results/phase4a_validation_2026-01-24.txt` |
| Stage config | `results/phase4a_stage_config_2026-01-24.json` |
| This report | `results/phase4a_validation_report_2026-01-24.md` |

---

## Conclusion

Phase 4a completed successfully with scientifically correct output:

- ✅ All required columns present including new lens model fields
- ✅ Proper holdout splits preserved from Phase 3
- ✅ Control fractions match configuration
- ✅ No O(n²) sampling bug
- ✅ Frozen randomness for reproducibility
- ✅ Correct flux units (nanomaggies with AB ZP=22.5)

**The data is ready for Phase 4b coadd caching.**


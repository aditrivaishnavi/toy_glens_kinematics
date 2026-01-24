# Phase 4a Validation Review Request

## Request for Independent Review

I have completed Phase 4a of my Dark Halo Scope project (gravitational lens detection in astronomical surveys). Phase 4a generates injection task manifests that will be used for:
1. Training a CNN lens detector (train tier)
2. Systematic completeness analysis (grid tier)  
3. Quick end-to-end testing (debug tier)

I need you to independently validate the Phase 4a output and confirm whether it's safe to proceed to Phase 4b (coadd caching from NERSC DR10).

---

## Project Context

### What is Dark Halo Scope?
A project to detect galaxy-galaxy strong gravitational lenses in the DESI Legacy Survey DR10 using machine learning. The pipeline:
- Phase 1-2: Defined LRG selection criteria
- Phase 3: Built parent catalog of ~20M LRGs with region-level train/val/test splits
- **Phase 4a** (current): Generate injection task manifests for lens simulation
- Phase 4b: Cache DR10 imaging coadds from NERSC
- Phase 4c: Generate injected lens images (synthetic training data)
- Phase 4d: Compute completeness summaries
- Phase 5: Train CNN lens detector

### Holdout Strategy
- Splits are at the **region level** (not object level) to prevent spatial leakage
- Train/Val/Test splits come from Phase 3 and are preserved through Phase 4

### Injection Approach
- Using **SIE (Singular Isothermal Ellipsoid)** lens model
- All randomness is **frozen in the manifest** for reproducibility
- Controls have `theta_e = 0` (no lensing) for false positive calibration
- Flux units: **nanomaggies** (Legacy Survey standard, AB ZP = 22.5)

---

## Files Attached for Review

1. **`phase4a_validation_2026-01-24.txt`** - Raw validation log from EMR Spark job
2. **`phase4a_stage_config_2026-01-24.json`** - Configuration used for this run
3. **`phase4a_validation_report_2026-01-24.md`** - Human-readable analysis report
4. **`spark_validate_phase4a.py`** - Validation script (for reference)
5. **`spark_phase4_pipeline.py`** - Main pipeline code (for reference, first ~200 lines)

---

## Validation Results Summary

### Raw Output from Validation Script

```
[validate] Reading manifests from 3 path(s): ['s3://...debug_stamp64_bandsgrz_gridgrid_small/', 
                                               's3://...grid_stamp64_bandsgrz_gridgrid_medium/', 
                                               's3://...train_stamp64_bandsgrz_gridgrid_small/']

[validate] Manifest columns: [42 columns present including lens_model, lens_e, lens_phi_rad, 
                              task_seed64, src_x_arcsec, etc.]

[validate] Key column null check: PASSED
[validate] Minimum row count check: PASSED
[validate] theta_e range: [0.0000, 1.2000] arcsec - PASSED
[validate] Control/non-control theta_e consistency: PASSED
[validate] Frozen randomness columns for non-controls: PASSED
[validate] shear_phi_rad range: PASSED
[validate] src color ranges: PASSED
[validate] Bricks manifest: 180,373 unique bricks
[validate] O(n_cfg^2) explosion check: PASSED
[validate] Lens parameter consistency: PASSED

============================================================
OK: Phase 4a manifests and bricks_manifest passed validation.
============================================================
```

### Row Counts by Tier/Split

| Tier | Train Split | Val Split | Test Split | Total |
|------|-------------|-----------|------------|-------|
| debug | 5,760 | 5,760 | 5,760 | 17,280 |
| grid | 330,480 | 345,600 | 344,880 | 1,020,960 |
| train | 2,768,516 | 4,198,484 | 3,681,570 | 10,648,570 |
| **TOTAL** | 3,104,756 | 4,549,844 | 4,032,210 | **11,686,810** |

### Control Fractions

| Tier | Split | Observed | Expected | Status |
|------|-------|----------|----------|--------|
| debug | all | 0.0% | 0.0% | ✅ |
| grid | train | 8.9% | 10.0% | ✅ |
| grid | val | 11.9% | 10.0% | ✅ |
| grid | test | 10.9% | 10.0% | ✅ |
| train | train | 49.9% | 50.0% | ✅ |
| train | val | 49.8% | 50.0% | ✅ |
| train | test | 49.9% | 50.0% | ✅ |

**Note**: The validation script showed WARN for train tier because it was checking against 25% (old default), but stage_config shows the actual configured value was 50%, which matches observed.

### Configuration Used

```json
{
  "stage": "4a",
  "variant": "v3_color_relaxed",
  "tiers": {
    "debug": { "n_per_config": 5, "control_frac": 0.0 },
    "grid": { "n_per_config": 40, "control_frac": 0.1 },
    "train": { "n_total_per_split": 200000, "control_frac": 0.5 }
  },
  "stamp_sizes": [64],
  "bandsets": ["grz"],
  "replicates": 2
}
```

---

## Questions for Your Review

### 1. Data Quality
- Are there any concerns about the null checks passing?
- Is the theta_e range [0, 1.2] arcsec appropriate for galaxy-galaxy lenses?
- Are 180,373 unique bricks a reasonable coverage?

### 2. Sample Size Assessment
- Is 2.77M training samples (train split) sufficient for training a ResNet-18/EfficientNet-B0 lens detector?
- Is 1M grid samples sufficient for systematic completeness analysis?
- Is 50% control fraction appropriate for binary classification?

### 3. Split Imbalance
- The splits are imbalanced (Val > Test > Train for train tier)
- This comes from Phase 3 region-level splits where large regions hashed to test
- Is this a problem? Does it prevent rigorous science?

### 4. Schema Completeness
- All 42 columns are present including:
  - `lens_model`, `lens_e`, `lens_phi_rad` (SIE parameters)
  - `task_seed64`, `src_x_arcsec`, `src_y_arcsec`, `src_phi_rad`, `shear_phi_rad` (frozen randomness)
  - `ab_zp_nmgy = 22.5`, `pipeline_version` (provenance)
- Is this schema sufficient for reproducible injections?

### 5. Scientific Correctness
- The O(n²) explosion check passed - does the validation logic look correct?
- Lens parameter consistency check passed - are the validation criteria appropriate?
- Controls have `lens_model = "CONTROL"`, `theta_e = 0`, `lens_e = 0` - is this correct?

### 6. Reproducibility
- All randomness is frozen via `task_seed64` derived from `xxhash64`
- Source positions, angles, colors are pre-computed in manifest
- Will Stage 4c produce identical outputs across reruns?

### 7. Ready for Phase 4b?
Based on all the above, should I proceed to Phase 4b (coadd caching)?

What are the risks if I proceed now?

---

## Specific Concerns to Address

### Concern A: No `_stage_config.json` in S3
The validation log shows a warning that `_stage_config.json` was not found at the expected S3 path. However:
- The config was saved locally (`phase4a_stage_config_2026-01-24.json`)
- The manifest data itself is valid
- Is this a blocker?

### Concern B: Train Split Size
The train split has only 2.77M rows while val has 4.2M. This is unusual (typically train > val). However:
- This is due to region-level splits from Phase 3
- We can still train effectively on 2.77M
- Is there any reason to rerun Phase 3 with different split ratios?

### Concern C: Grid Control Fraction Variance
Grid tier control fractions vary: 8.9%, 11.9%, 10.9% across splits. Is ±2% acceptable?

---

## Expected Phase 4b Command

If approved, I will run:

```bash
python3 emr/submit_phase4_pipeline_emr_cluster.py \
  --region us-east-2 --stage 4b \
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

This will cache coadd FITS files for 180,373 bricks (~100-200 GB estimated).

---

## Your Response Format

Please structure your response as:

1. **Overall Assessment**: GO / NO-GO / CONDITIONAL-GO
2. **Data Quality Findings**: Any issues with the validation results?
3. **Sample Size Verdict**: Sufficient for first-class research?
4. **Schema Completeness**: Missing anything for reproducible injections?
5. **Scientific Concerns**: Any physics or methodology issues?
6. **Recommendations**: What to watch for in 4b/4c/4d?
7. **Questions Back**: Anything unclear that I should clarify?

---

## Attached Files

Please review these files in order:
1. `phase4a_stage_config_2026-01-24.json` - Configuration (short)
2. `phase4a_validation_2026-01-24.txt` - Raw validation log
3. `phase4a_validation_report_2026-01-24.md` - Analysis report
4. `spark_validate_phase4a.py` - Validation script (optional, for code review)
5. `spark_phase4_pipeline.py` - Main pipeline (optional, for deep review)

Thank you for your independent review!


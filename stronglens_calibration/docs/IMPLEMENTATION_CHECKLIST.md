# Implementation Checklist - Strong Lens Calibration

**Created**: 2026-02-05  
**Last Updated**: 2026-02-07 (migrated to stronglens_calibration/)  
**Purpose**: Track every suggestion from the LLM conversation to ensure complete implementation  
**Status Legend**: ☐ PENDING | ⏳ IN PROGRESS | ✅ DONE | ❌ BLOCKED | ⚠️ NEEDS CLARIFICATION | 📋 DECISION MADE

**Project Directory**: `stronglens_calibration/`

---

## KEY DECISIONS FROM LLM (Summary)

| Topic | Decision | Source |
|-------|----------|--------|
| N1:N2 ratio | 85:15 within negatives | Section A |
| Controls | Deprecate as primary negative, keep as diagnostic only | Section A |
| HEALPix nside | 128 | Section B |
| Split stratification | Don't hard-stratify by PSF/depth, just verify balance | Section B |
| Label handling | Sample weights in loss (not label smoothing alone) | Section C |
| Tier-A weight | 0.9-1.0 | Section C |
| Tier-B weight | 0.3-0.6 | Section C |
| SLACS/BELLS | Evaluation/stress-test only if arc SNR below threshold | Section C |
| Cutout size | 101×101, center-crop to 64 if needed | Section F |
| Primary architecture | ResNet18 first | Section G |
| Epochs | 20-40 with cosine schedule | Section G |
| Batch size | 256 (64×64) or 128-192 (101×101) with AMP | Section G |
| θE grid | 0.5-3.0" in 0.25" steps (11 bins) | Section H |
| PSF grid | 0.9-1.8" in 0.15" steps (7 bins) | Section H |
| Depth grid | 22.5-24.5 in 0.5 mag steps (5 bins) | Section H |
| Injections per cell | Minimum 200 | Section H |
| Uncertainty method | Bayesian binomial, 68% intervals | Section H |
| Milestone approach | Parallel: quick baseline + proper EMR negatives | Section I |
| Inner images | Include but calibrate + ablation with suppressed | Section E |

---

## PHASE 0: Pre-Implementation Validation

| ID | Task | Status | File/Location | Notes |
|----|------|--------|---------------|-------|
| P0.1 | Download DESI DR1 spectroscopic catalogs (single-fiber, pairwise) | ✅ VERIFIED | `stronglens_calibration/data/external/desi_dr1/desi-sl-vac-v1.fits` | 2,176 entries, 1,603 high-quality. Migrated to stronglens_calibration 2026-02-07 |
| P0.1b | Download DESI imaging candidates from lenscat | ✅ VERIFIED | `stronglens_calibration/data/positives/desi_candidates.csv` | 5,104 entries (435 confident, 4,669 probable). Migrated 2026-02-07 |
| P0.2 | Crossmatch spectroscopic catalogs against imaging candidates | ✅ VERIFIED | `stronglens_calibration/scripts/analyze_desi_sl_catalog.py` | **822 matches (16.1%)**, ~781 independent for validation. Migrated 2026-02-07 |
| P0.3 | Verify Paper IV's 1,372 confirmed lens list availability | 📋 DECISION MADE | | LLM: Cannot reconstruct exactly, match broad categories instead |
| P0.4 | Define inner image handling strategy | 📋 DECISION MADE | | LLM: Include but calibrate visibility + run suppressed ablation |
| P0.5 | Obtain/verify Tier-A anchor catalog completeness | ✅ VERIFIED | `stronglens_calibration/data/positives/desi_candidates.csv` | **435 confident** from lenscat (grading="confident"). Migrated 2026-02-07 |
| P0.6 | Verify contaminant catalog categories are complete | ☐ NOT DONE | `stronglens_calibration/data/contaminants/` | Directory needs creation. Use two-step: Tractor mining + image filtering |

---

## PHASE 1: EMR Job for Negative Sampling

### 1A. Negative Pool Design
| ID | Task | Status | File/Location | Notes |
|----|------|--------|---------------|-------|
| 1A.1 | Implement Pool N1: Deployment-representative sampling | ⏳ PARTIAL | `stronglens_calibration/emr/spark_negative_sampling.py` | Pool catalog exists (114M rows), but stratified sampling to ~510K NOT done. Need 100:1 per stratum. |
| 1A.2 | Implement Pool N2: Hard confuser sampling | ❌ NOT WORKING | `stronglens_calibration/emr/sampling_utils.py` | classify_pool_n2() exists but produced 0 N2 entries. Thresholds broken or too restrictive. Audit 2026-02-08: N1=100%, N2=0%. |
| 1A.3 | Define N1:N2 ratio in training | 📋 DECISION MADE | `configs/negative_sampling_v1.yaml` | **85:15** within negative class |
| 1A.4 | Implement 100:1 negative:positive ratio per (nobs_z, type) bin | ⏳ IN PROGRESS | | Config ready, stratified sampling TBD |
| 1A.5 | Define control sample handling | 📋 DECISION MADE | | **Deprecate as primary negative**, keep as diagnostic only |
| 1A.6 | Implement galaxy de-duplication across sweep files | ⏳ IN PROGRESS | | Via galaxy_id uniqueness check in validation |
| 1A.7 | Implement exclusion of known/candidate lenses from negatives | ✅ DONE | `emr/sampling_utils.py` | is_near_known_lens() with 11" radius. Implemented 2026-02-07 |

### 1B. Spatial Splits
| ID | Task | Status | File/Location | Notes |
|----|------|--------|---------------|-------|
| 1B.1 | Integrate HEALPix splitting (replace hash-based) | ✅ DONE | `emr/sampling_utils.py` | compute_healpix() + assign_split(). Implemented 2026-02-07 |
| 1B.2 | Choose nside | 📋 DECISION MADE | `configs/negative_sampling_v1.yaml` | **nside=128** for safer "independent regions" claims |
| 1B.3 | Stratify HEALPix cells by observing conditions | 📋 DECISION MADE | | **Don't hard-stratify**, just verify and report balance |
| 1B.4 | Implement 70/15/15 train/val/test allocation | ✅ DONE | `emr/sampling_utils.py` | assign_split() with deterministic hash. Implemented 2026-02-07 |
| 1B.5 | Verify spatial disjointness (no healpix cell in multiple splits) | ⏳ IN PROGRESS | | Built into assign_split() determinism, needs unit test |

### 1C. Schema Implementation
| ID | Task | Status | File/Location | Notes |
|----|------|--------|---------------|-------|
| 1C.1 | Level 1: Galaxy Catalog Manifest - core columns | ✅ DONE | `emr/spark_negative_sampling.py` | galaxy_id, brickname, ra, dec, type. Implemented 2026-02-07 |
| 1C.2 | Level 1: Stratification columns | ✅ DONE | `emr/spark_negative_sampling.py` | nobs_z, nobs_z_bin, type_bin. Implemented 2026-02-07 |
| 1C.3 | Level 1: Photometry columns | ✅ DONE | `emr/spark_negative_sampling.py` | flux_g/r/z/w1, mag_g/r/z, colors. Implemented 2026-02-07 |
| 1C.4 | Level 1: Observing condition columns | ✅ DONE | `emr/spark_negative_sampling.py` | psfsize_g/r/z, psfdepth_g/r/z, galdepth_g/r/z, ebv. Implemented 2026-02-07 |
| 1C.5 | Level 1: Quality flag columns | ✅ DONE | `emr/spark_negative_sampling.py` | maskbits, fitbits, mw_transmission. Implemented 2026-02-07 |
| 1C.6 | Level 1: Spatial/split columns | ✅ DONE | `emr/spark_negative_sampling.py` | healpix_64, healpix_128, split. Implemented 2026-02-07 |
| 1C.7 | Level 1: Provenance columns | ✅ DONE | `emr/spark_negative_sampling.py` | sweep_file, row_index, pipeline_version, git_commit, extraction_timestamp. Implemented 2026-02-07 |
| 1C.8 | Level 2: Cutout core columns | ☐ PENDING | | galaxy_id, stamp_npz, stamp_size, bandset, cutout_ok |
| 1C.9 | Level 2: Quality metrics | ☐ PENDING | | has_nan, nan_pixel_count, bad_pixel_frac, wise_brightmask_frac |
| 1C.10 | Level 2: Per-cutout conditions | ☐ PENDING | | psf_fwhm_g/r/z at center, psfdepth_r_center |
| 1C.11 | Level 2: Shortcut detection metrics | ☐ PENDING | | core_brightness_r, outer_brightness_r, mad_r, median_r, clip_frac_r |
| 1C.12 | Level 2: Normalization stats | ☐ PENDING | | mean_r, std_r, percentile_1_r, percentile_99_r |
| 1C.13 | Level 2: Split assignment | ☐ PENDING | | split_assignment based on HEALPix |
| 1C.14 | Add injection metadata columns (for selection function) | ☐ PENDING | | source_mag_g/r/z, theta_e, q, shear, magnification, arc_snr |
| 1C.15 | Add masking/contamination descriptors | ☐ PENDING | | maskbit_frac_above_threshold, bright_star_dist, gaia_mag_nearest |
| 1C.16 | Add cutout provenance | ☐ PENDING | | cutout_url, download_timestamp, layer_version |

### 1D. Quality Gates
| ID | Task | Status | File/Location | Notes |
|----|------|--------|---------------|-------|
| 1D.1 | Implement NaN detection per cutout | ☐ PENDING | | `has_nan`, `nan_pixel_count`; exclude if NaN in central region |
| 1D.2 | Implement bad_pixel_frac calculation | ☐ PENDING | | |
| 1D.3 | Implement WISE bright star mask fraction | ☐ PENDING | | |
| 1D.4 | Implement shortcut detection metrics | ☐ PENDING | | `core_brightness_r`, `outer_brightness_r`, `mad_r` |
| 1D.5 | Implement per-partition QA summaries | ☐ PENDING | | arc SNR histogram, SB histogram - fail fast |
| 1D.6 | Define maskbit threshold for galaxy exclusion | 📋 DECISION MADE | | Exclude if bright star/saturation/bleeding in center (r<8px) OR masked_frac > 1-2%; track maskbits as covariate |

### 1E. EMR Job Stability
| ID | Task | Status | File/Location | Notes |
|----|------|--------|---------------|-------|
| 1E.1 | Implement deterministic seeding per object ID | ☐ PENDING | | For reproducibility |
| 1E.2 | Implement checkpointed output shards with manifest | ☐ PENDING | | Resume logic |
| 1E.3 | Implement idempotent processing (skip-if-exists) | ☐ PENDING | | |
| 1E.4 | Define Spark partitioning strategy | ☐ PENDING | | By brickname for cache locality |

---

## PHASE 2: Label Handling

| ID | Task | Status | File/Location | Notes |
|----|------|--------|---------------|-------|
| 2.1 | Implement tier-based weighting | 📋 DECISION MADE | `planc/training/dataset.py` | **Sample weights in loss** (primary), not label smoothing alone |
| 2.2 | Define weight values | 📋 DECISION MADE | | **Literature confirmed: 1.0, Tier-A: 0.9-1.0, Tier-B: 0.3-0.6** + optional label smoothing (1.0→0.95) for Tier-B |
| 2.3 | Add sample weights to loss function | ☐ PENDING | `planc/training/losses.py` | Use weighted BCE |
| 2.4 | Separate training positives from evaluation anchors | ☐ PENDING | | **Explicit ID exclusion + spatial split** |
| 2.5 | Track provenance of each positive (source catalog) | ☐ PENDING | | SLACS/BELLS/SL2S/DESI/etc |
| 2.6 | Handle SLACS/BELLS anchors with low DR10 visibility | 📋 DECISION MADE | | **Don't train on them if arc SNR below threshold**; use for evaluation/stress-test only |
| 2.7 | Define which candidates go in training vs held-out | ☐ PENDING | | Use arc-visibility selection function to filter |

---

## PHASE 3: Injection Realism (Phase 4c Calibration)

### 3A. Photometric Realism
| ID | Task | Status | File/Location | Notes |
|----|------|--------|---------------|-------|
| 3A.1 | Verify flux uses DR10 zeropoints (22.5) | ✅ DONE | `spark_phase4_pipeline_gen5.py` line 443 | Already correct |
| 3A.2 | Define source magnitude prior distribution | 📋 DECISION MADE | | **r-band 22-26 unlensed, μ=5-30, target annulus SNR 0-5 with tail to ~10** |
| 3A.3 | Sample source magnitudes from realistic prior | ☐ PENDING | | Enforce SNR distribution matches real |
| 3A.4 | Apply magnification with surface brightness conservation | ☐ PENDING | | Verify existing code |
| 3A.5 | Match arc annulus SNR distribution to real anchors | ☐ PENDING | | **Median, 10th, 90th percentile within 0.5× to 2×** |
| 3A.6 | Match color distribution (g-r, r-z) | ☐ PENDING | | **Median within ±0.2 mag** |

### 3B. PSF and Noise Realism
| ID | Task | Status | File/Location | Notes |
|----|------|--------|---------------|-------|
| 3B.1 | Use per-cutout PSF (band-specific) | ☐ PENDING | | Check existing implementation |
| 3B.2 | Add noise using measured background (MAD-based) | ☐ PENDING | | Outer annulus after source masking via sigma-clipping |
| 3B.3 | Verify noise histogram via KS test | ☐ PENDING | | **p > 0.05 for each band** |
| 3B.4 | Define PSF model | 📋 DECISION MADE | | Moffat; **bracket beta (2.5, 3.5, 4.5)** or per-cutout if available |

### 3C. Morphology Realism
| ID | Task | Status | File/Location | Notes |
|----|------|--------|---------------|-------|
| 3C.1 | Define source morphology model | 📋 DECISION MADE | | **Bracket with: (1) smooth Sersic, (2) clumpy**; COSMOS optional third |
| 3C.2 | Bracket with alternative morphologies | ☐ PENDING | | Sensitivity test required |

### 3D. Acceptance Diagnostics (GO/NO-GO)
| ID | Task | Status | File/Location | Notes |
|----|------|--------|---------------|-------|
| 3D.1 | Build arc annulus SNR histogram comparison | ☐ PENDING | | Real vs injected |
| 3D.2 | Define SNR acceptance threshold | 📋 DECISION MADE | | **Median, 10th, 90th percentile within 0.5× to 2×** |
| 3D.3 | Build color distribution comparison (g-r, r-z) | ☐ PENDING | | |
| 3D.4 | Define color acceptance threshold | 📋 DECISION MADE | | **Median within ±0.2 mag** |
| 3D.5 | Build noise histogram KS test | ☐ PENDING | | |
| 3D.6 | Define KS p-value threshold | 📋 DECISION MADE | | **p > 0.05** |
| 3D.7 | Build "injection realism report" | ☐ PENDING | | Paper appendix |
| 3D.8 | Define GO/NO-GO rule | 📋 DECISION MADE | | **All must pass**; also visual sanity panel (injections shouldn't look systematically cleaner) |
| 3D.9 | Build prior sensitivity analysis | ☐ PENDING | | 2-3 alternative priors |

---

## PHASE 4: Training

| ID | Task | Status | File/Location | Notes |
|----|------|--------|---------------|-------|
| 4.1 | Choose primary architecture | 📋 DECISION MADE | | **ResNet18 first**; add EfficientNet-B0 only if time allows and shows gain |
| 4.2 | Define minimum epochs | 📋 DECISION MADE | | **20-40 epochs with cosine schedule**, early stopping on spatial-val set |
| 4.3 | Define batch size | 📋 DECISION MADE | | **256 for 64×64**, **128-192 for 101×101**, use AMP |
| 4.4 | Train baseline with clean splits | ☐ PENDING | | |
| 4.5 | Implement calibration curves by stratum | ☐ PENDING | | ECE/RC curves |
| 4.6 | Evaluate on independent validation set (spectroscopic) | ☐ PENDING | | Domain shift analysis |
| 4.7 | Run annulus-only classifier (should be strong) | ☐ PENDING | | Good signal check |
| 4.8 | Run core-only classifier (should be weak) | ☐ PENDING | | No shortcut check |
| 4.9 | Freeze model before selection function work | ☐ PENDING | | No hyperparameter tuning on injections |
| 4.10 | Define safe augmentations | 📋 DECISION MADE | | **Safe**: rotation/flip, small translate, mild noise, mild PSF blur. **Risky**: aggressive brightness/contrast |

---

## PHASE 5: Selection Function

| ID | Task | Status | File/Location | Notes |
|----|------|--------|---------------|-------|
| 5.1 | Define injection-recovery grid axes | 📋 DECISION MADE | | **θE: 0.5-3.0" in 0.25" steps (11 bins), PSF: 0.9-1.8" in 0.15" steps (7 bins), depth: 22.5-24.5 in 0.5 mag steps (5 bins)** = 385 cells |
| 5.2 | Define minimum injection points per grid cell | 📋 DECISION MADE | | **Minimum 200 per cell** (~77,000 total minimum) |
| 5.3 | Run injections across grid | ☐ PENDING | | Stratified by DR10 conditions |
| 5.4 | Score injections with frozen detector | ☐ PENDING | | |
| 5.5 | Compute completeness surfaces with uncertainty | ☐ PENDING | | **Bayesian binomial intervals, 68%** (optionally 95% in appendix) |
| 5.6 | Handle low-N bins | 📋 DECISION MADE | | **Don't smooth unless justified; mark insufficient below Nmin or merge adjacent** |
| 5.7 | Produce lookup table artifact | ☐ PENDING | | Public release format |
| 5.8 | Run robustness check with different nobs_z binning | ☐ PENDING | | Show stability |
| 5.9 | Optional: add host type as 4th axis | ☐ PENDING | | Tractor TYPE bins if resources allow |

---

## PHASE 6: Failure Mode Analysis

| ID | Task | Status | File/Location | Notes |
|----|------|--------|---------------|-------|
| 6.1 | Score contaminant categories | ☐ PENDING | | FPR by type |
| 6.2 | Stratify FPR by conditions (PSF, depth, nobs) | ☐ PENDING | | |
| 6.3 | Identify completeness collapse bins | ☐ PENDING | | |
| 6.4 | Build failure mode gallery | ☐ PENDING | | Representative examples |
| 6.5 | Test causal hypotheses | ☐ PENDING | | PSF-blend, galaxy subtraction, etc |
| 6.6 | Run at least one "fix" experiment | ☐ PENDING | | Show selection function shift |

---

## PHASE 7: Paper Deliverables

| ID | Task | Status | File/Location | Notes |
|----|------|--------|---------------|-------|
| 7.1 | Quantitative completeness map C(θ_E, PSF, depth) | ☐ PENDING | | Primary deliverable |
| 7.2 | Failure-mode taxonomy with measurable covariates | ☐ PENDING | | Not just gallery |
| 7.3 | Reproducible audit protocol (code + gates) | ☐ PENDING | | Public release |
| 7.4 | Injection realism appendix | ☐ PENDING | | Diagnostics and tolerances |
| 7.5 | Limitations section (honest) | ☐ PENDING | | Label noise, anchor coverage, circularity |
| 7.6 | Must-have figures | 📋 DECISION MADE | | **(1) Data/split schematic, (2) Score distributions by stratum, (3) Selection function heatmaps C(θE,PSF) at fixed depth, (4) Failure mode gallery with counts, (5) Independent validation table** |
| 7.7 | Claims to avoid | 📋 DECISION MADE | | **Avoid: overall precision in survey, cosmology constraints, "complete" lens sample, outperforming Huang without matched protocol** |
| 7.8 | Novelty statement | 📋 DECISION MADE | | "We provide a detector-audit framework for DR10 strong-lens searches, including injection-calibrated completeness surfaces and a condition- and confuser-resolved false-positive taxonomy, enabling bias-aware use of ML lens catalogs." |
| 7.9 | Journal choice | 📋 DECISION MADE | | MNRAS or ApJ; don't optimize for venue, optimize for correctness |

---

## CROSS-CHECK: Things to Verify Post-Implementation

| ID | Verification | Status | Notes |
|----|-------------|--------|-------|
| X.1 | Negatives include hard confusers (rings, spirals, mergers) | ☐ | |
| X.2 | Spatial splits are truly disjoint (HEALPix, not hash) | ☐ | |
| X.3 | Labels distinguish confirmed vs probable | ☐ | |
| X.4 | Training positives excluded from evaluation anchors | ☐ | |
| X.5 | Injection SNR distribution matches real anchors | ☐ | |
| X.6 | Independent validation set used (spectroscopic) | ☐ | |
| X.7 | Core-only classifier is weak (no shortcut) | ☐ | |
| X.8 | Uncertainty reported on completeness surfaces | ☐ | |
| X.9 | Insufficient-data bins clearly marked | ☐ | |
| X.10 | All code released with paper | ☐ | |
| X.11 | No training on evaluation anchors | ☐ | |
| X.12 | Reproducibility verified (deterministic seeds) | ☐ | |

---

## REVERSE CHECK: Code That Might Conflict with LLM Recommendations

| ID | Existing Code Pattern | LLM Recommendation | Action Needed | File/Line |
|----|----------------------|-------------------|---------------|-----------|
| R.1 | `is_control` single pool | Two pools N1+N2 | Refactor | `spark_phase4_pipeline_gen5.py:1636-1661` |
| R.2 | Hash-based split | HEALPix disjoint | Replace | `spark_phase3_pipeline.py:899-910` |
| R.3 | Uniform label smoothing (0.05) | Tier-based (0.95/0.7-0.8) | Modify | `planc/training/train_baseline.py` |
| R.4 | No training/eval separation | Explicit exclusion | Add logic | Multiple files |
| R.5 | 64×64 cutouts | Consider 101×101 | Decision needed | |
| R.6 | Inner images included by default | Explicit choice + ablation | Add ablation | |
| R.7 | No DESI spectroscopic catalogs | Independent validation | ✅ RESOLVED: Downloaded desi-sl-vac-v1.fits, cross-matched with 822 overlaps identified | |
| R.8 | No per-partition QA summaries | Fail-fast diagnostics | Add to EMR job | |

---

## CRITICAL GAP CHECK (From LLM Section L)

| ID | Issue | Status | Notes |
|----|-------|--------|-------|
| G.1 | "How will we demonstrate independence from prior ML candidate selection?" | ☐ PENDING | **Need explicit section and experiment for this** |
| G.2 | Highest-risk failure mode | 📋 IDENTIFIED | Injections that pass visual but fail quantitative distribution matching |
| G.3 | If scope must be cut | 📋 DECISION MADE | **CUT**: full ensemble meta-learner, full-survey inference. **KEEP**: real-image baseline, independent validation, validated injection-recovery, failure-mode taxonomy |

---

## MILESTONE SEQUENCING (From LLM Section I)

| ID | Task | Status | Notes |
|----|------|--------|-------|
| M.1 | Path A: Quick baseline with existing negatives | ☐ PENDING | To validate training code, splits, evaluation pipeline |
| M.2 | Path B: Build EMR negative sampling in parallel | ☐ PENDING | Switch to proper N1+N2 once ready |
| M.3 | Do NOT interpret quick-baseline as scientific | 📋 NOTED | |

---

## CUTOUT DECISIONS (From LLM Section F)

| ID | Decision | Status | Notes |
|----|----------|--------|-------|
| F.1 | Cutout size | 📋 DECISION MADE | **101×101**, center-crop to 64 if needed |
| F.2 | Regenerate positives for consistency | 📋 DECISION MADE | **Yes**, if changing canonical size |
| F.3 | Require all bands | 📋 DECISION MADE | **Require g,r,z present**; exclude missing-band for first pass |
| F.4 | NaN handling | 📋 DECISION MADE | **Exclude if NaN in central region**; track if only in outer rim |

---

## Questions Resolved (No Longer Need LLM Clarification)

All major implementation questions have been answered. Remaining work is execution.

---

## Update Log

| Date | Updates |
|------|---------|
| 2026-02-05 | Initial checklist created from LLM conversation analysis |
| 2026-02-05 | Updated with all decisions from LLM response (Sections A-L) |
| 2026-02-07 | Downloaded lenscat imaging candidates (5,104) to `data/positives/desi_candidates.csv` |
| 2026-02-07 | Downloaded DESI spectroscopic catalog to `data/external/desi_dr1/desi-sl-vac-v1.fits` |
| 2026-02-07 | Completed cross-match: 822 overlaps, ~781 truly independent spectroscopic entries |
| 2026-02-07 | **FULL AUDIT**: Verified P0.1, P0.1b, P0.2, P0.5 with file existence checks. P0.6 NOT DONE. |
| 2026-02-07 | **MIGRATION**: Moved all docs, data, and scripts from `planc/` to `stronglens_calibration/`. Updated all paths. |
| 2026-02-07 | **PHASE 1 IMPL**: Created `emr/spark_negative_sampling.py` with N1/N2 pools, HEALPix splits, exclusion radius, full schema. Config in `configs/negative_sampling_v1.yaml`. Unit tests passing. |
| 2026-02-07 | **LOCAL TESTS**: All 12 unit tests (1A-1E) passing. Local pipeline test passed with 5000 rows. 5 quality checks passed. Ready for EMR. |
| 2026-02-08 | **AUDIT CORRECTION**: N1:N2 ratio previously claimed as 70:30 was FALSE. Actual manifest audit shows N1=100%, N2=0%. classify_pool_n2() thresholds broken. Items 1A.1 and 1A.2 status corrected. |

|| 2026-02-07 | **EMR PLAN**: Created `docs/EMR_FULL_RUN_PLAN.md`, `scripts/preflight_check.py`, `scripts/validate_output.py`. Full runbook with dependencies, gates, and rollback. |

---

## EMR EXECUTION PLAN

### Upstream Dependencies
| ID | Dependency | Type | Status | Verification Command |
|----|------------|------|--------|---------------------|
| D.1 | DR10 Sweep Files | Data | ⚠️ TBD | `aws s3 ls s3://darkhaloscope/dr10/sweeps/` |
| D.2 | Positive Catalog (`desi_candidates.csv`) | Data | ✅ READY | 5,104 rows verified |
| D.3 | Spectroscopic Catalog (`desi-sl-vac-v1.fits`) | Data | ✅ READY | 2,176 rows verified |
| D.4 | Configuration (`negative_sampling_v1.yaml`) | Config | ✅ READY | YAML validated |
| D.5 | AWS Credentials | Infra | ⚠️ TBD | `aws sts get-caller-identity` |
| D.6 | S3 Bucket Access (`s3://darkhaloscope`) | Infra | ⚠️ TBD | `aws s3 ls s3://darkhaloscope/` |
| D.7 | EMR Permissions | Infra | ⚠️ TBD | `aws emr list-clusters --active` |
| D.8 | EMR Roles (EMR_DefaultRole, EMR_EC2_DefaultRole) | Infra | ⚠️ TBD | `aws iam get-role` |
| D.9 | EC2 Quota (≥280 vCPUs) | Infra | ⚠️ TBD | EC2 quota dashboard |

### Pre-Flight Checklist
| ID | Check | Script | Status |
|----|-------|--------|--------|
| PF.1 | Unit tests pass | `python3 tests/test_phase1_local.py` | ✅ PASS |
| PF.2 | Pipeline test passes | `python3 tests/test_pipeline_local.py` | ✅ PASS |
| PF.3 | Code syntax valid | `python3 -m py_compile emr/*.py` | ✅ PASS |
| PF.4 | Config valid | YAML load + critical field check | ✅ PASS |
| PF.5 | AWS credentials | `aws sts get-caller-identity` | ☐ PENDING |
| PF.6 | S3 access | `aws s3 ls s3://darkhaloscope/` | ☐ PENDING |
| PF.7 | Sweep files available | Check S3 or NERSC | ☐ PENDING |
| PF.8 | Full pre-flight | `python3 scripts/preflight_check.py` | ☐ PENDING |

### EMR Mini-Test
| ID | Step | Command | Status |
|----|------|---------|--------|
| MT.1 | Upload code/config to S3 | `launch_negative_sampling.py --test` | ☐ PENDING |
| MT.2 | Launch mini cluster (2 workers) | Auto via launcher | ☐ PENDING |
| MT.3 | Submit Spark step | Auto via launcher | ☐ PENDING |
| MT.4 | Monitor completion (~10 min) | Check EMR console | ☐ PENDING |
| MT.5 | Validate output | `python3 scripts/validate_output.py --s3 ...` | ☐ PENDING |
| MT.6 | Review pool/split distribution | Check validation report | ☐ PENDING |
| MT.7 | Terminate mini cluster | `--terminate` | ☐ PENDING |

### EMR Full Run
| ID | Step | Command | Status |
|----|------|---------|--------|
| FR.1 | Launch full cluster (25 workers) | `python3 emr/launch_negative_sampling.py --full` | ☐ PENDING |
| FR.2 | Monitor step execution (~4 hours) | Poll EMR status | ☐ PENDING |
| FR.3 | Verify output on S3 | `aws s3 ls ...` | ☐ PENDING |
| FR.4 | Download and validate output | `scripts/validate_output.py --s3 ...` | ☐ PENDING |
| FR.5 | Verify row count (~250K) | Validation report | ☐ PENDING |
| FR.6 | Verify pool ratio (85:15 ± 10%) | Validation report | ☐ PENDING |
| FR.7 | Verify split ratio (70:15:15 ± 5%) | Validation report | ☐ PENDING |
| FR.8 | Verify no duplicates | Validation report | ☐ PENDING |
| FR.9 | Terminate cluster | `--terminate` | ☐ PENDING |
| FR.10 | Archive logs and save report | JSON output | ☐ PENDING |

### Post-EMR Validation Gates
| Gate | Criteria | Critical? |
|------|----------|-----------|
| Row count | ≥ 200,000 | Yes |
| Null values | 0 in critical columns | Yes |
| Duplicates | 0 duplicate galaxy_ids | Yes |
| Pool N1:N2 | 75-95% N1 | No |
| Split train | 65-75% | No |
| All types | ≥4 galaxy types | No |
| Coordinates | RA [0,360], Dec [-90,90] | Yes |
| Provenance | ≥2/3 columns populated | No |

### Scripts Created for EMR
| File | Purpose |
|------|---------|
| `docs/EMR_FULL_RUN_PLAN.md` | Complete runbook with dependencies, timeline, cost |
| `scripts/preflight_check.py` | Automated pre-flight validation (local + AWS) |
| `scripts/validate_output.py` | Post-run output validation (local or S3) |
| `emr/launch_negative_sampling.py` | EMR cluster launch and step submission |


---

## STEP 1 CROSSMATCH STATUS (2026-02-08) - ✅ COMPLETE

### Final Results
| Metric | Value |
|--------|-------|
| Total positives | 5,104 |
| Matched | 4,788 (93.8%) |
| Unmatched | 316 (6.2%) - outside DR10 coverage |
| Within 1" | 4,639 (97% of matches) |
| Within 5" | 4,788 (100% of matches) |
| Median separation | 0.059" |
| Mean separation | 0.157" |
| Max separation | 4.71" |
| Gates passed | ✅ YES (>90% match rate) |

### Tier Distribution
| Tier | Count | Description |
|------|-------|-------------|
| A | 389 | Confident lenses |
| B | 4,399 | Probable lenses |

### Type Distribution (Tractor morphology)
| Type | Count | Description |
|------|-------|-------------|
| SER | 2,909 | Sersic profile |
| DEV | 911 | de Vaucouleurs |
| REX | 724 | Round exponential |
| EXP | 200 | Exponential |
| PSF | 35 | Point source |
| DUP | 9 | Duplicate |

### EMR Job Details
- **Cluster**: j-1QNR9QBL0SN4R (30x m5.2xlarge workers)
- **Runtime**: 7 minutes
- **Output**: `s3://darkhaloscope/stronglens_calibration/positives_with_dr10/20260208_180524/`
- **Checkpointing**: ✅ Enabled (resume-capable)

### Files Created
- `emr/spark_crossmatch_positives_v2.py` - Spark job with checkpointing
- `emr/launch_crossmatch_positives_v2.py` - EMR launcher (boto3-based)
- `scripts/crossmatch_positives_sweeps.py` - Local sweep crossmatch (deprecated)
- `scripts/crossmatch_positives_local.py` - Original manifest-based (deprecated)


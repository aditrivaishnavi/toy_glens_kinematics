# Codebase Index

Quick-reference map of every directory and key file in `stronglens_calibration/`.
Updated 2026-02-14. See also `EXPERIMENT_REGISTRY.md` for run-level provenance.

---

## 1. Core Training Pipeline (`dhs/`)

The main package. Imported as `from dhs.<module> import ...` with `PYTHONPATH=.`.

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker |
| `constants.py` | Shared constants: `CUTOUT_SIZE=101`, `STAMP_SIZE=64`, `PIXEL_SCALE=0.262` |
| `data.py` | `StrongLensDataset` (PyTorch Dataset), cutout loading, augmentation |
| `model.py` | `build_model()` -- EfficientNetV2-S, ResNet-18, BottleneckedResNet |
| `train.py` | Training loop, checkpointing, weighted loss, LR scheduling |
| `preprocess.py` | `preprocess_stack()` -- raw_robust annulus normalization, cropping |
| `preprocess_spec.py` | `PreprocessSpec` dataclass -- ties checkpoint to preprocessing params |
| `scoring_utils.py` | `load_model_and_spec()` -- loads model + preprocessing kwargs from checkpoint |
| `injection_engine.py` | SIS+shear ray-tracing injection, Sersic source sampling, noise estimation |
| `selection_function_utils.py` | `m5_from_psfdepth()`, Bayesian binomial intervals, grid utilities |
| `s3io.py` | S3/local I/O helpers (`read_bytes`, `write_bytes`, `is_s3_uri`) |
| `calibration.py` | ECE, MCE, reliability diagram |
| `gates.py` | Quality gate checks for training data |
| `transforms.py` | Data augmentation transforms |
| `utils.py` | Miscellaneous utilities (seeding, logging, device setup) |
| `scripts/` | Internal helper scripts (build manifests, run evaluation, run gates) |

---

## 2. Training Configs (`configs/`)

Sorted by generation. See `EXPERIMENT_REGISTRY.md` for full provenance table.

### EfficientNetV2-S (Main Paper IV Line)

| Config | Gen ID | Key Change | Trained? |
|--------|--------|------------|----------|
| `paperIV_efficientnet_v2_s.yaml` | gen1 | Baseline, step LR | Yes |
| `paperIV_efficientnet_v2_s_v2.yaml` | gen2 | Freeze backbone 5ep + warmup | Yes |
| `paperIV_efficientnet_v2_s_v3_cosine.yaml` | gen3 | Cosine LR schedule | Yes |
| `paperIV_efficientnet_v2_s_v4_finetune.yaml` | gen4 | **Current best.** Finetune from gen2 | Yes |
| `gen5a_efficientnet_annulus_fix.yaml` | gen5a | From-scratch, corrected annulus (32.5,45) | Not yet |
| `gen5b_efficientnet_annulus_ft.yaml` | gen5b | Finetune gen4, corrected annulus | Not yet |
| `gen5c_efficientnet_weighted_loss.yaml` | gen5c | Corrected annulus + weighted loss | Not yet |

### BottleneckedResNet

| Config | Key Change | Trained? |
|--------|------------|----------|
| `paperIV_bottlenecked_resnet.yaml` | ~195K params baseline | Yes |
| `paperIV_bottlenecked_resnet_v2_annulus_fix.yaml` | Corrected annulus | Not yet |

### Other / Early Experiments

| Config | Purpose |
|--------|---------|
| `paperIV_resnet18.yaml` | ResNet-18 baseline comparison |
| `resnet18_baseline_v1.yaml` | Early ResNet-18 baseline |
| `paired_baseline.yaml` | R0 paired experiment |
| `paired_core_dropout.yaml` | R1 core dropout |
| `unpaired_matched_raw.yaml` | R3 unpaired raw |
| `unpaired_matched_raw_hardneg.yaml` | R4 unpaired + hard negatives |
| `unpaired_matched_residual.yaml` | R5 unpaired residual |
| `unpaired_matched_residual_hardneg.yaml` | R6 unpaired residual + hard negatives |
| `negative_sampling_v1.yaml` | Negative sampling config |
| `injection_priors.yaml` | **Single source of truth** for injection parameter ranges |

---

## 3. Injection Models

### Injection Model 1 (`injection_model_1/`) -- Random Host Selection

Independent SIE injection on random negatives from the val split.
This is the baseline injection model for the selection function.

| File | Purpose |
|------|---------|
| `README.md` | Overview of Model 1 design and usage |
| `INJECTION_MODEL_1_VALIDATION.md` | Validation results and findings |
| `engine/injection_engine.py` | Model 1's copy of the injection engine |
| `engine/selection_function_utils.py` | Grid utilities for Model 1 |
| `scripts/selection_function_grid.py` | Selection function grid (uses `scoring_utils`) |
| `scripts/sensitivity_analysis.py` | Prior sensitivity sweeps |
| `scripts/sim_to_real_validation.py` | Sim-to-real gap analysis |
| `scripts/validate_injections.py` | Injection quality checks |
| `tests/test_injection_engine.py` | Unit tests for Model 1 engine |
| `validation/bright_arc_injection_test.py` | Bright arc detection test |
| `validation/confuser_morphology_test.py` | Confuser morphology analysis |
| `validation/real_lens_scoring.py` | Score real lenses with Model 1 |
| `validation/sim_to_real_validation.py` | Cross-validation between sim and real |
| `validation/README.md` | Validation suite overview |

### Injection Model 2 (`injection_model_2/`) -- LRG-Conditioned Host Selection

Deflector-conditioned injection: q_lens conditioned on q_host.
Found 0.77pp WORSE than Model 1 (see Section 24 of training log).

| File | Purpose |
|------|---------|
| `README.md` | Overview of Model 2 design, host matching, and results |
| `host_matching.py` | `estimate_host_moments_rband()`, `map_host_to_lens_params()` |
| `host_selection.py` | `select_lrg_hosts()`, `select_random_hosts()` |
| `scripts/selection_function_grid_v2.py` | **Reference impl** of checkpoint-driven preprocessing |
| `scripts/host_conditioning_diagnostic.py` | Diagnostic for host conditioning effects |
| `tests/test_host_matching.py` | Unit tests for host matching |

---

## 4. Diagnostic and Analysis Scripts (`scripts/`)

### Pre-Retrain Diagnostics (D01)

| Script | Purpose | Runs On |
|--------|---------|---------|
| `split_balance_diagnostic.py` | PSF/depth balance + Tier-A counts per split | CPU |
| `masked_pixel_diagnostic.py` | NaN/zero pixel fraction in cutouts | CPU |
| `annulus_comparison.py` | Compare (20,32) vs (32.5,45) annulus stats | CPU |
| `mismatched_annulus_scoring.py` | Score gen4 with wrong annulus | GPU |
| `feature_space_analysis.py` | UMAP + linear probe on embeddings | GPU |
| `beta_frac_ceiling_diagnostic.py` | Beta-frac restricted detection analysis | GPU |

### Morphology and Injection Validation

| Script | Purpose |
|--------|---------|
| `arc_morphology_statistics.py` | Pixel-level stats for real vs injected arcs |
| `real_arc_morphology_experiment.py` | Inject real arc residuals instead of Sersic |
| `validate_injections.py` | Visual + statistical injection quality checks |

### Evaluation and Analysis

| Script | Purpose |
|--------|---------|
| `evaluate_parity.py` | Paper IV parity metrics (AUC, ECE, TPR@FPR) |
| `selection_function_grid.py` | Main selection function grid runner |
| `sensitivity_analysis.py` | Prior sensitivity sweeps |
| `compare_models_common_cells.py` | Model 1 vs Model 2 head-to-head |
| `sim_to_real_validation.py` | Sim-to-real distribution comparison |
| `bootstrap_eval.py` | Bootstrap confidence intervals |
| `meta_learner.py` | Meta-learner for ensemble combination |
| `negative_cleaning_scorer.py` | Score-based negative cleaning |

### Data Preparation

| Script | Purpose |
|--------|---------|
| `generate_training_manifest.py` | Build training manifest from cutouts |
| `generate_training_manifest_parallel.py` | Parallel version of above |
| `make_parity_manifest.py` | Build 70/30 parity manifest |
| `crossmatch_positives_local.py` | Cross-match positives with local catalog |
| `crossmatch_positives_sweeps.py` | Cross-match against sweeps catalog |
| `extract_desi_candidates.py` | Extract DESI lens candidates |
| `analyze_desi_sl_catalog.py` | Analyze DESI strong lens catalog |
| `fix_manifest_splits.py` | Fix split assignments in manifest |

### Validation and Debugging

| Script | Purpose |
|--------|---------|
| `validate_constants.py` | Verify shared constants consistency |
| `validate_cutout_integrity.py` | Check cutout file integrity |
| `validate_output.py` | Validate training output files |
| `validate_stratified_output.py` | Validate stratified output |
| `verify_splits.py` | Verify train/val/test split correctness |
| `check_ranges.py` | Check parameter ranges |
| `debug_eval.py` | Debug evaluation issues |
| `debug_nan.py` | Debug NaN issues in training |
| `preflight_check.py` | Pre-training sanity checks |

### Setup and Deploy (Shell Scripts)

| Script | Purpose |
|--------|---------|
| `setup_local_venv.sh` | Create local venv (CPU-only torch, Python 3.11) |
| `setup_lambda3_venv.sh` | Create Lambda3 venv (CUDA torch) |
| `sync_to_lambda3.sh` | Rsync code from laptop to Lambda3 NFS |
| `run_diagnostics.sh` | Run all D01 diagnostics in sequence on Lambda3 |
| `sync_data_to_s3.sh` | Sync data to S3 |

---

## 5. Sim-to-Real Validations (`sim_to_real_validations/`)

Standalone validation suite that tests the model against real data.

| File | Purpose |
|------|---------|
| `README.md` | Suite overview |
| `bright_arc_injection_test.py` | Beta-frac restricted bright arc test (Q2.1) |
| `confuser_morphology_test.py` | False positive morphology analysis |
| `real_lens_scoring.py` | Score real Tier-A/Tier-B lenses (with leakage guard) |
| `sim_to_real_validation.py` | Distribution comparison: sim vs real scores |

---

## 6. Tests (`tests/`)

Run with: `source .venv/bin/activate && PYTHONPATH=. pytest tests/ -v`

| File | Purpose |
|------|---------|
| `test_band_order.py` | Verify g=0, r=1, z=2 band ordering throughout pipeline |
| `test_injection_priors.py` | Assert code defaults match `injection_priors.yaml` |
| `test_injection_engine.py` | Unit tests for injection engine (28 tests) |
| `test_preprocess_regression.py` | Regression tests for preprocessing |
| `test_integration_mini_pipeline.py` | End-to-end mini pipeline test |
| `test_phase1_local.py` | Phase 1 local validation tests |
| `test_pipeline_local.py` | Local pipeline integration tests |
| `test_n2_classification.py` | N2 classification threshold tests |
| `test_negative_sampling.py` | Negative sampling tests |
| `test_dr10_extraction.py` | DR10 cutout extraction tests |
| `test_astropy_api.py` | Astropy API compatibility tests |
| `run_guard_check.py` | Standalone guard check runner |

---

## 7. Documentation (`docs/`)

### MNRAS Paper Documentation

| File | Purpose |
|------|---------|
| `MNRAS_SUPPLEMENTARY_TRAINING_LOG.md` | **Master training log.** Sections 1-28 covering all experiments, metrics, LLM findings, and MNRAS readiness checklist |
| `MNRAS_INJECTION_VALIDATION.md` | Injection validation methodology and results |
| `MNRAS_RAW_NOTES.md` | Raw working notes for MNRAS paper |

### Plans and Strategies

| File | Purpose |
|------|---------|
| `RETRAIN_PLAN_ANNULUS_FIX.md` | Detailed plan for gen5a/b/c retrain with annulus correction |
| `PRE_TRAINING_PLAN.md` | Pre-training checklist and setup |
| `EMR_FULL_RUN_PLAN.md` | EMR cluster run plan |
| `NO_GO_REMEDIATION_RUNBOOK.md` | What to do if retrain fails |

### Audits and Reviews

| File | Purpose |
|------|---------|
| `LLM_AUDIT_ACTIONABLE_INVENTORY.md` | Prioritized inventory of all LLM-identified action items |
| `LLM_CODE_PACK_AUDIT.md` | Code pack audit findings |
| `LLM_BLUEPRINT_RESPONSE.md` | LLM implementation blueprint response |
| `LLM_RECOMMENDATIONS_EXTRACTED.md` | Extracted LLM recommendations |
| `EVALUATION_HONEST_AUDIT.md` | Honest audit of evaluation methodology |
| `EVALUATION_4.5_4.6_LLM_REVIEW.md` | LLM review of evaluation sections 4.5-4.6 |
| `AUDIT_VS_LLM_BLUEPRINT.md` | Comparison: our audit vs LLM blueprint |

### Technical References

| File | Purpose |
|------|---------|
| `DATA_AND_VERIFICATION.md` | Data sources, verification procedures |
| `DESI_CATALOG_ANALYSIS_REPORT.md` | DESI strong lens catalog analysis |
| `LAMBDA_TRAINING_PATHS.md` | Lambda3 NFS paths for checkpoints and data |
| `TECHNICAL_SPECIFICATIONS.md` | Architecture and preprocessing specs |
| `SPLIT_ASSIGNMENT.md` | Train/val/test split methodology |
| `SHORTCUT_DETECTION.md` | Shortcut learning detection methods |
| `MODELS_PERFORMANCE.md` | Model performance comparison table |

### Status and Lessons

| File | Purpose |
|------|---------|
| `PROJECT_STATUS.md` | Current project status |
| `IMPLEMENTATION_CHECKLIST.md` | Implementation progress checklist |
| `LESSONS_LEARNED.md` | Lessons learned from the project |
| `PAPER_IV_FULL_PARITY_COURSE_CORRECTION.md` | Course correction notes |
| `PHASE1_TEST_REPORT.md` | Phase 1 test results |

### Conversation Logs

| File | Purpose |
|------|---------|
| `conversation_with_llm.txt` | First LLM conversation (initial code audit) |
| `second_conversation_with_llm.txt` | Second LLM conversation (injection review) |

### Data Files (JSON)

| File | Purpose |
|------|---------|
| `bootstrap_eval_test.json` | Bootstrap evaluation test output |
| `eval_resnet18_baseline_v1_sanitized.json` | ResNet-18 baseline eval results |
| `fpr_by_confuser_category.json` | FPR breakdown by confuser type |
| `split_verification_report.json` | Split assignment verification |
| `LLM_UPDATE_20260209.md` | Update notes from 2026-02-09 |

---

## 8. LLM Review Package (`llm_review_package/`)

Prompts sent to external LLMs for independent code review, listed in order.

| File | Prompt | Purpose |
|------|--------|---------|
| `PROMPT_SEQUENCE_README.md` | -- | Overview of all 4 prompts and their intent |
| `PROMPT_1_CODE_AUDIT.md` | Prompt 1 | Full code audit: preprocessing, training, evaluation |
| `PROMPT_2_INJECTION_PHYSICS.md` | Prompt 2 | Injection physics: Sersic priors, noise, beta_frac |
| `PROMPT_3_RETRAIN_DECISION.md` | Prompt 3 | Retrain decision: annulus bug, go/no-go criteria |
| `PROMPT_4_ROADMAP_AND_CODE.md` | Prompt 4 | Roadmap + code review: final assessment |
| `LLM_PROMPT_TRAINING_AND_EVAL_REVIEW.md` | Early | Training and evaluation review (pre-prompt-sequence) |
| `LLM_PROMPT_INJECTION_PIPELINE_REVIEW.md` | Early | Injection pipeline review |
| `LLM_PROMPT_INJECTION_MODEL_1_INDEPENDENT_REVIEW.md` | Early | Model 1 independent review |
| `LLM_PROMPT_MODEL1_MODEL2_REVIEW.md` | Early | Model 1 vs Model 2 comparison review |
| `LLM_PROMPT_COMPREHENSIVE_REVIEW_20260213.md` | Early | Comprehensive codebase review |
| `LLM_PROMPT_CODE_REVIEW_AND_RETRAIN_ASSESSMENT_20260213.md` | Early | Code review + retrain assessment |
| `LLM_PROMPT_LITERATURE_REVIEW.md` | Early | Literature comparison review |
| `MANIFEST.md` | -- | Manifest schema documentation |
| `MANIFEST_SCHEMA_TRAINING_V1.md` | -- | Training manifest v1 schema |

---

## 9. Registry and Change Logs (Root Level)

| File | Purpose |
|------|---------|
| `EXPERIMENT_REGISTRY.md` | **Master registry.** Maps gen IDs, injection models, diagnostic runs, and NFS paths |
| `CODEBASE_INDEX.md` | **This file.** Directory and file navigation map |
| `CHANGES_FOR_LLM_REVIEW.md` | Changes made in preparation for LLM review |
| `CHANGES_SINCE_PROMPT_1.md` | Changes implemented after Prompt 1 findings |
| `CHANGES_SINCE_PROMPT_3.md` | Changes implemented after Prompt 3 findings |
| `README.md` | Project overview and quick start |
| `requirements.txt` | Python dependencies (torch 2.7.0, torchvision 0.22.0, etc.) |
| `constants.py` | Top-level shared constants |

---

## 10. NFS Storage Layout (Lambda3)

All data and checkpoints live on NFS at:
`/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/`

```
checkpoints/
  paperIV_efficientnet_v2_s/          # gen1
  paperIV_efficientnet_v2_s_v2/       # gen2
  paperIV_efficientnet_v2_s_v3_cosine/ # gen3
  paperIV_efficientnet_v2_s_v4_finetune/ # gen4 (current best)
  gen5a_efficientnet_annulus_fix/      # gen5a (pending)
  gen5b_efficientnet_annulus_ft/       # gen5b (pending)
  gen5c_efficientnet_weighted_loss/    # gen5c (pending)
manifests/
  training_parity_70_30_v1.parquet    # 70/30 train/val manifest
  training_v1.parquet                 # 70/15/15 train/val/test manifest
cutouts/
  positives/                          # Tier-A + Tier-B positive cutouts
  negatives/20260210_025117/          # Negative cutouts
results/
  D01_YYYYMMDD_pre_retrain_diagnostics/ # Pre-retrain diagnostic outputs
```

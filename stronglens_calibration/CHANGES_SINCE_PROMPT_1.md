# Changes Since Prompt 1 — Complete Inventory

**Date:** 2026-02-13
**Trigger:** Both LLM1 and LLM2 independently reviewed the codebase (Prompt 1 of 4).
This document lists ALL changes made in response to their findings, organized by
the question numbers from the audit.

---

## Tier 1: Bug Fixes (6 items)

### 1. CRITICAL: Scoring scripts now auto-load preprocessing from checkpoint (Q3.1)

**Both LLMs: CRITICAL / FAIL**

- **NEW FILE: `dhs/preprocess_spec.py`** — Frozen `PreprocessSpec` dataclass with mode, crop, crop_size, clip_range, annulus_r_in, annulus_r_out. Includes validation (both annulus radii set together, r_in < r_out) and `to_dict()`/`from_dict()` for serialization.
- **NEW FILE: `dhs/scoring_utils.py`** — `load_model_and_spec()` loads model + preprocessing kwargs from checkpoint. Falls back to `ckpt["dataset"]` for v1-v4 checkpoints that lack `preprocess_spec`.
- **MODIFIED: `dhs/train.py`** — Builds `PreprocessSpec` from `DatasetConfig` and saves it in every checkpoint under the `"preprocess_spec"` key.
- **MODIFIED: 4 scoring scripts** — All now use `load_model_and_spec()` to automatically match preprocessing to the model:
  - `injection_model_2/scripts/selection_function_grid_v2.py`
  - `sim_to_real_validations/real_lens_scoring.py`
  - `sim_to_real_validations/bright_arc_injection_test.py`
  - `sim_to_real_validations/confuser_morphology_test.py`

### 2. HIGH: r_in < r_out validation + minimum pixel count (Q1.6)

**Both LLMs: FAIL / NEW BUG**

- **MODIFIED: `dhs/preprocess.py`** — `preprocess_stack` now raises `ValueError` if:
  - Exactly one of `annulus_r_in`/`annulus_r_out` is provided (partial specification)
  - `annulus_r_in >= annulus_r_out`
- **MODIFIED: `dhs/utils.py`** — `normalize_outer_annulus` now raises `ValueError` if:
  - `r_in >= r_out`
  - Annulus has fewer than 100 pixels

### 3. HIGH: v5 config missing freeze/warmup (Q1.22)

**LLM1: FAIL. LLM2: HIGH / NEW BUG**

- **MODIFIED: `configs/paperIV_efficientnet_v2_s_v5_annulus_fix.yaml`** — Added `freeze_backbone_epochs: 5` and `warmup_epochs: 5` to match the v2 recipe that produced best results.
- **Verified: `configs/paperIV_bottlenecked_resnet_v2_annulus_fix.yaml`** — Already had explicit `freeze_backbone_epochs: 0` and `warmup_epochs: 0` (correct for non-pretrained).
- **NEW FILE: `configs/paperIV_efficientnet_v2_s_v5ft_annulus_fix.yaml`** — Finetune from v4 best checkpoint with corrected annulus. Both LLMs recommended trying this as faster diagnostic before from-scratch (Q1.21/Q1.23).

### 4. MEDIUM: Logger warning floods (Q1.4)

**Both LLMs: FAIL / NEW BUG**

- **MODIFIED: `dhs/utils.py`** — Added module-level `_warned_annulus` flag. The annulus-mismatch warning now fires at most once per process, preventing ~152M warnings during training.

### 5. MEDIUM: Clumps params hardcoded, YAML values ignored (Q1.16)

**LLM1: FAIL / NEW BUG. LLM2: FAIL**

- **MODIFIED: `dhs/injection_engine.py`** — `sample_source_params()` now accepts explicit `clumps_n_range: Tuple[int, int] = (1, 4)` and `clumps_frac_range: Tuple[float, float] = (0.15, 0.45)` parameters. The hardcoded values in the function body now use these parameters.
- **MODIFIED: `tests/test_injection_priors.py`** — Added `test_clumps_n_range` and `test_clumps_frac_range` to validate against `configs/injection_priors.yaml`.

### 6. LOW: AST parser silently skips complex defaults (Q1.15)

**Both LLMs: CONCERN**

- **MODIFIED: `tests/test_injection_priors.py`** — `_extract_function_defaults` now raises `ValueError` if `_ast_to_value` returns `_SENTINEL` for any parameter. Added `test_source_params_extracted_keys_match_expected` to assert the extracted keys match a known set, catching silent omissions.

---

## Tier 2: Test Improvements (4 items)

### 7. MEDIUM: test_preprocessing_outer_sky_near_zero now has assertions (Q1.12)

**LLM1: FAIL ("test is not a test")**

- **MODIFIED: `tests/test_preprocess_regression.py`** — Added assertions:
  - `sky_median < 0.0` (galaxy contamination shifts sky negative)
  - `sky_median > -5.0` (unexpected change detection)
  - TODO comment for post-retraining assertion (`abs(sky_median) < 0.5`)

### 8. MEDIUM: De Vaucouleurs test galaxies (Q1.2/Q1.13)

**Both LLMs: CONCERN**

- **MODIFIED: `tests/test_preprocess_regression.py`** — Added `test_annulus_contamination_devaucouleurs_profiles` with de Vaucouleurs (n=4) profiles at R_e = 4, 8, 12 px. Validates that corrected annulus has less contamination and < 15% galaxy flux for all sizes.

### 9. MEDIUM: Checksum test replaced with tolerance-based regression (Q1.14/Q3.4)

**Both LLMs: CONCERN (brittle across platforms)**

- **MODIFIED: `tests/test_preprocess_regression.py`** — Replaced `test_raw_robust_no_crop_checksum` (bitwise SHA-256) with `test_raw_robust_no_crop_regression` using `assertAlmostEqual` with `delta=0.1` on per-band means, stds, and center pixel values. Platform-robust.

### 10. MEDIUM: Beta_frac diagnostic outputs binned detection curve (Q1.20)

**Both LLMs: recommended**

- **MODIFIED: `scripts/beta_frac_ceiling_diagnostic.py`** — Part 2 (injection experiment) now computes and prints P(detected | beta_frac) in bins [0.0-0.2, 0.2-0.3, ..., 0.9-1.0]. Also saves `binned_detection` in JSON output.
- Added note about clip_range=50 test to probe clipping artifacts (Q5.6).

---

## Tier 3: Documentation Fixes (6 items)

### 11. LOW: Sentinel 0.0 documented (Q1.8)
- `dhs/data.py`: DatasetConfig docstring now explicitly documents sentinel behavior.

### 12. LOW: Blank-host rates labeled as upper bounds (Q1.18)
- `scripts/beta_frac_ceiling_diagnostic.py`: Added note explaining blank-host rates are strict upper bounds.

### 13. LOW: HWC/CHW format boundary documented (Q3.2)
- `dhs/injection_engine.py`: `inject_sis_shear` docstring now documents input HWC, output CHW format boundary.

### 14. LOW: Pretrained weight loss for in_ch != 3 (Q4.2)
- `dhs/model.py`: Comment added to `build_efficientnet_v2_s` explaining pretrained weights are discarded when `in_ch != 3`.

### 15. LOW: PA convention docstring fixed (Q4.5)
- `injection_model_2/host_matching.py`: Changed "E of N convention" to "angle from x-axis in pixel coordinates".

### 16. LOW: b_n approximation documented (Q4.7)
- `dhs/injection_engine.py`: Already had detailed docstring (Ciotti & Bertin 1999). Added inline reference to `_sersic_source_integral` at second usage site.

---

## Tier 4: Minor Code Smells (1 item)

### 17. LOW: row.get(None, 1.0) fixed (Q4.3)
- `dhs/data.py`: `_get_file_and_label` now uses explicit `if self.dcfg.sample_weight_col is not None` check before `row.get()`.

---

## New Files Created

| File | Purpose |
|------|---------|
| `dhs/preprocess_spec.py` | PreprocessSpec dataclass for checkpoint-embedded preprocessing config |
| `dhs/scoring_utils.py` | `load_model_and_spec()` for consistent model+preprocessing loading |
| `configs/paperIV_efficientnet_v2_s_v5ft_annulus_fix.yaml` | Finetune-from-v4 config with corrected annulus |

---

## Test Results

**46 tests, 46 passed, 0 failed** (2026-02-13, Python 3.11.12, macOS ARM)

---

## Items Explicitly NOT Fixed (deferred, with justification)

- **Q1.3**: Annulus formula heuristic (0.65R, 0.90R) — paper discussion, not code fix
- **Q3.6**: Manifest cutout checksums — infrastructure change, do after retraining decision
- **Q3.7**: n_failed per cell correlation — already reported in output
- **Q4.6**: nearest_bin boundary hosts — edge case, not a bug
- **Q5.6a**: Clipping saturation finding — documented in diagnostic script notes
- **Q1.19**: SIS vs SIE caustic structure — physics discussion for Prompt 2

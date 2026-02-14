# Retraining Plan: Corrected Normalization Annulus

**Date:** 2026-02-13
**Root cause:** LLM review finding #1 — hardcoded annulus (20, 32) is wrong for 101x101

## Background

The outer-annulus normalization uses `(r_in=20, r_out=32)`. This was tuned for
64x64 stamps (crop=True). When the pipeline moved to 101x101 (crop=False, Paper
IV parity), the annulus was never updated. For 101x101, the annulus sits at
40-63% of the image half-width — inside the galaxy, not the sky.

**Impact:** MAD is inflated (galaxy variance > sky noise), arc contrast is
suppressed, and the normalization center (median) is biased by galaxy light
instead of being near zero. This affects all models trained with crop=False.

The corrected annulus from `default_annulus_radii(101, 101) = (32.5, 45.0)` sits
at 65-90% of the half-width, where galaxy flux fraction is ~6% instead of ~20%.

## Current State

- `dhs/utils.py`: `normalize_outer_annulus` defaults to (20, 32) — locked to match
  all existing trained models.
- `dhs/preprocess.py`: `preprocess_stack` accepts optional `annulus_r_in` /
  `annulus_r_out` kwargs, forwarded to `normalize_outer_annulus`.
- `dhs/data.py`: `DatasetConfig` has `annulus_r_in` / `annulus_r_out` fields
  (default 0.0 = use hardcoded). These are read from YAML config and passed
  through to `preprocess_stack`.
- A loud runtime warning fires whenever `r_out/half < 0.70` on images > 64px.
- `default_annulus_radii()` function is available but NOT wired as default.

## Step-by-step Plan

### Step 1: Pre-flight validation (local)

1. Run `tests/test_preprocess_regression.py` — all 25 tests should pass (they
   do as of 2026-02-13).
2. Run `tests/test_injection_priors.py` — all 17 tests should pass.
3. Verify the checksum test still produces `7e25b9e366471bda` for the old annulus.

### Step 2: Retrain both architectures (Lambda GPU)

New configs are ready:

| Config | Architecture | Annulus | Output dir |
|--------|-------------|---------|------------|
| `configs/paperIV_efficientnet_v2_s_v5_annulus_fix.yaml` | EfficientNetV2-S | (32.5, 45.0) | `checkpoints/paperIV_efficientnet_v2_s_v5_annulus_fix/` |
| `configs/paperIV_bottlenecked_resnet_v2_annulus_fix.yaml` | BottleneckedResNet | (32.5, 45.0) | `checkpoints/paperIV_bottlenecked_resnet_v2_annulus_fix/` |

Launch commands:
```bash
# On Lambda GPU instance
cd /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/code/stronglens_calibration

# EfficientNetV2-S (~6-8 hours on GH200)
PYTHONPATH=. python3 -m dhs.train --config configs/paperIV_efficientnet_v2_s_v5_annulus_fix.yaml

# BottleneckedResNet (~3-4 hours on GH200)
PYTHONPATH=. python3 -m dhs.train --config configs/paperIV_bottlenecked_resnet_v2_annulus_fix.yaml
```

### Step 3: Validate retrained models

1. Check val AUC is comparable to or better than old models:
   - EfficientNetV2-S v1: best_auc = 0.9978
   - BottleneckedResNet v1: best_auc = 0.9943 (after fix for memorization)
2. Run scoring on the DESI catalog with the new checkpoint **using the new annulus**.
3. Compare real-lens recall at fixed FPR to old models.

### Step 4: Rerun selection function grid

The selection function grid (completeness vs theta_E, r_mag, etc.) must be
regenerated with:
- The new model checkpoint
- The new annulus in preprocessing (annulus_r_in=32.5, annulus_r_out=45.0)

This means the injection-recovery pipeline must also preprocess hosts with the
new annulus. Update `scripts/selection_function_grid.py` (and the Model 2
variant) to pass annulus kwargs.

### Step 5: Update checksum

After retraining, run the preprocessing test with the new annulus to get the
new checksum, and update `EXPECTED_CHECKSUM` in
`tests/test_preprocess_regression.py::test_raw_robust_no_crop_checksum`.

### Step 6: Paper update

Update the methods section:
- Normalization annulus: "outer-annulus at (r_in, r_out) = (32.5, 45.0) pixels
  for 101x101 stamps (65-90% of image half-width)"
- Note: "Earlier models used (20, 32) which was appropriate for 64x64 center-
  cropped stamps but placed the annulus within the galaxy for 101x101 stamps."

## Risks

1. **Val AUC may change:** The normalization change affects the contrast of all
   features. If the model was implicitly compensating for the bad normalization,
   AUC might initially drop. Watch training curves carefully.
2. **Selection function values will change:** Because preprocessing changed,
   completeness at each grid point will differ. This is expected and correct.
3. **Cannot mix old and new checkpoints:** Any scoring or selection-function run
   must use EITHER all-old-annulus or all-new-annulus. Never mix.

## Definition of Done

- [ ] Both architectures retrained with corrected annulus
- [ ] Val AUC >= 0.995 for EfficientNetV2-S, >= 0.990 for BottleneckedResNet
- [ ] Selection function grid regenerated with new models + new annulus
- [ ] Checksum test updated in `test_preprocess_regression.py`
- [ ] Paper methods section updated
- [ ] Old (20, 32) models clearly labeled as deprecated in checkpoint dir

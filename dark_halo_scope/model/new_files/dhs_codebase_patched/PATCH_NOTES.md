# Patch Notes (January 30, 2026)

## Summary
This patch set addresses review-identified issues and hard bugs in the provided codebase.

## Changes

### 1) spark_phase4_pipeline.py
- Added PSF convolution options:
  - New CLI args: --psf-model {gaussian,moffat}, --moffat-beta
  - Implemented Moffat and Gaussian PSF kernels and FFT convolution for 64x64 stamps
  - Updated render_lensed_source and render_unlensed_source to use PSF FWHM (pixels), not sigma
- Default --psf-model is now "moffat" to reduce sim-to-real PSF mismatch.
- Default --unpaired-controls changed to 1 (when relevant to pipeline usage).
- Default --control-frac-train changed to 0.50 to match the project spec.

### 2) spark_phase4a_build_manifest_sota.py
- Default --unpaired_control_frac changed to 0.50 (review recommendation).

### 3) phase5_train_fullscale_gh200_v2.py
- Fixed DataLoader multi-worker duplication:
  - Streaming shards now partition by (distributed rank, worker_id).
- Added a hard guard against label-leaking meta_cols (arc_snr, injection parameters).

### 4) phase5_eval_stratified_fpr.py
- Fixed syntax errors in strata_masks() yield statements.

## How to use
See RUNBOOK.md.

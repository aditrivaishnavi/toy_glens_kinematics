#!/usr/bin/env bash
# run_d04_matched_comparison.sh — D04: Matched apples-to-apples comparison grids
#
# After the D03 post-review reinvestigation, we discovered:
#   1. The Poisson noise clamp(min=1.0) bug inflated annulus MAD by ~2.5x
#   2. The old grid used different depth ranges (24.0-25.5) vs D03 (22.5-24.5)
#   3. The summary reporting code averaged across source_mag_bin types
#
# This script runs three matched experiments:
#   1. No-Poisson baseline (same grid params as D03)
#   2. Fixed-Poisson grid (torch.poisson, same params)
#   3. Fixed-Poisson + clip_range=20 bright-arc diagnostic
#
# All three use the DEFAULT grid parameters:
#   depth:  22.5, 23.0, 23.5, 24.0, 24.5  (5 bins)
#   PSF:    0.9, 1.05, 1.2, 1.35, 1.5, 1.65, 1.8  (7 bins)
#   theta_E: 0.5 to 2.5 step 0.2  (11 bins)
#   injections/cell: 500
#   thresholds: 0.3, 0.5, 0.7
#   FPR targets: 0.001, 0.0001
#
# Run:
#   cd /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration
#   source .venv-lambda3/bin/activate
#   export PYTHONPATH=.
#   nohup bash scripts/run_d04_matched_comparison.sh > d04_log.txt 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Configuration ---
MANIFEST="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/manifests/training_parity_70_30_v1.parquet"
CHECKPOINT="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt"
RESULTS_BASE="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/results"
RUN_ID="D04_$(date +%Y%m%d)_matched_comparison"
OUT_DIR="$RESULTS_BASE/$RUN_ID"

echo "==========================================="
echo " D04 Matched Comparison Grids (Poisson Fix)"
echo " Run ID: $RUN_ID"
echo " Output: $OUT_DIR"
echo "==========================================="
mkdir -p "$OUT_DIR"

# Common grid args (identical for baseline and Poisson)
GRID_ARGS=(
    --checkpoint "$CHECKPOINT"
    --manifest "$MANIFEST"
    --host-split val
    --host-max 20000
    --thresholds 0.3 0.5 0.7
    --fpr-targets 0.001 0.0001
    --injections-per-cell 500
    --seed 1337
)

# ============================================================
# [1/3] No-Poisson Baseline Grid (matched parameters)
# ============================================================
# This produces the matched baseline for apples-to-apples Poisson comparison.
# Uses the SAME grid as D03 but WITHOUT Poisson noise.
echo ""
echo "--- [1/3] No-Poisson Baseline Grid (~4 GPU-hours) ---"
echo "  Start: $(date)"
python -u "$PROJECT_DIR/scripts/selection_function_grid.py" \
    "${GRID_ARGS[@]}" \
    --out-dir "$OUT_DIR/grid_no_poisson"
echo "  Done: $(date)"

# ============================================================
# [2/3] Fixed-Poisson Grid
# ============================================================
# Uses torch.poisson() instead of the buggy Gaussian approx with clamp(min=1.0).
# Zero-flux pixels now get zero noise (previously got sqrt(1.0) noise per pixel).
echo ""
echo "--- [2/3] Fixed-Poisson Grid (~4 GPU-hours) ---"
echo "  Start: $(date)"
python -u "$PROJECT_DIR/scripts/selection_function_grid.py" \
    "${GRID_ARGS[@]}" \
    --add-poisson-noise \
    --out-dir "$OUT_DIR/grid_poisson_fixed"
echo "  Done: $(date)"

# ============================================================
# [3/3] Fixed-Poisson + clip_range=20 Bright-Arc Diagnostic
# ============================================================
# Tests whether Poisson noise and wider clip_range have additive effects
# on bright-arc detection, using the fixed Poisson implementation.
echo ""
echo "--- [3/3] Fixed-Poisson + clip_range=20 Bright-Arc Diagnostic (~10 min GPU) ---"
echo "  Start: $(date)"
python -u "$PROJECT_DIR/sim_to_real_validations/bright_arc_injection_test.py" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/poisson_fixed_clip20_combined" \
    --beta-frac-range 0.1 0.55 \
    --add-poisson-noise \
    --clip-range 20 \
    --seed 42
echo "  Done: $(date)"

echo ""
echo "==========================================="
echo " D04 Complete"
echo " Results: $OUT_DIR"
echo "==========================================="
echo "  - grid_no_poisson/:             Matched baseline (no Poisson)"
echo "  - grid_poisson_fixed/:          Fixed Poisson (torch.poisson)"
echo "  - poisson_fixed_clip20_combined/: Bright-arc test (Poisson + clip=20)"
echo ""
echo " Next: rsync results back to laptop, build comparison table and figure."

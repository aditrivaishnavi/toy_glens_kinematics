#!/usr/bin/env bash
# run_diagnostics.sh — Run all D01 pre-retrain diagnostics on Lambda3
# Expects to be run from the stronglens_calibration/ directory with venv active.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Configuration ---
MANIFEST="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/manifests/training_parity_70_30_v1.parquet"
CHECKPOINT="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt"
RESULTS_BASE="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/results"
RUN_ID="D01_$(date +%Y%m%d)_pre_retrain_diagnostics"
OUT_DIR="$RESULTS_BASE/$RUN_ID"

echo "==========================================="
echo " D01 Pre-Retrain Diagnostics"
echo " Run ID: $RUN_ID"
echo " Output: $OUT_DIR"
echo "==========================================="
mkdir -p "$OUT_DIR"

# --- CPU diagnostics (no GPU needed) ---

echo ""
echo "--- [1/6] Split Balance Diagnostic (CPU, ~10s) ---"
python "$PROJECT_DIR/scripts/split_balance_diagnostic.py" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/split_balance_check" \
    2>&1 | tee "$OUT_DIR/split_balance_check.log"

echo ""
echo "--- [2/6] Masked Pixel Diagnostic (CPU, ~1-5 min) ---"
python "$PROJECT_DIR/scripts/masked_pixel_diagnostic.py" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/masked_pixel_check" \
    --n-samples 1000 \
    --threshold 0.05 \
    2>&1 | tee "$OUT_DIR/masked_pixel_check.log"

echo ""
echo "--- [3/6] Annulus Comparison (CPU, ~1-5 min) ---"
python "$PROJECT_DIR/scripts/annulus_comparison.py" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/q23_annulus_comparison" \
    --n-samples 1000 \
    2>&1 | tee "$OUT_DIR/q23_annulus_comparison.log"

# --- GPU diagnostics (need CUDA) ---

echo ""
echo "--- [4/6] Mismatched Annulus Scoring (GPU, ~10-30 min) ---"
python "$PROJECT_DIR/scripts/mismatched_annulus_scoring.py" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/q24_mismatched_scoring" \
    --n-samples 500 \
    2>&1 | tee "$OUT_DIR/q24_mismatched_scoring.log"

echo ""
echo "--- [5/6] Beta-Frac Restriction Test (GPU, ~30-60 min) ---"
python "$PROJECT_DIR/sim_to_real_validations/bright_arc_injection_test.py" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/q21_beta_frac" \
    --beta-frac-range 0.1 0.55 \
    2>&1 | tee "$OUT_DIR/q21_beta_frac.log"

echo ""
echo "--- [6/6] Embedding UMAP + Linear Probe (GPU, ~20-40 min) ---"
python "$PROJECT_DIR/scripts/feature_space_analysis.py" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR/q22_embedding_umap" \
    --n-samples 200 \
    2>&1 | tee "$OUT_DIR/q22_embedding_umap.log"

echo ""
echo "==========================================="
echo " D01 Diagnostics Complete"
echo " Results in: $OUT_DIR"
echo "==========================================="
ls -la "$OUT_DIR"/

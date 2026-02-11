#!/usr/bin/env bash
# Sync local-only data to EMR-launcher /data (optional) and upload to S3.
# S3 is the source of truth for production and EMR jobs.
#
# Usage:
#   From local (where data exists): ./scripts/sync_data_to_s3.sh
#   Or rsync to emr-launcher first, then on emr-launcher: ./scripts/sync_data_to_s3.sh --from /data/stronglens_calibration
#
# Requires: aws CLI configured (region us-east-2 for darkhaloscope bucket).
# Optional: EMR_LAUNCHER_HOST in env for rsync step.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUCKET="${S3_BUCKET:-darkhaloscope}"
REGION="${AWS_REGION:-us-east-2}"
PREFIX="stronglens_calibration"

# If data exists only on your local machine:
#   rsync -avz --progress data/external/ data/positives/ data/negatives/ \
#     EMR_LAUNCHER_HOST:/data/stronglens_calibration/
# Then on emr-launcher: cd /data/stronglens_calibration && ./scripts/sync_data_to_s3.sh --from /data/stronglens_calibration
#
# Optional: sync from local to emr-launcher /data first
FROM_DIR="${1:-}"
if [[ "${1:-}" == "--from" ]]; then
  FROM_DIR="${2:-}"
  shift 2 || true
fi

if [[ -n "$FROM_DIR" ]]; then
  echo "Using source directory: $FROM_DIR"
  DATA_ROOT="$FROM_DIR"
else
  DATA_ROOT="$REPO_ROOT"
fi

echo "Bucket: s3://${BUCKET}/${PREFIX}/"
echo "Region: $REGION"
echo ""

# 1) DESI DR1 spectroscopic catalog (not in git; .gitignore data/external/*.fits)
DESI_FITS_LOCAL="$DATA_ROOT/data/external/desi_dr1/desi-sl-vac-v1.fits"
DESI_S3="s3://${BUCKET}/${PREFIX}/data/external/desi_dr1/desi-sl-vac-v1.fits"
if [[ -f "$DESI_FITS_LOCAL" ]]; then
  echo "Uploading DESI spectroscopic catalog to S3..."
  aws s3 cp "$DESI_FITS_LOCAL" "$DESI_S3" --region "$REGION"
  echo "  -> $DESI_S3"
else
  echo "Skip (not found): $DESI_FITS_LOCAL"
  echo "  To add: place FITS file there, or rsync from local to emr-launcher then re-run with --from /data/stronglens_calibration"
fi
echo ""

# 2) Positive catalog (in git; optional upload so EMR can read from S3)
POSITIVE_CSV_LOCAL="$DATA_ROOT/data/positives/desi_candidates.csv"
POSITIVE_S3="s3://${BUCKET}/${PREFIX}/data/positives/desi_candidates.csv"
if [[ -f "$POSITIVE_CSV_LOCAL" ]]; then
  echo "Uploading positive catalog to S3..."
  aws s3 cp "$POSITIVE_CSV_LOCAL" "$POSITIVE_S3" --region "$REGION"
  echo "  -> $POSITIVE_S3"
else
  echo "Skip (not found): $POSITIVE_CSV_LOCAL"
fi
echo ""

# 3) Negative prototype (for local tests only; large, gitignored). Optional upload for EMR-launcher runs.
NEGATIVE_CSV_LOCAL="$DATA_ROOT/data/negatives/negative_catalog_prototype.csv"
NEGATIVE_S3="s3://${BUCKET}/${PREFIX}/data/negatives/negative_catalog_prototype.csv"
if [[ -f "$NEGATIVE_CSV_LOCAL" ]]; then
  echo "Uploading negative prototype to S3 (optional, for EMR-launcher)..."
  aws s3 cp "$NEGATIVE_CSV_LOCAL" "$NEGATIVE_S3" --region "$REGION"
  echo "  -> $NEGATIVE_S3"
else
  echo "Skip (not found): $NEGATIVE_CSV_LOCAL"
fi
echo ""

echo "Done. S3 source-of-truth paths documented in docs/DATA_AND_VERIFICATION.md"

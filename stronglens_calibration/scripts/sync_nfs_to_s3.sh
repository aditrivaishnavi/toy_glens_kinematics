#!/usr/bin/env bash
# sync_nfs_to_s3.sh — List NFS and S3 once, diff, upload missing files.
# Goal: get everything off NFS into S3 so we can decommission NFS.
#
# Usage (run on lambda3):
#   ./scripts/sync_nfs_to_s3.sh                  # default: dry-run (shows what would upload)
#   ./scripts/sync_nfs_to_s3.sh --list-only       # just list + diff, no upload
#   ./scripts/sync_nfs_to_s3.sh --execute          # actually upload missing files
#   ./scripts/sync_nfs_to_s3.sh --execute --parallel 16   # upload with 16 parallel workers
#
# Cost: S3 LIST ~$0.003 total, PUT ~$0.005/1000 files, data-in free.

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
NFS_BASE="/lambda/nfs/darkhaloscope-training-dc"
S3_BUCKET="darkhaloscope"
S3_URI="s3://${S3_BUCKET}"
AWS_REGION="${AWS_REGION:-us-east-2}"
PARALLEL="${PARALLEL:-8}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="/tmp/nfs_s3_sync_${TIMESTAMP}"
LOG_FILE="${WORK_DIR}/sync.log"

# Intermediate files
NFS_SORTED="${WORK_DIR}/nfs_files_sorted.txt"
S3_RAW="${WORK_DIR}/s3_raw.txt"
S3_SORTED="${WORK_DIR}/s3_files_sorted.txt"
MISSING_FILE="${WORK_DIR}/nfs_only_missing.txt"
MISSING_SIZES="${WORK_DIR}/nfs_only_missing_with_sizes.txt"

# ── Parse arguments ────────────────────────────────────────────────────────
MODE="dry-run"    # dry-run | list-only | execute
while [[ $# -gt 0 ]]; do
    case "$1" in
        --execute)   MODE="execute"; shift ;;
        --list-only) MODE="list-only"; shift ;;
        --dry-run)   MODE="dry-run"; shift ;;
        --parallel)  PARALLEL="$2"; shift 2 ;;
        --help|-h)
            head -12 "$0" | tail -8
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ── Helpers ────────────────────────────────────────────────────────────────
log() {
    local msg
    msg="$(date '+%Y-%m-%d %H:%M:%S') | $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

human_size() {
    # Convert bytes to human-readable (uses awk, no bc dependency)
    local bytes=$1
    awk -v b="$bytes" 'BEGIN {
        if      (b >= 1073741824) printf "%.1f GB", b/1073741824
        else if (b >= 1048576)    printf "%.1f MB", b/1048576
        else if (b >= 1024)       printf "%.1f KB", b/1024
        else                      printf "%d B", b
    }'
}

# ── Setup ──────────────────────────────────────────────────────────────────
mkdir -p "$WORK_DIR"
touch "$LOG_FILE"

log "============================================================"
log "NFS → S3 sync script"
log "Mode:       $MODE"
log "NFS base:   $NFS_BASE"
log "S3 target:  $S3_URI"
log "Region:     $AWS_REGION"
log "Parallel:   $PARALLEL"
log "Work dir:   $WORK_DIR"
log "============================================================"

# Verify NFS is accessible
if [[ ! -d "$NFS_BASE" ]]; then
    log "ERROR: NFS base $NFS_BASE not found. Are you on lambda3?"
    exit 1
fi

# Verify AWS credentials work
if ! aws sts get-caller-identity --region "$AWS_REGION" >/dev/null 2>&1; then
    log "ERROR: AWS credentials not configured. Run 'aws configure' or check ~/.aws/credentials"
    exit 1
fi
log "AWS credentials verified."

# ══════════════════════════════════════════════════════════════════════════
# PHASE 1: List both sides (one pass each)
# ══════════════════════════════════════════════════════════════════════════
log ""
log "──── Phase 1: Listing files ────"

# ── 1a. List NFS ──
log "Listing NFS files under $NFS_BASE ..."
NFS_START=$(date +%s)

find "$NFS_BASE" -type f \
    -not -path '*/.venv*' \
    -not -path '*/__pycache__/*' \
    -not -path '*/.git/*' \
    -not -name '*.pyc' \
    -not -name '.DS_Store' \
    -not -name '*.swp' \
    -not -name '*.swo' \
    -not -name '*~' \
    -not -name '.nfs*' \
    -not -name '*.lock' \
    -not -name '*.tmp' \
    | sed "s|^${NFS_BASE}/||" \
    | sort \
    > "$NFS_SORTED"

NFS_COUNT=$(wc -l < "$NFS_SORTED")
NFS_END=$(date +%s)
log "NFS listing complete: ${NFS_COUNT} files in $((NFS_END - NFS_START))s"
log "  Saved to: $NFS_SORTED"

# ── 1b. List S3 ──
log "Listing S3 objects under $S3_URI ..."
S3_START=$(date +%s)

aws s3 ls --recursive "${S3_URI}/" --region "$AWS_REGION" \
    > "$S3_RAW"

# Extract keys (field 4+, handles filenames with spaces)
awk '{$1=$2=$3=""; print substr($0,4)}' "$S3_RAW" \
    | sed 's/^ *//' \
    | sort \
    > "$S3_SORTED"

S3_COUNT=$(wc -l < "$S3_SORTED")
S3_END=$(date +%s)
log "S3 listing complete: ${S3_COUNT} objects in $((S3_END - S3_START))s"
log "  Saved to: $S3_SORTED"

# ══════════════════════════════════════════════════════════════════════════
# PHASE 2: Diff
# ══════════════════════════════════════════════════════════════════════════
log ""
log "──── Phase 2: Computing diff ────"

# Files on NFS but NOT on S3
comm -23 "$NFS_SORTED" "$S3_SORTED" > "$MISSING_FILE"
MISSING_COUNT=$(wc -l < "$MISSING_FILE")

# Files on S3 but NOT on NFS (informational)
S3_ONLY_COUNT=$(comm -13 "$NFS_SORTED" "$S3_SORTED" | wc -l)
COMMON_COUNT=$(comm -12 "$NFS_SORTED" "$S3_SORTED" | wc -l)

log "Summary:"
log "  NFS total:          ${NFS_COUNT}"
log "  S3 total:           ${S3_COUNT}"
log "  Common (both):      ${COMMON_COUNT}"
log "  NFS-only (MISSING): ${MISSING_COUNT}  <-- these need uploading"
log "  S3-only:            ${S3_ONLY_COUNT}  (already in S3, not on NFS)"
log "  Missing list:       $MISSING_FILE"

# Estimate total size of missing files
if [[ "$MISSING_COUNT" -gt 0 ]]; then
    log "Computing sizes of missing files ..."

    # Prepend NFS base to get absolute paths, batch-stat for sizes.
    # This script runs on lambda3 (Linux) so we use GNU stat.
    sed "s|^|${NFS_BASE}/|" "$MISSING_FILE" \
        | xargs -d '\n' stat --printf='%s\t%n\n' 2>/dev/null \
        | sed "s|${NFS_BASE}/||" \
        > "$MISSING_SIZES" || true

    if [[ -s "$MISSING_SIZES" ]]; then
        TOTAL_BYTES=$(awk -F'\t' '{s+=$1} END {print s+0}' "$MISSING_SIZES")
    else
        log "  (stat failed, falling back to du estimate)"
        TOTAL_BYTES=$(sed "s|^|${NFS_BASE}/|" "$MISSING_FILE" \
            | xargs -d '\n' du -sb 2>/dev/null \
            | awk '{s+=$1} END {print s+0}')
        TOTAL_BYTES="${TOTAL_BYTES:-0}"
    fi

    log "Total upload size: $(human_size $TOTAL_BYTES) (${TOTAL_BYTES} bytes) across ${MISSING_COUNT} files"

    # Show top 10 largest missing files
    if [[ -s "$MISSING_SIZES" ]]; then
        log ""
        log "Top 10 largest missing files:"
        sort -rn "$MISSING_SIZES" | head -10 | while IFS=$'\t' read -r sz path; do
            log "  $(human_size "$sz")  $path"
        done
    fi

    # Show breakdown by top-level directory
    log ""
    log "Missing files by top-level directory:"
    cut -d'/' -f1 "$MISSING_FILE" | sort | uniq -c | sort -rn | head -20 | while read -r cnt dir; do
        log "  ${cnt}  ${dir}"
    done
fi

# ══════════════════════════════════════════════════════════════════════════
# PHASE 3: Upload (if not list-only)
# ══════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "list-only" ]]; then
    log ""
    log "──── Mode: list-only. Skipping upload. ────"
    log "Review missing files at: $MISSING_FILE"
    log "Done."
    exit 0
fi

if [[ "$MISSING_COUNT" -eq 0 ]]; then
    log ""
    log "Nothing to upload. NFS and S3 are in sync."
    exit 0
fi

log ""
log "──── Phase 3: Upload ($MODE) ────"

if [[ "$MODE" == "dry-run" ]]; then
    log "DRY RUN: showing first 20 files that would be uploaded:"
    head -20 "$MISSING_FILE" | while IFS= read -r relpath; do
        log "  [would upload] ${NFS_BASE}/${relpath} → ${S3_URI}/${relpath}"
    done
    if [[ "$MISSING_COUNT" -gt 20 ]]; then
        log "  ... and $((MISSING_COUNT - 20)) more files"
    fi
    log ""
    log "To actually upload, re-run with --execute"
    log "Full missing list: $MISSING_FILE"
    exit 0
fi

# ── Execute mode: upload missing files ──
UPLOAD_LOG="${WORK_DIR}/upload_progress.log"
UPLOAD_ERRORS="${WORK_DIR}/upload_errors.log"
touch "$UPLOAD_LOG" "$UPLOAD_ERRORS"

upload_one_file() {
    local relpath="$1"
    local src="${NFS_BASE}/${relpath}"
    local dst="${S3_URI}/${relpath}"

    if aws s3 cp "$src" "$dst" --region "$AWS_REGION" --quiet 2>/dev/null; then
        echo "OK ${relpath}" >> "$UPLOAD_LOG"
    else
        echo "FAIL ${relpath}" >> "$UPLOAD_ERRORS"
        echo "FAIL ${relpath}" >> "$UPLOAD_LOG"
    fi
}
export -f upload_one_file
export NFS_BASE S3_URI AWS_REGION UPLOAD_LOG UPLOAD_ERRORS

log "Starting upload of ${MISSING_COUNT} files with ${PARALLEL} parallel workers ..."
log "Progress log: $UPLOAD_LOG"
log "Error log:    $UPLOAD_ERRORS"

# Upload with parallel workers — use tr+xargs -0 to handle filenames with quotes
tr '\n' '\0' < "$MISSING_FILE" \
    | xargs -0 -P "$PARALLEL" -n 1 bash -c 'upload_one_file "$1"' _ &

UPLOAD_PID=$!

# Monitor progress
while kill -0 "$UPLOAD_PID" 2>/dev/null; do
    sleep 10
    DONE=$(wc -l < "$UPLOAD_LOG" 2>/dev/null || echo 0)
    ERRS=$(wc -l < "$UPLOAD_ERRORS" 2>/dev/null || echo 0)
    log "Progress: ${DONE}/${MISSING_COUNT} uploaded, ${ERRS} errors"
done

wait "$UPLOAD_PID" || true

FINAL_DONE=$(wc -l < "$UPLOAD_LOG")
FINAL_ERRS=$(wc -l < "$UPLOAD_ERRORS")

log ""
log "============================================================"
log "Upload complete."
log "  Uploaded:  $((FINAL_DONE - FINAL_ERRS)) / ${MISSING_COUNT}"
log "  Errors:    ${FINAL_ERRS}"
if [[ "$FINAL_ERRS" -gt 0 ]]; then
    log "  Error list: $UPLOAD_ERRORS"
    log "  Re-run with --execute to retry failed files."
fi
log "  Full log:  $LOG_FILE"
log "============================================================"

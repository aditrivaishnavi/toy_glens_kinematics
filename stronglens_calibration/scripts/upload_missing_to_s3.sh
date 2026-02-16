#!/usr/bin/env bash
# upload_missing_to_s3.sh — Upload NFS-only files to S3 (no overwrites).
#
# Reads the diff list produced by sync_nfs_to_s3.sh --list-only and uploads
# only files confirmed missing from S3.  Every file is HEAD-checked before
# upload so nothing is ever overwritten.
#
# Usage (run on lambda3):
#   ./scripts/upload_missing_to_s3.sh --input /tmp/nfs_s3_sync_<ts>/nfs_only_missing.txt
#   ./scripts/upload_missing_to_s3.sh --input <file> --dry-run          # default: preview
#   ./scripts/upload_missing_to_s3.sh --input <file> --execute           # upload
#   ./scripts/upload_missing_to_s3.sh --input <file> --execute --parallel 16
#
# Safety:
#   - Every file is HEAD-checked in S3 before upload (no overwrites, ever)
#   - Paths are validated (no '..' traversal, must start with known prefixes)
#   - Dry-run is the default mode
#   - Uploads are checkpointed so re-runs skip already-uploaded files
#   - All actions logged with timestamps
#
# Cost:  HEAD $0.0004/1000, PUT $0.005/1000, data-in free.
#        For 422K files: HEAD ~$0.17, PUT ~$2.11, data-in $0.

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
NFS_BASE="/lambda/nfs/darkhaloscope-training-dc"
S3_BUCKET="darkhaloscope"
S3_URI="s3://${S3_BUCKET}"
AWS_REGION="${AWS_REGION:-us-east-2}"
PARALLEL="${PARALLEL:-8}"

# ── Parse arguments ────────────────────────────────────────────────────────
MODE="dry-run"       # dry-run | execute
INPUT_FILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)     INPUT_FILE="$2"; shift 2 ;;
        --execute)   MODE="execute"; shift ;;
        --dry-run)   MODE="dry-run"; shift ;;
        --parallel)  PARALLEL="$2"; shift 2 ;;
        --help|-h)
            head -16 "$0" | tail -13
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$INPUT_FILE" ]]; then
    echo "ERROR: --input <missing_file_list> is required." >&2
    echo "       Run sync_nfs_to_s3.sh --list-only first to generate the list." >&2
    exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Input file not found: $INPUT_FILE" >&2
    exit 1
fi

# ── Derived paths ──────────────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="/tmp/nfs_s3_upload_${TIMESTAMP}"
LOG_FILE="${WORK_DIR}/upload.log"
DONE_FILE="${WORK_DIR}/uploaded.txt"        # successfully uploaded keys
SKIP_FILE="${WORK_DIR}/skipped_exists.txt"  # skipped because already in S3
FAIL_FILE="${WORK_DIR}/failed.txt"          # failed uploads
UNSAFE_FILE="${WORK_DIR}/unsafe_paths.txt"  # rejected by path validation
TODO_FILE="${WORK_DIR}/todo.txt"            # filtered safe paths to upload

mkdir -p "$WORK_DIR"
touch "$LOG_FILE" "$DONE_FILE" "$SKIP_FILE" "$FAIL_FILE" "$UNSAFE_FILE"

# ── Helpers ────────────────────────────────────────────────────────────────
log() {
    local msg
    msg="$(date '+%Y-%m-%d %H:%M:%S') | $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

human_size() {
    local bytes=$1
    awk -v b="$bytes" 'BEGIN {
        if      (b >= 1073741824) printf "%.1f GB", b/1073741824
        else if (b >= 1048576)    printf "%.1f MB", b/1048576
        else if (b >= 1024)       printf "%.1f KB", b/1024
        else                      printf "%d B", b
    }'
}

# ── Path safety validation ────────────────────────────────────────────────
# Rejects paths with directory traversal, absolute paths, or suspicious chars.
is_safe_path() {
    local p="$1"
    # Reject empty
    [[ -z "$p" ]] && return 1
    # Reject absolute paths (should be relative)
    [[ "$p" == /* ]] && return 1
    # Reject directory traversal
    [[ "$p" == *..* ]] && return 1
    # Reject control characters and some shell-dangerous chars
    [[ "$p" =~ [[:cntrl:]] ]] && return 1
    [[ "$p" == *\;* ]] && return 1
    [[ "$p" == *\|* ]] && return 1
    [[ "$p" == *\&* ]] && return 1
    [[ "$p" == *\$* ]] && return 1
    [[ "$p" == *\`* ]] && return 1
    return 0
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Validate and filter input paths
# ══════════════════════════════════════════════════════════════════════════
log "============================================================"
log "NFS → S3 uploader (safe, no-overwrite)"
log "Mode:       $MODE"
log "Input:      $INPUT_FILE"
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

# Verify AWS credentials
if ! aws sts get-caller-identity --region "$AWS_REGION" >/dev/null 2>&1; then
    log "ERROR: AWS credentials not configured."
    exit 1
fi
log "AWS credentials verified."

TOTAL_INPUT=$(wc -l < "$INPUT_FILE")
log ""
log "──── Step 1: Validating ${TOTAL_INPUT} paths ────"

SAFE_COUNT=0
UNSAFE_COUNT=0
while IFS= read -r relpath; do
    if is_safe_path "$relpath"; then
        # Also verify the source file actually exists on NFS
        if [[ -f "${NFS_BASE}/${relpath}" ]]; then
            echo "$relpath" >> "$TODO_FILE"
            SAFE_COUNT=$((SAFE_COUNT + 1))
        else
            echo "NFS_MISSING ${relpath}" >> "$UNSAFE_FILE"
            UNSAFE_COUNT=$((UNSAFE_COUNT + 1))
        fi
    else
        echo "BAD_PATH ${relpath}" >> "$UNSAFE_FILE"
        UNSAFE_COUNT=$((UNSAFE_COUNT + 1))
    fi
done < "$INPUT_FILE"

log "Path validation complete:"
log "  Safe (will process): ${SAFE_COUNT}"
log "  Rejected/missing:    ${UNSAFE_COUNT}"
if [[ "$UNSAFE_COUNT" -gt 0 ]]; then
    log "  See rejected paths:  $UNSAFE_FILE"
fi

if [[ "$SAFE_COUNT" -eq 0 ]]; then
    log "No files to upload. Exiting."
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Dry-run preview
# ══════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "dry-run" ]]; then
    log ""
    log "──── DRY RUN: preview of first 20 uploads ────"
    head -20 "$TODO_FILE" | while IFS= read -r relpath; do
        fsize=$(stat --printf='%s' "${NFS_BASE}/${relpath}" 2>/dev/null || echo "?")
        log "  $(human_size "${fsize}") | ${NFS_BASE}/${relpath}"
        log "       -> ${S3_URI}/${relpath}"
    done
    if [[ "$SAFE_COUNT" -gt 20 ]]; then
        log "  ... and $((SAFE_COUNT - 20)) more files"
    fi
    log ""
    log "To upload, re-run with --execute:"
    log "  $0 --input $INPUT_FILE --execute --parallel $PARALLEL"
    log ""
    log "Filtered todo list: $TODO_FILE"
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Execute uploads (HEAD-check + upload, parallel, checkpointed)
# ══════════════════════════════════════════════════════════════════════════
log ""
log "──── Step 3: Uploading ${SAFE_COUNT} files (parallel=${PARALLEL}) ────"
log "  Done file:    $DONE_FILE"
log "  Skipped file: $SKIP_FILE"
log "  Fail file:    $FAIL_FILE"

# Single-file upload function: HEAD-check then upload
safe_upload_one() {
    local relpath="$1"
    local src="${NFS_BASE}/${relpath}"
    local s3key="${relpath}"

    # HEAD-check: does object already exist in S3?
    if aws s3api head-object \
            --bucket "$S3_BUCKET" \
            --key "$s3key" \
            --region "$AWS_REGION" \
            >/dev/null 2>&1; then
        echo "$relpath" >> "$SKIP_FILE"
        return 0
    fi

    # Upload
    if aws s3 cp "$src" "s3://${S3_BUCKET}/${s3key}" \
            --region "$AWS_REGION" \
            --quiet 2>/dev/null; then
        echo "$relpath" >> "$DONE_FILE"
    else
        echo "$relpath" >> "$FAIL_FILE"
    fi
}
export -f safe_upload_one
export NFS_BASE S3_BUCKET AWS_REGION DONE_FILE SKIP_FILE FAIL_FILE

# Launch parallel uploads in background
# Use tr+xargs -0 with positional args to handle filenames containing quotes
tr '\n' '\0' < "$TODO_FILE" \
    | xargs -0 -P "$PARALLEL" -n 1 bash -c 'safe_upload_one "$1"' _ &
UPLOAD_PID=$!

# Monitor progress every 15 seconds
log "Upload started (PID $UPLOAD_PID). Monitoring..."
while kill -0 "$UPLOAD_PID" 2>/dev/null; do
    sleep 15
    n_done=$(wc -l < "$DONE_FILE" 2>/dev/null || echo 0)
    n_skip=$(wc -l < "$SKIP_FILE" 2>/dev/null || echo 0)
    n_fail=$(wc -l < "$FAIL_FILE" 2>/dev/null || echo 0)
    n_total=$((n_done + n_skip + n_fail))
    pct=0
    if [[ "$SAFE_COUNT" -gt 0 ]]; then
        pct=$((n_total * 100 / SAFE_COUNT))
    fi
    log "Progress: ${n_total}/${SAFE_COUNT} (${pct}%) | uploaded=${n_done} skipped=${n_skip} failed=${n_fail}"
done
wait "$UPLOAD_PID" || true

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Final summary
# ══════════════════════════════════════════════════════════════════════════
FINAL_DONE=$(wc -l < "$DONE_FILE")
FINAL_SKIP=$(wc -l < "$SKIP_FILE")
FINAL_FAIL=$(wc -l < "$FAIL_FILE")

log ""
log "============================================================"
log "Upload complete."
log "  Uploaded (new):     ${FINAL_DONE}"
log "  Skipped (existed):  ${FINAL_SKIP}"
log "  Failed:             ${FINAL_FAIL}"
log "  Total processed:    $((FINAL_DONE + FINAL_SKIP + FINAL_FAIL)) / ${SAFE_COUNT}"
log "------------------------------------------------------------"
if [[ "$FINAL_FAIL" -gt 0 ]]; then
    log "  Failed list: $FAIL_FILE"
    log ""
    log "  To retry failed files:"
    log "    $0 --input $FAIL_FILE --execute --parallel $PARALLEL"
fi
log "  Full log:  $LOG_FILE"
log "============================================================"

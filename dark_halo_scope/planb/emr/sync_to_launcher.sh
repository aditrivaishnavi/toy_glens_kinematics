#!/bin/bash
# =============================================================================
# Sync Code to EMR-Launcher Machine
# =============================================================================
#
# This script syncs the planb code to the emr-launcher machine.
#
# Usage:
#   ./sync_to_launcher.sh [hostname]
#
# Example:
#   ./sync_to_launcher.sh emr-launcher
#   ./sync_to_launcher.sh ec2-user@10.0.1.100
#
# =============================================================================

set -e

# Configuration
LAUNCHER_HOST="${1:-emr-launcher}"
REMOTE_DIR="~/planb"
LOCAL_DIR="$(dirname "$(dirname "$(realpath "$0")")")"

echo "=============================================="
echo "Syncing Plan B code to EMR-Launcher"
echo "=============================================="
echo "Local:  $LOCAL_DIR"
echo "Remote: $LAUNCHER_HOST:$REMOTE_DIR"
echo ""

# Pre-flight: syntax check all Python files
echo "[1/4] Running syntax check on Python files..."
find "$LOCAL_DIR" -name "*.py" -type f | while read -r file; do
    python3 -m py_compile "$file" || {
        echo "ERROR: Syntax error in $file"
        exit 1
    }
done
echo "      All Python files pass syntax check."

# Create remote directory
echo "[2/4] Creating remote directory..."
ssh "$LAUNCHER_HOST" "mkdir -p $REMOTE_DIR"

# Sync code (excluding unnecessary files)
echo "[3/4] Syncing code..."
rsync -avz --progress \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='.DS_Store' \
    --exclude='*.egg-info' \
    --exclude='.pytest_cache' \
    --exclude='*.log' \
    --exclude='checkpoints/' \
    --exclude='results/' \
    "$LOCAL_DIR/" "$LAUNCHER_HOST:$REMOTE_DIR/"

# Verify sync
echo "[4/4] Verifying sync..."
ssh "$LAUNCHER_HOST" "ls -la $REMOTE_DIR/emr/"

echo ""
echo "=============================================="
echo "Sync complete!"
echo ""
echo "Next steps on $LAUNCHER_HOST:"
echo "  cd $REMOTE_DIR/emr"
echo "  python3 launcher.py list"
echo "=============================================="

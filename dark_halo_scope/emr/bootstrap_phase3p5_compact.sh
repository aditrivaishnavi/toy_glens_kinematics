#!/usr/bin/env bash
set -euo pipefail

# EMR bootstrap for Phase 3.5 compact job
# Minimal - no extra dependencies needed beyond what EMR provides
# (PySpark is installed by EMR after bootstrap completes)

echo "=== Phase 3.5 Bootstrap: Starting ==="
echo "Python version: $(python3 --version)"
echo "=== Phase 3.5 Bootstrap: Complete ==="

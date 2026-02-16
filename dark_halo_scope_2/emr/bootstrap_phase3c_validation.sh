#!/usr/bin/env bash
set -euo pipefail

# EMR bootstrap for Phase 3c Validation (PySpark)
# Minimal dependencies - just needs PySpark (pre-installed on EMR)

PY=python3

# Ensure pip is available
if ! $PY -m pip --version >/dev/null 2>&1; then
  sudo yum -y install python3-pip >/dev/null 2>&1 || true
fi

sudo $PY -m pip install --upgrade pip >/dev/null

# No additional dependencies needed - validation uses only PySpark
echo "Bootstrap complete for Phase 3c Validation"


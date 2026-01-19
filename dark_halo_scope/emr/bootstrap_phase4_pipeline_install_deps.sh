#!/usr/bin/env bash
set -euo pipefail

# Phase 4 bootstrap: keep compatible with EMR 6.x Python 3.7.
# Note: boto3 emits a Python 3.7 deprecation warning, but remains functional.

PY_BIN=${PY_BIN:-python3}

# System deps for some wheels (best-effort)
sudo yum -y -q install gcc gcc-c++ make || true

# Ensure pip is present and reasonably recent
$PY_BIN -m pip install --quiet --upgrade pip setuptools wheel || true

# Core runtime deps
$PY_BIN -m pip install --quiet \
  "numpy>=1.20,<2.0" \
  "requests>=2.28,<3.0" \
  "boto3>=1.28.0" \
  "astropy>=4.3,<5.0"

# Optional: faster FITS I/O (best-effort). If build fails, continue.
$PY_BIN -m pip install --quiet "fitsio>=1.1.2" || true

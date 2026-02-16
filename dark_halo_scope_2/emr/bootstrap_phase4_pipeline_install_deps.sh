#!/usr/bin/env bash
set -euo pipefail

# Phase 4 bootstrap: keep compatible with EMR 6.x Python 3.7.
# Note: boto3 emits a Python 3.7 deprecation warning, but remains functional.

echo "=== Phase 4 Bootstrap: Starting ==="

# System deps for some wheels (best-effort)
sudo yum -y -q install gcc gcc-c++ make || true

# Ensure pip is present and reasonably recent
sudo python3 -m pip install --quiet --upgrade pip setuptools wheel || true

# Pin urllib3<2.0 for compatibility with EMR's old OpenSSL
sudo python3 -m pip install --quiet 'urllib3<2.0' || true

# Core runtime deps - use sudo for system-wide install
sudo python3 -m pip install --quiet \
  "numpy>=1.20,<2.0" \
  "requests>=2.28,<3.0" \
  "boto3>=1.28.0" \
  "astropy>=4.3,<5.0"

# Optional: faster FITS I/O (best-effort). If build fails, continue.
sudo python3 -m pip install --quiet "fitsio>=1.1.2" || true

# =========================================================================
# LENSTRONOMY: Realistic lens modeling for SIE injections
# Used by stage 4c for physics-based lens injection and validation.
# If installation fails, the pipeline falls back to simplified SIS model.
# =========================================================================
echo "=== Phase 4 Bootstrap: Installing lenstronomy ==="
sudo python3 -m pip install --quiet "lenstronomy>=1.11.0" || {
  echo "WARNING: lenstronomy installation failed. Stage 4c will use simplified SIS model."
}

# Verify installations
echo "=== Phase 4 Bootstrap: Verifying installations ==="
python3 -c "import boto3; print(f'boto3 version: {boto3.__version__}')"
python3 -c "import numpy; print(f'numpy version: {numpy.__version__}')"
python3 -c "import astropy; print(f'astropy version: {astropy.__version__}')"

# Check lenstronomy (optional)
python3 -c "import lenstronomy; print(f'lenstronomy version: {lenstronomy.__version__}')" 2>/dev/null || {
  echo "lenstronomy not available - will use SIS fallback"
}

echo "=== Phase 4 Bootstrap: Complete ==="

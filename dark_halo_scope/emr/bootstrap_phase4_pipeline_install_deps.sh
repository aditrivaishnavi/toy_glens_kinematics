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

# Verify installations
echo "=== Phase 4 Bootstrap: Verifying installations ==="
python3 -c "import boto3; print(f'boto3 version: {boto3.__version__}')"
python3 -c "import numpy; print(f'numpy version: {numpy.__version__}')"
python3 -c "import astropy; print(f'astropy version: {astropy.__version__}')"

echo "=== Phase 4 Bootstrap: Complete ==="

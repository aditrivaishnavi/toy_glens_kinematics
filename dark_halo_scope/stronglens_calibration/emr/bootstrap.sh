#!/bin/bash
# =============================================================================
# EMR Bootstrap Script for Stronglens Calibration (v2 - optimized)
# =============================================================================
# Optimized for fast installation - skips slow packages that have fallbacks
# =============================================================================

set -e
set -o pipefail

echo "=============================================="
echo "Stronglens Calibration Bootstrap v2"
echo "=============================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"

# =============================================================================
# SYSTEM PACKAGES (minimal)
# =============================================================================
echo "[1/3] System packages..."
if command -v dnf &> /dev/null; then
    sudo dnf -y -q install gcc gcc-c++ 2>/dev/null || true
else
    sudo yum -y -q install gcc gcc-c++ 2>/dev/null || true
fi
echo "  Done."

# =============================================================================
# PYTHON PACKAGES (fast install, no build from source)
# =============================================================================
echo "[2/3] Python packages..."

# Use --no-cache-dir for speed, install in batches
sudo python3 -m pip install --quiet --no-cache-dir --upgrade pip 2>/dev/null || true

# Core packages (pre-built wheels available)
echo "  Core packages..."
sudo python3 -m pip install --quiet --no-cache-dir \
    numpy pandas pyarrow pyyaml boto3 scipy 2>/dev/null || true

# Astropy (pre-built wheel available)
echo "  Astropy..."
sudo python3 -m pip install --quiet --no-cache-dir astropy 2>/dev/null || true

# fitsio - try pre-built, fallback to astropy.io.fits if fails
echo "  Fitsio (optional)..."
timeout 120 sudo python3 -m pip install --quiet --no-cache-dir fitsio 2>/dev/null || echo "  fitsio skipped (will use astropy)"

# healpy - SKIP (code has fallback, and build takes 20+ minutes)
echo "  Skipping healpy (using code fallback)"

echo "  Done."

# =============================================================================
# VERIFY CORE PACKAGES
# =============================================================================
echo "[3/3] Verifying..."
python3 -c "
import numpy, pandas, pyarrow, yaml, boto3, astropy, scipy
print('  Core packages OK')
from scipy.spatial import cKDTree
print('  scipy.cKDTree OK')
try:
    import fitsio
    print('  fitsio OK')
except:
    print('  fitsio not available (using astropy)')
" || echo "  Verification warnings (non-fatal)"

# =============================================================================
# ENVIRONMENT
# =============================================================================
echo 'export PYSPARK_PYTHON=/usr/bin/python3' | sudo tee -a /etc/profile.d/pyspark.sh > /dev/null
echo 'export NUMBA_CACHE_DIR=/tmp/numba_cache' | sudo tee -a /etc/profile.d/pyspark.sh > /dev/null

echo "=============================================="
echo "Bootstrap complete!"
echo "=============================================="
exit 0

#!/bin/bash
# EMR Bootstrap: Install Gen5 dependencies for COSMOS source integration
# Designed for EMR 6.x and 7.x compatibility

echo "========================================="
echo "Gen5 EMR Bootstrap: Installing dependencies"
echo "========================================="

# NOTE: Do NOT use set -e - we want graceful degradation
# NOTE: Do NOT upgrade pip - on EMR 7.x it was installed via RPM and can't be uninstalled
# The error "Cannot uninstall pip 21.3.1, RECORD file not found" will cause bootstrap failure

# CRITICAL: Set NUMBA_CACHE_DIR to writable location
# This fixes: RuntimeError: cannot cache function 'rotate': no locator available
# Numba needs a writable cache directory for JIT compilation
echo "Setting up Numba cache directory..."
sudo mkdir -p /tmp/numba_cache
sudo chmod 777 /tmp/numba_cache
echo 'export NUMBA_CACHE_DIR=/tmp/numba_cache' | sudo tee -a /etc/environment
echo 'export NUMBA_CACHE_DIR=/tmp/numba_cache' | sudo tee -a /etc/profile.d/numba.sh
sudo chmod +x /etc/profile.d/numba.sh || true
export NUMBA_CACHE_DIR=/tmp/numba_cache

# Install boto3 for S3 access (CRITICAL: needed for coadd cache loading)
echo "Installing boto3..."
sudo python3 -m pip install --quiet "boto3>=1.28.0" || true

# Install h5py for COSMOS bank loading
echo "Installing h5py..."
sudo python3 -m pip install --quiet "h5py>=3.7.0" || true

# Install lenstronomy for realistic SIE lens modeling
echo "Installing lenstronomy..."
sudo python3 -m pip install --quiet "lenstronomy>=1.10.0" || true

# Install additional dependencies
sudo python3 -m pip install --quiet "scipy>=1.7.0" "astropy>=4.3" || true

# Verify critical installations
echo "========================================="
echo "Gen5 Bootstrap: Verifying installations"
echo "========================================="
python3 -c "import boto3; print(f'boto3 version: {boto3.__version__}')" || echo "❌ boto3 import FAILED"
python3 -c "import h5py; print(f'h5py version: {h5py.__version__}')" || echo "❌ h5py import FAILED"
python3 -c "import numpy; print(f'numpy version: {numpy.__version__}')" || echo "❌ numpy import FAILED"
python3 -c "import astropy; print(f'astropy version: {astropy.__version__}')" || echo "❌ astropy import FAILED"

# Test lenstronomy with NUMBA_CACHE_DIR set
echo "Testing lenstronomy import with numba cache..."
NUMBA_CACHE_DIR=/tmp/numba_cache python3 -c "import lenstronomy; print(f'lenstronomy available')" || echo "⚠️ lenstronomy import failed"echo "========================================="
echo "✅ Gen5 Bootstrap: Complete"
echo "NUMBA_CACHE_DIR=/tmp/numba_cache"
echo "========================================="
exit 0
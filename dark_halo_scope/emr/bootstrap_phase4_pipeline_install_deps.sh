#!/usr/bin/env bash
# Bootstrap for EMR 7.x (Amazon Linux 2023) - uses dnf not yum

echo "=== Phase 4 Bootstrap: Starting ==="

# System deps for some wheels (best-effort) - dnf for AL2023, yum for older
if command -v dnf &> /dev/null; then
    sudo dnf -y -q install gcc gcc-c++ make || true
else
    sudo yum -y -q install gcc gcc-c++ make || true
fi

# Ensure pip is present and reasonably recent
sudo python3 -m pip install --quiet --upgrade pip setuptools wheel || true

# Core runtime deps - use sudo for system-wide install
# Each on its own line with || true to avoid failing entire bootstrap
sudo python3 -m pip install --quiet "numpy>=1.20,<2.0" || true
sudo python3 -m pip install --quiet "requests>=2.28,<3.0" || true  
sudo python3 -m pip install --quiet "boto3>=1.28.0" || true
sudo python3 -m pip install --quiet "astropy>=4.3" || true

# Optional: faster FITS I/O (best-effort). If build fails, continue.
sudo python3 -m pip install --quiet "fitsio>=1.1.2" || true

# lenstronomy for SIE lens modeling (optional)
sudo python3 -m pip install --quiet "lenstronomy>=1.10.0" || true

# Verify installations
echo "=== Phase 4 Bootstrap: Verifying installations ==="
python3 -c "import boto3; print(f'boto3 version: {boto3.__version__}')" || echo "boto3 import failed"
python3 -c "import numpy; print(f'numpy version: {numpy.__version__}')" || echo "numpy import failed"
python3 -c "import astropy; print(f'astropy version: {astropy.__version__}')" || echo "astropy import failed"

echo "=== Phase 4 Bootstrap: Complete ==="
exit 0

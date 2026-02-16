#!/bin/bash
# =============================================================================
# EMR Bootstrap Script for Plan B
# =============================================================================
#
# This script is executed on each EMR node during cluster creation.
# It installs required Python packages and configures the environment.
#
# Lessons Learned:
# - L6.1: Always check exit codes
# - L5.4: Verify installations succeed
#
# Usage:
#   Automatically executed by EMR during cluster bootstrap
#   Can also be tested locally: bash bootstrap.sh
#
# =============================================================================

set -e  # Exit on any error
set -o pipefail  # Catch errors in pipes

# Logging
LOG_FILE="/tmp/bootstrap_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================="
echo "Plan B EMR Bootstrap Script"
echo "=============================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Node type: ${EMR_NODE_TYPE:-unknown}"
echo ""

# =============================================================================
# SYSTEM PACKAGES
# =============================================================================

echo "[1/5] Installing system packages..."

sudo yum install -y \
    python3-devel \
    gcc \
    gcc-c++ \
    make \
    || { echo "ERROR: Failed to install system packages"; exit 1; }

echo "System packages installed successfully."

# =============================================================================
# PYTHON PACKAGES
# =============================================================================

echo "[2/5] Installing Python packages..."

# Core packages
sudo pip3 install --upgrade pip

# Scientific computing
sudo pip3 install \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scipy==1.11.1 \
    pyarrow==12.0.1 \
    || { echo "ERROR: Failed to install scientific packages"; exit 1; }

# Astronomy packages
sudo pip3 install \
    astropy==5.3.1 \
    fitsio==1.1.8 \
    || { echo "ERROR: Failed to install astronomy packages"; exit 1; }

# AWS and cloud
sudo pip3 install \
    boto3==1.28.0 \
    s3fs==2023.6.0 \
    || { echo "ERROR: Failed to install AWS packages"; exit 1; }

# Utilities
sudo pip3 install \
    tqdm==4.65.0 \
    pyyaml==6.0 \
    || { echo "ERROR: Failed to install utility packages"; exit 1; }

echo "Python packages installed successfully."

# =============================================================================
# VERIFY INSTALLATIONS
# =============================================================================

echo "[3/5] Verifying installations..."

python3 -c "
import numpy as np
import pandas as pd
import pyarrow
import astropy
import fitsio
import boto3
import s3fs
print('All packages imported successfully')
print(f'  numpy: {np.__version__}')
print(f'  pandas: {pd.__version__}')
print(f'  pyarrow: {pyarrow.__version__}')
print(f'  astropy: {astropy.__version__}')
" || { echo "ERROR: Package verification failed"; exit 1; }

echo "Package verification passed."

# =============================================================================
# CONFIGURE SPARK ENVIRONMENT
# =============================================================================

echo "[4/5] Configuring Spark environment..."

# Set Python path for PySpark
echo 'export PYSPARK_PYTHON=/usr/bin/python3' | sudo tee -a /etc/profile.d/pyspark.sh
echo 'export PYSPARK_DRIVER_PYTHON=/usr/bin/python3' | sudo tee -a /etc/profile.d/pyspark.sh

# Increase file limits for S3 access
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

echo "Spark environment configured."

# =============================================================================
# MASTER-ONLY SETUP
# =============================================================================

if [ "${EMR_NODE_TYPE}" = "MASTER" ] || [ "$(hostname)" = *"master"* ]; then
    echo "[5/5] Running master-only setup..."
    
    # Create working directories
    sudo mkdir -p /tmp/planb/{code,data,output}
    sudo chmod 777 /tmp/planb/*
    
    echo "Master setup completed."
else
    echo "[5/5] Skipping master-only setup (this is a worker node)."
fi

# =============================================================================
# COMPLETION
# =============================================================================

echo ""
echo "=============================================="
echo "Bootstrap completed successfully!"
echo "Log saved to: $LOG_FILE"
echo "=============================================="

#!/bin/bash
# EMR Bootstrap: Install Gen5 dependencies for COSMOS source integration

set -e

echo "========================================="
echo "Gen5 EMR Bootstrap: Installing dependencies"
echo "========================================="

# Upgrade pip
sudo python3 -m pip install --upgrade pip

# Install h5py for COSMOS bank loading
echo "Installing h5py..."
sudo python3 -m pip install h5py==3.8.0

# Install lenstronomy for realistic SIE lens modeling
echo "Installing lenstronomy..."
sudo python3 -m pip install lenstronomy==1.11.6

# Install additional dependencies for lenstronomy
sudo python3 -m pip install scipy astropy

echo "✅ Gen5 dependencies installed successfully"
echo "   - h5py 3.8.0"
echo "   - lenstronomy 1.11.6"
echo "   - scipy, astropy"
echo "========================================="


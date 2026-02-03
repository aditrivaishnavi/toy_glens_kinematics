#!/bin/bash
set -e

echo "=========================================="
echo "Gen5 COSMOS Bank Builder Bootstrap"
echo "=========================================="

# Install dependencies
sudo pip3 install galsim h5py numpy astropy pydantic

# Create workspace
sudo mkdir -p /mnt/cosmos_workspace
sudo chmod 777 /mnt/cosmos_workspace

# Download GalSim COSMOS catalog (~2.3 GB)
echo "Downloading COSMOS catalog..."
cd /mnt/cosmos_workspace

# GalSim will auto-download to ~/.galsim/COSMOS_23.5_training_sample
# We'll trigger this in the actual script, but prepare the directory
mkdir -p /home/hadoop/.galsim
sudo chown -R hadoop:hadoop /home/hadoop/.galsim

echo "✅ Bootstrap complete"


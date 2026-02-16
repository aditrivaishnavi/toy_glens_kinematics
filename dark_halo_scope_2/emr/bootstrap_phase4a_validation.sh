#!/bin/bash
# Bootstrap script for Phase 4a validation EMR cluster
# Installs boto3 which is needed for S3 path discovery

set -ex

echo "=== Phase 4a Validation Bootstrap ==="
echo "Installing Python dependencies..."

# Install boto3 for S3 path discovery
sudo pip3 install --quiet boto3 "urllib3<2.0"

# Verify installation
python3 -c "import boto3; print(f'boto3 version: {boto3.__version__}')"

echo "=== Bootstrap complete ==="


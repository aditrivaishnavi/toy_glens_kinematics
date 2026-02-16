#!/usr/bin/env bash
set -euo pipefail
sudo python3 -m pip install --quiet --upgrade pip
sudo python3 -m pip install --quiet "astropy>=4.3,<5.0" "boto3>=1.28.0"

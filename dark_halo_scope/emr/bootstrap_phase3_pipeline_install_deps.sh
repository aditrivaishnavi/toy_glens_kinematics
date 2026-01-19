#!/bin/bash
set -euo pipefail

# EMR best practice: install Python deps once on each node.
# NOTE: EMR 6.15.x uses Python 3.7 by default; astropy>=5 requires Python>=3.9.

sudo python3 -m pip install --quiet --upgrade pip

# Minimal, EMR-compatible dependencies.
sudo python3 -m pip install --quiet \
  "numpy>=1.20" \
  "boto3>=1.28.0" \
  "astropy>=4.3,<5.0"


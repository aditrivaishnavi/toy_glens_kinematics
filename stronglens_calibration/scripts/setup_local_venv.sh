#!/usr/bin/env bash
# setup_local_venv.sh — Create local development venv with CPU-only torch
# Run from the stronglens_calibration/ directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

echo "=== Setting up local venv at $VENV_DIR ==="

# Use pyenv Python 3.11 if available, else system python3
if command -v pyenv &>/dev/null; then
    eval "$(pyenv init -)"
    PYTHON="$(pyenv which python3)"
    echo "Using pyenv Python: $($PYTHON --version)"
else
    PYTHON="python3"
    echo "Using system Python: $($PYTHON --version)"
fi

# Check Python version >= 3.10
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MINOR" -lt 10 ]; then
    echo "ERROR: Python 3.10+ required (torch 2.7.0). Found Python 3.$PY_MINOR"
    echo "Install Python 3.10+ via: pyenv install 3.11.12"
    exit 1
fi

# Create venv
if [ -d "$VENV_DIR" ]; then
    echo "Venv already exists at $VENV_DIR — skipping creation."
else
    $PYTHON -m venv "$VENV_DIR"
    echo "Created venv."
fi

# Activate and install
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

echo "Installing CPU-only torch..."
pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cpu -q

echo "Installing remaining requirements..."
pip install pandas pyarrow scikit-learn scipy healpy astropy pyyaml lenstronomy -q

echo ""
echo "=== Local venv ready ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run tests:     pytest tests/ -v"

#!/usr/bin/env bash
# setup_lambda3_venv.sh — Create venv on Lambda3 with CUDA torch
# Run ON Lambda3 after rsyncing code.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv-lambda3"

echo "=== Setting up Lambda3 venv at $VENV_DIR ==="

PYTHON="python3"
echo "Using: $($PYTHON --version)"

# Check Python version >= 3.10
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MINOR" -lt 10 ]; then
    echo "ERROR: Python 3.10+ required (torch 2.7.0). Found Python 3.$PY_MINOR"
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

echo "Installing CUDA torch (from requirements.txt)..."
pip install -r "$PROJECT_DIR/requirements.txt" -q

echo ""
echo "=== Lambda3 venv ready ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "GPU check:     python -c \"import torch; print(torch.cuda.is_available())\""

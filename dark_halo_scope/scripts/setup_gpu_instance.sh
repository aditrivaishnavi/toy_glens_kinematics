#!/bin/bash
# Phase 5 GPU Instance Setup Script
# For AWS P3 instances (p3.2xlarge or p3.8xlarge)
#
# Usage:
#   1. Launch a p3.2xlarge or p3.8xlarge with Deep Learning AMI
#   2. SSH into the instance
#   3. Run this script: bash setup_gpu_instance.sh
#
# Instance recommendations:
#   - p3.2xlarge: 8 vCPUs, 1x V100 (16GB) - smoke tests
#   - p3.8xlarge: 32 vCPUs, 4x V100 (64GB) - full training (uses full 32 vCPU quota)

set -e

echo "========================================"
echo "Phase 5 GPU Instance Setup"
echo "========================================"

# Check NVIDIA driver
echo ""
echo "[1/6] Checking NVIDIA driver..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Ensure you're using Deep Learning AMI."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "GPU check passed."

# Check PyTorch CUDA
echo ""
echo "[2/6] Checking PyTorch CUDA..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Install additional dependencies
echo ""
echo "[3/6] Installing additional dependencies..."
pip install --quiet s3fs tensorboard pyarrow fsspec

# Create data directory
echo ""
echo "[4/6] Setting up data directory..."
DATA_DIR="/data"
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating $DATA_DIR..."
    sudo mkdir -p $DATA_DIR
    sudo chown $(whoami):$(whoami) $DATA_DIR
fi

# Create subdirectories
mkdir -p $DATA_DIR/phase4c/stamps
mkdir -p $DATA_DIR/phase5/models
mkdir -p $DATA_DIR/phase5/scores
mkdir -p $DATA_DIR/phase5/logs

echo "Data directories created:"
ls -la $DATA_DIR/

# Download code
echo ""
echo "[5/6] Setting up code..."
CODE_DIR="$DATA_DIR/code"
if [ ! -d "$CODE_DIR/dark_halo_scope" ]; then
    mkdir -p $CODE_DIR
    cd $CODE_DIR
    echo "Clone the repository or rsync the code here."
    echo "Example: rsync -avz --progress user@local:~/code/toy_glens_kinematics/dark_halo_scope/ $CODE_DIR/dark_halo_scope/"
fi

# Print data staging commands
echo ""
echo "[6/6] Data staging commands..."
echo ""
echo "========================================"
echo "NEXT STEPS"
echo "========================================"
echo ""
echo "1. Stage Phase 4c stamps from S3 to local NVMe (CRITICAL for performance):"
echo ""
echo "   aws s3 sync s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/stamps/train_stamp64_bandsgrz_gridgrid_small $DATA_DIR/phase4c/stamps/ --quiet"
echo ""
echo "   Expected size: ~500GB for stamps, ~15 minutes to sync"
echo ""
echo "2. Copy contract file:"
echo ""
echo "   cp dark_halo_scope/model/phase5_required_columns_contract.json $DATA_DIR/"
echo ""
echo "3. Run smoke test (single GPU, 1 epoch, 100 steps):"
echo ""
echo "   cd $CODE_DIR"
echo "   python dark_halo_scope/model/phase5_train_lensfinder.py \\"
echo "     --data $DATA_DIR/phase4c/stamps \\"
echo "     --contract_json $DATA_DIR/phase5_required_columns_contract.json \\"
echo "     --split train \\"
echo "     --arch resnet18 \\"
echo "     --epochs 1 \\"
echo "     --steps_per_epoch 100 \\"
echo "     --batch_size 256 \\"
echo "     --out_dir $DATA_DIR/phase5/models/resnet18_smoke"
echo ""
echo "4. Run full training (4 GPUs with DDP):"
echo ""
echo "   torchrun --standalone --nproc_per_node=4 dark_halo_scope/model/phase5_train_lensfinder.py \\"
echo "     --data $DATA_DIR/phase4c/stamps \\"
echo "     --contract_json $DATA_DIR/phase5_required_columns_contract.json \\"
echo "     --split train \\"
echo "     --arch resnet18 \\"
echo "     --epochs 5 \\"
echo "     --steps_per_epoch 5000 \\"
echo "     --batch_size 256 \\"
echo "     --out_dir $DATA_DIR/phase5/models/resnet18_v1"
echo ""
echo "5. Monitor with TensorBoard:"
echo ""
echo "   tensorboard --logdir $DATA_DIR/phase5/models/resnet18_v1/tb --bind_all"
echo ""
echo "6. Upload checkpoints to S3:"
echo ""
echo "   aws s3 sync $DATA_DIR/phase5/models/resnet18_v1 s3://darkhaloscope/phase5/models/resnet18_v1/"
echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"


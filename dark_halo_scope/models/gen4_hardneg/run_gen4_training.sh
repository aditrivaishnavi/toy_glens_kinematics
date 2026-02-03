#!/bin/bash
#
# Gen4 Training with Hard Negatives
# Run this script on Lambda after copying to /lambda/nfs/darkhaloscope-training-dc/code/
#

set -e

cd /lambda/nfs/darkhaloscope-training-dc

# === Step 1: Prepare Hard Negatives ===
echo "=== Preparing Hard Negatives ==="

mkdir -p hard_negatives/merged

python3 << "PYEOF"
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import os

# Load hard negatives from Gen2 and Gen3
hn_records = []
for gen in ["gen2", "gen3"]:
    path = f"hard_negatives/{gen}_round0_hard_negatives_top20k.parquet"
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet")]
        if files:
            table = pq.read_table(files[0])
            df = table.to_pandas()
            df["source_model"] = gen
            hn_records.append(df)
            print(f"Loaded {len(df)} hard negatives from {gen}")

if not hn_records:
    print("ERROR: No hard negatives found!")
    exit(1)

merged = pd.concat(hn_records, ignore_index=True)
print(f"Total before dedup: {len(merged)}")

merged = merged.drop_duplicates(subset=["ra", "dec"])
print(f"Total after dedup: {len(merged)}")

# Create lookup keys (rounded for matching)
merged["ra_key"] = np.round(merged["ra"], 6)
merged["dec_key"] = np.round(merged["dec"], 6)

# Save lookup file
lookup = merged[["ra_key", "dec_key", "brickname", "source_model"]].copy()
lookup.to_parquet("hard_negatives/merged/hard_neg_lookup.parquet", index=False)
print(f"Saved lookup: {len(lookup)} entries to hard_negatives/merged/hard_neg_lookup.parquet")
PYEOF

# === Step 2: Create Output Directory ===
echo ""
echo "=== Setting Up Gen4 Output Directory ==="
mkdir -p models/gen4_hardneg

# === Step 3: Copy Training Script ===
echo ""
echo "=== Copying Training Script ==="
# The training script should already be in code/ directory

# === Step 4: Start Training ===
echo ""
echo "=== Starting Gen4 Training ==="

OUT_DIR=/lambda/nfs/darkhaloscope-training-dc/models/gen4_hardneg
DATA_DIR=/lambda/nfs/darkhaloscope-training-dc/phase4c_v4_sota_moffat
HARD_NEG=/lambda/nfs/darkhaloscope-training-dc/hard_negatives/merged/hard_neg_lookup.parquet

PYTHONUNBUFFERED=1 python3 -u code/phase5_train_gen4_hardneg.py \
    --data $DATA_DIR \
    --out_dir $OUT_DIR \
    --hard_neg_path $HARD_NEG \
    --hard_neg_weight 5 \
    --arch convnext_tiny \
    --epochs 50 \
    --batch_size 256 \
    --lr 3e-4 \
    --weight_decay 1e-2 \
    --dropout 0.1 \
    --use_bf16 \
    --augment \
    --loss focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --min_theta_over_psf 0.5 \
    --norm_method outer \
    --meta_cols psfsize_r,psfdepth_r \
    --early_stopping_patience 5 \
    2>&1 | tee $OUT_DIR/train.log

echo ""
echo "=== Training Complete ==="
echo "Checkpoints in: $OUT_DIR"


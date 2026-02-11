#!/usr/bin/env python3
"""Fix missing split assignments in training manifest."""
import pandas as pd
import numpy as np

MANIFEST_PATH = "/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/manifests/training_v1.parquet"

df = pd.read_parquet(MANIFEST_PATH)

# Get stats
pos_mask = df["label"] == 1
missing_split = df["split"].isna()

print(f"Total samples: {len(df)}")
print(f"Positives: {pos_mask.sum()}")
print(f"Missing split: {missing_split.sum()}")

# For positives, use deterministic random assignment based on galaxy_id
# 70% train, 15% val, 15% test
def assign_split(row):
    if pd.notna(row["split"]):
        return row["split"]
    
    # Deterministic based on galaxy_id
    h = hash(str(row["galaxy_id"])) % 100
    if h < 70:
        return "train"
    elif h < 85:
        return "val"
    else:
        return "test"

# Apply split assignment
print("\nAssigning splits...")
df["split"] = df.apply(assign_split, axis=1)

# Verify
print("\nFinal distribution:")
for split in ["train", "val", "test"]:
    sub = df[df["split"] == split]
    print(f"  {split}: positives={len(sub[sub['label']==1])}, negatives={len(sub[sub['label']==0])}")

# Save
df.to_parquet(MANIFEST_PATH, index=False)
print(f"\nManifest updated: {MANIFEST_PATH}")

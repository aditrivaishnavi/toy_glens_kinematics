#!/usr/bin/env python3
"""Check input data ranges."""
import sys
import numpy as np
sys.path.insert(0, "/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/code")

from dhs.data import LensDataset, DatasetConfig, SplitConfig
from dhs.transforms import AugmentConfig

dcfg = DatasetConfig(
    parquet_path="",
    manifest_path="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/manifests/training_v1.parquet",
    mode="file_manifest",
    preprocessing="raw_robust",
)
aug = AugmentConfig(hflip=False, vflip=False, rot90=False)
ds_train = LensDataset(dcfg, SplitConfig(split_value="train"), aug)

print(f"Train samples: {len(ds_train)}")

# Check input ranges for first 1000 samples
mins, maxs = [], []
for i in range(1000):
    x, y, w = ds_train[i]
    mins.append(x.min())
    maxs.append(x.max())

print(f"Min range: [{min(mins):.2f}, {max(mins):.2f}]")
print(f"Max range: [{min(maxs):.2f}, {max(maxs):.2f}]")
print(f"Overall: [{min(mins):.2f}, {max(maxs):.2f}]")

# Check for extreme values
extreme_count = sum(1 for m in maxs if m > 100)
print(f"Samples with max > 100: {extreme_count}/1000")

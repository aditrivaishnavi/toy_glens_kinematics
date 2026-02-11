#!/usr/bin/env python3
"""Debug NaN issues in data loading."""
import sys
import numpy as np
import torch
sys.path.insert(0, "/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/code")

from dhs.data import LensDataset, DatasetConfig, SplitConfig
from dhs.transforms import AugmentConfig
from dhs.model import build_resnet18

# Test data
dcfg = DatasetConfig(
    parquet_path="",
    manifest_path="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/manifests/training_v1.parquet",
    mode="file_manifest",
    preprocessing="raw_robust",
)
aug = AugmentConfig(hflip=False, vflip=False, rot90=False)
ds_val = LensDataset(dcfg, SplitConfig(split_value="val"), aug)

print(f"Val samples: {len(ds_val)}")

# Test first 100 samples
nan_count = 0
inf_count = 0
for i in range(min(100, len(ds_val))):
    try:
        x, y, w = ds_val[i]
        if np.isnan(x).any():
            nan_count += 1
            print(f"NaN at index {i}, count: {np.isnan(x).sum()}, path: {ds_val.df.iloc[i]['cutout_path']}")
        if np.isinf(x).any():
            inf_count += 1
            print(f"Inf at index {i}, count: {np.isinf(x).sum()}")
    except Exception as e:
        print(f"Error at index {i}: {e}")

print(f"\nOut of 100 samples: {nan_count} NaN, {inf_count} Inf")

# Test model
model = build_resnet18(3)
model.eval()
x_tensor = torch.from_numpy(ds_val[0][0]).unsqueeze(0).float()
print(f"\nInput shape: {x_tensor.shape}")
print(f"Input has NaN: {torch.isnan(x_tensor).any()}")
print(f"Input has Inf: {torch.isinf(x_tensor).any()}")
print(f"Input range: [{x_tensor.min():.3f}, {x_tensor.max():.3f}]")

with torch.no_grad():
    out = model(x_tensor)
    print(f"Output: {out.item():.4f}")
    print(f"Output has NaN: {torch.isnan(out).any()}")

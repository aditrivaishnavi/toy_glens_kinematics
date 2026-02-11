#!/usr/bin/env python3
"""Debug full evaluation pass."""
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, "/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/code")

from dhs.data import LensDataset, DatasetConfig, SplitConfig
from dhs.transforms import AugmentConfig
from dhs.model import build_resnet18

def _collate_weighted(batch):
    xs, ys, ws = zip(*batch)
    x = torch.from_numpy(np.stack(xs, axis=0)).float()
    y = torch.from_numpy(np.array(ys)).float().view(-1,1)
    w = torch.from_numpy(np.array(ws)).float().view(-1,1)
    return x, y, w

# Test data
dcfg = DatasetConfig(
    parquet_path="",
    manifest_path="/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/manifests/training_v1.parquet",
    mode="file_manifest",
    preprocessing="raw_robust",
)
aug = AugmentConfig(hflip=False, vflip=False, rot90=False)
ds_val = LensDataset(dcfg, SplitConfig(split_value="val"), aug)

dl_val = DataLoader(ds_val, batch_size=256, shuffle=False, num_workers=8, 
                    pin_memory=True, collate_fn=_collate_weighted)

print(f"Val samples: {len(ds_val)}")
print(f"Batches: {len(dl_val)}")

model = build_resnet18(3).cuda()
model.eval()

nan_batch = None
all_preds = []
all_labels = []

with torch.no_grad():
    for i, (x, y, w) in enumerate(dl_val):
        # Check input
        if torch.isnan(x).any():
            print(f"Batch {i}: Input has NaN")
        if torch.isinf(x).any():
            print(f"Batch {i}: Input has Inf")
        
        x = x.cuda()
        logits = model(x)
        
        # Check output
        if torch.isnan(logits).any():
            print(f"Batch {i}: Output has NaN")
            nan_batch = i
            break
        
        p = torch.sigmoid(logits).cpu().numpy().ravel()
        
        if np.isnan(p).any():
            print(f"Batch {i}: Sigmoid output has NaN")
            nan_batch = i
            break
        
        all_preds.extend(p)
        all_labels.extend(y.numpy().ravel())
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(dl_val)} batches")
        
        if i >= 10:  # Just test first 10 batches
            break

print(f"\nProcessed batches: {i+1}")
print(f"NaN batch: {nan_batch}")
print(f"Predictions range: [{min(all_preds):.4f}, {max(all_preds):.4f}]")
print(f"Label distribution: {sum(all_labels)} positives, {len(all_labels) - sum(all_labels)} negatives")

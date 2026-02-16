#!/usr/bin/env python3
"""Compare training data signal to anchor lenses."""
import numpy as np
import io
import pyarrow.dataset as ds
from pathlib import Path
from astropy.io import fits

# Training data
data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")
test_filter = (ds.field("region_split") == "test") & (ds.field("is_control") == 0) & (ds.field("cutout_ok") == 1)
table = dataset.to_table(filter=test_filter, columns=["stamp_npz", "arc_snr", "theta_e_arcsec"])
table = table.slice(0, 100)

print("=== TRAINING DATA (injected lenses) ===")
training_central = []
for i in range(min(100, table.num_rows)):
    blob = table["stamp_npz"][i].as_py()
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        r = z["image_r"].astype(np.float32)
    central_r = r[28:36, 28:36].max()
    training_central.append(central_r)

print(f"Central r-band max: min={min(training_central):.4f}, max={max(training_central):.4f}, mean={np.mean(training_central):.4f}")
print(f"Samples with central_r > 0.1: {sum(1 for x in training_central if x > 0.1)} / {len(training_central)}")
print(f"Samples with central_r > 0.05: {sum(1 for x in training_central if x > 0.05)} / {len(training_central)}")
print(f"Samples with central_r > 0.02: {sum(1 for x in training_central if x > 0.02)} / {len(training_central)}")

# Anchor lenses
anchor_dir = Path("/lambda/nfs/darkhaloscope-training-dc/anchor_cutouts/known_lenses")
anchor_central = []
for fits_file in sorted(anchor_dir.glob("*.fits")):
    with fits.open(fits_file) as hdu:
        data = hdu[0].data.astype(np.float32)
    central_r = data[1, 28:36, 28:36].max()  # r-band is index 1
    anchor_central.append(central_r)

print()
print("=== ANCHOR LENSES (real SLACS/BELLS) ===")
print(f"Central r-band max: min={min(anchor_central):.4f}, max={max(anchor_central):.4f}, mean={np.mean(anchor_central):.4f}")
print(f"Lenses with central_r > 0.1: {sum(1 for x in anchor_central if x > 0.1)} / {len(anchor_central)}")
print(f"Lenses with central_r > 0.05: {sum(1 for x in anchor_central if x > 0.05)} / {len(anchor_central)}")
print(f"Lenses with central_r > 0.02: {sum(1 for x in anchor_central if x > 0.02)} / {len(anchor_central)}")

print()
print("=== DISTRIBUTION COMPARISON ===")
print(f"Training: 25th={np.percentile(training_central, 25):.4f}, 50th={np.percentile(training_central, 50):.4f}, 75th={np.percentile(training_central, 75):.4f}")
print(f"Anchors:  25th={np.percentile(anchor_central, 25):.4f}, 50th={np.percentile(anchor_central, 50):.4f}, 75th={np.percentile(anchor_central, 75):.4f}")

print()
print("=== CONCLUSION ===")
if np.mean(training_central) > np.mean(anchor_central) * 5:
    print("PROBLEM: Training data has MUCH stronger central signals than real lenses!")
    print("This explains the sim-to-real gap. Our injected arcs are too bright.")
else:
    print("Training and anchor signals are similar in magnitude.")

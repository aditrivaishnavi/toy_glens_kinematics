#!/usr/bin/env python3
"""
Paired Controls Pilot: Generate 10K Paired Samples

This script creates a pilot dataset of paired positive/control samples
by re-fetching the base cutouts from S3 for existing positives.

Purpose:
1. Verify the salvage approach works
2. Run theta-aware gates on paired data
3. Confirm shortcut is eliminated before full-scale salvage

NOTE: This script runs on emr-launcher which has S3 access for re-fetching cutouts.
It reads positive samples from Lambda NFS data (synced to local) which has 3-band stamps.
"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import boto3
import io
from astropy.io import fits
from astropy.wcs import WCS
import json
from datetime import datetime, timezone
import os
import pandas as pd

RESULTS = {
    "test": "paired_pilot",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "n_target": 5000,  # Reduced for faster testing
}

print("=" * 70)
print("PAIRED CONTROLS PILOT: 5K SAMPLES")
print("=" * 70)

# Configuration
STAMP_SIZE = 64
S3_BUCKET = "darkhaloscope"
COADD_PREFIX = "dr10/coadd_cache"
N_PAIRS = 5000  # Generate 5K pairs for pilot
OUTPUT_DIR = "/data/paired_pilot"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# We'll read a sample of positives from S3 and re-fetch their base cutouts
# First, let's get the metadata from S3 (not the stamps, just coords)
print("\nReading positive sample metadata from S3...")
s3 = boto3.client('s3')

# Read from metrics (smaller files, have all coords)
prefix = "phase4_pipeline/phase4c/v5_cosmos_corrected/phase4c/cosmos_corrected/stamps/train_stamp64_bandsgrz_cosmos_corrected/region_split=train/"

response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, MaxKeys=3)
parquet_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.parquet')]

if not parquet_files:
    print("ERROR: No parquet files found!")
    exit(1)

print(f"Found {len(parquet_files)} parquet files")

# Read first parquet file
all_dfs = []
for key in parquet_files[:2]:  # Read 2 files for more samples
    print(f"Reading {key}...")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    df = pq.read_table(io.BytesIO(obj['Body'].read())).to_pandas()
    all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)
print(f"Columns: {list(df.columns)}")
print(f"Total rows: {len(df)}")

# Filter to positives only (is_control == 0)
positives = df[df['is_control'] == 0].copy()
print(f"Positives: {len(positives)}")

# Sample for pilot
np.random.seed(42)
sample_size = min(N_PAIRS, len(positives))
pilot_positives = positives.sample(n=sample_size)
print(f"Pilot sample: {sample_size} positives")

def decode_stamp(blob):
    """Decode stored NPZ. Handles both 3-band and single-band formats."""
    if blob is None:
        return None
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        keys = list(z.keys())
        if "image_g" in keys and "image_r" in keys and "image_z" in keys:
            # 3-band format
            g = z["image_g"].astype(np.float32)
            r = z["image_r"].astype(np.float32)
            zb = z["image_z"].astype(np.float32)
            return np.stack([g, r, zb], axis=0)
        elif "image_r" in keys:
            # Single-band r format - replicate for 3 channels
            r = z["image_r"].astype(np.float32)
            return np.stack([r, r, r], axis=0)  # Replicate r for all bands
        else:
            # Unknown format
            arr = z[keys[0]].astype(np.float32)
            if arr.ndim == 2:
                return np.stack([arr, arr, arr], axis=0)
            return arr

def encode_stamp(img_grz):
    """Encode (3, H, W) array to NPZ bytes."""
    bio = io.BytesIO()
    np.savez_compressed(bio, 
                        image_g=img_grz[0].astype(np.float32),
                        image_r=img_grz[1].astype(np.float32),
                        image_z=img_grz[2].astype(np.float32))
    bio.seek(0)
    return bio.read()

def fetch_base_cutout(ra, dec, brickname, bands=['g', 'r', 'z']):
    """Fetch base cutout from S3 (without injection)."""
    images = []
    for band in bands:
        s3_key = f"{COADD_PREFIX}/{brickname}/legacysurvey-{brickname}-image-{band}.fits.fz"
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            fits_bytes = obj['Body'].read()
            with fits.open(io.BytesIO(fits_bytes)) as hdul:
                img_data = hdul[1].data
                wcs = WCS(hdul[1].header)
                x, y = wcs.world_to_pixel_values(ra, dec)
                x, y = int(round(float(x))), int(round(float(y)))
                half = STAMP_SIZE // 2
                x0, x1 = x - half, x + half
                y0, y1 = y - half, y + half
                if x0 < 0 or y0 < 0 or x1 > img_data.shape[1] or y1 > img_data.shape[0]:
                    return None
                cutout = img_data[y0:y1, x0:x1].astype(np.float32)
                images.append(cutout)
        except Exception as e:
            return None
    return np.stack(images, axis=0)

# Generate paired samples
print("\nGenerating paired samples...")
paired_rows = []
n_success = 0
n_failed = 0

for idx, (_, row) in enumerate(pilot_positives.iterrows()):
    if idx % 500 == 0:
        print(f"  Progress: {idx}/{sample_size} ({n_success} success, {n_failed} failed)")
    
    ra = row['ra']
    dec = row['dec']
    brickname = row['brickname']
    
    # Get the positive stamp (with injection)
    pos_stamp_blob = row['stamp_npz']
    pos_stamp = decode_stamp(pos_stamp_blob)
    if pos_stamp is None:
        n_failed += 1
        continue
    
    # Fetch the base cutout (without injection) = paired control
    ctrl_stamp = fetch_base_cutout(ra, dec, brickname)
    if ctrl_stamp is None:
        n_failed += 1
        continue
    
    # Create pair_id
    pair_id = f"pair_{idx}"
    
    # Copy positive row and add pair_id
    pos_row = row.to_dict()
    pos_row['pair_id'] = pair_id
    pos_row['is_paired'] = 1
    paired_rows.append(pos_row)
    
    # Create control row with same metadata but different stamp and is_control
    ctrl_row = row.to_dict()
    ctrl_row['pair_id'] = pair_id
    ctrl_row['is_paired'] = 1
    ctrl_row['is_control'] = 1
    ctrl_row['stamp_npz'] = encode_stamp(ctrl_stamp)
    # Clear injection-specific columns for control
    for col in ['theta_e_arcsec', 'arc_snr', 'src_dmag', 'src_reff_arcsec', 
                'cosmos_index', 'magnification', 'physics_valid']:
        if col in ctrl_row:
            ctrl_row[col] = None
    paired_rows.append(ctrl_row)
    
    n_success += 1

print(f"\nGeneration complete:")
print(f"  Successful pairs: {n_success}")
print(f"  Failed: {n_failed}")
print(f"  Total rows: {len(paired_rows)}")

# Convert to DataFrame and save
import pandas as pd
paired_df = pd.DataFrame(paired_rows)

# Save as parquet
output_path = f"{OUTPUT_DIR}/paired_pilot.parquet"
paired_df.to_parquet(output_path, index=False)
print(f"\nSaved to {output_path}")

# Verify core brightness matching
print("\n" + "=" * 70)
print("CORE BRIGHTNESS VERIFICATION")
print("=" * 70)

# Quick check: compare core brightness between positives and controls
positives_in_pilot = paired_df[paired_df['is_control'] == 0]
controls_in_pilot = paired_df[paired_df['is_control'] == 1]

pos_core_brightness = []
ctrl_core_brightness = []

for _, row in positives_in_pilot.head(500).iterrows():
    stamp = decode_stamp(row['stamp_npz'])
    if stamp is not None:
        # Central 8 pixel radius
        h, w = stamp.shape[-2:]
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        core_mask = ((yy - cy)**2 + (xx - cx)**2) < 8**2
        pos_core_brightness.append(float(stamp[1][core_mask].mean()))

for _, row in controls_in_pilot.head(500).iterrows():
    stamp = decode_stamp(row['stamp_npz'])
    if stamp is not None:
        h, w = stamp.shape[-2:]
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        core_mask = ((yy - cy)**2 + (xx - cx)**2) < 8**2
        ctrl_core_brightness.append(float(stamp[1][core_mask].mean()))

pos_mean = np.mean(pos_core_brightness)
ctrl_mean = np.mean(ctrl_core_brightness)
ratio = pos_mean / ctrl_mean if ctrl_mean != 0 else float('inf')

print(f"Positives core_mean_r: {pos_mean:.6f}")
print(f"Controls core_mean_r:  {ctrl_mean:.6f}")
print(f"Ratio: {ratio:.4f}")

if 0.95 <= ratio <= 1.05:
    print("\n✓ PASS: Core brightness matched (ratio within 5%)")
    brightness_pass = True
else:
    print(f"\n✗ FAIL: Core brightness NOT matched (ratio={ratio:.4f})")
    brightness_pass = False

# But wait - positives have injection, controls don't!
# The ratio should NOT be 1.0 unless injection adds no core flux
# Let's check what the injection contributes

print("\n" + "=" * 70)
print("INJECTION CONTRIBUTION ANALYSIS")
print("=" * 70)

# For a true paired test, we need to compare:
# - Control core brightness (base only)
# - What the control WOULD have if we subtract what injection adds

# Actually, the key insight is:
# - In UNPAIRED data: positives were brighter because they came from DIFFERENT LRGs
# - In PAIRED data: the SAME LRG is used, so base brightness is identical
# - The only difference is the injected arc

# So we should check: is the DIFFERENCE due to arc (annulus) or core?
# If injection only adds to annulus, core should match

injection_core_contribution = []
for pair_id in paired_df['pair_id'].unique()[:200]:
    pair = paired_df[paired_df['pair_id'] == pair_id]
    if len(pair) != 2:
        continue
    pos = pair[pair['is_control'] == 0].iloc[0]
    ctrl = pair[pair['is_control'] == 1].iloc[0]
    
    pos_stamp = decode_stamp(pos['stamp_npz'])
    ctrl_stamp = decode_stamp(ctrl['stamp_npz'])
    
    if pos_stamp is None or ctrl_stamp is None:
        continue
    
    h, w = pos_stamp.shape[-2:]
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    core_mask = ((yy - cy)**2 + (xx - cx)**2) < 8**2
    
    # Injection contribution in core = pos - ctrl
    core_diff = float(pos_stamp[1][core_mask].mean() - ctrl_stamp[1][core_mask].mean())
    injection_core_contribution.append(core_diff)

mean_injection_core = np.mean(injection_core_contribution)
std_injection_core = np.std(injection_core_contribution)

print(f"Mean injection contribution to core: {mean_injection_core:.6f}")
print(f"Std: {std_injection_core:.6f}")
print(f"As fraction of control core: {mean_injection_core / ctrl_mean:.2%}")

if abs(mean_injection_core / ctrl_mean) < 0.1:
    print("\n✓ Injection adds <10% to core brightness")
else:
    print(f"\n⚠ Injection adds {abs(mean_injection_core / ctrl_mean):.1%} to core brightness")
    print("  This is expected for small theta_e where arc overlaps core")

# Save results
RESULTS["n_success"] = n_success
RESULTS["n_failed"] = n_failed
RESULTS["pos_core_mean"] = pos_mean
RESULTS["ctrl_core_mean"] = ctrl_mean
RESULTS["core_ratio"] = ratio
RESULTS["mean_injection_core_contribution"] = mean_injection_core
RESULTS["output_path"] = output_path

with open(f"{OUTPUT_DIR}/pilot_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR}/pilot_results.json")
print("\nNext step: Run theta-aware gates on this paired data")

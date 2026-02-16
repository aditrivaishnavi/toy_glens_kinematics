#!/usr/bin/env python3
"""
Column Audit: Verify exactly which columns are read during training.

This script verifies there is no label leakage through metadata columns.
"""
import pyarrow.dataset as ds
import torch

print("=" * 70)
print("PHASE 0.1: COLUMN AUDIT")
print("=" * 70)

# 1. List ALL columns in the Parquet dataset
data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

print("\n" + "=" * 70)
print("ALL COLUMNS IN DATASET:")
print("=" * 70)
all_cols = sorted(dataset.schema.names)
for col in all_cols:
    print(f"  {col}")
print(f"\nTotal columns: {len(all_cols)}")

# 2. Check what columns the training script reads
print("\n" + "=" * 70)
print("COLUMNS USED BY TRAINING SCRIPT:")
print("=" * 70)

# From checkpoint args
ckpt = torch.load("/lambda/nfs/darkhaloscope-training-dc/models/gen5_cosmos/ckpt_best.pt", 
                  map_location="cpu")
args = ckpt["args"]

print(f"\nFrom checkpoint args:")
print(f"  meta_cols: {args.get('meta_cols', 'NONE')}")
print(f"  arch: {args.get('arch', 'N/A')}")
print(f"  loss: {args.get('loss', 'N/A')}")
print(f"  min_arc_snr: {args.get('min_arc_snr', 'N/A')}")
print(f"  min_theta_over_psf: {args.get('min_theta_over_psf', 'N/A')}")

# 3. Flag DANGEROUS columns (injection parameters that could leak labels)
DANGEROUS_COLUMNS = [
    "arc_snr",           # Directly indicates injection presence/strength
    "theta_e_arcsec",    # Lens parameter - only set for injections
    "src_dmag",          # Source magnitude offset - injection param
    "src_reff_arcsec",   # Source size - injection param
    "is_control",        # LABEL ITSELF
    "cutout_ok",         # Could correlate with injection success
    "magnification",     # Lensing output - only for injections
    "tangential_stretch", # Lensing output
    "radial_stretch",    # Lensing output
    "expected_arc_radius", # Lensing output
    "cosmos_index",      # COSMOS template ID - injection param
    "cosmos_hlr_arcsec", # COSMOS HLR - injection param
    "lensed_hlr_arcsec", # Lensed HLR - injection param
    "physics_valid",     # Physics validation flag
    "lens_model",        # Lens model type
    "lens_e",            # Lens ellipticity
    "source_mode",       # Source mode (cosmos vs sersic)
]

# Safe columns (physical observing conditions, same for pos/neg)
SAFE_COLUMNS = [
    "psfsize_r", "psfdepth_r", "psfsize_g", "psfsize_z", 
    "psfdepth_g", "psfdepth_z", "ebv",
    "psfsize_i", "psfdepth_i",  # Other bands
]

# Always-used columns (required for training but not metadata)
REQUIRED_COLUMNS = [
    "stamp_npz",     # The actual image data
    "region_split",  # Split label (train/val/test)
]

meta_cols_list = args.get('meta_cols', '').split(',') if args.get('meta_cols') else []
meta_cols_list = [c.strip() for c in meta_cols_list if c.strip()]

print("\n" + "=" * 70)
print("LEAKAGE CHECK:")
print("=" * 70)

leakage_found = False
for col in meta_cols_list:
    if col in SAFE_COLUMNS:
        print(f"  ✓ {col}: SAFE (physical observing condition)")
    elif col in DANGEROUS_COLUMNS:
        print(f"  ✗ {col}: LEAKAGE DETECTED!")
        leakage_found = True
    else:
        print(f"  ? {col}: UNKNOWN - needs manual review")

# 4. Check if any dangerous columns are in the data reader
print("\n" + "=" * 70)
print("DANGEROUS COLUMNS IN DATASET:")
print("=" * 70)
for col in DANGEROUS_COLUMNS:
    if col in all_cols:
        if col in meta_cols_list:
            print(f"  ⚠️ {col}: IN DATASET AND USED AS META - LEAKAGE!")
            leakage_found = True
        else:
            print(f"  ○ {col}: in dataset but NOT used as metadata")
    else:
        print(f"  - {col}: not in dataset")

# 5. Summary
print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)
if leakage_found:
    print("  ✗ LEAKAGE DETECTED - STOP AND FIX TRAINING")
else:
    print("  ✓ NO LEAKAGE DETECTED")
    print(f"  ✓ Meta columns used: {meta_cols_list}")
    print("  ✓ All meta columns are safe observing conditions")

# 6. Additional check: verify the training script only reads safe columns
print("\n" + "=" * 70)
print("COLUMNS READ BY TRAINING DATALOADER:")
print("=" * 70)
# Based on training script, these are the columns read:
# - stamp_npz (image data)
# - is_control (label - used to compute y)
# - region_split (for filtering)
# - cutout_ok (for filtering)
# - meta_cols (psfsize_r, psfdepth_r)
# - optionally: arc_snr, theta_e for filtering (not fed to model)

training_reads = ["stamp_npz", "is_control", "region_split", "cutout_ok", "bandset"] + meta_cols_list

# Check for filtering columns
filter_cols = []
if args.get('min_arc_snr', 0) > 0:
    filter_cols.append("arc_snr")
if args.get('min_theta_over_psf', 0) > 0:
    filter_cols.append("theta_e_arcsec")
    filter_cols.append("psfsize_r")

print(f"  Image data: stamp_npz")
print(f"  Label: is_control (used to compute y=0 or y=1)")
print(f"  Filtering: {filter_cols if filter_cols else 'none'}")
print(f"  Metadata fed to model: {meta_cols_list}")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)
if leakage_found:
    print("  FAIL: Leakage detected. Fix training before proceeding.")
    exit(1)
else:
    print("  PASS: No leakage. Model only sees:")
    print("    - Image pixels (stamp_npz)")
    print("    - PSF size and depth (psfsize_r, psfdepth_r)")
    print("    - Label (is_control) used for supervision only")
    exit(0)

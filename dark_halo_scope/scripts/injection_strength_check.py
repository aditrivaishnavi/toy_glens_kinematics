#!/usr/bin/env python3
"""
Check the strength of injected signal vs real lens signal.
"""
import numpy as np
from pathlib import Path
from astropy.io import fits
import io
import pyarrow.dataset as ds

print("=" * 70)
print("INJECTION STRENGTH ANALYSIS")
print("=" * 70)

# ============================================================
# Part 1: What is the injection strength in training data?
# ============================================================
print("\n=== TRAINING DATA: INJECTION STRENGTH ===")

data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

# Get paired positive/negative from same location (if available)
pos_filter = (ds.field("region_split") == "test") & (ds.field("is_control") == 0) & (ds.field("cutout_ok") == 1)
pos_table = dataset.to_table(filter=pos_filter, columns=["stamp_npz", "arc_snr", "theta_e_arcsec", "src_dmag", "task_id"])
pos_table = pos_table.slice(0, 100)

def decode_stamp(blob):
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        g = z["image_g"].astype(np.float32)
        r = z["image_r"].astype(np.float32)
        zb = z["image_z"].astype(np.float32)
    return np.stack([g, r, zb], axis=0)

# Calculate the injected flux
print("\nInjected source magnitudes and arc SNR:")
print(f"{'src_rmag':<10} {'arc_snr':<10} {'theta_e':<10} {'r_max':<10} {'r_center':<10}")
print("-" * 60)

injection_stats = []
for i in range(min(50, pos_table.num_rows)):
    blob = pos_table["stamp_npz"][i].as_py()
    snr = pos_table["arc_snr"][i].as_py()
    te = pos_table["theta_e_arcsec"][i].as_py()
    src_mag = pos_table["src_dmag"][i].as_py() if "src_dmag" in pos_table.column_names else None
    
    img = decode_stamp(blob)
    r_max = img[1].max()
    r_center = img[1, 28:36, 28:36].max()
    
    injection_stats.append({
        "src_mag": src_mag,
        "arc_snr": snr,
        "theta_e": te,
        "r_max": r_max,
        "r_center": r_center
    })
    
    if i < 10:
        src_str = f"{src_mag:.2f}" if src_mag is not None else "N/A"
        print(f"{src_str:<10} {snr:<10.2f} {te:<10.2f} {r_max:<10.4f} {r_center:<10.4f}")

if "src_dmag" in pos_table.column_names:
    mags = [s["src_mag"] for s in injection_stats if s["src_mag"] is not None]
    print(f"\nSource magnitude distribution:")
    print(f"  min: {np.min(mags):.2f}")
    print(f"  median: {np.median(mags):.2f}")
    print(f"  max: {np.max(mags):.2f}")
    
    # Convert to flux
    flux_nmgy = 10.0 ** ((22.5 - np.array(mags)) / 2.5)
    print(f"\nSource flux distribution (nanomaggies):")
    print(f"  min: {np.min(flux_nmgy):.2f}")
    print(f"  median: {np.median(flux_nmgy):.2f}")
    print(f"  max: {np.max(flux_nmgy):.2f}")

# ============================================================
# Part 2: What is the lens galaxy brightness?
# ============================================================
print("\n=== LENS GALAXIES IN SLACS/BELLS ===")

# These are massive ellipticals - typically very bright
# SLACS lens galaxies are ~mag 17-19 in r-band
print("SLACS lens galaxies are typically r~17-19 mag")
print("Source arcs are typically r~21-24 mag")
print("Arc-to-lens flux ratio: 10^((17-23)/2.5) = 0.004 (arcs are 250x fainter)")

# ============================================================
# Part 3: Compare with anchor cutouts
# ============================================================
print("\n=== ANCHOR LENS SIGNAL ANALYSIS ===")

anchor_dir = Path("/lambda/nfs/darkhaloscope-training-dc/anchor_cutouts/known_lenses")

print("\nAnchor lens brightness:")
print(f"{'Name':<30} {'r_max':<10} {'r_center':<10} {'r_annulus':<10}")
print("-" * 70)

y, x = np.ogrid[:64, :64]
r = np.sqrt((y - 32)**2 + (x - 32)**2)
annulus_mask = (r >= 10) & (r <= 20)

for fits_file in sorted(anchor_dir.glob("*.fits"))[:15]:
    with fits.open(fits_file) as hdu:
        img = hdu[0].data.astype(np.float32)
    r_max = img[1].max()
    r_center = img[1, 28:36, 28:36].max()
    r_annulus = img[1, annulus_mask].mean()
    
    print(f"{fits_file.stem:<30} {r_max:<10.4f} {r_center:<10.4f} {r_annulus:<10.6f}")

# ============================================================
# Part 4: Direct comparison
# ============================================================
print("\n=== DIRECT COMPARISON ===")

# Training positives
pos_r_centers = [s["r_center"] for s in injection_stats]
pos_arc_snr = [s["arc_snr"] for s in injection_stats if s["arc_snr"] is not None]

print(f"Training positives (n={len(pos_r_centers)}):")
print(f"  Central r_max: mean={np.mean(pos_r_centers):.4f}, std={np.std(pos_r_centers):.4f}")
print(f"  Arc SNR: mean={np.mean(pos_arc_snr):.1f}, median={np.median(pos_arc_snr):.1f}")

# Anchor lenses
anchor_centers = []
for fits_file in anchor_dir.glob("*.fits"):
    with fits.open(fits_file) as hdu:
        img = hdu[0].data.astype(np.float32)
    anchor_centers.append(img[1, 28:36, 28:36].max())

print(f"\nAnchor lenses (n={len(anchor_centers)}):")
print(f"  Central r_max: mean={np.mean(anchor_centers):.4f}, std={np.std(anchor_centers):.4f}")

ratio = np.mean(pos_r_centers) / np.mean(anchor_centers)
print(f"\n  Ratio (training_pos / anchor): {ratio:.1f}x")

print("\n=== CONCLUSION ===")
if ratio > 2:
    print("Training positives have BRIGHTER centers than real anchor lenses.")
    print("This explains why model fails on real data - the signal distribution is different.")
elif ratio < 0.5:
    print("Training positives have FAINTER centers than real anchor lenses.")
    print("The model might be learning low-brightness features.")
else:
    print("Center brightness is similar. Issue may be elsewhere.")

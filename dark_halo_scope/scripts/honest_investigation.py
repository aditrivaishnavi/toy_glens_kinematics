#!/usr/bin/env python3
"""
Honest Investigation: What's Really Going On?

Questions to answer:
1. What features does the model respond to?
2. Are anchor cutouts correctly formatted?
3. Does the COSMOS injection produce realistic signals?
4. Is there a data format mismatch?
"""
import numpy as np
import io
from pathlib import Path
from astropy.io import fits
import pyarrow.dataset as ds

print("=" * 70)
print("HONEST INVESTIGATION: GEN5 REAL LENS PERFORMANCE")
print("=" * 70)

# ============================================================
# PART 1: Compare data formats between training and anchors
# ============================================================
print("\n" + "=" * 70)
print("PART 1: DATA FORMAT COMPARISON")
print("=" * 70)

# Load a few training samples
data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

# Get some positive (injected) and negative (control) samples
pos_filter = (ds.field("region_split") == "test") & (ds.field("is_control") == 0) & (ds.field("cutout_ok") == 1)
neg_filter = (ds.field("region_split") == "test") & (ds.field("is_control") == 1) & (ds.field("cutout_ok") == 1)

pos_table = dataset.to_table(filter=pos_filter, columns=["stamp_npz", "arc_snr", "theta_e_arcsec"])
pos_table = pos_table.slice(0, 10)

neg_table = dataset.to_table(filter=neg_filter, columns=["stamp_npz"])
neg_table = neg_table.slice(0, 10)

def decode_training_stamp(blob):
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        g = z["image_g"].astype(np.float32)
        r = z["image_r"].astype(np.float32)
        zb = z["image_z"].astype(np.float32)
    return np.stack([g, r, zb], axis=0)

print("\n--- Training POSITIVE samples (injected lenses) ---")
train_pos_stats = []
for i in range(min(10, pos_table.num_rows)):
    blob = pos_table["stamp_npz"][i].as_py()
    img = decode_training_stamp(blob)
    snr = pos_table["arc_snr"][i].as_py()
    te = pos_table["theta_e_arcsec"][i].as_py()
    
    stats = {
        "shape": img.shape,
        "dtype": str(img.dtype),
        "g_minmax": (img[0].min(), img[0].max()),
        "r_minmax": (img[1].min(), img[1].max()),
        "z_minmax": (img[2].min(), img[2].max()),
        "r_mean": img[1].mean(),
        "r_std": img[1].std(),
        "arc_snr": snr,
        "theta_e": te
    }
    train_pos_stats.append(stats)
    if i < 3:
        print(f"  Sample {i}: shape={img.shape}, r_range=[{img[1].min():.4f}, {img[1].max():.4f}], "
              f"arc_snr={snr:.1f}, theta_e={te:.2f}")

print("\n--- Training NEGATIVE samples (controls) ---")
train_neg_stats = []
for i in range(min(10, neg_table.num_rows)):
    blob = neg_table["stamp_npz"][i].as_py()
    img = decode_training_stamp(blob)
    
    stats = {
        "shape": img.shape,
        "r_minmax": (img[1].min(), img[1].max()),
        "r_mean": img[1].mean(),
        "r_std": img[1].std(),
    }
    train_neg_stats.append(stats)
    if i < 3:
        print(f"  Sample {i}: shape={img.shape}, r_range=[{img[1].min():.4f}, {img[1].max():.4f}]")

# Load anchor samples
anchor_dir = Path("/lambda/nfs/darkhaloscope-training-dc/anchor_cutouts/known_lenses")

print("\n--- ANCHOR samples (real SLACS/BELLS) ---")
anchor_stats = []
for fits_file in sorted(anchor_dir.glob("*.fits"))[:5]:
    with fits.open(fits_file) as hdu:
        img = hdu[0].data.astype(np.float32)
    
    stats = {
        "name": fits_file.stem,
        "shape": img.shape,
        "dtype": str(img.dtype),
        "g_minmax": (img[0].min(), img[0].max()),
        "r_minmax": (img[1].min(), img[1].max()),
        "z_minmax": (img[2].min(), img[2].max()),
        "r_mean": img[1].mean(),
        "r_std": img[1].std(),
    }
    anchor_stats.append(stats)
    print(f"  {fits_file.stem}: shape={img.shape}, r_range=[{img[1].min():.4f}, {img[1].max():.4f}]")

# ============================================================
# PART 2: Check if training positives look different from negatives
# ============================================================
print("\n" + "=" * 70)
print("PART 2: POSITIVE vs NEGATIVE DISCRIMINATION")
print("=" * 70)

pos_r_max = [s["r_minmax"][1] for s in train_pos_stats]
neg_r_max = [s["r_minmax"][1] for s in train_neg_stats]
anchor_r_max = [s["r_minmax"][1] for s in anchor_stats]

print(f"Training POSITIVES r_max: mean={np.mean(pos_r_max):.4f}, std={np.std(pos_r_max):.4f}")
print(f"Training NEGATIVES r_max: mean={np.mean(neg_r_max):.4f}, std={np.std(neg_r_max):.4f}")
print(f"ANCHOR LENSES r_max:      mean={np.mean(anchor_r_max):.4f}, std={np.std(anchor_r_max):.4f}")

if np.mean(pos_r_max) > np.mean(neg_r_max) * 1.5:
    print("\n⚠️  WARNING: Training positives have significantly higher r_max than negatives!")
    print("    The model may be learning 'brighter = lens' as a shortcut.")

# ============================================================
# PART 3: Check arc_snr distribution
# ============================================================
print("\n" + "=" * 70)
print("PART 3: ARC SNR DISTRIBUTION IN TRAINING DATA")
print("=" * 70)

# Get arc_snr distribution for positives
snr_filter = (ds.field("region_split") == "test") & (ds.field("is_control") == 0) & (ds.field("cutout_ok") == 1)
snr_table = dataset.to_table(filter=snr_filter, columns=["arc_snr"])
snr_table = snr_table.slice(0, 10000)

snr_values = [snr_table["arc_snr"][i].as_py() for i in range(snr_table.num_rows) if snr_table["arc_snr"][i].as_py() is not None]
snr_values = np.array(snr_values)

print(f"arc_snr distribution (n={len(snr_values)}):")
print(f"  min: {np.min(snr_values):.2f}")
print(f"  25th percentile: {np.percentile(snr_values, 25):.2f}")
print(f"  median: {np.median(snr_values):.2f}")
print(f"  75th percentile: {np.percentile(snr_values, 75):.2f}")
print(f"  max: {np.max(snr_values):.2f}")
print(f"  mean: {np.mean(snr_values):.2f}")

low_snr = np.sum(snr_values < 5)
print(f"\n  Samples with arc_snr < 5: {low_snr} ({100*low_snr/len(snr_values):.1f}%)")
print(f"  Samples with arc_snr < 1: {np.sum(snr_values < 1)} ({100*np.sum(snr_values < 1)/len(snr_values):.1f}%)")

# ============================================================
# PART 4: What do high-scoring anchor lenses look like?
# ============================================================
print("\n" + "=" * 70)
print("PART 4: HIGH vs LOW SCORING ANCHOR COMPARISON")
print("=" * 70)

# High scorer: SDSSJ1205+4910 (p_lens=0.91)
# Low scorer: SDSSJ2321-0939 (p_lens=very low)

high_path = anchor_dir / "SDSSJ1205+4910.fits"
low_path = anchor_dir / "SDSSJ2321-0939.fits"

if high_path.exists() and low_path.exists():
    with fits.open(high_path) as hdu:
        high_img = hdu[0].data.astype(np.float32)
    with fits.open(low_path) as hdu:
        low_img = hdu[0].data.astype(np.float32)
    
    print("\nHigh scorer (SDSSJ1205+4910, p_lens=0.91):")
    print(f"  r-band: min={high_img[1].min():.4f}, max={high_img[1].max():.4f}, mean={high_img[1].mean():.4f}")
    print(f"  Central 8x8: max={high_img[1, 28:36, 28:36].max():.4f}")
    print(f"  Total flux in r-band: {high_img[1].sum():.4f}")
    
    print("\nLow scorer (SDSSJ2321-0939, p_lens=very low):")
    print(f"  r-band: min={low_img[1].min():.4f}, max={low_img[1].max():.4f}, mean={low_img[1].mean():.4f}")
    print(f"  Central 8x8: max={low_img[1, 28:36, 28:36].max():.4f}")
    print(f"  Total flux in r-band: {low_img[1].sum():.4f}")
    
    print("\nDifference:")
    print(f"  r_max ratio: {high_img[1].max() / max(low_img[1].max(), 1e-10):.1f}x")
    print(f"  Total flux ratio: {high_img[1].sum() / max(abs(low_img[1].sum()), 1e-10):.1f}x")

# ============================================================
# PART 5: Check if anchor cutouts have the arc visible
# ============================================================
print("\n" + "=" * 70)
print("PART 5: ARC VISIBILITY IN ANCHOR CUTOUTS")
print("=" * 70)

# For each anchor, compute: is there extra flux in an annulus around center?
print("\nAnnulus analysis (looking for arc signal):")
print(f"{'Name':<25} {'Central':<10} {'Annulus':<10} {'Outer':<10} {'Arc Signal?':<12}")
print("-" * 70)

for fits_file in sorted(anchor_dir.glob("*.fits"))[:15]:
    with fits.open(fits_file) as hdu:
        img = hdu[0].data.astype(np.float32)
    
    r_band = img[1]
    h, w = r_band.shape
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r = np.sqrt((y - cy)**2 + (x - cx)**2)
    
    # Three regions: center (r<10), annulus (10<r<20), outer (r>25)
    center_mask = r < 10
    annulus_mask = (r >= 10) & (r < 20)
    outer_mask = r >= 25
    
    center_flux = r_band[center_mask].mean()
    annulus_flux = r_band[annulus_mask].mean()
    outer_flux = r_band[outer_mask].mean()
    
    # Arc should be visible as: annulus_flux > outer_flux
    arc_signal = "YES" if annulus_flux > outer_flux * 1.5 else "no"
    
    print(f"{fits_file.stem:<25} {center_flux:<10.5f} {annulus_flux:<10.5f} {outer_flux:<10.5f} {arc_signal:<12}")

# ============================================================
# PART 6: The Big Question - Is the model learning lensing?
# ============================================================
print("\n" + "=" * 70)
print("PART 6: DIAGNOSIS")
print("=" * 70)

print("""
Based on this analysis, here are the key findings:

1. DATA FORMAT: Training data and anchors are in similar format (3-channel, float32)

2. BRIGHTNESS DIFFERENCE: Need to check if training positives are systematically 
   brighter than negatives (would indicate model learning a shortcut)

3. ARC SNR: Check if we're training on low-SNR arcs or only high-SNR arcs

4. ANCHOR VISIBILITY: Check if the arcs in anchor cutouts are actually visible
   in the DR10 imaging

The fundamental question: Are SLACS/BELLS arcs detectable in DR10 imaging at all?
""")

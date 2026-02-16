#!/usr/bin/env python3
"""
Final diagnosis: What's the actual problem?
"""
import numpy as np
from pathlib import Path
from astropy.io import fits
import io
import pyarrow.dataset as ds

print("=" * 70)
print("FINAL DIAGNOSIS")
print("=" * 70)

# ============================================================
# Part 1: Compare training positives vs negatives (same base)
# ============================================================
print("\n=== TRAINING: POSITIVES vs NEGATIVES ===")

data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

pos_filter = (ds.field("region_split") == "test") & (ds.field("is_control") == 0) & (ds.field("cutout_ok") == 1)
neg_filter = (ds.field("region_split") == "test") & (ds.field("is_control") == 1) & (ds.field("cutout_ok") == 1)

pos_table = dataset.to_table(filter=pos_filter, columns=["stamp_npz"])
neg_table = dataset.to_table(filter=neg_filter, columns=["stamp_npz"])

def decode_stamp(blob):
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        g = z["image_g"].astype(np.float32)
        r = z["image_r"].astype(np.float32)
        zb = z["image_z"].astype(np.float32)
    return np.stack([g, r, zb], axis=0)

# Sample training data
pos_centers = []
neg_centers = []
for i in range(min(100, pos_table.num_rows)):
    img = decode_stamp(pos_table["stamp_npz"][i].as_py())
    pos_centers.append(img[1, 28:36, 28:36].max())
for i in range(min(100, neg_table.num_rows)):
    img = decode_stamp(neg_table["stamp_npz"][i].as_py())
    neg_centers.append(img[1, 28:36, 28:36].max())

print(f"Training POSITIVES (n={len(pos_centers)}):")
print(f"  Central r_max: mean={np.mean(pos_centers):.4f}, std={np.std(pos_centers):.4f}")
print(f"Training NEGATIVES (n={len(neg_centers)}):")  
print(f"  Central r_max: mean={np.mean(neg_centers):.4f}, std={np.std(neg_centers):.4f}")
print(f"Difference: {np.mean(pos_centers) - np.mean(neg_centers):.4f}")
print(f"Ratio: {np.mean(pos_centers) / np.mean(neg_centers):.2f}x")

# ============================================================
# Part 2: What are SLACS/BELLS galaxies?
# ============================================================
print("\n=== SLACS/BELLS LENS GALAXIES ===")
print("""
SLACS/BELLS selection:
- Found via SDSS spectroscopy: two redshifts in single fiber
- Lens galaxies are MASSIVE ELLIPTICALS (luminous red galaxies)
- Typical lens galaxy: Mr ~ -22 to -23 (very luminous)
- At z~0.2, this corresponds to r-band flux ~0.5-2 mJy (5000-20000 nanomaggies)

Our training LRGs:
- Selected from DR10 LRG sample
- Similar population: massive red galaxies
""")

# ============================================================
# Part 3: Check the actual pixel values in anchor vs training
# ============================================================
print("\n=== RAW PIXEL VALUE COMPARISON (nanomaggies) ===")

anchor_dir = Path("/lambda/nfs/darkhaloscope-training-dc/anchor_cutouts/known_lenses")

print("\nAnchor cutouts (r-band, central 8x8 max):")
for fits_file in sorted(anchor_dir.glob("*.fits"))[:10]:
    with fits.open(fits_file) as hdu:
        img = hdu[0].data.astype(np.float32)
    center = img[1, 28:36, 28:36].max()
    print(f"  {fits_file.stem}: {center:.4f} nMgy")

print("\nTraining samples (r-band, central 8x8 max):")
for i in range(5):
    img = decode_stamp(pos_table["stamp_npz"][i].as_py())
    center = img[1, 28:36, 28:36].max()
    print(f"  Positive {i}: {center:.4f} nMgy")
for i in range(5):
    img = decode_stamp(neg_table["stamp_npz"][i].as_py())
    center = img[1, 28:36, 28:36].max()
    print(f"  Negative {i}: {center:.4f} nMgy")

# ============================================================
# Part 4: THE KEY QUESTION
# ============================================================
print("\n" + "=" * 70)
print("THE KEY QUESTION: Why are anchor centers fainter?")
print("=" * 70)

print("""
Possibility 1: Different galaxy populations
- SLACS/BELLS are at z~0.2-0.5 (more distant, appear fainter)
- Our training LRGs might be at z~0.1-0.3 (closer, appear brighter)

Possibility 2: Different cutout processing
- Our pipeline may have different background subtraction
- Legacy Survey cutouts may have different processing

Possibility 3: The arcs ARE faint in DR10
- SLACS/BELLS were confirmed with HST (0.05"/pix)
- DR10 is ground-based (0.262"/pix, 5x coarser)
- The arcs may be below the noise floor in DR10

Let me check the background levels...
""")

# Check background levels
print("\n=== BACKGROUND LEVEL CHECK ===")
print("Outer region (r > 25 pixels from center) statistics:")

y, x = np.ogrid[:64, :64]
r = np.sqrt((y - 32)**2 + (x - 32)**2)
outer_mask = r > 25

print("\nAnchors:")
for fits_file in sorted(anchor_dir.glob("*.fits"))[:5]:
    with fits.open(fits_file) as hdu:
        img = hdu[0].data.astype(np.float32)
    outer = img[1, outer_mask]
    print(f"  {fits_file.stem}: mean={outer.mean():.6f}, std={outer.std():.6f}")

print("\nTraining positives:")
for i in range(5):
    img = decode_stamp(pos_table["stamp_npz"][i].as_py())
    outer = img[1, outer_mask]
    print(f"  Sample {i}: mean={outer.mean():.6f}, std={outer.std():.6f}")

# ============================================================
# FINAL ANSWER
# ============================================================
print("\n" + "=" * 70)
print("FINAL ANSWER")
print("=" * 70)

#!/usr/bin/env python3
"""
Check normalization effects in detail.
"""
import numpy as np
from pathlib import Path
from astropy.io import fits
import io
import pyarrow.dataset as ds

print("=" * 70)
print("NORMALIZATION DEEP DIVE")
print("=" * 70)

def robust_mad_norm_outer(x, clip=10.0, eps=1e-6, inner_frac=0.5):
    """Same normalization as training."""
    out = np.empty_like(x, dtype=np.float32)
    h, w = x.shape[-2:]
    cy, cx = h // 2, w // 2
    ri = int(min(h, w) * inner_frac / 2)
    yy, xx = np.ogrid[:h, :w]
    outer_mask = ((yy - cy)**2 + (xx - cx)**2) > ri**2
    
    for c in range(x.shape[0]):
        v = x[c]
        outer_v = v[outer_mask]
        med = np.median(outer_v)
        mad = np.median(np.abs(outer_v - med))
        scale = 1.4826 * mad + eps
        vv = (v - med) / scale
        if clip is not None:
            vv = np.clip(vv, -clip, clip)
        out[c] = vv.astype(np.float32)
    return out

def decode_stamp(blob):
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        g = z["image_g"].astype(np.float32)
        r = z["image_r"].astype(np.float32)
        zb = z["image_z"].astype(np.float32)
    return np.stack([g, r, zb], axis=0)

# Load data
data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")
pos_filter = (ds.field("region_split") == "test") & (ds.field("is_control") == 0) & (ds.field("cutout_ok") == 1)
pos_table = dataset.to_table(filter=pos_filter, columns=["stamp_npz"])
anchor_dir = Path("/lambda/nfs/darkhaloscope-training-dc/anchor_cutouts/known_lenses")

# ============================================================
# Part 1: Compare normalization parameters
# ============================================================
print("\n=== NORMALIZATION PARAMETERS ===")

# Training sample
train_img = decode_stamp(pos_table["stamp_npz"][0].as_py())
h, w = train_img.shape[-2:]
cy, cx = h // 2, w // 2
ri = int(min(h, w) * 0.5 / 2)
yy, xx = np.ogrid[:h, :w]
outer_mask = ((yy - cy)**2 + (xx - cx)**2) > ri**2

print("Training sample normalization (r-band):")
outer_v = train_img[1, outer_mask]
med = np.median(outer_v)
mad = np.median(np.abs(outer_v - med))
scale = 1.4826 * mad + 1e-6
print(f"  Outer median: {med:.6f}")
print(f"  Outer MAD: {mad:.6f}")
print(f"  Scale (1.4826*MAD): {scale:.6f}")
print(f"  Center max (raw): {train_img[1, 28:36, 28:36].max():.6f}")
print(f"  Center max (normalized): {(train_img[1, 28:36, 28:36].max() - med) / scale:.2f}")

# Anchor sample (high scorer)
with fits.open(anchor_dir / "SDSSJ1205+4910.fits") as hdu:
    anchor_high = hdu[0].data.astype(np.float32)

print("\nAnchor SDSSJ1205+4910 (high scorer, p=0.91):")
outer_v = anchor_high[1, outer_mask]
med = np.median(outer_v)
mad = np.median(np.abs(outer_v - med))
scale = 1.4826 * mad + 1e-6
print(f"  Outer median: {med:.6f}")
print(f"  Outer MAD: {mad:.6f}")
print(f"  Scale (1.4826*MAD): {scale:.6f}")
print(f"  Center max (raw): {anchor_high[1, 28:36, 28:36].max():.6f}")
print(f"  Center max (normalized): {(anchor_high[1, 28:36, 28:36].max() - med) / scale:.2f}")

# Anchor sample (low scorer)
with fits.open(anchor_dir / "SDSSJ2321-0939.fits") as hdu:
    anchor_low = hdu[0].data.astype(np.float32)

print("\nAnchor SDSSJ2321-0939 (low scorer, p=0.01):")
outer_v = anchor_low[1, outer_mask]
med = np.median(outer_v)
mad = np.median(np.abs(outer_v - med))
scale = 1.4826 * mad + 1e-6
print(f"  Outer median: {med:.6f}")
print(f"  Outer MAD: {mad:.6f}")
print(f"  Scale (1.4826*MAD): {scale:.6f}")
print(f"  Center max (raw): {anchor_low[1, 28:36, 28:36].max():.6f}")
print(f"  Center max (normalized): {(anchor_low[1, 28:36, 28:36].max() - med) / scale:.2f}")

# ============================================================
# Part 2: Compare full normalized images
# ============================================================
print("\n=== FULL NORMALIZED IMAGE COMPARISON ===")

train_norm = robust_mad_norm_outer(train_img)
anchor_high_norm = robust_mad_norm_outer(anchor_high)
anchor_low_norm = robust_mad_norm_outer(anchor_low)

print("\nAfter normalization (r-band):")
print(f"  Training: center_max={train_norm[1, 28:36, 28:36].max():.2f}, total_max={train_norm[1].max():.2f}")
print(f"  High anchor: center_max={anchor_high_norm[1, 28:36, 28:36].max():.2f}, total_max={anchor_high_norm[1].max():.2f}")
print(f"  Low anchor: center_max={anchor_low_norm[1, 28:36, 28:36].max():.2f}, total_max={anchor_low_norm[1].max():.2f}")

# ============================================================
# Part 3: The real issue - what patterns differ?
# ============================================================
print("\n=== PATTERN DIFFERENCE ANALYSIS ===")

# Compare statistics of normalized images
for name, img in [("Training", train_norm), ("High Anchor", anchor_high_norm), ("Low Anchor", anchor_low_norm)]:
    print(f"\n{name}:")
    for c, band in enumerate(["g", "r", "z"]):
        print(f"  {band}: mean={img[c].mean():.3f}, std={img[c].std():.3f}, max={img[c].max():.2f}")

# ============================================================
# Part 4: Look at specific regions
# ============================================================
print("\n=== REGION-BY-REGION ANALYSIS ===")

y, x = np.ogrid[:64, :64]
r = np.sqrt((y - 32)**2 + (x - 32)**2)
center_mask = r < 8
inner_ring_mask = (r >= 8) & (r < 16)
outer_ring_mask = (r >= 16) & (r < 24)

for name, img in [("Training", train_norm), ("High Anchor", anchor_high_norm), ("Low Anchor", anchor_low_norm)]:
    print(f"\n{name} (r-band):")
    print(f"  Center (r<8): mean={img[1, center_mask].mean():.3f}")
    print(f"  Inner ring (8<r<16): mean={img[1, inner_ring_mask].mean():.3f}")
    print(f"  Outer ring (16<r<24): mean={img[1, outer_ring_mask].mean():.3f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The model learned patterns at specific normalized signal levels.
The anchors have different noise/signal characteristics that
don't match what the model was trained on.
""")

#!/usr/bin/env python3
"""Check data units and preprocessing."""
import numpy as np
from astropy.io import fits
from pathlib import Path
import io
import pyarrow.dataset as ds

print("=" * 70)
print("CHECKING DATA UNITS AND PREPROCESSING")
print("=" * 70)

# Check anchor FITS headers
anchor_dir = Path("/lambda/nfs/darkhaloscope-training-dc/anchor_cutouts/known_lenses")

print("\n=== ANCHOR FITS HEADERS ===")
for fits_file in sorted(anchor_dir.glob("*.fits"))[:3]:
    print(f"\n{fits_file.name}:")
    with fits.open(fits_file) as hdu:
        h = hdu[0].header
        print(f"  All header keys: {list(h.keys())[:10]}...")
        print(f"  BUNIT: {h.get('BUNIT', 'NOT SET')}")
        data = hdu[0].data
        if data is not None:
            print(f"  Data shape: {data.shape}")
            print(f"  Data dtype: {data.dtype}")
            print(f"  Data range: [{data.min():.6f}, {data.max():.6f}]")

# Compare with training data
print("\n=== TRAINING DATA (from Parquet) ===")
data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")
filter_cond = (ds.field("region_split") == "test") & (ds.field("cutout_ok") == 1)
table = dataset.to_table(filter=filter_cond, columns=["stamp_npz", "is_control"])
table = table.slice(0, 5)

for i in range(min(3, table.num_rows)):
    blob = table["stamp_npz"][i].as_py()
    is_ctrl = table["is_control"][i].as_py()
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        r = z["image_r"].astype(np.float32)
    print(f"\nSample {i} (control={is_ctrl}):")
    print(f"  r-band shape: {r.shape}")
    print(f"  r-band dtype: {r.dtype}")
    print(f"  r-band range: [{r.min():.6f}, {r.max():.6f}]")

# Check what our Phase 4c pipeline outputs
print("\n=== HYPOTHESIS ===")
print("""
The anchor cutouts are downloaded DIRECTLY from Legacy Survey in nanomaggies.
The training data goes through our Phase 4c pipeline which may do preprocessing.

Let me check the training data source to understand the units.
""")

# Check a few more anchors to see the variation
print("\n=== ANCHOR DATA VARIATION ===")
print(f"{'Name':<30} {'r_min':<12} {'r_max':<12} {'r_mean':<12}")
print("-" * 70)
for fits_file in sorted(anchor_dir.glob("*.fits"))[:20]:
    with fits.open(fits_file) as hdu:
        data = hdu[0].data.astype(np.float32)
    r = data[1]  # r-band
    print(f"{fits_file.stem:<30} {r.min():<12.6f} {r.max():<12.6f} {r.mean():<12.6f}")

# Hypothesis: Maybe some anchors have a bright foreground galaxy
print("\n=== KEY INSIGHT ===")
print("""
Some anchors have VERY high r_max (9-11) while others have low r_max (0.01-0.05).

The high r_max anchors likely have:
1. A very bright lens galaxy in the center
2. Or a nearby bright star
3. Or are in units different from our training data

The ones with r_max ~ 0.01-0.05 are similar to our training data range.
""")

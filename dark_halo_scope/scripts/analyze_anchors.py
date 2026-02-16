#!/usr/bin/env python3
"""Analyze anchor lens signal strength."""
import numpy as np
from astropy.io import fits
from pathlib import Path

anchor_dir = Path("/lambda/nfs/darkhaloscope-training-dc/anchor_cutouts/known_lenses")

results = []
for fits_file in sorted(anchor_dir.glob("*.fits")):
    with fits.open(fits_file) as hdu:
        data = hdu[0].data.astype(np.float32)
        r_max = data[1].max()
        r_mean = data[1].mean()
        central_r = data[1, 28:36, 28:36].max()  # Central 8x8 region
        results.append({
            "name": fits_file.stem,
            "r_max": r_max,
            "r_mean": r_mean,
            "central_r_max": central_r
        })

results = sorted(results, key=lambda x: x["central_r_max"], reverse=True)

print("=== ANCHOR LENSES BY CENTRAL R-BAND SIGNAL ===")
print(f"{'Name':<35} {'Central_Max':<12} {'R_Max':<12} {'R_Mean':<12}")
print("-" * 75)
for r in results[:20]:
    name = r["name"][:33]
    print(f"{name:<35} {r['central_r_max']:<12.4f} {r['r_max']:<12.4f} {r['r_mean']:<12.6f}")

print()
print("... bottom 10 ...")
for r in results[-10:]:
    name = r["name"][:33]
    print(f"{name:<35} {r['central_r_max']:<12.4f} {r['r_max']:<12.4f} {r['r_mean']:<12.6f}")

# How many have very weak central signal?
weak = [r for r in results if r["central_r_max"] < 0.1]
print(f"\nLenses with central_r_max < 0.1: {len(weak)} / {len(results)} ({100*len(weak)/len(results):.0f}%)")

strong = [r for r in results if r["central_r_max"] > 1.0]
print(f"Lenses with central_r_max > 1.0: {len(strong)} / {len(results)} ({100*len(strong)/len(results):.0f}%)")

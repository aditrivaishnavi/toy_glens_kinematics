#!/usr/bin/env python3
"""
Test 1: Outer Annulus Pedestal Check

Purpose: If there's a systematic calibration mismatch between stamp and ctrl,
the outer annulus (far from any arc) should also show an offset.

If outer_mean > 0 consistently -> SYSTEMATIC MISMATCH (not PSF spreading)
If outer_mean ~ 0 -> PSF spreading only affects core
"""

import boto3
import io
import numpy as np
import pyarrow.parquet as pq

def radial_mask(shape, r_min, r_max):
    """Create a radial mask between r_min and r_max pixels from center."""
    cy, cx = shape[0]//2, shape[1]//2
    y, x = np.ogrid[:shape[0], :shape[1]]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    return (r >= r_min) & (r < r_max)

def main():
    s3 = boto3.client("s3", region_name="us-east-2")
    
    # List parquet files
    resp = s3.list_objects_v2(
        Bucket="darkhaloscope",
        Prefix="phase4_pipeline/phase4c/v5_cosmos_paired/train/",
        MaxKeys=10
    )
    files = [f for f in resp["Contents"] if f["Key"].endswith(".parquet")]
    
    outer_means = {"g": [], "r": [], "z": []}
    core_means = {"g": [], "r": [], "z": []}
    
    for fkey in files[:5]:
        print(f"Processing {fkey['Key']}...")
        obj = s3.get_object(Bucket="darkhaloscope", Key=fkey["Key"])
        table = pq.read_table(io.BytesIO(obj["Body"].read()))
        df = table.to_pandas()
        
        # Exclude problematic bricks
        df = df[~df["brickname"].isin(["0460m800", "3252p267"])]
        
        for idx, row in df.sample(min(100, len(df)), random_state=42).iterrows():
            try:
                stamp_data = np.load(io.BytesIO(row["stamp_npz"]))
                ctrl_data = np.load(io.BytesIO(row["ctrl_stamp_npz"]))
                
                for band in ["g", "r", "z"]:
                    stamp = stamp_data["image_" + band]
                    ctrl = ctrl_data["image_" + band]
                    
                    if np.any(np.isnan(stamp)) or np.any(np.isnan(ctrl)):
                        continue
                    
                    diff = stamp - ctrl
                    
                    # Outer annulus: r >= 20 pixels (far from any arc at theta_E < 3")
                    outer_mask = radial_mask(diff.shape, 20, 32)
                    outer_mean = float(np.mean(diff[outer_mask]))
                    
                    # Core: r < 5 pixels
                    core_mask = radial_mask(diff.shape, 0, 5)
                    core_mean = float(np.mean(diff[core_mask]))
                    
                    if np.isfinite(outer_mean) and np.isfinite(core_mean):
                        outer_means[band].append(outer_mean)
                        core_means[band].append(core_mean)
            except Exception as e:
                continue
    
    print("\n=== TEST 1: OUTER ANNULUS PEDESTAL CHECK ===")
    print("If outer_mean > 0 consistently -> SYSTEMATIC MISMATCH")
    print("If outer_mean ~ 0 -> PSF spreading only")
    print()
    
    for band in ["g", "r", "z"]:
        om = np.array(outer_means[band])
        cm = np.array(core_means[band])
        
        print(f"{band}-band (n={len(om)}):")
        print(f"  Outer (r>=20 pix):")
        print(f"    Mean: {np.mean(om):.6f}")
        print(f"    Median: {np.median(om):.6f}")
        print(f"    Std: {np.std(om):.6f}")
        print(f"    % positive: {100*np.mean(om > 0):.1f}%")
        print(f"  Core (r<5 pix):")
        print(f"    Mean: {np.mean(cm):.6f}")
        print()
    
    all_outer = np.concatenate([outer_means["g"], outer_means["r"], outer_means["z"]])
    all_core = np.concatenate([core_means["g"], core_means["r"], core_means["z"]])
    
    print("=== VERDICT ===")
    print(f"Overall outer mean: {np.mean(all_outer):.6f}")
    print(f"Overall core mean: {np.mean(all_core):.6f}")
    if np.mean(all_outer) != 0:
        print(f"Ratio (core/outer): {np.mean(all_core)/np.mean(all_outer):.2f}x")
    print()
    
    if np.mean(all_outer) > 0.001:
        print("OUTER ANNULUS IS POSITIVE -> SYSTEMATIC MISMATCH (BUG)")
    else:
        print("OUTER ANNULUS NEAR ZERO -> Core offset is likely PSF spreading")

if __name__ == "__main__":
    main()

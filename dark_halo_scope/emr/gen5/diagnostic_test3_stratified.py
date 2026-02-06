#!/usr/bin/env python3
"""
Test 3: Stratified Analysis by theta_E

Purpose: Check if core leakage decreases with larger Einstein radius.

If larger theta_E → less core leakage: PHYSICS
If leakage is constant across theta_E: BUG
"""

import boto3
import io
import numpy as np
import pyarrow.parquet as pq

def radial_mask(shape, r_min, r_max):
    cy, cx = shape[0]//2, shape[1]//2
    y, x = np.ogrid[:shape[0], :shape[1]]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    return (r >= r_min) & (r < r_max)

def main():
    s3 = boto3.client("s3", region_name="us-east-2")
    
    resp = s3.list_objects_v2(
        Bucket="darkhaloscope",
        Prefix="phase4_pipeline/phase4c/v5_cosmos_paired/train/",
        MaxKeys=10
    )
    files = [f for f in resp["Contents"] if f["Key"].endswith(".parquet")]
    
    data = []
    for fkey in files[:6]:
        print(f"Processing {fkey['Key']}...")
        obj = s3.get_object(Bucket="darkhaloscope", Key=fkey["Key"])
        table = pq.read_table(io.BytesIO(obj["Body"].read()))
        df = table.to_pandas()
        df = df[~df["brickname"].isin(["0460m800", "3252p267"])]
        
        for idx, row in df.sample(min(80, len(df)), random_state=42).iterrows():
            try:
                stamp_data = np.load(io.BytesIO(row["stamp_npz"]))
                ctrl_data = np.load(io.BytesIO(row["ctrl_stamp_npz"]))
                
                stamp_r = stamp_data["image_r"]
                ctrl_r = ctrl_data["image_r"]
                diff_r = stamp_r - ctrl_r
                
                core_mask = radial_mask(diff_r.shape, 0, 5)
                core_ctrl_flux = float(np.sum(ctrl_r[core_mask]))
                core_diff_flux = float(np.sum(diff_r[core_mask]))
                total_arc_flux = float(np.sum(diff_r))
                
                if core_ctrl_flux > 0 and total_arc_flux > 0:
                    data.append({
                        "theta_e": row["theta_e_arcsec"],
                        "theta_e_pix": row["theta_e_arcsec"] / 0.262,
                        "core_diff_pct": 100 * core_diff_flux / core_ctrl_flux,
                        "leakage_pct": 100 * core_diff_flux / total_arc_flux,
                    })
            except Exception as e:
                continue
    
    print(f"\n=== STRATIFIED BY THETA_E ===")
    print(f"Core radius = 5 pix = 1.31 arcsec")
    print(f"Total samples: {len(data)}")
    print()
    
    bins = [
        (0, 0.75, "theta<0.75 (arc IN core)"),
        (0.75, 1.0, "0.75<=theta<1"),
        (1.0, 1.5, "1<=theta<1.5"),
        (1.5, 2.0, "1.5<=theta<2"),
        (2.0, 5.0, "theta>=2 (arc FAR from core)")
    ]
    
    for low, high, label in bins:
        subset = [d for d in data if low <= d["theta_e"] < high]
        if len(subset) > 5:
            core_pcts = [d["core_diff_pct"] for d in subset]
            leak_pcts = [d["leakage_pct"] for d in subset]
            theta_pix = [d["theta_e_pix"] for d in subset]
            
            print(f"{label} (n={len(subset)}):")
            print(f"  Arc radius: {np.mean(theta_pix):.1f} pix")
            print(f"  Core diff as % of ctrl: {np.mean(core_pcts):.1f}%")
            print(f"  Arc flux that lands in core: {np.mean(leak_pcts):.1f}%")
            print()
    
    # Physics check for theta_E >= 2"
    print("=== PHYSICS CHECK FOR THETA_E >= 2 ===")
    large_theta = [d for d in data if d["theta_e"] >= 2.0]
    if len(large_theta) > 5:
        print(f"Samples: {len(large_theta)}")
        print(f"Mean arc radius: {np.mean([d['theta_e_pix'] for d in large_theta]):.1f} pix")
        print(f"Mean core offset: {np.mean([d['core_diff_pct'] for d in large_theta]):.1f}% of ctrl")
        print(f"Mean leakage: {np.mean([d['leakage_pct'] for d in large_theta]):.1f}% of arc")
        print()
        print("Expected PSF leakage at r=8pix with sigma=2.4pix:")
        print("  exp(-8^2/(2*2.4^2)) = exp(-5.6) = 0.4%")
        print()
        
        observed_leakage = np.mean([d["leakage_pct"] for d in large_theta])
        if observed_leakage > 5:
            print("OBSERVED >> EXPECTED")
            print("-> Either PSF is wider than assumed, or source profile is extended")
        else:
            print("OBSERVED ~ EXPECTED")
            print("-> Pure PSF physics")

if __name__ == "__main__":
    main()

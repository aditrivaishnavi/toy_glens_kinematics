#!/usr/bin/env python3
"""
Test 4: Radial Profile Analysis

Purpose: For samples where arc SHOULD be at r~8.6 pix (theta_E >= 2"),
check if the radial profile is peaked at theta_E (ring-like) or flat (diffuse).

Peaked at theta_E -> Simple arc at Einstein radius
Flat profile -> Extended source + lens physics
"""

import boto3
import io
import numpy as np
import pyarrow.parquet as pq

def main():
    s3 = boto3.client("s3", region_name="us-east-2")
    
    resp = s3.list_objects_v2(
        Bucket="darkhaloscope",
        Prefix="phase4_pipeline/phase4c/v5_cosmos_paired/train/",
        MaxKeys=5
    )
    files = [f for f in resp["Contents"] if f["Key"].endswith(".parquet")]
    
    profiles = []
    for fkey in files[:3]:
        print(f"Processing {fkey['Key']}...")
        obj = s3.get_object(Bucket="darkhaloscope", Key=fkey["Key"])
        table = pq.read_table(io.BytesIO(obj["Body"].read()))
        df = table.to_pandas()
        df = df[~df["brickname"].isin(["0460m800", "3252p267"])]
        df = df[df["theta_e_arcsec"] >= 2.0]  # Only large theta_E
        
        for idx, row in df.sample(min(30, len(df)), random_state=42).iterrows():
            try:
                stamp_data = np.load(io.BytesIO(row["stamp_npz"]))
                ctrl_data = np.load(io.BytesIO(row["ctrl_stamp_npz"]))
                
                diff = stamp_data["image_r"] - ctrl_data["image_r"]
                
                # Compute radial profile
                ny, nx = diff.shape
                cy, cx = ny//2, nx//2
                y, x = np.ogrid[:ny, :nx]
                r = np.sqrt((x - cx)**2 + (y - cy)**2)
                
                radii = np.arange(0, 25)
                profile = []
                for ri in radii:
                    mask = (r >= ri) & (r < ri + 1)
                    if np.sum(mask) > 0:
                        profile.append(float(np.mean(diff[mask])))
                    else:
                        profile.append(0)
                
                profiles.append({
                    "theta_e_pix": row["theta_e_arcsec"] / 0.262,
                    "profile": np.array(profile),
                })
            except Exception as e:
                continue
    
    print(f"\n=== RADIAL PROFILE OF (stamp - ctrl) FOR THETA_E >= 2 ===")
    print(f"Samples: {len(profiles)}")
    print()
    
    # Average profile
    mean_profile = np.mean([p["profile"] for p in profiles], axis=0)
    mean_theta = np.mean([p["theta_e_pix"] for p in profiles])
    
    print(f"Mean theta_E: {mean_theta:.1f} pix")
    print()
    print("Radius  | Mean Diff   | Note")
    print("-" * 45)
    for ri in range(18):
        note = ""
        if ri < 5:
            note = "CORE"
        elif abs(ri - mean_theta) < 2:
            note = "ARC REGION"
        print(f"{ri:4d}    | {mean_profile[ri]:.6f}    | {note}")
    
    inner_flux = np.sum(mean_profile[:5])
    arc_flux = np.sum(mean_profile[6:12])
    outer_flux = np.sum(mean_profile[15:20])
    
    print()
    print(f"Core (r<5) total: {inner_flux:.5f}")
    print(f"Arc (6-12) total: {arc_flux:.5f}")
    print(f"Outer (15-20) total: {outer_flux:.5f}")
    print()
    
    if arc_flux > 0:
        core_to_arc_ratio = inner_flux / arc_flux
        print(f"Core/Arc flux ratio: {core_to_arc_ratio:.2f}")
        print()
        if core_to_arc_ratio > 0.5:
            print("FLAT PROFILE: Core has significant flux even for large theta_E")
            print("-> Suggests extended source + lens physics, not thin ring")
        else:
            print("PEAKED PROFILE: Arc region dominates")
            print("-> Suggests thin ring at Einstein radius")

if __name__ == "__main__":
    main()

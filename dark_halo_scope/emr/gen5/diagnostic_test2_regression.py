#!/usr/bin/env python3
"""
Test 2: Core Offset Regression Analysis

Purpose: Check if core offset scales with arc properties (physics) or is constant (bug).

If R² is high -> offset depends on arc properties (PHYSICS)
If R² is low and offset is constant -> systematic mismatch (BUG)
"""

import boto3
import io
import numpy as np
import pyarrow.parquet as pq
from scipy import stats

def main():
    s3 = boto3.client("s3", region_name="us-east-2")
    
    resp = s3.list_objects_v2(
        Bucket="darkhaloscope",
        Prefix="phase4_pipeline/phase4c/v5_cosmos_paired/train/",
        MaxKeys=10
    )
    files = [f for f in resp["Contents"] if f["Key"].endswith(".parquet")]
    
    data = []
    for fkey in files[:5]:
        print(f"Processing {fkey['Key']}...")
        obj = s3.get_object(Bucket="darkhaloscope", Key=fkey["Key"])
        table = pq.read_table(io.BytesIO(obj["Body"].read()))
        df = table.to_pandas()
        df = df[~df["brickname"].isin(["0460m800", "3252p267"])]
        
        for idx, row in df.sample(min(50, len(df)), random_state=42).iterrows():
            try:
                stamp_data = np.load(io.BytesIO(row["stamp_npz"]))
                stamp = np.stack([stamp_data["image_g"], stamp_data["image_r"], stamp_data["image_z"]], axis=0)
                
                ctrl_data = np.load(io.BytesIO(row["ctrl_stamp_npz"]))
                ctrl = np.stack([ctrl_data["image_g"], ctrl_data["image_r"], ctrl_data["image_z"]], axis=0)
                
                # Skip if any NaN
                if np.any(np.isnan(stamp)) or np.any(np.isnan(ctrl)):
                    continue
                
                # Core region (central 10x10)
                core_stamp = stamp[:, 27:37, 27:37]
                core_ctrl = ctrl[:, 27:37, 27:37]
                
                # Total arc flux (stamp - ctrl, summed)
                diff = stamp - ctrl
                total_arc_flux = float(np.sum(diff))
                
                # Core offset
                core_offset = float(np.mean(core_stamp) - np.mean(core_ctrl))
                
                # Skip invalid
                if not np.isfinite(core_offset) or not np.isfinite(total_arc_flux):
                    continue
                if row["arc_snr"] is None or not np.isfinite(row["arc_snr"]):
                    continue
                    
                data.append({
                    "theta_e": float(row["theta_e_arcsec"]),
                    "arc_snr": float(row["arc_snr"]),
                    "total_arc_flux": total_arc_flux,
                    "core_offset": core_offset,
                })
            except Exception as e:
                continue
    
    print(f"\n=== DIAGNOSTIC TESTS: BUG vs PHYSICS ===")
    print(f"Valid samples: {len(data)}")
    print()
    
    theta_e = np.array([d["theta_e"] for d in data])
    arc_snr = np.array([d["arc_snr"] for d in data])
    total_flux = np.array([d["total_arc_flux"] for d in data])
    core_offset = np.array([d["core_offset"] for d in data])
    
    # Test 1: Core offset vs theta_E
    slope_theta, intercept, r_theta, p_val, _ = stats.linregress(theta_e, core_offset)
    print("TEST 1: Core offset vs theta_E")
    print(f"  Slope: {slope_theta:.6f}")
    print(f"  R²: {r_theta**2:.4f}")
    print(f"  p-value: {p_val:.6f}")
    if abs(r_theta**2) < 0.05:
        print("  → Suggests BUG (no theta_E dependence)")
    elif slope_theta < 0 and r_theta**2 > 0.1:
        print("  → Suggests PHYSICS (larger theta_E → less core leakage)")
    else:
        print("  → Inconclusive")
    print()
    
    # Test 2: Core offset vs total arc flux
    slope_flux, _, r_flux, p_val_flux, _ = stats.linregress(total_flux, core_offset)
    print("TEST 2: Core offset vs total arc flux")
    print(f"  Slope: {slope_flux:.8f}")
    print(f"  R²: {r_flux**2:.4f}")
    print(f"  p-value: {p_val_flux:.6f}")
    if r_flux**2 > 0.3:
        print("  → Suggests PHYSICS (core offset scales with arc flux)")
    elif r_flux**2 < 0.1:
        print("  → Suggests BUG (no flux dependence)")
    else:
        print("  → Inconclusive")
    print()
    
    # Test 3: Core offset vs arc_snr
    slope_snr, _, r_snr, p_val_snr, _ = stats.linregress(arc_snr, core_offset)
    print("TEST 3: Core offset vs arc_snr")
    print(f"  Slope: {slope_snr:.6f}")
    print(f"  R²: {r_snr**2:.4f}")
    print(f"  p-value: {p_val_snr:.6f}")
    if r_snr**2 > 0.3:
        print("  → Suggests PHYSICS (core offset scales with arc SNR)")
    elif r_snr**2 < 0.1:
        print("  → Suggests BUG (no SNR dependence)")
    else:
        print("  → Inconclusive")
    print()
    
    # Test 4: Is core offset constant?
    print("TEST 4: Core offset statistics")
    mean_offset = np.mean(core_offset)
    std_offset = np.std(core_offset)
    print(f"  Mean: {mean_offset:.6f}")
    print(f"  Std: {std_offset:.6f}")
    print(f"  CV (std/mean): {std_offset/mean_offset:.2f}")
    print(f"  Min: {np.min(core_offset):.6f}")
    print(f"  Max: {np.max(core_offset):.6f}")
    print(f"  % positive: {100*np.mean(core_offset > 0):.1f}%")
    print()
    
    print("=== VERDICT ===")
    bug_score = 0
    physics_score = 0
    
    if r_theta**2 < 0.05:
        bug_score += 1
        print("  - No theta_E dependence → BUG")
    if r_flux**2 < 0.1:
        bug_score += 1
        print("  - No flux dependence → BUG")
    if r_snr**2 < 0.1:
        bug_score += 1
        print("  - No SNR dependence → BUG")
    
    if r_theta**2 > 0.1 and slope_theta < 0:
        physics_score += 1
        print("  - Negative theta_E slope → PHYSICS")
    if r_flux**2 > 0.2:
        physics_score += 1
        print("  - Flux dependence → PHYSICS")
    if r_snr**2 > 0.2:
        physics_score += 1
        print("  - SNR dependence → PHYSICS")
    
    print()
    print(f"BUG indicators: {bug_score}/3")
    print(f"PHYSICS indicators: {physics_score}/3")

if __name__ == "__main__":
    main()

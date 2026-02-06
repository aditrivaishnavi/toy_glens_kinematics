#!/usr/bin/env python3
"""Test for cutout alignment mismatch between stamp and ctrl."""

import boto3
import io
import numpy as np
import pyarrow.parquet as pq
from scipy import ndimage


def phase_correlation_shift(img1, img2):
    """Compute subpixel shift between two images using phase correlation."""
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)
    cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10)
    correlation = np.fft.ifft2(cross_power).real
    
    max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    # Handle wraparound
    shift_y = max_idx[0] if max_idx[0] < correlation.shape[0] // 2 else max_idx[0] - correlation.shape[0]
    shift_x = max_idx[1] if max_idx[1] < correlation.shape[1] // 2 else max_idx[1] - correlation.shape[1]
    
    return shift_y, shift_x, np.max(correlation)


def test_alignment():
    """Test alignment between stamp and ctrl across multiple samples."""
    print("=== CUTOUT ALIGNMENT TEST ===")
    print()
    
    s3 = boto3.client("s3", region_name="us-east-2")
    
    resp = s3.list_objects_v2(
        Bucket="darkhaloscope",
        Prefix="phase4_pipeline/phase4c/v5_cosmos_paired/train/",
        MaxKeys=3
    )
    files = [f for f in resp["Contents"] if f["Key"].endswith(".parquet")]
    
    obj = s3.get_object(Bucket="darkhaloscope", Key=files[0]["Key"])
    table = pq.read_table(io.BytesIO(obj["Body"].read()))
    df = table.to_pandas()
    df = df[~df["brickname"].isin(["0460m800", "3252p267"])]
    
    # Test phase correlation on outer regions
    print("Testing phase correlation between stamp and ctrl")
    print("Using outer region (r >= 16) to avoid arc contamination")
    print()
    
    shifts = []
    for i in range(min(50, len(df))):
        sample = df.iloc[i]
        stamp_data = np.load(io.BytesIO(sample["stamp_npz"]))
        ctrl_data = np.load(io.BytesIO(sample["ctrl_stamp_npz"]))
        
        stamp_r = stamp_data["image_r"]
        ctrl_r = ctrl_data["image_r"]
        
        # Mask the central region
        mask = np.ones_like(stamp_r)
        mask[16:48, 16:48] = 0
        
        stamp_outer = stamp_r * mask
        ctrl_outer = ctrl_r * mask
        
        shift_y, shift_x, corr = phase_correlation_shift(stamp_outer, ctrl_outer)
        shifts.append((shift_y, shift_x))
    
    shifts = np.array(shifts)
    print(f"Samples tested: {len(shifts)}")
    print(f"Y-shift: mean={np.mean(shifts[:, 0]):.3f}, std={np.std(shifts[:, 0]):.3f}")
    print(f"X-shift: mean={np.mean(shifts[:, 1]):.3f}, std={np.std(shifts[:, 1]):.3f}")
    print()
    
    if abs(np.mean(shifts[:, 0])) > 0.5 or abs(np.mean(shifts[:, 1])) > 0.5:
        print("WARNING: Consistent misalignment detected!")
    else:
        print("PASS: No significant alignment mismatch")


def test_shift_sensitivity():
    """Test if shifting ctrl changes core difference."""
    print()
    print("=== SHIFT SENSITIVITY TEST ===")
    print()
    
    s3 = boto3.client("s3", region_name="us-east-2")
    
    resp = s3.list_objects_v2(
        Bucket="darkhaloscope",
        Prefix="phase4_pipeline/phase4c/v5_cosmos_paired/train/",
        MaxKeys=3
    )
    files = [f for f in resp["Contents"] if f["Key"].endswith(".parquet")]
    
    obj = s3.get_object(Bucket="darkhaloscope", Key=files[0]["Key"])
    table = pq.read_table(io.BytesIO(obj["Body"].read()))
    df = table.to_pandas()
    df = df[~df["brickname"].isin(["0460m800", "3252p267"])]
    
    sample = df.iloc[0]
    stamp_data = np.load(io.BytesIO(sample["stamp_npz"]))
    ctrl_data = np.load(io.BytesIO(sample["ctrl_stamp_npz"]))
    
    stamp_r = stamp_data["image_r"]
    ctrl_r = ctrl_data["image_r"]
    
    # Core region
    core_slice = (slice(27, 37), slice(27, 37))
    core_diff_orig = np.mean(stamp_r[core_slice] - ctrl_r[core_slice])
    
    print(f"Original core_diff: {core_diff_orig:.5f}")
    print()
    print("Core diff with shifted ctrl:")
    
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ctrl_shifted = ndimage.shift(ctrl_r, (dy, dx), mode="nearest")
            core_diff = np.mean(stamp_r[core_slice] - ctrl_shifted[core_slice])
            marker = " <-- unshifted" if (dy == 0 and dx == 0) else ""
            print(f"  Shift ({dy:2d}, {dx:2d}): core_diff = {core_diff:.5f}{marker}")
    
    print()
    print("If original matches (0,0), no alignment mismatch")


if __name__ == "__main__":
    test_alignment()
    test_shift_sensitivity()

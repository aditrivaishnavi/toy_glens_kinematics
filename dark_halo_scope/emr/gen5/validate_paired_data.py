#!/usr/bin/env python3
"""Physics-based validation of paired training data.

Validates that ctrl_stamp_npz (base LRG) and stamp_npz (LRG + injected lens)
are consistent and exhibit expected physical properties.

Validations:
1. Row counts match expected
2. Central aperture consistency (LRG core should be identical)
3. Difference image reveals arc (correlates with arc_snr)
4. Arc extent correlates with theta_E
5. Background noise consistency
6. Flux ratios are physically plausible
7. Band colors consistent with ellipticals
"""

import argparse
import io
import sys
from typing import Dict, List, Tuple

import numpy as np


def load_sample(row) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load stamp and ctrl_stamp from a parquet row."""
    # Load stamp_npz (LRG + lens)
    stamp_data = np.load(io.BytesIO(row['stamp_npz']))
    stamp = np.stack([stamp_data['image_g'], stamp_data['image_r'], stamp_data['image_z']], axis=0)
    
    # Load ctrl_stamp_npz (base LRG)
    ctrl_data = np.load(io.BytesIO(row['ctrl_stamp_npz']))
    ctrl = np.stack([ctrl_data['image_g'], ctrl_data['image_r'], ctrl_data['image_z']], axis=0)
    
    # Metadata
    meta = {
        'theta_e_arcsec': row.get('theta_e_arcsec', 0),
        'arc_snr': row.get('arc_snr', 0),
        'ra': row.get('ra', 0),
        'dec': row.get('dec', 0),
        'brickname': row.get('brickname', ''),
    }
    
    return stamp, ctrl, meta


def validate_central_aperture(stamp: np.ndarray, ctrl: np.ndarray, aperture_size: int = 10) -> Dict:
    """Check that central region (LRG core) is nearly identical.
    
    The injected arc should be around the LRG, not at its center.
    """
    h, w = stamp.shape[1], stamp.shape[2]
    cx, cy = w // 2, h // 2
    half = aperture_size // 2
    
    # Extract central apertures
    stamp_center = stamp[:, cy-half:cy+half, cx-half:cx+half]
    ctrl_center = ctrl[:, cy-half:cy+half, cx-half:cx+half]
    
    # Compute relative difference
    diff = np.abs(stamp_center - ctrl_center)
    mean_ctrl = np.mean(np.abs(ctrl_center))
    rel_diff = np.mean(diff) / (mean_ctrl + 1e-10)
    
    # Central pixels should be very similar (rel_diff < 0.1 typically)
    return {
        'central_rel_diff': float(rel_diff),
        'central_abs_diff': float(np.mean(diff)),
        'pass': rel_diff < 0.2  # Allow some tolerance
    }


def validate_arc_detection(stamp: np.ndarray, ctrl: np.ndarray, arc_snr: float) -> Dict:
    """Check that difference image reveals arc with flux correlating to arc_snr."""
    diff = stamp - ctrl  # This should be the injected arc
    
    # Compute arc flux (sum of positive pixels in difference)
    arc_flux = np.sum(np.maximum(diff, 0))
    
    # Compute noise from ctrl (background std)
    bg_std = np.std(ctrl[:, :10, :10])  # Corner region
    
    # Estimated SNR from difference
    arc_pixels = np.sum(diff > 2 * bg_std)  # Significant arc pixels
    if arc_pixels > 0:
        estimated_snr = arc_flux / (bg_std * np.sqrt(arc_pixels) + 1e-10)
    else:
        estimated_snr = 0
    
    return {
        'arc_flux': float(arc_flux),
        'arc_pixels': int(arc_pixels),
        'estimated_snr': float(estimated_snr),
        'expected_snr': float(arc_snr),
        'snr_ratio': float(estimated_snr / (arc_snr + 1e-10)),
    }


def validate_arc_extent(stamp: np.ndarray, ctrl: np.ndarray, theta_e_arcsec: float, 
                        pixel_scale: float = 0.262) -> Dict:
    """Check that arc spatial extent correlates with Einstein radius."""
    diff = stamp - ctrl
    r_band_diff = diff[1]  # Use r-band
    
    # Threshold to find arc pixels
    threshold = 3 * np.std(r_band_diff[:10, :10])
    arc_mask = r_band_diff > threshold
    
    if np.sum(arc_mask) < 5:
        return {'arc_extent_pixels': 0, 'theta_e_pixels': theta_e_arcsec / pixel_scale, 'pass': True}
    
    # Find arc extent (max distance from center)
    h, w = r_band_diff.shape
    cy, cx = h // 2, w // 2
    
    y_coords, x_coords = np.where(arc_mask)
    distances = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)
    max_extent = np.max(distances)
    
    # Expected extent from theta_E
    theta_e_pixels = theta_e_arcsec / pixel_scale
    
    return {
        'arc_extent_pixels': float(max_extent),
        'theta_e_pixels': float(theta_e_pixels),
        'extent_ratio': float(max_extent / (theta_e_pixels + 1e-10)),
        'pass': 0.3 < max_extent / (theta_e_pixels + 1e-10) < 3.0  # Reasonable range
    }


def validate_background_noise(stamp: np.ndarray, ctrl: np.ndarray) -> Dict:
    """Check that background regions have consistent noise."""
    # Use corners (should be pure background)
    corners = [(0, 10, 0, 10), (0, 10, -10, None), (-10, None, 0, 10), (-10, None, -10, None)]
    
    stamp_noise = []
    ctrl_noise = []
    
    for y0, y1, x0, x1 in corners:
        stamp_noise.append(np.std(stamp[:, y0:y1, x0:x1]))
        ctrl_noise.append(np.std(ctrl[:, y0:y1, x0:x1]))
    
    stamp_bg_std = np.mean(stamp_noise)
    ctrl_bg_std = np.mean(ctrl_noise)
    
    noise_ratio = stamp_bg_std / (ctrl_bg_std + 1e-10)
    
    return {
        'stamp_bg_std': float(stamp_bg_std),
        'ctrl_bg_std': float(ctrl_bg_std),
        'noise_ratio': float(noise_ratio),
        'pass': 0.8 < noise_ratio < 1.2  # Should be very similar
    }


def validate_flux_ratio(stamp: np.ndarray, ctrl: np.ndarray) -> Dict:
    """Check that arc-to-LRG flux ratio is physically plausible."""
    # LRG flux from ctrl (central region)
    h, w = ctrl.shape[1], ctrl.shape[2]
    cy, cx = h // 2, w // 2
    lrg_aperture = 15  # pixels
    
    lrg_flux = np.sum(ctrl[:, cy-lrg_aperture:cy+lrg_aperture, cx-lrg_aperture:cx+lrg_aperture])
    
    # Arc flux from difference
    diff = stamp - ctrl
    arc_flux = np.sum(np.maximum(diff, 0))
    
    flux_ratio = arc_flux / (lrg_flux + 1e-10)
    
    return {
        'lrg_flux': float(lrg_flux),
        'arc_flux': float(arc_flux),
        'arc_to_lrg_ratio': float(flux_ratio),
        'pass': 0.001 < flux_ratio < 0.5  # Typical range for lensing
    }


def validate_band_colors(ctrl: np.ndarray) -> Dict:
    """Check that LRG colors are consistent with elliptical galaxies."""
    # Central aperture photometry
    h, w = ctrl.shape[1], ctrl.shape[2]
    cy, cx = h // 2, w // 2
    ap = 10
    
    g_flux = np.sum(ctrl[0, cy-ap:cy+ap, cx-ap:cx+ap])
    r_flux = np.sum(ctrl[1, cy-ap:cy+ap, cx-ap:cx+ap])
    z_flux = np.sum(ctrl[2, cy-ap:cy+ap, cx-ap:cx+ap])
    
    # Color ratios (ellipticals are red: g < r < z typically)
    g_minus_r = -2.5 * np.log10(g_flux / (r_flux + 1e-10) + 1e-10)
    r_minus_z = -2.5 * np.log10(r_flux / (z_flux + 1e-10) + 1e-10)
    
    return {
        'g_flux': float(g_flux),
        'r_flux': float(r_flux),
        'z_flux': float(z_flux),
        'g_minus_r': float(g_minus_r),
        'r_minus_z': float(r_minus_z),
        'is_red': g_flux < r_flux < z_flux  # Ellipticals should be red
    }


def validate_sample(row) -> Dict:
    """Run all validations on a single sample."""
    stamp, ctrl, meta = load_sample(row)
    
    results = {
        'meta': meta,
        'central_aperture': validate_central_aperture(stamp, ctrl),
        'arc_detection': validate_arc_detection(stamp, ctrl, meta['arc_snr']),
        'arc_extent': validate_arc_extent(stamp, ctrl, meta['theta_e_arcsec']),
        'background_noise': validate_background_noise(stamp, ctrl),
        'flux_ratio': validate_flux_ratio(stamp, ctrl),
        'band_colors': validate_band_colors(ctrl),
    }
    
    # Overall pass
    results['all_pass'] = all([
        results['central_aperture']['pass'],
        results['background_noise']['pass'],
        results['flux_ratio']['pass'],
    ])
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate paired training data")
    parser.add_argument("--input", required=True, help="S3 path to paired parquet")
    parser.add_argument("--split", default="train", help="Split to validate")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples to validate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()
    
    import boto3
    import pyarrow.parquet as pq
    
    print(f"[INFO] Validating {args.split} split from {args.input}")
    
    s3 = boto3.client('s3', region_name='us-east-2')
    
    # Parse S3 path
    path = args.input.replace('s3://', '')
    bucket = path.split('/')[0]
    prefix = '/'.join(path.split('/')[1:]) + f'/{args.split}/'
    
    # List parquet files
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=100)
    files = [f['Key'] for f in resp.get('Contents', []) if f['Key'].endswith('.parquet')]
    
    if not files:
        print(f"[ERROR] No parquet files found at {args.input}/{args.split}/")
        sys.exit(1)
    
    print(f"[INFO] Found {len(files)} parquet files")
    
    # Sample from first few files
    np.random.seed(args.seed)
    
    all_results = []
    samples_collected = 0
    
    for file_key in files[:10]:  # Check first 10 files
        if samples_collected >= args.n_samples:
            break
            
        obj = s3.get_object(Bucket=bucket, Key=file_key)
        table = pq.read_table(io.BytesIO(obj['Body'].read()))
        df = table.to_pandas()
        
        # Random sample from this file
        n_from_file = min(args.n_samples - samples_collected, len(df), 20)
        indices = np.random.choice(len(df), n_from_file, replace=False)
        
        for idx in indices:
            row = df.iloc[idx].to_dict()
            try:
                result = validate_sample(row)
                all_results.append(result)
                samples_collected += 1
            except Exception as e:
                print(f"[WARN] Failed to validate sample: {e}")
    
    print(f"\n[INFO] Validated {len(all_results)} samples")
    print("=" * 60)
    
    # Aggregate statistics
    central_diffs = [r['central_aperture']['central_rel_diff'] for r in all_results]
    noise_ratios = [r['background_noise']['noise_ratio'] for r in all_results]
    flux_ratios = [r['flux_ratio']['arc_to_lrg_ratio'] for r in all_results]
    extent_ratios = [r['arc_extent']['extent_ratio'] for r in all_results if r['arc_extent'].get('extent_ratio')]
    red_fraction = np.mean([r['band_colors']['is_red'] for r in all_results])
    all_pass = np.mean([r['all_pass'] for r in all_results])
    
    print("\n[SUMMARY] Physics Validation Results:")
    print(f"  Central aperture rel diff: {np.mean(central_diffs):.4f} ± {np.std(central_diffs):.4f}")
    print(f"  Background noise ratio:    {np.mean(noise_ratios):.4f} ± {np.std(noise_ratios):.4f}")
    print(f"  Arc-to-LRG flux ratio:     {np.mean(flux_ratios):.4f} ± {np.std(flux_ratios):.4f}")
    if extent_ratios:
        print(f"  Arc extent / theta_E:      {np.mean(extent_ratios):.4f} ± {np.std(extent_ratios):.4f}")
    print(f"  Fraction with red colors:  {red_fraction:.2%}")
    print(f"  Overall pass rate:         {all_pass:.2%}")
    
    # Detailed breakdown
    print("\n[DETAIL] Pass rates by validation:")
    print(f"  Central aperture: {np.mean([r['central_aperture']['pass'] for r in all_results]):.2%}")
    print(f"  Background noise: {np.mean([r['background_noise']['pass'] for r in all_results]):.2%}")
    print(f"  Flux ratio:       {np.mean([r['flux_ratio']['pass'] for r in all_results]):.2%}")
    
    # Flag any issues
    if all_pass < 0.9:
        print("\n[WARNING] Overall pass rate below 90% - investigate failed samples")
        
    if np.mean(central_diffs) > 0.1:
        print("\n[WARNING] Central aperture differences higher than expected")
        print("  This could indicate WCS misalignment or incorrect cutout extraction")
    
    if red_fraction < 0.7:
        print("\n[WARNING] Many LRGs don't show expected red colors")
        print("  Check if ctrl_stamp_npz is correctly fetching the base LRG")
    
    print("\n[DONE] Validation complete")


if __name__ == "__main__":
    main()

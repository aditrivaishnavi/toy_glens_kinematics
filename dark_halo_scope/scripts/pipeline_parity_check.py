#!/usr/bin/env python3
"""
Pipeline Parity Check: Compare Legacy Survey cutout service vs EMR pipeline.

This script tests whether the two data sources produce equivalent pixel values
for the same sky coordinates.

Experiment 1: Fetch training LRG coordinates via BOTH methods and compare.
Experiment 3: Fetch SLACS/BELLS anchor coordinates via BOTH methods and compare.
"""
import numpy as np
import requests
import tempfile
import os
import io
import time
from astropy.io import fits
from astropy.wcs import WCS
import boto3

# ============================================================
# Configuration
# ============================================================
STAMP_SIZE = 64
PIXSCALE = 0.262  # arcsec/pixel

# S3 Configuration
S3_BUCKET = "darkhaloscope"
COADD_PREFIX = "dr10/coadd_cache"

# SLACS/BELLS anchor coordinates
ANCHOR_COORDS = [
    # SLACS lenses
    {"name": "SDSSJ0029-0055", "ra": 7.4543, "dec": -0.9254},
    {"name": "SDSSJ0037-0942", "ra": 9.3004, "dec": -9.7095},
    {"name": "SDSSJ0252+0039", "ra": 43.1313, "dec": 0.6651},
    {"name": "SDSSJ0330-0020", "ra": 52.5019, "dec": -0.3419},
    {"name": "SDSSJ0728+3835", "ra": 112.1879, "dec": 38.5900},
    {"name": "SDSSJ0737+3216", "ra": 114.4121, "dec": 32.2793},
    {"name": "SDSSJ0912+0029", "ra": 138.0200, "dec": 0.4894},
    {"name": "SDSSJ0959+0410", "ra": 149.7954, "dec": 4.1755},
    {"name": "SDSSJ1016+3859", "ra": 154.1092, "dec": 38.9937},
    {"name": "SDSSJ1020+1122", "ra": 155.1029, "dec": 11.3690},
    # BELLS lenses
    {"name": "SDSSJ0747+5055", "ra": 116.8850, "dec": 50.9261},
    {"name": "SDSSJ0755+3445", "ra": 118.8175, "dec": 34.7544},
    {"name": "SDSSJ0801+4727", "ra": 120.3808, "dec": 47.4569},
    {"name": "SDSSJ0830+5116", "ra": 127.5371, "dec": 51.2753},
    {"name": "SDSSJ0832+0404", "ra": 128.2038, "dec": 4.0725},
]

# ============================================================
# Helper Functions
# ============================================================
def get_brickname(ra, dec):
    """Get DR10 brickname for given coordinates."""
    # Bricknames are XXXDYYY where XXX.X is RA/10 and YYY is dec+0.5
    # Format: 4 digits for RA (XXXX) + sign + 3 digits for dec (YYY)
    ra_str = f"{int(ra * 10):04d}"
    if dec >= 0:
        dec_str = f"p{int(abs(dec) * 10 + 0.5):03d}"
    else:
        dec_str = f"m{int(abs(dec) * 10 + 0.5):03d}"
    return f"{ra_str}{dec_str}"

def fetch_via_cutout_service(ra, dec, size=STAMP_SIZE, retries=3):
    """
    Fetch cutout via Legacy Survey cutout service (same method as anchors).
    Returns (3, size, size) array in grz order, or None if failed.
    """
    url = f"https://www.legacysurvey.org/viewer/cutout.fits?ra={ra}&dec={dec}&size={size}&layer=ls-dr10&pixscale={PIXSCALE}&bands=grz"
    
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                time.sleep(1)
                continue
            
            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
                f.write(resp.content)
                tmp_path = f.name
            
            try:
                with fits.open(tmp_path) as hdul:
                    # Cutout service returns data in HDU 0
                    data = hdul[0].data
                    if data is None:
                        return None
                    # Should be (3, 64, 64) for grz
                    if data.shape != (3, size, size):
                        print(f"  Warning: unexpected shape {data.shape}")
                    return data.astype(np.float32)
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            print(f"  Cutout service attempt {attempt+1} failed: {e}")
            time.sleep(1)
    
    return None

def fetch_via_emr_pipeline(ra, dec, s3_client, size=STAMP_SIZE):
    """
    Fetch cutout via EMR pipeline logic (replicate training data extraction).
    Returns (3, size, size) array in grz order, or None if failed.
    """
    brickname = get_brickname(ra, dec)
    bands = ['g', 'r', 'z']
    images = []
    
    for band in bands:
        # S3 path to coadd
        s3_key = f"{COADD_PREFIX}/{brickname}/legacysurvey-{brickname}-image-{band}.fits.fz"
        
        try:
            # Download FITS from S3
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            fits_bytes = obj['Body'].read()
            
            with fits.open(io.BytesIO(fits_bytes)) as hdul:
                # Compressed FITS: data in HDU 1
                img_data = hdul[1].data
                wcs = WCS(hdul[1].header)
                
                # Convert RA/Dec to pixel
                # NOTE: wcs.world_to_pixel_values returns numpy arrays, need float() before round()
                x, y = wcs.world_to_pixel_values(ra, dec)
                x, y = int(round(float(x))), int(round(float(y)))
                
                # Extract cutout
                half = size // 2
                x0, x1 = x - half, x + half
                y0, y1 = y - half, y + half
                
                # Check bounds
                if x0 < 0 or y0 < 0 or x1 > img_data.shape[1] or y1 > img_data.shape[0]:
                    print(f"  Out of bounds for {brickname} band {band}")
                    return None
                
                cutout = img_data[y0:y1, x0:x1].copy()
                images.append(cutout.astype(np.float32))
        
        except Exception as e:
            print(f"  EMR fetch failed for {brickname}/{band}: {e}")
            return None
    
    return np.stack(images, axis=0)  # (3, 64, 64)

def compare_images(name, cutout_img, emr_img):
    """Compare two images and return statistics."""
    if cutout_img is None or emr_img is None:
        return None
    
    # r-band is index 1
    cutout_r = cutout_img[1]
    emr_r = emr_img[1]
    
    # Central 8x8 region
    c = cutout_img.shape[1] // 2
    cutout_center = cutout_r[c-4:c+4, c-4:c+4]
    emr_center = emr_r[c-4:c+4, c-4:c+4]
    
    return {
        "name": name,
        "cutout_r_mean": float(cutout_r.mean()),
        "emr_r_mean": float(emr_r.mean()),
        "cutout_center_max": float(cutout_center.max()),
        "emr_center_max": float(emr_center.max()),
        "mean_ratio": float(cutout_r.mean() / (emr_r.mean() + 1e-10)),
        "center_ratio": float(cutout_center.max() / (emr_center.max() + 1e-10)),
        "pixel_corr": float(np.corrcoef(cutout_r.flatten(), emr_r.flatten())[0, 1]),
        "max_abs_diff": float(np.max(np.abs(cutout_r - emr_r))),
    }

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("PIPELINE PARITY CHECK")
    print("=" * 70)
    
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    # --------------------------------------------------------
    # Experiment 1: Load training LRG coordinates from manifest
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: TRAINING LRG COORDINATES")
    print("=" * 70)
    print("(Will be run after extracting coordinates from manifest)")
    
    # For now, test with a few known coordinates from the training data
    # These are LRG positions we know are in the coadd cache
    
    # First, let's test the parity with anchor coordinates since we have them
    
    # --------------------------------------------------------
    # Experiment 3: SLACS/BELLS anchor coordinates
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: SLACS/BELLS ANCHOR COORDINATES")
    print("=" * 70)
    
    results = []
    for coord in ANCHOR_COORDS[:10]:  # Test first 10
        name = coord["name"]
        ra, dec = coord["ra"], coord["dec"]
        print(f"\nProcessing {name} (RA={ra:.4f}, Dec={dec:.4f})...")
        
        # Fetch via cutout service
        print("  Fetching via cutout service...")
        cutout_img = fetch_via_cutout_service(ra, dec)
        if cutout_img is not None:
            print(f"    Shape: {cutout_img.shape}, r-band mean: {cutout_img[1].mean():.6f}")
        else:
            print("    Failed!")
        
        # Fetch via EMR pipeline
        print("  Fetching via EMR pipeline...")
        emr_img = fetch_via_emr_pipeline(ra, dec, s3)
        if emr_img is not None:
            print(f"    Shape: {emr_img.shape}, r-band mean: {emr_img[1].mean():.6f}")
        else:
            print("    Failed (brick may not be in cache)")
        
        # Compare
        if cutout_img is not None and emr_img is not None:
            cmp = compare_images(name, cutout_img, emr_img)
            results.append(cmp)
            print(f"    Mean ratio (cutout/emr): {cmp['mean_ratio']:.4f}")
            print(f"    Pixel correlation: {cmp['pixel_corr']:.4f}")
        elif cutout_img is not None:
            # Record cutout-only result
            results.append({
                "name": name,
                "cutout_r_mean": float(cutout_img[1].mean()),
                "emr_r_mean": None,
                "cutout_center_max": float(cutout_img[1][28:36, 28:36].max()),
                "emr_center_max": None,
                "mean_ratio": None,
                "center_ratio": None,
                "pixel_corr": None,
            })
    
    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY: ANCHOR COORDINATES")
    print("=" * 70)
    
    valid_results = [r for r in results if r.get("mean_ratio") is not None]
    
    if valid_results:
        ratios = [r["mean_ratio"] for r in valid_results]
        corrs = [r["pixel_corr"] for r in valid_results]
        
        print(f"\n{'Name':<20} {'Cutout Mean':<14} {'EMR Mean':<14} {'Ratio':<10} {'Corr':<10}")
        print("-" * 70)
        for r in valid_results:
            print(f"{r['name']:<20} {r['cutout_r_mean']:<14.6f} {r['emr_r_mean']:<14.6f} {r['mean_ratio']:<10.4f} {r['pixel_corr']:<10.4f}")
        
        print(f"\nMean ratio: {np.mean(ratios):.4f} (+/- {np.std(ratios):.4f})")
        print(f"Mean correlation: {np.mean(corrs):.4f}")
        
        # Interpretation
        print("\n" + "=" * 70)
        print("INTERPRETATION:")
        print("=" * 70)
        
        mean_ratio = np.mean(ratios)
        if 0.9 <= mean_ratio <= 1.1:
            print("✓ PIPELINES AGREE (ratio ≈ 1.0)")
            print("  The 10x brightness gap is a REAL POPULATION DIFFERENCE.")
            print("  SLACS/BELLS galaxies are genuinely fainter than training LRGs.")
        elif mean_ratio < 0.5:
            print("✗ EMR PIPELINE PRODUCES BRIGHTER IMAGES")
            print("  This is a PROCESSING ARTIFACT in the EMR pipeline.")
        elif mean_ratio > 2.0:
            print("✗ CUTOUT SERVICE PRODUCES BRIGHTER IMAGES")
            print("  This is a PROCESSING ARTIFACT in the cutout service.")
        else:
            print("? MODERATE DIFFERENCE - needs investigation")
    else:
        print("No valid comparisons (EMR bricks may not be cached for anchors)")
        print("\nCutout-only results:")
        for r in results:
            if r.get("cutout_r_mean") is not None:
                print(f"  {r['name']}: r-band mean = {r['cutout_r_mean']:.6f}, center max = {r['cutout_center_max']:.6f}")

if __name__ == "__main__":
    main()

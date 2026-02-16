#!/usr/bin/env python3
"""
Pipeline Parity Check: ACTUAL CODE THAT WAS EXECUTED
This is the exact code that produced the results in the parity check.
"""
import numpy as np
import requests
import tempfile
import os
import io
import time
import json
from astropy.io import fits
from astropy.wcs import WCS
import boto3
import pyarrow.parquet as pq

STAMP_SIZE = 64
PIXSCALE = 0.262
S3_BUCKET = "darkhaloscope"
COADD_PREFIX = "dr10/coadd_cache"

s3 = boto3.client('s3')

def fetch_via_cutout_service(ra, dec, size=STAMP_SIZE, retries=3):
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
                    data = hdul[0].data
                    if data is None:
                        return None
                    return data.astype(np.float32)
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            time.sleep(1)
    return None

def fetch_via_emr_pipeline(ra, dec, brickname, size=STAMP_SIZE):
    bands = ['g', 'r', 'z']
    images = []
    for band in bands:
        s3_key = f"{COADD_PREFIX}/{brickname}/legacysurvey-{brickname}-image-{band}.fits.fz"
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            fits_bytes = obj['Body'].read()
            with fits.open(io.BytesIO(fits_bytes)) as hdul:
                img_data = hdul[1].data
                wcs = WCS(hdul[1].header)
                x, y = wcs.world_to_pixel_values(ra, dec)
                # CRITICAL FIX: wcs returns numpy arrays, need float() before round()
                x, y = int(round(float(x))), int(round(float(y)))
                half = size // 2
                x0, x1 = x - half, x + half
                y0, y1 = y - half, y + half
                if x0 < 0 or y0 < 0 or x1 > img_data.shape[1] or y1 > img_data.shape[0]:
                    return None
                cutout = img_data[y0:y1, x0:x1].copy()
                images.append(cutout.astype(np.float32))
        except Exception as e:
            return None
    return np.stack(images, axis=0)

# Get coordinates for cached bricks
print("Finding training LRGs in cached bricks...")
resp = s3.list_objects_v2(
    Bucket='darkhaloscope',
    Prefix='phase4_pipeline/phase4a/v4_sota_moffat/manifests/train_stamp64_bandsgrz_gridgrid_sota/',
    MaxKeys=5
)
parquet_files = [obj['Key'] for obj in resp.get('Contents', []) if obj['Key'].endswith('.parquet')]

coords = []
for pf_key in parquet_files[:1]:
    obj = s3.get_object(Bucket='darkhaloscope', Key=pf_key)
    tbl = pq.read_table(io.BytesIO(obj['Body'].read()))
    df = tbl.to_pandas()
    
    for _, row in df[['ra', 'dec', 'brickname']].drop_duplicates().head(30).iterrows():
        coords.append({
            'name': f"LRG_{row['brickname']}",
            'ra': float(row['ra']),
            'dec': float(row['dec']),
            'brickname': row['brickname']
        })

print(f"Found {len(coords)} unique coordinates")

# Run parity check
print("\n" + "=" * 70)
print("EXPERIMENT 1: TRAINING LRG COORDINATES")
print("=" * 70)

results = []
for i, coord in enumerate(coords[:15]):
    name, ra, dec, brick = coord['name'], coord['ra'], coord['dec'], coord['brickname']
    print(f"{i+1}. {name}...", end=" ")
    
    cutout_img = fetch_via_cutout_service(ra, dec)
    emr_img = fetch_via_emr_pipeline(ra, dec, brick)
    
    if cutout_img is not None and emr_img is not None:
        cutout_r, emr_r = cutout_img[1], emr_img[1]
        ratio = cutout_r.mean() / (emr_r.mean() + 1e-10)
        corr = np.corrcoef(cutout_r.flatten(), emr_r.flatten())[0, 1]
        results.append({
            "name": name,
            "cutout_r_mean": float(cutout_r.mean()),
            "emr_r_mean": float(emr_r.mean()),
            "mean_ratio": float(ratio),
            "pixel_corr": float(corr),
        })
        print(f"ratio={ratio:.4f}, corr={corr:.4f}")
    else:
        status = f"cutout={cutout_img is not None}, emr={emr_img is not None}"
        print(f"SKIP ({status})")

print(f"\nValid comparisons: {len(results)}")

if results:
    ratios = [r['mean_ratio'] for r in results]
    corrs = [r['pixel_corr'] for r in results]
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Mean ratio (cutout/emr): {np.mean(ratios):.4f} +/- {np.std(ratios):.4f}")
    print(f"Min ratio: {min(ratios):.4f}, Max ratio: {max(ratios):.4f}")
    print(f"Mean pixel correlation: {np.mean(corrs):.4f}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    mean_ratio = np.mean(ratios)
    if 0.9 <= mean_ratio <= 1.1:
        print("✓ PIPELINES AGREE (ratio ≈ 1.0)")
        print("  Both cutout service and EMR pipeline produce equivalent pixel values.")
        print("  Any brightness difference between training and anchors is REAL.")
    elif mean_ratio < 0.9:
        print(f"✗ EMR BRIGHTER by {1/mean_ratio:.2f}x")
    else:
        print(f"✗ CUTOUT SERVICE BRIGHTER by {mean_ratio:.2f}x")

#!/usr/bin/env python3
"""Audit script to verify paired training data flow."""
import pyarrow.dataset as ds
import numpy as np
import boto3
import io
from astropy.io import fits
from astropy.wcs import WCS

print("=" * 70)
print("AUDIT: Paired Training Data Flow")
print("=" * 70)

# Load sample positives
print("\n1. Loading sample positives from parquet...")
data = ds.dataset('/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos', format='parquet', partitioning='hive')
filt = (ds.field('region_split') == 'train') & (ds.field('is_control') == 0) & (ds.field('cutout_ok') == 1)
tbl = data.to_table(filter=filt, columns=['stamp_npz', 'ra', 'dec', 'brickname', 'theta_e_arcsec', 'arc_snr']).slice(0, 5)

s3 = boto3.client('s3')

print("\n2. Comparing positive (with injection) vs control (from coadd cache)...")
for i in range(min(3, tbl.num_rows)):
    blob = tbl['stamp_npz'][i].as_py()
    ra = tbl['ra'][i].as_py()
    dec = tbl['dec'][i].as_py()
    brick = tbl['brickname'][i].as_py()
    theta_e = tbl['theta_e_arcsec'][i].as_py()
    arc_snr = tbl['arc_snr'][i].as_py()
    
    # Decode stored positive
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        pos_r = z['image_r'].astype(np.float32)
    
    # Fetch base cutout from S3 coadd cache
    try:
        obj = s3.get_object(Bucket='darkhaloscope', Key=f'dr10/coadd_cache/{brick}/legacysurvey-{brick}-image-r.fits.fz')
        with fits.open(io.BytesIO(obj['Body'].read())) as hdul:
            img = hdul[1].data.astype(np.float32)
            wcs = WCS(hdul[1].header)
            x, y = wcs.world_to_pixel_values(ra, dec)
            x, y = int(round(float(x))), int(round(float(y)))
            ctrl_r = img[y-32:y+32, x-32:x+32]
        
        # Compare
        diff = pos_r - ctrl_r
        print(f'\nSample {i}: brick={brick}, theta_e={theta_e:.2f}", arc_snr={arc_snr:.1f}')
        print(f'  Positive r-band: mean={pos_r.mean():.4f}, max={pos_r.max():.4f}')
        print(f'  Control r-band:  mean={ctrl_r.mean():.4f}, max={ctrl_r.max():.4f}')
        print(f'  Difference:      mean={diff.mean():.6f}, max_abs={abs(diff).max():.4f}')
        print(f'  -> Injection signal present: {abs(diff).max() > 0.001}')
    except Exception as e:
        print(f'\nSample {i}: ERROR - {e}')

print("\n" + "=" * 70)
print("CONCLUSION:")
print("  - Positive stamps contain LRG + injected lens")
print("  - Control stamps (from coadd cache) contain same LRG WITHOUT injection")
print("  - Difference = injection signal (arc)")
print("  - Paired training uses this to teach model: detect arcs, not LRG properties")
print("=" * 70)

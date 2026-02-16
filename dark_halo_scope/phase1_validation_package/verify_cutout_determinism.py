#!/usr/bin/env python3
"""
Phase 1 Verification: Cutout Determinism Test

Tests whether re-fetching a cutout from DR10 coadd cache produces
identical pixel values to the stored stamp. This determines if we
can salvage the existing dataset with paired controls.

Scientific rigor requirements:
1. Test multiple samples across different bricks
2. Compare CONTROL samples only (no injection artifacts)
3. Report pixel-level correlation and max absolute difference
4. Document any discrepancies for LLM review
"""
import numpy as np
import pyarrow.dataset as ds
import boto3
import io
from astropy.io import fits
from astropy.wcs import WCS
import json
from datetime import datetime, timezone

RESULTS = {
    "test": "cutout_determinism",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "purpose": "Verify if we can salvage existing dataset by re-fetching paired controls",
    "samples": []
}

print("=" * 70)
print("PHASE 1 VERIFICATION: CUTOUT DETERMINISM TEST")
print("=" * 70)

# Configuration
STAMP_SIZE = 64
PIXSCALE = 0.262
S3_BUCKET = "darkhaloscope"
COADD_PREFIX = "dr10/coadd_cache"
N_SAMPLES = 20  # Test 20 samples for statistical confidence

# Load dataset - use CONTROLS only (no injection artifacts)
print("\nLoading control samples from dataset...")
data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

# Get controls with all metadata needed for re-fetch
filt = (ds.field("region_split") == "train") & (ds.field("cutout_ok") == 1) & (ds.field("is_control") == 1)
cols = ["stamp_npz", "ra", "dec", "brickname", "psfsize_r"]
table = dataset.to_table(filter=filt, columns=cols)

print(f"Total control samples: {table.num_rows}")

# Sample randomly
np.random.seed(42)
indices = np.random.choice(table.num_rows, min(N_SAMPLES, table.num_rows), replace=False)

def decode_stamp(blob):
    """Decode stored NPZ to (3, H, W) array."""
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        g = z["image_g"].astype(np.float32)
        r = z["image_r"].astype(np.float32)
        zb = z["image_z"].astype(np.float32)
    return np.stack([g, r, zb], axis=0)

def fetch_cutout_from_s3(ra, dec, brickname, s3_client, bands=['g', 'r', 'z']):
    """
    Re-fetch cutout from DR10 coadd cache using same logic as pipeline.
    Returns (3, 64, 64) array or None if failed.
    """
    images = []
    
    for band in bands:
        s3_key = f"{COADD_PREFIX}/{brickname}/legacysurvey-{brickname}-image-{band}.fits.fz"
        
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            fits_bytes = obj['Body'].read()
            
            with fits.open(io.BytesIO(fits_bytes)) as hdul:
                img_data = hdul[1].data
                wcs = WCS(hdul[1].header)
                
                # Convert RA/Dec to pixel
                x, y = wcs.world_to_pixel_values(ra, dec)
                x, y = int(round(float(x))), int(round(float(y)))
                
                # Extract cutout
                half = STAMP_SIZE // 2
                x0, x1 = x - half, x + half
                y0, y1 = y - half, y + half
                
                # Check bounds
                if x0 < 0 or y0 < 0 or x1 > img_data.shape[1] or y1 > img_data.shape[0]:
                    return None, f"Out of bounds: ({x0},{y0})-({x1},{y1}) vs shape {img_data.shape}"
                
                cutout = img_data[y0:y1, x0:x1].copy()
                images.append(cutout.astype(np.float32))
                
        except Exception as e:
            return None, f"S3 fetch error: {str(e)}"
    
    return np.stack(images, axis=0), None

# Initialize S3 client
s3 = boto3.client('s3')

print(f"\nTesting {len(indices)} samples...")
print("-" * 70)

matches = []
mismatches = []

for i, idx in enumerate(indices):
    idx = int(idx)
    
    # Get stored data
    blob = table["stamp_npz"][idx].as_py()
    ra = table["ra"][idx].as_py()
    dec = table["dec"][idx].as_py()
    brickname = table["brickname"][idx].as_py()
    
    if blob is None or ra is None or brickname is None:
        print(f"  {i+1}. SKIP - missing data")
        continue
    
    try:
        stored = decode_stamp(blob)
    except Exception as e:
        print(f"  {i+1}. SKIP - decode error: {e}")
        continue
    
    # Re-fetch from S3
    refetched, error = fetch_cutout_from_s3(ra, dec, brickname, s3)
    
    if refetched is None:
        print(f"  {i+1}. SKIP - {error}")
        continue
    
    # Compare
    if stored.shape != refetched.shape:
        result = {
            "idx": idx,
            "brickname": brickname,
            "ra": ra,
            "dec": dec,
            "match": False,
            "reason": f"Shape mismatch: {stored.shape} vs {refetched.shape}"
        }
        mismatches.append(result)
        print(f"  {i+1}. MISMATCH - shape: {stored.shape} vs {refetched.shape}")
        continue
    
    # Pixel-level comparison
    diff = stored - refetched
    max_abs_diff = float(np.max(np.abs(diff)))
    mean_abs_diff = float(np.mean(np.abs(diff)))
    
    # Correlation per band
    corr_g = float(np.corrcoef(stored[0].flatten(), refetched[0].flatten())[0, 1])
    corr_r = float(np.corrcoef(stored[1].flatten(), refetched[1].flatten())[0, 1])
    corr_z = float(np.corrcoef(stored[2].flatten(), refetched[2].flatten())[0, 1])
    
    # Exact match check (allowing for floating point)
    is_exact = max_abs_diff < 1e-6
    is_close = max_abs_diff < 0.01 and min(corr_g, corr_r, corr_z) > 0.999
    
    result = {
        "idx": idx,
        "brickname": brickname,
        "ra": ra,
        "dec": dec,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "corr_g": corr_g,
        "corr_r": corr_r,
        "corr_z": corr_z,
        "is_exact": is_exact,
        "is_close": is_close,
        "match": is_exact or is_close
    }
    
    if result["match"]:
        matches.append(result)
        status = "EXACT" if is_exact else "CLOSE"
        print(f"  {i+1}. {status} - max_diff={max_abs_diff:.2e}, corr_r={corr_r:.6f}")
    else:
        mismatches.append(result)
        print(f"  {i+1}. MISMATCH - max_diff={max_abs_diff:.4f}, corr_r={corr_r:.6f}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

n_tested = len(matches) + len(mismatches)
n_matches = len(matches)
n_mismatches = len(mismatches)
match_rate = n_matches / n_tested if n_tested > 0 else 0

print(f"Samples tested: {n_tested}")
print(f"Matches: {n_matches} ({match_rate:.1%})")
print(f"Mismatches: {n_mismatches}")

RESULTS["n_tested"] = n_tested
RESULTS["n_matches"] = n_matches
RESULTS["n_mismatches"] = n_mismatches
RESULTS["match_rate"] = match_rate
RESULTS["matches"] = matches
RESULTS["mismatches"] = mismatches

# Decision
print("\n" + "=" * 70)
print("DECISION")
print("=" * 70)

if match_rate >= 0.95:
    decision = "SALVAGE"
    explanation = "Cutout re-fetching is deterministic. We can salvage the existing dataset by re-fetching paired controls."
    RESULTS["salvage_viable"] = True
elif match_rate >= 0.80:
    decision = "SALVAGE_WITH_CAUTION"
    explanation = f"Cutout re-fetching mostly works ({match_rate:.1%}). Salvage is possible but some samples may need filtering."
    RESULTS["salvage_viable"] = True
else:
    decision = "REGENERATE"
    explanation = f"Cutout re-fetching is NOT deterministic ({match_rate:.1%}). Must regenerate Phase 4c with paired outputs."
    RESULTS["salvage_viable"] = False

print(f"Decision: {decision}")
print(f"Explanation: {explanation}")

RESULTS["decision"] = decision
RESULTS["explanation"] = explanation

# If mismatches exist, analyze why
if mismatches:
    print("\n" + "=" * 70)
    print("MISMATCH ANALYSIS")
    print("=" * 70)
    
    for m in mismatches[:5]:  # Show first 5
        print(f"  Brick: {m.get('brickname')}")
        print(f"    RA/Dec: {m.get('ra'):.6f}, {m.get('dec'):.6f}")
        if 'max_abs_diff' in m:
            print(f"    Max diff: {m['max_abs_diff']:.4f}")
            print(f"    Correlations: g={m['corr_g']:.4f}, r={m['corr_r']:.4f}, z={m['corr_z']:.4f}")
        if 'reason' in m:
            print(f"    Reason: {m['reason']}")
        print()

# Save results
output_path = "/lambda/nfs/darkhaloscope-training-dc/phase1_determinism_results.json"
with open(output_path, "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")

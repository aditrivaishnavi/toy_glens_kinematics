#!/usr/bin/env python3
"""
Crossmatch positives with DR10 sweep files directly.

The filtered manifest excludes sources near known lenses (by design),
so we must use the original sweep files for positive crossmatch.

Usage:
    python scripts/crossmatch_positives_sweeps.py \
        --sweeps s3://darkhaloscope/dr10/sweeps/ \
        --positives s3://darkhaloscope/stronglens_calibration/configs/positives/desi_candidates.csv \
        --output s3://darkhaloscope/stronglens_calibration/positives_with_dr10/

Author: Generated for stronglens_calibration project
Date: 2026-02-08
"""
import argparse
import gzip
import io
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
from astropy.io import fits

# =============================================================================
# CONSTANTS
# =============================================================================

PIPELINE_VERSION = "1.1.0"

# AWS Configuration (environment override supported)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

MAX_MATCH_RADIUS_ARCSEC = 5.0
HEALPIX_NSIDE = 128

TIER_WEIGHTS = {
    "confident": 0.95,
    "probable": 0.50,
}

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CrossmatchSweeps] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CrossmatchSweeps")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_sweep_filename(filename: str) -> Tuple[float, float, float, float]:
    """Parse sweep filename to get RA/Dec bounds."""
    # Format: sweep-{ra_min}{dec_sign}{dec_abs}-{ra_max}{dec_sign2}{dec_abs2}.fits.gz
    # Example: sweep-000p020-005p025.fits.gz -> RA 0-5, Dec 20-25
    base = filename.replace("sweep-", "").replace(".fits.gz", "")
    parts = base.split("-")
    
    def parse_coord(s):
        ra_part = s[:3]
        dec_sign = s[3]
        dec_part = s[4:]
        
        ra = float(ra_part)
        dec = float(dec_part)
        if dec_sign == "m":
            dec = -dec
        return ra, dec
    
    ra_min, dec_min = parse_coord(parts[0])
    ra_max, dec_max = parse_coord(parts[1])
    
    return ra_min, ra_max, dec_min, dec_max


def find_covering_sweep(ra: float, dec: float, sweep_files: List[str]) -> Optional[str]:
    """Find the sweep file that covers the given RA/Dec."""
    for sweep in sweep_files:
        filename = sweep.split("/")[-1]
        ra_min, ra_max, dec_min, dec_max = parse_sweep_filename(filename)
        
        # Handle RA wraparound
        if ra_min <= ra < ra_max and dec_min <= dec < dec_max:
            return sweep
    return None


def angular_separation_arcsec(ra1: float, dec1: float, ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    """Compute angular separation in arcseconds (vectorized)."""
    cos_dec = np.cos(np.radians(dec1))
    delta_ra = (ra2 - ra1) * cos_dec
    delta_dec = dec2 - dec1
    return np.sqrt(delta_ra**2 + delta_dec**2) * 3600


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def load_positives(s3_client, path: str) -> pd.DataFrame:
    """Load positive catalog from S3."""
    logger.info(f"Loading positives from: {path}")
    
    bucket = path.replace("s3://", "").split("/")[0]
    key = "/".join(path.replace("s3://", "").split("/")[1:])
    response = s3_client.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(response["Body"])
    
    logger.info(f"Loaded {len(df)} positives")
    
    # Add tier and weight
    df["tier"] = df["grading"].apply(lambda x: "A" if x == "confident" else "B")
    df["weight"] = df["tier"].apply(
        lambda x: TIER_WEIGHTS["confident"] if x == "A" else TIER_WEIGHTS["probable"]
    )
    
    return df


def list_sweep_files(s3_client, sweep_prefix: str) -> List[str]:
    """List all sweep files from S3."""
    logger.info(f"Listing sweep files from: {sweep_prefix}")
    
    bucket = sweep_prefix.replace("s3://", "").split("/")[0]
    prefix = "/".join(sweep_prefix.replace("s3://", "").split("/")[1:]).rstrip("/") + "/"
    
    sweep_files = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".fits.gz"):
                sweep_files.append(f"s3://{bucket}/{obj['Key']}")
    
    logger.info(f"Found {len(sweep_files)} sweep files")
    return sweep_files


def load_sweep_file(s3_client, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a sweep file and return RA, Dec, Type arrays."""
    import gc
    
    bucket = path.replace("s3://", "").split("/")[0]
    key = "/".join(path.replace("s3://", "").split("/")[1:])
    
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    decompressed = gzip.decompress(data)
    
    # Clear compressed data immediately
    del data
    gc.collect()
    
    buffer = io.BytesIO(decompressed)
    with fits.open(buffer, memmap=True) as hdu:
        # Extract only what we need as float32 to save memory
        ra = np.array(hdu[1].data["RA"], dtype=np.float32)
        dec = np.array(hdu[1].data["DEC"], dtype=np.float32)
        types = np.array([t.decode("utf-8").strip() if isinstance(t, bytes) else str(t).strip() 
                         for t in hdu[1].data["TYPE"]])
    
    # Clear decompressed data
    del decompressed, buffer
    gc.collect()
        
    return ra, dec, types


def crossmatch_positives(
    s3_client,
    positives: pd.DataFrame,
    sweep_files: List[str],
) -> pd.DataFrame:
    """Crossmatch positives with sweep files."""
    logger.info(f"Starting crossmatch of {len(positives)} positives")
    
    # Group positives by sweep file
    positives["sweep_file"] = positives.apply(
        lambda row: find_covering_sweep(row["ra"], row["dec"], sweep_files),
        axis=1
    )
    
    # Count coverage
    no_coverage = positives["sweep_file"].isna().sum()
    if no_coverage > 0:
        logger.warning(f"{no_coverage} positives have no sweep coverage")
    
    covered = positives[positives["sweep_file"].notna()]
    logger.info(f"{len(covered)} positives have sweep coverage")
    
    # Group by sweep file
    grouped = covered.groupby("sweep_file")
    logger.info(f"Positives span {len(grouped)} sweep files")
    
    import gc
    
    results = []
    files_processed = 0
    
    for sweep_path, group in grouped:
        # Load sweep file (no caching - not enough memory)
        filename = sweep_path.split("/")[-1]
        files_processed += 1
        logger.info(f"Loading {filename} ({files_processed}/{len(grouped)}, {len(group)} positives)...")
        
        try:
            ra_arr, dec_arr, type_arr = load_sweep_file(s3_client, sweep_path)
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            continue
        
        # Match each positive in this group
        for idx, pos in group.iterrows():
            pos_ra = pos["ra"]
            pos_dec = pos["dec"]
            
            # Compute separations
            seps = angular_separation_arcsec(pos_ra, pos_dec, ra_arr, dec_arr)
            
            # Find matches within radius
            within_radius = seps <= MAX_MATCH_RADIUS_ARCSEC
            
            if within_radius.sum() > 0:
                # Find nearest match
                nearest_idx = np.argmin(seps)
                nearest_sep = seps[nearest_idx]
                nearest_type = type_arr[nearest_idx]
                if isinstance(nearest_type, bytes):
                    nearest_type = nearest_type.decode("utf-8").strip()
                
                results.append({
                    "pos_name": pos["name"],
                    "pos_catalog_ra": pos_ra,
                    "pos_catalog_dec": pos_dec,
                    "zlens": pos.get("zlens"),
                    "pos_type": pos.get("type"),
                    "grading": pos["grading"],
                    "ref": pos.get("ref"),
                    "tier": pos["tier"],
                    "weight": pos["weight"],
                    "match_ra": float(ra_arr[nearest_idx]),
                    "match_dec": float(dec_arr[nearest_idx]),
                    "match_type": nearest_type,
                    "separation_arcsec": float(nearest_sep),
                    "n_within_radius": int(within_radius.sum()),
                })
            else:
                logger.warning(f"No match within {MAX_MATCH_RADIUS_ARCSEC}\" for {pos['name']}")
        
        # Clean up after each sweep file
        del ra_arr, dec_arr, type_arr
        gc.collect()
    
    if results:
        result_df = pd.DataFrame(results)
        logger.info(f"Matched {len(result_df)} of {len(positives)} positives")
        return result_df
    else:
        logger.error("No matches found!")
        return pd.DataFrame()


def save_output(s3_client, df: pd.DataFrame, path: str) -> None:
    """Save output to S3 as parquet."""
    logger.info(f"Saving output to: {path}")
    
    df["crossmatch_timestamp"] = datetime.now(timezone.utc).isoformat()
    df["crossmatch_version"] = PIPELINE_VERSION
    df["match_radius_arcsec"] = MAX_MATCH_RADIUS_ARCSEC
    
    bucket = path.replace("s3://", "").split("/")[0]
    key = "/".join(path.replace("s3://", "").split("/")[1:]).rstrip("/") + "/positives_with_dr10.parquet"
    
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow", compression="gzip", index=False)
    buffer.seek(0)
    
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    logger.info(f"Saved to s3://{bucket}/{key}")


def save_validation(s3_client, df: pd.DataFrame, positives_count: int, path: str) -> None:
    """Save validation report."""
    validation = {
        "total_positives": positives_count,
        "matched_count": len(df),
        "match_rate": len(df) / positives_count if positives_count > 0 else 0,
        "tier_distribution": df["tier"].value_counts().to_dict() if len(df) > 0 else {},
        "type_distribution": df["match_type"].value_counts().to_dict() if len(df) > 0 else {},
        "separation_stats": {
            "mean": float(df["separation_arcsec"].mean()) if len(df) > 0 else None,
            "median": float(df["separation_arcsec"].median()) if len(df) > 0 else None,
            "max": float(df["separation_arcsec"].max()) if len(df) > 0 else None,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    validation["gates_passed"] = validation["match_rate"] >= 0.95
    
    logger.info(f"Match rate: {validation['match_rate']*100:.1f}%")
    logger.info(f"Tier distribution: {validation['tier_distribution']}")
    logger.info(f"Type distribution: {validation['type_distribution']}")
    logger.info(f"Separation stats: {validation['separation_stats']}")
    
    bucket = path.replace("s3://", "").split("/")[0]
    key = "/".join(path.replace("s3://", "").split("/")[1:]).rstrip("/") + "/validation.json"
    s3_client.put_object(Bucket=bucket, Key=key, Body=json.dumps(validation, indent=2))
    logger.info(f"Validation saved to s3://{bucket}/{key}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Crossmatch positives with DR10 sweeps")
    parser.add_argument("--sweeps", required=True, help="S3 path to sweep files")
    parser.add_argument("--positives", required=True, help="S3 path to positives CSV")
    parser.add_argument("--output", required=True, help="S3 output path")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Starting Crossmatch with DR10 Sweep Files")
    logger.info("=" * 60)
    logger.info(f"Match radius: {MAX_MATCH_RADIUS_ARCSEC} arcsec")
    
    start_time = time.time()
    
    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        
        # Load positives
        positives = load_positives(s3, args.positives)
        
        # List sweep files
        sweep_files = list_sweep_files(s3, args.sweeps)
        
        # Crossmatch
        matched = crossmatch_positives(s3, positives, sweep_files)
        
        if len(matched) == 0:
            logger.error("No matches found")
            sys.exit(1)
        
        # Save output
        save_output(s3, matched, args.output)
        save_validation(s3, matched, len(positives), args.output)
        
        elapsed = time.time() - start_time
        logger.info(f"Completed in {elapsed:.1f}s")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

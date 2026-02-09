#!/usr/bin/env python3
"""
Local Crossmatch Script: Match positives with DR10 manifest.

This script runs locally on emr-launcher using PyArrow to read from S3.
It's simpler and faster for the small positive catalog (5,104 entries).

Usage:
    python scripts/crossmatch_positives_local.py \
        --manifest s3://darkhaloscope/stronglens_calibration/manifests/20260208_074343/ \
        --positives s3://darkhaloscope/stronglens_calibration/configs/positives/desi_candidates.csv \
        --output s3://darkhaloscope/stronglens_calibration/positives_with_dr10/

Author: Generated for stronglens_calibration project
Date: 2026-02-08
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import boto3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# =============================================================================
# CONSTANTS
# =============================================================================

PIPELINE_VERSION = "1.0.0"

# AWS Configuration (environment override supported)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

# Lens system coordinates are often offset from Tractor source centroids by several arcsec
# Using 5" radius to capture the deflector galaxy, will validate matches afterwards
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
    format="%(asctime)s [CrossmatchLocal] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CrossmatchLocal")


# =============================================================================
# HEALPIX FUNCTIONS
# =============================================================================

def compute_healpix(ra: float, dec: float, nside: int) -> int:
    """
    Compute HEALPix index for given coordinates.
    
    CRITICAL: The manifest was generated on EMR where healpy was NOT installed,
    so the hash-based fallback was used. We MUST use the same hash-based method
    here to ensure healpix values match.
    """
    # ALWAYS use hash-based method to match manifest (healpy was skipped on EMR)
    return int(hash((round(ra, 4), round(dec, 4), nside)) % (12 * nside * nside))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_positives(path: str) -> pd.DataFrame:
    """Load positive catalog from CSV."""
    logger.info(f"Loading positives from: {path}")
    
    s3 = boto3.client("s3", region_name=AWS_REGION)
    
    if path.startswith("s3://"):
        bucket = path.replace("s3://", "").split("/")[0]
        key = "/".join(path.replace("s3://", "").split("/")[1:])
        response = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(response["Body"])
    else:
        df = pd.read_csv(path)
    
    logger.info(f"Loaded {len(df)} positives")
    
    # Add healpix
    df["healpix_128"] = df.apply(
        lambda row: compute_healpix(row["ra"], row["dec"], HEALPIX_NSIDE), axis=1
    )
    
    # Add tier and weight
    df["tier"] = df["grading"].apply(lambda x: "A" if x == "confident" else "B")
    df["weight"] = df["tier"].apply(
        lambda x: TIER_WEIGHTS["confident"] if x == "A" else TIER_WEIGHTS["probable"]
    )
    
    return df


def build_kdtree(positives: pd.DataFrame):
    """Build a KD-tree from positive coordinates for efficient spatial lookup."""
    from scipy.spatial import cKDTree
    
    # Convert RA/Dec to Cartesian for KD-tree (approximate for small angles)
    # Use cos(dec) correction for RA
    ra_rad = np.radians(positives["ra"].values)
    dec_rad = np.radians(positives["dec"].values)
    
    # Project to 3D unit sphere
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    
    coords = np.column_stack([x, y, z])
    tree = cKDTree(coords)
    
    return tree


def crossmatch_streaming(positives: pd.DataFrame, manifest_path: str) -> pd.DataFrame:
    """
    Stream manifest files and find nearest match for each positive.
    
    Uses KD-tree for efficient spatial lookup. The hash-based healpix in manifest
    is NOT spatially meaningful, so we do direct spatial matching.
    """
    import io
    from scipy.spatial import cKDTree
    
    logger.info(f"Building KD-tree from {len(positives)} positives...")
    tree = build_kdtree(positives)
    
    # Match radius in radians (1 arcsec = 1/3600 deg)
    radius_deg = MAX_MATCH_RADIUS_ARCSEC / 3600.0
    radius_rad = np.radians(radius_deg)
    # Convert to chord distance for KD-tree (2*sin(angle/2) for unit sphere)
    radius_chord = 2 * np.sin(radius_rad / 2)
    
    logger.info(f"Match radius: {MAX_MATCH_RADIUS_ARCSEC}\" = {radius_chord:.6f} chord distance")
    
    s3 = boto3.client("s3", region_name=AWS_REGION)
    
    if manifest_path.startswith("s3://"):
        bucket = manifest_path.replace("s3://", "").split("/")[0]
        prefix = "/".join(manifest_path.replace("s3://", "").split("/")[1:]).rstrip("/") + "/"
    else:
        raise ValueError("Local manifest path not supported")
    
    # List all parquet files
    parquet_files = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                parquet_files.append(obj["Key"])
    
    logger.info(f"Found {len(parquet_files)} parquet files to scan")
    
    # Track best match per positive: {positive_idx: (distance, manifest_row)}
    best_matches = {}
    total_rows_scanned = 0
    
    for i, key in enumerate(parquet_files):
        if i % 20 == 0:
            logger.info(f"Processing file {i+1}/{len(parquet_files)} (scanned {total_rows_scanned:,} rows)...")
        
        # Download and read file
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        buffer = io.BytesIO(data)
        df = pq.read_table(buffer).to_pandas()
        total_rows_scanned += len(df)
        
        # Convert manifest coords to 3D
        ra_rad = np.radians(df["ra"].values)
        dec_rad = np.radians(df["dec"].values)
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)
        manifest_coords = np.column_stack([x, y, z])
        
        # Query tree for each manifest source
        distances, indices = tree.query(manifest_coords, distance_upper_bound=radius_chord)
        
        # Process matches (where index != tree.n means a match was found)
        n_positives = len(positives)
        for j in range(len(df)):
            if indices[j] < n_positives:  # Valid match
                pos_idx = indices[j]
                dist = distances[j]
                
                if pos_idx not in best_matches or dist < best_matches[pos_idx][0]:
                    best_matches[pos_idx] = (dist, df.iloc[j].to_dict())
    
    logger.info(f"Scanned {total_rows_scanned:,} manifest rows total")
    logger.info(f"Found matches for {len(best_matches)} of {len(positives)} positives")
    
    # Build result DataFrame
    results = []
    for pos_idx, (dist_chord, manifest_row) in best_matches.items():
        pos = positives.iloc[pos_idx]
        
        # Convert chord distance to arcsec
        sep_rad = 2 * np.arcsin(dist_chord / 2)
        sep_arcsec = np.degrees(sep_rad) * 3600
        
        row = manifest_row.copy()
        row["pos_name"] = pos["name"]
        row["pos_catalog_ra"] = pos["ra"]
        row["pos_catalog_dec"] = pos["dec"]
        row["zlens"] = pos.get("zlens", None)
        row["pos_type"] = pos.get("type", None)
        row["grading"] = pos["grading"]
        row["ref"] = pos.get("ref", None)
        row["tier"] = pos["tier"]
        row["weight"] = pos["weight"]
        row["separation_arcsec"] = sep_arcsec
        
        results.append(row)
    
    if results:
        return pd.DataFrame(results)
    else:
        logger.error("No matches found!")
        return pd.DataFrame()


# =============================================================================
# CROSSMATCH
# =============================================================================

def angular_separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Compute angular separation in arcseconds."""
    delta_ra = (ra2 - ra1) * np.cos(np.radians((dec1 + dec2) / 2))
    delta_dec = dec2 - dec1
    return np.sqrt(delta_ra**2 + delta_dec**2) * 3600


def crossmatch(positives: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Crossmatch positives with manifest.
    
    For each positive, find the nearest manifest row within MAX_MATCH_RADIUS_ARCSEC.
    """
    logger.info("Starting crossmatch...")
    
    results = []
    
    for idx, pos in positives.iterrows():
        healpix = pos["healpix_128"]
        pos_ra = pos["ra"]
        pos_dec = pos["dec"]
        
        # Get candidate matches from manifest
        candidates = manifest[manifest["healpix_128"] == healpix]
        
        if len(candidates) == 0:
            logger.warning(f"No candidates for {pos['name']} (healpix={healpix})")
            continue
        
        # Compute separations
        candidates = candidates.copy()
        candidates["separation_arcsec"] = candidates.apply(
            lambda row: angular_separation_arcsec(pos_ra, pos_dec, row["ra"], row["dec"]),
            axis=1
        )
        
        # Filter by radius
        candidates = candidates[candidates["separation_arcsec"] <= MAX_MATCH_RADIUS_ARCSEC]
        
        if len(candidates) == 0:
            logger.warning(f"No match within {MAX_MATCH_RADIUS_ARCSEC}\" for {pos['name']}")
            continue
        
        # Select nearest
        best_match = candidates.loc[candidates["separation_arcsec"].idxmin()].copy()
        
        # Add positive catalog info
        best_match["pos_name"] = pos["name"]
        best_match["pos_catalog_ra"] = pos_ra
        best_match["pos_catalog_dec"] = pos_dec
        best_match["zlens"] = pos.get("zlens", None)
        best_match["pos_type"] = pos.get("type", None)
        best_match["grading"] = pos["grading"]
        best_match["ref"] = pos.get("ref", None)
        best_match["tier"] = pos["tier"]
        best_match["weight"] = pos["weight"]
        
        results.append(best_match)
    
    if results:
        result_df = pd.DataFrame(results)
        logger.info(f"Matched {len(result_df)} of {len(positives)} positives")
        return result_df
    else:
        logger.error("No matches found!")
        return pd.DataFrame()


# =============================================================================
# OUTPUT
# =============================================================================

def save_output(df: pd.DataFrame, path: str) -> None:
    """Save output to S3 as parquet."""
    logger.info(f"Saving output to: {path}")
    
    # Add metadata
    df["crossmatch_timestamp"] = datetime.now(timezone.utc).isoformat()
    df["crossmatch_version"] = PIPELINE_VERSION
    df["match_radius_arcsec"] = MAX_MATCH_RADIUS_ARCSEC
    
    if path.startswith("s3://"):
        s3 = boto3.client("s3", region_name=AWS_REGION)
        bucket = path.replace("s3://", "").split("/")[0]
        key = "/".join(path.replace("s3://", "").split("/")[1:]).rstrip("/") + "/positives_with_dr10.parquet"
        
        # Write to bytes
        import io
        buffer = io.BytesIO()
        df.to_parquet(buffer, engine="pyarrow", compression="gzip", index=False)
        buffer.seek(0)
        
        s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        logger.info(f"Saved to s3://{bucket}/{key}")
    else:
        df.to_parquet(path, engine="pyarrow", compression="gzip", index=False)


def save_validation(df: pd.DataFrame, positives_count: int, path: str) -> None:
    """Save validation report."""
    validation = {
        "total_positives": positives_count,
        "matched_count": len(df),
        "match_rate": len(df) / positives_count if positives_count > 0 else 0,
        "tier_distribution": df["tier"].value_counts().to_dict() if len(df) > 0 else {},
        "split_distribution": df["split"].value_counts().to_dict() if "split" in df.columns and len(df) > 0 else {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    validation["gates_passed"] = validation["match_rate"] >= 0.95
    
    logger.info(f"Match rate: {validation['match_rate']*100:.1f}%")
    logger.info(f"Tier distribution: {validation['tier_distribution']}")
    logger.info(f"Split distribution: {validation['split_distribution']}")
    
    if path.startswith("s3://"):
        s3 = boto3.client("s3", region_name=AWS_REGION)
        bucket = path.replace("s3://", "").split("/")[0]
        key = "/".join(path.replace("s3://", "").split("/")[1:]).rstrip("/") + "/validation.json"
        s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(validation, indent=2))
        logger.info(f"Validation saved to s3://{bucket}/{key}")
    else:
        with open(path + "/validation.json", "w") as f:
            json.dump(validation, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Crossmatch positives with DR10 manifest")
    parser.add_argument("--manifest", required=True, help="S3 path to manifest")
    parser.add_argument("--positives", required=True, help="S3 path to positives CSV")
    parser.add_argument("--output", required=True, help="S3 output path")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Starting Local Crossmatch (Streaming KD-tree)")
    logger.info("=" * 60)
    logger.info(f"Match radius: {MAX_MATCH_RADIUS_ARCSEC} arcsec")
    
    start_time = time.time()
    
    try:
        # Load positives
        positives = load_positives(args.positives)
        
        # Stream manifest and crossmatch with KD-tree
        # (Note: healpix values in manifest are NOT spatially meaningful,
        # so we do brute-force spatial matching instead)
        matched = crossmatch_streaming(positives, args.manifest)
        
        if len(matched) == 0:
            logger.error("No matches found")
            sys.exit(1)
        
        # Save output
        save_output(matched, args.output)
        save_validation(matched, len(positives), args.output)
        
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

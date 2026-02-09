#!/usr/bin/env python3
"""
Spark Job: Crossmatch Positives with DR10 Sweep Files

This job matches the 5,104 positive lens candidates against original DR10 sweep files
to obtain Tractor source properties for each positive.

Why sweep files instead of manifest:
- The manifest intentionally excludes sources within 11" of known lenses (by design)
- Positives match Tractor sources with sub-arcsecond precision (0.06-0.16")
- Must use original sweep files to find the matching Tractor entries

Strategy:
- Broadcast the small positives catalog to all workers
- Each worker processes assigned sweep files in parallel
- For each sweep file, find nearest Tractor source to each positive in that region
- Aggregate results and select best match per positive

Usage:
    spark-submit --deploy-mode cluster spark_crossmatch_positives_v2.py \
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
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

PIPELINE_VERSION = "2.0.0"

# AWS Configuration (environment override supported)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

MAX_MATCH_RADIUS_ARCSEC = 5.0

TIER_WEIGHTS = {
    "confident": 0.95,
    "probable": 0.50,
}

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CrossmatchSpark] %(levelname)s: %(message)s",
)
logger = logging.getLogger("CrossmatchSpark")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_sweep_bounds(filename: str) -> Tuple[float, float, float, float]:
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


def check_already_processed(s3_client, checkpoint_bucket: str, checkpoint_prefix: str, filename: str) -> bool:
    """Check if a sweep file was already processed (checkpoint exists)."""
    checkpoint_key = f"{checkpoint_prefix}/checkpoints/{filename}.done"
    try:
        s3_client.head_object(Bucket=checkpoint_bucket, Key=checkpoint_key)
        return True
    except:
        return False


def write_checkpoint(s3_client, checkpoint_bucket: str, checkpoint_prefix: str, filename: str, results: List[Dict]):
    """Write checkpoint for a processed sweep file."""
    import json
    checkpoint_key = f"{checkpoint_prefix}/checkpoints/{filename}.done"
    results_key = f"{checkpoint_prefix}/partial/{filename}.json"
    
    # Write results
    s3_client.put_object(
        Bucket=checkpoint_bucket,
        Key=results_key,
        Body=json.dumps(results)
    )
    
    # Write checkpoint marker
    s3_client.put_object(
        Bucket=checkpoint_bucket,
        Key=checkpoint_key,
        Body=f"{len(results)} matches"
    )


def process_sweep_file(
    sweep_path: str,
    positives_bc: List[Dict],
    s3_bucket: str,
    checkpoint_prefix: str,
) -> Iterator[Dict]:
    """
    Process a single sweep file and find matches to positives.
    
    This runs on worker nodes - each worker processes its assigned sweep files.
    Includes checkpointing for resume capability.
    """
    import boto3
    from astropy.io import fits
    
    filename = sweep_path.split("/")[-1]
    s3 = boto3.client("s3", region_name=AWS_REGION)
    
    # Check if already processed (resume support)
    if check_already_processed(s3, s3_bucket, checkpoint_prefix, filename):
        logger.info(f"Skipping {filename} - already processed")
        # Load cached results
        try:
            import json
            results_key = f"{checkpoint_prefix}/partial/{filename}.json"
            obj = s3.get_object(Bucket=s3_bucket, Key=results_key)
            cached_results = json.loads(obj["Body"].read().decode("utf-8"))
            for r in cached_results:
                yield r
            return
        except Exception as e:
            logger.warning(f"Could not load cached results for {filename}: {e}")
    
    try:
        # Parse bounds from filename
        ra_min, ra_max, dec_min, dec_max = parse_sweep_bounds(filename)
        
        # Filter positives to those in this sweep's region
        positives_in_region = [
            p for p in positives_bc
            if ra_min <= p["ra"] < ra_max and dec_min <= p["dec"] < dec_max
        ]
        
        if not positives_in_region:
            # Write empty checkpoint
            write_checkpoint(s3, s3_bucket, checkpoint_prefix, filename, [])
            return
        
        # Download sweep file
        key = sweep_path.replace(f"s3://{s3_bucket}/", "")
        
        obj = s3.get_object(Bucket=s3_bucket, Key=key)
        data = obj["Body"].read()
        decompressed = gzip.decompress(data)
        del data
        
        # Read FITS
        buffer = io.BytesIO(decompressed)
        with fits.open(buffer, memmap=False) as hdu:
            sweep_data = hdu[1].data
            ra_arr = np.array(sweep_data["RA"], dtype=np.float32)
            dec_arr = np.array(sweep_data["DEC"], dtype=np.float32)
            type_arr = np.array([
                t.decode("utf-8").strip() if isinstance(t, bytes) else str(t).strip()
                for t in sweep_data["TYPE"]
            ])
            
            # Get additional columns for output
            brickname_arr = np.array([
                b.decode("utf-8").strip() if isinstance(b, bytes) else str(b).strip()
                for b in sweep_data["BRICKNAME"]
            ])
            objid_arr = np.array(sweep_data["OBJID"], dtype=np.int64)
            nobs_z_arr = np.array(sweep_data["NOBS_Z"], dtype=np.int32)
            
            # Flux columns
            flux_g_arr = np.array(sweep_data["FLUX_G"], dtype=np.float32)
            flux_r_arr = np.array(sweep_data["FLUX_R"], dtype=np.float32)
            flux_z_arr = np.array(sweep_data["FLUX_Z"], dtype=np.float32)
        
        del decompressed, buffer
        
        # Collect results for checkpointing
        results = []
        
        # Match each positive
        for pos in positives_in_region:
            pos_ra = pos["ra"]
            pos_dec = pos["dec"]
            
            # Compute angular separation in arcsec
            cos_dec = np.cos(np.radians(pos_dec))
            seps = np.sqrt(
                ((ra_arr - pos_ra) * cos_dec)**2 + (dec_arr - pos_dec)**2
            ) * 3600
            
            # Find nearest match
            nearest_idx = np.argmin(seps)
            nearest_sep = float(seps[nearest_idx])
            
            # Only yield if within radius
            if nearest_sep <= MAX_MATCH_RADIUS_ARCSEC:
                result = {
                    # Positive info
                    "pos_name": pos["name"],
                    "pos_catalog_ra": pos["ra"],
                    "pos_catalog_dec": pos["dec"],
                    "zlens": pos.get("zlens"),
                    "pos_type": pos.get("type"),
                    "grading": pos["grading"],
                    "ref": pos.get("ref"),
                    "tier": pos["tier"],
                    "weight": pos["weight"],
                    
                    # Match info
                    "match_ra": float(ra_arr[nearest_idx]),
                    "match_dec": float(dec_arr[nearest_idx]),
                    "match_type": type_arr[nearest_idx],
                    "brickname": brickname_arr[nearest_idx],
                    "objid": int(objid_arr[nearest_idx]),
                    "galaxy_id": f"{brickname_arr[nearest_idx]}_{objid_arr[nearest_idx]}",
                    "separation_arcsec": nearest_sep,
                    
                    # Tractor properties
                    "nobs_z": int(nobs_z_arr[nearest_idx]),
                    "flux_g": float(flux_g_arr[nearest_idx]),
                    "flux_r": float(flux_r_arr[nearest_idx]),
                    "flux_z": float(flux_z_arr[nearest_idx]),
                    
                    # Provenance
                    "sweep_file": filename,
                }
                results.append(result)
                yield result
        
        # Write checkpoint after processing
        write_checkpoint(s3, s3_bucket, checkpoint_prefix, filename, results)
    
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        return


def main():
    parser = argparse.ArgumentParser(description="Crossmatch positives with DR10 sweeps (Spark)")
    parser.add_argument("--sweeps", required=True, help="S3 path to sweep files")
    parser.add_argument("--positives", required=True, help="S3 path to positives CSV")
    parser.add_argument("--output", required=True, help="S3 output path")
    
    args = parser.parse_args()
    
    # Initialize Spark
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        StructType, StructField, StringType, FloatType, DoubleType, IntegerType, LongType
    )
    import pyspark.sql.functions as F
    
    spark = SparkSession.builder \
        .appName("CrossmatchPositivesWithSweeps") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    
    sc = spark.sparkContext
    
    logger.info("=" * 60)
    logger.info("Starting Spark Crossmatch Job")
    logger.info("=" * 60)
    logger.info(f"Match radius: {MAX_MATCH_RADIUS_ARCSEC} arcsec")
    logger.info(f"Sweeps: {args.sweeps}")
    logger.info(f"Positives: {args.positives}")
    logger.info(f"Output: {args.output}")
    
    start_time = time.time()
    
    try:
        import boto3
        import pandas as pd
        
        # Parse S3 paths
        sweeps_bucket = args.sweeps.replace("s3://", "").split("/")[0]
        sweeps_prefix = "/".join(args.sweeps.replace("s3://", "").split("/")[1:]).rstrip("/") + "/"
        
        # Load positives
        logger.info("Loading positives...")
        s3 = boto3.client("s3", region_name=AWS_REGION)
        pos_bucket = args.positives.replace("s3://", "").split("/")[0]
        pos_key = "/".join(args.positives.replace("s3://", "").split("/")[1:])
        
        response = s3.get_object(Bucket=pos_bucket, Key=pos_key)
        positives_df = pd.read_csv(response["Body"])
        
        # Add tier and weight
        positives_df["tier"] = positives_df["grading"].apply(lambda x: "A" if x == "confident" else "B")
        positives_df["weight"] = positives_df["tier"].apply(
            lambda x: TIER_WEIGHTS["confident"] if x == "A" else TIER_WEIGHTS["probable"]
        )
        
        positives_list = positives_df.to_dict("records")
        logger.info(f"Loaded {len(positives_list)} positives")
        
        # Broadcast positives to all workers
        positives_bc = sc.broadcast(positives_list)
        
        # List sweep files
        logger.info("Listing sweep files...")
        sweep_files = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=sweeps_bucket, Prefix=sweeps_prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".fits.gz"):
                    sweep_files.append(f"s3://{sweeps_bucket}/{obj['Key']}")
        
        logger.info(f"Found {len(sweep_files)} sweep files")
        
        # Checkpoint prefix for resume capability
        checkpoint_prefix = args.output.replace(f"s3://{sweeps_bucket}/", "").rstrip("/")
        checkpoint_prefix_bc = sc.broadcast(checkpoint_prefix)
        
        # Create RDD of sweep file paths
        sweep_rdd = sc.parallelize(sweep_files, numSlices=min(len(sweep_files), 200))
        
        # Process each sweep file in parallel (with checkpointing)
        logger.info("Processing sweep files in parallel (with checkpointing)...")
        results_rdd = sweep_rdd.flatMap(
            lambda path: process_sweep_file(path, positives_bc.value, sweeps_bucket, checkpoint_prefix_bc.value)
        )
        
        # Collect results
        results = results_rdd.collect()
        logger.info(f"Found {len(results)} matches")
        
        if not results:
            logger.error("No matches found!")
            sys.exit(1)
        
        # Convert to DataFrame
        result_df = pd.DataFrame(results)
        
        # If multiple matches per positive (from overlapping sweeps), keep best
        result_df = result_df.sort_values("separation_arcsec").drop_duplicates(
            subset=["pos_name"], keep="first"
        )
        logger.info(f"After deduplication: {len(result_df)} unique positives matched")
        
        # Add metadata
        result_df["crossmatch_timestamp"] = datetime.now(timezone.utc).isoformat()
        result_df["crossmatch_version"] = PIPELINE_VERSION
        result_df["match_radius_arcsec"] = MAX_MATCH_RADIUS_ARCSEC
        
        # Convert to Spark DataFrame for saving
        result_spark = spark.createDataFrame(result_df)
        
        # Save output
        output_path = args.output.rstrip("/") + "/positives_with_dr10.parquet"
        logger.info(f"Saving to: {output_path}")
        
        result_spark.coalesce(1).write.mode("overwrite") \
            .option("compression", "gzip") \
            .parquet(args.output.rstrip("/") + "/data/")
        
        # Save validation report
        validation = {
            "total_positives": len(positives_list),
            "matched_count": len(result_df),
            "match_rate": len(result_df) / len(positives_list),
            "sweep_files_processed": len(sweep_files),
            "tier_distribution": result_df["tier"].value_counts().to_dict(),
            "type_distribution": result_df["match_type"].value_counts().to_dict(),
            "separation_stats": {
                "min": float(result_df["separation_arcsec"].min()),
                "max": float(result_df["separation_arcsec"].max()),
                "mean": float(result_df["separation_arcsec"].mean()),
                "median": float(result_df["separation_arcsec"].median()),
            },
            "within_1_arcsec": int((result_df["separation_arcsec"] <= 1).sum()),
            "within_5_arcsec": int((result_df["separation_arcsec"] <= 5).sum()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": PIPELINE_VERSION,
        }
        
        validation["gates_passed"] = validation["match_rate"] >= 0.90  # 90% match rate gate
        
        logger.info(f"Match rate: {validation['match_rate']*100:.1f}%")
        logger.info(f"Tier distribution: {validation['tier_distribution']}")
        logger.info(f"Type distribution: {validation['type_distribution']}")
        logger.info(f"Separation stats: {validation['separation_stats']}")
        logger.info(f"Within 1\": {validation['within_1_arcsec']}")
        logger.info(f"Gates passed: {validation['gates_passed']}")
        
        # Save validation to S3
        val_key = args.output.replace(f"s3://{pos_bucket}/", "").rstrip("/") + "/validation.json"
        s3.put_object(
            Bucket=pos_bucket,
            Key=val_key,
            Body=json.dumps(validation, indent=2)
        )
        logger.info(f"Validation saved to s3://{pos_bucket}/{val_key}")
        
        elapsed = time.time() - start_time
        logger.info(f"Completed in {elapsed:.1f}s")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Job failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Spark Job: Crossmatch Positive Catalog with DR10 Manifest

This job crossmatches the 5,104 positive lens candidates with the DR10
negative pool manifest to obtain observing conditions (nobs_z, type_bin,
PSF/depth columns) for stratified sampling.

Why this is needed:
- The positive catalog only has: name, ra, dec, zlens, type, grading, ref
- For 100:1 per-stratum sampling, we need positives to have same stratification
  columns as negatives: nobs_z_bin, type_bin, healpix_128, split

Strategy:
- Read manifest (114M rows) with all DR10 columns
- Read positive catalog (5,104 rows) as Pandas (small enough)
- For each positive, compute healpix_128 and find matching manifest rows
- Select nearest match within radius for each positive
- Output: positives_with_dr10.parquet with all manifest columns + tier/weight

Usage:
    spark-submit --deploy-mode cluster spark_crossmatch_positives.py \
        --manifest s3://darkhaloscope/stronglens_calibration/manifests/20260208_074343/ \
        --positives s3://darkhaloscope/stronglens_calibration/configs/positives/desi_candidates.csv \
        --output s3://darkhaloscope/stronglens_calibration/positives_with_dr10/

Author: Generated for stronglens_calibration project
Date: 2026-02-08
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from pyspark import StorageLevel, broadcast
from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType,
    FloatType, DoubleType, BooleanType
)
from pyspark.sql.window import Window


# =============================================================================
# CONSTANTS
# =============================================================================

PIPELINE_VERSION = "1.0.0"
GIT_COMMIT = os.environ.get("GIT_COMMIT", "unknown")

# AWS Configuration (environment override supported)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

# Crossmatch parameters
MAX_MATCH_RADIUS_ARCSEC = 1.0  # Maximum separation for a valid match
HEALPIX_NSIDE = 128  # Use same nside as manifest for direct join

# Tier weights (from LLM Section C)
TIER_WEIGHTS = {
    "confident": 0.95,  # Tier-A
    "probable": 0.50,   # Tier-B
}


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(job_name: str) -> logging.Logger:
    """Set up logging for Spark job."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{job_name}] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(job_name)


# =============================================================================
# HELPER FUNCTIONS (no UDFs - pure Spark SQL)
# =============================================================================

def compute_healpix_scalar(ra: float, dec: float, nside: int) -> int:
    """Compute HEALPix index for given coordinates (scalar version)."""
    try:
        import healpy as hp
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        return int(hp.ang2pix(nside, theta, phi, nest=True))
    except ImportError:
        # Fallback without healpy
        return int(hash((round(ra, 4), round(dec, 4), nside)) % (12 * nside * nside))


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def load_positive_catalog_pandas(path: str, logger: logging.Logger) -> "pd.DataFrame":
    """
    Load positive catalog as Pandas DataFrame (small enough to fit in memory).
    Also computes HEALPix for each positive using local Python.
    """
    import pandas as pd
    import boto3
    import tempfile
    
    logger.info(f"Loading positive catalog from: {path}")
    
    # Handle S3 paths
    if path.startswith("s3://"):
        s3 = boto3.client("s3", region_name=AWS_REGION)
        bucket = path.replace("s3://", "").split("/")[0]
        key = "/".join(path.replace("s3://", "").split("/")[1:])
        
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            s3.download_file(bucket, key, tmp.name)
            df = pd.read_csv(tmp.name)
    else:
        df = pd.read_csv(path)
    
    logger.info(f"Loaded {len(df)} positives")
    
    # Compute HEALPix for each positive
    healpix_vals = []
    for _, row in df.iterrows():
        hp = compute_healpix_scalar(row["ra"], row["dec"], HEALPIX_NSIDE)
        healpix_vals.append(hp)
    df["healpix_128"] = healpix_vals
    
    # Add tier based on grading
    df["tier"] = df["grading"].apply(lambda x: "A" if x == "confident" else "B")
    
    # Add weight based on tier
    df["weight"] = df["tier"].apply(
        lambda x: TIER_WEIGHTS["confident"] if x == "A" else TIER_WEIGHTS["probable"]
    )
    
    # Rename to avoid conflicts with manifest
    df = df.rename(columns={
        "ra": "pos_ra",
        "dec": "pos_dec",
        "type": "pos_type"
    })
    
    logger.info(f"Computed HEALPix for {len(df)} positives")
    return df


def load_manifest(
    spark: SparkSession,
    path: str,
    logger: logging.Logger,
    sample_frac: Optional[float] = None
) -> DataFrame:
    """Load manifest from parquet."""
    logger.info(f"Loading manifest from: {path}")
    
    df = spark.read.parquet(path)
    
    if sample_frac:
        logger.info(f"Sampling {sample_frac*100:.1f}% for testing")
        df = df.sample(False, sample_frac, seed=42)
    
    initial_count = df.count()
    logger.info(f"Loaded {initial_count} manifest rows")
    
    return df


def crossmatch_with_broadcast(
    spark: SparkSession,
    positives_pdf: "pd.DataFrame",
    manifest_df: DataFrame,
    logger: logging.Logger,
    max_sep_arcsec: float = MAX_MATCH_RADIUS_ARCSEC
) -> DataFrame:
    """
    Crossmatch using broadcast join on HEALPix.
    
    Strategy:
    1. Convert positives to Spark DataFrame and broadcast (small)
    2. Join with manifest on healpix_128
    3. Compute angular separation using Spark SQL (no UDF)
    4. Select nearest match per positive
    """
    logger.info(f"Starting crossmatch with {max_sep_arcsec}\" radius")
    
    # Convert positives to Spark DataFrame
    positives_df = spark.createDataFrame(positives_pdf)
    logger.info(f"Created Spark DataFrame with {positives_df.count()} positives")
    
    # Join on healpix_128
    logger.info("Joining on healpix_128...")
    
    # Rename manifest columns to avoid conflicts
    manifest_cols = manifest_df.columns
    manifest_renamed = manifest_df
    for col in manifest_cols:
        if col not in ["healpix_128"]:
            manifest_renamed = manifest_renamed.withColumnRenamed(col, f"m_{col}")
    
    # Broadcast the small positives table
    joined = manifest_renamed.join(
        F.broadcast(positives_df),
        on="healpix_128",
        how="inner"
    )
    
    logger.info("Computing angular separations using Spark SQL...")
    
    # Use Spark SQL for angular separation calculation (Haversine approximation)
    # For small separations, this is accurate enough
    # sep = sqrt((delta_ra * cos(dec))^2 + delta_dec^2) * 3600
    joined = joined.withColumn(
        "delta_ra_deg",
        F.col("m_ra") - F.col("pos_ra")
    ).withColumn(
        "delta_dec_deg",
        F.col("m_dec") - F.col("pos_dec")
    ).withColumn(
        "cos_dec",
        F.cos(F.radians((F.col("m_dec") + F.col("pos_dec")) / 2))
    ).withColumn(
        "separation_arcsec",
        F.sqrt(
            F.pow(F.col("delta_ra_deg") * F.col("cos_dec"), 2) +
            F.pow(F.col("delta_dec_deg"), 2)
        ) * 3600
    )
    
    # Filter by max separation
    matched = joined.filter(F.col("separation_arcsec") <= max_sep_arcsec)
    
    # Select nearest match for each positive
    logger.info("Selecting nearest match for each positive...")
    window = Window.partitionBy("name").orderBy("separation_arcsec")
    matched = matched.withColumn("rank", F.row_number().over(window))
    matched = matched.filter(F.col("rank") == 1)
    
    # Drop helper columns
    matched = matched.drop("delta_ra_deg", "delta_dec_deg", "cos_dec", "rank")
    
    match_count = matched.count()
    logger.info(f"Found {match_count} matches")
    
    return matched


def prepare_output(
    matched_df: DataFrame,
    logger: logging.Logger
) -> DataFrame:
    """
    Prepare final output with renamed columns and added metadata.
    """
    logger.info("Preparing output...")
    
    # Rename manifest columns back (remove m_ prefix)
    result = matched_df
    for col in matched_df.columns:
        if col.startswith("m_"):
            new_name = col[2:]  # Remove m_ prefix
            result = result.withColumnRenamed(col, new_name)
    
    # Keep key positive columns
    result = result.withColumnRenamed("pos_ra", "pos_catalog_ra") \
                   .withColumnRenamed("pos_dec", "pos_catalog_dec")
    
    # Add metadata columns
    timestamp = datetime.now(timezone.utc).isoformat()
    result = result.withColumn("crossmatch_timestamp", F.lit(timestamp))
    result = result.withColumn("crossmatch_version", F.lit(PIPELINE_VERSION))
    result = result.withColumn("crossmatch_git_commit", F.lit(GIT_COMMIT))
    result = result.withColumn("match_radius_arcsec", F.lit(MAX_MATCH_RADIUS_ARCSEC))
    
    # Reorder columns - put positive catalog info first
    pos_cols = ["name", "pos_catalog_ra", "pos_catalog_dec", "zlens", "pos_type", 
                "grading", "ref", "tier", "weight", "separation_arcsec"]
    other_cols = [c for c in result.columns if c not in pos_cols and c != "healpix_128"]
    
    # Select in order, only including columns that exist
    select_cols = [c for c in pos_cols if c in result.columns] + sorted(other_cols)
    result = result.select(select_cols)
    
    return result


def validate_output(
    df: DataFrame,
    positives_count: int,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Validate crossmatch output for quality gates.
    """
    logger.info("Validating output...")
    
    matched_count = df.count()
    
    validation = {
        "total_positives": positives_count,
        "matched_count": matched_count,
        "match_rate": matched_count / positives_count if positives_count > 0 else 0.0,
        "gates_passed": True,
        "issues": [],
    }
    
    # Gate 1: Match rate should be high (>95%)
    if validation["match_rate"] < 0.95:
        validation["issues"].append(
            f"Low match rate: {validation['match_rate']*100:.1f}% (expected >95%)"
        )
        validation["gates_passed"] = False
    
    # Gate 2: All matched should have valid nobs_z_bin and type_bin
    if "nobs_z_bin" in df.columns:
        null_nobs = df.filter(F.col("nobs_z_bin").isNull()).count()
        if null_nobs > 0:
            validation["issues"].append(f"{null_nobs} rows with null nobs_z_bin")
    
    if "type_bin" in df.columns:
        null_type = df.filter(F.col("type_bin").isNull()).count()
        if null_type > 0:
            validation["issues"].append(f"{null_type} rows with null type_bin")
    
    # Gate 3: Tier distribution
    tier_counts = df.groupBy("tier").count().collect()
    validation["tier_distribution"] = {row["tier"]: row["count"] for row in tier_counts}
    
    # Gate 4: Split distribution
    if "split" in df.columns:
        split_counts = df.groupBy("split").count().collect()
        validation["split_distribution"] = {row["split"]: row["count"] for row in split_counts}
    
    # Log validation results
    logger.info(f"Match rate: {validation['match_rate']*100:.1f}%")
    logger.info(f"Tier distribution: {validation.get('tier_distribution', {})}")
    logger.info(f"Split distribution: {validation.get('split_distribution', {})}")
    
    if validation["issues"]:
        for issue in validation["issues"]:
            logger.warning(f"Validation issue: {issue}")
    
    return validation


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Crossmatch positives with DR10 manifest")
    parser.add_argument("--manifest", required=True,
                        help="S3 path to manifest parquet (e.g., s3://bucket/manifests/...)")
    parser.add_argument("--positives", required=True,
                        help="S3 path to positive catalog CSV")
    parser.add_argument("--output", required=True,
                        help="S3 output path for crossmatched parquet")
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Sample fraction of manifest for testing (e.g., 0.01)")
    parser.add_argument("--max-sep", type=float, default=MAX_MATCH_RADIUS_ARCSEC,
                        help=f"Max separation in arcsec (default: {MAX_MATCH_RADIUS_ARCSEC})")
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging("CrossmatchPositives")
    logger.info("=" * 60)
    logger.info("Starting Positive Catalog Crossmatch")
    logger.info("=" * 60)
    logger.info(f"Manifest: {args.manifest}")
    logger.info(f"Positives: {args.positives}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Max separation: {args.max_sep}\"")
    if args.sample_frac:
        logger.info(f"Sample fraction: {args.sample_frac}")
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("CrossmatchPositives") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    
    start_time = time.time()
    
    try:
        # Load positives as Pandas (small dataset)
        positives_pdf = load_positive_catalog_pandas(args.positives, logger)
        positives_count = len(positives_pdf)
        
        # Load manifest as Spark DataFrame
        manifest_df = load_manifest(spark, args.manifest, logger, args.sample_frac)
        
        # Crossmatch using broadcast join
        matched_df = crossmatch_with_broadcast(
            spark, positives_pdf, manifest_df, logger, args.max_sep
        )
        
        # Prepare output
        output_df = prepare_output(matched_df, logger)
        
        # Validate
        validation = validate_output(output_df, positives_count, logger)
        
        # Write output
        logger.info(f"Writing output to {args.output}")
        output_df.coalesce(1).write \
            .mode("overwrite") \
            .option("compression", "gzip") \
            .parquet(args.output)
        
        # Write validation report
        import json
        validation_json = json.dumps(validation, indent=2)
        
        # Save validation to S3
        if args.output.startswith("s3://"):
            import boto3
            s3 = boto3.client("s3", region_name=AWS_REGION)
            bucket = args.output.replace("s3://", "").split("/")[0]
            key = "/".join(args.output.replace("s3://", "").split("/")[1:]).rstrip("/") + "_validation.json"
            s3.put_object(Bucket=bucket, Key=key, Body=validation_json)
            logger.info(f"Validation report saved to s3://{bucket}/{key}")
        
        elapsed = time.time() - start_time
        logger.info(f"Completed in {elapsed:.1f}s")
        logger.info(f"Gates passed: {validation['gates_passed']}")
        
        if not validation["gates_passed"]:
            logger.warning("VALIDATION GATES FAILED - review issues above")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Job failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

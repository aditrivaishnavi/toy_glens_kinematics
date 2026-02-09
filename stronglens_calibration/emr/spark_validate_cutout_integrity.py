#!/usr/bin/env python3
"""
Spark Job: Validate Cutout Integrity

Validates that stored cutouts match fresh downloads from Legacy Survey
by re-downloading and comparing pixel-by-pixel.

This is critical QA for ensuring our FITS parsing and storage are correct.

Usage:
    spark-submit --deploy-mode cluster spark_validate_cutout_integrity.py \
        --input s3://darkhaloscope/stronglens_calibration/cutouts/positives/ \
        --output s3://darkhaloscope/stronglens_calibration/validation/positives/

Author: Generated for stronglens_calibration project
Date: 2026-02-08
"""
import argparse
import io
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Tuple
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

PIPELINE_VERSION = "1.0.0"

# AWS Configuration (environment override supported)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

# Cutout parameters (must match generation job)
CUTOUT_SIZE = 101  # pixels
PIXEL_SCALE = 0.262  # arcsec/pixel

# Legacy Survey cutout service
CUTOUT_URL_TEMPLATE = (
    "https://www.legacysurvey.org/viewer/fits-cutout"
    "?ra={ra}&dec={dec}&size={size}&layer=ls-dr10&pixscale={pixscale}&bands={bands}"
)

# Retry parameters
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds

# Tolerance for floating point comparison
ATOL = 1e-8
RTOL = 1e-5

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ValidateCutout] %(levelname)s: %(message)s",
)
logger = logging.getLogger("ValidateCutout")


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def download_fresh_cutout(
    ra: float,
    dec: float,
    size: int = CUTOUT_SIZE,
    bands: str = "grz",
    max_retries: int = MAX_RETRIES,
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Download fresh cutout from Legacy Survey for comparison.
    
    Uses IDENTICAL parameters to the original generation to ensure
    any discrepancy indicates a bug, not parameter differences.
    """
    url = CUTOUT_URL_TEMPLATE.format(
        ra=ra,
        dec=dec,
        size=size,
        pixscale=PIXEL_SCALE,
        bands=bands,
    )
    
    provenance = {
        "validation_url": url,
        "layer": "ls-dr10",
        "bands": bands,
        "pixel_scale": PIXEL_SCALE,
        "size": size,
    }
    
    for attempt in range(max_retries):
        try:
            with urlopen(url, timeout=60) as response:
                fits_data = response.read()
            
            # Parse FITS data - MUST match spark_generate_cutouts.py exactly
            from astropy.io import fits
            buffer = io.BytesIO(fits_data)
            with fits.open(buffer) as hdul:
                data = hdul[0].data  # Shape: (n_bands, H, W)
                
                if data is None:
                    provenance["error"] = "no_data_in_hdu0"
                    return None, provenance
                
                if data.ndim != 3:
                    provenance["error"] = f"wrong_ndim_{data.ndim}"
                    return None, provenance
                
                n_bands, h, w = data.shape
                if n_bands != 3:
                    provenance["error"] = f"wrong_n_bands_{n_bands}"
                    return None, provenance
                
                # Transpose to (H, W, bands) - MUST match generation
                cutout = np.transpose(data, (1, 2, 0)).astype(np.float32)
                
                return cutout, provenance
                
        except (URLError, HTTPError, TimeoutError) as e:
            if attempt < max_retries - 1:
                delay = RETRY_DELAY_BASE * (2 ** attempt)
                time.sleep(delay)
            else:
                provenance["error"] = f"download_failed_{type(e).__name__}"
                return None, provenance
        except Exception as e:
            provenance["error"] = f"parse_error_{type(e).__name__}"
            return None, provenance
    
    provenance["error"] = "max_retries_exceeded"
    return None, provenance


def compare_cutouts(stored: np.ndarray, fresh: np.ndarray) -> Dict:
    """
    Compare stored cutout against fresh download.
    
    Returns detailed metrics for debugging any discrepancies.
    """
    # Basic shape check
    if stored.shape != fresh.shape:
        return {
            "valid": False,
            "error": f"shape_mismatch_stored_{stored.shape}_fresh_{fresh.shape}",
            "identical": False,
        }
    
    # Handle NaN values - they should be in same positions
    stored_nan = np.isnan(stored)
    fresh_nan = np.isnan(fresh)
    nan_positions_match = np.array_equal(stored_nan, fresh_nan)
    
    # Pixel-wise difference (ignoring NaNs)
    valid_mask = ~stored_nan & ~fresh_nan
    if np.sum(valid_mask) == 0:
        return {
            "valid": True,
            "all_nan": True,
            "identical": True,
            "nan_positions_match": True,
        }
    
    diff = np.abs(stored - fresh)
    diff_valid = diff[valid_mask]
    
    # Metrics
    max_diff = float(np.max(diff_valid))
    mean_diff = float(np.mean(diff_valid))
    rmse = float(np.sqrt(np.mean(diff_valid**2)))
    
    # Check if identical within tolerance
    identical = np.allclose(
        stored[valid_mask], 
        fresh[valid_mask], 
        rtol=RTOL, 
        atol=ATOL
    ) and nan_positions_match
    
    # Per-band metrics
    band_metrics = {}
    for i, band in enumerate(["g", "r", "z"]):
        stored_band = stored[:, :, i]
        fresh_band = fresh[:, :, i]
        
        band_valid = ~np.isnan(stored_band) & ~np.isnan(fresh_band)
        if np.sum(band_valid) == 0:
            band_metrics[band] = {"all_nan": True}
            continue
        
        band_diff = np.abs(stored_band - fresh_band)[band_valid]
        
        # Correlation
        corr = np.corrcoef(
            stored_band[band_valid].ravel(),
            fresh_band[band_valid].ravel()
        )[0, 1]
        
        band_metrics[band] = {
            "max_diff": float(np.max(band_diff)),
            "mean_diff": float(np.mean(band_diff)),
            "correlation": float(corr),
            "identical": np.allclose(
                stored_band[band_valid],
                fresh_band[band_valid],
                rtol=RTOL,
                atol=ATOL
            ),
        }
    
    return {
        "valid": True,
        "identical": identical,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rmse": rmse,
        "nan_positions_match": nan_positions_match,
        "bands": band_metrics,
    }


def load_cutout_from_s3(s3_client, bucket: str, key: str) -> Tuple[Optional[np.ndarray], Dict]:
    """Load cutout and metadata from S3 NPZ file."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        buffer = io.BytesIO(response["Body"].read())
        npz = np.load(buffer)
        
        cutout = npz["cutout"]
        
        # Extract metadata
        metadata = {}
        for k in npz.files:
            if k.startswith("meta_"):
                val = npz[k]
                # Handle numpy scalars
                if val.ndim == 0:
                    val = val.item()
                metadata[k.replace("meta_", "")] = val
        
        return cutout, metadata
        
    except Exception as e:
        return None, {"error": str(e)}


# =============================================================================
# SPARK PROCESSING
# =============================================================================

def validate_cutout(
    s3_client,
    bucket: str,
    key: str,
) -> Dict:
    """
    Validate a single cutout by comparing with fresh download.
    
    Returns validation result with metrics.
    """
    galaxy_id = key.split("/")[-1].replace(".npz", "")
    
    result = {
        "galaxy_id": galaxy_id,
        "s3_key": key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    # Load stored cutout
    stored, metadata = load_cutout_from_s3(s3_client, bucket, key)
    
    if stored is None:
        result["status"] = "load_failed"
        result["error"] = metadata.get("error", "unknown")
        return result
    
    # Get coordinates
    ra = metadata.get("ra")
    dec = metadata.get("dec")
    
    if ra is None or dec is None:
        result["status"] = "missing_coordinates"
        result["error"] = f"ra={ra}, dec={dec}"
        return result
    
    result["ra"] = float(ra)
    result["dec"] = float(dec)
    
    # Validate stored cutout shape
    if stored.shape != (CUTOUT_SIZE, CUTOUT_SIZE, 3):
        result["status"] = "wrong_shape"
        result["error"] = f"stored_shape={stored.shape}"
        return result
    
    # Download fresh cutout
    fresh, provenance = download_fresh_cutout(ra, dec)
    
    if fresh is None:
        result["status"] = "download_failed"
        result["error"] = provenance.get("error", "unknown")
        return result
    
    # Compare
    comparison = compare_cutouts(stored, fresh)
    
    if not comparison.get("valid", False):
        result["status"] = "comparison_error"
        result["error"] = comparison.get("error", "unknown")
        return result
    
    # Success - record metrics
    result["status"] = "validated"
    result["identical"] = comparison["identical"]
    result["max_diff"] = comparison.get("max_diff", 0)
    result["mean_diff"] = comparison.get("mean_diff", 0)
    result["rmse"] = comparison.get("rmse", 0)
    result["nan_positions_match"] = comparison.get("nan_positions_match", True)
    
    # Per-band correlations (key quality metric)
    if "bands" in comparison:
        for band, metrics in comparison["bands"].items():
            if isinstance(metrics, dict):
                result[f"corr_{band}"] = metrics.get("correlation", 1.0)
                result[f"identical_{band}"] = metrics.get("identical", True)
    
    return result


def process_partition(
    keys: Iterator[str],
    bucket: str,
) -> Iterator[Dict]:
    """Process a partition of S3 keys."""
    import boto3
    
    s3 = boto3.client("s3", region_name=AWS_REGION)
    
    for key in keys:
        result = validate_cutout(s3, bucket, key)
        yield result
        
        # Rate limiting to avoid overwhelming Legacy Survey
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="Validate cutout integrity against Legacy Survey")
    parser.add_argument("--input", required=True, help="S3 path to cutouts directory")
    parser.add_argument("--output", required=True, help="S3 path for validation results")
    parser.add_argument("--sample-fraction", type=float, default=1.0,
                       help="Fraction of cutouts to validate (0-1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling")
    
    args = parser.parse_args()
    
    # Initialize Spark
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        StructType, StructField, StringType, FloatType, BooleanType, DoubleType
    )
    import pyspark.sql.functions as F
    
    spark = SparkSession.builder \
        .appName("ValidateCutoutIntegrity") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    sc = spark.sparkContext
    
    logger.info("=" * 60)
    logger.info("Starting Cutout Integrity Validation")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Sample fraction: {args.sample_fraction}")
    
    start_time = time.time()
    
    try:
        # Parse S3 paths
        input_bucket = args.input.replace("s3://", "").split("/")[0]
        input_prefix = "/".join(args.input.replace("s3://", "").split("/")[1:]).rstrip("/")
        
        output_bucket = args.output.replace("s3://", "").split("/")[0]
        output_prefix = "/".join(args.output.replace("s3://", "").split("/")[1:]).rstrip("/")
        
        # List all cutout files
        logger.info("Listing cutout files...")
        import boto3
        s3 = boto3.client("s3", region_name=AWS_REGION)
        
        cutout_keys = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=input_bucket, Prefix=input_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".npz"):
                    cutout_keys.append(key)
        
        total_cutouts = len(cutout_keys)
        logger.info(f"Found {total_cutouts} cutout files")
        
        if total_cutouts == 0:
            logger.warning("No cutouts found!")
            spark.stop()
            sys.exit(0)
        
        # Sample if requested
        if args.sample_fraction < 1.0:
            import random
            random.seed(args.seed)
            n_sample = max(1, int(total_cutouts * args.sample_fraction))
            cutout_keys = random.sample(cutout_keys, n_sample)
            logger.info(f"Sampled {len(cutout_keys)} cutouts for validation")
        
        # Broadcast bucket
        input_bucket_bc = sc.broadcast(input_bucket)
        
        # Distribute keys and process
        keys_rdd = sc.parallelize(cutout_keys, numSlices=min(100, len(cutout_keys)))
        
        results_rdd = keys_rdd.mapPartitions(
            lambda keys: process_partition(keys, input_bucket_bc.value)
        )
        
        # Collect results
        results = results_rdd.collect()
        
        # Compute statistics
        validated = [r for r in results if r["status"] == "validated"]
        identical = [r for r in validated if r.get("identical", False)]
        close_match = [r for r in validated 
                      if not r.get("identical", False) 
                      and r.get("max_diff", float("inf")) < 1e-4]
        issues = [r for r in results if r["status"] != "validated"]
        
        # Compute correlation stats
        correlations = {
            "g": [r.get("corr_g", 1.0) for r in validated if "corr_g" in r],
            "r": [r.get("corr_r", 1.0) for r in validated if "corr_r" in r],
            "z": [r.get("corr_z", 1.0) for r in validated if "corr_z" in r],
        }
        
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total cutouts:     {total_cutouts}")
        logger.info(f"Validated:         {len(validated)}")
        logger.info(f"  - Identical:     {len(identical)}")
        logger.info(f"  - Close match:   {len(close_match)}")
        logger.info(f"Issues:            {len(issues)}")
        
        for band in ["g", "r", "z"]:
            if correlations[band]:
                mean_corr = np.mean(correlations[band])
                min_corr = np.min(correlations[band])
                logger.info(f"  Correlation ({band}): mean={mean_corr:.6f}, min={min_corr:.6f}")
        
        # Save results
        summary = {
            "input_path": args.input,
            "total_cutouts": total_cutouts,
            "sample_fraction": args.sample_fraction,
            "cutouts_validated": len(validated),
            "identical": len(identical),
            "close_match": len(close_match),
            "issues": len(issues),
            "pass_rate": len(identical) / len(validated) if validated else 0,
            "close_match_rate": (len(identical) + len(close_match)) / len(validated) if validated else 0,
            "correlation_stats": {
                band: {
                    "mean": float(np.mean(v)) if v else None,
                    "min": float(np.min(v)) if v else None,
                    "std": float(np.std(v)) if v else None,
                }
                for band, v in correlations.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.time() - start_time,
            "pipeline_version": PIPELINE_VERSION,
        }
        
        # Quality gate
        min_correlation_threshold = 0.999
        all_correlations_ok = all(
            np.min(v) >= min_correlation_threshold
            for v in correlations.values()
            if v
        )
        gate_passed = len(issues) == 0 and all_correlations_ok
        summary["gate_passed"] = gate_passed
        
        logger.info(f"\nGate passed: {gate_passed}")
        if not gate_passed:
            if len(issues) > 0:
                logger.warning(f"Issues found in {len(issues)} cutouts")
                for issue in issues[:10]:  # Show first 10
                    logger.warning(f"  {issue['galaxy_id']}: {issue.get('error', 'unknown')}")
            if not all_correlations_ok:
                logger.warning(f"Some correlations below threshold {min_correlation_threshold}")
        
        # Save summary
        summary_key = f"{output_prefix}/summary.json"
        s3.put_object(
            Bucket=output_bucket,
            Key=summary_key,
            Body=json.dumps(summary, indent=2)
        )
        logger.info(f"Summary saved to s3://{output_bucket}/{summary_key}")
        
        # Save detailed results as parquet
        if results:
            # Create DataFrame from results
            results_df = spark.createDataFrame(results)
            results_df.coalesce(1).write.mode("overwrite").parquet(
                f"s3://{output_bucket}/{output_prefix}/details/"
            )
            logger.info(f"Details saved to s3://{output_bucket}/{output_prefix}/details/")
        
        # Save issues separately for easy inspection
        if issues:
            issues_key = f"{output_prefix}/issues.json"
            s3.put_object(
                Bucket=output_bucket,
                Key=issues_key,
                Body=json.dumps(issues, indent=2)
            )
            logger.info(f"Issues saved to s3://{output_bucket}/{issues_key}")
        
        elapsed = time.time() - start_time
        logger.info(f"\nCompleted in {elapsed:.1f}s")
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

#!/usr/bin/env python3
"""
Spark Job: Generate Cutouts from Legacy Survey

Downloads 101x101 pixel cutouts (g, r, z bands) from the Legacy Survey cutout service.
Works for both positive and negative samples.

Usage:
    spark-submit --deploy-mode cluster spark_generate_cutouts.py \
        --input s3://darkhaloscope/stronglens_calibration/positives_with_dr10/... \
        --output s3://darkhaloscope/stronglens_calibration/cutouts/positives/ \
        --cutout-type positive

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
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

PIPELINE_VERSION = "1.0.0"

# Cutout parameters
CUTOUT_SIZE = 101  # pixels
PIXEL_SCALE = 0.262  # arcsec/pixel
CUTOUT_ARCSEC = CUTOUT_SIZE * PIXEL_SCALE  # ~26.5 arcsec

# Legacy Survey cutout service
CUTOUT_URL_TEMPLATE = (
    "https://www.legacysurvey.org/viewer/fits-cutout"
    "?ra={ra}&dec={dec}&size={size}&layer=ls-dr10&pixscale={pixscale}&bands={bands}"
)

# Retry parameters
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CutoutGen] %(levelname)s: %(message)s",
)
logger = logging.getLogger("CutoutGen")


# =============================================================================
# CUTOUT DOWNLOAD
# =============================================================================

def download_cutout(
    ra: float,
    dec: float,
    size: int = CUTOUT_SIZE,
    bands: str = "grz",
    max_retries: int = MAX_RETRIES,
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Download cutout from Legacy Survey.
    
    Returns:
        Tuple of:
        - 3D numpy array of shape (size, size, 3) for g, r, z bands, or None if failed
        - Dict with provenance metadata (url, layer, etc.)
    
    FITS format: DR10 cutouts have bands stacked in primary HDU (HDU[0])
    with shape (n_bands, height, width). For "grz", shape is (3, size, size).
    """
    url = CUTOUT_URL_TEMPLATE.format(
        ra=ra,
        dec=dec,
        size=size,
        pixscale=PIXEL_SCALE,
        bands=bands,
    )
    
    provenance = {
        "cutout_url": url,
        "layer": "ls-dr10",
        "bands_requested": bands,
        "pixel_scale": PIXEL_SCALE,
        "size_requested": size,
    }
    
    for attempt in range(max_retries):
        try:
            with urlopen(url, timeout=30) as response:
                fits_data = response.read()
            
            # Parse FITS data
            from astropy.io import fits
            buffer = io.BytesIO(fits_data)
            with fits.open(buffer) as hdul:
                # DR10 cutouts: bands stacked in primary HDU[0]
                # Shape: (n_bands, height, width) = (3, size, size) for "grz"
                data = hdul[0].data
                
                if data is None:
                    provenance["error"] = "no_data_in_hdu0"
                    return None, provenance
                
                # Validate shape
                if data.ndim != 3:
                    provenance["error"] = f"wrong_ndim_{data.ndim}"
                    return None, provenance
                
                n_bands, h, w = data.shape
                if n_bands != 3:
                    provenance["error"] = f"wrong_n_bands_{n_bands}"
                    return None, provenance
                
                # Convert to float32 and transpose to (height, width, bands)
                # From (3, size, size) -> (size, size, 3)
                cutout = np.transpose(data, (1, 2, 0)).astype(np.float32)
                
                # Record actual dimensions
                provenance["actual_height"] = h
                provenance["actual_width"] = w
                provenance["actual_bands"] = n_bands
                
                return cutout, provenance
                    
        except (URLError, HTTPError, TimeoutError) as e:
            if attempt < max_retries - 1:
                delay = RETRY_DELAY_BASE * (2 ** attempt)
                time.sleep(delay)
            else:
                logger.warning(f"Failed to download cutout at ({ra}, {dec}): {e}")
                provenance["error"] = f"download_failed_{type(e).__name__}"
                return None, provenance
        except Exception as e:
            logger.warning(f"Error processing cutout at ({ra}, {dec}): {e}")
            provenance["error"] = f"parse_error_{type(e).__name__}"
            return None, provenance
    
    provenance["error"] = "max_retries_exceeded"
    return None, provenance


def compute_cutout_quality(cutout: np.ndarray) -> Dict:
    """
    Compute quality metrics for a cutout.
    
    Per LLM Section E recommendations:
    - Core brightness (shortcut detection)
    - Outer brightness (background)
    - Annulus brightness (where arcs appear)
    - MAD for noise estimation
    - Radial gradient
    
    Returns dict with quality metrics matching schema from LLM line 751.
    """
    h, w, c = cutout.shape
    center = h // 2
    
    # NaN analysis
    nan_mask = np.isnan(cutout)
    nan_count_per_band = [int(np.sum(nan_mask[:, :, i])) for i in range(c)]
    total_nan = int(np.sum(nan_mask))
    nan_frac = total_nan / (h * w * c)
    
    # Central region (inner 50x50 per LLM)
    margin = 25
    central = cutout[center-margin:center+margin, center-margin:center+margin, :]
    central_nan_frac = np.sum(np.isnan(central)) / central.size
    
    # Bands present
    bands_present = [not np.all(np.isnan(cutout[:, :, i])) for i in range(c)]
    
    # Create distance map from center for radial analysis
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - center)**2 + (x - center)**2)
    
    # Core (8 pixel radius per LLM)
    core_radius = 8
    core_mask = dist <= core_radius
    
    # Annulus (20-40 pixels where lensing arcs appear)
    annulus_inner = 20
    annulus_outer = 40
    annulus_mask = (dist >= annulus_inner) & (dist <= annulus_outer)
    
    # Outer region (edge pixels)
    outer_margin = h // 2 - margin
    outer_mask = dist >= outer_margin
    
    # r-band analysis (index 1)
    r_band = cutout[:, :, 1]
    valid_r = r_band[~np.isnan(r_band)]
    
    if len(valid_r) > 0:
        # Core stats
        core_pixels = r_band[core_mask]
        valid_core = core_pixels[~np.isnan(core_pixels)]
        core_brightness = float(np.median(valid_core)) if len(valid_core) > 0 else None
        core_max = float(np.max(valid_core)) if len(valid_core) > 0 else None
        
        # Annulus stats (critical for arc detection)
        annulus_pixels = r_band[annulus_mask]
        valid_annulus = annulus_pixels[~np.isnan(annulus_pixels)]
        annulus_brightness = float(np.median(valid_annulus)) if len(valid_annulus) > 0 else None
        annulus_std = float(np.std(valid_annulus)) if len(valid_annulus) > 0 else None
        
        # Outer (background) stats
        outer_pixels = r_band[outer_mask]
        valid_outer = outer_pixels[~np.isnan(outer_pixels)]
        outer_brightness = float(np.median(valid_outer)) if len(valid_outer) > 0 else None
        
        # Noise estimation (MAD is robust to outliers)
        mad_r = float(np.median(np.abs(valid_r - np.median(valid_r))))
        std_r = float(np.std(valid_r))
        
        # Radial gradient (shortcut detection per LLM)
        if core_brightness is not None and outer_brightness is not None:
            radial_gradient = core_brightness - outer_brightness
        else:
            radial_gradient = None
        
        # Normalization stats (LLM line 752)
        median_r = float(np.median(valid_r))
        mean_r = float(np.mean(valid_r))
        percentile_1 = float(np.percentile(valid_r, 1))
        percentile_99 = float(np.percentile(valid_r, 99))
        
        # Clip fraction (how much data is at extremes)
        clip_frac = float(np.sum((valid_r < percentile_1) | (valid_r > percentile_99)) / len(valid_r))
    else:
        core_brightness = core_max = None
        annulus_brightness = annulus_std = None
        outer_brightness = None
        mad_r = std_r = None
        radial_gradient = None
        median_r = mean_r = None
        percentile_1 = percentile_99 = None
        clip_frac = None
    
    # Quality gate: central region must be mostly valid
    quality_ok = (
        central_nan_frac < 0.02 and  # <2% NaN in center
        all(bands_present) and  # All bands present
        h == CUTOUT_SIZE and w == CUTOUT_SIZE  # Correct size
    )
    
    return {
        # NaN stats
        "nan_count_g": nan_count_per_band[0],
        "nan_count_r": nan_count_per_band[1],
        "nan_count_z": nan_count_per_band[2],
        "nan_frac": float(nan_frac),
        "central_nan_frac": float(central_nan_frac),
        # Band presence
        "has_g": bands_present[0],
        "has_r": bands_present[1],
        "has_z": bands_present[2],
        # Brightness features (shortcut detection)
        "core_brightness_r": core_brightness,
        "core_max_r": core_max,
        "annulus_brightness_r": annulus_brightness,
        "annulus_std_r": annulus_std,
        "outer_brightness_r": outer_brightness,
        "radial_gradient_r": radial_gradient,
        # Noise stats
        "mad_r": mad_r,
        "std_r": std_r,
        # Normalization stats
        "median_r": median_r,
        "mean_r": mean_r,
        "percentile_1_r": percentile_1,
        "percentile_99_r": percentile_99,
        "clip_frac_r": clip_frac,
        # Quality gate
        "quality_ok": quality_ok,
    }


# =============================================================================
# CHECKPOINT FUNCTIONS
# =============================================================================

def check_cutout_exists(s3_client, bucket: str, key: str) -> bool:
    """Check if cutout already exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False


def save_cutout_to_s3(
    s3_client,
    bucket: str,
    key: str,
    cutout: np.ndarray,
    metadata: Dict,
) -> bool:
    """Save cutout as compressed NPZ to S3."""
    try:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, cutout=cutout, **{f"meta_{k}": v for k, v in metadata.items()})
        buffer.seek(0)
        
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=buffer.getvalue(),
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to save cutout to {key}: {e}")
        return False


# =============================================================================
# SPARK PROCESSING
# =============================================================================

def process_batch(
    rows: List[Dict],
    output_bucket: str,
    output_prefix: str,
    cutout_type: str,
) -> Iterator[Dict]:
    """
    Process a batch of galaxies to generate cutouts.
    
    Yields result dictionaries with status and quality metrics.
    Implements resume support via S3 existence check.
    """
    import boto3
    
    s3 = boto3.client("s3", region_name="us-east-2")
    
    for row in rows:
        galaxy_id = row.get("galaxy_id") or row.get("pos_name", "unknown")
        ra = row.get("ra") or row.get("match_ra") or row.get("pos_catalog_ra")
        dec = row.get("dec") or row.get("match_dec") or row.get("pos_catalog_dec")
        
        if ra is None or dec is None:
            yield {
                "galaxy_id": galaxy_id,
                "status": "failed",
                "error": "missing_coordinates",
            }
            continue
        
        # Output key
        output_key = f"{output_prefix}/{galaxy_id}.npz"
        
        # Check if already exists (resume support)
        if check_cutout_exists(s3, output_bucket, output_key):
            yield {
                "galaxy_id": galaxy_id,
                "status": "skipped",
                "error": None,
            }
            continue
        
        # Download cutout (now returns tuple with provenance)
        cutout, provenance = download_cutout(ra, dec)
        
        if cutout is None:
            yield {
                "galaxy_id": galaxy_id,
                "status": "failed",
                "error": provenance.get("error", "download_failed"),
            }
            continue
        
        # Check size
        if cutout.shape != (CUTOUT_SIZE, CUTOUT_SIZE, 3):
            yield {
                "galaxy_id": galaxy_id,
                "status": "failed",
                "error": f"wrong_size_{cutout.shape}",
            }
            continue
        
        # Compute quality metrics
        quality = compute_cutout_quality(cutout)
        
        # Build metadata (per LLM recommendations, include provenance)
        metadata = {
            "galaxy_id": galaxy_id,
            "ra": float(ra),
            "dec": float(dec),
            "cutout_type": cutout_type,
            "size": CUTOUT_SIZE,
            "pixel_scale": PIXEL_SCALE,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": PIPELINE_VERSION,
        }
        
        # Add provenance metadata (LLM line 946-948)
        metadata.update({
            "cutout_url": provenance.get("cutout_url"),
            "layer": provenance.get("layer"),
            "bands_requested": provenance.get("bands_requested"),
        })
        
        # Add source metadata from input row
        for key in ["tier", "weight", "split", "pool", "confuser_category", 
                    "nobs_z", "nobs_z_bin", "type", "type_bin", "match_type", 
                    "brickname", "healpix_128", "psfsize_r", "psfdepth_r",
                    "galdepth_r", "ebv"]:
            if key in row and row[key] is not None:
                metadata[key] = row[key]
        
        # Merge quality metrics
        metadata.update(quality)
        
        # Save to S3
        success = save_cutout_to_s3(s3, output_bucket, output_key, cutout, metadata)
        
        if success:
            yield {
                "galaxy_id": galaxy_id,
                "status": "success",
                "error": None,
                "quality_ok": quality.get("quality_ok", False),
                **{k: v for k, v in quality.items() if k != "quality_ok"},
            }
        else:
            yield {
                "galaxy_id": galaxy_id,
                "status": "failed",
                "error": "save_failed",
            }


def main():
    parser = argparse.ArgumentParser(description="Generate cutouts from Legacy Survey")
    parser.add_argument("--input", required=True, help="S3 path to input parquet")
    parser.add_argument("--output", required=True, help="S3 output path for cutouts")
    parser.add_argument("--cutout-type", required=True, choices=["positive", "negative"],
                       help="Type of cutouts being generated")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size per partition")
    
    args = parser.parse_args()
    
    # Initialize Spark
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        StructType, StructField, StringType, FloatType, BooleanType, IntegerType
    )
    import pyspark.sql.functions as F
    
    spark = SparkSession.builder \
        .appName(f"GenerateCutouts-{args.cutout_type}") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    sc = spark.sparkContext
    
    logger.info("=" * 60)
    logger.info(f"Starting Cutout Generation ({args.cutout_type})")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    
    start_time = time.time()
    
    try:
        # Parse S3 paths
        output_bucket = args.output.replace("s3://", "").split("/")[0]
        output_prefix = "/".join(args.output.replace("s3://", "").split("/")[1:]).rstrip("/")
        
        # Broadcast parameters
        output_bucket_bc = sc.broadcast(output_bucket)
        output_prefix_bc = sc.broadcast(output_prefix)
        cutout_type_bc = sc.broadcast(args.cutout_type)
        
        # Load input
        logger.info("Loading input data...")
        input_df = spark.read.parquet(args.input)
        total_count = input_df.count()
        logger.info(f"Total inputs: {total_count}")
        
        # Convert to RDD of dictionaries
        rows_rdd = input_df.rdd.map(lambda row: row.asDict())
        
        # Process in batches
        def process_partition(iterator):
            batch = []
            for row in iterator:
                batch.append(row)
                if len(batch) >= 100:
                    yield from process_batch(
                        batch,
                        output_bucket_bc.value,
                        output_prefix_bc.value,
                        cutout_type_bc.value,
                    )
                    batch = []
            if batch:
                yield from process_batch(
                    batch,
                    output_bucket_bc.value,
                    output_prefix_bc.value,
                    cutout_type_bc.value,
                )
        
        results_rdd = rows_rdd.mapPartitions(process_partition)
        
        # Collect results
        results = results_rdd.collect()
        
        # Compute statistics
        success_count = sum(1 for r in results if r["status"] == "success")
        skipped_count = sum(1 for r in results if r["status"] == "skipped")
        failed_count = sum(1 for r in results if r["status"] == "failed")
        
        logger.info(f"Results: {success_count} success, {skipped_count} skipped, {failed_count} failed")
        
        # Save summary
        import boto3
        s3 = boto3.client("s3", region_name="us-east-2")
        
        summary = {
            "cutout_type": args.cutout_type,
            "total_inputs": total_count,
            "success_count": success_count,
            "skipped_count": skipped_count,
            "failed_count": failed_count,
            "success_rate": success_count / total_count if total_count > 0 else 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": PIPELINE_VERSION,
            "elapsed_seconds": time.time() - start_time,
        }
        
        # Quality stats for successful cutouts
        successful = [r for r in results if r["status"] == "success"]
        if successful:
            summary["quality_stats"] = {
                "mean_nan_frac": float(np.mean([r.get("nan_frac", 0) for r in successful])),
                "mean_central_nan_frac": float(np.mean([r.get("central_nan_frac", 0) for r in successful])),
                "all_bands_present": sum(1 for r in successful if r.get("has_g") and r.get("has_r") and r.get("has_z")),
            }
        
        summary_key = f"{output_prefix}/summary.json"
        s3.put_object(
            Bucket=output_bucket,
            Key=summary_key,
            Body=json.dumps(summary, indent=2)
        )
        logger.info(f"Summary saved to s3://{output_bucket}/{summary_key}")
        
        # Gate check
        if args.cutout_type == "positive":
            # All positives must have cutouts
            gate_passed = failed_count == 0
        else:
            # Allow 2% failure for negatives
            gate_passed = (success_count + skipped_count) / total_count >= 0.98
        
        logger.info(f"Gate passed: {gate_passed}")
        
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

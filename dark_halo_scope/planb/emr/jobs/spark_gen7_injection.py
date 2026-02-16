#!/usr/bin/env python3
"""
Phase 2: Gen7 Hybrid Source Injection via Spark

Generates training data with procedural hybrid sources (Sersic + clumps).

Input: Base stamps with ctrl (v5_cosmos_paired)
Output: New stamps with Gen7 injected sources

FEATURES:
- Fail-fast: Immediately abort on first error (default ON)
- Checkpointing: Skip already-processed rows on retry
- Chunked flushing: Avoid memory buildup by flushing every N rows

Usage:
    spark-submit --deploy-mode cluster spark_gen7_injection.py \
        --input s3://darkhaloscope/v5_cosmos_paired/train \
        --output s3://darkhaloscope/planb/gen7/train \
        --done s3://darkhaloscope/planb/gen7/done \
        --flush-rows 256

    # Disable fail-fast (not recommended for production):
    --no-fail-fast

Preset: large (35 workers)
Expected runtime: 2-4 hours for full dataset
"""
import argparse
import io
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Dict, Iterator, List, Set, Tuple

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, BinaryType, FloatType, IntegerType
)

# Constants
STAMP_SIZE = 64
PIX_SCALE_ARCSEC = 0.262
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "n_clumps_range": [2, 8],
    "clump_flux_frac_range": [0.05, 0.20],
    "clump_sigma_range": [0.5, 2.0],
    "gradient_strength": 0.15,
    "re_range_arcsec": [0.1, 0.5],
    "n_sersic_range": [0.5, 2.0],
    "q_range": [0.3, 1.0],
}


def setup_logging(job_name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{job_name}] %(levelname)s: %(message)s",
    )
    return logging.getLogger(job_name)


# =============================================================================
# GEN7 SOURCE GENERATION
# =============================================================================

def sersic_2d(
    size: int,
    re_pix: float,
    n: float,
    q: float,
    pa: float,
    center: Tuple[float, float] = None,
) -> np.ndarray:
    """Generate 2D Sersic profile."""
    if center is None:
        center = (size / 2, size / 2)
    
    y, x = np.mgrid[:size, :size]
    x = x - center[1]
    y = y - center[0]
    
    # Rotate and scale for ellipticity
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    x_rot = x * cos_pa + y * sin_pa
    y_rot = -x * sin_pa + y * cos_pa
    
    r = np.sqrt(x_rot**2 + (y_rot / q)**2)
    
    bn = 1.9992 * n - 0.3271  # Approximation
    intensity = np.exp(-bn * ((r / re_pix)**(1/n) - 1))
    
    return intensity.astype(np.float32)


def generate_hybrid_source(
    seed: int,
    config: Dict,
    size: int = STAMP_SIZE,
) -> Tuple[np.ndarray, Dict]:
    """
    Generate Gen7 hybrid source (Sersic + clumps + gradient).
    
    Returns:
        (source_image, metadata)
    """
    rng = np.random.default_rng(seed)
    
    # Sample parameters
    re_arcsec = rng.uniform(*config["re_range_arcsec"])
    re_pix = re_arcsec / PIX_SCALE_ARCSEC
    n_sersic = rng.uniform(*config["n_sersic_range"])
    q = rng.uniform(*config["q_range"])
    pa = rng.uniform(0, np.pi)
    
    # Generate base Sersic
    base = sersic_2d(size, re_pix, n_sersic, q, pa)
    base = base / base.sum()  # Normalize to unit flux
    
    # Add clumps
    n_clumps = rng.integers(*config["n_clumps_range"])
    clump_flux_frac = rng.uniform(*config["clump_flux_frac_range"])
    
    clumps = np.zeros((size, size), dtype=np.float32)
    base_flux = base.sum()
    total_clump_flux = clump_flux_frac * base_flux
    flux_per_clump = total_clump_flux / max(n_clumps, 1)
    
    y_grid, x_grid = np.mgrid[:size, :size]
    
    for _ in range(n_clumps):
        # Position within source region
        sigma_clump = rng.uniform(*config["clump_sigma_range"])
        
        # Sample position from base profile
        cx = size/2 + rng.normal(0, re_pix * 1.5)
        cy = size/2 + rng.normal(0, re_pix * 1.5)
        
        # Clamp to image
        cx = np.clip(cx, 5, size - 5)
        cy = np.clip(cy, 5, size - 5)
        
        # Add Gaussian clump
        gauss = np.exp(-0.5 * ((x_grid - cx)**2 + (y_grid - cy)**2) / sigma_clump**2)
        gauss = gauss / (2 * np.pi * sigma_clump**2)  # Normalize
        clumps += flux_per_clump * gauss
    
    # Combine
    result = base + clumps
    
    # Add color gradient (optional)
    if config.get("gradient_strength", 0) > 0:
        grad_angle = rng.uniform(0, 2 * np.pi)
        grad = (x_grid - size/2) * np.cos(grad_angle) + (y_grid - size/2) * np.sin(grad_angle)
        grad = grad / (size / 2)  # Normalize to [-1, 1]
        grad = 1 + config["gradient_strength"] * grad
        result = result * grad
    
    # Normalize
    result = result / result.sum()
    
    meta = {
        "re_pix": float(re_pix),
        "n_sersic": float(n_sersic),
        "q": float(q),
        "pa": float(pa),
        "n_clumps": int(n_clumps),
        "clump_flux_frac": float(clump_flux_frac),
    }
    
    return result, meta


# =============================================================================
# LENSING
# =============================================================================

def deflection_sie(x: np.ndarray, y: np.ndarray, theta_e: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """SIE deflection field."""
    r = np.sqrt(x**2 + y**2) + 1e-10
    
    if q < 0.99:
        alpha = theta_e * np.sqrt(q) / np.sqrt(1 - q**2)
        alpha_x = alpha * np.arctan(np.sqrt(1 - q**2) * x / (r + 1e-10))
        alpha_y = alpha * np.arctanh(np.sqrt(1 - q**2) * y / (r + 1e-10))
    else:
        alpha_x = theta_e * x / (r + 1e-10)
        alpha_y = theta_e * y / (r + 1e-10)
    
    return alpha_x, alpha_y


def ray_trace(
    source: np.ndarray,
    theta_e: float,
    q_lens: float = 0.8,
    source_offset: Tuple[float, float] = (0, 0),
) -> np.ndarray:
    """Ray-trace source through lens to create lensed image."""
    size = source.shape[0]
    
    # Image plane coordinates
    y_img, x_img = np.mgrid[:size, :size]
    x_img = (x_img - size/2).astype(np.float32)
    y_img = (y_img - size/2).astype(np.float32)
    
    # Deflection
    alpha_x, alpha_y = deflection_sie(x_img, y_img, theta_e, q_lens)
    
    # Source plane
    x_src = x_img - alpha_x + source_offset[0]
    y_src = y_img - alpha_y + source_offset[1]
    
    # Map back to pixel coords
    x_src_pix = x_src + size/2
    y_src_pix = y_src + size/2
    
    # Bilinear interpolation
    x0 = np.floor(x_src_pix).astype(int)
    y0 = np.floor(y_src_pix).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Weights
    wx = x_src_pix - x0
    wy = y_src_pix - y0
    
    # Clamp
    x0 = np.clip(x0, 0, size - 1)
    x1 = np.clip(x1, 0, size - 1)
    y0 = np.clip(y0, 0, size - 1)
    y1 = np.clip(y1, 0, size - 1)
    
    # Interpolate
    lensed = (
        (1 - wx) * (1 - wy) * source[y0, x0] +
        wx * (1 - wy) * source[y0, x1] +
        (1 - wx) * wy * source[y1, x0] +
        wx * wy * source[y1, x1]
    )
    
    return lensed.astype(np.float32)


# =============================================================================
# ENCODING UTILITIES
# =============================================================================

def decode_npz_blob(blob: bytes) -> np.ndarray:
    """Decode NPZ blob to (3, H, W) array."""
    z = np.load(io.BytesIO(blob), allow_pickle=False)
    return np.stack([z["image_g"], z["image_r"], z["image_z"]], axis=0)


def encode_npz_blob(stamp: np.ndarray) -> bytes:
    """Encode (3, H, W) array to NPZ blob."""
    out_buffer = io.BytesIO()
    np.savez_compressed(out_buffer,
                       image_g=stamp[0],
                       image_r=stamp[1],
                       image_z=stamp[2])
    return out_buffer.getvalue()


# =============================================================================
# CHECKPOINTING
# =============================================================================

def load_done_row_ids(spark: SparkSession, done_path: str, logger: logging.Logger) -> Set[str]:
    """Load set of already-processed row IDs from checkpoint manifest."""
    import boto3
    from botocore.exceptions import ClientError
    
    # Check if done_path exists
    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        # Parse S3 path
        if done_path.startswith("s3://"):
            parts = done_path[5:].split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
        else:
            logger.info("Done path is not S3, no checkpoint to load")
            return set()
        
        # Check if any files exist
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        if "Contents" not in response:
            logger.info(f"No checkpoint found at {done_path}")
            return set()
        
        # Load existing done manifest
        logger.info(f"Loading checkpoint from {done_path}...")
        done_df = spark.read.parquet(done_path)
        
        # Filter for OK status only
        done_ok = done_df.filter(F.col("status") == "OK")
        done_ids = set(row.row_id for row in done_ok.select("row_id").collect())
        logger.info(f"Loaded {len(done_ids):,} already-processed row IDs")
        return done_ids
        
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchBucket":
            logger.warning(f"Done bucket does not exist: {done_path}")
            return set()
        raise
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return set()


def write_done_records(
    spark: SparkSession,
    records: List[Dict],
    done_path: str,
    run_id: str,
    partition_id: int,
    logger: logging.Logger,
) -> None:
    """Write done records to checkpoint manifest."""
    if not records:
        return
    
    schema = StructType([
        StructField("row_id", StringType(), False),
        StructField("status", StringType(), False),
        StructField("run_id", StringType(), False),
        StructField("partition_id", IntegerType(), False),
        StructField("error", StringType(), True),
        StructField("timestamp", StringType(), False),
    ])
    
    timestamp = datetime.utcnow().isoformat()
    rows = [
        {
            "row_id": r["row_id"],
            "status": r["status"],
            "run_id": run_id,
            "partition_id": partition_id,
            "error": r.get("error"),
            "timestamp": timestamp,
        }
        for r in records
    ]
    
    df = spark.createDataFrame(rows, schema=schema)
    
    # Write with unique filename to avoid conflicts
    unique_suffix = f"{run_id}/part={partition_id}/done_{uuid.uuid4().hex[:8]}.parquet"
    output_path = f"{done_path.rstrip('/')}/{unique_suffix}"
    df.write.mode("append").parquet(output_path)


# =============================================================================
# PROCESSING
# =============================================================================

def process_row(row, config: Dict) -> Dict:
    """Process a single row: generate Gen7 source and inject."""
    try:
        # Decode control stamp
        ctrl = decode_npz_blob(row.ctrl_stamp_npz)
        
        # Get theta_e
        theta_e_arcsec = getattr(row, "theta_e_arcsec", 1.5)
        theta_e_pix = theta_e_arcsec / PIX_SCALE_ARCSEC
        
        # Generate Gen7 source
        seed = hash(row.task_id) % (2**31)
        source, source_meta = generate_hybrid_source(seed, config)
        
        # Lens the source
        lensed = ray_trace(source, theta_e_pix)
        
        # Scale to realistic flux
        src_flux = getattr(row, "src_total_flux_nmgy", 100.0)
        lensed = lensed * src_flux
        
        # Simple Gaussian PSF blur
        from scipy.ndimage import gaussian_filter
        psf_sigma = 1.5  # pixels
        lensed = gaussian_filter(lensed, psf_sigma)
        
        # Add to control (all bands)
        stamp = ctrl.copy()
        for i in range(3):
            stamp[i] = ctrl[i] + lensed
        
        # Encode output
        stamp_npz = encode_npz_blob(stamp)
        
        return {
            "task_id": row.task_id,
            "stamp_npz": stamp_npz,
            "ctrl_stamp_npz": row.ctrl_stamp_npz,
            "theta_e_arcsec": float(theta_e_arcsec),
            "gen7_re_pix": source_meta["re_pix"],
            "gen7_n_sersic": source_meta["n_sersic"],
            "gen7_n_clumps": source_meta["n_clumps"],
            "error": None,
        }
        
    except Exception as e:
        return {
            "task_id": getattr(row, "task_id", "unknown"),
            "stamp_npz": None,
            "ctrl_stamp_npz": None,
            "theta_e_arcsec": None,
            "gen7_re_pix": None,
            "gen7_n_sersic": None,
            "gen7_n_clumps": None,
            "error": str(e),
        }


def create_partition_processor(
    config: Dict,
    done_ids: Set[str],
    fail_fast: bool,
    flush_rows: int,
    output_path: str,
    done_path: str,
    run_id: str,
    spark_context,
):
    """Create a partition processor with checkpointing support."""
    
    def process_partition(partition_id: int, iterator: Iterator) -> Iterator:
        """Process partition with fail-fast and checkpointing."""
        import boto3
        from pyspark import TaskContext
        
        ctx = TaskContext.get()
        actual_partition_id = ctx.partitionId() if ctx else partition_id
        
        results_buffer = []
        done_buffer = []
        processed_count = 0
        skipped_count = 0
        
        for row in iterator:
            task_id = getattr(row, "task_id", None)
            
            # Skip if already processed
            if task_id and task_id in done_ids:
                skipped_count += 1
                continue
            
            # Process the row
            result = process_row(row, config)
            
            # Handle errors
            if result["error"] is not None:
                # Record error in done manifest
                done_buffer.append({
                    "row_id": task_id or "unknown",
                    "status": "ERROR",
                    "error": result["error"],
                })
                
                if fail_fast:
                    # Flush done buffer before failing
                    if done_buffer:
                        _flush_done_buffer(done_buffer, done_path, run_id, actual_partition_id)
                    
                    raise RuntimeError(
                        f"Gen7 injection failed for row_id={task_id}: {result['error']}\n"
                        f"FAIL-FAST enabled. Aborting. Processed {processed_count} rows, skipped {skipped_count}."
                    )
                continue
            
            # Success
            results_buffer.append(result)
            done_buffer.append({
                "row_id": task_id,
                "status": "OK",
                "error": None,
            })
            processed_count += 1
            
            # Flush buffers periodically
            if len(results_buffer) >= flush_rows:
                for r in results_buffer:
                    yield r
                results_buffer = []
                
                # Flush done records
                if done_buffer:
                    _flush_done_buffer(done_buffer, done_path, run_id, actual_partition_id)
                    done_buffer = []
        
        # Yield remaining results
        for r in results_buffer:
            yield r
        
        # Final done flush
        if done_buffer:
            _flush_done_buffer(done_buffer, done_path, run_id, actual_partition_id)
    
    return process_partition


def _flush_done_buffer(done_buffer: List[Dict], done_path: str, run_id: str, partition_id: int):
    """Flush done buffer to S3."""
    import boto3
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    if not done_buffer or not done_path:
        return
    
    # Parse S3 path
    if not done_path.startswith("s3://"):
        return
    
    parts = done_path[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    
    timestamp = datetime.utcnow().isoformat()
    
    # Build arrow table
    table = pa.table({
        "row_id": [r["row_id"] for r in done_buffer],
        "status": [r["status"] for r in done_buffer],
        "run_id": [run_id] * len(done_buffer),
        "partition_id": [partition_id] * len(done_buffer),
        "error": [r.get("error") for r in done_buffer],
        "timestamp": [timestamp] * len(done_buffer),
    })
    
    # Write to S3
    key = f"{prefix.rstrip('/')}/run_id={run_id}/part={partition_id}/done_{uuid.uuid4().hex[:8]}.parquet"
    
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    buffer.seek(0)
    
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--done", help="Checkpoint manifest path (S3)")
    parser.add_argument("--config", help="Config YAML path on S3")
    parser.add_argument("--limit", type=int, help="Limit rows for testing")
    parser.add_argument("--flush-rows", type=int, default=256, help="Flush interval (rows)")
    parser.add_argument("--run-id", help="Run ID for checkpointing")
    parser.add_argument("--no-fail-fast", action="store_true", help="Disable fail-fast (not recommended)")
    args = parser.parse_args()
    
    logger = setup_logging("gen7-injection")
    
    fail_fast = not args.no_fail_fast
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Fail-fast: {fail_fast}")
    logger.info(f"Flush interval: {args.flush_rows} rows")
    
    config = DEFAULT_CONFIG
    # TODO: Load config from S3 if provided
    
    spark = SparkSession.builder \
        .appName(f"gen7-injection-{run_id}") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        # Load checkpoint if exists
        done_ids = set()
        if args.done:
            done_ids = load_done_row_ids(spark, args.done, logger)
        
        logger.info(f"Reading from {args.input}")
        df = spark.read.parquet(args.input)
        
        if args.limit:
            df = df.limit(args.limit)
        
        input_count = df.count()
        logger.info(f"Input: {input_count:,} rows")
        logger.info(f"Already done: {len(done_ids):,} rows")
        logger.info(f"To process: ~{input_count - len(done_ids):,} rows")
        
        # Define output schema
        output_schema = StructType([
            StructField("task_id", StringType(), False),
            StructField("stamp_npz", BinaryType(), True),
            StructField("ctrl_stamp_npz", BinaryType(), True),
            StructField("theta_e_arcsec", FloatType(), True),
            StructField("gen7_re_pix", FloatType(), True),
            StructField("gen7_n_sersic", FloatType(), True),
            StructField("gen7_n_clumps", FloatType(), True),
            StructField("error", StringType(), True),
        ])
        
        # Create partition processor
        processor = create_partition_processor(
            config=config,
            done_ids=done_ids,
            fail_fast=fail_fast,
            flush_rows=args.flush_rows,
            output_path=args.output,
            done_path=args.done or "",
            run_id=run_id,
            spark_context=spark.sparkContext,
        )
        
        # Process
        start = time.time()
        result_rdd = df.rdd.mapPartitionsWithIndex(processor)
        result_df = spark.createDataFrame(result_rdd, schema=output_schema)
        
        # Filter out any errors (should be none if fail-fast is on)
        result_df = result_df.filter(F.col("error").isNull())
        
        # Append to output (don't overwrite, since we're checkpointing)
        unique_suffix = f"run_id={run_id}"
        output_with_run = f"{args.output.rstrip('/')}/{unique_suffix}"
        result_df.write.mode("append").parquet(output_with_run)
        
        elapsed = time.time() - start
        
        # Count outputs
        output_count = spark.read.parquet(args.output).count()
        
        logger.info("="*60)
        logger.info(f"Gen7 injection complete!")
        logger.info(f"  Run ID:  {run_id}")
        logger.info(f"  Input:   {input_count:,}")
        logger.info(f"  Skipped: {len(done_ids):,}")
        logger.info(f"  Output:  {output_count:,}")
        logger.info(f"  Time:    {elapsed/60:.1f} minutes")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Job failed: {e}")
        logger.error("Checkpoints saved. Retry will resume from last checkpoint.")
        raise
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 3: Gen8 Domain Randomization via Spark

Applies domain randomization artifacts to training data:
- PSF anisotropy
- Astrometric jitter
- Cosmic rays
- Saturation spikes
- Background residuals

FEATURES:
- Fail-fast: Immediately abort on first error (default ON)
- Checkpointing: Skip already-processed rows on retry
- Chunked flushing: Avoid memory buildup by flushing every N rows

Input: Base stamps with ctrl
Output: Stamps with Gen8 domain randomization

Usage:
    spark-submit --deploy-mode cluster spark_gen8_injection.py \
        --input s3://darkhaloscope/v5_cosmos_paired/train \
        --output s3://darkhaloscope/planb/gen8/train \
        --done s3://darkhaloscope/planb/gen8/done \
        --flush-rows 512

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
    StructType, StructField, StringType, BinaryType, FloatType, 
    BooleanType, IntegerType
)

# Constants
STAMP_SIZE = 64
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # From DR10 calibration
    "cosmic_rate": 0.12,
    "sat_rate": 0.06,
    
    # Astrometric jitter
    "jitter_sigma_pix": 0.25,
    
    # PSF anisotropy
    "psf_aniso_enabled": False,  # Already in injection
    
    # Background residuals
    "background_sigma": 0.02,
    "background_scale_pix": 10,
}


def setup_logging(job_name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{job_name}] %(levelname)s: %(message)s",
    )
    return logging.getLogger(job_name)


# =============================================================================
# ARTIFACT GENERATION
# =============================================================================

def add_cosmic_ray(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a cosmic ray artifact."""
    size = img.shape[0]
    
    # Random position
    x = rng.integers(0, size)
    y = rng.integers(0, size)
    
    # Random length and angle
    length = rng.integers(2, 8)
    angle = rng.uniform(0, 2 * np.pi)
    
    # Draw line
    for i in range(length):
        xi = int(x + i * np.cos(angle))
        yi = int(y + i * np.sin(angle))
        if 0 <= xi < size and 0 <= yi < size:
            # Very bright pixel
            img[yi, xi] = max(img[yi, xi], np.percentile(img, 99) * 10)
    
    return img


def add_saturation_spike(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add saturation diffraction spike."""
    size = img.shape[0]
    
    # Position near edge (simulating bright star off-frame)
    edge = rng.choice(["top", "bottom", "left", "right"])
    
    if edge == "top":
        x, y = rng.integers(10, size-10), 0
        dx, dy = 0, 1
    elif edge == "bottom":
        x, y = rng.integers(10, size-10), size-1
        dx, dy = 0, -1
    elif edge == "left":
        x, y = 0, rng.integers(10, size-10)
        dx, dy = 1, 0
    else:
        x, y = size-1, rng.integers(10, size-10)
        dx, dy = -1, 0
    
    # Draw spike
    length = rng.integers(10, 30)
    width = rng.integers(1, 3)
    amplitude = np.percentile(img, 95) * 5
    
    for i in range(length):
        for w in range(-width, width+1):
            xi = x + i * dx + w * (1 - abs(dx))
            yi = y + i * dy + w * (1 - abs(dy))
            if 0 <= xi < size and 0 <= yi < size:
                decay = 1.0 - i / length
                img[yi, xi] += amplitude * decay
    
    return img


def add_background_residual(
    img: np.ndarray, 
    rng: np.random.Generator,
    sigma: float = 0.02,
    scale: float = 10,
) -> np.ndarray:
    """Add correlated background residual."""
    size = img.shape[0]
    
    # Generate smooth random field
    small_size = max(size // int(scale), 2)
    small_noise = rng.normal(0, sigma, (small_size, small_size))
    
    # Upsample with interpolation
    from scipy.ndimage import zoom
    factor = size / small_size
    noise = zoom(small_noise, factor, order=1)
    
    # Crop to exact size
    noise = noise[:size, :size]
    
    # Scale by image std
    img_std = np.std(img)
    noise = noise * img_std
    
    return img + noise


def apply_jitter(img: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Apply sub-pixel astrometric jitter."""
    if sigma <= 0:
        return img
    
    dx = rng.normal(0, sigma)
    dy = rng.normal(0, sigma)
    
    # Sub-pixel shift via interpolation
    from scipy.ndimage import shift
    return shift(img, (dy, dx), order=1, mode='nearest')


def apply_domain_randomization(
    stamp: np.ndarray,
    ctrl: np.ndarray,
    seed: int,
    config: Dict,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Apply domain randomization to both stamp and ctrl.
    
    Returns:
        (augmented_stamp, augmented_ctrl, metadata)
    """
    rng = np.random.default_rng(seed)
    
    meta = {
        "has_cosmic": False,
        "has_spike": False,
        "jitter_dx": 0.0,
        "jitter_dy": 0.0,
    }
    
    # Apply same augmentations to both stamp and ctrl
    # (artifacts are in the base image, not the injection)
    
    for i in range(stamp.shape[0]):  # Per channel
        # Cosmic rays
        if rng.random() < config["cosmic_rate"]:
            stamp[i] = add_cosmic_ray(stamp[i].copy(), rng)
            ctrl[i] = add_cosmic_ray(ctrl[i].copy(), rng)
            meta["has_cosmic"] = True
        
        # Saturation spikes
        if rng.random() < config["sat_rate"]:
            stamp[i] = add_saturation_spike(stamp[i].copy(), rng)
            ctrl[i] = add_saturation_spike(ctrl[i].copy(), rng)
            meta["has_spike"] = True
        
        # Background residuals
        if config.get("background_sigma", 0) > 0:
            stamp[i] = add_background_residual(
                stamp[i], rng, 
                config["background_sigma"],
                config.get("background_scale_pix", 10)
            )
            ctrl[i] = add_background_residual(
                ctrl[i], rng,
                config["background_sigma"],
                config.get("background_scale_pix", 10)
            )
        
        # Astrometric jitter
        if config.get("jitter_sigma_pix", 0) > 0:
            stamp[i] = apply_jitter(stamp[i], rng, config["jitter_sigma_pix"])
            ctrl[i] = apply_jitter(ctrl[i], rng, config["jitter_sigma_pix"])
    
    return stamp, ctrl, meta


# =============================================================================
# ENCODING UTILITIES
# =============================================================================

def decode_npz_blob(blob: bytes) -> np.ndarray:
    """Decode NPZ blob to (3, H, W) array."""
    z = np.load(io.BytesIO(blob), allow_pickle=False)
    return np.stack([z["image_g"], z["image_r"], z["image_z"]], axis=0).astype(np.float32)


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


# =============================================================================
# PROCESSING
# =============================================================================

def process_row(row, config: Dict) -> Dict:
    """Process a single row: apply Gen8 domain randomization."""
    try:
        # Decode stamps
        stamp = decode_npz_blob(row.stamp_npz)
        ctrl = decode_npz_blob(row.ctrl_stamp_npz)
        
        # Apply domain randomization
        seed = hash(row.task_id) % (2**31)
        stamp_aug, ctrl_aug, meta = apply_domain_randomization(stamp, ctrl, seed, config)
        
        # Encode output
        stamp_npz = encode_npz_blob(stamp_aug)
        ctrl_npz = encode_npz_blob(ctrl_aug)
        
        return {
            "task_id": row.task_id,
            "stamp_npz": stamp_npz,
            "ctrl_stamp_npz": ctrl_npz,
            "theta_e_arcsec": getattr(row, "theta_e_arcsec", None),
            "arc_snr": getattr(row, "arc_snr", None),
            "gen8_has_cosmic": meta["has_cosmic"],
            "gen8_has_spike": meta["has_spike"],
            "error": None,
        }
        
    except Exception as e:
        return {
            "task_id": getattr(row, "task_id", "unknown"),
            "stamp_npz": None,
            "ctrl_stamp_npz": None,
            "theta_e_arcsec": None,
            "arc_snr": None,
            "gen8_has_cosmic": None,
            "gen8_has_spike": None,
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
):
    """Create a partition processor with checkpointing support."""
    
    def process_partition(partition_id: int, iterator: Iterator) -> Iterator:
        """Process partition with fail-fast and checkpointing."""
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
                        f"Gen8 domain randomization failed for row_id={task_id}: {result['error']}\n"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--done", help="Checkpoint manifest path (S3)")
    parser.add_argument("--cosmic-rate", type=float, default=0.12)
    parser.add_argument("--sat-rate", type=float, default=0.06)
    parser.add_argument("--jitter-sigma", type=float, default=0.25)
    parser.add_argument("--limit", type=int, help="Limit rows for testing")
    parser.add_argument("--flush-rows", type=int, default=512, help="Flush interval (rows)")
    parser.add_argument("--run-id", help="Run ID for checkpointing")
    parser.add_argument("--no-fail-fast", action="store_true", help="Disable fail-fast (not recommended)")
    args = parser.parse_args()
    
    logger = setup_logging("gen8-injection")
    
    fail_fast = not args.no_fail_fast
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Fail-fast: {fail_fast}")
    logger.info(f"Flush interval: {args.flush_rows} rows")
    
    config = DEFAULT_CONFIG.copy()
    config["cosmic_rate"] = args.cosmic_rate
    config["sat_rate"] = args.sat_rate
    config["jitter_sigma_pix"] = args.jitter_sigma
    
    logger.info(f"Config: cosmic_rate={config['cosmic_rate']}, sat_rate={config['sat_rate']}")
    
    spark = SparkSession.builder \
        .appName(f"gen8-injection-{run_id}") \
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
        
        # Output schema
        output_schema = StructType([
            StructField("task_id", StringType(), False),
            StructField("stamp_npz", BinaryType(), True),
            StructField("ctrl_stamp_npz", BinaryType(), True),
            StructField("theta_e_arcsec", FloatType(), True),
            StructField("arc_snr", FloatType(), True),
            StructField("gen8_has_cosmic", BooleanType(), True),
            StructField("gen8_has_spike", BooleanType(), True),
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
        logger.info(f"Gen8 domain randomization complete!")
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

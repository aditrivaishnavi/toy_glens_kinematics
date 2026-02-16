#!/usr/bin/env python3
"""
Phase 4: Score All Test Stamps via Spark

Scores all stamps for selection function analysis.
Uses a pre-trained model checkpoint.

Input: Test parquet + model checkpoint
Output: Parquet with scores

Usage:
    spark-submit --deploy-mode cluster spark_score_all.py \
        --input s3://darkhaloscope/v5_cosmos_paired/test \
        --checkpoint s3://darkhaloscope/planb/checkpoints/best_model.pt \
        --output s3://darkhaloscope/planb/scores/test

Preset: large (20 workers)
Expected runtime: 30-60 minutes
"""
import argparse
import gzip
import io
import logging
import sys
import time
from typing import Dict, Iterator

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType


def setup_logging(job_name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{job_name}] %(levelname)s: %(message)s",
    )
    return logging.getLogger(job_name)


# =============================================================================
# MODEL LOADING (on workers)
# =============================================================================

# Global variable for model (loaded once per worker)
_MODEL = None
_MODEL_PATH = None


def download_from_s3_with_retry(s3_uri: str, local_path: str, max_retries: int = 5) -> str:
    """
    Download file from S3 with retry and proper exception handling.
    
    Args:
        s3_uri: S3 URI (s3://bucket/key)
        local_path: Local path to save file
        max_retries: Maximum retry attempts
    
    Returns:
        Local path
    
    Raises:
        All errors are logged and re-raised (never silently eaten)
    """
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError, EndpointConnectionError
    
    # Configure retry
    config = Config(
        retries={"max_attempts": 3, "mode": "adaptive"},
        connect_timeout=10,
        read_timeout=60,
    )
    s3 = boto3.client("s3", config=config)
    
    bucket, key = s3_uri[5:].split("/", 1)
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {s3_uri} (attempt {attempt + 1}/{max_retries})")
            s3.download_file(bucket, key, local_path)
            logger.info(f"Successfully downloaded to {local_path}")
            return local_path
            
        except (EndpointConnectionError, ClientError) as e:
            error_str = str(e)
            
            # Check for credential errors - don't retry these
            if "ExpiredToken" in error_str or "InvalidClientTokenId" in error_str:
                logger.error(f"CREDENTIAL ERROR: {e}")
                logger.error("AWS credentials may have expired (valid for 24 hours)")
                raise RuntimeError(f"AWS credentials expired: {e}")
            
            # Retry for transient errors
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Download failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"Download failed after {max_retries} attempts: {e}")
                raise
                
        except Exception as e:
            # Don't silently eat - log and re-raise
            logger.error(f"Unexpected error downloading {s3_uri}: {type(e).__name__}: {e}")
            raise


def get_model(checkpoint_path: str):
    """
    Get or load model on worker.
    
    Model is loaded once per worker and cached.
    """
    global _MODEL, _MODEL_PATH
    
    if _MODEL is not None and _MODEL_PATH == checkpoint_path:
        return _MODEL
    
    import torch
    
    # For EMR, checkpoint is on S3 - download first
    if checkpoint_path.startswith("s3://"):
        local_path = f"/tmp/model_{hash(checkpoint_path)}.pt"
        download_from_s3_with_retry(checkpoint_path, local_path)
        checkpoint_path = local_path
    
    # Load model
    # This assumes a standard PyTorch checkpoint format
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Build model architecture (must match training)
    # For now, use a simple approach - in production, would load model config
    from torchvision.models import resnet18
    import torch.nn as nn
    
    model = resnet18()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 1)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    _MODEL = model
    _MODEL_PATH = checkpoint_path
    
    return model


# =============================================================================
# PREPROCESSING
# =============================================================================

def decode_and_normalize(blob: bytes) -> np.ndarray:
    """Decode NPZ and apply normalization."""
    z = np.load(io.BytesIO(blob), allow_pickle=False)
    img = np.stack([z["image_g"], z["image_r"], z["image_z"]], axis=0).astype(np.float32)
    
    # Robust normalization (same as training)
    c, h, w = img.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    outer_mask = r > 20
    
    normalized = np.zeros_like(img)
    for i in range(c):
        outer_values = img[i][outer_mask]
        median = np.median(outer_values)
        mad = np.median(np.abs(outer_values - median))
        std = 1.4826 * mad + 1e-10
        norm = (img[i] - median) / std
        norm = np.clip(norm, -5, 5)
        normalized[i] = norm
    
    return normalized


def score_row(row, checkpoint_path: str) -> Dict:
    """Score a single row."""
    try:
        import torch
        
        # Decode stamp
        stamp = decode_and_normalize(row.stamp_npz)
        
        # Get model
        model = get_model(checkpoint_path)
        
        # Score
        with torch.no_grad():
            x = torch.from_numpy(stamp).unsqueeze(0)  # (1, 3, 64, 64)
            logit = model(x)
            score = torch.sigmoid(logit).item()
        
        return {
            "task_id": row.task_id,
            "score": float(score),
            "theta_e_arcsec": getattr(row, "theta_e_arcsec", None),
            "arc_snr": getattr(row, "arc_snr", None),
            "psf_fwhm_arcsec": getattr(row, "psf_fwhm_arcsec", None),
            "error": None,
        }
        
    except Exception as e:
        return {
            "task_id": getattr(row, "task_id", "unknown"),
            "score": None,
            "theta_e_arcsec": None,
            "arc_snr": None,
            "psf_fwhm_arcsec": None,
            "error": str(e),
        }


def score_partition(checkpoint_path: str):
    """Return a partition scorer."""
    def _score(iterator: Iterator) -> Iterator:
        for row in iterator:
            yield score_row(row, checkpoint_path)
    return _score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path (S3 or local)")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--limit", type=int, help="Limit rows for testing")
    args = parser.parse_args()
    
    logger = setup_logging("score-all")
    
    spark = SparkSession.builder \
        .appName("score-all") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    # Broadcast checkpoint path
    checkpoint_path = args.checkpoint
    
    try:
        logger.info(f"Reading from {args.input}")
        df = spark.read.parquet(args.input)
        
        if args.limit:
            df = df.limit(args.limit)
        
        input_count = df.count()
        logger.info(f"Scoring {input_count:,} stamps...")
        
        # Output schema
        output_schema = StructType([
            StructField("task_id", StringType(), False),
            StructField("score", FloatType(), True),
            StructField("theta_e_arcsec", FloatType(), True),
            StructField("arc_snr", FloatType(), True),
            StructField("psf_fwhm_arcsec", FloatType(), True),
            StructField("error", StringType(), True),
        ])
        
        # Score
        start = time.time()
        result_rdd = df.rdd.mapPartitions(score_partition(checkpoint_path))
        result_df = spark.createDataFrame(result_rdd, schema=output_schema)
        
        # Keep all results (including errors for debugging)
        result_df.write.mode("overwrite").parquet(args.output)
        
        elapsed = time.time() - start
        
        # Stats
        result_df = spark.read.parquet(args.output)
        output_count = result_df.count()
        error_count = result_df.filter(F.col("error").isNotNull()).count()
        
        # Score distribution
        valid_scores = result_df.filter(F.col("score").isNotNull())
        score_stats = valid_scores.agg(
            F.avg("score").alias("mean"),
            F.stddev("score").alias("std"),
            F.min("score").alias("min"),
            F.max("score").alias("max"),
        ).collect()[0]
        
        logger.info("="*60)
        logger.info(f"Scoring complete!")
        logger.info(f"  Input:  {input_count:,}")
        logger.info(f"  Output: {output_count:,}")
        logger.info(f"  Errors: {error_count:,}")
        logger.info(f"  Time:   {elapsed/60:.1f} minutes")
        logger.info(f"\nScore distribution:")
        logger.info(f"  Mean: {score_stats['mean']:.4f}")
        logger.info(f"  Std:  {score_stats['std']:.4f}")
        logger.info(f"  Range: [{score_stats['min']:.4f}, {score_stats['max']:.4f}]")
        logger.info("="*60)
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

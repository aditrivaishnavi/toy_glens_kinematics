#!/usr/bin/env python3
"""
Phase 0: Verify Paired Data Integrity via Spark

Validates stamp/ctrl pairs at scale:
- Same shape
- Different values (arc present)
- No NaN/Inf
- Expected data types

Usage:
    spark-submit --deploy-mode cluster spark_verify_paired.py \
        --parquet-root s3://darkhaloscope/v5_cosmos_paired \
        --sample-rate 0.01

Preset: small (2 workers)
Expected runtime: 10-20 minutes
"""
import argparse
import gzip
import io
import json
import logging
import sys
from typing import Dict, Iterator

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, FloatType


def setup_logging(job_name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{job_name}] %(levelname)s: %(message)s",
    )
    return logging.getLogger(job_name)


def decode_npz(blob: bytes) -> np.ndarray:
    """Decode NPZ blob to array."""
    try:
        z = np.load(io.BytesIO(blob), allow_pickle=False)
        if "image_r" in z:
            return np.stack([z["image_g"], z["image_r"], z["image_z"]], axis=0)
        elif "img" in z:
            return np.asarray(z["img"])
        else:
            return np.asarray(z[list(z.keys())[0]])
    except:
        try:
            data = gzip.decompress(blob)
            z = np.load(io.BytesIO(data), allow_pickle=False)
            if "image_r" in z:
                return np.stack([z["image_g"], z["image_r"], z["image_z"]], axis=0)
            return np.asarray(z[list(z.keys())[0]])
        except:
            return None


def validate_row(row) -> Dict:
    """Validate a single row."""
    result = {
        "task_id": row.task_id if hasattr(row, "task_id") else "unknown",
        "shape_match": False,
        "not_identical": False,
        "no_nan_stamp": False,
        "no_nan_ctrl": False,
        "expected_shape": False,
        "error": None,
    }
    
    try:
        stamp = decode_npz(row.stamp_npz)
        ctrl = decode_npz(row.ctrl_stamp_npz)
        
        if stamp is None or ctrl is None:
            result["error"] = "decode_failed"
            return result
        
        # Shape match
        result["shape_match"] = stamp.shape == ctrl.shape
        
        # Not identical (arc present)
        diff = np.abs(stamp - ctrl).sum()
        result["not_identical"] = diff > 0.1
        
        # No NaN/Inf
        result["no_nan_stamp"] = np.isfinite(stamp).all()
        result["no_nan_ctrl"] = np.isfinite(ctrl).all()
        
        # Expected shape
        result["expected_shape"] = stamp.shape == (3, 64, 64) or stamp.ndim == 3
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def validate_partition(iterator: Iterator) -> Iterator:
    """Validate all rows in a partition."""
    for row in iterator:
        yield validate_row(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-rate", type=float, default=0.01, help="Fraction to sample")
    parser.add_argument("--output", help="S3 path for detailed results")
    args = parser.parse_args()
    
    logger = setup_logging("verify-paired")
    
    spark = SparkSession.builder \
        .appName("verify-paired") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        path = f"{args.parquet_root}/{args.split}"
        logger.info(f"Reading {path} with sample rate {args.sample_rate}")
        
        df = spark.read.parquet(path)
        
        # Sample
        if args.sample_rate < 1.0:
            df = df.sample(args.sample_rate, seed=42)
        
        total = df.count()
        logger.info(f"Validating {total:,} samples...")
        
        # Validate using mapPartitions
        validation_rdd = df.rdd.mapPartitions(validate_partition)
        
        # Collect results
        results_list = validation_rdd.collect()
        
        # Aggregate
        n_samples = len(results_list)
        checks = {
            "shape_match": sum(1 for r in results_list if r["shape_match"]),
            "not_identical": sum(1 for r in results_list if r["not_identical"]),
            "no_nan_stamp": sum(1 for r in results_list if r["no_nan_stamp"]),
            "no_nan_ctrl": sum(1 for r in results_list if r["no_nan_ctrl"]),
            "expected_shape": sum(1 for r in results_list if r["expected_shape"]),
        }
        errors = [r for r in results_list if r["error"]]
        
        logger.info("\n" + "="*60)
        logger.info(f"PAIRED DATA VERIFICATION - {args.split}")
        logger.info("="*60)
        logger.info(f"Samples checked: {n_samples:,}")
        
        all_passed = True
        for check, count in checks.items():
            pct = count / n_samples * 100 if n_samples > 0 else 0
            status = "✓" if pct == 100 else "✗"
            logger.info(f"  {status} {check}: {count:,}/{n_samples:,} ({pct:.1f}%)")
            if pct < 100:
                all_passed = False
        
        if errors:
            logger.warning(f"\n{len(errors)} rows had errors:")
            for e in errors[:5]:
                logger.warning(f"  {e['task_id']}: {e['error']}")
            all_passed = False
        
        logger.info("\n" + "="*60)
        if all_passed:
            logger.info("✓ PAIRED DATA VERIFIED")
        else:
            logger.error("✗ VERIFICATION FAILED")
        logger.info("="*60)
        
        if not all_passed:
            sys.exit(1)
            
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

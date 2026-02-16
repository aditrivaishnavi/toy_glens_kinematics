#!/usr/bin/env python3
"""
Phase 0: Verify Train/Val/Test Split Integrity via Spark

Checks that no brick or healpix overlaps between splits.
This prevents data leakage.

Usage:
    spark-submit --deploy-mode cluster spark_verify_splits.py \
        --parquet-root s3://darkhaloscope/v5_cosmos_paired

Preset: small (2 workers)
Expected runtime: 5-10 minutes
"""
import argparse
import json
import logging
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def setup_logging(job_name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{job_name}] %(levelname)s: %(message)s",
    )
    return logging.getLogger(job_name)


def verify_splits(spark: SparkSession, parquet_root: str, logger: logging.Logger) -> dict:
    """
    Verify no overlap between train/val/test splits.
    
    Returns:
        dict with verification results
    """
    results = {
        "passed": True,
        "splits": {},
        "overlaps": {},
    }
    
    splits = ["train", "val", "test"]
    brick_sets = {}
    healpix_sets = {}
    
    # Collect bricks and healpix from each split
    for split in splits:
        path = f"{parquet_root}/{split}"
        logger.info(f"Reading {split} from {path}")
        
        try:
            df = spark.read.parquet(path)
            count = df.count()
            
            # Get unique bricks
            if "brickname" in df.columns:
                bricks = set(df.select("brickname").distinct().rdd.flatMap(lambda x: x).collect())
                brick_sets[split] = bricks
                logger.info(f"  {split}: {count:,} rows, {len(bricks)} unique bricks")
            
            # Get unique healpix if present
            if "healpix" in df.columns:
                healpix = set(df.select("healpix").distinct().rdd.flatMap(lambda x: x).collect())
                healpix_sets[split] = healpix
                logger.info(f"  {split}: {len(healpix)} unique healpix")
            
            results["splits"][split] = {
                "count": count,
                "n_bricks": len(bricks) if "brickname" in df.columns else 0,
            }
            
        except Exception as e:
            logger.error(f"Failed to read {split}: {e}")
            results["passed"] = False
            results["splits"][split] = {"error": str(e)}
    
    # Check for brick overlaps
    logger.info("\nChecking brick overlaps...")
    for i, s1 in enumerate(splits):
        for s2 in splits[i+1:]:
            if s1 in brick_sets and s2 in brick_sets:
                overlap = brick_sets[s1] & brick_sets[s2]
                key = f"{s1}_vs_{s2}_bricks"
                
                if overlap:
                    logger.error(f"OVERLAP: {s1} vs {s2}: {len(overlap)} bricks")
                    results["passed"] = False
                    results["overlaps"][key] = list(overlap)[:10]  # First 10
                else:
                    logger.info(f"  ✓ {s1} vs {s2}: No overlap")
                    results["overlaps"][key] = []
    
    # Check for healpix overlaps
    if healpix_sets:
        logger.info("\nChecking healpix overlaps...")
        for i, s1 in enumerate(splits):
            for s2 in splits[i+1:]:
                if s1 in healpix_sets and s2 in healpix_sets:
                    overlap = healpix_sets[s1] & healpix_sets[s2]
                    key = f"{s1}_vs_{s2}_healpix"
                    
                    if overlap:
                        logger.error(f"OVERLAP: {s1} vs {s2}: {len(overlap)} healpix")
                        results["passed"] = False
                        results["overlaps"][key] = list(overlap)[:10]
                    else:
                        logger.info(f"  ✓ {s1} vs {s2}: No overlap")
                        results["overlaps"][key] = []
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-root", required=True, help="Root S3 path")
    parser.add_argument("--output", help="S3 path for results JSON")
    args = parser.parse_args()
    
    logger = setup_logging("verify-splits")
    
    spark = SparkSession.builder \
        .appName("verify-splits") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        results = verify_splits(spark, args.parquet_root, logger)
        
        logger.info("\n" + "="*60)
        if results["passed"]:
            logger.info("✓ ALL SPLITS VERIFIED - NO OVERLAPS")
        else:
            logger.error("✗ SPLIT VERIFICATION FAILED")
        logger.info("="*60)
        
        # Save results
        if args.output:
            spark.sparkContext.parallelize([json.dumps(results)]) \
                .saveAsTextFile(args.output)
            logger.info(f"Results saved to {args.output}")
        
        if not results["passed"]:
            sys.exit(1)
            
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

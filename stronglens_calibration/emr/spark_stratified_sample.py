#!/usr/bin/env python3
"""
Spark Job: Stratified Sampling for Negative Pool

Samples ~510K negatives from the 114M pool following:
- 100:1 negative:positive ratio per (nobs_z_bin, type_bin) stratum
- 85:15 N1:N2 ratio within sampled negatives

Usage:
    spark-submit --deploy-mode cluster spark_stratified_sample.py \
        --negatives s3://darkhaloscope/stronglens_calibration/manifests/TIMESTAMP/ \
        --positives s3://darkhaloscope/stronglens_calibration/positives_with_dr10/TIMESTAMP/data/ \
        --output s3://darkhaloscope/stronglens_calibration/sampled_negatives/TIMESTAMP/

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
from typing import Dict, List, Optional, Tuple

# =============================================================================
# CONSTANTS
# =============================================================================

PIPELINE_VERSION = "1.1.0"

# AWS Configuration (environment override supported)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

# Sampling ratios (from LLM recommendations)
NEG_POS_RATIO = 100  # 100 negatives per positive
N1_RATIO = 0.85  # 85% from N1 pool
N2_RATIO = 0.15  # 15% from N2 pool

# Strata definitions
NOBS_Z_BINS = ["1-2", "3-5", "6-10", "11+"]
TYPE_BINS = ["SER", "DEV", "REX"]  # Paper IV parity (EXP excluded)

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [StratifiedSample] %(levelname)s: %(message)s",
)
logger = logging.getLogger("StratifiedSample")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stratified sampling for negatives")
    parser.add_argument("--negatives", required=True, help="S3 path to negative pool manifest")
    parser.add_argument("--positives", required=True, help="S3 path to positives with DR10")
    parser.add_argument("--output", required=True, help="S3 output path")
    parser.add_argument("--neg-pos-ratio", type=int, default=NEG_POS_RATIO,
                       help="Negative:positive ratio per stratum")
    parser.add_argument("--n1-ratio", type=float, default=N1_RATIO,
                       help="N1 ratio within negatives (0.85 = 85%)")
    parser.add_argument("--test-limit", type=int, default=0,
                       help="Limit negative pool to N rows for testing (0 = no limit)")
    parser.add_argument("--force", action="store_true",
                       help="Force reprocessing, ignore existing checkpoints/output")
    
    args = parser.parse_args()
    
    # Initialize Spark
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    from pyspark.sql.window import Window
    from pyspark import StorageLevel
    
    spark = SparkSession.builder \
        .appName("StratifiedNegativeSampling") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    
    # Set checkpoint directory for fault tolerance (S3-based)
    checkpoint_dir = args.output.rstrip("/") + "/checkpoints/"
    spark.sparkContext.setCheckpointDir(checkpoint_dir)
    
    logger.info("=" * 60)
    logger.info("Starting Stratified Sampling")
    logger.info("=" * 60)
    logger.info(f"Negatives: {args.negatives}")
    logger.info(f"Positives: {args.positives}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Neg:Pos ratio: {args.neg_pos_ratio}")
    logger.info(f"N1:N2 ratio: {args.n1_ratio}:{1-args.n1_ratio}")
    logger.info(f"Test limit: {args.test_limit if args.test_limit > 0 else 'None (full run)'}")
    logger.info(f"Force reprocess: {args.force}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    
    start_time = time.time()
    
    try:
        # ----------------------------------------------------------
        # Check for existing output (skip if exists and not --force)
        # ----------------------------------------------------------
        output_data_path = args.output.rstrip("/") + "/data/"
        if not args.force:
            try:
                import boto3  # Import inside function (lesson 1.1)
                s3_check = boto3.client("s3", region_name=AWS_REGION)
                output_bucket = args.output.replace("s3://", "").split("/")[0]
                output_prefix = "/".join(args.output.replace("s3://", "").split("/")[1:]).rstrip("/") + "/data/"
                resp = s3_check.list_objects_v2(Bucket=output_bucket, Prefix=output_prefix, MaxKeys=1)
                if resp.get("KeyCount", 0) > 0:
                    logger.info(f"Output already exists at {output_data_path}")
                    logger.info("Use --force to override. Exiting.")
                    sys.exit(0)
            except Exception as e:
                logger.warning(f"Could not check existing output (proceeding): {e}")
        
        # Load positives (use data/ subdirectory with parquet filter)
        logger.info("\nLoading positives...")
        positives_path = args.positives.rstrip('/') + "/data/*.parquet"
        logger.info(f"Positives path: {positives_path}")
        positives_df = spark.read.parquet(positives_path)
        
        # Map positive types to type_bin if needed
        if "type_bin" not in positives_df.columns and "match_type" in positives_df.columns:
            positives_df = positives_df.withColumn("type_bin", F.col("match_type"))
        
        # Compute nobs_z_bin from nobs_z if not present
        if "nobs_z_bin" not in positives_df.columns and "nobs_z" in positives_df.columns:
            positives_df = positives_df.withColumn(
                "nobs_z_bin",
                F.when(F.col("nobs_z") <= 2, "1-2")
                .when(F.col("nobs_z") <= 5, "3-5")
                .when(F.col("nobs_z") <= 10, "6-10")
                .otherwise("11+")
            )
        
        total_positives = positives_df.count()
        logger.info(f"Total positives: {total_positives}")
        
        # Count positives per stratum
        pos_stratum_counts = positives_df.groupBy("nobs_z_bin", "type_bin").count()
        pos_counts = {
            (row["nobs_z_bin"], row["type_bin"]): row["count"]
            for row in pos_stratum_counts.collect()
        }
        
        logger.info(f"Positive stratum counts: {len(pos_counts)} strata")
        for (nobs_bin, type_bin), count in sorted(pos_counts.items()):
            logger.info(f"  ({nobs_bin}, {type_bin}): {count}")
        
        # Load negatives (filter for parquet files to avoid any JSON metadata)
        logger.info("\nLoading negative pool...")
        negatives_path = args.negatives.rstrip('/') + "/*.parquet"
        negatives_df = spark.read.parquet(negatives_path)
        
        # ----------------------------------------------------------
        # Test-limit: restrict to N rows for small-scale testing
        # Note: .limit() still scans all data (lesson 4.2), but for
        # testing correctness it's acceptable. For speed tests, use
        # partition pruning instead.
        # ----------------------------------------------------------
        if args.test_limit > 0:
            logger.info(f"TEST MODE: Limiting negatives to {args.test_limit} rows")
            negatives_df = negatives_df.limit(args.test_limit)
        
        # ----------------------------------------------------------
        # Repartition by (nobs_z_bin, type_bin) to distribute work
        # evenly across all workers and avoid data skew during
        # per-stratum sampling. 200 partitions for 30 workers.
        # ----------------------------------------------------------
        logger.info("Repartitioning negatives by (nobs_z_bin, type_bin)...")
        negatives_df = negatives_df.repartition(200, "nobs_z_bin", "type_bin")
        
        total_negatives = negatives_df.count()
        logger.info(f"Total negatives in pool: {total_negatives}")
        
        # Check N1/N2 distribution
        pool_dist = negatives_df.groupBy("pool").count().collect()
        pool_counts = {row["pool"]: row["count"] for row in pool_dist}
        logger.info(f"Pool distribution: {pool_counts}")
        
        n1_available = pool_counts.get("N1", 0)
        n2_available = pool_counts.get("N2", 0)
        
        if n2_available == 0:
            logger.warning("N2 pool is empty! Will sample 100% from N1.")
        
        # Compute target counts per stratum
        target_counts = {}
        for (nobs_bin, type_bin), pos_count in pos_counts.items():
            neg_target = pos_count * args.neg_pos_ratio
            n1_target = int(neg_target * args.n1_ratio)
            n2_target = neg_target - n1_target
            target_counts[(nobs_bin, type_bin)] = {
                "total": neg_target,
                "n1": n1_target,
                "n2": n2_target,
            }
        
        total_target = sum(t["total"] for t in target_counts.values())
        logger.info(f"\nTarget samples: {total_target}")
        
        # =================================================================
        # DETERMINISTIC SAMPLING VIA HASH ORDERING (LLM recommendation)
        # 
        # LLM: ".sample() + .limit() is NOT guaranteed stable across 
        # partitioning changes. Use a stable per-row key with hash 
        # ordering + Window for determinism."
        # =================================================================
        
        SEED = 42
        
        # Add deterministic hash column for stable ordering
        negatives_df = negatives_df.withColumn(
            "_sample_hash",
            F.abs(F.hash(F.col("galaxy_id"), F.lit(SEED)))
        )
        
        # ----------------------------------------------------------
        # Persist to avoid recomputation during per-stratum sampling.
        # DISK_ONLY to avoid OOM on large pools (114M rows).
        # ----------------------------------------------------------
        logger.info("Persisting negatives DataFrame (DISK_ONLY)...")
        negatives_df = negatives_df.persist(StorageLevel.DISK_ONLY)
        # Force materialization so persist happens before the loop
        neg_partition_count = negatives_df.rdd.getNumPartitions()
        logger.info(f"Negatives persisted across {neg_partition_count} partitions")
        
        sampled_dfs = []
        actual_counts = {"n1": 0, "n2": 0, "total": 0}
        stratum_results = {}
        
        for (nobs_bin, type_bin), targets in target_counts.items():
            logger.info(f"\nSampling stratum ({nobs_bin}, {type_bin})...")
            
            # Filter to this stratum
            stratum_df = negatives_df.filter(
                (F.col("nobs_z_bin") == nobs_bin) & (F.col("type_bin") == type_bin)
            )
            
            stratum_n1 = stratum_df.filter(F.col("pool") == "N1")
            stratum_n2 = stratum_df.filter(F.col("pool") == "N2")
            
            n1_count = stratum_n1.count()
            n2_count = stratum_n2.count()
            
            # Deterministic N1 sampling using Window + row_number
            n1_target = targets["n1"]
            if n1_count > 0 and n1_target > 0:
                # Order by hash for deterministic selection
                w = Window.orderBy("_sample_hash")
                n1_ranked = stratum_n1.withColumn("_rank", F.row_number().over(w))
                n1_sampled = n1_ranked.filter(F.col("_rank") <= n1_target).drop("_rank")
            else:
                n1_sampled = spark.createDataFrame([], negatives_df.schema)
            
            n1_actual = n1_sampled.count()
            
            # Deterministic N2 sampling
            n2_target = targets["n2"]
            n2_actual = 0
            
            if n2_count > 0 and n2_target > 0:
                w = Window.orderBy("_sample_hash")
                n2_ranked = stratum_n2.withColumn("_rank", F.row_number().over(w))
                n2_sampled = n2_ranked.filter(F.col("_rank") <= n2_target).drop("_rank")
                n2_actual = n2_sampled.count()
            else:
                n2_sampled = spark.createDataFrame([], negatives_df.schema)
            
            # Backfill from N1 if N2 insufficient (deterministic)
            backfill = 0
            if n2_actual < n2_target:
                shortfall = n2_target - n2_actual
                remaining_n1 = n1_count - n1_actual
                if remaining_n1 > 0:
                    # Select next N1 galaxies after those already sampled
                    w = Window.orderBy("_sample_hash")
                    n1_ranked = stratum_n1.withColumn("_rank", F.row_number().over(w))
                    # Take ranks from n1_actual+1 to n1_actual+shortfall
                    backfill_df = n1_ranked.filter(
                        (F.col("_rank") > n1_actual) & 
                        (F.col("_rank") <= n1_actual + shortfall)
                    ).drop("_rank")
                    backfill = backfill_df.count()
                    if backfill > 0:
                        n1_sampled = n1_sampled.union(backfill_df)
                        n1_actual += backfill
            
            # Union samples (drop hash column before union to keep schema clean)
            n1_clean = n1_sampled.drop("_sample_hash")
            n2_clean = n2_sampled.drop("_sample_hash")
            stratum_sampled = n1_clean.union(n2_clean)
            sampled_dfs.append(stratum_sampled)
            
            stratum_total = n1_actual + n2_actual
            actual_counts["n1"] += n1_actual
            actual_counts["n2"] += n2_actual
            actual_counts["total"] += stratum_total
            
            stratum_results[(nobs_bin, type_bin)] = {
                "target": targets["total"],
                "actual": stratum_total,
                "n1": n1_actual,
                "n2": n2_actual,
                "backfill": backfill,
                "n1_available": n1_count,
                "n2_available": n2_count,
            }
            
            logger.info(f"  Target: {targets['total']}, Actual: {stratum_total} (N1: {n1_actual}/{n1_count}, N2: {n2_actual}/{n2_count}, backfill: {backfill})")
        
        # Union all samples
        logger.info("\nCombining samples...")
        if sampled_dfs:
            sampled_df = sampled_dfs[0]
            for df in sampled_dfs[1:]:
                sampled_df = sampled_df.union(df)
        else:
            logger.error("No samples collected!")
            sys.exit(1)
        
        # Unpersist the large negatives DataFrame now that sampling is done
        negatives_df.unpersist()
        logger.info("Unpersisted negatives DataFrame")
        
        # Add sampling metadata
        sampled_df = sampled_df.withColumn(
            "sampling_timestamp",
            F.lit(datetime.now(timezone.utc).isoformat())
        ).withColumn(
            "sampling_version",
            F.lit(PIPELINE_VERSION)
        )
        
        # Repartition output to a reasonable number of files
        # (~510K rows -> 20 partitions -> ~25K rows per file, ~reasonable file sizes)
        output_partitions = max(10, actual_counts["total"] // 25000)
        logger.info(f"Repartitioning output to {output_partitions} partitions...")
        sampled_df = sampled_df.repartition(output_partitions)
        
        # Save output
        logger.info(f"\nSaving {actual_counts['total']} sampled negatives...")
        sampled_df.write.mode("overwrite") \
            .option("compression", "gzip") \
            .parquet(args.output.rstrip("/") + "/data/")
        
        # Save summary (import boto3 inside function -- lesson 1.1)
        import boto3
        s3 = boto3.client("s3", region_name=AWS_REGION)
        
        output_bucket = args.output.replace("s3://", "").split("/")[0]
        output_prefix = "/".join(args.output.replace("s3://", "").split("/")[1:]).rstrip("/")
        
        summary = {
            "total_positives": total_positives,
            "total_negative_pool": total_negatives,
            "pool_distribution": pool_counts,
            "neg_pos_ratio": args.neg_pos_ratio,
            "n1_ratio_target": args.n1_ratio,
            "n2_ratio_target": 1 - args.n1_ratio,
            "total_sampled": actual_counts["total"],
            "n1_sampled": actual_counts["n1"],
            "n2_sampled": actual_counts["n2"],
            "actual_n1_ratio": actual_counts["n1"] / actual_counts["total"] if actual_counts["total"] > 0 else 0,
            "actual_n2_ratio": actual_counts["n2"] / actual_counts["total"] if actual_counts["total"] > 0 else 0,
            "stratum_results": {f"{k[0]}_{k[1]}": v for k, v in stratum_results.items()},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": PIPELINE_VERSION,
            "elapsed_seconds": time.time() - start_time,
            "test_limit": args.test_limit if args.test_limit > 0 else None,
            "force": args.force,
        }
        
        # Gate checks
        summary["gates"] = {
            "total_count_ok": actual_counts["total"] >= total_positives * args.neg_pos_ratio * 0.9,
            "n1_n2_ratio_ok": 0.80 <= summary["actual_n1_ratio"] <= 0.92,
            "all_strata_represented": len(stratum_results) == len(pos_counts),
        }
        summary["gates_passed"] = all(summary["gates"].values())
        
        s3.put_object(
            Bucket=output_bucket,
            Key=f"{output_prefix}/summary.json",
            Body=json.dumps(summary, indent=2)
        )
        
        logger.info(f"\nSummary:")
        logger.info(f"  Total sampled: {actual_counts['total']}")
        logger.info(f"  N1: {actual_counts['n1']} ({summary['actual_n1_ratio']*100:.1f}%)")
        logger.info(f"  N2: {actual_counts['n2']} ({summary['actual_n2_ratio']*100:.1f}%)")
        logger.info(f"  Gates passed: {summary['gates_passed']}")
        
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

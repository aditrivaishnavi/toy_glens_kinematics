#!/usr/bin/env python3
"""
Phase 4p5: Compaction with Split Relabeling

This script performs two operations in a single pass:
1. Compacts many small parquet files from Phase 4c into fewer large files
2. Relabels splits to increase training data proportion

Split Relabeling Logic:
- test (35%)  → test (unchanged, frozen holdout)
- val (39%)   → 75% becomes train, 25% stays val
- train (26%) → train (unchanged)

Result: ~70% train, ~10% val, ~20% test

The relabeling uses deterministic hash-based assignment on task_id
for reproducibility across runs.

Usage:
    # Local Spark
    python phase4p5_compact_with_relabel.py \
        --input-s3 s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_corrected/stamps/... \
        --output-s3 s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_corrected_relabeled/ \
        --partitions 200

    # EMR
    spark-submit --deploy-mode cluster ... phase4p5_compact_with_relabel.py ...

Author: DarkHaloScope Team
Date: 2026-02-04
"""

import argparse
import hashlib
import sys
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


# =============================================================================
# SPLIT RELABELING LOGIC
# =============================================================================

def hash_based_split(task_id: str, original_split: str, val_to_train_pct: int = 75) -> str:
    """
    Deterministically relabel splits using hash-based assignment.
    
    Args:
        task_id: Unique identifier for the sample (used for hashing)
        original_split: Original split label ('train', 'val', 'test')
        val_to_train_pct: Percentage of val samples to move to train (default: 75%)
    
    Returns:
        New split label
    
    Logic:
        - 'test' → 'test' (always preserved as frozen holdout)
        - 'train' → 'train' (always preserved)
        - 'val' → hash(task_id) % 100 < val_to_train_pct ? 'train' : 'val'
    """
    if original_split == "test":
        return "test"
    elif original_split == "train":
        return "train"
    elif original_split == "val":
        # Use BLAKE2b hash for deterministic, reproducible assignment
        # Hash the task_id and take modulo 100
        h = hashlib.blake2b(str(task_id).encode(), digest_size=8)
        hash_val = int.from_bytes(h.digest(), 'little') % 100
        if hash_val < val_to_train_pct:
            return "train"
        else:
            return "val"
    else:
        # Unknown split, keep as-is
        return original_split


# Register as Spark UDF
hash_based_split_udf = F.udf(hash_based_split, StringType())


# =============================================================================
# COMPACTION WITH RELABELING
# =============================================================================

def compact_with_relabel(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    num_partitions: int = 200,
    val_to_train_pct: int = 75,
    dry_run: bool = False
) -> dict:
    """
    Compact parquet files and relabel splits in a single pass.
    
    Args:
        spark: SparkSession
        input_path: S3 path to Phase 4c output (e.g., s3://bucket/phase4c/.../stamps/...)
        output_path: S3 path for compacted output
        num_partitions: Number of output partitions per split
        val_to_train_pct: Percentage of val to move to train
        dry_run: If True, only compute stats without writing
    
    Returns:
        Dictionary with statistics
    """
    print("=" * 60)
    print("PHASE 4p5: COMPACTION WITH SPLIT RELABELING")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Partitions: {num_partitions}")
    print(f"Val → Train %: {val_to_train_pct}%")
    print(f"Dry run: {dry_run}")
    print()
    
    # Read input data
    print("Reading input data...")
    df = spark.read.parquet(input_path)
    
    # Get original split column name
    # Could be 'region_split' (Hive partition) or 'split' column
    if "region_split" in df.columns:
        split_col = "region_split"
    elif "split" in df.columns:
        split_col = "split"
    else:
        raise ValueError("No split column found. Expected 'region_split' or 'split'")
    
    print(f"Using split column: {split_col}")
    
    # Ensure task_id exists for hashing
    if "task_id" not in df.columns:
        # Create a synthetic task_id from available columns
        print("WARNING: 'task_id' not found, creating from row hash")
        df = df.withColumn("task_id", F.monotonically_increasing_id().cast("string"))
    
    # Count original splits
    print("\n=== ORIGINAL SPLIT DISTRIBUTION ===")
    original_counts = df.groupBy(split_col).count().collect()
    total_original = sum(row["count"] for row in original_counts)
    stats = {"original": {}, "relabeled": {}}
    
    for row in original_counts:
        split_name = row[split_col]
        count = row["count"]
        pct = 100 * count / total_original
        stats["original"][split_name] = count
        print(f"  {split_name}: {count:,} ({pct:.1f}%)")
    
    # Apply relabeling
    print(f"\n=== APPLYING RELABELING (val → {val_to_train_pct}% train) ===")
    
    # Preserve original split as 'original_split'
    df = df.withColumn("original_split", F.col(split_col))
    
    # Apply hash-based relabeling
    df = df.withColumn(
        split_col,
        hash_based_split_udf(F.col("task_id"), F.col("original_split"), F.lit(val_to_train_pct))
    )
    
    # Count relabeled splits
    print("\n=== RELABELED SPLIT DISTRIBUTION ===")
    relabeled_counts = df.groupBy(split_col).count().collect()
    total_relabeled = sum(row["count"] for row in relabeled_counts)
    
    for row in relabeled_counts:
        split_name = row[split_col]
        count = row["count"]
        pct = 100 * count / total_relabeled
        stats["relabeled"][split_name] = count
        print(f"  {split_name}: {count:,} ({pct:.1f}%)")
    
    # Verify test set unchanged
    test_original = stats["original"].get("test", 0)
    test_relabeled = stats["relabeled"].get("test", 0)
    if test_original != test_relabeled:
        raise RuntimeError(f"TEST SET CHANGED! Original: {test_original}, Relabeled: {test_relabeled}")
    print(f"\n✓ Test set preserved: {test_relabeled:,} samples (unchanged)")
    
    # Calculate movement
    val_original = stats["original"].get("val", 0)
    val_relabeled = stats["relabeled"].get("val", 0)
    val_moved_to_train = val_original - val_relabeled
    print(f"✓ Moved {val_moved_to_train:,} samples from val → train")
    
    if dry_run:
        print("\n[DRY RUN] Skipping write")
        return stats
    
    # Write output with partitioning
    print(f"\n=== WRITING OUTPUT ===")
    print(f"Partitioning by: {split_col}")
    print(f"Repartitioning to {num_partitions} files per split...")
    
    # Repartition and write
    df.repartition(num_partitions, split_col) \
        .write \
        .partitionBy(split_col) \
        .mode("overwrite") \
        .parquet(output_path)
    
    print(f"\n✓ Output written to: {output_path}")
    
    # Write metadata/stats
    stats_path = output_path.rstrip("/") + "/_split_stats.json"
    import json
    stats_json = json.dumps(stats, indent=2)
    print(f"\nSplit statistics:\n{stats_json}")
    
    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4p5: Compaction with Split Relabeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--input-s3", required=True,
                        help="S3 path to Phase 4c output")
    parser.add_argument("--output-s3", required=True,
                        help="S3 path for compacted output")
    parser.add_argument("--partitions", type=int, default=200,
                        help="Number of output partitions per split (default: 200)")
    parser.add_argument("--val-to-train-pct", type=int, default=75,
                        help="Percentage of val to move to train (default: 75)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only compute stats, don't write output")
    
    args = parser.parse_args()
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("Phase4p5-Compact-Relabel") \
        .config("spark.sql.parquet.compression.codec", "zstd") \
        .config("spark.sql.shuffle.partitions", "400") \
        .getOrCreate()
    
    try:
        stats = compact_with_relabel(
            spark=spark,
            input_path=args.input_s3,
            output_path=args.output_s3,
            num_partitions=args.partitions,
            val_to_train_pct=args.val_to_train_pct,
            dry_run=args.dry_run
        )
        print("\n" + "=" * 60)
        print("COMPACTION WITH RELABELING COMPLETE")
        print("=" * 60)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

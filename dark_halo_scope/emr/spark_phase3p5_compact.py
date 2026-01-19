#!/usr/bin/env python3
"""
Phase 3.5: Compact the Phase 3c parent catalog.

Problem:
  Phase 3c outputs ~188k small Parquet files (partitioned by region_split/region_id).
  This causes severe performance issues for downstream pipelines.

Solution:
  Rewrite into a compact layout with:
  - Only ~96-192 files total
  - Partition by region_split only (train/val/test)
  - Recompute LRG flags from stored mags to eliminate floating-point precision artifacts
  - Filter to ensure parent condition (v3_color_relaxed) is enforced
  - Add gmag_valid column for downstream convenience

Usage:
  spark-submit spark_phase3p5_compact.py \
    --input-path s3://darkhaloscope/phase3_pipeline/phase3c/v3_color_relaxed/parent_union_parquet \
    --output-path s3://darkhaloscope/phase3_pipeline/phase3p5/v3_color_relaxed/parent_compact \
    --num-partitions 96
"""

import argparse
import sys
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType


# -----------------------------------------------------------------------------
# LRG Variant Definitions (must match Phase 2/3 exactly)
# -----------------------------------------------------------------------------
LRG_VARIANTS = {
    "v1_pure_massive":   {"z_max": 20.0, "rz_min": 0.5, "zw1_min": 1.6},
    "v2_baseline_dr10":  {"z_max": 20.4, "rz_min": 0.4, "zw1_min": 1.6},
    "v3_color_relaxed":  {"z_max": 20.4, "rz_min": 0.4, "zw1_min": 0.8},
    "v4_mag_relaxed":    {"z_max": 21.0, "rz_min": 0.4, "zw1_min": 0.8},
    "v5_very_relaxed":   {"z_max": 21.5, "rz_min": 0.3, "zw1_min": 0.8},
}


def compute_lrg_flag(zmag_col, rz_col, zw1_col, variant: dict):
    """Compute LRG flag expression for a variant."""
    return (
        (F.col(zmag_col) < F.lit(variant["z_max"])) &
        (F.col(rz_col) > F.lit(variant["rz_min"])) &
        (F.col(zw1_col) > F.lit(variant["zw1_min"])) &
        F.col(zmag_col).isNotNull() &
        ~F.isnan(F.col(zmag_col)) &
        F.col(rz_col).isNotNull() &
        ~F.isnan(F.col(rz_col)) &
        F.col(zw1_col).isNotNull() &
        ~F.isnan(F.col(zw1_col))
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 3.5: Compact parent catalog")
    parser.add_argument("--input-path", required=True,
                        help="S3 path to Phase 3c parent_union_parquet")
    parser.add_argument("--output-path", required=True,
                        help="S3 path for compacted output")
    parser.add_argument("--num-partitions", type=int, default=96,
                        help="Number of output partitions (default: 96)")
    parser.add_argument("--parent-variant", default="v3_color_relaxed",
                        help="Parent selection variant to enforce (default: v3_color_relaxed)")
    parser.add_argument("--shuffle-partitions", type=int, default=200,
                        help="Spark shuffle partitions (default: 200)")
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 3.5: COMPACT PARENT CATALOG")
    print("=" * 80)
    print(f"Input:  {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Partitions: {args.num_partitions}")
    print(f"Parent variant: {args.parent_variant}")
    print(f"Started: {datetime.utcnow().isoformat()}Z")
    print("=" * 80)

    # Create Spark session
    spark = (SparkSession.builder
             .appName("DHS Phase3.5 Compact")
             .config("spark.sql.shuffle.partitions", args.shuffle_partitions)
             .getOrCreate())

    spark.sparkContext.setLogLevel("WARN")

    # Read input
    print(f"\n[1/5] Reading input from {args.input_path}")
    df = spark.read.parquet(args.input_path)
    
    input_count = df.count()
    print(f"      Input rows: {input_count:,}")

    # Show input schema
    print("\n[2/5] Input schema:")
    df.printSchema()

    # Recompute LRG flags from stored magnitudes/colors
    # This eliminates floating-point precision artifacts at boundaries
    print("\n[3/5] Recomputing LRG flags from stored mags/colors...")
    
    for variant_name, cuts in LRG_VARIANTS.items():
        col_name = f"is_{variant_name}"
        df = df.withColumn(col_name, compute_lrg_flag("zmag", "rz", "zw1", cuts))
        print(f"      Recomputed: {col_name}")

    # Add gmag_valid column for downstream convenience
    # Note: In PySpark, isfinite() doesn't exist - use isNotNull & ~isnan instead
    # (NaN is the only non-finite value we'd see in magnitude data)
    df = df.withColumn("gmag_valid", 
                       F.col("gmag").isNotNull() & 
                       ~F.isnan(F.col("gmag")))
    print("      Added: gmag_valid")

    # Enforce parent condition (filter to parent variant)
    parent_col = f"is_{args.parent_variant}"
    print(f"\n[4/5] Enforcing parent condition: {parent_col} == True")
    
    before_filter = df.count()
    df = df.filter(F.col(parent_col) == F.lit(True))
    after_filter = df.count()
    
    dropped = before_filter - after_filter
    print(f"      Before filter: {before_filter:,}")
    print(f"      After filter:  {after_filter:,}")
    print(f"      Dropped (boundary artifacts): {dropped:,}")

    # Verify variant distribution
    print("\n      Variant distribution after filtering:")
    for variant_name in LRG_VARIANTS.keys():
        col_name = f"is_{variant_name}"
        count = df.filter(F.col(col_name) == F.lit(True)).count()
        pct = 100.0 * count / after_filter if after_filter > 0 else 0
        print(f"        {variant_name}: {count:,} ({pct:.2f}%)")

    # Verify split distribution
    print("\n      Split distribution:")
    split_counts = df.groupBy("region_split").count().collect()
    for row in split_counts:
        pct = 100.0 * row["count"] / after_filter if after_filter > 0 else 0
        print(f"        {row['region_split']}: {row['count']:,} ({pct:.2f}%)")

    # Drop region_id from partitioning (keep as regular column)
    # Select columns in a clean order
    columns_ordered = [
        # Identifiers
        "brickname", "objid", "region_id", "region_split",
        # Coordinates
        "ra", "dec",
        # Raw magnitudes
        "gmag", "rmag", "zmag", "w1mag",
        # Raw colors
        "rz", "zw1",
        # MW-corrected magnitudes
        "gmag_mw", "rmag_mw", "zmag_mw", "w1mag_mw",
        # MW-corrected colors
        "rz_mw", "zw1_mw",
        # Metadata
        "maskbits", "type",
        # LRG flags
        "is_v1_pure_massive", "is_v2_baseline_dr10", "is_v3_color_relaxed",
        "is_v4_mag_relaxed", "is_v5_very_relaxed",
        # Convenience flags
        "gmag_valid",
    ]
    
    # Only select columns that exist
    existing_cols = set(df.columns)
    columns_to_select = [c for c in columns_ordered if c in existing_cols]
    df = df.select(columns_to_select)

    # Compact write
    print(f"\n[5/5] Writing compacted output to {args.output_path}")
    print(f"      Repartitioning to {args.num_partitions} partitions...")
    print(f"      Partitioning by: region_split only")
    
    (df
     .repartition(args.num_partitions)
     .write
     .mode("overwrite")
     .partitionBy("region_split")
     .parquet(args.output_path))

    print(f"      Write complete!")

    # Write metadata
    metadata = {
        "job": "phase3p5_compact",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input_path": args.input_path,
        "output_path": args.output_path,
        "input_rows": input_count,
        "output_rows": after_filter,
        "dropped_rows": dropped,
        "num_partitions": args.num_partitions,
        "parent_variant": args.parent_variant,
        "variants": list(LRG_VARIANTS.keys()),
    }
    
    import json
    metadata_path = args.output_path.rstrip("/") + "/_metadata.json"
    spark.sparkContext.parallelize([json.dumps(metadata, indent=2)]).coalesce(1).saveAsTextFile(
        metadata_path.replace("/_metadata.json", "/_metadata_temp")
    )
    print(f"      Metadata written")

    # Summary
    print("\n" + "=" * 80)
    print("PHASE 3.5 COMPLETE")
    print("=" * 80)
    print(f"Input:  {input_count:,} rows in ~188k files")
    print(f"Output: {after_filter:,} rows in ~{args.num_partitions * 3} files (3 splits)")
    print(f"Dropped: {dropped:,} boundary artifacts")
    print(f"Finished: {datetime.utcnow().isoformat()}Z")
    print("=" * 80)

    spark.stop()


if __name__ == "__main__":
    main()


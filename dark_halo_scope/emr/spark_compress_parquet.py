#!/usr/bin/env python3
"""
One-time safe Parquet compression migration job.

This job re-compresses existing parquet files from snappy (default) to gzip,
which typically reduces storage by 40-50%.

SAFETY FEATURES:
1. Writes to a NEW path (_gzip suffix) - never overwrites originals
2. Verifies row counts match before/after
3. Supports dry-run mode to estimate savings
4. Provides detailed size comparison

Usage:
  # Dry run (estimate savings without writing)
  python spark_compress_parquet.py \
    --source s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/stamps/train_stamp64_bandsgrz_gridgrid_small \
    --dry-run

  # Actual compression
  python spark_compress_parquet.py \
    --source s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/stamps/train_stamp64_bandsgrz_gridgrid_small \
    --target s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/stamps_gzip/train_stamp64_bandsgrz_gridgrid_small

  # With partitioning preserved
  python spark_compress_parquet.py \
    --source s3://...stamps/train_stamp64.../region_split=train \
    --target s3://...stamps_gzip/train_stamp64.../region_split=train \
    --partition-by region_split
"""

import argparse
import time
import sys

from pyspark.sql import SparkSession


def get_s3_size(spark, path: str) -> int:
    """Get total size of files at S3 path in bytes."""
    try:
        sc = spark.sparkContext
        hadoop_conf = sc._jsc.hadoopConfiguration()
        
        # Get filesystem
        uri = sc._jvm.java.net.URI(path)
        fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(uri, hadoop_conf)
        
        # Get content summary
        hadoop_path = sc._jvm.org.apache.hadoop.fs.Path(path)
        if fs.exists(hadoop_path):
            summary = fs.getContentSummary(hadoop_path)
            return summary.getLength()
        return 0
    except Exception as e:
        print(f"[WARN] Could not get size for {path}: {e}")
        return 0


def format_size(bytes_val: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(bytes_val) < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def main():
    parser = argparse.ArgumentParser(description="Safe Parquet compression migration")
    parser.add_argument("--source", required=True, help="Source parquet path (S3 or local)")
    parser.add_argument("--target", default=None, help="Target parquet path (default: source + _gzip suffix)")
    parser.add_argument("--compression", default="gzip", choices=["gzip", "zstd", "snappy", "lz4"],
                        help="Compression codec (default: gzip)")
    parser.add_argument("--partition-by", default=None, help="Partition column (e.g., region_split)")
    parser.add_argument("--coalesce", type=int, default=0, 
                        help="Coalesce to N partitions before write (0=auto)")
    parser.add_argument("--dry-run", action="store_true", help="Only estimate savings, don't write")
    parser.add_argument("--repartition", type=int, default=0,
                        help="Repartition to N partitions (0=keep original)")
    
    args = parser.parse_args()
    
    # Auto-generate target path if not specified
    if args.target is None:
        args.target = args.source.rstrip("/") + f"_{args.compression}"
    
    print("=" * 60)
    print("Parquet Compression Migration")
    print("=" * 60)
    print(f"Source:      {args.source}")
    print(f"Target:      {args.target}")
    print(f"Compression: {args.compression}")
    print(f"Partition:   {args.partition_by or 'None'}")
    print(f"Dry run:     {args.dry_run}")
    print("=" * 60)
    
    # Safety check: prevent overwriting source
    if args.source.rstrip("/") == args.target.rstrip("/"):
        print("[ERROR] Source and target paths are the same! This would overwrite data.")
        print("        Use a different target path for safety.")
        sys.exit(1)
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("ParquetCompression") \
        .config("spark.sql.parquet.compression.codec", args.compression) \
        .getOrCreate()
    
    try:
        # Get source size
        print("\n[INFO] Measuring source size...")
        source_size = get_s3_size(spark, args.source)
        print(f"[INFO] Source size: {format_size(source_size)}")
        
        # Read source
        print("\n[INFO] Reading source parquet...")
        t0 = time.time()
        df = spark.read.parquet(args.source)
        
        source_count = df.count()
        print(f"[INFO] Source row count: {source_count:,}")
        print(f"[INFO] Read time: {time.time() - t0:.1f}s")
        
        if args.dry_run:
            print("\n[DRY RUN] Estimating compression savings...")
            # Sample a small portion to estimate compression ratio
            sample_size = min(10000, source_count)
            sample_df = df.limit(sample_size)
            
            # Write sample to temporary location
            import uuid
            temp_path = f"{args.target}_sample_{uuid.uuid4().hex[:8]}"
            
            if args.partition_by:
                sample_df.write.mode("overwrite").option("compression", args.compression) \
                    .partitionBy(args.partition_by).parquet(temp_path)
            else:
                sample_df.write.mode("overwrite").option("compression", args.compression) \
                    .parquet(temp_path)
            
            sample_compressed_size = get_s3_size(spark, temp_path)
            
            # Estimate full size
            if sample_size > 0 and sample_compressed_size > 0:
                ratio = source_count / sample_size
                estimated_compressed = int(sample_compressed_size * ratio)
                savings = source_size - estimated_compressed
                savings_pct = (savings / source_size * 100) if source_size > 0 else 0
                
                print(f"\n[ESTIMATE] Compressed size: ~{format_size(estimated_compressed)}")
                print(f"[ESTIMATE] Savings: ~{format_size(savings)} ({savings_pct:.1f}%)")
                print(f"[ESTIMATE] Compression ratio: {source_size/estimated_compressed:.2f}x")
            
            # Clean up sample
            print(f"\n[INFO] Cleaning up sample at {temp_path}...")
            # Use Hadoop FS to delete
            sc = spark.sparkContext
            hadoop_conf = sc._jsc.hadoopConfiguration()
            uri = sc._jvm.java.net.URI(temp_path)
            fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(uri, hadoop_conf)
            fs.delete(sc._jvm.org.apache.hadoop.fs.Path(temp_path), True)
            
            print("\n[DRY RUN COMPLETE] To actually compress, run without --dry-run")
            return
        
        # Actual compression
        print(f"\n[INFO] Writing compressed parquet to {args.target}...")
        t0 = time.time()
        
        # Optional repartition
        if args.repartition > 0:
            print(f"[INFO] Repartitioning to {args.repartition} partitions...")
            df = df.repartition(args.repartition)
        elif args.coalesce > 0:
            print(f"[INFO] Coalescing to {args.coalesce} partitions...")
            df = df.coalesce(args.coalesce)
        
        # Write with compression
        writer = df.write.mode("overwrite").option("compression", args.compression)
        
        if args.partition_by:
            writer.partitionBy(args.partition_by).parquet(args.target)
        else:
            writer.parquet(args.target)
        
        write_time = time.time() - t0
        print(f"[INFO] Write time: {write_time:.1f}s")
        
        # Verify row count
        print("\n[INFO] Verifying row count...")
        target_df = spark.read.parquet(args.target)
        target_count = target_df.count()
        
        if target_count != source_count:
            print(f"[ERROR] Row count mismatch! Source: {source_count}, Target: {target_count}")
            print("[ERROR] DO NOT DELETE SOURCE DATA!")
            sys.exit(1)
        
        print(f"[INFO] Row count verified: {target_count:,} == {source_count:,} ✓")
        
        # Get target size
        print("\n[INFO] Measuring target size...")
        target_size = get_s3_size(spark, args.target)
        
        # Summary
        savings = source_size - target_size
        savings_pct = (savings / source_size * 100) if source_size > 0 else 0
        ratio = (source_size / target_size) if target_size > 0 else 0
        
        print("\n" + "=" * 60)
        print("COMPRESSION COMPLETE")
        print("=" * 60)
        print(f"Source size:     {format_size(source_size)}")
        print(f"Target size:     {format_size(target_size)}")
        print(f"Savings:         {format_size(savings)} ({savings_pct:.1f}%)")
        print(f"Compression:     {ratio:.2f}x")
        print(f"Row count:       {target_count:,} (verified)")
        print("=" * 60)
        print(f"\n[SUCCESS] Data written to: {args.target}")
        print(f"[INFO] Original data at {args.source} is PRESERVED")
        print(f"[INFO] After verification, you may delete the original.")
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()


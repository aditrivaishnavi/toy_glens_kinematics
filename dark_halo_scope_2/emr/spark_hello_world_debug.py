#!/usr/bin/env python
"""
Minimal PySpark debug script to verify EMR cluster is working.

This script:
  1. Creates a simple RDD
  2. Performs a basic map-reduce
  3. Writes a tiny output to S3
  4. Logs key diagnostics (memory, Python version, installed packages)

Use this to verify your EMR setup BEFORE running the full LRG density job.

Usage (as EMR step):
  spark-submit \
    --deploy-mode client \
    --master yarn \
    --driver-memory 2g \
    --executor-memory 2g \
    --executor-cores 2 \
    --conf spark.dynamicAllocation.enabled=false \
    --conf spark.executor.instances=2 \
    /mnt/dark_halo_scope_code/emr/spark_hello_world_debug.py \
    --output-prefix s3://YOUR_BUCKET/debug

Expected output:
  - A small CSV at s3://YOUR_BUCKET/debug/hello_world_result/
  - Diagnostic logs in stderr
"""

import argparse
import os
import sys
import platform
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType


def get_diagnostics() -> dict:
    """Collect system diagnostics."""
    diag = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    # Check available memory
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    diag["mem_total_kb"] = line.split()[1]
                elif line.startswith("MemAvailable:"):
                    diag["mem_available_kb"] = line.split()[1]
    except Exception as e:
        diag["mem_error"] = str(e)
    
    # Check if key packages are installed
    packages_to_check = ["numpy", "astropy", "requests"]
    for pkg in packages_to_check:
        try:
            mod = __import__(pkg)
            diag[f"{pkg}_version"] = getattr(mod, "__version__", "installed")
        except ImportError:
            diag[f"{pkg}_version"] = "NOT INSTALLED"
    
    return diag


def main():
    parser = argparse.ArgumentParser(description="EMR Hello World Debug")
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="S3 prefix for output (e.g., s3://bucket/debug)",
    )
    parser.add_argument(
        "--input-file",
        required=False,
        default=None,
        help="Optional: S3 path to a small text file to read (for testing S3 access)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("EMR HELLO WORLD DEBUG - STARTING")
    print("=" * 60)

    # Print diagnostics
    diag = get_diagnostics()
    print("\n[DIAGNOSTICS]")
    for k, v in diag.items():
        print(f"  {k}: {v}")

    # Create Spark session
    print("\n[SPARK] Creating SparkSession...")
    spark = SparkSession.builder.appName("dark-halo-scope-debug-hello-world").getOrCreate()
    sc = spark.sparkContext

    print(f"[SPARK] SparkContext created: {sc.applicationId}")
    print(f"[SPARK] Spark version: {sc.version}")
    print(f"[SPARK] Master: {sc.master}")
    print(f"[SPARK] Default parallelism: {sc.defaultParallelism}")

    # Simple RDD operation
    print("\n[TEST 1] Simple RDD map-reduce...")
    test_data = list(range(100))
    rdd = sc.parallelize(test_data, numSlices=4)
    total = rdd.map(lambda x: x * 2).reduce(lambda a, b: a + b)
    expected = sum(x * 2 for x in test_data)
    print(f"  Result: {total}, Expected: {expected}, Match: {total == expected}")

    # Test S3 read if input file provided
    if args.input_file:
        print(f"\n[TEST 2] Reading S3 file: {args.input_file}")
        try:
            lines_rdd = sc.textFile(args.input_file)
            line_count = lines_rdd.count()
            print(f"  Line count: {line_count}")
            if line_count > 0:
                first_lines = lines_rdd.take(3)
                print(f"  First 3 lines: {first_lines}")
        except Exception as e:
            print(f"  ERROR reading file: {e}")
    else:
        print("\n[TEST 2] Skipped (no --input-file provided)")

    # Test numpy inside executor
    print("\n[TEST 3] Testing numpy inside executor...")
    try:
        def test_numpy(x):
            import numpy as np
            return float(np.sqrt(x))

        rdd2 = sc.parallelize([4, 9, 16, 25], numSlices=2)
        sqrt_results = rdd2.map(test_numpy).collect()
        print(f"  sqrt([4,9,16,25]) = {sqrt_results}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Test astropy inside executor (if installed)
    print("\n[TEST 4] Testing astropy inside executor...")
    try:
        def test_astropy(x):
            from astropy.io import fits
            return f"astropy.io.fits loaded for input {x}"

        rdd3 = sc.parallelize([1], numSlices=1)
        astropy_result = rdd3.map(test_astropy).collect()
        print(f"  Result: {astropy_result}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Write output to S3
    print(f"\n[TEST 5] Writing output to S3: {args.output_prefix}")
    try:
        schema = StructType([
            StructField("test_name", StringType(), False),
            StructField("result", StringType(), False),
        ])
        
        result_data = [
            ("rdd_sum", str(total)),
            ("python_version", diag["python_version"]),
            ("spark_version", sc.version),
            ("numpy_version", diag.get("numpy_version", "N/A")),
            ("astropy_version", diag.get("astropy_version", "N/A")),
            ("mem_available_kb", diag.get("mem_available_kb", "N/A")),
        ]
        
        df = spark.createDataFrame(result_data, schema=schema)
        output_path = os.path.join(args.output_prefix.rstrip("/"), "hello_world_result")
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
        print(f"  SUCCESS: Output written to {output_path}")
    except Exception as e:
        print(f"  ERROR writing to S3: {e}")

    spark.stop()

    print("\n" + "=" * 60)
    print("EMR HELLO WORLD DEBUG - COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()




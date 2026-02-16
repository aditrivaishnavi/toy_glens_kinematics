#!/usr/bin/env python3
"""
Phase 3c COMPREHENSIVE Validation - PySpark EMR Job
=====================================================

This is an exhaustive validation of Phase 3c parent catalog output covering:

1. STRUCTURAL INTEGRITY
   - Parquet file counts, sizes, partitioning
   - Schema completeness and types
   - Partition key consistency

2. DATA QUALITY
   - Null/NaN analysis for all columns
   - Value range validation (magnitudes, coordinates, colors)
   - Outlier detection
   - Data type consistency

3. SCIENTIFIC CORRECTNESS
   - LRG variant flag recomputation and comparison
   - Variant hierarchy validation (v1 ⊂ v2 ⊂ v3 ⊂ v4 ⊂ v5)
   - Magnitude-to-color consistency (r-z, z-w1)
   - MW correction validation (if present)
   - Phase 2 LRG selection consistency

4. COVERAGE & COMPLETENESS
   - Region/brick/object counts by split
   - Expected vs actual brick coverage
   - Objects per brick distribution
   - Spatial coverage analysis

5. DATA INTEGRITY
   - Duplicate detection (by objid, by ra/dec)
   - Brick name format validation
   - Region ID consistency with Phase 3a/3b
   - Object ID uniqueness within bricks

6. CROSS-STAGE CONSISTENCY
   - Comparison with Phase 3a bricks_with_region
   - Comparison with Phase 3b region_selections
   - Missing/extra regions detection

7. STATISTICAL ANALYSIS
   - Distribution comparisons across splits (train/val/test)
   - Magnitude distributions by variant
   - Color-color distributions
   - Spatial uniformity tests

8. SAMPLE DATA
   - Random samples for manual inspection
   - Edge cases and potential issues

Usage:
    spark-submit spark_validate_phase3c_comprehensive.py \
        --phase3c-parquet s3://darkhaloscope/phase3_pipeline/phase3c/v3_color_relaxed/parent_union_parquet \
        --phase3a-bricks s3://darkhaloscope/phase3_pipeline/phase3a/v3_color_relaxed/bricks_with_region \
        --phase3b-selections s3://darkhaloscope/phase3_pipeline/phase3b/v3_color_relaxed/region_selections \
        --output-s3 s3://darkhaloscope/phase3_pipeline/phase3c/v3_color_relaxed/validation \
        --variant v3_color_relaxed
"""

import argparse
import json
import math
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel


# ---------------------------------------------------------------------------
# LRG Variant Definitions (MUST match Phase 2 and Phase 3 exactly)
# ---------------------------------------------------------------------------
LRG_VARIANTS = {
    "v1_pure_massive":   {"z_max": 20.0, "rz_min": 0.5, "zw1_min": 1.6},
    "v2_baseline_dr10":  {"z_max": 20.4, "rz_min": 0.4, "zw1_min": 1.6},
    "v3_color_relaxed":  {"z_max": 20.4, "rz_min": 0.4, "zw1_min": 0.8},
    "v4_mag_relaxed":    {"z_max": 21.0, "rz_min": 0.4, "zw1_min": 0.8},
    "v5_very_relaxed":   {"z_max": 21.5, "rz_min": 0.3, "zw1_min": 0.8},
}

# Expected columns in Phase 3c output
EXPECTED_COLUMNS = {
    # Core identifiers
    "region_id": T.IntegerType(),
    "region_split": T.StringType(),
    "brickname": T.StringType(),
    "objid": T.LongType(),
    "ra": T.DoubleType(),
    "dec": T.DoubleType(),
    # Raw magnitudes
    "gmag": T.FloatType(),
    "rmag": T.FloatType(),
    "zmag": T.FloatType(),
    "w1mag": T.FloatType(),
    # Colors (computed)
    "rz": T.FloatType(),
    "zw1": T.FloatType(),
    # MW-corrected magnitudes (optional but expected)
    "gmag_mw": T.FloatType(),
    "rmag_mw": T.FloatType(),
    "zmag_mw": T.FloatType(),
    "w1mag_mw": T.FloatType(),
    "rz_mw": T.FloatType(),
    "zw1_mw": T.FloatType(),
    # Metadata
    "maskbits": T.LongType(),
    "type": T.StringType(),
    # LRG variant flags
    "is_v1_pure_massive": T.BooleanType(),
    "is_v2_baseline_dr10": T.BooleanType(),
    "is_v3_color_relaxed": T.BooleanType(),
    "is_v4_mag_relaxed": T.BooleanType(),
    "is_v5_very_relaxed": T.BooleanType(),
}

# Valid magnitude ranges for galaxies
MAG_VALID_RANGE = {"min": 10.0, "max": 30.0}
COLOR_VALID_RANGE = {"min": -5.0, "max": 10.0}
RA_VALID_RANGE = {"min": 0.0, "max": 360.0}
DEC_VALID_RANGE = {"min": -90.0, "max": 90.0}


def make_spark(app_name: str, shuffle_partitions: int = 400) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )


# ===========================================================================
# SECTION 1: STRUCTURAL INTEGRITY
# ===========================================================================

def validate_structure(spark: SparkSession, parquet_path: str) -> Dict[str, Any]:
    """Analyze Parquet file structure, partitioning, and sizes."""
    print("  [1.1] Analyzing file structure...")
    
    # Get file listing via Hadoop FileSystem
    sc = spark.sparkContext
    hadoop_conf = sc._jsc.hadoopConfiguration()
    
    # Use Spark's internal file listing
    fs_path = sc._jvm.org.apache.hadoop.fs.Path(parquet_path)
    fs = fs_path.getFileSystem(hadoop_conf)
    
    file_statuses = []
    try:
        iterator = fs.listFiles(fs_path, True)  # recursive
        while iterator.hasNext():
            status = iterator.next()
            file_statuses.append({
                "path": str(status.getPath()),
                "size": status.getLen(),
                "is_parquet": str(status.getPath()).endswith(".parquet"),
            })
    except Exception as e:
        print(f"  [WARN] Could not list files: {e}")
        file_statuses = []
    
    parquet_files = [f for f in file_statuses if f["is_parquet"]]
    other_files = [f for f in file_statuses if not f["is_parquet"]]
    
    total_size_bytes = sum(f["size"] for f in parquet_files)
    
    # Analyze partitioning
    partitions = defaultdict(lambda: defaultdict(list))
    for f in parquet_files:
        path = f["path"]
        # Extract partition keys from path
        parts = path.split("/")
        partition_info = {}
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                partition_info[k] = v
        
        split = partition_info.get("region_split", "unknown")
        region = partition_info.get("region_id", "unknown")
        partitions[split][region].append(f)
    
    # File size distribution
    if parquet_files:
        sizes = [f["size"] for f in parquet_files]
        size_stats = {
            "min_bytes": min(sizes),
            "max_bytes": max(sizes),
            "mean_bytes": sum(sizes) / len(sizes),
            "total_bytes": total_size_bytes,
            "total_gb": round(total_size_bytes / (1024**3), 3),
        }
    else:
        size_stats = {}
    
    partition_summary = {}
    for split, regions in partitions.items():
        partition_summary[split] = {
            "n_regions": len(regions),
            "n_files": sum(len(files) for files in regions.values()),
            "total_bytes": sum(f["size"] for files in regions.values() for f in files),
        }
    
    return {
        "total_parquet_files": len(parquet_files),
        "total_other_files": len(other_files),
        "size_stats": size_stats,
        "partitions_by_split": partition_summary,
        "sample_paths": [f["path"] for f in parquet_files[:5]],
    }


def validate_schema(df: DataFrame) -> Dict[str, Any]:
    """Comprehensive schema validation."""
    print("  [1.2] Validating schema...")
    
    actual_schema = {f.name: f.dataType for f in df.schema.fields}
    actual_columns = set(actual_schema.keys())
    expected_columns = set(EXPECTED_COLUMNS.keys())
    
    missing = expected_columns - actual_columns
    extra = actual_columns - expected_columns
    
    # Type mismatches
    type_mismatches = []
    for col, expected_type in EXPECTED_COLUMNS.items():
        if col in actual_schema:
            actual_type = actual_schema[col]
            # Allow some flexibility (e.g., IntegerType vs LongType)
            if not _types_compatible(expected_type, actual_type):
                type_mismatches.append({
                    "column": col,
                    "expected": str(expected_type),
                    "actual": str(actual_type),
                })
    
    # Nullable analysis
    nullable_info = {f.name: f.nullable for f in df.schema.fields}
    
    return {
        "valid": len(missing) == 0,
        "missing_columns": list(missing),
        "extra_columns": list(extra),
        "type_mismatches": type_mismatches,
        "nullable_info": nullable_info,
        "full_schema": {f.name: {"type": str(f.dataType), "nullable": f.nullable} 
                        for f in df.schema.fields},
    }


def _types_compatible(expected: T.DataType, actual: T.DataType) -> bool:
    """Check if types are compatible (allowing some flexibility)."""
    # Exact match
    if type(expected) == type(actual):
        return True
    # Integer types
    int_types = (T.IntegerType, T.LongType, T.ShortType)
    if isinstance(expected, int_types) and isinstance(actual, int_types):
        return True
    # Float types
    float_types = (T.FloatType, T.DoubleType)
    if isinstance(expected, float_types) and isinstance(actual, float_types):
        return True
    return False


# ===========================================================================
# SECTION 2: DATA QUALITY
# ===========================================================================

def validate_data_quality(df: DataFrame) -> Dict[str, Any]:
    """Comprehensive data quality validation."""
    print("  [2.1] Analyzing nulls and NaNs...")
    
    n_rows = df.count()
    
    # Null and NaN counts for all columns
    null_exprs = []
    nan_exprs = []
    for c in df.columns:
        null_exprs.append(F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"null_{c}"))
        # NaN check for numeric columns
        if c in ["gmag", "rmag", "zmag", "w1mag", "ra", "dec", "rz", "zw1",
                 "gmag_mw", "rmag_mw", "zmag_mw", "w1mag_mw", "rz_mw", "zw1_mw"]:
            nan_exprs.append(F.sum(F.when(F.isnan(F.col(c)), 1).otherwise(0)).alias(f"nan_{c}"))
    
    null_row = df.select(null_exprs).collect()[0]
    null_counts = {c: int(null_row[f"null_{c}"]) for c in df.columns}
    
    nan_counts = {}
    if nan_exprs:
        nan_row = df.select(nan_exprs).collect()[0]
        for c in ["gmag", "rmag", "zmag", "w1mag", "ra", "dec", "rz", "zw1",
                  "gmag_mw", "rmag_mw", "zmag_mw", "w1mag_mw", "rz_mw", "zw1_mw"]:
            if f"nan_{c}" in nan_row.asDict():
                nan_counts[c] = int(nan_row[f"nan_{c}"])
    
    # Core columns check (should have zero nulls)
    core_columns = ["region_id", "region_split", "brickname", "objid", "ra", "dec",
                    "gmag", "rmag", "zmag", "w1mag", "rz", "zw1"]
    core_null_issues = {c: null_counts.get(c, 0) for c in core_columns if null_counts.get(c, 0) > 0}
    core_nan_issues = {c: nan_counts.get(c, 0) for c in core_columns if nan_counts.get(c, 0) > 0}
    
    print("  [2.2] Validating value ranges...")
    
    # Magnitude range validation
    mag_validation = {}
    for col in ["gmag", "rmag", "zmag", "w1mag"]:
        if col in df.columns:
            stats = df.select(
                F.min(col).alias("min"),
                F.max(col).alias("max"),
                F.mean(col).alias("mean"),
                F.stddev(col).alias("std"),
                F.percentile_approx(col, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).alias("percentiles"),
                F.count(F.when(F.col(col).isNotNull() & ~F.isnan(F.col(col)), 1)).alias("n_valid"),
                F.count(F.when((F.col(col) < MAG_VALID_RANGE["min"]) | 
                               (F.col(col) > MAG_VALID_RANGE["max"]), 1)).alias("n_out_of_range"),
            ).collect()[0]
            
            percentiles = stats["percentiles"] if stats["percentiles"] else [None]*7
            mag_validation[col] = {
                "min": round(float(stats["min"]), 4) if stats["min"] else None,
                "max": round(float(stats["max"]), 4) if stats["max"] else None,
                "mean": round(float(stats["mean"]), 4) if stats["mean"] else None,
                "std": round(float(stats["std"]), 4) if stats["std"] else None,
                "p01": round(float(percentiles[0]), 4) if percentiles[0] else None,
                "p05": round(float(percentiles[1]), 4) if percentiles[1] else None,
                "p25": round(float(percentiles[2]), 4) if percentiles[2] else None,
                "p50": round(float(percentiles[3]), 4) if percentiles[3] else None,
                "p75": round(float(percentiles[4]), 4) if percentiles[4] else None,
                "p95": round(float(percentiles[5]), 4) if percentiles[5] else None,
                "p99": round(float(percentiles[6]), 4) if percentiles[6] else None,
                "n_valid": int(stats["n_valid"]),
                "n_out_of_range": int(stats["n_out_of_range"]) if stats["n_out_of_range"] else 0,
                "in_valid_range": int(stats["n_out_of_range"] or 0) == 0,
            }
    
    # Color range validation
    color_validation = {}
    for col in ["rz", "zw1", "rz_mw", "zw1_mw"]:
        if col in df.columns:
            stats = df.select(
                F.min(col).alias("min"),
                F.max(col).alias("max"),
                F.mean(col).alias("mean"),
                F.percentile_approx(col, [0.1, 0.5, 0.9]).alias("percentiles"),
                F.count(F.when((F.col(col) < COLOR_VALID_RANGE["min"]) | 
                               (F.col(col) > COLOR_VALID_RANGE["max"]), 1)).alias("n_out_of_range"),
            ).collect()[0]
            
            percentiles = stats["percentiles"] if stats["percentiles"] else [None]*3
            color_validation[col] = {
                "min": round(float(stats["min"]), 4) if stats["min"] else None,
                "max": round(float(stats["max"]), 4) if stats["max"] else None,
                "mean": round(float(stats["mean"]), 4) if stats["mean"] else None,
                "p10": round(float(percentiles[0]), 4) if percentiles[0] else None,
                "p50": round(float(percentiles[1]), 4) if percentiles[1] else None,
                "p90": round(float(percentiles[2]), 4) if percentiles[2] else None,
                "n_out_of_range": int(stats["n_out_of_range"]) if stats["n_out_of_range"] else 0,
            }
    
    # Coordinate validation
    coord_validation = {}
    for col, valid_range in [("ra", RA_VALID_RANGE), ("dec", DEC_VALID_RANGE)]:
        if col in df.columns:
            stats = df.select(
                F.min(col).alias("min"),
                F.max(col).alias("max"),
                F.count(F.when((F.col(col) < valid_range["min"]) | 
                               (F.col(col) > valid_range["max"]), 1)).alias("n_out_of_range"),
            ).collect()[0]
            
            coord_validation[col] = {
                "min": round(float(stats["min"]), 4) if stats["min"] else None,
                "max": round(float(stats["max"]), 4) if stats["max"] else None,
                "valid_range": valid_range,
                "n_out_of_range": int(stats["n_out_of_range"]) if stats["n_out_of_range"] else 0,
                "in_valid_range": int(stats["n_out_of_range"] or 0) == 0,
            }
    
    print("  [2.3] Checking color consistency...")
    
    # Verify rz = rmag - zmag
    color_consistency = {}
    if all(c in df.columns for c in ["rz", "rmag", "zmag"]):
        mismatch = df.filter(
            F.abs(F.col("rz") - (F.col("rmag") - F.col("zmag"))) > 0.001
        ).count()
        color_consistency["rz_vs_rmag_zmag"] = {
            "n_mismatches": int(mismatch),
            "passed": mismatch == 0,
        }
    
    if all(c in df.columns for c in ["zw1", "zmag", "w1mag"]):
        mismatch = df.filter(
            F.abs(F.col("zw1") - (F.col("zmag") - F.col("w1mag"))) > 0.001
        ).count()
        color_consistency["zw1_vs_zmag_w1mag"] = {
            "n_mismatches": int(mismatch),
            "passed": mismatch == 0,
        }
    
    return {
        "n_rows": n_rows,
        "null_counts": null_counts,
        "nan_counts": nan_counts,
        "core_null_issues": core_null_issues,
        "core_nan_issues": core_nan_issues,
        "core_columns_passed": len(core_null_issues) == 0 and len(core_nan_issues) == 0,
        "magnitude_validation": mag_validation,
        "color_validation": color_validation,
        "coordinate_validation": coord_validation,
        "color_consistency": color_consistency,
    }


# ===========================================================================
# SECTION 3: SCIENTIFIC CORRECTNESS
# ===========================================================================

def validate_lrg_flags(df: DataFrame) -> Dict[str, Any]:
    """Validate LRG variant flags against recomputed values."""
    print("  [3.1] Recomputing LRG flags and comparing...")
    
    n_rows = df.count()
    results = {}
    
    # Add computed colors
    df_check = df.withColumn("rz_computed", F.col("rmag") - F.col("zmag")) \
                 .withColumn("zw1_computed", F.col("zmag") - F.col("w1mag"))
    
    for vname, cuts in LRG_VARIANTS.items():
        flag_col = f"is_{vname}"
        if flag_col not in df.columns:
            results[vname] = {"error": f"Missing column {flag_col}"}
            continue
        
        # Compute expected flag using exact same logic as Phase 2/3
        expected_expr = (
            (F.col("zmag") < F.lit(cuts["z_max"])) &
            (F.col("rz_computed") > F.lit(cuts["rz_min"])) &
            (F.col("zw1_computed") > F.lit(cuts["zw1_min"])) &
            F.col("zmag").isNotNull() & ~F.isnan(F.col("zmag")) &
            F.col("rz_computed").isNotNull() & ~F.isnan(F.col("rz_computed")) &
            F.col("zw1_computed").isNotNull() & ~F.isnan(F.col("zw1_computed"))
        )
        
        # Detailed comparison
        comparison = df_check.select(
            F.sum(F.when(expected_expr, 1).otherwise(0)).alias("n_expected_true"),
            F.sum(F.when(F.col(flag_col) == True, 1).otherwise(0)).alias("n_actual_true"),
            F.sum(F.when(expected_expr & (F.col(flag_col) == True), 1).otherwise(0)).alias("true_positive"),
            F.sum(F.when(~expected_expr & (F.col(flag_col) == False), 1).otherwise(0)).alias("true_negative"),
            F.sum(F.when(expected_expr & (F.col(flag_col) == False), 1).otherwise(0)).alias("false_negative"),
            F.sum(F.when(~expected_expr & (F.col(flag_col) == True), 1).otherwise(0)).alias("false_positive"),
        ).collect()[0]
        
        n_expected = int(comparison["n_expected_true"])
        n_actual = int(comparison["n_actual_true"])
        tp = int(comparison["true_positive"])
        tn = int(comparison["true_negative"])
        fn = int(comparison["false_negative"])
        fp = int(comparison["false_positive"])
        
        n_matches = tp + tn
        n_mismatches = fn + fp
        
        results[vname] = {
            "cuts": cuts,
            "n_rows": n_rows,
            "n_expected_true": n_expected,
            "n_actual_true": n_actual,
            "true_positive": tp,
            "true_negative": tn,
            "false_negative": fn,
            "false_positive": fp,
            "n_matches": n_matches,
            "n_mismatches": n_mismatches,
            "match_rate": round(n_matches / n_rows * 100, 4),
            "passed": n_mismatches == 0,
        }
    
    print("  [3.2] Checking variant hierarchy...")
    
    # Hierarchy check: stricter variant implies relaxed variant
    hierarchy_checks = []
    variant_order = ["v1_pure_massive", "v2_baseline_dr10", "v3_color_relaxed", 
                     "v4_mag_relaxed", "v5_very_relaxed"]
    
    for i in range(len(variant_order) - 1):
        strict_col = f"is_{variant_order[i]}"
        relaxed_col = f"is_{variant_order[i+1]}"
        
        if strict_col in df.columns and relaxed_col in df.columns:
            # Count violations: strict=True but relaxed=False
            violations_df = df.filter(
                (F.col(strict_col) == True) & (F.col(relaxed_col) == False)
            )
            n_violations = violations_df.count()
            
            # Sample violations if any
            sample_violations = []
            if n_violations > 0:
                samples = violations_df.select(
                    "brickname", "objid", "zmag", "rz", "zw1", strict_col, relaxed_col
                ).limit(5).collect()
                sample_violations = [row.asDict() for row in samples]
            
            hierarchy_checks.append({
                "strict": variant_order[i],
                "relaxed": variant_order[i+1],
                "n_violations": n_violations,
                "passed": n_violations == 0,
                "sample_violations": sample_violations,
            })
    
    results["hierarchy_checks"] = hierarchy_checks
    
    return results


def validate_mw_correction(df: DataFrame) -> Dict[str, Any]:
    """Validate MW-corrected magnitudes if present."""
    print("  [3.3] Validating MW correction...")
    
    mw_cols = ["gmag_mw", "rmag_mw", "zmag_mw", "w1mag_mw"]
    raw_cols = ["gmag", "rmag", "zmag", "w1mag"]
    
    if not all(c in df.columns for c in mw_cols):
        return {"status": "MW columns not present", "skipped": True}
    
    # MW-corrected should generally be brighter (lower mag) than raw
    # Because MW extinction dims objects
    validation = {}
    for mw_col, raw_col in zip(mw_cols, raw_cols):
        # Check that mw_mag <= raw_mag (with small tolerance for rounding)
        n_invalid = df.filter(
            F.col(mw_col).isNotNull() & 
            (F.col(mw_col) > F.col(raw_col) + 0.01)  # MW should make brighter, not dimmer
        ).count()
        
        # Statistics on correction magnitude
        stats = df.select(
            F.mean(F.col(raw_col) - F.col(mw_col)).alias("mean_correction"),
            F.min(F.col(raw_col) - F.col(mw_col)).alias("min_correction"),
            F.max(F.col(raw_col) - F.col(mw_col)).alias("max_correction"),
        ).collect()[0]
        
        validation[mw_col] = {
            "n_invalid_direction": n_invalid,
            "mean_correction_mag": round(float(stats["mean_correction"]), 4) if stats["mean_correction"] else None,
            "min_correction_mag": round(float(stats["min_correction"]), 4) if stats["min_correction"] else None,
            "max_correction_mag": round(float(stats["max_correction"]), 4) if stats["max_correction"] else None,
            "passed": n_invalid == 0,
        }
    
    return {"status": "validated", "skipped": False, "details": validation}


# ===========================================================================
# SECTION 4: COVERAGE & COMPLETENESS
# ===========================================================================

def validate_coverage(df: DataFrame) -> Dict[str, Any]:
    """Comprehensive coverage analysis."""
    print("  [4.1] Computing coverage statistics...")
    
    n_total = df.count()
    n_regions = df.select("region_id").distinct().count()
    n_bricks = df.select("brickname").distinct().count()
    
    # By split
    split_stats = df.groupBy("region_split").agg(
        F.countDistinct("region_id").alias("n_regions"),
        F.countDistinct("brickname").alias("n_bricks"),
        F.count("objid").alias("n_objects"),
        F.min("ra").alias("ra_min"),
        F.max("ra").alias("ra_max"),
        F.min("dec").alias("dec_min"),
        F.max("dec").alias("dec_max"),
    ).collect()
    
    by_split = {}
    for row in split_stats:
        by_split[row["region_split"]] = {
            "n_regions": int(row["n_regions"]),
            "n_bricks": int(row["n_bricks"]),
            "n_objects": int(row["n_objects"]),
            "pct_of_total": round(row["n_objects"] / n_total * 100, 2),
            "ra_range": [round(float(row["ra_min"]), 2), round(float(row["ra_max"]), 2)],
            "dec_range": [round(float(row["dec_min"]), 2), round(float(row["dec_max"]), 2)],
        }
    
    print("  [4.2] Analyzing region sizes...")
    
    # Region size distribution
    region_stats = df.groupBy("region_split", "region_id").agg(
        F.countDistinct("brickname").alias("n_bricks"),
        F.count("objid").alias("n_objects"),
    )
    
    region_size_stats = region_stats.select(
        F.min("n_bricks").alias("min_bricks"),
        F.max("n_bricks").alias("max_bricks"),
        F.mean("n_bricks").alias("mean_bricks"),
        F.min("n_objects").alias("min_objects"),
        F.max("n_objects").alias("max_objects"),
        F.mean("n_objects").alias("mean_objects"),
    ).collect()[0]
    
    region_distribution = {
        "bricks_per_region": {
            "min": int(region_size_stats["min_bricks"]),
            "max": int(region_size_stats["max_bricks"]),
            "mean": round(float(region_size_stats["mean_bricks"]), 2),
        },
        "objects_per_region": {
            "min": int(region_size_stats["min_objects"]),
            "max": int(region_size_stats["max_objects"]),
            "mean": round(float(region_size_stats["mean_objects"]), 2),
        },
    }
    
    # Top regions
    top_regions = region_stats.orderBy(F.desc("n_objects")).limit(20).collect()
    top_regions_list = [{
        "region_split": row["region_split"],
        "region_id": int(row["region_id"]),
        "n_bricks": int(row["n_bricks"]),
        "n_objects": int(row["n_objects"]),
    } for row in top_regions]
    
    print("  [4.3] Analyzing brick coverage...")
    
    # Objects per brick
    brick_stats = df.groupBy("brickname").agg(F.count("objid").alias("n_objects"))
    opb_stats = brick_stats.select(
        F.min("n_objects").alias("min"),
        F.max("n_objects").alias("max"),
        F.mean("n_objects").alias("mean"),
        F.percentile_approx("n_objects", [0.25, 0.5, 0.75, 0.9, 0.99]).alias("percentiles"),
    ).collect()[0]
    
    percentiles = opb_stats["percentiles"] if opb_stats["percentiles"] else [None]*5
    objects_per_brick = {
        "min": int(opb_stats["min"]),
        "max": int(opb_stats["max"]),
        "mean": round(float(opb_stats["mean"]), 2),
        "p25": int(percentiles[0]) if percentiles[0] else None,
        "p50": int(percentiles[1]) if percentiles[1] else None,
        "p75": int(percentiles[2]) if percentiles[2] else None,
        "p90": int(percentiles[3]) if percentiles[3] else None,
        "p99": int(percentiles[4]) if percentiles[4] else None,
    }
    
    return {
        "n_total_objects": n_total,
        "n_unique_regions": n_regions,
        "n_unique_bricks": n_bricks,
        "by_split": by_split,
        "region_distribution": region_distribution,
        "top_20_regions": top_regions_list,
        "objects_per_brick": objects_per_brick,
    }


# ===========================================================================
# SECTION 5: DATA INTEGRITY
# ===========================================================================

def validate_data_integrity(df: DataFrame) -> Dict[str, Any]:
    """Validate data integrity: duplicates, consistency, format."""
    print("  [5.1] Checking for duplicates...")
    
    n_total = df.count()
    
    # Exact duplicates (all columns)
    n_distinct = df.distinct().count()
    n_exact_duplicates = n_total - n_distinct
    
    # Duplicates by (brickname, objid)
    brick_objid_counts = df.groupBy("brickname", "objid").count()
    n_duplicate_brick_objid = brick_objid_counts.filter(F.col("count") > 1).count()
    
    # Sample duplicates
    sample_duplicates = []
    if n_duplicate_brick_objid > 0:
        dup_keys = brick_objid_counts.filter(F.col("count") > 1).limit(5).collect()
        sample_duplicates = [{"brickname": r["brickname"], "objid": r["objid"], "count": r["count"]} 
                            for r in dup_keys]
    
    print("  [5.2] Validating brickname format...")
    
    # Brickname format: should match pattern like "1234p567" or "1234m567"
    # Format: 4 digits + p/m + 3 digits
    brickname_pattern = r"^\d{4}[pm]\d{3}$"
    n_invalid_brickname = df.filter(
        ~F.col("brickname").rlike(brickname_pattern)
    ).count()
    
    # Sample invalid bricknames
    sample_invalid_bricks = []
    if n_invalid_brickname > 0:
        samples = df.filter(~F.col("brickname").rlike(brickname_pattern)) \
                    .select("brickname").distinct().limit(10).collect()
        sample_invalid_bricks = [r["brickname"] for r in samples]
    
    print("  [5.3] Validating region_split values...")
    
    # region_split should be one of train/val/test
    valid_splits = ["train", "val", "test"]
    split_counts = df.groupBy("region_split").count().collect()
    split_distribution = {r["region_split"]: int(r["count"]) for r in split_counts}
    
    invalid_splits = [s for s in split_distribution.keys() if s not in valid_splits]
    
    print("  [5.4] Validating TYPE values...")
    
    # TYPE should not include PSF
    type_counts = df.groupBy("type").count().orderBy(F.desc("count")).collect()
    type_distribution = {r["type"]: int(r["count"]) for r in type_counts}
    
    n_psf = type_distribution.get("PSF", 0) + type_distribution.get("psf", 0)
    
    # Valid galaxy types
    expected_types = {"DEV", "EXP", "REX", "SER", "COMP", "DUP"}
    unexpected_types = {t for t in type_distribution.keys() if t.upper() not in expected_types and t.upper() != "PSF"}
    
    return {
        "n_total_rows": n_total,
        "duplicates": {
            "n_exact_duplicates": n_exact_duplicates,
            "n_duplicate_brick_objid": n_duplicate_brick_objid,
            "has_duplicates": n_exact_duplicates > 0 or n_duplicate_brick_objid > 0,
            "sample_duplicates": sample_duplicates,
        },
        "brickname_validation": {
            "n_invalid_format": n_invalid_brickname,
            "passed": n_invalid_brickname == 0,
            "sample_invalid": sample_invalid_bricks,
        },
        "split_validation": {
            "distribution": split_distribution,
            "invalid_splits": invalid_splits,
            "passed": len(invalid_splits) == 0,
        },
        "type_validation": {
            "distribution": type_distribution,
            "n_psf": n_psf,
            "psf_filter_passed": n_psf == 0,
            "unexpected_types": list(unexpected_types),
        },
    }


# ===========================================================================
# SECTION 6: CROSS-STAGE CONSISTENCY
# ===========================================================================

def validate_cross_stage(
    df: DataFrame, 
    spark: SparkSession,
    phase3a_path: Optional[str],
    phase3b_path: Optional[str],
) -> Dict[str, Any]:
    """Validate consistency with Phase 3a and 3b outputs."""
    results = {}
    
    if phase3a_path:
        print("  [6.1] Comparing with Phase 3a bricks_with_region...")
        try:
            bricks_3a = spark.read.parquet(phase3a_path)
            
            # Expected bricks from 3a
            expected_bricks = set(r["brickname"] for r in bricks_3a.select("brickname").distinct().collect())
            actual_bricks = set(r["brickname"] for r in df.select("brickname").distinct().collect())
            
            # Region IDs from 3a vs 3c
            expected_regions = set(r["region_id"] for r in bricks_3a.select("region_id").distinct().collect())
            actual_regions = set(r["region_id"] for r in df.select("region_id").distinct().collect())
            
            results["phase3a_comparison"] = {
                "expected_bricks": len(expected_bricks),
                "actual_bricks": len(actual_bricks),
                "missing_bricks": len(expected_bricks - actual_bricks),
                "extra_bricks": len(actual_bricks - expected_bricks),
                "brick_coverage_pct": round(len(actual_bricks & expected_bricks) / len(expected_bricks) * 100, 2) if expected_bricks else 0,
                "expected_regions": len(expected_regions),
                "actual_regions": len(actual_regions),
                "missing_regions": len(expected_regions - actual_regions),
                "extra_regions": len(actual_regions - expected_regions),
                "sample_missing_bricks": list(expected_bricks - actual_bricks)[:10],
            }
        except Exception as e:
            results["phase3a_comparison"] = {"error": str(e)}
    else:
        results["phase3a_comparison"] = {"skipped": True, "reason": "path not provided"}
    
    if phase3b_path:
        print("  [6.2] Comparing with Phase 3b region_selections...")
        try:
            selections_3b = spark.read.parquet(phase3b_path)
            
            # Selected regions from 3b
            selected_regions = set(r["region_id"] for r in selections_3b.select("region_id").distinct().collect())
            actual_regions = set(r["region_id"] for r in df.select("region_id").distinct().collect())
            
            results["phase3b_comparison"] = {
                "selected_regions_3b": len(selected_regions),
                "actual_regions_3c": len(actual_regions),
                "missing_selected_regions": len(selected_regions - actual_regions),
                "extra_regions": len(actual_regions - selected_regions),
                "coverage_pct": round(len(actual_regions & selected_regions) / len(selected_regions) * 100, 2) if selected_regions else 0,
                "sample_missing_regions": list(selected_regions - actual_regions)[:10],
            }
        except Exception as e:
            results["phase3b_comparison"] = {"error": str(e)}
    else:
        results["phase3b_comparison"] = {"skipped": True, "reason": "path not provided"}
    
    return results


# ===========================================================================
# SECTION 7: STATISTICAL ANALYSIS
# ===========================================================================

def validate_statistics(df: DataFrame) -> Dict[str, Any]:
    """Statistical analysis of distributions."""
    print("  [7.1] Comparing distributions across splits...")
    
    # Magnitude distributions by split
    mag_by_split = {}
    for split in ["train", "val", "test"]:
        split_df = df.filter(F.col("region_split") == split)
        if split_df.count() == 0:
            continue
        
        stats = split_df.select(
            F.mean("zmag").alias("zmag_mean"),
            F.stddev("zmag").alias("zmag_std"),
            F.mean("rz").alias("rz_mean"),
            F.mean("zw1").alias("zw1_mean"),
        ).collect()[0]
        
        mag_by_split[split] = {
            "zmag_mean": round(float(stats["zmag_mean"]), 4) if stats["zmag_mean"] else None,
            "zmag_std": round(float(stats["zmag_std"]), 4) if stats["zmag_std"] else None,
            "rz_mean": round(float(stats["rz_mean"]), 4) if stats["rz_mean"] else None,
            "zw1_mean": round(float(stats["zw1_mean"]), 4) if stats["zw1_mean"] else None,
        }
    
    # Check for significant differences (simple comparison)
    split_consistency = {"passed": True, "notes": []}
    if len(mag_by_split) >= 2:
        splits = list(mag_by_split.keys())
        for i in range(len(splits)):
            for j in range(i+1, len(splits)):
                s1, s2 = splits[i], splits[j]
                for metric in ["zmag_mean", "rz_mean", "zw1_mean"]:
                    v1 = mag_by_split[s1].get(metric)
                    v2 = mag_by_split[s2].get(metric)
                    if v1 and v2:
                        diff = abs(v1 - v2)
                        # Flag large differences
                        threshold = 0.5 if "mag" in metric else 0.2
                        if diff > threshold:
                            split_consistency["passed"] = False
                            split_consistency["notes"].append(
                                f"{metric}: {s1}={v1:.3f} vs {s2}={v2:.3f} (diff={diff:.3f})"
                            )
    
    print("  [7.2] Analyzing variant distribution by split...")
    
    # Variant counts by split
    variant_by_split = {}
    for split in ["train", "val", "test"]:
        split_df = df.filter(F.col("region_split") == split)
        n_split = split_df.count()
        if n_split == 0:
            continue
        
        variant_counts = {}
        for vname in LRG_VARIANTS.keys():
            flag_col = f"is_{vname}"
            if flag_col in df.columns:
                n_v = split_df.filter(F.col(flag_col) == True).count()
                variant_counts[vname] = {
                    "count": n_v,
                    "pct": round(n_v / n_split * 100, 2),
                }
        variant_by_split[split] = variant_counts
    
    return {
        "magnitude_stats_by_split": mag_by_split,
        "split_consistency": split_consistency,
        "variant_distribution_by_split": variant_by_split,
    }


# ===========================================================================
# SECTION 8: SAMPLE DATA
# ===========================================================================

def extract_samples(df: DataFrame) -> Dict[str, Any]:
    """Extract sample rows for manual inspection."""
    print("  [8.1] Extracting random samples...")
    
    # Random sample
    random_sample = df.sample(fraction=0.0001, seed=42).limit(20).collect()
    random_sample_list = [row.asDict() for row in random_sample]
    
    print("  [8.2] Extracting edge cases...")
    
    # Brightest objects
    brightest = df.orderBy("zmag").limit(10).collect()
    brightest_list = [{"brickname": r["brickname"], "objid": r["objid"], 
                       "zmag": r["zmag"], "rz": r["rz"], "zw1": r["zw1"]} for r in brightest]
    
    # Faintest objects
    faintest = df.orderBy(F.desc("zmag")).limit(10).collect()
    faintest_list = [{"brickname": r["brickname"], "objid": r["objid"], 
                      "zmag": r["zmag"], "rz": r["rz"], "zw1": r["zw1"]} for r in faintest]
    
    # Reddest (highest r-z)
    reddest = df.orderBy(F.desc("rz")).limit(10).collect()
    reddest_list = [{"brickname": r["brickname"], "objid": r["objid"], 
                     "zmag": r["zmag"], "rz": r["rz"], "zw1": r["zw1"]} for r in reddest]
    
    # Objects near variant boundaries (close to cuts)
    # v3: z_max=20.4, rz_min=0.4, zw1_min=0.8
    boundary_v3 = df.filter(
        (F.abs(F.col("zmag") - 20.4) < 0.1) |
        (F.abs(F.col("rz") - 0.4) < 0.05) |
        (F.abs(F.col("zw1") - 0.8) < 0.05)
    ).limit(20).collect()
    boundary_list = [{"brickname": r["brickname"], "objid": r["objid"], 
                      "zmag": r["zmag"], "rz": r["rz"], "zw1": r["zw1"],
                      "is_v3": r["is_v3_color_relaxed"]} for r in boundary_v3]
    
    return {
        "random_sample": random_sample_list,
        "brightest_objects": brightest_list,
        "faintest_objects": faintest_list,
        "reddest_objects": reddest_list,
        "v3_boundary_objects": boundary_list,
    }


# ===========================================================================
# REPORT GENERATION
# ===========================================================================

def generate_comprehensive_report(results: Dict) -> str:
    """Generate comprehensive human-readable report."""
    lines = []
    
    lines.append("=" * 80)
    lines.append("PHASE 3C COMPREHENSIVE VALIDATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {results['timestamp']}")
    lines.append(f"Input: {results['input']}")
    lines.append(f"Variant: {results['variant']}")
    lines.append("")
    
    # =========== EXECUTIVE SUMMARY ===========
    lines.append("-" * 80)
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 80)
    
    checks = []
    all_passed = True
    
    # Schema
    schema = results.get("schema", {})
    if schema.get("valid"):
        checks.append("✅ Schema complete (all expected columns present)")
    else:
        checks.append(f"❌ Schema incomplete: missing {schema.get('missing_columns')}")
        all_passed = False
    
    # Data quality
    quality = results.get("data_quality", {})
    if quality.get("core_columns_passed"):
        checks.append("✅ Core columns have no nulls/NaNs")
    else:
        checks.append(f"❌ Core columns have issues: nulls={quality.get('core_null_issues')}, nans={quality.get('core_nan_issues')}")
        all_passed = False
    
    # LRG flags
    lrg = results.get("lrg_validation", {})
    lrg_passed = all(v.get("passed", False) for k, v in lrg.items() if k != "hierarchy_checks" and isinstance(v, dict))
    if lrg_passed:
        checks.append("✅ LRG flags match recomputed values (all 5 variants)")
    else:
        failed = [k for k, v in lrg.items() if k != "hierarchy_checks" and isinstance(v, dict) and not v.get("passed", True)]
        checks.append(f"❌ LRG flag mismatches: {failed}")
        all_passed = False
    
    # Hierarchy
    hierarchy = lrg.get("hierarchy_checks", [])
    hierarchy_passed = all(h.get("passed", False) for h in hierarchy)
    if hierarchy_passed:
        checks.append("✅ LRG variant hierarchy is consistent (v1⊂v2⊂v3⊂v4⊂v5)")
    else:
        checks.append("❌ LRG variant hierarchy has violations")
        all_passed = False
    
    # Integrity
    integrity = results.get("data_integrity", {})
    dup = integrity.get("duplicates", {})
    if not dup.get("has_duplicates"):
        checks.append("✅ No duplicate rows detected")
    else:
        checks.append(f"❌ Duplicates found: exact={dup.get('n_exact_duplicates')}, by_key={dup.get('n_duplicate_brick_objid')}")
        all_passed = False
    
    # Type filter
    type_val = integrity.get("type_validation", {})
    if type_val.get("psf_filter_passed"):
        checks.append("✅ No PSF (star) objects in catalog")
    else:
        checks.append(f"❌ Found {type_val.get('n_psf')} PSF objects")
        all_passed = False
    
    # Brickname format
    brick_val = integrity.get("brickname_validation", {})
    if brick_val.get("passed"):
        checks.append("✅ All bricknames have valid format")
    else:
        checks.append(f"❌ Invalid brickname format: {brick_val.get('n_invalid_format')} rows")
        all_passed = False
    
    for check in checks:
        lines.append(f"  {check}")
    
    lines.append("")
    if all_passed:
        lines.append("🎉 OVERALL STATUS: ALL CHECKS PASSED")
    else:
        lines.append("⚠️  OVERALL STATUS: SOME CHECKS FAILED - Review details below")
    lines.append("")
    
    # =========== DATA OVERVIEW ===========
    lines.append("-" * 80)
    lines.append("DATA OVERVIEW")
    lines.append("-" * 80)
    
    coverage = results.get("coverage", {})
    lines.append(f"  Total LRG Objects:  {coverage.get('n_total_objects', 'N/A'):,}")
    lines.append(f"  Unique Regions:     {coverage.get('n_unique_regions', 'N/A'):,}")
    lines.append(f"  Unique Bricks:      {coverage.get('n_unique_bricks', 'N/A'):,}")
    lines.append("")
    
    structure = results.get("structure", {})
    size_stats = structure.get("size_stats", {})
    lines.append(f"  Parquet Files:      {structure.get('total_parquet_files', 'N/A'):,}")
    lines.append(f"  Total Size:         {size_stats.get('total_gb', 'N/A')} GB")
    lines.append("")
    
    lines.append("  By Split:")
    for split, stats in coverage.get("by_split", {}).items():
        lines.append(f"    {split:5}: {stats['n_regions']:,} regions | {stats['n_bricks']:,} bricks | {stats['n_objects']:,} objects ({stats['pct_of_total']}%)")
    lines.append("")
    
    # =========== SCHEMA ===========
    lines.append("-" * 80)
    lines.append("SCHEMA VALIDATION")
    lines.append("-" * 80)
    
    if schema.get("missing_columns"):
        lines.append(f"  ❌ Missing columns: {schema['missing_columns']}")
    if schema.get("extra_columns"):
        lines.append(f"  ℹ️  Extra columns: {schema['extra_columns']}")
    if schema.get("type_mismatches"):
        lines.append(f"  ⚠️  Type mismatches:")
        for tm in schema["type_mismatches"]:
            lines.append(f"      {tm['column']}: expected {tm['expected']}, got {tm['actual']}")
    if not schema.get("missing_columns") and not schema.get("type_mismatches"):
        lines.append("  ✅ All columns present with expected types")
    lines.append("")
    
    # =========== DATA QUALITY ===========
    lines.append("-" * 80)
    lines.append("DATA QUALITY")
    lines.append("-" * 80)
    
    lines.append("  Magnitude Statistics:")
    lines.append("  " + "-" * 76)
    lines.append(f"  {'Band':<8} {'Min':>8} {'P05':>8} {'P50':>8} {'P95':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
    lines.append("  " + "-" * 76)
    for band, stats in quality.get("magnitude_validation", {}).items():
        lines.append(f"  {band:<8} {stats.get('min', 'N/A'):>8} {stats.get('p05', 'N/A'):>8} {stats.get('p50', 'N/A'):>8} {stats.get('p95', 'N/A'):>8} {stats.get('max', 'N/A'):>8} {stats.get('mean', 'N/A'):>8} {stats.get('std', 'N/A'):>8}")
    lines.append("")
    
    lines.append("  Color Statistics:")
    lines.append("  " + "-" * 60)
    lines.append(f"  {'Color':<8} {'Min':>8} {'P10':>8} {'P50':>8} {'P90':>8} {'Max':>8}")
    lines.append("  " + "-" * 60)
    for color, stats in quality.get("color_validation", {}).items():
        lines.append(f"  {color:<8} {stats.get('min', 'N/A'):>8} {stats.get('p10', 'N/A'):>8} {stats.get('p50', 'N/A'):>8} {stats.get('p90', 'N/A'):>8} {stats.get('max', 'N/A'):>8}")
    lines.append("")
    
    lines.append("  Coordinate Ranges:")
    for coord, stats in quality.get("coordinate_validation", {}).items():
        status = "✅" if stats.get("in_valid_range") else "❌"
        lines.append(f"    {coord}: {stats.get('min')} to {stats.get('max')} {status}")
    lines.append("")
    
    lines.append("  Color Consistency (r-z = rmag - zmag):")
    for check_name, check in quality.get("color_consistency", {}).items():
        status = "✅" if check.get("passed") else f"❌ {check.get('n_mismatches')} mismatches"
        lines.append(f"    {check_name}: {status}")
    lines.append("")
    
    # =========== LRG FLAG VALIDATION ===========
    lines.append("-" * 80)
    lines.append("LRG FLAG VALIDATION")
    lines.append("-" * 80)
    
    lines.append("  Recomputed vs Stored Flags:")
    lines.append("  " + "-" * 76)
    lines.append(f"  {'Variant':<20} {'Expected':>10} {'Actual':>10} {'Match%':>10} {'FP':>8} {'FN':>8} {'Status':>8}")
    lines.append("  " + "-" * 76)
    for vname in ["v1_pure_massive", "v2_baseline_dr10", "v3_color_relaxed", "v4_mag_relaxed", "v5_very_relaxed"]:
        v = lrg.get(vname, {})
        if "error" in v:
            lines.append(f"  {vname:<20} {'ERROR':>10} {v['error']}")
        else:
            status = "✅" if v.get("passed") else "❌"
            lines.append(f"  {vname:<20} {v.get('n_expected_true', 'N/A'):>10,} {v.get('n_actual_true', 'N/A'):>10,} {v.get('match_rate', 'N/A'):>9}% {v.get('false_positive', 'N/A'):>8} {v.get('false_negative', 'N/A'):>8} {status:>8}")
    lines.append("")
    
    lines.append("  Variant Hierarchy (v1 ⊂ v2 ⊂ v3 ⊂ v4 ⊂ v5):")
    for h in lrg.get("hierarchy_checks", []):
        status = "✅" if h.get("passed") else f"❌ {h.get('n_violations')} violations"
        lines.append(f"    {h['strict']} ⊂ {h['relaxed']}: {status}")
    lines.append("")
    
    # =========== DATA INTEGRITY ===========
    lines.append("-" * 80)
    lines.append("DATA INTEGRITY")
    lines.append("-" * 80)
    
    dup = integrity.get("duplicates", {})
    lines.append(f"  Exact duplicate rows: {dup.get('n_exact_duplicates', 0):,}")
    lines.append(f"  Duplicate (brickname, objid): {dup.get('n_duplicate_brick_objid', 0):,}")
    
    lines.append(f"  Invalid brickname format: {brick_val.get('n_invalid_format', 0):,}")
    if brick_val.get("sample_invalid"):
        lines.append(f"    Samples: {brick_val['sample_invalid'][:5]}")
    
    lines.append("")
    lines.append("  Split Distribution:")
    for split, count in integrity.get("split_validation", {}).get("distribution", {}).items():
        lines.append(f"    {split}: {count:,}")
    
    lines.append("")
    lines.append("  TYPE Distribution:")
    for typ, count in sorted(type_val.get("distribution", {}).items(), key=lambda x: -x[1])[:10]:
        lines.append(f"    {typ}: {count:,}")
    lines.append("")
    
    # =========== CROSS-STAGE CONSISTENCY ===========
    cross = results.get("cross_stage", {})
    
    lines.append("-" * 80)
    lines.append("CROSS-STAGE CONSISTENCY")
    lines.append("-" * 80)
    
    p3a = cross.get("phase3a_comparison", {})
    if p3a.get("skipped"):
        lines.append(f"  Phase 3a comparison: Skipped ({p3a.get('reason')})")
    elif p3a.get("error"):
        lines.append(f"  Phase 3a comparison: Error ({p3a.get('error')})")
    else:
        lines.append(f"  Phase 3a Bricks Comparison:")
        lines.append(f"    Expected bricks: {p3a.get('expected_bricks', 'N/A'):,}")
        lines.append(f"    Actual bricks:   {p3a.get('actual_bricks', 'N/A'):,}")
        lines.append(f"    Coverage:        {p3a.get('brick_coverage_pct', 'N/A')}%")
        if p3a.get("missing_bricks", 0) > 0:
            lines.append(f"    ⚠️  Missing {p3a.get('missing_bricks')} bricks from 3a")
    
    lines.append("")
    
    p3b = cross.get("phase3b_comparison", {})
    if p3b.get("skipped"):
        lines.append(f"  Phase 3b comparison: Skipped ({p3b.get('reason')})")
    elif p3b.get("error"):
        lines.append(f"  Phase 3b comparison: Error ({p3b.get('error')})")
    else:
        lines.append(f"  Phase 3b Regions Comparison:")
        lines.append(f"    Selected regions (3b): {p3b.get('selected_regions_3b', 'N/A'):,}")
        lines.append(f"    Actual regions (3c):   {p3b.get('actual_regions_3c', 'N/A'):,}")
        lines.append(f"    Coverage:              {p3b.get('coverage_pct', 'N/A')}%")
    lines.append("")
    
    # =========== STATISTICAL ANALYSIS ===========
    stats = results.get("statistics", {})
    
    lines.append("-" * 80)
    lines.append("STATISTICAL ANALYSIS")
    lines.append("-" * 80)
    
    lines.append("  Magnitude/Color Means by Split:")
    lines.append("  " + "-" * 60)
    lines.append(f"  {'Split':<8} {'zmag_mean':>12} {'zmag_std':>12} {'rz_mean':>12} {'zw1_mean':>12}")
    lines.append("  " + "-" * 60)
    for split, s in stats.get("magnitude_stats_by_split", {}).items():
        lines.append(f"  {split:<8} {s.get('zmag_mean', 'N/A'):>12} {s.get('zmag_std', 'N/A'):>12} {s.get('rz_mean', 'N/A'):>12} {s.get('zw1_mean', 'N/A'):>12}")
    lines.append("")
    
    consistency = stats.get("split_consistency", {})
    if consistency.get("passed"):
        lines.append("  ✅ Split distributions are consistent")
    else:
        lines.append("  ⚠️  Split distribution differences:")
        for note in consistency.get("notes", []):
            lines.append(f"      {note}")
    lines.append("")
    
    # =========== TOP REGIONS ===========
    lines.append("-" * 80)
    lines.append("TOP 20 REGIONS BY OBJECT COUNT")
    lines.append("-" * 80)
    
    lines.append(f"  {'Split':<8} {'Region':>8} {'Bricks':>10} {'Objects':>12}")
    lines.append("  " + "-" * 40)
    for r in coverage.get("top_20_regions", []):
        lines.append(f"  {r['region_split']:<8} {r['region_id']:>8} {r['n_bricks']:>10,} {r['n_objects']:>12,}")
    lines.append("")
    
    # =========== SAMPLE DATA ===========
    samples = results.get("samples", {})
    
    lines.append("-" * 80)
    lines.append("SAMPLE DATA")
    lines.append("-" * 80)
    
    lines.append("  Brightest Objects (lowest zmag):")
    for obj in samples.get("brightest_objects", [])[:5]:
        lines.append(f"    {obj['brickname']} objid={obj['objid']}: zmag={obj['zmag']:.2f}, r-z={obj['rz']:.2f}, z-w1={obj['zw1']:.2f}")
    lines.append("")
    
    lines.append("  Faintest Objects (highest zmag):")
    for obj in samples.get("faintest_objects", [])[:5]:
        lines.append(f"    {obj['brickname']} objid={obj['objid']}: zmag={obj['zmag']:.2f}, r-z={obj['rz']:.2f}, z-w1={obj['zw1']:.2f}")
    lines.append("")
    
    lines.append("  Objects Near v3 Cut Boundaries:")
    for obj in samples.get("v3_boundary_objects", [])[:5]:
        lines.append(f"    {obj['brickname']} objid={obj['objid']}: zmag={obj['zmag']:.2f}, r-z={obj['rz']:.2f}, z-w1={obj['zw1']:.2f}, is_v3={obj['is_v3']}")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    return "\n".join(lines)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Phase 3c Validation")
    parser.add_argument("--phase3c-parquet", required=True, help="S3 path to Phase 3c parent catalog")
    parser.add_argument("--phase3a-bricks", default=None, help="S3 path to Phase 3a bricks_with_region (optional)")
    parser.add_argument("--phase3b-selections", default=None, help="S3 path to Phase 3b region_selections (optional)")
    parser.add_argument("--output-s3", required=True, help="S3 path for validation outputs")
    parser.add_argument("--variant", default="v3_color_relaxed", help="Target LRG variant")
    parser.add_argument("--shuffle-partitions", type=int, default=400)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PHASE 3C COMPREHENSIVE VALIDATION")
    print("=" * 80)
    print(f"Input: {args.phase3c_parquet}")
    print(f"Output: {args.output_s3}")
    print()
    
    spark = make_spark("DHS Phase3c Comprehensive Validation", args.shuffle_partitions)
    
    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input": args.phase3c_parquet,
        "variant": args.variant,
        "phase3a_path": args.phase3a_bricks,
        "phase3b_path": args.phase3b_selections,
    }
    
    # 1. Structure
    print("[1/8] STRUCTURAL INTEGRITY")
    results["structure"] = validate_structure(spark, args.phase3c_parquet)
    
    # Load data
    print("[2/8] LOADING DATA")
    df = spark.read.parquet(args.phase3c_parquet)
    df.persist(StorageLevel.MEMORY_AND_DISK)
    n_rows = df.count()
    print(f"      Loaded {n_rows:,} rows")
    
    # 2. Schema
    print("[3/8] SCHEMA VALIDATION")
    results["schema"] = validate_schema(df)
    
    # 3. Data Quality
    print("[4/8] DATA QUALITY")
    results["data_quality"] = validate_data_quality(df)
    
    # 4. LRG Flags
    print("[5/8] LRG FLAG VALIDATION")
    results["lrg_validation"] = validate_lrg_flags(df)
    results["mw_correction"] = validate_mw_correction(df)
    
    # 5. Coverage
    print("[6/8] COVERAGE ANALYSIS")
    results["coverage"] = validate_coverage(df)
    
    # 6. Integrity
    print("[7/8] DATA INTEGRITY")
    results["data_integrity"] = validate_data_integrity(df)
    
    # 7. Cross-stage
    print("[8/8] CROSS-STAGE & STATISTICS")
    results["cross_stage"] = validate_cross_stage(df, spark, args.phase3a_bricks, args.phase3b_selections)
    results["statistics"] = validate_statistics(df)
    results["samples"] = extract_samples(df)
    
    # Generate outputs
    print()
    print("Generating outputs...")
    
    output_prefix = args.output_s3.rstrip("/")
    
    # JSON report
    json_str = json.dumps(results, indent=2, default=str)
    spark.sparkContext.parallelize([json_str]).coalesce(1).saveAsTextFile(f"{output_prefix}/validation_report_json")
    
    # Text report
    report_text = generate_comprehensive_report(results)
    spark.sparkContext.parallelize([report_text]).coalesce(1).saveAsTextFile(f"{output_prefix}/validation_report_txt")
    
    # Print report to console
    print()
    print(report_text)
    
    df.unpersist()
    spark.stop()
    
    print()
    print(f"Outputs written to: {output_prefix}/")
    print("  - validation_report_json/")
    print("  - validation_report_txt/")


if __name__ == "__main__":
    main()


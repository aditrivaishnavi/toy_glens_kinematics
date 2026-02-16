#!/usr/bin/env python3
"""
Phase 3c Validation - PySpark EMR Job
======================================

Validates Phase 3c parent catalog output using Spark for efficient parallel processing.

Usage (on EMR):
    spark-submit spark_validate_phase3c.py \
        --phase3c-parquet s3://darkhaloscope/phase3_pipeline/phase3c/v3_color_relaxed/parent_union_parquet \
        --output-s3 s3://darkhaloscope/phase3_pipeline/phase3c/v3_color_relaxed/validation \
        --variant v3_color_relaxed

Outputs:
    - validation_report.json - Complete validation results
    - validation_summary.txt - Human-readable summary
"""

import argparse
import json
from datetime import datetime
from typing import Dict, Any, List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window


# ---------------------------------------------------------------------------
# LRG Variant Definitions (must match Phase 2 and Phase 3)
# ---------------------------------------------------------------------------
LRG_VARIANTS = {
    "v1_pure_massive":   {"z_max": 20.0, "rz_min": 0.5, "zw1_min": 1.6},
    "v2_baseline_dr10":  {"z_max": 20.4, "rz_min": 0.4, "zw1_min": 1.6},
    "v3_color_relaxed":  {"z_max": 20.4, "rz_min": 0.4, "zw1_min": 0.8},
    "v4_mag_relaxed":    {"z_max": 21.0, "rz_min": 0.4, "zw1_min": 0.8},
    "v5_very_relaxed":   {"z_max": 21.5, "rz_min": 0.3, "zw1_min": 0.8},
}


def make_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def validate_schema(df: DataFrame) -> Dict[str, Any]:
    """Validate schema has expected columns."""
    expected_columns = {
        "region_id", "region_split", "brickname", "objid", "ra", "dec",
        "gmag", "rmag", "zmag", "w1mag", "rz", "zw1",
        "gmag_mw", "rmag_mw", "zmag_mw", "w1mag_mw", "rz_mw", "zw1_mw",
        "maskbits", "type",
        "is_v1_pure_massive", "is_v2_baseline_dr10", "is_v3_color_relaxed",
        "is_v4_mag_relaxed", "is_v5_very_relaxed",
    }
    
    actual_columns = set(df.columns)
    missing = expected_columns - actual_columns
    extra = actual_columns - expected_columns
    
    return {
        "valid": len(missing) == 0,
        "missing_columns": list(missing),
        "extra_columns": list(extra),
        "actual_columns": df.columns,
        "column_dtypes": {f.name: str(f.dataType) for f in df.schema.fields},
    }


def validate_data_quality(df: DataFrame) -> Dict[str, Any]:
    """Check data quality using Spark aggregations."""
    n_rows = df.count()
    
    # Null counts for all columns
    null_exprs = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"null_{c}") 
                  for c in df.columns]
    null_row = df.select(null_exprs).collect()[0]
    null_counts = {c: int(null_row[f"null_{c}"]) for c in df.columns}
    null_pcts = {c: round(v / n_rows * 100, 4) for c, v in null_counts.items()}
    
    # Core columns null check
    core_columns = ["region_id", "region_split", "brickname", "objid", "ra", "dec",
                    "gmag", "rmag", "zmag", "w1mag", "rz", "zw1"]
    core_nulls = {c: null_counts.get(c, 0) for c in core_columns if c in null_counts}
    
    # Magnitude statistics
    mag_stats = {}
    for col in ["gmag", "rmag", "zmag", "w1mag"]:
        if col in df.columns:
            stats = df.select(
                F.min(col).alias("min"),
                F.max(col).alias("max"),
                F.mean(col).alias("mean"),
                F.percentile_approx(col, 0.5).alias("median"),
                F.stddev(col).alias("std"),
                F.count(F.when(F.col(col).isNotNull(), 1)).alias("n_valid"),
            ).collect()[0]
            mag_stats[col] = {
                "min": round(float(stats["min"]), 3) if stats["min"] else None,
                "max": round(float(stats["max"]), 3) if stats["max"] else None,
                "mean": round(float(stats["mean"]), 3) if stats["mean"] else None,
                "median": round(float(stats["median"]), 3) if stats["median"] else None,
                "std": round(float(stats["std"]), 3) if stats["std"] else None,
                "n_valid": int(stats["n_valid"]),
            }
    
    # Coordinate ranges
    coord_stats = {}
    for col in ["ra", "dec"]:
        if col in df.columns:
            stats = df.select(F.min(col), F.max(col)).collect()[0]
            coord_stats[col] = {
                "min": round(float(stats[0]), 3) if stats[0] else None,
                "max": round(float(stats[1]), 3) if stats[1] else None,
            }
    
    # Color statistics
    color_stats = {}
    for col in ["rz", "zw1"]:
        if col in df.columns:
            stats = df.select(
                F.min(col).alias("min"),
                F.max(col).alias("max"),
                F.mean(col).alias("mean"),
                F.percentile_approx(col, 0.1).alias("p10"),
                F.percentile_approx(col, 0.5).alias("p50"),
                F.percentile_approx(col, 0.9).alias("p90"),
            ).collect()[0]
            color_stats[col] = {
                "min": round(float(stats["min"]), 3) if stats["min"] else None,
                "max": round(float(stats["max"]), 3) if stats["max"] else None,
                "mean": round(float(stats["mean"]), 3) if stats["mean"] else None,
                "p10": round(float(stats["p10"]), 3) if stats["p10"] else None,
                "p50": round(float(stats["p50"]), 3) if stats["p50"] else None,
                "p90": round(float(stats["p90"]), 3) if stats["p90"] else None,
            }
    
    return {
        "n_rows": n_rows,
        "null_counts": null_counts,
        "null_pcts": null_pcts,
        "core_columns_null_check": {
            "passed": all(v == 0 for v in core_nulls.values()),
            "details": core_nulls,
        },
        "magnitude_stats": mag_stats,
        "coordinate_stats": coord_stats,
        "color_stats": color_stats,
    }


def validate_lrg_flags(df: DataFrame) -> Dict[str, Any]:
    """Validate LRG flags are consistent with magnitude/color cuts."""
    results = {}
    n_rows = df.count()
    
    # Add recomputed flag columns
    df_check = df.select(
        "zmag", "rmag", "w1mag",
        (F.col("rmag") - F.col("zmag")).alias("rz_computed"),
        (F.col("zmag") - F.col("w1mag")).alias("zw1_computed"),
        "is_v1_pure_massive", "is_v2_baseline_dr10", "is_v3_color_relaxed",
        "is_v4_mag_relaxed", "is_v5_very_relaxed",
    )
    
    for vname, cuts in LRG_VARIANTS.items():
        flag_col = f"is_{vname}"
        if flag_col not in df.columns:
            results[vname] = {"error": f"Missing column {flag_col}"}
            continue
        
        # Compute expected flag
        expected_expr = (
            (F.col("zmag") < F.lit(cuts["z_max"])) &
            (F.col("rz_computed") > F.lit(cuts["rz_min"])) &
            (F.col("zw1_computed") > F.lit(cuts["zw1_min"])) &
            F.col("zmag").isNotNull() &
            F.col("rz_computed").isNotNull() &
            F.col("zw1_computed").isNotNull()
        )
        
        # Compare
        comparison = df_check.select(
            F.sum(F.when(expected_expr, 1).otherwise(0)).alias("n_expected_true"),
            F.sum(F.when(F.col(flag_col) == True, 1).otherwise(0)).alias("n_actual_true"),
            F.sum(F.when(expected_expr == F.col(flag_col), 1).otherwise(0)).alias("n_matches"),
        ).collect()[0]
        
        n_expected = int(comparison["n_expected_true"])
        n_actual = int(comparison["n_actual_true"])
        n_matches = int(comparison["n_matches"])
        n_mismatches = n_rows - n_matches
        
        results[vname] = {
            "n_rows": n_rows,
            "n_expected_true": n_expected,
            "n_actual_true": n_actual,
            "n_matches": n_matches,
            "n_mismatches": n_mismatches,
            "mismatch_pct": round(n_mismatches / n_rows * 100, 4),
            "passed": n_mismatches == 0,
            "cuts": cuts,
        }
    
    # Hierarchy check
    hierarchy_checks = []
    variant_order = ["v1_pure_massive", "v2_baseline_dr10", "v3_color_relaxed", 
                     "v4_mag_relaxed", "v5_very_relaxed"]
    for i in range(len(variant_order) - 1):
        strict_col = f"is_{variant_order[i]}"
        relaxed_col = f"is_{variant_order[i+1]}"
        if strict_col in df.columns and relaxed_col in df.columns:
            violations = df.filter(
                (F.col(strict_col) == True) & (F.col(relaxed_col) == False)
            ).count()
            hierarchy_checks.append({
                "strict": variant_order[i],
                "relaxed": variant_order[i+1],
                "violations": int(violations),
                "passed": violations == 0,
            })
    
    results["hierarchy_checks"] = hierarchy_checks
    return results


def validate_coverage(df: DataFrame) -> Dict[str, Any]:
    """Validate coverage by regions and splits."""
    # Unique counts
    n_unique_regions = df.select("region_id").distinct().count()
    n_unique_bricks = df.select("brickname").distinct().count()
    n_total_objects = df.count()
    
    # By split
    split_stats = df.groupBy("region_split").agg(
        F.countDistinct("region_id").alias("n_regions"),
        F.countDistinct("brickname").alias("n_bricks"),
        F.count("objid").alias("n_objects"),
    ).collect()
    
    by_split = {row["region_split"]: {
        "n_regions": int(row["n_regions"]),
        "n_bricks": int(row["n_bricks"]),
        "n_objects": int(row["n_objects"]),
    } for row in split_stats}
    
    # Top regions by object count
    top_regions = df.groupBy("region_split", "region_id").agg(
        F.countDistinct("brickname").alias("n_bricks"),
        F.count("objid").alias("n_objects"),
    ).orderBy(F.desc("n_objects")).limit(10).collect()
    
    top_regions_list = [{
        "region_split": row["region_split"],
        "region_id": int(row["region_id"]),
        "n_bricks": int(row["n_bricks"]),
        "n_objects": int(row["n_objects"]),
    } for row in top_regions]
    
    # Objects per brick
    opb = df.groupBy("brickname").agg(F.count("objid").alias("n_obj"))
    opb_stats = opb.select(
        F.min("n_obj").alias("min"),
        F.max("n_obj").alias("max"),
        F.mean("n_obj").alias("mean"),
        F.percentile_approx("n_obj", 0.5).alias("median"),
    ).collect()[0]
    
    return {
        "n_unique_regions": int(n_unique_regions),
        "n_unique_bricks": int(n_unique_bricks),
        "n_total_objects": int(n_total_objects),
        "by_split": by_split,
        "top_10_regions": top_regions_list,
        "objects_per_brick": {
            "min": int(opb_stats["min"]) if opb_stats["min"] else 0,
            "max": int(opb_stats["max"]) if opb_stats["max"] else 0,
            "mean": round(float(opb_stats["mean"]), 2) if opb_stats["mean"] else 0,
            "median": round(float(opb_stats["median"]), 2) if opb_stats["median"] else 0,
        },
    }


def validate_variant_distribution(df: DataFrame, target_variant: str) -> Dict[str, Any]:
    """Analyze LRG variant distribution."""
    n_total = df.count()
    
    variant_cols = [f"is_{v}" for v in LRG_VARIANTS.keys()]
    existing_cols = [c for c in variant_cols if c in df.columns]
    
    # Count each variant
    count_exprs = [F.sum(F.when(F.col(c) == True, 1).otherwise(0)).alias(c) for c in existing_cols]
    counts_row = df.select(count_exprs).collect()[0]
    
    variant_counts = {c: int(counts_row[c]) for c in existing_cols}
    variant_pcts = {c: round(counts_row[c] / n_total * 100, 2) for c in existing_cols}
    
    # Parent selection check
    target_col = f"is_{target_variant}"
    if target_col in df.columns:
        n_passing = df.filter(F.col(target_col) == True).count()
        parent_selection_check = {
            "target_variant": target_variant,
            "n_total": n_total,
            "n_passing": int(n_passing),
            "pct_passing": round(n_passing / n_total * 100, 2),
            "passed": n_passing == n_total,
        }
    else:
        parent_selection_check = {"error": f"Missing {target_col}"}
    
    return {
        "variant_counts": variant_counts,
        "variant_pcts": variant_pcts,
        "parent_selection_check": parent_selection_check,
    }


def validate_type_filter(df: DataFrame) -> Dict[str, Any]:
    """Verify PSF objects were excluded."""
    if "type" not in df.columns:
        return {"error": "Missing 'type' column"}
    
    type_counts_rows = df.groupBy("type").count().collect()
    type_counts = {row["type"]: int(row["count"]) for row in type_counts_rows}
    
    n_psf = type_counts.get("PSF", 0) + type_counts.get("psf", 0)
    
    return {
        "type_distribution": type_counts,
        "n_psf_found": n_psf,
        "psf_filter_passed": n_psf == 0,
    }


def generate_summary_text(results: Dict) -> str:
    """Generate human-readable summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("PHASE 3C VALIDATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append(f"Input: {results['input']}")
    lines.append("")
    
    # Executive Summary
    lines.append("-" * 70)
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 70)
    
    schema = results["schema"]
    quality = results["quality"]
    lrg_flags = results["lrg_flags"]
    coverage = results["coverage"]
    variant_dist = results["variant_distribution"]
    type_filter = results["type_filter"]
    
    checks = []
    all_passed = True
    
    if schema.get("valid"):
        checks.append("[PASS] Schema validation")
    else:
        checks.append(f"[FAIL] Schema - missing: {schema.get('missing_columns')}")
        all_passed = False
    
    if quality.get("core_columns_null_check", {}).get("passed"):
        checks.append("[PASS] Core columns have no nulls")
    else:
        checks.append("[FAIL] Core columns have nulls")
        all_passed = False
    
    lrg_passed = all(v.get("passed", False) for k, v in lrg_flags.items() if k != "hierarchy_checks")
    if lrg_passed:
        checks.append("[PASS] LRG flag validation (all variants)")
    else:
        failed = [k for k, v in lrg_flags.items() if k != "hierarchy_checks" and not v.get("passed", True)]
        checks.append(f"[FAIL] LRG flags failed for: {failed}")
        all_passed = False
    
    hierarchy_passed = all(h.get("passed", False) for h in lrg_flags.get("hierarchy_checks", []))
    if hierarchy_passed:
        checks.append("[PASS] LRG variant hierarchy consistent")
    else:
        checks.append("[FAIL] LRG variant hierarchy has violations")
        all_passed = False
    
    if variant_dist.get("parent_selection_check", {}).get("passed"):
        checks.append("[PASS] All rows pass parent variant filter")
    else:
        pct = variant_dist.get("parent_selection_check", {}).get("pct_passing", 0)
        checks.append(f"[WARN] Only {pct}% pass parent variant")
    
    if type_filter.get("psf_filter_passed"):
        checks.append("[PASS] PSF (star) filter")
    else:
        n = type_filter.get("n_psf_found", 0)
        checks.append(f"[FAIL] Found {n} PSF objects")
        all_passed = False
    
    for check in checks:
        lines.append(f"  {check}")
    
    lines.append("")
    if all_passed:
        lines.append("OVERALL STATUS: ALL CHECKS PASSED")
    else:
        lines.append("OVERALL STATUS: SOME CHECKS FAILED")
    
    # Data Overview
    lines.append("")
    lines.append("-" * 70)
    lines.append("DATA OVERVIEW")
    lines.append("-" * 70)
    lines.append(f"  Total LRG Objects: {coverage['n_total_objects']:,}")
    lines.append(f"  Unique Regions: {coverage['n_unique_regions']}")
    lines.append(f"  Unique Bricks: {coverage['n_unique_bricks']:,}")
    lines.append("")
    lines.append("  By Split:")
    for split, stats in coverage.get("by_split", {}).items():
        lines.append(f"    {split}: {stats['n_regions']} regions, {stats['n_bricks']:,} bricks, {stats['n_objects']:,} objects")
    
    # LRG Variant Distribution
    lines.append("")
    lines.append("-" * 70)
    lines.append("LRG VARIANT DISTRIBUTION")
    lines.append("-" * 70)
    for col, count in variant_dist.get("variant_counts", {}).items():
        pct = variant_dist.get("variant_pcts", {}).get(col, 0)
        lines.append(f"  {col}: {count:,} ({pct}%)")
    
    # Top Regions
    lines.append("")
    lines.append("-" * 70)
    lines.append("TOP 10 REGIONS BY OBJECT COUNT")
    lines.append("-" * 70)
    for r in coverage.get("top_10_regions", []):
        lines.append(f"  {r['region_split']:5} Region {r['region_id']:5}: {r['n_bricks']:,} bricks, {r['n_objects']:,} objects")
    
    # Type Distribution
    lines.append("")
    lines.append("-" * 70)
    lines.append("TYPE DISTRIBUTION")
    lines.append("-" * 70)
    for typ, count in sorted(type_filter.get("type_distribution", {}).items(), key=lambda x: -x[1]):
        lines.append(f"  {typ}: {count:,}")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 3c output using Spark")
    parser.add_argument("--phase3c-parquet", required=True, help="S3 path to Phase 3c parent catalog parquet")
    parser.add_argument("--output-s3", required=True, help="S3 path for validation outputs")
    parser.add_argument("--variant", default="v3_color_relaxed", help="Target LRG variant for parent selection")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Phase 3c Validation (Spark)")
    print("=" * 60)
    print()
    
    spark = make_spark("DHS Phase3c Validation")
    
    # Load data
    print("[1/6] Loading Parquet data...")
    df = spark.read.parquet(args.phase3c_parquet)
    df.cache()
    n_rows = df.count()
    print(f"      Loaded {n_rows:,} rows")
    
    # Validate schema
    print("[2/6] Validating schema...")
    schema = validate_schema(df)
    print(f"      Schema valid: {schema['valid']}")
    
    # Validate data quality
    print("[3/6] Checking data quality...")
    quality = validate_data_quality(df)
    print(f"      Core nulls check: {quality['core_columns_null_check']['passed']}")
    
    # Validate LRG flags
    print("[4/6] Validating LRG flags...")
    lrg_flags = validate_lrg_flags(df)
    n_passed = sum(1 for k, v in lrg_flags.items() if k != "hierarchy_checks" and v.get("passed"))
    print(f"      {n_passed}/5 variants passed flag validation")
    
    # Validate coverage
    print("[5/6] Analyzing coverage...")
    coverage = validate_coverage(df)
    print(f"      {coverage['n_unique_regions']} regions, {coverage['n_unique_bricks']:,} bricks")
    
    # Additional validations
    print("[6/6] Running additional validations...")
    variant_dist = validate_variant_distribution(df, args.variant)
    type_filter = validate_type_filter(df)
    print(f"      TYPE filter passed: {type_filter.get('psf_filter_passed', False)}")
    
    # Compile results
    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input": args.phase3c_parquet,
        "variant": args.variant,
        "schema": schema,
        "quality": quality,
        "lrg_flags": lrg_flags,
        "coverage": coverage,
        "variant_distribution": variant_dist,
        "type_filter": type_filter,
    }
    
    # Write outputs
    output_prefix = args.output_s3.rstrip("/")
    
    # JSON report
    json_path = f"{output_prefix}/validation_report.json"
    spark.sparkContext.parallelize([json.dumps(results, indent=2, default=str)]).coalesce(1).saveAsTextFile(json_path)
    print(f"JSON report written to: {json_path}")
    
    # Summary text
    summary_text = generate_summary_text(results)
    summary_path = f"{output_prefix}/validation_summary.txt"
    spark.sparkContext.parallelize([summary_text]).coalesce(1).saveAsTextFile(summary_path)
    print(f"Summary written to: {summary_path}")
    
    # Print summary to console
    print()
    print(summary_text)
    
    df.unpersist()
    spark.stop()
    
    print()
    print("Validation complete!")


if __name__ == "__main__":
    main()


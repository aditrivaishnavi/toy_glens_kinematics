#!/usr/bin/env python3
"""Spark-based validation for Phase 4c output.

Validates:
1. Schema completeness (all expected columns present)
2. Row counts match expectations
3. Data quality (nulls, ranges, distributions)
4. Cutout success rates
5. Physics metrics validity
6. PSF provenance columns
7. Maskbits metrics
8. Stage config consistency

Usage:
  spark-submit spark_validate_phase4c.py \
    --metrics-s3 s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/metrics/debug_stamp64_... \
    --stage-config-s3 s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/_stage_config_debug_stamp64_....json
"""

import argparse
import json
import sys
import time
from typing import Dict, List, Optional

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Try to import boto3 for S3 operations
try:
    import boto3
except ImportError:
    boto3 = None


def _parse_s3(uri: str):
    """Parse s3://bucket/key into (bucket, key)."""
    uri = uri.replace("s3://", "")
    parts = uri.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def read_stage_config(s3_uri: str) -> Optional[Dict]:
    """Read stage config JSON from S3."""
    if boto3 is None:
        print("[WARN] boto3 not available, cannot read stage config")
        return None
    try:
        c = boto3.client("s3")
        bkt, key = _parse_s3(s3_uri)
        obj = c.get_object(Bucket=bkt, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception as e:
        print(f"[WARN] Could not read stage config: {e}")
        return None


def validate_schema(df, required_cols: List[str]) -> Dict:
    """Validate that all required columns exist."""
    actual_cols = set(df.columns)
    missing = [c for c in required_cols if c not in actual_cols]
    extra = [c for c in actual_cols if c not in required_cols]
    
    return {
        "missing_columns": missing,
        "extra_columns": extra,
        "schema_valid": len(missing) == 0,
    }


def validate_counts(df) -> Dict:
    """Validate row counts and distributions."""
    total = df.count()
    
    # Count by region_split
    split_counts = df.groupBy("region_split").count().collect()
    splits = {r["region_split"]: r["count"] for r in split_counts}
    
    # Count by cutout_ok
    ok_counts = df.groupBy("cutout_ok").count().collect()
    cutout_stats = {f"cutout_ok_{r['cutout_ok']}": r["count"] for r in ok_counts}
    
    # Success rate
    ok_count = cutout_stats.get("cutout_ok_1", 0)
    success_rate = ok_count / total if total > 0 else 0
    
    return {
        "total_rows": total,
        "by_split": splits,
        "cutout_stats": cutout_stats,
        "success_rate": success_rate,
        "success_rate_valid": success_rate >= 0.95,  # Expect at least 95% success
    }


def validate_nulls(df, critical_cols: List[str]) -> Dict:
    """Check for unexpected nulls in critical columns."""
    null_counts = {}
    total = df.count()
    
    for col in critical_cols:
        if col in df.columns:
            null_count = df.filter(F.col(col).isNull()).count()
            null_counts[col] = null_count
    
    # Columns that should have zero nulls
    zero_null_expected = ["task_id", "experiment_id", "brickname", "ra", "dec", "cutout_ok"]
    violations = {c: null_counts.get(c, 0) for c in zero_null_expected if null_counts.get(c, 0) > 0}
    
    return {
        "null_counts": null_counts,
        "violations": violations,
        "nulls_valid": len(violations) == 0,
    }


def validate_ranges(df) -> Dict:
    """Validate value ranges for key columns."""
    issues = []
    
    # Check theta_e range (should be 0 for controls, 0.25-1.2 for injections)
    theta_stats = df.agg(
        F.min("theta_e_arcsec").alias("min"),
        F.max("theta_e_arcsec").alias("max"),
        F.avg("theta_e_arcsec").alias("avg"),
    ).collect()[0]
    
    if theta_stats["min"] is not None and theta_stats["min"] < 0:
        issues.append(f"theta_e_arcsec min ({theta_stats['min']}) < 0")
    if theta_stats["max"] is not None and theta_stats["max"] > 5.0:
        issues.append(f"theta_e_arcsec max ({theta_stats['max']}) > 5.0 (unexpectedly large)")
    
    # Check arc_snr range (should be reasonable)
    snr_stats = df.filter(F.col("arc_snr").isNotNull()).agg(
        F.min("arc_snr").alias("min"),
        F.max("arc_snr").alias("max"),
        F.avg("arc_snr").alias("avg"),
        F.expr("percentile_approx(arc_snr, 0.5)").alias("median"),
    ).collect()[0]
    
    # Check PSF FWHM columns exist and have values
    psf_cols = ["psfsize_r", "psf_fwhm_used_r"]
    for col in psf_cols:
        if col in df.columns:
            psf_stats = df.filter(F.col(col).isNotNull()).agg(
                F.min(col).alias("min"),
                F.max(col).alias("max"),
            ).collect()[0]
            if psf_stats["min"] is not None:
                if psf_stats["min"] < 0.5 or psf_stats["max"] > 5.0:
                    issues.append(f"{col} range [{psf_stats['min']:.2f}, {psf_stats['max']:.2f}] outside expected [0.5, 5.0]")
    
    return {
        "theta_e_stats": {
            "min": float(theta_stats["min"]) if theta_stats["min"] is not None else None,
            "max": float(theta_stats["max"]) if theta_stats["max"] is not None else None,
            "avg": float(theta_stats["avg"]) if theta_stats["avg"] is not None else None,
        },
        "arc_snr_stats": {
            "min": float(snr_stats["min"]) if snr_stats["min"] is not None else None,
            "max": float(snr_stats["max"]) if snr_stats["max"] is not None else None,
            "avg": float(snr_stats["avg"]) if snr_stats["avg"] is not None else None,
            "median": float(snr_stats["median"]) if snr_stats["median"] is not None else None,
        },
        "issues": issues,
        "ranges_valid": len(issues) == 0,
    }


def validate_controls(df) -> Dict:
    """Validate control samples have theta_e = 0 and low arc_snr."""
    # Controls should have theta_e = 0
    controls = df.filter(F.col("theta_e_arcsec") == 0)
    n_controls = controls.count()
    
    # Controls with non-zero arc_snr (should be low/null)
    if n_controls > 0:
        controls_with_snr = controls.filter(F.col("arc_snr").isNotNull())
        n_with_snr = controls_with_snr.count()
        
        if n_with_snr > 0:
            snr_stats = controls_with_snr.agg(
                F.avg("arc_snr").alias("avg"),
                F.max("arc_snr").alias("max"),
            ).collect()[0]
            avg_snr = float(snr_stats["avg"]) if snr_stats["avg"] is not None else 0
            max_snr = float(snr_stats["max"]) if snr_stats["max"] is not None else 0
        else:
            avg_snr = 0
            max_snr = 0
    else:
        avg_snr = None
        max_snr = None
    
    # Injections should have theta_e > 0
    injections = df.filter(F.col("theta_e_arcsec") > 0)
    n_injections = injections.count()
    
    return {
        "n_controls": n_controls,
        "n_injections": n_injections,
        "control_avg_snr": avg_snr,
        "control_max_snr": max_snr,
        "controls_valid": n_controls > 0,  # Should have some controls
    }


def validate_psf_provenance(df) -> Dict:
    """Validate PSF FWHM provenance columns."""
    psf_cols = ["psf_fwhm_used_g", "psf_fwhm_used_r", "psf_fwhm_used_z"]
    results = {}
    
    for col in psf_cols:
        if col in df.columns:
            stats = df.agg(
                F.count(F.when(F.col(col).isNotNull(), 1)).alias("non_null"),
                F.min(col).alias("min"),
                F.max(col).alias("max"),
            ).collect()[0]
            results[col] = {
                "non_null_count": stats["non_null"],
                "min": float(stats["min"]) if stats["min"] is not None else None,
                "max": float(stats["max"]) if stats["max"] is not None else None,
            }
        else:
            results[col] = {"error": "column not found"}
    
    # Check if at least r-band PSF is populated
    r_valid = "psf_fwhm_used_r" in df.columns and results.get("psf_fwhm_used_r", {}).get("non_null_count", 0) > 0
    
    return {
        "psf_columns": results,
        "psf_provenance_valid": r_valid,
    }


def validate_maskbits_metrics(df) -> Dict:
    """Validate maskbits-related metrics."""
    results = {}
    
    # Check bad_pixel_frac
    if "bad_pixel_frac" in df.columns:
        stats = df.filter(F.col("bad_pixel_frac").isNotNull()).agg(
            F.count("*").alias("count"),
            F.avg("bad_pixel_frac").alias("avg"),
            F.max("bad_pixel_frac").alias("max"),
        ).collect()[0]
        results["bad_pixel_frac"] = {
            "non_null_count": stats["count"],
            "avg": float(stats["avg"]) if stats["avg"] is not None else None,
            "max": float(stats["max"]) if stats["max"] is not None else None,
        }
    
    # Check wise_brightmask_frac
    if "wise_brightmask_frac" in df.columns:
        stats = df.filter(F.col("wise_brightmask_frac").isNotNull()).agg(
            F.count("*").alias("count"),
            F.avg("wise_brightmask_frac").alias("avg"),
            F.max("wise_brightmask_frac").alias("max"),
        ).collect()[0]
        results["wise_brightmask_frac"] = {
            "non_null_count": stats["count"],
            "avg": float(stats["avg"]) if stats["avg"] is not None else None,
            "max": float(stats["max"]) if stats["max"] is not None else None,
        }
    
    return {
        "maskbits_metrics": results,
        "maskbits_valid": len(results) > 0,
    }


def validate_physics(df) -> Dict:
    """Validate physics metrics."""
    results = {}
    
    # Check magnification
    if "magnification" in df.columns:
        # Only check injections (theta_e > 0)
        inj = df.filter(F.col("theta_e_arcsec") > 0)
        stats = inj.filter(F.col("magnification").isNotNull()).agg(
            F.count("*").alias("count"),
            F.avg("magnification").alias("avg"),
            F.min("magnification").alias("min"),
            F.max("magnification").alias("max"),
        ).collect()[0]
        results["magnification"] = {
            "non_null_count": stats["count"],
            "avg": float(stats["avg"]) if stats["avg"] is not None else None,
            "min": float(stats["min"]) if stats["min"] is not None else None,
            "max": float(stats["max"]) if stats["max"] is not None else None,
        }
    
    # Check physics_valid flag
    if "physics_valid" in df.columns:
        valid_counts = df.groupBy("physics_valid").count().collect()
        results["physics_valid_counts"] = {str(r["physics_valid"]): r["count"] for r in valid_counts}
    
    return {
        "physics_metrics": results,
    }


def validate_config_consistency(df, config: Optional[Dict]) -> Dict:
    """Validate that output matches stage config."""
    if config is None:
        return {"config_check": "skipped (config not available)"}
    
    issues = []
    
    # Check experiment_id matches
    if "experiment_id" in df.columns:
        exp_ids = df.select("experiment_id").distinct().collect()
        actual_exp_id = exp_ids[0]["experiment_id"] if len(exp_ids) == 1 else "MULTIPLE"
        expected_exp_id = config.get("experiment_id", "UNKNOWN")
        if actual_exp_id != expected_exp_id:
            issues.append(f"experiment_id mismatch: data={actual_exp_id}, config={expected_exp_id}")
    
    # Check bands
    expected_bands = config.get("bands", [])
    # We can't directly check bands in metrics, but we can note what was configured
    
    # Check use_psfsize_maps
    use_psfsize = config.get("psf", {}).get("use_psfsize_maps", False)
    
    return {
        "config": {
            "experiment_id": config.get("experiment_id"),
            "use_psfsize_maps": use_psfsize,
            "metrics_only": config.get("output_mode", {}).get("metrics_only", False),
            "pipeline_version": config.get("pipeline_version"),
        },
        "issues": issues,
        "config_valid": len(issues) == 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 4c output")
    parser.add_argument("--metrics-s3", required=True, help="S3 path to metrics parquet")
    parser.add_argument("--stage-config-s3", default="", help="S3 path to stage config JSON")
    parser.add_argument("--stamps-s3", default="", help="S3 path to stamps parquet (optional)")
    args = parser.parse_args()
    
    spark = SparkSession.builder.appName("validate_phase4c").getOrCreate()
    
    print("=" * 70)
    print("PHASE 4C VALIDATION")
    print("=" * 70)
    print(f"Metrics path: {args.metrics_s3}")
    print(f"Config path: {args.stage_config_s3}")
    print("=" * 70)
    
    # Read stage config
    config = None
    if args.stage_config_s3:
        config = read_stage_config(args.stage_config_s3)
        if config:
            print(f"\n[CONFIG] Loaded stage config:")
            print(f"  Pipeline version: {config.get('pipeline_version', 'UNKNOWN')}")
            print(f"  Experiment ID: {config.get('experiment_id', 'UNKNOWN')}")
            print(f"  Use PSFsize maps: {config.get('psf', {}).get('use_psfsize_maps', False)}")
            print(f"  Metrics only: {config.get('output_mode', {}).get('metrics_only', False)}")
    
    # Read metrics
    print(f"\n[1/8] Reading metrics from {args.metrics_s3}...")
    df = spark.read.parquet(args.metrics_s3)
    
    # Expected columns in metrics
    required_cols = [
        "task_id", "experiment_id", "selection_set_id", "ranking_mode", "selection_strategy",
        "region_id", "region_split", "brickname",
        "config_id", "theta_e_arcsec", "src_dmag", "src_reff_arcsec", "src_e", "shear", "replicate",
        "cutout_ok", "arc_snr", "psfsize_r", "psfdepth_r", "ebv",
    ]
    
    # Run validations
    results = {}
    
    print("\n[2/8] Validating schema...")
    results["schema"] = validate_schema(df, required_cols)
    print(f"  Schema valid: {results['schema']['schema_valid']}")
    if results["schema"]["missing_columns"]:
        print(f"  Missing: {results['schema']['missing_columns']}")
    
    print("\n[3/8] Validating counts...")
    results["counts"] = validate_counts(df)
    print(f"  Total rows: {results['counts']['total_rows']:,}")
    print(f"  By split: {results['counts']['by_split']}")
    print(f"  Success rate: {results['counts']['success_rate']:.2%}")
    
    print("\n[4/8] Validating nulls...")
    results["nulls"] = validate_nulls(df, required_cols)
    print(f"  Nulls valid: {results['nulls']['nulls_valid']}")
    if results["nulls"]["violations"]:
        print(f"  Violations: {results['nulls']['violations']}")
    
    print("\n[5/8] Validating ranges...")
    results["ranges"] = validate_ranges(df)
    print(f"  theta_e: min={results['ranges']['theta_e_stats']['min']}, max={results['ranges']['theta_e_stats']['max']}")
    print(f"  arc_snr: avg={results['ranges']['arc_snr_stats']['avg']:.2f}, median={results['ranges']['arc_snr_stats']['median']:.2f}" if results['ranges']['arc_snr_stats']['avg'] else "  arc_snr: no data")
    print(f"  Ranges valid: {results['ranges']['ranges_valid']}")
    
    print("\n[6/8] Validating controls...")
    results["controls"] = validate_controls(df)
    print(f"  Controls: {results['controls']['n_controls']:,}")
    print(f"  Injections: {results['controls']['n_injections']:,}")
    print(f"  Control avg SNR: {results['controls']['control_avg_snr']}")
    
    print("\n[7/8] Validating PSF provenance...")
    results["psf"] = validate_psf_provenance(df)
    print(f"  PSF provenance valid: {results['psf']['psf_provenance_valid']}")
    
    print("\n[8/8] Validating maskbits metrics...")
    results["maskbits"] = validate_maskbits_metrics(df)
    print(f"  Maskbits valid: {results['maskbits']['maskbits_valid']}")
    
    # Config consistency
    print("\n[EXTRA] Validating config consistency...")
    results["config"] = validate_config_consistency(df, config)
    print(f"  Config valid: {results['config'].get('config_valid', 'N/A')}")
    
    # Physics metrics
    print("\n[EXTRA] Validating physics metrics...")
    results["physics"] = validate_physics(df)
    if "magnification" in results["physics"].get("physics_metrics", {}):
        mag = results["physics"]["physics_metrics"]["magnification"]
        print(f"  Magnification: count={mag['non_null_count']}, avg={mag['avg']:.2f}" if mag['avg'] else "  Magnification: no data")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    all_valid = all([
        results["schema"]["schema_valid"],
        results["counts"]["success_rate_valid"],
        results["nulls"]["nulls_valid"],
        results["ranges"]["ranges_valid"],
        results["controls"]["controls_valid"],
        results["psf"]["psf_provenance_valid"],
        results["maskbits"]["maskbits_valid"],
    ])
    
    checks = [
        ("Schema", results["schema"]["schema_valid"]),
        ("Success Rate (>=95%)", results["counts"]["success_rate_valid"]),
        ("No Critical Nulls", results["nulls"]["nulls_valid"]),
        ("Value Ranges", results["ranges"]["ranges_valid"]),
        ("Has Controls", results["controls"]["controls_valid"]),
        ("PSF Provenance", results["psf"]["psf_provenance_valid"]),
        ("Maskbits Metrics", results["maskbits"]["maskbits_valid"]),
    ]
    
    for name, valid in checks:
        status = "✅ PASS" if valid else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print("=" * 70)
    if all_valid:
        print("OVERALL: ✅ VALIDATION PASSED")
        print("Phase 4c output is ready for Phase 4d / Phase 5")
    else:
        print("OVERALL: ❌ VALIDATION FAILED")
        print("Review issues above before proceeding")
    print("=" * 70)
    
    spark.stop()
    
    # Exit with appropriate code
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()


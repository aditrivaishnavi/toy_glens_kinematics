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
    fail_count = cutout_stats.get("cutout_ok_0", 0)
    success_rate = ok_count / total if total > 0 else 0
    
    # Analyze failure reasons if there are failures
    failure_reasons = {}
    if fail_count > 0 and "physics_warnings" in df.columns:
        # Get top 10 error messages from failed tasks
        failed = df.filter(F.col("cutout_ok") == 0).select("physics_warnings", "brickname")
        # Extract first 50 chars of error for grouping
        failed_grouped = failed.withColumn(
            "error_prefix", F.substring(F.col("physics_warnings"), 1, 80)
        ).groupBy("error_prefix").count().orderBy(F.desc("count")).limit(10).collect()
        failure_reasons = {r["error_prefix"]: r["count"] for r in failed_grouped if r["error_prefix"]}
    
    return {
        "total_rows": total,
        "by_split": splits,
        "cutout_stats": cutout_stats,
        "success_rate": success_rate,
        "success_rate_valid": success_rate >= 0.95,  # Expect at least 95% success
        "failure_reasons": failure_reasons,
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


def validate_controls(df, experiment_id: str = "") -> Dict:
    """Validate control samples have theta_e = 0 and low arc_snr."""
    # Controls should have theta_e = 0 (or is_control = 1 if column exists)
    if "is_control" in df.columns:
        controls = df.filter(F.col("is_control") == 1)
    else:
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
    
    # Debug tier is allowed to have 0 controls (control_frac_debug defaults to 0)
    is_debug = experiment_id.startswith("debug")
    controls_valid = n_controls > 0 or is_debug
    
    return {
        "n_controls": n_controls,
        "n_injections": n_injections,
        "control_avg_snr": avg_snr,
        "control_max_snr": max_snr,
        "controls_valid": controls_valid,
        "is_debug_tier": is_debug,
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
    
    # Only check injections (theta_e > 0)
    inj = df.filter(F.col("theta_e_arcsec") > 0)
    n_injections = inj.count()
    
    # Check magnification
    if "magnification" in df.columns:
        stats = inj.filter(F.col("magnification").isNotNull()).agg(
            F.count("*").alias("count"),
            F.avg("magnification").alias("avg"),
            F.min("magnification").alias("min"),
            F.max("magnification").alias("max"),
        ).collect()[0]
        results["magnification"] = {
            "non_null_count": stats["count"],
            "expected_non_null": n_injections,
            "coverage_pct": (stats["count"] / n_injections * 100) if n_injections > 0 else 0,
            "avg": float(stats["avg"]) if stats["avg"] is not None else None,
            "min": float(stats["min"]) if stats["min"] is not None else None,
            "max": float(stats["max"]) if stats["max"] is not None else None,
        }
    
    # Check physics_valid flag
    if "physics_valid" in df.columns:
        valid_counts = df.groupBy("physics_valid").count().collect()
        results["physics_valid_counts"] = {str(r["physics_valid"]): r["count"] for r in valid_counts}
    
    # Check expected_arc_radius
    if "expected_arc_radius" in df.columns:
        stats = inj.filter(F.col("expected_arc_radius").isNotNull()).agg(
            F.count("*").alias("count"),
            F.avg("expected_arc_radius").alias("avg"),
        ).collect()[0]
        results["expected_arc_radius"] = {
            "non_null_count": stats["count"],
            "avg": float(stats["avg"]) if stats["avg"] is not None else None,
        }
    
    return {
        "physics_metrics": results,
        "n_injections": n_injections,
    }


def validate_injection_params(df) -> Dict:
    """Validate injection parameter distributions."""
    inj = df.filter(F.col("theta_e_arcsec") > 0)
    
    # theta_e distribution
    theta_stats = inj.agg(
        F.min("theta_e_arcsec").alias("min"),
        F.max("theta_e_arcsec").alias("max"),
        F.avg("theta_e_arcsec").alias("avg"),
        F.expr("percentile_approx(theta_e_arcsec, array(0.25, 0.5, 0.75))").alias("quartiles"),
    ).collect()[0]
    
    # src_dmag distribution
    dmag_stats = inj.agg(
        F.min("src_dmag").alias("min"),
        F.max("src_dmag").alias("max"),
        F.avg("src_dmag").alias("avg"),
    ).collect()[0]
    
    # src_reff_arcsec distribution
    reff_stats = inj.agg(
        F.min("src_reff_arcsec").alias("min"),
        F.max("src_reff_arcsec").alias("max"),
        F.avg("src_reff_arcsec").alias("avg"),
    ).collect()[0]
    
    # src_e distribution
    e_stats = inj.agg(
        F.min("src_e").alias("min"),
        F.max("src_e").alias("max"),
        F.avg("src_e").alias("avg"),
    ).collect()[0]
    
    # shear distribution
    shear_stats = inj.agg(
        F.min("shear").alias("min"),
        F.max("shear").alias("max"),
        F.avg("shear").alias("avg"),
    ).collect()[0]
    
    return {
        "theta_e": {
            "min": float(theta_stats["min"]) if theta_stats["min"] is not None else None,
            "max": float(theta_stats["max"]) if theta_stats["max"] is not None else None,
            "avg": float(theta_stats["avg"]) if theta_stats["avg"] is not None else None,
            "quartiles": [float(q) for q in theta_stats["quartiles"]] if theta_stats["quartiles"] else None,
        },
        "src_dmag": {
            "min": float(dmag_stats["min"]) if dmag_stats["min"] is not None else None,
            "max": float(dmag_stats["max"]) if dmag_stats["max"] is not None else None,
            "avg": float(dmag_stats["avg"]) if dmag_stats["avg"] is not None else None,
        },
        "src_reff_arcsec": {
            "min": float(reff_stats["min"]) if reff_stats["min"] is not None else None,
            "max": float(reff_stats["max"]) if reff_stats["max"] is not None else None,
            "avg": float(reff_stats["avg"]) if reff_stats["avg"] is not None else None,
        },
        "src_e": {
            "min": float(e_stats["min"]) if e_stats["min"] is not None else None,
            "max": float(e_stats["max"]) if e_stats["max"] is not None else None,
            "avg": float(e_stats["avg"]) if e_stats["avg"] is not None else None,
        },
        "shear": {
            "min": float(shear_stats["min"]) if shear_stats["min"] is not None else None,
            "max": float(shear_stats["max"]) if shear_stats["max"] is not None else None,
            "avg": float(shear_stats["avg"]) if shear_stats["avg"] is not None else None,
        },
    }


def validate_lens_model_distribution(df) -> Dict:
    """Validate lens model distribution."""
    if "lens_model" not in df.columns:
        return {"error": "lens_model column not found"}
    
    dist = df.groupBy("lens_model").count().collect()
    model_counts = {r["lens_model"] if r["lens_model"] else "NULL": r["count"] for r in dist}
    
    return {
        "lens_model_counts": model_counts,
        "has_sie": model_counts.get("SIE", 0) > 0,
        "has_control": model_counts.get("CONTROL", 0) > 0 or model_counts.get("NULL", 0) > 0,
    }


def validate_snr_correlation(df) -> Dict:
    """Validate SNR behavior with theta_e for injections."""
    inj = df.filter((F.col("theta_e_arcsec") > 0) & (F.col("arc_snr").isNotNull()))
    
    if inj.count() == 0:
        return {"error": "No valid injections with SNR"}
    
    # Bin by theta_e and compute average SNR per bin
    binned = inj.withColumn(
        "theta_bin", F.floor(F.col("theta_e_arcsec") * 5) / 5  # 0.2 arcsec bins
    ).groupBy("theta_bin").agg(
        F.avg("arc_snr").alias("avg_snr"),
        F.count("*").alias("count"),
    ).orderBy("theta_bin").collect()
    
    bins = [(float(r["theta_bin"]), float(r["avg_snr"]), r["count"]) for r in binned]
    
    # Peak SNR may decrease with theta_e due to arc spreading (physically expected)
    # This is NOT a bug - larger arcs spread flux over more pixels
    if len(bins) >= 2:
        first_half_avg = sum(b[1] for b in bins[:len(bins)//2]) / max(len(bins)//2, 1)
        second_half_avg = sum(b[1] for b in bins[len(bins)//2:]) / max(len(bins) - len(bins)//2, 1)
        peak_snr_trend = "increasing" if second_half_avg > first_half_avg else "decreasing"
    else:
        peak_snr_trend = "unknown"
    
    return {
        "bins": bins,
        "n_bins": len(bins),
        "peak_snr_trend": peak_snr_trend,
        "note": "Peak SNR may decrease with theta_e due to arc spreading - this is expected physics"
    }


def validate_total_flux_correlation(df) -> Dict:
    """Validate TOTAL injected flux increases with theta_e (critical physics check)."""
    # This is the key physics test: magnification should increase with theta_e
    # for sources placed proportionally inside the Einstein radius
    
    if "total_injected_flux_r" not in df.columns:
        return {"error": "total_injected_flux_r column not found - run Stage 4c with updated code"}
    
    inj = df.filter((F.col("theta_e_arcsec") > 0) & (F.col("total_injected_flux_r").isNotNull()))
    
    n_with_flux = inj.count()
    if n_with_flux == 0:
        return {"error": "No injections have total_injected_flux_r - likely bug in Stage 4c"}
    
    # Bin by theta_e and compute average total flux per bin
    binned = inj.withColumn(
        "theta_bin", F.floor(F.col("theta_e_arcsec") * 5) / 5
    ).groupBy("theta_bin").agg(
        F.avg("total_injected_flux_r").alias("avg_flux"),
        F.count("*").alias("count"),
    ).orderBy("theta_bin").collect()
    
    bins = [(float(r["theta_bin"]), float(r["avg_flux"]), r["count"]) for r in binned]
    
    # CRITICAL: Total flux MUST increase with theta_e (magnification physics)
    if len(bins) >= 2:
        first_half_avg = sum(b[1] for b in bins[:len(bins)//2]) / max(len(bins)//2, 1)
        second_half_avg = sum(b[1] for b in bins[len(bins)//2:]) / max(len(bins) - len(bins)//2, 1)
        flux_trend_positive = second_half_avg > first_half_avg
    else:
        flux_trend_positive = None
    
    return {
        "n_with_flux": n_with_flux,
        "bins": bins,
        "flux_trend_positive": flux_trend_positive,
        "valid": flux_trend_positive == True,
        "interpretation": "✅ Total flux INCREASES with theta_e (physics correct)" if flux_trend_positive else "❌ Total flux does NOT increase with theta_e - MAGNIFICATION BUG!"
    }


def validate_control_injection_balance(df) -> Dict:
    """Validate control vs injection balance matches configuration."""
    if "is_control" not in df.columns:
        return {"error": "is_control column not found"}
    
    total = df.count()
    n_controls = df.filter(F.col("is_control") == 1).count()
    n_injections = df.filter(F.col("is_control") == 0).count()
    
    actual_frac = n_controls / total if total > 0 else 0
    
    # Per-split breakdown
    by_split = df.groupBy("region_split", "is_control").count().collect()
    split_breakdown = {}
    for r in by_split:
        split = r["region_split"]
        if split not in split_breakdown:
            split_breakdown[split] = {"controls": 0, "injections": 0}
        if r["is_control"] == 1:
            split_breakdown[split]["controls"] = r["count"]
        else:
            split_breakdown[split]["injections"] = r["count"]
    
    # Calculate per-split fractions
    for split in split_breakdown:
        total_split = split_breakdown[split]["controls"] + split_breakdown[split]["injections"]
        split_breakdown[split]["control_frac"] = split_breakdown[split]["controls"] / total_split if total_split > 0 else 0
    
    return {
        "total_controls": n_controls,
        "total_injections": n_injections,
        "overall_control_frac": actual_frac,
        "by_split": split_breakdown,
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
    
    # Expected columns in metrics = ALL stamps columns EXCEPT stamp_npz
    required_cols = [
        # Identifiers
        "task_id", "experiment_id", "selection_set_id", "ranking_mode", "selection_strategy",
        "region_id", "region_split", "brickname",
        # Galaxy coordinates
        "ra", "dec",
        # Stamp configuration
        "stamp_size", "bandset",
        # Injection parameters
        "config_id", "theta_e_arcsec", "src_dmag", "src_reff_arcsec", "src_e", "shear", "replicate",
        "is_control",
        # Frozen randomness
        "task_seed64", "src_x_arcsec", "src_y_arcsec", "src_phi_rad", "shear_phi_rad", "src_gr", "src_rz",
        # Binning
        "psf_bin", "depth_bin",
        # Provenance
        "ab_zp_nmgy", "pipeline_version",
        # Observing conditions
        "psfsize_r", "psfdepth_r", "ebv",
        # Quality metrics
        "cutout_ok", "arc_snr", "total_injected_flux_r", "metrics_ok",
        # Maskbits metrics
        "bad_pixel_frac", "wise_brightmask_frac",
        # PSF provenance
        "psf_fwhm_used_g", "psf_fwhm_used_r", "psf_fwhm_used_z",
        # Mode tracking
        "metrics_only",
        # Lens model provenance
        "lens_model", "lens_e", "lens_phi_rad",
        # Physics metrics
        "magnification", "tangential_stretch", "radial_stretch", "expected_arc_radius",
        "physics_valid", "physics_warnings",
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
    print(f"  Cutout stats: {results['counts']['cutout_stats']}")
    print(f"  Success rate: {results['counts']['success_rate']:.2%}")
    if results['counts'].get('failure_reasons'):
        print(f"  Top failure reasons:")
        for reason, count in list(results['counts']['failure_reasons'].items())[:5]:
            print(f"    - {reason}: {count}")
    
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
    experiment_id = config.get("experiment_id", "") if config else ""
    results["controls"] = validate_controls(df, experiment_id)
    print(f"  Controls: {results['controls']['n_controls']:,}")
    print(f"  Injections: {results['controls']['n_injections']:,}")
    print(f"  Control avg SNR: {results['controls']['control_avg_snr']}")
    if results['controls'].get('is_debug_tier'):
        print(f"  (Debug tier - 0 controls is expected)")
    
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
    print("\n[EXTRA 1/5] Validating physics metrics...")
    results["physics"] = validate_physics(df)
    if "magnification" in results["physics"].get("physics_metrics", {}):
        mag = results["physics"]["physics_metrics"]["magnification"]
        coverage = mag.get('coverage_pct', 0)
        print(f"  Magnification: count={mag['non_null_count']}, coverage={coverage:.1f}%")
        if mag['avg']:
            print(f"    avg={mag['avg']:.2f}, min={mag['min']:.2f}, max={mag['max']:.2f}")
        else:
            print(f"    ⚠️ NO MAGNIFICATION DATA - likely bug in Stage 4c!")
    
    # Injection parameter distributions
    print("\n[EXTRA 2/5] Validating injection parameter distributions...")
    results["injection_params"] = validate_injection_params(df)
    inj_p = results["injection_params"]
    print(f"  theta_e: min={inj_p['theta_e']['min']:.2f}, max={inj_p['theta_e']['max']:.2f}, avg={inj_p['theta_e']['avg']:.2f}")
    print(f"  src_dmag: min={inj_p['src_dmag']['min']:.1f}, max={inj_p['src_dmag']['max']:.1f}, avg={inj_p['src_dmag']['avg']:.1f}")
    print(f"  src_reff: min={inj_p['src_reff_arcsec']['min']:.2f}, max={inj_p['src_reff_arcsec']['max']:.2f}")
    print(f"  src_e: min={inj_p['src_e']['min']:.2f}, max={inj_p['src_e']['max']:.2f}")
    print(f"  shear: min={inj_p['shear']['min']:.3f}, max={inj_p['shear']['max']:.3f}")
    
    # Lens model distribution
    print("\n[EXTRA 3/5] Validating lens model distribution...")
    results["lens_model"] = validate_lens_model_distribution(df)
    if "lens_model_counts" in results["lens_model"]:
        print(f"  Model distribution: {results['lens_model']['lens_model_counts']}")
        print(f"  Has SIE: {results['lens_model'].get('has_sie', False)}")
    
    # SNR correlation with theta_e (peak SNR may decrease - expected physics)
    print("\n[EXTRA 4/6] Validating SNR-theta_e correlation...")
    results["snr_correlation"] = validate_snr_correlation(df)
    print(f"  Peak SNR trend: {results['snr_correlation'].get('peak_snr_trend', 'unknown')}")
    print(f"  Note: {results['snr_correlation'].get('note', '')}")
    if results["snr_correlation"].get("bins"):
        print(f"  Binned peak SNR:")
        for bin_val, avg_snr, count in results["snr_correlation"]["bins"][:5]:
            print(f"    theta={bin_val:.2f}: SNR={avg_snr:.1f} (n={count})")
    
    # CRITICAL: Total flux correlation with theta_e (must be positive)
    print("\n[EXTRA 5/6] Validating TOTAL FLUX-theta_e correlation (CRITICAL)...")
    results["total_flux_correlation"] = validate_total_flux_correlation(df)
    if "error" in results["total_flux_correlation"]:
        print(f"  ❌ {results['total_flux_correlation']['error']}")
    else:
        print(f"  {results['total_flux_correlation']['interpretation']}")
        if results["total_flux_correlation"].get("bins"):
            print(f"  Binned total flux:")
            for bin_val, avg_flux, count in results["total_flux_correlation"]["bins"][:5]:
                print(f"    theta={bin_val:.2f}: flux={avg_flux:.1f} nMgy (n={count})")
    
    # Control/injection balance
    print("\n[EXTRA 6/6] Validating control/injection balance...")
    results["balance"] = validate_control_injection_balance(df)
    print(f"  Controls: {results['balance']['total_controls']:,}")
    print(f"  Injections: {results['balance']['total_injections']:,}")
    print(f"  Overall control fraction: {results['balance']['overall_control_frac']:.2%}")
    print(f"  Per-split breakdown:")
    for split, data in results["balance"]["by_split"].items():
        print(f"    {split}: controls={data['controls']:,}, injections={data['injections']:,}, frac={data['control_frac']:.2%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    # Check magnification coverage (should be >0% for injections)
    mag_metrics = results["physics"].get("physics_metrics", {}).get("magnification", {})
    magnification_valid = mag_metrics.get("coverage_pct", 0) > 0
    
    # Check total flux trend (MUST be positive - critical physics check)
    total_flux_valid = results.get("total_flux_correlation", {}).get("valid", False)
    
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
        ("Magnification Data", magnification_valid),
        ("Total Flux ↑ with θ_E (CRITICAL)", total_flux_valid),
    ]
    
    for name, valid in checks:
        status = "✅ PASS" if valid else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print("=" * 70)
    
    # Count failures
    n_failures = sum(1 for _, v in checks if not v)
    
    # Critical physics checks
    physics_ok = magnification_valid and total_flux_valid
    
    if n_failures == 0:
        print("OVERALL: ✅ VALIDATION PASSED")
        print("Phase 4c output is ready for Phase 4d / Phase 5")
    elif not all_valid or not physics_ok:
        print("OVERALL: ❌ VALIDATION FAILED")
        if not magnification_valid:
            print("  - MAGNIFICATION DATA MISSING: Run 4c with fixed code")
        if not total_flux_valid:
            print("  - TOTAL FLUX DOES NOT INCREASE WITH θ_E: Injection physics broken!")
        print("DO NOT USE THIS DATA FOR TRAINING")
    else:
        print(f"OVERALL: ⚠️ VALIDATION PASSED WITH WARNINGS ({n_failures} checks failed)")
        print("Phase 4c output may proceed but investigate warnings")
    print("=" * 70)
    
    spark.stop()
    
    # Exit with appropriate code
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()


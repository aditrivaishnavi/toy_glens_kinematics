#!/usr/bin/env python3
"""Comprehensive Spark-based validation for Phase 4c train tier output.

Produces a complete validation report suitable for LLM review.

Usage:
  spark-submit spark_validate_phase4c_comprehensive.py \
    --metrics-s3 s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/metrics/train_stamp64_... \
    --output-s3 s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/validation/train_stamp64_...
"""

import argparse
import json
import sys
import time
from typing import Dict

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

try:
    import boto3
except ImportError:
    boto3 = None


def write_to_s3(uri: str, content: str):
    """Write text content to S3."""
    if boto3 is None:
        print(f"[WARN] boto3 not available, cannot write to {uri}")
        return
    uri = uri.replace("s3://", "")
    parts = uri.split("/", 1)
    bucket, key = parts[0], parts[1] if len(parts) > 1 else ""
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=content.encode("utf-8"))
    print(f"[INFO] Wrote to s3://{bucket}/{key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-s3", required=True, help="S3 path to Phase 4c metrics")
    parser.add_argument("--output-s3", required=True, help="S3 path to write validation report")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("phase4c_comprehensive_validation").getOrCreate()
    
    report = []
    report.append("=" * 80)
    report.append("PHASE 4C TRAIN TIER - COMPREHENSIVE VALIDATION REPORT")
    report.append("=" * 80)
    report.append(f"Metrics path: {args.metrics_s3}")
    report.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    report.append("")

    # Read metrics
    print(f"[INFO] Reading metrics from {args.metrics_s3}")
    df = spark.read.parquet(args.metrics_s3)
    df.cache()
    
    total_rows = df.count()
    num_columns = len(df.columns)
    
    report.append("=" * 80)
    report.append("1. DATASET OVERVIEW")
    report.append("=" * 80)
    report.append(f"Total rows: {total_rows:,}")
    report.append(f"Columns: {num_columns}")
    report.append(f"Column list: {', '.join(sorted(df.columns))}")
    report.append("")

    # =========================================================================
    # 2. SUCCESS RATES
    # =========================================================================
    report.append("=" * 80)
    report.append("2. SUCCESS RATES")
    report.append("=" * 80)
    
    cutout_stats = df.groupBy("cutout_ok").count().collect()
    cutout_dict = {r["cutout_ok"]: r["count"] for r in cutout_stats}
    ok_count = cutout_dict.get(1, 0)
    fail_count = cutout_dict.get(0, 0)
    success_rate = 100 * ok_count / total_rows if total_rows > 0 else 0
    
    report.append(f"cutout_ok=1 (success): {ok_count:,} ({success_rate:.2f}%)")
    report.append(f"cutout_ok=0 (failure): {fail_count:,} ({100-success_rate:.2f}%)")
    report.append(f"Success rate: {success_rate:.2f}%")
    report.append(f"PASS: {'YES' if success_rate >= 95 else 'NO'}")
    report.append("")

    # =========================================================================
    # 3. SPLIT DISTRIBUTION
    # =========================================================================
    report.append("=" * 80)
    report.append("3. SPLIT DISTRIBUTION")
    report.append("=" * 80)
    
    split_stats = df.groupBy("region_split").count().orderBy("region_split").collect()
    for r in split_stats:
        pct = 100 * r["count"] / total_rows
        report.append(f"  {r['region_split']}: {r['count']:,} ({pct:.1f}%)")
    report.append("")

    # =========================================================================
    # 4. CONTROL VS INJECTION SPLIT
    # =========================================================================
    report.append("=" * 80)
    report.append("4. CONTROL VS INJECTION SPLIT")
    report.append("=" * 80)
    
    model_stats = df.groupBy("lens_model").count().orderBy("lens_model").collect()
    model_dict = {r["lens_model"]: r["count"] for r in model_stats}
    
    control_count = model_dict.get("CONTROL", 0)
    injection_count = sum(v for k, v in model_dict.items() if k != "CONTROL")
    control_pct = 100 * control_count / total_rows if total_rows > 0 else 0
    
    report.append(f"Controls (CONTROL): {control_count:,} ({control_pct:.1f}%)")
    report.append(f"Injections: {injection_count:,} ({100-control_pct:.1f}%)")
    report.append(f"Lens model breakdown:")
    for model, cnt in sorted(model_dict.items()):
        report.append(f"  {model}: {cnt:,}")
    report.append(f"Control fraction ~50%: {'YES' if 45 <= control_pct <= 55 else 'NO'}")
    report.append("")

    # =========================================================================
    # 5. CONTROL VALIDATION
    # =========================================================================
    report.append("=" * 80)
    report.append("5. CONTROL VALIDATION")
    report.append("=" * 80)
    
    controls = df.filter(F.col("lens_model") == "CONTROL")
    controls.cache()
    n_controls = controls.count()
    
    # theta_e should be 0
    theta_zero = controls.filter(F.col("theta_e_arcsec") == 0).count()
    report.append(f"theta_e=0 for all controls: {theta_zero}/{n_controls} ({'PASS' if theta_zero == n_controls else 'FAIL'})")
    
    # arc_snr should be NULL
    arc_null = controls.filter(F.col("arc_snr").isNull()).count()
    report.append(f"arc_snr NULL for all controls: {arc_null}/{n_controls} ({'PASS' if arc_null == n_controls else 'FAIL'})")
    
    # magnification should be NULL
    mag_null = controls.filter(F.col("magnification").isNull()).count()
    report.append(f"magnification NULL for all controls: {mag_null}/{n_controls} ({'PASS' if mag_null == n_controls else 'FAIL'})")
    
    # total_injected_flux_r should be NULL
    flux_null = controls.filter(F.col("total_injected_flux_r").isNull()).count()
    report.append(f"total_injected_flux_r NULL for all controls: {flux_null}/{n_controls} ({'PASS' if flux_null == n_controls else 'FAIL'})")
    
    # cutout_ok success rate
    ctrl_ok = controls.filter(F.col("cutout_ok") == 1).count()
    ctrl_success = 100 * ctrl_ok / n_controls if n_controls > 0 else 0
    report.append(f"Control cutout success: {ctrl_ok}/{n_controls} ({ctrl_success:.2f}%)")
    
    controls.unpersist()
    report.append("")

    # =========================================================================
    # 6. INJECTION VALIDATION
    # =========================================================================
    report.append("=" * 80)
    report.append("6. INJECTION VALIDATION")
    report.append("=" * 80)
    
    injections = df.filter(F.col("lens_model") != "CONTROL")
    injections.cache()
    n_inj = injections.count()
    
    # arc_snr coverage
    arc_cov = injections.filter(F.col("arc_snr").isNotNull()).count()
    arc_pct = 100 * arc_cov / n_inj if n_inj > 0 else 0
    report.append(f"arc_snr coverage: {arc_cov}/{n_inj} ({arc_pct:.1f}%)")
    
    # magnification coverage
    mag_cov = injections.filter(F.col("magnification").isNotNull()).count()
    mag_pct = 100 * mag_cov / n_inj if n_inj > 0 else 0
    report.append(f"magnification coverage: {mag_cov}/{n_inj} ({mag_pct:.1f}%)")
    
    # total_injected_flux_r coverage
    flux_cov = injections.filter(F.col("total_injected_flux_r").isNotNull()).count()
    flux_pct = 100 * flux_cov / n_inj if n_inj > 0 else 0
    report.append(f"total_injected_flux_r coverage: {flux_cov}/{n_inj} ({flux_pct:.1f}%)")
    
    # cutout_ok success rate
    inj_ok = injections.filter(F.col("cutout_ok") == 1).count()
    inj_success = 100 * inj_ok / n_inj if n_inj > 0 else 0
    report.append(f"Injection cutout success: {inj_ok}/{n_inj} ({inj_success:.2f}%)")
    report.append("")

    # =========================================================================
    # 7. INJECTION PARAMETER DISTRIBUTIONS
    # =========================================================================
    report.append("=" * 80)
    report.append("7. INJECTION PARAMETER DISTRIBUTIONS")
    report.append("=" * 80)
    
    param_cols = ["theta_e_arcsec", "src_dmag", "src_reff_arcsec", "src_e", "shear"]
    for col in param_cols:
        if col in injections.columns:
            stats = injections.agg(
                F.min(col).alias("min"),
                F.max(col).alias("max"),
                F.avg(col).alias("avg"),
                F.expr(f"percentile_approx({col}, 0.5)").alias("median")
            ).collect()[0]
            report.append(f"{col}: min={stats['min']:.4f}, max={stats['max']:.4f}, avg={stats['avg']:.4f}, median={stats['median']:.4f}")
    report.append("")

    # =========================================================================
    # 8. PHYSICS METRICS
    # =========================================================================
    report.append("=" * 80)
    report.append("8. PHYSICS METRICS (Injections only)")
    report.append("=" * 80)
    
    # arc_snr statistics
    arc_stats = injections.filter(F.col("arc_snr").isNotNull()).agg(
        F.min("arc_snr").alias("min"),
        F.max("arc_snr").alias("max"),
        F.avg("arc_snr").alias("avg"),
        F.expr("percentile_approx(arc_snr, 0.5)").alias("median"),
        F.expr("percentile_approx(arc_snr, 0.25)").alias("p25"),
        F.expr("percentile_approx(arc_snr, 0.75)").alias("p75")
    ).collect()[0]
    report.append(f"arc_snr: min={arc_stats['min']:.2f}, p25={arc_stats['p25']:.2f}, median={arc_stats['median']:.2f}, p75={arc_stats['p75']:.2f}, max={arc_stats['max']:.2f}, avg={arc_stats['avg']:.2f}")
    
    # magnification statistics
    mag_stats = injections.filter(F.col("magnification").isNotNull()).agg(
        F.min("magnification").alias("min"),
        F.max("magnification").alias("max"),
        F.avg("magnification").alias("avg"),
        F.expr("percentile_approx(magnification, 0.5)").alias("median")
    ).collect()[0]
    report.append(f"magnification: min={mag_stats['min']:.3f}, median={mag_stats['median']:.2f}, max={mag_stats['max']:.2f}, avg={mag_stats['avg']:.2f}")
    
    # total_injected_flux_r statistics
    flux_stats = injections.filter(F.col("total_injected_flux_r").isNotNull()).agg(
        F.min("total_injected_flux_r").alias("min"),
        F.max("total_injected_flux_r").alias("max"),
        F.avg("total_injected_flux_r").alias("avg"),
        F.expr("percentile_approx(total_injected_flux_r, 0.5)").alias("median")
    ).collect()[0]
    report.append(f"total_injected_flux_r: min={flux_stats['min']:.3f}, median={flux_stats['median']:.2f}, max={flux_stats['max']:.2f}, avg={flux_stats['avg']:.2f}")
    
    # Magnification < 1 count
    low_mag = injections.filter(F.col("magnification") < 1.0).count()
    low_mag_pct = 100 * low_mag / n_inj if n_inj > 0 else 0
    report.append(f"\nMagnification < 1 cases: {low_mag} ({low_mag_pct:.2f}%) - Expected for sources near Einstein radius")
    report.append("")

    # =========================================================================
    # 9. SNR vs THETA_E BINNED ANALYSIS
    # =========================================================================
    report.append("=" * 80)
    report.append("9. ARC_SNR vs THETA_E BINNED ANALYSIS")
    report.append("=" * 80)
    
    binned_snr = injections.filter(F.col("arc_snr").isNotNull()).withColumn(
        "theta_bin", F.round(F.col("theta_e_arcsec"), 1)
    ).groupBy("theta_bin").agg(
        F.count("*").alias("count"),
        F.avg("arc_snr").alias("avg_snr"),
        F.expr("percentile_approx(arc_snr, 0.5)").alias("median_snr")
    ).orderBy("theta_bin").collect()
    
    report.append("theta_e_bin | count | avg_snr | median_snr")
    report.append("-" * 50)
    for r in binned_snr:
        report.append(f"{r['theta_bin']:.1f} | {r['count']:,} | {r['avg_snr']:.2f} | {r['median_snr']:.2f}")
    report.append("")

    # =========================================================================
    # 10. TOTAL FLUX vs THETA_E BINNED ANALYSIS (CRITICAL)
    # =========================================================================
    report.append("=" * 80)
    report.append("10. TOTAL_INJECTED_FLUX_R vs THETA_E BINNED ANALYSIS (CRITICAL)")
    report.append("=" * 80)
    
    binned_flux = injections.filter(F.col("total_injected_flux_r").isNotNull()).withColumn(
        "theta_bin", F.round(F.col("theta_e_arcsec"), 1)
    ).groupBy("theta_bin").agg(
        F.count("*").alias("count"),
        F.avg("total_injected_flux_r").alias("avg_flux"),
        F.expr("percentile_approx(total_injected_flux_r, 0.5)").alias("median_flux")
    ).orderBy("theta_bin").collect()
    
    report.append("theta_e_bin | count | avg_flux | median_flux")
    report.append("-" * 50)
    flux_by_theta = []
    for r in binned_flux:
        report.append(f"{r['theta_bin']:.1f} | {r['count']:,} | {r['avg_flux']:.3f} | {r['median_flux']:.3f}")
        flux_by_theta.append((r['theta_bin'], r['avg_flux']))
    
    # Check if flux increases with theta_e
    if len(flux_by_theta) >= 2:
        first_half = [f[1] for f in flux_by_theta[:len(flux_by_theta)//2]]
        second_half = [f[1] for f in flux_by_theta[len(flux_by_theta)//2:]]
        flux_increasing = sum(second_half)/len(second_half) > sum(first_half)/len(first_half)
        report.append(f"\nTotal flux increases with theta_e: {'YES (PASS)' if flux_increasing else 'NO (FAIL)'}")
    report.append("")

    # =========================================================================
    # 11. PSF PROVENANCE
    # =========================================================================
    report.append("=" * 80)
    report.append("11. PSF PROVENANCE")
    report.append("=" * 80)
    
    for col in ["psf_fwhm_used_g", "psf_fwhm_used_r", "psf_fwhm_used_z"]:
        if col in df.columns:
            # Injections should have PSF
            inj_cov = injections.filter(F.col(col).isNotNull()).count()
            inj_pct = 100 * inj_cov / n_inj if n_inj > 0 else 0
            
            # Controls should NOT have PSF (no convolution)
            ctrl_cov = df.filter((F.col("lens_model") == "CONTROL") & F.col(col).isNotNull()).count()
            ctrl_pct = 100 * ctrl_cov / n_controls if n_controls > 0 else 0
            
            report.append(f"{col}: injections={inj_pct:.0f}%, controls={ctrl_pct:.0f}%")
            
            # Stats for injections
            if inj_cov > 0:
                psf_stats = injections.filter(F.col(col).isNotNull()).agg(
                    F.min(col).alias("min"),
                    F.max(col).alias("max"),
                    F.avg(col).alias("avg")
                ).collect()[0]
                report.append(f"  Stats: min={psf_stats['min']:.3f}, max={psf_stats['max']:.3f}, avg={psf_stats['avg']:.3f}")
    report.append("")

    # =========================================================================
    # 12. MASKBITS METRICS
    # =========================================================================
    report.append("=" * 80)
    report.append("12. MASKBITS METRICS")
    report.append("=" * 80)
    
    for col in ["bad_pixel_frac", "wise_brightmask_frac"]:
        if col in df.columns:
            # Coverage
            cov = df.filter(F.col(col).isNotNull()).count()
            cov_pct = 100 * cov / total_rows if total_rows > 0 else 0
            
            # Stats
            stats = df.filter(F.col(col).isNotNull()).agg(
                F.min(col).alias("min"),
                F.max(col).alias("max"),
                F.avg(col).alias("avg"),
                F.expr(f"percentile_approx({col}, 0.5)").alias("median"),
                F.expr(f"percentile_approx({col}, 0.95)").alias("p95")
            ).collect()[0]
            
            report.append(f"{col}:")
            report.append(f"  Coverage: {cov:,}/{total_rows:,} ({cov_pct:.1f}%)")
            report.append(f"  Stats: min={stats['min']:.4f}, median={stats['median']:.4f}, avg={stats['avg']:.4f}, p95={stats['p95']:.4f}, max={stats['max']:.4f}")
    report.append("")

    # =========================================================================
    # 13. OBSERVING CONDITIONS
    # =========================================================================
    report.append("=" * 80)
    report.append("13. OBSERVING CONDITIONS")
    report.append("=" * 80)
    
    for col in ["psfsize_r", "psfdepth_r"]:
        if col in df.columns:
            stats = df.filter(F.col(col).isNotNull()).agg(
                F.min(col).alias("min"),
                F.max(col).alias("max"),
                F.avg(col).alias("avg"),
                F.expr(f"percentile_approx({col}, 0.5)").alias("median")
            ).collect()[0]
            report.append(f"{col}: min={stats['min']:.3f}, median={stats['median']:.3f}, avg={stats['avg']:.3f}, max={stats['max']:.3f}")
    report.append("")

    # =========================================================================
    # 14. PER-SPLIT BREAKDOWN
    # =========================================================================
    report.append("=" * 80)
    report.append("14. PER-SPLIT BREAKDOWN")
    report.append("=" * 80)
    
    for split in ["train", "val", "test"]:
        split_df = df.filter(F.col("region_split") == split)
        n_split = split_df.count()
        if n_split == 0:
            continue
            
        n_ctrl = split_df.filter(F.col("lens_model") == "CONTROL").count()
        n_inj_split = n_split - n_ctrl
        ctrl_pct_split = 100 * n_ctrl / n_split
        
        ok_split = split_df.filter(F.col("cutout_ok") == 1).count()
        success_split = 100 * ok_split / n_split
        
        report.append(f"{split}:")
        report.append(f"  Total: {n_split:,}")
        report.append(f"  Controls: {n_ctrl:,} ({ctrl_pct_split:.1f}%)")
        report.append(f"  Injections: {n_inj_split:,} ({100-ctrl_pct_split:.1f}%)")
        report.append(f"  Success rate: {success_split:.2f}%")
    report.append("")

    # =========================================================================
    # 15. FINAL SUMMARY
    # =========================================================================
    report.append("=" * 80)
    report.append("15. VALIDATION SUMMARY")
    report.append("=" * 80)
    
    checks = [
        ("Success rate >= 95%", success_rate >= 95),
        ("Control fraction 45-55%", 45 <= control_pct <= 55),
        ("Controls have theta_e=0", theta_zero == n_controls),
        ("Controls have NULL arc_snr", arc_null == n_controls),
        ("Controls have NULL magnification", mag_null == n_controls),
        ("Injection arc_snr coverage >= 99%", arc_pct >= 99),
        ("Injection magnification coverage >= 99%", mag_pct >= 99),
        ("Injection flux coverage >= 99%", flux_pct >= 99),
        ("PSF provenance for injections >= 99%", inj_pct >= 99),
        ("Total flux increases with theta_e", flux_increasing if 'flux_increasing' in dir() else False),
    ]
    
    n_pass = 0
    n_fail = 0
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
        else:
            n_fail += 1
        report.append(f"  [{status}] {name}")
    
    report.append("")
    report.append("=" * 80)
    if n_fail == 0:
        report.append("OVERALL: ✅ ALL CHECKS PASSED")
        report.append("Phase 4c train tier output is READY for Phase 4d and Phase 5")
    else:
        report.append(f"OVERALL: ❌ {n_fail} CHECK(S) FAILED")
        report.append("Review failures before proceeding")
    report.append("=" * 80)

    injections.unpersist()
    df.unpersist()
    
    # Write report
    report_text = "\n".join(report)
    print(report_text)
    
    # Write to S3
    write_to_s3(args.output_s3, report_text)
    
    # Exit with error if any checks failed
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()


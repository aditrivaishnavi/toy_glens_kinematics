#!/usr/bin/env python3
"""Comprehensive analysis of Phase 4d completeness surfaces.

Collects all metrics, patterns, and dimensions from the Phase 4d output
to inform Phase 5 training and Phase 6 science analysis.

Usage:
  spark-submit spark_analyze_phase4d.py \
    --surfaces-s3 s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/completeness_surfaces/train_stamp64_... \
    --region-agg-s3 s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/completeness_surfaces_region_agg/train_stamp64_... \
    --psf-provenance-s3 s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/psf_provenance/train_stamp64_... \
    --output-s3 s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/analysis/train_stamp64_...
"""

import argparse
import json
import sys
import time
from typing import Dict, List, Any

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
    parser.add_argument("--surfaces-s3", required=True, help="S3 path to completeness_surfaces")
    parser.add_argument("--region-agg-s3", required=True, help="S3 path to completeness_surfaces_region_agg")
    parser.add_argument("--psf-provenance-s3", required=True, help="S3 path to psf_provenance")
    parser.add_argument("--output-s3", required=True, help="S3 path to write analysis report")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("phase4d_comprehensive_analysis").getOrCreate()
    
    report = []
    report.append("=" * 100)
    report.append("PHASE 4D COMPLETENESS ANALYSIS - COMPREHENSIVE METRICS REPORT")
    report.append("=" * 100)
    report.append(f"Surfaces path: {args.surfaces_s3}")
    report.append(f"Region-agg path: {args.region_agg_s3}")
    report.append(f"PSF provenance path: {args.psf_provenance_s3}")
    report.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    report.append("")

    # =========================================================================
    # SECTION 1: SCHEMA ANALYSIS
    # =========================================================================
    report.append("=" * 100)
    report.append("SECTION 1: SCHEMA ANALYSIS")
    report.append("=" * 100)
    
    print("[INFO] Reading completeness surfaces...")
    df_surf = spark.read.parquet(args.surfaces_s3)
    df_surf.cache()
    
    report.append("\n1.1 Completeness Surfaces Schema:")
    report.append("-" * 50)
    for field in df_surf.schema.fields:
        report.append(f"  {field.name}: {field.dataType.simpleString()} (nullable={field.nullable})")
    
    print("[INFO] Reading region-aggregated surfaces...")
    df_agg = spark.read.parquet(args.region_agg_s3)
    df_agg.cache()
    
    report.append("\n1.2 Region-Aggregated Schema:")
    report.append("-" * 50)
    for field in df_agg.schema.fields:
        report.append(f"  {field.name}: {field.dataType.simpleString()} (nullable={field.nullable})")
    
    report.append("")
    
    # =========================================================================
    # SECTION 2: DATASET SIZE AND STRUCTURE
    # =========================================================================
    report.append("=" * 100)
    report.append("SECTION 2: DATASET SIZE AND STRUCTURE")
    report.append("=" * 100)
    
    n_surf_rows = df_surf.count()
    n_agg_rows = df_agg.count()
    
    report.append(f"\n2.1 Row Counts:")
    report.append(f"  Completeness surfaces (detailed): {n_surf_rows:,} rows")
    report.append(f"  Region-aggregated surfaces: {n_agg_rows:,} rows")
    
    # Unique regions
    n_regions = df_surf.select("region_id").distinct().count()
    report.append(f"\n2.2 Unique Regions: {n_regions}")
    
    # Split distribution
    report.append(f"\n2.3 Split Distribution (surfaces):")
    split_dist = df_surf.groupBy("region_split").agg(
        F.count("*").alias("n_rows"),
        F.sum("n_attempt").alias("total_attempt"),
        F.sum("n_valid_all").alias("total_valid_all"),
        F.sum("n_recovered_all").alias("total_recovered_all"),
    ).orderBy("region_split").collect()
    
    report.append("  split | rows | n_attempt | n_valid_all | n_recovered_all")
    report.append("  " + "-" * 70)
    for r in split_dist:
        report.append(f"  {r['region_split']} | {r['n_rows']:,} | {r['total_attempt']:,} | {r['total_valid_all']:,} | {r['total_recovered_all']:,}")
    
    report.append("")
    
    # =========================================================================
    # SECTION 3: OVERALL COMPLETENESS METRICS
    # =========================================================================
    report.append("=" * 100)
    report.append("SECTION 3: OVERALL COMPLETENESS METRICS")
    report.append("=" * 100)
    
    # Global totals
    totals = df_surf.agg(
        F.sum("n_attempt").alias("total_attempt"),
        F.sum("n_valid_all").alias("total_valid_all"),
        F.sum("n_valid_clean").alias("total_valid_clean"),
        F.sum("n_recovered_all").alias("total_recovered_all"),
        F.sum("n_recovered_clean").alias("total_recovered_clean"),
    ).collect()[0]
    
    total_attempt = totals["total_attempt"]
    total_valid_all = totals["total_valid_all"]
    total_valid_clean = totals["total_valid_clean"]
    total_recovered_all = totals["total_recovered_all"]
    total_recovered_clean = totals["total_recovered_clean"]
    
    report.append(f"\n3.1 Global Counts:")
    report.append(f"  Total attempted: {total_attempt:,}")
    report.append(f"  Total valid (all): {total_valid_all:,}")
    report.append(f"  Total valid (clean): {total_valid_clean:,}")
    report.append(f"  Total recovered (all): {total_recovered_all:,}")
    report.append(f"  Total recovered (clean): {total_recovered_clean:,}")
    
    # Global completeness
    comp_valid_all = total_recovered_all / total_valid_all if total_valid_all > 0 else 0
    comp_valid_clean = total_recovered_clean / total_valid_clean if total_valid_clean > 0 else 0
    comp_overall_all = total_recovered_all / total_attempt if total_attempt > 0 else 0
    valid_frac_all = total_valid_all / total_attempt if total_attempt > 0 else 0
    valid_frac_clean = total_valid_clean / total_attempt if total_attempt > 0 else 0
    
    report.append(f"\n3.2 Global Completeness:")
    report.append(f"  completeness_valid_all (recovered/valid): {comp_valid_all:.4f} ({comp_valid_all*100:.2f}%)")
    report.append(f"  completeness_valid_clean (recovered_clean/valid_clean): {comp_valid_clean:.4f} ({comp_valid_clean*100:.2f}%)")
    report.append(f"  completeness_overall (recovered/attempted): {comp_overall_all:.4f} ({comp_overall_all*100:.2f}%)")
    report.append(f"  valid_fraction_all: {valid_frac_all:.4f} ({valid_frac_all*100:.2f}%)")
    report.append(f"  valid_fraction_clean: {valid_frac_clean:.4f} ({valid_frac_clean*100:.2f}%)")
    
    report.append("")
    
    # =========================================================================
    # SECTION 4: COMPLETENESS BY THETA_E (Einstein Radius)
    # =========================================================================
    report.append("=" * 100)
    report.append("SECTION 4: COMPLETENESS BY THETA_E (Einstein Radius)")
    report.append("=" * 100)
    
    theta_comp = df_surf.groupBy("theta_e_arcsec").agg(
        F.sum("n_attempt").alias("n_attempt"),
        F.sum("n_valid_all").alias("n_valid_all"),
        F.sum("n_valid_clean").alias("n_valid_clean"),
        F.sum("n_recovered_all").alias("n_recovered_all"),
        F.sum("n_recovered_clean").alias("n_recovered_clean"),
        F.avg("arc_snr_mean").alias("arc_snr_mean"),
    ).orderBy("theta_e_arcsec").collect()
    
    report.append("\ntheta_e | n_attempt | n_valid_all | n_recovered_all | comp_valid_all | comp_valid_clean | arc_snr_mean")
    report.append("-" * 120)
    for r in theta_comp:
        comp_all = r["n_recovered_all"] / r["n_valid_all"] if r["n_valid_all"] > 0 else 0
        comp_clean = r["n_recovered_clean"] / r["n_valid_clean"] if r["n_valid_clean"] > 0 else 0
        report.append(f"{r['theta_e_arcsec']:.2f} | {r['n_attempt']:,} | {r['n_valid_all']:,} | {r['n_recovered_all']:,} | {comp_all:.4f} | {comp_clean:.4f} | {r['arc_snr_mean']:.2f}")
    
    report.append("")
    
    # =========================================================================
    # SECTION 5: COMPLETENESS BY RESOLUTION BIN
    # =========================================================================
    report.append("=" * 100)
    report.append("SECTION 5: COMPLETENESS BY RESOLUTION BIN (theta_e / psfsize_r)")
    report.append("=" * 100)
    
    res_comp = df_surf.groupBy("resolution_bin").agg(
        F.sum("n_attempt").alias("n_attempt"),
        F.sum("n_valid_all").alias("n_valid_all"),
        F.sum("n_valid_clean").alias("n_valid_clean"),
        F.sum("n_recovered_all").alias("n_recovered_all"),
        F.sum("n_recovered_clean").alias("n_recovered_clean"),
        F.avg("theta_over_psf_mean").alias("theta_over_psf_mean"),
        F.avg("arc_snr_mean").alias("arc_snr_mean"),
    ).orderBy("resolution_bin").collect()
    
    report.append("\nres_bin | n_attempt | n_valid_all | n_recovered_all | comp_valid_all | comp_valid_clean | theta/psf_mean | arc_snr")
    report.append("-" * 130)
    for r in res_comp:
        comp_all = r["n_recovered_all"] / r["n_valid_all"] if r["n_valid_all"] > 0 else 0
        comp_clean = r["n_recovered_clean"] / r["n_valid_clean"] if r["n_valid_clean"] > 0 else 0
        theta_psf = r["theta_over_psf_mean"] if r["theta_over_psf_mean"] else 0
        report.append(f"{r['resolution_bin']:8} | {r['n_attempt']:,} | {r['n_valid_all']:,} | {r['n_recovered_all']:,} | {comp_all:.4f} | {comp_clean:.4f} | {theta_psf:.3f} | {r['arc_snr_mean']:.2f}")
    
    report.append("")
    
    # =========================================================================
    # SECTION 6: COMPLETENESS BY SOURCE MAGNITUDE (src_dmag)
    # =========================================================================
    report.append("=" * 100)
    report.append("SECTION 6: COMPLETENESS BY SOURCE MAGNITUDE (src_dmag)")
    report.append("=" * 100)
    
    dmag_comp = df_surf.groupBy("src_dmag").agg(
        F.sum("n_attempt").alias("n_attempt"),
        F.sum("n_valid_all").alias("n_valid_all"),
        F.sum("n_recovered_all").alias("n_recovered_all"),
        F.sum("n_recovered_clean").alias("n_recovered_clean"),
        F.avg("arc_snr_mean").alias("arc_snr_mean"),
    ).orderBy("src_dmag").collect()
    
    report.append("\nsrc_dmag | n_attempt | n_valid_all | n_recovered_all | comp_valid_all | arc_snr_mean")
    report.append("-" * 100)
    for r in dmag_comp:
        comp_all = r["n_recovered_all"] / r["n_valid_all"] if r["n_valid_all"] > 0 else 0
        report.append(f"{r['src_dmag']:.2f} | {r['n_attempt']:,} | {r['n_valid_all']:,} | {r['n_recovered_all']:,} | {comp_all:.4f} | {r['arc_snr_mean']:.2f}")
    
    report.append("")
    
    # =========================================================================
    # SECTION 7: COMPLETENESS BY PSF SIZE BIN
    # =========================================================================
    report.append("=" * 100)
    report.append("SECTION 7: COMPLETENESS BY PSF SIZE BIN")
    report.append("=" * 100)
    
    psf_comp = df_surf.groupBy("psf_bin").agg(
        F.sum("n_attempt").alias("n_attempt"),
        F.sum("n_valid_all").alias("n_valid_all"),
        F.sum("n_recovered_all").alias("n_recovered_all"),
        F.avg("completeness_valid_all").alias("comp_mean"),
        F.avg("arc_snr_mean").alias("arc_snr_mean"),
    ).orderBy("psf_bin").collect()
    
    report.append("\npsf_bin (x0.1\") | n_attempt | n_valid_all | n_recovered_all | comp_valid_all | arc_snr_mean")
    report.append("-" * 100)
    for r in psf_comp:
        comp_all = r["n_recovered_all"] / r["n_valid_all"] if r["n_valid_all"] > 0 else 0
        psf_arcsec = r["psf_bin"] * 0.1 if r["psf_bin"] else 0
        report.append(f"{psf_arcsec:.1f}\" | {r['n_attempt']:,} | {r['n_valid_all']:,} | {r['n_recovered_all']:,} | {comp_all:.4f} | {r['arc_snr_mean']:.2f}")
    
    report.append("")
    
    # =========================================================================
    # SECTION 8: COMPLETENESS BY DEPTH BIN
    # =========================================================================
    report.append("=" * 100)
    report.append("SECTION 8: COMPLETENESS BY DEPTH BIN")
    report.append("=" * 100)
    
    depth_comp = df_surf.groupBy("depth_bin").agg(
        F.sum("n_attempt").alias("n_attempt"),
        F.sum("n_valid_all").alias("n_valid_all"),
        F.sum("n_recovered_all").alias("n_recovered_all"),
        F.avg("arc_snr_mean").alias("arc_snr_mean"),
    ).orderBy("depth_bin").collect()
    
    report.append("\ndepth_bin (x0.25mag) | n_attempt | n_valid_all | n_recovered_all | comp_valid_all | arc_snr_mean")
    report.append("-" * 100)
    for r in depth_comp:
        comp_all = r["n_recovered_all"] / r["n_valid_all"] if r["n_valid_all"] > 0 else 0
        depth_mag = r["depth_bin"] * 0.25 if r["depth_bin"] else 0
        report.append(f"{depth_mag:.2f}mag | {r['n_attempt']:,} | {r['n_valid_all']:,} | {r['n_recovered_all']:,} | {comp_all:.4f} | {r['arc_snr_mean']:.2f}")
    
    report.append("")
    
    # =========================================================================
    # SECTION 9: CROSS-TABULATION: THETA_E x RESOLUTION
    # =========================================================================
    report.append("=" * 100)
    report.append("SECTION 9: CROSS-TABULATION: THETA_E x RESOLUTION_BIN")
    report.append("=" * 100)
    
    cross_tab = df_surf.groupBy("theta_e_arcsec", "resolution_bin").agg(
        F.sum("n_attempt").alias("n_attempt"),
        F.sum("n_valid_all").alias("n_valid_all"),
        F.sum("n_recovered_all").alias("n_recovered_all"),
    ).orderBy("theta_e_arcsec", "resolution_bin").collect()
    
    report.append("\ntheta_e | res_bin | n_attempt | n_valid_all | n_recovered_all | completeness")
    report.append("-" * 90)
    for r in cross_tab:
        comp = r["n_recovered_all"] / r["n_valid_all"] if r["n_valid_all"] > 0 else 0
        report.append(f"{r['theta_e_arcsec']:.2f} | {r['resolution_bin']:8} | {r['n_attempt']:,} | {r['n_valid_all']:,} | {r['n_recovered_all']:,} | {comp:.4f}")
    
    report.append("")
    
    # =========================================================================
    # SECTION 10: REGION-LEVEL VARIANCE ANALYSIS
    # =========================================================================
    report.append("=" * 100)
    report.append("SECTION 10: REGION-LEVEL VARIANCE ANALYSIS")
    report.append("=" * 100)
    
    # Variance statistics from region-aggregated data
    variance_stats = df_agg.agg(
        F.avg("completeness_valid_all_mean").alias("comp_mean_of_means"),
        F.avg("completeness_valid_all_std").alias("comp_mean_std"),
        F.max("completeness_valid_all_std").alias("comp_max_std"),
        F.avg("n_regions").alias("avg_regions_per_bin"),
    ).collect()[0]
    
    report.append(f"\n10.1 Region Variance Summary:")
    report.append(f"  Mean completeness (across region means): {variance_stats['comp_mean_of_means']:.4f}")
    report.append(f"  Mean region-to-region std: {variance_stats['comp_mean_std']:.4f}")
    report.append(f"  Max region-to-region std: {variance_stats['comp_max_std']:.4f}")
    report.append(f"  Avg regions per bin: {variance_stats['avg_regions_per_bin']:.1f}")
    
    # Per-split variance
    report.append(f"\n10.2 Region Variance by Split:")
    split_var = df_agg.groupBy("region_split").agg(
        F.avg("completeness_valid_all_mean").alias("comp_mean"),
        F.avg("completeness_valid_all_std").alias("comp_std_mean"),
        F.sum("n_regions").alias("total_region_bins"),
    ).orderBy("region_split").collect()
    
    for r in split_var:
        report.append(f"  {r['region_split']}: mean_comp={r['comp_mean']:.4f}, mean_std={r['comp_std_mean']:.4f}, region_bins={r['total_region_bins']:,}")
    
    report.append("")
    
    # =========================================================================
    # SECTION 11: CONFIDENCE INTERVAL ANALYSIS
    # =========================================================================
    report.append("=" * 100)
    report.append("SECTION 11: CONFIDENCE INTERVAL ANALYSIS")
    report.append("=" * 100)
    
    ci_stats = df_surf.agg(
        F.avg(F.col("ci_high_valid_all") - F.col("ci_low_valid_all")).alias("avg_ci_width"),
        F.max(F.col("ci_high_valid_all") - F.col("ci_low_valid_all")).alias("max_ci_width"),
        F.min(F.col("ci_high_valid_all") - F.col("ci_low_valid_all")).alias("min_ci_width"),
        F.avg("ci_low_valid_all").alias("avg_ci_low"),
        F.avg("ci_high_valid_all").alias("avg_ci_high"),
    ).collect()[0]
    
    report.append(f"\n11.1 Wilson CI Statistics (completeness_valid_all):")
    report.append(f"  Average CI width: {ci_stats['avg_ci_width']:.4f}")
    report.append(f"  Min CI width: {ci_stats['min_ci_width']:.4f}")
    report.append(f"  Max CI width: {ci_stats['max_ci_width']:.4f}")
    report.append(f"  Average CI low bound: {ci_stats['avg_ci_low']:.4f}")
    report.append(f"  Average CI high bound: {ci_stats['avg_ci_high']:.4f}")
    
    # CI by theta_e
    report.append(f"\n11.2 CI Width by Theta_E:")
    ci_by_theta = df_surf.groupBy("theta_e_arcsec").agg(
        F.avg(F.col("ci_high_valid_all") - F.col("ci_low_valid_all")).alias("avg_ci_width"),
        F.avg("n_valid_all").alias("avg_n_valid"),
    ).orderBy("theta_e_arcsec").collect()
    
    for r in ci_by_theta:
        report.append(f"  theta_e={r['theta_e_arcsec']:.2f}: avg_ci_width={r['avg_ci_width']:.4f}, avg_n_valid={r['avg_n_valid']:.1f}")
    
    report.append("")
    
    # =========================================================================
    # SECTION 12: DATA QUALITY IMPACT (CLEAN VS ALL)
    # =========================================================================
    report.append("=" * 100)
    report.append("SECTION 12: DATA QUALITY IMPACT (CLEAN VS ALL)")
    report.append("=" * 100)
    
    # Clean fraction
    clean_frac = total_valid_clean / total_valid_all if total_valid_all > 0 else 0
    report.append(f"\n12.1 Clean Subset Statistics:")
    report.append(f"  Valid clean / Valid all: {total_valid_clean:,} / {total_valid_all:,} = {clean_frac:.4f} ({clean_frac*100:.2f}%)")
    
    # Completeness difference
    comp_diff = comp_valid_clean - comp_valid_all
    report.append(f"  Completeness (all): {comp_valid_all:.4f}")
    report.append(f"  Completeness (clean): {comp_valid_clean:.4f}")
    report.append(f"  Difference (clean - all): {comp_diff:+.4f}")
    
    # By resolution bin
    report.append(f"\n12.2 Clean vs All by Resolution Bin:")
    for r in res_comp:
        comp_all = r["n_recovered_all"] / r["n_valid_all"] if r["n_valid_all"] > 0 else 0
        comp_clean = r["n_recovered_clean"] / r["n_valid_clean"] if r["n_valid_clean"] > 0 else 0
        diff = comp_clean - comp_all
        report.append(f"  {r['resolution_bin']:8}: all={comp_all:.4f}, clean={comp_clean:.4f}, diff={diff:+.4f}")
    
    report.append("")
    
    # =========================================================================
    # SECTION 13: SNR DISTRIBUTION ANALYSIS
    # =========================================================================
    report.append("=" * 100)
    report.append("SECTION 13: ARC SNR DISTRIBUTION ANALYSIS")
    report.append("=" * 100)
    
    snr_stats = df_surf.agg(
        F.min("arc_snr_mean").alias("min"),
        F.max("arc_snr_mean").alias("max"),
        F.avg("arc_snr_mean").alias("avg"),
        F.expr("percentile_approx(arc_snr_mean, 0.25)").alias("p25"),
        F.expr("percentile_approx(arc_snr_mean, 0.5)").alias("p50"),
        F.expr("percentile_approx(arc_snr_mean, 0.75)").alias("p75"),
        F.expr("percentile_approx(arc_snr_mean, 0.95)").alias("p95"),
    ).collect()[0]
    
    report.append(f"\n13.1 Arc SNR Statistics (bin-level means):")
    report.append(f"  Min: {snr_stats['min']:.2f}")
    report.append(f"  P25: {snr_stats['p25']:.2f}")
    report.append(f"  Median: {snr_stats['p50']:.2f}")
    report.append(f"  P75: {snr_stats['p75']:.2f}")
    report.append(f"  P95: {snr_stats['p95']:.2f}")
    report.append(f"  Max: {snr_stats['max']:.2f}")
    report.append(f"  Mean: {snr_stats['avg']:.2f}")
    
    report.append("")
    
    # =========================================================================
    # SECTION 14: PSF PROVENANCE SUMMARY
    # =========================================================================
    report.append("=" * 100)
    report.append("SECTION 14: PSF PROVENANCE SUMMARY")
    report.append("=" * 100)
    
    for band in ["g", "r", "z"]:
        try:
            psf_path = f"{args.psf_provenance_s3}/psf_source_{band}"
            psf_df = spark.read.parquet(psf_path)
            psf_counts = psf_df.collect()
            
            report.append(f"\n14.{['g','r','z'].index(band)+1} PSF Source Distribution ({band}-band):")
            total = sum(r["count"] for r in psf_counts)
            for r in psf_counts:
                src = r[f"psf_source_{band}"]
                count = r["count"]
                pct = 100 * count / total if total > 0 else 0
                src_label = {0: "map", 1: "manifest", 2: "fallback_r", None: "NULL"}.get(src, f"unknown({src})")
                report.append(f"  {src_label}: {count:,} ({pct:.3f}%)")
        except Exception as e:
            report.append(f"\n14.{['g','r','z'].index(band)+1} PSF Source Distribution ({band}-band): Error reading - {str(e)[:100]}")
    
    report.append("")
    
    # =========================================================================
    # SECTION 15: SELECTION SET ANALYSIS
    # =========================================================================
    report.append("=" * 100)
    report.append("SECTION 15: SELECTION SET ANALYSIS")
    report.append("=" * 100)
    
    selset_comp = df_surf.groupBy("selection_set_id").agg(
        F.sum("n_attempt").alias("n_attempt"),
        F.sum("n_valid_all").alias("n_valid_all"),
        F.sum("n_recovered_all").alias("n_recovered_all"),
    ).orderBy("selection_set_id").collect()
    
    report.append("\nselection_set_id | n_attempt | n_valid_all | n_recovered_all | completeness")
    report.append("-" * 80)
    for r in selset_comp:
        comp = r["n_recovered_all"] / r["n_valid_all"] if r["n_valid_all"] > 0 else 0
        report.append(f"{r['selection_set_id']} | {r['n_attempt']:,} | {r['n_valid_all']:,} | {r['n_recovered_all']:,} | {comp:.4f}")
    
    report.append("")
    
    # =========================================================================
    # SECTION 16: BINS WITH EXTREME VALUES
    # =========================================================================
    report.append("=" * 100)
    report.append("SECTION 16: BINS WITH EXTREME VALUES")
    report.append("=" * 100)
    
    # Lowest completeness bins
    report.append("\n16.1 Top 10 Lowest Completeness Bins:")
    low_comp = df_surf.filter(F.col("n_valid_all") >= 100).orderBy("completeness_valid_all").limit(10).collect()
    for r in low_comp:
        report.append(f"  theta_e={r['theta_e_arcsec']:.2f}, res={r['resolution_bin']}, psf={r['psf_bin']}, comp={r['completeness_valid_all']:.4f}, n={r['n_valid_all']}")
    
    # Highest completeness bins
    report.append("\n16.2 Top 10 Highest Completeness Bins:")
    high_comp = df_surf.filter(F.col("n_valid_all") >= 100).orderBy(F.desc("completeness_valid_all")).limit(10).collect()
    for r in high_comp:
        report.append(f"  theta_e={r['theta_e_arcsec']:.2f}, res={r['resolution_bin']}, psf={r['psf_bin']}, comp={r['completeness_valid_all']:.4f}, n={r['n_valid_all']}")
    
    # Widest CI bins
    report.append("\n16.3 Top 10 Widest Confidence Interval Bins:")
    wide_ci = df_surf.withColumn("ci_width", F.col("ci_high_valid_all") - F.col("ci_low_valid_all")).orderBy(F.desc("ci_width")).limit(10).collect()
    for r in wide_ci:
        report.append(f"  theta_e={r['theta_e_arcsec']:.2f}, res={r['resolution_bin']}, ci_width={r['ci_width']:.4f}, n={r['n_valid_all']}")
    
    report.append("")
    
    # =========================================================================
    # SECTION 17: SUMMARY STATISTICS
    # =========================================================================
    report.append("=" * 100)
    report.append("SECTION 17: SUMMARY STATISTICS")
    report.append("=" * 100)
    
    report.append(f"""
17.1 Key Metrics Summary:
  - Total injections analyzed: {total_attempt:,}
  - Valid injections (all): {total_valid_all:,} ({valid_frac_all*100:.1f}%)
  - Valid injections (clean): {total_valid_clean:,} ({valid_frac_clean*100:.1f}%)
  - Recovered (all): {total_recovered_all:,}
  - Recovered (clean): {total_recovered_clean:,}
  - Overall completeness (valid_all): {comp_valid_all:.4f} ({comp_valid_all*100:.2f}%)
  - Overall completeness (valid_clean): {comp_valid_clean:.4f} ({comp_valid_clean*100:.2f}%)
  - Mean CI width: {ci_stats['avg_ci_width']:.4f}
  - Unique regions: {n_regions}
  - Mean region-to-region variance: {variance_stats['comp_mean_std']:.4f}
""")
    
    report.append("=" * 100)
    report.append("END OF PHASE 4D COMPREHENSIVE ANALYSIS REPORT")
    report.append("=" * 100)
    
    # Cleanup
    df_surf.unpersist()
    df_agg.unpersist()
    
    # Write report
    report_text = "\n".join(report)
    print(report_text)
    
    # Write to S3
    write_to_s3(f"{args.output_s3}/phase4d_analysis_report.txt", report_text)
    
    spark.stop()


if __name__ == "__main__":
    main()


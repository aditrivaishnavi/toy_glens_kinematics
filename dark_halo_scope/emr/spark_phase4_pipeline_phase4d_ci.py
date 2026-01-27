#!/usr/bin/env python3
"""
spark_phase4_pipeline_phase4d_ci.py

Phase 4d completeness estimation for Dark Halo Scope.

Reads Phase 4c metrics (injections only) and computes selection-function completeness surfaces
with Wilson score confidence intervals, plus a region-aggregated table (mean/std across regions
and pooled completeness).

Key outputs (Parquet):
- completeness_surfaces/<experiment_id>          (includes region_id for sky-variance analysis)
- completeness_surfaces_region_agg/<experiment_id>  (aggregated across region_id)
- psf_provenance/<experiment_id>/psf_source_*    (optional bandwise PSF-source counts)

Recovery proxy (default):
- arc_snr >= 5.0
- theta_e_arcsec / psfsize_r >= 0.8

"Clean" subset additionally requires:
- bad_pixel_frac <= 0.2
- wise_brightmask_frac <= 0.2

Notes
- This script is intentionally conservative and avoids "optimistic" validity definitions.
  By default, denominators require cutout_ok==1 and non-null metrics (arc_snr, psfsize_r, psfdepth_r).
- Bin definitions match the values shown in your Phase 4d report:
  psf_bin: floor(psfsize_r / psf_bin_width) * psf_bin_width  (default width 0.1 arcsec)
  depth_bin: floor(psfdepth_r / depth_bin_width) * depth_bin_width (default width 0.25 mag)
  resolution_bin: <0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0, >=1.0 (customizable edges)

Example
spark-submit spark_phase4_pipeline_phase4d_ci.py \
  --metrics-s3 s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/metrics/train_stamp64_bandsgrz_gridgrid_small \
  --output-s3 s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed \
  --experiment-id train_stamp64_bandsgrz_gridgrid_small
"""

from __future__ import annotations

import argparse
import math
from typing import List, Tuple

from pyspark.sql import SparkSession, functions as F, types as T


def _parse_edges(csv: str) -> List[float]:
    vals = []
    for x in csv.split(","):
        x = x.strip()
        if not x:
            continue
        if x.lower() in ("inf", "+inf", "infinity"):
            vals.append(float("inf"))
        else:
            vals.append(float(x))
    if len(vals) < 2:
        raise ValueError(f"Need >=2 edges, got: {csv}")
    # ensure sorted
    vals2 = sorted(vals)
    if vals2 != vals:
        vals = vals2
    return vals


def wilson_ci_cols(k_col, n_col, z: float = 1.96) -> Tuple[F.Column, F.Column]:
    """
    Wilson score interval for a binomial proportion k/n.

    Returns (low, high) as Spark Columns. For n<=0 -> NULL.
    """
    z2 = z * z
    n = n_col.cast("double")
    k = k_col.cast("double")
    p = F.when(n > 0, k / n).otherwise(F.lit(None).cast("double"))

    denom = (F.lit(1.0) + F.lit(z2) / n)
    center = (p + F.lit(z2) / (F.lit(2.0) * n)) / denom

    # half = z / denom * sqrt( (p(1-p) + z^2/(4n)) / n )
    half = (F.lit(z) / denom) * F.sqrt((p * (F.lit(1.0) - p) + F.lit(z2) / (F.lit(4.0) * n)) / n)

    low = F.when(n > 0, center - half).otherwise(F.lit(None).cast("double"))
    high = F.when(n > 0, center + half).otherwise(F.lit(None).cast("double"))
    return low, high


def make_resolution_bin(col: F.Column, edges: List[float]) -> F.Column:
    """
    Return a string bin label based on edges. Example edges:
    [0.0, 0.4, 0.6, 0.8, 1.0, inf]
    -> "<0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", ">=1.0"
    """
    # We assume edges sorted.
    labels = []
    for i in range(1, len(edges) - 1):
        labels.append(f"{edges[i]:g}-{edges[i+1]:g}")
    # Build conditions:
    expr = None
    # first bin: < edges[1]
    first = F.when(col < F.lit(edges[1]), F.lit(f"<{edges[1]:g}"))
    expr = first
    # middle bins: [edges[i], edges[i+1])
    for i in range(1, len(edges) - 2):
        lo, hi = edges[i], edges[i + 1]
        label = f"{lo:g}-{hi:g}"
        expr = expr.when((col >= F.lit(lo)) & (col < F.lit(hi)), F.lit(label))
    # last bin: >= edges[-2] (unless edges[-1] is not inf; still last-open)
    last_lo = edges[-2]
    expr = expr.otherwise(F.when(col >= F.lit(last_lo), F.lit(f">={last_lo:g}")).otherwise(F.lit(None)))
    return expr


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 4d completeness with Wilson confidence intervals (Spark)")
    p.add_argument("--metrics-s3", required=True, help="Phase 4c metrics parquet root (s3://...)")
    p.add_argument("--output-s3", required=True, help="Phase 4d output root (s3://...)")
    p.add_argument("--experiment-id", required=True, help="Experiment id name for output folders")
    p.add_argument("--force", type=int, default=0, help="Overwrite outputs if 1")
    p.add_argument("--write-psf-provenance", type=int, default=1, help="Write psf_source_* counts if 1")

    # Recovery criteria
    p.add_argument("--recovery-snr-thresh", type=float, default=5.0)
    p.add_argument("--recovery-theta-over-psf", type=float, default=0.8)

    # Clean subset cuts
    p.add_argument("--quality-bad-pixel-max", type=float, default=0.2)
    p.add_argument("--quality-wise-max", type=float, default=0.2)

    # Validity
    p.add_argument("--require-metrics-ok", type=int, default=1,
                   help="If 1, denominators require non-null metrics and positive psfsize_r/psfdepth_r (default 1)")

    # Binning
    p.add_argument("--psf-bin-width", type=float, default=0.1, help="arcsec bin width; bins use floor()")
    p.add_argument("--depth-bin-width", type=float, default=0.25, help="mag bin width; bins use floor()")
    p.add_argument("--resolution-edges", default="0.0,0.4,0.6,0.8,1.0,inf",
                   help="CSV edges for theta_over_psf bin labels")

    # CI
    p.add_argument("--ci-z", type=float, default=1.96, help="z-score for Wilson CI (default 1.96 ~ 95%)")

    return p


def main() -> None:
    args = build_parser().parse_args()
    spark = SparkSession.builder.appName(f"phase4d_ci_{args.experiment_id}").getOrCreate()
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    mode = "overwrite" if int(args.force) == 1 else "error"

    print(f"[4d] Reading metrics: {args.metrics_s3}")
    df = spark.read.parquet(args.metrics_s3)

    # Keep injections only
    # Expected: lens_model == "CONTROL" for controls; others are injections (e.g., "SIE")
    inj = df.filter(F.col("lens_model") != F.lit("CONTROL"))

    # Basic required columns check (fail fast with clear message)
    required = [
        "region_id", "region_split", "selection_set_id", "ranking_mode",
        "theta_e_arcsec", "src_dmag", "src_reff_arcsec",
        "arc_snr", "psfsize_r", "psfdepth_r",
        "bad_pixel_frac", "wise_brightmask_frac",
        "cutout_ok",
    ]
    missing = [c for c in required if c not in inj.columns]
    if missing:
        raise ValueError(f"Missing required columns in metrics: {missing}")

    # Derived: theta_over_psf
    inj = inj.withColumn("theta_over_psf", F.col("theta_e_arcsec") / F.col("psfsize_r"))

    # Validity definition
    metrics_ok = (
        (F.col("cutout_ok") == F.lit(1)) &
        F.col("arc_snr").isNotNull() &
        (F.col("psfsize_r").isNotNull()) & (F.col("psfsize_r") > F.lit(0.0)) &
        (F.col("psfdepth_r").isNotNull()) & (F.col("psfdepth_r") > F.lit(0.0))
    )
    if int(args.require_metrics_ok) == 0:
        metrics_ok = (F.col("cutout_ok") == F.lit(1))

    subset_all_ok = metrics_ok
    subset_clean_ok = metrics_ok & (F.col("bad_pixel_frac") <= F.lit(float(args.quality_bad_pixel_max))) & (F.col("wise_brightmask_frac") <= F.lit(float(args.quality_wise_max)))

    inj = inj.withColumn("subset_all_ok", subset_all_ok.cast("int"))
    inj = inj.withColumn("subset_clean_ok", subset_clean_ok.cast("int"))

    # Recovery proxy
    recovered_proxy = (F.col("arc_snr") >= F.lit(float(args.recovery_snr_thresh))) & (F.col("theta_over_psf") >= F.lit(float(args.recovery_theta_over_psf)))

    inj = inj.withColumn("recovered_all", (subset_all_ok & recovered_proxy).cast("int"))
    inj = inj.withColumn("recovered_clean", (subset_clean_ok & recovered_proxy).cast("int"))

    # Bins: psf_bin and depth_bin using floor
    psf_w = float(args.psf_bin_width)
    depth_w = float(args.depth_bin_width)

    # floor(psfsize_r / w) * w; keep as double, round to avoid float noise
    inj = inj.withColumn("psf_bin", F.round(F.floor(F.col("psfsize_r") / F.lit(psf_w)) * F.lit(psf_w), 2))
    inj = inj.withColumn("depth_bin", F.round(F.floor(F.col("psfdepth_r") / F.lit(depth_w)) * F.lit(depth_w), 2))

    # resolution_bin labels
    edges = _parse_edges(args.resolution_edges)
    inj = inj.withColumn("resolution_bin", make_resolution_bin(F.col("theta_over_psf"), edges))

    # Group keys
    group_cols = [
        "region_id",
        "region_split",
        "selection_set_id",
        "ranking_mode",
        "theta_e_arcsec",
        "src_dmag",
        "src_reff_arcsec",
        "psf_bin",
        "depth_bin",
        "resolution_bin",
    ]

    print("[4d] Computing completeness surfaces...")
    agg = (
        inj.groupBy(*group_cols)
        .agg(
            F.count("*").alias("n_attempt"),
            F.sum("subset_all_ok").alias("n_valid_all"),
            F.sum("subset_clean_ok").alias("n_valid_clean"),
            F.sum("recovered_all").alias("n_recovered_all"),
            F.sum("recovered_clean").alias("n_recovered_clean"),
            F.avg("arc_snr").alias("arc_snr_mean"),
            F.expr("percentile_approx(arc_snr, 0.5)").alias("arc_snr_p50"),
            F.avg("theta_over_psf").alias("theta_over_psf_mean"),
        )
    )

    def safe_div(num, den):
        return F.when(den > 0, (num.cast("double") / den.cast("double"))).otherwise(F.lit(None).cast("double"))

    agg = (
        agg
        .withColumn("completeness_valid_all", safe_div(F.col("n_recovered_all"), F.col("n_valid_all")))
        .withColumn("completeness_valid_clean", safe_div(F.col("n_recovered_clean"), F.col("n_valid_clean")))
        .withColumn("completeness_overall_all", safe_div(F.col("n_recovered_all"), F.col("n_attempt")))
        .withColumn("completeness_overall_clean", safe_div(F.col("n_recovered_clean"), F.col("n_attempt")))
        .withColumn("valid_frac_all", safe_div(F.col("n_valid_all"), F.col("n_attempt")))
        .withColumn("valid_frac_clean", safe_div(F.col("n_valid_clean"), F.col("n_attempt")))
    )

    # Wilson CI on completeness_valid_* (k = recovered, n = valid)
    z = float(args.ci_z)
    low_all, high_all = wilson_ci_cols(F.col("n_recovered_all"), F.col("n_valid_all"), z=z)
    low_clean, high_clean = wilson_ci_cols(F.col("n_recovered_clean"), F.col("n_valid_clean"), z=z)

    agg = (
        agg
        .withColumn("ci_low_valid_all", low_all)
        .withColumn("ci_high_valid_all", high_all)
        .withColumn("ci_low_valid_clean", low_clean)
        .withColumn("ci_high_valid_clean", high_clean)
    )

    out_surf = f"{args.output_s3.rstrip('/')}/completeness_surfaces/{args.experiment_id}"
    print(f"[4d] Writing completeness surfaces: {out_surf}")
    (
        agg.repartition("region_split")
        .write.mode(mode)
        .partitionBy("region_split")
        .parquet(out_surf)
    )

    # Region-aggregated: mean/std across regions plus pooled completeness + pooled Wilson CI
    print("[4d] Computing region-aggregated surfaces...")
    group_no_region = [c for c in group_cols if c != "region_id"]
    reg = (
        agg.groupBy(*group_no_region)
        .agg(
            F.countDistinct("region_id").alias("n_regions"),
            F.sum("n_attempt").alias("n_attempt_total"),
            F.sum("n_valid_all").alias("n_valid_all_total"),
            F.sum("n_valid_clean").alias("n_valid_clean_total"),
            F.sum("n_recovered_all").alias("n_recovered_all_total"),
            F.sum("n_recovered_clean").alias("n_recovered_clean_total"),
            F.avg("completeness_valid_all").alias("completeness_valid_all_mean"),
            F.stddev("completeness_valid_all").alias("completeness_valid_all_std"),
            F.avg("completeness_valid_clean").alias("completeness_valid_clean_mean"),
            F.stddev("completeness_valid_clean").alias("completeness_valid_clean_std"),
            F.avg("arc_snr_mean").alias("arc_snr_mean"),
            F.expr("percentile_approx(arc_snr_p50, 0.5)").alias("arc_snr_p50"),
            F.avg("theta_over_psf_mean").alias("theta_over_psf_mean"),
        )
        .withColumn("completeness_valid_all_pooled", safe_div(F.col("n_recovered_all_total"), F.col("n_valid_all_total")))
        .withColumn("completeness_valid_clean_pooled", safe_div(F.col("n_recovered_clean_total"), F.col("n_valid_clean_total")))
    )

    low_p_all, high_p_all = wilson_ci_cols(F.col("n_recovered_all_total"), F.col("n_valid_all_total"), z=z)
    low_p_clean, high_p_clean = wilson_ci_cols(F.col("n_recovered_clean_total"), F.col("n_valid_clean_total"), z=z)

    reg = (
        reg
        .withColumn("ci_low_pooled_all", low_p_all)
        .withColumn("ci_high_pooled_all", high_p_all)
        .withColumn("ci_low_pooled_clean", low_p_clean)
        .withColumn("ci_high_pooled_clean", high_p_clean)
    )

    out_reg = f"{args.output_s3.rstrip('/')}/completeness_surfaces_region_agg/{args.experiment_id}"
    print(f"[4d] Writing region-aggregated surfaces: {out_reg}")
    (
        reg.repartition("region_split")
        .write.mode(mode)
        .partitionBy("region_split")
        .parquet(out_reg)
    )

    if int(args.write_psf_provenance) == 1:
        # Optional: write counts for psf_source_* columns if present
        prov_root = f"{args.output_s3.rstrip('/')}/psf_provenance/{args.experiment_id}"
        for band in ["g", "r", "z"]:
            col = f"psf_source_{band}"
            if col in inj.columns:
                out = f"{prov_root}/{col}"
                print(f"[4d] Writing PSF provenance counts: {out}")
                (
                    inj.groupBy(col)
                    .agg(F.count("*").alias("n"))
                    .write.mode(mode)
                    .parquet(out)
                )

    print("[4d] Done.")
    spark.stop()


if __name__ == "__main__":
    main()

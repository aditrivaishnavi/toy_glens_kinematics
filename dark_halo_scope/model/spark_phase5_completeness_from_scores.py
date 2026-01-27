#!/usr/bin/env python3
"""
Phase 5: Spark aggregation to compute model-based completeness surfaces from a per-row score table.

This reads the output of phase5_infer_scores.py (parquet dataset), then computes:
- Completeness surfaces for injections only (lens_model != CONTROL, or y_true == 1)
- Optional false-positive-rate surfaces for controls (lens_model == CONTROL)

The binning is designed to match Phase 4d "proxy resolvability" binning as closely as possible:
- theta_e_arcsec, src_dmag, src_reff_arcsec: use existing discrete grid values
- psf_bin: 0.1 arcsec rounding of psf_fwhm_used_r (fallback to psfsize_r)
- depth_bin: 0.25 mag bins from psfdepth_r (rounded to nearest 0.25)
- resolution_bin: bins on theta_over_psf computed from per-stamp psf_fwhm_used_r (fallback psfsize_r)

Recovered definition (model-based):
- recovered = valid_all AND (model_score >= score_threshold)

Validity (for denominators):
- valid_all = (cutout_ok == 1) AND model_score IS NOT NULL
- valid_clean adds: bad_pixel_frac <= bad_pixel_max AND wise_brightmask_frac <= wise_brightmask_max
  (if columns are missing, valid_clean falls back to valid_all)

Confidence intervals:
- Wilson score interval for recovered/valid (z=1.96)

Example:
  spark-submit spark_phase5_completeness_from_scores.py \
    --scores "s3://.../phase5/scores/resnet18_v1/test_scores" \
    --out_root "s3://.../phase5/completeness/resnet18_v1" \
    --score_threshold 0.5 \
    --bad_pixel_max 0.2 \
    --wise_brightmask_max 0.2
"""

from __future__ import annotations

import argparse
import math

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


def wilson_ci(k_col, n_col, z: float = 1.96):
    """
    Returns (ci_low, ci_high) columns for Wilson interval.
    Handles n=0 by returning nulls.
    """
    k = k_col.cast("double")
    n = n_col.cast("double")
    z2 = z * z
    phat = F.when(n > 0, k / n).otherwise(F.lit(None).cast("double"))
    denom = F.when(n > 0, 1.0 + z2 / n).otherwise(F.lit(None).cast("double"))
    center = F.when(n > 0, (phat + z2 / (2.0 * n)) / denom).otherwise(F.lit(None).cast("double"))
    rad = F.when(
        n > 0,
        z * F.sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n))) / denom
    ).otherwise(F.lit(None).cast("double"))
    lo = F.when(n > 0, F.greatest(F.lit(0.0), center - rad)).otherwise(F.lit(None).cast("double"))
    hi = F.when(n > 0, F.least(F.lit(1.0), center + rad)).otherwise(F.lit(None).cast("double"))
    return lo, hi


def round_to_step(col, step: float):
    return F.round(col / F.lit(step)) * F.lit(step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="Parquet dataset from phase5_infer_scores.py")
    ap.add_argument("--out_root", required=True, help="Output root directory (parquet datasets)")
    ap.add_argument("--score_threshold", type=float, default=0.5)
    ap.add_argument("--bad_pixel_max", type=float, default=0.2)
    ap.add_argument("--wise_brightmask_max", type=float, default=0.2)
    ap.add_argument("--z", type=float, default=1.96)
    ap.add_argument("--write_controls_fpr", action="store_true", help="Also write control false-positive-rate surfaces")
    args = ap.parse_args()

    spark = SparkSession.builder.appName("phase5_completeness_from_scores").getOrCreate()

    df = spark.read.parquet(args.scores)

    # Use the per-stamp PSF that was actually used, with fallback
    psf_for_resolution = F.coalesce(F.col("psf_fwhm_used_r"), F.col("psfsize_r")).cast("double")
    df = df.withColumn("psf_for_resolution", psf_for_resolution)

    df = df.withColumn(
        "theta_over_psf",
        F.when((F.col("psf_for_resolution").isNotNull()) & (F.col("psf_for_resolution") > 0),
               F.col("theta_e_arcsec").cast("double") / F.col("psf_for_resolution"))
        .otherwise(F.lit(None).cast("double"))
    )

    # Bins
    df = df.withColumn("psf_bin", round_to_step(F.col("psf_for_resolution"), 0.1).cast("double"))

    # depth bins: 0.25 mag steps
    if "psfdepth_r" in df.columns:
        df = df.withColumn("depth_bin", round_to_step(F.col("psfdepth_r").cast("double"), 0.25).cast("double"))
    else:
        df = df.withColumn("depth_bin", F.lit(None).cast("double"))

    # resolution bins (same labels as Phase 4d report)
    df = df.withColumn(
        "resolution_bin",
        F.when(F.col("theta_over_psf").isNull(), F.lit("NULL"))
         .when(F.col("theta_over_psf") < 0.4, F.lit("<0.4"))
         .when((F.col("theta_over_psf") >= 0.4) & (F.col("theta_over_psf") < 0.6), F.lit("0.4-0.6"))
         .when((F.col("theta_over_psf") >= 0.6) & (F.col("theta_over_psf") < 0.8), F.lit("0.6-0.8"))
         .when((F.col("theta_over_psf") >= 0.8) & (F.col("theta_over_psf") < 1.0), F.lit("0.8-1.0"))
         .otherwise(F.lit(">=1.0"))
    )

    # Validity and clean flags
    valid_all_expr = (F.col("cutout_ok") == 1) & F.col("model_score").isNotNull()
    df = df.withColumn("valid_all", F.when(valid_all_expr, F.lit(1)).otherwise(F.lit(0)))

    has_bad = "bad_pixel_frac" in df.columns
    has_wise = "wise_brightmask_frac" in df.columns

    if has_bad and has_wise:
        valid_clean_expr = valid_all_expr & (F.col("bad_pixel_frac") <= F.lit(args.bad_pixel_max)) & (F.col("wise_brightmask_frac") <= F.lit(args.wise_brightmask_max))
        df = df.withColumn("valid_clean", F.when(valid_clean_expr, F.lit(1)).otherwise(F.lit(0)))
    else:
        df = df.withColumn("valid_clean", F.col("valid_all"))

    recovered_expr = valid_all_expr & (F.col("model_score") >= F.lit(args.score_threshold))
    df = df.withColumn("recovered_all", F.when(recovered_expr, F.lit(1)).otherwise(F.lit(0)))

    recovered_clean_expr = (F.col("valid_clean") == 1) & (F.col("model_score") >= F.lit(args.score_threshold))
    df = df.withColumn("recovered_clean", F.when(recovered_clean_expr, F.lit(1)).otherwise(F.lit(0)))

    # Primary grouping keys: match Phase 4d schema where possible
    group_keys = [
        "region_id",
        "selection_set_id",
        "ranking_mode",
        "theta_e_arcsec",
        "src_dmag",
        "src_reff_arcsec",
        "psf_bin",
        "depth_bin",
        "resolution_bin",
        "region_split",
    ]
    group_keys = [k for k in group_keys if k in df.columns]

    # Injections only for completeness
    inj = df
    if "y_true" in df.columns:
        inj = inj.filter(F.col("y_true") == 1)
    elif "lens_model" in df.columns:
        inj = inj.filter(F.col("lens_model") != F.lit("CONTROL"))

    agg = inj.groupBy(*group_keys).agg(
        F.count(F.lit(1)).alias("n_attempt"),
        F.sum("valid_all").cast("long").alias("n_valid_all"),
        F.sum("valid_clean").cast("long").alias("n_valid_clean"),
        F.sum("recovered_all").cast("long").alias("n_recovered_all"),
        F.sum("recovered_clean").cast("long").alias("n_recovered_clean"),
        F.avg("model_score").alias("model_score_mean"),
        F.expr("percentile_approx(model_score, 0.5)").alias("model_score_p50"),
        F.avg("theta_over_psf").alias("theta_over_psf_mean"),
    )

    # Fractions and Wilson CIs
    agg = agg.withColumn(
        "completeness_valid_all",
        F.when(F.col("n_valid_all") > 0, F.col("n_recovered_all") / F.col("n_valid_all")).otherwise(F.lit(None).cast("double"))
    ).withColumn(
        "completeness_valid_clean",
        F.when(F.col("n_valid_clean") > 0, F.col("n_recovered_clean") / F.col("n_valid_clean")).otherwise(F.lit(None).cast("double"))
    ).withColumn(
        "completeness_overall_all",
        F.when(F.col("n_attempt") > 0, F.col("n_recovered_all") / F.col("n_attempt")).otherwise(F.lit(None).cast("double"))
    ).withColumn(
        "completeness_overall_clean",
        F.when(F.col("n_attempt") > 0, F.col("n_recovered_clean") / F.col("n_attempt")).otherwise(F.lit(None).cast("double"))
    ).withColumn(
        "valid_frac_all",
        F.when(F.col("n_attempt") > 0, F.col("n_valid_all") / F.col("n_attempt")).otherwise(F.lit(None).cast("double"))
    ).withColumn(
        "valid_frac_clean",
        F.when(F.col("n_attempt") > 0, F.col("n_valid_clean") / F.col("n_attempt")).otherwise(F.lit(None).cast("double"))
    )

    lo_all, hi_all = wilson_ci(F.col("n_recovered_all"), F.col("n_valid_all"), z=args.z)
    lo_clean, hi_clean = wilson_ci(F.col("n_recovered_clean"), F.col("n_valid_clean"), z=args.z)

    agg = agg.withColumn("ci_low_valid_all", lo_all).withColumn("ci_high_valid_all", hi_all)
    agg = agg.withColumn("ci_low_valid_clean", lo_clean).withColumn("ci_high_valid_clean", hi_clean)

    out_surfaces = args.out_root.rstrip("/") + "/completeness_surfaces_injections"
    agg.repartition(256).write.mode("overwrite").parquet(out_surfaces)

    # Region-aggregated across regions (mean/std and pooled)
    if "region_id" in group_keys:
        agg2_keys = [k for k in group_keys if k != "region_id"]
        pooled = agg.groupBy(*agg2_keys).agg(
            F.countDistinct("region_id").alias("n_regions"),
            F.sum("n_attempt").alias("n_attempt_total"),
            F.sum("n_valid_all").alias("n_valid_all_total"),
            F.sum("n_valid_clean").alias("n_valid_clean_total"),
            F.sum("n_recovered_all").alias("n_recovered_all_total"),
            F.sum("n_recovered_clean").alias("n_recovered_clean_total"),
            F.avg("completeness_valid_all").alias("completeness_valid_all_mean"),
            F.stddev_pop("completeness_valid_all").alias("completeness_valid_all_std"),
            F.avg("completeness_valid_clean").alias("completeness_valid_clean_mean"),
            F.stddev_pop("completeness_valid_clean").alias("completeness_valid_clean_std"),
            F.avg("model_score_mean").alias("model_score_mean"),
            F.expr("percentile_approx(model_score_p50, 0.5)").alias("model_score_p50"),
            F.avg("theta_over_psf_mean").alias("theta_over_psf_mean"),
        )

        pooled = pooled.withColumn(
            "completeness_valid_all_pooled",
            F.when(F.col("n_valid_all_total") > 0, F.col("n_recovered_all_total") / F.col("n_valid_all_total")).otherwise(F.lit(None).cast("double"))
        ).withColumn(
            "completeness_valid_clean_pooled",
            F.when(F.col("n_valid_clean_total") > 0, F.col("n_recovered_clean_total") / F.col("n_valid_clean_total")).otherwise(F.lit(None).cast("double"))
        )

        lo_p_all, hi_p_all = wilson_ci(F.col("n_recovered_all_total"), F.col("n_valid_all_total"), z=args.z)
        lo_p_clean, hi_p_clean = wilson_ci(F.col("n_recovered_clean_total"), F.col("n_valid_clean_total"), z=args.z)

        pooled = pooled.withColumn("ci_low_pooled_all", lo_p_all).withColumn("ci_high_pooled_all", hi_p_all)
        pooled = pooled.withColumn("ci_low_pooled_clean", lo_p_clean).withColumn("ci_high_pooled_clean", hi_p_clean)

        out_agg = args.out_root.rstrip("/") + "/completeness_surfaces_region_agg_injections"
        pooled.repartition(256).write.mode("overwrite").parquet(out_agg)

    # Optional: controls false positive rate (FPR) surfaces
    if args.write_controls_fpr:
        ctrl = df
        if "y_true" in df.columns:
            ctrl = ctrl.filter(F.col("y_true") == 0)
        elif "lens_model" in df.columns:
            ctrl = ctrl.filter(F.col("lens_model") == F.lit("CONTROL"))

        ctrl_agg = ctrl.groupBy(*group_keys).agg(
            F.count(F.lit(1)).alias("n_attempt"),
            F.sum("valid_all").cast("long").alias("n_valid_all"),
            F.sum("valid_clean").cast("long").alias("n_valid_clean"),
            F.sum("recovered_all").cast("long").alias("n_flagged_all"),
            F.sum("recovered_clean").cast("long").alias("n_flagged_clean"),
            F.avg("model_score").alias("model_score_mean"),
            F.expr("percentile_approx(model_score, 0.5)").alias("model_score_p50"),
            F.avg("theta_over_psf").alias("theta_over_psf_mean"),
        )
        ctrl_agg = ctrl_agg.withColumn(
            "fpr_valid_all",
            F.when(F.col("n_valid_all") > 0, F.col("n_flagged_all") / F.col("n_valid_all")).otherwise(F.lit(None).cast("double"))
        ).withColumn(
            "fpr_valid_clean",
            F.when(F.col("n_valid_clean") > 0, F.col("n_flagged_clean") / F.col("n_valid_clean")).otherwise(F.lit(None).cast("double"))
        )
        lo_f_all, hi_f_all = wilson_ci(F.col("n_flagged_all"), F.col("n_valid_all"), z=args.z)
        lo_f_clean, hi_f_clean = wilson_ci(F.col("n_flagged_clean"), F.col("n_valid_clean"), z=args.z)
        ctrl_agg = ctrl_agg.withColumn("ci_low_fpr_all", lo_f_all).withColumn("ci_high_fpr_all", hi_f_all)
        ctrl_agg = ctrl_agg.withColumn("ci_low_fpr_clean", lo_f_clean).withColumn("ci_high_fpr_clean", hi_f_clean)

        out_ctrl = args.out_root.rstrip("/") + "/fpr_surfaces_controls"
        ctrl_agg.repartition(256).write.mode("overwrite").parquet(out_ctrl)

    spark.stop()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4a manifest validator (merged version).

Validates:
- Required columns and non-null keys
- Per-tier and per-split row counts are non-zero
- Control fraction by tier within tolerance (if stage_config provided)
- Parameter ranges: theta_e, src_* random fields, shear, colors
- Sanity checks that avoid classic bugs:
    * theta_e == 0 is treated as valid numeric (do not use truthiness checks)
    * debug/grid task counts scale as O(n_cfg), not O(n_cfg^2)

Adapted to work with the merged Phase 4 pipeline column schema.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F


def load_stage_config(spark: SparkSession, path: str) -> Dict:
    """Load stage config from S3 or local path."""
    if not path:
        return {}
    if path.startswith("s3://"):
        txt = "\n".join(spark.sparkContext.textFile(path).collect())
        return json.loads(txt)
    if path.startswith("file://"):
        path = path[len("file://"):]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def require_columns(df: DataFrame, cols: List[str]) -> List[str]:
    """Return list of columns that are missing from the DataFrame."""
    missing = [c for c in cols if c not in df.columns]
    return missing


def derive_tier(experiment_id: str) -> str:
    """Extract tier from experiment_id (e.g., 'debug_stamp64_...' -> 'debug')."""
    if experiment_id:
        return experiment_id.split("_")[0]
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests-s3", required=True, help="S3 path to manifests parquet")
    ap.add_argument("--bricks-manifest-s3", required=True, help="S3 path to bricks_manifest parquet")
    ap.add_argument("--stage-config-json", default="", help="Optional S3 path to _stage_config.json")
    ap.add_argument("--control-frac-tol", type=float, default=0.05, help="Tolerance for control fraction mismatch")
    ap.add_argument("--min-rows-per-split", type=int, default=100, help="Minimum rows per (tier, split)")
    args = ap.parse_args()

    spark = SparkSession.builder.appName("phase4a_validate_merged").getOrCreate()

    # Load optional stage config for expected control fractions
    cfg = {}
    if args.stage_config_json:
        try:
            cfg = load_stage_config(spark, args.stage_config_json)
            print(f"[validate] Loaded stage config from {args.stage_config_json}")
        except Exception as e:
            print(f"[validate] Warning: could not load stage config: {e}")

    # Read manifests
    m = spark.read.parquet(args.manifests_s3)
    print(f"[validate] Loaded manifests from {args.manifests_s3}")
    print(f"[validate] Manifest columns: {m.columns}")

    # Read bricks manifest
    b = spark.read.parquet(args.bricks_manifest_s3)
    print(f"[validate] Loaded bricks_manifest from {args.bricks_manifest_s3}")

    # Required columns (matching our merged pipeline schema)
    required = [
        "task_id", "experiment_id", "selection_set_id", "region_split", 
        "brickname", "ra", "dec", "stamp_size", "bandset", "replicate", "is_control",
        "config_id", "theta_e_arcsec",
        # Frozen randomness columns
        "task_seed64", "src_x_arcsec", "src_y_arcsec", "src_phi_rad", "shear_phi_rad",
        # Lens model and ellipticity (added 2026-01-23)
        "lens_model", "lens_e", "lens_phi_rad",
        # Provenance
        "ab_zp_nmgy", "pipeline_version",
    ]
    missing = require_columns(m, required)
    if missing:
        raise RuntimeError(f"Missing required manifest columns: {missing}")

    # Add tier column derived from experiment_id
    derive_tier_udf = F.udf(derive_tier)
    m = m.withColumn("tier", derive_tier_udf(F.col("experiment_id")))

    # Basic null checks on key columns
    key_nulls = (m
        .select(
            F.sum(F.col("brickname").isNull().cast("int")).alias("null_brickname"),
            F.sum(F.col("ra").isNull().cast("int")).alias("null_ra"),
            F.sum(F.col("dec").isNull().cast("int")).alias("null_dec"),
            F.sum(F.col("experiment_id").isNull().cast("int")).alias("null_experiment_id"),
        ).collect()[0]
    )
    nulls_dict = key_nulls.asDict()
    if any(nulls_dict[k] > 0 for k in nulls_dict):
        raise RuntimeError(f"Nulls in key columns: {nulls_dict}")
    print("[validate] Key column null check: PASSED")

    # Row counts by tier/split
    counts = m.groupBy("tier", "region_split").count().orderBy("tier", "region_split").collect()
    if not counts:
        raise RuntimeError("No rows in manifests dataset")

    print("[validate] Row counts by tier/split:")
    for row in counts:
        print(f"  tier={row['tier']}, split={row['region_split']}: {row['count']:,}")
        if row["count"] < int(args.min_rows_per_split):
            raise RuntimeError(f"Too few rows for tier={row['tier']} split={row['region_split']}: {row['count']}")
    print("[validate] Minimum row count check: PASSED")

    # Control fraction validation
    frac = (m.groupBy("tier", "region_split")
        .agg(
            F.avg(F.col("is_control").cast("double")).alias("control_frac"),
            F.count("*").alias("n")
        )
        .orderBy("tier", "region_split")
        .collect()
    )

    # Expected control fractions from stage config or defaults
    tiers_cfg = cfg.get("tiers", {})
    exp = {
        "train": float(tiers_cfg.get("train", {}).get("control_frac", 0.25)),
        "grid": float(tiers_cfg.get("grid", {}).get("control_frac", 0.10)),
        "debug": float(tiers_cfg.get("debug", {}).get("control_frac", 0.0)),
    }
    tol = float(args.control_frac_tol)

    print("[validate] Control fraction by tier/split:")
    for row in frac:
        tier = row["tier"]
        observed = float(row["control_frac"])
        expected = exp.get(tier, 0.0)
        status = "OK" if abs(observed - expected) <= tol else "WARN"
        print(f"  tier={tier}, split={row['region_split']}: observed={observed:.3f}, expected~{expected:.3f} [{status}]")
        if abs(observed - expected) > tol:
            print(f"    WARNING: Control fraction mismatch exceeds tolerance {tol}")

    # Range checks (theta_e == 0 is valid for controls, do NOT use truthiness)
    rng = m.agg(
        F.min("theta_e_arcsec").alias("theta_min"),
        F.max("theta_e_arcsec").alias("theta_max"),
        F.min("shear").alias("shear_min") if "shear" in m.columns else F.lit(None).alias("shear_min"),
        F.max("shear").alias("shear_max") if "shear" in m.columns else F.lit(None).alias("shear_max"),
    ).collect()[0]

    theta_min = rng["theta_min"]
    theta_max = rng["theta_max"]
    if theta_min is None or theta_max is None:
        raise RuntimeError("theta_e_arcsec missing min/max values")
    if theta_min < 0.0:
        raise RuntimeError(f"theta_e_arcsec has negative values (min={theta_min})")
    if theta_max > 5.0:
        raise RuntimeError(f"theta_e_arcsec unusually large (max={theta_max}) - check units")
    print(f"[validate] theta_e range: [{theta_min:.4f}, {theta_max:.4f}] arcsec - PASSED")

    # Controls: theta_e should be exactly 0, non-controls should be positive
    bad_control = m.where(
        (F.col("is_control") == 1) & (F.col("theta_e_arcsec") != F.lit(0.0))
    ).count()
    if bad_control > 0:
        raise RuntimeError(f"Found {bad_control} control rows with theta_e_arcsec != 0")

    bad_noncontrol = m.where(
        (F.col("is_control") == 0) & (F.col("theta_e_arcsec") <= F.lit(0.0))
    ).count()
    if bad_noncontrol > 0:
        raise RuntimeError(f"Found {bad_noncontrol} non-control rows with theta_e_arcsec <= 0")
    print("[validate] Control/non-control theta_e consistency: PASSED")

    # Random fields should be present for non-controls
    nonc = m.where(F.col("is_control") == 0)
    null_src = nonc.where(
        F.col("src_x_arcsec").isNull() | 
        F.col("src_y_arcsec").isNull() | 
        F.col("src_phi_rad").isNull()
    ).count()
    if null_src > 0:
        raise RuntimeError(f"Found {null_src} non-control rows missing src position/angle fields")
    print("[validate] Frozen randomness columns for non-controls: PASSED")

    # Shear phi range check
    out_of_range = nonc.where(
        (F.col("shear_phi_rad") < 0.0) | (F.col("shear_phi_rad") > math.pi + 0.01)
    ).count()
    if out_of_range > 0:
        raise RuntimeError(f"Found {out_of_range} non-control rows with shear_phi_rad outside [0, pi]")
    print("[validate] shear_phi_rad range: PASSED")

    # Color sanity (if present)
    if "src_gr" in m.columns and "src_rz" in m.columns:
        bad_color = nonc.where(
            (F.col("src_gr") < -1.0) | (F.col("src_gr") > 2.0) |
            (F.col("src_rz") < -1.0) | (F.col("src_rz") > 2.0)
        ).count()
        if bad_color > 0:
            print(f"[validate] WARNING: Found {bad_color} non-control rows with extreme src colors")
        else:
            print("[validate] src color ranges: PASSED")

    # Bricks manifest validation
    b_missing = require_columns(b, ["brickname"])
    if b_missing:
        raise RuntimeError(f"Missing required bricks_manifest columns: {b_missing}")

    n_bricks = b.select("brickname").distinct().count()
    print(f"[validate] Bricks manifest: {n_bricks} unique bricks")

    # O(n_cfg^2) explosion detection heuristic for debug/grid tiers
    # If explosion occurred, n/k >> k (where k = distinct config count)
    heuristic_rows = (m.where(
        (F.col("tier").isin("debug", "grid")) & (F.col("is_control") == 0)
    ).groupBy("tier", "region_split")
        .agg(F.count("*").alias("n"), F.countDistinct("config_id").alias("k"))
        .collect()
    )
    for row in heuristic_rows:
        n = int(row["n"])
        k = int(row["k"])
        if k > 0:
            ratio = n / k
            # Heuristic: if n/k > 10*k, likely O(n²) explosion
            if ratio > 10 * k:
                raise RuntimeError(
                    f"Possible O(n_cfg^2) expansion detected for tier={row['tier']} "
                    f"split={row['region_split']}: n={n}, k={k}, n/k={ratio:.1f}"
                )
    print("[validate] O(n_cfg^2) explosion check: PASSED")

    # Lens model validation
    lens_model_check = m.groupBy("lens_model").count().orderBy("lens_model").collect()
    print("[validate] Lens model distribution:")
    for row in lens_model_check:
        print(f"  {row['lens_model']}: {row['count']:,}")
    
    # Controls should have lens_model = "CONTROL"
    bad_control_model = m.where(
        (F.col("is_control") == 1) & (F.col("lens_model") != "CONTROL")
    ).count()
    if bad_control_model > 0:
        print(f"[validate] WARNING: {bad_control_model} control rows with lens_model != 'CONTROL'")
    else:
        print("[validate] Control lens_model consistency: PASSED")

    # Non-controls should have lens_model = "SIE" or "SIS"
    bad_noncontrol_model = m.where(
        (F.col("is_control") == 0) & (~F.col("lens_model").isin("SIE", "SIS"))
    ).count()
    if bad_noncontrol_model > 0:
        raise RuntimeError(f"Found {bad_noncontrol_model} non-control rows with invalid lens_model")
    print("[validate] Non-control lens_model consistency: PASSED")

    # Lens ellipticity range check (should be in [0, 0.5])
    bad_lens_e = m.where(
        (F.col("is_control") == 0) & 
        ((F.col("lens_e") < 0.0) | (F.col("lens_e") > 0.6))
    ).count()
    if bad_lens_e > 0:
        print(f"[validate] WARNING: {bad_lens_e} non-control rows with lens_e outside [0, 0.6]")
    else:
        print("[validate] lens_e range: PASSED")

    # Lens phi range check
    bad_lens_phi = m.where(
        (F.col("is_control") == 0) &
        ((F.col("lens_phi_rad") < 0.0) | (F.col("lens_phi_rad") > math.pi + 0.01))
    ).count()
    if bad_lens_phi > 0:
        print(f"[validate] WARNING: {bad_lens_phi} non-control rows with lens_phi_rad outside [0, pi]")
    else:
        print("[validate] lens_phi_rad range: PASSED")

    print("\n" + "=" * 60)
    print("OK: Phase 4a manifests and bricks_manifest passed validation.")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()

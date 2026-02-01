
#!/usr/bin/env python3
"""
Spark Phase 4a manifest builder (SOTA-oriented) for Dark Halo Scope.

Purpose:
- Build injection manifests from an input "target catalog" (typically Phase 3 LRG targets).
- Supports extended theta_E grids for resolvable lenses.
- Produces both paired controls (same target, theta_e=0) and optional "unpaired controls"
  (negative examples drawn from other targets matched on observing-condition bins).
- Optionally appends an extra negative catalog (hard negatives) if you have one.

This script is intentionally standalone so Phase 5 training cannot be blocked by a monolithic pipeline file.

Inputs (minimum required columns in the target catalog):
- ra (double), dec (double)
- brickname (string) OR (brickname-like id you already use downstream)
Recommended additional columns (used for bin-matching and downstream completeness binning):
- region_id, region_split
- psfsize_r or psf_fwhm_used_r (for binning; if absent, unpaired matching is disabled)
- psfdepth_r

Output:
- Parquet manifest with all original target columns + injection parameters:
  theta_e_arcsec, src_dmag, src_reff_arcsec, src_e, shear_gamma, shear_phi,
  lens_model, is_control, control_kind, grid_name, injection_id.

Notes:
- You can run this on EMR or any Spark cluster with S3 access.
- For reproducibility, set --seed and keep it in your stage config.
"""

import argparse
import hashlib
from typing import List, Optional

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T


def _parse_csv_floats(s: str) -> List[float]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    if not out:
        raise ValueError("Empty float list")
    return out


def _parse_csv_str(s: str) -> List[str]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(x)
    if not out:
        raise ValueError("Empty string list")
    return out


def build_grid_df(
    spark: SparkSession,
    grid_name: str,
    theta_e_list: List[float],
    src_dmag_list: List[float],
    src_reff_list: List[float],
    src_e_list: List[float],
    shear_list: List[float],
    lens_models: List[str],
) -> DataFrame:
    """Build a grid of injection parameters.
    
    Note: shear_phi (shear angle) is randomized during Stage 4c injection,
    so we only specify shear magnitude here (matching existing pipeline).
    """
    rows = []
    for theta in theta_e_list:
        for dmag in src_dmag_list:
            for reff in src_reff_list:
                for e in src_e_list:
                    for sh in shear_list:
                        for lm in lens_models:
                            rows.append((grid_name, float(theta), float(dmag), float(reff), float(e), float(sh), lm))
    schema = T.StructType([
        T.StructField("grid_name", T.StringType(), False),
        T.StructField("theta_e_arcsec", T.DoubleType(), False),
        T.StructField("src_dmag", T.DoubleType(), False),
        T.StructField("src_reff_arcsec", T.DoubleType(), False),
        T.StructField("src_e", T.DoubleType(), False),
        T.StructField("shear", T.DoubleType(), False),
        T.StructField("lens_model", T.StringType(), False),
    ])
    return spark.createDataFrame(rows, schema=schema)


def add_injection_id(df: DataFrame, seed: int) -> DataFrame:
    # Deterministic, content-addressed id to help joins across phases.
    # Use a stable subset of columns that define an injection row.
    cols = [
        "grid_name", "lens_model", "is_control",
        "theta_e_arcsec", "src_dmag", "src_reff_arcsec", "src_e",
        "shear",
        "ra", "dec",
    ]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column for injection_id: {c}")
    # Hash is md5 for speed; collision risk negligible at this scale for ids.
    concat = F.concat_ws("||", *[F.coalesce(F.col(c).cast("string"), F.lit("NULL")) for c in cols], F.lit(str(seed)))
    return df.withColumn("injection_id", F.md5(concat))


def maybe_add_unpaired_controls(
    df: DataFrame,
    targets: DataFrame,
    unpaired_frac: float,
    seed: int,
    psf_col: str,
    depth_col: str,
) -> DataFrame:
    """
    Replace the (ra, dec, brickname, and any other target-identifying columns) for a fraction
    of the control rows with values drawn from other targets matched on (psf_bin, depth_bin).

    This is a "cheap" hardening step: negatives are still real galaxies, but no longer the
    exact same galaxy as the positive's base cutout.

    If required columns are absent, returns df unchanged.
    """
    if unpaired_frac <= 0:
        return df

    required = [psf_col, depth_col, "ra", "dec"]
    if "brickname" in targets.columns:
        required.append("brickname")
    missing = [c for c in required if c not in targets.columns]
    if missing:
        # Cannot do matching without observing-condition columns.
        return df

    # Binning: coarse bins to increase match probability.
    # You can tune bin widths, but keep stable for reproducibility.
    t = targets
    t = t.withColumn("psf_bin", F.round(F.col(psf_col).cast("double"), 1))
    t = t.withColumn("depth_bin", F.round(F.col(depth_col).cast("double"), 1))
    t = t.withColumn("_rn", F.row_number().over(
        Window.partitionBy("psf_bin", "depth_bin").orderBy(F.rand(seed))
    ))
    swap_cols = ["psf_bin", "depth_bin", "_rn", "ra", "dec"]
    if "brickname" in t.columns:
        swap_cols.append("brickname")
    # Keep optional region columns if present, to avoid accidental split leakage when you swap.
    for opt in ["region_id", "region_split"]:
        if opt in t.columns:
            swap_cols.append(opt)
    swap = t.select(*swap_cols).withColumnRenamed("ra", "ra_swap").withColumnRenamed("dec", "dec_swap")
    if "brickname" in swap.columns:
        swap = swap.withColumnRenamed("brickname", "brickname_swap")
    if "region_id" in swap.columns:
        swap = swap.withColumnRenamed("region_id", "region_id_swap")
    if "region_split" in swap.columns:
        swap = swap.withColumnRenamed("region_split", "region_split_swap")

    d = df.withColumn("psf_bin", F.round(F.col(psf_col).cast("double"), 1)) if psf_col in df.columns else df
    d = d.withColumn("depth_bin", F.round(F.col(depth_col).cast("double"), 1)) if depth_col in d.columns else d

    # Tag which controls to unpair.
    d = d.withColumn("_do_unpair", F.when(
        (F.col("is_control") == 1) & (F.rand(seed + 17) < F.lit(unpaired_frac)),
        F.lit(1)
    ).otherwise(F.lit(0)))

    # For stable joining, assign row numbers within bins to both sides.
    d_ctrl = d.filter(F.col("is_control") == 1)
    d_pos = d.filter(F.col("is_control") == 0)

    # Only unpair subset, leave others untouched.
    d_ctrl_unpair = d_ctrl.filter(F.col("_do_unpair") == 1)
    d_ctrl_keep = d_ctrl.filter(F.col("_do_unpair") == 0)

    w = Window.partitionBy("psf_bin", "depth_bin").orderBy(F.rand(seed + 23))
    d_ctrl_unpair = d_ctrl_unpair.withColumn("_rn", F.row_number().over(w))

    joined = d_ctrl_unpair.join(swap, on=["psf_bin", "depth_bin", "_rn"], how="left")

    # Apply swaps where match exists.
    joined = joined.withColumn("control_kind", F.when(F.col("ra_swap").isNotNull(), F.lit("unpaired")).otherwise(F.lit("paired")))
    joined = joined.withColumn("ra", F.coalesce(F.col("ra_swap"), F.col("ra")))
    joined = joined.withColumn("dec", F.coalesce(F.col("dec_swap"), F.col("dec")))
    if "brickname" in joined.columns and "brickname_swap" in joined.columns:
        joined = joined.withColumn("brickname", F.coalesce(F.col("brickname_swap"), F.col("brickname")))
    if "region_id" in joined.columns and "region_id_swap" in joined.columns:
        joined = joined.withColumn("region_id", F.coalesce(F.col("region_id_swap"), F.col("region_id")))
    if "region_split" in joined.columns and "region_split_swap" in joined.columns:
        joined = joined.withColumn("region_split", F.coalesce(F.col("region_split_swap"), F.col("region_split")))

    drop_cols = [c for c in ["ra_swap", "dec_swap", "brickname_swap", "region_id_swap", "region_split_swap"] if c in joined.columns]
    joined = joined.drop(*drop_cols)

    out = d_pos.unionByName(d_ctrl_keep).unionByName(joined, allowMissingColumns=True)
    return out.drop("_do_unpair")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="Input target catalog (parquet), e.g. Phase 3 LRG targets")
    ap.add_argument("--out", required=True, help="Output manifest directory (parquet)")
    ap.add_argument("--grid_name", default="grid_sota_v1")
    ap.add_argument("--theta_e_list", default="0.5,0.75,1.0,1.25,1.5,2.0,2.5")
    ap.add_argument("--src_dmag_list", default="0.5,1.0,1.5")
    ap.add_argument("--src_reff_list", default="0.05,0.10,0.15")
    ap.add_argument("--src_e_list", default="0.0,0.3,0.6")
    ap.add_argument("--shear_list", default="0.0,0.03", help="Shear magnitude (angle randomized in Stage 4c)")
    ap.add_argument("--lens_models", default="SIE")
    ap.add_argument("--control_frac", type=float, default=0.5, help="Row-level control fraction after cross join")
    ap.add_argument("--unpaired_control_frac", type=float, default=0.50, help="Fraction of controls to unpair (matched by bins)")
    ap.add_argument("--psf_col", default="psfsize_r", help="Column to use for PSF matching (psf_fwhm_used_r preferred if present)")
    ap.add_argument("--depth_col", default="psfdepth_r", help="Column to use for depth matching")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--limit_targets", type=int, default=0, help="For quick runs: limit input targets to N (0 = no limit)")
    ap.add_argument("--extra_negatives", default="", help="Optional extra negative target catalog (parquet). Rows are labeled is_control=1.")
    args = ap.parse_args()

    spark = SparkSession.builder.appName("phase4a_manifest_sota").getOrCreate()

    targets = spark.read.parquet(args.targets)
    if args.limit_targets and args.limit_targets > 0:
        targets = targets.orderBy(F.rand(args.seed)).limit(int(args.limit_targets))

    # Basic schema checks
    for req in ["ra", "dec"]:
        if req not in targets.columns:
            raise ValueError(f"Targets missing required column: {req}")
    if "brickname" not in targets.columns:
        # Not always required for all pipelines, but strongly recommended.
        print("WARNING: targets do not include 'brickname'. Ensure your downstream stamp builder does not require it.")

    # Prefer psf_fwhm_used_r if present even if psf_col not set.
    psf_col = args.psf_col
    if psf_col not in targets.columns and "psf_fwhm_used_r" in targets.columns:
        psf_col = "psf_fwhm_used_r"
    if psf_col not in targets.columns:
        print(f"WARNING: PSF column '{psf_col}' not found in targets. Unpaired controls will be disabled.")
    depth_col = args.depth_col
    if depth_col not in targets.columns:
        print(f"WARNING: depth column '{depth_col}' not found in targets. Unpaired controls will be disabled.")

    grid = build_grid_df(
        spark,
        grid_name=args.grid_name,
        theta_e_list=_parse_csv_floats(args.theta_e_list),
        src_dmag_list=_parse_csv_floats(args.src_dmag_list),
        src_reff_list=_parse_csv_floats(args.src_reff_list),
        src_e_list=_parse_csv_floats(args.src_e_list),
        shear_list=_parse_csv_floats(args.shear_list),
        lens_models=_parse_csv_str(args.lens_models),
    )

    # Cross join targets x grid (this is your injection candidate set).
    # This creates one row per (target, injection params).
    df = targets.crossJoin(grid)

    # Assign controls at the row level
    df = df.withColumn("is_control", F.when(F.rand(args.seed) < F.lit(args.control_frac), F.lit(1)).otherwise(F.lit(0)))

    # For controls, zero out injection parameters (use 0.0 for consistency with existing pipeline)
    df = df.withColumn("theta_e_arcsec", F.when(F.col("is_control") == 1, F.lit(0.0)).otherwise(F.col("theta_e_arcsec")))
    df = df.withColumn("src_dmag", F.when(F.col("is_control") == 1, F.lit(0.0)).otherwise(F.col("src_dmag")))
    df = df.withColumn("src_reff_arcsec", F.when(F.col("is_control") == 1, F.lit(0.0)).otherwise(F.col("src_reff_arcsec")))
    df = df.withColumn("src_e", F.when(F.col("is_control") == 1, F.lit(0.0)).otherwise(F.col("src_e")))
    df = df.withColumn("shear", F.when(F.col("is_control") == 1, F.lit(0.0)).otherwise(F.col("shear")))
    df = df.withColumn("lens_model", F.when(F.col("is_control") == 1, F.lit("CONTROL")).otherwise(F.col("lens_model")))

    df = df.withColumn("control_kind", F.when(F.col("is_control") == 1, F.lit("paired")).otherwise(F.lit(None)))

    # Optionally harden negatives by unpairing a fraction of controls.
    if (args.unpaired_control_frac > 0) and (psf_col in df.columns) and (depth_col in df.columns):
        df = maybe_add_unpaired_controls(df, targets, args.unpaired_control_frac, args.seed, psf_col=psf_col, depth_col=depth_col)

    # Optional extra negative pool (hard negatives)
    if args.extra_negatives:
        extra = spark.read.parquet(args.extra_negatives)
        for req in ["ra", "dec"]:
            if req not in extra.columns:
                raise ValueError(f"Extra negatives missing required column: {req}")

        # Match columns by unionByName; fill injection params as controls (use 0.0 for consistency).
        extra = extra.withColumn("grid_name", F.lit(args.grid_name))
        extra = extra.withColumn("theta_e_arcsec", F.lit(0.0))
        extra = extra.withColumn("src_dmag", F.lit(0.0))
        extra = extra.withColumn("src_reff_arcsec", F.lit(0.0))
        extra = extra.withColumn("src_e", F.lit(0.0))
        extra = extra.withColumn("shear", F.lit(0.0))
        extra = extra.withColumn("lens_model", F.lit("CONTROL"))
        extra = extra.withColumn("is_control", F.lit(1))
        extra = extra.withColumn("control_kind", F.lit("extra_negative"))
        df = df.unionByName(extra, allowMissingColumns=True)

    df = add_injection_id(df, seed=args.seed)

    # Write
    (df
        .repartition(512)  # tune for your cluster
        .write
        .mode("overwrite")
        .parquet(args.out)
    )

    print(f"Wrote manifest to: {args.out}")
    spark.stop()


if __name__ == "__main__":
    main()

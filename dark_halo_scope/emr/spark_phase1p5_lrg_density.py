#!/usr/bin/env python

"""
PySpark Phase 1.5: LRG density per brick from DR10 SWEEPs.

This job is designed to run on an EMR cluster. It:

  1. Reads a list of DR10 SWEEP FITS URLs or S3 paths.
  2. Distributes these paths across Spark executors.
  3. For each SWEEP file:
       - Downloads to local /tmp.
       - Loads the FITS table (HDU 1).
       - Applies DESI-like LRG color/magnitude cuts.
       - Groups by BRICKNAME and returns (brickname, lrg_count) pairs.
  4. Reduces counts per brick across all SWEEPs.
  5. Writes a compact per-brick CSV to S3 with columns:
       - brickname
       - lrg_count

You will combine this with brick QA info (seeing, depth, area_deg2)
on your laptop or another step to compute LRG densities and select regions.
"""

import argparse
import os
import tempfile
from typing import Iterable, List, Tuple

import numpy as np
from astropy.io import fits
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType


def nanomaggies_to_mag(flux: np.ndarray, zero_point: float = 22.5) -> np.ndarray:
    """Convert nanomaggies to AB magnitudes. flux <= 0 => NaN."""
    flux = np.array(flux, dtype=float)
    mag = np.full_like(flux, np.nan, dtype=float)
    mask = flux > 0
    mag[mask] = zero_point - 2.5 * np.log10(flux[mask])
    return mag


def build_lrg_mask(
    flux_r: np.ndarray,
    flux_z: np.ndarray,
    flux_w1: np.ndarray,
    lrg_z_mag_max: float,
    lrg_min_r_minus_z: float,
    lrg_min_z_minus_w1: float,
) -> np.ndarray:
    """
    DESI-like LRG proxy:

      z < lrg_z_mag_max
      r - z > lrg_min_r_minus_z
      z - W1 > lrg_min_z_minus_w1
    """
    mag_r = nanomaggies_to_mag(flux_r)
    mag_z = nanomaggies_to_mag(flux_z)
    mag_w1 = nanomaggies_to_mag(flux_w1)

    r_minus_z = mag_r - mag_z
    z_minus_w1 = mag_z - mag_w1

    mask = (
        (mag_z < lrg_z_mag_max)
        & (r_minus_z > lrg_min_r_minus_z)
        & (z_minus_w1 > lrg_min_z_minus_w1)
    )
    return mask


def process_single_sweep(
    path: str,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    lrg_z_mag_max: float,
    lrg_min_r_minus_z: float,
    lrg_min_z_minus_w1: float,
) -> List[Tuple[str, int]]:
    """
    Load a single SWEEP FITS from a local or remote path, apply the
    LRG mask, and return (brickname, count) pairs.

    The path can be:
      - A local filesystem path (e.g. /mnt/.../sweep-*.fits)
      - An S3 URL supported by EMR's Hadoop client (s3:// or s3a://)
      - An HTTP(S) URL (we download it explicitly to /tmp)
    """
    import urllib.parse
    import shutil
    import requests

    # Normalize: if HTTP(S), download to local temp; otherwise assume Spark
    # can read the path via the local filesystem.
    url = urllib.parse.urlparse(path)
    local_path = path
    temp_file_created = False

    if url.scheme in ("http", "https"):
        # Download once per executor task to a temp file
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
            local_path = tmp.name
            temp_file_created = True
            with requests.get(path, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                shutil.copyfileobj(resp.raw, tmp)

    if not os.path.exists(local_path):
        # Nothing to do
        return []

    try:
        with fits.open(local_path, memmap=True) as hdul:
            data = hdul[1].data

            # Normalize column names to lower case
            colnames = [c.lower() for c in data.columns.names]
            lower_to_idx = {name: i for i, name in enumerate(colnames)}

            # Required columns
            required = ["ra", "dec", "flux_r", "flux_z", "flux_w1", "brickname"]
            missing = [c for c in required if c not in lower_to_idx]
            if missing:
                return []

            ra = np.array(data.field(lower_to_idx["ra"]))
            dec = np.array(data.field(lower_to_idx["dec"]))
            flux_r = np.array(data.field(lower_to_idx["flux_r"]))
            flux_z = np.array(data.field(lower_to_idx["flux_z"]))
            flux_w1 = np.array(data.field(lower_to_idx["flux_w1"]))
            brickname = np.array(data.field(lower_to_idx["brickname"])).astype(str)

            # Optional: footprint filter
            mask_sky = (
                (ra >= ra_min)
                & (ra <= ra_max)
                & (dec >= dec_min)
                & (dec <= dec_max)
            )

            if not np.any(mask_sky):
                return []

            ra = ra[mask_sky]
            dec = dec[mask_sky]
            flux_r = flux_r[mask_sky]
            flux_z = flux_z[mask_sky]
            flux_w1 = flux_w1[mask_sky]
            brickname = brickname[mask_sky]

            # LRG selection
            lrg_mask = build_lrg_mask(
                flux_r=flux_r,
                flux_z=flux_z,
                flux_w1=flux_w1,
                lrg_z_mag_max=lrg_z_mag_max,
                lrg_min_r_minus_z=lrg_min_r_minus_z,
                lrg_min_z_minus_w1=lrg_min_z_minus_w1,
            )

            if not np.any(lrg_mask):
                return []

            bricknames_lrg = brickname[lrg_mask]

            # Count per brickname
            unique, counts = np.unique(bricknames_lrg, return_counts=True)
            return list(zip(unique.tolist(), counts.astype(int).tolist()))
    finally:
        # Clean up temp file if we created one
        if temp_file_created and os.path.exists(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass


def process_sweep_partition(
    paths: Iterable[str],
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    lrg_z_mag_max: float,
    lrg_min_r_minus_z: float,
    lrg_min_z_minus_w1: float,
) -> Iterable[Tuple[str, int]]:
    """
    MapPartitions function: processes multiple SWEEP paths inside a single
    executor partition.
    """
    all_pairs: List[Tuple[str, int]] = []
    for path in paths:
        path = path.strip()
        if not path:
            continue
        try:
            pairs = process_single_sweep(
                path=path,
                ra_min=ra_min,
                ra_max=ra_max,
                dec_min=dec_min,
                dec_max=dec_max,
                lrg_z_mag_max=lrg_z_mag_max,
                lrg_min_r_minus_z=lrg_min_r_minus_z,
                lrg_min_z_minus_w1=lrg_min_z_minus_w1,
            )
            all_pairs.extend(pairs)
        except Exception as exc:
            # Avoid killing the job for one bad file; just log to executor stderr
            print(f"[WARN] Failed processing SWEEP {path}: {exc!r}")
    return all_pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.5: EMR PySpark job to compute LRG counts per brick."
    )

    parser.add_argument(
        "--sweep-index-path",
        required=True,
        help=(
            "Path to a newline-delimited file listing SWEEP FITS paths or URLs. "
            "Can be local (on EMR HDFS/instance) or an S3 URL supported by Spark."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help=(
            "S3 prefix for outputs, e.g. s3://my-bucket/dark_halo_scope/phase1p5. "
            "The job will create output_prefix + '/lrg_counts_per_brick/'."
        ),
    )

    # Phase 1.5 footprint and LRG proxy cuts
    parser.add_argument("--ra-min", type=float, default=150.0)
    parser.add_argument("--ra-max", type=float, default=250.0)
    parser.add_argument("--dec-min", type=float, default=0.0)
    parser.add_argument("--dec-max", type=float, default=30.0)

    parser.add_argument("--lrg-z-mag-max", type=float, default=20.4)
    parser.add_argument("--lrg-min-r-minus-z", type=float, default=0.4)
    parser.add_argument("--lrg-min-z-minus-w1", type=float, default=1.6)

    parser.add_argument(
        "--num-partitions",
        type=int,
        default=64,
        help="Number of Spark partitions to use for SWEEP file list.",
    )

    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName("dark-halo-scope-phase1p5-lrg-density")
        .getOrCreate()
    )
    sc = spark.sparkContext

    # Read sweep index as lines (paths)
    paths_rdd = sc.textFile(args.sweep_index_path, minPartitions=args.num_partitions)

    # MapPartitions: (brickname, count)
    pairs_rdd = paths_rdd.mapPartitions(
        lambda it: process_sweep_partition(
            paths=it,
            ra_min=args.ra_min,
            ra_max=args.ra_max,
            dec_min=args.dec_min,
            dec_max=args.dec_max,
            lrg_z_mag_max=args.lrg_z_mag_max,
            lrg_min_r_minus_z=args.lrg_min_r_minus_z,
            lrg_min_z_minus_w1=args.lrg_min_z_minus_w1,
        )
    )

    # Reduce by brickname
    reduced_rdd = pairs_rdd.reduceByKey(lambda a, b: a + b)

    # Convert to a small Spark DataFrame and write to S3 as CSV
    schema = StructType(
        [
            StructField("brickname", StringType(), nullable=False),
            StructField("lrg_count", LongType(), nullable=False),
        ]
    )

    df = spark.createDataFrame(reduced_rdd, schema=schema)

    output_dir = os.path.join(args.output_prefix.rstrip("/"), "lrg_counts_per_brick")
    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(output_dir)
    )

    spark.stop()


if __name__ == "__main__":
    main()


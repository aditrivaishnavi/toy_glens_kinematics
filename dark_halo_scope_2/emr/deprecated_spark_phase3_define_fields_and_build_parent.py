#!/usr/bin/env python3
"""
Dark Halo Scope: Phase 3 (EMR / PySpark)
=======================================

Single Spark job that does BOTH:
  1) Region selection: pick top-K regions for multiple ranking modes using Phase 2 region summaries.
  2) Parent sample build: scan DR10 South SWEEP FITS files from an S3 cache (supports .fits and .fits.gz),
     select v3 LRGs, and attach (ranking_mode, region_id, region_rank).

Inputs (all in S3):
  - phase2_analysis/{variant}/phase2_regions_summary.csv
  - phase2_analysis/{variant}/phase2_regions_bricks.csv
  - sweep_index.txt : list of s3://... sweep FITS (one per line)

Outputs (to output_s3):
  - {output_s3}/{variant}/{mode}/phase3_target_bricks/ (CSV)
  - {output_s3}/{variant}/{mode}/phase3_selected_regions/ (CSV)
  - {output_s3}/{variant}/parent_catalog/ (Parquet partitioned by ranking mode)
  - {output_s3}/{variant}/{mode}/phase3_lrg_parent_catalog/ (CSV per mode)

Scientific notes:
  - LRG magnitudes are computed from nanomaggies WITHOUT Galactic-extinction correction, matching Phase 2 Spark job.
  - Parent selection uses the Phase 2 v3 definition (hard-coded numeric thresholds below) and excludes TYPE='PSF'.
  - Pure "density" ranking is often dominated by pathological bricks (high dust / poor PSF / survey edges).
    The psf_weighted mode exists specifically to downweight those systematics.

Python: 3.9+
"""

from __future__ import annotations

import argparse
import gc
import gzip
import os
import shutil
import sys
import tempfile
import time
import urllib.parse
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Set, Tuple

import boto3
from boto3.s3.transfer import TransferConfig
import numpy as np
from astropy.io import fits

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.window import Window
from pyspark import StorageLevel


# ---------------------------
# Hard-coded LRG hypergrid thresholds (must match Phase 2)
# ---------------------------

@dataclass(frozen=True)
class LRGVariant:
    name: str
    z_mag_max: float
    rz_min: float
    zw1_min: float


LRG_VARIANTS: List[LRGVariant] = [
    LRGVariant("v1_pure_massive",   z_mag_max=20.0, rz_min=0.5, zw1_min=1.6),
    LRGVariant("v2_baseline_dr10",  z_mag_max=20.4, rz_min=0.4, zw1_min=1.6),
    LRGVariant("v3_color_relaxed",  z_mag_max=20.4, rz_min=0.4, zw1_min=0.8),
    LRGVariant("v4_mag_relaxed",    z_mag_max=21.0, rz_min=0.4, zw1_min=0.8),
    LRGVariant("v5_very_relaxed",   z_mag_max=21.5, rz_min=0.3, zw1_min=0.8),
]

BASELINE_VARIANT = "v3_color_relaxed"


# ---------------------------
# Utilities: S3 + local caching on executors
# ---------------------------

def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Not an s3:// URI: {s3_uri}")
    parsed = urllib.parse.urlparse(s3_uri)
    return parsed.netloc, parsed.path.lstrip("/")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _s3_object_exists(s3_client, bucket: str, key: str) -> bool:
    """Check if an S3 object exists using head_object (fast, no download)."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError:
        return False


def s3_download_to_local(
    s3_client,
    s3_uri: str,
    local_cache_dir: str,
    transfer_config: Optional[TransferConfig] = None,
) -> str:
    """
    Download an S3 object to a local cache file (executor-local disk).
    Returns the local file path.
    Uses TransferConfig for efficient large file transfers.
    """
    bucket, key = parse_s3_uri(s3_uri)
    ensure_dir(local_cache_dir)
    fname = os.path.basename(key) or "object"
    local_path = os.path.join(local_cache_dir, fname)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    if transfer_config:
        s3_client.download_file(bucket, key, local_path, Config=transfer_config)
    else:
        s3_client.download_file(bucket, key, local_path)
    return local_path


def download_sweep_with_cache(
    s3_client,
    sweep_url: str,
    s3_sweep_cache_prefix: Optional[str],
    local_cache_dir: str,
    transfer_config: Optional[TransferConfig] = None,
) -> str:
    """
    Download a sweep FITS file, checking S3 cache first.
    
    Priority:
    1. Check S3 sweep cache for gzipped file ({cache_prefix}/{basename}.gz)
    2. If not in cache, download from original sweep_url
    
    Returns local path to the (possibly gzipped) FITS file.
    """
    ensure_dir(local_cache_dir)
    
    # Extract base filename
    basename = os.path.basename(sweep_url)
    if basename.endswith(".gz"):
        basename_no_gz = basename[:-3]
    else:
        basename_no_gz = basename
    
    # Local paths
    local_path_gz = os.path.join(local_cache_dir, basename_no_gz + ".gz")
    local_path_raw = os.path.join(local_cache_dir, basename_no_gz)
    
    # Check if already cached locally (prefer gzipped)
    if os.path.exists(local_path_gz) and os.path.getsize(local_path_gz) > 0:
        return local_path_gz
    if os.path.exists(local_path_raw) and os.path.getsize(local_path_raw) > 0:
        return local_path_raw
    
    # Check S3 sweep cache if prefix provided
    if s3_sweep_cache_prefix:
        cache_bucket, cache_key_prefix = parse_s3_uri(s3_sweep_cache_prefix)
        cache_key_prefix = cache_key_prefix.rstrip("/")
        gz_key = f"{cache_key_prefix}/{basename_no_gz}.gz"
        
        if _s3_object_exists(s3_client, cache_bucket, gz_key):
            print(f"[CACHE-HIT] Found in S3 cache: s3://{cache_bucket}/{gz_key}", file=sys.stderr)
            sys.stderr.flush()
            if transfer_config:
                s3_client.download_file(cache_bucket, gz_key, local_path_gz, Config=transfer_config)
            else:
                s3_client.download_file(cache_bucket, gz_key, local_path_gz)
            return local_path_gz
    
    # Not in cache - download from original URL
    print(f"[CACHE-MISS] Downloading from source: {sweep_url}", file=sys.stderr)
    sys.stderr.flush()
    
    if sweep_url.startswith("s3://"):
        # S3 URL
        local_path = s3_download_to_local(s3_client, sweep_url, local_cache_dir, transfer_config)
        return local_path
    elif sweep_url.startswith("http://") or sweep_url.startswith("https://"):
        # HTTP URL - use urllib
        import urllib.request
        target_path = local_path_gz if sweep_url.endswith(".gz") else local_path_raw
        urllib.request.urlretrieve(sweep_url, target_path)
        return target_path
    else:
        raise ValueError(f"Unsupported URL scheme: {sweep_url}")


def maybe_decompress_gzip(local_path: str) -> str:
    """
    If local_path ends with .gz, decompress to a temp file and return that path.
    Otherwise return local_path.
    Uses streaming decompression to avoid loading entire file into memory.
    """
    if not local_path.endswith(".gz"):
        return local_path
    out_path = local_path[:-3]
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    # Use streaming copy to avoid memory spike for large files
    with gzip.open(local_path, "rb") as fin, open(out_path, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)  # 16MB chunks
    return out_path


def nanomaggies_to_mag(flux: np.ndarray, zero_point: float = 22.5) -> np.ndarray:
    """
    Convert nanomaggies to AB magnitudes.
    For flux <= 0, returns NaN.
    """
    flux = np.asarray(flux, dtype=np.float64)
    out = np.full_like(flux, np.nan, dtype=np.float64)
    m = flux > 0
    out[m] = zero_point - 2.5 * np.log10(flux[m])
    return out


def get_col(data: np.ndarray, names: List[str], desired: List[str]) -> np.ndarray:
    """
    Find a column in a FITS recarray by case-insensitive match against 'desired'.
    """
    lower = {n.lower(): n for n in names}
    for d in desired:
        k = d.lower()
        if k in lower:
            return data[lower[k]]
    raise KeyError(f"Missing FITS column. Tried {desired}. Have {names[:30]}...")


# ---------------------------
# Region scoring
# ---------------------------

def compute_depth_normalization_quantiles(regions_df, depth_col: str) -> Tuple[float, float]:
    # robust linear normalization using p10 and p90
    qs = regions_df.approxQuantile(depth_col, [0.10, 0.90], 0.001)
    if len(qs) != 2:
        return (0.0, 1.0)
    lo, hi = qs
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def add_region_scores(
    regions_df,
    variant: str,
    psf_ref: float,
    sigma_psf: float,
    k_ebv: float,
    depth_p10: float,
    depth_p90: float,
) :
    """
    Add score columns for multiple ranking modes.
    """
    dens_col = f"mean_lrg_density_{variant}"
    cnt_col  = f"total_n_lrg_{variant}"

    df = regions_df

    df = df.withColumn("score_density", F.col(dens_col))
    df = df.withColumn("score_n_lrg", F.col(cnt_col))
    df = df.withColumn("score_area_weighted", F.col(dens_col) * F.sqrt(F.col("total_area_deg2")))

    # weights
    w_psf = F.exp(-(F.col("median_psf_r_arcsec") - F.lit(psf_ref)) / F.lit(sigma_psf))
    w_psf = F.when(w_psf > 1.0, 1.0).when(w_psf < 0.0, 0.0).otherwise(w_psf)

    w_ebv = F.exp(-F.lit(k_ebv) * F.col("median_ebv"))

    w_depth = (F.col("median_psfdepth_r") - F.lit(depth_p10)) / F.lit(depth_p90 - depth_p10)
    w_depth = F.when(w_depth > 1.0, 1.0).when(w_depth < 0.0, 0.0).otherwise(w_depth)

    df = df.withColumn("score_psf_weighted", F.col(dens_col) * w_psf * w_depth * w_ebv)
    return df


def select_top_regions(
    regions_df,
    mode: str,
    k: int,
) :
    score_col = {
        "density": "score_density",
        "n_lrg": "score_n_lrg",
        "area_weighted": "score_area_weighted",
        "psf_weighted": "score_psf_weighted",
    }[mode]

    w = F.row_number().over(Window.orderBy(F.col(score_col).desc()))

    top = regions_df.select(
        "region_id",
        "n_bricks",
        "total_area_deg2",
        "total_n_gal",
        "ra_center_deg",
        "dec_center_deg",
        "median_psf_r_arcsec",
        "median_psfdepth_r",
        "median_ebv",
        score_col,
    ).withColumn("phase3_ranking_mode", F.lit(mode)) \
     .withColumn("phase3_score", F.col(score_col)) \
     .withColumn("phase3_region_rank", w) \
     .where(F.col("phase3_region_rank") <= F.lit(k))

    return top.orderBy(F.col("phase3_score").desc())


# ---------------------------
# Parent sample scan (mapPartitions)
# ---------------------------

PARENT_SCHEMA = T.StructType([
    T.StructField("RA", T.DoubleType(), False),
    T.StructField("DEC", T.DoubleType(), False),
    T.StructField("BRICKNAME", T.StringType(), False),
    T.StructField("OBJID", T.LongType(), True),
    T.StructField("TYPE", T.StringType(), True),
    T.StructField("FLUX_G", T.DoubleType(), True),
    T.StructField("FLUX_R", T.DoubleType(), True),
    T.StructField("FLUX_Z", T.DoubleType(), True),
    T.StructField("FLUX_W1", T.DoubleType(), True),
    T.StructField("g_mag", T.DoubleType(), True),
    T.StructField("r_mag", T.DoubleType(), True),
    T.StructField("z_mag", T.DoubleType(), True),
    T.StructField("w1_mag", T.DoubleType(), True),
    T.StructField("r_minus_z", T.DoubleType(), True),
    T.StructField("z_minus_w1", T.DoubleType(), True),
    T.StructField("is_lrg_v1", T.BooleanType(), True),
    T.StructField("is_lrg_v2", T.BooleanType(), True),
    T.StructField("is_lrg_v3", T.BooleanType(), True),
    T.StructField("is_lrg_v4", T.BooleanType(), True),
    T.StructField("is_lrg_v5", T.BooleanType(), True),
    T.StructField("phase3_ranking_mode", T.StringType(), False),
    T.StructField("region_id", T.IntegerType(), False),
    T.StructField("phase3_region_rank", T.IntegerType(), False),
])


def sweep_partition_iterator(
    urls: Iterator[str],
    brick_to_modes: Dict[str, List[Tuple[str, int, int]]],
    local_cache_dir: str,
    chunk_size: int,
    checkpoint_s3_prefix: Optional[str] = None,
    s3_sweep_cache_prefix: Optional[str] = None,
) -> Iterator[dict]:
    """
    Iterate over sweep URLs and yield parent-catalog rows for all modes for which the brick is selected.
    
    Features:
    - S3 sweep cache: checks cache before downloading from source
    - TransferConfig for efficient large file downloads
    - Streaming gzip decompression
    - Per-sweep checkpointing (skip if _SUCCESS marker exists)
    - Detailed progress logging to stderr
    """
    # executor-local clients
    s3 = boto3.client("s3")
    
    # TransferConfig for large file transfers (1.8GB FITS files)
    transfer_config = TransferConfig(
        multipart_threshold=100 * 1024 * 1024,  # 100MB
        max_concurrency=10,
        multipart_chunksize=64 * 1024 * 1024,   # 64MB chunks
    )
    
    # Parse checkpoint prefix if provided
    ckpt_bucket = None
    ckpt_key_prefix = None
    if checkpoint_s3_prefix:
        ckpt_bucket, ckpt_key_prefix = parse_s3_uri(checkpoint_s3_prefix)
        ckpt_key_prefix = ckpt_key_prefix.rstrip("/")
    
    url_list = list(urls)
    total_urls = len(url_list)
    
    # Pre-cache brick keys as a set for O(1) lookup (avoid creating list every chunk)
    brick_keys_set: Set[str] = set(brick_to_modes.keys())
    brick_keys_list: List[str] = list(brick_keys_set)  # For np.isin
    
    for idx, url in enumerate(url_list):
        url = url.strip()
        if not url or url.startswith("#"):
            continue
        
        sweep_name = os.path.basename(url).replace(".fits.gz", "").replace(".fits", "")
        start_time = time.time()
        
        # Check for checkpoint (skip if already processed)
        if ckpt_bucket and ckpt_key_prefix:
            success_key = f"{ckpt_key_prefix}/{sweep_name}/_SUCCESS"
            try:
                s3.head_object(Bucket=ckpt_bucket, Key=success_key)
                print(f"[SWEEP {idx+1}/{total_urls}] {sweep_name}: SKIP (checkpointed)", file=sys.stderr)
                sys.stderr.flush()
                continue
            except s3.exceptions.ClientError:
                pass  # Not checkpointed, proceed
        
        print(f"[SWEEP {idx+1}/{total_urls}] {sweep_name}: Processing...", file=sys.stderr)
        sys.stderr.flush()

        rows_yielded = 0
        local_file = None
        local_fits = None
        try:
            # Download from S3 cache first, fallback to original URL
            local_file = download_sweep_with_cache(
                s3, url, s3_sweep_cache_prefix, local_cache_dir, transfer_config
            )
            download_time = time.time() - start_time
            print(f"[SWEEP {idx+1}/{total_urls}] {sweep_name}: Downloaded in {download_time:.1f}s", file=sys.stderr)
            sys.stderr.flush()
            
            local_fits = maybe_decompress_gzip(local_file)

            with fits.open(local_fits, memmap=True) as hdul:
                data = hdul[1].data
                names = list(data.columns.names)

                ra = get_col(data, names, ["ra"])
                dec = get_col(data, names, ["dec"])
                brick = get_col(data, names, ["brickname"])
                typ = get_col(data, names, ["type"])
                objid = None
                try:
                    objid = get_col(data, names, ["objid"])
                except KeyError:
                    pass

                flux_g = get_col(data, names, ["flux_g"])
                flux_r = get_col(data, names, ["flux_r"])
                flux_z = get_col(data, names, ["flux_z"])
                flux_w1 = get_col(data, names, ["flux_w1"])

                n = len(ra)
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)

                    brick_chunk = np.array(brick[start:end]).astype(str)

                    # filter to selected bricks (use pre-cached list)
                    in_sel = np.isin(brick_chunk, brick_keys_list)
                    if not np.any(in_sel):
                        continue

                    # TYPE != 'PSF'
                    type_chunk = np.array(typ[start:end]).astype(str)
                    non_psf = type_chunk != "PSF"

                    fr = np.asarray(flux_r[start:end], dtype=np.float64)
                    fz = np.asarray(flux_z[start:end], dtype=np.float64)
                    fw = np.asarray(flux_w1[start:end], dtype=np.float64)

                    good_flux = (fr > 0) & (fz > 0) & (fw > 0)

                    keep = in_sel & non_psf & good_flux
                    if not np.any(keep):
                        continue

                    fg = np.asarray(flux_g[start:end], dtype=np.float64)
                    ra_c = np.asarray(ra[start:end], dtype=np.float64)
                    dec_c = np.asarray(dec[start:end], dtype=np.float64)

                    mag_g = nanomaggies_to_mag(fg)
                    mag_r = nanomaggies_to_mag(fr)
                    mag_z = nanomaggies_to_mag(fz)
                    mag_w1 = nanomaggies_to_mag(fw)

                    r_minus_z = mag_r - mag_z
                    z_minus_w1 = mag_z - mag_w1

                    # variants
                    is_v = {}
                    for v in LRG_VARIANTS:
                        is_v[v.name] = (
                            (mag_z < v.z_mag_max)
                            & (r_minus_z > v.rz_min)
                            & (z_minus_w1 > v.zw1_min)
                        )

                    # baseline parent selection: v3 only
                    base = is_v[BASELINE_VARIANT] & keep
                    idx_arr = np.where(base)[0]
                    if idx_arr.size == 0:
                        continue

                    for i in idx_arr:
                        bname = brick_chunk[i]
                        mode_entries = brick_to_modes.get(bname)
                        if not mode_entries:
                            continue

                        row_common = {
                            "RA": float(ra_c[i]),
                            "DEC": float(dec_c[i]),
                            "BRICKNAME": str(bname),
                            "OBJID": int(objid[start + i]) if objid is not None else None,
                            "TYPE": str(type_chunk[i]),
                            "FLUX_G": float(fg[i]),
                            "FLUX_R": float(fr[i]),
                            "FLUX_Z": float(fz[i]),
                            "FLUX_W1": float(fw[i]),
                            "g_mag": float(mag_g[i]) if np.isfinite(mag_g[i]) else None,
                            "r_mag": float(mag_r[i]) if np.isfinite(mag_r[i]) else None,
                            "z_mag": float(mag_z[i]) if np.isfinite(mag_z[i]) else None,
                            "w1_mag": float(mag_w1[i]) if np.isfinite(mag_w1[i]) else None,
                            "r_minus_z": float(r_minus_z[i]) if np.isfinite(r_minus_z[i]) else None,
                            "z_minus_w1": float(z_minus_w1[i]) if np.isfinite(z_minus_w1[i]) else None,
                            "is_lrg_v1": bool(is_v["v1_pure_massive"][i]),
                            "is_lrg_v2": bool(is_v["v2_baseline_dr10"][i]),
                            "is_lrg_v3": bool(is_v["v3_color_relaxed"][i]),
                            "is_lrg_v4": bool(is_v["v4_mag_relaxed"][i]),
                            "is_lrg_v5": bool(is_v["v5_very_relaxed"][i]),
                        }

                        for (mode, region_id, region_rank) in mode_entries:
                            out = dict(row_common)
                            out["phase3_ranking_mode"] = mode
                            out["region_id"] = int(region_id)
                            out["phase3_region_rank"] = int(region_rank)
                            rows_yielded += 1
                            yield out

            # Write checkpoint marker after successful processing
            if ckpt_bucket and ckpt_key_prefix:
                success_key = f"{ckpt_key_prefix}/{sweep_name}/_SUCCESS"
                try:
                    s3.put_object(Bucket=ckpt_bucket, Key=success_key, Body=b"")
                except Exception as ckpt_err:
                    print(f"[SWEEP {idx+1}/{total_urls}] {sweep_name}: WARN checkpoint write failed: {ckpt_err}", file=sys.stderr)
                    sys.stderr.flush()

            elapsed = time.time() - start_time
            print(f"[SWEEP {idx+1}/{total_urls}] {sweep_name}: DONE in {elapsed:.1f}s, yielded {rows_yielded} LRGs", file=sys.stderr)
            sys.stderr.flush()

        except Exception as e:
            # Log error but continue to next file
            elapsed = time.time() - start_time
            print(f"[SWEEP {idx+1}/{total_urls}] {sweep_name}: ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}", file=sys.stderr)
            sys.stderr.flush()
        
        finally:
            # Cleanup: delete decompressed files to free disk space
            # This is critical when processing many files concurrently
            try:
                if local_file and local_file.endswith('.gz'):
                    decompressed = local_file[:-3]
                    if os.path.exists(decompressed):
                        os.remove(decompressed)
                        print(f"[SWEEP {idx+1}/{total_urls}] {sweep_name}: Cleaned up decompressed file", file=sys.stderr)
                        sys.stderr.flush()
            except Exception:
                pass  # Cleanup failure is not critical
            
            # Force garbage collection to free numpy arrays
            gc.collect()


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--phase2-analysis-s3", required=True,
                    help="S3 prefix containing phase2_analysis outputs (must contain {variant}/phase2_regions_summary.csv and {variant}/phase2_regions_bricks.csv)")
    ap.add_argument("--variant", default=BASELINE_VARIANT,
                    help="Variant directory name under phase2_analysis (default: v3_color_relaxed)")
    ap.add_argument("--sweep-index-s3", required=True,
                    help="S3 URL of text file listing sweep S3 URLs (one per line). Entries may end with .fits or .fits.gz")
    ap.add_argument("--output-s3", required=True, help="S3 prefix to write Phase 3 outputs")

    ap.add_argument("--num-regions", type=int, default=30)
    ap.add_argument("--ranking-modes", default="n_lrg,area_weighted,psf_weighted",
                    help="Comma-separated ranking modes: density,n_lrg,area_weighted,psf_weighted")

    # Optional region-level quality cuts (applied BEFORE ranking)
    ap.add_argument("--max-ebv", type=float, default=0.12)
    ap.add_argument("--max-psf-r-arcsec", type=float, default=1.60)
    ap.add_argument("--min-psfdepth-r", type=float, default=23.6)

    # psf_weighted parameters
    ap.add_argument("--psf-ref-arcsec", type=float, default=1.25)
    ap.add_argument("--sigma-psf-arcsec", type=float, default=0.15)
    ap.add_argument("--k-ebv", type=float, default=8.0)

    # sweep scan
    ap.add_argument("--chunk-size", type=int, default=100000)
    ap.add_argument("--local-cache-dir", default="/mnt/tmp/dhs_cache")
    ap.add_argument("--sweep-partitions", type=int, default=600,
                    help="Number of RDD partitions for sweep scan (tune for cluster size)")
    ap.add_argument("--max-sweeps", type=int, default=0,
                    help="For debugging: limit number of sweeps scanned (0 = all)")
    ap.add_argument("--checkpoint-s3", default="",
                    help="S3 prefix for per-sweep checkpointing. If set, completed sweeps write a _SUCCESS marker and are skipped on retry.")
    ap.add_argument("--s3-sweep-cache-prefix", default="",
                    help="S3 prefix where gzipped sweep FITS are cached (e.g., s3://bucket/sweep_fits_dump/). Checked before downloading from original URL.")

    args = ap.parse_args()

    spark = (
        SparkSession.builder
        .appName("dark-halo-scope-phase3")
        .getOrCreate()
    )

    variant = args.variant
    phase2_prefix = args.phase2_analysis_s3.rstrip("/")

    regions_path = f"{phase2_prefix}/{variant}/phase2_regions_summary.csv"
    bricks_path  = f"{phase2_prefix}/{variant}/phase2_regions_bricks.csv"

    regions = spark.read.option("header", True).csv(regions_path, inferSchema=True)
    bricks  = spark.read.option("header", True).csv(bricks_path, inferSchema=True)

    # Filter to plausible regions (quality cuts only - area window filter removed as it filters out all regions)
    # The keep_in_area_window column exists but all values are False in the Phase 2 output
    # If you need to filter by area window, uncomment the following:
    # if "keep_in_area_window" in regions.columns:
    #     regions = regions.where(F.col("keep_in_area_window") == F.lit(True))

    print(f"[MAIN] Total regions before quality cuts: {regions.count()}", file=sys.stderr)
    sys.stderr.flush()
    
    regions = regions.where(F.col("median_ebv") <= F.lit(args.max_ebv)) \
                     .where(F.col("median_psf_r_arcsec") <= F.lit(args.max_psf_r_arcsec)) \
                     .where(F.col("median_psfdepth_r") >= F.lit(args.min_psfdepth_r))
    
    print(f"[MAIN] Regions after quality cuts (ebv<={args.max_ebv}, psf<={args.max_psf_r_arcsec}, depth>={args.min_psfdepth_r}): {regions.count()}", file=sys.stderr)
    sys.stderr.flush()

    depth_p10, depth_p90 = compute_depth_normalization_quantiles(regions, "median_psfdepth_r")
    regions_scored = add_region_scores(
        regions_df=regions,
        variant=variant,
        psf_ref=args.psf_ref_arcsec,
        sigma_psf=args.sigma_psf_arcsec,
        k_ebv=args.k_ebv,
        depth_p10=depth_p10,
        depth_p90=depth_p90,
    ).cache()

    modes = [m.strip() for m in args.ranking_modes.split(",") if m.strip()]
    for m in modes:
        if m not in {"density", "n_lrg", "area_weighted", "psf_weighted"}:
            raise ValueError(f"Unknown ranking mode: {m}")

    # Select top regions per mode + write per-mode target bricks
    selected_regions_per_mode = {}
    for mode in modes:
        top = select_top_regions(regions_scored, mode=mode, k=args.num_regions).cache()
        selected_regions_per_mode[mode] = top

        out_reg = f"{args.output_s3.rstrip('/')}/{variant}/{mode}/phase3_selected_regions"
        top.coalesce(1).write.mode("overwrite").option("header", True).csv(out_reg)

        sel_region_ids = [r["region_id"] for r in top.select("region_id").collect()]
        target = bricks.where(F.col("region_id").isin(sel_region_ids)) \
                       .join(top.select("region_id", "phase3_ranking_mode", "phase3_region_rank"),
                             on="region_id", how="inner")

        out_tb = f"{args.output_s3.rstrip('/')}/{variant}/{mode}/phase3_target_bricks"
        target.coalesce(1).write.mode("overwrite").option("header", True).csv(out_tb)

    # Build brick -> list[(mode, region_id, region_rank)] mapping for a SINGLE sweep scan.
    brick_to_modes: Dict[str, List[Tuple[str, int, int]]] = {}
    for mode, top in selected_regions_per_mode.items():
        # Map region_id -> rank
        reg_rank = {int(r["region_id"]): int(r["phase3_region_rank"]) for r in top.select("region_id", "phase3_region_rank").collect()}

        # bricks for these regions
        sel = bricks.where(F.col("region_id").isin(list(reg_rank.keys()))).select("region_id", "brickname").collect()
        for r in sel:
            rid = int(r["region_id"])
            b = str(r["brickname"])
            brick_to_modes.setdefault(b, []).append((mode, rid, reg_rank[rid]))

    sc = spark.sparkContext
    bc_map = sc.broadcast(brick_to_modes)
    
    # Determine checkpoint prefix (use output_s3 subfolder if not explicitly set)
    checkpoint_prefix = args.checkpoint_s3 if args.checkpoint_s3 else f"{args.output_s3.rstrip('/')}/{variant}/checkpoints"
    bc_checkpoint_prefix = sc.broadcast(checkpoint_prefix)
    
    # Broadcast S3 sweep cache prefix (empty string if not set)
    bc_s3_sweep_cache = sc.broadcast(args.s3_sweep_cache_prefix if args.s3_sweep_cache_prefix else None)

    # Load sweep urls list (driver) then distribute
    s3 = boto3.client("s3")
    bucket, key = parse_s3_uri(args.sweep_index_s3)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    s3.download_file(bucket, key, tmp.name)
    with open(tmp.name, "r") as f:
        urls = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    os.unlink(tmp.name)

    if args.max_sweeps and args.max_sweeps > 0:
        urls = urls[: args.max_sweeps]
    
    print(f"[MAIN] Processing {len(urls)} sweep files with {args.sweep_partitions} partitions", file=sys.stderr)
    print(f"[MAIN] Checkpoint prefix: {checkpoint_prefix}", file=sys.stderr)
    print(f"[MAIN] S3 sweep cache: {args.s3_sweep_cache_prefix or '(none)'}", file=sys.stderr)
    print(f"[MAIN] Target bricks: {len(brick_to_modes)}", file=sys.stderr)
    sys.stderr.flush()

    rdd = sc.parallelize(urls, numSlices=max(1, args.sweep_partitions))

    rows_rdd = rdd.mapPartitions(
        lambda it: sweep_partition_iterator(
            it,
            brick_to_modes=bc_map.value,
            local_cache_dir=args.local_cache_dir,
            chunk_size=args.chunk_size,
            checkpoint_s3_prefix=bc_checkpoint_prefix.value,
            s3_sweep_cache_prefix=bc_s3_sweep_cache.value,
        )
    )

    parent_df = spark.createDataFrame(rows_rdd, schema=PARENT_SCHEMA)

    # Deduplicate within each mode by (BRICKNAME, OBJID) if OBJID is present.
    if "OBJID" in parent_df.columns:
        parent_df = parent_df.dropDuplicates(["phase3_ranking_mode", "BRICKNAME", "OBJID"])
    else:
        parent_df = parent_df.dropDuplicates(["phase3_ranking_mode", "BRICKNAME", "RA", "DEC"])

    # Persist before first action to avoid recomputation
    parent_df = parent_df.persist(StorageLevel.MEMORY_AND_DISK)
    
    # Count triggers computation and caches the result
    total_count = parent_df.count()
    print(f"[MAIN] Total LRGs in parent catalog: {total_count}", file=sys.stderr)
    sys.stderr.flush()

    # Write master parquet partitioned by mode (best for later phases)
    out_parquet = f"{args.output_s3.rstrip('/')}/{variant}/parent_catalog_parquet"
    print(f"[MAIN] Writing Parquet to: {out_parquet}", file=sys.stderr)
    sys.stderr.flush()
    parent_df.write.mode("overwrite").partitionBy("phase3_ranking_mode").parquet(out_parquet)

    # Also write per-mode CSV for convenience
    for mode in modes:
        out_csv = f"{args.output_s3.rstrip('/')}/{variant}/{mode}/phase3_lrg_parent_catalog"
        mode_count = parent_df.where(F.col("phase3_ranking_mode") == F.lit(mode)).count()
        print(f"[MAIN] Writing CSV for mode '{mode}': {mode_count} LRGs -> {out_csv}", file=sys.stderr)
        sys.stderr.flush()
        parent_df.where(F.col("phase3_ranking_mode") == F.lit(mode)) \
                 .coalesce(1).write.mode("overwrite").option("header", True).csv(out_csv)
    
    # Unpersist to free memory
    parent_df.unpersist()
    print(f"[MAIN] Phase 3 complete.", file=sys.stderr)
    sys.stderr.flush()

    spark.stop()


if __name__ == "__main__":
    main()

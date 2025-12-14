#!/usr/bin/env python3
"""
PySpark Phase 2: Multi-cut LRG density per DR10 brick (Checkpointed Version)

This script generalizes the Phase 1.5 EMR job by:
  * Applying several physically motivated LRG-like selection functions
    in a SINGLE pass over each DR10 SWEEP file.
  * Optionally restricting to a rectangular RA/Dec footprint.
  * Writing intermediate CSV per SWEEP file with a _SUCCESS marker.
  * Supporting resumable runs - skips already-completed SWEEPs.
  * Final aggregation step combines all intermediate CSVs.

Checkpointing Design:
  - Each SWEEP file gets its own output subfolder: {output_prefix}/intermediate/{sweep_basename}/
  - After processing, writes CSV + _SUCCESS marker to that folder
  - On restart, checks for _SUCCESS marker and skips completed SWEEPs
  - Final step reads all intermediate CSVs, aggregates, and writes sorted result

Assumptions about DR10 SWEEP schema
-----------------------------------
We assume each SWEEP FITS file has (case-insensitive) columns:

  ra, dec           : sky coordinates in deg
  brickname         : DR10 brick name (string)
  type              : morphology label ('PSF', 'REX', 'EXP', etc.)
  flux_r, flux_z    : Tractor fluxes in nanomaggies
  flux_w1           : W1 flux in nanomaggies

Author: Dark Halo Scope pipeline, Phase 2
"""

import argparse
import os
import sys
import tempfile
import urllib.parse
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyspark.sql import SparkSession

# ----------------------------------------------------------------------
# Hyper-parameter grid of LRG-like selections
# ----------------------------------------------------------------------


def build_lrg_hypergrid():
    """
    Return a list of LRG selection definitions.

    Each definition is a dict with:
      name       : short identifier used in column names
      z_mag_max  : faint-end z-band limit (AB mag)
      rz_min     : minimum (r - z) color
      zw1_min    : minimum (z - W1) color
    """
    cuts = [
        {
            "name": "v1_pure_massive",
            "z_mag_max": 20.0,
            "rz_min": 0.5,
            "zw1_min": 1.6,
        },
        {
            "name": "v2_baseline_dr10",
            "z_mag_max": 20.4,
            "rz_min": 0.4,
            "zw1_min": 1.6,
        },
        {
            "name": "v3_color_relaxed",
            "z_mag_max": 20.4,
            "rz_min": 0.4,
            "zw1_min": 0.8,
        },
        {
            "name": "v4_mag_relaxed",
            "z_mag_max": 21.0,
            "rz_min": 0.4,
            "zw1_min": 0.8,
        },
        {
            "name": "v5_very_relaxed",
            "z_mag_max": 21.5,
            "rz_min": 0.3,
            "zw1_min": 0.8,
        },
    ]
    return cuts


# ----------------------------------------------------------------------
# FITS utilities (run on executors)
# ----------------------------------------------------------------------


COLUMN_ALIASES = {
    "ra": ["ra", "RA"],
    "dec": ["dec", "DEC"],
    "brickname": ["brickname", "BRICKNAME"],
    "type": ["type", "TYPE"],
    "flux_r": ["flux_r", "FLUX_R"],
    "flux_z": ["flux_z", "FLUX_Z"],
    "flux_w1": ["flux_w1", "FLUX_W1"],
}


def _resolve_column(df, logical_name: str):
    """Return the actual column name in `df` matching logical_name."""
    candidates = COLUMN_ALIASES.get(logical_name, [logical_name])
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    raise KeyError(
        f"Could not find a column corresponding to '{logical_name}' "
        f"in SWEEP file. Available columns: {list(df.columns)}"
    )


def nanomaggies_to_mag(flux: np.ndarray, zero_point: float = 22.5) -> np.ndarray:
    """
    Convert nanomaggies to AB magnitudes.
    For flux <= 0, return NaN.
    """
    flux = np.asarray(flux, dtype=float)
    mag = np.full_like(flux, np.nan, dtype=float)
    mask = flux > 0
    mag[mask] = zero_point - 2.5 * np.log10(flux[mask])
    return mag


def _parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    parsed = urllib.parse.urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def _get_s3_transfer_config():
    """
    Get TransferConfig optimized for large FITS files (1-2 GB).
    
    - multipart_threshold: Use multipart for files > 100MB
    - multipart_chunksize: 100MB chunks (reduces API calls for large files)
    - max_concurrency: 10 parallel threads for faster transfers
    - use_threads: Enable threading for parallel chunk transfers
    """
    from boto3.s3.transfer import TransferConfig
    
    return TransferConfig(
        multipart_threshold=100 * 1024 * 1024,  # 100 MB
        multipart_chunksize=100 * 1024 * 1024,  # 100 MB chunks
        max_concurrency=10,
        use_threads=True,
    )


def _s3_object_exists(s3_client, bucket: str, key: str) -> bool:
    """Check if an S3 object exists using head_object (fast)."""
    from botocore.exceptions import ClientError
    
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def download_to_local(path: str, s3_cache_prefix: Optional[str] = None) -> Tuple[str, bool]:
    """
    Resolve a SWEEP path to a local on-disk file.

    S3 Cache Lookup Order:
      1. Check for gzipped file: {s3_cache_prefix}/{basename}.gz
      2. Check for non-gzipped file: {s3_cache_prefix}/{basename}
      3. If neither found, download from HTTP

    After downloading from HTTP, the file is gzip-compressed and uploaded
    to S3 cache with .gz extension for future runs.
    
    Uses boto3 for all S3 operations.
    """
    import gzip
    import shutil
    import time
    from pathlib import Path
    
    import boto3
    from botocore.exceptions import ClientError

    parsed = urllib.parse.urlparse(path)
    scheme = parsed.scheme.lower()

    if scheme in ("", "file"):
        return path, False

    tmp_dir = Path("/mnt/tmp/phase2_sweeps")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    file_basename = os.path.basename(parsed.path) or "sweep.fits"
    local_path = tmp_dir / file_basename

    # Create S3 client and transfer config for large files (1-2 GB)
    s3_client = boto3.client("s3")
    transfer_config = _get_s3_transfer_config()

    if scheme in ("s3", "s3a"):
        bucket, key = _parse_s3_uri(path)
        print(f"[SWEEP-IO] Copying from S3: s3://{bucket}/{key}", file=sys.stderr)
        try:
            s3_client.download_file(bucket, key, str(local_path), Config=transfer_config)
        except ClientError as e:
            raise RuntimeError(f"Failed to copy {path} from S3: {e}")
        return str(local_path), True

    if scheme in ("http", "https"):
        import requests

        cache_hit = False

        # Check S3 cache if prefix provided
        if s3_cache_prefix:
            cache_bucket, cache_key_prefix = _parse_s3_uri(s3_cache_prefix)
            cache_key_prefix = cache_key_prefix.rstrip("/")

            # --- Try 1: Check for gzipped file (.gz extension) ---
            gz_key = f"{cache_key_prefix}/{file_basename}.gz"
            local_gz_path = str(local_path) + ".gz"
            print(f"[SWEEP-IO] Checking S3 cache (gzipped): s3://{cache_bucket}/{gz_key}", file=sys.stderr)
            sys.stderr.flush()

            if _s3_object_exists(s3_client, cache_bucket, gz_key):
                # File exists - download it (use transfer_config for large files)
                print(f"[SWEEP-IO] Found in cache, downloading...", file=sys.stderr)
                try:
                    s3_client.download_file(cache_bucket, gz_key, local_gz_path, Config=transfer_config)
                    
                    if os.path.exists(local_gz_path) and os.path.getsize(local_gz_path) > 0:
                        gz_size = os.path.getsize(local_gz_path)
                        print(
                            f"[SWEEP-IO] ✓ Cache HIT (gzipped): {file_basename}.gz ({gz_size / 1e6:.1f} MB), decompressing...",
                            file=sys.stderr,
                        )

                        with gzip.open(local_gz_path, "rb") as f_in:
                            with open(local_path, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)

                        os.remove(local_gz_path)

                        file_size = os.path.getsize(local_path)
                        print(f"[SWEEP-IO] ✓ Decompressed to {file_size / 1e6:.1f} MB", file=sys.stderr)
                        cache_hit = True
                except ClientError as e:
                    print(f"[SWEEP-IO] ⚠ Download failed: {e}", file=sys.stderr)
                    if os.path.exists(local_gz_path):
                        os.remove(local_gz_path)

            if not cache_hit:
                # --- Try 2: Check for non-gzipped file (no .gz extension) ---
                raw_key = f"{cache_key_prefix}/{file_basename}"
                print(f"[SWEEP-IO] Checking S3 cache (raw): s3://{cache_bucket}/{raw_key}", file=sys.stderr)
                sys.stderr.flush()

                if _s3_object_exists(s3_client, cache_bucket, raw_key):
                    # File exists - download it (use transfer_config for large files)
                    print(f"[SWEEP-IO] Found raw file in cache, downloading...", file=sys.stderr)
                    try:
                        s3_client.download_file(cache_bucket, raw_key, str(local_path), Config=transfer_config)
                        
                        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                            file_size = os.path.getsize(local_path)
                            print(
                                f"[SWEEP-IO] ✓ Cache HIT (raw): {file_basename} ({file_size / 1e6:.1f} MB)",
                                file=sys.stderr,
                            )
                            sys.stderr.flush()
                            cache_hit = True

                            # --- Migrate: compress to gzip, upload, delete raw ---
                            print(f"[SWEEP-IO] Migrating raw file to gzipped format...", file=sys.stderr)
                            sys.stderr.flush()
                            local_gz_path = str(local_path) + ".gz"

                            # Compress
                            compress_start = time.time()
                            with open(local_path, "rb") as f_in:
                                with gzip.open(local_gz_path, "wb", compresslevel=6) as f_out:
                                    shutil.copyfileobj(f_in, f_out)

                            gz_size = os.path.getsize(local_gz_path)
                            compress_time = time.time() - compress_start
                            ratio = 100.0 * gz_size / file_size
                            print(
                                f"[SWEEP-IO] Compressed {file_size / 1e6:.1f} MB → {gz_size / 1e6:.1f} MB ({ratio:.1f}%) in {compress_time:.1f}s",
                                file=sys.stderr,
                            )

                            # Upload gzipped version (use transfer_config for large files)
                            print(f"[SWEEP-IO] Uploading gzipped version: s3://{cache_bucket}/{gz_key}", file=sys.stderr)
                            try:
                                s3_client.upload_file(local_gz_path, cache_bucket, gz_key, Config=transfer_config)
                                print(f"[SWEEP-IO] ✓ Uploaded gzipped: s3://{cache_bucket}/{gz_key}", file=sys.stderr)

                                # Delete raw file from S3
                                print(f"[SWEEP-IO] Deleting raw file from S3: s3://{cache_bucket}/{raw_key}", file=sys.stderr)
                                sys.stderr.flush()
                                s3_client.delete_object(Bucket=cache_bucket, Key=raw_key)
                                print(f"[SWEEP-IO] ✓ Deleted raw file", file=sys.stderr)
                                sys.stderr.flush()
                            except ClientError as e:
                                print(f"[SWEEP-IO] ⚠ Migration failed: {e}", file=sys.stderr)
                                sys.stderr.flush()
                            finally:
                                if os.path.exists(local_gz_path):
                                    os.remove(local_gz_path)
                    except ClientError as e:
                        print(f"[SWEEP-IO] ⚠ Download failed: {e}", file=sys.stderr)
                        if os.path.exists(local_path):
                            os.remove(local_path)
                else:
                    print(f"[SWEEP-IO] Cache MISS: {file_basename} not found in S3 cache", file=sys.stderr)

        # Download from HTTP if not in cache
        if not cache_hit:
            start_time = time.time()
            print(f"[SWEEP-IO] Downloading from HTTP: {path}", file=sys.stderr)
            resp = requests.get(path, stream=True, timeout=600)
            resp.raise_for_status()
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0

            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and downloaded % (100 * 1024 * 1024) < (1 << 20):
                            pct = 100.0 * downloaded / total_size
                            print(
                                f"[SWEEP-IO] Download progress: {downloaded / 1e6:.0f} MB / {total_size / 1e6:.0f} MB ({pct:.0f}%)",
                                file=sys.stderr,
                            )

            file_size = os.path.getsize(local_path)
            download_time = time.time() - start_time
            print(
                f"[SWEEP-IO] ✓ Downloaded {file_size / 1e6:.1f} MB in {download_time:.1f}s",
                file=sys.stderr,
            )

            # Upload to S3 cache as gzip-compressed
            if s3_cache_prefix and file_size > 0:
                cache_bucket, cache_key_prefix = _parse_s3_uri(s3_cache_prefix)
                cache_key_prefix = cache_key_prefix.rstrip("/")
                gz_key = f"{cache_key_prefix}/{file_basename}.gz"
                local_gz_path = str(local_path) + ".gz"

                print(f"[SWEEP-IO] Compressing for S3 cache...", file=sys.stderr)
                compress_start = time.time()

                with open(local_path, "rb") as f_in:
                    with gzip.open(local_gz_path, "wb", compresslevel=6) as f_out:
                        shutil.copyfileobj(f_in, f_out)

                gz_size = os.path.getsize(local_gz_path)
                compress_time = time.time() - compress_start
                ratio = 100.0 * gz_size / file_size
                print(
                    f"[SWEEP-IO] Compressed {file_size / 1e6:.1f} MB → {gz_size / 1e6:.1f} MB ({ratio:.1f}%) in {compress_time:.1f}s",
                    file=sys.stderr,
                )

                print(f"[SWEEP-IO] Uploading to S3 cache: s3://{cache_bucket}/{gz_key}", file=sys.stderr)
                try:
                    s3_client.upload_file(local_gz_path, cache_bucket, gz_key, Config=transfer_config)
                    print(f"[SWEEP-IO] ✓ Cached to S3: s3://{cache_bucket}/{gz_key}", file=sys.stderr)
                except ClientError as e:
                    print(f"[SWEEP-IO] ⚠ Failed to cache to S3: {e}", file=sys.stderr)
                finally:
                    if os.path.exists(local_gz_path):
                        os.remove(local_gz_path)

        return str(local_path), True

    raise ValueError(f"Unsupported URL scheme for path: {path}")


def check_success_marker(output_prefix: str, sweep_basename: str) -> bool:
    """
    Check if a _SUCCESS marker exists for this SWEEP file.
    Returns True if already completed (should skip), False otherwise.
    
    Uses boto3 head_object for fast existence check.
    """
    import boto3
    from botocore.exceptions import ClientError
    
    bucket, key_prefix = _parse_s3_uri(output_prefix)
    key_prefix = key_prefix.rstrip("/")
    success_key = f"{key_prefix}/intermediate/{sweep_basename}/_SUCCESS"
    
    s3_client = boto3.client("s3")
    
    try:
        s3_client.head_object(Bucket=bucket, Key=success_key)
        print(f"[CHECKPOINT] ✓ Found _SUCCESS marker for {sweep_basename}, skipping", file=sys.stderr)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        # Re-raise unexpected errors
        raise


def write_intermediate_csv(
    output_prefix: str,
    sweep_basename: str,
    counts: Dict[str, np.ndarray],
    lrg_cuts: List[dict],
) -> None:
    """
    Write intermediate CSV for a single SWEEP file to S3.
    Creates: {output_prefix}/intermediate/{sweep_basename}/part-00000.csv
    Then writes a _SUCCESS marker.
    
    Uses boto3 for all S3 operations.
    """
    import csv
    import tempfile
    import boto3
    from botocore.exceptions import ClientError
    
    bucket, key_prefix = _parse_s3_uri(output_prefix)
    key_prefix = key_prefix.rstrip("/")
    
    s3_client = boto3.client("s3")
    
    if not counts:
        print(f"[CHECKPOINT] No data to write for {sweep_basename}", file=sys.stderr)
        # Still write success marker to avoid re-processing empty files
        success_key = f"{key_prefix}/intermediate/{sweep_basename}/_SUCCESS"
        try:
            s3_client.put_object(Bucket=bucket, Key=success_key, Body=b"")
        except ClientError as e:
            print(f"[CHECKPOINT] ⚠ Failed to write empty success marker: {e}", file=sys.stderr)
        return
    
    cut_names = [cut["name"] for cut in lrg_cuts]
    
    # Write CSV to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        # Header
        header = ["brickname", "n_gal"] + [f"n_lrg_{name}" for name in cut_names]
        writer.writerow(header)
        
        # Data rows
        for brick, vec in counts.items():
            row = [brick, int(vec[0])] + [int(vec[i]) for i in range(1, len(vec))]
            writer.writerow(row)
        
        temp_path = f.name
    
    # Upload CSV to S3
    csv_key = f"{key_prefix}/intermediate/{sweep_basename}/part-00000.csv"
    print(f"[CHECKPOINT] Writing intermediate CSV: s3://{bucket}/{csv_key}", file=sys.stderr)
    
    try:
        s3_client.upload_file(temp_path, bucket, csv_key)
    except ClientError as e:
        print(f"[CHECKPOINT] ⚠ Failed to write CSV: {e}", file=sys.stderr)
        os.unlink(temp_path)
        return
    
    os.unlink(temp_path)
    
    # Write _SUCCESS marker
    success_key = f"{key_prefix}/intermediate/{sweep_basename}/_SUCCESS"
    success_content = f"Completed processing {sweep_basename}\n"
    
    try:
        s3_client.put_object(Bucket=bucket, Key=success_key, Body=success_content.encode("utf-8"))
    except ClientError as e:
        print(f"[CHECKPOINT] ⚠ Failed to write success marker: {e}", file=sys.stderr)
        return
    
    print(f"[CHECKPOINT] ✓ Wrote {len(counts)} bricks + _SUCCESS for {sweep_basename}", file=sys.stderr)


def process_single_sweep(
    path: str,
    ra_min: Optional[float],
    ra_max: Optional[float],
    dec_min: Optional[float],
    dec_max: Optional[float],
    chunk_size: int,
    s3_cache_prefix: Optional[str],
    lrg_cuts: List[dict],
    output_prefix: str,
) -> None:
    """
    Process one SWEEP FITS file:
    1. Check if already completed (via _SUCCESS marker) - if so, skip
    2. Download and process the SWEEP file
    3. Write intermediate CSV to S3
    4. Write _SUCCESS marker
    
    Returns nothing - all output goes to S3.
    """
    from astropy.io import fits

    # Extract sweep basename for folder naming
    parsed = urllib.parse.urlparse(path)
    sweep_basename = os.path.basename(parsed.path).replace(".fits", "").replace(".gz", "")
    
    # Check if already completed
    if check_success_marker(output_prefix, sweep_basename):
        return  # Skip - already done
    
    print(f"[SWEEP] Starting: {path}", file=sys.stderr)
    local_path, is_temp = download_to_local(path, s3_cache_prefix=s3_cache_prefix)
    print(f"[SWEEP] Using local file: {local_path}", file=sys.stderr)

    counts: Dict[str, np.ndarray] = {}
    n_cuts = len(lrg_cuts)

    try:
        with fits.open(local_path, memmap=True) as hdul:
            data = hdul[1].data
            n_rows = len(data)
            print(f"[SWEEP] {path} contains {n_rows} rows", file=sys.stderr)

            # Resolve column names once (they don't change within a file)
            colnames_lower = {c.lower(): c for c in data.columns.names}
            
            def get_col(logical_name: str) -> str:
                """Get actual column name from logical name."""
                for alias in COLUMN_ALIASES.get(logical_name, [logical_name]):
                    if alias.lower() in colnames_lower:
                        return colnames_lower[alias.lower()]
                raise KeyError(f"Column '{logical_name}' not found. Available: {list(colnames_lower.values())}")
            
            try:
                col_ra = get_col("ra")
                col_dec = get_col("dec")
                col_brick = get_col("brickname")
                col_type = get_col("type")
                col_flux_r = get_col("flux_r")
                col_flux_z = get_col("flux_z")
                col_flux_w1 = get_col("flux_w1")
            except KeyError as e:
                print(f"[SWEEP-ERROR] {e}", file=sys.stderr)
                return

            for start in range(0, n_rows, chunk_size):
                stop = min(start + chunk_size, n_rows)
                chunk = data[start:stop]
                if len(chunk) == 0:
                    continue

                # Extract only needed columns as numpy arrays (no full DataFrame copy)
                # Use .astype() to handle byte order conversion efficiently
                ra = np.asarray(chunk[col_ra], dtype=np.float64)
                dec = np.asarray(chunk[col_dec], dtype=np.float64)
                flux_r = np.asarray(chunk[col_flux_r], dtype=np.float64)
                flux_z = np.asarray(chunk[col_flux_z], dtype=np.float64)
                flux_w1 = np.asarray(chunk[col_flux_w1], dtype=np.float64)
                obj_type = np.asarray(chunk[col_type], dtype=str)
                brickname_arr = np.asarray(chunk[col_brick], dtype=str)

                # Apply footprint filter if requested
                if None not in (ra_min, ra_max, dec_min, dec_max):
                    mask_sky = (
                        (ra >= ra_min) & (ra <= ra_max) &
                        (dec >= dec_min) & (dec <= dec_max)
                    )
                    if not np.any(mask_sky):
                        continue
                    # Apply mask to all arrays
                    ra = ra[mask_sky]
                    dec = dec[mask_sky]
                    flux_r = flux_r[mask_sky]
                    flux_z = flux_z[mask_sky]
                    flux_w1 = flux_w1[mask_sky]
                    obj_type = obj_type[mask_sky]
                    brickname_arr = brickname_arr[mask_sky]

                # Morphology and flux sanity cuts
                # Strip and uppercase for comparison
                obj_type_clean = np.char.strip(np.char.upper(obj_type))
                is_gal = obj_type_clean != "PSF"
                positive_flux = (flux_r > 0) & (flux_z > 0) & (flux_w1 > 0)

                base_mask = is_gal & positive_flux
                if not np.any(base_mask):
                    continue

                # Apply base mask
                flux_r = flux_r[base_mask]
                flux_z = flux_z[base_mask]
                flux_w1 = flux_w1[base_mask]
                bricknames = brickname_arr[base_mask]

                # Compute magnitudes
                mag_r = nanomaggies_to_mag(flux_r)
                mag_z = nanomaggies_to_mag(flux_z)
                mag_w1 = nanomaggies_to_mag(flux_w1)

                r_minus_z = mag_r - mag_z
                z_minus_w1 = mag_z - mag_w1

                # Build cut masks
                cut_masks = []
                for cut in lrg_cuts:
                    m = (
                        (mag_z < cut["z_mag_max"])
                        & (r_minus_z > cut["rz_min"])
                        & (z_minus_w1 > cut["zw1_min"])
                    )
                    cut_masks.append(m)

                # Aggregate by brick using vectorized operations
                unique_bricks, inverse = np.unique(bricknames, return_inverse=True)
                
                # Use bincount for fast counting (O(n) instead of O(n*k))
                n_unique = len(unique_bricks)
                total_per_brick = np.bincount(inverse, minlength=n_unique)
                
                for i, brick in enumerate(unique_bricks):
                    vec = counts.get(brick)
                    if vec is None:
                        vec = np.zeros(1 + n_cuts, dtype=np.int64)
                        counts[brick] = vec

                    vec[0] += int(total_per_brick[i])
                    
                    # For each cut, count how many pass in this brick
                    for j, m in enumerate(cut_masks, start=1):
                        # bincount with mask: count only where cut passes
                        cut_counts = np.bincount(inverse, weights=m.astype(np.int64), minlength=n_unique)
                        vec[j] += int(cut_counts[i])
                
                # Explicit cleanup to help garbage collector
                del ra, dec, flux_r, flux_z, flux_w1, obj_type, brickname_arr
                del mag_r, mag_z, mag_w1, r_minus_z, z_minus_w1, cut_masks, bricknames

    finally:
        # Clean up temporary file
        if is_temp and os.path.exists(local_path):
            try:
                os.remove(local_path)
                print(f"[SWEEP] Deleted temp file: {local_path}", file=sys.stderr)
            except OSError as e:
                print(f"[SWEEP] ⚠ Failed to delete temp file {local_path}: {e}", file=sys.stderr)
        sys.stderr.flush()

    # Write intermediate CSV and success marker
    write_intermediate_csv(output_prefix, sweep_basename, counts, lrg_cuts)


def aggregate_intermediate_csvs(spark, output_prefix: str, lrg_cuts: List[dict]) -> None:
    """
    Read all intermediate CSVs from S3, aggregate by brickname,
    and write final sorted result.
    """
    from pyspark.sql import functions as F
    from pyspark import StorageLevel
    
    intermediate_path = f"{output_prefix.rstrip('/')}/intermediate/*/part-*.csv"
    print(f"[AGGREGATE] Reading intermediate CSVs from: {intermediate_path}", file=sys.stderr)
    
    # Read all intermediate CSVs
    df = spark.read.option("header", "true").csv(intermediate_path)
    
    # Check if empty using head() instead of rdd.isEmpty() (more efficient)
    if len(df.head(1)) == 0:
        print("[AGGREGATE] No intermediate data found!", file=sys.stderr)
        return
    
    # Get column names for aggregation
    cut_names = [cut["name"] for cut in lrg_cuts]
    agg_cols = ["n_gal"] + [f"n_lrg_{name}" for name in cut_names]
    
    # Cast to integers and aggregate
    for col in agg_cols:
        df = df.withColumn(col, F.col(col).cast("long"))
    
    # Group by brickname and sum all count columns
    agg_exprs = [F.sum(F.col(c)).alias(c) for c in agg_cols]
    result_df = df.groupBy("brickname").agg(*agg_exprs)
    
    # Sort by brickname for ordered output
    result_df = result_df.orderBy("brickname")
    
    # Persist the result before triggering actions to avoid recomputation
    result_df = result_df.persist(StorageLevel.MEMORY_AND_DISK)
    
    # Count first (triggers computation and caches result)
    n_bricks = result_df.count()
    print(f"[AGGREGATE] Aggregated {n_bricks} unique bricks", file=sys.stderr)
    
    # Write final result (uses cached data, no recomputation)
    final_path = f"{output_prefix.rstrip('/')}/phase2_lrg_hypergrid.csv"
    print(f"[AGGREGATE] Writing final result to: {final_path}", file=sys.stderr)
    
    result_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(final_path)
    
    # Unpersist to free memory
    result_df.unpersist()
    
    print(f"[AGGREGATE] ✓ Wrote {n_bricks} bricks to {final_path}", file=sys.stderr)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: Multi-cut LRG density per brick from DR10 SWEEPs (Checkpointed)"
    )
    parser.add_argument(
        "--sweep-index-path",
        required=True,
        help="Text file (local or S3) with one SWEEP FITS path/URL per line",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix (local or S3) for CSV, e.g. s3://bucket/phase2_lrg_hypergrid",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Number of rows per FITS chunk (default: 100000)",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=0,
        help="Optional limit on number of sweeps to process (0 = all)",
    )
    parser.add_argument(
        "--ra-min",
        type=float,
        default=None,
        help="Minimum RA (deg) for footprint filter (optional)",
    )
    parser.add_argument(
        "--ra-max",
        type=float,
        default=None,
        help="Maximum RA (deg) for footprint filter (optional)",
    )
    parser.add_argument(
        "--dec-min",
        type=float,
        default=None,
        help="Minimum Dec (deg) for footprint filter (optional)",
    )
    parser.add_argument(
        "--dec-max",
        type=float,
        default=None,
        help="Maximum Dec (deg) for footprint filter (optional)",
    )
    parser.add_argument(
        "--s3-cache-prefix",
        type=str,
        default=None,
        help="Optional S3 prefix to cache HTTP downloads of SWEEP files",
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=128,
        help="Number of Spark partitions (Sweeps tasks). Default: 128",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Phase 2: Multi-cut LRG density per brick (Checkpointed Version)")
    print("=" * 70)
    print(f"  sweep-index-path : {args.sweep_index_path}")
    print(f"  output-prefix    : {args.output_prefix}")
    print(f"  chunk-size       : {args.chunk_size}")
    print(f"  max-sweeps       : {args.max_sweeps}")
    print(f"  RA/Dec footprint : RA [{args.ra_min}, {args.ra_max}], Dec [{args.dec_min}, {args.dec_max}]")
    print(f"  s3-cache-prefix  : {args.s3_cache_prefix}")
    print(f"  num-partitions   : {args.num_partitions}")
    print("")
    print("  Checkpointing: Writes intermediate CSV per SWEEP + _SUCCESS marker")
    print("  Resumable: Skips SWEEPs with existing _SUCCESS markers")
    print("")

    lrg_cuts = build_lrg_hypergrid()
    print("  LRG hyper-parameter grid (applied simultaneously):")
    for cut in lrg_cuts:
        print(
            f"    - {cut['name']}: z < {cut['z_mag_max']:.2f}, "
            f"(r - z) > {cut['rz_min']:.2f}, (z - W1) > {cut['zw1_min']:.2f}"
        )
    print("")

    # Initialize Spark
    spark = SparkSession.builder.appName("phase2_lrg_hypergrid_checkpointed").getOrCreate()
    sc = spark.sparkContext

    # Read sweep index file
    def read_index_lines(path: str) -> List[str]:
        parsed = urllib.parse.urlparse(path)
        scheme = parsed.scheme.lower()
        if scheme in ("", "file"):
            with open(path, "r") as f:
                return [line.strip() for line in f if line.strip()]
        if scheme in ("s3", "s3a"):
            import boto3
            
            bucket, key = _parse_s3_uri(path)
            print(f"[INDEX] Reading index from S3: s3://{bucket}/{key}", file=sys.stderr)
            s3_client = boto3.client("s3")
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read().decode("utf-8")
            return [line.strip() for line in content.splitlines() if line.strip()]
        raise ValueError(f"Unsupported URL scheme for sweep-index-path: {path}")

    sweep_paths = read_index_lines(args.sweep_index_path)
    if args.max_sweeps > 0:
        sweep_paths = sweep_paths[: args.max_sweeps]

    n_sweeps = len(sweep_paths)
    if n_sweeps == 0:
        print("[MAIN] No sweep paths found. Exiting.")
        sys.exit(1)

    print(f"[MAIN] Total SWEEP files to process: {n_sweeps}")

    # Broadcast configuration
    bc_ra_min = sc.broadcast(args.ra_min)
    bc_ra_max = sc.broadcast(args.ra_max)
    bc_dec_min = sc.broadcast(args.dec_min)
    bc_dec_max = sc.broadcast(args.dec_max)
    bc_chunk_size = sc.broadcast(args.chunk_size)
    bc_s3_cache_prefix = sc.broadcast(args.s3_cache_prefix)
    bc_lrg_cuts = sc.broadcast(lrg_cuts)
    bc_output_prefix = sc.broadcast(args.output_prefix)

    rdd = sc.parallelize(sweep_paths, numSlices=args.num_partitions)

    def map_fn(path: str) -> None:
        """Process a single SWEEP and write results to S3. Returns nothing."""
        import traceback
        
        try:
            print(f"[TASK] Starting task for: {path}", file=sys.stderr)
            sys.stderr.flush()
            
            process_single_sweep(
                path=path,
                ra_min=bc_ra_min.value,
                ra_max=bc_ra_max.value,
                dec_min=bc_dec_min.value,
                dec_max=bc_dec_max.value,
                chunk_size=int(bc_chunk_size.value),
                s3_cache_prefix=bc_s3_cache_prefix.value,
                lrg_cuts=bc_lrg_cuts.value,
                output_prefix=bc_output_prefix.value,
            )
            
            print(f"[TASK] Completed task for: {path}", file=sys.stderr)
            sys.stderr.flush()
        except Exception as e:
            print(f"[TASK-ERROR] Exception processing {path}: {e}", file=sys.stderr)
            print(f"[TASK-ERROR] Traceback:\n{traceback.format_exc()}", file=sys.stderr)
            sys.stderr.flush()
            raise  # Re-raise so Spark knows the task failed
        
        return None

    # Process all SWEEPs (each writes its own intermediate CSV)
    print("[MAIN] Processing SWEEP files (checkpointed)...", file=sys.stderr)
    rdd.foreach(map_fn)
    
    print("[MAIN] All SWEEP files processed. Aggregating results...", file=sys.stderr)
    
    # Aggregate all intermediate CSVs into final result
    aggregate_intermediate_csvs(spark, args.output_prefix, lrg_cuts)

    print("=" * 70)
    print("[MAIN] Phase 2 complete!")
    print(f"[MAIN] Intermediate CSVs: {args.output_prefix}/intermediate/")
    print(f"[MAIN] Final result: {args.output_prefix}/phase2_lrg_hypergrid.csv/")
    print("=" * 70)

    spark.stop()


if __name__ == "__main__":
    main()

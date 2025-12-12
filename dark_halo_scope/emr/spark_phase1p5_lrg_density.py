#!/usr/bin/env python
"""
PySpark Phase 1.5: LRG density per brick from DR10 SWEEPs.

This job is designed to run on an EMR cluster. It:

  1. Reads a list of DR10 SWEEP FITS URLs or S3 paths.
  2. Distributes these paths across Spark executors.
  3. For each SWEEP file:
       - Downloads to local /tmp (with size limit to prevent OOM).
       - Loads the FITS table (HDU 1).
       - Applies DESI-like LRG color/magnitude cuts.
       - Groups by BRICKNAME and returns (brickname, lrg_count) pairs.
  4. Reduces counts per brick across all SWEEPs.
  5. Writes a compact per-brick CSV to S3 with columns:
       - brickname
       - lrg_count

You will combine this with brick QA info (seeing, depth, area_deg2)
on your laptop or another step to compute LRG densities and select regions.

MEMORY CONSIDERATIONS:
- Each SWEEP file is 100-300 MB compressed
- astropy loads the entire table into memory (~500MB-1GB per file)
- With 2GB executor memory, only 1 file can be processed at a time per executor
- Use --num-partitions equal to the number of SWEEP files for best parallelism

RECOMMENDED SPARK-SUBMIT:
  spark-submit \\
    --deploy-mode client \\
    --master yarn \\
    --driver-memory 4g \\
    --executor-memory 4g \\
    --executor-cores 2 \\
    --conf spark.executor.memoryOverhead=1g \\
    --conf spark.driver.memoryOverhead=1g \\
    --conf spark.dynamicAllocation.enabled=true \\
    --conf spark.shuffle.service.enabled=true \\
    --conf spark.sql.shuffle.partitions=100 \\
    --py-files /mnt/dark_halo_scope_code.tgz \\
    /mnt/dark_halo_scope_code/emr/spark_phase1p5_lrg_density.py \\
    --sweep-index-path s3://bucket/sweep_urls.txt \\
    --output-prefix s3://bucket/phase1p5
"""

import argparse
import gc
import os
import sys
import tempfile
import traceback
from datetime import datetime
from typing import Iterable, List, Tuple

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType

# Import astropy lazily in executor functions to avoid driver-side memory issues
# from astropy.io import fits  # Moved to inside functions


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


def process_single_sweep_chunked(
    path: str,
    ra_min: float,  # None means no filter
    ra_max: float,  # None means no filter
    dec_min: float,  # None means no filter
    dec_max: float,  # None means no filter
    lrg_z_mag_max: float,
    lrg_min_r_minus_z: float,
    lrg_min_z_minus_w1: float,
    chunk_size: int = 500_000,
    s3_cache_prefix: str = None,
) -> List[Tuple[str, int]]:
    """
    Load a single SWEEP FITS and process it in CHUNKS to avoid OOM.

    This is the MEMORY-EFFICIENT version that processes the table in chunks
    of `chunk_size` rows at a time, never loading the entire file into RAM.

    For a 1.6 GB FITS file with 20M rows:
    - Old approach: loads ALL columns → 3-5 GB RAM
    - This approach: loads 500K rows at a time → ~100-200 MB RAM

    The path can be:
      - A local filesystem path (e.g. /mnt/.../sweep-*.fits)
      - An S3 URL (s3:// or s3a://) - downloaded via aws cli
      - An HTTP(S) URL - downloaded via requests
    
    S3 Caching:
      If s3_cache_prefix is provided (e.g., "s3://bucket/sweep-cache/"), files
      downloaded from HTTP will be uploaded to S3 for future runs. On subsequent
      runs, the job will check S3 first before downloading from HTTP.
    """
    import urllib.parse
    import time
    from collections import Counter
    
    # Import astropy lazily inside the executor
    from astropy.io import fits

    original_path = path.strip()
    local_path = original_path
    temp_file_created = False
    file_basename = ""
    
    print(f"[SWEEP-DEBUG] Starting process_single_sweep_chunked for: {original_path[:80]}...", file=sys.stderr)
    print(f"[SWEEP-DEBUG] Parameters: ra_min={ra_min}, ra_max={ra_max}, dec_min={dec_min}, dec_max={dec_max}", file=sys.stderr)
    print(f"[SWEEP-DEBUG] LRG cuts: z_mag_max={lrg_z_mag_max}, r-z>{lrg_min_r_minus_z}, z-W1>{lrg_min_z_minus_w1}", file=sys.stderr)
    print(f"[SWEEP-DEBUG] chunk_size={chunk_size}, s3_cache_prefix={s3_cache_prefix}", file=sys.stderr)
    sys.stderr.flush()
    
    # Accumulate (brickname -> count) across all chunks
    brick_counts: Counter = Counter()
    n_total = 0
    n_in_footprint = 0
    n_lrg = 0

    url = urllib.parse.urlparse(local_path)
    start_time = time.time()
    
    print(f"[SWEEP-DEBUG] URL scheme: {url.scheme}", file=sys.stderr)
    sys.stderr.flush()

    try:
        # ---- Step 1: Download file if remote ----
        if url.scheme in ("http", "https"):
            import requests
            import subprocess
            
            file_basename = os.path.basename(url.path) or "sweep.fits"
            temp_dir = tempfile.gettempdir()
            local_path = os.path.join(temp_dir, f"sweep_{os.getpid()}_{file_basename}")
            temp_file_created = True
            
            # Check S3 cache first if prefix provided
            s3_cached_path = None
            if s3_cache_prefix:
                s3_cached_path = f"{s3_cache_prefix.rstrip('/')}/{file_basename}"
                print(f"[SWEEP] Checking S3 cache: {s3_cached_path}", file=sys.stderr)
                
                # Try to copy from S3 cache
                result_cp = subprocess.run(
                    ["aws", "s3", "cp", s3_cached_path, local_path],
                    capture_output=True,
                    text=True,
                    timeout=900
                )
                
                if result_cp.returncode == 0 and os.path.exists(local_path):
                    file_size = os.path.getsize(local_path)
                    print(f"[SWEEP] ✓ Cache HIT: {file_basename} ({file_size / 1e6:.1f} MB from S3 cache)", file=sys.stderr)
                else:
                    print(f"[SWEEP] Cache MISS: {file_basename}, downloading from source...", file=sys.stderr)
                    s3_cached_path = None  # Will trigger upload after download
            
            # If not in cache (or no cache), download from HTTP
            if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
                print(f"[SWEEP] Downloading from HTTP: {original_path}", file=sys.stderr)
                
                with requests.get(original_path, stream=True, timeout=600) as resp:
                    resp.raise_for_status()
                    total_size = int(resp.headers.get('content-length', 0))
                    downloaded = 0
                    
                    with open(local_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                # Log progress every 100 MB
                                if total_size > 0 and downloaded % (100 * 1024 * 1024) < 1024 * 1024:
                                    pct = 100.0 * downloaded / total_size
                                    print(f"[SWEEP] Download progress: {downloaded / 1e6:.0f} MB / {total_size / 1e6:.0f} MB ({pct:.0f}%)", file=sys.stderr)
                
                file_size = os.path.getsize(local_path)
                download_time = time.time() - start_time
                print(f"[SWEEP] ✓ Downloaded {file_size / 1e6:.1f} MB in {download_time:.1f}s ({file_size / download_time / 1e6:.1f} MB/s)", file=sys.stderr)
                
                # Upload to S3 cache for future runs
                if s3_cache_prefix and file_size > 0:
                    s3_dest = f"{s3_cache_prefix.rstrip('/')}/{file_basename}"
                    print(f"[SWEEP] Uploading to S3 cache: {s3_dest}", file=sys.stderr)
                    
                    result_upload = subprocess.run(
                        ["aws", "s3", "cp", local_path, s3_dest],
                        capture_output=True,
                        text=True,
                        timeout=900
                    )
                    
                    if result_upload.returncode == 0:
                        print(f"[SWEEP] ✓ Cached to S3: {s3_dest}", file=sys.stderr)
                    else:
                        print(f"[SWEEP] ⚠ Failed to cache to S3: {result_upload.stderr}", file=sys.stderr)
        
        elif url.scheme in ("s3", "s3a"):
            import subprocess
            
            file_basename = os.path.basename(url.path) or "sweep.fits"
            temp_dir = tempfile.gettempdir()
            local_path_temp = os.path.join(temp_dir, f"sweep_{os.getpid()}_{file_basename}")
            
            print(f"[SWEEP] Copying from S3: {original_path}", file=sys.stderr)
            
            result_cp = subprocess.run(
                ["aws", "s3", "cp", original_path, local_path_temp],
                capture_output=True,
                text=True,
                timeout=900  # 15 min for large files
            )
            if result_cp.returncode != 0:
                print(f"[SWEEP] S3 copy failed: {result_cp.stderr}", file=sys.stderr)
                return []
            
            local_path = local_path_temp
            temp_file_created = True
            file_size = os.path.getsize(local_path)
            download_time = time.time() - start_time
            print(f"[SWEEP] ✓ Copied {file_size / 1e6:.1f} MB from S3 in {download_time:.1f}s", file=sys.stderr)
        else:
            # Local file
            file_basename = os.path.basename(local_path)

        if not os.path.exists(local_path):
            print(f"[SWEEP] ERROR: File not found: {local_path}", file=sys.stderr)
            return []

        # ---- Step 2: Open FITS and get metadata ----
        process_start = time.time()
        
        with fits.open(local_path, memmap=True) as hdul:
            data = hdul[1].data
            nrows = len(data)
            n_total = nrows

            # Normalize column names
            colnames = [c.lower() for c in data.columns.names]
            lower_to_idx = {name: i for i, name in enumerate(colnames)}

            # Check required columns
            required = ["ra", "dec", "flux_r", "flux_z", "flux_w1", "brickname"]
            missing = [c for c in required if c not in lower_to_idx]
            if missing:
                print(f"[SWEEP] ERROR: Missing columns {missing} in {original_path}", file=sys.stderr)
                return []

            # ---- Step 3: Process in chunks ----
            n_chunks = (nrows + chunk_size - 1) // chunk_size
            print(f"[SWEEP] Processing {nrows:,} rows in {n_chunks} chunks of {chunk_size:,}", file=sys.stderr)

            for chunk_idx in range(n_chunks):
                chunk_start_time = time.time()
                start_row = chunk_idx * chunk_size
                end_row = min(start_row + chunk_size, nrows)
                
                # Read only this chunk's rows
                # FITS slicing with memmap reads only the requested rows!
                chunk_slice = slice(start_row, end_row)
                
                ra_chunk = np.array(data.field(lower_to_idx["ra"])[chunk_slice])
                dec_chunk = np.array(data.field(lower_to_idx["dec"])[chunk_slice])
                flux_r_chunk = np.array(data.field(lower_to_idx["flux_r"])[chunk_slice])
                flux_z_chunk = np.array(data.field(lower_to_idx["flux_z"])[chunk_slice])
                flux_w1_chunk = np.array(data.field(lower_to_idx["flux_w1"])[chunk_slice])
                brickname_chunk = np.array(data.field(lower_to_idx["brickname"])[chunk_slice]).astype(str)

                # Footprint filter (OPTIONAL - if all bounds are None, process all objects)
                apply_footprint = (
                    ra_min is not None or ra_max is not None or 
                    dec_min is not None or dec_max is not None
                )
                
                if apply_footprint:
                    mask_sky = np.ones(len(ra_chunk), dtype=bool)
                    if ra_min is not None:
                        mask_sky &= (ra_chunk >= ra_min)
                    if ra_max is not None:
                        mask_sky &= (ra_chunk <= ra_max)
                    if dec_min is not None:
                        mask_sky &= (dec_chunk >= dec_min)
                    if dec_max is not None:
                        mask_sky &= (dec_chunk <= dec_max)
                    chunk_in_footprint = int(np.sum(mask_sky))
                else:
                    # No footprint filter - process ALL objects
                    mask_sky = None
                    chunk_in_footprint = len(ra_chunk)
                
                n_in_footprint += chunk_in_footprint
                chunk_lrg = 0

                if chunk_in_footprint == 0:
                    # Free chunk memory and continue
                    del ra_chunk, dec_chunk, flux_r_chunk, flux_z_chunk, flux_w1_chunk, brickname_chunk
                    if mask_sky is not None:
                        del mask_sky
                else:
                    # Apply footprint mask (or use all if no filter)
                    if mask_sky is not None:
                        flux_r_filt = flux_r_chunk[mask_sky]
                        flux_z_filt = flux_z_chunk[mask_sky]
                        flux_w1_filt = flux_w1_chunk[mask_sky]
                        brickname_filt = brickname_chunk[mask_sky]
                        del ra_chunk, dec_chunk, flux_r_chunk, flux_z_chunk, flux_w1_chunk, brickname_chunk, mask_sky
                    else:
                        # No filter - use all data
                        flux_r_filt = flux_r_chunk
                        flux_z_filt = flux_z_chunk
                        flux_w1_filt = flux_w1_chunk
                        brickname_filt = brickname_chunk
                        del ra_chunk, dec_chunk

                    # LRG selection
                    lrg_mask = build_lrg_mask(
                        flux_r=flux_r_filt,
                        flux_z=flux_z_filt,
                        flux_w1=flux_w1_filt,
                        lrg_z_mag_max=lrg_z_mag_max,
                        lrg_min_r_minus_z=lrg_min_r_minus_z,
                        lrg_min_z_minus_w1=lrg_min_z_minus_w1,
                    )

                    chunk_lrg = int(np.sum(lrg_mask))
                    n_lrg += chunk_lrg

                    if chunk_lrg > 0:
                        bricknames_lrg = brickname_filt[lrg_mask]
                        # Update counter
                        unique, counts = np.unique(bricknames_lrg, return_counts=True)
                        for bname, cnt in zip(unique, counts):
                            brick_counts[bname] += int(cnt)
                        del bricknames_lrg, unique, counts

                    # Free filtered arrays
                    del flux_r_filt, flux_z_filt, flux_w1_filt, brickname_filt, lrg_mask

                # Log chunk progress
                chunk_time = time.time() - chunk_start_time
                print(
                    f"[CHUNK] {file_basename} chunk {chunk_idx + 1}/{n_chunks}: "
                    f"rows {start_row:,}-{end_row:,}, "
                    f"{chunk_in_footprint:,} in footprint, "
                    f"{chunk_lrg:,} LRGs, "
                    f"{chunk_time:.2f}s",
                    file=sys.stderr
                )

            # End of chunk loop
            gc.collect()

        # ---- Step 4: Convert counter to list of tuples ----
        # IMPORTANT: Convert numpy integers to plain Python int to avoid
        # pickle serialization errors when Spark creates DataFrame
        result = [(str(k), int(v)) for k, v in brick_counts.items()]
        
        total_time = time.time() - start_time
        process_time = time.time() - process_start
        
        print(
            f"[SWEEP] ✓ DONE {file_basename}: {n_total:,} rows, "
            f"{n_in_footprint:,} in footprint, {n_lrg:,} LRGs in {len(result)} bricks "
            f"(process: {process_time:.1f}s, total: {total_time:.1f}s)",
            file=sys.stderr
        )

    except Exception as e:
        print(f"[SWEEP] ERROR processing {original_path}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        result = []

    finally:
        if temp_file_created and os.path.exists(local_path):
            try:
                os.remove(local_path)
            except OSError as e:
                print(f"[SWEEP] Cleanup failed: {e}", file=sys.stderr)
        gc.collect()

    return result


# Default chunk size - conservative for reliability
# 100K rows ≈ 30-50 MB RAM per chunk (very safe for 2GB executors)
DEFAULT_CHUNK_SIZE = 100_000


def process_single_sweep(
    path: str,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    lrg_z_mag_max: float,
    lrg_min_r_minus_z: float,
    lrg_min_z_minus_w1: float,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    s3_cache_prefix: str = None,
) -> List[Tuple[str, int]]:
    """Wrapper that calls the chunked implementation."""
    return process_single_sweep_chunked(
        path=path,
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=dec_min,
        dec_max=dec_max,
        lrg_z_mag_max=lrg_z_mag_max,
        lrg_min_r_minus_z=lrg_min_r_minus_z,
        lrg_min_z_minus_w1=lrg_min_z_minus_w1,
        chunk_size=chunk_size,
        s3_cache_prefix=s3_cache_prefix,
    )


def process_sweep_partition(
    paths: Iterable[str],
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    lrg_z_mag_max: float,
    lrg_min_r_minus_z: float,
    lrg_min_z_minus_w1: float,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    s3_cache_prefix: str = "",
) -> Iterable[Tuple[str, int]]:
    """
    MapPartitions function: processes multiple SWEEP paths inside a single
    executor partition.
    
    Each SWEEP file is processed in chunks to avoid OOM. The chunk_size
    parameter controls memory usage vs speed tradeoff.
    
    If s3_cache_prefix is provided, HTTP downloads are cached to S3.
    """
    import socket
    hostname = socket.gethostname()
    
    print(f"[PARTITION-START] Host: {hostname}, ra_min={ra_min}, ra_max={ra_max}, dec_min={dec_min}, dec_max={dec_max}", file=sys.stderr)
    print(f"[PARTITION-START] LRG cuts: z_mag_max={lrg_z_mag_max}, r-z>{lrg_min_r_minus_z}, z-W1>{lrg_min_z_minus_w1}", file=sys.stderr)
    print(f"[PARTITION-START] chunk_size={chunk_size}, s3_cache_prefix={s3_cache_prefix}", file=sys.stderr)
    sys.stderr.flush()
    
    all_pairs: List[Tuple[str, int]] = []
    file_count = 0
    paths_list = list(paths)  # Convert iterator to list to see count
    
    print(f"[PARTITION] Received {len(paths_list)} paths to process", file=sys.stderr)
    for i, p in enumerate(paths_list[:5]):  # Show first 5 paths
        print(f"[PARTITION] Path {i}: {p[:100]}...", file=sys.stderr)
    sys.stderr.flush()
    
    for path in paths_list:
        path = path.strip()
        if not path or path.startswith("#"):
            print(f"[PARTITION] Skipping empty/comment path: '{path[:50]}'", file=sys.stderr)
            continue
        
        file_count += 1
        print(f"[PARTITION] Processing file {file_count}: {path}", file=sys.stderr)
        sys.stderr.flush()
        
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
                chunk_size=chunk_size,
                s3_cache_prefix=s3_cache_prefix or None,
            )
            print(f"[PARTITION] File {file_count} returned {len(pairs)} (brick, count) pairs", file=sys.stderr)
            all_pairs.extend(pairs)
        except Exception as exc:
            # Avoid killing the job for one bad file; just log to executor stderr
            print(f"[PARTITION-ERROR] Failed processing SWEEP {path}: {exc!r}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
    
    print(f"[PARTITION-END] Processed {file_count} files, found {len(all_pairs)} (brick, count) pairs total", file=sys.stderr)
    sys.stderr.flush()
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

    # Phase 1.5 footprint filter (OPTIONAL - omit for full-sky comprehensive scan)
    parser.add_argument(
        "--ra-min", type=float, default=None,
        help="Min RA in degrees. If omitted, no RA filter (full sky).",
    )
    parser.add_argument(
        "--ra-max", type=float, default=None,
        help="Max RA in degrees. If omitted, no RA filter (full sky).",
    )
    parser.add_argument(
        "--dec-min", type=float, default=None,
        help="Min Dec in degrees. If omitted, no Dec filter (full sky).",
    )
    parser.add_argument(
        "--dec-max", type=float, default=None,
        help="Max Dec in degrees. If omitted, no Dec filter (full sky).",
    )

    parser.add_argument("--lrg-z-mag-max", type=float, default=20.4)
    parser.add_argument("--lrg-min-r-minus-z", type=float, default=0.4)
    parser.add_argument("--lrg-min-z-minus-w1", type=float, default=0.8)

    parser.add_argument(
        "--num-partitions",
        type=int,
        default=0,
        help=(
            "Number of Spark partitions to use for SWEEP file list. "
            "If 0, will auto-set to number of SWEEP files (recommended)."
        ),
    )
    
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=0,
        help=(
            "Maximum number of SWEEP files to process. "
            "If 0, process all files (default). Use for testing."
        ),
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            f"Number of rows to process at a time per SWEEP file. "
            f"Lower = less memory, more iterations. "
            f"Default: {DEFAULT_CHUNK_SIZE:,} (~30-50 MB RAM). "
            f"Use 50000 for very low memory, 500000 for faster processing."
        ),
    )
    
    parser.add_argument(
        "--s3-cache-prefix",
        type=str,
        default="",
        help=(
            "S3 prefix to cache downloaded SWEEP files. "
            "If provided, HTTP downloads will be cached to S3 for future runs. "
            "Example: s3://my-bucket/sweep-cache/ "
            "On subsequent runs, job checks S3 cache before downloading from HTTP."
        ),
    )

    args = parser.parse_args()

    print("=" * 70, file=sys.stderr)
    print("PHASE 1.5: LRG DENSITY PER BRICK - STARTING", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Timestamp: {datetime.utcnow().isoformat()}", file=sys.stderr)
    print(f"sweep-index-path: {args.sweep_index_path}", file=sys.stderr)
    print(f"output-prefix: {args.output_prefix}", file=sys.stderr)
    
    # Footprint filter status
    has_footprint = (
        args.ra_min is not None or args.ra_max is not None or
        args.dec_min is not None or args.dec_max is not None
    )
    if has_footprint:
        print(f"Footprint: RA [{args.ra_min}, {args.ra_max}], Dec [{args.dec_min}, {args.dec_max}]", file=sys.stderr)
    else:
        print("Footprint: FULL SKY (no RA/Dec filter - comprehensive scan)", file=sys.stderr)
    print(f"LRG cuts: z_mag_max={args.lrg_z_mag_max}, r-z>{args.lrg_min_r_minus_z}, z-W1>{args.lrg_min_z_minus_w1}", file=sys.stderr)
    print(f"num-partitions: {args.num_partitions} (0=auto)", file=sys.stderr)
    print(f"max-sweeps: {args.max_sweeps} (0=all)", file=sys.stderr)
    print(f"chunk-size: {args.chunk_size:,} rows (~{args.chunk_size * 50 // 1_000_000} MB RAM per chunk)", file=sys.stderr)
    print(f"s3-cache-prefix: {args.s3_cache_prefix or '(disabled)'}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Create Spark session
    spark = (
        SparkSession.builder
        .appName("dark-halo-scope-phase1p5-lrg-density")
        .getOrCreate()
    )
    sc = spark.sparkContext

    # Log Spark configuration
    print(f"[SPARK] Application ID: {sc.applicationId}", file=sys.stderr)
    print(f"[SPARK] Spark version: {sc.version}", file=sys.stderr)
    print(f"[SPARK] Master: {sc.master}", file=sys.stderr)
    print(f"[SPARK] Default parallelism: {sc.defaultParallelism}", file=sys.stderr)

    # Read sweep index as lines (paths)
    print(f"[SPARK] Reading sweep index from: {args.sweep_index_path}", file=sys.stderr)
    paths_rdd = sc.textFile(args.sweep_index_path)
    
    # Filter out empty lines and comments
    paths_rdd = paths_rdd.filter(lambda x: x.strip() and not x.strip().startswith("#"))
    
    # Count total paths
    total_paths = paths_rdd.count()
    print(f"[SPARK] Found {total_paths} SWEEP paths in index", file=sys.stderr)
    
    if total_paths == 0:
        print("[ERROR] No SWEEP paths found! Check your index file.", file=sys.stderr)
        spark.stop()
        sys.exit(1)

    # Limit number of sweeps if requested (for testing)
    if args.max_sweeps > 0 and args.max_sweeps < total_paths:
        print(f"[SPARK] Limiting to first {args.max_sweeps} SWEEP files (--max-sweeps)", file=sys.stderr)
        # Take first N paths
        paths_list = paths_rdd.take(args.max_sweeps)
        paths_rdd = sc.parallelize(paths_list)
        total_paths = args.max_sweeps

    # Set partitions (ideally 1 file per partition for memory efficiency)
    num_partitions = args.num_partitions if args.num_partitions > 0 else max(total_paths, 1)
    print(f"[SPARK] Using {num_partitions} partitions", file=sys.stderr)
    print(f"[SPARK] Chunk size: {args.chunk_size:,} rows per chunk", file=sys.stderr)
    
    # DEBUG: Show the actual paths being processed
    print("[DEBUG] First 5 paths to be processed:", file=sys.stderr)
    sample_paths = paths_rdd.take(5)
    for i, p in enumerate(sample_paths):
        print(f"[DEBUG]   {i}: {p}", file=sys.stderr)
    sys.stderr.flush()
    
    # Repartition for parallel processing
    if paths_rdd.getNumPartitions() != num_partitions:
        paths_rdd = paths_rdd.repartition(num_partitions)

    # Capture values for use in lambda (avoid late binding issues)
    chunk_size = args.chunk_size
    s3_cache_prefix = args.s3_cache_prefix
    
    # Capture RA/Dec for debug
    ra_min_val = args.ra_min
    ra_max_val = args.ra_max
    dec_min_val = args.dec_min
    dec_max_val = args.dec_max
    
    print(f"[DEBUG] RA filter: {ra_min_val} to {ra_max_val}", file=sys.stderr)
    print(f"[DEBUG] Dec filter: {dec_min_val} to {dec_max_val}", file=sys.stderr)
    print(f"[DEBUG] s3_cache_prefix: {s3_cache_prefix}", file=sys.stderr)
    sys.stderr.flush()

    # MapPartitions: (brickname, count)
    print("[SPARK] Starting map phase (processing SWEEP files)...", file=sys.stderr)
    sys.stderr.flush()
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
            chunk_size=chunk_size,
            s3_cache_prefix=s3_cache_prefix,
        )
    )

    # Reduce by brickname
    print("[SPARK] Starting reduce phase (aggregating by brickname)...", file=sys.stderr)
    reduced_rdd = pairs_rdd.reduceByKey(lambda a, b: a + b)
    
    # Ensure all values are native Python types (not numpy) for Spark serialization
    # This is critical to avoid pickle errors when creating DataFrame
    reduced_rdd = reduced_rdd.map(lambda x: (str(x[0]), int(x[1])))

    # Count results
    result_count = reduced_rdd.count()
    print(f"[SPARK] Found LRGs in {result_count} unique bricks", file=sys.stderr)

    # Convert to a small Spark DataFrame and write to S3 as CSV
    schema = StructType(
        [
            StructField("brickname", StringType(), nullable=False),
            StructField("lrg_count", LongType(), nullable=False),
        ]
    )

    df = spark.createDataFrame(reduced_rdd, schema=schema)

    output_dir = os.path.join(args.output_prefix.rstrip("/"), "lrg_counts_per_brick")
    print(f"[SPARK] Writing output to: {output_dir}", file=sys.stderr)
    
    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(output_dir)
    )

    print("=" * 70, file=sys.stderr)
    print("PHASE 1.5: LRG DENSITY PER BRICK - COMPLETE", file=sys.stderr)
    print(f"Timestamp: {datetime.utcnow().isoformat()}", file=sys.stderr)
    print(f"Processed {total_paths} SWEEP files", file=sys.stderr)
    print(f"Found LRGs in {result_count} bricks", file=sys.stderr)
    print(f"Output: {output_dir}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    spark.stop()


if __name__ == "__main__":
    main()


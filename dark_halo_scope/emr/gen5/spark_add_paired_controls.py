#!/usr/bin/env python3
"""EMR Job: Add paired controls to phase4c parquet.

OPTIMIZED VERSION v2: 
- Parallel FITS downloads (3 bands concurrently via ThreadPoolExecutor)
- Retry with exponential backoff (boto3 adaptive mode, 5 attempts)
- Explicit error handling (FetchError exception, no silent failures)
- Spark accumulators for global stats tracking
- Optimized write (coalesce instead of repartition)

For each positive sample (is_control=0), fetches the base LRG cutout
(without injection) from the coadd cache and stores it as ctrl_stamp_npz.
This enables true paired training without runtime fetching.

Input: s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_production/.../stamps/train_stamp64_bandsgrz_cosmos
Output: s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_paired/

Schema additions:
- ctrl_stamp_npz: Binary NPZ of base LRG cutout (C, H, W) without injection

Validation checks applied (from lessons_learned.md):
- NaN/Inf validation on fetched cutouts (Lesson 4.4)
- boto3/astropy imported inside functions (Lesson 1.1)
- stamp_size from data, not hardcoded (Lesson: match input schema)
- test-limit option for smoke testing (Lesson 4.2)

Performance optimizations:
- Parallel S3 downloads (3x speedup per brick)
- Per-partition LRU cache for FITS files (max 3 bricks to avoid OOM)
- Repartition by brickname for cache locality
"""

import argparse
import io
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql import types as T

# Constants
DEFAULT_STAMP_SIZE = 64  # Default, but use stamp_size from data when available
BANDS = ("g", "r", "z")
PIXEL_SCALE = 0.262  # arcsec/pixel

# LRU cache size: Each brick's FITS data is ~150MB (3 bands), limit to 3 bricks = ~450MB per executor
MAX_CACHED_BRICKS = 3

# S3 paths
S3_COADD_CACHE = "s3://darkhaloscope/dr10/coadd_cache"
S3_INPUT = "s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_production/phase4c/v5_cosmos_production/stamps/train_stamp64_bandsgrz_cosmos"
S3_OUTPUT = "s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_paired"

# S3 bucket
S3_BUCKET = "darkhaloscope"


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class FetchError(Exception):
    """Explicit error for S3 fetch failures. Not silently swallowed."""
    pass


# =============================================================================
# LRU CACHE
# =============================================================================

class LRUCache(OrderedDict):
    """Simple LRU cache using OrderedDict."""
    
    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize
    
    def get(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None
    
    def put(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.maxsize:
            self.popitem(last=False)


# =============================================================================
# S3 FETCH FUNCTIONS WITH RETRY AND PARALLEL DOWNLOAD
# =============================================================================

def _get_s3_client():
    """Create S3 client with retry configuration.
    
    Uses adaptive retry mode with 5 attempts and exponential backoff.
    Imports boto3 inside function per Lesson 1.1.
    """
    import boto3
    from botocore.config import Config
    
    config = Config(
        retries={
            'max_attempts': 5,
            'mode': 'adaptive'  # Exponential backoff with jitter
        },
        connect_timeout=10,
        read_timeout=60  # FITS files can be large
    )
    
    return boto3.client("s3", config=config, region_name="us-east-2")


def _fetch_single_fits(brickname: str, band: str) -> Tuple[np.ndarray, 'WCS']:
    """Fetch a single FITS image and WCS for a brick/band.
    
    Returns (image_array, wcs).
    Raises FetchError on any failure (no silent returns).
    """
    from botocore.exceptions import ClientError
    
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except ImportError as e:
        raise FetchError(f"astropy not available: {e}")
    
    s3 = _get_s3_client()
    key = f"dr10/coadd_cache/{brickname}/legacysurvey-{brickname}-image-{band}.fits.fz"
    
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=key)
        data = response["Body"].read()
        
        with fits.open(io.BytesIO(data)) as hdul:
            img = hdul[1].data.astype(np.float32)
            wcs = WCS(hdul[1].header)
            return (img, wcs)
            
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        raise FetchError(f"S3 ClientError {error_code} for {brickname}/{band}: {e}")
    except Exception as e:
        raise FetchError(f"Unexpected error fetching {brickname}/{band}: {type(e).__name__}: {e}")


def _fetch_brick_data_parallel(brickname: str, bands: Tuple[str, ...] = BANDS) -> Dict:
    """Fetch all band data for a brick IN PARALLEL.
    
    Uses ThreadPoolExecutor to fetch g, r, z bands concurrently.
    Returns dict with 'images' list and 'wcs' (from r-band).
    Raises FetchError if any band fails.
    """
    results = {}
    errors = []
    
    # Fetch all 3 bands in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_fetch_single_fits, brickname, band): band 
            for band in bands
        }
        
        for future in as_completed(futures):
            band = futures[future]
            try:
                result = future.result()
                results[band] = result
            except FetchError as e:
                errors.append(str(e))
            except Exception as e:
                errors.append(f"Unexpected error for {brickname}/{band}: {e}")
    
    # If any band failed, raise error with all failure details
    if errors:
        raise FetchError(f"Failed to fetch brick {brickname}: {'; '.join(errors)}")
    
    # Build output structure
    images = [results[band][0] for band in bands]  # Ordered g, r, z
    wcs_main = results["r"][1]  # Use r-band WCS as reference
    
    return {"images": images, "wcs": wcs_main}


# =============================================================================
# CUTOUT EXTRACTION AND ENCODING
# =============================================================================

def _extract_cutout(
    brick_data: Dict,
    ra: float,
    dec: float,
    stamp_size: int,
) -> Optional[np.ndarray]:
    """Extract cutout from cached brick data.
    
    Returns (C, H, W) array or None if out of bounds.
    """
    images = brick_data["images"]
    wcs = brick_data["wcs"]
    
    # Convert RA/Dec to pixel coordinates
    x, y = wcs.all_world2pix(ra, dec, 0)
    x, y = int(np.round(float(x))), int(np.round(float(y)))
    
    half = stamp_size // 2
    y0, y1 = y - half, y + half
    x0, x1 = x - half, x + half
    
    # Check bounds
    h, w = images[0].shape
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        return None
    
    cutouts = []
    for img in images:
        cutout = img[y0:y1, x0:x1]
        if cutout.shape != (stamp_size, stamp_size):
            return None
        cutouts.append(cutout)
    
    return np.stack(cutouts, axis=0)  # (C, H, W)


def _encode_npz(arr: np.ndarray) -> bytes:
    """Encode numpy array to NPZ bytes.
    
    Match the format expected by training code:
    Keys: image_g, image_r, image_z (each 64x64)
    Input arr is (C, H, W) with C=3 for g, r, z bands.
    """
    buf = io.BytesIO()
    np.savez_compressed(buf, 
                        image_g=arr[0],
                        image_r=arr[1],
                        image_z=arr[2])
    return buf.getvalue()


# =============================================================================
# PARTITION PROCESSOR WITH ACCUMULATORS
# =============================================================================

def make_process_partition(success_acc, failure_acc, cache_hit_acc, cache_miss_acc):
    """Create partition processor closure with accumulator access.
    
    Returns a function that processes a partition while updating global accumulators.
    """
    
    def process_partition(rows: Iterator[Row]) -> Iterator[Row]:
        """Process a partition of rows with LRU-cached FITS files.
        
        This function is called once per partition. It maintains a cache
        of recently-used brick FITS data to avoid redundant S3 fetches.
        """
        # Initialize LRU cache for this partition
        cache = LRUCache(MAX_CACHED_BRICKS)
        
        # Local stats (for logging)
        local_processed = 0
        local_success = 0
        local_failed = 0
        local_cache_hits = 0
        local_cache_misses = 0
        
        for row in rows:
            local_processed += 1
            
            # Extract row fields
            ra = row.ra
            dec = row.dec
            brickname = row.brickname
            stamp_size = row.stamp_size if hasattr(row, 'stamp_size') and row.stamp_size else DEFAULT_STAMP_SIZE
            
            # Try to get brick data from cache
            brick_data = cache.get(brickname)
            
            if brick_data is None:
                # Cache miss: fetch from S3 with parallel download
                local_cache_misses += 1
                
                try:
                    brick_data = _fetch_brick_data_parallel(brickname)
                except FetchError as e:
                    # Log explicit error - not silently swallowed
                    print(f"[FETCH_FAILED] {e}")
                    local_failed += 1
                    failure_acc.add(1)
                    continue
                
                # Add to cache
                cache.put(brickname, brick_data)
            else:
                local_cache_hits += 1
            
            # Extract cutout from cached brick data
            cutout = _extract_cutout(brick_data, ra, dec, stamp_size)
            
            if cutout is None:
                print(f"[CUTOUT_FAILED] Out of bounds: {brickname} ra={ra:.6f} dec={dec:.6f}")
                local_failed += 1
                failure_acc.add(1)
                continue
            
            # Validate no NaN/Inf (Lesson 4.4)
            if not np.isfinite(cutout).all():
                print(f"[VALIDATION_FAILED] NaN/Inf in cutout: {brickname} ra={ra:.6f} dec={dec:.6f}")
                local_failed += 1
                failure_acc.add(1)
                continue
            
            # Encode to NPZ
            ctrl_stamp_npz = _encode_npz(cutout)
            
            # Create new row with ctrl_stamp_npz added
            row_dict = row.asDict()
            row_dict["ctrl_stamp_npz"] = ctrl_stamp_npz
            
            local_success += 1
            success_acc.add(1)
            yield Row(**row_dict)
        
        # Update cache accumulators
        cache_hit_acc.add(local_cache_hits)
        cache_miss_acc.add(local_cache_misses)
        
        # Log partition stats
        if local_processed > 0:
            hit_rate = local_cache_hits / local_processed * 100 if local_processed > 0 else 0
            print(f"[PARTITION_COMPLETE] Processed: {local_processed}, Success: {local_success}, "
                  f"Failed: {local_failed}, Cache Hits: {local_cache_hits}, "
                  f"Cache Misses: {local_cache_misses}, Hit Rate: {hit_rate:.1f}%")
    
    return process_partition


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Add paired controls to phase4c parquet")
    parser.add_argument("--input", default=S3_INPUT, help="Input parquet path")
    parser.add_argument("--output", default=S3_OUTPUT, help="Output parquet path")
    parser.add_argument("--split", default=None, help="Process only this split (train/val/test)")
    parser.add_argument("--dry-run", action="store_true", help="Just count rows, don't process")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit rows for smoke test")
    parser.add_argument("--partitions", type=int, default=1000, help="Output partitions")
    parser.add_argument("--shuffle-partitions", type=int, default=2000, help="Shuffle partitions for sorting")
    args = parser.parse_args()
    
    spark = SparkSession.builder \
        .appName("AddPairedControls_v2") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions)) \
        .getOrCreate()
    
    # Create accumulators for global stats tracking
    success_acc = spark.sparkContext.accumulator(0)
    failure_acc = spark.sparkContext.accumulator(0)
    cache_hit_acc = spark.sparkContext.accumulator(0)
    cache_miss_acc = spark.sparkContext.accumulator(0)
    
    # Read input
    print(f"[INFO] Reading from {args.input}")
    df = spark.read.parquet(args.input)
    
    if args.split:
        df = df.filter(F.col("region_split") == args.split)
        print(f"[INFO] Filtered to split={args.split}")
    
    # Filter to positives only (is_control=0)
    # Controls become redundant since ctrl_stamp_npz serves that purpose
    df = df.filter(F.col("is_control") == 0)
    
    # Apply test limit for smoke testing (Lesson 4.2)
    if args.test_limit:
        df = df.limit(args.test_limit)
        print(f"[INFO] Limited to {args.test_limit} rows for smoke test")
    
    if args.dry_run:
        # Show schema and sample
        total_count = df.count()
        print(f"[INFO] Total positive samples: {total_count:,}")
        df.printSchema()
        df.select("ra", "dec", "brickname", "is_control", "region_split", "theta_e_arcsec").show(10)
        
        # Count unique bricknames for optimization insight
        brick_count = df.select("brickname").distinct().count()
        print(f"[INFO] Unique bricknames: {brick_count:,}")
        print(f"[INFO] Estimated S3 fetches: {brick_count * 3:,} (3 bands per brick)")
        print(f"[INFO] Rows per brick (avg): {total_count / brick_count:.1f}")
        
        spark.stop()
        return
    
    # OPTIMIZATION: Repartition by brickname so all rows for the same brick
    # go to the same partition, then sort within partition for cache locality
    print("[INFO] Repartitioning by brickname to maximize cache hits...")
    
    # Use shuffle partitions directly - Spark adaptive will coalesce if needed
    df_sorted = df.repartition(args.shuffle_partitions, "brickname").sortWithinPartitions("brickname")
    
    # Get the output schema (add ctrl_stamp_npz column)
    output_schema = df.schema.add(T.StructField("ctrl_stamp_npz", T.BinaryType(), True))
    
    # Create partition processor with accumulator access
    process_fn = make_process_partition(success_acc, failure_acc, cache_hit_acc, cache_miss_acc)
    
    # Process using mapPartitions with LRU cache
    print("[INFO] Processing with parallel FITS downloads and LRU cache...")
    result_rdd = df_sorted.rdd.mapPartitions(process_fn)
    
    # Convert back to DataFrame
    result_df = spark.createDataFrame(result_rdd, output_schema)
    
    # Write output - use coalesce instead of repartition (avoids full shuffle)
    output_path = args.output
    if args.split:
        output_path = f"{output_path}/{args.split}"
    
    print(f"[INFO] Writing to {output_path}")
    
    # If we have more partitions than needed, coalesce down
    # This is more efficient than repartition as it avoids a full shuffle
    if args.partitions < args.shuffle_partitions:
        result_df = result_df.coalesce(args.partitions)
    
    result_df.write.mode("overwrite").parquet(output_path)
    
    # Print final stats from accumulators
    print("=" * 60)
    print("[FINAL STATS]")
    print(f"  Success: {success_acc.value:,}")
    print(f"  Failed: {failure_acc.value:,}")
    print(f"  Cache Hits: {cache_hit_acc.value:,}")
    print(f"  Cache Misses: {cache_miss_acc.value:,}")
    if success_acc.value + failure_acc.value > 0:
        success_rate = success_acc.value / (success_acc.value + failure_acc.value) * 100
        print(f"  Success Rate: {success_rate:.2f}%")
    if cache_hit_acc.value + cache_miss_acc.value > 0:
        cache_hit_rate = cache_hit_acc.value / (cache_hit_acc.value + cache_miss_acc.value) * 100
        print(f"  Cache Hit Rate: {cache_hit_rate:.2f}%")
    print("=" * 60)
    
    # Quick verification (read count from output - single action)
    out_count = spark.read.parquet(output_path).count()
    print(f"[INFO] Output rows verified: {out_count:,}")
    print(f"[INFO] Each row now has: stamp_npz (LRG+injection), ctrl_stamp_npz (base LRG)")
    
    spark.stop()
    print("[DONE] Paired controls added successfully")


if __name__ == "__main__":
    main()

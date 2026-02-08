#!/usr/bin/env python3
"""
Spark job to download DR10 sweep files from NERSC to S3 with compression.

Features:
- Parallel download across EMR workers
- Gzip compression before S3 upload
- Resumable: skips files already in S3
- Retry with exponential backoff for NERSC throttling
- Clean S3 exception handling with retries
- Final manifest with download status

Usage:
    spark-submit spark_download_sweeps.py \
        --sweep-urls s3://darkhaloscope/dr10/sweep_urls.txt \
        --output-prefix s3://darkhaloscope/dr10/sweeps/ \
        --manifest-output s3://darkhaloscope/dr10/sweeps_manifest/
"""

import argparse
import gzip
import io
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

PIPELINE_VERSION = "1.0.0"
GIT_COMMIT = os.environ.get("GIT_COMMIT", "unknown")

# Retry configuration
MAX_RETRIES_NERSC = 5
INITIAL_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0
BACKOFF_MULTIPLIER = 2.0

# S3 configuration
S3_MAX_RETRIES = 10
S3_RETRY_MODE = "adaptive"


# =============================================================================
# S3 UTILITIES
# =============================================================================

def get_s3_client():
    """Get S3 client with retry configuration."""
    import boto3
    from botocore.config import Config
    
    config = Config(
        retries={
            "max_attempts": S3_MAX_RETRIES,
            "mode": S3_RETRY_MODE,
        },
        connect_timeout=30,
        read_timeout=60,
    )
    
    return boto3.client("s3", region_name="us-east-2", config=config)


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    if s3_uri.startswith("s3://"):
        path = s3_uri[5:]
    elif s3_uri.startswith("s3a://"):
        path = s3_uri[6:]
    else:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    
    parts = path.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def check_s3_exists(s3_client, bucket: str, key: str) -> Optional[int]:
    """
    Check if file exists in S3 and return its size.
    Returns None if not exists, size in bytes if exists.
    """
    import botocore.exceptions
    
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        return response["ContentLength"]
    except botocore.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ["404", "NoSuchKey"]:
            return None
        raise


def upload_to_s3_with_retry(
    s3_client,
    data: bytes,
    bucket: str,
    key: str,
    content_type: str = "application/gzip",
) -> int:
    """
    Upload bytes to S3 with retry handling and jitter.
    Returns file size in bytes.
    """
    import botocore.exceptions
    import random
    
    last_error = None
    for attempt in range(S3_MAX_RETRIES):
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
            return len(data)
        except botocore.exceptions.ClientError as e:
            last_error = e
            error_code = e.response.get("Error", {}).get("Code", "")
            
            # Retry on throttling or transient errors
            if error_code in ["SlowDown", "ServiceUnavailable", "InternalError", "RequestTimeout"]:
                base_wait = min(
                    INITIAL_BACKOFF_SECONDS * (BACKOFF_MULTIPLIER ** attempt),
                    MAX_BACKOFF_SECONDS
                )
                wait_time = base_wait + (base_wait * random.uniform(0, 0.5))
                print(f"[S3] Retry {attempt + 1}/{S3_MAX_RETRIES} after {wait_time:.1f}s: {error_code}")
                time.sleep(wait_time)
                continue
            raise
        except Exception as e:
            last_error = e
            base_wait = min(
                INITIAL_BACKOFF_SECONDS * (BACKOFF_MULTIPLIER ** attempt),
                MAX_BACKOFF_SECONDS
            )
            wait_time = base_wait + (base_wait * random.uniform(0, 0.5))
            print(f"[S3] Retry {attempt + 1}/{S3_MAX_RETRIES} after {wait_time:.1f}s: {e}")
            time.sleep(wait_time)
    
    raise last_error


# =============================================================================
# NERSC DOWNLOAD
# =============================================================================

def get_backoff_with_jitter(attempt: int) -> float:
    """Calculate exponential backoff with random jitter to avoid thundering herd."""
    import random
    base_wait = min(
        INITIAL_BACKOFF_SECONDS * (BACKOFF_MULTIPLIER ** attempt),
        MAX_BACKOFF_SECONDS
    )
    # Add 0-50% jitter
    jitter = base_wait * random.uniform(0, 0.5)
    return base_wait + jitter


def download_from_nersc_with_retry(url: str) -> bytes:
    """
    Download file from NERSC with retry and exponential backoff + jitter.
    Returns raw bytes.
    """
    import requests
    
    last_error = None
    for attempt in range(MAX_RETRIES_NERSC):
        try:
            # Use timeout tuple: (connect_timeout, read_timeout)
            response = requests.get(
                url,
                timeout=(30, 300),  # 30s connect, 5min read for large files
            )
            
            if response.status_code == 200:
                return response.content
            
            elif response.status_code == 429:
                # Rate limited - back off with jitter
                wait_time = get_backoff_with_jitter(attempt)
                print(f"[NERSC] Rate limited, retry {attempt + 1}/{MAX_RETRIES_NERSC} after {wait_time:.1f}s")
                time.sleep(wait_time)
                continue
            
            elif response.status_code in [500, 502, 503, 504]:
                # Server error - retry with jitter
                wait_time = get_backoff_with_jitter(attempt)
                print(f"[NERSC] Server error {response.status_code}, retry {attempt + 1}/{MAX_RETRIES_NERSC} after {wait_time:.1f}s")
                time.sleep(wait_time)
                continue
            
            else:
                raise Exception(f"HTTP {response.status_code}: {response.reason}")
                
        except requests.exceptions.Timeout as e:
            last_error = e
            wait_time = get_backoff_with_jitter(attempt)
            print(f"[NERSC] Timeout, retry {attempt + 1}/{MAX_RETRIES_NERSC} after {wait_time:.1f}s")
            time.sleep(wait_time)
            
        except requests.exceptions.ConnectionError as e:
            last_error = e
            wait_time = get_backoff_with_jitter(attempt)
            print(f"[NERSC] Connection error, retry {attempt + 1}/{MAX_RETRIES_NERSC} after {wait_time:.1f}s")
            time.sleep(wait_time)
            
        except Exception as e:
            last_error = e
            wait_time = get_backoff_with_jitter(attempt)
            print(f"[NERSC] Error: {e}, retry {attempt + 1}/{MAX_RETRIES_NERSC} after {wait_time:.1f}s")
            time.sleep(wait_time)
    
    raise Exception(f"Failed to download {url} after {MAX_RETRIES_NERSC} retries: {last_error}")


def compress_fits_data(data: bytes) -> bytes:
    """Compress FITS data with gzip."""
    compressed = gzip.compress(data, compresslevel=6)
    return compressed


# =============================================================================
# WORKER FUNCTION
# =============================================================================

def process_url_partition(
    urls: Iterator[str],
    output_bucket: str,
    output_prefix: str,
) -> Iterator[Dict]:
    """Process a partition of URLs with shared S3 client."""
    # Create S3 client once per partition for connection reuse
    s3_client = get_s3_client()
    
    for url in urls:
        result = process_url_with_client(url, output_bucket, output_prefix, s3_client)
        yield result


def process_url_with_client(
    url: str,
    output_bucket: str,
    output_prefix: str,
    s3_client,
) -> Dict:
    """
    Process a single NERSC URL with provided S3 client.
    """
    import traceback
    
    # Extract filename from URL
    filename = os.path.basename(urlparse(url).path)
    compressed_filename = filename + ".gz" if not filename.endswith(".gz") else filename
    
    # S3 target key - output_prefix is just the key prefix (e.g., "dr10/sweeps")
    s3_key = output_prefix.rstrip("/") + "/" + compressed_filename
    
    result = {
        "source_url": url,
        "s3_uri": f"s3://{output_bucket}/{s3_key}",
        "filename": filename,
        "compressed_filename": compressed_filename,
        "original_size_bytes": 0,
        "compressed_size_bytes": 0,
        "compression_ratio": 0.0,
        "status": "unknown",
        "error_message": "",
        "download_time_seconds": 0.0,
        "upload_time_seconds": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    try:
        # Check if already exists in S3
        existing_size = check_s3_exists(s3_client, output_bucket, s3_key)
        if existing_size is not None and existing_size > 0:
            result["status"] = "skipped"
            result["compressed_size_bytes"] = existing_size
            print(f"[SKIP] {filename} already exists in S3 ({existing_size:,} bytes)")
            return result
        
        # Download from NERSC
        print(f"[DOWNLOAD] {filename} from NERSC...")
        t_download_start = time.time()
        raw_data = download_from_nersc_with_retry(url)
        result["download_time_seconds"] = time.time() - t_download_start
        result["original_size_bytes"] = len(raw_data)
        print(f"[DOWNLOAD] {filename}: {len(raw_data):,} bytes in {result['download_time_seconds']:.1f}s")
        
        # Compress
        print(f"[COMPRESS] {filename}...")
        compressed_data = compress_fits_data(raw_data)
        result["compressed_size_bytes"] = len(compressed_data)
        result["compression_ratio"] = len(raw_data) / len(compressed_data) if len(compressed_data) > 0 else 0
        print(f"[COMPRESS] {filename}: {len(raw_data):,} -> {len(compressed_data):,} bytes ({result['compression_ratio']:.2f}x)")
        
        # Free memory
        del raw_data
        
        # Upload to S3
        print(f"[UPLOAD] {compressed_filename} to S3...")
        t_upload_start = time.time()
        upload_to_s3_with_retry(s3_client, compressed_data, output_bucket, s3_key)
        result["upload_time_seconds"] = time.time() - t_upload_start
        print(f"[UPLOAD] {compressed_filename}: {len(compressed_data):,} bytes in {result['upload_time_seconds']:.1f}s")
        
        # Free memory
        del compressed_data
        
        result["status"] = "success"
        print(f"[SUCCESS] {filename} -> {compressed_filename}")
        
    except Exception as e:
        result["status"] = "failed"
        result["error_message"] = str(e)
        print(f"[FAILED] {filename}: {e}")
        traceback.print_exc()
    
    return result


# =============================================================================
# MAIN
# =============================================================================

def create_manifest_schema():
    """Create Spark schema for manifest DataFrame."""
    from pyspark.sql.types import (
        StructType, StructField, StringType, LongType, DoubleType
    )
    
    return StructType([
        StructField("source_url", StringType(), False),
        StructField("s3_uri", StringType(), False),
        StructField("filename", StringType(), False),
        StructField("compressed_filename", StringType(), False),
        StructField("original_size_bytes", LongType(), False),
        StructField("compressed_size_bytes", LongType(), False),
        StructField("compression_ratio", DoubleType(), False),
        StructField("status", StringType(), False),
        StructField("error_message", StringType(), True),
        StructField("download_time_seconds", DoubleType(), False),
        StructField("upload_time_seconds", DoubleType(), False),
        StructField("timestamp", StringType(), False),
    ])


def main():
    parser = argparse.ArgumentParser(description="Download DR10 sweeps from NERSC to S3")
    
    parser.add_argument(
        "--sweep-urls", required=True,
        help="S3 path to sweep_urls.txt file"
    )
    parser.add_argument(
        "--output-prefix", required=True,
        help="S3 prefix for output files (e.g., s3://bucket/dr10/sweeps/)"
    )
    parser.add_argument(
        "--manifest-output", required=True,
        help="S3 path for manifest output"
    )
    parser.add_argument(
        "--test-limit", type=int, default=None,
        help="Limit number of files for testing"
    )
    parser.add_argument(
        "--job-name", type=str, default="sweep-download",
        help="Job name for logging"
    )
    
    args = parser.parse_args()
    
    # Initialize Spark
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder \
        .appName(f"SweepDownload-{args.job_name}") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    try:
        logger.info("=" * 60)
        logger.info(f"Starting sweep download job: {args.job_name}")
        logger.info(f"Pipeline version: {PIPELINE_VERSION}")
        logger.info(f"Sweep URLs: {args.sweep_urls}")
        logger.info(f"Output prefix: {args.output_prefix}")
        logger.info(f"Manifest output: {args.manifest_output}")
        logger.info("=" * 60)
        
        # Parse output S3 location
        output_bucket, output_prefix = parse_s3_uri(args.output_prefix)
        
        # Load sweep URLs
        logger.info("Loading sweep URLs...")
        urls_df = spark.read.text(args.sweep_urls)
        urls = [row.value.strip() for row in urls_df.collect() if row.value.strip()]
        
        logger.info(f"Found {len(urls)} URLs")
        
        # Apply test limit
        if args.test_limit and args.test_limit < len(urls):
            logger.info(f"Applying test limit: {args.test_limit} URLs")
            urls = urls[:args.test_limit]
        
        # Create RDD of URLs with 1:1 partitioning
        num_partitions = min(len(urls), 50000)
        urls_rdd = spark.sparkContext.parallelize(urls, num_partitions)
        
        logger.info(f"Distributing {len(urls)} URLs across {num_partitions} partitions")
        
        # Process URLs
        start_time = time.time()
        
        # Broadcast output location
        output_bucket_bc = spark.sparkContext.broadcast(output_bucket)
        output_prefix_bc = spark.sparkContext.broadcast(output_prefix)
        
        results_rdd = urls_rdd.mapPartitions(
            lambda urls_iter: process_url_partition(
                urls_iter,
                output_bucket_bc.value,
                output_prefix_bc.value,
            )
        )
        
        # Create manifest DataFrame
        manifest_schema = create_manifest_schema()
        manifest_df = spark.createDataFrame(results_rdd, schema=manifest_schema)
        
        # Cache for aggregation
        manifest_df.cache()
        
        # Compute statistics
        total_count = manifest_df.count()
        elapsed_time = time.time() - start_time
        
        success_count = manifest_df.filter(manifest_df.status == "success").count()
        skipped_count = manifest_df.filter(manifest_df.status == "skipped").count()
        failed_count = manifest_df.filter(manifest_df.status == "failed").count()
        
        # Aggregate sizes
        from pyspark.sql.functions import sum as spark_sum, avg
        
        stats = manifest_df.agg(
            spark_sum("original_size_bytes").alias("total_original_bytes"),
            spark_sum("compressed_size_bytes").alias("total_compressed_bytes"),
            avg("compression_ratio").alias("avg_compression_ratio"),
            spark_sum("download_time_seconds").alias("total_download_time"),
            spark_sum("upload_time_seconds").alias("total_upload_time"),
        ).collect()[0]
        
        # Handle null stats safely
        total_original = stats['total_original_bytes'] or 0
        total_compressed = stats['total_compressed_bytes'] or 0
        avg_ratio = stats['avg_compression_ratio'] or 0.0
        
        logger.info("=" * 60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total URLs: {total_count}")
        logger.info(f"Success: {success_count}")
        logger.info(f"Skipped (already in S3): {skipped_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Total original size: {total_original / 1e9:.2f} GB")
        logger.info(f"Total compressed size: {total_compressed / 1e9:.2f} GB")
        logger.info(f"Average compression ratio: {avg_ratio:.2f}x")
        logger.info(f"Elapsed time: {elapsed_time:.1f}s")
        logger.info("=" * 60)
        
        # Save manifest
        logger.info(f"Saving manifest to: {args.manifest_output}")
        manifest_df.coalesce(1).write.mode("overwrite").parquet(args.manifest_output)
        
        # Also save as CSV for easy viewing
        csv_output = args.manifest_output.rstrip("/") + "_csv/"
        manifest_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(csv_output)
        logger.info(f"Saved manifest CSV to: {csv_output}")
        
        # Print failed URLs if any
        if failed_count > 0:
            logger.warning(f"Failed URLs ({failed_count}):")
            failed_df = manifest_df.filter(manifest_df.status == "failed").select("source_url", "error_message")
            for row in failed_df.collect():
                logger.warning(f"  {row.source_url}: {row.error_message}")
        
        logger.info("Sweep download job completed successfully")
        
    except Exception as e:
        logger.error(f"Job failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        try:
            spark.stop()
        except NameError:
            pass


if __name__ == "__main__":
    main()

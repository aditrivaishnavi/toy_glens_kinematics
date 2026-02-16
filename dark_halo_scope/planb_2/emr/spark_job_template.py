#!/usr/bin/env python3
"""
Spark Job Template for Plan B EMR Jobs

This is a template for creating new Spark jobs. Copy and customize.

Usage:
    spark-submit --deploy-mode cluster spark_job_template.py \
        --input s3://bucket/input \
        --output s3://bucket/output

Lessons Learned Incorporated:
- L1.4: Validate data for NaN/Inf
- L6.1: Proper error handling
- L5.2: Comprehensive logging
- L3.3: Don't use .limit() in production
"""
import argparse
import logging
import sys
import time
from datetime import datetime
from typing import Dict, Iterator, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, BinaryType, FloatType


# =============================================================================
# CONFIGURATION
# =============================================================================

# Expected input schema (customize for your job)
INPUT_SCHEMA = StructType([
    StructField("task_id", StringType(), False),
    StructField("stamp_npz", BinaryType(), False),
    StructField("theta_e_arcsec", FloatType(), True),
    StructField("arc_snr", FloatType(), True),
])


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(job_name: str) -> logging.Logger:
    """Set up logging for Spark job."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{job_name}] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(job_name)


# =============================================================================
# SPARK SESSION
# =============================================================================

def create_spark_session(app_name: str) -> SparkSession:
    """Create configured Spark session."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )
    
    # Set log level
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


# =============================================================================
# DATA OPERATIONS
# =============================================================================

def load_input_data(
    spark: SparkSession,
    input_path: str,
    logger: logging.Logger,
) -> DataFrame:
    """
    Load input data from S3.
    
    Returns:
        Spark DataFrame
    """
    logger.info(f"Loading data from: {input_path}")
    
    df = spark.read.parquet(input_path)
    
    # Log schema
    logger.info(f"Schema: {df.schema.simpleString()}")
    
    # Log count (be careful with large datasets)
    count = df.count()
    logger.info(f"Row count: {count:,}")
    
    return df


def validate_data(
    df: DataFrame,
    logger: logging.Logger,
) -> Dict[str, int]:
    """
    Validate data quality.
    
    Lesson L1.4: Never assume clean data.
    
    Returns:
        Dict with validation statistics
    """
    logger.info("Validating data quality...")
    
    stats = {}
    
    # Check for nulls in required columns
    for col_name in df.columns:
        null_count = df.filter(F.col(col_name).isNull()).count()
        stats[f"{col_name}_nulls"] = null_count
        if null_count > 0:
            logger.warning(f"Column {col_name} has {null_count:,} null values")
    
    # Check for duplicate task_ids if present
    if "task_id" in df.columns:
        total = df.count()
        unique = df.select("task_id").distinct().count()
        duplicates = total - unique
        stats["duplicate_task_ids"] = duplicates
        if duplicates > 0:
            logger.warning(f"Found {duplicates:,} duplicate task_ids")
    
    logger.info(f"Validation stats: {stats}")
    
    return stats


def process_partition(iterator: Iterator) -> Iterator:
    """
    Process a partition of data.
    
    This function is called on each worker for each partition.
    Put your main processing logic here.
    
    Yields:
        Processed rows
    """
    # Import any needed modules here (they'll be available on workers)
    import numpy as np
    
    for row in iterator:
        try:
            # Example processing logic - customize this
            result = {
                "task_id": row.task_id,
                "processed": True,
                # Add your output fields here
            }
            yield result
            
        except Exception as e:
            # Log error and skip row (or raise depending on requirements)
            print(f"ERROR processing {row.task_id}: {e}")
            yield {
                "task_id": row.task_id,
                "processed": False,
                "error": str(e),
            }


def run_processing(
    df: DataFrame,
    logger: logging.Logger,
) -> DataFrame:
    """
    Main processing logic.
    
    Customize this function for your specific job.
    """
    logger.info("Starting processing...")
    start_time = time.time()
    
    # Example: use mapPartitions for efficient processing
    # Uncomment and customize:
    # 
    # from pyspark.sql.types import StructType, StructField, StringType, BooleanType
    # 
    # output_schema = StructType([
    #     StructField("task_id", StringType(), False),
    #     StructField("processed", BooleanType(), False),
    # ])
    # 
    # result_rdd = df.rdd.mapPartitions(process_partition)
    # result_df = spark.createDataFrame(result_rdd, schema=output_schema)
    
    # For now, just pass through with a timestamp
    result_df = df.withColumn("processed_at", F.lit(datetime.now().isoformat()))
    
    elapsed = time.time() - start_time
    logger.info(f"Processing completed in {elapsed:.1f}s")
    
    return result_df


def save_output(
    df: DataFrame,
    output_path: str,
    logger: logging.Logger,
    partition_by: Optional[str] = None,
    num_partitions: Optional[int] = None,
) -> None:
    """
    Save output to S3.
    """
    logger.info(f"Saving output to: {output_path}")
    
    writer = df.write.mode("overwrite")
    
    if num_partitions:
        df = df.repartition(num_partitions)
    
    if partition_by:
        writer = writer.partitionBy(partition_by)
    
    writer.parquet(output_path)
    
    logger.info("Output saved successfully")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Spark Job Template")
    parser.add_argument("--input", required=True, help="Input S3 path")
    parser.add_argument("--output", required=True, help="Output S3 path")
    parser.add_argument("--job-name", default="spark-job", help="Job name for logging")
    parser.add_argument("--num-partitions", type=int, help="Output partitions")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't process")
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.job_name)
    logger.info("="*60)
    logger.info(f"Starting job: {args.job_name}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info("="*60)
    
    try:
        # Create Spark session
        spark = create_spark_session(args.job_name)
        
        # Load data
        df = load_input_data(spark, args.input, logger)
        
        # Validate
        stats = validate_data(df, logger)
        
        if args.dry_run:
            logger.info("Dry run complete. Exiting without processing.")
            return
        
        # Process
        result_df = run_processing(df, logger)
        
        # Validate output
        validate_data(result_df, logger)
        
        # Save
        save_output(result_df, args.output, logger, num_partitions=args.num_partitions)
        
        logger.info("="*60)
        logger.info("Job completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Job failed: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

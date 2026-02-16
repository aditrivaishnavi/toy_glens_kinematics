#!/usr/bin/env python

"""
Submit an EMR cluster + PySpark step for Phase 1.5 LRG density estimation.

Usage (from an EC2 host or anywhere with AWS credentials):

  python -m emr.submit_phase1p5_emr_cluster \
      --region us-west-2 \
      --sweep-index-s3 s3://MY_BUCKET/dr10/sweeps_ra150_250_dec0_30_10.1.txt \
      --output-prefix s3://MY_BUCKET/dark_halo_scope/phase1p5 \
      --code-archive-s3 s3://MY_BUCKET/code/dark_halo_scope_code.tgz

The EMR step will:

  spark-submit \
    --deploy-mode cluster \
    --py-files /mnt/dark_halo_scope_code.tgz \
    /mnt/dark_halo_scope_code/emr/spark_phase1p5_lrg_density.py \
    --sweep-index-path s3://.../sweeps_ra150_250_dec0_30_10.1.txt \
    --output-prefix s3://.../phase1p5

You are responsible for ensuring:
  - IAM roles exist (EMR_DefaultRole, EMR_EC2_DefaultRole or your equivalents).
  - The S3 paths and bucket names are correct and accessible.
"""

import argparse
import os
import sys
from typing import Dict, Any, List

import boto3

# Add parent directory to path for imports when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Phase1p5Config


def build_emr_step_args(
    config: Phase1p5Config,
    sweep_index_s3: str,
    output_prefix: str,
    ra_min: float = None,  # None = no filter (full sky)
    ra_max: float = None,  # None = no filter (full sky)
    dec_min: float = None,  # None = no filter (full sky)
    dec_max: float = None,  # None = no filter (full sky)
    max_sweeps: int = 0,
    s3_cache_prefix: str = "",
    chunk_size: int = 100000,
) -> Dict[str, Any]:
    """
    Build the EMR step definition for the Phase 1.5 PySpark job.
    
    IMPORTANT: Uses --deploy-mode client for better stability and easier debugging.
    In client mode, the driver runs on the master node and logs are more accessible.
    
    Memory configuration is tuned for processing SWEEP FITS files:
    - Each SWEEP file is 100-300 MB and expands to ~500MB-1GB in memory
    - Driver needs 4G for coordination
    - Executors need 4G each to handle one file at a time safely
    """
    # Where the code archive will be placed on EMR master node via bootstrap
    local_code_archive = "/mnt/dark_halo_scope_code.tgz"
    local_extracted_dir = "/mnt/dark_halo_scope_code"

    # PySpark driver path inside the extracted tree
    driver_path = os.path.join(local_extracted_dir, config.emr_pyspark_driver_path)

    # Spark step command with explicit memory configuration
    # Using client mode for better log visibility and stability
    spark_args = [
        "spark-submit",
        "--deploy-mode", "client",
        "--master", "yarn",
        # Driver memory (runs on master node in client mode)
        # Note: m5.xlarge has 16GB, we request 2G driver + 512M overhead = 2.5GB
        # This leaves room for OS and other processes
        "--driver-memory", "2g",
        "--driver-cores", "1",
        # Executor memory (runs on core nodes)
        # m5.xlarge has 16GB, we can fit 2-3 executors with 4G each
        "--executor-memory", "4g",
        "--executor-cores", "2",
        # Memory overhead for Python/numpy/astropy (important!)
        "--conf", "spark.executor.memoryOverhead=1g",
        "--conf", "spark.driver.memoryOverhead=512m",
        # Dynamic allocation for efficient resource use
        "--conf", "spark.dynamicAllocation.enabled=true",
        "--conf", "spark.dynamicAllocation.minExecutors=1",
        "--conf", "spark.dynamicAllocation.maxExecutors=10",
        "--conf", "spark.shuffle.service.enabled=true",
        # Better shuffle settings
        "--conf", "spark.sql.shuffle.partitions=100",
        # Python files
        "--py-files", local_code_archive,
        # Driver script
        driver_path,
        # Script arguments
        "--sweep-index-path", sweep_index_s3,
        "--output-prefix", output_prefix,
    ]
    
    # Add RA/Dec filters ONLY if specified (omit for full-sky comprehensive scan)
    if ra_min is not None:
        spark_args.extend(["--ra-min", str(ra_min)])
    if ra_max is not None:
        spark_args.extend(["--ra-max", str(ra_max)])
    if dec_min is not None:
        spark_args.extend(["--dec-min", str(dec_min)])
    if dec_max is not None:
        spark_args.extend(["--dec-max", str(dec_max)])
    
    # Add max-sweeps if specified (for testing)
    if max_sweeps > 0:
        spark_args.extend(["--max-sweeps", str(max_sweeps)])
    
    # Add S3 cache prefix if specified
    if s3_cache_prefix:
        spark_args.extend(["--s3-cache-prefix", s3_cache_prefix])
    
    # Add chunk size
    spark_args.extend(["--chunk-size", str(chunk_size)])

    spark_step = {
        "Name": f"{config.emr_job_name}-pyspark-step",
        "ActionOnFailure": "TERMINATE_CLUSTER",
        "HadoopJarStep": {
            "Jar": "command-runner.jar",
            "Args": spark_args,
        },
    }

    return spark_step


def build_bootstrap_actions(config: Phase1p5Config, code_archive_s3: str, bootstrap_script_s3: str) -> List[Dict[str, Any]]:
    """
    Bootstrap actions to:
      1. Install Python dependencies (astropy, requests, numpy) on ALL nodes.
      2. Download the project archive from S3 and unpack it on ALL nodes.

    We install deps on all nodes because executors also run Python code.
    The code archive is needed on all nodes for --py-files to work properly.
    
    Args:
        config: Phase1p5Config instance
        code_archive_s3: S3 path to the code tarball
        bootstrap_script_s3: S3 path where we'll upload the bootstrap script
    """
    return [
        {
            "Name": "phase1p5-install-deps-and-code",
            "ScriptBootstrapAction": {
                "Path": bootstrap_script_s3,
                "Args": [code_archive_s3],
            },
        }
    ]


def upload_bootstrap_script(s3_bucket: str, s3_prefix: str, code_archive_s3: str) -> str:
    """
    Create and upload a bootstrap script to S3.
    
    Returns the S3 URI of the uploaded script.
    """
    import tempfile
    
    # Bootstrap script content
    # $1 is the code archive S3 path passed as argument
    script_content = """#!/bin/bash
set -euo pipefail

CODE_ARCHIVE_S3="$1"

echo "=== Bootstrap: Installing Python dependencies ==="
sudo python3 -m pip install --upgrade pip --quiet

# IMPORTANT: Pin urllib3<2.0 for compatibility with EMR's old OpenSSL
# EMR 6.x has Python 3.7 with OpenSSL 1.0.2k which doesn't support urllib3 v2.0
sudo python3 -m pip install 'urllib3<2.0' --quiet
sudo python3 -m pip install numpy astropy requests --quiet

echo "=== Bootstrap: Verifying installations ==="
python3 -c "import urllib3; print(f'urllib3 version: {urllib3.__version__}')"
python3 -c "import requests; print(f'requests version: {requests.__version__}')"
python3 -c "import numpy; print(f'numpy version: {numpy.__version__}')"
python3 -c "import astropy; print(f'astropy version: {astropy.__version__}')"

echo "=== Bootstrap: Downloading code archive from $CODE_ARCHIVE_S3 ==="
aws s3 cp "$CODE_ARCHIVE_S3" /mnt/dark_halo_scope_code.tgz

echo "=== Bootstrap: Extracting code archive ==="
mkdir -p /mnt/dark_halo_scope_code
# Use --strip-components=1 to remove the top-level dark_halo_scope/ directory
# This puts emr/, src/, etc. directly under /mnt/dark_halo_scope_code/
tar -xzf /mnt/dark_halo_scope_code.tgz -C /mnt/dark_halo_scope_code --strip-components=1

echo "=== Bootstrap: Complete ==="
ls -la /mnt/dark_halo_scope_code/
echo "=== Verifying driver script exists ==="
ls -la /mnt/dark_halo_scope_code/emr/spark_phase1p5_lrg_density.py
"""
    
    # Write to temp file and upload
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(script_content)
        temp_path = f.name
    
    s3_key = f"{s3_prefix.strip('/')}/bootstrap_phase1p5.sh"
    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    
    s3_client = boto3.client('s3')
    
    print(f"Uploading bootstrap script to {s3_uri}")
    s3_client.upload_file(temp_path, s3_bucket, s3_key)
    
    # Clean up temp file
    os.unlink(temp_path)
    
    return s3_uri


def submit_emr_cluster(
    region: str,
    sweep_index_s3: str,
    output_prefix: str,
    code_archive_s3: str,
    ra_min: float = None,  # None = full sky
    ra_max: float = None,  # None = full sky
    dec_min: float = None,  # None = full sky
    dec_max: float = None,  # None = full sky
    max_sweeps: int = 0,
    instance_type: str = None,
    master_instance_type: str = "m5.xlarge",
    core_count: int = None,
    log_uri: str = None,
    s3_cache_prefix: str = "",
    chunk_size: int = 100000,
    ec2_key_name: str = None,
) -> str:
    """
    Submit the EMR cluster and return the JobFlowId.
    
    IMPORTANT: Default instance type is m5.2xlarge (32 GB RAM) which is 
    much safer for running astropy/numpy workloads than m5.xlarge (16 GB).
    """
    config = Phase1p5Config()

    emr = boto3.client("emr", region_name=region)

    # Instance types: master can be smaller, cores need more RAM for processing
    master_type = master_instance_type or "m5.xlarge"
    core_type = instance_type or "m5.2xlarge"  # 32 GB RAM for FITS processing
    core_instance_count = core_count or config.emr_core_instance_count

    print(f"EMR Configuration:")
    print(f"  Master: 1 x {master_type}")
    print(f"  Core: {core_instance_count} x {core_type}")
    
    # Footprint status
    has_footprint = ra_min is not None or ra_max is not None or dec_min is not None or dec_max is not None
    if has_footprint:
        print(f"  Footprint: RA [{ra_min}, {ra_max}], Dec [{dec_min}, {dec_max}]")
    else:
        print(f"  Footprint: FULL SKY (comprehensive scan - no RA/Dec filter)")
    
    print(f"  Chunk size: {chunk_size:,} rows")
    if max_sweeps > 0:
        print(f"  Max SWEEPs: {max_sweeps} (testing mode)")
    if s3_cache_prefix:
        print(f"  S3 Cache: {s3_cache_prefix}")

    instances = {
        "InstanceGroups": [
            {
                "Name": "Master nodes",
                "Market": "ON_DEMAND",
                "InstanceRole": "MASTER",
                "InstanceType": master_type,
                "InstanceCount": 1,
            },
            {
                "Name": "Core nodes",
                "Market": "ON_DEMAND",
                "InstanceRole": "CORE",
                "InstanceType": core_type,
                "InstanceCount": core_instance_count,
            },
        ],
        "KeepJobFlowAliveWhenNoSteps": False,
        "TerminationProtected": False,
    }
    
    # Add EC2 key pair for SSH access if provided
    if ec2_key_name:
        instances["Ec2KeyName"] = ec2_key_name
        print(f"  SSH Key: {ec2_key_name}")

    # Use provided log_uri, or derive from output_prefix
    if log_uri:
        effective_log_uri = log_uri
    else:
        effective_log_uri = f"{output_prefix.rstrip('/')}/emr-logs/"
    
    print(f"  Logs: {effective_log_uri}")

    step = build_emr_step_args(
        config=config,
        sweep_index_s3=sweep_index_s3,
        output_prefix=output_prefix,
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=dec_min,
        dec_max=dec_max,
        max_sweeps=max_sweeps,
        s3_cache_prefix=s3_cache_prefix,
        chunk_size=chunk_size,
    )

    # Extract bucket and prefix from output_prefix for bootstrap script upload
    # output_prefix is like s3://bucket/path/to/output
    import re
    s3_match = re.match(r's3://([^/]+)/?(.*)', output_prefix)
    if not s3_match:
        raise ValueError(f"Invalid S3 output prefix: {output_prefix}")
    
    s3_bucket = s3_match.group(1)
    s3_prefix = s3_match.group(2) or "phase1p5"
    
    # Upload bootstrap script to S3
    bootstrap_script_s3 = upload_bootstrap_script(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        code_archive_s3=code_archive_s3,
    )

    bootstrap_actions = build_bootstrap_actions(
        config=config,
        code_archive_s3=code_archive_s3,
        bootstrap_script_s3=bootstrap_script_s3,
    )

    response = emr.run_job_flow(
        Name=config.emr_job_name,
        ReleaseLabel=config.emr_release_label,
        Applications=[{"Name": "Spark"}],
        LogUri=effective_log_uri,
        Instances=instances,
        BootstrapActions=bootstrap_actions,
        Steps=[step],
        JobFlowRole=config.emr_job_flow_role,
        ServiceRole=config.emr_service_role,
        VisibleToAllUsers=True,
    )

    job_flow_id = response["JobFlowId"]
    print(f"\n{'='*60}")
    print(f"EMR CLUSTER SUBMITTED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"  JobFlowId: {job_flow_id}")
    print(f"  Region: {region}")
    print(f"  Logs: {effective_log_uri}")
    print(f"  Bootstrap script: {bootstrap_script_s3}")
    print(f"  Output: {output_prefix}/lrg_counts_per_brick/")
    print(f"\nMonitor with:")
    print(f"  aws emr describe-cluster --cluster-id {job_flow_id} --region {region}")
    print(f"{'='*60}")
    return job_flow_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit an EMR cluster for Phase 1.5 PySpark LRG density job.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

  # Full production run for a specific region:
  python -m emr.submit_phase1p5_emr_cluster \\
      --region us-east-2 \\
      --sweep-index-s3 s3://darkhaloscope/dr10/sweep_urls.txt \\
      --output-prefix s3://darkhaloscope/phase1p5 \\
      --code-archive-s3 s3://darkhaloscope/code/dark_halo_scope_code.tgz \\
      --ra-min 150 --ra-max 250 --dec-min 0 --dec-max 30

  # Small test run (only 5 SWEEPs):
  python -m emr.submit_phase1p5_emr_cluster \\
      --region us-east-2 \\
      --sweep-index-s3 s3://darkhaloscope/dr10/sweep_urls.txt \\
      --output-prefix s3://darkhaloscope/phase1p5_test \\
      --code-archive-s3 s3://darkhaloscope/code/dark_halo_scope_code.tgz \\
      --max-sweeps 5

  # Use smaller (cheaper) instances for testing:
  python -m emr.submit_phase1p5_emr_cluster \\
      --region us-east-2 \\
      --sweep-index-s3 s3://darkhaloscope/dr10/sweep_urls.txt \\
      --output-prefix s3://darkhaloscope/phase1p5_test \\
      --code-archive-s3 s3://darkhaloscope/code/dark_halo_scope_code.tgz \\
      --max-sweeps 3 \\
      --instance-type m5.xlarge \\
      --core-count 2
        """
    )

    parser.add_argument(
        "--region",
        required=True,
        help="AWS region for EMR (e.g. us-east-2, us-west-2).",
    )
    parser.add_argument(
        "--sweep-index-s3",
        required=True,
        help=(
            "S3 URI of the SWEEP index file (newline-delimited list of SWEEP paths). "
            "Example: s3://MY_BUCKET/dr10/sweeps_ra150_250_dec0_30_10.1.txt"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help=(
            "S3 prefix for Phase 1.5 outputs. Example: "
            "s3://MY_BUCKET/dark_halo_scope/phase1p5"
        ),
    )
    parser.add_argument(
        "--code-archive-s3",
        required=True,
        help=(
            "S3 URI of the project code archive (e.g. dark_halo_scope_code.tgz) "
            "that contains this repo and the emr/spark_phase1p5_lrg_density.py file."
        ),
    )
    
    # Footprint arguments (OPTIONAL - omit for full-sky comprehensive scan)
    parser.add_argument(
        "--ra-min", type=float, default=None,
        help="Min RA in degrees. Omit for full sky.",
    )
    parser.add_argument(
        "--ra-max", type=float, default=None,
        help="Max RA in degrees. Omit for full sky.",
    )
    parser.add_argument(
        "--dec-min", type=float, default=None,
        help="Min Dec in degrees. Omit for full sky.",
    )
    parser.add_argument(
        "--dec-max", type=float, default=None,
        help="Max Dec in degrees. Omit for full sky.",
    )
    
    # Testing arguments
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=0,
        help="Max SWEEP files to process (0=all). Use for testing.",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default=None,
        help="EC2 instance type for core nodes (default: m5.2xlarge).",
    )
    parser.add_argument(
        "--master-instance-type",
        type=str,
        default="m5.xlarge",
        help="EC2 instance type for master node (default: m5.xlarge). Needs at least 8GB RAM for driver.",
    )
    parser.add_argument(
        "--core-count",
        type=int,
        default=None,
        help="Number of core nodes (default: from config, typically 3).",
    )
    
    parser.add_argument(
        "--log-uri",
        type=str,
        default=None,
        help=(
            "S3 URI for EMR logs. If not specified, uses output-prefix + '/emr-logs/'. "
            "Example: s3://my-bucket/emr-logs/"
        ),
    )
    
    parser.add_argument(
        "--s3-cache-prefix",
        type=str,
        default="",
        help=(
            "S3 prefix to cache downloaded SWEEP files for future runs. "
            "Example: s3://darkhaloscope/sweep-cache/ "
            "This saves HTTP downloads on subsequent runs."
        ),
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Rows per chunk when processing SWEEP files. Default: 100000",
    )
    
    parser.add_argument(
        "--ec2-key-name",
        type=str,
        default=None,
        help="EC2 key pair name for SSH access to cluster nodes. Example: --ec2-key-name root",
    )

    args = parser.parse_args()

    submit_emr_cluster(
        region=args.region,
        sweep_index_s3=args.sweep_index_s3,
        output_prefix=args.output_prefix,
        code_archive_s3=args.code_archive_s3,
        ra_min=args.ra_min,
        ra_max=args.ra_max,
        dec_min=args.dec_min,
        dec_max=args.dec_max,
        max_sweeps=args.max_sweeps,
        instance_type=args.instance_type,
        master_instance_type=args.master_instance_type,
        core_count=args.core_count,
        log_uri=args.log_uri,
        s3_cache_prefix=args.s3_cache_prefix,
        chunk_size=args.chunk_size,
        ec2_key_name=args.ec2_key_name,
    )


if __name__ == "__main__":
    main()


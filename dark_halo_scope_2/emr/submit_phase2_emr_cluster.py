#!/usr/bin/env python3
"""
Submit an EMR cluster + PySpark step for Phase 2 multi-cut LRG density.

This script creates a TRANSIENT EMR cluster that auto-terminates after the
job completes, just like the Phase 1.5 submitter.

Usage (from an EC2 host or anywhere with AWS credentials):

  python -m emr.submit_phase2_emr_cluster \
      --region us-east-2 \
      --sweep-index-s3 s3://MY_BUCKET/dr10/sweep_urls.txt \
      --output-prefix s3://MY_BUCKET/phase2_hypergrid \
      --code-archive-s3 s3://MY_BUCKET/code/dark_halo_scope_code.tgz

The EMR cluster will:
  1. Bootstrap: install numpy, astropy, requests, pandas
  2. Bootstrap: download and extract the code archive
  3. Run spark-submit on spark_phase2_lrg_hypergrid.py
  4. Auto-terminate after job completion

You are responsible for ensuring:
  - IAM roles exist (EMR_DefaultRole, EMR_EC2_DefaultRole or your equivalents).
  - The S3 paths and bucket names are correct and accessible.
"""

import argparse
import os
import re
import sys
import tempfile
from typing import Any, Dict, List

import boto3

# Add parent directory to path for imports when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Phase1p5Config  # Reuse EMR config from Phase1p5


def build_emr_step_args(
    config: Phase1p5Config,
    sweep_index_s3: str,
    output_prefix: str,
    ra_min: float = None,
    ra_max: float = None,
    dec_min: float = None,
    dec_max: float = None,
    max_sweeps: int = 0,
    s3_cache_prefix: str = "",
    chunk_size: int = 100000,
    num_partitions: int = 128,
) -> Dict[str, Any]:
    """
    Build the EMR step definition for the Phase 2 PySpark job.

    Phase 2 needs slightly more memory than Phase 1.5 because it maintains
    multiple selection masks simultaneously:
      - Driver: 3g (vs 2g for Phase 1.5)
      - Executor: 6g (vs 4g for Phase 1.5)
      - Overhead: 2g executor, 1g driver
    """
    # Where the code archive will be placed on EMR nodes via bootstrap
    local_code_archive = "/mnt/dark_halo_scope_code.tgz"
    local_extracted_dir = "/mnt/dark_halo_scope_code"

    # PySpark driver path for Phase 2
    driver_path = os.path.join(local_extracted_dir, "emr/spark_phase2_lrg_hypergrid.py")

    # Spark step command with memory configuration tuned for Phase 2
    spark_args = [
        "spark-submit",
        "--deploy-mode", "client",
        "--master", "yarn",
        # Driver memory (runs on master node in client mode)
        "--driver-memory", "3g",
        "--driver-cores", "1",
        # Executor memory (Phase 2 needs more for multiple masks)
        "--executor-memory", "6g",
        "--executor-cores", "2",
        # Memory overhead for Python/numpy/astropy/pandas
        "--conf", "spark.executor.memoryOverhead=2g",
        "--conf", "spark.driver.memoryOverhead=1g",
        # Dynamic allocation for efficient resource use
        "--conf", "spark.dynamicAllocation.enabled=true",
        "--conf", "spark.dynamicAllocation.minExecutors=1",
        "--conf", "spark.dynamicAllocation.maxExecutors=10",
        "--conf", "spark.shuffle.service.enabled=true",
        # Better shuffle settings
        "--conf", "spark.sql.shuffle.partitions=200",
        # Python files
        "--py-files", local_code_archive,
        # Driver script
        driver_path,
        # Script arguments
        "--sweep-index-path", sweep_index_s3,
        "--output-prefix", output_prefix,
        "--chunk-size", str(chunk_size),
        "--num-partitions", str(num_partitions),
    ]

    # Add RA/Dec filters if specified
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

    spark_step = {
        "Name": "dark-halo-scope-phase2-lrg-hypergrid-step",
        "ActionOnFailure": "TERMINATE_CLUSTER",
        "HadoopJarStep": {
            "Jar": "command-runner.jar",
            "Args": spark_args,
        },
    }

    return spark_step


def build_bootstrap_actions(
    code_archive_s3: str,
    bootstrap_script_s3: str,
) -> List[Dict[str, Any]]:
    """
    Bootstrap actions to:
      1. Install Python dependencies (astropy, requests, numpy, pandas) on ALL nodes.
      2. Download the project archive from S3 and unpack it on ALL nodes.

    We install deps on all nodes because executors also run Python code.
    """
    return [
        {
            "Name": "phase2-install-deps-and-code",
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
    # Bootstrap script content - includes pandas for Phase 2
    script_content = """#!/bin/bash
set -euo pipefail

CODE_ARCHIVE_S3="$1"

echo "=== Bootstrap Phase 2: Installing Python dependencies ==="
sudo python3 -m pip install --upgrade pip --quiet

# IMPORTANT: Pin urllib3<2.0 for compatibility with EMR's old OpenSSL
sudo python3 -m pip install 'urllib3<2.0' --quiet
# boto3 is pre-installed on EMR but we include it for completeness
sudo python3 -m pip install numpy astropy requests boto3 --quiet

echo "=== Bootstrap Phase 2: Verifying installations ==="
python3 -c "import urllib3; print(f'urllib3 version: {urllib3.__version__}')"
python3 -c "import requests; print(f'requests version: {requests.__version__}')"
python3 -c "import numpy; print(f'numpy version: {numpy.__version__}')"
python3 -c "import astropy; print(f'astropy version: {astropy.__version__}')"
python3 -c "import boto3; print(f'boto3 version: {boto3.__version__}')"

echo "=== Bootstrap Phase 2: Downloading code archive from $CODE_ARCHIVE_S3 ==="
aws s3 cp "$CODE_ARCHIVE_S3" /mnt/dark_halo_scope_code.tgz

echo "=== Bootstrap Phase 2: Extracting code archive ==="
mkdir -p /mnt/dark_halo_scope_code
# Use --strip-components=1 to remove the top-level directory
tar -xzf /mnt/dark_halo_scope_code.tgz -C /mnt/dark_halo_scope_code --strip-components=1

echo "=== Bootstrap Phase 2: Complete ==="
ls -la /mnt/dark_halo_scope_code/
echo "=== Verifying Phase 2 driver script exists ==="
ls -la /mnt/dark_halo_scope_code/emr/spark_phase2_lrg_hypergrid.py
"""

    # Write to temp file and upload
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        temp_path = f.name

    s3_key = f"{s3_prefix.strip('/')}/bootstrap_phase2.sh"
    s3_uri = f"s3://{s3_bucket}/{s3_key}"

    s3_client = boto3.client("s3")

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
    ra_min: float = None,
    ra_max: float = None,
    dec_min: float = None,
    dec_max: float = None,
    max_sweeps: int = 0,
    instance_type: str = None,
    master_instance_type: str = "m5.xlarge",
    core_count: int = None,
    log_uri: str = None,
    s3_cache_prefix: str = "",
    chunk_size: int = 100000,
    num_partitions: int = 256,
    ec2_key_name: str = None,
) -> str:
    """
    Submit the EMR cluster and return the JobFlowId.

    Default configuration optimized for 256 vCPU account limit:
      - Master: 1 x m5.xlarge (4 vCPUs)
      - Core: 30 x m5.2xlarge (8 vCPUs each = 240 vCPUs)
      - Total: 244 vCPUs (within 256 limit)

    Phase 2 defaults to m5.2xlarge for core nodes (32 GB RAM) which provides
    comfortable headroom for the multi-cut LRG processing.
    """
    config = Phase1p5Config()

    emr = boto3.client("emr", region_name=region)

    # Instance types optimized for 256 vCPU limit
    # m5.xlarge = 4 vCPU, m5.2xlarge = 8 vCPU
    # 1 master (4 vCPU) + 30 core (240 vCPU) = 244 vCPU total
    master_type = master_instance_type or "m5.xlarge"
    core_type = instance_type or "m5.2xlarge"  # 32 GB RAM, 8 vCPU
    core_instance_count = core_count if core_count is not None else 30  # Default 30 for 256 vCPU limit

    # Calculate vCPU usage for display
    vcpu_map = {"m5.xlarge": 4, "m5.2xlarge": 8, "m5.4xlarge": 16, "r5.xlarge": 4, "r5.2xlarge": 8}
    master_vcpu = vcpu_map.get(master_type, 4)
    core_vcpu = vcpu_map.get(core_type, 8)
    total_vcpu = master_vcpu + (core_instance_count * core_vcpu)

    print(f"Phase 2 EMR Configuration:")
    print(f"  Master: 1 x {master_type} ({master_vcpu} vCPU)")
    print(f"  Core: {core_instance_count} x {core_type} ({core_instance_count * core_vcpu} vCPU)")
    print(f"  Total vCPUs: {total_vcpu} (limit: 256)")

    # Footprint status
    has_footprint = (
        ra_min is not None
        or ra_max is not None
        or dec_min is not None
        or dec_max is not None
    )
    if has_footprint:
        print(f"  Footprint: RA [{ra_min}, {ra_max}], Dec [{dec_min}, {dec_max}]")
    else:
        print(f"  Footprint: FULL SKY (no RA/Dec filter)")

    print(f"  Chunk size: {chunk_size:,} rows")
    print(f"  Num partitions: {num_partitions}")
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
        num_partitions=num_partitions,
    )

    # Extract bucket and prefix from output_prefix for bootstrap script upload
    s3_match = re.match(r"s3://([^/]+)/?(.*)", output_prefix)
    if not s3_match:
        raise ValueError(f"Invalid S3 output prefix: {output_prefix}")

    s3_bucket = s3_match.group(1)
    s3_prefix = s3_match.group(2) or "phase2"

    # Upload bootstrap script to S3
    bootstrap_script_s3 = upload_bootstrap_script(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        code_archive_s3=code_archive_s3,
    )

    bootstrap_actions = build_bootstrap_actions(
        code_archive_s3=code_archive_s3,
        bootstrap_script_s3=bootstrap_script_s3,
    )

    response = emr.run_job_flow(
        Name="dark-halo-scope-phase2-lrg-hypergrid",
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
    print(f"PHASE 2 EMR CLUSTER SUBMITTED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"  JobFlowId: {job_flow_id}")
    print(f"  Region: {region}")
    print(f"  Logs: {effective_log_uri}")
    print(f"  Bootstrap script: {bootstrap_script_s3}")
    print(f"  Output: {output_prefix}/phase2_lrg_hypergrid.csv/")
    print(f"\nLRG selections computed (in one pass):")
    print(f"  - v1_pure_massive:   z<20.0, r-z>0.5, z-W1>1.6")
    print(f"  - v2_baseline_dr10:  z<20.4, r-z>0.4, z-W1>1.6")
    print(f"  - v3_color_relaxed:  z<20.4, r-z>0.4, z-W1>0.8")
    print(f"  - v4_mag_relaxed:    z<21.0, r-z>0.4, z-W1>0.8")
    print(f"  - v5_very_relaxed:   z<21.5, r-z>0.3, z-W1>0.8")
    print(f"\nMonitor with:")
    print(f"  aws emr describe-cluster --cluster-id {job_flow_id} --region {region}")
    print(f"{'='*60}")
    return job_flow_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit an EMR cluster for Phase 2 multi-cut LRG density job.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

  # Full production run for a specific region:
  python -m emr.submit_phase2_emr_cluster \\
      --region us-east-2 \\
      --sweep-index-s3 s3://darkhaloscope/dr10/sweep_urls.txt \\
      --output-prefix s3://darkhaloscope/phase2_hypergrid \\
      --code-archive-s3 s3://darkhaloscope/code/dark_halo_scope_code.tgz \\
      --ra-min 150 --ra-max 250 --dec-min 0 --dec-max 30

  # Small test run (only 5 SWEEPs):
  python -m emr.submit_phase2_emr_cluster \\
      --region us-east-2 \\
      --sweep-index-s3 s3://darkhaloscope/dr10/sweep_urls.txt \\
      --output-prefix s3://darkhaloscope/phase2_test \\
      --code-archive-s3 s3://darkhaloscope/code/dark_halo_scope_code.tgz \\
      --max-sweeps 5

  # Use smaller (cheaper) instances for testing:
  python -m emr.submit_phase2_emr_cluster \\
      --region us-east-2 \\
      --sweep-index-s3 s3://darkhaloscope/dr10/sweep_urls.txt \\
      --output-prefix s3://darkhaloscope/phase2_test \\
      --code-archive-s3 s3://darkhaloscope/code/dark_halo_scope_code.tgz \\
      --max-sweeps 3 \\
      --instance-type m5.xlarge \\
      --core-count 2
        """,
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
            "Example: s3://MY_BUCKET/dr10/sweep_urls.txt"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help=(
            "S3 prefix for Phase 2 outputs. Example: "
            "s3://MY_BUCKET/dark_halo_scope/phase2_hypergrid"
        ),
    )
    parser.add_argument(
        "--code-archive-s3",
        required=True,
        help=(
            "S3 URI of the project code archive (e.g. dark_halo_scope_code.tgz) "
            "that contains this repo and the emr/spark_phase2_lrg_hypergrid.py file."
        ),
    )

    # Footprint arguments (OPTIONAL)
    parser.add_argument(
        "--ra-min",
        type=float,
        default=None,
        help="Min RA in degrees. Omit for full sky.",
    )
    parser.add_argument(
        "--ra-max",
        type=float,
        default=None,
        help="Max RA in degrees. Omit for full sky.",
    )
    parser.add_argument(
        "--dec-min",
        type=float,
        default=None,
        help="Min Dec in degrees. Omit for full sky.",
    )
    parser.add_argument(
        "--dec-max",
        type=float,
        default=None,
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
        help="EC2 instance type for master node (default: m5.xlarge).",
    )
    parser.add_argument(
        "--core-count",
        type=int,
        default=None,
        help="Number of core nodes (default: 30 for 256 vCPU limit with m5.2xlarge).",
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
            "Example: s3://darkhaloscope/sweep-cache/"
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Rows per chunk when processing SWEEP files. Default: 100000",
    )

    parser.add_argument(
        "--num-partitions",
        type=int,
        default=256,
        help="Number of Spark partitions (parallel tasks). Default: 256",
    )

    parser.add_argument(
        "--ec2-key-name",
        type=str,
        default=None,
        help="EC2 key pair name for SSH access to cluster nodes.",
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
        num_partitions=args.num_partitions,
        ec2_key_name=args.ec2_key_name,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Launch EMR job to download DR10 sweep files from NERSC to S3.

This is a one-time job to populate S3 with compressed sweep files.

Usage:
    # Test with 5 files
    python emr/launch_sweep_download.py --test
    
    # Full download (1436 files)
    python emr/launch_sweep_download.py --full --preset large
    
    # Resume/retry failed downloads
    python emr/launch_sweep_download.py --full --preset large

The job is resumable - it skips files already in S3.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
S3_BUCKET = os.environ.get("S3_BUCKET", "darkhaloscope")

# S3 Paths
S3_CODE_PREFIX = "stronglens_calibration/emr/code"
S3_LOGS_PREFIX = "stronglens_calibration/emr/logs"

# Input/Output paths
SWEEP_URLS_PATH = f"s3://{S3_BUCKET}/dr10/sweep_urls.txt"
SWEEP_OUTPUT_PREFIX = f"s3://{S3_BUCKET}/dr10/sweeps/"
MANIFEST_OUTPUT_PREFIX = f"s3://{S3_BUCKET}/dr10/sweeps_manifest/"

# EMR Configuration
EMR_RELEASE_LABEL = "emr-7.0.0"
EMR_SUBNET_ID = os.environ.get("EMR_SUBNET_ID", "")

# Instance presets - larger instances for download-heavy workload
INSTANCE_PRESETS = {
    "test": {
        "name": "test",
        "master_type": "m5.xlarge",
        "worker_type": "m5.xlarge",
        "worker_count": 2,
        "executor_memory": "4g",
        "timeout_hours": 1,
    },
    "medium": {
        "name": "medium",
        "master_type": "m5.xlarge",
        "worker_type": "m5.2xlarge",
        "worker_count": 20,
        "executor_memory": "8g",
        "timeout_hours": 8,
    },
    "large": {
        "name": "large",
        "master_type": "m5.xlarge",
        "worker_type": "m5.2xlarge",
        "worker_count": 30,
        "executor_memory": "8g",
        "timeout_hours": 12,
    },
}

# Project paths
EMR_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = EMR_DIR.parent


# =============================================================================
# HELPERS
# =============================================================================

def check_aws_credentials() -> bool:
    """Verify AWS credentials are configured."""
    try:
        import boto3
        sts = boto3.client("sts", region_name=AWS_REGION)
        identity = sts.get_caller_identity()
        print(f"AWS Account: {identity['Account']}")
        return True
    except Exception as e:
        print(f"ERROR: AWS credentials not configured: {e}")
        return False


def upload_to_s3(local_path: Path, s3_key: str) -> str:
    """Upload file to S3 and return S3 URI."""
    import boto3
    
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
    print(f"  Uploaded: {local_path.name} -> {s3_uri}")
    return s3_uri


def prepare_and_upload_code() -> Dict[str, str]:
    """Upload job code to S3."""
    print("\n[1/4] Uploading code to S3...")
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    uploads = {}
    
    # Upload main script
    script_path = EMR_DIR / "spark_download_sweeps.py"
    s3_key = f"{S3_CODE_PREFIX}/{timestamp}/spark_download_sweeps.py"
    uploads["script"] = upload_to_s3(script_path, s3_key)
    
    # Upload bootstrap script
    bootstrap_path = EMR_DIR / "bootstrap.sh"
    s3_key = f"{S3_CODE_PREFIX}/{timestamp}/bootstrap.sh"
    if bootstrap_path.exists():
        uploads["bootstrap"] = upload_to_s3(bootstrap_path, s3_key)
    
    print(f"  Uploaded {len(uploads)} files")
    return uploads


def launch_emr_cluster(preset: str, uploads: Dict[str, str]) -> str:
    """Launch EMR cluster and return cluster ID."""
    import boto3
    
    config = INSTANCE_PRESETS[preset]
    print(f"\n[2/4] Launching EMR cluster (preset: {preset})...")
    print(f"  Master:  1x {config['master_type']}")
    print(f"  Workers: {config['worker_count']}x {config['worker_type']}")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    
    cluster_config = {
        "Name": f"sweep-download-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        "ReleaseLabel": EMR_RELEASE_LABEL,
        "LogUri": f"s3://{S3_BUCKET}/{S3_LOGS_PREFIX}/",
        "Applications": [
            {"Name": "Spark"},
            {"Name": "Hadoop"},
        ],
        "Instances": {
            "InstanceGroups": [
                {
                    "Name": "Master",
                    "Market": "ON_DEMAND",
                    "InstanceRole": "MASTER",
                    "InstanceType": config["master_type"],
                    "InstanceCount": 1,
                },
                {
                    "Name": "Core",
                    "Market": "ON_DEMAND",
                    "InstanceRole": "CORE",
                    "InstanceType": config["worker_type"],
                    "InstanceCount": config["worker_count"],
                },
            ],
            "KeepJobFlowAliveWhenNoSteps": True,
            "TerminationProtected": False,
        },
        "JobFlowRole": "EMR_EC2_DefaultRole",
        "ServiceRole": "EMR_DefaultRole",
        "Tags": [
            {"Key": "Project", "Value": "stronglens_calibration"},
            {"Key": "Job", "Value": "sweep_download"},
        ],
    }
    
    # Add subnet if configured
    if EMR_SUBNET_ID:
        cluster_config["Instances"]["Ec2SubnetId"] = EMR_SUBNET_ID
    
    # Add bootstrap action
    if uploads.get("bootstrap"):
        cluster_config["BootstrapActions"] = [
            {
                "Name": "Install dependencies",
                "ScriptBootstrapAction": {
                    "Path": uploads["bootstrap"],
                },
            },
        ]
    
    response = emr.run_job_flow(**cluster_config)
    cluster_id = response["JobFlowId"]
    
    print(f"  Cluster ID: {cluster_id}")
    return cluster_id


def wait_for_cluster(cluster_id: str, timeout_minutes: int = 15) -> bool:
    """Wait for cluster to be ready."""
    import boto3
    
    print(f"\n[3/4] Waiting for cluster to be ready...")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    start_time = time.time()
    
    while True:
        response = emr.describe_cluster(ClusterId=cluster_id)
        state = response["Cluster"]["Status"]["State"]
        
        print(f"  Cluster state: {state}")
        
        if state == "WAITING":
            print("  Cluster is ready!")
            return True
        
        if state in ["TERMINATED", "TERMINATED_WITH_ERRORS"]:
            print(f"  ERROR: Cluster terminated")
            return False
        
        if time.time() - start_time > timeout_minutes * 60:
            print(f"  ERROR: Timeout waiting for cluster")
            return False
        
        time.sleep(30)


def submit_spark_step(
    cluster_id: str,
    uploads: Dict[str, str],
    test_limit: Optional[int] = None,
) -> str:
    """Submit Spark step and return step ID."""
    import boto3
    
    print(f"\n[4/4] Submitting Spark step...")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    manifest_output = f"{MANIFEST_OUTPUT_PREFIX}{timestamp}/"
    
    step_args = [
        "spark-submit",
        "--deploy-mode", "cluster",
        "--conf", "spark.executor.memory=6g",
        "--conf", "spark.driver.memory=4g",
        "--conf", "spark.executor.cores=1",  # 1 task per executor to avoid OOM (each file is ~2GB)
        "--conf", "spark.task.cpus=1",
        "--conf", "spark.sql.shuffle.partitions=200",
        "--conf", "spark.network.timeout=600s",  # Longer timeout for large file transfers
        "--conf", "spark.executor.heartbeatInterval=60s",
        uploads["script"],
        "--sweep-urls", SWEEP_URLS_PATH,
        "--output-prefix", SWEEP_OUTPUT_PREFIX,
        "--manifest-output", manifest_output,
        "--job-name", f"download-{timestamp}",
    ]
    
    if test_limit:
        step_args.extend(["--test-limit", str(test_limit)])
    
    response = emr.add_job_flow_steps(
        JobFlowId=cluster_id,
        Steps=[
            {
                "Name": "SweepDownload",
                "ActionOnFailure": "CONTINUE",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": step_args,
                },
            },
        ],
    )
    
    step_id = response["StepIds"][0]
    print(f"  Step ID: {step_id}")
    print(f"  Manifest output: {manifest_output}")
    
    return step_id


def monitor_step(cluster_id: str, step_id: str, timeout_hours: float) -> bool:
    """Monitor step execution."""
    import boto3
    
    print(f"\nMonitoring step execution (timeout: {timeout_hours}h)...")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    start_time = time.time()
    timeout_seconds = timeout_hours * 3600
    
    while True:
        response = emr.describe_step(ClusterId=cluster_id, StepId=step_id)
        state = response["Step"]["Status"]["State"]
        
        elapsed = time.time() - start_time
        print(f"  [{elapsed/60:.0f}m] Step state: {state}")
        
        if state == "COMPLETED":
            print("\n  Step completed successfully!")
            return True
        
        if state in ["FAILED", "CANCELLED"]:
            print(f"\n  Step {state}")
            if "FailureDetails" in response["Step"]["Status"]:
                details = response["Step"]["Status"]["FailureDetails"]
                print(f"  Reason: {details.get('Reason', 'Unknown')}")
                print(f"  Message: {details.get('Message', 'Unknown')}")
            return False
        
        if elapsed > timeout_seconds:
            print(f"\n  Timeout after {timeout_hours} hours")
            return False
        
        time.sleep(60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Launch EMR sweep download job")
    
    parser.add_argument("--test", action="store_true",
                        help="Run test with 5 files")
    parser.add_argument("--full", action="store_true",
                        help="Run full download (1436 files)")
    parser.add_argument("--status", action="store_true",
                        help="Check cluster/step status")
    parser.add_argument("--terminate", action="store_true",
                        help="Terminate cluster")
    
    parser.add_argument("--cluster-id", type=str,
                        help="Existing cluster ID")
    parser.add_argument("--step-id", type=str,
                        help="Step ID to monitor")
    parser.add_argument("--preset", type=str, default="medium",
                        choices=list(INSTANCE_PRESETS.keys()),
                        help="Instance preset")
    parser.add_argument("--test-limit", type=int, default=None,
                        help="Limit number of files to download")
    
    args = parser.parse_args()
    
    # Check credentials
    if not check_aws_credentials():
        sys.exit(1)
    
    # Handle status check
    if args.status:
        if not args.cluster_id:
            print("ERROR: --cluster-id required")
            sys.exit(1)
        
        import boto3
        emr = boto3.client("emr", region_name=AWS_REGION)
        
        response = emr.describe_cluster(ClusterId=args.cluster_id)
        print(f"\nCluster Status: {response['Cluster']['Status']['State']}")
        
        if args.step_id:
            response = emr.describe_step(ClusterId=args.cluster_id, StepId=args.step_id)
            print(f"Step Status: {response['Step']['Status']['State']}")
        
        sys.exit(0)
    
    # Handle terminate
    if args.terminate:
        if not args.cluster_id:
            print("ERROR: --cluster-id required")
            sys.exit(1)
        
        import boto3
        emr = boto3.client("emr", region_name=AWS_REGION)
        emr.terminate_job_flows(JobFlowIds=[args.cluster_id])
        print(f"Terminating cluster: {args.cluster_id}")
        sys.exit(0)
    
    # Run job
    if not args.test and not args.full:
        print("ERROR: Specify --test or --full")
        sys.exit(1)
    
    # Upload code
    uploads = prepare_and_upload_code()
    
    # Launch or use existing cluster
    if args.cluster_id:
        cluster_id = args.cluster_id
        print(f"Using existing cluster: {cluster_id}")
    else:
        preset = "test" if args.test else args.preset
        cluster_id = launch_emr_cluster(preset, uploads)
        
        if not wait_for_cluster(cluster_id):
            print("ERROR: Cluster failed to start")
            sys.exit(1)
    
    # Determine test limit
    if args.test:
        test_limit = args.test_limit or 5
    else:
        test_limit = args.test_limit
    
    # Submit step
    step_id = submit_spark_step(
        cluster_id=cluster_id,
        uploads=uploads,
        test_limit=test_limit,
    )
    
    # Monitor
    config = INSTANCE_PRESETS[args.preset if not args.test else "test"]
    success = monitor_step(cluster_id, step_id, config["timeout_hours"])
    
    if success:
        print(f"\nOutput location: {SWEEP_OUTPUT_PREFIX}")
        print(f"\nTo verify:")
        print(f"  aws s3 ls {SWEEP_OUTPUT_PREFIX} | wc -l")
        print(f"  aws s3 ls {MANIFEST_OUTPUT_PREFIX} --recursive")
    
    print(f"\nCluster ID: {cluster_id}")
    print(f"To terminate: python emr/launch_sweep_download.py --terminate --cluster-id {cluster_id}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

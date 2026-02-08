#!/usr/bin/env python3
"""
EMR Launch Script for Positive Catalog Crossmatch Job

This script launches the crossmatch Spark job on EMR to match the 5,104
positive lens candidates with the DR10 manifest to obtain observing conditions.

Prerequisites:
1. AWS CLI configured with valid credentials
2. boto3 installed: pip install boto3
3. S3 bucket access configured

Usage:
    # Mini test (sample 1% of manifest)
    python emr/launch_crossmatch_positives.py --test
    
    # Full run
    python emr/launch_crossmatch_positives.py --full
    
    # Check status
    python emr/launch_crossmatch_positives.py --status --cluster-id j-XXXXX

Author: Generated for stronglens_calibration project
Date: 2026-02-08
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

# AWS Configuration
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
S3_BUCKET = os.environ.get("S3_BUCKET", "darkhaloscope")

# S3 Paths
S3_CODE_PREFIX = "stronglens_calibration/emr/code"
S3_OUTPUT_PREFIX = "stronglens_calibration/positives_with_dr10"
S3_LOGS_PREFIX = "stronglens_calibration/emr/logs"

# Default paths
DEFAULT_MANIFEST_PATH = "s3://darkhaloscope/stronglens_calibration/manifests/20260208_074343/"
DEFAULT_POSITIVES_PATH = "s3://darkhaloscope/stronglens_calibration/configs/positives/desi_candidates.csv"

# EMR Configuration
EMR_RELEASE_LABEL = "emr-7.0.0"
EMR_SUBNET_ID = os.environ.get("EMR_SUBNET_ID", "")

# Instance Configuration - smaller since this is a simple join
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
        "worker_count": 5,
        "executor_memory": "8g",
        "timeout_hours": 2,
    },
    "large": {
        "name": "large",
        "master_type": "m5.xlarge",
        "worker_type": "m5.2xlarge",
        "worker_count": 10,
        "executor_memory": "8g",
        "timeout_hours": 4,
    },
}

# Project root
EMR_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = EMR_DIR.parent


# =============================================================================
# HELPERS
# =============================================================================

def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except:
        return "unknown"


def check_aws_credentials() -> bool:
    """Check if AWS credentials are valid."""
    try:
        import boto3
        sts = boto3.client("sts", region_name=AWS_REGION)
        identity = sts.get_caller_identity()
        print(f"✓ AWS credentials valid")
        print(f"  Account: {identity['Account']}")
        print(f"  Region:  {AWS_REGION}")
        return True
    except ImportError:
        print("ERROR: boto3 not installed. Run: pip install boto3")
        return False
    except Exception as e:
        print(f"ERROR: AWS credentials invalid: {e}")
        return False


def upload_to_s3(local_path: Path, s3_key: str) -> str:
    """Upload file to S3 and return S3 URI."""
    import boto3
    
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
    
    print(f"  Uploading: {local_path.name} -> {s3_uri}")
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    
    # Verify upload
    s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
    
    return s3_uri


def verify_inputs_exist() -> bool:
    """Verify that required input files exist in S3."""
    import boto3
    
    s3 = boto3.client("s3", region_name=AWS_REGION)
    
    # Check manifest
    manifest_bucket = DEFAULT_MANIFEST_PATH.replace("s3://", "").split("/")[0]
    manifest_key = "/".join(DEFAULT_MANIFEST_PATH.replace("s3://", "").split("/")[1:])
    
    try:
        # List to check if directory has content
        response = s3.list_objects_v2(
            Bucket=manifest_bucket,
            Prefix=manifest_key,
            MaxKeys=1
        )
        if response.get("KeyCount", 0) == 0:
            print(f"ERROR: Manifest not found at {DEFAULT_MANIFEST_PATH}")
            return False
        print(f"✓ Manifest found at {DEFAULT_MANIFEST_PATH}")
    except Exception as e:
        print(f"ERROR checking manifest: {e}")
        return False
    
    # Check positives
    positives_bucket = DEFAULT_POSITIVES_PATH.replace("s3://", "").split("/")[0]
    positives_key = "/".join(DEFAULT_POSITIVES_PATH.replace("s3://", "").split("/")[1:])
    
    try:
        s3.head_object(Bucket=positives_bucket, Key=positives_key)
        print(f"✓ Positives found at {DEFAULT_POSITIVES_PATH}")
    except Exception as e:
        print(f"ERROR: Positives not found at {DEFAULT_POSITIVES_PATH}: {e}")
        return False
    
    return True


# =============================================================================
# MAIN OPERATIONS
# =============================================================================

def prepare_and_upload_code() -> Dict[str, str]:
    """Upload code to S3."""
    print("\n[1/4] Preparing and uploading code...")
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    git_commit = get_git_commit()
    
    uploads = {}
    
    # Upload bootstrap script
    bootstrap_path = EMR_DIR / "bootstrap.sh"
    if bootstrap_path.exists():
        s3_key = f"{S3_CODE_PREFIX}/{timestamp}_{git_commit}/bootstrap.sh"
        uploads["bootstrap"] = upload_to_s3(bootstrap_path, s3_key)
    
    # Upload main script
    script_path = EMR_DIR / "spark_crossmatch_positives.py"
    s3_key = f"{S3_CODE_PREFIX}/{timestamp}_{git_commit}/spark_crossmatch_positives.py"
    uploads["script"] = upload_to_s3(script_path, s3_key)
    
    # Upload utilities
    utils_path = EMR_DIR / "sampling_utils.py"
    s3_key = f"{S3_CODE_PREFIX}/{timestamp}_{git_commit}/sampling_utils.py"
    uploads["utils"] = upload_to_s3(utils_path, s3_key)
    
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
        "Name": f"stronglens-crossmatch-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
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
        "VisibleToAllUsers": True,
        "JobFlowRole": "EMR_EC2_DefaultRole",
        "ServiceRole": "EMR_DefaultRole",
        "Configurations": [
            {
                "Classification": "spark-defaults",
                "Properties": {
                    "spark.executor.memory": config["executor_memory"],
                    "spark.executor.cores": "4",
                    "spark.dynamicAllocation.enabled": "true",
                    "spark.sql.adaptive.enabled": "true",
                },
            },
            {
                "Classification": "spark-env",
                "Configurations": [
                    {
                        "Classification": "export",
                        "Properties": {
                            "NUMBA_CACHE_DIR": "/tmp/numba_cache",
                            "PYSPARK_PYTHON": "/usr/bin/python3",
                            "PYSPARK_DRIVER_PYTHON": "/usr/bin/python3",
                        },
                    },
                ],
            },
        ],
    }
    
    if "bootstrap" in uploads:
        cluster_config["BootstrapActions"] = [
            {
                "Name": "InstallDependencies",
                "ScriptBootstrapAction": {
                    "Path": uploads["bootstrap"],
                },
            },
        ]
    
    if EMR_SUBNET_ID:
        cluster_config["Instances"]["Ec2SubnetId"] = EMR_SUBNET_ID
    
    response = emr.run_job_flow(**cluster_config)
    cluster_id = response["JobFlowId"]
    
    print(f"  Cluster ID: {cluster_id}")
    
    return cluster_id


def wait_for_cluster(cluster_id: str, timeout_minutes: int = 20) -> bool:
    """Wait for cluster to be ready."""
    import boto3
    
    print(f"\n[3/4] Waiting for cluster to be ready...")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    while True:
        response = emr.describe_cluster(ClusterId=cluster_id)
        status = response["Cluster"]["Status"]["State"]
        
        print(f"  Status: {status}")
        
        if status == "WAITING":
            print("  Cluster ready!")
            return True
        elif status in ("TERMINATED", "TERMINATED_WITH_ERRORS"):
            print(f"  ERROR: Cluster failed with status: {status}")
            return False
        
        if time.time() - start_time > timeout_seconds:
            print(f"  ERROR: Timeout waiting for cluster")
            return False
        
        time.sleep(30)


def submit_spark_step(
    cluster_id: str,
    uploads: Dict[str, str],
    manifest_path: str,
    positives_path: str,
    output_path: str,
    sample_frac: Optional[float] = None,
) -> str:
    """Submit Spark step and return step ID."""
    import boto3
    
    print(f"\n[4/4] Submitting Spark step...")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    
    step_args = [
        "spark-submit",
        "--deploy-mode", "cluster",
        "--py-files", uploads["utils"],
        uploads["script"],
        "--manifest", manifest_path,
        "--positives", positives_path,
        "--output", output_path,
    ]
    
    if sample_frac:
        step_args.extend(["--sample-frac", str(sample_frac)])
    
    response = emr.add_job_flow_steps(
        JobFlowId=cluster_id,
        Steps=[
            {
                "Name": "CrossmatchPositives",
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
    
    return step_id


def monitor_step(cluster_id: str, step_id: str, timeout_hours: int = 2) -> bool:
    """Monitor step until completion."""
    import boto3
    
    print("\nMonitoring step progress...")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    start_time = time.time()
    timeout_seconds = timeout_hours * 3600
    
    while True:
        response = emr.describe_step(ClusterId=cluster_id, StepId=step_id)
        status = response["Step"]["Status"]["State"]
        
        elapsed = (time.time() - start_time) / 60
        print(f"  [{elapsed:.0f}m] Status: {status}")
        
        if status == "COMPLETED":
            print("\n  Step completed successfully!")
            return True
        elif status in ("FAILED", "CANCELLED"):
            print(f"\n  Step failed with status: {status}")
            if "FailureDetails" in response["Step"]["Status"]:
                details = response["Step"]["Status"]["FailureDetails"]
                print(f"  Reason: {details.get('Reason', 'Unknown')}")
                print(f"  Message: {details.get('Message', 'Unknown')}")
            return False
        
        if time.time() - start_time > timeout_seconds:
            print(f"\n  Timeout after {timeout_hours} hours")
            return False
        
        time.sleep(30)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Launch EMR crossmatch job")
    
    parser.add_argument("--test", action="store_true",
                        help="Run test with 1%% manifest sample")
    parser.add_argument("--full", action="store_true",
                        help="Run full job")
    parser.add_argument("--status", action="store_true",
                        help="Check cluster/step status")
    parser.add_argument("--terminate", action="store_true",
                        help="Terminate cluster")
    
    parser.add_argument("--cluster-id", type=str,
                        help="Existing cluster ID")
    parser.add_argument("--step-id", type=str,
                        help="Step ID to monitor")
    parser.add_argument("--preset", type=str, default="test",
                        choices=list(INSTANCE_PRESETS.keys()),
                        help="Instance preset")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST_PATH,
                        help="S3 path to manifest")
    parser.add_argument("--positives", type=str, default=DEFAULT_POSITIVES_PATH,
                        help="S3 path to positive catalog")
    
    args = parser.parse_args()
    
    # Check credentials first
    if not check_aws_credentials():
        sys.exit(1)
    
    # Handle status check
    if args.status:
        if not args.cluster_id:
            print("ERROR: --cluster-id required for status check")
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
            print("ERROR: --cluster-id required for terminate")
            sys.exit(1)
        
        import boto3
        emr = boto3.client("emr", region_name=AWS_REGION)
        emr.terminate_job_flows(JobFlowIds=[args.cluster_id])
        print(f"Terminating cluster: {args.cluster_id}")
        sys.exit(0)
    
    # Verify inputs exist
    if not verify_inputs_exist():
        sys.exit(1)
    
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
    
    # Submit step
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = f"s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}/{timestamp}/"
    
    # For test, sample 1% of manifest
    sample_frac = 0.01 if args.test else None
    
    step_id = submit_spark_step(
        cluster_id=cluster_id,
        uploads=uploads,
        manifest_path=args.manifest,
        positives_path=args.positives,
        output_path=output_path,
        sample_frac=sample_frac,
    )
    
    # Monitor
    config = INSTANCE_PRESETS[args.preset if not args.test else "test"]
    success = monitor_step(cluster_id, step_id, config["timeout_hours"])
    
    if success:
        print(f"\nOutput: {output_path}")
        print("\nTo verify output:")
        print(f"  aws s3 ls {output_path}")
    
    print(f"\nCluster ID: {cluster_id}")
    print("To terminate: python emr/launch_crossmatch_positives.py --terminate --cluster-id " + cluster_id)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

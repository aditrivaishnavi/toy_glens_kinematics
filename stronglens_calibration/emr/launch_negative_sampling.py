#!/usr/bin/env python3
"""
EMR Launch Script for Negative Sampling Job

This script launches the negative sampling Spark job on EMR.

Prerequisites:
1. AWS CLI configured with valid credentials
2. boto3 installed: pip install boto3
3. S3 bucket access configured

Usage:
    # Small test run (2 partitions)
    python emr/launch_negative_sampling.py --test
    
    # Full run
    python emr/launch_negative_sampling.py --full
    
    # Check status
    python emr/launch_negative_sampling.py --status --cluster-id j-XXXXX

Lessons Applied:
- L6.1: Local testing before EMR
- L6.2: Verify S3 uploads
- L5.1: Don't declare victory prematurely
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
S3_CONFIG_PREFIX = "stronglens_calibration/configs"
S3_OUTPUT_PREFIX = "stronglens_calibration/manifests"
S3_LOGS_PREFIX = "stronglens_calibration/emr/logs"
S3_CHECKPOINT_PREFIX = "stronglens_calibration/checkpoints"

# EMR Configuration
EMR_RELEASE_LABEL = "emr-7.0.0"
EMR_SUBNET_ID = os.environ.get("EMR_SUBNET_ID", "")  # Must be set

# Instance Configuration
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
        "worker_count": 10,
        "executor_memory": "8g",
        "timeout_hours": 4,
    },
    "large": {
        "name": "large",
        "master_type": "m5.xlarge", 
        "worker_type": "m5.2xlarge",
        "worker_count": 30,
        "executor_memory": "8g",
        "timeout_hours": 8,
    },
}

# Project root - the directory containing this script
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
        print("\nTo fix:")
        print("  1. Run: aws configure")
        print("  2. Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
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


def verify_s3_upload(s3_uri: str) -> bool:
    """Verify file exists in S3."""
    import boto3
    
    s3 = boto3.client("s3", region_name=AWS_REGION)
    
    # Parse S3 URI
    if s3_uri.startswith("s3://"):
        s3_uri = s3_uri[5:]
    bucket, key = s3_uri.split("/", 1)
    
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False


# =============================================================================
# MAIN OPERATIONS
# =============================================================================

def prepare_and_upload_code() -> Dict[str, str]:
    """
    Upload code, config, and bootstrap script to S3.
    
    Returns dict of S3 URIs.
    """
    print("\n[1/4] Preparing and uploading code...")
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    git_commit = get_git_commit()
    
    uploads = {}
    
    # Upload bootstrap script (needed for cluster creation)
    bootstrap_path = EMR_DIR / "bootstrap.sh"
    s3_key = f"{S3_CODE_PREFIX}/{timestamp}_{git_commit}/bootstrap.sh"
    uploads["bootstrap"] = upload_to_s3(bootstrap_path, s3_key)
    
    # Upload main script
    script_path = EMR_DIR / "spark_negative_sampling.py"
    s3_key = f"{S3_CODE_PREFIX}/{timestamp}_{git_commit}/spark_negative_sampling.py"
    uploads["script"] = upload_to_s3(script_path, s3_key)
    
    # Upload utilities
    utils_path = EMR_DIR / "sampling_utils.py"
    s3_key = f"{S3_CODE_PREFIX}/{timestamp}_{git_commit}/sampling_utils.py"
    uploads["utils"] = upload_to_s3(utils_path, s3_key)
    
    # Upload sweep utilities
    sweep_utils_path = EMR_DIR / "sweep_utils.py"
    if sweep_utils_path.exists():
        s3_key = f"{S3_CODE_PREFIX}/{timestamp}_{git_commit}/sweep_utils.py"
        uploads["sweep_utils"] = upload_to_s3(sweep_utils_path, s3_key)
    
    # Upload config - check both locations
    config_path = EMR_DIR / "configs" / "negative_sampling_v1.yaml"
    if not config_path.exists():
        config_path = PROJECT_ROOT / "configs" / "negative_sampling_v1.yaml"
    s3_key = f"{S3_CONFIG_PREFIX}/{timestamp}/negative_sampling_v1.yaml"
    if config_path.exists():
        uploads["config"] = upload_to_s3(config_path, s3_key)
    else:
        print(f"  WARNING: Config not found at {config_path}")
    
    # Upload positive catalog for exclusion - optional
    positive_path = EMR_DIR / "data" / "positives" / "desi_candidates.csv"
    if not positive_path.exists():
        positive_path = PROJECT_ROOT / "data" / "positives" / "desi_candidates.csv"
    if positive_path.exists():
        s3_key = f"{S3_CONFIG_PREFIX}/positives/desi_candidates.csv"
        uploads["positives"] = upload_to_s3(positive_path, s3_key)
    else:
        print(f"  NOTE: Positive catalog not found (optional)")
    
    print(f"  Uploaded {len(uploads)} files")
    
    return uploads


def launch_emr_cluster(preset: str, uploads: Dict[str, str]) -> str:
    """
    Launch EMR cluster with bootstrap action and return cluster ID.
    
    Args:
        preset: Instance preset name (test, medium, large)
        uploads: Dict of S3 URIs including bootstrap script
    """
    import boto3
    
    config = INSTANCE_PRESETS[preset]
    print(f"\n[2/4] Launching EMR cluster (preset: {preset})...")
    print(f"  Master:  1x {config['master_type']}")
    print(f"  Workers: {config['worker_count']}x {config['worker_type']}")
    print(f"  Bootstrap: {uploads.get('bootstrap', 'NOT PROVIDED')}")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    
    # Build cluster configuration
    cluster_config = {
        "Name": f"stronglens-negative-sampling-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
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
            "KeepJobFlowAliveWhenNoSteps": True,  # Keep alive for step submission (terminate manually)
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
    
    # Add bootstrap action to install dependencies
    if "bootstrap" in uploads:
        cluster_config["BootstrapActions"] = [
            {
                "Name": "InstallDependencies",
                "ScriptBootstrapAction": {
                    "Path": uploads["bootstrap"],
                },
            },
        ]
        print(f"  Bootstrap action configured")
    else:
        print("  WARNING: No bootstrap script - dependencies may not be installed!")
    
    # Add subnet if specified
    if EMR_SUBNET_ID:
        cluster_config["Instances"]["Ec2SubnetId"] = EMR_SUBNET_ID
    
    response = emr.run_job_flow(**cluster_config)
    cluster_id = response["JobFlowId"]
    
    print(f"  Cluster ID: {cluster_id}")
    
    return cluster_id


def wait_for_cluster(cluster_id: str, timeout_minutes: int = 25) -> bool:
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
    sweep_input: str,
    output_path: str,
    test_limit: Optional[int] = None,
) -> str:
    """Submit Spark step and return step ID."""
    import boto3
    
    print(f"\n[4/4] Submitting Spark step...")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    
    # Collect all Python dependency files
    py_files = [uploads["utils"]]
    if "sweep_utils" in uploads:
        py_files.append(uploads["sweep_utils"])
    py_files_str = ",".join(py_files)
    
    # Build step arguments
    step_args = [
        "spark-submit",
        "--deploy-mode", "cluster",
        "--py-files", py_files_str,
        uploads["script"],
        "--config", uploads["config"],
        "--sweep-input", sweep_input,
        "--output", output_path,
    ]
    
    # Only add --positive-catalog if we have a catalog
    if uploads.get("positives"):
        step_args.extend(["--positive-catalog", uploads["positives"]])
    
    if test_limit:
        step_args.extend(["--test-limit", str(test_limit)])
    
    response = emr.add_job_flow_steps(
        JobFlowId=cluster_id,
        Steps=[
            {
                "Name": "NegativeSampling",
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


def monitor_step(cluster_id: str, step_id: str, timeout_hours: int = 4) -> bool:
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
            # Print failure reason
            if "FailureDetails" in response["Step"]["Status"]:
                details = response["Step"]["Status"]["FailureDetails"]
                print(f"  Reason: {details.get('Reason', 'Unknown')}")
                print(f"  Message: {details.get('Message', 'Unknown')}")
            return False
        
        if time.time() - start_time > timeout_seconds:
            print(f"\n  Timeout after {timeout_hours} hours")
            return False
        
        time.sleep(60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Launch EMR negative sampling job")
    
    parser.add_argument("--test", action="store_true",
                        help="Run small test (2 partitions)")
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
    parser.add_argument("--sweep-input", type=str,
                        default="s3://darkhaloscope/dr10/sweeps/",
                        help="S3 path to sweep files")
    
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
    
    test_limit = 2 if args.test else None
    
    step_id = submit_spark_step(
        cluster_id=cluster_id,
        uploads=uploads,
        sweep_input=args.sweep_input,
        output_path=output_path,
        test_limit=test_limit,
    )
    
    # Monitor
    config = INSTANCE_PRESETS[args.preset if not args.test else "test"]
    success = monitor_step(cluster_id, step_id, config["timeout_hours"])
    
    if success:
        print(f"\nOutput: {output_path}")
        print("\nTo verify output:")
        print(f"  aws s3 ls {output_path}")
    
    # Auto-terminate cluster after step completion
    print(f"\nTerminating cluster: {cluster_id}")
    import boto3
    emr = boto3.client("emr", region_name=AWS_REGION)
    emr.terminate_job_flows(JobFlowIds=[cluster_id])
    print("Cluster termination initiated.")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

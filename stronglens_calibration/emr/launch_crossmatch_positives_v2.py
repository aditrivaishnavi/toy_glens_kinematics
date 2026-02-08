#!/usr/bin/env python3
"""
EMR Launcher: Crossmatch Positives with DR10 Sweep Files

This script launches an EMR cluster to run the crossmatch job.
Memory-intensive sweep file processing is distributed across workers.

Usage:
    python emr/launch_crossmatch_positives_v2.py --preset test
    python emr/launch_crossmatch_positives_v2.py --preset medium
    python emr/launch_crossmatch_positives_v2.py --preset full

Author: Generated for stronglens_calibration project
Date: 2026-02-08
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import boto3

# =============================================================================
# CONFIGURATION
# =============================================================================

AWS_REGION = "us-east-2"
S3_BUCKET = "darkhaloscope"
S3_CODE_PREFIX = "stronglens_calibration/emr/code"
S3_OUTPUT_PREFIX = "stronglens_calibration/positives_with_dr10"
S3_LOGS_PREFIX = "stronglens_calibration/emr/logs"

EMR_RELEASE = "emr-7.0.0"

# Instance presets
PRESETS = {
    "test": {
        "master_type": "m5.xlarge",
        "worker_type": "m5.xlarge",
        "worker_count": 2,
        "executor_memory": "4g",
        "description": "Test run with 2 workers",
    },
    "medium": {
        "master_type": "m5.xlarge",
        "worker_type": "m5.2xlarge",
        "worker_count": 5,
        "executor_memory": "12g",
        "description": "Medium run with 5 workers (32 GB each)",
    },
    "full": {
        "master_type": "m5.xlarge",
        "worker_type": "m5.2xlarge",
        "worker_count": 30,
        "executor_memory": "12g",
        "description": "Full run with 30 workers",
    },
}

# Paths
SWEEPS_PATH = f"s3://{S3_BUCKET}/dr10/sweeps/"
POSITIVES_PATH = f"s3://{S3_BUCKET}/stronglens_calibration/configs/positives/desi_candidates.csv"

EMR_DIR = Path(__file__).parent.resolve()


# =============================================================================
# HELPERS
# =============================================================================

def upload_to_s3(local_path: Path, s3_key: str) -> str:
    """Upload file to S3 and return S3 URI."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
    
    print(f"  Uploading: {local_path.name} -> {s3_uri}")
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    
    # Verify upload
    s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
    
    return s3_uri


def prepare_and_upload_code() -> Dict[str, str]:
    """Upload code and bootstrap to S3."""
    print("\n[1/4] Preparing and uploading code...")
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    uploads = {}
    
    # Upload bootstrap script
    bootstrap_path = EMR_DIR / "bootstrap.sh"
    s3_key = f"{S3_CODE_PREFIX}/{timestamp}/bootstrap.sh"
    uploads["bootstrap"] = upload_to_s3(bootstrap_path, s3_key)
    
    # Upload main script
    script_path = EMR_DIR / "spark_crossmatch_positives_v2.py"
    s3_key = f"{S3_CODE_PREFIX}/{timestamp}/spark_crossmatch_positives_v2.py"
    uploads["script"] = upload_to_s3(script_path, s3_key)
    
    # Upload utilities
    utils_path = EMR_DIR / "sampling_utils.py"
    s3_key = f"{S3_CODE_PREFIX}/{timestamp}/sampling_utils.py"
    uploads["utils"] = upload_to_s3(utils_path, s3_key)
    
    uploads["timestamp"] = timestamp
    print(f"  Uploaded {len(uploads) - 1} files")
    
    return uploads


def launch_emr_cluster(preset: str, uploads: Dict[str, str]) -> str:
    """Launch EMR cluster with bootstrap action."""
    config = PRESETS[preset]
    timestamp = uploads["timestamp"]
    
    print(f"\n[2/4] Launching EMR cluster (preset: {preset})...")
    print(f"  Master:  1x {config['master_type']}")
    print(f"  Workers: {config['worker_count']}x {config['worker_type']}")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    
    cluster_config = {
        "Name": f"crossmatch-positives-{timestamp}",
        "ReleaseLabel": EMR_RELEASE,
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
                            "PYSPARK_PYTHON": "/usr/bin/python3",
                            "PYSPARK_DRIVER_PYTHON": "/usr/bin/python3",
                        },
                    },
                ],
            },
        ],
        "BootstrapActions": [
            {
                "Name": "InstallDependencies",
                "ScriptBootstrapAction": {
                    "Path": uploads["bootstrap"],
                },
            },
        ],
    }
    
    response = emr.run_job_flow(**cluster_config)
    cluster_id = response["JobFlowId"]
    
    print(f"  Cluster ID: {cluster_id}")
    
    return cluster_id


def wait_for_cluster(cluster_id: str, timeout_minutes: int = 20) -> bool:
    """Wait for cluster to be ready."""
    print(f"\n[3/4] Waiting for cluster to be ready...")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    while True:
        response = emr.describe_cluster(ClusterId=cluster_id)
        status = response["Cluster"]["Status"]["State"]
        
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Status: {status}")
        
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


def submit_spark_step(cluster_id: str, uploads: Dict[str, str]) -> str:
    """Submit Spark step and return step ID."""
    timestamp = uploads["timestamp"]
    output_path = f"s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}/{timestamp}/"
    
    print(f"\n[4/4] Submitting Spark step...")
    print(f"  Output: {output_path}")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    
    step_args = [
        "spark-submit",
        "--deploy-mode", "cluster",
        "--driver-memory", "4g",
        "--executor-memory", "12g",
        "--executor-cores", "4",
        "--conf", "spark.dynamicAllocation.enabled=true",
        "--py-files", uploads["utils"],
        uploads["script"],
        "--sweeps", SWEEPS_PATH,
        "--positives", POSITIVES_PATH,
        "--output", output_path,
    ]
    
    response = emr.add_job_flow_steps(
        JobFlowId=cluster_id,
        Steps=[
            {
                "Name": "CrossmatchPositives",
                "ActionOnFailure": "TERMINATE_CLUSTER",
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
    print("\nMonitoring step progress...")
    print(f"EMR Console: https://us-east-2.console.aws.amazon.com/emr/home?region=us-east-2#/clusterDetails/{cluster_id}")
    
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
        
        time.sleep(60)


def terminate_cluster(cluster_id: str):
    """Terminate cluster."""
    print(f"\nTerminating cluster {cluster_id}...")
    emr = boto3.client("emr", region_name=AWS_REGION)
    emr.terminate_job_flows(JobFlowIds=[cluster_id])
    print("  Termination initiated")


def main():
    parser = argparse.ArgumentParser(description="Launch EMR crossmatch job")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="test",
                       help="Instance preset")
    parser.add_argument("--no-monitor", action="store_true",
                       help="Don't monitor after launch")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EMR Crossmatch Positives Launcher")
    print("=" * 60)
    print(f"Preset: {args.preset} - {PRESETS[args.preset]['description']}")
    
    # Check credentials
    try:
        sts = boto3.client("sts", region_name=AWS_REGION)
        identity = sts.get_caller_identity()
        print(f"AWS Account: {identity['Account']}")
    except Exception as e:
        print(f"ERROR: AWS credentials invalid: {e}")
        sys.exit(1)
    
    # Upload code
    uploads = prepare_and_upload_code()
    
    # Launch cluster
    cluster_id = launch_emr_cluster(args.preset, uploads)
    
    # Wait for cluster
    if not wait_for_cluster(cluster_id):
        print("Cluster failed to start")
        sys.exit(1)
    
    # Submit step
    step_id = submit_spark_step(cluster_id, uploads)
    
    print(f"\n" + "=" * 60)
    print(f"Cluster ID: {cluster_id}")
    print(f"Step ID:    {step_id}")
    print(f"Output:     s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}/{uploads['timestamp']}/")
    print("=" * 60)
    
    if not args.no_monitor:
        success = monitor_step(cluster_id, step_id)
        
        # Terminate cluster when done
        terminate_cluster(cluster_id)
        
        sys.exit(0 if success else 1)
    else:
        print("\nNot monitoring (--no-monitor).")
        print(f"Check EMR console: https://us-east-2.console.aws.amazon.com/emr/home?region=us-east-2")
        print(f"To terminate: aws emr terminate-clusters --cluster-ids {cluster_id} --region us-east-2")


if __name__ == "__main__":
    main()

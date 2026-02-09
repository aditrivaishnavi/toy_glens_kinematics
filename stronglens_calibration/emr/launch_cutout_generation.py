#!/usr/bin/env python3
"""
EMR Launcher: Generate Cutouts from Legacy Survey

Launches EMR cluster to download cutouts for positives or negatives.

Usage:
    python emr/launch_cutout_generation.py --type positive --preset medium
    python emr/launch_cutout_generation.py --type negative --preset full

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

# Import central constants
sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import (
    AWS_REGION,
    S3_BUCKET,
    S3_CODE_PREFIX,
    S3_CUTOUTS_PREFIX,
    S3_LOGS_PREFIX,
    EMR_RELEASE,
    EMR_PRESETS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Use presets from constants with added descriptions
PRESETS = {
    name: {**cfg, "description": cfg.get("description", f"{name} preset")}
    for name, cfg in EMR_PRESETS.items()
    if name in ("test", "medium", "full")
}

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
    s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
    
    return s3_uri


def prepare_and_upload_code() -> Dict[str, str]:
    """Upload code to S3."""
    print("\n[1/4] Preparing and uploading code...")
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    uploads = {}
    
    # Upload bootstrap
    bootstrap_path = EMR_DIR / "bootstrap.sh"
    s3_key = f"{S3_CODE_PREFIX}/{timestamp}/bootstrap.sh"
    uploads["bootstrap"] = upload_to_s3(bootstrap_path, s3_key)
    
    # Upload main script
    script_path = EMR_DIR / "spark_generate_cutouts.py"
    s3_key = f"{S3_CODE_PREFIX}/{timestamp}/spark_generate_cutouts.py"
    uploads["script"] = upload_to_s3(script_path, s3_key)
    
    uploads["timestamp"] = timestamp
    print(f"  Uploaded {len(uploads) - 1} files")
    
    return uploads


def launch_emr_cluster(preset: str, uploads: Dict[str, str]) -> str:
    """Launch EMR cluster."""
    config = PRESETS[preset]
    timestamp = uploads["timestamp"]
    
    print(f"\n[2/4] Launching EMR cluster (preset: {preset})...")
    print(f"  Master:  1x {config['master_type']}")
    print(f"  Workers: {config['worker_count']}x {config['worker_type']}")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    
    cluster_config = {
        "Name": f"cutout-generation-{timestamp}",
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
                    "spark.executor.cores": "2",
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


def submit_spark_step(
    cluster_id: str,
    uploads: Dict[str, str],
    cutout_type: str,
    input_path: str,
) -> str:
    """Submit Spark step."""
    timestamp = uploads["timestamp"]
    output_path = f"s3://{S3_BUCKET}/{S3_CUTOUTS_PREFIX}/{cutout_type}s/{timestamp}/"
    
    print(f"\n[4/4] Submitting Spark step...")
    print(f"  Type: {cutout_type}")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    
    step_args = [
        "spark-submit",
        "--deploy-mode", "cluster",
        "--driver-memory", "4g",
        "--executor-memory", "8g",
        "--executor-cores", "2",
        "--conf", "spark.dynamicAllocation.enabled=true",
        uploads["script"],
        "--input", input_path,
        "--output", output_path,
        "--cutout-type", cutout_type,
    ]
    
    response = emr.add_job_flow_steps(
        JobFlowId=cluster_id,
        Steps=[
            {
                "Name": f"GenerateCutouts-{cutout_type}",
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


def monitor_step(cluster_id: str, step_id: str, timeout_hours: int = 8) -> bool:
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
    parser = argparse.ArgumentParser(description="Launch EMR cutout generation job")
    parser.add_argument("--type", required=True, choices=["positive", "negative"],
                       help="Cutout type")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="medium",
                       help="Instance preset")
    parser.add_argument("--input", help="Override input path")
    parser.add_argument("--no-monitor", action="store_true",
                       help="Don't monitor after launch")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EMR Cutout Generation Launcher")
    print("=" * 60)
    print(f"Type: {args.type}")
    print(f"Preset: {args.preset} - {PRESETS[args.preset]['description']}")
    
    # Determine input path (required)
    if not args.input:
        print("ERROR: --input is required (path to parquet with ra/dec)")
        sys.exit(1)
    input_path = args.input
    
    print(f"Input: {input_path}")
    
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
    step_id = submit_spark_step(cluster_id, uploads, args.type, input_path)
    
    print(f"\n" + "=" * 60)
    print(f"Cluster ID: {cluster_id}")
    print(f"Step ID:    {step_id}")
    print(f"Output:     s3://{S3_BUCKET}/{S3_CUTOUTS_PREFIX}/{args.type}s/{uploads['timestamp']}/")
    print("=" * 60)
    
    if not args.no_monitor:
        success = monitor_step(cluster_id, step_id)
        terminate_cluster(cluster_id)
        sys.exit(0 if success else 1)
    else:
        print("\nNot monitoring (--no-monitor).")
        print(f"To terminate: aws emr terminate-clusters --cluster-ids {cluster_id} --region us-east-2")


if __name__ == "__main__":
    main()

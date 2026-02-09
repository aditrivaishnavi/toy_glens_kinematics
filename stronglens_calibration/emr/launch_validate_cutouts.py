#!/usr/bin/env python3
"""
EMR Launcher: Validate Cutouts and Detect Shortcuts

Runs quality validation and shortcut detection on generated cutouts.

Usage:
    python emr/launch_validate_cutouts.py \
        --positives s3://darkhaloscope/stronglens_calibration/cutouts/positives/TIMESTAMP/ \
        --negatives s3://darkhaloscope/stronglens_calibration/cutouts/negatives/TIMESTAMP/ \
        --preset medium

Author: Generated for stronglens_calibration project
Date: 2026-02-08
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import boto3

# Import central constants
sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import (
    AWS_REGION,
    S3_BUCKET,
    S3_CODE_PREFIX,
    S3_LOGS_PREFIX,
    S3_VALIDATION_PREFIX,
    EMR_RELEASE,
    EMR_PRESETS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Use presets from constants
PRESETS = {
    name: {**cfg, "description": cfg.get("description", f"{name} preset")}
    for name, cfg in EMR_PRESETS.items()
    if name in ("test", "medium", "large")
}

EMR_DIR = Path(__file__).parent.resolve()


# =============================================================================
# HELPERS
# =============================================================================

def upload_to_s3(local_path: Path, s3_key: str) -> str:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
    print(f"  Uploading: {local_path.name} -> {s3_uri}")
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    return s3_uri


def prepare_and_upload_code():
    print("\n[1/4] Preparing and uploading code...")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    uploads = {}
    
    # Bootstrap
    bootstrap_path = EMR_DIR / "bootstrap.sh"
    s3_key = f"{S3_CODE_PREFIX}/{timestamp}/bootstrap.sh"
    uploads["bootstrap"] = upload_to_s3(bootstrap_path, s3_key)
    
    # Main script
    script_path = EMR_DIR / "spark_validate_cutouts.py"
    s3_key = f"{S3_CODE_PREFIX}/{timestamp}/spark_validate_cutouts.py"
    uploads["script"] = upload_to_s3(script_path, s3_key)
    
    uploads["timestamp"] = timestamp
    print(f"  Uploaded {len(uploads) - 1} files")
    return uploads


def launch_emr_cluster(preset: str, uploads):
    config = PRESETS[preset]
    timestamp = uploads["timestamp"]
    
    print(f"\n[2/4] Launching EMR cluster (preset: {preset})...")
    print(f"  Workers: {config['worker_count']}x {config['worker_type']}")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    
    response = emr.run_job_flow(
        Name=f"validate-cutouts-{timestamp}",
        ReleaseLabel=EMR_RELEASE,
        LogUri=f"s3://{S3_BUCKET}/{S3_LOGS_PREFIX}/",
        Applications=[{"Name": "Spark"}, {"Name": "Hadoop"}],
        Instances={
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
        VisibleToAllUsers=True,
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
        Configurations=[
            {
                "Classification": "spark-defaults",
                "Properties": {
                    "spark.executor.memory": config["executor_memory"],
                    "spark.dynamicAllocation.enabled": "true",
                },
            },
            {
                "Classification": "spark-env",
                "Configurations": [
                    {"Classification": "export", "Properties": {"PYSPARK_PYTHON": "/usr/bin/python3"}},
                ],
            },
        ],
        BootstrapActions=[
            {"Name": "InstallDependencies", "ScriptBootstrapAction": {"Path": uploads["bootstrap"]}},
        ],
    )
    
    cluster_id = response["JobFlowId"]
    print(f"  Cluster ID: {cluster_id}")
    return cluster_id


def wait_for_cluster(cluster_id: str, timeout_minutes: int = 15):
    print("\n[3/4] Waiting for cluster to be ready...")
    emr = boto3.client("emr", region_name=AWS_REGION)
    start = time.time()
    
    while time.time() - start < timeout_minutes * 60:
        response = emr.describe_cluster(ClusterId=cluster_id)
        status = response["Cluster"]["Status"]["State"]
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {status}")
        
        if status == "WAITING":
            return True
        elif status in ("TERMINATED", "TERMINATED_WITH_ERRORS"):
            return False
        time.sleep(30)
    return False


def submit_spark_step(cluster_id: str, uploads, positives: str, negatives: str):
    timestamp = uploads["timestamp"]
    output_path = f"s3://{S3_BUCKET}/{S3_VALIDATION_PREFIX}/{timestamp}/"
    
    print(f"\n[4/4] Submitting Spark step...")
    print(f"  Output: {output_path}")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    
    response = emr.add_job_flow_steps(
        JobFlowId=cluster_id,
        Steps=[
            {
                "Name": "ValidateCutouts",
                "ActionOnFailure": "CONTINUE",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": [
                        "spark-submit",
                        "--deploy-mode", "cluster",
                        "--executor-memory", "8g",
                        uploads["script"],
                        "--positives", positives,
                        "--negatives", negatives,
                        "--output", output_path,
                    ],
                },
            },
        ],
    )
    
    step_id = response["StepIds"][0]
    print(f"  Step ID: {step_id}")
    return step_id, output_path


def monitor_step(cluster_id: str, step_id: str, timeout_minutes: int = 60):
    print("\nMonitoring...")
    print(f"Console: https://us-east-2.console.aws.amazon.com/emr/home?region=us-east-2#/clusterDetails/{cluster_id}")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    start = time.time()
    
    while time.time() - start < timeout_minutes * 60:
        response = emr.describe_step(ClusterId=cluster_id, StepId=step_id)
        status = response["Step"]["Status"]["State"]
        elapsed = (time.time() - start) / 60
        print(f"  [{elapsed:.0f}m] {status}")
        
        if status == "COMPLETED":
            return True
        elif status in ("FAILED", "CANCELLED"):
            return False
        time.sleep(60)
    return False


def terminate_cluster(cluster_id: str):
    print(f"\nTerminating {cluster_id}...")
    emr = boto3.client("emr", region_name=AWS_REGION)
    emr.terminate_job_flows(JobFlowIds=[cluster_id])


def main():
    parser = argparse.ArgumentParser(description="Launch cutout validation EMR job")
    parser.add_argument("--positives", required=True, help="S3 path to positive cutouts")
    parser.add_argument("--negatives", required=True, help="S3 path to negative cutouts")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="medium")
    parser.add_argument("--no-monitor", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EMR Cutout Validation Launcher")
    print("=" * 60)
    
    # Check AWS
    try:
        sts = boto3.client("sts", region_name=AWS_REGION)
        print(f"AWS Account: {sts.get_caller_identity()['Account']}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    uploads = prepare_and_upload_code()
    cluster_id = launch_emr_cluster(args.preset, uploads)
    
    if not wait_for_cluster(cluster_id):
        print("Cluster failed")
        sys.exit(1)
    
    step_id, output_path = submit_spark_step(cluster_id, uploads, args.positives, args.negatives)
    
    print(f"\n" + "=" * 60)
    print(f"Cluster: {cluster_id}")
    print(f"Output:  {output_path}")
    print("=" * 60)
    
    if not args.no_monitor:
        success = monitor_step(cluster_id, step_id)
        terminate_cluster(cluster_id)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

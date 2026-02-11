#!/usr/bin/env python3
"""
EMR Launcher: Stratified Sampling for Negative Pool

Samples ~510K negatives maintaining 100:1 per-stratum ratio with 85:15 N1:N2.

Usage:
    python emr/launch_stratified_sample.py --preset medium --negatives s3://...
    python emr/launch_stratified_sample.py --preset test --negatives s3://...

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
    S3_SAMPLED_NEGATIVES_PREFIX,
    EMR_RELEASE,
    EMR_PRESETS,
    get_emr_console_url,
    get_emr_terminate_cmd,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Use presets from constants
PRESETS = {
    name: {**cfg, "description": cfg.get("description", f"{name} preset")}
    for name, cfg in EMR_PRESETS.items()
    if name in ("test", "medium")
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
    
    return s3_uri


def prepare_and_upload_code():
    """Upload code to S3."""
    print("\n[1/4] Preparing and uploading code...")
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    uploads = {}
    
    # Bootstrap
    bootstrap_path = EMR_DIR / "bootstrap.sh"
    s3_key = f"{S3_CODE_PREFIX}/{timestamp}/bootstrap.sh"
    uploads["bootstrap"] = upload_to_s3(bootstrap_path, s3_key)
    
    # Main script
    script_path = EMR_DIR / "spark_stratified_sample.py"
    s3_key = f"{S3_CODE_PREFIX}/{timestamp}/spark_stratified_sample.py"
    uploads["script"] = upload_to_s3(script_path, s3_key)
    
    uploads["timestamp"] = timestamp
    print(f"  Uploaded {len(uploads) - 1} files")
    
    return uploads


def launch_emr_cluster_with_step(preset: str, uploads, negatives_path: str, positives_path: str):
    """Launch EMR cluster with step included (for auto-termination)."""
    config = PRESETS[preset]
    timestamp = uploads["timestamp"]
    output_path = f"s3://{S3_BUCKET}/{S3_SAMPLED_NEGATIVES_PREFIX}/{timestamp}/"
    
    print(f"\n[2/3] Launching EMR cluster with step (preset: {preset})...")
    print(f"  Master:  1x {config['master_type']}")
    print(f"  Workers: {config['worker_count']}x {config['worker_type']}")
    print(f"  Output: {output_path}")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    
    # Build step args
    step_args = [
        "spark-submit",
        "--deploy-mode", "cluster",
        "--driver-memory", "4g",
        "--executor-memory", "8g",
        "--executor-cores", "2",
        "--conf", "spark.dynamicAllocation.enabled=true",
        "--conf", "spark.sql.shuffle.partitions=200",
        uploads["script"],
        "--negatives", negatives_path,
        "--positives", positives_path,
        "--output", output_path,
    ]
    
    cluster_config = {
        "Name": f"stratified-sample-{timestamp}",
        "ReleaseLabel": EMR_RELEASE,
        "LogUri": f"s3://{S3_BUCKET}/{S3_LOGS_PREFIX}/",
        "Applications": [
            {"Name": "Spark"},
            {"Name": "Hadoop"},
        ],
        "Steps": [
            {
                "Name": "StratifiedSample",
                "ActionOnFailure": "TERMINATE_CLUSTER",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": step_args,
                },
            },
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
            "KeepJobFlowAliveWhenNoSteps": False,  # Auto-terminate when step completes
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
                    "spark.sql.shuffle.partitions": "200",
                },
            },
            {
                "Classification": "spark-env",
                "Configurations": [
                    {
                        "Classification": "export",
                        "Properties": {
                            "PYSPARK_PYTHON": "/usr/bin/python3",
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
    print(f"  Output: {output_path}")
    
    return cluster_id, output_path


def monitor_step(cluster_id: str, step_id: str, timeout_minutes: int = 60):
    """Monitor step until completion."""
    print("\nMonitoring step progress...")
    print(f"EMR Console: {get_emr_console_url(cluster_id)}")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    start_time = time.time()
    
    while True:
        response = emr.describe_step(ClusterId=cluster_id, StepId=step_id)
        status = response["Step"]["Status"]["State"]
        
        elapsed = (time.time() - start_time) / 60
        print(f"  [{elapsed:.0f}m] Status: {status}")
        
        if status == "COMPLETED":
            print("\n  Step completed!")
            return True
        elif status in ("FAILED", "CANCELLED"):
            print(f"\n  Step failed: {status}")
            return False
        
        if time.time() - start_time > timeout_minutes * 60:
            print("\n  Timeout!")
            return False
        
        time.sleep(60)


def terminate_cluster(cluster_id: str):
    """Terminate cluster."""
    print(f"\nTerminating cluster {cluster_id}...")
    emr = boto3.client("emr", region_name=AWS_REGION)
    emr.terminate_job_flows(JobFlowIds=[cluster_id])
    print("  Termination initiated")


def main():
    parser = argparse.ArgumentParser(description="Launch stratified sampling EMR job")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="medium")
    parser.add_argument("--negatives", required=True, help="S3 path to negative manifest")
    parser.add_argument("--positives", required=True, help="S3 path to positives with DR10")
    parser.add_argument("--no-monitor", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EMR Stratified Sampling Launcher")
    print("=" * 60)
    print(f"Preset: {args.preset} - {PRESETS[args.preset]['description']}")
    print(f"Negatives: {args.negatives}")
    print(f"Positives: {args.positives}")
    
    # Check AWS
    try:
        sts = boto3.client("sts", region_name=AWS_REGION)
        identity = sts.get_caller_identity()
        print(f"AWS Account: {identity['Account']}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Upload
    uploads = prepare_and_upload_code()
    
    # Launch cluster with step (auto-terminates when done)
    cluster_id, output_path = launch_emr_cluster_with_step(
        args.preset, uploads, args.negatives, args.positives
    )
    
    print(f"\n" + "=" * 60)
    print(f"Cluster ID: {cluster_id}")
    print(f"Output:     {output_path}")
    print("=" * 60)
    print(f"\nCluster will auto-terminate after step completes.")
    print(f"EMR Console: {get_emr_console_url(cluster_id)}")
    print(f"\nMonitor with:")
    print(f"  aws emr describe-cluster --cluster-id {cluster_id} --region {AWS_REGION}")
    print(f"  aws emr list-steps --cluster-id {cluster_id} --region {AWS_REGION}")


if __name__ == "__main__":
    main()

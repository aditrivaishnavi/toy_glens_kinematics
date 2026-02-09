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
    if name in ("test", "small", "medium", "medium-large", "large-xlarge", "large")
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


def launch_emr_with_step(preset: str, uploads, positives: str, negatives: str,
                         sample: int = 0, bootstrap: int = 200):
    """
    Launch EMR cluster with step embedded in run_job_flow.
    
    This is more robust than add_job_flow_steps - the step runs even if
    the launcher disconnects.
    """
    config = PRESETS[preset]
    timestamp = uploads["timestamp"]
    output_path = f"s3://{S3_BUCKET}/{S3_VALIDATION_PREFIX}/{timestamp}/"
    
    print(f"\n[2/3] Launching EMR cluster with step (preset: {preset})...")
    print(f"  Workers: {config['worker_count']}x {config['worker_type']}")
    print(f"  Output: {output_path}")
    if sample > 0:
        print(f"  SAMPLE MODE: {sample} files per class")
    print(f"  Bootstrap samples: {bootstrap}")
    
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
            "KeepJobFlowAliveWhenNoSteps": False,  # Auto-terminate when done
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
                    "spark.driver.maxResultSize": "8g",  # For collecting 420K+ results
                    "spark.driver.memory": "16g",  # Larger driver for result aggregation
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
        # Embed step directly - runs even if launcher disconnects
        Steps=[
            {
                "Name": "ValidateCutouts",
                "ActionOnFailure": "TERMINATE_CLUSTER",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": [
                        "spark-submit",
                        "--deploy-mode", "cluster",
                        # Use 10g driver for 2xlarge, 4g for xlarge instances
                        "--driver-memory", "10g" if "2xlarge" in config["worker_type"] else "4g",
                        "--executor-memory", config["executor_memory"],
                        "--conf", f"spark.driver.maxResultSize={'8g' if '2xlarge' in config['worker_type'] else '2g'}",
                        uploads["script"],
                        "--positives", positives,
                        "--negatives", negatives,
                        "--output", output_path,
                        "--sample", str(sample),
                        "--bootstrap", str(bootstrap),
                    ],
                },
            },
        ],
    )
    
    cluster_id = response["JobFlowId"]
    print(f"  Cluster ID: {cluster_id}")
    return cluster_id, output_path


def monitor_cluster(cluster_id: str, timeout_minutes: int = 90):
    """Monitor cluster until step completes or fails."""
    print("\n[3/3] Monitoring...")
    print(f"Console: {get_emr_console_url(cluster_id)}")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    start = time.time()
    
    while time.time() - start < timeout_minutes * 60:
        # Check cluster state
        response = emr.describe_cluster(ClusterId=cluster_id)
        cluster_status = response["Cluster"]["Status"]["State"]
        elapsed = (time.time() - start) / 60
        
        # Check step state
        steps = emr.list_steps(ClusterId=cluster_id)["Steps"]
        step_status = steps[0]["Status"]["State"] if steps else "PENDING"
        
        print(f"  [{elapsed:.0f}m] Cluster: {cluster_status}, Step: {step_status}")
        
        if step_status == "COMPLETED":
            return True
        elif step_status in ("FAILED", "CANCELLED"):
            return False
        elif cluster_status in ("TERMINATED", "TERMINATED_WITH_ERRORS"):
            return step_status == "COMPLETED"
        
        time.sleep(60)
    
    print("  Timeout - check EMR console")
    return False


def main():
    parser = argparse.ArgumentParser(description="Launch cutout validation EMR job")
    parser.add_argument("--positives", required=True, help="S3 path to positive cutouts")
    parser.add_argument("--negatives", required=True, help="S3 path to negative cutouts")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="medium")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample N files per class for mini run (0 = all)")
    parser.add_argument("--bootstrap", type=int, default=200,
                        help="Bootstrap samples for AUC CI (default: 200, use 20 for mini)")
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
    
    # Launch cluster with step embedded (robust to launcher disconnect)
    cluster_id, output_path = launch_emr_with_step(
        args.preset, 
        uploads, 
        args.positives, 
        args.negatives,
        sample=args.sample,
        bootstrap=args.bootstrap,
    )
    
    print(f"\n" + "=" * 60)
    print(f"Cluster: {cluster_id}")
    print(f"Output:  {output_path}")
    print(f"Console: {get_emr_console_url(cluster_id)}")
    print("=" * 60)
    print("\nCluster will auto-terminate when step completes.")
    
    if not args.no_monitor:
        success = monitor_cluster(cluster_id)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
EMR Launcher: Validate Cutout Integrity

Launches EMR cluster to validate positive cutouts against fresh Legacy Survey downloads.
Performs pixel-by-pixel comparison to ensure FITS parsing and storage correctness.

Usage:
    python emr/launch_validate_cutout_integrity.py --preset small
    python emr/launch_validate_cutout_integrity.py --preset small --sample-fraction 0.1

Author: Generated for stronglens_calibration project
Date: 2026-02-08
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import boto3

# Import central constants
# When running locally, add parent to path for import
sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import (
    AWS_REGION,
    S3_BUCKET,
    S3_CODE_PREFIX,
    S3_LOGS_PREFIX,
    S3_VALIDATION_PREFIX,
    S3_CUTOUTS_POSITIVES_PREFIX,
    EMR_RELEASE,
    EMR_PRESETS,
    s3_uri,
)

EMR_DIR = Path(__file__).parent.resolve()


# =============================================================================
# HELPERS
# =============================================================================

def find_latest_cutouts_dir(bucket: str, prefix: str) -> str:
    """Find the most recent cutouts directory with /data/ suffix."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    
    # List subdirectories (timestamps)
    paginator = s3.get_paginator("list_objects_v2")
    
    # Look for summary.json files to identify completed runs
    summaries = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for common_prefix in page.get("CommonPrefixes", []):
            subdir = common_prefix["Prefix"]
            summary_key = f"{subdir}summary.json"
            try:
                s3.head_object(Bucket=bucket, Key=summary_key)
                # Extract timestamp from path
                timestamp = subdir.rstrip("/").split("/")[-1]
                summaries.append((timestamp, subdir))
            except:
                pass
    
    if not summaries:
        return None
    
    # Sort by timestamp (descending) and return latest
    summaries.sort(reverse=True)
    latest_dir = summaries[0][1]
    # The cutouts are stored directly in the timestamp directory, not in /data/
    return f"s3://{bucket}/{latest_dir}"


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
    script_path = EMR_DIR / "spark_validate_cutout_integrity.py"
    s3_key = f"{S3_CODE_PREFIX}/{timestamp}/spark_validate_cutout_integrity.py"
    uploads["script"] = upload_to_s3(script_path, s3_key)
    
    uploads["timestamp"] = timestamp
    print(f"  Uploaded {len(uploads) - 1} files")
    
    return uploads


def launch_emr_cluster(preset: str, uploads: Dict[str, str]) -> str:
    """Launch EMR cluster."""
    config = EMR_PRESETS[preset]
    timestamp = uploads["timestamp"]
    
    print(f"\n[2/4] Launching EMR cluster (preset: {preset})...")
    print(f"  Master:  1x {config['master_type']}")
    print(f"  Workers: {config['worker_count']}x {config['worker_type']}")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    
    cluster_config = {
        "Name": f"validate-cutout-integrity-{timestamp}",
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
            "KeepJobFlowAliveWhenNoSteps": False,  # Auto-terminate when done
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
    input_path: str,
    sample_fraction: float,
) -> str:
    """Submit Spark step."""
    timestamp = uploads["timestamp"]
    output_path = f"s3://{S3_BUCKET}/{S3_VALIDATION_PREFIX}/positives/{timestamp}/"
    
    print(f"\n[4/4] Submitting Spark step...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Sample fraction: {sample_fraction}")
    
    emr = boto3.client("emr", region_name=AWS_REGION)
    
    step_args = [
        "spark-submit",
        "--deploy-mode", "cluster",
        "--driver-memory", "4g",
        "--executor-memory", "4g",
        "--executor-cores", "2",
        "--conf", "spark.dynamicAllocation.enabled=true",
        uploads["script"],
        "--input", input_path,
        "--output", output_path,
        "--sample-fraction", str(sample_fraction),
    ]
    
    response = emr.add_job_flow_steps(
        JobFlowId=cluster_id,
        Steps=[
            {
                "Name": "ValidateCutoutIntegrity",
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
        try:
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
                
        except emr.exceptions.ClusterNotFound:
            # Cluster terminated (expected with KeepJobFlowAliveWhenNoSteps=False)
            print("\n  Cluster terminated (expected behavior)")
            return True
        except Exception as e:
            if "Cluster" in str(e) and "terminated" in str(e).lower():
                print("\n  Cluster terminated (expected behavior)")
                return True
            raise
        
        time.sleep(60)


def check_validation_results(output_path: str):
    """Check validation results after job completion."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    
    bucket = output_path.replace("s3://", "").split("/")[0]
    prefix = "/".join(output_path.replace("s3://", "").split("/")[1:]).rstrip("/")
    
    summary_key = f"{prefix}/summary.json"
    
    try:
        response = s3.get_object(Bucket=bucket, Key=summary_key)
        summary = response["Body"].read().decode("utf-8")
        import json
        summary_data = json.loads(summary)
        
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        print(f"Total cutouts:   {summary_data.get('total_cutouts', 'N/A')}")
        print(f"Validated:       {summary_data.get('cutouts_validated', 'N/A')}")
        print(f"Identical:       {summary_data.get('identical', 'N/A')}")
        print(f"Close match:     {summary_data.get('close_match', 'N/A')}")
        print(f"Issues:          {summary_data.get('issues', 'N/A')}")
        print(f"Pass rate:       {summary_data.get('pass_rate', 0)*100:.1f}%")
        print(f"Gate passed:     {summary_data.get('gate_passed', 'N/A')}")
        
        if summary_data.get("correlation_stats"):
            print("\nCorrelation stats:")
            for band, stats in summary_data["correlation_stats"].items():
                if stats.get("mean"):
                    print(f"  {band}: mean={stats['mean']:.6f}, min={stats.get('min', 'N/A')}")
        
        return summary_data.get("gate_passed", False)
        
    except Exception as e:
        print(f"Could not retrieve results: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Launch EMR cutout integrity validation job")
    parser.add_argument("--preset", choices=list(EMR_PRESETS.keys()), default="small",
                       help="Instance preset")
    parser.add_argument("--input", help="Override input path (cutouts directory)")
    parser.add_argument("--sample-fraction", type=float, default=1.0,
                       help="Fraction of cutouts to validate (0-1)")
    parser.add_argument("--no-monitor", action="store_true",
                       help="Don't monitor after launch")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EMR Cutout Integrity Validation Launcher")
    print("=" * 60)
    print(f"Preset: {args.preset} - {EMR_PRESETS[args.preset]['description']}")
    
    # Determine input path
    if args.input:
        input_path = args.input
    else:
        # Auto-detect latest cutouts directory
        print("\nAuto-detecting latest cutouts directory...")
        input_path = find_latest_cutouts_dir(
            S3_BUCKET, 
            "stronglens_calibration/cutouts/positives/"
        )
        if not input_path:
            print("ERROR: No cutouts found. Specify --input explicitly.")
            sys.exit(1)
    
    print(f"Input: {input_path}")
    print(f"Sample fraction: {args.sample_fraction}")
    
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
    step_id = submit_spark_step(cluster_id, uploads, input_path, args.sample_fraction)
    
    output_path = f"s3://{S3_BUCKET}/{S3_VALIDATION_PREFIX}/positives/{uploads['timestamp']}/"
    
    print(f"\n" + "=" * 60)
    print(f"Cluster ID: {cluster_id}")
    print(f"Step ID:    {step_id}")
    print(f"Output:     {output_path}")
    print("=" * 60)
    
    if not args.no_monitor:
        success = monitor_step(cluster_id, step_id)
        
        if success:
            # Check results
            gate_passed = check_validation_results(output_path)
            if gate_passed:
                print("\n✅ VALIDATION PASSED - All cutouts match Legacy Survey")
            elif gate_passed is False:
                print("\n⚠️  VALIDATION ISSUES DETECTED - Check issues.json")
            
        sys.exit(0 if success else 1)
    else:
        print("\nNot monitoring (--no-monitor).")
        print("Cluster will auto-terminate when done (KeepJobFlowAliveWhenNoSteps=False)")


if __name__ == "__main__":
    main()

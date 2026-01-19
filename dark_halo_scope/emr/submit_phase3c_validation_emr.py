#!/usr/bin/env python3
"""
Submit Phase 3c Comprehensive Validation to EMR
================================================

Creates a transient EMR cluster to run comprehensive validation of Phase 3c outputs.

Usage:
    python emr/submit_phase3c_validation_emr.py \
        --region us-east-2 \
        --phase3c-parquet s3://darkhaloscope/phase3_pipeline/phase3c/v3_color_relaxed/parent_union_parquet \
        --output-s3 s3://darkhaloscope/phase3_pipeline/phase3c/v3_color_relaxed/validation \
        --variant v3_color_relaxed
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


def upload_to_s3(local_path: str, s3_path: str, region: str) -> None:
    """Upload a file to S3."""
    print(f"  Uploading {local_path} -> {s3_path}")
    subprocess.run(["aws", "s3", "cp", local_path, s3_path, "--region", region], check=True)


def main():
    parser = argparse.ArgumentParser(description="Submit Phase 3c Validation to EMR")
    
    # AWS settings
    parser.add_argument("--region", default="us-east-2", help="AWS region")
    parser.add_argument("--subnet-id", default="subnet-01ca3ae3325cec025", help="VPC subnet ID")
    parser.add_argument("--ec2-key-name", default="root", help="EC2 key pair name")
    parser.add_argument("--service-role", default="EMR_DefaultRole")
    parser.add_argument("--jobflow-role", default="EMR_EC2_DefaultRole")
    parser.add_argument("--log-uri", default="s3://darkhaloscope/emr-logs/validation/")
    
    # Cluster settings
    parser.add_argument("--master-instance-type", default="m5.xlarge")
    parser.add_argument("--core-instance-type", default="m5.xlarge")
    parser.add_argument("--core-instance-count", type=int, default=4)
    parser.add_argument("--emr-release", default="emr-6.15.0")
    
    # Validation settings
    parser.add_argument("--phase3c-parquet", required=True, help="S3 path to Phase 3c parent catalog")
    parser.add_argument("--phase3a-bricks", default=None, help="S3 path to Phase 3a bricks_with_region (optional)")
    parser.add_argument("--phase3b-selections", default=None, help="S3 path to Phase 3b region_selections (optional)")
    parser.add_argument("--output-s3", required=True, help="S3 path for validation outputs")
    parser.add_argument("--variant", default="v3_color_relaxed", help="Target LRG variant")
    
    # Code locations
    parser.add_argument("--code-s3-prefix", default="s3://darkhaloscope/phase3/code/",
                        help="S3 prefix for uploading code")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Phase 3c Comprehensive Validation - EMR Submission")
    print("=" * 60)
    print()
    
    # Find script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    validation_script = os.path.join(script_dir, "spark_validate_phase3c_comprehensive.py")
    bootstrap_script = os.path.join(script_dir, "bootstrap_phase3c_validation.sh")
    
    if not os.path.exists(validation_script):
        print(f"ERROR: Validation script not found: {validation_script}")
        sys.exit(1)
    
    if not os.path.exists(bootstrap_script):
        print(f"ERROR: Bootstrap script not found: {bootstrap_script}")
        sys.exit(1)
    
    # Upload scripts to S3
    print("[1/3] Uploading scripts to S3...")
    code_prefix = args.code_s3_prefix.rstrip("/")
    
    validation_s3 = f"{code_prefix}/spark_validate_phase3c_comprehensive.py"
    bootstrap_s3 = f"{code_prefix}/bootstrap_phase3c_validation.sh"
    
    upload_to_s3(validation_script, validation_s3, args.region)
    upload_to_s3(bootstrap_script, bootstrap_s3, args.region)
    
    # Build step arguments
    print("[2/3] Building cluster configuration...")
    
    step_args = [
        "spark-submit",
        "--deploy-mode", "client",
        "--driver-memory", "4g",
        "--executor-memory", "6g",
        "--executor-cores", "2",
        "--conf", "spark.dynamicAllocation.enabled=true",
        "--conf", "spark.sql.adaptive.enabled=true",
        validation_s3,
        "--phase3c-parquet", args.phase3c_parquet,
        "--output-s3", args.output_s3,
        "--variant", args.variant,
    ]
    
    if args.phase3a_bricks:
        step_args.extend(["--phase3a-bricks", args.phase3a_bricks])
    
    if args.phase3b_selections:
        step_args.extend(["--phase3b-selections", args.phase3b_selections])
    
    # Build cluster name
    cluster_name = f"Phase3c-Validation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Launch cluster
    print("[3/3] Launching EMR cluster...")
    print()
    print(f"  Cluster: {cluster_name}")
    print(f"  Master: {args.master_instance_type} x 1")
    print(f"  Core: {args.core_instance_type} x {args.core_instance_count}")
    print(f"  Input: {args.phase3c_parquet}")
    print(f"  Output: {args.output_s3}")
    print()
    
    # Build instance groups JSON
    instance_groups = json.dumps([
        {
            "Name": "Master",
            "InstanceGroupType": "MASTER",
            "InstanceType": args.master_instance_type,
            "InstanceCount": 1,
        },
        {
            "Name": "Core",
            "InstanceGroupType": "CORE",
            "InstanceType": args.core_instance_type,
            "InstanceCount": args.core_instance_count,
        },
    ])
    
    # Build step JSON
    steps = json.dumps([
        {
            "Name": "Phase3c Comprehensive Validation",
            "ActionOnFailure": "TERMINATE_CLUSTER",
            "Type": "CUSTOM_JAR",
            "Jar": "command-runner.jar",
            "Args": step_args,
        }
    ])
    
    # Build bootstrap actions JSON
    bootstrap_actions = json.dumps([
        {
            "Name": "Install dependencies",
            "Path": bootstrap_s3,
        }
    ])
    
    cmd = [
        "aws", "emr", "create-cluster",
        "--region", args.region,
        "--name", cluster_name,
        "--release-label", args.emr_release,
        "--applications", "Name=Spark",
        "--instance-groups", instance_groups,
        "--ec2-attributes", f"SubnetId={args.subnet_id},KeyName={args.ec2_key_name}",
        "--service-role", args.service_role,
        "--ec2-attributes", f"InstanceProfile={args.jobflow_role}",
        "--log-uri", args.log_uri,
        "--steps", steps,
        "--bootstrap-actions", bootstrap_actions,
        "--auto-terminate",
        "--visible-to-all-users",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("ERROR: Failed to create cluster")
        print(result.stderr)
        sys.exit(1)
    
    response = json.loads(result.stdout)
    cluster_id = response.get("ClusterId")
    
    print("=" * 60)
    print(f"Cluster launched: {cluster_id}")
    print("=" * 60)
    print()
    print("Monitor progress:")
    print(f"  aws emr describe-cluster --cluster-id {cluster_id} --region {args.region}")
    print()
    print("View logs:")
    print(f"  {args.log_uri}{cluster_id}/")
    print()
    print("Outputs will be written to:")
    print(f"  {args.output_s3}/validation_report_json/")
    print(f"  {args.output_s3}/validation_report_txt/")
    print()
    print("To read the text report after completion:")
    print(f"  aws s3 cp {args.output_s3}/validation_report_txt/part-00000 - | head -200")


if __name__ == "__main__":
    main()


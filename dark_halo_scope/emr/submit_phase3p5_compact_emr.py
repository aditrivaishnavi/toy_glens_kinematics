#!/usr/bin/env python3
"""Submit Phase 3.5 compact job to EMR using boto3.

This job compacts the ~188k small Parquet files from Phase 3c into
a clean, performant layout with only ~100-300 files.

Usage:
    python3 emr/submit_phase3p5_compact_emr.py \
        --region us-east-2 \
        --subnet-id subnet-01ca3ae3325cec025 \
        --ec2-key-name root \
        --variant v3_color_relaxed \
        --num-partitions 96
"""

import argparse
import subprocess
import sys
from typing import List

import boto3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Submit Phase 3.5 compact job to EMR")
    
    # AWS / EMR basics
    p.add_argument("--region", default="us-east-2")
    p.add_argument("--release-label", default="emr-6.15.0")
    p.add_argument("--log-uri", default="s3://darkhaloscope/emr-logs/phase3p5/")
    p.add_argument("--service-role", default="EMR_DefaultRole")
    p.add_argument("--jobflow-role", default="EMR_EC2_DefaultRole")
    
    p.add_argument("--subnet-id", default="subnet-01ca3ae3325cec025")
    p.add_argument("--ec2-key-name", default="root")
    
    # Cluster sizing
    p.add_argument("--master-instance-type", default="m5.xlarge")
    p.add_argument("--core-instance-type", default="m5.xlarge")
    p.add_argument("--core-instance-count", type=int, default=4,
                   help="Number of core nodes (default: 4)")
    
    # S3 paths
    p.add_argument("--s3-bucket", default="darkhaloscope")
    p.add_argument("--variant", default="v3_color_relaxed")
    
    # Job config
    p.add_argument("--num-partitions", type=int, default=96,
                   help="Number of output partitions (default: 96)")
    
    return p


def upload_to_s3(local_path: str, s3_path: str, region: str) -> None:
    """Upload a file to S3."""
    cmd = ["aws", "s3", "cp", "--region", region, local_path, s3_path]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)


def main() -> None:
    args = build_parser().parse_args()
    
    # Derived paths
    input_path = f"s3://{args.s3_bucket}/phase3_pipeline/phase3c/{args.variant}/parent_union_parquet"
    output_path = f"s3://{args.s3_bucket}/phase3_pipeline/phase3p5/{args.variant}/parent_compact"
    
    code_s3_prefix = f"s3://{args.s3_bucket}/emr_code/phase3p5"
    script_s3 = f"{code_s3_prefix}/spark_phase3p5_compact.py"
    bootstrap_s3 = f"{code_s3_prefix}/bootstrap_phase3p5_compact.sh"
    
    print("=" * 80)
    print("PHASE 3.5: SUBMIT COMPACT JOB TO EMR")
    print("=" * 80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Partitions: {args.num_partitions}")
    print(f"Cluster: {args.core_instance_count} x {args.core_instance_type}")
    print("=" * 80)
    
    # Upload scripts to S3
    print("\n[1/2] Uploading scripts to S3...")
    upload_to_s3("emr/spark_phase3p5_compact.py", script_s3, args.region)
    upload_to_s3("emr/bootstrap_phase3p5_compact.sh", bootstrap_s3, args.region)
    
    # Build Spark submit command
    spark_submit: List[str] = [
        "spark-submit",
        "--deploy-mode", "client",
        "--master", "yarn",
        "--driver-memory", "4g",
        "--executor-memory", "6g",
        "--executor-cores", "2",
        "--conf", "spark.sql.shuffle.partitions=200",
        "--conf", "spark.dynamicAllocation.enabled=true",
        "--conf", "spark.dynamicAllocation.minExecutors=1",
        "--conf", "spark.dynamicAllocation.maxExecutors=20",
        "--conf", "spark.shuffle.service.enabled=true",
        script_s3,
        "--input-path", input_path,
        "--output-path", output_path,
        "--num-partitions", str(args.num_partitions),
        "--parent-variant", args.variant,
    ]
    
    # Bootstrap action
    bootstrap_action = {
        "Name": "install-deps",
        "ScriptBootstrapAction": {"Path": bootstrap_s3},
    }
    
    # Step definition
    step = {
        "Name": "Phase3p5-Compact",
        "ActionOnFailure": "TERMINATE_CLUSTER",
        "HadoopJarStep": {
            "Jar": "command-runner.jar",
            "Args": spark_submit,
        },
    }
    
    # Job flow definition
    job_flow = {
        "Name": f"DHS-Phase3p5-Compact-{args.variant}",
        "ReleaseLabel": args.release_label,
        "LogUri": args.log_uri,
        "ServiceRole": args.service_role,
        "JobFlowRole": args.jobflow_role,
        "VisibleToAllUsers": True,
        "Applications": [{"Name": "Spark"}],
        "Instances": {
            "InstanceGroups": [
                {
                    "Name": "Master",
                    "Market": "ON_DEMAND",
                    "InstanceRole": "MASTER",
                    "InstanceType": args.master_instance_type,
                    "InstanceCount": 1,
                },
                {
                    "Name": "Core",
                    "Market": "ON_DEMAND",
                    "InstanceRole": "CORE",
                    "InstanceType": args.core_instance_type,
                    "InstanceCount": args.core_instance_count,
                },
            ],
            "KeepJobFlowAliveWhenNoSteps": False,
            "TerminationProtected": False,
            "Ec2SubnetId": args.subnet_id,
            "Ec2KeyName": args.ec2_key_name,
        },
        "BootstrapActions": [bootstrap_action],
        "Steps": [step],
    }
    
    # Create cluster using boto3
    print("\n[2/2] Creating EMR cluster...")
    emr = boto3.client("emr", region_name=args.region)
    resp = emr.run_job_flow(**job_flow)
    cluster_id = resp.get("JobFlowId")
    
    print("\n" + "=" * 80)
    print("CLUSTER LAUNCHED!")
    print("=" * 80)
    print(f"Cluster ID: {cluster_id}")
    print(f"")
    print("Monitor:")
    print(f"  aws emr describe-cluster --region {args.region} --cluster-id {cluster_id}")
    print(f"")
    print("Logs:")
    print(f"  {args.log_uri}")
    print("=" * 80)


if __name__ == "__main__":
    main()

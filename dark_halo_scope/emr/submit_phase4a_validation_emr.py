#!/usr/bin/env python3
"""
Submit Phase 4a validation as a Spark job on EMR.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime

import boto3


def upload_to_s3(local_path: str, s3_uri: str, region: str):
    """Upload a file to S3."""
    cmd = f"aws s3 cp {local_path} {s3_uri} --region {region}"
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser(description="Submit Phase 4a validation EMR job")
    
    # S3 paths
    parser.add_argument("--manifests-s3", required=True, help="S3 path to manifests")
    parser.add_argument("--bricks-manifest-s3", required=True, help="S3 path to bricks manifest")
    parser.add_argument("--stage-config-s3", required=True, help="S3 path to stage config JSON")
    parser.add_argument("--output-s3", required=True, help="S3 path for validation report JSON")
    parser.add_argument("--output-md-s3", help="S3 path for markdown summary")
    
    # EMR config
    parser.add_argument("--code-s3-prefix", default="s3://darkhaloscope/code/phase4_validation",
                        help="S3 prefix for uploading code")
    parser.add_argument("--log-uri", default="s3://darkhaloscope/emr-logs/phase4_validation/",
                        help="S3 path for EMR logs")
    parser.add_argument("--region", default="us-east-2", help="AWS region")
    parser.add_argument("--release-label", default="emr-6.15.0", help="EMR release")
    parser.add_argument("--master-instance-type", default="m5.xlarge")
    parser.add_argument("--core-instance-type", default="m5.xlarge")
    parser.add_argument("--core-instance-count", type=int, default=5)
    parser.add_argument("--subnet-id", default="subnet-01ca3ae3325cec025")
    parser.add_argument("--ec2-key-name", default="root")
    parser.add_argument("--service-role", default="EMR_DefaultRole")
    parser.add_argument("--jobflow-role", default="EMR_EC2_DefaultRole")
    
    args = parser.parse_args()
    
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    cluster_name = f"phase4a-validation-{timestamp}"
    
    print("=" * 60)
    print("Phase 4a Validation - EMR Submission")
    print("=" * 60)
    
    # 1. Upload Spark script and bootstrap to S3
    script_s3 = f"{args.code_s3_prefix}/spark_validate_phase4a.py"
    bootstrap_s3 = f"{args.code_s3_prefix}/bootstrap_phase4a_validation.sh"
    
    print("\n1. Uploading scripts to S3...")
    upload_to_s3("emr/spark_validate_phase4a.py", script_s3, args.region)
    upload_to_s3("emr/bootstrap_phase4a_validation.sh", bootstrap_s3, args.region)
    
    # 2. Build Spark step
    spark_args = [
        "--manifests-s3", args.manifests_s3,
        "--bricks-manifest-s3", args.bricks_manifest_s3,
        "--stage-config-s3", args.stage_config_s3,
        "--output-s3", args.output_s3,
    ]
    if args.output_md_s3:
        spark_args.extend(["--output-md-s3", args.output_md_s3])
    
    step = {
        "Name": "phase4a-validation",
        "ActionOnFailure": "TERMINATE_CLUSTER",
        "HadoopJarStep": {
            "Jar": "command-runner.jar",
            "Args": [
                "spark-submit",
                "--deploy-mode", "cluster",
                "--conf", "spark.yarn.maxAppAttempts=1",
                "--conf", "spark.sql.parquet.mergeSchema=false",
                "--conf", "spark.sql.adaptive.enabled=true",
                "--executor-memory", "4g",
                "--executor-cores", "2",
                script_s3,
            ] + spark_args,
        },
    }
    
    # 3. Create EMR cluster
    print("\n2. Creating EMR cluster...")
    
    emr = boto3.client("emr", region_name=args.region)
    
    response = emr.run_job_flow(
        Name=cluster_name,
        ReleaseLabel=args.release_label,
        LogUri=args.log_uri,
        VisibleToAllUsers=True,
        JobFlowRole=args.jobflow_role,
        ServiceRole=args.service_role,
        Applications=[{"Name": "Spark"}],
        BootstrapActions=[
            {
                "Name": "Install Python dependencies",
                "ScriptBootstrapAction": {
                    "Path": bootstrap_s3,
                },
            },
        ],
        Instances={
            "Ec2SubnetId": args.subnet_id,
            "Ec2KeyName": args.ec2_key_name,
            "KeepJobFlowAliveWhenNoSteps": False,
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
        },
        Steps=[step],
        Tags=[
            {"Key": "Project", "Value": "DarkHaloScope"},
            {"Key": "Phase", "Value": "4a-validation"},
        ],
    )
    
    cluster_id = response["JobFlowId"]
    
    print(f"\n{'='*60}")
    print(f"EMR Cluster Created: {cluster_id}")
    print(f"{'='*60}")
    print(f"Cluster Name: {cluster_name}")
    print(f"Region: {args.region}")
    print(f"Nodes: 1 master + {args.core_instance_count} core ({args.core_instance_type})")
    print(f"\nMonitor:")
    print(f"  https://{args.region}.console.aws.amazon.com/emr/home?region={args.region}#/clusterDetails/{cluster_id}")
    print(f"\nOutput will be written to:")
    print(f"  JSON: {args.output_s3}")
    if args.output_md_s3:
        print(f"  Markdown: {args.output_md_s3}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Submit EMR job to add paired controls to phase4c parquet.

Usage:
    python submit_paired_controls_emr.py --split train
    python submit_paired_controls_emr.py  # all splits
"""

import argparse
import boto3
import json
import sys
from datetime import datetime

# EMR Configuration
EMR_RELEASE = "emr-6.10.0"
INSTANCE_TYPE_DRIVER = "m5.2xlarge"
INSTANCE_TYPE_WORKER = "m5.2xlarge"
NUM_WORKERS = 20  # 20 workers for parallelism

# S3 paths
S3_BUCKET = "darkhaloscope"
S3_CODE_PREFIX = "emr_code/paired_controls"
S3_LOGS_PREFIX = "emr_logs/paired_controls"

# Bootstrap script content
BOOTSTRAP_SCRIPT = """#!/bin/bash
set -ex
sudo pip3 install boto3 astropy numpy h5py
"""


def upload_code(s3_client, local_path: str, s3_key: str) -> str:
    """Upload file to S3 and return s3:// URI."""
    with open(local_path, 'rb') as f:
        s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=f.read())
    return f"s3://{S3_BUCKET}/{s3_key}"


def create_bootstrap_script(s3_client) -> str:
    """Create and upload bootstrap script."""
    key = f"{S3_CODE_PREFIX}/bootstrap.sh"
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=BOOTSTRAP_SCRIPT.encode('utf-8')
    )
    return f"s3://{S3_BUCKET}/{key}"


def submit_emr_job(args):
    """Submit EMR cluster with Spark job."""
    emr = boto3.client('emr', region_name='us-east-2')
    s3 = boto3.client('s3', region_name='us-east-2')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cluster_name = f"paired-controls-{args.split or 'all'}-{timestamp}"
    
    # Upload code
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    spark_script = os.path.join(script_dir, "spark_add_paired_controls.py")
    
    code_key = f"{S3_CODE_PREFIX}/spark_add_paired_controls_{timestamp}.py"
    code_uri = upload_code(s3, spark_script, code_key)
    print(f"[INFO] Uploaded code to {code_uri}")
    
    # Upload bootstrap
    bootstrap_uri = create_bootstrap_script(s3)
    print(f"[INFO] Created bootstrap at {bootstrap_uri}")
    
    # Build step arguments
    step_args = [
        "spark-submit",
        "--deploy-mode", "cluster",
        "--conf", "spark.executor.memory=12g",
        "--conf", "spark.executor.cores=4",
        "--conf", "spark.driver.memory=8g",
        "--conf", "spark.dynamicAllocation.enabled=true",
        "--conf", "spark.dynamicAllocation.minExecutors=5",
        "--conf", "spark.dynamicAllocation.maxExecutors=100",
        "--conf", "spark.sql.shuffle.partitions=2000",
        code_uri,
        "--partitions", str(args.partitions),
    ]
    
    if args.split:
        step_args.extend(["--split", args.split])
    
    if args.test_limit:
        step_args.extend(["--test-limit", str(args.test_limit)])
    
    if args.dry_run:
        step_args.append("--dry-run")
    
    # EMR cluster configuration
    cluster_config = {
        'Name': cluster_name,
        'ReleaseLabel': EMR_RELEASE,
        'LogUri': f"s3://{S3_BUCKET}/{S3_LOGS_PREFIX}/{timestamp}/",
        'Applications': [
            {'Name': 'Spark'},
            {'Name': 'Hadoop'},
        ],
        'Instances': {
            'InstanceGroups': [
                {
                    'Name': 'Driver',
                    'InstanceRole': 'MASTER',
                    'InstanceType': INSTANCE_TYPE_DRIVER,
                    'InstanceCount': 1,
                },
                {
                    'Name': 'Workers',
                    'InstanceRole': 'CORE',
                    'InstanceType': INSTANCE_TYPE_WORKER,
                    'InstanceCount': NUM_WORKERS,
                },
            ],
            'KeepJobFlowAliveWhenNoSteps': False,
            'TerminationProtected': False,
            'Ec2SubnetId': args.subnet_id,
        },
        'BootstrapActions': [
            {
                'Name': 'Install dependencies',
                'ScriptBootstrapAction': {
                    'Path': bootstrap_uri,
                }
            }
        ],
        'Steps': [
            {
                'Name': 'Add Paired Controls',
                'ActionOnFailure': 'TERMINATE_CLUSTER',
                'HadoopJarStep': {
                    'Jar': 'command-runner.jar',
                    'Args': step_args,
                }
            }
        ],
        'ServiceRole': 'EMR_DefaultRole',
        'JobFlowRole': 'EMR_EC2_DefaultRole',
        'VisibleToAllUsers': True,
        'Tags': [
            {'Key': 'Project', 'Value': 'DarkHaloScope'},
            {'Key': 'Purpose', 'Value': 'PairedControls'},
        ],
    }
    
    if args.dry_run_emr:
        print("[DRY-RUN] Would submit cluster:")
        print(json.dumps(cluster_config, indent=2, default=str))
        return
    
    response = emr.run_job_flow(**cluster_config)
    cluster_id = response['JobFlowId']
    
    print(f"\n{'='*60}")
    print(f"EMR Cluster Submitted Successfully!")
    print(f"{'='*60}")
    print(f"Cluster ID: {cluster_id}")
    print(f"Cluster Name: {cluster_name}")
    print(f"Split: {args.split or 'all'}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"\nMonitor at:")
    print(f"  https://us-east-2.console.aws.amazon.com/emr/home?region=us-east-2#/clusterDetails/{cluster_id}")
    print(f"\nLogs at:")
    print(f"  s3://{S3_BUCKET}/{S3_LOGS_PREFIX}/{timestamp}/")
    print(f"{'='*60}")
    
    return cluster_id


def main():
    parser = argparse.ArgumentParser(description="Submit EMR job for paired controls")
    parser.add_argument("--split", choices=["train", "val", "test"], help="Process single split")
    parser.add_argument("--partitions", type=int, default=1000, help="Output partitions")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit rows for smoke test")
    parser.add_argument("--dry-run", action="store_true", help="Spark dry-run (count only)")
    parser.add_argument("--dry-run-emr", action="store_true", help="Print EMR config without submitting")
    parser.add_argument("--subnet-id", default="subnet-01ca3ae3325cec025", 
                        help="EC2 subnet ID for EMR cluster")
    args = parser.parse_args()
    
    submit_emr_job(args)


if __name__ == "__main__":
    main()

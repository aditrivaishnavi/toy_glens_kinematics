#!/usr/bin/env python

"""
Submit an EMR cluster + PySpark step for Phase 1.5 LRG density estimation.

Usage (from an EC2 host or anywhere with AWS credentials):

  python -m emr.submit_phase1p5_emr_cluster \
      --region us-west-2 \
      --sweep-index-s3 s3://MY_BUCKET/dr10/sweeps_ra150_250_dec0_30_10.1.txt \
      --output-prefix s3://MY_BUCKET/dark_halo_scope/phase1p5 \
      --code-archive-s3 s3://MY_BUCKET/code/dark_halo_scope_code.tgz

The EMR step will:

  spark-submit \
    --deploy-mode cluster \
    --py-files /mnt/dark_halo_scope_code.tgz \
    /mnt/dark_halo_scope_code/emr/spark_phase1p5_lrg_density.py \
    --sweep-index-path s3://.../sweeps_ra150_250_dec0_30_10.1.txt \
    --output-prefix s3://.../phase1p5

You are responsible for ensuring:
  - IAM roles exist (EMR_DefaultRole, EMR_EC2_DefaultRole or your equivalents).
  - The S3 paths and bucket names are correct and accessible.
"""

import argparse
import os
import sys
from typing import Dict, Any, List

import boto3

# Add parent directory to path for imports when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Phase1p5Config


def build_emr_step_args(
    config: Phase1p5Config,
    sweep_index_s3: str,
    output_prefix: str,
) -> Dict[str, Any]:
    """
    Build the EMR step definition for the Phase 1.5 PySpark job.
    """
    # Where the code archive will be placed on EMR master node via bootstrap
    local_code_archive = "/mnt/dark_halo_scope_code.tgz"
    local_extracted_dir = "/mnt/dark_halo_scope_code"

    # PySpark driver path inside the extracted tree
    driver_path = os.path.join(local_extracted_dir, config.emr_pyspark_driver_path)

    # Spark step command: we use command-runner.jar so EMR handles spark-submit
    spark_step = {
        "Name": f"{config.emr_job_name}-pyspark-step",
        "ActionOnFailure": "TERMINATE_CLUSTER",
        "HadoopJarStep": {
            "Jar": "command-runner.jar",
            "Args": [
                "spark-submit",
                "--deploy-mode",
                "cluster",
                "--py-files",
                local_code_archive,
                driver_path,
                "--sweep-index-path",
                sweep_index_s3,
                "--output-prefix",
                output_prefix,
            ],
        },
    }

    return spark_step


def build_bootstrap_actions(config: Phase1p5Config, code_archive_s3: str) -> List[Dict[str, Any]]:
    """
    Bootstrap actions to:
      1. Install Python dependencies (astropy, requests, numpy).
      2. Download the project archive from S3 to the master node and unpack it.

    NOTE: This uses an inline script approach. You may need to adapt
    depending on your EMR configuration or use a separate bootstrap script
    uploaded to S3.
    """
    # The bootstrap script as a shell script
    # EMR runs this on all nodes before the cluster is ready
    bootstrap_script_content = f"""#!/bin/bash
set -euo pipefail

# Install Python deps needed by the PySpark job
sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install numpy astropy requests

# Download project code archive from S3 and unpack to /mnt
aws s3 cp "{code_archive_s3}" /mnt/dark_halo_scope_code.tgz
mkdir -p /mnt/dark_halo_scope_code
tar -xzf /mnt/dark_halo_scope_code.tgz -C /mnt/dark_halo_scope_code
"""

    # Write bootstrap script to a local temp file and upload to S3
    # For simplicity, we'll use an S3 path for the bootstrap script
    # In practice, you might want to upload this script separately
    
    # For now, we'll use an inline approach with a simple bootstrap
    # that assumes the bootstrap script is already in S3
    # You'll need to upload this script to S3 first
    
    return [
        {
            "Name": "phase1p5-install-deps",
            "ScriptBootstrapAction": {
                "Path": "s3://elasticmapreduce/bootstrap-actions/run-if",
                "Args": [
                    "instance.isMaster=true",
                    "bash",
                    "-c",
                    f"""
sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install numpy astropy requests
aws s3 cp {code_archive_s3} /mnt/dark_halo_scope_code.tgz
mkdir -p /mnt/dark_halo_scope_code
tar -xzf /mnt/dark_halo_scope_code.tgz -C /mnt/dark_halo_scope_code
                    """.strip(),
                ],
            },
        }
    ]


def submit_emr_cluster(
    region: str,
    sweep_index_s3: str,
    output_prefix: str,
    code_archive_s3: str,
) -> str:
    """
    Submit the EMR cluster and return the JobFlowId.
    """
    config = Phase1p5Config()

    emr = boto3.client("emr", region_name=region)

    instances = {
        "InstanceGroups": [
            {
                "Name": "Master nodes",
                "Market": "ON_DEMAND",
                "InstanceRole": "MASTER",
                "InstanceType": config.emr_master_instance_type,
                "InstanceCount": 1,
            },
            {
                "Name": "Core nodes",
                "Market": "ON_DEMAND",
                "InstanceRole": "CORE",
                "InstanceType": config.emr_core_instance_type,
                "InstanceCount": config.emr_core_instance_count,
            },
        ],
        "KeepJobFlowAliveWhenNoSteps": False,
        "TerminationProtected": False,
    }

    log_uri = config.emr_s3_log_prefix

    step = build_emr_step_args(
        config=config,
        sweep_index_s3=sweep_index_s3,
        output_prefix=output_prefix,
    )

    bootstrap_actions = build_bootstrap_actions(
        config=config,
        code_archive_s3=code_archive_s3,
    )

    response = emr.run_job_flow(
        Name=config.emr_job_name,
        ReleaseLabel=config.emr_release_label,
        Applications=[{"Name": "Spark"}],
        LogUri=log_uri,
        Instances=instances,
        BootstrapActions=bootstrap_actions,
        Steps=[step],
        JobFlowRole=config.emr_job_flow_role,
        ServiceRole=config.emr_service_role,
        VisibleToAllUsers=True,
    )

    job_flow_id = response["JobFlowId"]
    print(f"Submitted EMR cluster with JobFlowId: {job_flow_id}")
    print(f"  Region: {region}")
    print(f"  Logs: {log_uri}")
    print(f"  Output will be written to: {output_prefix}/lrg_counts_per_brick/")
    return job_flow_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit an EMR cluster for Phase 1.5 PySpark LRG density job."
    )

    parser.add_argument(
        "--region",
        required=True,
        help="AWS region for EMR (e.g. us-west-2).",
    )
    parser.add_argument(
        "--sweep-index-s3",
        required=True,
        help=(
            "S3 URI of the SWEEP index file (newline-delimited list of SWEEP paths). "
            "Example: s3://MY_BUCKET/dr10/sweeps_ra150_250_dec0_30_10.1.txt"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help=(
            "S3 prefix for Phase 1.5 outputs. Example: "
            "s3://MY_BUCKET/dark_halo_scope/phase1p5"
        ),
    )
    parser.add_argument(
        "--code-archive-s3",
        required=True,
        help=(
            "S3 URI of the project code archive (e.g. dark_halo_scope_code.tgz) "
            "that contains this repo and the emr/spark_phase1p5_lrg_density.py file."
        ),
    )

    args = parser.parse_args()

    submit_emr_cluster(
        region=args.region,
        sweep_index_s3=args.sweep_index_s3,
        output_prefix=args.output_prefix,
        code_archive_s3=args.code_archive_s3,
    )


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Submit Phase 4c validation job to EMR.

Usage:
  python3 submit_phase4c_validation_emr.py \
    --experiment-id debug_stamp64_bandsgrz_gridgrid_small \
    --variant v3_color_relaxed
"""

import argparse
import json

import boto3


def main():
    parser = argparse.ArgumentParser(description="Submit Phase 4c validation to EMR")
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--experiment-id", required=True, help="Experiment ID to validate")
    parser.add_argument("--variant", default="v3_color_relaxed")
    parser.add_argument("--log-uri", default="s3://darkhaloscope/emr-logs/phase4/")
    parser.add_argument("--subnet-id", default="subnet-01ca3ae3325cec025")
    parser.add_argument("--ec2-key-name", default="root")
    parser.add_argument("--core-instance-count", type=int, default=2)
    parser.add_argument("--script-s3", default="s3://darkhaloscope/phase4/code/spark_validate_phase4c.py")
    parser.add_argument("--bootstrap-s3", default="s3://darkhaloscope/phase4/code/bootstrap_phase4_pipeline_install_deps.sh")
    args = parser.parse_args()
    
    # Build paths
    base = f"s3://darkhaloscope/phase4_pipeline/phase4c/{args.variant}"
    metrics_s3 = f"{base}/metrics/{args.experiment_id}"
    config_s3 = f"{base}/_stage_config_{args.experiment_id}.json"
    
    emr = boto3.client("emr", region_name=args.region)
    
    cluster_name = f"darkhaloscope-validate-4c-{args.experiment_id[:20]}"
    
    # Build step
    step = {
        "Name": "validate-phase4c",
        "ActionOnFailure": "TERMINATE_CLUSTER",
        "HadoopJarStep": {
            "Jar": "command-runner.jar",
            "Args": [
                "spark-submit",
                "--deploy-mode", "cluster",
                "--conf", "spark.yarn.maxAppAttempts=1",
                args.script_s3,
                "--metrics-s3", metrics_s3,
                "--stage-config-s3", config_s3,
            ],
        },
    }
    
    resp = emr.run_job_flow(
        Name=cluster_name,
        ReleaseLabel="emr-6.15.0",
        LogUri=args.log_uri,
        VisibleToAllUsers=True,
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
        Applications=[{"Name": "Spark"}],
        Instances={
            "Ec2SubnetId": args.subnet_id,
            "Ec2KeyName": args.ec2_key_name,
            "KeepJobFlowAliveWhenNoSteps": False,
            "InstanceGroups": [
                {
                    "Name": "Master nodes",
                    "Market": "ON_DEMAND",
                    "InstanceRole": "MASTER",
                    "InstanceType": "m5.xlarge",
                    "InstanceCount": 1,
                },
                {
                    "Name": "Core nodes",
                    "Market": "ON_DEMAND",
                    "InstanceRole": "CORE",
                    "InstanceType": "m5.xlarge",
                    "InstanceCount": args.core_instance_count,
                },
            ],
        },
        BootstrapActions=[
            {
                "Name": "install-deps",
                "ScriptBootstrapAction": {
                    "Path": args.bootstrap_s3,
                    "Args": [],
                },
            }
        ],
        Steps=[step],
        Tags=[{"Key": "project", "Value": "darkhaloscope"}, {"Key": "phase", "Value": "4c-validation"}],
    )
    
    print(json.dumps({
        "job_flow_id": resp.get("JobFlowId"),
        "name": cluster_name,
        "metrics_s3": metrics_s3,
        "config_s3": config_s3,
    }, indent=2))


if __name__ == "__main__":
    main()


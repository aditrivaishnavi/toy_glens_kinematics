#!/usr/bin/env python3
"""Submit Phase 4 pipeline stages to EMR.

This is patterned after the Phase 3 launcher: it creates a transient cluster
and runs a single Spark step.

Usage examples are at the bottom of this file.
"""

import argparse
import json
import shlex
from typing import Dict, List

import boto3


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--region", default="us-east-2")
    p.add_argument("--stage", required=True, choices=["4a", "4b", "4c", "4d", "4p5"])
    p.add_argument("--log-uri", required=True)
    p.add_argument("--service-role", default="EMR_DefaultRole")
    p.add_argument("--jobflow-role", default="EMR_EC2_DefaultRole")
    p.add_argument("--subnet-id", required=True)
    p.add_argument("--ec2-key-name", required=True)

    p.add_argument("--name", default=None)
    p.add_argument("--release-label", default="emr-6.15.0")

    p.add_argument("--master-instance-type", default="m5.xlarge")
    p.add_argument("--core-instance-type", default="m5.2xlarge")
    p.add_argument("--core-instance-count", type=int, default=10)

    p.add_argument("--script-s3", required=True)
    p.add_argument("--bootstrap-s3", required=True)

    # Spark tuning
    p.add_argument("--executor-memory", default=None)
    p.add_argument("--executor-cores", type=int, default=None)
    p.add_argument("--executor-overhead", default=None)

    # String forwarded verbatim to the phase4 script
    p.add_argument("--spark-args", required=True)

    return p.parse_args()


def _build_step(script_s3: str, stage: str, spark_args: str, tuning: Dict[str, str]) -> Dict:
    base = [
        "spark-submit",
        "--deploy-mode",
        "cluster",
        "--conf",
        "spark.yarn.maxAppAttempts=1",
        "--conf",
        "spark.sql.parquet.mergeSchema=false",
    ]

    if tuning.get("executor_memory"):
        base += ["--executor-memory", tuning["executor_memory"]]
    if tuning.get("executor_cores"):
        base += ["--executor-cores", str(tuning["executor_cores"]) ]
    if tuning.get("executor_overhead"):
        base += ["--conf", f"spark.executor.memoryOverhead={tuning['executor_overhead']}"]

    # Script invocation
    cmd = base + [script_s3, "--stage", stage] + shlex.split(spark_args)

    return {
        "Name": f"phase4-{stage}",
        "ActionOnFailure": "TERMINATE_CLUSTER",
        "HadoopJarStep": {
            "Jar": "command-runner.jar",
            "Args": cmd,
        },
    }


def main() -> None:
    args = _parse_args()

    emr = boto3.client("emr", region_name=args.region)

    cluster_name = args.name or f"darkhaloscope-phase4-{args.stage}"

    tuning = {
        "executor_memory": args.executor_memory,
        "executor_cores": args.executor_cores,
        "executor_overhead": args.executor_overhead,
    }

    step = _build_step(args.script_s3, args.stage, args.spark_args, tuning)

    resp = emr.run_job_flow(
        Name=cluster_name,
        ReleaseLabel=args.release_label,
        LogUri=args.log_uri,
        VisibleToAllUsers=True,
        JobFlowRole=args.jobflow_role,
        ServiceRole=args.service_role,
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
                    "InstanceType": args.master_instance_type,
                    "InstanceCount": 1,
                },
                {
                    "Name": "Core nodes",
                    "Market": "ON_DEMAND",
                    "InstanceRole": "CORE",
                    "InstanceType": args.core_instance_type,
                    "InstanceCount": args.core_instance_count,
                },
            ],
        },
        BootstrapActions=[
            {
                "Name": "install-phase4-deps",
                "ScriptBootstrapAction": {
                    "Path": args.bootstrap_s3,
                    "Args": [],
                },
            }
        ],
        Steps=[step],
        Tags=[{"Key": "project", "Value": "darkhaloscope"}, {"Key": "phase", "Value": "4"}],
    )

    print(json.dumps({"job_flow_id": resp.get("JobFlowId"), "name": cluster_name}, indent=2))


if __name__ == "__main__":
    main()

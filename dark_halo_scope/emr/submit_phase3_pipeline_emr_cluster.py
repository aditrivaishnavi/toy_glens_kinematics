#!/usr/bin/env python3
"""Submit Phase 3 pipeline stages (3a/3b/3c) to EMR.

This script:
  1) Creates an ephemeral EMR cluster (one per stage invocation)
  2) Adds a single Spark step that runs spark_phase3_pipeline.py with --stage {3a|3b|3c}
  3) Terminates the cluster after the step (KeepJobFlowAliveWhenNoSteps = False)

Notes:
- Defaults are aligned to earlier Phase 2 settings (EMR 6.15.x, Python 3.7).
- Tune core-instance-count and Spark memory depending on Stage.
"""

import argparse
import shlex
from typing import List

import boto3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # AWS / EMR basics
    p.add_argument("--region", required=True)
    p.add_argument("--release-label", default="emr-6.15.0")

    p.add_argument("--log-uri", required=True, help="s3://... EMR log URI")
    p.add_argument("--service-role", default="EMR_DefaultRole")
    p.add_argument("--jobflow-role", default="EMR_EC2_DefaultRole")

    p.add_argument("--subnet-id", required=True)
    p.add_argument("--ec2-key-name", required=True)

    # Cluster sizing
    p.add_argument("--master-instance-type", default="m5.xlarge")
    p.add_argument("--core-instance-type", default="m5.xlarge")
    p.add_argument("--core-instance-count", type=int, default=10)

    # Spark submit conf
    p.add_argument("--executor-memory", default="4g")
    p.add_argument("--executor-overhead", default="10g")
    p.add_argument("--executor-cores", type=int, default=2)
    p.add_argument("--python-worker-memory", default="1g")
    p.add_argument("--dynamic-allocation", type=int, default=1)

    # Stage + code locations
    p.add_argument("--stage", required=True, choices=["3a", "3b", "3c"])
    p.add_argument("--script-s3", required=True, help="s3://.../spark_phase3_pipeline.py")
    p.add_argument("--bootstrap-s3", required=True, help="s3://.../bootstrap_phase3_pipeline_install_deps.sh")

    # Arguments forwarded to spark_phase3_pipeline.py (as a single string)
    p.add_argument(
        "--spark-args",
        required=True,
        help=(
            "Arguments passed to spark_phase3_pipeline.py, e.g. "
            "\"--output-s3 s3://... --phase2-results-s3 s3://... --bricks-fits-s3 s3://... --variant v3_color_relaxed\""
        ),
    )

    return p


def main() -> None:
    args = build_parser().parse_args()
    emr = boto3.client("emr", region_name=args.region)

    bootstrap_action = {
        "Name": "install-python-deps",
        "ScriptBootstrapAction": {"Path": args.bootstrap_s3},
    }

    spark_submit: List[str] = [
        "spark-submit",
        "--deploy-mode",
        "cluster",
        "--master",
        "yarn",
        "--conf",
        f"spark.executor.memory={args.executor_memory}",
        "--conf",
        f"spark.executor.memoryOverhead={args.executor_overhead}",
        "--conf",
        f"spark.executor.cores={args.executor_cores}",
        "--conf",
        f"spark.python.worker.memory={args.python_worker_memory}",
        "--conf",
        f"spark.dynamicAllocation.enabled={'true' if args.dynamic_allocation else 'false'}",
        args.script_s3,
        "--stage",
        args.stage,
    ]

    # Append forwarded args to the application.
    spark_submit.extend(shlex.split(args.spark_args))

    step = {
        "Name": f"phase3_{args.stage}",
        "ActionOnFailure": "TERMINATE_CLUSTER",
        "HadoopJarStep": {
            "Jar": "command-runner.jar",
            "Args": spark_submit,
        },
    }

    job_flow = {
        "Name": f"darkhaloscope-phase3-{args.stage}",
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

    resp = emr.run_job_flow(**job_flow)
    cluster_id = resp.get("JobFlowId")
    print(cluster_id)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Submit an EMR cluster + step for Dark Halo Scope Phase 3 (PySpark).

This mirrors the style of submit_phase2_emr_cluster.py:
  - Uploads the Spark script to S3
  - Creates an EMR cluster
  - Runs a single spark-submit step
  - Terminates on failure (or optionally keep cluster alive)

Python: 3.9+
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.parse

import boto3


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Not an s3:// URI: {s3_uri}")
    p = urllib.parse.urlparse(s3_uri)
    return p.netloc, p.path.lstrip("/")


def s3_join(prefix: str, *parts: str) -> str:
    prefix = prefix.rstrip("/")
    return "/".join([prefix] + [p.strip("/") for p in parts if p])


def upload_file(s3_client, local_path: str, s3_uri: str) -> None:
    bucket, key = parse_s3_uri(s3_uri)
    s3_client.upload_file(local_path, bucket, key)


def build_bootstrap_action(s3_bootstrap_uri: str) -> List[Dict]:
    # Installs required Python packages on all nodes.
    # Keep pinned versions minimal. EMR already ships numpy/pandas; astropy is typically missing.
    return [
        {
            "Name": "Install Python deps (astropy)",
            "ScriptBootstrapAction": {
                "Path": s3_bootstrap_uri,
                "Args": [],
            },
        }
    ]


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--aws-region", default="us-east-2", help="AWS region (default: us-east-2)")
    ap.add_argument("--release-label", default="emr-6.15.0", help="EMR release label")
    ap.add_argument("--ec2-key-name", default="", help="Optional EC2 key pair name")
    ap.add_argument("--subnet-id", default="", help="Optional subnet ID")
    ap.add_argument("--log-uri", default="", help="Optional s3://... EMR log URI")
    ap.add_argument("--keep-alive", action="store_true", help="Do not auto-terminate after step")

    ap.add_argument("--master-instance-type", default="m5.xlarge")
    ap.add_argument("--core-instance-type", default="m5.2xlarge")
    ap.add_argument("--core-count", type=int, default=20)
    
    # Spark memory tuning for PySpark with large FITS files
    ap.add_argument("--executor-memory", default="6g",
                    help="Spark executor memory (default: 6g)")
    ap.add_argument("--executor-memory-overhead", default="4g",
                    help="Spark executor memory overhead for Python/off-heap (default: 4g)")
    ap.add_argument("--executor-cores", type=int, default=2,
                    help="Cores per executor (fewer = less concurrent FITS processing, less OOM risk)")

    ap.add_argument("--phase2-analysis-s3", required=True,
                    help="s3://... prefix containing phase2_analysis outputs")
    ap.add_argument("--variant", default="v3_color_relaxed",
                    help="Variant directory under phase2_analysis (default: v3_color_relaxed)")
    ap.add_argument("--sweep-index-s3", required=True,
                    help="s3://... text file listing sweep S3 URLs (one per line)")
    ap.add_argument("--output-s3", required=True,
                    help="s3://... prefix for phase3 outputs (also used to store code/bootstrap artifacts)")

    ap.add_argument("--num-regions", type=int, default=30)
    ap.add_argument("--ranking-modes", default="n_lrg,area_weighted,psf_weighted")

    # Optional pass-through knobs for Spark job (kept small; edit the script if you need more)
    ap.add_argument("--max-ebv", type=float, default=0.12)
    ap.add_argument("--max-psf-r-arcsec", type=float, default=1.60)
    ap.add_argument("--min-psfdepth-r", type=float, default=23.6)
    ap.add_argument("--sweep-partitions", type=int, default=600)
    ap.add_argument("--chunk-size", type=int, default=100000)
    ap.add_argument("--max-sweeps", type=int, default=0,
                    help="For debugging: limit number of sweeps (0 = all)")
    ap.add_argument("--checkpoint-s3", default="",
                    help="S3 prefix for checkpointing. If empty, uses {output-s3}/{variant}/checkpoints")
    ap.add_argument("--s3-sweep-cache-prefix", default="",
                    help="S3 prefix where gzipped sweep FITS are cached (e.g., s3://bucket/sweep_fits_dump/)")

    args = ap.parse_args()

    session = boto3.session.Session(region_name=args.aws_region)
    s3 = session.client("s3")
    emr = session.client("emr")

    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # Upload Phase 2 results to S3 if they exist locally
    repo_root = Path(__file__).parent.parent
    local_phase2_dir = repo_root / "results" / "phase2_analysis" / args.variant
    
    print(f"\n{'='*60}")
    print("PHASE 2 RESULTS UPLOAD")
    print(f"{'='*60}")
    print(f"Looking for Phase 2 results in: {local_phase2_dir}")
    
    if local_phase2_dir.exists():
        phase2_bucket, phase2_prefix = parse_s3_uri(args.phase2_analysis_s3)
        phase2_prefix = phase2_prefix.rstrip("/")
        
        files_to_upload = [
            "phase2_regions_summary.csv",
            "phase2_regions_bricks.csv",
            "phase2_hypergrid_bricks_merged.csv",
            "phase2_variant_stats.csv",
            "phase2_hypergrid_analysis.md",
        ]
        
        uploaded_count = 0
        for fname in files_to_upload:
            local_file = local_phase2_dir / fname
            if local_file.exists():
                s3_uri = f"s3://{phase2_bucket}/{phase2_prefix}/{args.variant}/{fname}"
                print(f"  aws s3 cp {local_file} {s3_uri}")
                s3.upload_file(str(local_file), phase2_bucket, f"{phase2_prefix}/{args.variant}/{fname}")
                uploaded_count += 1
            else:
                print(f"  [SKIP] {fname} not found locally")
        
        print(f"\nUploaded {uploaded_count} files to s3://{phase2_bucket}/{phase2_prefix}/{args.variant}/")
    else:
        print(f"  [WARN] Directory not found: {local_phase2_dir}")
        print(f"  Assuming Phase 2 results are already in S3")
    print(f"{'='*60}\n")

    # Upload Spark script + bootstrap script to S3
    local_spark = str(Path(__file__).with_name("spark_phase3_define_fields_and_build_parent.py"))
    if not os.path.exists(local_spark):
        raise FileNotFoundError(f"Missing Spark script next to submitter: {local_spark}")

    code_prefix = s3_join(args.output_s3, "code", "phase3", timestamp)
    spark_script_s3 = s3_join(code_prefix, "spark_phase3_define_fields_and_build_parent.py")
    upload_file(s3, local_spark, spark_script_s3)

    # Bootstrap script (inline small shell)
    bootstrap_local = str(Path(__file__).with_name("bootstrap_phase3_install_deps.sh"))
    if not os.path.exists(bootstrap_local):
        Path(bootstrap_local).write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "sudo python3 -m pip install --quiet --upgrade pip\n"
            'sudo python3 -m pip install --quiet "astropy>=4.3,<5.0" "boto3>=1.28.0"\n'
        )
    bootstrap_s3 = s3_join(code_prefix, "bootstrap_phase3_install_deps.sh")
    upload_file(s3, bootstrap_local, bootstrap_s3)

    # Build step args with Spark memory configuration
    step_args = [
        "spark-submit",
        "--deploy-mode", "cluster",
        "--conf", f"spark.executor.memory={args.executor_memory}",
        "--conf", f"spark.executor.memoryOverhead={args.executor_memory_overhead}",
        "--conf", f"spark.executor.cores={args.executor_cores}",
        "--conf", "spark.python.worker.memory=1g",
        "--conf", "spark.dynamicAllocation.enabled=true",
        spark_script_s3,
        "--phase2-analysis-s3", args.phase2_analysis_s3,
        "--variant", args.variant,
        "--sweep-index-s3", args.sweep_index_s3,
        "--output-s3", args.output_s3,
        "--num-regions", str(args.num_regions),
        "--ranking-modes", args.ranking_modes,
        "--max-ebv", str(args.max_ebv),
        "--max-psf-r-arcsec", str(args.max_psf_r_arcsec),
        "--min-psfdepth-r", str(args.min_psfdepth_r),
        "--sweep-partitions", str(args.sweep_partitions),
        "--chunk-size", str(args.chunk_size),
    ]
    
    if args.max_sweeps > 0:
        step_args.extend(["--max-sweeps", str(args.max_sweeps)])
    
    if args.checkpoint_s3:
        step_args.extend(["--checkpoint-s3", args.checkpoint_s3])
    
    if args.s3_sweep_cache_prefix:
        step_args.extend(["--s3-sweep-cache-prefix", args.s3_sweep_cache_prefix])

    steps = [
        {
            "Name": "Phase3: define fields + build parent sample",
            "ActionOnFailure": "TERMINATE_CLUSTER",
            "HadoopJarStep": {
                "Jar": "command-runner.jar",
                "Args": step_args,
            },
        }
    ]

    instances = {
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
                "InstanceCount": args.core_count,
            },
        ],
        "KeepJobFlowAliveWhenNoSteps": bool(args.keep_alive),
        "TerminationProtected": False,
    }
    if args.ec2_key_name:
        instances["Ec2KeyName"] = args.ec2_key_name
    if args.subnet_id:
        instances["Ec2SubnetId"] = args.subnet_id

    cluster_args = {
        "Name": f"dark-halo-scope-phase3-{timestamp}",
        "ReleaseLabel": args.release_label,
        "Applications": [{"Name": "Spark"}],
        "Instances": instances,
        "Steps": steps,
        "BootstrapActions": build_bootstrap_action(bootstrap_s3),
        "VisibleToAllUsers": True,
        "JobFlowRole": "EMR_EC2_DefaultRole",
        "ServiceRole": "EMR_DefaultRole",
    }
    if args.log_uri:
        cluster_args["LogUri"] = args.log_uri

    resp = emr.run_job_flow(**cluster_args)
    cluster_id = resp["JobFlowId"]
    print(cluster_id)


if __name__ == "__main__":
    main()

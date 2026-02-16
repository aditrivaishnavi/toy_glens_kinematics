#!/usr/bin/env python3
"""
EMR Job Launcher

Reusable launcher for submitting Spark jobs to EMR.

Lessons Learned Incorporated:
- L3.1: Verify cluster is ready before submitting
- L5.4: Validate code syntax before uploading
- L6.1: Proper error handling and retries
- L5.2: Check logs immediately after submission

Usage:
    python launcher.py submit \
        --job-name my-job \
        --script path/to/script.py \
        --preset medium \
        --args "--input s3://bucket/input --output s3://bucket/output"
    
    python launcher.py status --cluster-id j-XXXXX
    
    python launcher.py terminate --cluster-id j-XXXXX
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import NoCredentialsError, ClientError, EndpointConnectionError
except ImportError:
    print("ERROR: boto3 not installed. Run: pip install boto3")
    sys.exit(1)

from constants import (
    AWS_REGION, EMR_RELEASE_LABEL,
    S3_BUCKET, S3_CODE_PREFIX, S3_LOGS_PREFIX,
    PRESETS, InstanceConfig, SparkConfig, MAX_VCORES,
)

from aws_utils import (
    validate_credentials,
    s3_upload_file,
    s3_head_bucket,
    get_emr_client,
    get_s3_client,
    handle_aws_error,
    is_credential_error,
    RETRY_CONFIG,
    S3_RETRY_CONFIG,
    CREDENTIAL_ERROR_MSG,
)


def check_aws_credentials() -> bool:
    """
    Check if AWS credentials are valid.
    
    IMPORTANT: AWS credentials expire after 24 hours.
    If this fails, STOP and ask user for new credentials.
    
    Returns:
        True if credentials are valid
    """
    try:
        validate_credentials()
        print(f"✓ AWS credentials valid")
        print(f"  Region: {AWS_REGION}")
        return True
    except SystemExit:
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        print(CREDENTIAL_ERROR_MSG)
        return False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class JobConfig:
    """Configuration for an EMR job."""
    job_name: str
    script_path: str
    preset: str = "medium"
    script_args: str = ""
    
    # Cluster settings (override preset)
    worker_count: Optional[int] = None
    worker_instance_type: Optional[str] = None
    
    # Additional settings
    keep_alive: bool = False
    tags: Dict[str, str] = None
    
    # IAM roles
    service_role: str = "EMR_DefaultRole"
    instance_profile: str = "EMR_EC2_DefaultRole"
    
    # Networking
    subnet_id: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        
        # Add default tags
        self.tags.update({
            "Project": "PlanB",
            "JobName": self.job_name,
            "CreatedAt": datetime.now().isoformat(),
        })


# =============================================================================
# VALIDATION
# =============================================================================

def validate_script(script_path: str) -> Tuple[bool, str]:
    """
    Validate script before uploading.
    
    Lesson L5.4: Validate code before expensive operations.
    
    Returns:
        (success, message)
    """
    path = Path(script_path)
    
    # Check exists
    if not path.exists():
        return False, f"Script not found: {script_path}"
    
    # Check syntax
    result = subprocess.run(
        ["python3", "-m", "py_compile", str(path)],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        return False, f"Syntax error in script:\n{result.stderr}"
    
    # Check for common issues
    content = path.read_text()
    
    issues = []
    if "print(" in content and "logger" not in content.lower():
        issues.append("WARNING: Using print() instead of logger")
    
    if "import pdb" in content or "breakpoint()" in content:
        issues.append("ERROR: Debug statements found (pdb/breakpoint)")
        return False, "\n".join(issues)
    
    if issues:
        print("\n".join(issues))
    
    return True, "Script validation passed"


def validate_s3_path(s3_path: str) -> Tuple[bool, str]:
    """Validate S3 path is accessible."""
    if not s3_path.startswith("s3://"):
        return False, f"Invalid S3 path: {s3_path}"
    
    # Parse bucket and key
    parts = s3_path[5:].split("/", 1)
    bucket = parts[0]
    
    try:
        if s3_head_bucket(bucket):
            return True, "S3 path accessible"
        else:
            return False, f"S3 bucket not found: {bucket}"
    except SystemExit:
        # Credential error - re-raise
        raise
    except Exception as e:
        # Don't silently eat - return the error
        return False, f"Cannot access S3 bucket: {e}"


# =============================================================================
# S3 OPERATIONS
# =============================================================================

def upload_to_s3(local_path: str, s3_key: str) -> str:
    """
    Upload file to S3 with retry and proper exception handling.
    
    Returns:
        S3 URI of uploaded file
    
    Raises:
        All errors are logged and re-raised (never silently eaten)
    """
    # Use the shared utility which has retries and proper error handling
    return s3_upload_file(local_path, S3_BUCKET, s3_key, max_retries=3)


def upload_code_package(script_path: str, job_name: str) -> Dict[str, str]:
    """
    Upload script and dependencies to S3.
    
    Returns:
        Dict with S3 URIs for uploaded files
    
    Raises:
        All errors are logged and re-raised (never silently eaten)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{S3_CODE_PREFIX}/{job_name}/{timestamp}"
    
    uploads = {}
    
    try:
        # Upload main script
        script_name = Path(script_path).name
        print(f"  Uploading main script: {script_name}")
        uploads["script"] = upload_to_s3(script_path, f"{prefix}/{script_name}")
        
        # Upload bootstrap script
        bootstrap_path = Path(__file__).parent / "bootstrap.sh"
        if bootstrap_path.exists():
            print(f"  Uploading bootstrap script")
            uploads["bootstrap"] = upload_to_s3(
                str(bootstrap_path), 
                f"{prefix}/bootstrap.sh"
            )
        else:
            print(f"  WARNING: bootstrap.sh not found at {bootstrap_path}")
        
        # Upload shared module if exists
        shared_dir = Path(__file__).parent.parent / "shared"
        if shared_dir.exists():
            print(f"  Uploading shared module")
            import shutil
            zip_path = f"/tmp/shared_{timestamp}.zip"
            shutil.make_archive(zip_path.replace(".zip", ""), "zip", shared_dir.parent, "shared")
            uploads["shared"] = upload_to_s3(zip_path, f"{prefix}/shared.zip")
        
        print(f"  ✓ All files uploaded to s3://{S3_BUCKET}/{prefix}/")
        return uploads
        
    except SystemExit:
        # Credential error - re-raise
        raise
    except Exception as e:
        # Log but re-raise - never silently eat
        print(f"ERROR uploading code package: {e}")
        raise


# =============================================================================
# EMR OPERATIONS
# =============================================================================

class EMRLauncher:
    """EMR cluster launcher and manager."""
    
    def __init__(self, region: str = AWS_REGION):
        self.region = region
        # Use shared client factory with proper retry configuration
        self.emr = get_emr_client()
    
    def create_cluster(self, config: JobConfig) -> str:
        """
        Create EMR cluster with specified configuration.
        
        Returns:
            Cluster ID
        """
        # Get preset or create custom config
        preset = PRESETS.get(config.preset, PRESETS["medium"])
        instance_config = InstanceConfig(
            worker_count=config.worker_count or preset.instance_config.worker_count,
            worker_instance_type=config.worker_instance_type or preset.instance_config.worker_instance_type,
        )
        spark_config = preset.spark_config
        
        # Upload code
        print("\n[1/4] Uploading code to S3...")
        uploads = upload_code_package(config.script_path, config.job_name)
        
        # Build cluster configuration
        cluster_config = {
            "Name": f"planb-{config.job_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "ReleaseLabel": EMR_RELEASE_LABEL,
            "Applications": [
                {"Name": "Spark"},
                {"Name": "Hadoop"},
            ],
            "Configurations": spark_config.to_spark_defaults(),
            "Instances": {
                "InstanceFleets": instance_config.to_instance_fleet_config(),
                "KeepJobFlowAliveWhenNoSteps": config.keep_alive,
                "TerminationProtected": False,
            },
            "ServiceRole": config.service_role,
            "JobFlowRole": config.instance_profile,
            "LogUri": f"s3://{S3_BUCKET}/{S3_LOGS_PREFIX}/{config.job_name}/",
            "Tags": [{"Key": k, "Value": v} for k, v in config.tags.items()],
            "VisibleToAllUsers": True,
            "StepConcurrencyLevel": 1,
        }
        
        # Add subnet if specified
        if config.subnet_id:
            cluster_config["Instances"]["Ec2SubnetId"] = config.subnet_id
        
        # Add bootstrap action
        if "bootstrap" in uploads:
            cluster_config["BootstrapActions"] = [{
                "Name": "Install dependencies",
                "ScriptBootstrapAction": {
                    "Path": uploads["bootstrap"],
                },
            }]
        
        # Add spark step
        step_args = [
            "spark-submit",
            "--deploy-mode", "cluster",
            "--master", "yarn",
            "--conf", f"spark.executor.memory={spark_config.executor_memory}",
            "--conf", f"spark.executor.cores={spark_config.executor_cores}",
        ]
        
        # Add shared module to py-files if uploaded
        if "shared" in uploads:
            step_args.extend(["--py-files", uploads["shared"]])
        
        step_args.append(uploads["script"])
        
        # Add script arguments
        if config.script_args:
            step_args.extend(config.script_args.split())
        
        cluster_config["Steps"] = [{
            "Name": config.job_name,
            "ActionOnFailure": "TERMINATE_CLUSTER" if not config.keep_alive else "CONTINUE",
            "HadoopJarStep": {
                "Jar": "command-runner.jar",
                "Args": step_args,
            },
        }]
        
        # Create cluster
        print("\n[2/4] Creating EMR cluster...")
        response = self.emr.run_job_flow(**cluster_config)
        cluster_id = response["JobFlowId"]
        
        print(f"\n[3/4] Cluster created: {cluster_id}")
        print(f"       Logs: s3://{S3_BUCKET}/{S3_LOGS_PREFIX}/{config.job_name}/")
        
        return cluster_id
    
    def get_cluster_status(self, cluster_id: str) -> Dict:
        """Get cluster status details."""
        response = self.emr.describe_cluster(ClusterId=cluster_id)
        cluster = response["Cluster"]
        
        status = {
            "id": cluster_id,
            "name": cluster["Name"],
            "state": cluster["Status"]["State"],
            "state_change_reason": cluster["Status"].get("StateChangeReason", {}),
            "created_at": str(cluster.get("Status", {}).get("Timeline", {}).get("CreationDateTime")),
        }
        
        # Get step status
        steps_response = self.emr.list_steps(ClusterId=cluster_id)
        status["steps"] = [
            {
                "name": step["Name"],
                "state": step["Status"]["State"],
                "failure_reason": step["Status"].get("FailureDetails", {}).get("Reason"),
            }
            for step in steps_response["Steps"]
        ]
        
        return status
    
    def wait_for_cluster(
        self,
        cluster_id: str,
        target_states: List[str] = None,
        timeout_minutes: int = 60,
        poll_interval: int = 30,
    ) -> str:
        """
        Wait for cluster to reach target state.
        
        Lesson L3.1: Verify cluster is ready.
        """
        if target_states is None:
            target_states = ["WAITING", "RUNNING", "TERMINATED", "TERMINATED_WITH_ERRORS"]
        
        terminal_states = ["TERMINATED", "TERMINATED_WITH_ERRORS"]
        
        start_time = time.time()
        last_state = None
        
        print(f"\n[4/4] Waiting for cluster {cluster_id}...")
        
        while True:
            elapsed = (time.time() - start_time) / 60
            
            if elapsed > timeout_minutes:
                raise TimeoutError(f"Cluster did not reach target state in {timeout_minutes} minutes")
            
            status = self.get_cluster_status(cluster_id)
            state = status["state"]
            
            if state != last_state:
                print(f"       [{elapsed:.1f}m] State: {state}")
                last_state = state
            
            if state in target_states:
                if state in terminal_states:
                    # Check if steps succeeded
                    for step in status["steps"]:
                        if step["state"] == "FAILED":
                            print(f"       Step failed: {step['name']}")
                            print(f"       Reason: {step['failure_reason']}")
                            return state
                
                return state
            
            time.sleep(poll_interval)
    
    def terminate_cluster(self, cluster_id: str) -> None:
        """Terminate a cluster."""
        print(f"Terminating cluster {cluster_id}...")
        self.emr.terminate_job_flows(JobFlowIds=[cluster_id])
        print("Termination initiated.")
    
    def list_clusters(self, states: List[str] = None) -> List[Dict]:
        """List clusters with optional state filter."""
        if states is None:
            states = ["STARTING", "BOOTSTRAPPING", "RUNNING", "WAITING"]
        
        response = self.emr.list_clusters(ClusterStates=states)
        
        return [
            {
                "id": c["Id"],
                "name": c["Name"],
                "state": c["Status"]["State"],
            }
            for c in response["Clusters"]
        ]


# =============================================================================
# CLI
# =============================================================================

def cmd_submit(args):
    """Submit a new EMR job."""
    # Step 0: Check credentials first
    print("\n" + "="*60)
    print("EMR JOB SUBMISSION")
    print("="*60)
    
    print("\n[0/5] Checking AWS credentials...")
    if not check_aws_credentials():
        print("\n*** STOP: Ask user for new AWS credentials ***")
        sys.exit(1)
    
    # Step 1: Validate script
    print("\n[1/5] Validating script...")
    valid, msg = validate_script(args.script)
    if not valid:
        print(f"ERROR: {msg}")
        sys.exit(1)
    print(f"       {msg}")
    
    # Create job config
    config = JobConfig(
        job_name=args.job_name,
        script_path=args.script,
        preset=args.preset,
        script_args=args.args or "",
        worker_count=args.workers,
        worker_instance_type=args.instance_type,
        keep_alive=args.keep_alive,
        subnet_id=args.subnet,
    )
    
    # Step 2: Validate vCore budget
    print(f"\n[2/5] Checking vCore budget (max {MAX_VCORES})...")
    preset = PRESETS.get(config.preset, PRESETS["medium"])
    instance_config = InstanceConfig(
        worker_count=config.worker_count or preset.instance_config.worker_count,
        worker_instance_type=config.worker_instance_type or preset.instance_config.worker_instance_type,
    )
    total_vcores = instance_config.total_vcores()
    print(f"       Requested: {total_vcores} vCores")
    
    try:
        instance_config.validate_vcore_budget()
        print(f"       ✓ Within budget ({total_vcores}/{MAX_VCORES})")
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Launch cluster
    launcher = EMRLauncher()
    cluster_id = launcher.create_cluster(config)
    
    # Wait if requested
    if args.wait:
        final_state = launcher.wait_for_cluster(cluster_id, timeout_minutes=args.timeout)
        print(f"\nFinal state: {final_state}")
        
        if final_state == "TERMINATED_WITH_ERRORS":
            sys.exit(1)
    
    print(f"\nCluster ID: {cluster_id}")
    print(f"Monitor: aws emr describe-cluster --cluster-id {cluster_id}")


def cmd_status(args):
    """Get cluster status."""
    launcher = EMRLauncher()
    status = launcher.get_cluster_status(args.cluster_id)
    
    print(json.dumps(status, indent=2, default=str))


def cmd_terminate(args):
    """Terminate a cluster."""
    launcher = EMRLauncher()
    launcher.terminate_cluster(args.cluster_id)


def cmd_list(args):
    """List clusters."""
    launcher = EMRLauncher()
    clusters = launcher.list_clusters()
    
    if not clusters:
        print("No active clusters.")
        return
    
    print(f"{'ID':<20} {'Name':<50} {'State':<15}")
    print("-" * 85)
    for c in clusters:
        print(f"{c['id']:<20} {c['name']:<50} {c['state']:<15}")


def main():
    parser = argparse.ArgumentParser(description="EMR Job Launcher")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit a new job")
    submit_parser.add_argument("--job-name", required=True, help="Job name")
    submit_parser.add_argument("--script", required=True, help="Path to Python script")
    submit_parser.add_argument("--preset", default="medium", choices=["small", "medium", "large"])
    submit_parser.add_argument("--args", help="Arguments to pass to script")
    submit_parser.add_argument("--workers", type=int, help="Override worker count")
    submit_parser.add_argument("--instance-type", help="Override instance type")
    submit_parser.add_argument("--keep-alive", action="store_true", help="Keep cluster alive after job")
    submit_parser.add_argument("--wait", action="store_true", help="Wait for job completion")
    submit_parser.add_argument("--timeout", type=int, default=60, help="Wait timeout in minutes")
    submit_parser.add_argument("--subnet", help="EC2 subnet ID")
    submit_parser.set_defaults(func=cmd_submit)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get cluster status")
    status_parser.add_argument("--cluster-id", required=True, help="Cluster ID")
    status_parser.set_defaults(func=cmd_status)
    
    # Terminate command
    term_parser = subparsers.add_parser("terminate", help="Terminate a cluster")
    term_parser.add_argument("--cluster-id", required=True, help="Cluster ID")
    term_parser.set_defaults(func=cmd_terminate)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List active clusters")
    list_parser.set_defaults(func=cmd_list)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

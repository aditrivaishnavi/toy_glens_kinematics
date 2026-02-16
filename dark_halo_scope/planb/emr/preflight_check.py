#!/usr/bin/env python3
"""
EMR Pre-flight Check

Validates environment and configuration before submitting EMR jobs.

Usage:
    python preflight_check.py
    python preflight_check.py --verbose
    python preflight_check.py --fix  # Attempt to fix issues

Lessons Learned:
- L5.4: Validate before expensive operations
- L3.1: Verify cluster readiness
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor} (need 3.8+)"


def check_aws_credentials() -> Tuple[bool, str]:
    """
    Check AWS credentials are configured and valid.
    
    IMPORTANT: AWS credentials expire after 24 hours.
    If expired, STOP and ask user for new credentials.
    """
    try:
        import boto3
    except ImportError:
        return False, "boto3 not installed"
    
    try:
        from botocore.exceptions import NoCredentialsError, ClientError
    except ImportError:
        return False, "botocore not installed"
    
    try:
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        account = identity["Account"]
        
        # Check region
        from constants import AWS_REGION
        
        return True, f"AWS Account: {account}, Region: {AWS_REGION}"
    except NoCredentialsError:
        return False, (
            "No AWS credentials found.\n"
            "        *** AWS credentials expire after 24 hours ***\n"
            "        Ask user for new credentials if expired.\n"
            "        Run: aws configure"
        )
    except ClientError as e:
        if "ExpiredToken" in str(e):
            return False, (
                "AWS credentials EXPIRED.\n"
                "        *** STOP: Ask user for new credentials ***\n"
                "        Credentials are valid for only 24 hours."
            )
        return False, f"AWS credentials error: {e}"
    except Exception as e:
        return False, f"AWS credentials error: {e}"


def check_s3_access(bucket: str = "darkhaloscope") -> Tuple[bool, str]:
    """Check S3 bucket is accessible."""
    try:
        from aws_utils import get_s3_client, is_credential_error
        s3 = get_s3_client()
        s3.head_bucket(Bucket=bucket)
        return True, f"S3 bucket '{bucket}' accessible"
    except ImportError:
        # Fall back to basic boto3 if aws_utils not available
        try:
            import boto3
            s3 = boto3.client("s3")
            s3.head_bucket(Bucket=bucket)
            return True, f"S3 bucket '{bucket}' accessible"
        except Exception as e:
            return False, f"S3 access error: {e}"
    except Exception as e:
        # Don't silently eat - include full error
        error_str = str(e)
        if "ExpiredToken" in error_str or "InvalidClientTokenId" in error_str:
            return False, f"S3 access failed - CREDENTIALS EXPIRED: {e}"
        return False, f"S3 access error: {e}"


def check_emr_access() -> Tuple[bool, str]:
    """Check EMR service is accessible."""
    try:
        from aws_utils import get_emr_client
        emr = get_emr_client()
        clusters = emr.list_clusters(ClusterStates=["WAITING", "RUNNING"])
        count = len(clusters.get("Clusters", []))
        return True, f"EMR accessible ({count} active clusters)"
    except ImportError:
        try:
            import boto3
            emr = boto3.client("emr")
            clusters = emr.list_clusters(ClusterStates=["WAITING", "RUNNING"])
            count = len(clusters.get("Clusters", []))
            return True, f"EMR accessible ({count} active clusters)"
        except Exception as e:
            return False, f"EMR access error: {e}"
    except Exception as e:
        error_str = str(e)
        if "ExpiredToken" in error_str or "InvalidClientTokenId" in error_str:
            return False, f"EMR access failed - CREDENTIALS EXPIRED: {e}"
        return False, f"EMR access error: {e}"


def check_iam_roles() -> Tuple[bool, str]:
    """Check required IAM roles exist."""
    try:
        from aws_utils import get_iam_client
        iam = get_iam_client()
    except ImportError:
        try:
            import boto3
            iam = boto3.client("iam")
        except Exception as e:
            return False, f"IAM client error: {e}"
    
    try:
        required_roles = ["EMR_DefaultRole", "EMR_EC2_DefaultRole"]
        missing = []
        
        for role in required_roles:
            try:
                iam.get_role(RoleName=role)
            except Exception as role_error:
                # Log which role failed and why - don't silently eat
                error_str = str(role_error)
                if "NoSuchEntity" in error_str:
                    missing.append(role)
                elif "ExpiredToken" in error_str or "InvalidClientTokenId" in error_str:
                    return False, f"IAM check failed - CREDENTIALS EXPIRED: {role_error}"
                else:
                    missing.append(f"{role} (error: {role_error})")
        
        if missing:
            return False, f"Missing IAM roles: {missing}"
        return True, "All required IAM roles exist"
    except Exception as e:
        return False, f"IAM check error: {e}"


def check_required_packages() -> Tuple[bool, str]:
    """Check required Python packages are installed."""
    required = ["boto3", "botocore"]
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        return False, f"Missing packages: {missing}"
    return True, "All required packages installed"


def check_local_files() -> Tuple[bool, str]:
    """Check required local files exist."""
    emr_dir = Path(__file__).parent
    
    required_files = [
        "launcher.py",
        "constants.py",
        "bootstrap.sh",
        "spark_job_template.py",
    ]
    
    missing = []
    for f in required_files:
        if not (emr_dir / f).exists():
            missing.append(f)
    
    if missing:
        return False, f"Missing files: {missing}"
    return True, "All required files present"


def check_bootstrap_uploaded(bucket: str = "darkhaloscope") -> Tuple[bool, str]:
    """Check bootstrap.sh is uploaded to S3."""
    try:
        import boto3
        s3 = boto3.client("s3")
        key = "planb/emr/code/bootstrap.sh"
        s3.head_object(Bucket=bucket, Key=key)
        return True, f"Bootstrap script exists at s3://{bucket}/{key}"
    except:
        return False, "Bootstrap script not uploaded to S3"


def check_syntax_all_scripts() -> Tuple[bool, str]:
    """Syntax check all Python scripts in emr/."""
    emr_dir = Path(__file__).parent
    errors = []
    
    for py_file in emr_dir.glob("*.py"):
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(py_file)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            errors.append(f"{py_file.name}: {result.stderr}")
    
    # Also check jobs/ directory
    jobs_dir = emr_dir / "jobs"
    if jobs_dir.exists():
        for py_file in jobs_dir.glob("*.py"):
            result = subprocess.run(
                ["python3", "-m", "py_compile", str(py_file)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                errors.append(f"jobs/{py_file.name}: {result.stderr}")
    
    if errors:
        return False, f"Syntax errors:\n" + "\n".join(errors)
    return True, "All scripts pass syntax check"


def check_vcore_budget() -> Tuple[bool, str]:
    """Verify all presets are within vCore budget."""
    try:
        from constants import PRESETS, MAX_VCORES
        
        lines = [f"Max vCore budget: {MAX_VCORES}"]
        all_ok = True
        
        for name, preset in PRESETS.items():
            total = preset.instance_config.total_vcores()
            ok = total <= MAX_VCORES
            status = "✓" if ok else "✗ EXCEEDS"
            lines.append(f"  {name}: {total} vCores {status}")
            if not ok:
                all_ok = False
        
        if all_ok:
            return True, "\n".join(lines)
        else:
            return False, "\n".join(lines)
    except Exception as e:
        return False, f"vCore check error: {e}"


def fix_bootstrap_upload(bucket: str = "darkhaloscope") -> bool:
    """Upload bootstrap.sh to S3."""
    try:
        import boto3
        s3 = boto3.client("s3")
        bootstrap_path = Path(__file__).parent / "bootstrap.sh"
        
        if not bootstrap_path.exists():
            print("ERROR: bootstrap.sh not found locally")
            return False
        
        s3.upload_file(
            str(bootstrap_path),
            bucket,
            "planb/emr/code/bootstrap.sh",
        )
        print(f"Uploaded bootstrap.sh to s3://{bucket}/planb/emr/code/bootstrap.sh")
        return True
    except Exception as e:
        print(f"ERROR uploading: {e}")
        return False


def fix_create_iam_roles() -> bool:
    """Create default EMR IAM roles."""
    try:
        result = subprocess.run(
            ["aws", "emr", "create-default-roles"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("Created default EMR roles")
            return True
        else:
            print(f"ERROR: {result.stderr}")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def run_all_checks(verbose: bool = False) -> Dict[str, Tuple[bool, str]]:
    """Run all pre-flight checks."""
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("Local Files", check_local_files),
        ("Script Syntax", check_syntax_all_scripts),
        ("vCore Budget", check_vcore_budget),
        ("AWS Credentials", check_aws_credentials),
        ("S3 Access", check_s3_access),
        ("EMR Access", check_emr_access),
        ("IAM Roles", check_iam_roles),
        ("Bootstrap Uploaded", check_bootstrap_uploaded),
    ]
    
    results = {}
    
    print("=" * 60)
    print("EMR PRE-FLIGHT CHECK")
    print("=" * 60)
    print()
    
    for name, check_fn in checks:
        try:
            passed, message = check_fn()
        except Exception as e:
            passed, message = False, f"Error: {e}"
        
        results[name] = (passed, message)
        
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if verbose or not passed:
            print(f"      {message}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="EMR Pre-flight Check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show details for all checks")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix issues")
    args = parser.parse_args()
    
    results = run_all_checks(verbose=args.verbose)
    
    failed = [name for name, (passed, _) in results.items() if not passed]
    
    print()
    
    if failed:
        print(f"✗ {len(failed)} check(s) failed: {failed}")
        
        if args.fix:
            print("\nAttempting to fix issues...")
            
            if "Bootstrap Uploaded" in failed:
                if fix_bootstrap_upload():
                    failed.remove("Bootstrap Uploaded")
            
            if "IAM Roles" in failed:
                if fix_create_iam_roles():
                    failed.remove("IAM Roles")
            
            if failed:
                print(f"\nCould not fix: {failed}")
                sys.exit(1)
            else:
                print("\nAll issues fixed!")
                sys.exit(0)
        
        print("\nRun with --fix to attempt automatic fixes")
        sys.exit(1)
    else:
        print("✓ ALL CHECKS PASSED")
        print("\nReady to submit EMR jobs!")
        sys.exit(0)


if __name__ == "__main__":
    main()

"""
AWS Utility Functions with Proper Retry and Exception Handling

IMPORTANT:
- All boto3 clients use retry configuration
- NO silent exception eating - all errors are logged and re-raised or returned explicitly
- AWS credentials expire after 24 hours - check for ExpiredToken errors
"""
import logging
import sys
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import (
        BotoCoreError,
        ClientError,
        NoCredentialsError,
        CredentialRetrievalError,
        EndpointConnectionError,
    )
except ImportError:
    print("ERROR: boto3 not installed. Run: pip install boto3")
    sys.exit(1)

from constants import AWS_REGION, S3_BUCKET


# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# RETRY CONFIGURATION
# =============================================================================

# Standard retry config for all AWS clients
RETRY_CONFIG = Config(
    retries={
        "max_attempts": 5,
        "mode": "adaptive",  # Adapts to throttling
    },
    connect_timeout=10,
    read_timeout=30,
)

# S3-specific config with more retries for data operations
S3_RETRY_CONFIG = Config(
    retries={
        "max_attempts": 10,
        "mode": "adaptive",
    },
    connect_timeout=10,
    read_timeout=60,
    max_pool_connections=50,
)


# =============================================================================
# CREDENTIAL ERROR HANDLING
# =============================================================================

CREDENTIAL_ERROR_MSG = """
================================================================================
AWS CREDENTIAL ERROR

Your AWS credentials may have expired (they are valid for only 24 hours).

*** STOP: Ask user for new credentials ***

To fix:
1. Get new credentials from your AWS SSO/IAM console
2. Run: aws configure
   OR set environment variables:
   export AWS_ACCESS_KEY_ID=...
   export AWS_SECRET_ACCESS_KEY=...
   export AWS_SESSION_TOKEN=... (if using temporary credentials)

Then retry the operation.
================================================================================
"""


def is_credential_error(error: Exception) -> bool:
    """Check if error is related to credentials/auth."""
    error_str = str(error)
    credential_indicators = [
        "ExpiredToken",
        "InvalidClientTokenId",
        "SignatureDoesNotMatch",
        "InvalidAccessKeyId",
        "AuthFailure",
        "NoCredentialsError",
        "CredentialRetrievalError",
    ]
    return any(indicator in error_str for indicator in credential_indicators)


def handle_aws_error(error: Exception, operation: str) -> None:
    """
    Handle AWS error with proper logging.
    
    NEVER silently eats exceptions - always logs and re-raises.
    """
    if is_credential_error(error):
        logger.error(CREDENTIAL_ERROR_MSG)
        logger.error(f"Operation that failed: {operation}")
        raise SystemExit(1)  # Force stop for credential errors
    
    if isinstance(error, EndpointConnectionError):
        logger.error(f"Network error during {operation}: {error}")
        logger.error("Check your internet connection and AWS region setting.")
        raise
    
    if isinstance(error, ClientError):
        error_code = error.response.get("Error", {}).get("Code", "Unknown")
        error_msg = error.response.get("Error", {}).get("Message", str(error))
        logger.error(f"AWS ClientError during {operation}: [{error_code}] {error_msg}")
        raise
    
    if isinstance(error, BotoCoreError):
        logger.error(f"AWS BotoCoreError during {operation}: {error}")
        raise
    
    # Unknown error - log and re-raise
    logger.error(f"Unexpected error during {operation}: {type(error).__name__}: {error}")
    raise


# =============================================================================
# CLIENT FACTORIES
# =============================================================================

def get_s3_client():
    """
    Get S3 client with retry configuration.
    
    Returns:
        boto3 S3 client with retries configured
    """
    return boto3.client("s3", region_name=AWS_REGION, config=S3_RETRY_CONFIG)


def get_emr_client():
    """
    Get EMR client with retry configuration.
    
    Returns:
        boto3 EMR client with retries configured
    """
    return boto3.client("emr", region_name=AWS_REGION, config=RETRY_CONFIG)


def get_sts_client():
    """
    Get STS client with retry configuration.
    
    Returns:
        boto3 STS client with retries configured
    """
    return boto3.client("sts", region_name=AWS_REGION, config=RETRY_CONFIG)


def get_iam_client():
    """
    Get IAM client with retry configuration.
    
    Returns:
        boto3 IAM client with retries configured
    """
    return boto3.client("iam", region_name=AWS_REGION, config=RETRY_CONFIG)


# =============================================================================
# S3 OPERATIONS WITH RETRY AND EXCEPTION HANDLING
# =============================================================================

def s3_upload_file(
    local_path: str,
    bucket: str,
    key: str,
    max_retries: int = 3,
) -> str:
    """
    Upload file to S3 with retry and proper exception handling.
    
    Args:
        local_path: Local file path
        bucket: S3 bucket name
        key: S3 object key
        max_retries: Maximum retry attempts for transient errors
    
    Returns:
        S3 URI (s3://bucket/key)
    
    Raises:
        All errors are logged and re-raised (never silently eaten)
    """
    s3 = get_s3_client()
    s3_uri = f"s3://{bucket}/{key}"
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Uploading {local_path} to {s3_uri} (attempt {attempt + 1}/{max_retries})")
            s3.upload_file(local_path, bucket, key)
            logger.info(f"Successfully uploaded to {s3_uri}")
            return s3_uri
            
        except (EndpointConnectionError, ClientError) as e:
            # Check for credential errors first
            if is_credential_error(e):
                handle_aws_error(e, f"s3_upload_file({local_path} -> {s3_uri})")
            
            # Retry for transient errors
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Upload failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"Upload failed after {max_retries} attempts: {e}")
                raise
                
        except Exception as e:
            handle_aws_error(e, f"s3_upload_file({local_path} -> {s3_uri})")


def s3_download_file(
    bucket: str,
    key: str,
    local_path: str,
    max_retries: int = 3,
) -> str:
    """
    Download file from S3 with retry and proper exception handling.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        local_path: Local file path to save to
        max_retries: Maximum retry attempts
    
    Returns:
        Local file path
    
    Raises:
        All errors are logged and re-raised (never silently eaten)
    """
    s3 = get_s3_client()
    s3_uri = f"s3://{bucket}/{key}"
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {s3_uri} to {local_path} (attempt {attempt + 1}/{max_retries})")
            s3.download_file(bucket, key, local_path)
            logger.info(f"Successfully downloaded to {local_path}")
            return local_path
            
        except (EndpointConnectionError, ClientError) as e:
            if is_credential_error(e):
                handle_aws_error(e, f"s3_download_file({s3_uri} -> {local_path})")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Download failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"Download failed after {max_retries} attempts: {e}")
                raise
                
        except Exception as e:
            handle_aws_error(e, f"s3_download_file({s3_uri} -> {local_path})")


def s3_head_object(bucket: str, key: str) -> Optional[dict]:
    """
    Check if S3 object exists and get metadata.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
    
    Returns:
        Object metadata dict, or None if not found
    
    Raises:
        All errors except 404 are logged and re-raised
    """
    s3 = get_s3_client()
    s3_uri = f"s3://{bucket}/{key}"
    
    try:
        response = s3.head_object(Bucket=bucket, Key=key)
        return response
        
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "404" or error_code == "NoSuchKey":
            logger.debug(f"Object not found: {s3_uri}")
            return None
        handle_aws_error(e, f"s3_head_object({s3_uri})")
        
    except Exception as e:
        handle_aws_error(e, f"s3_head_object({s3_uri})")


def s3_head_bucket(bucket: str) -> bool:
    """
    Check if S3 bucket exists and is accessible.
    
    Args:
        bucket: S3 bucket name
    
    Returns:
        True if accessible, False if not found
    
    Raises:
        Permission and credential errors are logged and re-raised
    """
    s3 = get_s3_client()
    
    try:
        s3.head_bucket(Bucket=bucket)
        return True
        
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "404" or error_code == "NoSuchBucket":
            logger.warning(f"Bucket not found: {bucket}")
            return False
        # Permission and credential errors should be raised
        handle_aws_error(e, f"s3_head_bucket({bucket})")
        
    except Exception as e:
        handle_aws_error(e, f"s3_head_bucket({bucket})")


def s3_list_objects(
    bucket: str,
    prefix: str,
    max_keys: int = 1000,
) -> list:
    """
    List objects in S3 with proper error handling.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 key prefix
        max_keys: Maximum number of keys to return
    
    Returns:
        List of object keys
    
    Raises:
        All errors are logged and re-raised
    """
    s3 = get_s3_client()
    
    try:
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=max_keys,
        )
        
        contents = response.get("Contents", [])
        return [obj["Key"] for obj in contents]
        
    except Exception as e:
        handle_aws_error(e, f"s3_list_objects({bucket}/{prefix})")


# =============================================================================
# CREDENTIAL VALIDATION
# =============================================================================

def validate_credentials() -> bool:
    """
    Validate AWS credentials are valid and not expired.
    
    Returns:
        True if credentials are valid
    
    Raises:
        SystemExit if credentials are invalid/expired (STOP and ask user)
    """
    try:
        sts = get_sts_client()
        identity = sts.get_caller_identity()
        logger.info(f"AWS credentials valid - Account: {identity['Account']}, Region: {AWS_REGION}")
        return True
        
    except NoCredentialsError:
        logger.error(CREDENTIAL_ERROR_MSG)
        raise SystemExit(1)
        
    except ClientError as e:
        if is_credential_error(e):
            logger.error(CREDENTIAL_ERROR_MSG)
            raise SystemExit(1)
        handle_aws_error(e, "validate_credentials")
        
    except Exception as e:
        handle_aws_error(e, "validate_credentials")

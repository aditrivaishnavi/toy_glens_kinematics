#!/usr/bin/env python3
"""
Validate that constants.py S3 paths match the actual S3 structure.

Run this script to verify all S3 prefixes exist before making changes.

Usage:
    python scripts/validate_constants.py
"""
import sys
from pathlib import Path

# Add parent to path for import
sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import (
    S3_BUCKET,
    AWS_REGION,
    # Global paths
    S3_DR10_PREFIX,
    S3_DR10_SWEEPS_PREFIX,
    S3_DR10_COADD_CACHE_PREFIX,
    S3_SWEEP_FITS_DUMP_PREFIX,
    S3_COSMOS_BANKS_PREFIX,
    S3_MODELS_PREFIX,
    S3_PLANB_PREFIX,
    # Stronglens paths
    S3_BASE_PREFIX,
    S3_CODE_PREFIX,
    S3_LOGS_PREFIX,
    S3_CONFIG_PREFIX,
    S3_POSITIVES_PREFIX,
    S3_MANIFESTS_PREFIX,
    S3_SAMPLED_NEGATIVES_PREFIX,
    S3_CUTOUTS_PREFIX,
    S3_CUTOUTS_POSITIVES_PREFIX,
    S3_CUTOUTS_NEGATIVES_PREFIX,
    S3_VALIDATION_PREFIX,
)

import boto3


def check_prefix_exists(s3, bucket: str, prefix: str) -> dict:
    """Check if an S3 prefix has any objects."""
    try:
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=1
        )
        exists = response.get("KeyCount", 0) > 0
        return {"exists": exists, "error": None}
    except Exception as e:
        return {"exists": False, "error": str(e)}


def main():
    print("=" * 60)
    print("Validating constants.py S3 paths")
    print("=" * 60)
    print(f"Bucket: {S3_BUCKET}")
    print(f"Region: {AWS_REGION}")
    print()
    
    s3 = boto3.client("s3", region_name=AWS_REGION)
    
    # Define prefixes to check with descriptions
    prefixes = [
        # Global paths
        ("DR10 sweeps", S3_DR10_SWEEPS_PREFIX),
        ("DR10 coadd cache", S3_DR10_COADD_CACHE_PREFIX),
        ("Sweep FITS dump", S3_SWEEP_FITS_DUMP_PREFIX),
        ("Cosmos banks", S3_COSMOS_BANKS_PREFIX),
        ("Models", S3_MODELS_PREFIX),
        ("PlanB", S3_PLANB_PREFIX),
        # Stronglens paths
        ("EMR code", S3_CODE_PREFIX),
        ("EMR logs", S3_LOGS_PREFIX),
        ("Configs", S3_CONFIG_PREFIX),
        ("Positives with DR10", S3_POSITIVES_PREFIX),
        ("Manifests", S3_MANIFESTS_PREFIX),
        ("Sampled negatives", S3_SAMPLED_NEGATIVES_PREFIX),
        ("Cutouts positives", S3_CUTOUTS_POSITIVES_PREFIX),
        ("Cutouts negatives", S3_CUTOUTS_NEGATIVES_PREFIX),
        ("Validation", S3_VALIDATION_PREFIX),
    ]
    
    all_ok = True
    
    for name, prefix in prefixes:
        result = check_prefix_exists(s3, S3_BUCKET, prefix)
        status = "✓" if result["exists"] else "✗"
        
        if result["error"]:
            print(f"  {status} {name}: {prefix} (ERROR: {result['error']})")
            all_ok = False
        elif result["exists"]:
            print(f"  {status} {name}: {prefix}")
        else:
            # Not an error if it doesn't exist yet (e.g., validation directory)
            print(f"  - {name}: {prefix} (empty/not created yet)")
    
    print()
    print("=" * 60)
    if all_ok:
        print("All paths validated successfully!")
    else:
        print("Some paths have issues - check above.")
    print("=" * 60)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

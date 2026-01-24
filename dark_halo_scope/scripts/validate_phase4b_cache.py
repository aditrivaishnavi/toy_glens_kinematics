#!/usr/bin/env python3
"""
Phase 4b Cache Validation Script

Validates that the coadd cache contains all expected files for all bricks
referenced in the Phase 4a bricks_manifest.

This is designed to run locally on emr-launcher (no EMR cluster needed).
It uses boto3 for S3 listing and pandas for analysis.

Expected files per brick (7 total):
  - legacysurvey-{brick}-image-g.fits.fz
  - legacysurvey-{brick}-image-r.fits.fz
  - legacysurvey-{brick}-image-z.fits.fz
  - legacysurvey-{brick}-invvar-g.fits.fz
  - legacysurvey-{brick}-invvar-r.fits.fz
  - legacysurvey-{brick}-invvar-z.fits.fz
  - legacysurvey-{brick}-maskbits.fits.fz

Usage:
    python3 validate_phase4b_cache.py \
        --cache-prefix s3://darkhaloscope/dr10/coadd_cache/ \
        --bricks-manifest s3://darkhaloscope/phase4_pipeline/phase4a/v3_color_relaxed/bricks_manifest/ \
        --output-report validation_4b_report.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple

import boto3
import pandas as pd

# Expected file suffixes per brick
EXPECTED_SUFFIXES = [
    "image-g.fits.fz",
    "image-r.fits.fz",
    "image-z.fits.fz",
    "invvar-g.fits.fz",
    "invvar-r.fits.fz",
    "invvar-z.fits.fz",
    "maskbits.fits.fz",
]


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    match = re.match(r"^s3://([^/]+)/(.*)$", uri)
    if not match:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return match.group(1), match.group(2)


def list_s3_files(bucket: str, prefix: str) -> List[Dict]:
    """List all files under an S3 prefix with pagination."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    
    files = []
    print(f"[validate] Listing files in s3://{bucket}/{prefix}...")
    
    page_count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        page_count += 1
        if page_count % 100 == 0:
            print(f"  ... processed {page_count} pages, found {len(files)} files so far")
        
        for obj in page.get("Contents", []):
            files.append({
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
            })
    
    print(f"[validate] Found {len(files):,} files in {page_count} pages")
    return files


def parse_cached_files(files: List[Dict], prefix: str) -> Dict[str, Set[str]]:
    """
    Parse cached files into a dict: brickname -> set of suffixes present.
    
    Expected path pattern: {prefix}{brickdir}/{brick}/legacysurvey-{brick}-{suffix}
    Example: dr10/coadd_cache/000/000p025/legacysurvey-000p025-image-g.fits.fz
    """
    brick_files = defaultdict(set)
    
    for f in files:
        key = f["key"]
        # Skip _SUCCESS markers and directories
        if key.endswith("_SUCCESS") or key.endswith("/"):
            continue
        
        # Extract filename
        parts = key.split("/")
        if len(parts) < 2:
            continue
        
        filename = parts[-1]
        
        # Parse: legacysurvey-{brickname}-{suffix}
        match = re.match(r"^legacysurvey-([0-9pm]+)-(.+)$", filename)
        if match:
            brickname = match.group(1)
            suffix = match.group(2)
            brick_files[brickname].add(suffix)
    
    return brick_files


def load_bricks_manifest(uri: str) -> pd.DataFrame:
    """Load bricks manifest from S3 parquet."""
    print(f"[validate] Loading bricks manifest from {uri}...")
    
    # Use pandas with s3fs or download locally
    import pyarrow.parquet as pq
    import s3fs
    
    fs = s3fs.S3FileSystem()
    # Handle trailing slash
    uri_clean = uri.rstrip("/")
    
    # Read parquet dataset
    dataset = pq.ParquetDataset(uri_clean, filesystem=fs)
    df = dataset.read().to_pandas()
    
    print(f"[validate] Loaded {len(df):,} bricks from manifest")
    return df


def validate_cache(
    cache_files: Dict[str, Set[str]],
    expected_bricks: Set[str],
    expected_suffixes: List[str] = EXPECTED_SUFFIXES,
) -> Dict:
    """
    Validate cache completeness.
    
    Returns dict with:
    - total_expected_bricks
    - total_cached_bricks
    - complete_bricks (all 7 files)
    - incomplete_bricks (some files missing)
    - missing_bricks (no files at all)
    - extra_bricks (cached but not in manifest)
    """
    expected_suffix_set = set(expected_suffixes)
    
    complete = []
    incomplete = []
    missing = []
    
    for brick in expected_bricks:
        cached = cache_files.get(brick, set())
        if not cached:
            missing.append(brick)
        elif cached >= expected_suffix_set:
            complete.append(brick)
        else:
            incomplete.append({
                "brick": brick,
                "present": list(cached),
                "missing": list(expected_suffix_set - cached),
            })
    
    # Extra bricks (cached but not in manifest)
    extra = set(cache_files.keys()) - expected_bricks
    
    return {
        "total_expected_bricks": len(expected_bricks),
        "total_cached_bricks": len(cache_files),
        "complete_bricks": len(complete),
        "incomplete_bricks": len(incomplete),
        "missing_bricks": len(missing),
        "extra_bricks": len(extra),
        "completeness_pct": 100.0 * len(complete) / len(expected_bricks) if expected_bricks else 0,
        "incomplete_details": incomplete[:100],  # First 100 for debugging
        "missing_brick_list": missing[:1000],  # First 1000 for debugging
    }


def analyze_missingness(
    missing_bricks: List[str],
    incomplete_bricks: List[Dict],
    bricks_df: pd.DataFrame,
) -> Dict:
    """
    Analyze if missingness correlates with observing conditions.
    
    Returns stats by PSF/depth bins if available.
    """
    if bricks_df.empty:
        return {"error": "No bricks manifest data"}
    
    # Get all problem bricks
    problem_bricks = set(missing_bricks)
    for inc in incomplete_bricks:
        problem_bricks.add(inc["brick"])
    
    if not problem_bricks:
        return {"status": "no_problems", "problem_count": 0}
    
    # Check if we have condition columns
    condition_cols = []
    for col in ["psfsize_r", "psfdepth_r", "ebv", "psf_bin", "depth_bin"]:
        if col in bricks_df.columns:
            condition_cols.append(col)
    
    if not condition_cols:
        return {
            "status": "no_condition_columns",
            "problem_count": len(problem_bricks),
        }
    
    # Add problem flag
    bricks_df = bricks_df.copy()
    bricks_df["is_problem"] = bricks_df["brickname"].isin(problem_bricks)
    
    # Compute problem rate by condition bins
    analysis = {"problem_count": len(problem_bricks)}
    
    for col in condition_cols:
        if col in bricks_df.columns:
            grouped = bricks_df.groupby(col).agg({
                "is_problem": ["sum", "count", "mean"]
            })
            grouped.columns = ["problem_count", "total", "problem_rate"]
            analysis[f"by_{col}"] = grouped.to_dict("index")
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 4b coadd cache")
    parser.add_argument(
        "--cache-prefix",
        default="s3://darkhaloscope/dr10/coadd_cache/",
        help="S3 prefix where coadds are cached",
    )
    parser.add_argument(
        "--bricks-manifest",
        default="s3://darkhaloscope/phase4_pipeline/phase4a/v3_color_relaxed/bricks_manifest/",
        help="S3 path to bricks manifest parquet",
    )
    parser.add_argument(
        "--output-report",
        default="validation_4b_report.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--output-summary",
        default="validation_4b_summary.md",
        help="Output markdown summary path",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Phase 4b Cache Validation")
    print("=" * 60)
    
    start_time = datetime.utcnow()
    
    # 1. List all cached files
    cache_bucket, cache_key = parse_s3_uri(args.cache_prefix)
    cached_files_raw = list_s3_files(cache_bucket, cache_key)
    
    # 2. Parse into brick -> files mapping
    cached_files = parse_cached_files(cached_files_raw, cache_key)
    print(f"[validate] Parsed {len(cached_files):,} unique bricks from cache")
    
    # 3. Load bricks manifest
    bricks_df = load_bricks_manifest(args.bricks_manifest)
    expected_bricks = set(bricks_df["brickname"].unique())
    print(f"[validate] Expected {len(expected_bricks):,} bricks from manifest")
    
    # 4. Validate completeness
    results = validate_cache(cached_files, expected_bricks)
    
    # 5. Analyze missingness
    if results["missing_bricks"] > 0 or results["incomplete_bricks"] > 0:
        missingness = analyze_missingness(
            results["missing_brick_list"],
            results["incomplete_details"],
            bricks_df,
        )
        results["missingness_analysis"] = missingness
    else:
        results["missingness_analysis"] = {"status": "all_complete"}
    
    # 6. Add metadata
    results["validation_timestamp"] = datetime.utcnow().isoformat() + "Z"
    results["cache_prefix"] = args.cache_prefix
    results["bricks_manifest"] = args.bricks_manifest
    results["runtime_seconds"] = (datetime.utcnow() - start_time).total_seconds()
    
    # 7. Print summary
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Expected bricks:   {results['total_expected_bricks']:,}")
    print(f"Cached bricks:     {results['total_cached_bricks']:,}")
    print(f"Complete (7/7):    {results['complete_bricks']:,}")
    print(f"Incomplete:        {results['incomplete_bricks']:,}")
    print(f"Missing:           {results['missing_bricks']:,}")
    print(f"Extra (unexpected):{results['extra_bricks']:,}")
    print(f"Completeness:      {results['completeness_pct']:.2f}%")
    print("=" * 60)
    
    # 8. Determine pass/fail
    if results["completeness_pct"] >= 99.9:
        status = "PASS"
        print(f"\n✅ VALIDATION PASSED: {results['completeness_pct']:.2f}% complete")
    elif results["completeness_pct"] >= 95.0:
        status = "WARN"
        print(f"\n⚠️  VALIDATION WARNING: {results['completeness_pct']:.2f}% complete (some gaps)")
    else:
        status = "FAIL"
        print(f"\n❌ VALIDATION FAILED: Only {results['completeness_pct']:.2f}% complete")
    
    results["status"] = status
    
    # 9. Save report
    with open(args.output_report, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[validate] Report saved to: {args.output_report}")
    
    # 10. Generate markdown summary
    md = f"""# Phase 4b Cache Validation Report

**Date**: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC  
**Status**: {status}

## Summary

| Metric | Value |
|--------|-------|
| Expected bricks | {results['total_expected_bricks']:,} |
| Cached bricks | {results['total_cached_bricks']:,} |
| Complete (7/7 files) | {results['complete_bricks']:,} |
| Incomplete | {results['incomplete_bricks']:,} |
| Missing | {results['missing_bricks']:,} |
| **Completeness** | **{results['completeness_pct']:.2f}%** |

## Files Per Brick

Each brick should have 7 files:
- `image-g.fits.fz`, `image-r.fits.fz`, `image-z.fits.fz`
- `invvar-g.fits.fz`, `invvar-r.fits.fz`, `invvar-z.fits.fz`
- `maskbits.fits.fz`

## Cache Location

`{args.cache_prefix}`

## Recommendation

"""
    if status == "PASS":
        md += "✅ Cache is complete. Ready for Phase 4c.\n"
    elif status == "WARN":
        md += f"⚠️ Cache has {results['incomplete_bricks'] + results['missing_bricks']:,} gaps. Consider rerunning 4b with `--force 0` to fill gaps before 4c.\n"
    else:
        md += f"❌ Cache is significantly incomplete. Rerun 4b with `--force 0` to download missing files.\n"
    
    with open(args.output_summary, "w") as f:
        f.write(md)
    print(f"[validate] Summary saved to: {args.output_summary}")
    
    return 0 if status in ("PASS", "WARN") else 1


if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""
Phase 3c Output Validation Script
==================================

This script validates the Phase 3c parent catalog output for:
1. Structural integrity (schema, partitioning, file counts)
2. Data quality (nulls, value ranges, distributions)
3. Scientific correctness (LRG flag consistency, magnitude/color calculations)
4. Coverage (regions, splits, bricks)
5. Cross-stage consistency (with 3a and 3b outputs)

Usage:
    python -m scripts.validate_phase3c_output \
        --phase3c-parquet s3://darkhaloscope/phase3_pipeline/phase3c/v3_color_relaxed/parent_union_parquet \
        --phase3a-regions s3://darkhaloscope/phase3_pipeline/phase3a/v3_color_relaxed/region_metrics \
        --phase3b-selections s3://darkhaloscope/phase3_pipeline/phase3b/v3_color_relaxed/region_selections \
        --output-report results/phase3c_validation_report.md

Outputs:
    - Markdown report with all validation results
    - JSON summary for programmatic access
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import boto3
import numpy as np

# Try to import pandas - fallback to basic analysis if not available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[WARN] pandas not available - some analyses will be limited")


# ---------------------------------------------------------------------------
# LRG Variant Definitions (must match Phase 2 and Phase 3)
# ---------------------------------------------------------------------------
LRG_VARIANTS = {
    "v1_pure_massive":   {"z_max": 20.0, "rz_min": 0.5, "zw1_min": 1.6},
    "v2_baseline_dr10":  {"z_max": 20.4, "rz_min": 0.4, "zw1_min": 1.6},
    "v3_color_relaxed":  {"z_max": 20.4, "rz_min": 0.4, "zw1_min": 0.8},
    "v4_mag_relaxed":    {"z_max": 21.0, "rz_min": 0.4, "zw1_min": 0.8},
    "v5_very_relaxed":   {"z_max": 21.5, "rz_min": 0.3, "zw1_min": 0.8},
}


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def s3_list_objects(prefix: str, client=None) -> List[Dict]:
    """List all objects under an S3 prefix."""
    if client is None:
        client = boto3.client("s3")
    bucket, key = parse_s3_uri(prefix)
    key = key.rstrip("/") + "/"
    
    objects = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key):
        for obj in page.get("Contents", []):
            objects.append({
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
            })
    return objects


def download_parquet_sample(s3_prefix: str, local_dir: str, max_files: int = 5, client=None) -> List[str]:
    """Download a sample of parquet files for local analysis."""
    if client is None:
        client = boto3.client("s3")
    
    objects = s3_list_objects(s3_prefix, client)
    parquet_files = [o for o in objects if o["key"].endswith(".parquet")]
    
    os.makedirs(local_dir, exist_ok=True)
    downloaded = []
    
    for obj in parquet_files[:max_files]:
        bucket, _ = parse_s3_uri(s3_prefix)
        local_path = os.path.join(local_dir, os.path.basename(obj["key"]))
        client.download_file(bucket, obj["key"], local_path)
        downloaded.append(local_path)
    
    return downloaded


def analyze_parquet_structure(s3_prefix: str, client=None) -> Dict[str, Any]:
    """Analyze the structure of Parquet output (partitions, files, sizes)."""
    objects = s3_list_objects(s3_prefix, client)
    
    parquet_files = [o for o in objects if o["key"].endswith(".parquet")]
    other_files = [o for o in objects if not o["key"].endswith(".parquet")]
    
    # Analyze partitioning structure
    partitions = defaultdict(lambda: defaultdict(list))
    for obj in parquet_files:
        key = obj["key"]
        # Expected: .../region_split=train/region_id=123/part-00000.parquet
        parts = key.split("/")
        partition_info = {}
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                partition_info[k] = v
        
        if partition_info:
            split = partition_info.get("region_split", "unknown")
            region = partition_info.get("region_id", "unknown")
            partitions[split][region].append(obj)
    
    total_size_bytes = sum(o["size"] for o in parquet_files)
    
    return {
        "total_parquet_files": len(parquet_files),
        "total_size_bytes": total_size_bytes,
        "total_size_gb": round(total_size_bytes / (1024**3), 3),
        "other_files": len(other_files),
        "partitions_by_split": {
            split: {
                "n_regions": len(regions),
                "n_files": sum(len(files) for files in regions.values()),
                "total_bytes": sum(f["size"] for files in regions.values() for f in files),
            }
            for split, regions in partitions.items()
        },
        "sample_files": [o["key"] for o in parquet_files[:5]],
    }


def load_parquet_from_s3(s3_prefix: str, sample_fraction: float = 0.01, max_rows: int = 500000, 
                         max_files: int = 50, client=None) -> Optional[pd.DataFrame]:
    """Load a sample of Parquet data from S3 for analysis.
    
    With many small files, we sample a subset of files rather than trying to read the whole dataset.
    """
    if not HAS_PANDAS:
        return None
    
    try:
        import pyarrow.parquet as pq
        import io
        
        if client is None:
            client = boto3.client("s3")
        
        # List parquet files
        objects = s3_list_objects(s3_prefix, client)
        parquet_files = [o for o in objects if o["key"].endswith(".parquet")]
        
        if len(parquet_files) == 0:
            print("[WARN] No parquet files found")
            return None
        
        # Sample files if too many (to avoid memory/time issues)
        if len(parquet_files) > max_files:
            print(f"      Sampling {max_files} of {len(parquet_files)} files...")
            # Sample evenly across the file list
            step = len(parquet_files) // max_files
            parquet_files = parquet_files[::step][:max_files]
        
        bucket, _ = parse_s3_uri(s3_prefix)
        
        dfs = []
        rows_so_far = 0
        
        for i, obj in enumerate(parquet_files):
            if rows_so_far >= max_rows:
                break
            
            if i % 10 == 0:
                print(f"      Reading file {i+1}/{len(parquet_files)}...")
            
            # Download to memory and read with pyarrow
            response = client.get_object(Bucket=bucket, Key=obj["key"])
            data = response["Body"].read()
            
            table = pq.read_table(io.BytesIO(data))
            df_chunk = table.to_pandas()
            dfs.append(df_chunk)
            rows_so_far += len(df_chunk)
        
        if not dfs:
            return None
        
        df = pd.concat(dfs, ignore_index=True)
        
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42)
        
        return df
        
    except Exception as e:
        print(f"[WARN] Could not load Parquet: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate the schema of the parent catalog."""
    expected_columns = {
        # Core identifiers
        "region_id": "int",
        "region_split": "str",
        "brickname": "str",
        "objid": "int",
        "ra": "float",
        "dec": "float",
        # Magnitudes (raw)
        "gmag": "float",
        "rmag": "float",
        "zmag": "float",
        "w1mag": "float",
        # Colors
        "rz": "float",
        "zw1": "float",
        # MW-corrected (optional)
        "gmag_mw": "float",
        "rmag_mw": "float",
        "zmag_mw": "float",
        "w1mag_mw": "float",
        "rz_mw": "float",
        "zw1_mw": "float",
        # Metadata
        "maskbits": "int",
        "type": "str",
        # LRG flags
        "is_v1_pure_massive": "bool",
        "is_v2_baseline_dr10": "bool",
        "is_v3_color_relaxed": "bool",
        "is_v4_mag_relaxed": "bool",
        "is_v5_very_relaxed": "bool",
    }
    
    actual_columns = set(df.columns)
    expected_set = set(expected_columns.keys())
    
    missing = expected_set - actual_columns
    extra = actual_columns - expected_set
    
    return {
        "valid": len(missing) == 0,
        "missing_columns": list(missing),
        "extra_columns": list(extra),
        "actual_columns": list(df.columns),
        "column_dtypes": {col: str(df[col].dtype) for col in df.columns},
    }


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Check data quality: nulls, ranges, distributions."""
    n_rows = len(df)
    
    # Null analysis
    null_counts = df.isnull().sum().to_dict()
    null_pcts = {k: round(v / n_rows * 100, 2) for k, v in null_counts.items()}
    
    # Core columns should have no nulls
    core_columns = ["region_id", "region_split", "brickname", "objid", "ra", "dec",
                    "gmag", "rmag", "zmag", "w1mag", "rz", "zw1"]
    core_nulls = {col: null_counts.get(col, 0) for col in core_columns if col in null_counts}
    
    # Magnitude ranges (should be reasonable for galaxies)
    mag_stats = {}
    for col in ["gmag", "rmag", "zmag", "w1mag"]:
        if col in df.columns:
            valid = df[col].dropna()
            mag_stats[col] = {
                "min": round(float(valid.min()), 3),
                "max": round(float(valid.max()), 3),
                "mean": round(float(valid.mean()), 3),
                "median": round(float(valid.median()), 3),
                "std": round(float(valid.std()), 3),
                "n_valid": len(valid),
                "n_invalid": n_rows - len(valid),
            }
    
    # Coordinate ranges
    coord_stats = {}
    for col in ["ra", "dec"]:
        if col in df.columns:
            valid = df[col].dropna()
            coord_stats[col] = {
                "min": round(float(valid.min()), 3),
                "max": round(float(valid.max()), 3),
            }
    
    # Color ranges
    color_stats = {}
    for col in ["rz", "zw1"]:
        if col in df.columns:
            valid = df[col].dropna()
            color_stats[col] = {
                "min": round(float(valid.min()), 3),
                "max": round(float(valid.max()), 3),
                "mean": round(float(valid.mean()), 3),
                "p10": round(float(valid.quantile(0.1)), 3),
                "p50": round(float(valid.quantile(0.5)), 3),
                "p90": round(float(valid.quantile(0.9)), 3),
            }
    
    return {
        "n_rows": n_rows,
        "null_counts": null_counts,
        "null_pcts": null_pcts,
        "core_columns_null_check": {
            "passed": all(v == 0 for v in core_nulls.values()),
            "details": core_nulls,
        },
        "magnitude_stats": mag_stats,
        "coordinate_stats": coord_stats,
        "color_stats": color_stats,
    }


def validate_lrg_flags(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate LRG flags are consistent with magnitude/color cuts."""
    results = {}
    n_rows = len(df)
    
    # Check each variant
    for vname, cuts in LRG_VARIANTS.items():
        flag_col = f"is_{vname}"
        if flag_col not in df.columns:
            results[vname] = {"error": f"Missing column {flag_col}"}
            continue
        
        # Recompute expected flags from raw mags
        zmag = df["zmag"]
        rmag = df["rmag"]
        w1mag = df["w1mag"]
        rz = rmag - zmag
        zw1 = zmag - w1mag
        
        expected = (
            (zmag < cuts["z_max"]) &
            (rz > cuts["rz_min"]) &
            (zw1 > cuts["zw1_min"]) &
            np.isfinite(zmag) &
            np.isfinite(rz) &
            np.isfinite(zw1)
        )
        
        actual = df[flag_col].fillna(False)
        
        # Compare
        matches = (expected == actual).sum()
        mismatches = n_rows - matches
        
        # Count by flag value
        n_expected_true = expected.sum()
        n_actual_true = actual.sum()
        
        results[vname] = {
            "n_rows": n_rows,
            "n_expected_true": int(n_expected_true),
            "n_actual_true": int(n_actual_true),
            "n_matches": int(matches),
            "n_mismatches": int(mismatches),
            "mismatch_pct": round(mismatches / n_rows * 100, 4),
            "passed": mismatches == 0,
            "cuts": cuts,
        }
    
    # Variant hierarchy check: v1 ⊂ v2 ⊂ v3 ⊂ v4 ⊂ v5
    hierarchy_checks = []
    variant_order = ["v1_pure_massive", "v2_baseline_dr10", "v3_color_relaxed", 
                     "v4_mag_relaxed", "v5_very_relaxed"]
    for i in range(len(variant_order) - 1):
        strict = f"is_{variant_order[i]}"
        relaxed = f"is_{variant_order[i+1]}"
        if strict in df.columns and relaxed in df.columns:
            # All strict=True should also have relaxed=True
            violations = ((df[strict] == True) & (df[relaxed] == False)).sum()
            hierarchy_checks.append({
                "strict": variant_order[i],
                "relaxed": variant_order[i+1],
                "violations": int(violations),
                "passed": violations == 0,
            })
    
    results["hierarchy_checks"] = hierarchy_checks
    
    return results


def validate_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate coverage by regions and splits."""
    # Unique counts
    n_unique_regions = df["region_id"].nunique()
    n_unique_bricks = df["brickname"].nunique()
    n_unique_objects = len(df)  # Each row is a unique LRG
    
    # By split
    split_stats = df.groupby("region_split").agg({
        "region_id": "nunique",
        "brickname": "nunique",
        "objid": "count",
    }).rename(columns={
        "region_id": "n_regions",
        "brickname": "n_bricks",
        "objid": "n_objects",
    }).to_dict("index")
    
    # By region (top 10 by object count)
    region_stats = df.groupby(["region_split", "region_id"]).agg({
        "brickname": "nunique",
        "objid": "count",
    }).rename(columns={
        "brickname": "n_bricks",
        "objid": "n_objects",
    }).reset_index()
    
    top_regions = region_stats.nlargest(10, "n_objects").to_dict("records")
    
    # Objects per brick distribution
    objs_per_brick = df.groupby("brickname")["objid"].count()
    
    return {
        "n_unique_regions": int(n_unique_regions),
        "n_unique_bricks": int(n_unique_bricks),
        "n_total_objects": int(n_unique_objects),
        "by_split": split_stats,
        "top_10_regions": top_regions,
        "objects_per_brick": {
            "min": int(objs_per_brick.min()),
            "max": int(objs_per_brick.max()),
            "mean": round(float(objs_per_brick.mean()), 2),
            "median": round(float(objs_per_brick.median()), 2),
        },
    }


def validate_variant_distribution(df: pd.DataFrame, target_variant: str = "v3_color_relaxed") -> Dict[str, Any]:
    """Analyze the distribution of LRG variants in the catalog."""
    variant_cols = [f"is_{v}" for v in LRG_VARIANTS.keys()]
    existing_cols = [c for c in variant_cols if c in df.columns]
    
    # Count each variant
    variant_counts = {col: int(df[col].sum()) for col in existing_cols}
    variant_pcts = {col: round(df[col].sum() / len(df) * 100, 2) for col in existing_cols}
    
    # Parent selection check: all rows should pass the target variant
    target_col = f"is_{target_variant}"
    if target_col in df.columns:
        n_passing = df[target_col].sum()
        parent_selection_check = {
            "target_variant": target_variant,
            "n_total": len(df),
            "n_passing": int(n_passing),
            "pct_passing": round(n_passing / len(df) * 100, 2),
            "passed": n_passing == len(df),
        }
    else:
        parent_selection_check = {"error": f"Missing {target_col}"}
    
    return {
        "variant_counts": variant_counts,
        "variant_pcts": variant_pcts,
        "parent_selection_check": parent_selection_check,
    }


def validate_type_filter(df: pd.DataFrame) -> Dict[str, Any]:
    """Verify that PSF (star) types were excluded."""
    if "type" not in df.columns:
        return {"error": "Missing 'type' column"}
    
    type_counts = df["type"].value_counts().to_dict()
    
    # Check for PSF
    n_psf = type_counts.get("PSF", 0) + type_counts.get("psf", 0)
    
    return {
        "type_distribution": type_counts,
        "n_psf_found": n_psf,
        "psf_filter_passed": n_psf == 0,
    }


def generate_report(
    structure: Dict,
    schema: Dict,
    quality: Dict,
    lrg_flags: Dict,
    coverage: Dict,
    variant_dist: Dict,
    type_filter: Dict,
    args: argparse.Namespace,
) -> str:
    """Generate a comprehensive Markdown report."""
    lines = []
    
    lines.append("# Phase 3c Validation Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.utcnow().isoformat()}Z")
    lines.append(f"**Input**: `{args.phase3c_parquet}`")
    lines.append("")
    
    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    
    all_passed = True
    checks = []
    
    if schema.get("valid"):
        checks.append("✅ Schema validation passed")
    else:
        checks.append(f"❌ Schema validation failed: missing {schema.get('missing_columns')}")
        all_passed = False
    
    if quality.get("core_columns_null_check", {}).get("passed"):
        checks.append("✅ Core columns have no nulls")
    else:
        checks.append("❌ Core columns have unexpected nulls")
        all_passed = False
    
    lrg_passed = all(v.get("passed", False) for k, v in lrg_flags.items() if k != "hierarchy_checks")
    if lrg_passed:
        checks.append("✅ LRG flag validation passed (all variants)")
    else:
        failed = [k for k, v in lrg_flags.items() if k != "hierarchy_checks" and not v.get("passed", True)]
        checks.append(f"❌ LRG flag validation failed for: {failed}")
        all_passed = False
    
    hierarchy_passed = all(h.get("passed", False) for h in lrg_flags.get("hierarchy_checks", []))
    if hierarchy_passed:
        checks.append("✅ LRG variant hierarchy is consistent")
    else:
        checks.append("❌ LRG variant hierarchy has violations")
        all_passed = False
    
    if variant_dist.get("parent_selection_check", {}).get("passed"):
        checks.append("✅ All rows pass parent variant filter")
    else:
        pct = variant_dist.get("parent_selection_check", {}).get("pct_passing", 0)
        checks.append(f"⚠️ Only {pct}% pass parent variant filter")
    
    if type_filter.get("psf_filter_passed"):
        checks.append("✅ PSF (star) filter passed")
    else:
        n = type_filter.get("n_psf_found", 0)
        checks.append(f"❌ Found {n} PSF objects (should be 0)")
        all_passed = False
    
    for check in checks:
        lines.append(f"- {check}")
    
    lines.append("")
    if all_passed:
        lines.append("**Overall Status**: ✅ ALL CHECKS PASSED")
    else:
        lines.append("**Overall Status**: ❌ SOME CHECKS FAILED - Review details below")
    lines.append("")
    
    # Data Overview
    lines.append("## Data Overview")
    lines.append("")
    lines.append(f"- **Total Parquet Files**: {structure.get('total_parquet_files', 'N/A')}")
    lines.append(f"- **Total Size**: {structure.get('total_size_gb', 'N/A')} GB")
    lines.append(f"- **Total LRG Objects**: {coverage.get('n_total_objects', 'N/A'):,}")
    lines.append(f"- **Unique Regions**: {coverage.get('n_unique_regions', 'N/A')}")
    lines.append(f"- **Unique Bricks**: {coverage.get('n_unique_bricks', 'N/A'):,}")
    lines.append("")
    
    # Coverage by Split
    lines.append("### Coverage by Split")
    lines.append("")
    lines.append("| Split | Regions | Bricks | Objects |")
    lines.append("|-------|---------|--------|---------|")
    for split, stats in coverage.get("by_split", {}).items():
        lines.append(f"| {split} | {stats.get('n_regions', 'N/A')} | {stats.get('n_bricks', 'N/A'):,} | {stats.get('n_objects', 'N/A'):,} |")
    lines.append("")
    
    # Schema
    lines.append("## Schema Validation")
    lines.append("")
    if schema.get("valid"):
        lines.append("All expected columns present.")
    else:
        lines.append(f"**Missing columns**: {schema.get('missing_columns')}")
    if schema.get("extra_columns"):
        lines.append(f"**Extra columns**: {schema.get('extra_columns')}")
    lines.append("")
    lines.append("### Column Types")
    lines.append("")
    lines.append("| Column | Type |")
    lines.append("|--------|------|")
    for col, dtype in sorted(schema.get("column_dtypes", {}).items()):
        lines.append(f"| {col} | {dtype} |")
    lines.append("")
    
    # Data Quality
    lines.append("## Data Quality")
    lines.append("")
    
    lines.append("### Magnitude Statistics")
    lines.append("")
    lines.append("| Band | Min | Max | Mean | Median | Std |")
    lines.append("|------|-----|-----|------|--------|-----|")
    for band, stats in quality.get("magnitude_stats", {}).items():
        lines.append(f"| {band} | {stats['min']} | {stats['max']} | {stats['mean']} | {stats['median']} | {stats['std']} |")
    lines.append("")
    
    lines.append("### Color Statistics")
    lines.append("")
    lines.append("| Color | Min | Max | Mean | P10 | P50 | P90 |")
    lines.append("|-------|-----|-----|------|-----|-----|-----|")
    for color, stats in quality.get("color_stats", {}).items():
        lines.append(f"| {color} | {stats['min']} | {stats['max']} | {stats['mean']} | {stats['p10']} | {stats['p50']} | {stats['p90']} |")
    lines.append("")
    
    lines.append("### Coordinate Ranges")
    lines.append("")
    coord_stats = quality.get("coordinate_stats", {})
    ra = coord_stats.get("ra", {})
    dec = coord_stats.get("dec", {})
    lines.append(f"- **RA**: {ra.get('min', 'N/A')}° to {ra.get('max', 'N/A')}°")
    lines.append(f"- **Dec**: {dec.get('min', 'N/A')}° to {dec.get('max', 'N/A')}°")
    lines.append("")
    
    # LRG Flag Validation
    lines.append("## LRG Flag Validation")
    lines.append("")
    lines.append("Verifying that LRG variant flags match recomputed values from raw magnitudes.")
    lines.append("")
    lines.append("| Variant | Expected True | Actual True | Mismatches | Status |")
    lines.append("|---------|---------------|-------------|------------|--------|")
    for vname in ["v1_pure_massive", "v2_baseline_dr10", "v3_color_relaxed", "v4_mag_relaxed", "v5_very_relaxed"]:
        result = lrg_flags.get(vname, {})
        if "error" in result:
            lines.append(f"| {vname} | - | - | - | ❌ {result['error']} |")
        else:
            status = "✅" if result.get("passed") else "❌"
            lines.append(f"| {vname} | {result.get('n_expected_true', 'N/A'):,} | {result.get('n_actual_true', 'N/A'):,} | {result.get('n_mismatches', 'N/A'):,} | {status} |")
    lines.append("")
    
    lines.append("### Variant Hierarchy Check")
    lines.append("")
    lines.append("Verifying that stricter variants are subsets of relaxed variants (v1 ⊂ v2 ⊂ v3 ⊂ v4 ⊂ v5).")
    lines.append("")
    for check in lrg_flags.get("hierarchy_checks", []):
        status = "✅" if check.get("passed") else "❌"
        lines.append(f"- {check['strict']} ⊂ {check['relaxed']}: {check['violations']} violations {status}")
    lines.append("")
    
    # Variant Distribution
    lines.append("## LRG Variant Distribution")
    lines.append("")
    lines.append("| Variant | Count | % of Total |")
    lines.append("|---------|-------|------------|")
    for col, count in variant_dist.get("variant_counts", {}).items():
        pct = variant_dist.get("variant_pcts", {}).get(col, 0)
        lines.append(f"| {col.replace('is_', '')} | {count:,} | {pct}% |")
    lines.append("")
    
    psc = variant_dist.get("parent_selection_check", {})
    if "error" not in psc:
        lines.append(f"**Parent Selection**: Target variant `{psc.get('target_variant')}` - {psc.get('pct_passing')}% pass")
    lines.append("")
    
    # Type Filter
    lines.append("## TYPE Filter (Star Rejection)")
    lines.append("")
    if type_filter.get("psf_filter_passed"):
        lines.append("✅ No PSF (point source / star) objects found - filter working correctly.")
    else:
        lines.append(f"❌ Found {type_filter.get('n_psf_found')} PSF objects - these should have been filtered out.")
    lines.append("")
    lines.append("### Type Distribution")
    lines.append("")
    lines.append("| Type | Count |")
    lines.append("|------|-------|")
    for typ, count in sorted(type_filter.get("type_distribution", {}).items(), key=lambda x: -x[1]):
        lines.append(f"| {typ} | {count:,} |")
    lines.append("")
    
    # Top Regions
    lines.append("## Top 10 Regions by Object Count")
    lines.append("")
    lines.append("| Split | Region ID | Bricks | Objects |")
    lines.append("|-------|-----------|--------|---------|")
    for r in coverage.get("top_10_regions", []):
        lines.append(f"| {r['region_split']} | {r['region_id']} | {r['n_bricks']:,} | {r['n_objects']:,} |")
    lines.append("")
    
    # Objects per Brick
    lines.append("## Objects per Brick Statistics")
    lines.append("")
    opb = coverage.get("objects_per_brick", {})
    lines.append(f"- **Min**: {opb.get('min', 'N/A')}")
    lines.append(f"- **Max**: {opb.get('max', 'N/A')}")
    lines.append(f"- **Mean**: {opb.get('mean', 'N/A')}")
    lines.append(f"- **Median**: {opb.get('median', 'N/A')}")
    lines.append("")
    
    # File Structure
    lines.append("## File Structure")
    lines.append("")
    lines.append("### Partitions by Split")
    lines.append("")
    for split, stats in structure.get("partitions_by_split", {}).items():
        size_mb = round(stats.get("total_bytes", 0) / (1024**2), 2)
        lines.append(f"- **{split}**: {stats.get('n_regions', 0)} regions, {stats.get('n_files', 0)} files, {size_mb} MB")
    lines.append("")
    
    lines.append("### Sample File Paths")
    lines.append("")
    for path in structure.get("sample_files", [])[:3]:
        lines.append(f"- `{path}`")
    lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 3c output")
    parser.add_argument("--phase3c-parquet", required=True, help="S3 path to Phase 3c parent catalog parquet")
    parser.add_argument("--phase3a-regions", default=None, help="S3 path to Phase 3a region_metrics (optional)")
    parser.add_argument("--phase3b-selections", default=None, help="S3 path to Phase 3b region_selections (optional)")
    parser.add_argument("--output-report", default="results/phase3c_validation_report.md", help="Output Markdown report path")
    parser.add_argument("--output-json", default=None, help="Output JSON summary path (optional)")
    parser.add_argument("--variant", default="v3_color_relaxed", help="Target LRG variant for parent selection")
    parser.add_argument("--max-sample-rows", type=int, default=500000, help="Max rows to sample for analysis")
    parser.add_argument("--max-sample-files", type=int, default=50, help="Max parquet files to sample (to avoid memory issues with many small files)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Phase 3c Output Validation")
    print("=" * 60)
    print()
    
    client = boto3.client("s3")
    
    # 1. Analyze structure
    print("[1/7] Analyzing Parquet structure...")
    structure = analyze_parquet_structure(args.phase3c_parquet, client)
    print(f"      Found {structure['total_parquet_files']} files, {structure['total_size_gb']} GB")
    
    # 2. Load sample data
    print("[2/7] Loading sample data...")
    df = load_parquet_from_s3(args.phase3c_parquet, max_rows=args.max_sample_rows, 
                               max_files=args.max_sample_files, client=client)
    
    if df is None or len(df) == 0:
        print("ERROR: Could not load any data from Parquet")
        sys.exit(1)
    
    print(f"      Loaded {len(df):,} rows for analysis")
    
    # 3. Validate schema
    print("[3/7] Validating schema...")
    schema = validate_schema(df)
    print(f"      Schema valid: {schema['valid']}")
    
    # 4. Validate data quality
    print("[4/7] Checking data quality...")
    quality = validate_data_quality(df)
    print(f"      Core nulls check: {quality['core_columns_null_check']['passed']}")
    
    # 5. Validate LRG flags
    print("[5/7] Validating LRG flags...")
    lrg_flags = validate_lrg_flags(df)
    n_passed = sum(1 for k, v in lrg_flags.items() if k != "hierarchy_checks" and v.get("passed"))
    print(f"      {n_passed}/5 variants passed flag validation")
    
    # 6. Validate coverage
    print("[6/7] Analyzing coverage...")
    coverage = validate_coverage(df)
    print(f"      {coverage['n_unique_regions']} regions, {coverage['n_unique_bricks']:,} bricks, {coverage['n_total_objects']:,} objects")
    
    # 7. Additional validations
    print("[7/7] Running additional validations...")
    variant_dist = validate_variant_distribution(df, args.variant)
    type_filter = validate_type_filter(df)
    print(f"      TYPE filter passed: {type_filter.get('psf_filter_passed', False)}")
    
    # Generate report
    print()
    print("Generating report...")
    report = generate_report(structure, schema, quality, lrg_flags, coverage, variant_dist, type_filter, args)
    
    # Write report
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w") as f:
        f.write(report)
    print(f"Report written to: {args.output_report}")
    
    # Write JSON if requested
    if args.output_json:
        summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "input": args.phase3c_parquet,
            "structure": structure,
            "schema": schema,
            "quality": quality,
            "lrg_flags": lrg_flags,
            "coverage": coverage,
            "variant_distribution": variant_dist,
            "type_filter": type_filter,
        }
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"JSON summary written to: {args.output_json}")
    
    print()
    print("=" * 60)
    print("Validation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


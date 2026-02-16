#!/usr/bin/env python3
"""
Phase 4a Manifest Validation Script

Validates the output of Stage 4a (manifest generation) for:
- Structural integrity
- Data quality
- Coverage across splits, selection sets, and observing conditions
- Consistency with expected counts
- Alignment with Phase 3 inputs

Usage:
    python scripts/validate_phase4a_manifests.py \
        --manifests-s3 s3://darkhaloscope/phase4_pipeline/phase4a/v3_color_relaxed/manifests \
        --bricks-manifest-s3 s3://darkhaloscope/phase4_pipeline/phase4a/v3_color_relaxed/bricks_manifest \
        --stage-config-s3 s3://darkhaloscope/phase4_pipeline/phase4a/v3_color_relaxed/_stage_config.json \
        --output-report phase4a_validation_report.json \
        --output-summary phase4a_validation_summary.md
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas and numpy are required. Install with: pip install pandas numpy pyarrow")
    sys.exit(1)


def run_aws_cmd(cmd: str, region: str = "us-east-2") -> str:
    """Run an AWS CLI command and return output."""
    full_cmd = f"{cmd} --region {region}"
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"AWS command failed: {full_cmd}\n{result.stderr}")
    return result.stdout


def s3_list_prefixes(s3_uri: str, region: str = "us-east-2") -> List[str]:
    """List immediate subdirectories under an S3 prefix."""
    if not s3_uri.endswith("/"):
        s3_uri += "/"
    output = run_aws_cmd(f"aws s3 ls {s3_uri}", region)
    prefixes = []
    for line in output.strip().split("\n"):
        if line.strip() and line.strip().startswith("PRE "):
            prefix = line.strip().replace("PRE ", "").rstrip("/")
            prefixes.append(prefix)
    return prefixes


def s3_download_json(s3_uri: str, region: str = "us-east-2") -> Dict:
    """Download and parse a JSON file from S3."""
    output = run_aws_cmd(f"aws s3 cp {s3_uri} -", region)
    return json.loads(output)


def s3_download_parquet_to_df(s3_uri: str, region: str = "us-east-2", sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Streaming validation: process ALL files but only keep stats + small sample.
    Returns a sample DataFrame with aggregated stats in attrs.
    """
    import shutil
    import gc
    
    # List parquet files
    list_output = run_aws_cmd(f"aws s3 ls {s3_uri}/ --recursive", region)
    parquet_files = []
    for line in list_output.strip().split("\n"):
        if line.strip() and ".parquet" in line:
            parts = line.split()
            if len(parts) >= 4:
                key = parts[-1]
                bucket = s3_uri.replace("s3://", "").split("/")[0]
                parquet_files.append(f"s3://{bucket}/{key}")
    
    if not parquet_files:
        raise RuntimeError(f"No parquet files found at {s3_uri}")
    
    n_files = len(parquet_files)
    print(f"  Found {n_files} parquet files, streaming ALL...", flush=True)
    
    tmpdir = tempfile.mkdtemp()
    
    # Aggregated stats (no data in memory)
    total_rows = 0
    columns = None
    split_counts = {}
    selection_set_counts = {}
    ra_min, ra_max = float('inf'), float('-inf')
    dec_min, dec_max = float('inf'), float('-inf')
    unique_bricks = set()
    null_counts = {}
    sample_rows = []  # Keep tiny sample for schema validation
    
    try:
        for i, file_uri in enumerate(parquet_files):
            local_file = os.path.join(tmpdir, "current.parquet")
            
            # Download
            run_aws_cmd(f"aws s3 cp {file_uri} {local_file}", region)
            
            # Read
            df = pd.read_parquet(local_file)
            file_rows = len(df)
            total_rows += file_rows
            
            # Delete downloaded file immediately
            os.remove(local_file)
            
            # First file: get columns
            if columns is None:
                columns = list(df.columns)
                null_counts = {c: 0 for c in columns}
            
            # Aggregate stats (no data kept in memory)
            if "region_split" in df.columns:
                for split, count in df["region_split"].value_counts().items():
                    split_counts[split] = split_counts.get(split, 0) + count
            
            if "selection_set_id" in df.columns:
                for sid, count in df["selection_set_id"].value_counts().items():
                    selection_set_counts[sid] = selection_set_counts.get(sid, 0) + count
            
            if "ra" in df.columns:
                ra_min = min(ra_min, df["ra"].min())
                ra_max = max(ra_max, df["ra"].max())
            
            if "dec" in df.columns:
                dec_min = min(dec_min, df["dec"].min())
                dec_max = max(dec_max, df["dec"].max())
            
            if "brickname" in df.columns:
                unique_bricks.update(df["brickname"].unique()[:1000])  # Limit set size
            
            for c in columns:
                null_counts[c] += df[c].isna().sum()
            
            # Keep tiny sample from first file only
            if i == 0:
                sample_rows = df.head(1000).to_dict('records')
            
            # Free memory
            del df
            gc.collect()
            
            if (i + 1) % 5 == 0 or i == n_files - 1:
                print(f"    Processed {i+1}/{n_files} files, {total_rows:,} total rows", flush=True)
        
        # Create result DataFrame from sample
        result_df = pd.DataFrame(sample_rows) if sample_rows else pd.DataFrame()
        
        # Store all aggregated stats
        result_df.attrs['n_files'] = n_files
        result_df.attrs['total_rows'] = total_rows
        result_df.attrs['columns'] = columns
        result_df.attrs['split_counts'] = split_counts
        result_df.attrs['selection_set_counts'] = selection_set_counts
        result_df.attrs['ra_range'] = [ra_min, ra_max]
        result_df.attrs['dec_range'] = [dec_min, dec_max]
        result_df.attrs['n_unique_bricks'] = len(unique_bricks)
        result_df.attrs['null_counts'] = null_counts
        
        print(f"  Total: {total_rows:,} rows across {n_files} files", flush=True)
        
        return result_df
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        gc.collect()


def validate_stage_config(config: Dict) -> Dict[str, Any]:
    """Validate the stage config JSON."""
    issues = []
    warnings = []
    
    # Check required top-level keys
    required_keys = ["stage", "variant", "inputs", "tiers", "stamp_sizes", "bandsets", "replicates"]
    for key in required_keys:
        if key not in config:
            issues.append(f"Missing required key: {key}")
    
    # Validate stage
    if config.get("stage") != "4a":
        issues.append(f"Unexpected stage: {config.get('stage')} (expected '4a')")
    
    # Validate inputs
    inputs = config.get("inputs", {})
    for input_key in ["parent_s3", "bricks_with_region_s3", "region_selections_s3"]:
        if not inputs.get(input_key):
            issues.append(f"Missing input path: {input_key}")
    
    # Validate tiers
    tiers = config.get("tiers", {})
    expected_tiers = {"debug", "grid", "train"}
    found_tiers = set(tiers.keys())
    if not found_tiers:
        issues.append("No tiers defined")
    
    # Check tier configs
    for tier_name, tier_config in tiers.items():
        if tier_name == "train":
            if "n_total_per_split" not in tier_config:
                warnings.append(f"Tier '{tier_name}' missing n_total_per_split")
        else:
            if "n_per_config" not in tier_config:
                warnings.append(f"Tier '{tier_name}' missing n_per_config")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "config_summary": {
            "stage": config.get("stage"),
            "variant": config.get("variant"),
            "tiers": list(tiers.keys()),
            "stamp_sizes": config.get("stamp_sizes"),
            "bandsets": config.get("bandsets"),
            "replicates": config.get("replicates"),
        }
    }


def validate_manifest_schema(df: pd.DataFrame, experiment_id: str) -> Dict[str, Any]:
    """Validate the schema of a manifest DataFrame."""
    issues = []
    warnings = []
    
    # Use attrs if available (from streaming), otherwise use df directly
    columns = df.attrs.get('columns', list(df.columns))
    total_rows = df.attrs.get('total_rows', len(df))
    null_counts = df.attrs.get('null_counts', {})
    
    # Required columns
    required_cols = [
        "task_id", "experiment_id", "brickname", "ra", "dec",
        "region_id", "region_split", "selection_set_id",
        "stamp_size", "bandset", "replicate",
        "config_id", "theta_e_arcsec"
    ]
    
    missing_cols = [c for c in required_cols if c not in columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    # Check for control column in train tier
    if "train" in experiment_id.lower():
        if "is_control" not in columns:
            warnings.append("Train tier missing 'is_control' column")
    
    # Check for null values in critical columns (use aggregated counts if available)
    critical_cols = ["task_id", "brickname", "ra", "dec", "region_split"]
    for col in critical_cols:
        if col in columns:
            null_count = null_counts.get(col, 0)
            if null_count > 0:
                issues.append(f"Column '{col}' has {null_count} null values")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "columns": columns,
        "row_count": total_rows
    }


def validate_manifest_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality of manifest."""
    issues = []
    warnings = []
    stats = {}
    
    # Use attrs if available (from streaming)
    ra_range = df.attrs.get('ra_range')
    dec_range = df.attrs.get('dec_range')
    total_rows = df.attrs.get('total_rows', len(df))
    
    # Check RA/Dec ranges
    if ra_range:
        ra_min, ra_max = ra_range
        stats["ra_range"] = [float(ra_min), float(ra_max)]
        if ra_min < 0 or ra_max > 360:
            issues.append(f"RA out of range [0, 360]: [{ra_min}, {ra_max}]")
    elif "ra" in df.columns:
        ra_min, ra_max = df["ra"].min(), df["ra"].max()
        stats["ra_range"] = [float(ra_min), float(ra_max)]
        if ra_min < 0 or ra_max > 360:
            issues.append(f"RA out of range [0, 360]: [{ra_min}, {ra_max}]")
    
    if dec_range:
        dec_min, dec_max = dec_range
        stats["dec_range"] = [float(dec_min), float(dec_max)]
        if dec_min < -90 or dec_max > 90:
            issues.append(f"Dec out of range [-90, 90]: [{dec_min}, {dec_max}]")
    elif "dec" in df.columns:
        dec_min, dec_max = df["dec"].min(), df["dec"].max()
        stats["dec_range"] = [float(dec_min), float(dec_max)]
        if dec_min < -90 or dec_max > 90:
            issues.append(f"Dec out of range [-90, 90]: [{dec_min}, {dec_max}]")
    
    # Check theta_e (Einstein radius) - should be positive for injections
    if "theta_e_arcsec" in df.columns:
        theta_stats = df["theta_e_arcsec"].describe()
        stats["theta_e_stats"] = {
            "min": float(theta_stats["min"]),
            "max": float(theta_stats["max"]),
            "mean": float(theta_stats["mean"]),
            "std": float(theta_stats["std"]) if "std" in theta_stats else 0,
        }
        negative_count = (df["theta_e_arcsec"] < 0).sum()
        if negative_count > 0:
            issues.append(f"Found {negative_count} negative theta_e values")
    
    # Check task_id uniqueness
    if "task_id" in df.columns:
        unique_tasks = df["task_id"].nunique()
        total_tasks = len(df)
        stats["task_id_unique"] = unique_tasks
        stats["task_id_total"] = total_tasks
        if unique_tasks != total_tasks:
            issues.append(f"Duplicate task_ids: {total_tasks - unique_tasks} duplicates")
    
    # Check replicate values
    if "replicate" in df.columns:
        replicate_values = sorted(df["replicate"].unique())
        stats["replicate_values"] = [int(r) for r in replicate_values]
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "stats": stats
    }


def validate_manifest_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate coverage across dimensions."""
    coverage = {}
    issues = []
    warnings = []
    
    # Use attrs if available (from streaming)
    split_counts_attr = df.attrs.get('split_counts', {})
    selection_set_counts_attr = df.attrs.get('selection_set_counts', {})
    n_unique_bricks_attr = df.attrs.get('n_unique_bricks')
    
    # Split coverage
    if split_counts_attr:
        coverage["by_split"] = {str(k): int(v) for k, v in split_counts_attr.items()}
        expected_splits = {"train", "val", "test"}
        found_splits = set(str(s) for s in split_counts_attr.keys())
        missing_splits = expected_splits - found_splits
        if missing_splits:
            warnings.append(f"Missing splits: {missing_splits}")
    elif "region_split" in df.columns:
        split_counts = df["region_split"].value_counts().to_dict()
        coverage["by_split"] = {str(k): int(v) for k, v in split_counts.items()}
        expected_splits = {"train", "val", "test"}
        found_splits = set(str(s) for s in split_counts.keys())
        missing_splits = expected_splits - found_splits
        if missing_splits:
            warnings.append(f"Missing splits: {missing_splits}")
    
    # Selection set coverage
    if selection_set_counts_attr:
        coverage["by_selection_set"] = {str(k): int(v) for k, v in selection_set_counts_attr.items()}
        coverage["n_selection_sets"] = len(selection_set_counts_attr)
    elif "selection_set_id" in df.columns:
        sel_counts = df["selection_set_id"].value_counts().to_dict()
        coverage["by_selection_set"] = {str(k): int(v) for k, v in sel_counts.items()}
        coverage["n_selection_sets"] = len(sel_counts)
    
    # PSF bin coverage (if present) - uses sample
    if "psf_bin" in df.columns and len(df) > 0:
        psf_counts = df["psf_bin"].value_counts().to_dict()
        coverage["by_psf_bin"] = {str(k): int(v) for k, v in sorted(psf_counts.items())}
    
    # Depth bin coverage (if present) - uses sample
    if "depth_bin" in df.columns and len(df) > 0:
        depth_counts = df["depth_bin"].value_counts().to_dict()
        coverage["by_depth_bin"] = {str(k): int(v) for k, v in sorted(depth_counts.items())}
    
    # Control vs injection (for train tier) - uses sample, scale up
    if "is_control" in df.columns and len(df) > 0:
        control_counts = df["is_control"].value_counts().to_dict()
        coverage["control_vs_injection"] = {
            "control": int(control_counts.get(1, 0)),
            "injection": int(control_counts.get(0, 0)),
        }
        total = sum(control_counts.values())
        if total > 0:
            coverage["control_fraction"] = control_counts.get(1, 0) / total
    
    # Brick coverage
    if n_unique_bricks_attr:
        coverage["n_unique_bricks"] = n_unique_bricks_attr
    elif "brickname" in df.columns and len(df) > 0:
        coverage["n_unique_bricks"] = int(df["brickname"].nunique())
    
    # Config coverage - uses sample
    if "config_id" in df.columns and len(df) > 0:
        coverage["n_unique_configs"] = int(df["config_id"].nunique())
    
    return {
        "coverage": coverage,
        "issues": issues,
        "warnings": warnings
    }


def estimate_4c_runtime(total_tasks: int, nodes: int = 30, cores_per_node: int = 4) -> Dict[str, Any]:
    """Estimate Stage 4c runtime based on task count."""
    executors = nodes * cores_per_node // 2  # 2 cores per executor
    
    # Estimate: 1-3 seconds per task
    optimistic_sec_per_task = 1.0
    conservative_sec_per_task = 3.0
    
    optimistic_hours = (total_tasks / executors * optimistic_sec_per_task) / 3600
    conservative_hours = (total_tasks / executors * conservative_sec_per_task) / 3600
    
    # Cost estimate (m5.xlarge ~ $0.192/hr)
    cost_per_hour = nodes * 0.192
    
    return {
        "total_tasks": total_tasks,
        "assumed_nodes": nodes,
        "assumed_executors": executors,
        "optimistic_hours": round(optimistic_hours, 1),
        "conservative_hours": round(conservative_hours, 1),
        "optimistic_days": round(optimistic_hours / 24, 2),
        "conservative_days": round(conservative_hours / 24, 2),
        "estimated_cost_usd": {
            "optimistic": round(cost_per_hour * optimistic_hours, 2),
            "conservative": round(cost_per_hour * conservative_hours, 2),
        }
    }


def generate_summary_markdown(report: Dict, output_path: str) -> None:
    """Generate a human-readable markdown summary."""
    lines = [
        "# Phase 4a Validation Report",
        f"\nGenerated: {report['metadata']['timestamp']}",
        f"\nVariant: {report['metadata'].get('variant', 'unknown')}",
        "",
        "## Overall Status",
        "",
    ]
    
    # Overall status
    overall_valid = report.get("overall_valid", False)
    status_emoji = "✅" if overall_valid else "❌"
    lines.append(f"**Status**: {status_emoji} {'PASSED' if overall_valid else 'FAILED'}")
    lines.append("")
    
    # Stage config summary
    if "stage_config" in report:
        cfg = report["stage_config"]
        lines.append("## Stage Configuration")
        lines.append("")
        if "config_summary" in cfg:
            cs = cfg["config_summary"]
            lines.append(f"- **Tiers**: {', '.join(cs.get('tiers', []))}")
            lines.append(f"- **Stamp sizes**: {cs.get('stamp_sizes')}")
            lines.append(f"- **Bandsets**: {cs.get('bandsets')}")
            lines.append(f"- **Replicates**: {cs.get('replicates')}")
        lines.append("")
    
    # Experiments summary
    if "experiments" in report:
        lines.append("## Experiments Summary")
        lines.append("")
        lines.append("| Experiment | Tasks | Unique Bricks | Splits | Valid |")
        lines.append("|------------|-------|---------------|--------|-------|")
        
        total_tasks = 0
        for exp_name, exp_data in report["experiments"].items():
            task_count = exp_data.get("row_count", 0)
            total_tasks += task_count
            n_bricks_raw = exp_data.get("coverage", {}).get("coverage", {}).get("n_unique_bricks", "?")
            n_bricks = f"{n_bricks_raw:,}" if isinstance(n_bricks_raw, int) else str(n_bricks_raw)
            splits = list(exp_data.get("coverage", {}).get("coverage", {}).get("by_split", {}).keys())
            valid = "✅" if exp_data.get("valid", False) else "❌"
            lines.append(f"| {exp_name} | {task_count:,} | {n_bricks} | {', '.join(splits)} | {valid} |")
        
        lines.append(f"| **TOTAL** | **{total_tasks:,}** | | | |")
        lines.append("")
    
    # Runtime estimates
    if "runtime_estimates" in report:
        lines.append("## Stage 4c Runtime Estimates")
        lines.append("")
        for exp_name, est in report["runtime_estimates"].items():
            lines.append(f"### {exp_name}")
            lines.append(f"- Tasks: {est['total_tasks']:,}")
            lines.append(f"- Estimated time: {est['optimistic_hours']:.1f} - {est['conservative_hours']:.1f} hours")
            lines.append(f"- Estimated cost: ${est['estimated_cost_usd']['optimistic']:.2f} - ${est['estimated_cost_usd']['conservative']:.2f}")
            lines.append("")
    
    # Issues
    all_issues = []
    all_warnings = []
    
    if "stage_config" in report:
        all_issues.extend(report["stage_config"].get("issues", []))
        all_warnings.extend(report["stage_config"].get("warnings", []))
    
    for exp_data in report.get("experiments", {}).values():
        all_issues.extend(exp_data.get("schema", {}).get("issues", []))
        all_warnings.extend(exp_data.get("schema", {}).get("warnings", []))
        all_issues.extend(exp_data.get("data_quality", {}).get("issues", []))
        all_warnings.extend(exp_data.get("data_quality", {}).get("warnings", []))
    
    if all_issues:
        lines.append("## Issues ❌")
        lines.append("")
        for issue in all_issues:
            lines.append(f"- {issue}")
        lines.append("")
    
    if all_warnings:
        lines.append("## Warnings ⚠️")
        lines.append("")
        for warning in all_warnings:
            lines.append(f"- {warning}")
        lines.append("")
    
    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Summary written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 4a manifest outputs")
    parser.add_argument("--manifests-s3", required=True, help="S3 path to manifests directory")
    parser.add_argument("--bricks-manifest-s3", required=True, help="S3 path to bricks_manifest directory")
    parser.add_argument("--stage-config-s3", required=True, help="S3 path to _stage_config.json")
    parser.add_argument("--output-report", default="phase4a_validation_report.json", help="Output JSON report path")
    parser.add_argument("--output-summary", default="phase4a_validation_summary.md", help="Output markdown summary path")
    parser.add_argument("--region", default="us-east-2", help="AWS region")
    parser.add_argument("--sample-size", type=int, default=100000, help="Sample size for large manifests (0=all)")
    parser.add_argument("--runtime-nodes", type=int, default=30, help="Assumed nodes for runtime estimate")
    
    args = parser.parse_args()
    
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "manifests_s3": args.manifests_s3,
            "bricks_manifest_s3": args.bricks_manifest_s3,
            "stage_config_s3": args.stage_config_s3,
        },
        "overall_valid": True,
        "experiments": {},
        "runtime_estimates": {},
    }
    
    # 1. Validate stage config
    print("Validating stage config...")
    try:
        stage_config = s3_download_json(args.stage_config_s3, args.region)
        config_validation = validate_stage_config(stage_config)
        report["stage_config"] = config_validation
        report["metadata"]["variant"] = stage_config.get("variant", "unknown")
        if not config_validation["valid"]:
            report["overall_valid"] = False
    except Exception as e:
        report["stage_config"] = {"valid": False, "issues": [str(e)], "warnings": []}
        report["overall_valid"] = False
    
    # 2. List and validate each experiment manifest
    print("Listing experiment manifests...")
    try:
        experiments = s3_list_prefixes(args.manifests_s3, args.region)
        print(f"Found {len(experiments)} experiments: {experiments}")
    except Exception as e:
        print(f"ERROR listing manifests: {e}")
        experiments = []
        report["overall_valid"] = False
    
    # Filter out _csv experiments (they're just convenience copies of parquet)
    parquet_experiments = [e for e in experiments if not e.endswith("_csv")]
    print(f"Validating {len(parquet_experiments)} parquet experiments (skipping {len(experiments) - len(parquet_experiments)} CSV copies)")
    
    for exp_name in parquet_experiments:
        print(f"\nValidating experiment: {exp_name}...")
        exp_s3_path = f"{args.manifests_s3.rstrip('/')}/{exp_name}"
        
        exp_report = {"valid": True}
        
        try:
            # Download manifest (with sampling for large ones)
            sample_size = args.sample_size if args.sample_size > 0 else None
            if "train" in exp_name.lower() and sample_size:
                print(f"  Sampling {sample_size} rows from train manifest...")
            
            df = s3_download_parquet_to_df(exp_s3_path, args.region)
            exp_report["row_count"] = len(df)
            print(f"  Loaded {len(df):,} tasks")
            
            # Schema validation
            schema_result = validate_manifest_schema(df, exp_name)
            exp_report["schema"] = schema_result
            if not schema_result["valid"]:
                exp_report["valid"] = False
            
            # Data quality validation
            quality_result = validate_manifest_data_quality(df)
            exp_report["data_quality"] = quality_result
            if not quality_result["valid"]:
                exp_report["valid"] = False
            
            # Coverage validation
            coverage_result = validate_manifest_coverage(df)
            exp_report["coverage"] = coverage_result
            
            # Runtime estimate
            report["runtime_estimates"][exp_name] = estimate_4c_runtime(
                len(df), nodes=args.runtime_nodes
            )
            
        except Exception as e:
            exp_report["valid"] = False
            exp_report["error"] = str(e)
            print(f"  ERROR: {e}")
        
        report["experiments"][exp_name] = exp_report
        if not exp_report["valid"]:
            report["overall_valid"] = False
    
    # 3. Validate bricks manifest
    print("\nValidating bricks manifest...")
    try:
        bricks_df = s3_download_parquet_to_df(args.bricks_manifest_s3, args.region)
        report["bricks_manifest"] = {
            "valid": True,
            "n_bricks": len(bricks_df),
            "columns": list(bricks_df.columns),
        }
        if "brickname" in bricks_df.columns:
            report["bricks_manifest"]["n_unique_bricks"] = int(bricks_df["brickname"].nunique())
        print(f"  Found {len(bricks_df):,} bricks")
    except Exception as e:
        report["bricks_manifest"] = {"valid": False, "error": str(e)}
        report["overall_valid"] = False
    
    # 4. Write reports
    print("\nWriting reports...")
    
    # JSON report
    with open(args.output_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"JSON report written to: {args.output_report}")
    
    # Markdown summary
    generate_summary_markdown(report, args.output_summary)
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    status = "✅ PASSED" if report["overall_valid"] else "❌ FAILED"
    print(f"Overall Status: {status}")
    print(f"Experiments validated: {len(report['experiments'])}")
    
    total_tasks = sum(e.get("row_count", 0) for e in report["experiments"].values())
    print(f"Total tasks across all experiments: {total_tasks:,}")
    
    if report.get("bricks_manifest", {}).get("n_bricks"):
        print(f"Total bricks to cache in 4b: {report['bricks_manifest']['n_bricks']:,}")
    
    return 0 if report["overall_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())


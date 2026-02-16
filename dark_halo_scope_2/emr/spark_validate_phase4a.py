#!/usr/bin/env python3
"""
Spark-based validation for Phase 4a manifests.
Runs distributed validation across the cluster - no OOM issues.
"""

import argparse
import json
import sys
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType


def discover_experiments(s3_prefix: str) -> list:
    """Discover experiment subdirectories under manifests/.
    
    Auto-discovers actual experiment names from S3 instead of hardcoding.
    Filters out _csv exports (those are secondary outputs).
    FAILS HARD if no experiments are found.
    """
    import boto3
    from urllib.parse import urlparse
    
    parsed = urlparse(s3_prefix)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    if not prefix.endswith("/"):
        prefix += "/"
    
    print(f"Discovering experiments under s3://{bucket}/{prefix}")
    
    client = boto3.client("s3", region_name="us-east-2")
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
    
    experiments = []
    for cp in resp.get("CommonPrefixes", []):
        name = cp["Prefix"].replace(prefix, "").rstrip("/")
        if not name.endswith("_csv"):  # Skip CSV exports
            experiments.append(name)
    
    if not experiments:
        raise RuntimeError(f"FATAL: No experiments found under {s3_prefix}. "
                           "Check that Phase 4a ran successfully and the path is correct.")
    
    print(f"Discovered {len(experiments)} experiments: {experiments}")
    return sorted(experiments)


def make_spark(app_name: str = "Phase4aValidation") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.parquet.mergeSchema", "false")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def validate_manifests(spark: SparkSession, args, stage_config: dict = None) -> dict:
    """Validate all manifest experiments.
    
    Args:
        spark: SparkSession
        args: CLI arguments
        stage_config: Optional stage config dict to extract expected control fraction
    """
    
    results = {
        "validation_time": datetime.utcnow().isoformat(),
        "manifests_s3": args.manifests_s3,
        "experiments": {},
        "summary": {},
    }
    
    # Extract expected control fraction from stage config (default 0.25)
    # This fixes the bug where we hardcoded 0.5 but pipeline default is 0.25
    expected_control_frac = 0.25  # Pipeline default
    if stage_config and stage_config.get("valid"):
        config = stage_config.get("config", {})
        # Try to get from tier-specific config or global
        if "control_frac_train" in config:
            expected_control_frac = config["control_frac_train"]
        elif "tiers" in config:
            # Check train tier config
            train_tier = config.get("tiers", {}).get("train", {})
            if "control_frac" in train_tier:
                expected_control_frac = train_tier["control_frac"]
    print(f"Expected control fraction: {expected_control_frac}")
    
    # Discover experiments dynamically from S3 (no hardcoding)
    experiments = discover_experiments(args.manifests_s3)
    
    total_tasks = 0
    all_issues = []
    
    for exp in experiments:
        exp_path = f"{args.manifests_s3}/{exp}"
        print(f"\n{'='*60}")
        print(f"Validating experiment: {exp}")
        print(f"Path: {exp_path}")
        print(f"{'='*60}")
        
        # FAIL HARD if path doesn't exist - no silent skipping
        df = spark.read.parquet(exp_path)
        df.cache()
        
        exp_result = validate_single_manifest(df, exp, expected_control_frac=expected_control_frac)
        results["experiments"][exp] = exp_result
        total_tasks += exp_result.get("row_count", 0)
        all_issues.extend(exp_result.get("issues", []))
        
        df.unpersist()
    
    # Summary
    results["summary"] = {
        "total_experiments": len(experiments),
        "total_tasks": total_tasks,
        "total_issues": len(all_issues),
        "issues": all_issues[:20],  # First 20 issues
    }
    
    return results


def validate_single_manifest(df, experiment_id: str, expected_control_frac: float = 0.25) -> dict:
    """Validate a single manifest DataFrame using Spark aggregations.
    
    Args:
        df: DataFrame to validate
        experiment_id: Experiment identifier
        expected_control_frac: Expected control fraction from stage config (default 0.25)
    """
    
    result = {
        "experiment_id": experiment_id,
        "issues": [],
        "warnings": [],
    }
    
    # 1. Row count
    row_count = df.count()
    result["row_count"] = row_count
    print(f"  Row count: {row_count:,}")
    
    # 2. Schema validation
    columns = df.columns
    result["columns"] = columns
    
    required_cols = [
        "task_id", "experiment_id", "brickname", "ra", "dec",
        "region_id", "region_split", "selection_set_id",
        "stamp_size", "bandset", "replicate", "config_id", "theta_e_arcsec"
    ]
    missing_cols = [c for c in required_cols if c not in columns]
    if missing_cols:
        result["issues"].append(f"Missing columns: {missing_cols}")
    
    # 3. Null checks (distributed)
    critical_cols = ["task_id", "brickname", "ra", "dec", "region_split"]
    null_counts = {}
    for col in critical_cols:
        if col in columns:
            null_count = df.filter(F.col(col).isNull()).count()
            null_counts[col] = null_count
            if null_count > 0:
                result["issues"].append(f"Column '{col}' has {null_count} nulls")
    result["null_counts"] = null_counts
    
    # 4. RA/Dec range validation
    if "ra" in columns and "dec" in columns:
        ranges = df.agg(
            F.min("ra").alias("ra_min"),
            F.max("ra").alias("ra_max"),
            F.min("dec").alias("dec_min"),
            F.max("dec").alias("dec_max"),
        ).collect()[0]
        
        result["ra_range"] = [float(ranges["ra_min"]), float(ranges["ra_max"])]
        result["dec_range"] = [float(ranges["dec_min"]), float(ranges["dec_max"])]
        
        if ranges["ra_min"] < 0 or ranges["ra_max"] > 360:
            result["issues"].append(f"RA out of [0,360]: {result['ra_range']}")
        if ranges["dec_min"] < -90 or ranges["dec_max"] > 90:
            result["issues"].append(f"Dec out of [-90,90]: {result['dec_range']}")
        
        print(f"  RA range: [{ranges['ra_min']:.2f}, {ranges['ra_max']:.2f}]")
        print(f"  Dec range: [{ranges['dec_min']:.2f}, {ranges['dec_max']:.2f}]")
    
    # 5. Split distribution
    if "region_split" in columns:
        split_counts = df.groupBy("region_split").count().collect()
        result["split_distribution"] = {row["region_split"]: row["count"] for row in split_counts}
        print(f"  Split distribution: {result['split_distribution']}")
        
        expected = {"train", "val", "test"}
        found = set(result["split_distribution"].keys())
        missing = expected - found
        if missing:
            result["warnings"].append(f"Missing splits: {missing}")
    
    # 6. Selection set distribution
    if "selection_set_id" in columns:
        n_sets = df.select("selection_set_id").distinct().count()
        result["n_selection_sets"] = n_sets
        print(f"  Selection sets: {n_sets}")
    
    # 7. Unique bricks
    if "brickname" in columns:
        n_bricks = df.select("brickname").distinct().count()
        result["n_unique_bricks"] = n_bricks
        print(f"  Unique bricks: {n_bricks}")
    
    # 8. Unique regions
    if "region_id" in columns:
        n_regions = df.select("region_id").distinct().count()
        result["n_unique_regions"] = n_regions
        print(f"  Unique regions: {n_regions}")
    
    # 9. Task ID uniqueness
    if "task_id" in columns:
        n_unique_tasks = df.select("task_id").distinct().count()
        if n_unique_tasks != row_count:
            dups = row_count - n_unique_tasks
            result["issues"].append(f"Duplicate task_ids: {dups}")
        print(f"  Task ID uniqueness: {n_unique_tasks}/{row_count}")
    
    # 10. Theta_e stats (for injection tasks)
    # =========================================================================
    # NOTE ON CONTROL SAMPLES:
    # Control samples have theta_e_arcsec = 0.0 (no lens injection).
    # This is valid data, not missing data. We must use "is not None" checks
    # to avoid treating 0.0 as falsy/null.
    #
    # BUG FIX (2026-01-22): Previously used `if theta_stats["min"]` which
    # treated 0.0 as falsy, causing controls to show min=null. Fixed below.
    # =========================================================================
    if "theta_e_arcsec" in columns:
        theta_stats = df.agg(
            F.min("theta_e_arcsec").alias("min"),
            F.max("theta_e_arcsec").alias("max"),
            F.mean("theta_e_arcsec").alias("mean"),
            F.stddev("theta_e_arcsec").alias("std"),
        ).collect()[0]
        
        # FIX: Use "is not None" to correctly handle theta_e=0.0 (control samples)
        result["theta_e_stats"] = {
            "min": float(theta_stats["min"]) if theta_stats["min"] is not None else None,
            "max": float(theta_stats["max"]) if theta_stats["max"] is not None else None,
            "mean": float(theta_stats["mean"]) if theta_stats["mean"] is not None else None,
            "std": float(theta_stats["std"]) if theta_stats["std"] is not None else None,
        }
        
        # Safe print that handles None values
        min_str = f"{theta_stats['min']:.3f}" if theta_stats['min'] is not None else "None"
        max_str = f"{theta_stats['max']:.3f}" if theta_stats['max'] is not None else "None"
        mean_str = f"{theta_stats['mean']:.3f}" if theta_stats['mean'] is not None else "None"
        print(f"  Theta_e: min={min_str}, max={max_str}, mean={mean_str}")
        
        neg_count = df.filter(F.col("theta_e_arcsec") < 0).count()
        if neg_count > 0:
            result["issues"].append(f"Negative theta_e values: {neg_count}")
    
    # 11. Control fraction validation (for train tiers)
    # =========================================================================
    # Train tier should have ~50% controls (configurable via control_frac).
    # Controls are samples with theta_e=0 (no lens injection) - negative examples.
    # We validate that the actual control fraction is within 5% of expected.
    # =========================================================================
    if "is_control" in columns:
        control_counts = df.groupBy("is_control").count().collect()
        control_dist = {str(row["is_control"]): row["count"] for row in control_counts}
        result["control_distribution"] = control_dist
        total = sum(control_dist.values())
        if total > 0:
            # Count controls (is_control=1 or True)
            ctrl = control_dist.get("1", control_dist.get("True", 0))
            if isinstance(ctrl, str):
                ctrl = int(ctrl)
            actual_frac = ctrl / total
            result["control_fraction"] = actual_frac
            
            # Validate against expected (from stage_config, default 25%)
            # FIX (2026-01-22): Now uses expected_control_frac passed from stage_config
            # instead of hardcoded 0.5. Pipeline default is 0.25.
            tolerance = 0.05  # Allow 5% deviation
            if abs(actual_frac - expected_control_frac) > tolerance:
                result["warnings"].append(
                    f"Control fraction {actual_frac:.1%} differs from expected {expected_control_frac:.0%} by more than {tolerance:.0%}"
                )
            print(f"  Control fraction: {actual_frac:.1%} (expected ~{expected_control_frac:.0%})")
        print(f"  Control distribution: {control_dist}")
    
    # 12. Config coverage and per-config count validation
    # =========================================================================
    # For debug/grid tiers, each config should have roughly equal representation.
    # For train tier, configs are randomly assigned so distribution may vary more.
    # =========================================================================
    if "config_id" in columns:
        n_configs = df.select("config_id").distinct().count()
        result["n_unique_configs"] = n_configs
        print(f"  Unique configs: {n_configs}")
        
        # Compute per-config counts for validation
        config_counts = df.groupBy("config_id").count()
        config_stats = config_counts.agg(
            F.min("count").alias("min"),
            F.max("count").alias("max"),
            F.avg("count").alias("avg"),
            F.stddev("count").alias("std"),
        ).collect()[0]
        
        result["per_config_stats"] = {
            "min": int(config_stats["min"]) if config_stats["min"] is not None else None,
            "max": int(config_stats["max"]) if config_stats["max"] is not None else None,
            "avg": float(config_stats["avg"]) if config_stats["avg"] is not None else None,
            "std": float(config_stats["std"]) if config_stats["std"] is not None else None,
        }
        
        # For debug/grid tiers, check if distribution is reasonably uniform
        if config_stats["avg"] and config_stats["std"]:
            cv = config_stats["std"] / config_stats["avg"]  # Coefficient of variation
            result["per_config_cv"] = cv
            # CV > 0.5 suggests highly uneven distribution
            if cv > 0.5:
                result["warnings"].append(
                    f"Per-config count has high variability (CV={cv:.2f}). "
                    f"Range: {config_stats['min']}-{config_stats['max']}, avg={config_stats['avg']:.0f}"
                )
        
        print(f"  Per-config: min={config_stats['min']}, max={config_stats['max']}, avg={config_stats['avg']:.0f}")
    
    # 13. Stamp size distribution
    if "stamp_size" in columns:
        stamp_counts = df.groupBy("stamp_size").count().collect()
        result["stamp_size_distribution"] = {row["stamp_size"]: row["count"] for row in stamp_counts}
        print(f"  Stamp sizes: {result['stamp_size_distribution']}")
    
    # 14. Bandset distribution
    if "bandset" in columns:
        bandset_counts = df.groupBy("bandset").count().collect()
        result["bandset_distribution"] = {row["bandset"]: row["count"] for row in bandset_counts}
        print(f"  Bandsets: {result['bandset_distribution']}")
    
    result["valid"] = len(result["issues"]) == 0
    print(f"  Valid: {result['valid']} (issues: {len(result['issues'])}, warnings: {len(result['warnings'])})")
    
    return result


def validate_bricks_manifest(spark: SparkSession, args) -> dict:
    """Validate the bricks manifest. FAILS HARD if path doesn't exist."""
    
    print(f"\n{'='*60}")
    print("Validating bricks manifest")
    print(f"Path: {args.bricks_manifest_s3}")
    print(f"{'='*60}")
    
    result = {"issues": [], "warnings": []}
    
    # FAIL HARD if path doesn't exist - no silent skipping
    df = spark.read.parquet(args.bricks_manifest_s3)
    df.cache()
    
    row_count = df.count()
    result["row_count"] = row_count
    result["columns"] = df.columns
    print(f"  Row count: {row_count:,}")
    print(f"  Columns: {df.columns}")
    
    # Unique bricks
    if "brickname" in df.columns:
        n_bricks = df.select("brickname").distinct().count()
        result["n_unique_bricks"] = n_bricks
        print(f"  Unique bricks: {n_bricks}")
    
    df.unpersist()
    result["valid"] = True
    
    return result


def load_stage_config(spark: SparkSession, config_s3: str) -> dict:
    """Load stage config JSON from S3."""
    import boto3
    
    print(f"\nLoading stage config from: {config_s3}")
    
    try:
        # Parse S3 URI
        parts = config_s3.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        config = json.loads(obj["Body"].read().decode("utf-8"))
        
        print(f"  Stage: {config.get('stage')}")
        print(f"  Variant: {config.get('variant')}")
        print(f"  Tiers: {list(config.get('tiers', {}).keys())}")
        
        return {"valid": True, "config": config}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"valid": False, "error": str(e)}


def write_results(spark: SparkSession, results: dict, output_s3: str):
    """Write validation results to S3."""
    import boto3
    
    parts = output_s3.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(results, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"\nResults written to: {output_s3}")


def generate_summary_markdown(results: dict) -> str:
    """Generate markdown summary."""
    
    lines = [
        "# Phase 4a Validation Report",
        f"\n**Generated**: {results.get('validation_time', 'N/A')}",
        f"\n**Manifests S3**: `{results.get('manifests_s3', 'N/A')}`",
        "\n## Summary\n",
    ]
    
    summary = results.get("summary", {})
    lines.append(f"- **Total Experiments**: {summary.get('total_experiments', 0)}")
    lines.append(f"- **Total Tasks**: {summary.get('total_tasks', 0):,}")
    lines.append(f"- **Total Issues**: {summary.get('total_issues', 0)}")
    
    if summary.get("issues"):
        lines.append("\n### Issues\n")
        for issue in summary["issues"]:
            lines.append(f"- {issue}")
    
    lines.append("\n## Experiment Details\n")
    
    for exp_id, exp_data in results.get("experiments", {}).items():
        lines.append(f"\n### {exp_id}\n")
        
        if "error" in exp_data:
            lines.append(f"**ERROR**: {exp_data['error']}")
            continue
        
        lines.append(f"- **Rows**: {exp_data.get('row_count', 0):,}")
        lines.append(f"- **Valid**: {exp_data.get('valid', False)}")
        
        if exp_data.get("split_distribution"):
            lines.append(f"- **Splits**: {exp_data['split_distribution']}")
        
        if exp_data.get("n_unique_bricks"):
            lines.append(f"- **Unique Bricks**: {exp_data['n_unique_bricks']:,}")
        
        if exp_data.get("n_unique_regions"):
            lines.append(f"- **Unique Regions**: {exp_data['n_unique_regions']:,}")
        
        if exp_data.get("issues"):
            lines.append("\n**Issues:**")
            for issue in exp_data["issues"]:
                lines.append(f"  - {issue}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Spark-based Phase 4a validation")
    parser.add_argument("--manifests-s3", required=True, help="S3 path to manifests directory")
    parser.add_argument("--bricks-manifest-s3", required=True, help="S3 path to bricks manifest")
    parser.add_argument("--stage-config-s3", required=True, help="S3 path to stage config JSON")
    parser.add_argument("--output-s3", required=True, help="S3 path for output JSON report")
    parser.add_argument("--output-md-s3", help="S3 path for output markdown summary")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Phase 4a Spark Validation")
    print("=" * 60)
    print(f"Manifests: {args.manifests_s3}")
    print(f"Bricks Manifest: {args.bricks_manifest_s3}")
    print(f"Stage Config: {args.stage_config_s3}")
    print(f"Output: {args.output_s3}")
    
    spark = make_spark()
    
    results = {
        "validation_time": datetime.utcnow().isoformat(),
        "manifests_s3": args.manifests_s3,
    }
    
    # 1. Validate stage config
    config_result = load_stage_config(spark, args.stage_config_s3)
    results["stage_config"] = config_result
    
    # 2. Validate manifests (pass stage_config to get expected control fraction)
    manifest_results = validate_manifests(spark, args, stage_config=config_result)
    results.update(manifest_results)
    
    # 3. Validate bricks manifest
    bricks_result = validate_bricks_manifest(spark, args)
    results["bricks_manifest"] = bricks_result
    
    # 4. Write results
    write_results(spark, results, args.output_s3)
    
    # 5. Write markdown summary
    if args.output_md_s3:
        md_content = generate_summary_markdown(results)
        parts = args.output_md_s3.replace("s3://", "").split("/", 1)
        import boto3
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=parts[0],
            Key=parts[1],
            Body=md_content.encode("utf-8"),
            ContentType="text/markdown",
        )
        print(f"Markdown summary written to: {args.output_md_s3}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    total_issues = results.get("summary", {}).get("total_issues", 0)
    print(f"Total issues: {total_issues}")
    
    spark.stop()
    
    sys.exit(0 if total_issues == 0 else 1)


if __name__ == "__main__":
    main()


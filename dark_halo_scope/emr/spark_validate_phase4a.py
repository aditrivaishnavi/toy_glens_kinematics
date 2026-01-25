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


def make_spark(app_name: str = "Phase4aValidation") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.parquet.mergeSchema", "false")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def validate_manifests(spark: SparkSession, args) -> dict:
    """Validate all manifest experiments."""
    
    results = {
        "validation_time": datetime.utcnow().isoformat(),
        "manifests_s3": args.manifests_s3,
        "experiments": {},
        "summary": {},
    }
    
    # List experiments (subdirectories under manifests/)
    experiments = ["debug", "grid", "train_test", "train_val", "train_train"]
    
    total_tasks = 0
    all_issues = []
    
    for exp in experiments:
        exp_path = f"{args.manifests_s3}/{exp}"
        print(f"\n{'='*60}")
        print(f"Validating experiment: {exp}")
        print(f"Path: {exp_path}")
        print(f"{'='*60}")
        
        try:
            df = spark.read.parquet(exp_path)
            df.cache()
            
            exp_result = validate_single_manifest(df, exp)
            results["experiments"][exp] = exp_result
            total_tasks += exp_result.get("row_count", 0)
            all_issues.extend(exp_result.get("issues", []))
            
            df.unpersist()
            
        except Exception as e:
            error_msg = f"Failed to read {exp}: {str(e)}"
            print(f"  ERROR: {error_msg}")
            results["experiments"][exp] = {"error": error_msg}
    
    # Summary
    results["summary"] = {
        "total_experiments": len(experiments),
        "total_tasks": total_tasks,
        "total_issues": len(all_issues),
        "issues": all_issues[:20],  # First 20 issues
    }
    
    return results


def validate_single_manifest(df, experiment_id: str) -> dict:
    """Validate a single manifest DataFrame using Spark aggregations."""
    
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
    if "theta_e_arcsec" in columns:
        theta_stats = df.agg(
            F.min("theta_e_arcsec").alias("min"),
            F.max("theta_e_arcsec").alias("max"),
            F.mean("theta_e_arcsec").alias("mean"),
            F.stddev("theta_e_arcsec").alias("std"),
        ).collect()[0]
        result["theta_e_stats"] = {
            "min": float(theta_stats["min"]) if theta_stats["min"] else None,
            "max": float(theta_stats["max"]) if theta_stats["max"] else None,
            "mean": float(theta_stats["mean"]) if theta_stats["mean"] else None,
            "std": float(theta_stats["std"]) if theta_stats["std"] else None,
        }
        print(f"  Theta_e: min={theta_stats['min']:.3f}, max={theta_stats['max']:.3f}, mean={theta_stats['mean']:.3f}")
        
        neg_count = df.filter(F.col("theta_e_arcsec") < 0).count()
        if neg_count > 0:
            result["issues"].append(f"Negative theta_e values: {neg_count}")
    
    # 11. Control fraction (for train tiers)
    if "is_control" in columns:
        control_counts = df.groupBy("is_control").count().collect()
        control_dist = {str(row["is_control"]): row["count"] for row in control_counts}
        result["control_distribution"] = control_dist
        total = sum(control_dist.values())
        if total > 0:
            ctrl = control_dist.get("1", control_dist.get("True", 0))
            result["control_fraction"] = ctrl / total
        print(f"  Control distribution: {control_dist}")
    
    # 12. Config coverage
    if "config_id" in columns:
        n_configs = df.select("config_id").distinct().count()
        result["n_unique_configs"] = n_configs
        print(f"  Unique configs: {n_configs}")
    
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
    """Validate the bricks manifest."""
    
    print(f"\n{'='*60}")
    print("Validating bricks manifest")
    print(f"Path: {args.bricks_manifest_s3}")
    print(f"{'='*60}")
    
    result = {"issues": [], "warnings": []}
    
    try:
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
        
    except Exception as e:
        result["error"] = str(e)
        result["valid"] = False
        print(f"  ERROR: {e}")
    
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
    
    # 2. Validate manifests
    manifest_results = validate_manifests(spark, args)
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


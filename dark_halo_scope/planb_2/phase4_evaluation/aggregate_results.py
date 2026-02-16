#!/usr/bin/env python3
"""
Phase 4.1: Aggregate all ablation results into master comparison table.

Usage:
    python aggregate_results.py \
        --checkpoint-dir checkpoints/ \
        --output-csv results/master_comparison.csv \
        --output-json results/master_comparison.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np


def load_validation_results(checkpoint_dir: Path) -> Dict[str, Any]:
    """Load validation results from a checkpoint directory."""
    validation_file = checkpoint_dir / "validation_results.json"
    
    if validation_file.exists():
        with open(validation_file) as f:
            return json.load(f)
    
    # Try to find any JSON file with results
    json_files = list(checkpoint_dir.glob("*_validation.json"))
    if json_files:
        with open(json_files[0]) as f:
            return json.load(f)
    
    return None


def aggregate_all_results(
    checkpoint_base: str,
    variants: List[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Aggregate results from all ablation variants.
    
    Returns:
        DataFrame with comparison table
    """
    checkpoint_base = Path(checkpoint_base)
    
    # Default variants if not specified
    if variants is None:
        variants = [
            "gen5_prime_baseline",
            "ablate_no_hardneg",
            "ablate_no_coredrop",
            "ablate_minimal",
            "gen7_hybrid",
            "gen8_domain_rand",
        ]
    
    results = []
    
    for variant in variants:
        variant_dir = checkpoint_base / variant
        
        if not variant_dir.exists():
            if verbose:
                print(f"WARNING: {variant_dir} not found, skipping")
            continue
        
        validation = load_validation_results(variant_dir)
        
        if validation is None:
            if verbose:
                print(f"WARNING: No validation results for {variant}")
            continue
        
        row = {
            "variant": variant,
        }
        
        # Extract metrics
        if "metrics" in validation:
            row.update(validation["metrics"])
        
        # Extract gate status
        if "gates" in validation:
            for gate_name, gate_result in validation["gates"].items():
                row[f"gate_{gate_name}"] = gate_result.get("passed", False)
        
        results.append(row)
    
    if not results:
        if verbose:
            print("ERROR: No results found")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Compute deltas vs baseline
    baseline_name = "gen5_prime_baseline"
    if baseline_name in df.variant.values:
        baseline = df[df.variant == baseline_name].iloc[0]
        
        for col in ["auroc_synth", "auroc_core_masked", "hardneg_auroc"]:
            if col in df.columns:
                df[f"delta_{col}"] = df[col] - baseline[col]
    
    if verbose:
        print("\n" + "="*60)
        print("AGGREGATED RESULTS")
        print("="*60)
        print(df.to_markdown(index=False))
    
    return df


def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics from results."""
    summary = {
        "n_variants": len(df),
        "variants": df.variant.tolist(),
    }
    
    # Best variant by each metric
    for col in ["auroc_synth", "auroc_core_masked", "hardneg_auroc"]:
        if col in df.columns:
            best_idx = df[col].idxmax()
            summary[f"best_{col}"] = {
                "variant": df.loc[best_idx, "variant"],
                "value": float(df.loc[best_idx, col]),
            }
    
    # Gate pass rates
    gate_cols = [c for c in df.columns if c.startswith("gate_")]
    if gate_cols:
        summary["gate_pass_rate"] = {}
        for col in gate_cols:
            summary["gate_pass_rate"][col] = float(df[col].mean())
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Aggregate ablation results")
    parser.add_argument("--checkpoint-dir", required=True, help="Base checkpoint directory")
    parser.add_argument("--variants", nargs="+", help="Specific variants to include")
    parser.add_argument("--output-csv", help="Save comparison table to CSV")
    parser.add_argument("--output-json", help="Save full results to JSON")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    
    df = aggregate_all_results(
        args.checkpoint_dir,
        variants=args.variants,
        verbose=not args.quiet
    )
    
    if df.empty:
        sys.exit(1)
    
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved to: {args.output_csv}")
    
    if args.output_json:
        summary = generate_summary_stats(df)
        summary["comparison_table"] = df.to_dict(orient="records")
        
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to: {args.output_json}")


if __name__ == "__main__":
    main()

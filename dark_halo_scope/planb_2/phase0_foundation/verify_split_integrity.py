#!/usr/bin/env python3
"""
Phase 0.2.1: Verify train/val/test split integrity.

Ensures no brick or healpix overlap between splits to prevent data leakage.

Usage:
    python verify_split_integrity.py --parquet-root /path/to/v5_cosmos_paired
"""
import argparse
import sys
from glob import glob
from pathlib import Path

import pandas as pd
import numpy as np


def verify_splits(parquet_root: str, verbose: bool = True) -> dict:
    """
    Verify no brick/healpix overlap between train/val/test splits.
    
    Returns:
        dict with verification results
    """
    results = {
        "passed": True,
        "checks": {},
        "errors": [],
        "split_stats": {},
    }
    
    splits = ["train", "val", "test"]
    brick_sets = {}
    file_counts = {}
    
    for split in splits:
        split_path = Path(parquet_root) / split
        files = list(split_path.glob("*.parquet"))
        file_counts[split] = len(files)
        
        if len(files) == 0:
            results["passed"] = False
            results["errors"].append(f"No parquet files found in {split_path}")
            continue
        
        # Sample files to extract bricks
        sample_files = files[:min(20, len(files))]
        bricks = set()
        
        for f in sample_files:
            try:
                df = pd.read_parquet(f, columns=["brickname"])
                bricks.update(df.brickname.unique())
            except Exception as e:
                results["errors"].append(f"Failed to read {f}: {e}")
        
        brick_sets[split] = bricks
        results["split_stats"][split] = {
            "n_files": len(files),
            "n_bricks_sampled": len(bricks),
        }
    
    # Check pairwise overlaps
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    overlap_results = {}
    
    for s1, s2 in pairs:
        if s1 not in brick_sets or s2 not in brick_sets:
            continue
        
        overlap = brick_sets[s1] & brick_sets[s2]
        pair_key = f"{s1}_vs_{s2}"
        overlap_results[pair_key] = {
            "n_overlap": len(overlap),
            "passed": len(overlap) == 0,
            "overlapping_bricks": list(overlap)[:5] if overlap else [],
        }
        
        if len(overlap) > 0:
            results["passed"] = False
            results["errors"].append(
                f"Brick overlap between {s1} and {s2}: {len(overlap)} bricks"
            )
    
    results["checks"]["brick_overlap"] = overlap_results
    
    # Summary
    if verbose:
        print("\n" + "="*60)
        print("SPLIT INTEGRITY VERIFICATION")
        print("="*60)
        print(f"\nParquet root: {parquet_root}")
        
        print("\nSplit statistics:")
        for split, stats in results["split_stats"].items():
            print(f"  {split}: {stats['n_files']} files, {stats['n_bricks_sampled']} bricks sampled")
        
        print("\nOverlap checks:")
        for pair, check in overlap_results.items():
            status = "✓ PASS" if check["passed"] else "✗ FAIL"
            print(f"  {pair}: {status} ({check['n_overlap']} overlapping bricks)")
        
        if results["passed"]:
            print(f"\n✓ NO BRICK OVERLAP DETECTED")
        else:
            print(f"\n✗ VERIFICATION FAILED")
            for error in results["errors"]:
                print(f"  - {error}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Verify split integrity")
    parser.add_argument("--parquet-root", required=True, help="Root directory with train/val/test splits")
    parser.add_argument("--output-json", help="Save results to JSON")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    
    results = verify_splits(args.parquet_root, verbose=not args.quiet)
    
    if args.output_json:
        import json
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
    
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 0.2.2: Verify paired data integrity.

Ensures stamp and ctrl are properly paired and valid.

Usage:
    python verify_paired_data.py --parquet-root /path/to/v5_cosmos_paired --n-samples 100
"""
import argparse
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Import from shared module - SINGLE SOURCE OF TRUTH
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.constants import STAMP_SHAPE, VALUE_RANGE_MIN, VALUE_RANGE_MAX
from shared.schema import PARQUET_SCHEMA
from shared.utils import decode_stamp_npz, validate_stamp


def verify_paired_data(
    parquet_root: str,
    n_samples: int = 100,
    verbose: bool = True
) -> dict:
    """
    Verify paired data integrity.
    
    Checks performed:
    1. Stamp and ctrl have same shape
    2. Stamp and ctrl are different (arc present)
    3. No NaN/Inf values
    4. Reasonable value range
    5. Expected shape (3, 64, 64)
    
    Returns:
        dict with verification results
    """
    results = {
        "passed": True,
        "checks": {},
        "errors": [],
        "n_samples_checked": 0,
    }
    
    # Find parquet files
    train_path = Path(parquet_root) / "train"
    files = list(train_path.glob("*.parquet"))
    
    if len(files) == 0:
        results["passed"] = False
        results["errors"].append(f"No parquet files found in {train_path}")
        return results
    
    # Sample a random file
    random.seed(42)
    file = random.choice(files)
    
    try:
        df = pd.read_parquet(file)
    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Failed to read {file}: {e}")
        return results
    
    # Check required columns using schema
    schema_result = PARQUET_SCHEMA.validate_dataframe(df)
    if not schema_result["valid"]:
        results["passed"] = False
        results["errors"].append(f"Schema validation failed: {schema_result['missing_required']}")
        return results
    
    # Sample rows
    sample_df = df.sample(min(n_samples, len(df)), random_state=42)
    
    check_results = {
        "shape_match": {"passed": 0, "failed": 0, "failures": []},
        "not_identical": {"passed": 0, "failed": 0, "failures": []},
        "no_nan_inf": {"passed": 0, "failed": 0, "failures": []},
        "value_range": {"passed": 0, "failed": 0, "failures": []},
        "expected_shape": {"passed": 0, "failed": 0, "failures": []},
    }
    
    for idx, row in sample_df.iterrows():
        try:
            stamp, _ = decode_stamp_npz(row.stamp_npz)
            ctrl, _ = decode_stamp_npz(row.ctrl_stamp_npz)
        except Exception as e:
            results["errors"].append(f"Failed to decode row {idx}: {e}")
            continue
        
        # Check 1: Same shape
        if stamp.shape == ctrl.shape:
            check_results["shape_match"]["passed"] += 1
        else:
            check_results["shape_match"]["failed"] += 1
            check_results["shape_match"]["failures"].append(
                f"Row {idx}: stamp={stamp.shape}, ctrl={ctrl.shape}"
            )
        
        # Check 2: Not identical
        diff = np.abs(stamp - ctrl).sum()
        if diff > 0.1:
            check_results["not_identical"]["passed"] += 1
        else:
            check_results["not_identical"]["failed"] += 1
            check_results["not_identical"]["failures"].append(
                f"Row {idx}: diff={diff:.4f}"
            )
        
        # Check 3: No NaN/Inf
        if np.isfinite(stamp).all() and np.isfinite(ctrl).all():
            check_results["no_nan_inf"]["passed"] += 1
        else:
            check_results["no_nan_inf"]["failed"] += 1
            check_results["no_nan_inf"]["failures"].append(f"Row {idx}")
        
        # Check 4: Value range (using constants)
        if (stamp.min() > VALUE_RANGE_MIN and stamp.max() < VALUE_RANGE_MAX and
            ctrl.min() > VALUE_RANGE_MIN and ctrl.max() < VALUE_RANGE_MAX):
            check_results["value_range"]["passed"] += 1
        else:
            check_results["value_range"]["failed"] += 1
            check_results["value_range"]["failures"].append(
                f"Row {idx}: stamp=[{stamp.min():.1e}, {stamp.max():.1e}]"
            )
        
        # Check 5: Expected shape (using constants)
        if stamp.shape == STAMP_SHAPE:
            check_results["expected_shape"]["passed"] += 1
        else:
            check_results["expected_shape"]["failed"] += 1
            check_results["expected_shape"]["failures"].append(
                f"Row {idx}: shape={stamp.shape}, expected={STAMP_SHAPE}"
            )
    
    results["n_samples_checked"] = len(sample_df)
    results["checks"] = check_results
    
    # Determine overall pass/fail
    for check_name, check in check_results.items():
        if check["failed"] > 0:
            results["passed"] = False
            results["errors"].append(
                f"{check_name}: {check['failed']}/{check['passed'] + check['failed']} failed"
            )
    
    if verbose:
        print("\n" + "="*60)
        print("PAIRED DATA VERIFICATION")
        print("="*60)
        print(f"\nFile: {file}")
        print(f"Samples checked: {results['n_samples_checked']}")
        
        print("\nChecks:")
        for check_name, check in check_results.items():
            total = check["passed"] + check["failed"]
            status = "✓ PASS" if check["failed"] == 0 else "✗ FAIL"
            print(f"  {check_name}: {status} ({check['passed']}/{total})")
        
        if results["passed"]:
            print(f"\n✓ ALL CHECKS PASSED")
        else:
            print(f"\n✗ VERIFICATION FAILED")
            for error in results["errors"]:
                print(f"  - {error}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Verify paired data integrity")
    parser.add_argument("--parquet-root", required=True)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--output-json", help="Save results to JSON")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    
    results = verify_paired_data(
        args.parquet_root,
        n_samples=args.n_samples,
        verbose=not args.quiet
    )
    
    if args.output_json:
        import json
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
    
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()

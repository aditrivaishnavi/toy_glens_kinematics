#!/usr/bin/env python3
"""
Phase 0.1.1: Validate anchor lens catalog.

This script verifies the anchor set is complete and valid before any training.
All checks must pass before proceeding to Phase 1.

Usage:
    python validate_anchors.py --anchor-csv anchors/tier_a_anchors.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Import from shared module - SINGLE SOURCE OF TRUTH
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.constants import MIN_ANCHORS, MIN_THETA_E_ARCSEC
from shared.schema import ANCHOR_SCHEMA


def get_brick_for_coords(ra: float, dec: float) -> str:
    """Get DR10 brick name for given coordinates."""
    # DR10 brick naming convention
    # brickname = DDDDsMDD where DDDD is RA/10, s is sign, MDD is abs(dec)
    ra_idx = int(np.floor(ra / 0.25)) * 0.25
    dec_idx = int(np.floor(dec / 0.25)) * 0.25
    
    if dec_idx >= 0:
        sign = 'p'
    else:
        sign = 'm'
    
    brickname = f"{int(ra_idx*10):04d}{sign}{int(abs(dec_idx)*10):03d}"
    return brickname


def check_dr10_coverage(ra: float, dec: float) -> bool:
    """Check if coordinates have DR10 coverage."""
    # DR10 roughly covers dec > -20 and various RA ranges
    # This is a simplified check - production should query actual footprint
    if dec < -20:
        return False
    return True


def validate_anchors(anchor_csv: str, verbose: bool = True) -> dict:
    """
    Validate anchor catalog with all required checks.
    
    Checks performed:
    1. Minimum count (>= 30 anchors)
    2. Required columns present
    3. No duplicate names
    4. Theta_e >= 0.5" (detectable at DR10 resolution)
    5. DR10 coverage for sample
    
    Returns:
        dict with validation results
    """
    results = {
        "passed": True,
        "checks": {},
        "n_anchors": 0,
        "errors": [],
    }
    
    # Load catalog
    try:
        df = pd.read_csv(anchor_csv)
    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Failed to load CSV: {e}")
        return results
    
    results["n_anchors"] = len(df)
    
    # Check 1: Minimum count (using constant)
    check1_pass = len(df) >= MIN_ANCHORS
    results["checks"]["min_count"] = {
        "passed": check1_pass,
        "value": len(df),
        "threshold": MIN_ANCHORS,
    }
    if not check1_pass:
        results["passed"] = False
        results["errors"].append(f"Need >= {MIN_ANCHORS} anchors, have {len(df)}")
    
    # Check 2: Required columns
    required = ["name", "ra", "dec", "theta_e_arcsec", "source"]
    missing = [col for col in required if col not in df.columns]
    check2_pass = len(missing) == 0
    results["checks"]["required_columns"] = {
        "passed": check2_pass,
        "missing": missing,
    }
    if not check2_pass:
        results["passed"] = False
        results["errors"].append(f"Missing columns: {missing}")
        return results  # Cannot continue without required columns
    
    # Check 3: No duplicates
    n_unique = df.name.nunique()
    check3_pass = n_unique == len(df)
    results["checks"]["no_duplicates"] = {
        "passed": check3_pass,
        "unique_names": n_unique,
        "total_rows": len(df),
    }
    if not check3_pass:
        results["passed"] = False
        duplicates = df[df.name.duplicated()].name.tolist()
        results["errors"].append(f"Duplicate names: {duplicates[:5]}")
    
    # Check 4: Theta_e range (using constant)
    small_theta = df[df.theta_e_arcsec < MIN_THETA_E_ARCSEC]
    check4_pass = len(small_theta) == 0
    results["checks"]["theta_e_range"] = {
        "passed": check4_pass,
        "n_below_threshold": len(small_theta),
        "threshold": MIN_THETA_E_ARCSEC,
        "min_theta_e": float(df.theta_e_arcsec.min()),
        "max_theta_e": float(df.theta_e_arcsec.max()),
    }
    if not check4_pass:
        results["passed"] = False
        results["errors"].append(f"{len(small_theta)} anchors have theta_e < {MIN_THETA_E_ARCSEC}\"")
    
    # Check 5: DR10 coverage (sample 10)
    sample_size = min(10, len(df))
    sample_df = df.sample(sample_size, random_state=42)
    coverage_checks = []
    for _, row in sample_df.iterrows():
        has_coverage = check_dr10_coverage(row.ra, row.dec)
        coverage_checks.append({
            "name": row["name"],
            "ra": row.ra,
            "dec": row.dec,
            "has_coverage": has_coverage,
        })
    
    n_covered = sum(c["has_coverage"] for c in coverage_checks)
    check5_pass = n_covered == sample_size
    results["checks"]["dr10_coverage"] = {
        "passed": check5_pass,
        "n_checked": sample_size,
        "n_covered": n_covered,
        "details": coverage_checks,
    }
    if not check5_pass:
        results["passed"] = False
        missing_coverage = [c["name"] for c in coverage_checks if not c["has_coverage"]]
        results["errors"].append(f"Missing DR10 coverage: {missing_coverage}")
    
    # Summary statistics
    results["summary"] = {
        "n_anchors": len(df),
        "sources": df.source.value_counts().to_dict(),
        "theta_e_median": float(df.theta_e_arcsec.median()),
        "theta_e_range": [float(df.theta_e_arcsec.min()), float(df.theta_e_arcsec.max())],
    }
    
    if verbose:
        print("\n" + "="*60)
        print("ANCHOR VALIDATION RESULTS")
        print("="*60)
        print(f"\nFile: {anchor_csv}")
        print(f"Total anchors: {len(df)}")
        print(f"\nChecks:")
        for check_name, check_result in results["checks"].items():
            status = "✓ PASS" if check_result["passed"] else "✗ FAIL"
            print(f"  {check_name}: {status}")
        
        if results["passed"]:
            print(f"\n✓ ALL CHECKS PASSED")
        else:
            print(f"\n✗ VALIDATION FAILED")
            for error in results["errors"]:
                print(f"  - {error}")
        
        print("\nSummary:")
        print(f"  Sources: {results['summary']['sources']}")
        print(f"  Theta_e median: {results['summary']['theta_e_median']:.2f}\"")
        print(f"  Theta_e range: {results['summary']['theta_e_range'][0]:.2f}\" - {results['summary']['theta_e_range'][1]:.2f}\"")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate anchor lens catalog")
    parser.add_argument("--anchor-csv", required=True, help="Path to anchor CSV file")
    parser.add_argument("--output-json", help="Save results to JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()
    
    results = validate_anchors(args.anchor_csv, verbose=not args.quiet)
    
    if args.output_json:
        import json
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")
    
    # Exit with error code if validation failed
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()

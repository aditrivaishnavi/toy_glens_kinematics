#!/usr/bin/env python3
"""
Phase 0.1.2: Validate contaminant catalog.

Contaminants are known non-lenses used to measure false positive rates.
Categories: ring galaxies, face-on spirals, mergers, diffraction spikes.

Usage:
    python validate_contaminants.py --contaminant-csv contaminants/catalog.csv --anchor-csv anchors/tier_a.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Import from shared module - SINGLE SOURCE OF TRUTH
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.constants import (
    MIN_RING_CONTAMINANTS, MIN_SPIRAL_CONTAMINANTS, MIN_MERGER_CONTAMINANTS
)
from shared.schema import CONTAMINANT_SCHEMA


def validate_contaminants(
    contaminant_csv: str,
    anchor_csv: str = None,
    verbose: bool = True
) -> dict:
    """
    Validate contaminant catalog.
    
    Checks performed:
    1. Required columns present
    2. Category distribution meets minimum requirements
    3. No overlap with anchor set
    4. Coordinates are valid
    
    Returns:
        dict with validation results
    """
    results = {
        "passed": True,
        "checks": {},
        "n_contaminants": 0,
        "errors": [],
    }
    
    # Load contaminant catalog
    try:
        df = pd.read_csv(contaminant_csv)
    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Failed to load contaminant CSV: {e}")
        return results
    
    results["n_contaminants"] = len(df)
    
    # Check 1: Required columns
    required = ["name", "ra", "dec", "category"]
    missing = [col for col in required if col not in df.columns]
    check1_pass = len(missing) == 0
    results["checks"]["required_columns"] = {
        "passed": check1_pass,
        "missing": missing,
    }
    if not check1_pass:
        results["passed"] = False
        results["errors"].append(f"Missing columns: {missing}")
        return results
    
    # Check 2: Category distribution (using constants)
    category_counts = df.category.value_counts().to_dict()
    min_requirements = {
        "ring": MIN_RING_CONTAMINANTS,
        "spiral": MIN_SPIRAL_CONTAMINANTS,
        "merger": MIN_MERGER_CONTAMINANTS,
    }
    
    category_issues = []
    for cat, min_count in min_requirements.items():
        actual = category_counts.get(cat, 0)
        if actual < min_count:
            category_issues.append(f"{cat}: {actual}/{min_count}")
    
    check2_pass = len(category_issues) == 0
    results["checks"]["category_distribution"] = {
        "passed": check2_pass,
        "counts": category_counts,
        "requirements": min_requirements,
        "issues": category_issues,
    }
    if not check2_pass:
        results["passed"] = False
        results["errors"].append(f"Category requirements not met: {category_issues}")
    
    # Check 3: No overlap with anchors
    if anchor_csv:
        try:
            anchor_df = pd.read_csv(anchor_csv)
            
            # Create coordinate sets (rounded to 4 decimal places for matching)
            anchor_coords = set(zip(
                anchor_df.ra.round(4),
                anchor_df.dec.round(4)
            ))
            contam_coords = set(zip(
                df.ra.round(4),
                df.dec.round(4)
            ))
            
            overlap = anchor_coords & contam_coords
            check3_pass = len(overlap) == 0
            results["checks"]["no_anchor_overlap"] = {
                "passed": check3_pass,
                "n_overlap": len(overlap),
                "overlapping_coords": list(overlap)[:5] if overlap else [],
            }
            if not check3_pass:
                results["passed"] = False
                results["errors"].append(f"{len(overlap)} contaminants overlap with anchors")
        except Exception as e:
            results["checks"]["no_anchor_overlap"] = {
                "passed": False,
                "error": str(e),
            }
    else:
        results["checks"]["no_anchor_overlap"] = {
            "passed": True,
            "note": "No anchor file provided, skipping check",
        }
    
    # Check 4: Valid coordinates
    valid_ra = (df.ra >= 0) & (df.ra < 360)
    valid_dec = (df.dec >= -90) & (df.dec <= 90)
    n_invalid = (~valid_ra | ~valid_dec).sum()
    
    check4_pass = n_invalid == 0
    results["checks"]["valid_coordinates"] = {
        "passed": check4_pass,
        "n_invalid": int(n_invalid),
    }
    if not check4_pass:
        results["passed"] = False
        results["errors"].append(f"{n_invalid} contaminants have invalid coordinates")
    
    # Summary
    results["summary"] = {
        "n_contaminants": len(df),
        "categories": category_counts,
    }
    
    if verbose:
        print("\n" + "="*60)
        print("CONTAMINANT VALIDATION RESULTS")
        print("="*60)
        print(f"\nFile: {contaminant_csv}")
        print(f"Total contaminants: {len(df)}")
        print(f"\nCategory distribution:")
        for cat, count in category_counts.items():
            req = min_requirements.get(cat, "-")
            print(f"  {cat}: {count} (required: {req})")
        
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
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate contaminant catalog")
    parser.add_argument("--contaminant-csv", required=True, help="Path to contaminant CSV")
    parser.add_argument("--anchor-csv", help="Path to anchor CSV for overlap check")
    parser.add_argument("--output-json", help="Save results to JSON")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    
    results = validate_contaminants(
        args.contaminant_csv,
        args.anchor_csv,
        verbose=not args.quiet
    )
    
    if args.output_json:
        import json
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
    
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()

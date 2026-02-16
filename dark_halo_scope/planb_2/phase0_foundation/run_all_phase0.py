#!/usr/bin/env python3
"""
Phase 0: Run all foundation checks.

This script runs ALL Phase 0 validation checks and reports a summary.
All checks must pass before proceeding to Phase 1 training.

Usage:
    python run_all_phase0.py \
        --parquet-root /path/to/v5_cosmos_paired \
        --anchor-csv anchors/tier_a_anchors.csv \
        --contaminant-csv contaminants/catalog.csv \
        --output-dir phase0_results
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_check(name: str, cmd: list, output_dir: str) -> dict:
    """Run a single check and capture results."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    result = {
        "name": name,
        "command": " ".join(cmd),
        "passed": False,
        "output": "",
        "error": "",
    }
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        result["output"] = proc.stdout
        result["error"] = proc.stderr
        result["exit_code"] = proc.returncode
        result["passed"] = proc.returncode == 0
        
        print(proc.stdout)
        if proc.stderr:
            print(f"STDERR: {proc.stderr}")
        
    except subprocess.TimeoutExpired:
        result["error"] = "Timeout after 300 seconds"
        print(f"TIMEOUT: {name}")
    except Exception as e:
        result["error"] = str(e)
        print(f"ERROR: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Run all Phase 0 checks")
    parser.add_argument("--parquet-root", required=True, help="Path to paired parquet data")
    parser.add_argument("--anchor-csv", help="Path to anchor CSV")
    parser.add_argument("--contaminant-csv", help="Path to contaminant CSV")
    parser.add_argument("--output-dir", default="phase0_results", help="Output directory")
    parser.add_argument("--skip-missing", action="store_true", help="Skip checks if files missing")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Define all checks
    checks = []
    
    # Check 1: Anchor validation
    if args.anchor_csv and Path(args.anchor_csv).exists():
        checks.append({
            "name": "anchor_validation",
            "cmd": [
                sys.executable,
                str(script_dir / "validate_anchors.py"),
                "--anchor-csv", args.anchor_csv,
                "--output-json", str(output_dir / "anchor_validation.json"),
            ],
        })
    elif not args.skip_missing:
        print(f"WARNING: Anchor CSV not provided or not found: {args.anchor_csv}")
    
    # Check 2: Contaminant validation
    if args.contaminant_csv and Path(args.contaminant_csv).exists():
        cmd = [
            sys.executable,
            str(script_dir / "validate_contaminants.py"),
            "--contaminant-csv", args.contaminant_csv,
            "--output-json", str(output_dir / "contaminant_validation.json"),
        ]
        if args.anchor_csv and Path(args.anchor_csv).exists():
            cmd.extend(["--anchor-csv", args.anchor_csv])
        checks.append({
            "name": "contaminant_validation",
            "cmd": cmd,
        })
    elif not args.skip_missing:
        print(f"WARNING: Contaminant CSV not provided or not found: {args.contaminant_csv}")
    
    # Check 3: Split integrity
    checks.append({
        "name": "split_integrity",
        "cmd": [
            sys.executable,
            str(script_dir / "verify_split_integrity.py"),
            "--parquet-root", args.parquet_root,
            "--output-json", str(output_dir / "split_integrity.json"),
        ],
    })
    
    # Check 4: Paired data integrity
    checks.append({
        "name": "paired_data_integrity",
        "cmd": [
            sys.executable,
            str(script_dir / "verify_paired_data.py"),
            "--parquet-root", args.parquet_root,
            "--n-samples", "100",
            "--output-json", str(output_dir / "paired_data_integrity.json"),
        ],
    })
    
    # Check 5: Data loading test
    test_dir = script_dir.parent / "tests"
    checks.append({
        "name": "data_loading_test",
        "cmd": [
            sys.executable,
            str(test_dir / "test_data_loading.py"),
            "--parquet-root", args.parquet_root,
        ],
    })
    
    # Run all checks
    results = []
    for check in checks:
        result = run_check(check["name"], check["cmd"], str(output_dir))
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("PHASE 0 SUMMARY")
    print("="*60)
    
    n_passed = sum(1 for r in results if r["passed"])
    n_total = len(results)
    
    for result in results:
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"  {result['name']}: {status}")
    
    print(f"\nTotal: {n_passed}/{n_total} checks passed")
    
    all_passed = n_passed == n_total
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ ALL PHASE 0 CHECKS PASSED")
        print("="*60)
        print("\nYou may proceed to Phase 1: Baseline Training")
    else:
        print("\n" + "="*60)
        print("✗ PHASE 0 FAILED")
        print("="*60)
        print("\nFix the failing checks before proceeding.")
        print("See individual JSON files in output directory for details.")
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "parquet_root": args.parquet_root,
        "anchor_csv": args.anchor_csv,
        "contaminant_csv": args.contaminant_csv,
        "n_passed": n_passed,
        "n_total": n_total,
        "all_passed": all_passed,
        "checks": [
            {"name": r["name"], "passed": r["passed"]}
            for r in results
        ],
    }
    
    with open(output_dir / "phase0_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

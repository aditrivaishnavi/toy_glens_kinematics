#!/usr/bin/env python3
"""
Post-run validation script for EMR negative sampling output.
Run this after EMR job completes to verify output quality.

Usage:
    # Validate local output
    python3 scripts/validate_output.py --input output/full_run/
    
    # Validate S3 output (downloads sample)
    python3 scripts/validate_output.py --s3 s3://darkhaloscope/stronglens_calibration/manifests/TIMESTAMP/
    
    # Strict mode (fail on any warning)
    python3 scripts/validate_output.py --input output/ --strict
"""

import argparse
import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def check_mark(passed: bool) -> str:
    return f"{Colors.GREEN}✓{Colors.END}" if passed else f"{Colors.RED}✗{Colors.END}"

def warn_mark() -> str:
    return f"{Colors.YELLOW}⚠{Colors.END}"

def print_header(title: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")


@dataclass
class ValidationResult:
    name: str
    passed: bool
    critical: bool
    message: str
    details: Optional[Dict] = None


class OutputValidator:
    """Validates EMR negative sampling output."""
    
    # Expected schema columns
    CRITICAL_COLUMNS = ['galaxy_id', 'ra', 'dec', 'type', 'nobs_z', 'pool', 'split']
    EXPECTED_COLUMNS = [
        'galaxy_id', 'ra', 'dec', 'type', 'type_bin', 
        'nobs_z', 'nobs_z_bin', 'pool', 'pool_subtype',
        'split', 'healpix_128', 'brickid', 'objid',
        'flux_g', 'flux_r', 'flux_z', 'flux_w1', 'flux_w2',
        'shape_r', 'shape_e1', 'shape_e2',
        'psfsize_g', 'psfsize_r', 'psfsize_z',
        'psfdepth_g', 'psfdepth_r', 'psfdepth_z',
        'maskbits', 'fitbits',
        'pipeline_version', 'git_commit', 'extraction_timestamp'
    ]
    
    # Expected distributions
    EXPECTED_POOL_N1_RATIO = (75, 95)  # 85 ± 10%
    EXPECTED_TRAIN_RATIO = (65, 75)    # 70 ± 5%
    EXPECTED_VAL_RATIO = (10, 20)      # 15 ± 5%
    EXPECTED_TEST_RATIO = (10, 20)     # 15 ± 5%
    
    def __init__(self, input_path: Path, strict: bool = False):
        self.input_path = input_path
        self.strict = strict
        self.results: List[ValidationResult] = []
        self.df: Optional[pd.DataFrame] = None
        
    def load_data(self) -> bool:
        """Load parquet files from input path."""
        try:
            parquet_files = list(self.input_path.glob('*.parquet'))
            if not parquet_files:
                parquet_files = list(self.input_path.glob('**/*.parquet'))
            
            if not parquet_files:
                self.results.append(ValidationResult(
                    'Load data', False, True,
                    'No parquet files found',
                    {'path': str(self.input_path)}
                ))
                return False
            
            print(f"Loading {len(parquet_files)} parquet files...")
            dfs = [pd.read_parquet(f) for f in parquet_files]
            self.df = pd.concat(dfs, ignore_index=True)
            
            self.results.append(ValidationResult(
                'Load data', True, True,
                f'Loaded {len(self.df):,} rows from {len(parquet_files)} files',
                {'files': len(parquet_files), 'rows': len(self.df)}
            ))
            return True
            
        except Exception as e:
            self.results.append(ValidationResult(
                'Load data', False, True,
                f'Error loading data: {e}'
            ))
            return False
    
    def check_row_count(self, min_rows: int = 10000) -> bool:
        """Check minimum row count."""
        row_count = len(self.df)
        passed = row_count >= min_rows
        
        self.results.append(ValidationResult(
            'Row count', passed, True,
            f'{row_count:,} rows' + ('' if passed else f' (need {min_rows:,})'),
            {'count': row_count, 'minimum': min_rows}
        ))
        return passed
    
    def check_schema(self) -> bool:
        """Check all expected columns are present."""
        actual_cols = set(self.df.columns)
        
        # Check critical columns
        missing_critical = [c for c in self.CRITICAL_COLUMNS if c not in actual_cols]
        
        # Check expected columns
        missing_expected = [c for c in self.EXPECTED_COLUMNS if c not in actual_cols]
        extra_cols = [c for c in actual_cols if c not in self.EXPECTED_COLUMNS]
        
        critical_passed = len(missing_critical) == 0
        expected_passed = len(missing_expected) == 0
        
        self.results.append(ValidationResult(
            'Critical columns', critical_passed, True,
            'All present' if critical_passed else f'Missing: {missing_critical}',
            {'missing': missing_critical}
        ))
        
        self.results.append(ValidationResult(
            'Expected columns', expected_passed, False,
            'All present' if expected_passed else f'Missing: {missing_expected}',
            {'missing': missing_expected, 'extra': extra_cols}
        ))
        
        return critical_passed
    
    def check_null_values(self) -> bool:
        """Check for null values in critical columns."""
        null_counts = {}
        for col in self.CRITICAL_COLUMNS:
            if col in self.df.columns:
                null_count = self.df[col].isna().sum()
                if null_count > 0:
                    null_counts[col] = int(null_count)
        
        passed = len(null_counts) == 0
        
        self.results.append(ValidationResult(
            'Null values', passed, True,
            'No nulls in critical columns' if passed else f'Nulls found: {null_counts}',
            {'null_counts': null_counts}
        ))
        return passed
    
    def check_duplicates(self) -> bool:
        """Check for duplicate galaxy IDs."""
        if 'galaxy_id' not in self.df.columns:
            self.results.append(ValidationResult(
                'Duplicates', False, True,
                'galaxy_id column not found'
            ))
            return False
        
        unique_count = self.df['galaxy_id'].nunique()
        total_count = len(self.df)
        dup_count = total_count - unique_count
        dup_pct = (dup_count / total_count) * 100 if total_count > 0 else 0
        
        passed = dup_count == 0
        
        self.results.append(ValidationResult(
            'Duplicates', passed, True,
            f'No duplicates' if passed else f'{dup_count:,} duplicates ({dup_pct:.2f}%)',
            {'duplicates': dup_count, 'unique': unique_count}
        ))
        return passed
    
    def check_pool_distribution(self) -> bool:
        """Check N1/N2 pool distribution is ~85:15."""
        if 'pool' not in self.df.columns:
            self.results.append(ValidationResult(
                'Pool distribution', False, True,
                'pool column not found'
            ))
            return False
        
        pool_dist = self.df['pool'].value_counts(normalize=True) * 100
        n1_pct = pool_dist.get('N1', 0)
        n2_pct = pool_dist.get('N2', 0)
        
        min_ratio, max_ratio = self.EXPECTED_POOL_N1_RATIO
        passed = min_ratio <= n1_pct <= max_ratio
        
        self.results.append(ValidationResult(
            'Pool distribution', passed, False,
            f'N1: {n1_pct:.1f}%, N2: {n2_pct:.1f}%' + 
            ('' if passed else f' (expected N1: {min_ratio}-{max_ratio}%)'),
            {'N1_pct': round(n1_pct, 2), 'N2_pct': round(n2_pct, 2)}
        ))
        return passed
    
    def check_split_distribution(self) -> bool:
        """Check train/val/test split is ~70:15:15."""
        if 'split' not in self.df.columns:
            self.results.append(ValidationResult(
                'Split distribution', False, True,
                'split column not found'
            ))
            return False
        
        split_dist = self.df['split'].value_counts(normalize=True) * 100
        train_pct = split_dist.get('train', 0)
        val_pct = split_dist.get('val', 0)
        test_pct = split_dist.get('test', 0)
        
        train_ok = self.EXPECTED_TRAIN_RATIO[0] <= train_pct <= self.EXPECTED_TRAIN_RATIO[1]
        val_ok = self.EXPECTED_VAL_RATIO[0] <= val_pct <= self.EXPECTED_VAL_RATIO[1]
        test_ok = self.EXPECTED_TEST_RATIO[0] <= test_pct <= self.EXPECTED_TEST_RATIO[1]
        
        passed = train_ok and val_ok and test_ok
        
        self.results.append(ValidationResult(
            'Split distribution', passed, False,
            f'train: {train_pct:.1f}%, val: {val_pct:.1f}%, test: {test_pct:.1f}%',
            {'train': round(train_pct, 2), 'val': round(val_pct, 2), 'test': round(test_pct, 2)}
        ))
        return passed
    
    def check_type_distribution(self) -> bool:
        """Check all expected galaxy types are present."""
        expected_types = {'SER', 'DEV', 'REX', 'EXP', 'PSF', 'DUP'}
        
        if 'type_bin' in self.df.columns:
            actual_types = set(self.df['type_bin'].unique())
        elif 'type' in self.df.columns:
            actual_types = set(self.df['type'].unique())
        else:
            self.results.append(ValidationResult(
                'Type distribution', False, True,
                'type/type_bin column not found'
            ))
            return False
        
        # At least have the main extended types
        main_types = {'SER', 'DEV', 'REX', 'EXP'}
        has_main = len(main_types & actual_types) >= 3  # At least 3 of 4
        
        type_counts = self.df['type_bin' if 'type_bin' in self.df.columns else 'type'].value_counts()
        
        self.results.append(ValidationResult(
            'Type distribution', has_main, False,
            f'{len(actual_types)} types found',
            {'types': {k: int(v) for k, v in type_counts.items()}}
        ))
        return has_main
    
    def check_coordinate_ranges(self) -> bool:
        """Check RA/Dec are within valid ranges."""
        if 'ra' not in self.df.columns or 'dec' not in self.df.columns:
            self.results.append(ValidationResult(
                'Coordinate ranges', False, True,
                'ra/dec columns not found'
            ))
            return False
        
        ra_valid = self.df['ra'].between(0, 360).all()
        dec_valid = self.df['dec'].between(-90, 90).all()
        
        ra_range = (float(self.df['ra'].min()), float(self.df['ra'].max()))
        dec_range = (float(self.df['dec'].min()), float(self.df['dec'].max()))
        
        passed = ra_valid and dec_valid
        
        self.results.append(ValidationResult(
            'Coordinate ranges', passed, True,
            f'RA: [{ra_range[0]:.2f}, {ra_range[1]:.2f}], Dec: [{dec_range[0]:.2f}, {dec_range[1]:.2f}]',
            {'ra_range': ra_range, 'dec_range': dec_range}
        ))
        return passed
    
    def check_provenance(self) -> bool:
        """Check provenance columns are present and populated."""
        prov_cols = ['pipeline_version', 'git_commit', 'extraction_timestamp']
        
        present = [c for c in prov_cols if c in self.df.columns]
        populated = []
        
        for col in present:
            non_null = self.df[col].notna().sum()
            if non_null > len(self.df) * 0.9:  # 90% populated
                populated.append(col)
        
        passed = len(populated) >= 2  # At least 2 of 3
        
        git_commits = self.df['git_commit'].unique().tolist() if 'git_commit' in self.df.columns else []
        
        self.results.append(ValidationResult(
            'Provenance', passed, False,
            f'{len(populated)}/{len(prov_cols)} provenance columns populated',
            {'present': present, 'populated': populated, 'git_commits': git_commits[:5]}
        ))
        return passed
    
    def check_stratification(self) -> bool:
        """Check nobs_z bins are populated."""
        if 'nobs_z_bin' not in self.df.columns:
            self.results.append(ValidationResult(
                'Stratification', False, False,
                'nobs_z_bin column not found'
            ))
            return False
        
        nobs_dist = self.df['nobs_z_bin'].value_counts()
        n_bins = len(nobs_dist)
        
        passed = n_bins >= 2
        
        self.results.append(ValidationResult(
            'Stratification', passed, False,
            f'{n_bins} nobs_z bins found',
            {'bins': {k: int(v) for k, v in nobs_dist.items()}}
        ))
        return passed
    
    def run_all(self) -> Tuple[int, int, int]:
        """Run all validation checks. Returns (passed, failed, warnings)."""
        
        print_header("LOADING DATA")
        if not self.load_data():
            return 0, 1, 0
        
        print_header("VALIDATION CHECKS")
        
        self.check_row_count()
        self.check_schema()
        self.check_null_values()
        self.check_duplicates()
        self.check_pool_distribution()
        self.check_split_distribution()
        self.check_type_distribution()
        self.check_coordinate_ranges()
        self.check_provenance()
        self.check_stratification()
        
        # Print results
        print_header("RESULTS")
        
        passed = 0
        failed = 0
        warnings = 0
        
        for result in self.results:
            if result.passed:
                status = check_mark(True)
                passed += 1
            elif result.critical:
                status = check_mark(False)
                failed += 1
            else:
                status = warn_mark()
                warnings += 1
                if self.strict:
                    failed += 1
            
            critical_str = " [CRITICAL]" if result.critical and not result.passed else ""
            print(f"  {status} {result.name}: {result.message}{critical_str}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"  {Colors.GREEN}Passed: {passed}{Colors.END}")
        if warnings > 0:
            print(f"  {Colors.YELLOW}Warnings: {warnings}{Colors.END}")
        if failed > 0:
            print(f"  {Colors.RED}Failed: {failed}{Colors.END}")
        print(f"{'='*60}")
        
        if failed == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ OUTPUT VALIDATION PASSED{Colors.END}")
            print("  Output is ready for downstream processing")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ OUTPUT VALIDATION FAILED{Colors.END}")
            print("  Review failures before using this output")
        
        return passed, failed, warnings
    
    def save_report(self, output_path: Path):
        """Save validation report to JSON."""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'input_path': str(self.input_path),
            'row_count': len(self.df) if self.df is not None else 0,
            'checks': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'critical': r.critical,
                    'message': r.message,
                    'details': r.details
                }
                for r in self.results
            ],
            'summary': {
                'passed': sum(1 for r in self.results if r.passed),
                'failed': sum(1 for r in self.results if not r.passed and r.critical),
                'warnings': sum(1 for r in self.results if not r.passed and not r.critical),
            }
        }
        
        # Add data summary if loaded
        if self.df is not None:
            report['data_summary'] = {
                'rows': len(self.df),
                'columns': list(self.df.columns),
                'pool_distribution': self.df['pool'].value_counts().to_dict() if 'pool' in self.df.columns else {},
                'split_distribution': self.df['split'].value_counts().to_dict() if 'split' in self.df.columns else {},
            }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to: {output_path}")


def download_from_s3(s3_path: str, local_dir: Path) -> bool:
    """Download files from S3 to local directory."""
    print(f"Downloading from {s3_path}...")
    
    result = subprocess.run(
        ['aws', 's3', 'sync', s3_path, str(local_dir), '--region', 'us-west-2'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error downloading: {result.stderr}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Validate EMR output')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=str, help='Local input directory')
    group.add_argument('--s3', type=str, help='S3 path to download and validate')
    
    parser.add_argument('--strict', action='store_true',
                        help='Fail on warnings too')
    parser.add_argument('--min-rows', type=int, default=10000,
                        help='Minimum expected row count')
    parser.add_argument('--save-report', type=str, default=None,
                        help='Save report to JSON file')
    
    args = parser.parse_args()
    
    print(f"{Colors.BOLD}EMR Output Validation{Colors.END}")
    print(f"Time: {datetime.now().isoformat()}")
    
    # Determine input path
    if args.s3:
        # Download from S3 to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            if not download_from_s3(args.s3, temp_path):
                print("Failed to download from S3")
                sys.exit(1)
            
            validator = OutputValidator(temp_path, strict=args.strict)
            passed, failed, warnings = validator.run_all()
            
            if args.save_report:
                validator.save_report(Path(args.save_report))
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Input path does not exist: {input_path}")
            sys.exit(1)
        
        validator = OutputValidator(input_path, strict=args.strict)
        passed, failed, warnings = validator.run_all()
        
        if args.save_report:
            validator.save_report(Path(args.save_report))
    
    # Exit code
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()

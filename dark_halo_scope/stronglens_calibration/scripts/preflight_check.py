#!/usr/bin/env python3
"""
Pre-flight check script for EMR negative sampling job.
Run this before launching EMR to verify all dependencies are ready.

Usage:
    python3 scripts/preflight_check.py [--skip-aws] [--verbose]
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def check_mark(passed: bool) -> str:
    if passed:
        return f"{Colors.GREEN}✓{Colors.END}"
    return f"{Colors.RED}✗{Colors.END}"

def warn_mark() -> str:
    return f"{Colors.YELLOW}⚠{Colors.END}"

def print_header(title: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")


class PreflightChecker:
    def __init__(self, base_dir: Path, skip_aws: bool = False, verbose: bool = False):
        self.base_dir = base_dir
        self.skip_aws = skip_aws
        self.verbose = verbose
        self.results: Dict[str, Dict] = {}
        
    def run_command(self, cmd: List[str], timeout: int = 30) -> Tuple[bool, str, str]:
        """Run a shell command and return (success, stdout, stderr)."""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except FileNotFoundError:
            return False, "", f"Command not found: {cmd[0]}"
        except Exception as e:
            return False, "", str(e)
    
    def check_file_exists(self, path: Path, description: str) -> bool:
        """Check if a file exists."""
        exists = path.exists()
        self.results[description] = {
            'passed': exists,
            'path': str(path),
            'message': 'File found' if exists else 'File not found'
        }
        return exists
    
    def check_file_content(self, path: Path, min_lines: int, description: str) -> bool:
        """Check if file exists and has minimum content."""
        if not path.exists():
            self.results[description] = {
                'passed': False,
                'message': 'File not found'
            }
            return False
        
        with open(path) as f:
            lines = sum(1 for _ in f)
        
        passed = lines >= min_lines
        self.results[description] = {
            'passed': passed,
            'lines': lines,
            'message': f'{lines} lines' + ('' if passed else f' (need {min_lines})')
        }
        return passed
    
    # ===========================================================
    # Section 1: Local Environment Checks
    # ===========================================================
    
    def check_python_version(self) -> bool:
        """Verify Python version is 3.8+."""
        version = sys.version_info
        passed = version.major == 3 and version.minor >= 8
        self.results['Python version'] = {
            'passed': passed,
            'version': f'{version.major}.{version.minor}.{version.micro}',
            'message': 'OK' if passed else 'Need Python 3.8+'
        }
        return passed
    
    def check_required_packages(self) -> bool:
        """Check required Python packages are installed."""
        packages = ['yaml', 'pandas', 'numpy']
        optional = ['healpy', 'astropy', 'boto3']
        
        all_passed = True
        for pkg in packages:
            try:
                __import__(pkg if pkg != 'yaml' else 'yaml')
                self.results[f'Package: {pkg}'] = {'passed': True, 'message': 'Installed'}
            except ImportError:
                self.results[f'Package: {pkg}'] = {'passed': False, 'message': 'NOT INSTALLED'}
                all_passed = False
        
        for pkg in optional:
            try:
                __import__(pkg)
                self.results[f'Package: {pkg} (optional)'] = {'passed': True, 'message': 'Installed'}
            except ImportError:
                self.results[f'Package: {pkg} (optional)'] = {
                    'passed': True,  # Optional, so still pass
                    'message': 'Not installed (will be installed on EMR)'
                }
        
        return all_passed
    
    def check_code_syntax(self) -> bool:
        """Check Python files for syntax errors."""
        files = [
            self.base_dir / 'emr' / 'spark_negative_sampling.py',
            self.base_dir / 'emr' / 'sampling_utils.py',
            self.base_dir / 'emr' / 'launch_negative_sampling.py',
        ]
        
        all_passed = True
        for file in files:
            if not file.exists():
                self.results[f'Syntax: {file.name}'] = {'passed': False, 'message': 'File not found'}
                all_passed = False
                continue
            
            success, _, stderr = self.run_command(['python3', '-m', 'py_compile', str(file)])
            self.results[f'Syntax: {file.name}'] = {
                'passed': success,
                'message': 'OK' if success else stderr.strip()
            }
            if not success:
                all_passed = False
        
        return all_passed
    
    def check_unit_tests(self) -> bool:
        """Run integration tests (test_phase1_local.py)."""
        # Use integration test which doesn't require pytest
        test_file = self.base_dir / 'tests' / 'test_phase1_local.py'
        if not test_file.exists():
            self.results['Integration tests'] = {'passed': False, 'message': 'Test file not found'}
            return False
        
        # Run tests (may take ~60s to load prototype data)
        success, stdout, stderr = self.run_command(
            ['python3', str(test_file)],
            timeout=180  # 3 minutes for large data loading
        )
        
        output = stdout + stderr
        
        # Parse output to find pass count
        passed_count = 0
        failed_count = 0
        for line in output.split('\n'):
            if 'Passed:' in line:
                try:
                    passed_count = int(line.split(':')[1].strip())
                except:
                    pass
            if 'Failed:' in line:
                try:
                    failed_count = int(line.split(':')[1].strip())
                except:
                    pass
        
        all_passed = 'ALL TESTS PASSED' in output or (passed_count > 0 and failed_count == 0)
        
        self.results['Integration tests'] = {
            'passed': all_passed,
            'message': f'{passed_count} tests passed' if all_passed else f'{failed_count} tests failed',
            'output': output[-500:] if self.verbose else None
        }
        return all_passed
    
    def check_config_file(self) -> bool:
        """Validate configuration file."""
        config_path = self.base_dir / 'configs' / 'negative_sampling_v1.yaml'
        if not config_path.exists():
            self.results['Configuration file'] = {'passed': False, 'message': 'Config not found'}
            return False
        
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Validate critical fields
            checks = {
                'version': config.get('version') == '1.0.0',
                'n1_n2_ratio': config.get('negative_pools', {}).get('n1_n2_ratio') == [85, 15],
                'neg_pos_ratio': config.get('negative_pools', {}).get('neg_pos_ratio') == 100,
                'nside': config.get('spatial_splits', {}).get('nside') == 128,
            }
            
            all_valid = all(checks.values())
            failed = [k for k, v in checks.items() if not v]
            
            self.results['Configuration file'] = {
                'passed': all_valid,
                'message': 'Valid' if all_valid else f'Invalid: {", ".join(failed)}'
            }
            return all_valid
            
        except Exception as e:
            self.results['Configuration file'] = {'passed': False, 'message': str(e)}
            return False
    
    # ===========================================================
    # Section 2: Data File Checks
    # ===========================================================
    
    def check_positive_catalog(self) -> bool:
        """Check positive catalog exists and has expected content."""
        path = self.base_dir / 'data' / 'positives' / 'desi_candidates.csv'
        return self.check_file_content(path, 5000, 'Positive catalog (desi_candidates.csv)')
    
    def check_spectroscopic_catalog(self) -> bool:
        """Check spectroscopic catalog exists."""
        path = self.base_dir / 'data' / 'external' / 'desi_dr1' / 'desi-sl-vac-v1.fits'
        return self.check_file_exists(path, 'Spectroscopic catalog (desi-sl-vac-v1.fits)')
    
    def check_prototype_data(self) -> bool:
        """Check prototype negative data exists (for local testing)."""
        path = self.base_dir / 'data' / 'negatives' / 'negative_catalog_prototype.csv'
        if path.exists():
            return self.check_file_content(path, 100000, 'Prototype data (for testing)')
        else:
            self.results['Prototype data (for testing)'] = {
                'passed': True,  # Not critical
                'message': 'Not present (OK for production runs)'
            }
            return True
    
    # ===========================================================
    # Section 3: AWS Checks
    # ===========================================================
    
    def check_aws_credentials(self) -> bool:
        """Check AWS credentials are configured."""
        success, stdout, stderr = self.run_command(['aws', 'sts', 'get-caller-identity'])
        
        if success:
            try:
                identity = json.loads(stdout)
                account = identity.get('Account', 'unknown')
                self.results['AWS credentials'] = {
                    'passed': True,
                    'account': account,
                    'message': f'Account: {account}'
                }
                return True
            except:
                pass
        
        self.results['AWS credentials'] = {
            'passed': False,
            'message': 'Not configured or invalid'
        }
        return False
    
    def check_s3_access(self) -> bool:
        """Check S3 bucket access."""
        bucket = 'darkhaloscope'
        success, stdout, stderr = self.run_command(
            ['aws', 's3', 'ls', f's3://{bucket}/', '--region', 'us-west-2']
        )
        
        self.results['S3 bucket access'] = {
            'passed': success,
            'bucket': bucket,
            'message': 'Accessible' if success else 'Access denied or not found'
        }
        return success
    
    def check_emr_permissions(self) -> bool:
        """Check EMR permissions."""
        success, stdout, stderr = self.run_command(
            ['aws', 'emr', 'list-clusters', '--region', 'us-west-2', '--active']
        )
        
        self.results['EMR permissions'] = {
            'passed': success,
            'message': 'EMR access OK' if success else 'Cannot list clusters'
        }
        return success
    
    def check_sweep_files_s3(self) -> bool:
        """Check if sweep files are available in S3."""
        # Check main sweep location
        success, stdout, stderr = self.run_command([
            'aws', 's3', 'ls', 's3://darkhaloscope/dr10/sweeps/', 
            '--region', 'us-west-2'
        ])
        
        if success and stdout.strip():
            # Count files
            file_count = len([l for l in stdout.strip().split('\n') if '.fits' in l])
            self.results['Sweep files in S3'] = {
                'passed': True,
                'count': file_count,
                'message': f'{file_count} files found'
            }
            return True
        
        # Check alternative location
        success, stdout, stderr = self.run_command([
            'aws', 's3', 'ls', 's3://darkhaloscope/stronglens_calibration/test_input/',
            '--region', 'us-west-2'
        ])
        
        if success and stdout.strip():
            self.results['Sweep files in S3'] = {
                'passed': True,
                'message': 'Test input available (not full dataset)'
            }
            return True
        
        self.results['Sweep files in S3'] = {
            'passed': False,
            'message': 'No sweep data found in S3'
        }
        return False
    
    # ===========================================================
    # Run All Checks
    # ===========================================================
    
    def run_all(self) -> Tuple[int, int, int]:
        """Run all pre-flight checks. Returns (passed, failed, warnings)."""
        
        print_header("1. LOCAL ENVIRONMENT")
        self.check_python_version()
        self.check_required_packages()
        self.check_code_syntax()
        self.check_config_file()
        self.check_unit_tests()
        
        print_header("2. DATA FILES")
        self.check_positive_catalog()
        self.check_spectroscopic_catalog()
        self.check_prototype_data()
        
        if not self.skip_aws:
            print_header("3. AWS INFRASTRUCTURE")
            self.check_aws_credentials()
            self.check_s3_access()
            self.check_emr_permissions()
            self.check_sweep_files_s3()
        else:
            print_header("3. AWS INFRASTRUCTURE (SKIPPED)")
            print(f"  {warn_mark()} Skipped AWS checks (--skip-aws)")
        
        # Print results
        print_header("RESULTS")
        
        passed = 0
        failed = 0
        warnings = 0
        
        for check, result in self.results.items():
            status = check_mark(result['passed'])
            message = result.get('message', '')
            
            if not result['passed'] and '(optional)' in check:
                status = warn_mark()
                warnings += 1
            elif result['passed']:
                passed += 1
            else:
                failed += 1
            
            print(f"  {status} {check}: {message}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"  {Colors.GREEN}Passed: {passed}{Colors.END}")
        if warnings > 0:
            print(f"  {Colors.YELLOW}Warnings: {warnings}{Colors.END}")
        if failed > 0:
            print(f"  {Colors.RED}Failed: {failed}{Colors.END}")
        print(f"{'='*60}")
        
        if failed == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL CRITICAL CHECKS PASSED{Colors.END}")
            print("  Ready to proceed with EMR launch")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ SOME CHECKS FAILED{Colors.END}")
            print("  Fix issues before launching EMR")
        
        return passed, failed, warnings
    
    def save_report(self, output_path: Path):
        """Save results to JSON file."""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'base_dir': str(self.base_dir),
            'checks': self.results,
            'summary': {
                'passed': sum(1 for r in self.results.values() if r['passed']),
                'failed': sum(1 for r in self.results.values() if not r['passed']),
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Pre-flight check for EMR job')
    parser.add_argument('--skip-aws', action='store_true', 
                        help='Skip AWS-related checks')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show verbose output')
    parser.add_argument('--save-report', type=str, default=None,
                        help='Save report to JSON file')
    args = parser.parse_args()
    
    # Determine base directory
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    
    print(f"{Colors.BOLD}Pre-Flight Check for EMR Negative Sampling{Colors.END}")
    print(f"Base directory: {base_dir}")
    print(f"Time: {datetime.now().isoformat()}")
    
    checker = PreflightChecker(base_dir, skip_aws=args.skip_aws, verbose=args.verbose)
    passed, failed, warnings = checker.run_all()
    
    if args.save_report:
        checker.save_report(Path(args.save_report))
    
    # Exit code
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()

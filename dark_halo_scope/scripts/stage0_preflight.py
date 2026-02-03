#!/usr/bin/env python3
"""
Stage 0 Pre-flight Checks

Run these checks BEFORE executing the anchor baseline to validate:
1. Catalog data integrity
2. Model checkpoint compatibility
3. Infrastructure availability
4. Config consistency

Usage:
    python3 stage0_preflight.py --model_path /path/to/ckpt.pt --check_network
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pre-flight Check Results
# =============================================================================

@dataclass
class CheckResult:
    """Result of a single pre-flight check."""
    name: str
    passed: bool
    message: str
    severity: str = "error"  # "error", "warning", "info"
    details: Dict = field(default_factory=dict)


@dataclass
class PreflightReport:
    """Complete pre-flight report."""
    checks: List[CheckResult] = field(default_factory=list)
    
    def add(self, result: CheckResult):
        self.checks.append(result)
    
    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "error")
    
    @property
    def error_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "error")
    
    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "warning")
    
    def print_report(self):
        """Print formatted report."""
        print("\n" + "=" * 70)
        print("STAGE 0 PRE-FLIGHT CHECK REPORT")
        print("=" * 70)
        
        for check in self.checks:
            status = "✅ PASS" if check.passed else ("⚠️ WARN" if check.severity == "warning" else "❌ FAIL")
            print(f"\n{status}: {check.name}")
            print(f"       {check.message}")
            if check.details:
                for k, v in check.details.items():
                    print(f"       - {k}: {v}")
        
        print("\n" + "-" * 70)
        if self.all_passed:
            print("✅ ALL CRITICAL CHECKS PASSED - Ready to execute Stage 0")
        else:
            print(f"❌ {self.error_count} ERRORS, {self.warning_count} WARNINGS")
            print("   Fix errors before proceeding!")
        print("=" * 70 + "\n")


# =============================================================================
# Check 1: Catalog Data Integrity
# =============================================================================

def check_catalog_integrity(report: PreflightReport):
    """Verify catalog data is valid and complete."""
    # Handle both package import and standalone execution
    try:
        from scripts.stage0_anchor_baseline import (
            SLACS_LENSES, BELLS_LENSES, create_known_lens_catalog
        )
    except ModuleNotFoundError:
        # Running standalone - import from same directory
        from stage0_anchor_baseline import (
            SLACS_LENSES, BELLS_LENSES, create_known_lens_catalog
        )
    
    # Check 1.1: SLACS count
    expected_slacs = 48
    actual_slacs = len(SLACS_LENSES)
    report.add(CheckResult(
        name="SLACS Catalog Count",
        passed=abs(actual_slacs - expected_slacs) <= 5,
        message=f"Found {actual_slacs} SLACS lenses (expected ~{expected_slacs})",
        details={"count": actual_slacs, "expected": expected_slacs}
    ))
    
    # Check 1.2: BELLS count
    expected_bells = 20
    actual_bells = len(BELLS_LENSES)
    report.add(CheckResult(
        name="BELLS Catalog Count",
        passed=abs(actual_bells - expected_bells) <= 5,
        message=f"Found {actual_bells} BELLS lenses (expected ~{expected_bells})",
        details={"count": actual_bells, "expected": expected_bells}
    ))
    
    # Check 1.3: Coordinate validity
    df = create_known_lens_catalog()
    ra_valid = (df['ra'] >= 0) & (df['ra'] <= 360)
    dec_valid = (df['dec'] >= -90) & (df['dec'] <= 90)
    all_valid = ra_valid.all() and dec_valid.all()
    
    report.add(CheckResult(
        name="Coordinate Validity",
        passed=all_valid,
        message=f"All coordinates in valid range" if all_valid else "Invalid coordinates found",
        details={
            "ra_range": f"[{df['ra'].min():.2f}, {df['ra'].max():.2f}]",
            "dec_range": f"[{df['dec'].min():.2f}, {df['dec'].max():.2f}]"
        }
    ))
    
    # Check 1.4: No duplicate coordinates
    df['coord_key'] = df['ra'].round(4).astype(str) + "_" + df['dec'].round(4).astype(str)
    n_unique = df['coord_key'].nunique()
    no_dups = n_unique == len(df)
    
    report.add(CheckResult(
        name="No Duplicate Coordinates",
        passed=no_dups,
        message=f"{n_unique} unique positions out of {len(df)} lenses",
        details={"unique": n_unique, "total": len(df)}
    ))
    
    # Check 1.5: Einstein radii in expected range
    theta_e_valid = (df['theta_e'] > 0.3) & (df['theta_e'] < 3.0)
    theta_ok = theta_e_valid.all()
    
    report.add(CheckResult(
        name="Einstein Radii Range",
        passed=theta_ok,
        message=f"θ_e range: [{df['theta_e'].min():.2f}, {df['theta_e'].max():.2f}] arcsec",
        severity="warning" if not theta_ok else "error",
        details={
            "min": df['theta_e'].min(),
            "max": df['theta_e'].max(),
            "median": df['theta_e'].median()
        }
    ))
    
    # Check 1.6: Redshift ordering (z_source > z_lens)
    z_order_ok = (df['z_source'] > df['z_lens']).all()
    
    report.add(CheckResult(
        name="Redshift Ordering",
        passed=z_order_ok,
        message="All sources behind lenses" if z_order_ok else "Some z_source <= z_lens!",
        details={
            "z_lens_range": f"[{df['z_lens'].min():.2f}, {df['z_lens'].max():.2f}]",
            "z_source_range": f"[{df['z_source'].min():.2f}, {df['z_source'].max():.2f}]"
        }
    ))
    
    # Check 1.7: Known reference lens present
    reference_lens = "SDSSJ0912+0029"  # Well-known SLACS lens
    has_reference = any(reference_lens.replace("SDSS", "") in lens[0] for lens in SLACS_LENSES)
    
    report.add(CheckResult(
        name="Reference Lens Present",
        passed=has_reference,
        message=f"{reference_lens} found in catalog" if has_reference else f"{reference_lens} MISSING!",
        details={"reference_lens": reference_lens}
    ))


# =============================================================================
# Check 2: Model Checkpoint Compatibility
# =============================================================================

def check_model_checkpoint(report: PreflightReport, model_path: str):
    """Verify model checkpoint is compatible."""
    import torch
    
    # Check 2.1: File exists
    exists = os.path.exists(model_path)
    report.add(CheckResult(
        name="Checkpoint Exists",
        passed=exists,
        message=f"Found: {model_path}" if exists else f"NOT FOUND: {model_path}",
        details={"path": model_path}
    ))
    
    if not exists:
        return  # Skip remaining checks
    
    # Check 2.2: Load checkpoint
    try:
        ckpt = torch.load(model_path, map_location="cpu")
        load_ok = True
    except Exception as e:
        load_ok = False
        report.add(CheckResult(
            name="Checkpoint Loadable",
            passed=False,
            message=f"Failed to load: {e}",
            details={"error": str(e)}
        ))
        return
    
    report.add(CheckResult(
        name="Checkpoint Loadable",
        passed=True,
        message=f"Successfully loaded checkpoint",
        details={"keys": list(ckpt.keys())}
    ))
    
    # Check 2.3: Has required keys
    required_keys = ['model', 'args']
    has_keys = all(k in ckpt for k in required_keys)
    
    report.add(CheckResult(
        name="Required Keys Present",
        passed=has_keys,
        message=f"Found keys: {list(ckpt.keys())}",
        details={"required": required_keys, "found": list(ckpt.keys())}
    ))
    
    if not has_keys:
        return
    
    # Check 2.4: Architecture match
    args = ckpt.get('args', {})
    arch = args.get('arch', 'unknown')
    expected_arch = 'convnext_tiny'
    
    report.add(CheckResult(
        name="Architecture Match",
        passed=arch == expected_arch,
        message=f"Architecture: {arch}",
        details={"found": arch, "expected": expected_arch}
    ))
    
    # Check 2.5: Metadata columns
    meta_cols = args.get('meta_cols', '')
    has_meta = bool(meta_cols)
    
    report.add(CheckResult(
        name="Metadata Fusion Config",
        passed=True,  # Informational
        message=f"meta_cols: '{meta_cols}'" if meta_cols else "No metadata fusion",
        severity="info",
        details={"meta_cols": meta_cols, "has_meta": has_meta}
    ))
    
    # Check 2.6: Training data variant
    data_path = args.get('data', 'unknown')
    
    report.add(CheckResult(
        name="Training Data Path",
        passed=True,  # Informational
        message=f"Trained on: {data_path}",
        severity="info",
        details={"data_path": data_path}
    ))
    
    # Check 2.7: State dict structure
    state_dict = ckpt.get('model', {})
    has_backbone = any('backbone' in k for k in state_dict.keys())
    has_head = any('head' in k for k in state_dict.keys())
    
    report.add(CheckResult(
        name="Model Structure Valid",
        passed=has_backbone and has_head,
        message=f"Backbone: {has_backbone}, Head: {has_head}",
        details={
            "num_params": len(state_dict),
            "sample_keys": list(state_dict.keys())[:5]
        }
    ))


# =============================================================================
# Check 3: Infrastructure Availability
# =============================================================================

def check_infrastructure(report: PreflightReport, check_network: bool = False):
    """Verify required infrastructure is available."""
    
    # Check 3.1: PyTorch available
    try:
        import torch
        torch_ok = True
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        torch_ok = False
        torch_version = "N/A"
        cuda_available = False
    
    report.add(CheckResult(
        name="PyTorch Available",
        passed=torch_ok,
        message=f"PyTorch {torch_version}, CUDA: {cuda_available}",
        details={"version": torch_version, "cuda": cuda_available}
    ))
    
    # Check 3.2: Astropy available (for FITS)
    try:
        from astropy.io import fits
        astropy_ok = True
    except ImportError:
        astropy_ok = False
    
    report.add(CheckResult(
        name="Astropy Available",
        passed=astropy_ok,
        message="Astropy FITS support available" if astropy_ok else "Astropy NOT installed",
        severity="error"
    ))
    
    # Check 3.3: Network connectivity (optional)
    if check_network:
        try:
            import urllib.request
            url = "https://www.legacysurvey.org/viewer/cutout.fits?ra=150&dec=30&size=10&layer=ls-dr10&pixscale=0.262&bands=grz"
            urllib.request.urlopen(url, timeout=10)
            network_ok = True
        except Exception as e:
            network_ok = False
            
        report.add(CheckResult(
            name="Legacy Survey API Accessible",
            passed=network_ok,
            message="Can reach cutout service" if network_ok else f"Cannot reach: {e}",
            severity="error"
        ))
    
    # Check 3.4: GPU memory (if CUDA available)
    if torch_ok and cuda_available:
        try:
            import torch
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            report.add(CheckResult(
                name="GPU Memory",
                passed=total_mem > 8,
                message=f"{total_mem:.1f} GB available",
                severity="warning" if total_mem < 16 else "info",
                details={"gpu_memory_gb": total_mem}
            ))
        except Exception:
            pass


# =============================================================================
# Check 4: Config Consistency
# =============================================================================

def check_config_consistency(report: PreflightReport, model_path: str, output_dir: str):
    """Verify configuration is internally consistent."""
    import torch
    
    # Check 4.1: Output directory writable
    try:
        os.makedirs(output_dir, exist_ok=True)
        test_file = os.path.join(output_dir, ".preflight_test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        dir_ok = True
    except Exception as e:
        dir_ok = False
    
    report.add(CheckResult(
        name="Output Directory Writable",
        passed=dir_ok,
        message=f"Can write to {output_dir}" if dir_ok else f"Cannot write to {output_dir}",
        details={"path": output_dir}
    ))
    
    # Check 4.2: Model normalization matches expected
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location="cpu")
        args = ckpt.get('args', {})
        
        norm_method = args.get('norm_method', 'full')
        expected_norm = 'outer'  # Our Stage 0 script uses outer
        
        report.add(CheckResult(
            name="Normalization Method Match",
            passed=norm_method == expected_norm,
            message=f"Model trained with '{norm_method}', Stage 0 uses '{expected_norm}'",
            severity="warning" if norm_method != expected_norm else "info",
            details={"model_norm": norm_method, "stage0_norm": expected_norm}
        ))
        
        # Check 4.3: MAD clip value
        mad_clip = args.get('mad_clip', 10.0)
        expected_clip = 10.0
        
        report.add(CheckResult(
            name="MAD Clip Value Match",
            passed=mad_clip == expected_clip,
            message=f"Model clip={mad_clip}, Stage 0 clip={expected_clip}",
            severity="warning" if mad_clip != expected_clip else "info"
        ))


# =============================================================================
# Check 5: Sanity Checks on Known Values
# =============================================================================

def check_known_values(report: PreflightReport):
    """Verify known reference values match expected."""
    try:
        from scripts.stage0_anchor_baseline import SLACS_LENSES
    except ModuleNotFoundError:
        from stage0_anchor_baseline import SLACS_LENSES
    
    # Check 5.1: SDSSJ0912+0029 coordinates (well-documented lens)
    # Published: RA=138.1417, Dec=+0.4861, theta_e=1.63"
    reference = None
    for lens in SLACS_LENSES:
        if "0912" in lens[0]:
            reference = lens
            break
    
    if reference:
        name, ra, dec, theta_e, z_l, z_s = reference
        
        ra_ok = abs(ra - 138.14) < 0.1
        dec_ok = abs(dec - 0.49) < 0.1
        theta_ok = abs(theta_e - 1.63) < 0.1
        
        report.add(CheckResult(
            name="Reference Lens J0912+0029",
            passed=ra_ok and dec_ok and theta_ok,
            message=f"RA={ra}, Dec={dec}, θ_e={theta_e}\" vs published RA=138.14, Dec=0.49, θ_e=1.63\"",
            details={
                "found_ra": ra, "expected_ra": 138.14,
                "found_dec": dec, "expected_dec": 0.49,
                "found_theta_e": theta_e, "expected_theta_e": 1.63
            }
        ))
    else:
        report.add(CheckResult(
            name="Reference Lens J0912+0029",
            passed=False,
            message="Reference lens NOT FOUND in catalog!"
        ))


# =============================================================================
# Main
# =============================================================================

def run_preflight(model_path: str, output_dir: str, check_network: bool = False) -> PreflightReport:
    """Run all pre-flight checks."""
    report = PreflightReport()
    
    logger.info("Running Stage 0 pre-flight checks...")
    
    # Run all checks
    check_catalog_integrity(report)
    check_model_checkpoint(report, model_path)
    check_infrastructure(report, check_network)
    check_config_consistency(report, model_path, output_dir)
    check_known_values(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Stage 0 Pre-flight Checks")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", default="/tmp/anchor_baseline", help="Output directory")
    parser.add_argument("--check_network", action="store_true", help="Check network connectivity")
    args = parser.parse_args()
    
    report = run_preflight(args.model_path, args.output_dir, args.check_network)
    report.print_report()
    
    # Exit with appropriate code
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()


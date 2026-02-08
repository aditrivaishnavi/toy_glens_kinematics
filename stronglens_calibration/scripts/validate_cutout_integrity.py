#!/usr/bin/env python3
"""
Automated Cutout Validation: Compare stored cutouts with fresh downloads.

This script validates that our cutouts match the original Legacy Survey data
by re-downloading and comparing pixel-by-pixel.

Usage:
    python validate_cutout_integrity.py --samples 10
    python validate_cutout_integrity.py --cutout-dir /path/to/cutouts --samples 50
"""
import argparse
import io
import json
import sys
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

import numpy as np


def download_fresh_cutout(ra: float, dec: float, size: int = 101, bands: str = "grz"):
    """Download cutout directly from Legacy Survey."""
    url = (
        f"https://www.legacysurvey.org/viewer/fits-cutout?"
        f"ra={ra}&dec={dec}&size={size}&layer=ls-dr10&pixscale=0.262&bands={bands}"
    )
    
    with urlopen(url, timeout=60) as response:
        fits_data = response.read()
    
    from astropy.io import fits
    buffer = io.BytesIO(fits_data)
    with fits.open(buffer) as hdul:
        data = hdul[0].data  # Shape: (n_bands, H, W)
        # Transpose to (H, W, n_bands)
        cutout = np.transpose(data, (1, 2, 0)).astype(np.float32)
    
    return cutout


def compare_cutouts(stored: np.ndarray, fresh: np.ndarray) -> dict:
    """Compare two cutouts and return metrics."""
    # Pixel-wise difference
    diff = stored - fresh
    
    # Metrics
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    
    # Check if identical (within floating point tolerance)
    identical = np.allclose(stored, fresh, rtol=1e-5, atol=1e-8)
    
    # Per-band comparison
    band_metrics = {}
    for i, band in enumerate(["g", "r", "z"]):
        band_diff = stored[:, :, i] - fresh[:, :, i]
        band_metrics[band] = {
            "max_diff": float(np.max(np.abs(band_diff))),
            "mean_diff": float(np.mean(np.abs(band_diff))),
            "correlation": float(np.corrcoef(stored[:, :, i].ravel(), fresh[:, :, i].ravel())[0, 1]),
            "identical": np.allclose(stored[:, :, i], fresh[:, :, i], rtol=1e-5, atol=1e-8),
        }
    
    return {
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "rmse": float(rmse),
        "identical": identical,
        "bands": band_metrics,
    }


def validate_cutouts(cutout_dir: Path, n_samples: int = 10, verbose: bool = True):
    """Validate a sample of cutouts against fresh downloads."""
    
    # Find all NPZ files
    npz_files = sorted(cutout_dir.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {cutout_dir}")
        return None
    
    # Sample evenly across the dataset
    step = max(1, len(npz_files) // n_samples)
    sample_files = npz_files[::step][:n_samples]
    
    if verbose:
        print(f"Validating {len(sample_files)} cutouts from {len(npz_files)} total")
        print("=" * 60)
    
    results = {}
    issues = []
    
    for npz_file in sample_files:
        name = npz_file.stem
        
        try:
            # Load stored cutout
            npz = np.load(npz_file)
            stored = npz["cutout"]
            ra = float(npz["meta_ra"])
            dec = float(npz["meta_dec"])
            
            if verbose:
                print(f"\n{name}: RA={ra:.6f}, Dec={dec:.6f}")
            
            # Download fresh cutout
            fresh = download_fresh_cutout(ra, dec)
            
            # Compare
            metrics = compare_cutouts(stored, fresh)
            results[name] = metrics
            
            if metrics["identical"]:
                if verbose:
                    print(f"  ✅ IDENTICAL (max_diff={metrics['max_diff']:.2e})")
            else:
                if verbose:
                    print(f"  ⚠️  DIFFERS: max_diff={metrics['max_diff']:.4f}, rmse={metrics['rmse']:.6f}")
                    for band, bm in metrics["bands"].items():
                        status = "✓" if bm["identical"] else "✗"
                        print(f"    {band}: {status} max={bm['max_diff']:.4f}, corr={bm['correlation']:.6f}")
                
                # Only flag as issue if correlation is low (not just floating point differences)
                if any(bm["correlation"] < 0.999 for bm in metrics["bands"].values()):
                    issues.append(name)
                    
        except (URLError, HTTPError) as e:
            if verbose:
                print(f"  ❌ NETWORK ERROR: {e}")
            issues.append(name)
        except Exception as e:
            if verbose:
                print(f"  ❌ ERROR: {e}")
            issues.append(name)
    
    # Summary
    n_identical = sum(1 for r in results.values() if r["identical"])
    n_close = sum(1 for r in results.values() 
                  if not r["identical"] and all(bm["correlation"] > 0.999 for bm in r["bands"].values()))
    
    summary = {
        "total_tested": len(results),
        "identical": n_identical,
        "close_match": n_close,  # Different due to floating point, but highly correlated
        "issues": len(issues),
        "issue_names": issues,
        "pass_rate": (n_identical + n_close) / len(results) if results else 0,
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total tested:   {summary['total_tested']}")
        print(f"Identical:      {summary['identical']}")
        print(f"Close match:    {summary['close_match']} (>99.9% correlation)")
        print(f"Issues:         {summary['issues']}")
        print(f"Pass rate:      {summary['pass_rate']*100:.1f}%")
        
        if summary["issues"] == 0:
            print("\n✅ ALL CUTOUTS VALIDATED SUCCESSFULLY")
        else:
            print(f"\n⚠️  Issues found in: {issues}")
    
    return {"summary": summary, "details": results}


def main():
    parser = argparse.ArgumentParser(description="Validate cutout integrity against Legacy Survey")
    parser.add_argument("--cutout-dir", type=Path, default=Path("."), 
                       help="Directory containing .npz cutouts")
    parser.add_argument("--samples", type=int, default=10,
                       help="Number of samples to validate")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    
    results = validate_cutouts(args.cutout_dir, args.samples, verbose=not args.quiet)
    
    if results and args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Exit with error if issues found
    if results and results["summary"]["issues"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

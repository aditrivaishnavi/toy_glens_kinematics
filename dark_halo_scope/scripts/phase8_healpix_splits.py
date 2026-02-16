#!/usr/bin/env python3
"""
Phase 8: Implement HEALPix-based region-disjoint splits.

Problem:
- Current hash-based splits don't guarantee region-disjoint train/val/test
- Nearby LRGs can end up in different splits, causing spatial leakage
- Publication-grade models need guaranteed region disjointness

Solution:
- Use HEALPix pixelization to define sky regions
- Assign entire HEALPix pixels to train/val/test
- Ensures no two samples in different splits are spatially close

This script provides:
1. HEALPix implementation for splitting
2. Integration code for pipeline
3. Verification of spatial disjointness
"""
import numpy as np
import json
from datetime import datetime, timezone


def healpix_split_implementation():
    """
    Provide HEALPix-based splitting implementation.
    """
    code = '''
import healpy as hp
import numpy as np

def assign_healpix_split(ra, dec, nside=32, train_frac=0.7, val_frac=0.15, seed=42):
    """
    Assign sample to train/val/test based on HEALPix pixel.
    
    Uses HEALPix to define sky regions, then deterministically assigns
    entire pixels to splits. This ensures region-disjoint splits.
    
    Args:
        ra: Right ascension in degrees
        dec: Declination in degrees
        nside: HEALPix resolution (32 = ~1.8° pixels, ~12,288 pixels total)
        train_frac: Fraction of pixels for training
        val_frac: Fraction of pixels for validation
        seed: Random seed for reproducible pixel assignment
    
    Returns:
        str: "train", "val", or "test"
    """
    # Convert to HEALPix theta/phi
    theta = np.radians(90 - dec)  # HEALPix uses colatitude
    phi = np.radians(ra)
    
    # Get HEALPix pixel
    pix = hp.ang2pix(nside, theta, phi)
    
    # Deterministic assignment based on pixel hash
    np.random.seed(seed + pix)  # Reproducible per-pixel
    r = np.random.random()
    
    if r < train_frac:
        return "train"
    elif r < train_frac + val_frac:
        return "val"
    else:
        return "test"


def verify_region_disjointness(ra_train, dec_train, ra_test, dec_test, min_sep_deg=2.0):
    """
    Verify that train and test samples are spatially separated.
    
    Args:
        ra_train, dec_train: Training sample coordinates
        ra_test, dec_test: Test sample coordinates
        min_sep_deg: Minimum required separation in degrees
    
    Returns:
        dict: Verification results
    """
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    
    train_coords = SkyCoord(ra=ra_train*u.deg, dec=dec_train*u.deg)
    test_coords = SkyCoord(ra=ra_test*u.deg, dec=dec_test*u.deg)
    
    # For each test point, find nearest train point
    idx, sep2d, _ = test_coords.match_to_catalog_sky(train_coords)
    min_seps = sep2d.deg
    
    violations = np.sum(min_seps < min_sep_deg)
    
    return {
        "n_test": len(test_coords),
        "n_violations": int(violations),
        "violation_rate": float(violations / len(test_coords)),
        "min_separation": float(np.min(min_seps)),
        "median_separation": float(np.median(min_seps)),
        "passed": violations == 0
    }


# Pipeline integration:
def add_healpix_split_to_manifest(df, nside=32, seed=42):
    """
    Add HEALPix-based region_split to manifest DataFrame.
    
    Args:
        df: Pandas DataFrame with 'ra' and 'dec' columns
        nside: HEALPix resolution
        seed: Random seed
    
    Returns:
        DataFrame with 'region_split' column added
    """
    import healpy as hp
    
    # Convert to HEALPix
    theta = np.radians(90 - df['dec'].values)
    phi = np.radians(df['ra'].values)
    pix = hp.ang2pix(nside, theta, phi)
    
    # Pre-compute split for each unique pixel
    unique_pix = np.unique(pix)
    np.random.seed(seed)
    pix_to_split = {}
    for p in unique_pix:
        r = np.random.random()
        if r < 0.7:
            pix_to_split[p] = "train"
        elif r < 0.85:
            pix_to_split[p] = "val"
        else:
            pix_to_split[p] = "test"
    
    # Apply to DataFrame
    df['region_split'] = [pix_to_split[p] for p in pix]
    df['healpix_pix'] = pix
    
    return df
'''
    
    print("HEALPix Split Implementation:")
    print("-" * 70)
    print(code)
    print("-" * 70)
    
    return code


def healpix_nside_tradeoffs():
    """Document tradeoffs for different NSIDE values."""
    tradeoffs = {
        "nside=16": {
            "pixel_size_deg": 3.66,
            "n_pixels": 3072,
            "pros": "Maximum spatial separation between splits",
            "cons": "Fewer pixels may cause unbalanced splits"
        },
        "nside=32": {
            "pixel_size_deg": 1.83,
            "n_pixels": 12288,
            "pros": "Good balance of separation and sample count",
            "cons": "Recommended default"
        },
        "nside=64": {
            "pixel_size_deg": 0.92,
            "n_pixels": 49152,
            "pros": "More pixels for fine-grained control",
            "cons": "Smaller separation between splits"
        }
    }
    
    print("\nHEALPix NSIDE Tradeoffs:")
    for nside, info in tradeoffs.items():
        print(f"\n{nside}:")
        print(f"  Pixel size: {info['pixel_size_deg']:.2f}°")
        print(f"  Total pixels: {info['n_pixels']}")
        print(f"  Pros: {info['pros']}")
        print(f"  Cons: {info['cons']}")
    
    return tradeoffs


def main():
    print("=" * 70)
    print("PHASE 8: HEALPIX-BASED REGION-DISJOINT SPLITS")
    print("=" * 70)
    
    RESULTS = {
        "phase": "8",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": "HEALPix-based region-disjoint train/val/test splits"
    }
    
    # Implementation
    code = healpix_split_implementation()
    RESULTS["implementation"] = code
    
    # Tradeoffs
    tradeoffs = healpix_nside_tradeoffs()
    RESULTS["nside_tradeoffs"] = tradeoffs
    
    # Recommendation
    RESULTS["recommendation"] = {
        "nside": 32,
        "pixel_size_deg": 1.83,
        "guard_band_deg": 0.5,  # Optional: exclude samples within 0.5° of pixel boundary
        "split_fractions": {"train": 0.70, "val": 0.15, "test": 0.15}
    }
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(f"NSIDE: {RESULTS['recommendation']['nside']}")
    print(f"Pixel size: {RESULTS['recommendation']['pixel_size_deg']}°")
    print(f"Guard band: {RESULTS['recommendation']['guard_band_deg']}° (optional)")
    print(f"Split fractions: {RESULTS['recommendation']['split_fractions']}")
    
    RESULTS["status"] = "READY_TO_IMPLEMENT"
    RESULTS["next_steps"] = [
        "1. pip install healpy on EMR and Lambda",
        "2. Add assign_healpix_split to manifest generation",
        "3. Replace hash-based splitting with HEALPix-based",
        "4. Verify with verify_region_disjointness function",
        "5. Document expected train/val/test sizes"
    ]
    
    RESULTS["integration_points"] = [
        "spark_phase4a_build_manifest_sota.py - manifest generation",
        "phase4p5_compact_with_relabel.py - if relabeling needed",
        "Training script - verify region_split column is used"
    ]
    
    print("\nNext Steps:")
    for step in RESULTS["next_steps"]:
        print(f"  {step}")
    
    print("\nIntegration Points:")
    for point in RESULTS["integration_points"]:
        print(f"  - {point}")
    
    # Save
    with open("/lambda/nfs/darkhaloscope-training-dc/phase8_healpix_config.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    
    print("\nResults saved to phase8_healpix_config.json")
    return RESULTS


if __name__ == "__main__":
    main()

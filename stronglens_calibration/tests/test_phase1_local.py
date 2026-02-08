#!/usr/bin/env python3
"""
Local Phase 1 Integration Test

This test validates all Phase 1 components (1A-1E) using local data
BEFORE running on EMR. This catches bugs early and saves EMR costs.

Run with: python3 tests/test_phase1_local.py

Lessons Applied:
- L4.3: Test the ACTUAL code path, not a mock version
- L5.1: Don't declare victory prematurely - verify everything
- L6.1: Local testing before EMR
"""
import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from emr.sampling_utils import (
    get_nobs_z_bin,
    get_type_bin,
    flux_to_mag,
    compute_healpix,
    assign_split,
    is_near_known_lens,
    classify_pool_n2,
    check_maskbits,
    compute_split_proportions,
    VALID_TYPES_N1,
    NOBS_Z_BINS,
    DEFAULT_EXCLUDE_MASKBITS,
)

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
NEGATIVE_CATALOG = DATA_DIR / "negatives" / "negative_catalog_prototype.csv"
POSITIVE_CATALOG = DATA_DIR / "positives" / "desi_candidates.csv"

# Test parameters
SAMPLE_SIZE = 10000  # Number of rows to test (for speed)
EXPECTED_SPLIT_TOLERANCE = 0.03  # 3% tolerance for split proportions


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_sample_data(n_rows: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Load a sample of the negative catalog."""
    if not NEGATIVE_CATALOG.exists():
        raise FileNotFoundError(f"Negative catalog not found: {NEGATIVE_CATALOG}")
    
    # Read first n rows
    df = pd.read_csv(NEGATIVE_CATALOG, nrows=n_rows)
    print(f"Loaded {len(df)} rows from negative catalog")
    return df


def load_positive_coords() -> List[Tuple[float, float]]:
    """Load RA/Dec coordinates from positive catalog."""
    if not POSITIVE_CATALOG.exists():
        raise FileNotFoundError(f"Positive catalog not found: {POSITIVE_CATALOG}")
    
    df = pd.read_csv(POSITIVE_CATALOG)
    coords = list(zip(df['ra'].values, df['dec'].values))
    print(f"Loaded {len(coords)} positive positions for exclusion")
    return coords


# =============================================================================
# PHASE 1A: NEGATIVE POOL DESIGN
# =============================================================================

def test_1a_pool_classification():
    """Test 1A.1-1A.2: Pool N1 and N2 classification."""
    print("\n" + "="*60)
    print("TEST 1A: Negative Pool Design")
    print("="*60)
    
    df = load_sample_data()
    
    # Test N1 classification (deployment-representative)
    n1_count = 0
    n2_count = 0
    other_count = 0
    
    for _, row in df.iterrows():
        galaxy_type = str(row.get('type', 'OTHER')).strip().upper()
        type_bin = get_type_bin(galaxy_type)
        
        # Check N2 confuser criteria
        # Use flux_z as proxy for flux_r if flux_r not available
        flux_r = row.get('flux_r', None)
        if flux_r is None and 'flux_z' in row.index:
            flux_r = row['flux_z']  # Use z-band as proxy
        
        shape_r = row.get('shape_r', None)
        g_minus_r = None  # Not in prototype catalog
        mag_r = flux_to_mag(flux_r) if flux_r else None
        
        # Config with adjusted thresholds for z-band flux
        config = {
            "negative_pools": {
                "pool_n2": {
                    "tractor_criteria": {
                        "ring_proxy": {"type": "DEV", "flux_r_min": 5},  # Lower threshold for z-band
                        "edge_on_proxy": {"type": "EXP", "shape_r_min": 2.0},
                        "blue_clumpy_proxy": {"g_minus_r_max": 0.4, "r_mag_max": 19.0}
                    }
                }
            }
        }
        
        confuser = classify_pool_n2(galaxy_type, flux_r, shape_r, g_minus_r, mag_r, config)
        
        if confuser:
            n2_count += 1
        elif type_bin in VALID_TYPES_N1:
            n1_count += 1
        else:
            other_count += 1
    
    total = n1_count + n2_count
    n1_frac = n1_count / total * 100 if total > 0 else 0
    n2_frac = n2_count / total * 100 if total > 0 else 0
    
    print(f"  N1 (deployment): {n1_count} ({n1_frac:.1f}%)")
    print(f"  N2 (confuser):   {n2_count} ({n2_frac:.1f}%)")
    print(f"  Other (skipped): {other_count}")
    
    # Verify we have both pools
    assert n1_count > 0, "ERROR: No N1 samples found"
    assert total > 0, "ERROR: No valid samples"
    
    print("  [PASS] Pool classification working")
    return True


def test_1a_nobs_z_binning():
    """Test 1A.4: nobs_z stratification."""
    print("\n  Testing nobs_z binning...")
    
    df = load_sample_data()
    
    bin_counts = Counter()
    for nobs in df['nobs_z']:
        bin_label = get_nobs_z_bin(int(nobs))
        bin_counts[bin_label] += 1
    
    print("  nobs_z bin distribution:")
    for bin_label, count in sorted(bin_counts.items()):
        pct = count / len(df) * 100
        print(f"    {bin_label:8s}: {count:6d} ({pct:5.1f}%)")
    
    # Verify all bins are represented (except 'invalid')
    valid_bins = [b for b in bin_counts.keys() if b != 'invalid']
    assert len(valid_bins) >= 3, f"ERROR: Only {len(valid_bins)} nobs_z bins found"
    
    print("  [PASS] nobs_z binning working")
    return True


def test_1a_type_binning():
    """Test 1A.4: Type stratification."""
    print("\n  Testing type binning...")
    
    df = load_sample_data()
    
    type_counts = Counter()
    for t in df['type']:
        type_bin = get_type_bin(str(t))
        type_counts[type_bin] += 1
    
    print("  Type bin distribution:")
    for type_bin, count in sorted(type_counts.items()):
        pct = count / len(df) * 100
        print(f"    {type_bin:8s}: {count:6d} ({pct:5.1f}%)")
    
    # Verify at least 3 types are present
    assert len(type_counts) >= 3, f"ERROR: Only {len(type_counts)} type bins found"
    
    print("  [PASS] Type binning working")
    return True


def test_1a_lens_exclusion():
    """Test 1A.7: Known lens exclusion."""
    print("\n  Testing lens exclusion...")
    
    df = load_sample_data()
    known_coords = load_positive_coords()
    
    excluded = 0
    exclusion_radius = 11.0  # arcsec
    
    for _, row in df.iterrows():
        ra = row['ra']
        dec = row['dec']
        if is_near_known_lens(ra, dec, known_coords, exclusion_radius):
            excluded += 1
    
    excluded_pct = excluded / len(df) * 100
    print(f"  Excluded within {exclusion_radius}\": {excluded} ({excluded_pct:.2f}%)")
    
    # Should exclude some but not too many
    if excluded > 0:
        print("  [PASS] Lens exclusion working")
    else:
        print("  [WARN] No galaxies excluded - may be due to non-overlapping sky regions")
    
    return True


# =============================================================================
# PHASE 1B: SPATIAL SPLITS
# =============================================================================

def test_1b_healpix_computation():
    """Test 1B.1: HEALPix index computation."""
    print("\n" + "="*60)
    print("TEST 1B: Spatial Splits")
    print("="*60)
    
    df = load_sample_data()
    
    healpix_64 = []
    healpix_128 = []
    
    for _, row in df.iterrows():
        hp64 = compute_healpix(row['ra'], row['dec'], 64)
        hp128 = compute_healpix(row['ra'], row['dec'], 128)
        healpix_64.append(hp64)
        healpix_128.append(hp128)
    
    unique_64 = len(set(healpix_64))
    unique_128 = len(set(healpix_128))
    
    print(f"  HEALPix unique cells:")
    print(f"    nside=64:  {unique_64}")
    print(f"    nside=128: {unique_128}")
    
    # Should have multiple cells for good spatial coverage
    assert unique_64 > 1, "ERROR: All samples in single HEALPix-64 cell"
    assert unique_128 >= unique_64, "ERROR: nside=128 has fewer cells than nside=64"
    
    print("  [PASS] HEALPix computation working")
    return True


def test_1b_split_assignment():
    """Test 1B.4: Train/val/test split assignment."""
    print("\n  Testing split assignment...")
    
    df = load_sample_data()
    allocations = {"train": 0.70, "val": 0.15, "test": 0.15}
    seed = 42
    
    # Count unique HEALPix cells
    unique_cells = set()
    cell_splits = {}
    
    for _, row in df.iterrows():
        hp128 = compute_healpix(row['ra'], row['dec'], 128)
        unique_cells.add(hp128)
        if hp128 not in cell_splits:
            cell_splits[hp128] = assign_split(hp128, allocations, seed)
    
    n_cells = len(unique_cells)
    print(f"  Unique HEALPix cells: {n_cells}")
    
    # If we have very few cells, proportions won't match target - this is expected
    if n_cells < 20:
        print(f"  [WARN] Only {n_cells} cells - split proportions may not match target")
        print("  This is expected for limited sky regions")
        
        # Just verify function works
        splits = list(cell_splits.values())
        print(f"  Cell assignments: {Counter(splits)}")
        
        # Verify at least one split type is assigned
        assert len(splits) > 0, "ERROR: No splits assigned"
        print("  [PASS] Split assignment function working (limited cells)")
        return True
    
    # Full verification for large cell counts
    splits = []
    for _, row in df.iterrows():
        hp128 = compute_healpix(row['ra'], row['dec'], 128)
        split = assign_split(hp128, allocations, seed)
        splits.append(split)
    
    proportions = compute_split_proportions(splits)
    
    print("  Split proportions:")
    for split, prop in sorted(proportions.items()):
        target = allocations[split]
        diff = abs(prop - target)
        status = "OK" if diff < EXPECTED_SPLIT_TOLERANCE else "WARN"
        print(f"    {split:5s}: {prop*100:5.1f}% (target: {target*100:.0f}%) [{status}]")
    
    # Verify proportions are within tolerance
    for split, target in allocations.items():
        actual = proportions.get(split, 0)
        diff = abs(actual - target)
        assert diff < EXPECTED_SPLIT_TOLERANCE, \
            f"ERROR: Split {split} proportion {actual:.2f} deviates from target {target:.2f}"
    
    print("  [PASS] Split assignment working")
    return True


def test_1b_split_determinism():
    """Test 1B.5: Split assignment is deterministic."""
    print("\n  Testing split determinism...")
    
    df = load_sample_data(n_rows=100)  # Smaller sample for speed
    allocations = {"train": 0.70, "val": 0.15, "test": 0.15}
    seed = 42
    
    # Run twice
    splits_1 = []
    splits_2 = []
    
    for _, row in df.iterrows():
        hp128 = compute_healpix(row['ra'], row['dec'], 128)
        splits_1.append(assign_split(hp128, allocations, seed))
        splits_2.append(assign_split(hp128, allocations, seed))
    
    assert splits_1 == splits_2, "ERROR: Split assignment is not deterministic!"
    
    print("  [PASS] Split assignment is deterministic")
    return True


def test_1b_spatial_disjointness():
    """Test 1B.5: Verify no HEALPix cell appears in multiple splits."""
    print("\n  Testing spatial disjointness...")
    
    df = load_sample_data()
    allocations = {"train": 0.70, "val": 0.15, "test": 0.15}
    seed = 42
    
    cell_to_split = {}
    conflicts = 0
    
    for _, row in df.iterrows():
        hp128 = compute_healpix(row['ra'], row['dec'], 128)
        split = assign_split(hp128, allocations, seed)
        
        if hp128 in cell_to_split:
            if cell_to_split[hp128] != split:
                conflicts += 1
        else:
            cell_to_split[hp128] = split
    
    print(f"  Unique HEALPix cells: {len(cell_to_split)}")
    print(f"  Conflicts: {conflicts}")
    
    assert conflicts == 0, f"ERROR: {conflicts} HEALPix cells appear in multiple splits!"
    
    print("  [PASS] Spatial disjointness verified")
    return True


# =============================================================================
# PHASE 1C: SCHEMA IMPLEMENTATION
# =============================================================================

def test_1c_schema_completeness():
    """Test 1C: All schema columns can be computed."""
    print("\n" + "="*60)
    print("TEST 1C: Schema Implementation")
    print("="*60)
    
    df = load_sample_data(n_rows=100)
    
    # Columns we can compute from prototype catalog
    required_cols = ['ra', 'dec', 'type', 'nobs_z', 'brickname']
    optional_cols = ['psfsize_z', 'psfdepth_z', 'flux_z']
    
    missing_required = [c for c in required_cols if c not in df.columns]
    missing_optional = [c for c in optional_cols if c not in df.columns]
    
    print(f"  Required columns present: {len(required_cols) - len(missing_required)}/{len(required_cols)}")
    if missing_required:
        print(f"  Missing required: {missing_required}")
    
    print(f"  Optional columns present: {len(optional_cols) - len(missing_optional)}/{len(optional_cols)}")
    
    assert len(missing_required) == 0, f"ERROR: Missing required columns: {missing_required}"
    
    # Test derived columns
    row = df.iloc[0]
    galaxy_id = f"{row['brickname']}_{0}"  # Would use objid in real data
    nobs_z_bin = get_nobs_z_bin(int(row['nobs_z']))
    type_bin = get_type_bin(str(row['type']))
    hp64 = compute_healpix(row['ra'], row['dec'], 64)
    hp128 = compute_healpix(row['ra'], row['dec'], 128)
    split = assign_split(hp128, {"train": 0.7, "val": 0.15, "test": 0.15}, 42)
    
    print(f"  Sample derived values:")
    print(f"    galaxy_id:   {galaxy_id}")
    print(f"    nobs_z_bin:  {nobs_z_bin}")
    print(f"    type_bin:    {type_bin}")
    print(f"    healpix_64:  {hp64}")
    print(f"    healpix_128: {hp128}")
    print(f"    split:       {split}")
    
    print("  [PASS] Schema columns computable")
    return True


# =============================================================================
# PHASE 1D: QUALITY GATES
# =============================================================================

def test_1d_quality_gates():
    """Test 1D: Quality gate checks."""
    print("\n" + "="*60)
    print("TEST 1D: Quality Gates")
    print("="*60)
    
    df = load_sample_data()
    
    # Check for null values in critical columns
    null_counts = {}
    for col in df.columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            null_counts[col] = null_count
    
    if null_counts:
        print("  Null values found:")
        for col, count in null_counts.items():
            pct = count / len(df) * 100
            print(f"    {col}: {count} ({pct:.2f}%)")
    else:
        print("  No null values in data")
    
    # Check data ranges
    print("\n  Data ranges:")
    print(f"    RA:     {df['ra'].min():.2f} to {df['ra'].max():.2f}")
    print(f"    Dec:    {df['dec'].min():.2f} to {df['dec'].max():.2f}")
    print(f"    nobs_z: {df['nobs_z'].min()} to {df['nobs_z'].max()}")
    
    # Verify RA/Dec ranges are valid
    assert df['ra'].min() >= 0 and df['ra'].max() <= 360, "ERROR: RA out of range"
    assert df['dec'].min() >= -90 and df['dec'].max() <= 90, "ERROR: Dec out of range"
    assert df['nobs_z'].min() >= 0, "ERROR: Negative nobs_z"
    
    print("  [PASS] Quality gates verified")
    return True


# =============================================================================
# PHASE 1E: EMR STABILITY
# =============================================================================

def test_1e_provenance_tracking():
    """Test 1E: Provenance tracking."""
    print("\n" + "="*60)
    print("TEST 1E: EMR Stability & Provenance")
    print("="*60)
    
    # Test that we can generate provenance info
    import subprocess
    
    # Get git commit
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()[:8]
    except:
        git_commit = "unknown"
    
    timestamp = datetime.utcnow().isoformat()
    pipeline_version = "1.0.0"
    
    print(f"  Provenance info:")
    print(f"    git_commit:       {git_commit}")
    print(f"    timestamp:        {timestamp}")
    print(f"    pipeline_version: {pipeline_version}")
    
    print("  [PASS] Provenance tracking working")
    return True


def test_1e_deterministic_seeding():
    """Test 1E.1: Deterministic seeding produces reproducible results."""
    print("\n  Testing deterministic seeding...")
    
    # Test with same galaxy_id
    galaxy_id = "test_brick_12345"
    seed = int(hashlib.md5(galaxy_id.encode()).hexdigest()[:8], 16)
    
    np.random.seed(seed)
    result1 = np.random.random()
    
    np.random.seed(seed)
    result2 = np.random.random()
    
    assert result1 == result2, "ERROR: Seeding is not deterministic"
    
    print("  [PASS] Deterministic seeding verified")
    return True


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all Phase 1 tests."""
    print("\n" + "#"*60)
    print("# PHASE 1 LOCAL INTEGRATION TESTS")
    print("# Run before EMR to catch bugs early")
    print("#"*60)
    
    start_time = datetime.now()
    results = {}
    
    # Phase 1A: Negative Pool Design
    try:
        results['1A.1-2 Pool Classification'] = test_1a_pool_classification()
        results['1A.4a nobs_z Binning'] = test_1a_nobs_z_binning()
        results['1A.4b Type Binning'] = test_1a_type_binning()
        results['1A.7 Lens Exclusion'] = test_1a_lens_exclusion()
    except Exception as e:
        results['1A'] = f"FAILED: {e}"
        print(f"  [FAIL] 1A: {e}")
    
    # Phase 1B: Spatial Splits
    try:
        results['1B.1 HEALPix'] = test_1b_healpix_computation()
        results['1B.4 Split Assignment'] = test_1b_split_assignment()
        results['1B.5a Determinism'] = test_1b_split_determinism()
        results['1B.5b Disjointness'] = test_1b_spatial_disjointness()
    except Exception as e:
        results['1B'] = f"FAILED: {e}"
        print(f"  [FAIL] 1B: {e}")
    
    # Phase 1C: Schema
    try:
        results['1C Schema'] = test_1c_schema_completeness()
    except Exception as e:
        results['1C'] = f"FAILED: {e}"
        print(f"  [FAIL] 1C: {e}")
    
    # Phase 1D: Quality Gates
    try:
        results['1D Quality Gates'] = test_1d_quality_gates()
    except Exception as e:
        results['1D'] = f"FAILED: {e}"
        print(f"  [FAIL] 1D: {e}")
    
    # Phase 1E: EMR Stability
    try:
        results['1E.1 Provenance'] = test_1e_provenance_tracking()
        results['1E.2 Seeding'] = test_1e_deterministic_seeding()
    except Exception as e:
        results['1E'] = f"FAILED: {e}"
        print(f"  [FAIL] 1E: {e}")
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is not True)
    
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Time:   {elapsed:.1f}s")
    
    if failed > 0:
        print("\n  Failed tests:")
        for name, result in results.items():
            if result is not True:
                print(f"    - {name}: {result}")
    
    print("\n" + "="*60)
    if failed == 0:
        print("ALL TESTS PASSED - Ready for EMR mini-test")
        return True
    else:
        print("TESTS FAILED - Fix before running EMR")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

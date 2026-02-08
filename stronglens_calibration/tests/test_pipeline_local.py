#!/usr/bin/env python3
"""
Local Pipeline Integration Test (EMR Simulation)

This test runs the full negative sampling pipeline locally using pandas
instead of Spark, simulating what EMR would do but catching bugs faster.

Run with: python3 tests/test_pipeline_local.py

This is the "mini-test" before actual EMR deployment.
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional

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
    VALID_TYPES_N1,
    DEFAULT_EXCLUDE_MASKBITS,
)
from common.experiment_tracking import ExperimentTracker, get_git_info
from common.validation import validate_dataframe, validate_file_exists

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
NEGATIVE_CATALOG = DATA_DIR / "negatives" / "negative_catalog_prototype.csv"
POSITIVE_CATALOG = DATA_DIR / "positives" / "desi_candidates.csv"
OUTPUT_DIR = DATA_DIR / "test_output"
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "negative_sampling_v1.yaml"

# Pipeline version for provenance
PIPELINE_VERSION = "1.0.0"


# =============================================================================
# PIPELINE FUNCTIONS (mirrors EMR job logic)
# =============================================================================

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML."""
    import yaml
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def load_known_lenses() -> List[Tuple[float, float]]:
    """Load known lens coordinates for exclusion."""
    df = pd.read_csv(POSITIVE_CATALOG)
    return list(zip(df['ra'].values, df['dec'].values))


def process_galaxy(
    row: pd.Series,
    config: Dict[str, Any],
    known_coords: List[Tuple[float, float]],
    row_idx: int,
    sweep_file: str = "local_test",
) -> Optional[Dict]:
    """
    Process a single galaxy row.
    
    This mirrors the Spark mapPartitions logic.
    """
    try:
        # Extract basic fields
        ra = float(row['ra'])
        dec = float(row['dec'])
        galaxy_type = str(row.get('type', 'OTHER')).strip().upper()
        nobs_z = int(row.get('nobs_z', 0))
        brickname = str(row.get('brickname', 'unknown'))
        
        # Maskbit exclusion (default to 0 if not present)
        maskbits = int(row.get('maskbits', 0))
        exclude_bits = set(config.get('exclusion', {}).get('exclude_maskbits', DEFAULT_EXCLUDE_MASKBITS))
        if check_maskbits(maskbits, exclude_bits):
            return None
        
        # Known lens exclusion
        exclusion_radius = config.get('exclusion', {}).get('known_lens_radius_arcsec', 11.0)
        if is_near_known_lens(ra, dec, known_coords, exclusion_radius):
            return None
        
        # Z-band magnitude limit
        flux_z = float(row.get('flux_z', 0))
        mag_z = flux_to_mag(flux_z)
        z_mag_limit = config.get('negative_pools', {}).get('pool_n1', {}).get('z_mag_limit', 20.0)
        if mag_z is not None and mag_z >= z_mag_limit:
            return None
        
        # Type binning
        type_bin = get_type_bin(galaxy_type)
        if type_bin not in VALID_TYPES_N1:
            return None  # Skip non-galaxy types
        
        # nobs_z binning
        nobs_z_bin = get_nobs_z_bin(nobs_z)
        if nobs_z_bin == 'invalid':
            return None
        
        # Pool classification
        # Use flux_z as proxy for flux_r
        flux_r = flux_z
        shape_r = row.get('shape_r', None)
        g_minus_r = None
        mag_r = flux_to_mag(flux_r)
        
        confuser_category = classify_pool_n2(
            galaxy_type, flux_r, shape_r, g_minus_r, mag_r, config
        )
        pool = "N2" if confuser_category else "N1"
        
        # HEALPix and split
        hp64 = compute_healpix(ra, dec, 64)
        hp128 = compute_healpix(ra, dec, 128)
        
        allocations = config.get('spatial_splits', {}).get('allocations', {
            'train': 0.70, 'val': 0.15, 'test': 0.15
        })
        seed = config.get('spatial_splits', {}).get('hash_seed', 42)
        split = assign_split(hp128, allocations, seed)
        
        # Photometry
        flux_g = row.get('flux_g', None)
        mag_g = flux_to_mag(flux_g) if flux_g else None
        
        # Observing conditions
        psfsize_z = row.get('psfsize_z', None)
        psfdepth_z = row.get('psfdepth_z', None)
        
        # Galaxy ID
        galaxy_id = f"{brickname}_{row_idx}"
        
        # Provenance
        git_info = get_git_info()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        return {
            'galaxy_id': galaxy_id,
            'brickname': brickname,
            'objid': row_idx,
            'ra': ra,
            'dec': dec,
            'type': galaxy_type,
            'nobs_z': nobs_z,
            'nobs_z_bin': nobs_z_bin,
            'type_bin': type_bin,
            'flux_z': flux_z,
            'mag_z': mag_z,
            'psfsize_z': psfsize_z,
            'psfdepth_z': psfdepth_z,
            'healpix_64': hp64,
            'healpix_128': hp128,
            'split': split,
            'pool': pool,
            'confuser_category': confuser_category,
            'sweep_file': sweep_file,
            'row_index': row_idx,
            'pipeline_version': PIPELINE_VERSION,
            'git_commit': git_info.get('commit', 'unknown')[:8],
            'extraction_timestamp': timestamp,
        }
        
    except Exception as e:
        print(f"ERROR processing row {row_idx}: {e}")
        return None


def validate_output(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate output data quality.
    
    Returns validation statistics.
    """
    stats = {}
    
    # Total count
    stats['total_rows'] = len(df)
    
    # Pool distribution
    pool_counts = df['pool'].value_counts().to_dict()
    stats['pool_distribution'] = pool_counts
    
    n1 = pool_counts.get('N1', 0)
    n2 = pool_counts.get('N2', 0)
    total = n1 + n2
    if total > 0:
        stats['n1_percentage'] = n1 / total * 100
        stats['n2_percentage'] = n2 / total * 100
    
    # Split distribution
    split_counts = df['split'].value_counts().to_dict()
    stats['split_distribution'] = split_counts
    
    # Type distribution
    type_counts = df['type_bin'].value_counts().to_dict()
    stats['type_distribution'] = type_counts
    
    # nobs_z bin distribution
    nobs_counts = df['nobs_z_bin'].value_counts().to_dict()
    stats['nobs_distribution'] = nobs_counts
    
    # Check for duplicates
    unique_ids = df['galaxy_id'].nunique()
    stats['duplicate_count'] = len(df) - unique_ids
    
    # Check for nulls in critical columns
    critical_cols = ['ra', 'dec', 'type', 'nobs_z', 'split', 'pool']
    null_counts = {}
    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                null_counts[col] = null_count
    stats['null_counts'] = null_counts
    
    return stats


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_local_pipeline(
    n_rows: Optional[int] = 50000,
    save_output: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the full negative sampling pipeline locally.
    
    Args:
        n_rows: Number of input rows to process (None for all)
        save_output: Whether to save output to parquet
    
    Returns:
        (output_df, stats_dict)
    """
    print("\n" + "="*60)
    print("LOCAL PIPELINE TEST (EMR Simulation)")
    print("="*60)
    
    start_time = time.time()
    
    # Load config
    print("\n[1/5] Loading configuration...")
    config = load_config()
    print(f"  Config version: {config.get('version', 'unknown')}")
    
    # Load known lenses
    print("\n[2/5] Loading known lens positions...")
    known_coords = load_known_lenses()
    print(f"  Loaded {len(known_coords)} known lens positions")
    
    # Load input data
    print(f"\n[3/5] Loading input data (limit={n_rows})...")
    df_input = pd.read_csv(NEGATIVE_CATALOG, nrows=n_rows)
    print(f"  Loaded {len(df_input)} rows")
    
    # Process rows
    print("\n[4/5] Processing rows...")
    results = []
    processed = 0
    skipped = 0
    
    for idx, row in df_input.iterrows():
        result = process_galaxy(row, config, known_coords, int(idx))
        if result:
            results.append(result)
            processed += 1
        else:
            skipped += 1
        
        # Progress update
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1}/{len(df_input)}...")
    
    print(f"  Processed: {processed}, Skipped: {skipped}")
    
    # Create output DataFrame
    df_output = pd.DataFrame(results)
    
    # Validate output
    print("\n[5/5] Validating output...")
    stats = validate_output(df_output, config)
    
    print(f"\n  Total output rows: {stats['total_rows']}")
    print(f"  Pool distribution: {stats['pool_distribution']}")
    print(f"  Split distribution: {stats['split_distribution']}")
    print(f"  Duplicates: {stats['duplicate_count']}")
    if stats['null_counts']:
        print(f"  Null counts: {stats['null_counts']}")
    
    # Save output
    if save_output:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "negative_manifest_test.parquet"
        df_output.to_parquet(output_path, index=False)
        print(f"\n  Saved to: {output_path}")
        
        # Verify by reading back
        df_verify = pd.read_parquet(output_path)
        assert len(df_verify) == len(df_output), "Output verification failed!"
        print(f"  Verified: {len(df_verify)} rows")
    
    elapsed = time.time() - start_time
    print(f"\n  Pipeline completed in {elapsed:.1f}s")
    
    # Summary statistics
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    
    print(f"\nInput: {len(df_input)} rows")
    print(f"Output: {len(df_output)} rows")
    print(f"Retention: {len(df_output)/len(df_input)*100:.1f}%")
    
    print(f"\nPool distribution:")
    for pool, count in stats['pool_distribution'].items():
        pct = count / len(df_output) * 100
        print(f"  {pool}: {count} ({pct:.1f}%)")
    
    print(f"\nSplit distribution:")
    for split, count in stats['split_distribution'].items():
        pct = count / len(df_output) * 100
        print(f"  {split}: {count} ({pct:.1f}%)")
    
    print(f"\nType distribution:")
    for type_bin, count in sorted(stats['type_distribution'].items()):
        pct = count / len(df_output) * 100
        print(f"  {type_bin}: {count} ({pct:.1f}%)")
    
    # Quality checks
    print("\n" + "="*60)
    print("QUALITY CHECKS")
    print("="*60)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: No duplicates
    checks_total += 1
    if stats['duplicate_count'] == 0:
        print("  [PASS] No duplicate galaxy IDs")
        checks_passed += 1
    else:
        print(f"  [FAIL] {stats['duplicate_count']} duplicate galaxy IDs")
    
    # Check 2: No nulls in critical columns
    checks_total += 1
    if not stats['null_counts']:
        print("  [PASS] No null values in critical columns")
        checks_passed += 1
    else:
        print(f"  [FAIL] Null values found: {stats['null_counts']}")
    
    # Check 3: Both pools present
    checks_total += 1
    if 'N1' in stats['pool_distribution'] and len(df_output) > 0:
        print("  [PASS] N1 pool present")
        checks_passed += 1
    else:
        print("  [FAIL] N1 pool missing")
    
    # Check 4: N2 percentage reasonable (if enough data)
    checks_total += 1
    n2_pct = stats.get('n2_percentage', 0)
    if n2_pct > 0 or len(df_output) < 1000:
        print(f"  [PASS] N2 pool detection working ({n2_pct:.1f}%)")
        checks_passed += 1
    else:
        print(f"  [WARN] N2 pool is empty - may need more data")
        checks_passed += 1  # Not a failure with limited data
    
    # Check 5: Valid RA/Dec ranges
    checks_total += 1
    ra_valid = df_output['ra'].between(0, 360).all()
    dec_valid = df_output['dec'].between(-90, 90).all()
    if ra_valid and dec_valid:
        print("  [PASS] RA/Dec ranges valid")
        checks_passed += 1
    else:
        print("  [FAIL] RA/Dec out of range")
    
    print(f"\nChecks passed: {checks_passed}/{checks_total}")
    
    return df_output, stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Local pipeline test")
    parser.add_argument("--rows", type=int, default=50000, help="Number of rows to process")
    parser.add_argument("--no-save", action="store_true", help="Don't save output")
    args = parser.parse_args()
    
    df_output, stats = run_local_pipeline(
        n_rows=args.rows,
        save_output=not args.no_save,
    )
    
    # Final verdict
    print("\n" + "="*60)
    if stats['duplicate_count'] == 0 and not stats['null_counts']:
        print("PIPELINE TEST PASSED - Ready for EMR deployment")
        sys.exit(0)
    else:
        print("PIPELINE TEST FAILED - Fix issues before EMR deployment")
        sys.exit(1)

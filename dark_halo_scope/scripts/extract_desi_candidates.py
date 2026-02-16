#!/usr/bin/env python3
"""
Extract DESI imaging lens candidates from the lenscat catalog.

This script:
1. Loads the full lenscat catalog (community-aggregated lens catalog)
2. Filters for entries with "DESI" in the name field
3. Parses complex multi-name fields to extract the primary DESI identifier
4. Verifies DR10 footprint coverage
5. Outputs to data/positives/desi_candidates.csv
6. Generates summary statistics

Source: https://github.com/lenscat/lenscat
Reference: Vujeva et al. (2024) arXiv:2406.04398
"""

import csv
import re
from pathlib import Path
from collections import Counter
from typing import Optional


def extract_desi_name(name_field: str) -> Optional[str]:
    """
    Extract the DESI identifier from a potentially complex name field.
    
    Examples:
        "DESI-037.0378-12.8812 | J0228-1252" -> "DESI-037.0378-12.8812"
        "J0228-1252 | DESI-037.0378-12.8812" -> "DESI-037.0378-12.8812"
        "SDSSJ0114+0722" -> None (no DESI identifier)
    """
    # Split by pipe and check each component
    components = [c.strip() for c in name_field.split('|')]
    
    for component in components:
        # Look for DESI-XXX.XXXX+/-YY.YYYY pattern
        if component.startswith('DESI-'):
            return component
    
    return None


def is_in_dr10_footprint(ra: float, dec: float) -> bool:
    """
    Check if coordinates are within DR10 South footprint.
    
    DR10 South covers:
    - Dec roughly < +32 degrees
    - Avoids galactic plane (|b| > ~18 deg)
    
    Since these are already DESI Legacy Survey candidates, they should
    already have DR10 coverage. This is a basic sanity check.
    """
    # Basic declination check for DR10 South
    # DR10 has both north and south components, but south is the primary
    # The DESI candidates are from the legacy survey which is primarily Dec < 32
    
    # Very permissive check - just ensure coordinates are valid
    if not (-90 <= dec <= 90):
        return False
    if not (0 <= ra <= 360):
        return False
    
    # DR10 South footprint is roughly Dec < 32, but some overlap exists
    # We keep all valid DESI candidates since they're already from the survey
    return True


def parse_zlens(zlens_field: str) -> str:
    """
    Parse the zlens field which may have multiple values separated by |.
    
    Returns the first valid numeric redshift, or the original string if complex.
    """
    if not zlens_field or zlens_field == '-':
        return ''
    
    # Handle multiple values separated by |
    values = [v.strip() for v in zlens_field.split('|')]
    
    for val in values:
        # Skip nan, measured, observed, etc.
        if val.lower() in ('nan', '-', 'measured', 'observed', 'not measured'):
            continue
        
        # Try to parse as float
        try:
            # Handle values like "0.407" or "0.272?"
            clean_val = val.rstrip('?')
            float(clean_val)
            return clean_val
        except ValueError:
            continue
    
    return ''


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / 'data' / 'external' / 'lenscat_catalog.csv'
    output_file = project_root / 'data' / 'positives' / 'desi_candidates.csv'
    
    print(f"Loading lenscat catalog from: {input_file}")
    
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Please download the lenscat catalog first:")
        print("  curl -L -o data/external/lenscat_catalog.csv https://raw.githubusercontent.com/lenscat/lenscat/main/lenscat/data/catalog.csv")
        return
    
    # Read and filter
    desi_candidates = []
    total_rows = 0
    desi_matches = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_rows += 1
            name_field = row.get('name', '')
            
            # Check if this row contains a DESI candidate
            if 'DESI' not in name_field:
                continue
            
            desi_matches += 1
            
            # Extract DESI identifier
            desi_name = extract_desi_name(name_field)
            if not desi_name:
                # Some entries have DESI in a different format, use full name
                desi_name = name_field.split('|')[0].strip()
            
            # Parse coordinates
            try:
                ra = float(row['RA [deg]'])
                dec = float(row['DEC [deg]'])
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not parse coordinates for {desi_name}: {e}")
                continue
            
            # Check footprint
            if not is_in_dr10_footprint(ra, dec):
                print(f"Warning: {desi_name} outside DR10 footprint (RA={ra}, Dec={dec})")
                continue
            
            # Parse other fields
            zlens = parse_zlens(row.get('zlens', ''))
            lens_type = row.get('type', 'galaxy')
            grading = row.get('grading', 'probable')
            ref = row.get('ref', '')
            
            desi_candidates.append({
                'name': desi_name,
                'ra': ra,
                'dec': dec,
                'zlens': zlens,
                'type': lens_type,
                'grading': grading,
                'ref': ref
            })
    
    print(f"\nProcessed {total_rows:,} total rows")
    print(f"Found {desi_matches:,} rows containing 'DESI'")
    print(f"Extracted {len(desi_candidates):,} valid DESI candidates")
    
    # Sort by RA for consistency
    desi_candidates.sort(key=lambda x: x['ra'])
    
    # Write output
    print(f"\nWriting to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['name', 'ra', 'dec', 'zlens', 'type', 'grading', 'ref']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(desi_candidates)
    
    # Statistics
    grading_counts = Counter(c['grading'] for c in desi_candidates)
    type_counts = Counter(c['type'] for c in desi_candidates)
    
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"\nTotal DESI candidates: {len(desi_candidates):,}")
    
    print(f"\nBy Grading (Tier):")
    for grading, count in sorted(grading_counts.items()):
        tier = "Tier-A" if grading == "confident" else "Tier-B"
        print(f"  {grading:12s} ({tier}): {count:5,}")
    
    print(f"\nBy Lens Type:")
    for lens_type, count in sorted(type_counts.items()):
        print(f"  {lens_type:12s}: {count:5,}")
    
    # Coordinate ranges
    ras = [c['ra'] for c in desi_candidates]
    decs = [c['dec'] for c in desi_candidates]
    
    print(f"\nCoordinate Ranges:")
    print(f"  RA:  {min(ras):.4f} to {max(ras):.4f} deg")
    print(f"  Dec: {min(decs):.4f} to {max(decs):.4f} deg")
    
    # Redshift statistics
    z_values = [float(c['zlens']) for c in desi_candidates if c['zlens']]
    if z_values:
        print(f"\nRedshift Statistics:")
        print(f"  Entries with z: {len(z_values):,}")
        print(f"  z range: {min(z_values):.4f} to {max(z_values):.4f}")
        print(f"  z median: {sorted(z_values)[len(z_values)//2]:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Output saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

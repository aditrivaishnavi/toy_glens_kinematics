#!/usr/bin/env python3
"""
Analyze DESI Strong Lensing Catalog for independent validation use.

This script:
1. Loads the DESI DR1 Strong Lensing VAC
2. Reports summary statistics
3. Cross-matches against our existing candidates (if available)
4. Identifies usable samples for independent validation
"""

import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    from astropy import units as u
except ImportError:
    print("ERROR: astropy not installed. Run: pip install astropy")
    sys.exit(1)


def load_desi_catalog(filepath: str) -> dict:
    """Load DESI SL catalog and return as dict of arrays."""
    with fits.open(filepath) as hdu:
        data = hdu[1].data
        columns = hdu[1].columns.names
        
        result = {
            'n_rows': len(data),
            'columns': columns,
            'data': {}
        }
        
        # Extract key columns
        key_cols = [
            'TARGET_RA', 'TARGET_DEC', 'Z', 'ZERR', 'ZWARN', 
            'SPECTYPE', 'BRICKNAME', 'MORPHTYPE', 
            'FLUX_G', 'FLUX_R', 'FLUX_Z',
            'HEALPIX', 'DELTACHI2', 'ZCAT_PRIMARY'
        ]
        
        for col in key_cols:
            if col in columns:
                result['data'][col] = np.array(data[col])
            else:
                print(f"  Warning: Column {col} not found")
        
        return result


def analyze_catalog(catalog: dict) -> dict:
    """Analyze catalog and return statistics."""
    data = catalog['data']
    n = catalog['n_rows']
    
    stats = {
        'total_rows': n,
        'total_columns': len(catalog['columns']),
    }
    
    # Redshift analysis
    if 'Z' in data:
        z = data['Z']
        z_valid = z[(z > 0) & (z < 10)]
        stats['redshift'] = {
            'n_valid': len(z_valid),
            'n_invalid': n - len(z_valid),
            'min': float(np.min(z_valid)) if len(z_valid) > 0 else None,
            'max': float(np.max(z_valid)) if len(z_valid) > 0 else None,
            'median': float(np.median(z_valid)) if len(z_valid) > 0 else None,
            'mean': float(np.mean(z_valid)) if len(z_valid) > 0 else None,
        }
    
    # ZWARN analysis
    if 'ZWARN' in data:
        zwarn = data['ZWARN']
        stats['zwarn'] = {
            'n_zwarn_0': int(np.sum(zwarn == 0)),
            'n_zwarn_nonzero': int(np.sum(zwarn != 0)),
            'unique_values': list(np.unique(zwarn)[:10]),  # First 10 unique
        }
    
    # SPECTYPE analysis
    if 'SPECTYPE' in data:
        spectype = data['SPECTYPE']
        unique, counts = np.unique(spectype, return_counts=True)
        stats['spectype'] = {str(t.strip()): int(c) for t, c in zip(unique, counts)}
    
    # MORPHTYPE analysis
    if 'MORPHTYPE' in data:
        morphtype = data['MORPHTYPE']
        unique, counts = np.unique(morphtype, return_counts=True)
        stats['morphtype'] = {str(t.strip()): int(c) for t, c in zip(unique, counts)}
    
    # Coordinate coverage
    if 'TARGET_RA' in data and 'TARGET_DEC' in data:
        ra = data['TARGET_RA']
        dec = data['TARGET_DEC']
        stats['coordinates'] = {
            'ra_min': float(np.min(ra)),
            'ra_max': float(np.max(ra)),
            'dec_min': float(np.min(dec)),
            'dec_max': float(np.max(dec)),
        }
    
    # Flux statistics (for magnitude calculation)
    if 'FLUX_R' in data:
        flux_r = data['FLUX_R']
        flux_valid = flux_r[flux_r > 0]
        if len(flux_valid) > 0:
            mag_r = 22.5 - 2.5 * np.log10(flux_valid)
            stats['r_magnitude'] = {
                'n_valid': int(len(flux_valid)),
                'min': float(np.min(mag_r)),
                'max': float(np.max(mag_r)),
                'median': float(np.median(mag_r)),
            }
    
    # HEALPix distribution
    if 'HEALPIX' in data:
        healpix = data['HEALPIX']
        stats['healpix'] = {
            'n_unique_cells': int(len(np.unique(healpix))),
            'nside': 64,  # From documentation
        }
    
    # Primary flag
    if 'ZCAT_PRIMARY' in data:
        primary = data['ZCAT_PRIMARY']
        stats['primary_flag'] = {
            'n_primary': int(np.sum(primary)),
            'n_secondary': int(np.sum(~primary)),
        }
    
    return stats


def identify_validation_samples(catalog: dict) -> dict:
    """Identify samples suitable for independent validation."""
    data = catalog['data']
    n = catalog['n_rows']
    
    # Create boolean masks for quality cuts
    masks = {}
    
    # 1. Valid redshift
    if 'Z' in data:
        masks['valid_z'] = (data['Z'] > 0) & (data['Z'] < 10)
    
    # 2. No redshift warning
    if 'ZWARN' in data:
        masks['zwarn_0'] = data['ZWARN'] == 0
    
    # 3. Galaxy spectype
    if 'SPECTYPE' in data:
        spectype = np.array([s.strip() for s in data['SPECTYPE']])
        masks['is_galaxy'] = spectype == 'GALAXY'
    
    # 4. Primary observation
    if 'ZCAT_PRIMARY' in data:
        masks['is_primary'] = data['ZCAT_PRIMARY']
    
    # Combine masks
    combined = np.ones(n, dtype=bool)
    for name, mask in masks.items():
        combined = combined & mask
    
    validation_samples = {
        'n_total': n,
        'quality_cuts': {name: int(np.sum(mask)) for name, mask in masks.items()},
        'n_passing_all': int(np.sum(combined)),
        'passing_indices': np.where(combined)[0],
    }
    
    # Get sample info for passing samples
    if np.sum(combined) > 0:
        passing = combined
        validation_samples['passing_sample'] = {
            'ra': data['TARGET_RA'][passing].tolist()[:10],  # First 10
            'dec': data['TARGET_DEC'][passing].tolist()[:10],
            'z': data['Z'][passing].tolist()[:10],
            'brickname': [str(b) for b in data['BRICKNAME'][passing][:10]],
        }
    
    return validation_samples


def crossmatch_with_candidates(catalog: dict, candidates_path: str = None) -> dict:
    """Cross-match DESI catalog with our imaging candidates."""
    
    # Try to find our candidates file
    if candidates_path is None:
        possible_paths = [
            project_root / 'data' / 'positives' / 'desi_candidates.csv',
            project_root / 'data' / 'positives' / 'desi_candidates_enriched.csv',
        ]
        for p in possible_paths:
            if p.exists():
                candidates_path = str(p)
                break
    
    result = {
        'candidates_path': candidates_path,
        'candidates_found': False,
        'crossmatch_done': False,
    }
    
    if candidates_path is None or not Path(candidates_path).exists():
        result['message'] = "No imaging candidates file found. Cross-match skipped."
        return result
    
    # If we found candidates, do cross-match
    try:
        import pandas as pd
        candidates = pd.read_csv(candidates_path)
        result['candidates_found'] = True
        result['n_candidates'] = len(candidates)
        
        # Create SkyCoord for both
        desi_coords = SkyCoord(
            ra=catalog['data']['TARGET_RA'] * u.deg,
            dec=catalog['data']['TARGET_DEC'] * u.deg
        )
        
        cand_coords = SkyCoord(
            ra=candidates['ra'].values * u.deg,
            dec=candidates['dec'].values * u.deg
        )
        
        # Cross-match with 5 arcsec radius
        idx, sep, _ = cand_coords.match_to_catalog_sky(desi_coords)
        matches = sep < 5 * u.arcsec
        
        result['crossmatch_done'] = True
        result['match_radius_arcsec'] = 5.0
        result['n_matches'] = int(np.sum(matches))
        result['n_unique_desi_matched'] = int(len(np.unique(idx[matches])))
        result['match_fraction'] = float(np.sum(matches)) / len(candidates)
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def generate_report(catalog: dict, stats: dict, validation: dict, crossmatch: dict) -> str:
    """Generate a markdown report."""
    
    report = []
    report.append("# DESI DR1 Strong Lensing Catalog Analysis Report")
    report.append(f"\n**Generated**: {__import__('datetime').datetime.now().isoformat()}")
    report.append(f"**Purpose**: Assess catalog for independent validation of our lens finder")
    report.append("")
    
    # Summary
    report.append("## 1. Catalog Summary")
    report.append("")
    report.append(f"- **Total rows**: {stats['total_rows']}")
    report.append(f"- **Total columns**: {stats['total_columns']}")
    report.append("")
    
    # Redshift
    if 'redshift' in stats:
        z = stats['redshift']
        report.append("### Redshift Distribution")
        report.append(f"- Valid redshifts: {z['n_valid']} ({100*z['n_valid']/stats['total_rows']:.1f}%)")
        report.append(f"- Invalid/missing: {z['n_invalid']}")
        if z['median']:
            report.append(f"- Range: {z['min']:.4f} - {z['max']:.4f}")
            report.append(f"- Median: {z['median']:.4f}")
            report.append(f"- Mean: {z['mean']:.4f}")
        report.append("")
    
    # ZWARN
    if 'zwarn' in stats:
        zw = stats['zwarn']
        report.append("### Redshift Warning Flags")
        report.append(f"- ZWARN=0 (reliable): {zw['n_zwarn_0']} ({100*zw['n_zwarn_0']/stats['total_rows']:.1f}%)")
        report.append(f"- ZWARN≠0 (caution): {zw['n_zwarn_nonzero']}")
        report.append("")
    
    # Spectral types
    if 'spectype' in stats:
        report.append("### Spectral Type Distribution")
        for t, c in sorted(stats['spectype'].items(), key=lambda x: -x[1]):
            report.append(f"- {t}: {c} ({100*c/stats['total_rows']:.1f}%)")
        report.append("")
    
    # Morphological types
    if 'morphtype' in stats:
        report.append("### Morphological Type Distribution (Tractor)")
        for t, c in sorted(stats['morphtype'].items(), key=lambda x: -x[1]):
            report.append(f"- {t}: {c} ({100*c/stats['total_rows']:.1f}%)")
        report.append("")
    
    # Magnitude
    if 'r_magnitude' in stats:
        m = stats['r_magnitude']
        report.append("### r-band Magnitude")
        report.append(f"- Valid flux measurements: {m['n_valid']}")
        report.append(f"- Range: {m['min']:.2f} - {m['max']:.2f} mag")
        report.append(f"- Median: {m['median']:.2f} mag")
        report.append("")
    
    # Coordinates
    if 'coordinates' in stats:
        c = stats['coordinates']
        report.append("### Sky Coverage")
        report.append(f"- RA: {c['ra_min']:.2f}° - {c['ra_max']:.2f}°")
        report.append(f"- Dec: {c['dec_min']:.2f}° - {c['dec_max']:.2f}°")
        report.append("")
    
    # HEALPix
    if 'healpix' in stats:
        report.append("### Spatial Distribution")
        report.append(f"- Unique HEALPix cells (nside={stats['healpix']['nside']}): {stats['healpix']['n_unique_cells']}")
        report.append("")
    
    # Validation samples
    report.append("## 2. Samples for Independent Validation")
    report.append("")
    report.append("### Quality Cuts Applied")
    for cut, n in validation['quality_cuts'].items():
        report.append(f"- {cut}: {n} pass ({100*n/validation['n_total']:.1f}%)")
    report.append("")
    report.append(f"**Samples passing ALL cuts**: {validation['n_passing_all']} ({100*validation['n_passing_all']/validation['n_total']:.1f}%)")
    report.append("")
    
    if 'passing_sample' in validation:
        report.append("### Sample of Passing Entries (first 10)")
        report.append("| RA | Dec | z | Brickname |")
        report.append("|-----|-----|-----|-----------|")
        sample = validation['passing_sample']
        for i in range(min(10, len(sample['ra']))):
            report.append(f"| {sample['ra'][i]:.4f} | {sample['dec'][i]:.4f} | {sample['z'][i]:.4f} | {sample['brickname'][i]} |")
        report.append("")
    
    # Cross-match
    report.append("## 3. Cross-match with Our Imaging Candidates")
    report.append("")
    if crossmatch['candidates_found']:
        report.append(f"- Imaging candidates file: `{crossmatch['candidates_path']}`")
        report.append(f"- Number of imaging candidates: {crossmatch['n_candidates']}")
        if crossmatch['crossmatch_done']:
            report.append(f"- Match radius: {crossmatch['match_radius_arcsec']} arcsec")
            report.append(f"- **Matches found**: {crossmatch['n_matches']} ({100*crossmatch['match_fraction']:.1f}% of imaging candidates)")
            report.append(f"- Unique DESI entries matched: {crossmatch['n_unique_desi_matched']}")
    else:
        report.append(f"**Note**: {crossmatch.get('message', 'Cross-match not performed')}")
    report.append("")
    
    # Recommendations
    report.append("## 4. Recommendations for Use")
    report.append("")
    report.append("### As Independent Validation Set")
    report.append(f"- **{validation['n_passing_all']} high-quality spectroscopic observations** available")
    report.append("- These are spectroscopically-selected (different method from imaging-CNN)")
    report.append("- Breaks circularity with Paper IV ML candidates")
    report.append("")
    report.append("### Quality Criteria for Selection")
    report.append("- Use only `ZWARN=0` for reliable redshifts")
    report.append("- Use only `ZCAT_PRIMARY=True` to avoid duplicates")
    report.append("- Filter `SPECTYPE='GALAXY'` for galaxy-galaxy lenses")
    report.append("")
    report.append("### Integration Steps")
    report.append("1. Download catalog: `desi-sl-vac-v1.fits`")
    report.append("2. Apply quality cuts (ZWARN=0, ZCAT_PRIMARY=True)")
    report.append("3. Cross-match with our imaging candidates")
    report.append("4. Exclude overlapping systems from training")
    report.append("5. Use remaining DESI systems as independent validation")
    report.append("")
    
    return "\n".join(report)


def main():
    """Main function."""
    print("=" * 60)
    print("DESI DR1 Strong Lensing Catalog Analysis")
    print("=" * 60)
    print()
    
    # File path
    catalog_path = project_root / 'data' / 'external' / 'desi_dr1' / 'desi-sl-vac-v1.fits'
    
    if not catalog_path.exists():
        print(f"ERROR: Catalog not found at {catalog_path}")
        print("Please download from: https://data.desi.lbl.gov/public/dr1/vac/dr1/strong-lensing/v1.0/desi-sl-vac-v1.fits")
        return 1
    
    print(f"Loading catalog: {catalog_path}")
    catalog = load_desi_catalog(str(catalog_path))
    print(f"  Loaded {catalog['n_rows']} rows, {len(catalog['columns'])} columns")
    print()
    
    print("Analyzing catalog...")
    stats = analyze_catalog(catalog)
    print()
    
    print("Identifying validation samples...")
    validation = identify_validation_samples(catalog)
    print(f"  {validation['n_passing_all']} samples pass all quality cuts")
    print()
    
    print("Cross-matching with imaging candidates...")
    crossmatch = crossmatch_with_candidates(catalog)
    if crossmatch['crossmatch_done']:
        print(f"  Found {crossmatch['n_matches']} matches")
    else:
        print(f"  {crossmatch.get('message', 'Skipped')}")
    print()
    
    print("Generating report...")
    report = generate_report(catalog, stats, validation, crossmatch)
    
    # Save report
    report_path = project_root / 'docs' / 'DESI_CATALOG_ANALYSIS_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved to: {report_path}")
    print()
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total catalog entries: {stats['total_rows']}")
    print(f"High-quality validation samples: {validation['n_passing_all']}")
    if 'redshift' in stats and stats['redshift']['median']:
        print(f"Median redshift: {stats['redshift']['median']:.3f}")
    print()
    print("Report saved to: docs/DESI_CATALOG_ANALYSIS_REPORT.md")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

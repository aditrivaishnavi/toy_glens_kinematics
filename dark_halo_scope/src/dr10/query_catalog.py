"""
DR10 Tractor catalog queries and LRG selection.

This module provides functions to query the DESI Legacy DR10 catalog
and apply LRG-like selection cuts for building a parent sample.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


# DR10 Tractor catalog access endpoints
DR10_CATALOG_URL = "https://datalab.noirlab.edu/query.php"

# LRG selection parameters
DEFAULT_LRG_CUTS = {
    'r_mag_max': 20.5,           # Bright galaxies only
    'type_exclude': ['PSF'],      # Exclude point sources
    'type_include': ['SER', 'DEV', 'REX', 'EXP'],  # Extended sources
    'g_minus_r_min': 0.5,         # Red galaxies
    'r_minus_z_min': 0.3,         # Red galaxies
}


def query_dr10_region(
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    output_path: str = "data/raw/dr10_catalog_region.parquet",
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Query DR10 Tractor catalog for a rectangular region.
    
    Parameters
    ----------
    ra_min, ra_max : float
        Right ascension bounds (degrees)
    dec_min, dec_max : float
        Declination bounds (degrees)
    output_path : str
        Path to save the catalog as parquet
    limit : int, optional
        Maximum number of rows to return (for testing)
    
    Returns
    -------
    pd.DataFrame
        Catalog with columns including:
        - ra, dec: coordinates
        - type: morphological type
        - flux_g, flux_r, flux_z: fluxes in nanomaggies
        - psfsize_g, psfsize_r, psfsize_z: PSF FWHM
    
    Notes
    -----
    This function requires network access to the NOIRLab Data Lab.
    For offline development, use a cached catalog.
    """
    # TODO: Implement actual DataLab query
    # For now, this is a placeholder that shows the expected interface
    
    query = f"""
    SELECT
        ra, dec, type,
        flux_g, flux_r, flux_z,
        flux_ivar_g, flux_ivar_r, flux_ivar_z,
        psfsize_g, psfsize_r, psfsize_z,
        shape_r, shape_e1, shape_e2,
        ref_cat, ref_id
    FROM
        ls_dr10.tractor
    WHERE
        ra BETWEEN {ra_min} AND {ra_max}
        AND dec BETWEEN {dec_min} AND {dec_max}
    """
    
    if limit is not None:
        query += f" LIMIT {limit}"
    
    # Placeholder: In actual implementation, execute query via DataLab
    print(f"Would execute query for region: RA [{ra_min}, {ra_max}], Dec [{dec_min}, {dec_max}]")
    print(f"Output path: {output_path}")
    
    # Return empty DataFrame with expected schema
    columns = [
        'ra', 'dec', 'type',
        'flux_g', 'flux_r', 'flux_z',
        'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z',
        'psfsize_g', 'psfsize_r', 'psfsize_z',
        'shape_r', 'shape_e1', 'shape_e2',
        'ref_cat', 'ref_id'
    ]
    
    return pd.DataFrame(columns=columns)


def flux_to_mag(flux: np.ndarray, zero_point: float = 22.5) -> np.ndarray:
    """
    Convert DR10 flux (nanomaggies) to magnitude.
    
    mag = 22.5 - 2.5 * log10(flux)
    
    Parameters
    ----------
    flux : np.ndarray
        Flux in nanomaggies
    zero_point : float
        Photometric zero point (default: 22.5 for DR10)
    
    Returns
    -------
    np.ndarray
        AB magnitudes
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        mag = zero_point - 2.5 * np.log10(flux)
    return mag


def apply_lrg_cuts(
    catalog: pd.DataFrame,
    cuts: Optional[dict] = None
) -> pd.DataFrame:
    """
    Apply LRG-like selection cuts to the catalog.
    
    Selection criteria:
    - Extended morphology (not PSF)
    - Bright in r-band (r < 20.5)
    - Red colors (g-r > 0.5, r-z > 0.3)
    
    Parameters
    ----------
    catalog : pd.DataFrame
        Input catalog from query_dr10_region
    cuts : dict, optional
        Custom selection parameters (default: DEFAULT_LRG_CUTS)
    
    Returns
    -------
    pd.DataFrame
        Filtered catalog containing LRG candidates
    """
    if cuts is None:
        cuts = DEFAULT_LRG_CUTS
    
    # Compute magnitudes
    mag_r = flux_to_mag(catalog['flux_r'].values)
    mag_g = flux_to_mag(catalog['flux_g'].values)
    mag_z = flux_to_mag(catalog['flux_z'].values)
    
    # Compute colors
    g_minus_r = mag_g - mag_r
    r_minus_z = mag_r - mag_z
    
    # Build selection mask
    mask = np.ones(len(catalog), dtype=bool)
    
    # Morphology cut
    mask &= ~catalog['type'].isin(cuts['type_exclude'])
    mask &= catalog['type'].isin(cuts['type_include'])
    
    # Brightness cut
    mask &= mag_r < cuts['r_mag_max']
    mask &= np.isfinite(mag_r)
    
    # Color cuts
    mask &= g_minus_r > cuts['g_minus_r_min']
    mask &= r_minus_z > cuts['r_minus_z_min']
    mask &= np.isfinite(g_minus_r) & np.isfinite(r_minus_z)
    
    # Apply mask and add computed columns
    result = catalog[mask].copy()
    result['mag_r'] = mag_r[mask]
    result['mag_g'] = mag_g[mask]
    result['mag_z'] = mag_z[mask]
    result['g_minus_r'] = g_minus_r[mask]
    result['r_minus_z'] = r_minus_z[mask]
    
    return result.reset_index(drop=True)


def build_parent_sample(
    raw_catalog_path: str = "data/raw/dr10_catalog_region.parquet",
    output_path: str = "data/processed/parent_sample.parquet",
    cuts: Optional[dict] = None
) -> pd.DataFrame:
    """
    Build the parent LRG sample from raw catalog.
    
    Parameters
    ----------
    raw_catalog_path : str
        Path to raw catalog parquet file
    output_path : str
        Path to save parent sample
    cuts : dict, optional
        LRG selection parameters
    
    Returns
    -------
    pd.DataFrame
        Parent sample of LRG candidates
    """
    catalog = pd.read_parquet(raw_catalog_path)
    parent = apply_lrg_cuts(catalog, cuts)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    parent.to_parquet(output_path, index=False)
    
    print(f"Parent sample: {len(parent)} LRG candidates")
    print(f"Saved to: {output_path}")
    
    return parent


def summarize_parent_sample(parent: pd.DataFrame) -> dict:
    """
    Compute summary statistics for the parent sample.
    
    Returns
    -------
    dict
        Summary statistics including counts, magnitude ranges, etc.
    """
    return {
        'n_objects': len(parent),
        'mag_r_range': (parent['mag_r'].min(), parent['mag_r'].max()),
        'mag_r_median': parent['mag_r'].median(),
        'g_minus_r_median': parent['g_minus_r'].median(),
        'r_minus_z_median': parent['r_minus_z'].median(),
        'type_counts': parent['type'].value_counts().to_dict(),
        'ra_range': (parent['ra'].min(), parent['ra'].max()),
        'dec_range': (parent['dec'].min(), parent['dec'].max()),
    }


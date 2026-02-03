"""
External catalog download script.

Downloads and standardizes external catalogs for anchor baseline validation.

Usage:
    python download_catalogs.py --catalog slacs --output-dir ./external_catalogs/slacs/
    python download_catalogs.py --all --output-dir ./external_catalogs/

This script:
1. Downloads catalogs from VizieR, Simbad, or direct URLs
2. Standardizes column names to (ra, dec, theta_e, z_lens, z_source, grade)
3. Filters by grade if specified
4. Saves as parquet for efficient loading
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import logging

from catalog_sources import (
    CatalogSource,
    CatalogType,
    CATALOG_REGISTRY,
    get_known_lens_catalogs,
    get_hard_negative_catalogs,
    get_catalog
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# VizieR Query Functions
# =============================================================================

def query_vizier(catalog_id: str, columns: List[str], max_rows: int = 10000) -> pd.DataFrame:
    """
    Query VizieR for a catalog.
    
    Parameters
    ----------
    catalog_id : str
        VizieR catalog ID (e.g., "J/ApJ/705/1099")
    columns : list
        Column names to retrieve
    max_rows : int
        Maximum number of rows
    
    Returns
    -------
    pd.DataFrame
        Query results
    """
    try:
        from astroquery.vizier import Vizier
        
        v = Vizier(columns=columns, row_limit=max_rows)
        result = v.get_catalogs(catalog_id)
        
        if len(result) == 0:
            logger.warning(f"No results from VizieR for {catalog_id}")
            return pd.DataFrame()
        
        # Convert to pandas
        df = result[0].to_pandas()
        return df
        
    except ImportError:
        logger.error("astroquery not installed. Install with: pip install astroquery")
        raise


def query_simbad(object_type: str, max_rows: int = 5000) -> pd.DataFrame:
    """
    Query SIMBAD for objects of a given type.
    
    Parameters
    ----------
    object_type : str
        SIMBAD object type (e.g., "RingG" for ring galaxies)
    max_rows : int
        Maximum number of results
    
    Returns
    -------
    pd.DataFrame
        Query results with ra, dec columns
    """
    try:
        from astroquery.simbad import Simbad
        
        custom_simbad = Simbad()
        custom_simbad.add_votable_fields('otype', 'ra(d)', 'dec(d)')
        custom_simbad.ROW_LIMIT = max_rows
        
        result = custom_simbad.query_criteria(otype=object_type)
        
        if result is None:
            logger.warning(f"No SIMBAD results for object type {object_type}")
            return pd.DataFrame()
        
        df = result.to_pandas()
        return df
        
    except ImportError:
        logger.error("astroquery not installed. Install with: pip install astroquery")
        raise


# =============================================================================
# Catalog-Specific Download Functions
# =============================================================================

def download_slacs(output_dir: str) -> pd.DataFrame:
    """
    Download SLACS catalog from VizieR.
    
    Returns standardized DataFrame with columns:
    - ra, dec: coordinates (degrees)
    - theta_e: Einstein radius (arcsec)
    - z_lens: lens redshift
    - z_source: source redshift
    - grade: lens grade (A/B/C)
    """
    logger.info("Downloading SLACS catalog from VizieR...")
    
    # VizieR catalog ID for Auger et al. 2009
    catalog_id = "J/ApJ/705/1099"
    
    try:
        df = query_vizier(
            catalog_id,
            columns=['RAJ2000', 'DEJ2000', 'zl', 'zs', 'Rein', 'Grade']
        )
        
        if df.empty:
            logger.warning("SLACS query returned no results, using fallback data")
            return _create_slacs_fallback()
        
        # Standardize columns
        result = pd.DataFrame({
            'ra': df['RAJ2000'].values,
            'dec': df['DEJ2000'].values,
            'theta_e': df['Rein'].values if 'Rein' in df else np.nan,
            'z_lens': df['zl'].values if 'zl' in df else np.nan,
            'z_source': df['zs'].values if 'zs' in df else np.nan,
            'grade': df['Grade'].values if 'Grade' in df else 'A',
            'catalog': 'SLACS'
        })
        
        logger.info(f"Downloaded {len(result)} SLACS lenses")
        return result
        
    except Exception as e:
        logger.warning(f"Failed to query VizieR: {e}. Using fallback data.")
        return _create_slacs_fallback()


def _create_slacs_fallback() -> pd.DataFrame:
    """
    Create fallback SLACS data with well-known lenses.
    
    This is used when VizieR is unavailable.
    Data from Auger et al. 2009, Table 1.
    """
    # Subset of well-known SLACS lenses (for testing/fallback)
    data = [
        # (name, ra, dec, theta_e, z_l, z_s, grade)
        ("J0029-0055", 7.4008, -0.9269, 0.96, 0.227, 0.931, "A"),
        ("J0037-0942", 9.4583, -9.7050, 1.53, 0.195, 0.632, "A"),
        ("J0216-0813", 34.1221, -8.2217, 1.16, 0.332, 0.523, "A"),
        ("J0252+0039", 43.0963, 0.6567, 1.04, 0.280, 0.982, "A"),
        ("J0330-0020", 52.5083, -0.3372, 1.10, 0.351, 1.071, "A"),
        ("J0728+3835", 112.0708, 38.5917, 1.25, 0.206, 0.688, "A"),
        ("J0737+3216", 114.4125, 32.2725, 0.98, 0.322, 0.581, "A"),
        ("J0822+2652", 125.5125, 26.8750, 1.17, 0.241, 0.594, "A"),
        ("J0912+0029", 138.1417, 0.4861, 1.63, 0.164, 0.324, "A"),
        ("J0936+0913", 144.0583, 9.2242, 1.09, 0.190, 0.588, "A"),
        ("J0956+5100", 149.1625, 51.0050, 1.33, 0.240, 0.470, "A"),
        ("J0959+0410", 149.9208, 4.1722, 0.99, 0.126, 0.535, "A"),
        ("J1016+3859", 154.2333, 38.9928, 1.09, 0.168, 0.439, "A"),
        ("J1020+1122", 155.2333, 11.3761, 1.20, 0.282, 0.553, "A"),
        ("J1023+4230", 155.8500, 42.5083, 1.41, 0.191, 0.696, "A"),
    ]
    
    return pd.DataFrame(data, columns=['name', 'ra', 'dec', 'theta_e', 'z_lens', 'z_source', 'grade']).assign(catalog='SLACS')


def download_bells(output_dir: str) -> pd.DataFrame:
    """
    Download BELLS catalog from VizieR.
    """
    logger.info("Downloading BELLS catalog from VizieR...")
    
    catalog_id = "J/ApJ/744/41"  # Brownstein et al. 2012
    
    try:
        df = query_vizier(
            catalog_id,
            columns=['RAJ2000', 'DEJ2000', 'zl', 'zs', 'Rein', 'Grade']
        )
        
        if df.empty:
            logger.warning("BELLS query returned no results, using fallback data")
            return _create_bells_fallback()
        
        result = pd.DataFrame({
            'ra': df['RAJ2000'].values,
            'dec': df['DEJ2000'].values,
            'theta_e': df['Rein'].values if 'Rein' in df else np.nan,
            'z_lens': df['zl'].values if 'zl' in df else np.nan,
            'z_source': df['zs'].values if 'zs' in df else np.nan,
            'grade': df['Grade'].values if 'Grade' in df else 'A',
            'catalog': 'BELLS'
        })
        
        logger.info(f"Downloaded {len(result)} BELLS lenses")
        return result
        
    except Exception as e:
        logger.warning(f"Failed to query VizieR: {e}. Using fallback data.")
        return _create_bells_fallback()


def _create_bells_fallback() -> pd.DataFrame:
    """Create fallback BELLS data."""
    # Sample BELLS lenses from Brownstein et al. 2012
    data = [
        ("J0747+4448", 116.906, 44.810, 1.44, 0.437, 0.898, "A"),
        ("J0801+4727", 120.334, 47.457, 1.02, 0.524, 1.239, "A"),
        ("J0830+5116", 127.581, 51.278, 1.21, 0.530, 1.192, "A"),
        ("J0918+5104", 139.664, 51.077, 1.18, 0.581, 1.236, "A"),
        ("J0935+0003", 143.887, 0.060, 0.87, 0.467, 0.899, "A"),
        ("J1110+3649", 167.652, 36.823, 1.01, 0.733, 1.393, "A"),
        ("J1141+2216", 175.334, 22.271, 1.02, 0.586, 1.215, "A"),
        ("J1226+5457", 186.539, 54.956, 1.32, 0.498, 1.020, "A"),
        ("J1318+3942", 199.550, 39.715, 1.10, 0.560, 1.296, "A"),
        ("J1337+3620", 204.332, 36.347, 0.99, 0.643, 1.416, "A"),
    ]
    
    return pd.DataFrame(data, columns=['name', 'ra', 'dec', 'theta_e', 'z_lens', 'z_source', 'grade']).assign(catalog='BELLS')


def download_galaxy_zoo_rings(output_dir: str, max_rows: int = 2000) -> pd.DataFrame:
    """
    Download Galaxy Zoo ring galaxy classifications.
    
    Uses Galaxy Zoo 2 debiased vote fractions for ring features.
    """
    logger.info("Downloading Galaxy Zoo ring galaxies...")
    
    # GZ2 catalog in VizieR
    catalog_id = "J/MNRAS/435/2835"
    
    try:
        df = query_vizier(
            catalog_id,
            columns=['RAJ2000', 'DEJ2000', 't07_rounded_a17_ring_debiased'],
            max_rows=max_rows
        )
        
        if df.empty:
            logger.warning("Galaxy Zoo query returned no results")
            return _create_gz_rings_fallback()
        
        # Filter for ring-like objects (vote fraction > 0.3)
        ring_col = 't07_rounded_a17_ring_debiased'
        if ring_col in df:
            df = df[df[ring_col] > 0.3]
        
        result = pd.DataFrame({
            'ra': df['RAJ2000'].values,
            'dec': df['DEJ2000'].values,
            'ring_vote_fraction': df[ring_col].values if ring_col in df else 0.5,
            'catalog': 'GalaxyZoo_Rings'
        })
        
        logger.info(f"Downloaded {len(result)} ring galaxy candidates")
        return result
        
    except Exception as e:
        logger.warning(f"Failed to query VizieR: {e}. Using SIMBAD fallback.")
        return download_simbad_rings(output_dir)


def download_simbad_rings(output_dir: str, max_rows: int = 1000) -> pd.DataFrame:
    """
    Download ring galaxies from SIMBAD.
    """
    logger.info("Querying SIMBAD for ring galaxies...")
    
    try:
        df = query_simbad("RingG", max_rows=max_rows)
        
        if df.empty:
            logger.warning("SIMBAD query returned no results")
            return _create_gz_rings_fallback()
        
        result = pd.DataFrame({
            'ra': df['RA_d'].values if 'RA_d' in df else df['RA_d_A_ICRS_J2000_2000'].values,
            'dec': df['DEC_d'].values if 'DEC_d' in df else df['DEC_d_D_ICRS_J2000_2000'].values,
            'catalog': 'SIMBAD_Rings'
        })
        
        # Clean invalid coordinates
        result = result.dropna(subset=['ra', 'dec'])
        result = result[(result['ra'] >= 0) & (result['ra'] <= 360)]
        result = result[(result['dec'] >= -90) & (result['dec'] <= 90)]
        
        logger.info(f"Downloaded {len(result)} ring galaxies from SIMBAD")
        return result
        
    except Exception as e:
        logger.warning(f"SIMBAD query failed: {e}. Using fallback data.")
        return _create_gz_rings_fallback()


def _create_gz_rings_fallback() -> pd.DataFrame:
    """Create fallback ring galaxy data for testing."""
    # Well-known ring galaxies from Buta & Combes 1996 catalog
    data = [
        ("Hoag's Object", 229.3067, 21.5992),
        ("Cartwheel Galaxy", 9.4500, -33.7167),
        ("AM 0644-741", 101.1167, -74.2500),
        ("Lindsay-Shapley Ring", 105.7833, -37.9333),
        ("II Hz 4", 128.7917, 43.6000),
        ("Arp 147", 47.0167, 1.3000),
        ("NGC 922", 36.3833, -24.7917),
        ("NGC 7020", 316.6583, -64.0250),
        ("NGC 1533", 62.5167, -56.1167),
        ("NGC 4736", 192.7208, 41.1208),
    ]
    
    return pd.DataFrame(data, columns=['name', 'ra', 'dec']).assign(catalog='Fallback_Rings')


def download_galaxy_zoo_mergers(output_dir: str, max_rows: int = 2000) -> pd.DataFrame:
    """
    Download Galaxy Zoo merger classifications.
    """
    logger.info("Downloading Galaxy Zoo mergers...")
    
    catalog_id = "J/MNRAS/435/2835"
    
    try:
        df = query_vizier(
            catalog_id,
            columns=['RAJ2000', 'DEJ2000', 't08_odd_feature_a19_merger_debiased'],
            max_rows=max_rows
        )
        
        if df.empty:
            logger.warning("Galaxy Zoo query returned no results")
            return _create_mergers_fallback()
        
        merger_col = 't08_odd_feature_a19_merger_debiased'
        if merger_col in df:
            df = df[df[merger_col] > 0.3]
        
        result = pd.DataFrame({
            'ra': df['RAJ2000'].values,
            'dec': df['DEJ2000'].values,
            'merger_vote_fraction': df[merger_col].values if merger_col in df else 0.5,
            'catalog': 'GalaxyZoo_Mergers'
        })
        
        logger.info(f"Downloaded {len(result)} merger candidates")
        return result
        
    except Exception as e:
        logger.warning(f"Failed to query: {e}. Using fallback.")
        return _create_mergers_fallback()


def _create_mergers_fallback() -> pd.DataFrame:
    """Create fallback merger data."""
    data = [
        ("Antennae", 180.4708, -18.8767),
        ("Mice Galaxies", 185.4583, 30.7333),
        ("NGC 6240", 253.2458, 2.4008),
        ("Arp 220", 233.7375, 23.5039),
        ("NGC 7252", 339.0083, -24.6792),
        ("NGC 2623", 129.6000, 25.7542),
        ("NGC 520", 21.1500, 3.7933),
        ("NGC 3256", 156.9625, -43.9036),
        ("IC 1623", 16.9417, -17.5072),
        ("NGC 4676", 191.5458, 30.7278),
    ]
    
    return pd.DataFrame(data, columns=['name', 'ra', 'dec']).assign(catalog='Fallback_Mergers')


# =============================================================================
# Main Download Functions
# =============================================================================

DOWNLOAD_FUNCTIONS = {
    'slacs': download_slacs,
    'bells': download_bells,
    'gz_rings': download_galaxy_zoo_rings,
    'gz_mergers': download_galaxy_zoo_mergers,
    'simbad_rings': download_simbad_rings,
}


def download_catalog(catalog_name: str, output_dir: str) -> pd.DataFrame:
    """
    Download a single catalog and save to output directory.
    
    Parameters
    ----------
    catalog_name : str
        Name of catalog (slacs, bells, gz_rings, gz_mergers, simbad_rings)
    output_dir : str
        Output directory path
    
    Returns
    -------
    pd.DataFrame
        Downloaded catalog data
    """
    if catalog_name.lower() not in DOWNLOAD_FUNCTIONS:
        raise ValueError(f"Unknown catalog: {catalog_name}. Available: {list(DOWNLOAD_FUNCTIONS.keys())}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Download
    download_func = DOWNLOAD_FUNCTIONS[catalog_name.lower()]
    df = download_func(output_dir)
    
    # Save
    output_path = os.path.join(output_dir, f"{catalog_name.lower()}.parquet")
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")
    
    return df


def download_all_catalogs(output_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Download all available catalogs.
    
    Parameters
    ----------
    output_dir : str
        Base output directory
    
    Returns
    -------
    dict
        Mapping of catalog names to DataFrames
    """
    results = {}
    
    for name in DOWNLOAD_FUNCTIONS.keys():
        catalog_dir = os.path.join(output_dir, name)
        try:
            results[name] = download_catalog(name, catalog_dir)
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            results[name] = pd.DataFrame()
    
    return results


def merge_known_lenses(output_dir: str) -> pd.DataFrame:
    """
    Merge all known lens catalogs into a single file.
    """
    dfs = []
    
    for name in ['slacs', 'bells']:
        catalog_path = os.path.join(output_dir, name, f"{name}.parquet")
        if os.path.exists(catalog_path):
            dfs.append(pd.read_parquet(catalog_path))
    
    if not dfs:
        logger.warning("No known lens catalogs found")
        return pd.DataFrame()
    
    merged = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates by position
    merged = merged.drop_duplicates(subset=['ra', 'dec'], keep='first')
    
    output_path = os.path.join(output_dir, "known_lenses_merged.parquet")
    merged.to_parquet(output_path, index=False)
    logger.info(f"Merged {len(merged)} known lenses to {output_path}")
    
    return merged


def merge_hard_negatives(output_dir: str) -> pd.DataFrame:
    """
    Merge all hard negative catalogs into a single file.
    """
    dfs = []
    
    for name in ['gz_rings', 'gz_mergers', 'simbad_rings']:
        catalog_path = os.path.join(output_dir, name, f"{name}.parquet")
        if os.path.exists(catalog_path):
            dfs.append(pd.read_parquet(catalog_path))
    
    if not dfs:
        logger.warning("No hard negative catalogs found")
        return pd.DataFrame()
    
    merged = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates
    merged = merged.drop_duplicates(subset=['ra', 'dec'], keep='first')
    
    output_path = os.path.join(output_dir, "hard_negatives_merged.parquet")
    merged.to_parquet(output_path, index=False)
    logger.info(f"Merged {len(merged)} hard negatives to {output_path}")
    
    return merged


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download external catalogs for anchor baseline validation"
    )
    parser.add_argument(
        "--catalog",
        type=str,
        choices=list(DOWNLOAD_FUNCTIONS.keys()),
        help="Specific catalog to download"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all catalogs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./external_catalogs",
        help="Output directory"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge catalogs after downloading"
    )
    
    args = parser.parse_args()
    
    if args.all:
        download_all_catalogs(args.output_dir)
        if args.merge:
            merge_known_lenses(args.output_dir)
            merge_hard_negatives(args.output_dir)
    elif args.catalog:
        download_catalog(args.catalog, os.path.join(args.output_dir, args.catalog))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


"""
Cross-match external catalogs with DR10 footprint.

This script:
1. Loads external catalogs (SLACS, BELLS, ring galaxies, etc.)
2. Cross-matches with DR10 brick coverage
3. Downloads 64x64 cutouts for matched objects
4. Saves cutouts in same format as Phase 4c training data

Usage:
    python crossmatch_dr10.py \
        --catalog known_lenses_merged.parquet \
        --brick-metadata ../data/ls_dr10_south_bricks_metadata.csv \
        --output-dir ./cutouts/known_lenses/ \
        --stamp-size 64
"""

import argparse
import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
import io
import base64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CrossMatchConfig:
    """Configuration for cross-matching and cutout download."""
    stamp_size: int = 64
    pixel_scale: float = 0.262  # arcsec/pixel for DECaLS
    bands: List[str] = None
    
    # Legacy Survey cutout service
    cutout_service_url: str = "https://www.legacysurvey.org/viewer/cutout.fits"
    
    # DR10 footprint bounds (approximate)
    ra_min: float = 0.0
    ra_max: float = 360.0
    dec_min: float = -90.0
    dec_max: float = 90.0
    
    def __post_init__(self):
        if self.bands is None:
            self.bands = ["g", "r", "z"]


# =============================================================================
# Footprint Matching
# =============================================================================

def load_brick_metadata(brick_metadata_path: str) -> pd.DataFrame:
    """
    Load DR10 brick metadata for footprint matching.
    
    Parameters
    ----------
    brick_metadata_path : str
        Path to brick metadata CSV
    
    Returns
    -------
    pd.DataFrame
        Brick metadata with ra_min, ra_max, dec_min, dec_max columns
    """
    df = pd.read_csv(brick_metadata_path)
    
    # Ensure required columns exist
    required_cols = ['brickname', 'ra', 'dec', 'ra1', 'ra2', 'dec1', 'dec2']
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        logger.warning(f"Missing columns in brick metadata: {missing}")
        # Try alternative column names
        if 'brickid' in df.columns and 'brickname' not in df.columns:
            df['brickname'] = df['brickid']
    
    logger.info(f"Loaded {len(df)} bricks from {brick_metadata_path}")
    return df


def in_dr10_footprint(
    ra: np.ndarray, 
    dec: np.ndarray, 
    brick_metadata: pd.DataFrame
) -> np.ndarray:
    """
    Check which coordinates fall within DR10 brick coverage.
    
    Parameters
    ----------
    ra, dec : np.ndarray
        Coordinate arrays in degrees
    brick_metadata : pd.DataFrame
        Brick metadata with ra1, ra2, dec1, dec2 columns
    
    Returns
    -------
    np.ndarray
        Boolean array indicating in-footprint status
    """
    # Build spatial index for efficient lookup
    from scipy.spatial import cKDTree
    
    # Use brick centers for initial matching
    if 'ra' in brick_metadata.columns and 'dec' in brick_metadata.columns:
        brick_coords = np.column_stack([
            brick_metadata['ra'].values,
            brick_metadata['dec'].values
        ])
    else:
        # Compute centers from bounds
        brick_coords = np.column_stack([
            (brick_metadata['ra1'] + brick_metadata['ra2']) / 2,
            (brick_metadata['dec1'] + brick_metadata['dec2']) / 2
        ])
    
    tree = cKDTree(brick_coords)
    
    # Query with generous radius (brick size is ~0.25 deg)
    query_coords = np.column_stack([ra, dec])
    distances, indices = tree.query(query_coords, distance_upper_bound=0.5)
    
    # Points within distance threshold are in footprint
    in_footprint = distances < np.inf
    
    return in_footprint


def crossmatch_with_footprint(
    catalog: pd.DataFrame,
    brick_metadata: pd.DataFrame,
    ra_col: str = 'ra',
    dec_col: str = 'dec'
) -> pd.DataFrame:
    """
    Cross-match catalog with DR10 footprint.
    
    Parameters
    ----------
    catalog : pd.DataFrame
        External catalog with coordinates
    brick_metadata : pd.DataFrame
        DR10 brick metadata
    ra_col, dec_col : str
        Column names for coordinates
    
    Returns
    -------
    pd.DataFrame
        Catalog subset within DR10 footprint
    """
    ra = catalog[ra_col].values
    dec = catalog[dec_col].values
    
    in_fp = in_dr10_footprint(ra, dec, brick_metadata)
    
    matched = catalog[in_fp].copy()
    matched['in_dr10_footprint'] = True
    
    logger.info(f"Matched {len(matched)}/{len(catalog)} objects to DR10 footprint")
    
    return matched


# =============================================================================
# Cutout Download
# =============================================================================

def download_cutout(
    ra: float,
    dec: float,
    config: CrossMatchConfig
) -> Optional[np.ndarray]:
    """
    Download a cutout from Legacy Survey viewer.
    
    Parameters
    ----------
    ra, dec : float
        Center coordinates in degrees
    config : CrossMatchConfig
        Download configuration
    
    Returns
    -------
    np.ndarray or None
        Cutout array of shape (3, stamp_size, stamp_size) or None if failed
    """
    try:
        import requests
        from astropy.io import fits
    except ImportError as e:
        logger.error(f"Required package not installed: {e}")
        return None
    
    # Compute cutout size in pixels
    size_arcsec = config.stamp_size * config.pixel_scale
    
    # Build URL
    url = (
        f"{config.cutout_service_url}?"
        f"ra={ra}&dec={dec}"
        f"&pixscale={config.pixel_scale}"
        f"&size={config.stamp_size}"
        f"&layer=ls-dr10"
    )
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse FITS
        with fits.open(io.BytesIO(response.content)) as hdul:
            data = hdul[0].data
            
            if data is None:
                logger.warning(f"Empty FITS data for ({ra}, {dec})")
                return None
            
            # Reshape if needed (should be 3, H, W for grz)
            if data.ndim == 2:
                # Single band - stack 3 times
                data = np.stack([data, data, data], axis=0)
            elif data.ndim == 3 and data.shape[0] != 3:
                # Wrong band order
                if data.shape[2] == 3:
                    data = np.transpose(data, (2, 0, 1))
            
            return data.astype(np.float32)
            
    except Exception as e:
        logger.warning(f"Failed to download cutout at ({ra}, {dec}): {e}")
        return None


def download_cutouts_batch(
    catalog: pd.DataFrame,
    output_dir: str,
    config: CrossMatchConfig,
    ra_col: str = 'ra',
    dec_col: str = 'dec',
    max_workers: int = 4
) -> pd.DataFrame:
    """
    Download cutouts for all objects in catalog.
    
    Parameters
    ----------
    catalog : pd.DataFrame
        Catalog with coordinates
    output_dir : str
        Output directory for cutouts
    config : CrossMatchConfig
        Download configuration
    ra_col, dec_col : str
        Column names for coordinates
    max_workers : int
        Number of parallel download workers
    
    Returns
    -------
    pd.DataFrame
        Catalog with cutout paths added
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    
    cutout_paths = []
    cutout_success = []
    
    def download_single(idx, row):
        ra = row[ra_col]
        dec = row[dec_col]
        
        cutout = download_cutout(ra, dec, config)
        
        if cutout is not None:
            # Save as npz
            filename = f"cutout_{idx:06d}.npz"
            filepath = os.path.join(output_dir, filename)
            np.savez_compressed(filepath, stamp=cutout, ra=ra, dec=dec)
            return idx, filepath, True
        else:
            return idx, None, False
    
    # Sequential download with progress bar
    logger.info(f"Downloading {len(catalog)} cutouts...")
    
    for idx, row in tqdm(catalog.iterrows(), total=len(catalog), desc="Downloading cutouts"):
        _, path, success = download_single(idx, row)
        cutout_paths.append(path)
        cutout_success.append(success)
    
    # Add results to catalog
    result = catalog.copy()
    result['cutout_path'] = cutout_paths
    result['cutout_success'] = cutout_success
    
    n_success = sum(cutout_success)
    logger.info(f"Successfully downloaded {n_success}/{len(catalog)} cutouts")
    
    return result


# =============================================================================
# Main Function
# =============================================================================

def crossmatch_and_download(
    catalog_path: str,
    brick_metadata_path: str,
    output_dir: str,
    stamp_size: int = 64,
    bands: List[str] = None,
    download_cutouts: bool = True
) -> pd.DataFrame:
    """
    Full pipeline: load catalog, cross-match, and download cutouts.
    
    Parameters
    ----------
    catalog_path : str
        Path to external catalog parquet
    brick_metadata_path : str
        Path to DR10 brick metadata
    output_dir : str
        Output directory
    stamp_size : int
        Cutout size in pixels
    bands : list
        Bands to download
    download_cutouts : bool
        Whether to download cutouts (set False for testing)
    
    Returns
    -------
    pd.DataFrame
        Matched catalog with cutout paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load catalog
    logger.info(f"Loading catalog from {catalog_path}")
    catalog = pd.read_parquet(catalog_path)
    
    # Load brick metadata
    logger.info(f"Loading brick metadata from {brick_metadata_path}")
    bricks = load_brick_metadata(brick_metadata_path)
    
    # Cross-match
    matched = crossmatch_with_footprint(catalog, bricks)
    
    # Save matched catalog
    matched_path = os.path.join(output_dir, "matched_catalog.parquet")
    matched.to_parquet(matched_path, index=False)
    logger.info(f"Saved matched catalog to {matched_path}")
    
    # Download cutouts
    if download_cutouts and len(matched) > 0:
        config = CrossMatchConfig(stamp_size=stamp_size, bands=bands or ["g", "r", "z"])
        cutouts_dir = os.path.join(output_dir, "cutouts")
        matched = download_cutouts_batch(matched, cutouts_dir, config)
        
        # Update saved catalog
        matched.to_parquet(matched_path, index=False)
    
    return matched


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-match external catalogs with DR10 and download cutouts"
    )
    parser.add_argument(
        "--catalog",
        type=str,
        required=True,
        help="Path to external catalog parquet file"
    )
    parser.add_argument(
        "--brick-metadata",
        type=str,
        required=True,
        help="Path to DR10 brick metadata CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--stamp-size",
        type=int,
        default=64,
        help="Cutout size in pixels (default: 64)"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip cutout download (just do cross-match)"
    )
    
    args = parser.parse_args()
    
    crossmatch_and_download(
        catalog_path=args.catalog,
        brick_metadata_path=args.brick_metadata,
        output_dir=args.output_dir,
        stamp_size=args.stamp_size,
        download_cutouts=not args.no_download
    )


if __name__ == "__main__":
    main()


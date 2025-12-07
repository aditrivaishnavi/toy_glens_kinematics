"""
DR10 image cutout downloader.

This module provides functions to download g, r, z cutouts
from the DESI Legacy Survey image server.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import requests
from io import BytesIO


# Legacy Survey cutout service
CUTOUT_BASE_URL = "https://www.legacysurvey.org/viewer/cutout.fits"

# Default cutout parameters
DEFAULT_CUTOUT_SIZE = 128  # pixels
DEFAULT_PIXEL_SCALE = 0.262  # arcsec/pixel


def get_cutout_url(
    ra: float,
    dec: float,
    size: int = DEFAULT_CUTOUT_SIZE,
    pixscale: float = DEFAULT_PIXEL_SCALE,
    layer: str = "ls-dr10",
    bands: str = "grz"
) -> str:
    """
    Construct URL for Legacy Survey cutout service.
    
    Parameters
    ----------
    ra, dec : float
        Center coordinates (degrees)
    size : int
        Cutout size in pixels
    pixscale : float
        Pixel scale in arcsec/pixel
    layer : str
        Data layer (e.g., "ls-dr10")
    bands : str
        Bands to include (e.g., "grz")
    
    Returns
    -------
    str
        URL for the cutout service
    """
    return (
        f"{CUTOUT_BASE_URL}?"
        f"ra={ra}&dec={dec}&"
        f"size={size}&pixscale={pixscale}&"
        f"layer={layer}&bands={bands}"
    )


def download_single_cutout(
    ra: float,
    dec: float,
    size: int = DEFAULT_CUTOUT_SIZE,
    pixscale: float = DEFAULT_PIXEL_SCALE,
    timeout: int = 30
) -> Optional[dict]:
    """
    Download a single g, r, z cutout.
    
    Parameters
    ----------
    ra, dec : float
        Center coordinates (degrees)
    size : int
        Cutout size in pixels
    pixscale : float
        Pixel scale in arcsec/pixel
    timeout : int
        Request timeout in seconds
    
    Returns
    -------
    dict or None
        Dictionary with keys 'g', 'r', 'z' containing 2D arrays,
        or None if download failed
    """
    url = get_cutout_url(ra, dec, size, pixscale)
    
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        # Parse FITS from response
        from astropy.io import fits
        
        with fits.open(BytesIO(response.content)) as hdul:
            # DR10 cutouts have bands stacked in the primary HDU
            # Shape: (3, size, size) for grz
            data = hdul[0].data
            
            if data is None or data.ndim != 3 or data.shape[0] != 3:
                return None
            
            return {
                'g': data[0].astype(np.float32),
                'r': data[1].astype(np.float32),
                'z': data[2].astype(np.float32)
            }
    
    except Exception as e:
        print(f"Failed to download cutout at ({ra}, {dec}): {e}")
        return None


def download_cutouts(
    parent_sample: pd.DataFrame,
    output_dir: str = "data/raw/cutouts/",
    size: int = DEFAULT_CUTOUT_SIZE,
    pixscale: float = DEFAULT_PIXEL_SCALE,
    max_workers: int = 4,
    skip_existing: bool = True
) -> pd.DataFrame:
    """
    Download cutouts for all objects in parent sample.
    
    Parameters
    ----------
    parent_sample : pd.DataFrame
        Parent sample with 'ra', 'dec' columns
    output_dir : str
        Directory to save cutouts as .npz files
    size : int
        Cutout size in pixels
    pixscale : float
        Pixel scale in arcsec/pixel
    max_workers : int
        Number of parallel download threads
    skip_existing : bool
        Skip objects with existing cutout files
    
    Returns
    -------
    pd.DataFrame
        Updated parent sample with 'cutout_path' and 'cutout_status' columns
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Add status columns
    parent_sample = parent_sample.copy()
    parent_sample['cutout_path'] = ""
    parent_sample['cutout_status'] = "pending"
    
    from tqdm import tqdm
    
    for idx, row in tqdm(parent_sample.iterrows(), total=len(parent_sample), 
                         desc="Downloading cutouts"):
        # Generate filename
        filename = f"cutout_{idx:06d}.npz"
        filepath = output_path / filename
        
        # Skip if exists
        if skip_existing and filepath.exists():
            parent_sample.loc[idx, 'cutout_path'] = str(filepath)
            parent_sample.loc[idx, 'cutout_status'] = "cached"
            continue
        
        # Download
        cutout = download_single_cutout(row['ra'], row['dec'], size, pixscale)
        
        if cutout is not None:
            # Save as compressed numpy
            np.savez_compressed(
                filepath,
                g=cutout['g'],
                r=cutout['r'],
                z=cutout['z'],
                ra=row['ra'],
                dec=row['dec']
            )
            parent_sample.loc[idx, 'cutout_path'] = str(filepath)
            parent_sample.loc[idx, 'cutout_status'] = "success"
        else:
            parent_sample.loc[idx, 'cutout_status'] = "failed"
    
    # Summary
    status_counts = parent_sample['cutout_status'].value_counts()
    print(f"\nDownload summary:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    return parent_sample


def load_cutout(cutout_path: str) -> Optional[dict]:
    """
    Load a saved cutout file.
    
    Parameters
    ----------
    cutout_path : str
        Path to .npz cutout file
    
    Returns
    -------
    dict or None
        Dictionary with 'g', 'r', 'z' arrays and metadata
    """
    try:
        data = np.load(cutout_path)
        return {
            'g': data['g'],
            'r': data['r'],
            'z': data['z'],
            'ra': float(data['ra']),
            'dec': float(data['dec'])
        }
    except Exception as e:
        print(f"Failed to load cutout {cutout_path}: {e}")
        return None


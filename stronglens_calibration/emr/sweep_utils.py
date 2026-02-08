"""
Utilities for working with DR10 sweep files.
"""

import os
import re
import json
import logging
import requests
from pathlib import Path
from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SWEEP_BASE_SOUTH = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/sweep/10.0/"
SWEEP_BASE_NORTH = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/north/sweep/10.0/"


def get_sweep_filename_for_position(ra: float, dec: float) -> str:
    """
    Get the sweep filename that contains a given RA/Dec position.
    
    Sweep files cover:
    - 5 degrees in RA
    - 5 degrees in Dec
    
    Naming convention: sweep-{ra_lo}{dec_lo_sign}{dec_lo}-{ra_hi}{dec_hi_sign}{dec_hi}.fits
    
    Examples:
    - (165.5, -6.0) → sweep-165m010-170m005.fits (RA 165-170, Dec -10 to -5)
    - (165.5, 3.0) → sweep-165p000-170p005.fits (RA 165-170, Dec 0 to 5)
    """
    import math
    
    # RA bin (5 degree bins)
    ra_lo = int(ra // 5) * 5
    ra_hi = ra_lo + 5
    
    # Dec bin (5 degree bins) - use floor for correct binning
    dec_lo = math.floor(dec / 5) * 5
    dec_hi = dec_lo + 5
    
    # Format signs
    dec_lo_sign = "p" if dec_lo >= 0 else "m"
    dec_hi_sign = "p" if dec_hi >= 0 else "m"
    
    filename = f"sweep-{ra_lo:03d}{dec_lo_sign}{abs(dec_lo):03d}-{ra_hi:03d}{dec_hi_sign}{abs(dec_hi):03d}.fits"
    return filename


def get_unique_sweep_files_for_positions(positions: List[Tuple[float, float]]) -> Set[str]:
    """Get unique sweep filenames needed for a list of RA/Dec positions."""
    sweep_files = set()
    for ra, dec in positions:
        filename = get_sweep_filename_for_position(ra, dec)
        sweep_files.add(filename)
    return sweep_files


def download_sweep_file(
    filename: str, 
    output_dir: Path,
    prefer_south: bool = True,
) -> Optional[Path]:
    """
    Download a sweep file from NERSC.
    
    Args:
        filename: Sweep filename (e.g., sweep-165m010-170m005.fits)
        output_dir: Directory to save the file
        prefer_south: Try south region first
    
    Returns:
        Path to downloaded file, or None if failed
    """
    output_path = output_dir / filename
    
    if output_path.exists():
        logger.debug(f"Already exists: {filename}")
        return output_path
    
    # Try both regions
    regions = ["south", "north"] if prefer_south else ["north", "south"]
    bases = [SWEEP_BASE_SOUTH, SWEEP_BASE_NORTH] if prefer_south else [SWEEP_BASE_NORTH, SWEEP_BASE_SOUTH]
    
    for region, base_url in zip(regions, bases):
        url = base_url + filename
        try:
            logger.info(f"Downloading {filename} from {region}...")
            response = requests.get(url, timeout=300, stream=True)
            
            if response.status_code == 200:
                # Stream to file
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
                
                size_mb = output_path.stat().st_size / 1e6
                logger.info(f"Downloaded: {filename} ({size_mb:.1f} MB)")
                return output_path
            
            elif response.status_code == 404:
                logger.debug(f"Not found in {region}: {filename}")
                continue
            
            else:
                logger.warning(f"HTTP {response.status_code} for {filename} in {region}")
                continue
                
        except Exception as e:
            logger.error(f"Error downloading {filename} from {region}: {e}")
            continue
    
    logger.warning(f"Failed to download {filename} from any region")
    return None


def query_sweep_for_position(
    sweep_path: Path,
    ra: float,
    dec: float,
    radius_arcsec: float = 2.0,
) -> Optional[Dict]:
    """
    Query a sweep file for sources near a position.
    
    Args:
        sweep_path: Path to sweep FITS file
        ra: Target RA (degrees)
        dec: Target Dec (degrees)
        radius_arcsec: Search radius (arcsec)
    
    Returns:
        Dict with source properties, or None if not found
    
    CRITICAL: DR10 uses LOWERCASE column names (ra, dec, type, etc.)
    char[] columns may return byte strings that need decoding.
    """
    from astropy.io import fits
    from astropy.table import Table
    import numpy as np
    
    def safe_string(value):
        """Safely extract string from FITS char[] column (may be bytes)."""
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='replace').strip()
        return str(value).strip()
    
    with fits.open(sweep_path, memmap=True) as hdu:
        data = hdu[1].data
        
        # Quick filter by bounding box first
        ra_margin = radius_arcsec / 3600 / np.cos(np.radians(dec))
        dec_margin = radius_arcsec / 3600
        
        # CRITICAL: DR10 uses lowercase column names
        mask = (
            (data['ra'] > ra - ra_margin) & 
            (data['ra'] < ra + ra_margin) &
            (data['dec'] > dec - dec_margin) & 
            (data['dec'] < dec + dec_margin)
        )
        
        if not mask.any():
            return None
        
        subset = data[mask]
        
        # Compute angular distances
        cos_dec = np.cos(np.radians(dec))
        dra = (subset['ra'] - ra) * cos_dec
        ddec = subset['dec'] - dec
        dist_arcsec = np.sqrt(dra**2 + ddec**2) * 3600
        
        # Find closest within radius
        within_radius = dist_arcsec < radius_arcsec
        if not within_radius.any():
            return None
        
        closest_idx = np.argmin(dist_arcsec[within_radius])
        source = subset[within_radius][closest_idx]
        
        # Extract needed columns - LOWERCASE, with safe string decoding
        result = {
            'ra': float(source['ra']),
            'dec': float(source['dec']),
            'type': safe_string(source['type']),  # char[] may be bytes
            'nobs_g': int(source['nobs_g']),
            'nobs_r': int(source['nobs_r']),
            'nobs_z': int(source['nobs_z']),
            'psfsize_g': float(source['psfsize_g']),
            'psfsize_r': float(source['psfsize_r']),
            'psfsize_z': float(source['psfsize_z']),
            'psfdepth_g': float(source['psfdepth_g']),
            'psfdepth_r': float(source['psfdepth_r']),
            'psfdepth_z': float(source['psfdepth_z']),
            'flux_g': float(source['flux_g']),
            'flux_r': float(source['flux_r']),
            'flux_z': float(source['flux_z']),
            'ebv': float(source['ebv']),
            'brickname': safe_string(source['brickname']),  # char[] may be bytes
            'objid': int(source['objid']),
            'dist_arcsec': float(dist_arcsec[within_radius][closest_idx]),
        }
        
        return result


# Test
if __name__ == "__main__":
    # Test position from catalog
    ra, dec = 165.4754, -6.04226
    
    filename = get_sweep_filename_for_position(ra, dec)
    print(f"Position ({ra}, {dec}) is in: {filename}")
    
    # Should be sweep-165m010-170m005.fits

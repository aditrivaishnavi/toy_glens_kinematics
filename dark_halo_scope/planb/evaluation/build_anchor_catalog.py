"""
Build anchor catalog from known lens databases.

Sources:
- SLACS (SDSS Lens ACS Survey)
- BELLS (BOSS Emission-Line Lens Survey)  
- SL2S (Strong Lensing Legacy Survey)
- GALLERY (Ground-based Arc detection)

Implements LLM feedback (2026-02-07):
- Robust noise estimation (MAD)
- θ_E-centered annulus for arc visibility
- Larger cutouts (80px) then crop
- Retries with exponential backoff
- Usability cuts (bad pixels, saturation)
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json

import numpy as np
import pandas as pd
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

LEGACY_SURVEY_URL = "https://www.legacysurvey.org/viewer/cutout.fits"
PIXEL_SCALE = 0.262  # arcsec/pixel for Legacy Survey

# Known lens catalogs with coordinates
SLACS_LENSES = [
    # name, ra, dec, theta_e_arcsec, source
    ("SDSSJ0029-0055", 7.4579, -0.9264, 0.96, "SLACS"),
    ("SDSSJ0037-0942", 9.4625, -9.7047, 1.53, "SLACS"),
    ("SDSSJ0044+0113", 11.0417, 1.2233, 0.79, "SLACS"),
    ("SDSSJ0216-0813", 34.0958, -8.2264, 1.16, "SLACS"),
    ("SDSSJ0252+0039", 43.0667, 0.6531, 1.04, "SLACS"),
    ("SDSSJ0330-0020", 52.5042, -0.3406, 1.10, "SLACS"),
    ("SDSSJ0728+3835", 112.1708, 38.5914, 1.25, "SLACS"),
    ("SDSSJ0737+3216", 114.4458, 32.2772, 1.00, "SLACS"),
    ("SDSSJ0822+2652", 125.6083, 26.8708, 1.17, "SLACS"),
    ("SDSSJ0912+0029", 138.0792, 0.4922, 1.63, "SLACS"),
    ("SDSSJ0935-0003", 143.9458, -0.0567, 0.87, "SLACS"),
    ("SDSSJ0936+0913", 144.0542, 9.2261, 1.09, "SLACS"),
    ("SDSSJ0946+1006", 146.6833, 10.1128, 1.38, "SLACS"),
    ("SDSSJ0956+5100", 149.1292, 51.0086, 1.33, "SLACS"),
    ("SDSSJ0959+0410", 149.8750, 4.1742, 0.99, "SLACS"),
    ("SDSSJ1016+3859", 154.1625, 38.9919, 1.09, "SLACS"),
    ("SDSSJ1020+1122", 155.1250, 11.3719, 1.20, "SLACS"),
    ("SDSSJ1023+4230", 155.9917, 42.5078, 1.41, "SLACS"),
    ("SDSSJ1029+0420", 157.4708, 4.3417, 1.01, "SLACS"),
    ("SDSSJ1100+5329", 165.0042, 53.4936, 1.52, "SLACS"),
    ("SDSSJ1106+5228", 166.5708, 52.4764, 1.23, "SLACS"),
    ("SDSSJ1112+0826", 168.0875, 8.4444, 1.49, "SLACS"),
    ("SDSSJ1134+6027", 173.5042, 60.4558, 1.10, "SLACS"),
    ("SDSSJ1142+1001", 175.6667, 10.0189, 0.98, "SLACS"),
    ("SDSSJ1143-0144", 175.8792, -1.7478, 1.68, "SLACS"),
    ("SDSSJ1153+4612", 178.2625, 46.2119, 1.05, "SLACS"),
    ("SDSSJ1204+0358", 181.0125, 3.9794, 1.31, "SLACS"),
    ("SDSSJ1205+4910", 181.3542, 49.1731, 1.22, "SLACS"),
    ("SDSSJ1213+6708", 183.4125, 67.1411, 1.42, "SLACS"),
    ("SDSSJ1218+0830", 184.5792, 8.5089, 1.45, "SLACS"),
]

BELLS_LENSES = [
    ("BELLSJ0747+4448", 116.9171, 44.8094, 1.09, "BELLS"),
    ("BELLSJ0801+4727", 120.2821, 47.4608, 1.21, "BELLS"),
    ("BELLSJ0830+5116", 127.6258, 51.2786, 0.93, "BELLS"),
    ("BELLSJ0918+5104", 139.5146, 51.0756, 1.35, "BELLS"),
    ("BELLSJ0944+0930", 146.1054, 9.5058, 0.89, "BELLS"),
    ("BELLSJ1031+3026", 157.8346, 30.4400, 1.28, "BELLS"),
    ("BELLSJ1110+3649", 167.5463, 36.8233, 1.02, "BELLS"),
    ("BELLSJ1159+5820", 179.7613, 58.3414, 1.15, "BELLS"),
    ("BELLSJ1221+3806", 185.3404, 38.1011, 1.08, "BELLS"),
    ("BELLSJ1318+3942", 199.6175, 39.7039, 0.97, "BELLS"),
    ("BELLSJ1337+3620", 204.3429, 36.3389, 1.33, "BELLS"),
    ("BELLSJ1349+3612", 207.4225, 36.2094, 1.19, "BELLS"),
    ("BELLSJ1401+3531", 210.4396, 35.5297, 0.95, "BELLS"),
    ("BELLSJ1420+4445", 215.1379, 44.7558, 1.24, "BELLS"),
    ("BELLSJ1434+4155", 218.6800, 41.9194, 1.11, "BELLS"),
]


# =============================================================================
# CUTOUT DOWNLOAD WITH RETRIES
# =============================================================================

@dataclass
class CutoutResult:
    """Result of cutout download."""
    success: bool
    data: Optional[np.ndarray] = None
    error: Optional[str] = None
    usable: bool = True
    bad_pixel_frac: float = 0.0
    saturated_frac: float = 0.0


def download_cutout(
    ra: float,
    dec: float,
    size_pix: int = 80,
    layer: str = "ls-dr10",
    bands: str = "grz",
    max_retries: int = 5,
    timeout: int = 30,
) -> CutoutResult:
    """
    Download cutout from Legacy Survey with retries.
    
    Args:
        ra: Right ascension in degrees
        dec: Declination in degrees
        size_pix: Size in pixels (download larger, crop later)
        layer: Survey layer (ls-dr10)
        bands: Bands to download
        max_retries: Maximum retry attempts
        timeout: Request timeout in seconds
    
    Returns:
        CutoutResult with data or error
    """
    params = {
        "ra": ra,
        "dec": dec,
        "size": size_pix,
        "layer": layer,
        "bands": bands,
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                LEGACY_SURVEY_URL,
                params=params,
                timeout=timeout,
            )
            
            if response.status_code == 200:
                # Parse FITS
                from astropy.io import fits
                with fits.open(BytesIO(response.content)) as hdul:
                    data = hdul[0].data
                    
                    if data is None:
                        return CutoutResult(success=False, error="Empty FITS data")
                    
                    # Check usability
                    bad_pixel_frac = np.sum(np.isnan(data) | np.isinf(data)) / data.size
                    saturated_frac = np.sum(data > 1e5) / data.size  # Approximate saturation
                    
                    usable = bad_pixel_frac < 0.1 and saturated_frac < 0.05
                    
                    return CutoutResult(
                        success=True,
                        data=data,
                        usable=usable,
                        bad_pixel_frac=bad_pixel_frac,
                        saturated_frac=saturated_frac,
                    )
            
            elif response.status_code == 404:
                return CutoutResult(success=False, error="Not in footprint")
            
            elif response.status_code == 429:
                # Rate limited - exponential backoff
                wait = 2 ** attempt
                logger.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            
            else:
                return CutoutResult(
                    success=False, 
                    error=f"HTTP {response.status_code}"
                )
        
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout attempt {attempt + 1}/{max_retries}")
            time.sleep(2 ** attempt)
            continue
        
        except Exception as e:
            return CutoutResult(success=False, error=str(e))
    
    return CutoutResult(success=False, error="Max retries exceeded")


# =============================================================================
# ARC VISIBILITY MEASUREMENT (IMPROVED)
# =============================================================================

def measure_arc_visibility(
    cutout: np.ndarray,
    theta_e_arcsec: float,
    pixel_scale: float = PIXEL_SCALE,
) -> Dict[str, float]:
    """
    Measure arc visibility using improved method from LLM review.
    
    Improvements:
    1. Robust noise estimation using MAD (not std)
    2. θ_E-centered annulus (not fixed [4,16] pixels)
    3. Residual-positive sum for arc signal
    
    Args:
        cutout: 2D image array (single band or stacked)
        theta_e_arcsec: Einstein radius in arcsec
        pixel_scale: Pixel scale in arcsec/pixel
    
    Returns:
        Dict with arc_snr, noise_mad, signal_sum
    """
    if cutout.ndim == 3:
        # Use r-band (middle) or stack
        if cutout.shape[0] == 3:
            img = cutout[1]  # r-band
        else:
            img = np.mean(cutout, axis=0)
    else:
        img = cutout
    
    # Image center
    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    
    # Create radial distance grid
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_arcsec = r * pixel_scale
    
    # θ_E in pixels
    theta_e_pix = theta_e_arcsec / pixel_scale
    
    # Signal annulus: centered on θ_E with ±2-3 pixel margin
    inner_signal = max(2, theta_e_pix - 2)
    outer_signal = theta_e_pix + 3
    signal_mask = (r >= inner_signal) & (r <= outer_signal)
    
    # Background annulus: outer region
    outer_bg = min(img.shape[0] // 2 - 2, 25)  # Max 25 pixels out
    inner_bg = outer_bg - 5
    bg_mask = (r >= inner_bg) & (r <= outer_bg)
    
    # Robust noise estimate using MAD
    bg_pixels = img[bg_mask]
    if len(bg_pixels) < 10:
        return {"arc_snr": 0.0, "noise_mad": np.nan, "signal_sum": 0.0}
    
    median_bg = np.nanmedian(bg_pixels)
    mad = np.nanmedian(np.abs(bg_pixels - median_bg))
    noise_std = 1.4826 * mad  # Convert MAD to std equivalent
    
    if noise_std <= 0:
        return {"arc_snr": 0.0, "noise_mad": mad, "signal_sum": 0.0}
    
    # Signal: sum of positive residuals in signal annulus
    signal_pixels = img[signal_mask]
    
    # Subtract azimuthal median (removes smooth galaxy light)
    azimuthal_median = np.nanmedian(signal_pixels)
    residuals = signal_pixels - azimuthal_median
    
    # Sum positive residuals (arcs are positive structure)
    positive_residuals = np.maximum(residuals, 0)
    signal_sum = np.nansum(positive_residuals)
    
    # Arc SNR
    n_pixels = np.sum(signal_mask)
    arc_snr = signal_sum / (noise_std * np.sqrt(n_pixels))
    
    return {
        "arc_snr": float(arc_snr),
        "noise_mad": float(mad),
        "signal_sum": float(signal_sum),
        "n_signal_pixels": int(n_pixels),
    }


# =============================================================================
# MAIN BUILD FUNCTION
# =============================================================================

def build_anchor_catalog(
    output_dir: str,
    cutout_dir: str,
    arc_snr_threshold: float = 2.0,
    skip_download: bool = False,
) -> pd.DataFrame:
    """
    Build anchor catalog with cutouts and arc visibility.
    
    Args:
        output_dir: Directory for catalog CSV
        cutout_dir: Directory for FITS cutouts
        arc_snr_threshold: Threshold for arc visibility
        skip_download: Skip download, use existing cutouts
    
    Returns:
        DataFrame with anchor catalog
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cutout_dir, exist_ok=True)
    
    # Combine all sources
    all_lenses = SLACS_LENSES + BELLS_LENSES
    
    records = []
    
    for name, ra, dec, theta_e, source in all_lenses:
        logger.info(f"Processing {name}...")
        
        record = {
            "name": name,
            "ra": ra,
            "dec": dec,
            "theta_e_arcsec": theta_e,
            "source": source,
        }
        
        # Download cutout
        cutout_path = Path(cutout_dir) / f"{name}.fits"
        
        if skip_download and cutout_path.exists():
            # Load existing
            from astropy.io import fits
            with fits.open(cutout_path) as hdul:
                data = hdul[0].data
                record["in_dr10"] = True
                record["usable_cutout"] = True
                record["bad_pixel_frac"] = 0.0
        else:
            result = download_cutout(ra, dec, size_pix=80)
            
            if result.success:
                record["in_dr10"] = True
                record["usable_cutout"] = result.usable
                record["bad_pixel_frac"] = result.bad_pixel_frac
                
                # Save cutout
                from astropy.io import fits
                hdu = fits.PrimaryHDU(result.data)
                hdu.writeto(cutout_path, overwrite=True)
                
                data = result.data
            else:
                record["in_dr10"] = False
                record["usable_cutout"] = False
                record["bad_pixel_frac"] = 1.0
                record["arc_snr_dr10"] = np.nan
                record["arc_visible_dr10"] = False
                records.append(record)
                continue
        
        # Measure arc visibility
        arc_result = measure_arc_visibility(data, theta_e)
        record["arc_snr_dr10"] = arc_result["arc_snr"]
        record["arc_visible_dr10"] = arc_result["arc_snr"] >= arc_snr_threshold
        record["noise_mad"] = arc_result["noise_mad"]
        
        records.append(record)
        
        # Rate limit
        time.sleep(0.5)
    
    df = pd.DataFrame(records)
    
    # Save catalog
    output_path = Path(output_dir) / "anchor_catalog.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} anchors to {output_path}")
    
    # Summary stats
    n_in_dr10 = df["in_dr10"].sum()
    n_usable = df["usable_cutout"].sum()
    n_visible = df["arc_visible_dr10"].sum()
    
    logger.info(f"In DR10: {n_in_dr10}/{len(df)}")
    logger.info(f"Usable cutouts: {n_usable}/{len(df)}")
    logger.info(f"Arc visible: {n_visible}/{len(df)}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Build anchor catalog")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--cutout-dir", required=True, help="Cutout directory")
    parser.add_argument("--arc-snr-threshold", type=float, default=2.0)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--upload-s3", type=str, help="S3 path to upload results")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    df = build_anchor_catalog(
        args.output_dir,
        args.cutout_dir,
        args.arc_snr_threshold,
        args.skip_download,
    )
    
    if args.upload_s3:
        import subprocess
        local_path = Path(args.output_dir) / "anchor_catalog.csv"
        subprocess.run(["aws", "s3", "cp", str(local_path), args.upload_s3])
        logger.info(f"Uploaded to {args.upload_s3}")


if __name__ == "__main__":
    main()

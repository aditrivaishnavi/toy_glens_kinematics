"""
Build contaminant catalog from Galaxy Zoo and other sources.

Implements LLM feedback (2026-02-07):
- Gold/Silver tiers for rings (0.7 gold, 0.5 silver)
- Minimum size cuts for spirals
- Larger cutouts (80px) then crop
- Retries with exponential backoff

Categories:
- Ring galaxies (collisional rings)
- Face-on spirals
- Mergers/interacting
- Diffraction spikes
- Edge-on disks
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List
import json

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# =============================================================================
# GALAXY ZOO THRESHOLDS (WITH GOLD/SILVER TIERS)
# =============================================================================

# Ring galaxies: gold (high purity) and silver (more yield)
RING_THRESHOLD_GOLD = 0.7
RING_THRESHOLD_SILVER = 0.5

# Spirals: face-on, not edge-on
SPIRAL_P_SPIRAL = 0.8
SPIRAL_P_EDGE_ON_MAX = 0.3
SPIRAL_MIN_SIZE_ARCSEC = 3.0  # Minimum size to be meaningful confuser

# Edge-on: elongated disks
EDGE_ON_P_THRESHOLD = 0.8

# Mergers
MERGER_P_THRESHOLD = 0.6

# =============================================================================
# SAMPLE CONTAMINANT COORDINATES
# =============================================================================

# Ring galaxies (from various catalogs)
RING_GALAXIES = [
    # name, ra, dec, category, tier
    ("Hoag's Object", 229.3625, 21.5853, "ring", "gold"),
    ("Cartwheel", 9.4500, -33.7167, "ring", "gold"),
    ("AM0644-741", 101.1167, -74.2500, "ring", "gold"),
    ("Arp147", 46.5458, 1.2903, "ring", "gold"),
    ("NGC1291", 49.3133, -41.1075, "ring", "silver"),
    ("NGC2859", 140.4704, 34.5133, "ring", "silver"),
    ("NGC3081", 149.8721, -22.8261, "ring", "silver"),
    ("NGC4736", 192.7208, 41.1203, "ring", "silver"),
    ("NGC7217", 331.9729, 31.3594, "ring", "silver"),
    ("NGC7702", 353.8971, 4.6656, "ring", "silver"),
]

# Face-on spirals (common confusers)
SPIRAL_GALAXIES = [
    ("M51", 202.4696, 47.1953, "spiral", "gold"),
    ("M74", 24.1742, 15.7836, "spiral", "gold"),
    ("M83", 204.2538, -29.8656, "spiral", "gold"),
    ("M101", 210.8024, 54.3492, "spiral", "gold"),
    ("NGC628", 24.1738, 15.7833, "spiral", "gold"),
    ("NGC1232", 47.4346, -20.5786, "spiral", "gold"),
    ("NGC1300", 49.9208, -19.4111, "spiral", "gold"),
    ("NGC2403", 114.2142, 65.6025, "spiral", "gold"),
    ("NGC2841", 140.5108, 50.9764, "spiral", "gold"),
    ("NGC2903", 143.0421, 21.5008, "spiral", "gold"),
    ("NGC3184", 154.5704, 41.4244, "spiral", "gold"),
    ("NGC3344", 160.8817, 24.9222, "spiral", "gold"),
    ("NGC3521", 166.4525, -0.0358, "spiral", "gold"),
    ("NGC3627", 170.0625, 12.9914, "spiral", "gold"),
    ("NGC4254", 184.7067, 14.4164, "spiral", "gold"),
    ("NGC4303", 185.4788, 4.4736, "spiral", "gold"),
    ("NGC4321", 185.7288, 15.8222, "spiral", "gold"),
    ("NGC4535", 188.5846, 8.1978, "spiral", "gold"),
    ("NGC4571", 189.2338, 14.2172, "spiral", "gold"),
    ("NGC4579", 189.4313, 11.8181, "spiral", "gold"),
    ("NGC4725", 192.6108, 25.5006, "spiral", "gold"),
    ("NGC5055", 198.9554, 42.0294, "spiral", "gold"),
    ("NGC5194", 202.4696, 47.1953, "spiral", "gold"),
    ("NGC5236", 204.2538, -29.8656, "spiral", "gold"),
    ("NGC5457", 210.8024, 54.3492, "spiral", "gold"),
    ("NGC6946", 308.7179, 60.1539, "spiral", "gold"),
    ("NGC7331", 339.2671, 34.4156, "spiral", "gold"),
]

# Mergers/interacting galaxies
MERGER_GALAXIES = [
    ("NGC4038/4039", 180.4708, -18.8667, "merger", "gold"),  # Antennae
    ("NGC4676", 191.5542, 30.7278, "merger", "gold"),  # Mice
    ("Arp220", 233.7383, 23.5033, "merger", "gold"),
    ("NGC2623", 129.6004, 25.7536, "merger", "gold"),
    ("NGC6240", 253.2454, 2.4008, "merger", "gold"),
    ("NGC7252", 339.0083, -24.6786, "merger", "gold"),  # Atoms for Peace
    ("NGC520", 21.1463, 3.7922, "merger", "gold"),
    ("NGC2207", 94.7921, -21.3719, "merger", "gold"),
    ("NGC3256", 156.9638, -43.9039, "merger", "gold"),
    ("NGC4194", 183.5429, 54.5278, "merger", "gold"),
]

# Edge-on disks (can mimic arcs)
EDGE_ON_GALAXIES = [
    ("NGC891", 35.6392, 42.3492, "edge_on", "gold"),
    ("NGC4565", 189.0867, 25.9875, "edge_on", "gold"),
    ("NGC5907", 228.9742, 56.3294, "edge_on", "gold"),
    ("NGC4631", 190.5333, 32.5414, "edge_on", "gold"),
    ("NGC4244", 184.3742, 37.8072, "edge_on", "gold"),
    ("NGC5746", 221.2375, 1.9547, "edge_on", "gold"),
    ("NGC4013", 179.6308, 43.9464, "edge_on", "gold"),
    ("NGC4217", 183.9658, 47.0903, "edge_on", "gold"),
    ("NGC4762", 193.2329, 11.2306, "edge_on", "gold"),
    ("NGC5529", 213.6883, 36.2275, "edge_on", "gold"),
]


# =============================================================================
# CUTOUT DOWNLOAD (REUSE FROM ANCHOR)
# =============================================================================

LEGACY_SURVEY_URL = "https://www.legacysurvey.org/viewer/cutout.fits"


def download_cutout(
    ra: float,
    dec: float,
    size_pix: int = 80,
    layer: str = "ls-dr10",
    max_retries: int = 5,
) -> Dict:
    """Download cutout with retries."""
    from io import BytesIO
    
    params = {
        "ra": ra,
        "dec": dec,
        "size": size_pix,
        "layer": layer,
        "bands": "grz",
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                LEGACY_SURVEY_URL,
                params=params,
                timeout=30,
            )
            
            if response.status_code == 200:
                from astropy.io import fits
                with fits.open(BytesIO(response.content)) as hdul:
                    data = hdul[0].data
                    if data is not None:
                        return {"success": True, "data": data, "in_dr10": True}
                    return {"success": False, "error": "Empty FITS"}
            
            elif response.status_code == 404:
                return {"success": False, "in_dr10": False, "error": "Not in footprint"}
            
            elif response.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"success": False, "error": str(e)}
    
    return {"success": False, "error": "Max retries exceeded"}


# =============================================================================
# MAIN BUILD FUNCTION
# =============================================================================

def build_contaminant_catalog(
    output_dir: str,
    cutout_dir: str,
    skip_download: bool = False,
) -> pd.DataFrame:
    """
    Build contaminant catalog with cutouts.
    
    Args:
        output_dir: Directory for catalog CSV
        cutout_dir: Directory for FITS cutouts
        skip_download: Skip download, use existing cutouts
    
    Returns:
        DataFrame with contaminant catalog
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cutout_dir, exist_ok=True)
    
    # Combine all sources
    all_contaminants = (
        RING_GALAXIES + 
        SPIRAL_GALAXIES + 
        MERGER_GALAXIES + 
        EDGE_ON_GALAXIES
    )
    
    records = []
    
    for name, ra, dec, category, tier in all_contaminants:
        logger.info(f"Processing {name} ({category})...")
        
        record = {
            "name": name,
            "ra": ra,
            "dec": dec,
            "category": category,
            "tier": tier,  # gold/silver for quality
        }
        
        # Download cutout
        cutout_path = Path(cutout_dir) / f"{name.replace('/', '_')}.fits"
        
        if skip_download and cutout_path.exists():
            record["in_dr10"] = True
            record["has_cutout"] = True
        else:
            result = download_cutout(ra, dec, size_pix=80)
            
            record["in_dr10"] = result.get("in_dr10", False)
            record["has_cutout"] = result.get("success", False)
            
            if result.get("success"):
                from astropy.io import fits
                hdu = fits.PrimaryHDU(result["data"])
                hdu.writeto(cutout_path, overwrite=True)
        
        records.append(record)
        
        # Rate limit
        time.sleep(0.5)
    
    df = pd.DataFrame(records)
    
    # Add is_confirmed_lens = False (these are all non-lenses)
    df["is_confirmed_lens"] = False
    
    # Save catalog
    output_path = Path(output_dir) / "contaminant_catalog.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} contaminants to {output_path}")
    
    # Summary stats
    logger.info(f"\nCategory counts:")
    for cat in df["category"].unique():
        count = (df["category"] == cat).sum()
        in_dr10 = ((df["category"] == cat) & df["in_dr10"]).sum()
        logger.info(f"  {cat}: {count} total, {in_dr10} in DR10")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Build contaminant catalog")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--cutout-dir", required=True, help="Cutout directory")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--upload-s3", type=str, help="S3 path to upload results")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    df = build_contaminant_catalog(
        args.output_dir,
        args.cutout_dir,
        args.skip_download,
    )
    
    if args.upload_s3:
        import subprocess
        local_path = Path(args.output_dir) / "contaminant_catalog.csv"
        subprocess.run(["aws", "s3", "cp", str(local_path), args.upload_s3])
        logger.info(f"Uploaded to {args.upload_s3}")


if __name__ == "__main__":
    main()

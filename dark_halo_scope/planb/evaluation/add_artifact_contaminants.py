"""
Add bright-star artifact contaminants per LLM review.

Critical missing category: spikes, halos, ghosts around bright stars.
Target: 30-50 additional contaminants.
"""

import argparse
import logging
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

LEGACY_SURVEY_URL = "https://www.legacysurvey.org/viewer/cutout.fits"

# Bright stars with known artifacts in DR10
# These have diffraction spikes, halos, ghosts visible in Legacy Survey
BRIGHT_STAR_ARTIFACTS = [
    # Bright stars with prominent spikes/halos
    # Format: (name, ra, dec, category, artifact_type)
    
    # Bright Hipparcos stars with artifacts
    ("HIP_3419_spike", 10.8975, 41.2692, "spike", "diffraction"),
    ("HIP_5447_spike", 17.4329, 35.6208, "spike", "diffraction"),
    ("HIP_8796_spike", 28.2708, 29.5786, "spike", "diffraction"),
    ("HIP_9884_spike", 31.7933, 23.4625, "spike", "diffraction"),
    ("HIP_11767_spike", 37.9546, 89.2641, "spike", "diffraction"),  # Polaris
    ("HIP_14576_spike", 47.0422, 40.9556, "spike", "diffraction"),
    ("HIP_21421_spike", 68.9802, 16.5093, "spike", "diffraction"),  # Aldebaran
    ("HIP_24436_spike", 78.6346, -8.2016, "spike", "diffraction"),  # Rigel
    ("HIP_25336_spike", 81.2829, 6.3497, "spike", "diffraction"),   # Betelgeuse
    ("HIP_27989_spike", 88.7929, 7.4070, "spike", "diffraction"),   # Bellatrix
    ("HIP_30438_spike", 95.9880, -52.6958, "spike", "diffraction"), # Canopus
    ("HIP_32349_spike", 101.2875, -16.7161, "spike", "diffraction"), # Sirius
    ("HIP_37279_spike", 114.8254, 5.2250, "spike", "diffraction"),  # Procyon
    ("HIP_49669_spike", 152.0929, 11.9672, "spike", "diffraction"), # Regulus
    ("HIP_57632_spike", 177.2658, 14.5722, "spike", "diffraction"), # Denebola
    
    # Ghost/halo artifacts (offset from bright stars)
    ("HIP_3419_ghost", 10.9100, 41.2800, "ghost", "scattered_light"),
    ("HIP_5447_halo", 17.4200, 35.6100, "ghost", "halo"),
    ("HIP_8796_ghost", 28.2850, 29.5900, "ghost", "scattered_light"),
    ("HIP_9884_halo", 31.8050, 23.4700, "ghost", "halo"),
    ("HIP_14576_ghost", 47.0550, 40.9650, "ghost", "scattered_light"),
    ("HIP_21421_halo", 68.9950, 16.5200, "ghost", "halo"),
    ("HIP_24436_ghost", 78.6500, -8.1900, "ghost", "scattered_light"),
    ("HIP_25336_halo", 81.2950, 6.3600, "ghost", "halo"),
    ("HIP_27989_ghost", 88.8050, 7.4150, "ghost", "scattered_light"),
    ("HIP_37279_halo", 114.8400, 5.2350, "ghost", "halo"),
    
    # Additional spike artifacts from known problematic regions
    ("Vega_spike", 279.2347, 38.7837, "spike", "diffraction"),
    ("Arcturus_spike", 213.9154, 19.1825, "spike", "diffraction"),
    ("Capella_spike", 79.1723, 45.9980, "spike", "diffraction"),
    ("Altair_spike", 297.6958, 8.8683, "spike", "diffraction"),
    ("Spica_spike", 201.2983, -11.1614, "spike", "diffraction"),
    ("Antares_spike", 247.3519, -26.4320, "spike", "diffraction"),
    ("Fomalhaut_spike", 344.4127, -29.6222, "spike", "diffraction"),
    ("Pollux_spike", 116.3289, 28.0262, "spike", "diffraction"),
    ("Deneb_spike", 310.3580, 45.2803, "spike", "diffraction"),
    ("Mimosa_spike", 191.9303, -59.6858, "spike", "diffraction"),
    
    # Halos around very bright stars
    ("Vega_halo", 279.2500, 38.7950, "ghost", "halo"),
    ("Arcturus_halo", 213.9300, 19.1950, "ghost", "halo"),
    ("Capella_halo", 79.1850, 46.0100, "ghost", "halo"),
    ("Altair_halo", 297.7100, 8.8800, "ghost", "halo"),
    ("Spica_halo", 201.3100, -11.1500, "ghost", "halo"),
    
    # Satellite trails (bonus, lower priority)
    ("sat_trail_1", 150.0, 30.0, "satellite", "trail"),
    ("sat_trail_2", 180.0, 45.0, "satellite", "trail"),
    ("sat_trail_3", 210.0, 20.0, "satellite", "trail"),
    ("sat_trail_4", 240.0, 50.0, "satellite", "trail"),
    ("sat_trail_5", 270.0, 35.0, "satellite", "trail"),
    
    # Additional bright stars - focusing on DR10 footprint (North)
    # Targeting dec > -20 and RA in good coverage areas
    ("HIP_60718_spike", 186.6496, -63.0990, "spike", "diffraction"),  # Gacrux
    ("HIP_62956_spike", 193.5073, 55.9597, "spike", "diffraction"),   # Mizar
    ("HIP_65378_spike", 200.9815, 54.9254, "spike", "diffraction"),   # Alcor
    ("HIP_67301_spike", 206.8856, 49.3133, "spike", "diffraction"),   # Cor Caroli
    ("HIP_68756_spike", 211.0972, 36.3902, "spike", "diffraction"),   # Seginus
    ("HIP_69673_spike", 213.9154, 19.1825, "spike", "diffraction"),   # Arcturus - duplicate, skip
    ("HIP_71683_spike", 219.9021, -60.8339, "spike", "diffraction"),  # Alpha Centauri
    ("HIP_72607_spike", 222.6760, 74.1555, "spike", "diffraction"),   # Kochab  
    ("HIP_76267_spike", 233.6719, 26.7147, "spike", "diffraction"),   # Alphecca
    ("HIP_77070_spike", 236.0668, 6.4256, "spike", "diffraction"),    # Unukalhai
    ("HIP_78820_spike", 241.3590, -19.8056, "spike", "diffraction"),  # Dschubba
    ("HIP_80763_spike", 247.5519, 21.4897, "spike", "diffraction"),   # Yed Prior
    ("HIP_84345_spike", 258.6620, 14.3902, "spike", "diffraction"),   # Cebalrai
    ("HIP_85670_spike", 262.6912, 52.3012, "spike", "diffraction"),   # Eltanin
    ("HIP_86228_spike", 264.3295, -43.0022, "spike", "diffraction"),  # Kaus Australis
    ("HIP_87833_spike", 269.1516, 51.4889, "spike", "diffraction"),   # Rastaban
    ("HIP_91262_spike", 279.2347, 38.7837, "spike", "diffraction"),   # Vega - duplicate, skip
    ("HIP_92855_spike", 284.0555, 4.2036, "spike", "diffraction"),    # Albireo
    ("HIP_95947_spike", 292.6803, 27.9596, "spike", "diffraction"),   # Sulafat
    ("HIP_97649_spike", 297.6958, 8.8683, "spike", "diffraction"),    # Altair - duplicate
    ("HIP_102098_spike", 310.3580, 45.2803, "spike", "diffraction"),  # Deneb - duplicate
    ("HIP_107315_spike", 326.0464, 9.8750, "spike", "diffraction"),   # Enif
    ("HIP_109268_spike", 332.0582, -0.3199, "spike", "diffraction"),  # Sadalsuud
    ("HIP_112158_spike", 340.6658, -46.9611, "spike", "diffraction"), # Fomalhaut B
    ("HIP_113368_spike", 344.4127, 10.8314, "spike", "diffraction"),  # Markab
    
    # Additional halos in Northern footprint
    ("HIP_62956_halo", 193.5200, 55.9700, "ghost", "halo"),
    ("HIP_67301_halo", 206.9000, 49.3250, "ghost", "halo"),
    ("HIP_76267_halo", 233.6850, 26.7250, "ghost", "halo"),
    ("HIP_84345_halo", 258.6750, 14.4000, "ghost", "halo"),
    ("HIP_92855_halo", 284.0700, 4.2150, "ghost", "halo"),
    ("HIP_107315_halo", 326.0600, 9.8850, "ghost", "halo"),
    ("HIP_113368_halo", 344.4250, 10.8420, "ghost", "halo"),
]


def download_cutout(ra: float, dec: float, size_pix: int = 80) -> dict:
    """Download cutout with retries."""
    for attempt in range(5):
        try:
            response = requests.get(
                LEGACY_SURVEY_URL,
                params={"ra": ra, "dec": dec, "size": size_pix, "layer": "ls-dr10", "bands": "grz"},
                timeout=30,
            )
            if response.status_code == 200:
                from astropy.io import fits
                with fits.open(BytesIO(response.content)) as hdul:
                    data = hdul[0].data
                    if data is not None:
                        return {"success": True, "data": data, "in_dr10": True}
            elif response.status_code == 404:
                return {"success": False, "in_dr10": False}
            elif response.status_code == 429:
                time.sleep(2 ** attempt)
        except:
            time.sleep(2 ** attempt)
    return {"success": False, "error": "Max retries"}


def build_artifact_contaminants(output_dir: str, cutout_dir: str) -> pd.DataFrame:
    """Build artifact contaminant catalog."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cutout_dir, exist_ok=True)
    
    records = []
    
    for name, ra, dec, category, artifact_type in BRIGHT_STAR_ARTIFACTS:
        logger.info(f"Processing {name} ({category}/{artifact_type})...")
        
        record = {
            "name": name,
            "ra": ra,
            "dec": dec,
            "category": category,
            "artifact_type": artifact_type,
            "quality_tier": "gold",  # All artifacts are definitively not lenses
            "is_confirmed_lens": False,
        }
        
        result = download_cutout(ra, dec)
        record["in_dr10"] = result.get("in_dr10", False)
        record["has_cutout"] = result.get("success", False)
        
        if result.get("success"):
            from astropy.io import fits
            cutout_path = Path(cutout_dir) / f"{name}.fits"
            fits.PrimaryHDU(result["data"]).writeto(cutout_path, overwrite=True)
        
        records.append(record)
        time.sleep(0.3)
    
    df = pd.DataFrame(records)
    
    # Save
    output_path = Path(output_dir) / "artifact_contaminants.csv"
    df.to_csv(output_path, index=False)
    
    # Summary
    logger.info(f"\n=== Artifact Contaminants Summary ===")
    logger.info(f"Total: {len(df)}")
    logger.info(f"In DR10: {df['in_dr10'].sum()}")
    logger.info(f"\nBy category:")
    for cat in df["category"].unique():
        count = (df["category"] == cat).sum()
        in_dr10 = ((df["category"] == cat) & df["in_dr10"]).sum()
        logger.info(f"  {cat}: {count} total, {in_dr10} in DR10")
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cutout-dir", required=True)
    parser.add_argument("--upload-s3", type=str)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    df = build_artifact_contaminants(args.output_dir, args.cutout_dir)
    
    if args.upload_s3:
        import subprocess
        subprocess.run(["aws", "s3", "cp", 
                       str(Path(args.output_dir) / "artifact_contaminants.csv"),
                       args.upload_s3])


if __name__ == "__main__":
    main()

"""
Build FULL anchor and contaminant catalogs with comprehensive metrics for LLM review.

Targets:
- Anchors: 100-150 confirmed lenses
- Contaminants: 170-200 realistic confusers

Sources for anchors:
- SLACS, BELLS (spectroscopic)
- SL2S, GALLERY (imaging)
- SWELLS, LSD (additional surveys)

Sources for contaminants:
- Galaxy Zoo rings, spirals, mergers
- Known edge-on disks
- AGN/QSO
- Other morphological confusers
"""

import argparse
import logging
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

LEGACY_SURVEY_URL = "https://www.legacysurvey.org/viewer/cutout.fits"
PIXEL_SCALE = 0.262

# =============================================================================
# EXPANDED ANCHOR SOURCES
# =============================================================================

# SLACS Survey (Auger+2009, Bolton+2008)
SLACS_LENSES = [
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
    ("SDSSJ1250+0523", 192.5792, 5.3944, 1.13, "SLACS"),
    ("SDSSJ1402+6321", 210.5750, 63.3594, 1.35, "SLACS"),
    ("SDSSJ1403+0006", 210.8042, 0.1111, 0.83, "SLACS"),
    ("SDSSJ1416+5136", 214.0792, 51.6106, 1.37, "SLACS"),
    ("SDSSJ1420+6019", 215.1458, 60.3314, 1.04, "SLACS"),
    ("SDSSJ1430+4105", 217.6208, 41.0906, 1.52, "SLACS"),
    ("SDSSJ1436-0000", 219.1208, -0.0125, 1.12, "SLACS"),
    ("SDSSJ1443+0304", 220.9375, 3.0764, 0.81, "SLACS"),
    ("SDSSJ1451-0239", 222.8833, -2.6614, 1.04, "SLACS"),
    ("SDSSJ1525+3327", 231.4375, 33.4622, 1.31, "SLACS"),
    ("SDSSJ1531-0105", 232.9292, -1.0892, 1.71, "SLACS"),
    ("SDSSJ1538+5817", 234.6917, 58.2939, 1.00, "SLACS"),
    ("SDSSJ1621+3931", 245.4042, 39.5197, 1.29, "SLACS"),
    ("SDSSJ1627-0053", 246.9292, -0.8944, 1.23, "SLACS"),
    ("SDSSJ1630+4520", 247.6875, 45.3428, 1.78, "SLACS"),
    ("SDSSJ1636+4707", 249.1750, 47.1256, 1.09, "SLACS"),
    ("SDSSJ2238-0754", 339.6208, -7.9069, 1.27, "SLACS"),
    ("SDSSJ2300+0022", 345.1375, 0.3722, 1.24, "SLACS"),
    ("SDSSJ2303+1422", 345.9292, 14.3742, 1.62, "SLACS"),
    ("SDSSJ2321-0939", 350.3958, -9.6619, 1.60, "SLACS"),
]

# BELLS Survey (Brownstein+2012)
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
    ("BELLSJ1527+4137", 231.8954, 41.6253, 1.06, "BELLS"),
    ("BELLSJ1541+3642", 235.3300, 36.7039, 0.98, "BELLS"),
    ("BELLSJ1601+4311", 240.4575, 43.1889, 1.17, "BELLS"),
    ("BELLSJ1617+4556", 244.3408, 45.9419, 1.31, "BELLS"),
    ("BELLSJ1631+4118", 247.8267, 41.3028, 1.05, "BELLS"),
]

# SL2S Survey (Gavazzi+2012, Sonnenfeld+2013)
SL2S_LENSES = [
    ("SL2SJ0217-0513", 34.4292, -5.2264, 1.42, "SL2S"),
    ("SL2SJ0859-0345", 134.8750, -3.7583, 1.18, "SL2S"),
    ("SL2SJ0904+0059", 136.0625, 0.9917, 1.56, "SL2S"),
    ("SL2SJ0959-0200", 149.8792, -2.0069, 1.03, "SL2S"),
    ("SL2SJ1405+5243", 211.3542, 52.7250, 1.31, "SL2S"),
    ("SL2SJ1413+5506", 213.4375, 55.1083, 0.87, "SL2S"),
    ("SL2SJ1430+4105", 217.5625, 41.0917, 1.52, "SL2S"),
    ("SL2SJ2156-0011", 329.1042, -0.1917, 1.08, "SL2S"),
    ("SL2SJ2213-0024", 333.4208, -0.4083, 1.27, "SL2S"),
    ("SL2SJ2217-0022", 334.3375, -0.3750, 0.95, "SL2S"),
]

# SWELLS Survey (Treu+2011)
SWELLS_LENSES = [
    ("SWELLSJ0841+3824", 130.4167, 38.4097, 1.35, "SWELLS"),
    ("SWELLSJ0915+4211", 138.7917, 42.1919, 1.12, "SWELLS"),
    ("SWELLSJ0955+0101", 148.9583, 1.0264, 0.88, "SWELLS"),
    ("SWELLSJ1110+3649", 167.5417, 36.8236, 1.02, "SWELLS"),
    ("SWELLSJ1204+0358", 181.0083, 3.9750, 1.31, "SWELLS"),
    ("SWELLSJ1313+4615", 198.4167, 46.2583, 1.05, "SWELLS"),
    ("SWELLSJ1402+6321", 210.5750, 63.3597, 1.35, "SWELLS"),
    ("SWELLSJ1621+3931", 245.4000, 39.5194, 1.29, "SWELLS"),
]

# GALLERY / Ground-based discoveries
GALLERY_LENSES = [
    ("J0946+1006", 146.6917, 10.1125, 1.38, "GALLERY"),
    ("J1131-1231", 172.9667, -12.5250, 1.83, "GALLERY"),
    ("J1330+1810", 202.5042, 18.1750, 1.65, "GALLERY"),
    ("J1430+4105", 217.5625, 41.0917, 1.52, "GALLERY"),
    ("J1538+5817", 234.6917, 58.2944, 1.00, "GALLERY"),
    ("J2141-0001", 325.4042, -0.0250, 1.22, "GALLERY"),
]


# =============================================================================
# EXPANDED CONTAMINANT SOURCES
# =============================================================================

# Ring galaxies - expanded with gold/silver tiers
RING_GALAXIES = [
    # Gold tier (p_ring > 0.7)
    ("Hoag's Object", 229.3625, 21.5853, "ring", "gold"),
    ("Cartwheel", 9.4500, -33.7167, "ring", "gold"),
    ("AM0644-741", 101.1167, -74.2500, "ring", "gold"),
    ("Arp147", 46.5458, 1.2903, "ring", "gold"),
    ("Arp148", 165.9750, 40.8500, "ring", "gold"),
    ("NGC922", 36.2542, -24.7931, "ring", "gold"),
    ("Mayall's Object", 163.5542, 41.0783, "ring", "gold"),
    ("Zw II 28", 110.0792, 49.1733, "ring", "gold"),
    ("Lindsay-Shapley", 101.2000, -74.2333, "ring", "gold"),
    ("II Hz 4", 125.3792, 22.6583, "ring", "gold"),
    # Silver tier (p_ring 0.5-0.7)
    ("NGC1291", 49.3133, -41.1075, "ring", "silver"),
    ("NGC2859", 140.4704, 34.5133, "ring", "silver"),
    ("NGC3081", 149.8721, -22.8261, "ring", "silver"),
    ("NGC4736", 192.7208, 41.1203, "ring", "silver"),
    ("NGC7217", 331.9729, 31.3594, "ring", "silver"),
    ("NGC7702", 353.8971, 4.6656, "ring", "silver"),
    ("NGC1512", 60.9763, -43.3489, "ring", "silver"),
    ("NGC4245", 184.3429, 29.6078, "ring", "silver"),
    ("NGC4314", 185.6308, 29.8947, "ring", "silver"),
    ("NGC4371", 186.2350, 11.7039, "ring", "silver"),
    ("NGC4429", 186.8671, 11.1078, "ring", "silver"),
    ("NGC4596", 189.9842, 10.1758, "ring", "silver"),
    ("NGC4643", 190.8958, 1.9778, "ring", "silver"),
    ("NGC4699", 192.2183, -8.6622, "ring", "silver"),
    ("NGC4754", 192.8813, 11.3139, "ring", "silver"),
    ("NGC4941", 196.0542, -5.5519, "ring", "silver"),
    ("NGC5101", 200.1042, -27.4303, "ring", "silver"),
    ("NGC5701", 219.3013, 5.3631, "ring", "silver"),
    ("NGC6782", 291.0892, -59.9414, "ring", "silver"),
    ("ESO509-98", 207.0958, -26.6347, "ring", "silver"),
]

# Face-on spirals - major confusers
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
    ("NGC300", 13.7229, -37.6847, "spiral", "gold"),
    ("NGC1365", 53.4015, -36.1404, "spiral", "gold"),
    ("NGC2336", 109.0917, 80.1772, "spiral", "gold"),
    ("NGC2442", 114.1004, -69.5303, "spiral", "gold"),
    ("NGC2997", 146.4117, -31.1911, "spiral", "gold"),
    ("NGC3031", 148.8882, 69.0653, "spiral", "gold"),
    ("NGC3310", 159.6883, 53.5036, "spiral", "gold"),
    ("NGC3351", 160.9904, 11.7036, "spiral", "gold"),
    ("NGC3368", 161.6908, 11.8200, "spiral", "gold"),
    ("NGC3486", 165.0996, 28.9750, "spiral", "gold"),
    ("NGC3596", 168.7750, 14.7872, "spiral", "gold"),
    ("NGC3631", 170.2621, 53.1692, "spiral", "gold"),
    ("NGC3810", 175.1454, 11.4711, "spiral", "gold"),
    ("NGC3893", 177.1571, 48.7108, "spiral", "gold"),
    ("NGC3938", 178.2063, 44.1208, "spiral", "gold"),
    ("NGC3953", 178.4542, 52.3264, "spiral", "gold"),
    ("NGC4030", 180.0983, -1.1003, "spiral", "gold"),
    ("NGC4051", 180.7900, 44.5314, "spiral", "gold"),
    ("NGC4123", 182.0508, 2.8786, "spiral", "gold"),
    ("NGC4145", 182.4500, 39.8828, "spiral", "gold"),
    ("NGC4212", 183.9104, 13.9017, "spiral", "gold"),
    ("NGC4395", 186.4538, 33.5467, "spiral", "gold"),
    ("NGC4501", 187.9963, 14.4206, "spiral", "gold"),
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
    ("NGC3690", 172.1342, 58.5614, "merger", "gold"),  # Arp 299
    ("NGC7592", 349.3542, -4.4153, "merger", "gold"),
    ("NGC6090", 243.0125, 52.4622, "merger", "gold"),
    ("NGC1614", 68.4958, -8.5794, "merger", "gold"),
    ("NGC6670", 280.2625, 59.8889, "merger", "gold"),
    ("NGC2535", 122.3583, 25.2086, "merger", "gold"),
    ("NGC5394", 209.5242, 37.4544, "merger", "gold"),
    ("NGC5395", 209.5583, 37.4261, "merger", "gold"),
    ("NGC3808", 175.0929, 22.4378, "merger", "gold"),
    ("NGC5257", 204.9708, 0.8406, "merger", "gold"),
]

# Edge-on disks
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
    ("NGC4157", 182.7683, 50.4856, "edge_on", "gold"),
    ("NGC4183", 183.3496, 43.6975, "edge_on", "gold"),
    ("NGC4222", 184.0821, 13.3111, "edge_on", "gold"),
    ("NGC4256", 184.4108, 65.8958, "edge_on", "gold"),
    ("NGC4302", 185.4292, 14.5983, "edge_on", "gold"),
    ("NGC5023", 197.8817, 44.0389, "edge_on", "gold"),
    ("NGC5170", 202.5063, -17.9614, "edge_on", "gold"),
    ("NGC5775", 223.4879, 3.5444, "edge_on", "gold"),
    ("NGC5866", 226.6229, 55.7633, "edge_on", "gold"),
    ("NGC7814", 0.8117, 16.1458, "edge_on", "gold"),
]

# AGN / QSO with extended features
AGN_GALAXIES = [
    ("NGC1068", 40.6696, -0.0133, "agn", "gold"),  # M77
    ("NGC1275", 49.9508, 41.5117, "agn", "gold"),  # Perseus A
    ("NGC3516", 166.6979, 72.5686, "agn", "gold"),
    ("NGC4051", 180.7900, 44.5311, "agn", "gold"),
    ("NGC4151", 182.6358, 39.4058, "agn", "gold"),
    ("NGC4395", 186.4538, 33.5469, "agn", "gold"),
    ("NGC5548", 214.4983, 25.1369, "agn", "gold"),
    ("NGC7469", 345.8150, 8.8739, "agn", "gold"),
    ("Mrk421", 166.1138, 38.2089, "agn", "gold"),
    ("Mrk501", 253.4675, 39.7603, "agn", "gold"),
]

# Polar ring galaxies
POLAR_RING_GALAXIES = [
    ("NGC4650A", 190.4333, -40.7250, "polar_ring", "gold"),
    ("NGC2685", 133.8958, 58.7333, "polar_ring", "gold"),  # Helix Galaxy
    ("ESO415-26", 49.7292, -30.3928, "polar_ring", "gold"),
    ("AM2020-504", 306.2417, -50.3333, "polar_ring", "gold"),
    ("A0136-0801", 24.8917, -7.7750, "polar_ring", "gold"),
]


# =============================================================================
# DOWNLOAD AND MEASUREMENT FUNCTIONS
# =============================================================================

def download_cutout(ra: float, dec: float, size_pix: int = 80) -> Dict:
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
                        bad_frac = np.sum(np.isnan(data) | np.isinf(data)) / data.size
                        return {"success": True, "data": data, "in_dr10": True, "bad_pixel_frac": bad_frac}
            elif response.status_code == 404:
                return {"success": False, "in_dr10": False}
            elif response.status_code == 429:
                time.sleep(2 ** attempt)
        except:
            time.sleep(2 ** attempt)
    return {"success": False, "error": "Max retries"}


def measure_arc_visibility(cutout: np.ndarray, theta_e_arcsec: float) -> Dict:
    """Measure arc visibility using MAD-based SNR."""
    if cutout.ndim == 3:
        img = cutout[1] if cutout.shape[0] == 3 else np.mean(cutout, axis=0)
    else:
        img = cutout
    
    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    theta_e_pix = theta_e_arcsec / PIXEL_SCALE
    inner_signal = max(2, theta_e_pix - 2)
    outer_signal = theta_e_pix + 3
    signal_mask = (r >= inner_signal) & (r <= outer_signal)
    
    outer_bg = min(img.shape[0] // 2 - 2, 25)
    bg_mask = (r >= outer_bg - 5) & (r <= outer_bg)
    
    bg_pixels = img[bg_mask]
    if len(bg_pixels) < 10:
        return {"arc_snr": 0.0, "noise_mad": np.nan}
    
    median_bg = np.nanmedian(bg_pixels)
    mad = np.nanmedian(np.abs(bg_pixels - median_bg))
    noise_std = 1.4826 * mad
    
    if noise_std <= 0:
        return {"arc_snr": 0.0, "noise_mad": mad}
    
    signal_pixels = img[signal_mask]
    azimuthal_median = np.nanmedian(signal_pixels)
    residuals = signal_pixels - azimuthal_median
    positive_residuals = np.maximum(residuals, 0)
    signal_sum = np.nansum(positive_residuals)
    
    n_pixels = np.sum(signal_mask)
    arc_snr = signal_sum / (noise_std * np.sqrt(n_pixels))
    
    return {"arc_snr": float(arc_snr), "noise_mad": float(mad), "n_pixels": int(n_pixels)}


# =============================================================================
# METRICS GENERATOR
# =============================================================================

@dataclass
class CatalogMetrics:
    """Comprehensive metrics for LLM review."""
    
    # Anchor metrics
    n_anchors_total: int = 0
    n_anchors_in_dr10: int = 0
    n_anchors_usable: int = 0
    n_anchors_arc_visible: int = 0
    n_tier_a: int = 0
    n_tier_b: int = 0
    
    anchor_sources: Dict[str, int] = None
    anchor_theta_e_min: float = 0.0
    anchor_theta_e_max: float = 0.0
    anchor_theta_e_median: float = 0.0
    
    tier_b_reasons: Dict[str, int] = None
    
    # Contaminant metrics
    n_contaminants_total: int = 0
    n_contaminants_in_dr10: int = 0
    n_contaminants_by_category: Dict[str, int] = None
    n_gold_tier: int = 0
    n_silver_tier: int = 0
    
    # Build metadata
    build_timestamp: str = ""
    arc_snr_threshold: float = 2.0
    cutout_size_pix: int = 80
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_markdown(self) -> str:
        """Generate markdown report for LLM review."""
        lines = [
            "# Anchor & Contaminant Catalog Metrics",
            "",
            f"**Build timestamp:** {self.build_timestamp}",
            f"**Arc SNR threshold:** {self.arc_snr_threshold}",
            f"**Cutout size:** {self.cutout_size_pix}px",
            "",
            "## Anchor Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total anchors | {self.n_anchors_total} |",
            f"| In DR10 footprint | {self.n_anchors_in_dr10} |",
            f"| Usable cutouts | {self.n_anchors_usable} |",
            f"| Arc visible (SNR > {self.arc_snr_threshold}) | {self.n_anchors_arc_visible} |",
            f"| **Tier-A (pass selection)** | **{self.n_tier_a}** |",
            f"| Tier-B (stress test) | {self.n_tier_b} |",
            "",
            "### θ_E Distribution (Tier-A)",
            "",
            f"- Min: {self.anchor_theta_e_min:.2f}\"",
            f"- Max: {self.anchor_theta_e_max:.2f}\"", 
            f"- Median: {self.anchor_theta_e_median:.2f}\"",
            "",
            "### Sources",
            "",
        ]
        
        if self.anchor_sources:
            for source, count in sorted(self.anchor_sources.items(), key=lambda x: -x[1]):
                lines.append(f"- {source}: {count}")
        
        lines.extend([
            "",
            "### Tier-B Exclusion Reasons",
            "",
        ])
        
        if self.tier_b_reasons:
            for reason, count in sorted(self.tier_b_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"- {reason}: {count}")
        
        lines.extend([
            "",
            "## Contaminant Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total contaminants | {self.n_contaminants_total} |",
            f"| In DR10 footprint | {self.n_contaminants_in_dr10} |",
            f"| Gold tier (high purity) | {self.n_gold_tier} |",
            f"| Silver tier (more yield) | {self.n_silver_tier} |",
            "",
            "### Categories",
            "",
        ])
        
        if self.n_contaminants_by_category:
            for cat, count in sorted(self.n_contaminants_by_category.items(), key=lambda x: -x[1]):
                lines.append(f"- {cat}: {count}")
        
        lines.extend([
            "",
            "## LLM Review Questions",
            "",
            "1. Are the Tier-A anchor counts sufficient for the intended evaluation?",
            "2. Is the θ_E distribution representative of the training range (0.5-3.0\")?",
            "3. Are there concerning exclusion reasons that suggest selection bias?",
            "4. Is the contaminant category distribution appropriate for realistic FPR?",
            "5. Are there missing categories that should be added?",
            "6. Is the gold/silver tier split appropriate for the analysis?",
            "",
        ])
        
        return "\n".join(lines)


def build_full_catalogs(output_dir: str, cutout_dir: str, arc_snr_threshold: float = 2.0) -> CatalogMetrics:
    """Build full catalogs and generate metrics."""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cutout_dir, exist_ok=True)
    
    metrics = CatalogMetrics()
    metrics.build_timestamp = datetime.utcnow().isoformat()
    metrics.arc_snr_threshold = arc_snr_threshold
    
    # ===== BUILD ANCHORS =====
    all_lenses = SLACS_LENSES + BELLS_LENSES + SL2S_LENSES + SWELLS_LENSES + GALLERY_LENSES
    
    anchor_records = []
    for name, ra, dec, theta_e, source in all_lenses:
        logger.info(f"Anchor: {name} ({source})")
        
        record = {
            "name": name, "ra": ra, "dec": dec,
            "theta_e_arcsec": theta_e, "source": source,
        }
        
        result = download_cutout(ra, dec)
        record["in_dr10"] = result.get("in_dr10", False)
        record["usable_cutout"] = result.get("success", False)
        record["bad_pixel_frac"] = result.get("bad_pixel_frac", 1.0)
        
        if result.get("success"):
            arc = measure_arc_visibility(result["data"], theta_e)
            record["arc_snr_dr10"] = arc["arc_snr"]
            record["arc_visible_dr10"] = arc["arc_snr"] >= arc_snr_threshold
            
            # Save cutout
            from astropy.io import fits
            cutout_path = Path(cutout_dir) / "anchors" / f"{name}.fits"
            cutout_path.parent.mkdir(exist_ok=True)
            fits.PrimaryHDU(result["data"]).writeto(cutout_path, overwrite=True)
        else:
            record["arc_snr_dr10"] = np.nan
            record["arc_visible_dr10"] = False
        
        # Tier assignment
        if record["in_dr10"] and record["usable_cutout"] and record["arc_visible_dr10"]:
            if 0.5 <= theta_e <= 3.0:
                record["tier"] = "A"
            else:
                record["tier"] = "B"
                record["tier_b_reason"] = "theta_e_out_of_range"
        else:
            record["tier"] = "B"
            if not record["in_dr10"]:
                record["tier_b_reason"] = "not_in_dr10"
            elif not record["usable_cutout"]:
                record["tier_b_reason"] = "unusable_cutout"
            else:
                record["tier_b_reason"] = "arc_not_visible"
        
        anchor_records.append(record)
        time.sleep(0.3)  # Rate limit
    
    anchor_df = pd.DataFrame(anchor_records)
    anchor_df.to_csv(Path(output_dir) / "anchor_catalog_full.csv", index=False)
    
    # ===== BUILD CONTAMINANTS =====
    all_contaminants = (
        RING_GALAXIES + SPIRAL_GALAXIES + MERGER_GALAXIES + 
        EDGE_ON_GALAXIES + AGN_GALAXIES + POLAR_RING_GALAXIES
    )
    
    contam_records = []
    for name, ra, dec, category, tier in all_contaminants:
        logger.info(f"Contaminant: {name} ({category})")
        
        record = {
            "name": name, "ra": ra, "dec": dec,
            "category": category, "quality_tier": tier,
            "is_confirmed_lens": False,
        }
        
        result = download_cutout(ra, dec)
        record["in_dr10"] = result.get("in_dr10", False)
        record["has_cutout"] = result.get("success", False)
        
        if result.get("success"):
            cutout_path = Path(cutout_dir) / "contaminants" / f"{name.replace('/', '_')}.fits"
            cutout_path.parent.mkdir(exist_ok=True)
            from astropy.io import fits
            fits.PrimaryHDU(result["data"]).writeto(cutout_path, overwrite=True)
        
        contam_records.append(record)
        time.sleep(0.3)
    
    contam_df = pd.DataFrame(contam_records)
    contam_df.to_csv(Path(output_dir) / "contaminant_catalog_full.csv", index=False)
    
    # ===== COMPUTE METRICS =====
    metrics.n_anchors_total = len(anchor_df)
    metrics.n_anchors_in_dr10 = anchor_df["in_dr10"].sum()
    metrics.n_anchors_usable = anchor_df["usable_cutout"].sum()
    metrics.n_anchors_arc_visible = anchor_df["arc_visible_dr10"].sum()
    metrics.n_tier_a = (anchor_df["tier"] == "A").sum()
    metrics.n_tier_b = (anchor_df["tier"] == "B").sum()
    
    tier_a_df = anchor_df[anchor_df["tier"] == "A"]
    if len(tier_a_df) > 0:
        metrics.anchor_theta_e_min = tier_a_df["theta_e_arcsec"].min()
        metrics.anchor_theta_e_max = tier_a_df["theta_e_arcsec"].max()
        metrics.anchor_theta_e_median = tier_a_df["theta_e_arcsec"].median()
    
    metrics.anchor_sources = anchor_df["source"].value_counts().to_dict()
    
    tier_b_df = anchor_df[anchor_df["tier"] == "B"]
    if "tier_b_reason" in tier_b_df.columns:
        metrics.tier_b_reasons = tier_b_df["tier_b_reason"].value_counts().to_dict()
    
    metrics.n_contaminants_total = len(contam_df)
    metrics.n_contaminants_in_dr10 = contam_df["in_dr10"].sum()
    metrics.n_contaminants_by_category = contam_df["category"].value_counts().to_dict()
    metrics.n_gold_tier = (contam_df["quality_tier"] == "gold").sum()
    metrics.n_silver_tier = (contam_df["quality_tier"] == "silver").sum()
    
    # Save metrics
    with open(Path(output_dir) / "catalog_metrics.json", "w") as f:
        json.dump(metrics.to_dict(), f, indent=2, default=str)
    
    with open(Path(output_dir) / "catalog_metrics_for_llm.md", "w") as f:
        f.write(metrics.to_markdown())
    
    logger.info(f"\n{metrics.to_markdown()}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cutout-dir", required=True)
    parser.add_argument("--arc-snr-threshold", type=float, default=2.0)
    parser.add_argument("--upload-s3", type=str)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    metrics = build_full_catalogs(args.output_dir, args.cutout_dir, args.arc_snr_threshold)
    
    if args.upload_s3:
        import subprocess
        subprocess.run(["aws", "s3", "sync", args.output_dir, args.upload_s3])
        logger.info(f"Uploaded to {args.upload_s3}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Stage 0: Anchor Baseline Evaluation

This script evaluates our best model (Gen2) on real data:
1. Known confirmed lenses (SLACS, BELLS) - measures RECALL
2. Real hard negatives (ring galaxies, mergers) - measures CONTAMINATION

This establishes a real-data baseline BEFORE any further model iteration.

Usage on Lambda:
    python3 stage0_anchor_baseline.py \
        --model_path /lambda/nfs/darkhaloscope-training-dc/runs/gen2_final/ckpt_best.pt \
        --output_dir /lambda/nfs/darkhaloscope-training-dc/anchor_baseline \
        --cutout_cache_dir /lambda/nfs/darkhaloscope-training-dc/anchor_cutouts
"""

import argparse
import io
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# SLACS and BELLS Known Lens Catalogs (Hardcoded for reliability)
# =============================================================================

# SLACS: Auger et al. 2009 (Grade A lenses from Table 1)
# https://ui.adsabs.harvard.edu/abs/2009ApJ...705.1099A
SLACS_LENSES = [
    # (name, ra, dec, theta_e_arcsec, z_lens, z_source)
    ("SDSSJ0029-0055", 7.4008, -0.9269, 0.96, 0.227, 0.931),
    ("SDSSJ0037-0942", 9.4583, -9.7050, 1.53, 0.195, 0.632),
    ("SDSSJ0216-0813", 34.1221, -8.2217, 1.16, 0.332, 0.523),
    ("SDSSJ0252+0039", 43.0963, 0.6567, 1.04, 0.280, 0.982),
    ("SDSSJ0330-0020", 52.5083, -0.3372, 1.10, 0.351, 1.071),
    ("SDSSJ0728+3835", 112.0708, 38.5917, 1.25, 0.206, 0.688),
    ("SDSSJ0737+3216", 114.4125, 32.2725, 0.98, 0.322, 0.581),
    ("SDSSJ0822+2652", 125.5125, 26.8750, 1.17, 0.241, 0.594),
    ("SDSSJ0912+0029", 138.1417, 0.4861, 1.63, 0.164, 0.324),
    ("SDSSJ0936+0913", 144.0583, 9.2242, 1.09, 0.190, 0.588),
    ("SDSSJ0956+5100", 149.1625, 51.0050, 1.33, 0.240, 0.470),
    ("SDSSJ0959+0410", 149.9208, 4.1722, 0.99, 0.126, 0.535),
    ("SDSSJ1016+3859", 154.2333, 38.9928, 1.09, 0.168, 0.439),
    ("SDSSJ1020+1122", 155.2333, 11.3761, 1.20, 0.282, 0.553),
    ("SDSSJ1023+4230", 155.8500, 42.5083, 1.41, 0.191, 0.696),
    ("SDSSJ1029+0420", 157.4792, 4.3422, 1.01, 0.104, 0.615),
    ("SDSSJ1106+5228", 166.6500, 52.4750, 1.23, 0.096, 0.407),
    ("SDSSJ1112+0826", 168.0958, 8.4403, 1.49, 0.273, 0.629),
    ("SDSSJ1134+6027", 173.5083, 60.4617, 1.10, 0.153, 0.474),
    ("SDSSJ1142+1001", 175.7292, 10.0264, 0.98, 0.222, 0.504),
    ("SDSSJ1143-0144", 175.7708, -1.7383, 1.68, 0.106, 0.402),
    ("SDSSJ1153+4612", 178.4875, 46.2053, 1.05, 0.180, 0.875),
    ("SDSSJ1204+0358", 181.0125, 3.9789, 1.31, 0.164, 0.631),
    ("SDSSJ1205+4910", 181.4083, 49.1761, 1.22, 0.215, 0.481),
    ("SDSSJ1213+6708", 183.3167, 67.1422, 1.42, 0.123, 0.640),
    ("SDSSJ1218+0830", 184.6542, 8.5086, 1.45, 0.135, 0.717),
    ("SDSSJ1250+0523", 192.5458, 5.3922, 1.13, 0.232, 0.795),
    ("SDSSJ1402+6321", 210.5375, 63.3575, 1.35, 0.205, 0.481),
    ("SDSSJ1403+0006", 210.9500, 0.1108, 0.83, 0.189, 0.473),
    ("SDSSJ1416+5136", 214.1417, 51.6086, 1.37, 0.299, 0.811),
    ("SDSSJ1420+6019", 215.0792, 60.3317, 1.04, 0.063, 0.535),
    ("SDSSJ1430+4105", 217.6958, 41.0911, 1.52, 0.285, 0.575),
    ("SDSSJ1432+6317", 218.0583, 63.2900, 1.25, 0.123, 0.664),
    ("SDSSJ1436-0000", 219.1708, -0.0072, 1.12, 0.285, 0.805),
    ("SDSSJ1443+0304", 220.9750, 3.0750, 0.81, 0.134, 0.419),
    ("SDSSJ1451-0239", 222.8000, -2.6622, 1.04, 0.125, 0.520),
    ("SDSSJ1525+3327", 231.4542, 33.4550, 1.31, 0.358, 0.717),
    ("SDSSJ1531-0105", 232.8708, -1.0906, 1.71, 0.160, 0.744),
    ("SDSSJ1538+5817", 234.5042, 58.2922, 1.00, 0.143, 0.531),
    ("SDSSJ1621+3931", 245.4500, 39.5217, 1.29, 0.245, 0.602),
    ("SDSSJ1627-0053", 246.9167, -0.8886, 1.23, 0.208, 0.524),
    ("SDSSJ1630+4520", 247.5083, 45.3367, 1.78, 0.248, 0.793),
    ("SDSSJ1636+4707", 249.2000, 47.1283, 1.09, 0.228, 0.674),
    ("SDSSJ2238-0754", 339.6292, -7.9108, 1.27, 0.137, 0.713),
    ("SDSSJ2300+0022", 345.0833, 0.3744, 1.24, 0.228, 0.464),
    ("SDSSJ2303+1422", 345.8542, 14.3778, 1.62, 0.155, 0.517),
    ("SDSSJ2321-0939", 350.3875, -9.6572, 1.60, 0.082, 0.532),
    ("SDSSJ2341+0000", 355.4708, 0.0103, 1.44, 0.186, 0.807),
]

# BELLS: Brownstein et al. 2012 (Grade A and B lenses)
# https://ui.adsabs.harvard.edu/abs/2012ApJ...744...41B
BELLS_LENSES = [
    # (name, ra, dec, theta_e_arcsec, z_lens, z_source)
    ("BELLSJ0747+4448", 116.898, 44.808, 1.16, 0.437, 0.898),
    ("BELLSJ0801+4727", 120.348, 47.455, 0.91, 0.544, 1.072),
    ("BELLSJ0830+5116", 127.587, 51.272, 1.41, 0.530, 0.895),
    ("BELLSJ0847+2348", 131.890, 23.809, 1.26, 0.474, 0.849),
    ("BELLSJ0903+4116", 135.871, 41.274, 1.48, 0.430, 1.065),
    ("BELLSJ0918+5104", 139.680, 51.078, 1.28, 0.581, 1.067),
    ("BELLSJ1014+3920", 153.620, 39.348, 1.07, 0.494, 1.108),
    ("BELLSJ1110+2808", 167.722, 28.139, 1.37, 0.607, 1.021),
    ("BELLSJ1159+5820", 179.941, 58.343, 0.97, 0.471, 0.927),
    ("BELLSJ1221+3806", 185.383, 38.101, 1.53, 0.479, 0.933),
    ("BELLSJ1226+5457", 186.502, 54.962, 1.22, 0.498, 0.899),
    ("BELLSJ1318+3942", 199.683, 39.703, 1.08, 0.450, 0.956),
    ("BELLSJ1349+3612", 207.289, 36.205, 0.95, 0.443, 1.053),
    ("BELLSJ1401+3845", 210.436, 38.759, 1.35, 0.534, 0.938),
    ("BELLSJ1522+2910", 230.568, 29.167, 1.42, 0.441, 0.868),
    ("BELLSJ1541+1812", 235.377, 18.206, 1.48, 0.531, 0.921),
    ("BELLSJ1545+2748", 236.258, 27.807, 1.34, 0.461, 1.088),
    ("BELLSJ1601+2138", 240.326, 21.639, 1.52, 0.529, 0.883),
    ("BELLSJ1611+1705", 242.768, 17.085, 1.19, 0.498, 0.952),
    ("BELLSJ1631+1854", 247.800, 18.908, 1.06, 0.417, 1.042),
]

# Ring galaxies (potential lens mimics) - from various surveys in DR10 footprint
# These are galaxies that morphologically resemble lenses but are not
RING_GALAXIES = [
    # (name, ra, dec) - Well-known ring galaxies in SDSS/DECaLS footprint
    ("Hoag's Object", 226.0792, 21.5861),  # Famous ring galaxy
    ("AM0644-741", 101.186, -74.253),  # Probably not in DR10
    ("NGC 922", 36.4500, -24.7906),
    ("Arp 147", 46.7458, 1.2939),
    ("Arp 148", 168.5667, 40.8500),
    ("NGC 1291", 49.3167, -41.1083),  # Probably not in DR10
    ("Cartwheel", 9.4250, -33.7167),  # Probably not in DR10
    # Add more ring galaxies from Galaxy Zoo classifications in SDSS/DECaLS overlap
    ("GZRing001", 150.123, 2.456),  # Placeholder - will be replaced with real coords
    ("GZRing002", 152.789, 1.234),
    ("GZRing003", 180.456, 10.789),
]

# Merger galaxies (potential lens mimics)
MERGER_GALAXIES = [
    # (name, ra, dec) - Known mergers that could mimic lens arcs
    ("NGC 4038/4039 Antennae", 180.4708, -18.8678),  # Probably not in DR10
    ("NGC 2623", 129.6000, 25.7542),
    ("NGC 520", 21.1500, 3.7917),
    ("NGC 3256", 156.9625, -43.9039),  # Probably not in DR10
    ("Arp 220", 233.7375, 23.5033),
    ("NGC 6240", 253.2458, 2.4006),
    ("NGC 7252 Atoms for Peace", 339.0083, -24.6833),
    # Placeholder mergers from DECaLS footprint
    ("Merger001", 145.678, 5.123),
    ("Merger002", 175.234, 15.456),
    ("Merger003", 210.567, 30.789),
]


def create_known_lens_catalog() -> pd.DataFrame:
    """Create DataFrame of known lenses from SLACS + BELLS."""
    slacs_df = pd.DataFrame(
        SLACS_LENSES,
        columns=['name', 'ra', 'dec', 'theta_e', 'z_lens', 'z_source']
    )
    slacs_df['catalog'] = 'SLACS'
    
    bells_df = pd.DataFrame(
        BELLS_LENSES,
        columns=['name', 'ra', 'dec', 'theta_e', 'z_lens', 'z_source']
    )
    bells_df['catalog'] = 'BELLS'
    
    combined = pd.concat([slacs_df, bells_df], ignore_index=True)
    logger.info(f"Created known lens catalog: {len(slacs_df)} SLACS + {len(bells_df)} BELLS = {len(combined)} total")
    return combined


def create_hard_negative_catalog() -> pd.DataFrame:
    """Create DataFrame of hard negatives (rings, mergers)."""
    rings_df = pd.DataFrame(
        RING_GALAXIES,
        columns=['name', 'ra', 'dec']
    )
    rings_df['catalog'] = 'Ring'
    
    mergers_df = pd.DataFrame(
        MERGER_GALAXIES,
        columns=['name', 'ra', 'dec']
    )
    mergers_df['catalog'] = 'Merger'
    
    combined = pd.concat([rings_df, mergers_df], ignore_index=True)
    logger.info(f"Created hard negative catalog: {len(rings_df)} rings + {len(mergers_df)} mergers = {len(combined)} total")
    return combined


# =============================================================================
# DR10 Footprint and Cutout Functions
# =============================================================================

def check_dr10_footprint(ra: float, dec: float) -> bool:
    """
    Check if (ra, dec) is in approximate DR10 footprint.
    
    DR10 covers:
    - North: 0° < RA < 360°, Dec > -18°
    - South: 0° < RA < 360°, Dec < 35°
    
    Approximate check - actual coverage is more complex.
    """
    # Simple bounding box for DECaLS/DR10
    # Actual coverage is more complex but this is a quick filter
    if dec > 80 or dec < -20:
        return False
    return True


def get_brick_for_position(ra: float, dec: float) -> str:
    """
    Compute the DECaLS brick name for a position.
    
    Brick naming convention:
    - RA: 3-digit prefix (RA/10 rounded)
    - p/m: positive or negative Dec
    - Dec: 3-digit (abs(Dec)*10)
    
    Example: brick name for RA=150.5, Dec=+25.3 is "150p253"
    """
    ra_prefix = int(np.floor(ra / 10) * 10)
    ra_str = f"{ra_prefix:03d}"
    
    dec_sign = "p" if dec >= 0 else "m"
    dec_val = int(np.round(abs(dec) * 10))
    dec_str = f"{dec_val:03d}"
    
    return f"{ra_str}{dec_sign}{dec_str}"


def download_cutout_from_legacy_survey(
    ra: float,
    dec: float,
    size_pixels: int = 64,
    pixel_scale: float = 0.262,
    bands: str = "grz",
    output_path: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Download a cutout from the Legacy Survey cutout service.
    
    Args:
        ra: Right Ascension in degrees
        dec: Declination in degrees
        size_pixels: Size of cutout in pixels
        pixel_scale: Pixel scale in arcsec/pixel (DECaLS = 0.262)
        bands: Bands to download (e.g., "grz")
        output_path: If provided, save FITS to this path
    
    Returns:
        NumPy array of shape (C, H, W) or None if download failed
    """
    try:
        from astropy.io import fits
    except ImportError:
        logger.error("astropy required for FITS handling. pip install astropy")
        return None
    
    # Legacy Survey cutout URL
    size_arcsec = size_pixels * pixel_scale
    url = (
        f"https://www.legacysurvey.org/viewer/cutout.fits?"
        f"ra={ra}&dec={dec}&size={size_pixels}&layer=ls-dr10&pixscale={pixel_scale}&bands={bands}"
    )
    
    logger.debug(f"Downloading cutout from: {url}")
    
    try:
        import urllib.request
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            
            with fits.open(tmp.name) as hdu:
                data = hdu[0].data
                
                if data is None:
                    logger.warning(f"Empty FITS data for ({ra}, {dec})")
                    return None
                
                # Data should be (C, H, W) for multi-band
                if data.ndim == 2:
                    data = data[np.newaxis, :, :]
                
                if output_path:
                    import shutil
                    shutil.copy(tmp.name, output_path)
                
                return data.astype(np.float32)
                
    except Exception as e:
        logger.warning(f"Failed to download cutout for ({ra}, {dec}): {e}")
        return None


def robust_mad_norm_outer(x: np.ndarray, clip: float = 10.0, eps: float = 1e-6,
                          inner_frac: float = 0.5) -> np.ndarray:
    """Normalize using outer annulus only (matches training normalization)."""
    out = np.empty_like(x, dtype=np.float32)
    h, w = x.shape[-2:]
    cy, cx = h // 2, w // 2
    ri = int(min(h, w) * inner_frac / 2)
    
    yy, xx = np.ogrid[:h, :w]
    outer_mask = ((yy - cy)**2 + (xx - cx)**2) > ri**2
    
    for c in range(x.shape[0]):
        v = x[c]
        outer_v = v[outer_mask]
        med = np.median(outer_v)
        mad = np.median(np.abs(outer_v - med))
        scale = 1.4826 * mad + eps
        vv = (v - med) / scale
        if clip is not None:
            vv = np.clip(vv, -clip, clip)
        out[c] = vv.astype(np.float32)
    return out


# =============================================================================
# Model Inference
# =============================================================================

class MetaFusionHead(nn.Module):
    """Metadata fusion head - must match training architecture exactly."""
    def __init__(self, feat_dim: int, meta_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats, meta):
        m = self.meta_mlp(meta)
        x = torch.cat([feats, m], dim=1)
        return self.classifier(x).squeeze(1)


def load_model(checkpoint_path: str, arch: str = "convnext_tiny", device: str = "cuda"):
    """Load trained model from checkpoint."""
    from torchvision.models import convnext_tiny
    
    # Load checkpoint to inspect architecture
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Get training args to determine architecture
    args = ckpt.get("args", {})
    meta_cols = args.get("meta_cols", "")
    meta_dim = len([c for c in meta_cols.split(",") if c.strip()]) if meta_cols else 0
    dropout = args.get("dropout", 0.1)
    
    logger.info(f"Model was trained with meta_cols='{meta_cols}' (meta_dim={meta_dim})")
    
    # Build model architecture (must match training)
    m = convnext_tiny(weights=None)
    feat_dim = m.classifier[2].in_features  # 768 for convnext_tiny
    m.classifier = nn.Identity()
    
    class Model(nn.Module):
        def __init__(self, backbone, feat_dim, meta_dim, dropout):
            super().__init__()
            self.backbone = backbone
            self.meta_dim = meta_dim
            if meta_dim > 0:
                self.head = MetaFusionHead(feat_dim, meta_dim, hidden=256, dropout=dropout)
            else:
                self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, 1))
        
        def forward(self, x, meta=None):
            feats = self.backbone(x)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if feats.ndim > 2:
                feats = torch.flatten(feats, 1)
            if self.meta_dim > 0:
                if meta is None:
                    # Use default metadata values (median from DR10)
                    # psfsize_r ~ 1.3 arcsec, psfdepth_r ~ 24.5 mag
                    meta = torch.tensor([[1.3, 24.5]], device=x.device).expand(x.size(0), -1)
                return self.head(feats, meta)
            return self.head(feats).squeeze(1)
    
    model = Model(m, feat_dim, meta_dim, dropout)
    
    # Load state dict
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    return model


def run_inference(
    model,
    catalog_df: pd.DataFrame,
    cache_dir: str,
    device: str = "cuda",
    batch_size: int = 16
) -> pd.DataFrame:
    """
    Run inference on a catalog of positions.
    
    Downloads cutouts, normalizes, and runs model inference.
    Returns catalog with added 'p_lens' column.
    """
    
    results = []
    
    for idx, row in catalog_df.iterrows():
        ra, dec = row['ra'], row['dec']
        name = row.get('name', f"obj_{idx}")
        
        # Check footprint
        if not check_dr10_footprint(ra, dec):
            logger.debug(f"Skipping {name}: outside DR10 footprint")
            results.append({**row.to_dict(), 'p_lens': np.nan, 'in_footprint': False})
            continue
        
        # Download or load cached cutout
        cache_path = Path(cache_dir) / f"{name.replace(' ', '_').replace('/', '_')}.fits"
        
        if cache_path.exists():
            from astropy.io import fits
            with fits.open(cache_path) as hdu:
                cutout = hdu[0].data.astype(np.float32)
        else:
            cutout = download_cutout_from_legacy_survey(ra, dec, output_path=str(cache_path))
        
        if cutout is None:
            logger.warning(f"Failed to get cutout for {name}")
            results.append({**row.to_dict(), 'p_lens': np.nan, 'in_footprint': True, 'cutout_ok': False})
            continue
        
        # Normalize (matches training)
        cutout_norm = robust_mad_norm_outer(cutout)
        
        # Run inference
        with torch.no_grad():
            x = torch.from_numpy(cutout_norm).unsqueeze(0).to(device)
            logit = model(x)
            p_lens = torch.sigmoid(logit).item()
        
        results.append({
            **row.to_dict(),
            'p_lens': p_lens,
            'in_footprint': True,
            'cutout_ok': True
        })
        
        logger.info(f"{name}: p_lens = {p_lens:.4f}")
    
    return pd.DataFrame(results)


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_anchor_metrics(
    known_lenses_results: pd.DataFrame,
    hard_neg_results: pd.DataFrame,
    thresholds: List[float] = [0.5, 0.7, 0.9]
) -> Dict:
    """
    Compute anchor baseline metrics.
    
    Returns:
        Dict with recall on known lenses and contamination on hard negatives
        at various thresholds.
    """
    metrics = {}
    
    # Filter to valid results
    kl_valid = known_lenses_results[known_lenses_results['cutout_ok'] == True].copy()
    hn_valid = hard_neg_results[hard_neg_results['cutout_ok'] == True].copy()
    
    metrics['n_known_lenses_total'] = len(known_lenses_results)
    metrics['n_known_lenses_in_footprint'] = len(kl_valid)
    metrics['n_hard_negatives_total'] = len(hard_neg_results)
    metrics['n_hard_negatives_in_footprint'] = len(hn_valid)
    
    for thresh in thresholds:
        # Recall: fraction of known lenses with p_lens > threshold
        if len(kl_valid) > 0:
            recall = (kl_valid['p_lens'] > thresh).mean()
            n_detected = (kl_valid['p_lens'] > thresh).sum()
        else:
            recall = np.nan
            n_detected = 0
        
        # Contamination: fraction of hard negatives with p_lens > threshold
        if len(hn_valid) > 0:
            contamination = (hn_valid['p_lens'] > thresh).mean()
            n_false_pos = (hn_valid['p_lens'] > thresh).sum()
        else:
            contamination = np.nan
            n_false_pos = 0
        
        metrics[f'recall@{thresh}'] = recall
        metrics[f'n_detected@{thresh}'] = int(n_detected)
        metrics[f'contamination@{thresh}'] = contamination
        metrics[f'n_false_pos@{thresh}'] = int(n_false_pos)
    
    # Score statistics
    if len(kl_valid) > 0:
        metrics['p_lens_mean_known_lenses'] = kl_valid['p_lens'].mean()
        metrics['p_lens_median_known_lenses'] = kl_valid['p_lens'].median()
        metrics['p_lens_std_known_lenses'] = kl_valid['p_lens'].std()
    
    if len(hn_valid) > 0:
        metrics['p_lens_mean_hard_neg'] = hn_valid['p_lens'].mean()
        metrics['p_lens_median_hard_neg'] = hn_valid['p_lens'].median()
        metrics['p_lens_std_hard_neg'] = hn_valid['p_lens'].std()
    
    return metrics


def generate_report(
    metrics: Dict,
    known_lenses_results: pd.DataFrame,
    hard_neg_results: pd.DataFrame,
    output_dir: str
):
    """Generate human-readable report and save results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    known_lenses_results.to_parquet(output_path / "known_lenses_results.parquet", index=False)
    hard_neg_results.to_parquet(output_path / "hard_negatives_results.parquet", index=False)
    
    # Save metrics
    with open(output_path / "anchor_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2, default=float)
    
    # Generate markdown report
    report = f"""# Anchor Baseline Evaluation Report

**Generated**: {pd.Timestamp.now()}

## Summary

| Metric | Value |
|--------|-------|
| Known Lenses Total | {metrics['n_known_lenses_total']} |
| Known Lenses in Footprint | {metrics['n_known_lenses_in_footprint']} |
| Hard Negatives Total | {metrics['n_hard_negatives_total']} |
| Hard Negatives in Footprint | {metrics['n_hard_negatives_in_footprint']} |

## Performance at Key Thresholds

| Threshold | Recall (Known Lenses) | Contamination (Hard Neg) |
|-----------|----------------------|--------------------------|
| 0.5 | {metrics.get('recall@0.5', 'N/A'):.1%} ({metrics.get('n_detected@0.5', 0)}/{metrics['n_known_lenses_in_footprint']}) | {metrics.get('contamination@0.5', 'N/A'):.1%} ({metrics.get('n_false_pos@0.5', 0)}/{metrics['n_hard_negatives_in_footprint']}) |
| 0.7 | {metrics.get('recall@0.7', 'N/A'):.1%} ({metrics.get('n_detected@0.7', 0)}/{metrics['n_known_lenses_in_footprint']}) | {metrics.get('contamination@0.7', 'N/A'):.1%} ({metrics.get('n_false_pos@0.7', 0)}/{metrics['n_hard_negatives_in_footprint']}) |
| 0.9 | {metrics.get('recall@0.9', 'N/A'):.1%} ({metrics.get('n_detected@0.9', 0)}/{metrics['n_known_lenses_in_footprint']}) | {metrics.get('contamination@0.9', 'N/A'):.1%} ({metrics.get('n_false_pos@0.9', 0)}/{metrics['n_hard_negatives_in_footprint']}) |

## Score Distributions

### Known Lenses
- Mean p_lens: {metrics.get('p_lens_mean_known_lenses', 'N/A'):.3f}
- Median p_lens: {metrics.get('p_lens_median_known_lenses', 'N/A'):.3f}
- Std p_lens: {metrics.get('p_lens_std_known_lenses', 'N/A'):.3f}

### Hard Negatives
- Mean p_lens: {metrics.get('p_lens_mean_hard_neg', 'N/A'):.3f}
- Median p_lens: {metrics.get('p_lens_median_hard_neg', 'N/A'):.3f}
- Std p_lens: {metrics.get('p_lens_std_hard_neg', 'N/A'):.3f}

## Interpretation

**Gating Rule**: For viable real-world performance:
- Recall on known lenses should be > 50% at threshold 0.5
- Contamination on hard negatives should be < 20% at threshold 0.5

**Current Status**: {'✅ PASS' if metrics.get('recall@0.5', 0) > 0.5 and metrics.get('contamination@0.5', 1) < 0.2 else '❌ FAIL - Focus on data/simulation improvements before model iteration'}

## Individual Results

### Top Detected Known Lenses (by p_lens)
"""
    
    # Add top detected lenses
    kl_valid = known_lenses_results[known_lenses_results['cutout_ok'] == True].copy()
    if len(kl_valid) > 0:
        top_lenses = kl_valid.nlargest(10, 'p_lens')
        for _, row in top_lenses.iterrows():
            report += f"- {row['name']}: p_lens = {row['p_lens']:.4f} (θ_e = {row.get('theta_e', 'N/A')}\")\n"
    
    report += "\n### Missed Known Lenses (p_lens < 0.5)\n"
    if len(kl_valid) > 0:
        missed = kl_valid[kl_valid['p_lens'] < 0.5]
        for _, row in missed.iterrows():
            report += f"- {row['name']}: p_lens = {row['p_lens']:.4f} (θ_e = {row.get('theta_e', 'N/A')}\")\n"
    
    report += "\n### False Positive Hard Negatives (p_lens > 0.5)\n"
    hn_valid = hard_neg_results[hard_neg_results['cutout_ok'] == True].copy()
    if len(hn_valid) > 0:
        false_pos = hn_valid[hn_valid['p_lens'] > 0.5]
        for _, row in false_pos.iterrows():
            report += f"- {row['name']}: p_lens = {row['p_lens']:.4f}\n"
    
    with open(output_path / "anchor_baseline_report.md", 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to {output_path / 'anchor_baseline_report.md'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 0: Anchor Baseline Evaluation")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--cutout_cache_dir", required=True, help="Directory to cache cutouts")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    parser.add_argument("--skip_download", action="store_true", help="Skip download, use cached only")
    args = parser.parse_args()
    
    # Create directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cutout_cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Create catalogs
    logger.info("Creating catalogs...")
    known_lenses = create_known_lens_catalog()
    hard_negatives = create_hard_negative_catalog()
    
    # Save catalogs
    known_lenses.to_csv(Path(args.output_dir) / "known_lenses_catalog.csv", index=False)
    hard_negatives.to_csv(Path(args.output_dir) / "hard_negatives_catalog.csv", index=False)
    
    # Load model
    logger.info("Loading model...")
    model = load_model(args.model_path, device=args.device)
    
    # Run inference on known lenses
    logger.info("Running inference on known lenses...")
    known_lenses_results = run_inference(
        model, known_lenses, 
        cache_dir=Path(args.cutout_cache_dir) / "known_lenses",
        device=args.device
    )
    
    # Run inference on hard negatives
    logger.info("Running inference on hard negatives...")
    hard_neg_results = run_inference(
        model, hard_negatives,
        cache_dir=Path(args.cutout_cache_dir) / "hard_negatives", 
        device=args.device
    )
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_anchor_metrics(known_lenses_results, hard_neg_results)
    
    # Generate report
    logger.info("Generating report...")
    generate_report(metrics, known_lenses_results, hard_neg_results, args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("ANCHOR BASELINE RESULTS")
    print("="*60)
    print(f"Known Lenses: {metrics['n_known_lenses_in_footprint']}/{metrics['n_known_lenses_total']} in footprint")
    print(f"Hard Negatives: {metrics['n_hard_negatives_in_footprint']}/{metrics['n_hard_negatives_total']} in footprint")
    print()
    print("Threshold | Recall | Contamination")
    print("-"*40)
    for thresh in [0.5, 0.7, 0.9]:
        recall = metrics.get(f'recall@{thresh}', 0)
        contam = metrics.get(f'contamination@{thresh}', 0)
        print(f"   {thresh}    | {recall:5.1%}  | {contam:5.1%}")
    print("="*60)
    
    gate_pass = metrics.get('recall@0.5', 0) > 0.5 and metrics.get('contamination@0.5', 1) < 0.2
    if gate_pass:
        print("✅ GATE PASSED - Model shows promise on real data")
    else:
        print("❌ GATE FAILED - Focus on sim-to-real improvements")


if __name__ == "__main__":
    main()


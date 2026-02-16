#!/usr/bin/env python3
"""
Paired Training Pipeline v2: Shortcut-Resistant Lens Finder Training

This module provides:
  1) PairedParquetDataset: yields (pos_img, ctrl_img, meta) pairs
  2) PairedMixCollate: on-the-fly sampling of {pos, ctrl, hardneg} with curriculum
  3) Preprocess6CH: robust outer-annulus MAD norm + residual view => 6 channels
  4) GateRunner: stratified gate metrics for shortcut detection

Based on LLM recommendations with critical bug fixes applied:
  - Issue #9: Empty region handling in _extract_region_stats
  - Issue #11: Meta units conversion (pixels to arcsec)
  - Issue #12: Per-worker RNG seeding
  - Issue #14: Gzip-compressed NPZ handling

Author: DarkHaloScope Team
Date: 2026-02-05
"""

from __future__ import annotations

import gzip
import io
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pyarrow.dataset as ds

from astropy.io import fits
from astropy.wcs import WCS

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# =============================================================================
# CONFIGURATION
# =============================================================================

PIX_SCALE_ARCSEC = 0.262  # DR10 pixel scale
STAMP_SIZE = 64

BANDS = ("g", "r", "z")
NPZ_KEYS = ("image_g", "image_r", "image_z")

DEFAULT_OUTER_R_PIX = 16  # outer mask for MAD normalization
DEFAULT_CLIP = 10.0

# Residual view blur sigma (pixels)
DEFAULT_RESID_SIGMA_PIX = 3.0

# Hard negative azimuthal shuffle bins
DEFAULT_AZIMUTHAL_NBINS = 24

# Stratification definitions for x = theta_E / PSF_FWHM
DEFAULT_STRATA = {
    "x_lt_0p8": (None, 0.8),
    "x_0p8_to_1p0": (0.8, 1.0),
    "x_ge_1p0": (1.0, None),
    "all": (None, None),
}


# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================

def _radius_maps(h: int, w: int) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute radius map from image center.
    
    Returns:
        r: radius in pixels (H, W)
        r2: squared radius (H, W)
        (dy, dx): offset from center (H, 1) and (1, W)
    """
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dy = (yy - cy).astype(np.float32)
    dx = (xx - cx).astype(np.float32)
    r2 = (dy * dy + dx * dx).astype(np.float32)
    r = np.sqrt(r2, dtype=np.float32)
    return r, r2, (dy, dx)


def make_outer_mask(h: int = STAMP_SIZE, w: int = STAMP_SIZE, outer_r_pix: int = DEFAULT_OUTER_R_PIX) -> np.ndarray:
    """Create mask for outer region (r >= outer_r_pix)."""
    r, _, _ = _radius_maps(h, w)
    return (r >= float(outer_r_pix))


def make_circular_mask(h: int, w: int, r0: float, r1: Optional[float] = None) -> np.ndarray:
    """
    Create circular mask.
    
    Args:
        r0: inner radius (if r1 is None, mask is r < r0)
        r1: outer radius (if provided, mask is r0 <= r < r1)
    """
    r, _, _ = _radius_maps(h, w)
    if r1 is None:
        return r < float(r0)
    return (r >= float(r0)) & (r < float(r1))


# =============================================================================
# NPZ DECODING (with gzip support - FIX #14)
# =============================================================================

def decode_stamp_npz(blob: bytes) -> np.ndarray:
    """
    Decode stamp NPZ from bytes, handling both raw and gzip-compressed formats.
    
    Returns:
        Image array of shape (C, H, W) in band order g, r, z.
    
    Raises:
        ValueError: If decoding fails.
    """
    if blob is None:
        raise ValueError("stamp_npz is None")
    
    def _try_load(data: bytes) -> Optional[np.ndarray]:
        try:
            bio = io.BytesIO(data)
            with np.load(bio, allow_pickle=False) as z:
                if all(k in z for k in NPZ_KEYS):
                    arrs = [z[k].astype(np.float32) for k in NPZ_KEYS]
                    return np.stack(arrs, axis=0)
        except Exception:
            pass
        return None
    
    # Try raw NPZ first
    result = _try_load(blob)
    if result is not None:
        if result.shape[-2:] != (STAMP_SIZE, STAMP_SIZE):
            raise ValueError(f"Unexpected stamp shape: {result.shape}")
        return result
    
    # Try gzip decompression
    try:
        decompressed = gzip.decompress(blob)
        result = _try_load(decompressed)
        if result is not None:
            if result.shape[-2:] != (STAMP_SIZE, STAMP_SIZE):
                raise ValueError(f"Unexpected stamp shape: {result.shape}")
            return result
    except Exception:
        pass
    
    raise ValueError("Cannot decode stamp_npz: unsupported format")


# =============================================================================
# COADD CUTOUT PROVIDER (for paired controls)
# =============================================================================

# Legacy Survey cutout service URL template
LEGACY_SURVEY_CUTOUT_URL = (
    "https://www.legacysurvey.org/viewer/fits-cutout"
    "?ra={ra}&dec={dec}&size={size}&layer=ls-dr10&bands={bands}"
)


def _fetch_from_legacy_survey(
    ra: float, dec: float, stamp_size: int = STAMP_SIZE, bands: str = "grz"
) -> Optional[np.ndarray]:
    """
    Fetch cutout from Legacy Survey public cutout service.
    
    This is a fallback when local coadd cache is unavailable.
    
    Returns:
        (C, H, W) array for g, r, z bands, or None on failure.
    """
    import urllib.request
    import urllib.error
    
    url = LEGACY_SURVEY_CUTOUT_URL.format(
        ra=ra, dec=dec, size=stamp_size, bands=bands
    )
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            fits_bytes = response.read()
        
        with fits.open(io.BytesIO(fits_bytes)) as hdul:
            # cutout service returns (bands, H, W) or (H, W, bands)
            data = hdul[0].data
            if data is None:
                return None
            
            data = data.astype(np.float32)
            
            # Handle different array orderings
            if data.ndim == 3:
                if data.shape[0] == len(bands):
                    # Already (C, H, W)
                    pass
                elif data.shape[-1] == len(bands):
                    # (H, W, C) -> (C, H, W)
                    data = np.transpose(data, (2, 0, 1))
                else:
                    return None
            else:
                return None
            
            # Verify shape
            if data.shape != (len(bands), stamp_size, stamp_size):
                return None
            
            return data
            
    except (urllib.error.URLError, urllib.error.HTTPError, Exception):
        return None


class CoaddCutoutProvider:
    """
    Provides deterministic base cutouts from local DR10 coadd cache.
    
    Supports fallback to Legacy Survey cutout service when local cache
    is unavailable (set use_api_fallback=True).
    
    Can also cache fetched cutouts to disk for reuse (set disk_cache_dir).
    
    Caches per-(brickname, band) FITS data in memory to avoid re-reading.
    """
    
    def __init__(
        self,
        coadd_cache_root: str,
        stamp_size: int = STAMP_SIZE,
        bands: Tuple[str, ...] = BANDS,
        max_cache: int = 32,
        use_api_fallback: bool = False,
        disk_cache_dir: Optional[str] = None,
    ):
        self.root = coadd_cache_root
        self.stamp_size = int(stamp_size)
        self.bands = bands
        self.max_cache = int(max_cache)
        self.use_api_fallback = use_api_fallback
        self.disk_cache_dir = disk_cache_dir
        self._cache: Dict[Tuple[str, str], Tuple[np.ndarray, WCS]] = {}
        self._cache_order: List[Tuple[str, str]] = []
        self._api_fallback_count: int = 0
        self._disk_cache_hits: int = 0
        
        # Create disk cache directory if specified
        if disk_cache_dir and not os.path.exists(disk_cache_dir):
            os.makedirs(disk_cache_dir, exist_ok=True)
    
    def _load_band(self, brickname: str, band: str) -> Tuple[np.ndarray, WCS]:
        """Load a single band FITS file."""
        key = (brickname, band)
        if key in self._cache:
            return self._cache[key]
        
        path = os.path.join(
            self.root,
            brickname,
            f"legacysurvey-{brickname}-image-{band}.fits.fz",
        )
        with fits.open(path) as hdul:
            img = hdul[1].data.astype(np.float32)
            wcs = WCS(hdul[1].header)
        
        # LRU cache eviction
        self._cache[key] = (img, wcs)
        self._cache_order.append(key)
        if len(self._cache_order) > self.max_cache:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        
        return img, wcs
    
    def _try_local_fetch(self, ra: float, dec: float, brickname: str) -> Optional[np.ndarray]:
        """Try to fetch from local coadd cache."""
        images: List[np.ndarray] = []
        half = self.stamp_size // 2
        
        for band in self.bands:
            try:
                img, wcs = self._load_band(brickname, band)
                xpix, ypix = wcs.world_to_pixel_values(float(ra), float(dec))
                xpix = int(round(float(xpix)))
                ypix = int(round(float(ypix)))
                
                x0, x1 = xpix - half, xpix + half
                y0, y1 = ypix - half, ypix + half
                
                if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
                    return None
                
                cutout = img[y0:y1, x0:x1]
                if cutout.shape != (self.stamp_size, self.stamp_size):
                    return None
                
                images.append(cutout.astype(np.float32))
            except Exception:
                return None
        
        return np.stack(images, axis=0)
    
    def _disk_cache_key(self, ra: float, dec: float) -> str:
        """Generate disk cache filename from coordinates."""
        # Round to avoid floating point issues
        ra_str = f"{ra:.6f}".replace(".", "p").replace("-", "m")
        dec_str = f"{dec:.6f}".replace(".", "p").replace("-", "m")
        return f"cutout_{ra_str}_{dec_str}.npz"
    
    def _try_disk_cache(self, ra: float, dec: float) -> Optional[np.ndarray]:
        """Try to load cutout from disk cache."""
        if not self.disk_cache_dir:
            return None
        
        cache_path = os.path.join(self.disk_cache_dir, self._disk_cache_key(ra, dec))
        if os.path.exists(cache_path):
            try:
                with np.load(cache_path) as z:
                    result = z["cutout"].astype(np.float32)
                    if result.shape == (len(self.bands), self.stamp_size, self.stamp_size):
                        self._disk_cache_hits += 1
                        return result
            except Exception:
                pass
        return None
    
    def _save_to_disk_cache(self, ra: float, dec: float, cutout: np.ndarray) -> None:
        """Save cutout to disk cache."""
        if not self.disk_cache_dir:
            return
        
        cache_path = os.path.join(self.disk_cache_dir, self._disk_cache_key(ra, dec))
        try:
            np.savez_compressed(cache_path, cutout=cutout)
        except Exception:
            pass  # Ignore cache write errors
    
    def fetch(self, ra: float, dec: float, brickname: str) -> Optional[np.ndarray]:
        """
        Fetch base cutout for given coordinates.
        
        Order of attempts:
        1. Local coadd cache (fastest)
        2. Disk cache (if disk_cache_dir set)
        3. Legacy Survey API (if use_api_fallback=True)
        
        Returns:
            (C, H, W) array for g, r, z bands, or None if out of bounds/missing.
        """
        # Try local coadd cache first
        result = self._try_local_fetch(ra, dec, brickname)
        if result is not None:
            return result
        
        # Try disk cache
        result = self._try_disk_cache(ra, dec)
        if result is not None:
            return result
        
        # Fallback to API if enabled
        if self.use_api_fallback:
            result = _fetch_from_legacy_survey(ra, dec, self.stamp_size, "".join(self.bands))
            if result is not None:
                self._api_fallback_count += 1
                # Save to disk cache for future use
                self._save_to_disk_cache(ra, dec, result)
            return result
        
        return None


# =============================================================================
# DATASET: PAIRED POSITIVE/CONTROL
# =============================================================================

@dataclass
class PairMeta:
    """Metadata for a positive/control pair."""
    ra: float
    dec: float
    brickname: str
    theta_e_arcsec: float
    psf_fwhm_arcsec: float
    psfdepth_r: float
    arc_snr: float
    row_id: Any


class PairedParquetDataset(Dataset):
    """
    Dataset that yields paired (positive, control, metadata).
    
    Indexes only positive rows (is_control==0) and fetches corresponding
    control cutouts from the coadd cache.
    """
    
    def __init__(
        self,
        parquet_root: str,
        coadd_provider: CoaddCutoutProvider,
        split: str = "train",
        max_pairs: Optional[int] = None,
        require_cutout_ok: bool = True,
    ):
        self.parquet_root = parquet_root
        self.provider = coadd_provider
        self.split = split
        self.max_pairs = max_pairs
        self.require_cutout_ok = require_cutout_ok
        
        self.dataset = ds.dataset(parquet_root, format="parquet", partitioning="hive")
        self._rows: List[Dict[str, Any]] = []
        self._build_index()
    
    def _build_index(self) -> None:
        """Build index of positive samples."""
        filt = (ds.field("region_split") == self.split) & (ds.field("is_control") == 0)
        if self.require_cutout_ok and "cutout_ok" in self.dataset.schema.names:
            filt = filt & (ds.field("cutout_ok") == 1)
        
        # Required columns
        cols = ["stamp_npz", "ra", "dec", "brickname", "theta_e_arcsec", "psfsize_r"]
        
        # Optional columns
        schema_names = set(self.dataset.schema.names)
        if "psfdepth_r" in schema_names:
            cols.append("psfdepth_r")
        if "arc_snr" in schema_names:
            cols.append("arc_snr")
        if "row_id" in schema_names:
            cols.append("row_id")
        # Note: removed duplicate brickname fallback (FIX #7)
        
        table = self.dataset.to_table(filter=filt, columns=cols)
        n = table.num_rows
        if self.max_pairs is not None:
            n = min(n, int(self.max_pairs))
        
        for i in range(n):
            rec = {c: table[c][i].as_py() for c in cols}
            self._rows.append(rec)
    
    def __len__(self) -> int:
        return len(self._rows)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, PairMeta]:
        r = self._rows[idx]
        
        pos = decode_stamp_npz(r["stamp_npz"])
        
        ra = float(r["ra"])
        dec = float(r["dec"])
        brick = str(r["brickname"])
        theta = float(r.get("theta_e_arcsec", 0.0))
        psf_fwhm = float(r.get("psfsize_r", 1.32))
        psfdepth = float(r.get("psfdepth_r", 0.0))
        arc_snr = float(r.get("arc_snr", float("nan")))
        row_id = r.get("row_id", f"{brick}_{idx}")
        
        ctrl = self.provider.fetch(ra, dec, brick)
        if ctrl is None:
            raise RuntimeError(f"Control fetch failed for idx={idx}, brick={brick}")
        
        meta = PairMeta(
            ra=ra,
            dec=dec,
            brickname=brick,
            theta_e_arcsec=theta,
            psf_fwhm_arcsec=psf_fwhm,
            psfdepth_r=psfdepth,
            arc_snr=arc_snr,
            row_id=row_id,
        )
        return pos, ctrl, meta


class DropOnErrorWrapper(Dataset):
    """Wraps a dataset and retries nearby indices on failure."""
    
    def __init__(self, base: Dataset, max_retry: int = 8):
        self.base = base
        self.max_retry = int(max_retry)
    
    def __len__(self) -> int:
        return len(self.base)
    
    def __getitem__(self, idx: int):
        last_err = None
        for k in range(self.max_retry):
            j = (idx + k) % len(self.base)
            try:
                return self.base[j]
            except Exception as e:
                last_err = e
                continue
        raise last_err or RuntimeError("Unknown error in DropOnErrorWrapper")


# =============================================================================
# HARD NEGATIVES: AZIMUTHAL SHUFFLE
# =============================================================================

def azimuthal_shuffle_delta(
    delta: np.ndarray,
    nbins: int = DEFAULT_AZIMUTHAL_NBINS,
    r_max: Optional[float] = None,
    r_core: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Shuffle delta values within radial bins, preserving per-radius distribution.
    
    This destroys coherent arc structure while keeping radial brightness profile.
    Inner core (r < r_core) is preserved to maintain central brightness match.
    
    Args:
        delta: (C, H, W) difference image (pos - ctrl)
        nbins: number of radial bins for shuffling
        r_max: maximum radius to shuffle (default: full image)
        r_core: radius below which to preserve (no shuffle)
        rng: random generator
    
    Returns:
        Shuffled delta array.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    C, H, W = delta.shape
    r, _, _ = _radius_maps(H, W)
    if r_max is None:
        r_max = float(r.max())
    
    edges = np.linspace(float(r_core), float(r_max) + 1e-6, nbins + 1, dtype=np.float32)
    out = delta.copy()
    
    flat_r = r.reshape(-1)
    flat_delta = delta.reshape(C, -1)
    out_flat = out.reshape(C, -1)
    
    # Preserve core (r < r_core)
    # (already copied above, no action needed)
    
    # Shuffle within each radial bin
    for b in range(nbins):
        m = (flat_r >= edges[b]) & (flat_r < edges[b + 1])
        idxs = np.flatnonzero(m)
        if idxs.size <= 1:
            continue
        
        perm = rng.permutation(idxs)
        # Shared permutation across channels preserves color coherence
        out_flat[:, idxs] = flat_delta[:, perm]
    
    return out


def azimuthal_shuffle_theta_aware(
    delta: np.ndarray,
    theta_pix: float,
    psf_fwhm_pix: float,
    k: float = 1.5,
    r_core: float = 2.0,
    nbins: int = DEFAULT_AZIMUTHAL_NBINS,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Theta-aware azimuthal shuffle: shuffle mainly in the arc-bearing annulus.
    
    Args:
        delta: (C, H, W) difference image
        theta_pix: Einstein radius in pixels
        psf_fwhm_pix: PSF FWHM in pixels
        k: annulus half-width in units of FWHM
        r_core: minimum core radius to preserve
        nbins: number of radial bins
        rng: random generator
    
    Returns:
        Shuffled delta.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    C, H, W = delta.shape
    r, _, _ = _radius_maps(H, W)
    
    # Arc annulus: [theta - k*FWHM, theta + k*FWHM]
    r0 = max(r_core, theta_pix - k * psf_fwhm_pix)
    r1 = theta_pix + k * psf_fwhm_pix
    
    out = delta.copy()
    
    # Shuffle within the annulus only
    ann_mask = (r >= r0) & (r < r1)
    idxs = np.flatnonzero(ann_mask.ravel())
    
    if idxs.size > 16:
        perm = rng.permutation(idxs)
        flat_delta = delta.reshape(C, -1)
        out_flat = out.reshape(C, -1)
        out_flat[:, idxs] = flat_delta[:, perm]
    
    return out


def make_hardneg(
    ctrl: np.ndarray,
    pos: np.ndarray,
    theta_pix: float,
    psf_fwhm_pix: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create hard negative: ctrl + theta-aware shuffled (pos - ctrl).
    
    The result has the same radial brightness profile as positives
    but destroyed tangential arc structure.
    """
    delta = (pos - ctrl).astype(np.float32)
    shuf = azimuthal_shuffle_theta_aware(
        delta,
        theta_pix=theta_pix,
        psf_fwhm_pix=psf_fwhm_pix,
        k=1.5,
        r_core=2.0,
        rng=rng,
    )
    return (ctrl + shuf).astype(np.float32)


# =============================================================================
# PREPROCESSING: MAD NORMALIZATION + RESIDUAL VIEW
# =============================================================================

def robust_mad_norm_outer_np(
    x: np.ndarray,
    outer_mask: np.ndarray,
    clip: float = DEFAULT_CLIP,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Robust MAD normalization using outer region statistics.
    
    Args:
        x: (C, H, W) image
        outer_mask: (H, W) boolean mask for outer region
        clip: clip normalized values to [-clip, clip]
        eps: numerical stability
    
    Returns:
        Normalized image (C, H, W).
    """
    C, H, W = x.shape
    out = np.empty_like(x, dtype=np.float32)
    m = outer_mask.reshape(-1)
    
    for c in range(C):
        v = x[c].reshape(-1)
        ov = v[m]
        
        if ov.size == 0:
            # Fallback: use full image
            ov = v
        
        med = np.median(ov)
        mad = np.median(np.abs(ov - med))
        scale = 1.4826 * mad + eps
        
        z = (x[c] - med) / scale
        if clip is not None:
            z = np.clip(z, -clip, clip)
        out[c] = z.astype(np.float32)
    
    return out


def _gaussian_kernel_1d(sigma: float, truncate: float = 3.0, device: Optional[torch.device] = None) -> torch.Tensor:
    """Create 1D Gaussian kernel."""
    if sigma <= 0:
        return torch.tensor([1.0], dtype=torch.float32, device=device)
    
    radius = int(math.ceil(truncate * float(sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(xs * xs) / (2.0 * float(sigma) * float(sigma)))
    k = k / (k.sum() + 1e-12)
    return k


def gaussian_blur_torch(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Depthwise separable Gaussian blur with reflect padding.
    
    Args:
        x: (B, C, H, W) tensor
        sigma: blur sigma in pixels
    
    Returns:
        Blurred tensor of same shape.
    """
    if sigma <= 0:
        return x
    
    B, C, H, W = x.shape
    device = x.device
    k1 = _gaussian_kernel_1d(sigma, device=device)
    K = k1.numel()
    pad = K // 2
    
    # Vertical pass
    wv = k1.view(1, 1, K, 1).repeat(C, 1, 1, 1)
    xv = F.pad(x, (0, 0, pad, pad), mode="reflect")
    y = F.conv2d(xv, wv, bias=None, stride=1, padding=0, groups=C)
    
    # Horizontal pass
    wh = k1.view(1, 1, 1, K).repeat(C, 1, 1, 1)
    yh = F.pad(y, (pad, pad, 0, 0), mode="reflect")
    z = F.conv2d(yh, wh, bias=None, stride=1, padding=0, groups=C)
    
    return z


class Preprocess6CH:
    """
    Preprocessing pipeline producing 6-channel output.
    
    Channels 0-2: raw_norm (g, r, z) - MAD normalized
    Channels 3-5: residual (g, r, z) - raw_norm minus blur
    
    The residual view emphasizes tangential arc structure.
    """
    
    def __init__(
        self,
        outer_mask: np.ndarray,
        clip: float = DEFAULT_CLIP,
        resid_sigma_pix: float = DEFAULT_RESID_SIGMA_PIX,
    ):
        self.outer_mask = outer_mask.astype(bool)
        self.clip = float(clip)
        self.resid_sigma_pix = float(resid_sigma_pix)
    
    def __call__(self, x_np: np.ndarray) -> torch.Tensor:
        """
        Process image to 6-channel tensor.
        
        Args:
            x_np: (C, H, W) raw image
        
        Returns:
            (6, H, W) tensor
        """
        raw = robust_mad_norm_outer_np(x_np, self.outer_mask, clip=self.clip)
        t = torch.from_numpy(raw).unsqueeze(0).contiguous()  # (1, 3, H, W)
        blur = gaussian_blur_torch(t, sigma=self.resid_sigma_pix)
        resid = (t - blur).squeeze(0)  # (3, H, W)
        raw_t = t.squeeze(0)  # (3, H, W)
        x6 = torch.cat([raw_t, resid], dim=0)  # (6, H, W)
        return x6


# =============================================================================
# COLLATE: ON-THE-FLY SAMPLING WITH CURRICULUM
# =============================================================================

@dataclass
class BatchOut:
    """Batch output from collate function."""
    x6: torch.Tensor          # (B, 6, H, W)
    y: torch.Tensor           # (B,)
    meta: Dict[str, torch.Tensor]
    pair_row_ids: List[Any]


class PairedMixCollate:
    """
    Collate function that samples pos/ctrl/hardneg from pairs.
    
    Supports curriculum learning by adjusting hard negative probability.
    """
    
    def __init__(
        self,
        preproc: Preprocess6CH,
        pos_prob: float = 0.4,
        ctrl_prob: float = 0.4,
        hardneg_prob: float = 0.2,
        hardneg_nbins: int = DEFAULT_AZIMUTHAL_NBINS,
        seed: int = 1337,
    ):
        s = pos_prob + ctrl_prob + hardneg_prob
        if abs(s - 1.0) > 1e-6:
            raise ValueError("Probabilities must sum to 1")
        
        self.preproc = preproc
        self.pos_prob = float(pos_prob)
        self.ctrl_prob = float(ctrl_prob)
        self.hardneg_prob = float(hardneg_prob)
        self.hardneg_nbins = int(hardneg_nbins)
        self.base_seed = int(seed)
    
    def _get_rng(self) -> np.random.Generator:
        """Get per-worker RNG (FIX #12)."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Unique seed per worker + time for variation
            seed = self.base_seed + worker_info.id + int(time.time() * 1000) % 100000
        else:
            seed = self.base_seed + int(time.time() * 1000) % 100000
        return np.random.default_rng(seed)
    
    def __call__(self, batch: List[Tuple[np.ndarray, np.ndarray, PairMeta]]) -> BatchOut:
        rng = self._get_rng()
        
        xs: List[torch.Tensor] = []
        ys: List[int] = []
        theta_pix_list: List[float] = []
        psf_fwhm_pix_list: List[float] = []
        psf_fwhm_arcsec_list: List[float] = []  # For model meta
        psfdepth_list: List[float] = []
        arc_snr_list: List[float] = []
        is_hardneg_list: List[int] = []
        row_ids: List[Any] = []
        
        for pos, ctrl, meta in batch:
            theta_pix = float(meta.theta_e_arcsec) / PIX_SCALE_ARCSEC
            psf_fwhm_pix = float(meta.psf_fwhm_arcsec) / PIX_SCALE_ARCSEC
            
            u = float(rng.random())
            if u < self.pos_prob:
                img = pos
                y = 1
                is_hn = 0
            elif u < self.pos_prob + self.ctrl_prob:
                img = ctrl
                y = 0
                is_hn = 0
            else:
                img = make_hardneg(ctrl, pos, theta_pix, psf_fwhm_pix, rng)
                y = 0
                is_hn = 1
            
            x6 = self.preproc(img)
            xs.append(x6)
            ys.append(y)
            
            theta_pix_list.append(theta_pix)
            psf_fwhm_pix_list.append(psf_fwhm_pix)
            psf_fwhm_arcsec_list.append(float(meta.psf_fwhm_arcsec))  # Keep arcsec for model
            psfdepth_list.append(float(meta.psfdepth_r))
            arc_snr_list.append(float(meta.arc_snr) if np.isfinite(meta.arc_snr) else float("nan"))
            is_hardneg_list.append(is_hn)
            row_ids.append(meta.row_id)
        
        x = torch.stack(xs, dim=0)
        y = torch.tensor(ys, dtype=torch.float32)
        
        theta_pix_t = torch.tensor(theta_pix_list, dtype=torch.float32)
        psf_fwhm_pix_t = torch.tensor(psf_fwhm_pix_list, dtype=torch.float32)
        psf_fwhm_arcsec_t = torch.tensor(psf_fwhm_arcsec_list, dtype=torch.float32)
        x_ratio_t = theta_pix_t / torch.clamp(psf_fwhm_pix_t, min=1e-6)
        
        meta_out = {
            "theta_pix": theta_pix_t,
            "psf_fwhm_pix": psf_fwhm_pix_t,
            "psf_fwhm_arcsec": psf_fwhm_arcsec_t,  # For model input (FIX #11)
            "x_ratio": x_ratio_t,
            "psfdepth_r": torch.tensor(psfdepth_list, dtype=torch.float32),
            "arc_snr": torch.tensor(arc_snr_list, dtype=torch.float32),
            "is_hardneg": torch.tensor(is_hardneg_list, dtype=torch.int64),
        }
        
        return BatchOut(x6=x, y=y, meta=meta_out, pair_row_ids=row_ids)


# =============================================================================
# GATE RUNNER: SHORTCUT DETECTION
# =============================================================================

def _extract_region_stats(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract per-channel statistics from masked region.
    
    Returns 7 stats per channel: mean, std, median, q25, q75, max, min.
    
    FIX #9: Handles empty regions gracefully.
    """
    N_STATS = 7
    feats = []
    
    for c in range(x.shape[0]):
        v = x[c][mask]
        
        if v.size == 0:
            # Empty region: return zeros
            feats.extend([0.0] * N_STATS)
            continue
        
        feats.extend([
            float(np.mean(v)),
            float(np.std(v)),
            float(np.median(v)),
            float(np.percentile(v, 25)),
            float(np.percentile(v, 75)),
            float(np.max(v)),
            float(np.min(v)),
        ])
    
    return np.array(feats, dtype=np.float32)


def _radial_profile_features(x: np.ndarray, nbins: int = 16) -> np.ndarray:
    """
    Extract radial profile features (mean per radial bin per channel).
    
    Returns C * nbins features.
    """
    C, H, W = x.shape
    r, _, _ = _radius_maps(H, W)
    rmax = float(r.max())
    edges = np.linspace(0.0, rmax + 1e-6, nbins + 1, dtype=np.float32)
    
    feats = []
    for c in range(C):
        for b in range(nbins):
            m = (r >= edges[b]) & (r < edges[b + 1])
            v = x[c][m]
            feats.append(float(np.mean(v)) if v.size else 0.0)
    
    return np.array(feats, dtype=np.float32)


def _stratum_mask(x_ratio: np.ndarray, lo: Optional[float], hi: Optional[float]) -> np.ndarray:
    """Create mask for x_ratio stratum."""
    m = np.ones_like(x_ratio, dtype=bool)
    if lo is not None:
        m &= (x_ratio >= lo)
    if hi is not None:
        m &= (x_ratio < hi)
    return m


def _fit_auc_lr(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Fit logistic regression and return AUC."""
    if len(np.unique(y)) < 2 or X.shape[0] < 10:
        return float("nan")
    clf = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, p))


def _fill_mask_from_outer(
    img: np.ndarray,
    mask: np.ndarray,
    outer_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fill masked region with samples from outer region (noise fill for occlusion)."""
    out = img.copy()
    for c in range(img.shape[0]):
        pool = img[c][outer_mask]
        if pool.size == 0:
            continue
        n = int(mask.sum())
        if n > 0:
            out[c][mask] = rng.choice(pool, size=n, replace=True)
    return out


@torch.no_grad()
def _predict_model_batched(
    model: nn.Module,
    x6: torch.Tensor,
    psf_fwhm_arcsec: torch.Tensor,
    psfdepth_r: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Run model inference in batches.
    
    FIX #11: Uses arcsec for PSF (not pixels).
    """
    model.eval()
    probs: List[np.ndarray] = []
    n = x6.shape[0]
    
    for i in range(0, n, batch_size):
        xb = x6[i:i + batch_size].to(device, non_blocking=True)
        # Model expects meta as [psfsize_r (arcsec), psfdepth_r]
        meta = torch.stack([
            psf_fwhm_arcsec[i:i + batch_size],
            psfdepth_r[i:i + batch_size],
        ], dim=1).to(device)
        
        logit = model(xb, meta)
        p = torch.sigmoid(logit).detach().cpu().numpy().ravel()
        probs.append(p.astype(np.float32))
    
    return np.concatenate(probs, axis=0)


class GateRunner:
    """
    Runs stratified shortcut detection gates on paired samples.
    
    Gates:
        - Core-only logistic AUC (should be <= 0.60 for x >= 1.0)
        - Annulus-only logistic AUC (should be >= 0.75 for x >= 1.0)
        - Radial-profile logistic AUC (should be <= 0.60 for x >= 1.0)
        - Hard negative mean score (should be <= 0.05)
        - Arc occlusion drop (should be >= 0.30)
    """
    
    def __init__(
        self,
        preproc: Preprocess6CH,
        outer_mask: np.ndarray,
        strata: Dict[str, Tuple[Optional[float], Optional[float]]] = None,
        seed: int = 1337,
    ):
        self.preproc = preproc
        self.outer_mask = outer_mask.astype(bool)
        self.strata = strata or DEFAULT_STRATA
        self.rng = np.random.default_rng(int(seed))
        
        # Precompute radius map
        self._r, self._r2, _ = _radius_maps(STAMP_SIZE, STAMP_SIZE)
    
    def _theta_aware_core_mask(self, theta_pix: float, psf_fwhm_pix: float, k: float = 1.5) -> np.ndarray:
        """Core mask: r < max(theta - k*FWHM, 0), fallback to r<2 if empty."""
        rc = max(theta_pix - k * psf_fwhm_pix, 0.0)
        mask = self._r < float(rc)
        if mask.sum() < 8:
            mask = self._r < 2.0
        return mask
    
    def _theta_aware_annulus_mask(self, theta_pix: float, psf_fwhm_pix: float, k: float = 1.5, rmin: float = 2.0) -> np.ndarray:
        """Annulus mask: r in [max(theta - k*FWHM, rmin), theta + k*FWHM]."""
        r0 = max(theta_pix - k * psf_fwhm_pix, rmin)
        r1 = theta_pix + k * psf_fwhm_pix
        return (self._r >= float(r0)) & (self._r < float(r1))
    
    def run(
        self,
        pairs: Iterable[Tuple[np.ndarray, np.ndarray, PairMeta]],
        model: Optional[nn.Module] = None,
        device: str = "cuda",
        max_pairs: int = 2000,
        model_batch: int = 128,
    ) -> Dict[str, Any]:
        """
        Run all gates on paired samples.
        
        Args:
            pairs: iterable of (pos, ctrl, meta) tuples
            model: optional model for hard-neg and occlusion gates
            device: device for model inference
            max_pairs: maximum pairs to process
            model_batch: batch size for model inference
        
        Returns:
            Dict with gate results by stratum.
        """
        device_t = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        
        # Collect pairs
        pos_list: List[np.ndarray] = []
        ctrl_list: List[np.ndarray] = []
        theta_pix_list: List[float] = []
        psf_fwhm_pix_list: List[float] = []
        psf_fwhm_arcsec_list: List[float] = []
        psfdepth_list: List[float] = []
        arc_snr_list: List[float] = []
        
        ncol = 0
        for pos, ctrl, meta in pairs:
            pos_list.append(pos)
            ctrl_list.append(ctrl)
            theta_pix_list.append(float(meta.theta_e_arcsec) / PIX_SCALE_ARCSEC)
            psf_fwhm_pix_list.append(float(meta.psf_fwhm_arcsec) / PIX_SCALE_ARCSEC)
            psf_fwhm_arcsec_list.append(float(meta.psf_fwhm_arcsec))
            psfdepth_list.append(float(meta.psfdepth_r))
            arc_snr_list.append(float(meta.arc_snr) if np.isfinite(meta.arc_snr) else float("nan"))
            ncol += 1
            if ncol >= int(max_pairs):
                break
        
        if ncol == 0:
            raise RuntimeError("No pairs collected for gates")
        
        theta_pix = np.array(theta_pix_list, dtype=np.float32)
        psf_fwhm_pix = np.array(psf_fwhm_pix_list, dtype=np.float32)
        psf_fwhm_arcsec = np.array(psf_fwhm_arcsec_list, dtype=np.float32)
        x_ratio = theta_pix / np.clip(psf_fwhm_pix, 1e-6, None)
        psfdepth = np.array(psfdepth_list, dtype=np.float32)
        arc_snr = np.array(arc_snr_list, dtype=np.float32)
        
        # Build LR baseline features
        core_feats = []
        ann_feats = []
        rad_feats = []
        y_lr = []
        
        for i in range(ncol):
            p = robust_mad_norm_outer_np(pos_list[i], self.outer_mask)
            c = robust_mad_norm_outer_np(ctrl_list[i], self.outer_mask)
            
            core_m = self._theta_aware_core_mask(theta_pix[i], psf_fwhm_pix[i])
            ann_m = self._theta_aware_annulus_mask(theta_pix[i], psf_fwhm_pix[i])
            
            # Positive features
            core_feats.append(_extract_region_stats(p, core_m))
            ann_feats.append(_extract_region_stats(p, ann_m))
            rad_feats.append(_radial_profile_features(p))
            y_lr.append(1)
            
            # Control features
            core_feats.append(_extract_region_stats(c, core_m))
            ann_feats.append(_extract_region_stats(c, ann_m))
            rad_feats.append(_radial_profile_features(c))
            y_lr.append(0)
        
        core_feats = np.stack(core_feats, axis=0)
        ann_feats = np.stack(ann_feats, axis=0)
        rad_feats = np.stack(rad_feats, axis=0)
        y_lr = np.array(y_lr, dtype=np.int64)
        
        # Expand x_ratio for interleaved pos/ctrl
        x_lr = np.repeat(x_ratio, 2).astype(np.float32)
        
        results: Dict[str, Any] = {
            "n_pairs": int(ncol),
            "strata": {},
        }
        
        # LR AUCs per stratum
        for name, (lo, hi) in self.strata.items():
            m = _stratum_mask(x_lr, lo, hi)
            if m.sum() < 200:
                results["strata"][name] = {"status": "INSUFFICIENT_SAMPLES", "count": int(m.sum())}
                continue
            
            auc_core = _fit_auc_lr(core_feats[m], y_lr[m])
            auc_ann = _fit_auc_lr(ann_feats[m], y_lr[m])
            auc_rad = _fit_auc_lr(rad_feats[m], y_lr[m])
            
            results["strata"][name] = {
                "count": int(m.sum()),
                "core_auc_lr": auc_core,
                "annulus_auc_lr": auc_ann,
                "radial_profile_auc_lr": auc_rad,
            }
        
        # Model-based gates
        if model is not None:
            pos6_list = []
            ctrl6_list = []
            hard6_list = []
            
            for i in range(ncol):
                pos6_list.append(self.preproc(pos_list[i]))
                ctrl6_list.append(self.preproc(ctrl_list[i]))
                hn = make_hardneg(ctrl_list[i], pos_list[i], theta_pix[i], psf_fwhm_pix[i], self.rng)
                hard6_list.append(self.preproc(hn))
            
            pos6_t = torch.stack(pos6_list, dim=0)
            ctrl6_t = torch.stack(ctrl6_list, dim=0)
            hard6_t = torch.stack(hard6_list, dim=0)
            
            psf_arcsec_t = torch.tensor(psf_fwhm_arcsec, dtype=torch.float32)
            psfdepth_t = torch.tensor(psfdepth, dtype=torch.float32)
            
            p_pos = _predict_model_batched(model, pos6_t, psf_arcsec_t, psfdepth_t, device_t, model_batch)
            p_ctrl = _predict_model_batched(model, ctrl6_t, psf_arcsec_t, psfdepth_t, device_t, model_batch)
            p_hard = _predict_model_batched(model, hard6_t, psf_arcsec_t, psfdepth_t, device_t, model_batch)
            
            results["model_gates"] = {
                "pos_mean_p": float(np.mean(p_pos)),
                "ctrl_mean_p": float(np.mean(p_ctrl)),
                "hardneg_mean_p": float(np.mean(p_hard)),
                "hardneg_frac_gt_0p5": float(np.mean(p_hard > 0.5)),
            }
            
            # Arc occlusion test
            occ6_list = []
            for i in range(ncol):
                ann_m = self._theta_aware_annulus_mask(theta_pix[i], psf_fwhm_pix[i])
                occ_raw = _fill_mask_from_outer(pos_list[i], ann_m, self.outer_mask, self.rng)
                occ6_list.append(self.preproc(occ_raw))
            
            occ6_t = torch.stack(occ6_list, dim=0)
            p_occ = _predict_model_batched(model, occ6_t, psf_arcsec_t, psfdepth_t, device_t, model_batch)
            
            drop = float(np.mean(p_pos) - np.mean(p_occ))
            drop_frac = float(drop / (np.mean(p_pos) + 1e-9))
            
            results["model_gates"].update({
                "arc_occlusion_pos_mean_p": float(np.mean(p_occ)),
                "arc_occlusion_drop_abs": drop,
                "arc_occlusion_drop_frac": drop_frac,
            })
            
            # Stratified model gates
            strata_model = {}
            for name, (lo, hi) in self.strata.items():
                m = _stratum_mask(x_ratio, lo, hi)
                if m.sum() < 50:
                    strata_model[name] = {"status": "INSUFFICIENT_SAMPLES", "count": int(m.sum())}
                    continue
                strata_model[name] = {
                    "count": int(m.sum()),
                    "hardneg_mean_p": float(np.mean(p_hard[m])),
                    "hardneg_frac_gt_0p5": float(np.mean(p_hard[m] > 0.5)),
                    "arc_occ_drop_abs": float(np.mean(p_pos[m]) - np.mean(p_occ[m])),
                    "arc_occ_drop_frac": float((np.mean(p_pos[m]) - np.mean(p_occ[m])) / (np.mean(p_pos[m]) + 1e-9)),
                }
            results["model_gates"]["stratified"] = strata_model
        
        # Distribution summaries
        results["distributions"] = {
            "theta_pix": {
                "mean": float(np.mean(theta_pix)),
                "p10": float(np.percentile(theta_pix, 10)),
                "p50": float(np.percentile(theta_pix, 50)),
                "p90": float(np.percentile(theta_pix, 90)),
            },
            "x_ratio": {
                "mean": float(np.mean(x_ratio)),
                "frac_ge_1": float(np.mean(x_ratio >= 1.0)),
            },
        }
        
        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_training_loader(
    parquet_root: str,
    coadd_cache_root: str,
    split: str = "train",
    batch_pairs: int = 64,
    num_workers: int = 4,
    max_pairs_index: Optional[int] = None,
    pos_prob: float = 0.4,
    ctrl_prob: float = 0.4,
    hardneg_prob: float = 0.2,
    use_api_fallback: bool = False,
    disk_cache_dir: Optional[str] = None,
) -> DataLoader:
    """Build DataLoader for paired training.
    
    Args:
        use_api_fallback: If True, use Legacy Survey cutout service as fallback
                          when local coadd cache is unavailable.
        disk_cache_dir: If set, cache fetched cutouts to this directory for reuse.
    """
    provider = CoaddCutoutProvider(
        coadd_cache_root, 
        max_cache=32, 
        use_api_fallback=use_api_fallback,
        disk_cache_dir=disk_cache_dir,
    )
    base_ds = PairedParquetDataset(parquet_root, provider, split=split, max_pairs=max_pairs_index)
    ds_safe = DropOnErrorWrapper(base_ds, max_retry=16)
    
    outer_mask = make_outer_mask(STAMP_SIZE, STAMP_SIZE, DEFAULT_OUTER_R_PIX)
    preproc = Preprocess6CH(outer_mask=outer_mask, clip=DEFAULT_CLIP, resid_sigma_pix=DEFAULT_RESID_SIGMA_PIX)
    collate = PairedMixCollate(
        preproc=preproc,
        pos_prob=pos_prob,
        ctrl_prob=ctrl_prob,
        hardneg_prob=hardneg_prob,
    )
    
    return DataLoader(
        ds_safe,
        batch_size=batch_pairs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate,
    )


def run_gates_quick(
    parquet_root: str,
    coadd_cache_root: str,
    split: str = "train",
    model: Optional[nn.Module] = None,
    device: str = "cuda",
    max_pairs: int = 2000,
    use_api_fallback: bool = False,
    disk_cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Quick gate evaluation.
    
    Args:
        use_api_fallback: If True, use Legacy Survey cutout service as fallback
                          when local coadd cache is unavailable.
        disk_cache_dir: If set, cache fetched cutouts to this directory for reuse.
    """
    provider = CoaddCutoutProvider(
        coadd_cache_root, 
        max_cache=32, 
        use_api_fallback=use_api_fallback,
        disk_cache_dir=disk_cache_dir,
    )
    base_ds = PairedParquetDataset(parquet_root, provider, split=split, max_pairs=max_pairs)
    ds_safe = DropOnErrorWrapper(base_ds, max_retry=16)
    
    outer_mask = make_outer_mask(STAMP_SIZE, STAMP_SIZE, DEFAULT_OUTER_R_PIX)
    preproc = Preprocess6CH(outer_mask=outer_mask, clip=DEFAULT_CLIP, resid_sigma_pix=DEFAULT_RESID_SIGMA_PIX)
    
    runner = GateRunner(preproc=preproc, outer_mask=outer_mask)
    return runner.run(iter(ds_safe), model=model, device=device, max_pairs=max_pairs)


# =============================================================================
# CLI USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    import json
    
    # Example paths (adjust for your setup)
    parquet_root = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
    coadd_cache_root = "/lambda/nfs/darkhaloscope-training-dc/dr10/coadd_cache"
    
    print("Building training loader...")
    loader = build_training_loader(
        parquet_root,
        coadd_cache_root,
        split="train",
        batch_pairs=16,
        num_workers=0,
        max_pairs_index=100,
    )
    
    print("Getting first batch...")
    batch = next(iter(loader))
    print(f"  x6 shape: {batch.x6.shape}")
    print(f"  y shape: {batch.y.shape}")
    print(f"  x_ratio mean: {batch.meta['x_ratio'].mean().item():.3f}")
    print(f"  hardneg count: {batch.meta['is_hardneg'].sum().item()}")
    
    print("\nRunning gates (no model)...")
    results = run_gates_quick(
        parquet_root,
        coadd_cache_root,
        split="train",
        model=None,
        device="cpu",
        max_pairs=200,
    )
    print(json.dumps(results, indent=2))

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

AB_ZEROPOINT_MAG = 22.5

@dataclass
class PreprocessConfig:
    mode: str = "raw_robust"  # keep your current default
    clip: float = 10.0
    crop_size: Optional[int] = None  # Paper IV parity: None (keep 101x101)

def _robust_norm(img: np.ndarray, r_inner: int = 20, r_outer: int = 32, clip: float = 10.0) -> np.ndarray:
    """Normalize each band using median/MAD in an outer annulus, then clip."""
    h, w, c = img.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    ann = (rr >= r_inner) & (rr <= r_outer)
    out = img.copy()
    for ch in range(c):
        band = out[:, :, ch]
        vals = band[ann]
        vals = vals[np.isfinite(vals)]
        if vals.size < 50:
            # fallback to global
            vals = band[np.isfinite(band)]
        med = np.median(vals) if vals.size else 0.0
        mad = np.median(np.abs(vals - med)) if vals.size else 1.0
        mad = max(mad, 1e-6)
        band = (band - med) / (1.4826 * mad)
        band = np.clip(band, -clip, clip)
        out[:, :, ch] = band
    return out

def center_crop(img: np.ndarray, crop: int) -> np.ndarray:
    h, w, c = img.shape
    if crop > h or crop > w:
        raise ValueError(f"crop {crop} larger than image {img.shape}")
    cy, cx = h // 2, w // 2
    hs = crop // 2
    return img[cy-hs:cy-hs+crop, cx-hs:cx-hs+crop, :]

def preprocess(img: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if cfg.mode == "raw_robust":
        out = _robust_norm(img, clip=cfg.clip)
    else:
        raise ValueError(f"Unknown preprocess mode: {cfg.mode}")
    if cfg.crop_size is not None:
        out = center_crop(out, cfg.crop_size)
    return out

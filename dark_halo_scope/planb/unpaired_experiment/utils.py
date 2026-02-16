"""Utility functions for unpaired experiment."""
from __future__ import annotations
import io
import numpy as np
from typing import Tuple

from .constants import STAMP_SIZE


def decode_npz_blob(blob: bytes) -> dict[str, np.ndarray]:
    """Decode npz blob to dict of arrays."""
    with np.load(io.BytesIO(blob)) as z:
        return {k: z[k] for k in z.files}


def radial_rmap(H: int = STAMP_SIZE, W: int = STAMP_SIZE) -> np.ndarray:
    """Create radial distance map from center."""
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2)


def radial_mask(H: int, W: int, r_in: float, r_out: float) -> np.ndarray:
    """Create annular mask."""
    r = radial_rmap(H, W)
    return (r >= r_in) & (r < r_out)


def center_slices(H: int, W: int, box: int) -> Tuple[slice, slice]:
    """Get slices for central box."""
    cy, cx = H // 2, W // 2
    half = box // 2
    return slice(cy - half, cy + half), slice(cx - half, cx + half)


def robust_median_mad(x: np.ndarray, eps: float = 1e-8) -> Tuple[float, float]:
    """Compute median and MAD (median absolute deviation), ignoring NaN."""
    valid = x[~np.isnan(x)]
    if len(valid) == 0:
        return 0.0, 1.0
    med = float(np.median(valid))
    mad = float(np.median(np.abs(valid - med))) + eps
    return med, mad


def normalize_outer_annulus(img: np.ndarray, r_in: float = 20, r_out: float = 32) -> np.ndarray:
    """Normalize image by outer annulus median/MAD, handling NaN."""
    H, W = img.shape
    m = radial_mask(H, W, r_in, r_out)
    med, mad = robust_median_mad(img[m])
    result = (img - med) / mad
    # Replace any remaining NaN with 0
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def azimuthal_median_profile(img: np.ndarray, r_max: int = 32) -> np.ndarray:
    """Compute azimuthally averaged median profile, handling NaN."""
    H, W = img.shape
    r = radial_rmap(H, W)
    prof = np.zeros(r_max, dtype=np.float32)
    for ri in range(r_max):
        m = (r >= ri) & (r < ri + 1)
        if np.any(m):
            vals = img[m]
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                prof[ri] = np.median(valid)
    return prof


def radial_profile_model(img: np.ndarray, r_max: int = 32) -> np.ndarray:
    """Build radial profile model (for residual subtraction)."""
    H, W = img.shape
    r = radial_rmap(H, W)
    prof = azimuthal_median_profile(img, r_max=r_max)
    rr = np.clip(np.floor(r).astype(int), 0, r_max - 1)
    return prof[rr].astype(np.float32)

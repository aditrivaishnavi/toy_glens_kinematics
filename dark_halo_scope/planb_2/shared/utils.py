"""
Shared utility functions for Plan B codebase.

All phases MUST use these functions instead of reimplementing.
This ensures consistency across the codebase.

Lessons Learned:
- L1.2: Duplicate function definitions caused bugs
- L9.1: Data format mismatches between assumed and actual
"""
import io
import gzip
from typing import Dict, Tuple, Optional

import numpy as np

from .constants import (
    STAMP_SIZE, NUM_CHANNELS, OUTER_RADIUS_PIX, CLIP_SIGMA,
    MAD_TO_STD, CORE_RADIUS_PIX, VALUE_RANGE_MIN, VALUE_RANGE_MAX,
    AZIMUTHAL_SHUFFLE_BINS,
)
from .schema import STAMP_SCHEMA


# =============================================================================
# NPZ DECODING
# =============================================================================
# SINGLE implementation - do NOT duplicate elsewhere

def decode_stamp_npz(blob: bytes) -> Tuple[np.ndarray, str]:
    """
    Decode stamp NPZ blob to (C, H, W) array and bandset string.
    
    This is the ONLY implementation of NPZ decoding.
    Do NOT reimplement in other modules.
    
    Handles:
    - Multi-band format: image_g, image_r, image_z keys
    - Legacy single-key format: 'img' key
    - Fallback: first available key
    - Gzip compression
    
    Args:
        blob: Raw bytes from parquet column
    
    Returns:
        Tuple of (image array shape (C,H,W), bandset string like "grz")
    
    Raises:
        ValueError: If decoding fails
    """
    def _decode(z: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, str]:
        # Check for multi-band format (our production format)
        if "image_r" in z:
            bands = []
            bandset = ""
            for key, char in [("image_g", "g"), ("image_r", "r"), ("image_z", "z")]:
                if key in z:
                    bands.append(np.asarray(z[key], dtype=np.float32))
                    bandset += char
            
            if len(bands) == 0:
                raise ValueError("No band keys found in NPZ")
            
            arr = np.stack(bands, axis=0)
            return arr, bandset
        
        # Legacy single-key format
        elif "img" in z:
            arr = np.asarray(z["img"], dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]
            return arr, "unknown"
        
        # Fallback to first key
        else:
            keys = list(z.keys())
            if len(keys) == 0:
                raise ValueError("Empty NPZ archive")
            
            key = keys[0]
            arr = np.asarray(z[key], dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]
            return arr, "unknown"
    
    # Try uncompressed first
    try:
        z = np.load(io.BytesIO(blob), allow_pickle=False)
        return _decode(z)
    except Exception:
        pass
    
    # Try gzip compressed
    try:
        data = gzip.decompress(blob)
        z = np.load(io.BytesIO(data), allow_pickle=False)
        return _decode(z)
    except Exception as e:
        raise ValueError(f"Failed to decode stamp NPZ: {e}")


def validate_stamp(stamp: np.ndarray, name: str = "stamp") -> Dict[str, bool]:
    """
    Validate stamp array against schema.
    
    Uses STAMP_SCHEMA for consistent validation.
    
    Args:
        stamp: Array to validate (C, H, W)
        name: Name for error messages
    
    Returns:
        Dict with validation results including 'valid' key
    """
    result = STAMP_SCHEMA.validate_array(stamp, name)
    
    # Convert to simple bool dict for compatibility
    return {
        "has_correct_shape": stamp.shape == STAMP_SCHEMA.expected_shape or stamp.shape[0] == NUM_CHANNELS,
        "no_nan": not np.isnan(stamp).any(),
        "no_inf": not np.isinf(stamp).any(),
        "has_variance": stamp.std() > STAMP_SCHEMA.min_variance,
        "reasonable_range": stamp.min() > VALUE_RANGE_MIN and stamp.max() < VALUE_RANGE_MAX,
        "valid": result["valid"],
        "errors": result.get("errors", []),
        "warnings": result.get("warnings", []),
    }


# =============================================================================
# NORMALIZATION
# =============================================================================

def create_radial_mask(
    height: int,
    width: int,
    radius: float,
    inside: bool = True
) -> np.ndarray:
    """
    Create a circular mask.
    
    Args:
        height: Image height
        width: Image width
        radius: Mask radius in pixels
        inside: If True, mask is True inside radius; if False, True outside
    
    Returns:
        Boolean mask array (H, W)
    """
    cy, cx = height // 2, width // 2
    yy, xx = np.ogrid[:height, :width]
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    
    if inside:
        return r < radius
    else:
        return r >= radius


def robust_normalize(
    img: np.ndarray,
    outer_radius: int = OUTER_RADIUS_PIX,
    clip_sigma: float = CLIP_SIGMA
) -> np.ndarray:
    """
    Per-sample robust normalization using outer annulus statistics.
    
    Uses median and MAD from outer region (r > outer_radius) to normalize.
    This reduces sensitivity to core brightness variations.
    
    Args:
        img: Input image (C, H, W)
        outer_radius: Pixels with r > outer_radius used for stats
        clip_sigma: Clip values beyond this many sigma
    
    Returns:
        Normalized image (C, H, W), dtype float32
    """
    c, h, w = img.shape
    outer_mask = create_radial_mask(h, w, outer_radius, inside=False)
    
    normalized = np.zeros_like(img, dtype=np.float32)
    
    for i in range(c):
        outer_values = img[i][outer_mask]
        
        if len(outer_values) == 0:
            # Fallback if no outer pixels
            median = np.median(img[i])
            mad = np.median(np.abs(img[i] - median))
        else:
            median = np.median(outer_values)
            mad = np.median(np.abs(outer_values - median))
        
        # Convert MAD to std-equivalent
        std = MAD_TO_STD * mad + 1e-10
        
        # Normalize and clip
        norm = (img[i] - median) / std
        norm = np.clip(norm, -clip_sigma, clip_sigma)
        normalized[i] = norm
    
    return normalized


# =============================================================================
# HARD NEGATIVE GENERATION
# =============================================================================

def azimuthal_shuffle(
    diff: np.ndarray,
    n_bins: int = AZIMUTHAL_SHUFFLE_BINS,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Create hard negative by shuffling pixels within radial bins.
    
    Preserves:
    - Radial profile (brightness at each radius)
    - Total flux
    - Statistical properties
    
    Destroys:
    - Arc coherent structure
    - Morphology
    
    Args:
        diff: Difference image (stamp - ctrl), shape (C, H, W)
        n_bins: Number of radial bins
        seed: Random seed for reproducibility
    
    Returns:
        Shuffled difference image
    """
    if seed is not None:
        np.random.seed(seed)
    
    c, h, w = diff.shape
    cy, cx = h // 2, w // 2
    
    # Create radial coordinate
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    
    # Bin edges
    r_max = np.sqrt(cx**2 + cy**2)
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    
    # Shuffle within each bin
    shuffled = np.zeros_like(diff)
    
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        n_pixels = mask.sum()
        
        if n_pixels == 0:
            continue
        
        for c_idx in range(c):
            values = diff[c_idx][mask].copy()
            np.random.shuffle(values)
            shuffled[c_idx][mask] = values
    
    return shuffled


# =============================================================================
# CORE DROPOUT
# =============================================================================

def apply_core_dropout(
    img: np.ndarray,
    radius: int = CORE_RADIUS_PIX,
    fill_mode: str = "outer_median"
) -> np.ndarray:
    """
    Apply core dropout by masking/replacing central pixels.
    
    Forces model to learn from arc morphology in outer regions.
    
    Args:
        img: Input image (C, H, W)
        radius: Mask pixels with r < radius
        fill_mode: How to fill masked region
            - "outer_median": Fill with outer annulus median
            - "zero": Fill with zeros
            - "noise": Fill with noise matched to outer region
    
    Returns:
        Image with core masked
    """
    c, h, w = img.shape
    
    core_mask = create_radial_mask(h, w, radius, inside=True)
    outer_mask = ~core_mask
    
    result = img.copy()
    
    for i in range(c):
        if fill_mode == "outer_median":
            fill_value = np.median(img[i][outer_mask])
            result[i][core_mask] = fill_value
        elif fill_mode == "zero":
            result[i][core_mask] = 0.0
        elif fill_mode == "noise":
            outer_std = np.std(img[i][outer_mask])
            outer_mean = np.mean(img[i][outer_mask])
            noise = np.random.normal(outer_mean, outer_std, core_mask.sum())
            result[i][core_mask] = noise
        else:
            result[i][core_mask] = 0.0
    
    return result


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def format_check_result(name: str, passed: bool, details: str = "") -> str:
    """Format a check result for logging."""
    status = "✓ PASS" if passed else "✗ FAIL"
    if details:
        return f"  {name}: {status} ({details})"
    return f"  {name}: {status}"

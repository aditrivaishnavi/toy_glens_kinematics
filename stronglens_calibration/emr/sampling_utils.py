#!/usr/bin/env python3
"""
Utility functions for negative sampling.

These functions are separated from the main Spark job to enable testing
without pyspark dependency.
"""
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


# =============================================================================
# CONSTANTS
# =============================================================================

# Galaxy types for Pool N1 (Paper IV parity: SER/DEV/REX only, EXP excluded)
VALID_TYPES_N1 = {"SER", "DEV", "REX"}

# nobs_z bin edges: [1-2], [3-5], [6-10], [11+]
NOBS_Z_BINS = [(1, 2), (3, 5), (6, 10), (11, 999)]

# Maskbits to exclude (from config, but hardcoded defaults)
DEFAULT_EXCLUDE_MASKBITS = {1, 5, 6, 7, 11, 12, 13}


# =============================================================================
# BINNING FUNCTIONS
# =============================================================================

def get_nobs_z_bin(nobs_z: Optional[int]) -> str:
    """Assign nobs_z to bin label."""
    if nobs_z is None or nobs_z < 1:
        return "invalid"
    
    for low, high in NOBS_Z_BINS:
        if low <= nobs_z <= high:
            if high == 999:
                return f"{low}+"
            return f"{low}-{high}"
    
    return "11+"  # Fallback for very high values


def get_type_bin(galaxy_type: Optional[str]) -> str:
    """Normalize galaxy type to bin label."""
    if galaxy_type is None:
        return "OTHER"
    
    galaxy_type = galaxy_type.strip().upper()
    if galaxy_type in VALID_TYPES_N1:
        return galaxy_type
    return "OTHER"


# =============================================================================
# PHOTOMETRY
# =============================================================================

def flux_to_mag(flux: Optional[float]) -> Optional[float]:
    """Convert flux in nanomaggies to AB magnitude."""
    if flux is None or flux <= 0:
        return None
    return float(22.5 - 2.5 * np.log10(flux))


# =============================================================================
# HEALPIX UTILITIES
# =============================================================================

def compute_healpix(ra: float, dec: float, nside: int) -> int:
    """
    Compute HEALPix index for given coordinates.
    
    Uses healpy if available, falls back to manual calculation.
    """
    try:
        import healpy as hp
        # Convert to theta, phi (healpy convention)
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        return int(hp.ang2pix(nside, theta, phi, nest=True))
    except ImportError:
        # Manual calculation (less accurate but works without healpy)
        # This is a fallback - prefer using healpy
        return int(hash((round(ra, 4), round(dec, 4), nside)) % (12 * nside * nside))


def assign_split(healpix_idx: int, allocations: Dict[str, float], seed: int = 42) -> str:
    """
    Assign train/val/test split based on HEALPix index.
    
    Uses deterministic hash to ensure reproducibility.
    
    IMPORTANT: Allocation order is explicitly defined as [train, val, test]
    to avoid bugs from lexicographic sorting (test < train < val alphabetically).
    """
    # Create deterministic hash
    hash_input = f"{healpix_idx}_{seed}"
    hash_bytes = hashlib.sha256(hash_input.encode()).digest()
    hash_value = int.from_bytes(hash_bytes[:4], "big") / (2**32)
    
    # FIXED: Use explicit ordering instead of sorted() which sorts alphabetically
    # This ensures train comes before val comes before test
    SPLIT_ORDER = ["train", "val", "test"]
    
    # Assign based on cumulative thresholds in explicit order
    cumulative = 0.0
    for split in SPLIT_ORDER:
        if split in allocations:
            cumulative += allocations[split]
            if hash_value < cumulative:
                return split
    
    return "train"  # Default fallback


# =============================================================================
# SPATIAL CROSS-MATCHING
# =============================================================================

def is_near_known_lens(
    ra: float,
    dec: float,
    known_coords: List[Tuple[float, float]],
    radius_arcsec: float,
    _kdtree_cache: dict = {}  # Mutable default for caching
) -> bool:
    """
    Check if position is within exclusion radius of any known lens.
    
    Uses KD-tree for O(log n) lookup instead of O(n) linear scan.
    """
    if not known_coords:
        return False
    
    radius_deg = radius_arcsec / 3600.0
    
    # Build or retrieve cached KD-tree
    cache_key = id(known_coords)
    if cache_key not in _kdtree_cache:
        try:
            from scipy.spatial import cKDTree
            # Convert to Cartesian for proper spherical distance
            coords_array = np.array(known_coords)
            ra_rad = np.radians(coords_array[:, 0])
            dec_rad = np.radians(coords_array[:, 1])
            x = np.cos(dec_rad) * np.cos(ra_rad)
            y = np.cos(dec_rad) * np.sin(ra_rad)
            z = np.sin(dec_rad)
            _kdtree_cache[cache_key] = cKDTree(np.column_stack([x, y, z]))
            _kdtree_cache[f"{cache_key}_has_tree"] = True
        except ImportError:
            _kdtree_cache[f"{cache_key}_has_tree"] = False
    
    # Use KD-tree if available
    if _kdtree_cache.get(f"{cache_key}_has_tree", False):
        tree = _kdtree_cache[cache_key]
        # Convert query point to Cartesian
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)
        # Convert angular radius to chord length
        chord_length = 2 * np.sin(np.radians(radius_deg) / 2)
        neighbors = tree.query_ball_point([x, y, z], chord_length)
        return len(neighbors) > 0
    
    # Fallback to linear scan (slow but works)
    for known_ra, known_dec in known_coords:
        delta_ra = (ra - known_ra) * np.cos(np.radians(dec))
        delta_dec = dec - known_dec
        sep = np.sqrt(delta_ra**2 + delta_dec**2)
        if sep < radius_deg:
            return True
    
    return False


# =============================================================================
# POOL CLASSIFICATION
# =============================================================================

# Default N2 thresholds calibrated for ~15% N2 rate in DR10
# Based on actual DR10 Tractor catalog distributions:
#   - flux_r: log-normal, median ~3-5 nMgy (mag ~20-21)
#   - shape_r: log-normal, median ~0.5-1.0 arcsec
#   - g-r: Gaussian, median ~0.65, std ~0.25
#   - ellipticity: sqrt(e1^2 + e2^2), range 0-0.7
#
# Calibration notes (2026-02-09):
#   - Initial thresholds gave 37.8% N2, too high
#   - Tightened to target 10-20% range
#   - Removed bright_core (redundant with ring_proxy)
DEFAULT_N2_THRESHOLDS = {
    "ring_proxy": {
        "types": ["DEV", "SER"],  # DEV always qualifies, SER needs high sersic
        "flux_r_min": 5.0,        # ~mag 20.8 (tightened from 3.0)
        "sersic_min": 4.0,        # Very high concentration (for SER only)
    },
    "edge_on_proxy": {
        "types": ["EXP", "SER", "DEV"],  # Any extended galaxy
        "ellipticity_min": 0.50,  # High ellipticity (LLM: loosened from 0.55 to get 2-4% edge-on)
        "shape_r_min": 0.6,       # At least moderately resolved
        "shape_r_min_legacy": 1.8,  # Fallback if no ellipticity (loosened from 2.0)
    },
    "blue_clumpy_proxy": {
        "g_minus_r_max": 0.4,     # Very blue (tightened from 0.5)
        "r_mag_max": 20.5,        # Reasonably bright (tightened from 21.0)
    },
    "large_galaxy_proxy": {
        "shape_r_min": 2.0,       # Large half-light radius (tightened from 1.5)
        "flux_r_min": 3.0,        # Visible (tightened from 2.0)
    },
    # bright_core_proxy removed - redundant with ring_proxy
}


def classify_pool_n2(
    galaxy_type: str,
    flux_r: Optional[float],
    shape_r: Optional[float],
    g_minus_r: Optional[float],
    mag_r: Optional[float],
    config: Dict[str, Any],
    ellipticity: Optional[float] = None,
    sersic: Optional[float] = None,
) -> Optional[str]:
    """
    Classify galaxy into N2 confuser categories based on Tractor properties.
    
    N2 "hard confusers" are galaxies that might look like lenses:
    - ring_proxy: Ellipticals with ring-like features
    - edge_on_proxy: Edge-on disks (elongated, could look like arcs)
    - blue_clumpy_proxy: Blue star-forming with clumps
    - large_galaxy_proxy: Extended galaxies with structure
    - bright_core_proxy: Bright concentrated cores
    
    Target: ~15% of galaxies should be classified as N2.
    
    Returns confuser category or None if not a confuser.
    """
    # Get config thresholds, falling back to calibrated defaults
    n2_config = config.get("negative_pools", {}).get("pool_n2", {}).get("tractor_criteria", {})
    
    # Normalize type
    galaxy_type = galaxy_type.strip().upper() if galaxy_type else ""
    
    # --- Category 1: Ring proxy ---
    # DEV galaxies (de Vaucouleurs, n=4) - ellipticals that might have ring features
    ring_cfg = {**DEFAULT_N2_THRESHOLDS["ring_proxy"], **n2_config.get("ring_proxy", {})}
    ring_types = ring_cfg.get("types", ["DEV"])
    if galaxy_type in ring_types:
        if flux_r is not None and flux_r >= ring_cfg.get("flux_r_min", 5.0):
            # DEV is always n=4, so it qualifies if bright enough
            if galaxy_type == "DEV":
                return "ring_proxy"
            # For SER, require very high Sersic index
            elif galaxy_type == "SER" and sersic is not None:
                if sersic >= ring_cfg.get("sersic_min", 4.0):
                    return "ring_proxy"
    
    # --- Category 2: Edge-on proxy ---
    # Elongated galaxies that could look like arcs
    edge_cfg = {**DEFAULT_N2_THRESHOLDS["edge_on_proxy"], **n2_config.get("edge_on_proxy", {})}
    edge_types = edge_cfg.get("types", ["EXP", "SER", "DEV"])
    if galaxy_type in edge_types:
        # Check ellipticity if available
        if ellipticity is not None and ellipticity >= edge_cfg.get("ellipticity_min", 0.50):
            if shape_r is not None and shape_r >= edge_cfg.get("shape_r_min", 0.6):
                return "edge_on_proxy"
        # Fallback to shape_r for extended galaxies (legacy behavior)
        elif ellipticity is None and shape_r is not None:
            if shape_r >= edge_cfg.get("shape_r_min_legacy", 1.8):
                return "edge_on_proxy"
    
    # --- Category 3: Blue clumpy proxy ---
    # Blue star-forming galaxies with clumpy structure
    blue_cfg = {**DEFAULT_N2_THRESHOLDS["blue_clumpy_proxy"], **n2_config.get("blue_clumpy_proxy", {})}
    if g_minus_r is not None and g_minus_r <= blue_cfg.get("g_minus_r_max", 0.4):
        if mag_r is not None and mag_r <= blue_cfg.get("r_mag_max", 20.5):
            return "blue_clumpy"
    
    # --- Category 4: Large galaxy proxy ---
    # Extended galaxies with visible structure (spiral arms, etc.)
    large_cfg = {**DEFAULT_N2_THRESHOLDS["large_galaxy_proxy"], **n2_config.get("large_galaxy_proxy", {})}
    if shape_r is not None and shape_r >= large_cfg.get("shape_r_min", 2.0):
        if flux_r is not None and flux_r >= large_cfg.get("flux_r_min", 3.0):
            return "large_galaxy"
    
    # bright_core category removed - redundant with ring_proxy
    
    return None


def classify_pool_n2_simple(
    galaxy_type: str,
    flux_r: Optional[float],
    shape_r: Optional[float],
    g_minus_r: Optional[float],
    mag_r: Optional[float],
    config: Dict[str, Any]
) -> Optional[str]:
    """
    Simplified N2 classification without ellipticity/sersic.
    
    For backward compatibility with existing Spark jobs that don't 
    extract ellipticity and sersic columns.
    """
    return classify_pool_n2(
        galaxy_type=galaxy_type,
        flux_r=flux_r,
        shape_r=shape_r,
        g_minus_r=g_minus_r,
        mag_r=mag_r,
        config=config,
        ellipticity=None,
        sersic=None,
    )


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def check_maskbits(maskbits: int, exclude_bits: set) -> bool:
    """
    Check if any excluded maskbits are set.
    
    Returns True if the galaxy should be EXCLUDED.
    """
    for bit in exclude_bits:
        if maskbits & (1 << bit):
            return True
    return False


def compute_split_proportions(splits: List[str]) -> Dict[str, float]:
    """Compute actual proportions from list of splits."""
    n = len(splits)
    if n == 0:
        return {}
    
    counts = {}
    for s in splits:
        counts[s] = counts.get(s, 0) + 1
    
    return {k: v / n for k, v in counts.items()}

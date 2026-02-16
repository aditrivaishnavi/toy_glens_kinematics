#!/usr/bin/env python3
"""
Dark Halo Scope: Phase 3 Core Functions (Spark-Free)
=====================================================

Pure functions extracted from spark_phase3_define_fields_and_build_parent.py
for local testing and debugging without Spark.

This module contains:
- S3 URI parsing
- Magnitude conversions
- FITS column extraction
- LRG variant definitions and selection
- Sweep chunk processing

All functions are pure (no Spark, no side effects) and can be unit tested.
"""

from __future__ import annotations

import gzip
import os
import shutil
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
from astropy.io import fits


# ---------------------------
# Hard-coded LRG hypergrid thresholds (must match Phase 2)
# ---------------------------

@dataclass(frozen=True)
class LRGVariant:
    """Defines an LRG selection variant with magnitude/color cuts."""
    name: str
    z_mag_max: float
    rz_min: float
    zw1_min: float


# These must match Phase 2 exactly
LRG_VARIANTS: List[LRGVariant] = [
    LRGVariant("v1_pure_massive",   z_mag_max=20.0, rz_min=0.5, zw1_min=1.6),
    LRGVariant("v2_baseline_dr10",  z_mag_max=20.4, rz_min=0.4, zw1_min=1.6),
    LRGVariant("v3_color_relaxed",  z_mag_max=20.4, rz_min=0.4, zw1_min=0.8),
    LRGVariant("v4_mag_relaxed",    z_mag_max=21.0, rz_min=0.4, zw1_min=0.8),
    LRGVariant("v5_very_relaxed",   z_mag_max=21.5, rz_min=0.3, zw1_min=0.8),
]

BASELINE_VARIANT = "v3_color_relaxed"


def get_variant_by_name(name: str) -> Optional[LRGVariant]:
    """Get an LRG variant by name."""
    for v in LRG_VARIANTS:
        if v.name == name:
            return v
    return None


# ---------------------------
# S3 URI Parsing
# ---------------------------

def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """
    Parse an S3 URI into (bucket, key).
    
    Args:
        s3_uri: S3 URI like 's3://bucket/path/to/object'
    
    Returns:
        Tuple of (bucket_name, key)
    
    Raises:
        ValueError: If the URI doesn't start with 's3://'
    
    Examples:
        >>> parse_s3_uri('s3://mybucket/path/to/file.csv')
        ('mybucket', 'path/to/file.csv')
        >>> parse_s3_uri('s3://bucket/key')
        ('bucket', 'key')
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Not an s3:// URI: {s3_uri}")
    parsed = urllib.parse.urlparse(s3_uri)
    return parsed.netloc, parsed.path.lstrip("/")


# ---------------------------
# Magnitude Conversion
# ---------------------------

def nanomaggies_to_mag(flux: np.ndarray, zero_point: float = 22.5) -> np.ndarray:
    """
    Convert nanomaggies to AB magnitudes.
    
    Args:
        flux: Array of fluxes in nanomaggies
        zero_point: Zeropoint magnitude (default 22.5 for AB system)
    
    Returns:
        Array of AB magnitudes. For flux <= 0 or NaN, returns NaN.
    
    Examples:
        >>> nanomaggies_to_mag(np.array([1.0]))  # 1 nanomaggie = 22.5 mag
        array([22.5])
        >>> nanomaggies_to_mag(np.array([0.0]))  # Invalid flux
        array([nan])
    """
    flux = np.asarray(flux, dtype=np.float64)
    out = np.full_like(flux, np.nan, dtype=np.float64)
    m = flux > 0
    out[m] = zero_point - 2.5 * np.log10(flux[m])
    return out


# ---------------------------
# FITS Column Extraction
# ---------------------------

def get_col(data: np.ndarray, names: List[str], desired: List[str]) -> np.ndarray:
    """
    Find a column in a FITS recarray by case-insensitive match against 'desired'.
    
    Args:
        data: FITS table data (recarray)
        names: List of column names in the FITS file
        desired: List of possible column names to look for (in priority order)
    
    Returns:
        The column data array
    
    Raises:
        KeyError: If none of the desired columns are found
    
    Examples:
        >>> names = ['RA', 'DEC', 'BRICKNAME', 'TYPE']
        >>> get_col(data, names, ['brickname', 'BRICKNAME'])  # Case insensitive
        array([...])
    """
    lower = {n.lower(): n for n in names}
    for d in desired:
        k = d.lower()
        if k in lower:
            return data[lower[k]]
    raise KeyError(f"Missing FITS column. Tried {desired}. Have {names[:30]}...")


# ---------------------------
# LRG Selection
# ---------------------------

def compute_lrg_flags(
    mag_z: np.ndarray,
    r_minus_z: np.ndarray,
    z_minus_w1: np.ndarray,
    variants: Optional[List[LRGVariant]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute LRG selection flags for all variants.
    
    Args:
        mag_z: Array of z-band magnitudes
        r_minus_z: Array of r-z colors
        z_minus_w1: Array of z-W1 colors
        variants: List of LRG variants to compute (default: all 5)
    
    Returns:
        Dictionary mapping variant name to boolean mask array
    
    Examples:
        >>> flags = compute_lrg_flags(mag_z, r_z, z_w1)
        >>> flags['v3_color_relaxed']  # Boolean array
        array([True, False, True, ...])
    """
    if variants is None:
        variants = LRG_VARIANTS
    
    result = {}
    for v in variants:
        mask = (
            (mag_z < v.z_mag_max)
            & (r_minus_z > v.rz_min)
            & (z_minus_w1 > v.zw1_min)
        )
        result[v.name] = mask
    return result


# ---------------------------
# Sweep Processing
# ---------------------------

def maybe_decompress_gzip(local_path: str) -> str:
    """
    If local_path ends with .gz, decompress to a file without .gz extension.
    
    Args:
        local_path: Path to a file (may or may not be gzipped)
    
    Returns:
        Path to the uncompressed file
    
    Notes:
        Uses streaming decompression to avoid loading entire file into memory.
    """
    if not local_path.endswith(".gz"):
        return local_path
    out_path = local_path[:-3]
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    # Use streaming copy to avoid memory spike for large files
    with gzip.open(local_path, "rb") as fin, open(out_path, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)  # 16MB chunks
    return out_path


def process_sweep_chunk(
    fits_path: str,
    brick_to_modes: Dict[str, List[Tuple[str, int, int]]],
    chunk_size: int = 100000,
    baseline_variant: str = BASELINE_VARIANT,
) -> Iterator[Dict[str, Any]]:
    """
    Process a single FITS sweep file and yield LRG rows.
    
    This is the core processing logic extracted from sweep_partition_iterator,
    without any S3/checkpointing logic.
    
    Args:
        fits_path: Local path to the FITS file
        brick_to_modes: Dictionary mapping brickname -> list of (mode, region_id, region_rank)
        chunk_size: Number of rows to process at a time
        baseline_variant: Which variant to use for parent selection (default: v3_color_relaxed)
    
    Yields:
        Dictionary rows ready for DataFrame creation
    
    Notes:
        - Filters to TYPE != 'PSF'
        - Filters to positive fluxes in r, z, W1
        - Filters to bricks present in brick_to_modes
        - Applies baseline variant LRG cut
    """
    # Decompress if needed
    if fits_path.endswith(".gz"):
        fits_path = maybe_decompress_gzip(fits_path)
    
    # Pre-cache brick keys for efficient lookup
    brick_keys_set: Set[str] = set(brick_to_modes.keys())
    brick_keys_list: List[str] = list(brick_keys_set)
    
    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul[1].data
        names = list(data.columns.names)
        
        # Extract columns (case-insensitive)
        ra = get_col(data, names, ["ra"])
        dec = get_col(data, names, ["dec"])
        brick = get_col(data, names, ["brickname"])
        typ = get_col(data, names, ["type"])
        
        objid = None
        try:
            objid = get_col(data, names, ["objid"])
        except KeyError:
            pass
        
        flux_g = get_col(data, names, ["flux_g"])
        flux_r = get_col(data, names, ["flux_r"])
        flux_z = get_col(data, names, ["flux_z"])
        flux_w1 = get_col(data, names, ["flux_w1"])
        
        n = len(ra)
        
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            
            # Convert bricknames to strings
            brick_chunk = np.array(brick[start:end]).astype(str)
            
            # Filter to selected bricks
            in_sel = np.isin(brick_chunk, brick_keys_list)
            if not np.any(in_sel):
                continue
            
            # TYPE != 'PSF' filter
            type_chunk = np.array(typ[start:end]).astype(str)
            non_psf = type_chunk != "PSF"
            
            # Flux arrays
            fr = np.asarray(flux_r[start:end], dtype=np.float64)
            fz = np.asarray(flux_z[start:end], dtype=np.float64)
            fw = np.asarray(flux_w1[start:end], dtype=np.float64)
            
            # Positive flux filter
            good_flux = (fr > 0) & (fz > 0) & (fw > 0)
            
            # Combined filter
            keep = in_sel & non_psf & good_flux
            if not np.any(keep):
                continue
            
            # G-band flux and positions
            fg = np.asarray(flux_g[start:end], dtype=np.float64)
            ra_c = np.asarray(ra[start:end], dtype=np.float64)
            dec_c = np.asarray(dec[start:end], dtype=np.float64)
            
            # Compute magnitudes
            mag_g = nanomaggies_to_mag(fg)
            mag_r = nanomaggies_to_mag(fr)
            mag_z = nanomaggies_to_mag(fz)
            mag_w1 = nanomaggies_to_mag(fw)
            
            # Compute colors
            r_minus_z = mag_r - mag_z
            z_minus_w1 = mag_z - mag_w1
            
            # Compute all variant flags
            is_v = compute_lrg_flags(mag_z, r_minus_z, z_minus_w1)
            
            # Baseline parent selection
            base = is_v[baseline_variant] & keep
            idx_arr = np.where(base)[0]
            if idx_arr.size == 0:
                continue
            
            # Yield rows for each matching object
            for i in idx_arr:
                bname = brick_chunk[i]
                mode_entries = brick_to_modes.get(bname)
                if not mode_entries:
                    continue
                
                row_common = {
                    "RA": float(ra_c[i]),
                    "DEC": float(dec_c[i]),
                    "BRICKNAME": str(bname),
                    "OBJID": int(objid[start + i]) if objid is not None else None,
                    "TYPE": str(type_chunk[i]),
                    "FLUX_G": float(fg[i]),
                    "FLUX_R": float(fr[i]),
                    "FLUX_Z": float(fz[i]),
                    "FLUX_W1": float(fw[i]),
                    "g_mag": float(mag_g[i]) if np.isfinite(mag_g[i]) else None,
                    "r_mag": float(mag_r[i]) if np.isfinite(mag_r[i]) else None,
                    "z_mag": float(mag_z[i]) if np.isfinite(mag_z[i]) else None,
                    "w1_mag": float(mag_w1[i]) if np.isfinite(mag_w1[i]) else None,
                    "r_minus_z": float(r_minus_z[i]) if np.isfinite(r_minus_z[i]) else None,
                    "z_minus_w1": float(z_minus_w1[i]) if np.isfinite(z_minus_w1[i]) else None,
                    "is_lrg_v1": bool(is_v["v1_pure_massive"][i]),
                    "is_lrg_v2": bool(is_v["v2_baseline_dr10"][i]),
                    "is_lrg_v3": bool(is_v["v3_color_relaxed"][i]),
                    "is_lrg_v4": bool(is_v["v4_mag_relaxed"][i]),
                    "is_lrg_v5": bool(is_v["v5_very_relaxed"][i]),
                }
                
                for (mode, region_id, region_rank) in mode_entries:
                    out = dict(row_common)
                    out["phase3_ranking_mode"] = mode
                    out["region_id"] = int(region_id)
                    out["phase3_region_rank"] = int(region_rank)
                    yield out


# ---------------------------
# Region Scoring (Pure Python, no Spark)
# ---------------------------

def compute_region_scores(
    regions_df: "pd.DataFrame",
    variant: str,
    psf_ref: float = 1.25,
    sigma_psf: float = 0.15,
    k_ebv: float = 8.0,
) -> "pd.DataFrame":
    """
    Compute region scores for all ranking modes.
    
    Args:
        regions_df: DataFrame with region summary columns
        variant: LRG variant name (e.g., 'v3_color_relaxed')
        psf_ref: Reference PSF for weighting
        sigma_psf: PSF sigma for weighting
        k_ebv: E(B-V) weighting factor
    
    Returns:
        DataFrame with additional score columns
    """
    import pandas as pd
    
    df = regions_df.copy()
    
    dens_col = f"mean_lrg_density_{variant}"
    cnt_col = f"total_n_lrg_{variant}"
    
    df["score_density"] = df[dens_col]
    df["score_n_lrg"] = df[cnt_col]
    df["score_area_weighted"] = df[dens_col] * np.sqrt(df["total_area_deg2"])
    
    # PSF weighting
    w_psf = np.exp(-(df["median_psf_r_arcsec"] - psf_ref) / sigma_psf)
    w_psf = np.clip(w_psf, 0, 1)
    
    # E(B-V) weighting
    w_ebv = np.exp(-k_ebv * df["median_ebv"])
    
    # Depth weighting (normalize by p10/p90)
    depth = df["median_psfdepth_r"]
    depth_p10, depth_p90 = np.percentile(depth.dropna(), [10, 90])
    if depth_p90 <= depth_p10:
        depth_p90 = depth_p10 + 1
    w_depth = (depth - depth_p10) / (depth_p90 - depth_p10)
    w_depth = np.clip(w_depth, 0, 1)
    
    df["score_psf_weighted"] = df[dens_col] * w_psf * w_depth * w_ebv
    
    return df


def select_top_regions(
    scored_df: "pd.DataFrame",
    mode: str,
    k: int,
) -> "pd.DataFrame":
    """
    Select top-K regions for a given ranking mode.
    
    Args:
        scored_df: DataFrame with score columns
        mode: Ranking mode ('density', 'n_lrg', 'area_weighted', 'psf_weighted')
        k: Number of regions to select
    
    Returns:
        DataFrame of top-K regions with rank column
    """
    score_col = {
        "density": "score_density",
        "n_lrg": "score_n_lrg",
        "area_weighted": "score_area_weighted",
        "psf_weighted": "score_psf_weighted",
    }[mode]
    
    # Sort by score descending, take top K
    df = scored_df.sort_values(score_col, ascending=False).head(k).copy()
    df["phase3_ranking_mode"] = mode
    df["phase3_score"] = df[score_col]
    df["phase3_region_rank"] = range(1, len(df) + 1)
    
    return df


def build_brick_to_modes(
    regions_summary_df: "pd.DataFrame",
    regions_bricks_df: "pd.DataFrame",
    variant: str,
    ranking_modes: List[str],
    num_regions: int,
    max_ebv: float = 0.12,
    max_psf_r_arcsec: float = 1.60,
    min_psfdepth_r: float = 23.6,
) -> Dict[str, List[Tuple[str, int, int]]]:
    """
    Build the brick-to-modes dictionary that maps bricknames to ranking modes.
    
    This is the key function that creates the mapping used by sweep processing.
    
    Args:
        regions_summary_df: DataFrame with region-level summaries from Phase 2
        regions_bricks_df: DataFrame with brick-to-region mapping from Phase 2
        variant: LRG variant name
        ranking_modes: List of ranking modes to use
        num_regions: Number of top regions to select per mode
        max_ebv: Maximum E(B-V) for quality cut
        max_psf_r_arcsec: Maximum PSF for quality cut
        min_psfdepth_r: Minimum depth for quality cut
    
    Returns:
        Dictionary mapping brickname -> list of (mode, region_id, region_rank)
    """
    import pandas as pd
    
    # Apply quality cuts
    regions = regions_summary_df.copy()
    n_before = len(regions)
    
    regions = regions[regions["median_ebv"] <= max_ebv]
    regions = regions[regions["median_psf_r_arcsec"] <= max_psf_r_arcsec]
    regions = regions[regions["median_psfdepth_r"] >= min_psfdepth_r]
    
    n_after = len(regions)
    print(f"Quality cuts: {n_before} -> {n_after} regions")
    
    if n_after == 0:
        print("WARNING: All regions filtered out by quality cuts!")
        return {}
    
    # Compute scores
    scored = compute_region_scores(regions, variant)
    
    # Build brick_to_modes
    brick_to_modes: Dict[str, List[Tuple[str, int, int]]] = {}
    
    for mode in ranking_modes:
        top = select_top_regions(scored, mode, num_regions)
        
        # Get region IDs and their ranks
        reg_rank = dict(zip(top["region_id"], top["phase3_region_rank"]))
        
        # Get bricks for these regions
        sel_region_ids = list(reg_rank.keys())
        matching_bricks = regions_bricks_df[
            regions_bricks_df["region_id"].isin(sel_region_ids)
        ]
        
        for _, row in matching_bricks.iterrows():
            rid = int(row["region_id"])
            brickname = str(row["brickname"])
            rank = int(reg_rank[rid])
            
            brick_to_modes.setdefault(brickname, []).append((mode, rid, rank))
    
    return brick_to_modes


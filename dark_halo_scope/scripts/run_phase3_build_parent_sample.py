#!/usr/bin/env python3
"""
Phase 3 - Build LRG parent catalog in selected regions

This script uses the brick-level region definitions from Phase 3 field
selection and the DR10 sweep files to build an object-level parent catalog
of LRGs according to the v3_color_relaxed selection.

Supports multiple ranking modes from run_phase3_define_fields.py:
- area_weighted_v3
- density_v3
- total_lrg_v3
- manual (explicit region IDs)

Performance optimizations:
- Single-pass processing: All modes are processed together, each sweep is
  downloaded and parsed only once even if multiple modes share regions.
- Local FITS cache: Downloaded FITS files are cached to avoid re-downloading.
- Efficient memory usage: Only needed columns are loaded from FITS files.

For each ranking mode, outputs are written to:
  {output_dir}/{variant}/{ranking_mode}/phase3_lrg_parent_catalog.csv

Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import shutil
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("phase3.build_parent_sample")


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

RANKING_MODES_ALL = ("area_weighted_v3", "density_v3", "total_lrg_v3", "manual")


# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Phase 3 LRG parent catalog builder (multi-mode, optimized)"
    )

    parser.add_argument(
        "--phase3-dir",
        default="results/phase3",
        help=(
            "Base directory containing Phase 3 outputs from run_phase3_define_fields.py. "
            "Structure: {phase3-dir}/{variant}/{ranking_mode}/phase3_target_bricks.csv"
        ),
    )
    parser.add_argument(
        "--variant",
        default="v3_color_relaxed",
        help="Variant subdirectory name (e.g., v3_color_relaxed).",
    )
    parser.add_argument(
        "--ranking-modes",
        default="",
        help=(
            "Comma-separated list of ranking modes to process "
            "(area_weighted_v3, density_v3, total_lrg_v3, manual). "
            "If empty, auto-detects available modes from phase3-dir/{variant}/ subdirectories."
        ),
    )
    parser.add_argument(
        "--sweep-index-path",
        required=True,
        help=(
            "Path to a text file containing a list of DR10 sweep FITS URLs or "
            "filesystem paths, one per line."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.expanduser("~/data/sweep_cache"),
        help=(
            "Directory to cache downloaded FITS files. "
            "Default: ~/data/sweep_cache"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Output directory. If empty, uses phase3-dir/{variant}. "
            "Outputs go to {output-dir}/{ranking_mode}/phase3_lrg_parent_catalog.csv"
        ),
    )
    parser.add_argument(
        "--save-parquet",
        action="store_true",
        help="If set, also save a Parquet copy of the parent catalog.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, only show download plan without actually downloading.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="If set, re-download files even if cached.",
    )

    return parser.parse_args(argv)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def nanomaggies_to_mag(flux: np.ndarray) -> np.ndarray:
    """Convert nanomaggies to AB magnitudes. Non-positive flux -> NaN."""
    flux = np.asarray(flux, dtype=float)
    mag = np.full_like(flux, np.nan, dtype=float)
    mask = np.isfinite(flux) & (flux > 0)
    mag[mask] = 22.5 - 2.5 * np.log10(flux[mask])
    return mag


def read_sweep_paths(index_path: str) -> List[str]:
    """Read sweep FITS paths/URLs from a text file."""
    with open(index_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def get_sweep_bounds(sweep_name: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Parse sweep filename to extract RA/Dec bounds.
    Returns (ra_min, ra_max, dec_min, dec_max) or None if parsing fails.
    """
    base = os.path.basename(sweep_name)
    # Handle .fits.gz extension
    if base.endswith(".fits.gz"):
        base = base[:-3]  # Remove .gz
    if not base.startswith("sweep-") or not base.endswith(".fits"):
        return None

    try:
        core = base[len("sweep-") : -len(".fits")]
        part1, part2 = core.split("-")

        def parse_part(part: str) -> Tuple[float, float]:
            ra_str = part[:3]
            sign_char = part[3]
            dec_str = part[4:]
            ra = float(ra_str)
            dec = float(dec_str)
            if sign_char == "m":
                dec = -dec
            return ra, dec

        ra1, dec1 = parse_part(part1)
        ra2, dec2 = parse_part(part2)

        return (min(ra1, ra2), max(ra1, ra2), min(dec1, dec2), max(dec1, dec2))
    except Exception:
        return None


def sweep_overlaps_bricks(sweep_name: str, brick_positions: List[Tuple[float, float]]) -> bool:
    """Check if a sweep overlaps any of the actual brick positions."""
    bounds = get_sweep_bounds(sweep_name)
    if bounds is None:
        return True  # Can't parse, assume overlap
    
    swe_ra_min, swe_ra_max, swe_dec_min, swe_dec_max = bounds
    
    for ra, dec in brick_positions:
        if swe_ra_min <= ra <= swe_ra_max and swe_dec_min <= dec <= swe_dec_max:
            return True
    
    return False


def get_cache_path(url_or_path: str, cache_dir: str) -> str:
    """Generate a cache file path for a given URL or path."""
    # Use the basename as the cache filename
    basename = os.path.basename(url_or_path)
    # Handle .gz extension
    if basename.endswith(".gz"):
        basename = basename[:-3]
    return os.path.join(cache_dir, basename)


def download_with_cache(
    url_or_path: str,
    cache_dir: str,
    force_download: bool = False,
) -> str:
    """
    Download a FITS file to cache if not already cached.
    Returns the local path to the cached file.
    """
    cache_path = get_cache_path(url_or_path, cache_dir)
    
    # Check if already cached
    if os.path.exists(cache_path) and not force_download:
        logger.debug(f"Cache hit: {cache_path}")
        return cache_path
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        logger.info(f"  Downloading: {os.path.basename(url_or_path)}")
        temp_path = cache_path + ".tmp"
        try:
            urllib.request.urlretrieve(url_or_path, temp_path)
            
            # If downloaded file is gzipped, decompress it
            if url_or_path.endswith(".gz"):
                logger.debug(f"  Decompressing gzipped file...")
                with gzip.open(temp_path, 'rb') as f_in:
                    with open(cache_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(temp_path)
            else:
                os.rename(temp_path, cache_path)
                
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
    elif url_or_path.startswith("s3://"):
        # Use boto3 for S3
        import boto3
        from boto3.s3.transfer import TransferConfig
        
        logger.info(f"  Downloading from S3: {os.path.basename(url_or_path)}")
        s3 = boto3.client("s3")
        
        # Parse S3 URL
        parts = url_or_path[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        
        config = TransferConfig(
            multipart_threshold=100 * 1024 * 1024,
            max_concurrency=10,
        )
        
        temp_path = cache_path + ".tmp"
        try:
            s3.download_file(bucket, key, temp_path, Config=config)
            
            # Decompress if gzipped
            if url_or_path.endswith(".gz"):
                with gzip.open(temp_path, 'rb') as f_in:
                    with open(cache_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(temp_path)
            else:
                os.rename(temp_path, cache_path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
    else:
        # Local file - just copy or link
        if os.path.exists(url_or_path):
            logger.info(f"  Copying local file: {os.path.basename(url_or_path)}")
            shutil.copy2(url_or_path, cache_path)
        else:
            raise FileNotFoundError(f"Local file not found: {url_or_path}")
    
    return cache_path


def load_sweep_fits(path: str) -> pd.DataFrame:
    """
    Load a DR10 sweep FITS file into a Pandas DataFrame.
    Only loads columns needed for Phase 3 to minimize memory.
    """
    needed_cols = ["RA", "DEC", "BRICKNAME", "FLUX_G", "FLUX_R", "FLUX_Z", "FLUX_W1", "TYPE"]
    
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        
        result = {}
        available_cols = data.dtype.names
        
        for col in needed_cols:
            if col in available_cols:
                col_data = data[col]
                # Convert to native byte order efficiently
                if col_data.dtype.byteorder not in ('=', '|', '<' if sys.byteorder == 'little' else '>'):
                    result[col] = np.asarray(col_data, dtype=col_data.dtype.newbyteorder('='))
                else:
                    result[col] = np.asarray(col_data)
        
        df = pd.DataFrame(result)

    return df


def compute_lrg_flags_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute magnitudes, colors, and LRG selection flags.
    Filters to v3 LRGs and returns only those rows.
    
    Memory-efficient: modifies in place where possible.
    """
    # Compute magnitudes
    g_mag = nanomaggies_to_mag(df["FLUX_G"].values)
    r_mag = nanomaggies_to_mag(df["FLUX_R"].values)
    z_mag = nanomaggies_to_mag(df["FLUX_Z"].values)
    w1_mag = nanomaggies_to_mag(df["FLUX_W1"].values)

    r_minus_z = r_mag - z_mag
    z_minus_w1 = z_mag - w1_mag

    # v3 selection (our primary filter)
    is_lrg_v3 = (z_mag < 20.4) & (r_minus_z > 0.4) & (z_minus_w1 > 0.8)
    
    # Filter first, then compute other flags (saves memory)
    v3_mask = is_lrg_v3
    if not np.any(v3_mask):
        return pd.DataFrame()  # Empty result
    
    # Create output with only v3 LRGs
    out = df.loc[v3_mask].copy()
    
    # Add computed columns for v3 rows only
    out["g_mag"] = g_mag[v3_mask]
    out["r_mag"] = r_mag[v3_mask]
    out["z_mag"] = z_mag[v3_mask]
    out["w1_mag"] = w1_mag[v3_mask]
    out["r_minus_z"] = r_minus_z[v3_mask]
    out["z_minus_w1"] = z_minus_w1[v3_mask]
    
    # Now compute other variant flags for v3 LRGs
    z_v3 = out["z_mag"].values
    rz_v3 = out["r_minus_z"].values
    zw1_v3 = out["z_minus_w1"].values
    
    out["is_lrg_v1"] = (z_v3 < 20.0) & (rz_v3 > 0.5) & (zw1_v3 > 1.6)
    out["is_lrg_v2"] = (z_v3 < 20.4) & (rz_v3 > 0.4) & (zw1_v3 > 1.6)
    out["is_lrg_v3"] = True  # All rows are v3 by construction
    out["is_lrg_v4"] = (z_v3 < 21.0) & (rz_v3 > 0.4) & (zw1_v3 > 0.8)
    out["is_lrg_v5"] = (z_v3 < 21.5) & (rz_v3 > 0.3) & (zw1_v3 > 0.8)

    return out


def detect_ranking_modes(phase3_dir: str, variant: str) -> List[str]:
    """Auto-detect available ranking modes."""
    variant_dir = os.path.join(phase3_dir, variant)
    modes = []
    for mode in RANKING_MODES_ALL:
        bricks_path = os.path.join(variant_dir, mode, "phase3_target_bricks.csv")
        if os.path.exists(bricks_path):
            modes.append(mode)
    return modes


def format_size(bytes_val: float) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(bytes_val) < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}hr"


# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    
    phase3_dir = args.phase3_dir
    variant = args.variant
    variant_dir = os.path.join(phase3_dir, variant)
    output_base = args.output_dir if args.output_dir else variant_dir
    cache_dir = os.path.expanduser(args.cache_dir)
    
    # Determine ranking modes to process
    if args.ranking_modes:
        modes = [m.strip() for m in args.ranking_modes.split(",") if m.strip()]
        for m in modes:
            if m not in RANKING_MODES_ALL:
                raise ValueError(f"Unknown ranking mode '{m}'. Valid: {RANKING_MODES_ALL}")
    else:
        modes = detect_ranking_modes(phase3_dir, variant)
        if not modes:
            raise ValueError(
                f"No ranking modes detected in {variant_dir}. "
                "Run run_phase3_define_fields.py first."
            )
    
    logger.info("=" * 70)
    logger.info("Phase 3 - LRG Parent Catalog Builder (Optimized)")
    logger.info("=" * 70)
    logger.info(f"Phase 3 directory : {phase3_dir}")
    logger.info(f"Variant           : {variant}")
    logger.info(f"Output base       : {output_base}")
    logger.info(f"Cache directory   : {cache_dir}")
    logger.info(f"Ranking modes     : {modes}")
    logger.info(f"Sweep index       : {args.sweep_index_path}")
    logger.info(f"Dry run           : {args.dry_run}")
    logger.info(f"Force download    : {args.force_download}")
    
    # -------------------------------------------------------------------------
    # Step 1: Load all bricks across all modes
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("-" * 70)
    logger.info("Step 1: Loading target bricks for all modes")
    logger.info("-" * 70)
    
    # Structure: mode -> DataFrame of bricks
    mode_bricks: Dict[str, pd.DataFrame] = {}
    
    # Unified set of all bricks and their positions
    all_bricknames: Set[str] = set()
    all_brick_positions: List[Tuple[float, float]] = []
    brick_position_map: Dict[str, Tuple[float, float]] = {}
    
    # Brick -> (mode, region_id, rank) mappings
    brick_to_mode_info: Dict[str, List[Tuple[str, int, int]]] = {}
    
    for mode in modes:
        bricks_csv_path = os.path.join(variant_dir, mode, "phase3_target_bricks.csv")
        
        if not os.path.exists(bricks_csv_path):
            logger.warning(f"Bricks file not found for mode '{mode}': {bricks_csv_path}")
            continue
        
        bricks = pd.read_csv(bricks_csv_path)
        mode_bricks[mode] = bricks
        
        # Find column names
        brickname_col = next((c for c in ["brickname", "BRICKNAME"] if c in bricks.columns), None)
        ra_col = next((c for c in ["ra", "brick_ra_center", "RA"] if c in bricks.columns), None)
        dec_col = next((c for c in ["dec", "brick_dec_center", "DEC"] if c in bricks.columns), None)
        
        if not all([brickname_col, ra_col, dec_col]):
            raise ValueError(f"Bricks CSV missing required columns. Available: {list(bricks.columns)}")
        
        has_rank = "phase3_region_rank" in bricks.columns
        
        for _, row in bricks.iterrows():
            bn = str(row[brickname_col])
            ra = float(row[ra_col])
            dec = float(row[dec_col])
            region_id = int(row["region_id"])
            rank = int(row["phase3_region_rank"]) if has_rank else 0
            
            all_bricknames.add(bn)
            if bn not in brick_position_map:
                brick_position_map[bn] = (ra, dec)
                all_brick_positions.append((ra, dec))
            
            if bn not in brick_to_mode_info:
                brick_to_mode_info[bn] = []
            brick_to_mode_info[bn].append((mode, region_id, rank))
        
        logger.info(f"  Mode '{mode}': {len(bricks)} bricks")
    
    if not mode_bricks:
        raise ValueError("No valid mode bricks found.")
    
    total_unique_bricks = len(all_bricknames)
    logger.info(f"  Total unique bricks across all modes: {total_unique_bricks}")
    
    # -------------------------------------------------------------------------
    # Step 2: Find overlapping sweeps
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("-" * 70)
    logger.info("Step 2: Identifying sweeps that overlap target bricks")
    logger.info("-" * 70)
    
    sweep_paths = read_sweep_paths(args.sweep_index_path)
    logger.info(f"  Total sweeps in index: {len(sweep_paths)}")
    
    overlapping_sweeps = []
    for path in sweep_paths:
        if sweep_overlaps_bricks(path, all_brick_positions):
            overlapping_sweeps.append(path)
    
    logger.info(f"  Sweeps overlapping target bricks: {len(overlapping_sweeps)}")
    
    # Check cache status
    cached_count = 0
    to_download_count = 0
    for path in overlapping_sweeps:
        cache_path = get_cache_path(path, cache_dir)
        if os.path.exists(cache_path) and not args.force_download:
            cached_count += 1
        else:
            to_download_count += 1
    
    # Estimate download size (assuming ~300MB per sweep on average)
    est_size_per_sweep = 300 * 1024 * 1024  # 300 MB
    est_total_download = to_download_count * est_size_per_sweep
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("DOWNLOAD PLAN")
    logger.info("=" * 70)
    logger.info(f"  Total sweeps needed       : {len(overlapping_sweeps)}")
    logger.info(f"  Already cached            : {cached_count}")
    logger.info(f"  To download               : {to_download_count}")
    logger.info(f"  Estimated download size   : {format_size(est_total_download)}")
    logger.info(f"  Cache directory           : {cache_dir}")
    logger.info("")
    logger.info("Sweeps to process:")
    for i, path in enumerate(overlapping_sweeps):
        bounds = get_sweep_bounds(path)
        cache_path = get_cache_path(path, cache_dir)
        cached = os.path.exists(cache_path) and not args.force_download
        status = "[cached]" if cached else "[to download]"
        if bounds:
            logger.info(f"  {i+1:3d}. {os.path.basename(path)} "
                       f"(RA {bounds[0]:.0f}-{bounds[1]:.0f}, Dec {bounds[2]:.0f}-{bounds[3]:.0f}) {status}")
        else:
            logger.info(f"  {i+1:3d}. {os.path.basename(path)} {status}")
    logger.info("=" * 70)
    
    if args.dry_run:
        logger.info("")
        logger.info("DRY RUN MODE - No files will be downloaded or processed.")
        logger.info("Remove --dry-run to execute.")
        return
    
    # -------------------------------------------------------------------------
    # Step 3: Process sweeps and build parent catalogs
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("-" * 70)
    logger.info("Step 3: Processing sweeps and extracting LRGs")
    logger.info("-" * 70)
    
    # Initialize result containers for each mode
    mode_results: Dict[str, List[pd.DataFrame]] = {mode: [] for mode in modes}
    mode_lrg_counts: Dict[str, int] = {mode: 0 for mode in modes}
    
    start_time = time.time()
    total_lrgs_found = 0
    sweeps_with_data = 0
    
    for i, sweep_path in enumerate(overlapping_sweeps):
        elapsed = time.time() - start_time
        
        # Progress logging with ETA
        if i > 0:
            avg_time = elapsed / i
            remaining = avg_time * (len(overlapping_sweeps) - i)
            logger.info(f"[{i+1}/{len(overlapping_sweeps)}] {os.path.basename(sweep_path)} "
                       f"[{format_time(elapsed)} elapsed, ~{format_time(remaining)} remaining]")
        else:
            logger.info(f"[{i+1}/{len(overlapping_sweeps)}] {os.path.basename(sweep_path)}")
        
        # Download/cache the file
        try:
            local_path = download_with_cache(
                sweep_path, 
                cache_dir, 
                force_download=args.force_download
            )
        except Exception as e:
            logger.warning(f"  Failed to download {sweep_path}: {e}")
            continue
        
        # Load FITS data
        try:
            df = load_sweep_fits(local_path)
        except Exception as e:
            logger.warning(f"  Failed to read {local_path}: {e}")
            continue
        
        # Filter to target bricks
        if "BRICKNAME" not in df.columns:
            logger.warning(f"  No BRICKNAME column in {local_path}")
            continue
        
        mask = df["BRICKNAME"].astype(str).isin(all_bricknames)
        df_filtered = df.loc[mask]
        
        if df_filtered.empty:
            logger.debug(f"  No objects in target bricks")
            continue
        
        # Compute LRG flags and filter to v3
        df_lrg = compute_lrg_flags_inplace(df_filtered)
        
        if df_lrg.empty:
            logger.debug(f"  No v3 LRGs found")
            continue
        
        sweeps_with_data += 1
        n_lrg_this_sweep = len(df_lrg)
        total_lrgs_found += n_lrg_this_sweep
        
        # Assign LRGs to modes based on their brickname
        for mode in modes:
            # Find which bricks in this mode are present
            mode_brick_set = set(mode_bricks[mode][
                next(c for c in ["brickname", "BRICKNAME"] if c in mode_bricks[mode].columns)
            ].astype(str))
            
            mode_mask = df_lrg["BRICKNAME"].astype(str).isin(mode_brick_set)
            mode_df = df_lrg.loc[mode_mask].copy()
            
            if mode_df.empty:
                continue
            
            # Add mode-specific metadata
            mode_df["phase3_ranking_mode"] = mode
            
            # Add region_id and rank from brick mapping
            def get_mode_info(bn: str, m: str):
                infos = brick_to_mode_info.get(bn, [])
                for mode_name, region_id, rank in infos:
                    if mode_name == m:
                        return region_id, rank
                return None, None
            
            region_ids = []
            ranks = []
            for bn in mode_df["BRICKNAME"].astype(str):
                rid, rank = get_mode_info(bn, mode)
                region_ids.append(rid)
                ranks.append(rank)
            
            mode_df["region_id"] = region_ids
            mode_df["phase3_region_rank"] = ranks
            
            mode_results[mode].append(mode_df)
            mode_lrg_counts[mode] += len(mode_df)
        
        logger.info(f"  ✓ Found {n_lrg_this_sweep} v3 LRGs (total: {total_lrgs_found})")
    
    # -------------------------------------------------------------------------
    # Step 4: Write output catalogs
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("-" * 70)
    logger.info("Step 4: Writing output catalogs")
    logger.info("-" * 70)
    
    for mode in modes:
        mode_output_dir = os.path.join(output_base, mode)
        os.makedirs(mode_output_dir, exist_ok=True)
        
        if not mode_results[mode]:
            logger.warning(f"  Mode '{mode}': No LRGs found, skipping output.")
            continue
        
        parent_df = pd.concat(mode_results[mode], ignore_index=True)
        
        csv_path = os.path.join(mode_output_dir, "phase3_lrg_parent_catalog.csv")
        parent_df.to_csv(csv_path, index=False)
        logger.info(f"  Mode '{mode}': {len(parent_df)} LRGs -> {csv_path}")
        
        if args.save_parquet:
            try:
                parquet_path = os.path.join(mode_output_dir, "phase3_lrg_parent_catalog.parquet")
                parent_df.to_parquet(parquet_path, index=False)
                logger.info(f"  Mode '{mode}': Parquet -> {parquet_path}")
            except Exception as e:
                logger.warning(f"  Mode '{mode}': Failed to save Parquet: {e}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    total_time = time.time() - start_time
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Total time                : {format_time(total_time)}")
    logger.info(f"  Sweeps processed          : {len(overlapping_sweeps)}")
    logger.info(f"  Sweeps with data          : {sweeps_with_data}")
    logger.info(f"  Total v3 LRGs found       : {total_lrgs_found}")
    logger.info("")
    logger.info("  LRGs per mode:")
    for mode in modes:
        logger.info(f"    {mode}: {mode_lrg_counts[mode]}")
    logger.info("")
    logger.info(f"  Output directory: {output_base}")
    logger.info("=" * 70)
    logger.info("Done.")


if __name__ == "__main__":
    main()

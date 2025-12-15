#!/usr/bin/env python3
"""
Phase 3 - Build LRG parent catalog in selected regions

This script uses the brick level region definitions from Phase 3 field
selection and the DR10 sweep files to build an object level parent catalog
of LRGs according to the v3_color_relaxed selection.

Outputs:
- results/phase3/phase3_lrg_parent_catalog.csv

One row per v3 LRG in the Phase 3 bricks, with photometry, colors,
morphology (if available), region_id, and selection variant flags.

Important:
- The LRG selection cuts for v1 through v5 are kept strictly identical to
  those in spark_phase2_lrg_hypergrid.py. Do not change them here without
  also changing Phase 2 and re-running the hypergrid and analysis.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Set

import numpy as np
import pandas as pd
from astropy.io import fits


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Phase 3 LRG parent catalog builder")

    parser.add_argument(
        "--phase3-bricks-csv",
        default="results/phase3/phase3_target_bricks.csv",
        help="CSV listing bricks and regions selected for Phase 3",
    )
    parser.add_argument(
        "--sweep-index-path",
        required=True,
        help=(
            "Path to a text file containing a list of DR10 sweep FITS URLs or "
            "filesystem paths, one per line. This should be the same list used "
            "for the Phase 1.5 and Phase 2 EMR jobs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/phase3",
        help="Directory where phase3_lrg_parent_catalog.* will be written",
    )
    parser.add_argument(
        "--ra-min",
        type=float,
        default=None,
        help="Optional RA min for Phase 3 footprint. If omitted, derived from bricks.",
    )
    parser.add_argument(
        "--ra-max",
        type=float,
        default=None,
        help="Optional RA max for Phase 3 footprint. If omitted, derived from bricks.",
    )
    parser.add_argument(
        "--dec-min",
        type=float,
        default=None,
        help="Optional Dec min for Phase 3 footprint. If omitted, derived from bricks.",
    )
    parser.add_argument(
        "--dec-max",
        type=float,
        default=None,
        help="Optional Dec max for Phase 3 footprint. If omitted, derived from bricks.",
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

    return parser.parse_args(argv)


def nanomaggies_to_mag(flux: np.ndarray) -> np.ndarray:
    """
    Convert nanomaggies to AB magnitudes.

    Non positive or non finite flux values are mapped to NaN magnitudes.
    """
    flux = np.asarray(flux, dtype=float)
    mag = np.full_like(flux, np.nan, dtype=float)

    mask = np.isfinite(flux) & (flux > 0)
    mag[mask] = 22.5 - 2.5 * np.log10(flux[mask])
    return mag


def read_sweep_paths(index_path: str) -> List[str]:
    """
    Read the sweep FITS paths or URLs from a text file.

    The file should have one path or URL per line.
    """
    with open(index_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def derive_footprint_from_bricks(bricks: pd.DataFrame, args) -> tuple:
    """
    Use the Phase 3 bricks to derive RA and Dec footprint bounds if
    the user did not specify them explicitly.
    """
    # Handle column name variations from Phase 2 output
    ra_col = None
    for col in ["ra", "brick_ra_center", "RA"]:
        if col in bricks.columns:
            ra_col = col
            break
    dec_col = None
    for col in ["dec", "brick_dec_center", "DEC"]:
        if col in bricks.columns:
            dec_col = col
            break
    
    if ra_col is None or dec_col is None:
        raise ValueError(
            f"Phase 3 bricks CSV must contain RA and Dec columns. "
            f"Available columns: {list(bricks.columns)}"
        )

    ra_min = bricks[ra_col].min()
    ra_max = bricks[ra_col].max()
    dec_min = bricks[dec_col].min()
    dec_max = bricks[dec_col].max()

    pad_ra = 0.2
    pad_dec = 0.2

    ra_min -= pad_ra
    ra_max += pad_ra
    dec_min -= pad_dec
    dec_max += pad_dec

    if args.ra_min is not None:
        ra_min = args.ra_min
    if args.ra_max is not None:
        ra_max = args.ra_max
    if args.dec_min is not None:
        dec_min = args.dec_min
    if args.dec_max is not None:
        dec_max = args.dec_max

    return ra_min, ra_max, dec_min, dec_max


def get_sweep_bounds(sweep_name: str) -> Optional[tuple]:
    """
    Parse sweep filename to extract RA/Dec bounds.
    Returns (ra_min, ra_max, dec_min, dec_max) or None if parsing fails.
    """
    base = os.path.basename(sweep_name)
    if not base.startswith("sweep-") or not base.endswith(".fits"):
        return None

    try:
        core = base[len("sweep-") : -len(".fits")]
        part1, part2 = core.split("-")

        def parse_part(part: str) -> tuple:
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


def sweep_overlaps_bricks(sweep_name: str, brick_positions: List[tuple]) -> bool:
    """
    Check if a sweep overlaps any of the actual brick positions.
    brick_positions is a list of (ra, dec) tuples.
    
    This is more precise than bounding-box overlap when bricks are scattered.
    """
    bounds = get_sweep_bounds(sweep_name)
    if bounds is None:
        return True  # Can't parse, assume overlap
    
    swe_ra_min, swe_ra_max, swe_dec_min, swe_dec_max = bounds
    
    # Check if any brick center is inside the sweep bounds
    for ra, dec in brick_positions:
        if swe_ra_min <= ra <= swe_ra_max and swe_dec_min <= dec <= swe_dec_max:
            return True
    
    return False


def sweep_overlaps_footprint(sweep_name: str, ra_min: float, ra_max: float, dec_min: float, dec_max: float) -> bool:
    """
    Heuristic check whether a sweep tile likely overlaps the Phase 3 RA Dec footprint,
    based on its name.

    Assumes standard DR10 sweep naming convention:
      sweep-<RRR><p/m><DDD>-<RRR><p/m><DDD>.fits
      
    Where RRR is RA in degrees (000-360) and DDD is Dec in degrees (000-090).
    The sign character 'p' means positive Dec, 'm' means negative Dec.
    
    Example: sweep-035p015-040p020.fits → RA 35-40°, Dec +15° to +20°

    This is a performance optimization to skip obviously unrelated sweeps.
    If the name cannot be parsed, this returns True and later RA Dec filters
    handle any non overlapping data.
    """
    base = os.path.basename(sweep_name)
    if not base.startswith("sweep-") or not base.endswith(".fits"):
        return True

    try:
        core = base[len("sweep-") : -len(".fits")]
        part1, part2 = core.split("-")

        def parse_part(part: str) -> tuple:
            # Part format: RRR[pm]DDD, e.g., "035p015" or "320m025"
            ra_str = part[:3]
            sign_char = part[3]
            dec_str = part[4:]
            
            # RA and Dec are in whole degrees, NOT divided by 10
            ra = float(ra_str)
            dec = float(dec_str)
            
            if sign_char == "m":
                dec = -dec
            # 'p' means positive (no change needed)
            
            return ra, dec

        ra1, dec1 = parse_part(part1)
        ra2, dec2 = parse_part(part2)

        swe_ra_min = min(ra1, ra2)
        swe_ra_max = max(ra1, ra2)
        swe_dec_min = min(dec1, dec2)
        swe_dec_max = max(dec1, dec2)

        overlaps = not (
            (swe_ra_max < ra_min)
            or (swe_ra_min > ra_max)
            or (swe_dec_max < dec_min)
            or (swe_dec_min > dec_max)
        )
        return overlaps
    except Exception:
        return True


def load_sweep_fits(path_or_url: str) -> pd.DataFrame:
    """
    Load a DR10 sweep FITS file into a Pandas DataFrame.

    This assumes the FITS file has a main table extension with
    columns such as RA, DEC, FLUX_G, FLUX_R, FLUX_Z, FLUX_W1, TYPE, BRICKNAME.

    Only loads the columns needed for Phase 3 to minimize memory usage.
    """
    # Columns we need for Phase 3
    needed_cols = ["RA", "DEC", "BRICKNAME", "FLUX_G", "FLUX_R", "FLUX_Z", "FLUX_W1", "TYPE"]
    
    with fits.open(path_or_url, memmap=True) as hdul:
        data = hdul[1].data
        
        # Extract only needed columns to avoid full data copy
        result = {}
        available_cols = data.dtype.names
        
        for col in needed_cols:
            if col in available_cols:
                # Convert to native byte order efficiently
                col_data = data[col]
                if col_data.dtype.byteorder not in ('=', '|', '<' if sys.byteorder == 'little' else '>'):
                    result[col] = np.asarray(col_data, dtype=col_data.dtype.newbyteorder('='))
                else:
                    result[col] = np.asarray(col_data)
        
        df = pd.DataFrame(result)

    return df


def filter_to_phase3_bricks(
    df: pd.DataFrame,
    bricknames: Set[str],
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
) -> pd.DataFrame:
    """
    Restrict a sweep DataFrame to objects:

    - within the Phase 3 RA Dec footprint, and
    - whose BRICKNAME is one of the Phase 3 bricks.
    """
    required_cols = ["RA", "DEC", "BRICKNAME"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' in sweep FITS table. "
                "If your schema uses a different name, update filter_to_phase3_bricks."
            )

    mask = (
        (df["RA"] >= ra_min)
        & (df["RA"] <= ra_max)
        & (df["DEC"] >= dec_min)
        & (df["DEC"] <= dec_max)
        & (df["BRICKNAME"].astype(str).isin(bricknames))
    )
    return df.loc[mask].copy()


def compute_lrg_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute magnitudes, colors, and LRG selection flags for all five
    Phase 2 variants.

    Only rows passing v3_color_relaxed will be kept in the parent catalog,
    but flags for v1, v2, v4, v5 are also attached.

    The cuts in this function are kept identical to those in
    spark_phase2_lrg_hypergrid.py:

    v1_pure_massive:
        z_mag < 20.0
        r_minus_z > 0.5
        z_minus_w1 > 1.6

    v2_baseline_dr10:
        z_mag < 20.4
        r_minus_z > 0.4
        z_minus_w1 > 1.6

    v3_color_relaxed:
        z_mag < 20.4
        r_minus_z > 0.4
        z_minus_w1 > 0.8

    v4_mag_relaxed:
        z_mag < 21.0
        r_minus_z > 0.4
        z_minus_w1 > 0.8

    v5_very_relaxed:
        z_mag < 21.5
        r_minus_z > 0.3
        z_minus_w1 > 0.8
    """
    for col in ["FLUX_G", "FLUX_R", "FLUX_Z", "FLUX_W1"]:
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' in sweep FITS table. "
                "If your schema differs, update compute_lrg_flags accordingly."
            )

    g_mag = nanomaggies_to_mag(df["FLUX_G"].values)
    r_mag = nanomaggies_to_mag(df["FLUX_R"].values)
    z_mag = nanomaggies_to_mag(df["FLUX_Z"].values)
    w1_mag = nanomaggies_to_mag(df["FLUX_W1"].values)

    r_minus_z = r_mag - z_mag
    z_minus_w1 = z_mag - w1_mag

    # v1: pure massive, stricter in magnitude and color
    is_lrg_v1 = (
        (z_mag < 20.0)
        & (r_minus_z > 0.5)
        & (z_minus_w1 > 1.6)
    )

    # v2: baseline DR10 style
    is_lrg_v2 = (
        (z_mag < 20.4)
        & (r_minus_z > 0.4)
        & (z_minus_w1 > 1.6)
    )

    # v3: color relaxed (relative to v2): same r_minus_z, but looser z_minus_w1
    is_lrg_v3 = (
        (z_mag < 20.4)
        & (r_minus_z > 0.4)
        & (z_minus_w1 > 0.8)
    )

    # v4: magnitude relaxed relative to v3
    is_lrg_v4 = (
        (z_mag < 21.0)
        & (r_minus_z > 0.4)
        & (z_minus_w1 > 0.8)
    )

    # v5: very relaxed (faints and slightly bluer in r_minus_z, but keeps z_minus_w1 > 0.8)
    is_lrg_v5 = (
        (z_mag < 21.5)
        & (r_minus_z > 0.3)
        & (z_minus_w1 > 0.8)
    )

    out = df.copy()
    out["g_mag"] = g_mag
    out["r_mag"] = r_mag
    out["z_mag"] = z_mag
    out["w1_mag"] = w1_mag
    out["r_minus_z"] = r_minus_z
    out["z_minus_w1"] = z_minus_w1

    out["is_lrg_v1"] = is_lrg_v1
    out["is_lrg_v2"] = is_lrg_v2
    out["is_lrg_v3"] = is_lrg_v3
    out["is_lrg_v4"] = is_lrg_v4
    out["is_lrg_v5"] = is_lrg_v5

    out = out[out["is_lrg_v3"]].copy()

    return out


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    bricks = pd.read_csv(args.phase3_bricks_csv)
    
    # Handle column name variations (Phase 2 uses lowercase 'brickname')
    brickname_col = None
    for col in ["brickname", "BRICKNAME"]:
        if col in bricks.columns:
            brickname_col = col
            break
    if brickname_col is None:
        raise ValueError(
            f"Phase 3 bricks CSV must contain a brickname column. "
            f"Available columns: {list(bricks.columns)}"
        )
    
    if "region_id" not in bricks.columns:
        raise ValueError("Phase 3 bricks CSV must contain 'region_id' column")

    bricknames: Set[str] = set(bricks[brickname_col].astype(str).tolist())

    ra_min, ra_max, dec_min, dec_max = derive_footprint_from_bricks(bricks, args)
    
    start_time = time.time()
    print("============================================================")
    print("Phase 3 - LRG parent catalog")
    print("============================================================")
    print(f"Using bricks CSV: {args.phase3_bricks_csv}")
    print(f"Brickname column: {brickname_col}")
    print(f"Number of target bricks: {len(bricknames)}")
    print(f"RA range:  [{ra_min:.3f}, {ra_max:.3f}] deg")
    print(f"Dec range: [{dec_min:.3f}, {dec_max:.3f}] deg")
    print(f"Sweep index: {args.sweep_index_path}")
    print("")

    sweep_paths = read_sweep_paths(args.sweep_index_path)
    print(f"Total sweeps in index: {len(sweep_paths)}")

    # Get actual brick positions for precise overlap checking
    ra_col = None
    for col in ["ra", "brick_ra_center", "RA"]:
        if col in bricks.columns:
            ra_col = col
            break
    dec_col = None
    for col in ["dec", "brick_dec_center", "DEC"]:
        if col in bricks.columns:
            dec_col = col
            break
    
    brick_positions = list(zip(bricks[ra_col].values, bricks[dec_col].values))
    
    # First pass: identify which sweeps overlap actual brick positions (not just bounding box)
    print(f"\nScanning for overlapping sweeps (using precise brick positions)...")
    overlapping_sweeps = []
    for path in sweep_paths:
        if sweep_overlaps_bricks(path, brick_positions):
            overlapping_sweeps.append(path)
    
    # Also show what bounding-box method would have matched (for comparison)
    bbox_sweeps = [p for p in sweep_paths if sweep_overlaps_footprint(p, ra_min, ra_max, dec_min, dec_max)]
    
    print(f"\n{'='*60}")
    print(f"DOWNLOAD PLAN")
    print(f"{'='*60}")
    print(f"Bounding box footprint: RA [{ra_min:.1f}, {ra_max:.1f}], Dec [{dec_min:.1f}, {dec_max:.1f}]")
    print(f"Sweeps matching bounding box: {len(bbox_sweeps)} (inefficient!)")
    print(f"Sweeps matching actual bricks: {len(overlapping_sweeps)} (optimized)")
    print(f"\nNote: Each sweep file is 100-500 MB. Estimated download: {len(overlapping_sweeps) * 0.3:.1f} GB")
    print(f"\nSweeps to download:")
    for i, path in enumerate(overlapping_sweeps):
        bounds = get_sweep_bounds(path)
        if bounds:
            print(f"  {i+1}. {os.path.basename(path)} (RA {bounds[0]:.0f}-{bounds[1]:.0f}, Dec {bounds[2]:.0f}-{bounds[3]:.0f})")
        else:
            print(f"  {i+1}. {os.path.basename(path)}")
    print(f"{'='*60}\n")
    
    if args.dry_run:
        print("Dry run mode - exiting without downloading.")
        return

    selected_rows: List[pd.DataFrame] = []
    n_sweeps_used = 0
    n_sweeps_checked = 0
    n_lrg_total = 0

    # Precompute brick -> region mapping once (handle column name)
    brick_to_region = bricks.set_index(brickname_col)["region_id"].to_dict()

    for i, path in enumerate(overlapping_sweeps):
        n_sweeps_checked += 1
        elapsed = time.time() - start_time
        
        # Progress with timing
        if n_sweeps_checked > 1:
            avg_time = elapsed / n_sweeps_checked
            remaining = avg_time * (len(overlapping_sweeps) - n_sweeps_checked)
            print(f"[{n_sweeps_checked}/{len(overlapping_sweeps)}] Reading sweep: {os.path.basename(path)} [{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")
        else:
            print(f"[{n_sweeps_checked}/{len(overlapping_sweeps)}] Reading sweep: {os.path.basename(path)}")
        
        try:
            df = load_sweep_fits(path)
        except Exception as e:
            print(f"  Warning: failed to read {path}: {e}")
            continue

        df_sub = filter_to_phase3_bricks(df, bricknames, ra_min, ra_max, dec_min, dec_max)
        if df_sub.empty:
            continue

        df_lrg = compute_lrg_flags(df_sub)
        if df_lrg.empty:
            continue

        n_sweeps_used += 1
        n_lrg_total += len(df_lrg)
        print(f"  ✓ Selected {len(df_lrg)} v3 LRGs (total: {n_lrg_total})")

        df_lrg["region_id"] = df_lrg["BRICKNAME"].astype(str).map(brick_to_region)
        selected_rows.append(df_lrg)

    if not selected_rows:
        print("No LRGs selected in Phase 3 footprint. Check region choices and cuts.")
        sys.exit(1)

    parent_df = pd.concat(selected_rows, ignore_index=True)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "phase3_lrg_parent_catalog.csv")
    parent_df.to_csv(csv_path, index=False)

    if args.save_parquet:
        try:
            import pyarrow  # type: ignore
            parquet_path = os.path.join(args.output_dir, "phase3_lrg_parent_catalog.parquet")
            parent_df.to_parquet(parquet_path, index=False)
            print(f"Saved Parquet parent catalog to: {parquet_path}")
        except Exception as e:
            print(f"Warning: failed to save Parquet file: {e}")

    total_time = time.time() - start_time
    print("============================================================")
    print("Phase 3 parent catalog completed")
    print("============================================================")
    print(f"Number of sweeps checked: {n_sweeps_checked}")
    print(f"Number of sweeps with data: {n_sweeps_used}")
    print(f"Total v3 LRGs in parent catalog: {n_lrg_total}")
    print(f"Wrote CSV parent catalog to: {csv_path}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()


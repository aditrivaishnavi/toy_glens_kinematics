"""
Region scouting module for Phase 1.5.

Queries DR10 bricks via TAP or loads from local files, and selects
high-quality regions for lens search.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from math import radians, sin
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

try:
    from astropy.io import fits
except ImportError:
    fits = None

try:
    import pyvo
    from pyvo.dal import DALQueryError
except ImportError:
    pyvo = None
    DALQueryError = Exception  # fallback for type hints

from .config import Phase1p5Config


def get_tap_service(tap_url: str) -> "pyvo.dal.TAPService":
    """Return a pyvo TAP service for the given URL."""
    if pyvo is None:
        raise ImportError(
            "pyvo is required for TAP access. Install with: pip install pyvo"
        )
    return pyvo.dal.TAPService(tap_url)


@dataclass
class BrickRecord:
    """
    Record for a single DR10 brick with QA metrics.
    
    Column names match the ls_dr10.bricks_s schema exactly.
    """
    brickname: str
    ra: float
    dec: float
    ra1: float
    ra2: float
    dec1: float
    dec2: float
    psfsize_g: float
    psfsize_r: float
    psfsize_z: float
    psfdepth_g: float
    psfdepth_r: float
    psfdepth_z: float
    ebv: float
    area_deg2: float
    lrg_count: int = 0
    lrg_density: float = 0.0

    @property
    def center_ra_deg(self) -> float:
        return float(self.ra)

    @property
    def center_dec_deg(self) -> float:
        return float(self.dec)


def _compute_brick_area_deg2(ra1: float, ra2: float, dec1: float, dec2: float) -> float:
    """
    Compute the approximate spherical area of a brick in deg^2 using its RA/Dec bounds.

    Uses the formula:
      area = (ra2 - ra1) * (sin(dec2) - sin(dec1))   [in radians^2]
      then converts to deg^2.

    This is more accurate than assuming a fixed 0.25° × 0.25° for all bricks.
    """
    ra1_rad = radians(ra1)
    ra2_rad = radians(ra2)
    dec1_rad = radians(dec1)
    dec2_rad = radians(dec2)

    d_ra = ra2_rad - ra1_rad
    d_sin_dec = sin(dec2_rad) - sin(dec1_rad)
    area_sr = abs(d_ra * d_sin_dec)  # steradians

    # 1 sr = (180/pi)^2 deg^2
    steradian_to_deg2 = (180.0 / np.pi) ** 2
    return float(area_sr * steradian_to_deg2)


def _ra_delta_deg(ra1: float, ra2: float) -> float:
    """
    Smallest RA difference in degrees on the sphere, accounting for wrap at 0/360.
    """
    d = abs(ra1 - ra2)
    return min(d, 360.0 - d)


# ---------------------------------------------------------------------------
# Checkpoint helpers for resumable LRG density estimation
# ---------------------------------------------------------------------------


def _load_density_checkpoint(checkpoint_path: Path) -> set:
    """
    Load already-processed brick names from a checkpoint CSV.

    Returns a set of brickname strings that have been successfully processed.
    If the file does not exist, returns an empty set.
    """
    if not checkpoint_path.exists():
        return set()

    try:
        df = pd.read_csv(checkpoint_path)
        if "brickname" in df.columns:
            return set(df["brickname"].astype(str).values)
        return set()
    except Exception as e:
        print(f"[CHECKPOINT] Warning: Could not read checkpoint file: {e}", flush=True)
        return set()


def _append_density_checkpoint(
    checkpoint_path: Path,
    brickname: str,
    lrg_count: int,
    lrg_density: float,
    area_deg2: float,
) -> None:
    """
    Append one brick's LRG density result to the checkpoint CSV.

    Creates the file with a header if it doesn't exist.
    Flushes immediately to ensure data is persisted.
    """
    write_header = not checkpoint_path.exists()

    with open(checkpoint_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("brickname,lrg_count,lrg_density,area_deg2\n")
        f.write(f"{brickname},{lrg_count},{lrg_density:.6f},{area_deg2:.6f}\n")
        f.flush()


def _load_checkpoint_as_dataframe(checkpoint_path: Path) -> pd.DataFrame:
    """
    Load checkpoint CSV as a DataFrame for merging into final results.

    Returns an empty DataFrame if the file doesn't exist.
    """
    if not checkpoint_path.exists():
        return pd.DataFrame(columns=["brickname", "lrg_count", "lrg_density", "area_deg2"])

    try:
        return pd.read_csv(checkpoint_path)
    except Exception as e:
        print(f"[CHECKPOINT] Warning: Could not read checkpoint as DataFrame: {e}", flush=True)
        return pd.DataFrame(columns=["brickname", "lrg_count", "lrg_density", "area_deg2"])


def fetch_bricks(config: Phase1p5Config) -> pd.DataFrame:
    """
    Fetch DR10 bricks in the scouting footprint using the correct column names
    from ls_dr10.bricks_s.

    Uses synchronous TAP (run_sync) because the NOIRLab TAP async/UWS
    implementation does not provide a result URI compatible with pyvo's
    async job API.

    We work at the brick level (native survey unit) and pull only robust QA
    columns: seeing, depth, and extinction.

    Returns
    -------
    pandas.DataFrame
        One row per brick with columns:
        ['brickname', 'ra', 'dec', 'ra1', 'ra2', 'dec1', 'dec2',
         'psfsize_g', 'psfsize_r', 'psfsize_z',
         'psfdepth_g', 'psfdepth_r', 'psfdepth_z',
         'ebv', 'area_deg2']
    """
    import time
    
    print(f"    Connecting to TAP service: {config.tap_url}", flush=True)
    tap = get_tap_service(config.tap_url)
    print("    TAP service connected.", flush=True)

    # Column names match ls_dr10.bricks_s exactly
    query = f"""
        SELECT
            brickname,
            ra,
            dec,
            ra1,
            ra2,
            dec1,
            dec2,
            psfsize_g,
            psfsize_r,
            psfsize_z,
            psfdepth_g,
            psfdepth_r,
            psfdepth_z,
            ebv
        FROM {config.bricks_table}
        WHERE ra  BETWEEN {config.ra_min}  AND {config.ra_max}
          AND dec BETWEEN {config.dec_min} AND {config.dec_max}
    """

    print(f"    Querying {config.bricks_table}...", flush=True)
    print(f"    Footprint: RA [{config.ra_min}, {config.ra_max}], Dec [{config.dec_min}, {config.dec_max}]", flush=True)
    
    start_time = time.time()
    try:
        result = tap.run_sync(query)
    except DALQueryError as exc:
        raise RuntimeError(
            f"Failed to fetch bricks from {config.bricks_table} via TAP: {exc}"
        ) from exc
    
    elapsed = time.time() - start_time
    print(f"    Query completed in {elapsed:.1f} seconds.", flush=True)

    try:
        table = result.to_table()
    except AttributeError:
        table = result
    df = table.to_pandas()

    # Sanity check: ensure expected columns are present
    expected_cols = [
        "brickname",
        "ra",
        "dec",
        "ra1",
        "ra2",
        "dec1",
        "dec2",
        "psfsize_g",
        "psfsize_r",
        "psfsize_z",
        "psfdepth_g",
        "psfdepth_r",
        "psfdepth_z",
        "ebv",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Bricks query returned missing columns: {missing}. "
            f"Got columns: {list(df.columns)}"
        )

    if len(df) == 0:
        raise RuntimeError(
            "Bricks query returned zero rows. "
            "Check the footprint (RA/Dec) and table name in Phase1p5Config."
        )

    print(f"    Retrieved {len(df)} bricks in scouting footprint.")

    # Compute brick area from true RA/Dec bounds
    df["area_deg2"] = df.apply(
        lambda row: _compute_brick_area_deg2(
            float(row["ra1"]),
            float(row["ra2"]),
            float(row["dec1"]),
            float(row["dec2"]),
        ),
        axis=1,
    )

    return df


def load_bricks_from_local_fits(config: Phase1p5Config) -> pd.DataFrame:
    """
    Load DR10 bricks from a local survey-bricks FITS file and apply
    the RA/Dec footprint filter.

    This avoids TAP for bricks and uses the same columns as fetch_bricks.
    """
    if fits is None:
        raise ImportError(
            "astropy is required to load local FITS files. "
            "Install with: pip install astropy"
        )

    bricks_path = Path(config.bricks_fits)
    if not bricks_path.exists():
        raise FileNotFoundError(
            f"Local bricks FITS not found at {bricks_path}. "
            "Run scripts/download_dr10_region_data.py first or set use_local_bricks=False."
        )

    print(f"    Loading bricks from local FITS: {bricks_path}", flush=True)
    with fits.open(bricks_path) as hdul:
        # survey-bricks table is in the first extension
        table = hdul[1].data
        df = pd.DataFrame(table.byteswap().newbyteorder())

    # Normalize column names to lower case to be robust
    df.columns = [c.lower() for c in df.columns]

    # Sanity check
    required = {
        "brickname",
        "ra",
        "dec",
        "ra1",
        "ra2",
        "dec1",
        "dec2",
        "psfsize_g",
        "psfsize_r",
        "psfsize_z",
        "psfdepth_g",
        "psfdepth_r",
        "psfdepth_z",
        "ebv",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            f"Local bricks FITS is missing columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Apply RA/Dec footprint
    mask = (
        (df["ra"] >= config.ra_min)
        & (df["ra"] <= config.ra_max)
        & (df["dec"] >= config.dec_min)
        & (df["dec"] <= config.dec_max)
    )
    df = df.loc[mask].copy()
    if df.empty:
        raise RuntimeError(
            "Local bricks selection returned zero rows. "
            "Check RA/Dec footprint in Phase1p5Config."
        )

    # Compute area_deg2 as before
    df["area_deg2"] = df.apply(
        lambda row: _compute_brick_area_deg2(
            float(row["ra1"]),
            float(row["ra2"]),
            float(row["dec1"]),
            float(row["dec2"]),
        ),
        axis=1,
    )

    print(f"    Loaded {len(df)} bricks from local FITS in scouting footprint.", flush=True)
    return df


def apply_brick_quality_cuts(df: pd.DataFrame, config: Phase1p5Config) -> pd.DataFrame:
    """
    Apply hard, physically motivated quality cuts:
      - seeing (psfsize_r)
      - depth (psfdepth_r)
      - extinction (ebv)
    
    Note: nexp_* columns are not always available in the bricks table,
    so we skip that cut here. Can be added later if schema is confirmed.
    """
    mask = (
        (df["psfsize_r"] <= config.max_psfsize_r)
        & (df["psfdepth_r"] >= config.min_psfdepth_r)
        & (df["ebv"] <= config.max_ebv)
    )
    return df.loc[mask].copy()


def _nanomaggies_to_mag(flux: np.ndarray, zero_point: float = 22.5) -> np.ndarray:
    """
    Convert nanomaggies to AB magnitudes using the Legacy Surveys convention:

      mag = zero_point - 2.5 * log10(flux)

    For flux <= 0, returns NaN.
    """
    flux = np.array(flux, dtype=float)
    mag = np.full_like(flux, np.nan, dtype=float)
    positive = flux > 0
    mag[positive] = zero_point - 2.5 * np.log10(flux[positive])
    return mag


def _build_lrg_mask(
    df: pd.DataFrame,
    config: Phase1p5Config,
) -> np.ndarray:
    """
    DESI-like LRG proxy mask using magnitudes and colors.

    We work in AB magnitudes derived from Tractor fluxes (nanomaggies).
    We do NOT apply an explicit extinction correction here; instead,
    Phase 1.5 restricts to low-EBV bricks so that relative LRG densities
    are comparable across the footprint.

    Steps:
      1. Convert fluxes (nanomaggies) to AB mags.
      2. Compute colors: r-z, z-W1.
      3. Apply cuts:
         - z < lrg_z_mag_max
         - r - z > lrg_min_r_minus_z
         - z - W1 > lrg_min_z_minus_w1
         - Optional: photo_z in range (only if column exists AND use_photo_z=True)

    Note: We skip MW extinction correction here because mw_transmission_*
    columns may not be available. For brick-ranking purposes, this is acceptable
    since we are comparing relative densities across similar-extinction regions.
    """
    # Convert fluxes to magnitudes
    # Legacy Surveys convention: mag = 22.5 - 2.5 * log10(flux)
    mag_r = _nanomaggies_to_mag(df["flux_r"].values)
    mag_z = _nanomaggies_to_mag(df["flux_z"].values)
    mag_w1 = _nanomaggies_to_mag(df["flux_w1"].values)

    r_minus_z = mag_r - mag_z
    z_minus_w1 = mag_z - mag_w1

    mask = (
        (mag_z < config.lrg_z_mag_max)
        & (r_minus_z > config.lrg_min_r_minus_z)
        & (z_minus_w1 > config.lrg_min_z_minus_w1)
    )

    # Optional photo-z cut (only if column exists AND flag is True)
    if config.use_photo_z and "photo_z" in df.columns:
        z_phot = df["photo_z"].values
        mask &= (
            (z_phot >= config.lrg_z_phot_min)
            & (z_phot <= config.lrg_z_phot_max)
        )

    return mask


def estimate_lrg_density_for_bricks(
    bricks_df: pd.DataFrame,
    config: Phase1p5Config,
    pilot_limit: int = 0,
    checkpoint_path: Path = None,
) -> pd.DataFrame:
    """
    Estimate LRG density per brick.

    If config.use_sweeps_for_lrg is True, use DR10 SWEEP files in a streaming,
    mildly parallel way that does not accumulate large FITS files on disk.

    Otherwise, fall back to the TAP-based ls_dr10.tractor_s query method.

    Parameters
    ----------
    bricks_df : pd.DataFrame
        DataFrame of bricks with quality cuts applied.
    config : Phase1p5Config
        Configuration object.
    pilot_limit : int, optional
        If > 0, limit to this many bricks for testing.
    checkpoint_path : Path, optional
        If provided, enables resumable processing. Already-processed bricks
        are skipped, and results are saved incrementally.
    """
    # Prefer SWEEP-based method if requested
    if getattr(config, "use_sweeps_for_lrg", False):
        from .sweep_lrg_density import estimate_lrg_density_from_sweeps as sweep_estimate

        print("[LRG] Using SWEEP-based LRG density estimation", flush=True)
        return sweep_estimate(
            bricks_df=bricks_df,
            config=config,
            pilot_limit=pilot_limit,
        )

    # Fallback: existing TAP-based logic
    return _estimate_lrg_density_via_tap(bricks_df, config, pilot_limit, checkpoint_path)


def _estimate_lrg_density_via_tap(
    bricks_df: pd.DataFrame,
    config: Phase1p5Config,
    pilot_limit: int = 0,
    checkpoint_path: Path = None,
) -> pd.DataFrame:
    """
    TAP-based LRG density estimation with optional checkpointing.

    For up to max_bricks_for_lrg_density bricks, query the Tractor catalog and
    estimate LRG density (number per deg^2) using a DESI-like proxy.

    If checkpoint_path is provided, already-processed bricks are skipped
    and results are saved incrementally after each brick.
    """
    import time

    print("[LRG] Using TAP-based LRG density estimation", flush=True)

    # Load checkpoint if available
    completed_bricks = set()
    if checkpoint_path is not None:
        completed_bricks = _load_density_checkpoint(checkpoint_path)
        if completed_bricks:
            print(f"[CHECKPOINT] Loaded {len(completed_bricks)} already-processed bricks", flush=True)
        else:
            print("[CHECKPOINT] No existing checkpoint found, starting fresh", flush=True)

    print("[LRG] Connecting to TAP service...", flush=True)
    tap = get_tap_service(config.tap_url)
    print("[LRG] TAP service connected.", flush=True)

    # For reproducibility, sort bricks by brickname and take the first N
    max_bricks = pilot_limit if pilot_limit > 0 else config.max_bricks_for_lrg_density
    bricks_for_density = bricks_df.sort_values("brickname").head(max_bricks)
    n_bricks_total = len(bricks_for_density)

    # Filter out already-completed bricks
    bricks_to_process = bricks_for_density[
        ~bricks_for_density["brickname"].astype(str).isin(completed_bricks)
    ]
    n_remaining = len(bricks_to_process)
    n_skipped = n_bricks_total - n_remaining

    if pilot_limit > 0:
        print(f"[LRG] PILOT MODE: limiting to {n_bricks_total} bricks for testing", flush=True)

    if n_skipped > 0:
        print(f"[CHECKPOINT] Skipping {n_skipped} already-completed bricks", flush=True)

    print(f"[LRG] Will process {n_remaining} bricks (of {n_bricks_total} total)", flush=True)
    print(f"[LRG] Tractor table: {config.tractor_table}", flush=True)
    print("[LRG] " + "=" * 50, flush=True)

    failed_bricks = []
    start_time = time.perf_counter()
    processed_count = 0

    for idx, (_, row) in enumerate(bricks_to_process.iterrows()):
        brickname = str(row["brickname"])
        area_deg2 = float(row["area_deg2"])

        processed_count += 1
        print(f"[LRG] {processed_count}/{n_remaining} STARTING brick {brickname}...", flush=True)
        t0 = time.perf_counter()

        query = f"""
            SELECT
                ra, dec,
                flux_g, flux_r, flux_z, flux_w1
            FROM {config.tractor_table}
            WHERE brickname = '{brickname}'
              AND brick_primary = 1
              AND flux_r > 0
              AND flux_z > 0
              AND flux_w1 > 0
        """

        try:
            result = tap.run_sync(query)
            t1 = time.perf_counter()
            query_time = t1 - t0

            if query_time > 60.0:
                print(
                    f"[LRG] {processed_count}/{n_remaining} WARNING: query took {query_time:.1f}s (>60s)",
                    flush=True,
                )

            try:
                table = result.to_table()
            except AttributeError:
                table = result
            tdf = table.to_pandas()

        except Exception as e:
            t1 = time.perf_counter()
            print(f"[LRG] {processed_count}/{n_remaining} FAILED brick {brickname}: {e!r}", flush=True)
            print(f"[LRG] {processed_count}/{n_remaining} (failed after {t1-t0:.1f}s)", flush=True)
            failed_bricks.append(brickname)
            # Checkpoint failed bricks with 0 counts so we don't retry them
            if checkpoint_path is not None:
                _append_density_checkpoint(checkpoint_path, brickname, 0, 0.0, area_deg2)
            continue

        if tdf.empty:
            n_obj = 0
            n_lrg = 0
            density = 0.0
        else:
            n_obj = len(tdf)
            lrg_mask = _build_lrg_mask(tdf, config)
            n_lrg = int(np.count_nonzero(lrg_mask))
            density = n_lrg / area_deg2 if area_deg2 > 0 else 0.0

        # Checkpoint immediately after processing
        if checkpoint_path is not None:
            _append_density_checkpoint(checkpoint_path, brickname, n_lrg, density, area_deg2)

        t1 = time.perf_counter()
        brick_elapsed = t1 - t0
        total_elapsed = t1 - start_time
        avg_per_brick = total_elapsed / processed_count
        remaining_bricks = n_remaining - processed_count
        eta_seconds = avg_per_brick * remaining_bricks

        print(
            f"[LRG] {processed_count}/{n_remaining} FINISHED brick {brickname}: "
            f"{n_obj} obj, {n_lrg} LRGs, {brick_elapsed:.1f}s "
            f"(ETA: {eta_seconds/60:.1f} min)",
            flush=True,
        )

    print("[LRG] " + "=" * 50, flush=True)
    total_time = time.perf_counter() - start_time
    print(f"[LRG] Completed {processed_count} bricks in {total_time/60:.1f} minutes", flush=True)
    if failed_bricks:
        print(f"[LRG] Failed bricks ({len(failed_bricks)}): {failed_bricks[:10]}...", flush=True)

    # Load all checkpoint data and merge with bricks_df
    if checkpoint_path is not None:
        checkpoint_df = _load_checkpoint_as_dataframe(checkpoint_path)
        # Ensure brickname is string type for merge
        checkpoint_df["brickname"] = checkpoint_df["brickname"].astype(str)
        merged = bricks_df.copy()
        merged["brickname"] = merged["brickname"].astype(str)
        merged = merged.merge(
            checkpoint_df[["brickname", "lrg_count", "lrg_density"]],
            on="brickname",
            how="left",
        )
    else:
        # No checkpoint: build results from what we just processed
        # (This path is for backwards compatibility if checkpoint_path is None)
        lrg_counts = []
        lrg_densities = []
        for _, row in bricks_for_density.iterrows():
            brickname = str(row["brickname"])
            if brickname in completed_bricks:
                lrg_counts.append(0)
                lrg_densities.append(0.0)
            else:
                lrg_counts.append(0)
                lrg_densities.append(0.0)
        bricks_for_density = bricks_for_density.copy()
        bricks_for_density["lrg_count"] = lrg_counts
        bricks_for_density["lrg_density"] = lrg_densities
        merged = bricks_df.merge(
            bricks_for_density[["brickname", "lrg_count", "lrg_density"]],
            on="brickname",
            how="left",
        )

    merged["lrg_count"] = merged["lrg_count"].fillna(0).astype(int)
    merged["lrg_density"] = merged["lrg_density"].fillna(0.0)

    print(f"    Completed LRG density estimation for {n_bricks_total} bricks.")
    return merged


def estimate_lrg_density_from_sweeps(
    bricks_df: pd.DataFrame,
    config: Phase1p5Config,
) -> pd.DataFrame:
    """
    Estimate LRG density per brick using local SWEEP FITS files.

    This processes one SWEEP file at a time, accumulates LRG counts per brick,
    and never needs all SWEEPs in memory at once.

    Assumptions:
      - SWEEP FITS files live under config.sweeps_dir
      - Each SWEEP has columns RA, DEC, FLUX_G/R/Z/W1 and BRICKNAME (case-insensitive)
    """
    if fits is None:
        raise ImportError(
            "astropy is required to load local FITS files. "
            "Install with: pip install astropy"
        )

    sweeps_dir = Path(config.sweeps_dir)
    if not sweeps_dir.exists():
        raise FileNotFoundError(
            f"SWEEP directory {sweeps_dir} does not exist. "
            "Run scripts/download_dr10_region_data.py first or set use_local_sweeps=False."
        )

    sweep_files = sorted(sweeps_dir.glob("sweep-*.fits"))
    if not sweep_files:
        raise RuntimeError(
            f"No SWEEP FITS files found under {sweeps_dir}. "
            "Download a subset first."
        )

    print(f"[LRG-LOCAL] Using {len(sweep_files)} SWEEP files from {sweeps_dir}", flush=True)

    # Map brickname -> LRG count
    lrg_counts: Dict[str, int] = {}

    # Precompute a lookup of brickname -> area for density calculation
    brick_area = {
        str(row["brickname"]): float(row["area_deg2"])
        for _, row in bricks_df.iterrows()
    }

    for i, sweep_path in enumerate(sweep_files, start=1):
        print(f"[LRG-LOCAL] {i}/{len(sweep_files)} processing {sweep_path.name}...", flush=True)
        with fits.open(sweep_path, memmap=True) as hdul:
            data = hdul[1].data
            tdf = pd.DataFrame(data.byteswap().newbyteorder())

        # Normalize column names to lower case
        tdf.columns = [c.lower() for c in tdf.columns]

        # Identify brick column
        if "brickname" not in tdf.columns:
            print(
                f"[LRG-LOCAL] WARNING: SWEEP file {sweep_path.name} missing 'brickname', skipping",
                flush=True
            )
            continue

        # Basic flux selection
        required_flux_cols = ["flux_r", "flux_z", "flux_w1"]
        missing_cols = [c for c in required_flux_cols if c not in tdf.columns]
        if missing_cols:
            print(
                f"[LRG-LOCAL] WARNING: SWEEP file {sweep_path.name} missing {missing_cols}, skipping",
                flush=True
            )
            continue

        # Optional RA/Dec footprint refinement
        if "ra" in tdf.columns and "dec" in tdf.columns:
            mask_sky = (
                (tdf["ra"] >= config.ra_min)
                & (tdf["ra"] <= config.ra_max)
                & (tdf["dec"] >= config.dec_min)
                & (tdf["dec"] <= config.dec_max)
            )
            tdf = tdf.loc[mask_sky]

        if tdf.empty:
            if config.delete_sweeps_after_use:
                try:
                    os.remove(sweep_path)
                except OSError:
                    pass
            continue

        # Build LRG mask and group by brickname
        lrg_mask = _build_lrg_mask(tdf, config)
        if not np.any(lrg_mask):
            if config.delete_sweeps_after_use:
                try:
                    os.remove(sweep_path)
                except OSError:
                    pass
            continue

        tdf_lrg = tdf.loc[lrg_mask, ["brickname"]]
        grouped = tdf_lrg.groupby("brickname").size()

        # Accumulate counts
        for brickname, count in grouped.items():
            key = str(brickname)
            lrg_counts[key] = lrg_counts.get(key, 0) + int(count)

        n_lrg_this_sweep = int(lrg_mask.sum())
        print(
            f"[LRG-LOCAL] {i}/{len(sweep_files)} found {n_lrg_this_sweep} LRGs in {sweep_path.name}",
            flush=True
        )

        if config.delete_sweeps_after_use:
            try:
                os.remove(sweep_path)
                print(f"[LRG-LOCAL] Deleted {sweep_path.name} after use", flush=True)
            except OSError as e:
                print(f"[LRG-LOCAL] Warning: could not delete {sweep_path}: {e!r}", flush=True)

    # Attach LRG counts and densities back to bricks_df
    df = bricks_df.copy()
    df["lrg_count"] = [
        int(lrg_counts.get(str(bn), 0)) for bn in df["brickname"].astype(str).values
    ]
    df["lrg_density"] = [
        (df["lrg_count"].iloc[i] / brick_area.get(str(bn), 1.0))
        if brick_area.get(str(bn), 0.0) > 0
        else 0.0
        for i, bn in enumerate(df["brickname"].astype(str).values)
    ]

    total_lrg = sum(lrg_counts.values())
    print(
        f"[LRG-LOCAL] Finished: {total_lrg} total LRGs across {len(lrg_counts)} bricks "
        f"using {len(sweep_files)} SWEEP files.",
        flush=True,
    )
    return df


def _build_adjacency(bricks: List[BrickRecord]) -> Dict[int, List[int]]:
    """
    Build a simple adjacency graph between bricks:
      - Two bricks are adjacent if their centers are within
        ~0.3 deg in RA and ~0.3 deg in Dec.

    This is approximate but adequate for grouping bricks into contiguous regions.
    """
    adj: Dict[int, List[int]] = {i: [] for i in range(len(bricks))}
    for i, bi in enumerate(bricks):
        for j in range(i + 1, len(bricks)):
            bj = bricks[j]
            dra = _ra_delta_deg(bi.center_ra_deg, bj.center_ra_deg)
            ddec = abs(bi.center_dec_deg - bj.center_dec_deg)
            if dra <= 0.3 and ddec <= 0.3:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _connected_components(adj: Dict[int, List[int]]) -> List[List[int]]:
    """
    Basic DFS/BFS connected components on an undirected graph.
    """
    visited = set()
    components: List[List[int]] = []

    for node in adj:
        if node in visited:
            continue
        stack = [node]
        comp: List[int] = []
        visited.add(node)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        components.append(comp)

    return components


def select_regions(
    df_with_density: pd.DataFrame,
    config: Phase1p5Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build BrickRecord objects from the DataFrame, group them into contiguous
    regions, and choose a primary and optional backup region based on:

      - Total area between [min_region_area_deg2, max_region_area_deg2]
      - Highest LRG density (total LRGs / total area)

    Returns:
      (primary_region_df, backup_region_df)
    """
    records: List[BrickRecord] = []
    for _, row in df_with_density.iterrows():
        records.append(
            BrickRecord(
                brickname=str(row["brickname"]),
                ra=float(row["ra"]),
                dec=float(row["dec"]),
                ra1=float(row["ra1"]),
                ra2=float(row["ra2"]),
                dec1=float(row["dec1"]),
                dec2=float(row["dec2"]),
                psfsize_g=float(row["psfsize_g"]),
                psfsize_r=float(row["psfsize_r"]),
                psfsize_z=float(row["psfsize_z"]),
                psfdepth_g=float(row["psfdepth_g"]),
                psfdepth_r=float(row["psfdepth_r"]),
                psfdepth_z=float(row["psfdepth_z"]),
                ebv=float(row["ebv"]),
                area_deg2=float(row["area_deg2"]),
                lrg_count=int(row["lrg_count"]),
                lrg_density=float(row["lrg_density"]),
            )
        )

    if not records:
        return pd.DataFrame(), pd.DataFrame()

    adj = _build_adjacency(records)
    components = _connected_components(adj)

    component_summaries = []
    for comp in components:
        bricks_in_comp = [records[i] for i in comp]
        total_area = sum(b.area_deg2 for b in bricks_in_comp)
        total_lrg = sum(b.lrg_count for b in bricks_in_comp)
        density = total_lrg / total_area if total_area > 0 else 0.0
        component_summaries.append(
            {
                "component_index": len(component_summaries),
                "brick_indices": comp,
                "total_area_deg2": total_area,
                "total_lrg_count": total_lrg,
                "lrg_surface_density": density,
            }
        )

    # Filter components by area window
    area_ok = [
        c for c in component_summaries
        if (c["total_area_deg2"] >= config.min_region_area_deg2)
        and (c["total_area_deg2"] <= config.max_region_area_deg2)
    ]

    # If none are in the target area range, relax and just pick the largest few
    if not area_ok:
        area_ok = sorted(
            component_summaries,
            key=lambda c: c["total_area_deg2"],
            reverse=True,
        )[:3]

    # Rank by LRG surface density
    area_ok_sorted = sorted(
        area_ok,
        key=lambda c: c["lrg_surface_density"],
        reverse=True,
    )

    primary = area_ok_sorted[0]
    backup = area_ok_sorted[1] if len(area_ok_sorted) > 1 else None

    # Build DataFrames
    def _component_to_df(summary_dict) -> pd.DataFrame:
        indices = summary_dict["brick_indices"]
        recs = [records[i] for i in indices]
        data = {
            "brickname": [b.brickname for b in recs],
            "ra": [b.ra for b in recs],
            "dec": [b.dec for b in recs],
            "ra1": [b.ra1 for b in recs],
            "ra2": [b.ra2 for b in recs],
            "dec1": [b.dec1 for b in recs],
            "dec2": [b.dec2 for b in recs],
            "psfsize_g": [b.psfsize_g for b in recs],
            "psfsize_r": [b.psfsize_r for b in recs],
            "psfsize_z": [b.psfsize_z for b in recs],
            "psfdepth_g": [b.psfdepth_g for b in recs],
            "psfdepth_r": [b.psfdepth_r for b in recs],
            "psfdepth_z": [b.psfdepth_z for b in recs],
            "ebv": [b.ebv for b in recs],
            "area_deg2": [b.area_deg2 for b in recs],
            "lrg_count": [b.lrg_count for b in recs],
            "lrg_density": [b.lrg_density for b in recs],
        }
        return pd.DataFrame(data)

    primary_df = _component_to_df(primary)
    backup_df = _component_to_df(backup) if backup is not None else pd.DataFrame()

    return primary_df, backup_df

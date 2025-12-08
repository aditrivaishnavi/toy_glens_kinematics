"""
SWEEP-based LRG density estimation for Phase 1.5.

This module provides a streaming, mildly parallel approach to estimating
LRG density per brick using DR10 SWEEP files. It downloads SWEEP files
on-demand (if they are URLs), processes them one at a time, and deletes
downloaded files after use to keep disk usage bounded.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from .config import Phase1p5Config
from .region_scout import _build_lrg_mask  # reuse existing LRG mask logic


def _load_sweep_index(path: str) -> List[str]:
    """
    Read a text file with one SWEEP path or URL per line.
    Ignores blank lines and comments starting with '#'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SWEEP index file not found: {path}"
        )
    entries: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(line)
    if not entries:
        raise RuntimeError(f"SWEEP index file {path} contains no usable entries.")
    return entries


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _download_if_needed(sweep_entry: str, download_dir: str) -> Tuple[str, bool]:
    """
    Given a SWEEP entry, return the local path and whether it was downloaded.

    If sweep_entry is an http/https URL, streams it to download_dir / basename.
    If sweep_entry is a local path, returns it unchanged with downloaded=False.
    """
    import requests

    if not _is_url(sweep_entry):
        # treat as local path
        return sweep_entry, False

    os.makedirs(download_dir, exist_ok=True)
    basename = os.path.basename(sweep_entry)
    local_path = str(Path(download_dir) / basename)

    if os.path.exists(local_path):
        return local_path, False

    resp = requests.get(sweep_entry, stream=True, timeout=300)
    resp.raise_for_status()

    with open(local_path, "wb") as out:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            out.write(chunk)

    return local_path, True


def _process_single_sweep(
    sweep_entry: str,
    bricks_of_interest: Set[str],
    config: Phase1p5Config,
    counts: Dict[str, int],
    lock: threading.Lock,
) -> Tuple[str, int]:
    """
    Download (if needed), process, and optionally delete one SWEEP file.

    Returns (sweep_entry, n_lrg_in_this_sweep) for logging.
    """
    from astropy.table import Table

    local_path, downloaded = _download_if_needed(
        sweep_entry,
        config.sweep_download_dir,
    )
    n_lrg_total = 0

    try:
        # Read the table; SWEEP files usually store the main catalog in HDU 1
        table = Table.read(local_path, hdu=1)
        df = table.to_pandas()

        # Harmonize column names: SWEEPs typically use uppercase like BRICKNAME, FLUX_R, etc.
        cols = {c.upper(): c for c in df.columns}
        if "BRICKNAME" not in cols:
            raise RuntimeError(
                f"SWEEP file {local_path} does not contain BRICKNAME column."
            )

        # Build a smaller DataFrame with only the bricks we care about
        brick_col_name = cols["BRICKNAME"]
        mask_brick = df[brick_col_name].isin(bricks_of_interest)
        if not mask_brick.any():
            return sweep_entry, 0

        sub = df.loc[mask_brick].copy()

        # Map to the column names expected by _build_lrg_mask:
        # flux_r, flux_z, flux_w1
        def _find_flux_col(band: str) -> str:
            # band in ["R", "Z", "W1"]
            candidates = [
                f"FLUX_{band}",
                f"FLUX{band}",
            ]
            for cand in candidates:
                if cand in cols:
                    return cols[cand]
            raise RuntimeError(
                f"Could not find a flux column for band {band} in {local_path}"
            )

        flux_r_col = _find_flux_col("R")
        flux_z_col = _find_flux_col("Z")
        flux_w1_col = _find_flux_col("W1")

        # Construct a minimal view with column names that match _build_lrg_mask
        mini = pd.DataFrame(
            {
                "brickname": sub[brick_col_name].astype(str).values,
                "flux_r": sub[flux_r_col].values,
                "flux_z": sub[flux_z_col].values,
                "flux_w1": sub[flux_w1_col].values,
            }
        )

        # Compute LRG mask using the existing logic
        lrg_mask = _build_lrg_mask(mini, config)
        if not np.any(lrg_mask):
            return sweep_entry, 0

        # Accumulate counts per brick
        lrg_bricks = mini.loc[lrg_mask, "brickname"].values
        with lock:
            for bname in lrg_bricks:
                counts[bname] = counts.get(bname, 0) + 1
                n_lrg_total += 1

        return sweep_entry, n_lrg_total

    finally:
        # Clean up downloaded file to avoid filling the disk
        if downloaded:
            try:
                Path(local_path).unlink()
            except OSError:
                pass


def estimate_lrg_density_from_sweeps(
    bricks_df: pd.DataFrame,
    config: Phase1p5Config,
    pilot_limit: int = 0,
) -> pd.DataFrame:
    """
    Estimate LRG density per brick using DR10 SWEEP files in a streaming,
    mildly parallel fashion.

    - Reads SWEEP entries from config.sweep_index_path.
    - Processes up to config.max_sweeps_for_lrg_density entries (or all if 0),
      overridden by pilot_limit if > 0.
    - Uses max_parallel_sweeps threads for concurrent download + processing.
    - Deletes any SWEEP files that were downloaded by this script.

    Returns a copy of bricks_df with `lrg_count` and `lrg_density` columns added.
    """
    import time

    print("[SWEEP] Loading SWEEP index...", flush=True)
    sweep_entries = _load_sweep_index(config.sweep_index_path)
    n_total = len(sweep_entries)

    # Determine how many sweeps to use
    max_sweeps = config.max_sweeps_for_lrg_density
    if pilot_limit > 0:
        max_sweeps = pilot_limit
    if max_sweeps <= 0 or max_sweeps > n_total:
        max_sweeps = n_total

    sweep_entries = sweep_entries[:max_sweeps]
    print(f"[SWEEP] Using {len(sweep_entries)} SWEEP entries out of {n_total} listed.", flush=True)

    # Bricks we care about
    bricks_of_interest: Set[str] = set(bricks_df["brickname"].astype(str).values)

    # Shared counts dict and lock for thread safety
    counts: Dict[str, int] = {}
    lock = threading.Lock()

    start = time.perf_counter()
    n_workers = max(1, int(config.max_parallel_sweeps))
    print(f"[SWEEP] Starting ThreadPool with {n_workers} workers", flush=True)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(
                _process_single_sweep,
                entry,
                bricks_of_interest,
                config,
                counts,
                lock,
            ): entry
            for entry in sweep_entries
        }

        for i, fut in enumerate(as_completed(futures), start=1):
            entry = futures[fut]
            try:
                sweep_name, n_lrg = fut.result()
                elapsed = time.perf_counter() - start
                print(
                    f"[SWEEP] {i}/{len(sweep_entries)} done: {os.path.basename(sweep_name)} "
                    f"→ {n_lrg} LRGs (elapsed {elapsed/60:.1f} min)",
                    flush=True,
                )
            except Exception as e:
                elapsed = time.perf_counter() - start
                print(
                    f"[SWEEP] {i}/{len(sweep_entries)} FAILED {entry!r}: {e!r} "
                    f"(elapsed {elapsed/60:.1f} min)",
                    flush=True,
                )

    total_elapsed = time.perf_counter() - start
    print(f"[SWEEP] Completed SWEEP processing in {total_elapsed/60:.1f} minutes", flush=True)
    print(f"[SWEEP] Non-zero LRG counts for {len(counts)} bricks", flush=True)

    # Attach counts and densities to bricks_df
    bricks_copy = bricks_df.copy()
    bricks_copy["lrg_count"] = bricks_copy["brickname"].astype(str).map(counts).fillna(0).astype(int)
    bricks_copy["lrg_density"] = bricks_copy.apply(
        lambda row: (
            row["lrg_count"] / row["area_deg2"] if row["area_deg2"] > 0 else 0.0
        ),
        axis=1,
    )

    print(
        f"[SWEEP] Example density range: "
        f"{bricks_copy['lrg_density'].min():.2f} – {bricks_copy['lrg_density'].max():.2f} LRG/deg^2",
        flush=True,
    )
    return bricks_copy


#!/usr/bin/env python3
"""
Phase 2 hypergrid analysis

Takes the EMR Phase 2 hypergrid output (per brick LRG counts for multiple
selection variants), joins it to DR10 bricks geometry and QA, computes
densities and fractions, and identifies contiguous high density regions.

This is meant to be run locally after the EMR hypergrid job completes.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import pyvo
except ImportError:
    pyvo = None


# -----------------------------
# Helpers for bricks and areas
# -----------------------------

def _compute_brick_area_deg2(ra1: float, ra2: float, dec1: float, dec2: float) -> float:
    """
    Compute approximate spherical area of a brick in deg^2 from RA/Dec bounds.

    area_sr = (ra2 - ra1) * (sin(dec2) - sin(dec1)) in radians^2
    1 sr = (180 / pi)^2 deg^2
    """
    ra1_rad = math.radians(ra1)
    ra2_rad = math.radians(ra2)
    dec1_rad = math.radians(dec1)
    dec2_rad = math.radians(dec2)

    d_ra = ra2_rad - ra1_rad
    d_sin_dec = math.sin(dec2_rad) - math.sin(dec1_rad)
    area_sr = abs(d_ra * d_sin_dec)
    steradian_to_deg2 = (180.0 / math.pi) ** 2
    return float(area_sr * steradian_to_deg2)


def _get_tap_service(tap_url: str):
    if pyvo is None:
        raise ImportError(
            "pyvo is required for TAP access. Install with `pip install pyvo` "
            "or provide a local bricks CSV via --bricks-csv."
        )
    return pyvo.dal.TAPService(tap_url)


def fetch_bricks_for_hypergrid(
    bricknames: List[str],
    tap_url: str = "https://datalab.noirlab.edu/tap",
    batch_size: int = 500,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch geometry and QA for the given bricknames from ls_dr10.bricks_s
    via TAP in batches.

    This pulls only the bricks you actually have from Phase 2, not the full table.
    
    If cache_path is provided:
      - Reads existing cached bricks
      - Only fetches missing bricks from TAP
      - Streams each batch to cache file (resumable on crash)
    """
    tap = _get_tap_service(tap_url)

    cols = [
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

    unique_bricks = sorted(set(str(b) for b in bricknames))
    
    # Check what we already have cached
    cached_bricks: set = set()
    existing_columns: Optional[List[str]] = None
    
    if cache_path and cache_path.exists():
        existing_df = pd.read_csv(cache_path)
        cached_bricks = set(existing_df["brickname"].astype(str).tolist())
        existing_columns = list(existing_df.columns)  # Preserve column order for appending
        print(f"[BRICKS] Found {len(cached_bricks)} bricks already cached", flush=True)
    
    # Determine what's missing
    bricks_to_fetch = [b for b in unique_bricks if b not in cached_bricks]
    
    if not bricks_to_fetch:
        print(f"[BRICKS] All {len(unique_bricks)} bricks already cached!", flush=True)
        return pd.read_csv(cache_path)
    
    import time
    
    n_total = len(bricks_to_fetch)
    n_batches = (n_total + batch_size - 1) // batch_size
    
    print(f"[BRICKS] Need to fetch: {n_total} bricks in {n_batches} batches", flush=True)
    if cached_bricks:
        print(f"[BRICKS] (Skipping {len(cached_bricks)} already cached)", flush=True)

    # Prepare cache file for appending
    # If file exists, we append without header and use existing column order
    write_header = not (cache_path and cache_path.exists())
    
    fetched_count = 0
    pending_chunks: List[pd.DataFrame] = []
    flush_every = 10  # Flush to disk every N batches
    start_time = time.time()

    for i in range(0, n_total, batch_size):
        batch = bricks_to_fetch[i : i + batch_size]
        in_list = ",".join(f"'{b}'" for b in batch)
        query = f"""
            SELECT
                {', '.join(cols)}
            FROM ls_dr10.bricks_s
            WHERE brickname IN ({in_list})
        """

        batch_num = i // batch_size + 1
        pct = 100.0 * batch_num / n_batches
        elapsed = time.time() - start_time
        
        # Estimate remaining time
        if batch_num > 1:
            avg_per_batch = elapsed / (batch_num - 1)
            remaining = avg_per_batch * (n_batches - batch_num + 1)
            eta_str = f", ETA: {remaining/60:.1f}min"
        else:
            eta_str = ""

        print(
            f"[BRICKS] Batch {batch_num}/{n_batches} ({pct:.0f}%) - {len(batch)} bricks [{elapsed:.0f}s{eta_str}]",
            flush=True,
        )
        result = tap.run_sync(query)
        try:
            table = result.to_table()
        except AttributeError:
            table = result
        df = table.to_pandas()
        
        # Compute area
        if "area_deg2" not in df.columns and len(df) > 0:
            df["area_deg2"] = df.apply(
                lambda row: _compute_brick_area_deg2(
                    float(row["ra1"]),
                    float(row["ra2"]),
                    float(row["dec1"]),
                    float(row["dec2"]),
                ),
                axis=1,
            )
        
        if len(df) > 0:
            pending_chunks.append(df)
            fetched_count += len(df)
        
        # Flush to disk periodically
        if cache_path and len(pending_chunks) >= flush_every:
            combined = pd.concat(pending_chunks, ignore_index=True)
            
            # Ensure column order matches existing file when appending
            if existing_columns is not None and not write_header:
                # Reorder columns to match existing file
                combined = combined[[c for c in existing_columns if c in combined.columns]]
            
            combined.to_csv(cache_path, mode='a', header=write_header, index=False)
            
            # After first write, capture column order for subsequent appends
            if write_header:
                existing_columns = list(combined.columns)
            write_header = False
            
            elapsed = time.time() - start_time
            print(f"[BRICKS] 💾 Flushed {len(combined)} bricks to cache [{elapsed:.0f}s]", flush=True)
            pending_chunks = []
    
    # Flush any remaining
    if cache_path and pending_chunks:
        combined = pd.concat(pending_chunks, ignore_index=True)
        
        # Ensure column order matches existing file when appending
        if existing_columns is not None and not write_header:
            combined = combined[[c for c in existing_columns if c in combined.columns]]
        
        combined.to_csv(cache_path, mode='a', header=write_header, index=False)
        elapsed = time.time() - start_time
        print(f"[BRICKS] 💾 Flushed final {len(combined)} bricks to cache [{elapsed:.0f}s]", flush=True)

    total_time = time.time() - start_time
    print(f"[BRICKS] ✓ Fetched {fetched_count} new bricks total in {total_time:.1f}s ({total_time/60:.1f}min)", flush=True)
    
    # Read back the complete cache
    if cache_path and cache_path.exists():
        bricks_df = pd.read_csv(cache_path)
    else:
        raise RuntimeError("No bricks fetched and no cache available.")

    # Sanity check
    missing = [b for b in unique_bricks if b not in set(bricks_df["brickname"].astype(str))]
    if missing:
        print(
            f"[BRICKS] Warning: {len(missing)} bricknames not found in ls_dr10.bricks_s "
            f"(first few: {missing[:10]})",
            file=sys.stderr,
            flush=True,
        )

    return bricks_df


# -----------------------------
# Region adjacency and grouping
# -----------------------------

@dataclass
class BrickNode:
    index: int
    brickname: str
    ra: float
    dec: float
    area_deg2: float
    lrg_density_baseline: float


def _ra_delta_deg(ra1: float, ra2: float) -> float:
    d = abs(ra1 - ra2)
    return min(d, 360.0 - d)


def build_adjacency(nodes: List[BrickNode], max_sep_deg: float = 0.3, verify: bool = False) -> Dict[int, List[int]]:
    """
    Build adjacency graph between bricks using spatial indexing (KDTree).
    Two bricks are adjacent if their centers are within max_sep_deg
    in both RA and Dec. This mirrors the Phase 1.5 adjacency logic.
    
    Uses scipy.spatial.cKDTree for O(n log n) performance instead of O(n^2).
    
    Args:
        nodes: List of BrickNode objects
        max_sep_deg: Maximum separation in degrees for adjacency
        verify: If True, verify KDTree result against O(n²) on a sample
    """
    import time
    from scipy.spatial import cKDTree
    
    start_time = time.time()
    n_nodes = len(nodes)
    
    if n_nodes == 0:
        return {}
    
    print(f"[ADJACENCY] Building adjacency graph for {n_nodes} nodes...", flush=True)
    
    adj: Dict[int, List[int]] = {n.index: [] for n in nodes}
    
    # For small datasets, use simple O(n^2) approach
    if n_nodes < 500:
        print(f"[ADJACENCY] Using simple O(n²) for small dataset...", flush=True)
    for i, bi in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            bj = nodes[j]
            dra = _ra_delta_deg(bi.ra, bj.ra)
            ddec = abs(bi.dec - bj.dec)
            if dra <= max_sep_deg and ddec <= max_sep_deg:
                adj[bi.index].append(bj.index)
                adj[bj.index].append(bi.index)
        elapsed = time.time() - start_time
        total_edges = sum(len(v) for v in adj.values()) // 2
        print(f"[ADJACENCY] ✓ Built graph: {total_edges} edges in {elapsed:.1f}s", flush=True)
        return adj
    
    # For larger datasets, use KDTree with Dec-based spatial indexing
    # Note: RA wraps around, so we handle it specially
    print(f"[ADJACENCY] Using KDTree spatial indexing for large dataset...", flush=True)
    
    # Create coordinate array (Dec, RA) for KDTree
    # We'll query by Dec first (no wraparound), then filter by RA
    coords = np.array([[n.dec, n.ra] for n in nodes])
    
    # Build KDTree on Dec coordinate only for initial neighbor search
    dec_coords = coords[:, 0].reshape(-1, 1)
    tree = cKDTree(dec_coords)
    
    # Query all pairs within max_sep_deg in Dec
    # This gives us candidate pairs that we then filter by RA
    pairs = tree.query_pairs(r=max_sep_deg, output_type='ndarray')
    
    print(f"[ADJACENCY] Found {len(pairs)} Dec-candidate pairs, filtering by RA...", flush=True)
    
    # Filter by RA distance
    edge_count = 0
    for i, j in pairs:
        bi = nodes[i]
        bj = nodes[j]
        dra = _ra_delta_deg(bi.ra, bj.ra)
        if dra <= max_sep_deg:
            adj[bi.index].append(bj.index)
            adj[bj.index].append(bi.index)
            edge_count += 1
    
    elapsed = time.time() - start_time
    print(f"[ADJACENCY] ✓ Built graph: {edge_count} edges in {elapsed:.1f}s", flush=True)
    
    # Optional verification against O(n²) on a sample
    # Note: O(n²) takes ~10-20s for 3000 nodes, acceptable for verification
    if verify and n_nodes <= 4000:
        print(f"[ADJACENCY] Verifying against O(n²) brute force...", flush=True)
        expected_edges = set()
        for i, bi in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                bj = nodes[j]
                dra = _ra_delta_deg(bi.ra, bj.ra)
                ddec = abs(bi.dec - bj.dec)
                if dra <= max_sep_deg and ddec <= max_sep_deg:
                    expected_edges.add((min(bi.index, bj.index), max(bi.index, bj.index)))
        
        actual_edges = set()
        for node_idx, neighbors in adj.items():
            for neighbor_idx in neighbors:
                actual_edges.add((min(node_idx, neighbor_idx), max(node_idx, neighbor_idx)))
        
        if expected_edges == actual_edges:
            print(f"[ADJACENCY] ✓ Verification PASSED: {len(expected_edges)} edges match", flush=True)
        else:
            missing = expected_edges - actual_edges
            extra = actual_edges - expected_edges
            print(f"[ADJACENCY] ✗ Verification FAILED!", flush=True)
            print(f"[ADJACENCY]   Missing edges: {len(missing)}", flush=True)
            print(f"[ADJACENCY]   Extra edges: {len(extra)}", flush=True)
            if missing:
                print(f"[ADJACENCY]   First missing: {list(missing)[:5]}", flush=True)
    
    return adj


def connected_components(adj: Dict[int, List[int]]) -> List[List[int]]:
    """
    DFS connected components on the brick adjacency graph.
    """
    import time
    
    start_time = time.time()
    n_nodes = len(adj)
    print(f"[COMPONENTS] Finding connected components in {n_nodes} nodes...", flush=True)
    
    visited = set()
    components: List[List[int]] = []

    for start in adj.keys():
        if start in visited:
            continue
        stack = [start]
        comp: List[int] = []
        visited.add(start)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        components.append(comp)

    elapsed = time.time() - start_time
    print(f"[COMPONENTS] ✓ Found {len(components)} components in {elapsed:.2f}s", flush=True)
    return components


# -----------------------------
# Main analysis logic
# -----------------------------

def analyze_hypergrid(
    hypergrid_csv: Path,
    output_dir: Path,
    bricks_csv: Optional[Path] = None,
    tap_url: str = "https://datalab.noirlab.edu/tap",
    ra_min: Optional[float] = None,
    ra_max: Optional[float] = None,
    dec_min: Optional[float] = None,
    dec_max: Optional[float] = None,
    baseline_variant: str = "n_lrg_v2_baseline_dr10",
    high_percentiles: Tuple[float, ...] = (90.0, 95.0, 99.0),
    min_region_area_deg2: float = 2.0,
    max_region_area_deg2: float = 400.0,
    verify_adjacency: bool = False,
) -> None:
    """
    Core analysis routine.

    - Loads hypergrid CSV (per brick LRG counts for multiple variants)
    - Joins to bricks geometry and QA
    - Computes densities and LRG fractions
    - Identifies contiguous high density regions
    - Writes merged bricks table, region summaries, and a markdown report
    """
    import time as _time
    
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_start = _time.time()

    print(f"[PHASE2] === Step 1/6: Loading hypergrid CSV ===", flush=True)
    print(f"[PHASE2] Loading hypergrid CSV: {hypergrid_csv}", flush=True)
    step_start = _time.time()
    hg = pd.read_csv(hypergrid_csv)
    print(f"[PHASE2] Loaded {len(hg)} rows in {_time.time() - step_start:.1f}s", flush=True)

    if "brickname" not in hg.columns:
        raise ValueError("Hypergrid CSV must contain a 'brickname' column.")

    if "n_gal" not in hg.columns:
        raise ValueError("Hypergrid CSV must contain an 'n_gal' column.")

    # Identify variant columns
    lrg_cols = [c for c in hg.columns if c.startswith("n_lrg_v")]
    if not lrg_cols:
        raise ValueError("Hypergrid CSV contains no 'n_lrg_v*' columns.")

    print(f"[PHASE2] Found LRG variant columns: {lrg_cols}", flush=True)

    # Load bricks geometry and QA
    print(f"[PHASE2] === Step 2/6: Loading bricks geometry ===", flush=True)
    step_start = _time.time()
    if bricks_csv is not None:
        print(f"[PHASE2] Loading bricks from local CSV: {bricks_csv}", flush=True)
        bricks = pd.read_csv(bricks_csv)
        # Expect at least these columns
        required_brick_cols = {
            "brickname",
            "ra",
            "dec",
            "ra1",
            "ra2",
            "dec1",
            "dec2",
            "psfsize_r",
            "psfdepth_r",
            "ebv",
        }
        missing = required_brick_cols - set(bricks.columns)
        if missing:
            raise ValueError(
                f"Bricks CSV missing required columns: {sorted(missing)}"
            )

        if "area_deg2" not in bricks.columns:
            print("[PHASE2] Bricks CSV missing area_deg2; computing from RA/Dec bounds", flush=True)
            bricks["area_deg2"] = bricks.apply(
                lambda row: _compute_brick_area_deg2(
                    float(row["ra1"]),
                    float(row["ra2"]),
                    float(row["dec1"]),
                    float(row["dec2"]),
                ),
                axis=1,
            )
    else:
        print("[PHASE2] No bricks CSV provided; fetching bricks via TAP", flush=True)
        bricks = fetch_bricks_for_hypergrid(
            bricknames=hg["brickname"].astype(str).tolist(),
            tap_url=tap_url,
        )

    # Optional RA/Dec filter at brick level
    if ra_min is not None and ra_max is not None and dec_min is not None and dec_max is not None:
        mask = (
            (bricks["ra"] >= ra_min)
            & (bricks["ra"] <= ra_max)
            & (bricks["dec"] >= dec_min)
            & (bricks["dec"] <= dec_max)
        )
        bricks = bricks.loc[mask].copy()
        print(
            f"[PHASE2] After RA/Dec filter: {len(bricks)} bricks remain in [{ra_min}, {ra_max}] x [{dec_min}, {dec_max}]",
            flush=True,
        )

    print(f"[PHASE2] Bricks loaded in {_time.time() - step_start:.1f}s", flush=True)

    # Merge hypergrid with bricks
    print(f"[PHASE2] === Step 3/6: Merging and computing densities ===", flush=True)
    step_start = _time.time()
    merged = pd.merge(
        bricks,
        hg,
        on="brickname",
        how="inner",
        validate="one_to_one",
    )
    print(
        f"[PHASE2] Merged hypergrid with bricks: {len(merged)} bricks "
        f"(from {len(bricks)} bricks with geometry and {len(hg)} hypergrid rows)",
        flush=True,
    )

    if merged.empty:
        raise RuntimeError("Merged table is empty. Check RA/Dec filter and brickname overlap.")

    # Compute densities and fractions for each LRG variant
    for col in lrg_cols:
        base_name = col.replace("n_lrg_", "")
        dens_col = f"lrg_density_{base_name}"
        frac_col = f"lrg_frac_{base_name}"

        merged[dens_col] = merged[col] / merged["area_deg2"]
        merged[frac_col] = merged[col] / merged["n_gal"].where(merged["n_gal"] > 0, np.nan)

    # Sanity stats for each variant
    variant_stats = []
    for col in lrg_cols:
        base_name = col.replace("n_lrg_", "")
        dens_col = f"lrg_density_{base_name}"
        frac_col = f"lrg_frac_{base_name}"

        dens = merged[dens_col].values
        dens_pos = dens[dens > 0]

        if dens_pos.size == 0:
            stats = {
                "variant": base_name,
                "n_bricks": len(merged),
                "n_nonzero": 0,
                "median_density": 0.0,
                "p90_density": 0.0,
                "p95_density": 0.0,
                "p99_density": 0.0,
                "mean_frac": float(merged[frac_col].mean(skipna=True)),
                "median_frac": float(merged[frac_col].median(skipna=True)),
                "description": col,
            }
        else:
            stats = {
                "variant": base_name,
                "n_bricks": len(merged),
                "n_nonzero": int((dens > 0).sum()),
                "median_density": float(np.median(dens_pos)),
                "p90_density": float(np.percentile(dens_pos, 90)),
                "p95_density": float(np.percentile(dens_pos, 95)),
                "p99_density": float(np.percentile(dens_pos, 99)),
                "mean_frac": float(merged[frac_col].mean(skipna=True)),
                "median_frac": float(merged[frac_col].median(skipna=True)),
                "description": col,
            }
        variant_stats.append(stats)

    variant_stats_df = pd.DataFrame(variant_stats)
    variant_stats_path = output_dir / "phase2_variant_stats.csv"
    variant_stats_df.to_csv(variant_stats_path, index=False)
    print(f"[PHASE2] Wrote variant stats to {variant_stats_path}", flush=True)
    print(f"[PHASE2] Step 3 completed in {_time.time() - step_start:.1f}s", flush=True)

    # -----------------------------
    # Region finding on baseline
    # -----------------------------
    print(f"[PHASE2] === Step 4/6: Finding high-density regions ===", flush=True)
    step_start = _time.time()
    
    if baseline_variant not in merged.columns:
        raise ValueError(
            f"Baseline variant column '{baseline_variant}' not found in hypergrid file. "
            f"Available: {list(hg.columns)}"
        )

    # Use density for baseline variant
    base_name = baseline_variant.replace("n_lrg_", "")
    base_dens_col = f"lrg_density_{base_name}"
    if base_dens_col not in merged.columns:
        raise ValueError(
            f"Baseline density column '{base_dens_col}' not computed. "
            f"Check variant naming."
        )

    dens = merged[base_dens_col].values
    dens_pos = dens[dens > 0]
    if dens_pos.size == 0:
        print("[PHASE2] Warning: baseline variant has zero density everywhere. Skipping region selection.", flush=True)
        regions_summary_df = pd.DataFrame()
        regions_bricks_df = pd.DataFrame()
    else:
        # Determine high density threshold from requested percentiles
        high_percentiles = sorted(high_percentiles)
        perc_values = {
            p: float(np.percentile(dens_pos, p)) for p in high_percentiles
        }

        print("[PHASE2] Baseline density percentiles (nonzero):", flush=True)
        for p in high_percentiles:
            print(f"         p{p:.0f} = {perc_values[p]:.3f} LRG / deg^2", flush=True)

        # Tag bricks in the highest percentile band (for example use the highest given percentile)
        top_p = max(high_percentiles)
        top_thr = perc_values[top_p]
        merged["is_high_density_baseline"] = merged[base_dens_col] >= top_thr
        high_df = merged.loc[merged["is_high_density_baseline"]].copy()

        print(
            f"[PHASE2] High density bricks (>= p{top_p:.0f} in baseline): "
            f"{len(high_df)} out of {len(merged)}",
            flush=True,
        )

        # Build nodes for adjacency graph using only high density bricks
        nodes: List[BrickNode] = []
        idx_map: Dict[int, int] = {}
        for idx, row in high_df.reset_index(drop=True).iterrows():
            node_index = idx
            node = BrickNode(
                index=node_index,
                brickname=str(row["brickname"]),
                ra=float(row["ra"]),
                dec=float(row["dec"]),
                area_deg2=float(row["area_deg2"]),
                lrg_density_baseline=float(row[base_dens_col]),
            )
            nodes.append(node)
            idx_map[node_index] = int(row.name)

        if not nodes:
            print("[PHASE2] No high density bricks identified. Region list will be empty.", flush=True)
            regions_summary_df = pd.DataFrame()
            regions_bricks_df = pd.DataFrame()
        else:
            adj = build_adjacency(nodes, max_sep_deg=0.3, verify=verify_adjacency)
            comps = connected_components(adj)

            print(
                f"[PHASE2] Found {len(comps)} contiguous components among high density bricks.",
                flush=True,
            )

            # Map from local node index to original merged row index
            high_df_reset = high_df.reset_index(drop=True)

            region_rows = []
            region_brick_rows = []

            for region_id, comp in enumerate(comps):
                # comp contains node indices; map back to high_df rows
                comp_indices = comp
                bricks_in_region = high_df_reset.iloc[comp_indices]

                total_area = float(bricks_in_region["area_deg2"].sum())
                total_n_gal = int(bricks_in_region["n_gal"].sum())

                # Aggregate LRG counts for each variant
                region_counts = {}
                region_densities = {}
                for col in lrg_cols:
                    variant_name = col.replace("n_lrg_", "")
                    dens_col = f"lrg_density_{variant_name}"
                    total_n_lrg = int(bricks_in_region[col].sum())
                    region_counts[variant_name] = total_n_lrg
                    region_densities[variant_name] = (
                        total_n_lrg / total_area if total_area > 0 else 0.0
                    )

                # QA metrics
                median_psf_r = float(bricks_in_region["psfsize_r"].median())
                median_depth_r = float(bricks_in_region["psfdepth_r"].median())
                median_ebv = float(bricks_in_region["ebv"].median())

                # Area filter
                if total_area < min_region_area_deg2 or total_area > max_region_area_deg2:
                    keep_flag = False
                else:
                    keep_flag = True

                # Simple area weighted center
                weights = bricks_in_region["area_deg2"].values
                ra_center = float(np.average(bricks_in_region["ra"].values, weights=weights))
                dec_center = float(np.average(bricks_in_region["dec"].values, weights=weights))

                row = {
                    "region_id": region_id,
                    "n_bricks": len(bricks_in_region),
                    "total_area_deg2": total_area,
                    "total_n_gal": total_n_gal,
                    "ra_center_deg": ra_center,
                    "dec_center_deg": dec_center,
                    "median_psf_r_arcsec": median_psf_r,
                    "median_psfdepth_r": median_depth_r,
                    "median_ebv": median_ebv,
                    "keep_in_area_window": keep_flag,
                }

                # Attach variant counts and densities
                for variant_name in region_counts:
                    row[f"total_n_lrg_{variant_name}"] = region_counts[variant_name]
                    row[f"mean_lrg_density_{variant_name}"] = region_densities[variant_name]

                region_rows.append(row)

                # Also export brick membership per region
                for _, b_row in bricks_in_region.iterrows():
                    region_brick_rows.append(
                        {
                            "region_id": region_id,
                            "brickname": b_row["brickname"],
                            "ra": b_row["ra"],
                            "dec": b_row["dec"],
                            "area_deg2": b_row["area_deg2"],
                            "n_gal": b_row["n_gal"],
                            **{c: b_row[c] for c in lrg_cols},
                        }
                    )

            regions_summary_df = pd.DataFrame(region_rows)
            regions_bricks_df = pd.DataFrame(region_brick_rows)

            # Sort regions by baseline mean density within area window
            if not regions_summary_df.empty:
                regions_summary_df = regions_summary_df.sort_values(
                    by=[f"mean_lrg_density_{base_name}", "total_area_deg2"],
                    ascending=[False, False],
                )

    print(f"[PHASE2] Step 4 completed in {_time.time() - step_start:.1f}s", flush=True)

    # -----------------------------
    # Write outputs
    # -----------------------------
    print(f"[PHASE2] === Step 5/6: Writing output files ===", flush=True)
    step_start = _time.time()

    merged_path = output_dir / "phase2_hypergrid_bricks_merged.csv"
    merged.to_csv(merged_path, index=False)
    print(f"[PHASE2] Wrote merged bricks table to {merged_path}", flush=True)

    regions_summary_path = output_dir / "phase2_regions_summary.csv"
    regions_summary_df.to_csv(regions_summary_path, index=False)
    print(f"[PHASE2] Wrote regions summary to {regions_summary_path}", flush=True)

    regions_bricks_path = output_dir / "phase2_regions_bricks.csv"
    regions_bricks_df.to_csv(regions_bricks_path, index=False)
    print(f"[PHASE2] Wrote region membership table to {regions_bricks_path}", flush=True)

    print(f"[PHASE2] Step 5 completed in {_time.time() - step_start:.1f}s", flush=True)

    # -----------------------------
    # Markdown report
    # -----------------------------
    print(f"[PHASE2] === Step 6/6: Generating markdown report ===", flush=True)
    step_start = _time.time()
    
    md_path = output_dir / "phase2_hypergrid_analysis.md"
    
    # Get variant definition for detailed parameters
    variant_defn = get_variant_definition(baseline_variant)

    with md_path.open("w") as f:
        f.write("# Phase 2 Hypergrid Analysis Report\n\n")
        
        f.write("## Input Parameters\n\n")
        f.write(f"- **Hypergrid CSV**: `{hypergrid_csv}`\n")
        f.write(f"- **Output directory**: `{output_dir}`\n")
        f.write(f"- **Baseline variant**: `{baseline_variant}`\n")
        f.write(f"- **Number of merged bricks**: {len(merged)}\n")
        f.write(f"- **Percentiles for high-density threshold**: {high_percentiles}\n")
        f.write(f"- **Min region area**: {min_region_area_deg2} deg²\n")
        f.write(f"- **Max region area**: {max_region_area_deg2} deg²\n\n")
        
        if variant_defn:
            f.write("## LRG Selection Criteria (for this variant)\n\n")
            f.write(f"| Parameter | Value |\n")
            f.write(f"|-----------|-------|\n")
            f.write(f"| Variant name | `{variant_defn['name']}` |\n")
            f.write(f"| z-band magnitude limit | z < {variant_defn['z_mag_max']} AB mag |\n")
            f.write(f"| r-z color cut | r - z > {variant_defn['rz_min']} |\n")
            f.write(f"| z-W1 color cut | z - W1 > {variant_defn['zw1_min']} |\n")
            f.write(f"| Description | {variant_defn['description']} |\n\n")
        
        f.write("## EMR Phase 2 Job Parameters\n\n")
        f.write("The input CSV was generated by `spark_phase2_lrg_hypergrid.py` with:\n\n")
        f.write("| Variant | z_mag_max | r-z min | z-W1 min | Description |\n")
        f.write("|---------|-----------|---------|----------|-------------|\n")
        for defn in LRG_HYPERGRID_DEFINITIONS:
            marker = " ← **this report**" if defn['name'] == baseline_variant.replace("n_lrg_", "") else ""
            f.write(
                f"| {defn['name']} | {defn['z_mag_max']} | {defn['rz_min']} | "
                f"{defn['zw1_min']} | {defn['description']}{marker} |\n"
            )
        f.write("\n")

        if ra_min is not None and ra_max is not None and dec_min is not None and dec_max is not None:
            f.write("## Footprint Filter\n\n")
            f.write(
                f"- RA range: **[{ra_min}, {ra_max}] deg**\n"
                f"- Dec range: **[{dec_min}, {dec_max}] deg**\n\n"
            )
        else:
            f.write("## Footprint Filter\n\n")
            f.write("- **No RA/Dec filter applied** - analyzing full sky coverage\n\n")

        f.write("## LRG Variant Summary\n\n")
        f.write(
            "For each LRG selection variant, we list the number of bricks, "
            "the number of bricks with nonzero LRG density, and key density percentiles.\n\n"
        )
        f.write(variant_stats_df.to_markdown(index=False))
        f.write("\n\n")

        if not regions_summary_df.empty:
            f.write("## High density regions (baseline selection)\n\n")
            f.write(
                f"- Baseline variant: `{baseline_variant}` "
                f"(density column `{base_dens_col}`)\n"
            )
            f.write(
                f"- High density threshold: top **{top_p:.0f}th percentile** of nonzero density\n"
            )
            f.write(
                f"- Number of high density bricks: **{len(high_df)}** out of **{len(merged)}**\n"
            )
            f.write(
                f"- Number of contiguous components among high density bricks: "
                f"**{len(regions_summary_df)}**\n\n"
            )

            kept = regions_summary_df[regions_summary_df["keep_in_area_window"]]
            if not kept.empty:
                f.write(
                    "### Candidate regions within the area window\n\n"
                    f"- Area window: **[{min_region_area_deg2}, {max_region_area_deg2}] deg²**\n"
                    "- Regions below show total area, center coordinates, and mean LRG density per variant.\n\n"
                )
                # Show top 10 by baseline mean density
                top_regions = kept.head(10)
                f.write(top_regions.to_markdown(index=False))
                f.write("\n\n")
            else:
                f.write(
                    "No regions passed the specified area window filter. "
                    "You may want to relax `min_region_area_deg2` or `max_region_area_deg2` and rerun.\n\n"
                )
        else:
            f.write(
                "## High density regions\n\n"
                "Baseline selection has zero or negligible LRG density in this footprint, "
                "so no contiguous high density regions were identified.\n\n"
            )

        f.write("## Notes on physical interpretation\n\n")
        f.write(
            "- Each LRG variant corresponds to a different cut in color and magnitude space.\n"
            "  The stricter variants (for example `pure_massive`) trace the most massive halos, "
            "  while the more relaxed variants trace a more complete LRG population.\n"
        )
        f.write(
            "- Per brick LRG surface densities (in LRG per square degree) can be compared across variants "
            "  to study how selection choices trade purity versus completeness.\n"
        )
        f.write(
            "- The candidate regions listed above are contiguous sets of bricks that sit in the high tail "
            "  of the baseline LRG density distribution, which is where the projected dark matter halo "
            "  surface density is expected to be highest and where strong lensing is most likely.\n"
        )
        f.write(
            "- For ISEF documentation, you can use these tables to generate sky maps, density histograms, "
            "  and comparison plots between variants to quantify how lens search sensitivity depends on "
            "  LRG selection.\n"
        )

    print(f"[PHASE2] Wrote markdown report to {md_path}", flush=True)
    print(f"[PHASE2] Step 6 completed in {_time.time() - step_start:.1f}s", flush=True)
    
    total_elapsed = _time.time() - analysis_start
    print(f"[PHASE2] ✓ Analysis complete! Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)", flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze Phase 2 LRG hypergrid output and build candidate regions."
    )
    p.add_argument(
        "--hypergrid-csv",
        type=Path,
        required=True,
        help="Path to Phase 2 hypergrid CSV (EMR output per brick).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write analysis outputs.",
    )
    p.add_argument(
        "--bricks-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to a local bricks CSV with geometry and QA "
            "(contains brickname, ra, dec, ra1, ra2, dec1, dec2, psfsize_r, psfdepth_r, ebv). "
            "If not provided, bricks are fetched via TAP."
        ),
    )
    p.add_argument(
        "--tap-url",
        type=str,
        default="https://datalab.noirlab.edu/tap",
        help="TAP service URL for DR10 bricks (used if --bricks-csv is not given).",
    )
    p.add_argument("--ra-min", type=float, default=None, help="Optional RA minimum for analysis.")
    p.add_argument("--ra-max", type=float, default=None, help="Optional RA maximum for analysis.")
    p.add_argument("--dec-min", type=float, default=None, help="Optional Dec minimum for analysis.")
    p.add_argument("--dec-max", type=float, default=None, help="Optional Dec maximum for analysis.")
    p.add_argument(
        "--baseline-variant",
        type=str,
        default=None,
        help="Name of the LRG count column to use as the baseline selection. "
             "If not provided, analysis is run for ALL variants.",
    )
    p.add_argument(
        "--all-variants",
        action="store_true",
        help="Run analysis for ALL LRG variants as baseline (generates separate reports).",
    )
    p.add_argument(
        "--percentiles",
        type=str,
        default="90,95,99",
        help="Comma separated list of percentiles for density thresholds, for example '90,95,99'.",
    )
    p.add_argument(
        "--min-region-area-deg2",
        type=float,
        default=2.0,
        help="Minimum total area of a region to keep, in square degrees.",
    )
    p.add_argument(
        "--max-region-area-deg2",
        type=float,
        default=400.0,
        help="Maximum total area of a region to keep, in square degrees.",
    )
    p.add_argument(
        "--verify-adjacency",
        action="store_true",
        help="Verify KDTree adjacency against O(n²) brute force (for debugging).",
    )

    args = p.parse_args(argv)

    percents = [float(pct.strip()) for pct in args.percentiles.split(",") if pct.strip()]
    args.percentiles = tuple(percents)

    return args


# LRG hypergrid definitions (must match spark_phase2_lrg_hypergrid.py)
LRG_HYPERGRID_DEFINITIONS = [
    {
        "name": "v1_pure_massive",
        "z_mag_max": 20.0,
        "rz_min": 0.5,
        "zw1_min": 1.6,
        "description": "Strictest cut: pure massive LRGs only",
    },
    {
        "name": "v2_baseline_dr10",
        "z_mag_max": 20.4,
        "rz_min": 0.4,
        "zw1_min": 1.6,
        "description": "Baseline DR10 LRG selection",
    },
    {
        "name": "v3_color_relaxed",
        "z_mag_max": 20.4,
        "rz_min": 0.4,
        "zw1_min": 0.8,
        "description": "Relaxed z-W1 color cut",
    },
    {
        "name": "v4_mag_relaxed",
        "z_mag_max": 21.0,
        "rz_min": 0.4,
        "zw1_min": 0.8,
        "description": "Deeper magnitude limit (z < 21)",
    },
    {
        "name": "v5_very_relaxed",
        "z_mag_max": 21.5,
        "rz_min": 0.3,
        "zw1_min": 0.8,
        "description": "Most inclusive: faint + relaxed colors",
    },
]


def get_variant_definition(variant_name: str) -> Optional[dict]:
    """Get the hypergrid definition for a variant by name."""
    # Handle both 'n_lrg_v1_pure_massive' and 'v1_pure_massive' formats
    clean_name = variant_name.replace("n_lrg_", "")
    for defn in LRG_HYPERGRID_DEFINITIONS:
        if defn["name"] == clean_name:
            return defn
    return None


def _get_local_bricks_cache_path() -> Path:
    """
    Get the local cache path for DR10 South bricks metadata.
    
    Single cache file: dark_halo_scope/data/ls_dr10_south_bricks_metadata.csv
    
    Contains brick geometry (ra, dec, bounds) and QA info (PSF size, depth, EBV)
    from the ls_dr10.bricks_s TAP table.
    """
    # Find the dark_halo_scope root directory
    script_dir = Path(__file__).resolve().parent  # scripts/
    project_root = script_dir.parent  # dark_halo_scope/
    
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    return data_dir / "ls_dr10_south_bricks_metadata.csv"


def load_or_fetch_bricks(
    hypergrid_csv: Path,
    bricks_csv: Optional[Path],
    tap_url: str,
) -> Path:
    """
    Load bricks from local cache or fetch from TAP if not cached.
    
    DR10 bricks metadata is the same for all experiments, so we use a single
    cache file at: dark_halo_scope/data/dr10_south_bricks.csv
    
    Returns the path to the bricks CSV (either provided, cached, or newly fetched).
    """
    # If user provided a bricks CSV, use it directly
    if bricks_csv is not None:
        print(f"[BRICKS] Using provided bricks CSV: {bricks_csv}", flush=True)
        return bricks_csv
    
    # Check for local cache
    cache_path = _get_local_bricks_cache_path()
    
    # Load hypergrid to get required bricknames
    hg = pd.read_csv(hypergrid_csv)
    required_bricks = set(hg["brickname"].astype(str).tolist())
    
    if cache_path.exists():
        # Check if cache has all required bricks
        cached_df = pd.read_csv(cache_path)
        cached_bricks = set(cached_df["brickname"].astype(str).tolist())
        missing = required_bricks - cached_bricks
        
        if not missing:
            print(f"[BRICKS] ✓ Using local cache: {cache_path} ({len(cached_df)} bricks)", flush=True)
            return cache_path
        else:
            print(f"[BRICKS] Cache exists but missing {len(missing)} bricks, fetching all...", flush=True)
    
    # Fetch from TAP (streams to cache file periodically)
    print(f"[BRICKS] Fetching bricks from TAP...", flush=True)
    print(f"[BRICKS] Cache location: {cache_path}", flush=True)
    
    bricks_df = fetch_bricks_for_hypergrid(
        bricknames=list(required_bricks),
        tap_url=tap_url,
        cache_path=cache_path,
    )
    
    cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"[BRICKS] ✓ Cache complete: {len(bricks_df)} bricks ({cache_size_mb:.1f} MB)", flush=True)
    
    return cache_path


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    
    # Determine which variants to analyze
    if args.all_variants or args.baseline_variant is None:
        # Run for all variants
        variants_to_run = [f"n_lrg_{d['name']}" for d in LRG_HYPERGRID_DEFINITIONS]
        print(f"[PHASE2] Running analysis for ALL {len(variants_to_run)} variants", flush=True)
    else:
        variants_to_run = [args.baseline_variant]
    
    # Load or fetch bricks ONCE (uses local cache if available)
    bricks_csv_to_use = load_or_fetch_bricks(
        hypergrid_csv=args.hypergrid_csv,
        bricks_csv=args.bricks_csv,
        tap_url=args.tap_url,
    )
    
    for variant in variants_to_run:
        variant_short = variant.replace("n_lrg_", "")
        variant_output_dir = args.output_dir / variant_short
        
        print(f"\n{'='*70}", flush=True)
        print(f"[PHASE2] Analyzing variant: {variant}", flush=True)
        print(f"[PHASE2] Output directory: {variant_output_dir}", flush=True)
        print(f"{'='*70}\n", flush=True)
        
        analyze_hypergrid(
            hypergrid_csv=args.hypergrid_csv,
            output_dir=variant_output_dir,
            bricks_csv=bricks_csv_to_use,
        tap_url=args.tap_url,
        ra_min=args.ra_min,
        ra_max=args.ra_max,
        dec_min=args.dec_min,
        dec_max=args.dec_max,
            baseline_variant=variant,
        high_percentiles=args.percentiles,
        min_region_area_deg2=args.min_region_area_deg2,
        max_region_area_deg2=args.max_region_area_deg2,
            verify_adjacency=args.verify_adjacency,
        )
    
    # Write a master summary if running all variants
    if len(variants_to_run) > 1:
        master_md_path = args.output_dir / "phase2_all_variants_summary.md"
        with master_md_path.open("w") as f:
            f.write("# Phase 2 Hypergrid Analysis - All Variants Summary\n\n")
            f.write("## Analysis Parameters\n\n")
            f.write(f"- **Hypergrid CSV**: `{args.hypergrid_csv}`\n")
            f.write(f"- **Output directory**: `{args.output_dir}`\n")
            if args.bricks_csv:
                f.write(f"- **Bricks CSV**: `{args.bricks_csv}`\n")
            else:
                f.write(f"- **Bricks source**: TAP ({args.tap_url})\n")
            f.write(f"- **Percentiles**: {args.percentiles}\n")
            f.write(f"- **Min region area**: {args.min_region_area_deg2} deg²\n")
            f.write(f"- **Max region area**: {args.max_region_area_deg2} deg²\n")
            
            if args.ra_min is not None:
                f.write(f"- **RA range**: [{args.ra_min}, {args.ra_max}] deg\n")
                f.write(f"- **Dec range**: [{args.dec_min}, {args.dec_max}] deg\n")
            else:
                f.write("- **RA/Dec filter**: None (full sky)\n")
            
            f.write("\n## LRG Selection Variants\n\n")
            f.write("| Variant | z_mag_max | r-z min | z-W1 min | Description |\n")
            f.write("|---------|-----------|---------|----------|-------------|\n")
            for defn in LRG_HYPERGRID_DEFINITIONS:
                f.write(
                    f"| {defn['name']} | {defn['z_mag_max']} | {defn['rz_min']} | "
                    f"{defn['zw1_min']} | {defn['description']} |\n"
                )
            
            f.write("\n## Individual Variant Reports\n\n")
            for variant in variants_to_run:
                variant_short = variant.replace("n_lrg_", "")
                f.write(f"- [{variant_short}](./{variant_short}/phase2_hypergrid_analysis.md)\n")
        
        print(f"\n[PHASE2] Wrote master summary: {master_md_path}", flush=True)


if __name__ == "__main__":
    main()


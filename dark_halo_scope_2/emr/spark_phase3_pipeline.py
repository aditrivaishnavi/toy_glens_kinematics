
#!/usr/bin/env python3
""" 
Dark Halo Scope - Phase 3 (EMR / PySpark)
========================================

Design intent
------------
Phase 3 is where you transition from Phase 2 "per-brick" aggregate maps to:
  (3a) a *comprehensive* region metrics table for *all* connected components (after brick-quality cuts),
  (3b) one or more region-selection lists (top-K, stratified, train/val/test holdouts),
  (3c) parent catalogs built from DR10 South sweeps for the selected regions.

This file is a *single* script that can be invoked in 3 modes, so you can run it as three
separate EMR steps while keeping the logic in one place:

  --stage 3a   Region metrics for all connected components
  --stage 3b   Region selection (produces lists of regions/bricks)
  --stage 3c   Build parent catalogs (one sweep scan) for selected regions

Why this structure matters
--------------------------
Your constraint is that EMR iteration is slow/expensive. The code is therefore written to:
  - produce *rich* intermediate artifacts (metrics, scores, splits, multiple selection sets)
  - avoid reruns by capturing downstream-required metadata as early as possible
  - keep 3b lightweight (can run locally) by ensuring 3a outputs everything needed

Key scientific principles enforced
----------------------------------
  - Brick-quality cuts are applied before connected-component construction.
  - Region splits are deterministic (hash-based) to avoid hidden leakage across reruns.
  - Region scores are computed from *region-level* (area-weighted) densities, not per-brick means.

Assumptions
-----------
  - Phase 2 produced a per-brick table with at least:
        brickname, ra, dec, area_deg2,
        n_gal, n_lrg_v1..v5, lrg_density_v1..v5
    If area_deg2 is missing, we can compute it if bricks FITS is provided.
  - Optional: you provide a DR10 South bricks FITS (survey-bricks-dr10-south.fits.gz) in S3.
    If present, Phase 3a will join in Tier-2 metadata (nexp_g/r/z) and will also prefer brick
    quality columns from bricks FITS (psfsize_*, psfdepth_*, ebv).
  - Sweeps are read from an S3 cache prefix (fits or fits.gz). We also accept a sweep index of URLs,
    but will first look for the cached basename under the S3 prefix.

Python compatibility: 3.9.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import posixpath
import re
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import boto3
import numpy as np

from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql import functions as F
from pyspark.sql import types as T


# ----------------------------
# LRG variant definitions
# ----------------------------
# NOTE: These must match Phase 2 exactly (do NOT hand-wave). Keep in one place.
LRG_VARIANTS = {
    "v1_pure_massive":      {"z_max": 20.0, "rz_min": 0.5, "zw1_min": 1.6},
    "v2_baseline_dr10":    {"z_max": 20.4, "rz_min": 0.4, "zw1_min": 1.6},
    "v3_color_relaxed":    {"z_max": 20.4, "rz_min": 0.4, "zw1_min": 0.8},
    "v4_mag_relaxed":      {"z_max": 21.0, "rz_min": 0.4, "zw1_min": 0.8},
    "v5_very_relaxed":     {"z_max": 21.5, "rz_min": 0.3, "zw1_min": 0.8},
}

# Valid ranking modes for Stage 3b
VALID_RANKING_MODES = {"density", "n_lrg", "area_weighted", "psf_weighted"}

# Score column mapping
RANKING_MODE_TO_SCORE_COL = {
    "density": "score_density",
    "n_lrg": "score_n_lrg",
    "area_weighted": "score_area_weighted",
    "psf_weighted": "score_psf_weighted",
}


@dataclass
class RankingModeConfig:
    """Configuration for a single ranking mode in Stage 3b."""
    mode: str
    k_top: int
    k_stratified: int

    def __post_init__(self):
        if self.mode not in VALID_RANKING_MODES:
            raise ValueError(f"Invalid ranking mode '{self.mode}'. Valid: {VALID_RANKING_MODES}")
        if self.k_top < 0:
            raise ValueError(f"k_top must be >= 0, got {self.k_top}")
        if self.k_stratified < 0:
            raise ValueError(f"k_stratified must be >= 0, got {self.k_stratified}")


def parse_ranking_config(config_str: str) -> List[RankingModeConfig]:
    """Parse --ranking-config string into list of RankingModeConfig.

    Format: "mode:k_top:k_strat,mode:k_top:k_strat,..."
    Example: "n_lrg:100:100,psf_weighted:100:100,density:30:0,area_weighted:30:0"

    Returns list of RankingModeConfig objects.
    """
    configs = []
    for part in config_str.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(":")
        if len(tokens) != 3:
            raise ValueError(
                f"Invalid ranking-config entry '{part}'. "
                f"Expected format 'mode:k_top:k_strat'. Example: 'n_lrg:100:50'"
            )
        mode, k_top_str, k_strat_str = tokens
        mode = mode.strip().lower()
        try:
            k_top = int(k_top_str.strip())
            k_strat = int(k_strat_str.strip())
        except ValueError:
            raise ValueError(f"Invalid k values in '{part}'. k_top and k_strat must be integers.")
        configs.append(RankingModeConfig(mode=mode, k_top=k_top, k_stratified=k_strat))
    if not configs:
        raise ValueError("ranking-config is empty. Provide at least one mode.")
    return configs


def build_ranking_configs_from_args(args) -> List[RankingModeConfig]:
    """Build list of RankingModeConfig from argparse namespace.

    If --ranking-config is provided, parse it.
    Otherwise, use --ranking-modes with global --k-top and --k-stratified.
    """
    if args.ranking_config:
        return parse_ranking_config(args.ranking_config)
    else:
        # Fall back to legacy mode
        modes = [m.strip().lower() for m in args.ranking_modes.split(",") if m.strip()]
        return [
            RankingModeConfig(mode=m, k_top=args.k_top, k_stratified=args.k_stratified)
            for m in modes
        ]


# ----------------------------
# Stage config JSON writer
# ----------------------------

def write_stage_config_json(spark: "SparkSession", s3_path: str, stage: str, args, ranking_configs: Optional[List[RankingModeConfig]] = None) -> None:
    """Write a JSON file capturing all configuration parameters for reproducibility.

    This creates a _stage_config.json file at the given S3 path with:
    - stage identifier (3a, 3b, or 3c)
    - timestamp of the run
    - all CLI arguments used
    - parsed ranking configurations (for 3b)
    - selection counts (for 3b)

    Args:
        spark: SparkSession (needed for Spark-based JSON write)
        s3_path: S3 output prefix (e.g., s3://bucket/phase3b/v3_color_relaxed)
        stage: Stage identifier ("3a", "3b", or "3c")
        args: argparse Namespace with all CLI args
        ranking_configs: List of RankingModeConfig (for stage 3b only)
    """
    from datetime import datetime
    config_data = {
        "stage": stage,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "variant": getattr(args, "variant", "v3_color_relaxed"),
        "output_s3": args.output_s3,
    }

    # Stage-specific configs
    if stage == "3a":
        config_data["stage_3a"] = {
            "phase2_results_s3": getattr(args, "phase2_results_s3", None),
            "bricks_fits_s3": getattr(args, "bricks_fits_s3", None),
            "bricks_fits_columns": getattr(args, "bricks_fits_columns", None),
            "max_ebv": getattr(args, "max_ebv", None),
            "max_psf_r_arcsec": getattr(args, "max_psf_r_arcsec", None),
            "min_psfdepth_r": getattr(args, "min_psfdepth_r", None),
            "adj_ra_eps": getattr(args, "adj_ra_eps", None),
            "adj_dec_eps": getattr(args, "adj_dec_eps", None),
            "adj_cell_size": getattr(args, "adj_cell_size", None),
            "split_seed": getattr(args, "split_seed", None),
            "test_frac": getattr(args, "test_frac", None),
            "val_frac": getattr(args, "val_frac", None),
            "area_weighted_alpha": getattr(args, "area_weighted_alpha", None),
            "psf_ref": getattr(args, "psf_ref", None),
            "psf_sigma": getattr(args, "psf_sigma", None),
            "depth_ref": getattr(args, "depth_ref", None),
            "depth_sigma": getattr(args, "depth_sigma", None),
            "ebv_k": getattr(args, "ebv_k", None),
        }

    elif stage == "3b":
        # Ranking configuration
        ranking_list = []
        if ranking_configs:
            for rc in ranking_configs:
                ranking_list.append({
                    "mode": rc.mode,
                    "k_top": rc.k_top,
                    "k_stratified": rc.k_stratified,
                })
        config_data["stage_3b"] = {
            "ranking_config_raw": getattr(args, "ranking_config", None),
            "ranking_modes_raw": getattr(args, "ranking_modes", None),
            "k_top_global": getattr(args, "k_top", None),
            "k_stratified_global": getattr(args, "k_stratified", None),
            "ranking_configs_parsed": ranking_list,
            "emit_balanced_stratified": getattr(args, "emit_balanced_stratified", 1),
            "strata_quantiles": getattr(args, "strata_quantiles", None),
            "strata_weights": getattr(args, "strata_weights", None),
            "exclude_splits": getattr(args, "exclude_splits", None),
        }

    elif stage == "3c":
        config_data["stage_3c"] = {
            "region_selection_s3": getattr(args, "region_selection_s3", None),
            "only_selection_sets": getattr(args, "only_selection_sets", None),
            "parent_output_mode": getattr(args, "parent_output_mode", "union"),
            "max_selected_bricks": getattr(args, "max_selected_bricks", None),
            "max_selected_regions": getattr(args, "max_selected_regions", None),
            "sweep_index_s3": getattr(args, "sweep_index_s3", None),
            "s3_sweep_cache_prefix": getattr(args, "s3_sweep_cache_prefix", None),
            "sweep_partitions": getattr(args, "sweep_partitions", None),
            "chunk_size": getattr(args, "chunk_size", None),
            "use_mw_correction": getattr(args, "use_mw_correction", 0),
            "emit_mw_corrected_mags": getattr(args, "emit_mw_corrected_mags", 1),
        }

    # Idempotency flags
    config_data["idempotency"] = {
        "skip_if_exists": getattr(args, "skip_if_exists", 1),
        "force": getattr(args, "force", 0),
    }

    # LRG variants (same for all stages, for reference)
    config_data["lrg_variants"] = LRG_VARIANTS

    # Write as JSON to S3
    import json
    json_str = json.dumps(config_data, indent=2, default=str)
    json_path = s3_path.rstrip("/") + "/_stage_config.json"

    # Use boto3 to write JSON directly
    try:
        bucket, key = parse_s3_uri(json_path)
        import boto3
        s3 = boto3.client("s3")
        s3.put_object(Bucket=bucket, Key=key, Body=json_str.encode("utf-8"), ContentType="application/json")
        print(f"[{stage}] Wrote config JSON: {json_path}")
    except Exception as e:
        print(f"[{stage}] Warning: failed to write config JSON to {json_path}: {e}")


# ----------------------------
# S3 utilities
# ----------------------------
_S3_URI_RE = re.compile(r"^s3://([^/]+)/?(.*)$")


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    m = _S3_URI_RE.match(uri)
    if not m:
        raise ValueError(f"Not an s3:// URI: {uri}")
    bucket, key = m.group(1), m.group(2)
    return bucket, key


def s3_basename(uri_or_url: str) -> str:
    # works for s3://, https://, and plain paths
    return posixpath.basename(uri_or_url.strip())

def s3_prefix_exists(prefix_uri: str, *, client=None) -> bool:
    '''Return True if the given S3 prefix contains at least one object.'''
    if not prefix_uri:
        return False
    if not prefix_uri.startswith('s3://'):
        raise ValueError(f'Expected s3:// prefix, got: {prefix_uri!r}')
    bucket, key_prefix = parse_s3_uri(prefix_uri)
    key_prefix = key_prefix.rstrip('/') + '/'
    c = client or boto3.client('s3')
    resp = c.list_objects_v2(Bucket=bucket, Prefix=key_prefix, MaxKeys=1)
    return bool(resp.get('KeyCount', 0))


def download_s3_to_local(s3_uri: str, local_path: str, *, client=None) -> str:
    bucket, key = parse_s3_uri(s3_uri)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if client is None:
        client = boto3.client("s3")
    client.download_file(bucket, key, local_path)
    return local_path


def try_download_sweep_to_local(
    sweep_entry: str,
    s3_cache_prefix: str,
    local_dir: str,
    *,
    client=None,
) -> Optional[str]:
    """Try to fetch a DR10 sweep file into a local cache directory.

    Inputs:
      - sweep_entry: either an https URL (NERSC portal) or an s3:// URI
      - s3_cache_prefix: s3:// bucket/prefix where you have cached sweep files
      - local_dir: local temp dir on executor

    Behavior:
      - If sweep_entry is an https URL, we map it to the S3 cache using the basename.
      - We try BOTH .fits and .fits.gz forms, because your cache may store gzipped FITS.

    Returns local path on success, or None.
    """
    entry = (sweep_entry or '').strip()
    if not entry:
        return None

    base = s3_basename(entry)
    # Strip query params if any
    if '?' in base:
        base = base.split('?', 1)[0]

    # Candidate basenames to try
    cand = [base]
    if base.endswith('.fits') and not base.endswith('.fits.gz'):
        cand.append(base + '.gz')
        cand.append(base[:-5] + '.fits.gz')
    elif base.endswith('.fits.gz'):
        cand.append(base[:-3])

    # De-duplicate, preserve order
    seen = set()
    cand = [c for c in cand if not (c in seen or seen.add(c))]

    s3_candidates = []

    # Prefer cached S3 sweeps if a cache prefix is given.
    if s3_cache_prefix:
        pref = s3_cache_prefix.rstrip('/')
        for c in cand:
            s3_candidates.append(pref + '/' + c)

    # If the entry itself is s3://, also try it plus extension variants.
    if entry.startswith('s3://'):
        s3_candidates.append(entry)
        try:
            bucket, key = parse_s3_uri(entry)
            key_prefix = key.rsplit('/', 1)[0] if '/' in key else ''
            for c in cand:
                new_key = (key_prefix + '/' if key_prefix else '') + c
                s3_candidates.append(f's3://{bucket}/{new_key}')
        except Exception:
            pass

    # De-duplicate s3 URIs
    seen2 = set()
    s3_candidates = [u for u in s3_candidates if not (u in seen2 or seen2.add(u))]

    for s3_uri in s3_candidates:
        try:
            local_name = s3_basename(s3_uri)
            if '?' in local_name:
                local_name = local_name.split('?', 1)[0]
            local_path = os.path.join(local_dir, local_name)
            download_s3_to_local(s3_uri, local_path, client=client)
            return local_path
        except Exception:
            continue

    return None


# ----------------------------
# Connected components on driver
# ----------------------------
@dataclass
class BrickForGraph:
    brickname: str
    ra: float
    dec: float


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _delta_ra_deg(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 360.0 - d)


def build_connected_components(
    bricks: List[BrickForGraph],
    *,
    ra_eps: float = 0.30,
    dec_eps: float = 0.30,
    cell_size: float = 0.25,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Approximate brick adjacency via a simple sky grid hash.

    Two bricks are neighbors if their centers are within about one brick in RA and Dec.

    Returns:
      - brickname -> region_id (deterministic)
      - brickname -> neighbor_degree (within the adjacency graph)
    """
    n = len(bricks)
    uf = UnionFind(n)

    # Spatial hash: (ira, idec) -> indices
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for i, b in enumerate(bricks):
        ira = int(math.floor(b.ra / cell_size))
        idec = int(math.floor((b.dec + 90.0) / cell_size))
        buckets.setdefault((ira, idec), []).append(i)

    # Check neighbors within nearby buckets
    degree = [0] * n
    for i, b in enumerate(bricks):
        ira = int(math.floor(b.ra / cell_size))
        idec = int(math.floor((b.dec + 90.0) / cell_size))
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                key = (ira + di, idec + dj)
                for j in buckets.get(key, []):
                    if j <= i:
                        continue
                    other = bricks[j]
                    if abs(b.dec - other.dec) <= dec_eps and _delta_ra_deg(b.ra, other.ra) <= ra_eps:
                        uf.union(i, j)
                        degree[i] += 1
                        degree[j] += 1

    # Assign region IDs deterministically
    comp_to_bricks: Dict[int, List[int]] = {}
    for i in range(n):
        root = uf.find(i)
        comp_to_bricks.setdefault(root, []).append(i)

    # Sort components by minimum brickname
    comp_list = sorted(comp_to_bricks.keys(), key=lambda c: min(bricks[i].brickname for i in comp_to_bricks[c]))

    brick_to_region: Dict[str, int] = {}
    brick_to_degree: Dict[str, int] = {}
    for region_id, comp in enumerate(comp_list):
        for i in comp_to_bricks[comp]:
            brick_to_region[bricks[i].brickname] = region_id
            brick_to_degree[bricks[i].brickname] = degree[i]

    return brick_to_region, brick_to_degree


# ----------------------------
# Math helpers
# ----------------------------

def nanomaggies_to_mag(flux_nmgy: np.ndarray) -> np.ndarray:
    """Convert nanomaggies to AB magnitude. Returns nan for non-positive flux."""
    flux = np.asarray(flux_nmgy, dtype=float)
    out = np.full_like(flux, np.nan, dtype=float)
    m = (flux > 0) & np.isfinite(flux)
    out[m] = 22.5 - 2.5 * np.log10(flux[m])
    return out


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise division with nan for division by zero or invalid inputs."""
    out = np.full_like(a, np.nan, dtype=np.float32)
    m = (b != 0) & np.isfinite(a) & np.isfinite(b)
    out[m] = a[m] / b[m]
    return out


def load_bricks_fits_subset(
    bricks_fits_s3: str,
    wanted_bricknames: Sequence[str],
    *,
    client,
    columns: Optional[Sequence[str]] = None,
) -> "np.ndarray":
    """Load a subset of the DR10 South bricks table from a FITS file stored in S3.

    This is intentionally done on the DRIVER (not distributed), because:
      - We only need a subset of rows (the bricks present in Phase 2 results, ~1e4).
      - Avoids distributing FITS parsing across executors and repeated downloads.

    Returns a numpy structured array with requested columns (if available).

    NOTE: if columns is None, we will keep all columns (not recommended).
    """
    from astropy.io import fits  # imported lazily for EMR bootstrap friendliness

    wanted = set(wanted_bricknames)
    with tempfile.TemporaryDirectory() as td:
        local_path = os.path.join(td, "bricks.fits.gz")
        download_s3_to_local(bricks_fits_s3, local_path, client=client)
        with fits.open(local_path, memmap=False) as hdul:
            data = hdul[1].data
            # Normalize brickname column
            brickname_col = None
            for cand in ("BRICKNAME", "brickname"):
                if cand in data.names:
                    brickname_col = cand
                    break
            if brickname_col is None:
                raise RuntimeError("Bricks FITS missing BRICKNAME column")

            names = list(data.names)
            if columns is not None:
                # Case-insensitive column matching
                names_upper = {n.upper(): n for n in names}
                keep = []
                for c in columns:
                    actual_name = names_upper.get(c.upper())
                    if actual_name:
                        keep.append(actual_name)
                # Always keep brickname
                if brickname_col not in keep:
                    keep = [brickname_col] + keep
            else:
                keep = names

            # Filter rows
            bricknames = np.array(data[brickname_col]).astype(str)
            mask = np.array([bn in wanted for bn in bricknames])
            sub = data[mask]

            # Build structured array of kept columns
            out = np.zeros(len(sub), dtype=[(c, sub[c].dtype) for c in keep])
            for c in keep:
                out[c] = sub[c]
            return out


def bricks_subset_to_spark_df(spark: SparkSession, arr: "np.ndarray") -> DataFrame:
    rows = []
    for i in range(len(arr)):
        d = {name.lower(): (arr[name][i].item() if hasattr(arr[name][i], "item") else arr[name][i]) for name in arr.dtype.names}
        # Ensure brickname normalized
        rows.append(d)
    return spark.createDataFrame(rows)


# ----------------------------
# Spark setup
# ----------------------------

def make_spark(app_name: str, shuffle_partitions: int) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ----------------------------
# Stage 3a: region metrics
# ----------------------------

def stage_3a_region_metrics(args: argparse.Namespace) -> None:
    # SKIP: stage 3a
    out3a = f"{args.output_s3.rstrip('/')}/phase3a/{args.variant}"
    out_bricks = out3a + '/bricks_with_region'
    out_regions = out3a + '/region_metrics'
    if args.skip_if_exists and (not args.force) and s3_prefix_exists(out_bricks) and s3_prefix_exists(out_regions):
        print(f"[SKIP] Stage 3a outputs already exist under {out3a} (use --force 1 to overwrite).")
        return

    spark = make_spark("DHS Phase3a RegionMetrics", args.shuffle_partitions)
    # Phase 2 per-brick metrics
    p2_raw = (
        spark.read.option("header", "true").option("inferSchema", "true")
        .csv(args.phase2_results_s3)
    )

    # --- Normalize Phase 2 column names (robust to naming differences) ---
    # We standardize to:
    #   brickname, ra, dec, area_deg2, n_gal,
    #   n_lrg_v1..v5, lrg_density_v1..v5

    def _find_col(cols, candidates):
        lower = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand in cols:
                return cand
            if cand.lower() in lower:
                return lower[cand.lower()]
        return None

    cols = p2_raw.columns

    # brickname
    brick_col = _find_col(cols, ["brickname", "BRICKNAME", "brick"])
    if brick_col is None:
        raise RuntimeError("Phase 2 results missing brickname column")

    # area and total-galaxy counts
    area_col = _find_col(cols, ["area_deg2", "brick_area_deg2", "area", "brick_area"])  # may be None
    ngal_col = _find_col(cols, ["n_gal", "n_galaxies", "n_objects", "n_all", "n_total"])  # may be None

    if ngal_col is None:
        raise RuntimeError("Phase 2 results missing a total-galaxy count column (expected one of n_gal, n_objects, ...).")

    if area_col is None and not args.bricks_fits_s3:
        raise RuntimeError("Phase 2 results missing brick area; provide --bricks-fits-s3 so area can be computed from ra1/ra2/dec1/dec2.")

    # brick center coordinates (optional, can be recovered from bricks meta)
    ra_col = _find_col(cols, ["ra", "brick_ra", "ra_center", "ra_centroid"])
    dec_col = _find_col(cols, ["dec", "brick_dec", "dec_center", "dec_centroid"])

    # variant count columns
    var_order = ["v1_pure_massive", "v2_baseline_dr10", "v3_color_relaxed", "v4_mag_relaxed", "v5_very_relaxed"]
    count_cols = {}
    dens_cols = {}
    for i, vname in enumerate(var_order, start=1):
        count_cols[i] = _find_col(
            cols,
            [
                f"n_lrg_v{i}",
                f"n_lrg_{vname}",
                f"count_{vname}",
                f"n_{vname}",
                f"n_lrg_{vname.replace('v', 'v')}",
            ],
        )
        dens_cols[i] = _find_col(
            cols,
            [
                f"lrg_density_v{i}",
                f"lrg_density_{vname}",
                f"density_{vname}",
            ],
        )

    missing_counts = [i for i in range(1, 6) if count_cols[i] is None]
    if missing_counts:
        raise RuntimeError(
            "Phase 2 results missing LRG count columns for: " + ",".join(str(i) for i in missing_counts)
        )

    p2 = p2_raw.select(
        F.col(brick_col).alias("brickname"),
        (F.col(ra_col).cast("double").alias("ra") if ra_col else F.lit(None).cast("double").alias("ra")),
        (F.col(dec_col).cast("double").alias("dec") if dec_col else F.lit(None).cast("double").alias("dec")),
        (F.col(area_col).cast("double").alias("area_deg2") if area_col else F.lit(None).cast("double").alias("area_deg2")),
        (F.col(ngal_col).cast("long").alias("n_gal") if ngal_col else F.lit(None).cast("long").alias("n_gal")),
        *[F.col(count_cols[i]).cast("long").alias(f"n_lrg_v{i}") for i in range(1, 6)],
        *[(F.col(dens_cols[i]).cast("double").alias(f"lrg_density_v{i}") if dens_cols[i] else F.lit(None).cast("double").alias(f"lrg_density_v{i}")) for i in range(1, 6)],
    )

    # Collect list of bricknames for bricks-FITS subset join
    bricknames = [r[0] for r in p2.select("brickname").distinct().collect()]

    bricks_meta_df = None
    meta_cols_present: List[str] = []
    if args.bricks_fits_s3:
        client = boto3.client("s3")
        cols = [c.strip() for c in args.bricks_fits_columns.split(",") if c.strip()]
        arr = load_bricks_fits_subset(args.bricks_fits_s3, bricknames, client=client, columns=cols)
        bricks_meta_df = bricks_subset_to_spark_df(spark, arr)
        meta_cols_present = bricks_meta_df.columns

        # Avoid duplicate column names on join: if Phase2 already has a REAL column (not a null placeholder), keep Phase2's copy
        # But keep bricks_meta versions of ra/dec/area if Phase2 doesn't have them (ra_col/dec_col/area_col were None)
        cols_phase2_actually_has = set(p2.columns)
        cols_to_drop_from_p2 = []
        if ra_col is None:
            cols_phase2_actually_has.discard("ra")
            cols_to_drop_from_p2.append("ra")
        if dec_col is None:
            cols_phase2_actually_has.discard("dec")
            cols_to_drop_from_p2.append("dec")
        if area_col is None:
            cols_phase2_actually_has.discard("area_deg2")
            cols_to_drop_from_p2.append("area_deg2")
        
        # Drop NULL placeholder columns from p2 so bricks_meta can provide them
        if cols_to_drop_from_p2:
            p2 = p2.drop(*cols_to_drop_from_p2)
        
        overlap = [c for c in bricks_meta_df.columns if c != 'brickname' and c in cols_phase2_actually_has]
        if overlap:
            bricks_meta_df = bricks_meta_df.drop(*overlap)

    # Join meta
    bricks = p2
    if bricks_meta_df is not None:
        # bricks_meta_df columns are lower-case; ensure join key
        if "brickname" not in bricks_meta_df.columns:
            raise RuntimeError("Bricks meta subset is missing brickname")
        bricks = bricks.join(F.broadcast(bricks_meta_df), on="brickname", how="left")

    # Prefer quality columns from bricks metadata if present
    # (Phase2 results may already contain these; but if both exist, keep the meta-sourced ones)
    # Normalize names to expected: psfsize_r, psfdepth_r, ebv, nexp_r/g/z
    rename_map = {}
    for c in bricks.columns:
        lc = c.lower()
        if lc == "psfsize_r" or lc == "psfdepth_r" or lc == "ebv":
            continue
        # Common bricks-fits naming: PSFSIZE_R, PSFDEPTH_R, EBV, NEXP_R
        if lc == "psfsize_r":
            rename_map[c] = "psfsize_r"
        if lc == "psfdepth_r":
            rename_map[c] = "psfdepth_r"
        if lc == "nexp_r":
            rename_map[c] = "nexp_r"
    for old, new in rename_map.items():
        bricks = bricks.withColumnRenamed(old, new)

    # If area_deg2 missing, attempt compute from ra1/ra2/dec1/dec2 in meta
    if "area_deg2" not in bricks.columns:
        needed = {"ra1", "ra2", "dec1", "dec2"}
        if not needed.issubset(set(bricks.columns)):
            raise RuntimeError("area_deg2 missing and cannot be computed: need ra1,ra2,dec1,dec2 from bricks meta")
        # area in deg^2: (ra2-ra1) * (sin dec2 - sin dec1) in radians converted to deg^2
        ra1 = F.radians(F.col("ra1"))
        ra2 = F.radians(F.col("ra2"))
        dec1 = F.radians(F.col("dec1"))
        dec2 = F.radians(F.col("dec2"))
        area_sr = (ra2 - ra1) * (F.sin(dec2) - F.sin(dec1))
        area_deg2 = area_sr * F.lit((180.0 / math.pi) ** 2)
        bricks = bricks.withColumn("area_deg2", area_deg2)

    # Ensure per-brick LRG densities exist for each variant. If Phase 2 did not emit per-brick densities, we compute count/area here.
    for i in range(1, 6):
        dens_col = f"lrg_density_v {i}"
        cnt_col = f"n_lrg_v{i}"
        if dens_col not in bricks.columns:
            bricks = bricks.withColumn(dens_col, F.col(cnt_col) / F.col("area_deg2"))
        else:
            bricks = bricks.withColumn(
                dens_col,
                F.when(F.col(dens_col).isNull(), F.col(cnt_col) / F.col("area_deg2")).otherwise(F.col(dens_col)),
            )

    # Ensure coordinates exist (needed for contiguity graph). If missing here, provide them via bricks-fits columns (ra, dec).
    if bricks.filter(F.col("ra").isNull() | F.col("dec").isNull()).limit(1).count() > 0:
        raise RuntimeError("Brick RA/Dec missing after join. Provide ra/dec in Phase 2 results or include them in --bricks-fits-columns.")

    # Derived brick-level diagnostics
    bricks = bricks.withColumn("gal_density", F.col("n_gal") / F.col("area_deg2"))
    bricks = bricks.withColumn("lrg_frac_v3", F.when(F.col("n_gal") > 0, F.col("n_lrg_v3") / F.col("n_gal")).otherwise(F.lit(None)))

    # Base quality cuts (brick-level)
    # These cuts govern the connected-component graph. Bad bricks are excluded entirely.
    q = (
        (F.col("ebv") <= F.lit(args.max_ebv))
        & (F.col("psfsize_r") <= F.lit(args.max_psf_r_arcsec))
        & (F.col("psfdepth_r") >= F.lit(args.min_psfdepth_r))
    )
    bricks_q = bricks.filter(q)

    # Build connected components on driver using only (brickname, ra, dec)
    # Avoid pandas dependency and keep memory bounded.
    rows = bricks_q.select("brickname", "ra", "dec").collect()
    brick_list: List[BrickForGraph] = [
        BrickForGraph(str(r["brickname"]), float(r["ra"]), float(r["dec"]))
        for r in rows
    ]

    brick_to_region, brick_to_degree = build_connected_components(
        brick_list,
        ra_eps=args.adj_ra_eps,
        dec_eps=args.adj_dec_eps,
        cell_size=args.adj_cell_size,
    )

    # Spark mapping tables
    mapping_rows = [(k, int(v), int(brick_to_degree.get(k, 0))) for k, v in brick_to_region.items()]
    map_schema = T.StructType([
        T.StructField("brickname", T.StringType(), False),
        T.StructField("region_id", T.IntegerType(), False),
        T.StructField("brick_degree", T.IntegerType(), False),
    ])
    brick_region_df = spark.createDataFrame(mapping_rows, schema=map_schema)

    bricks_q = bricks_q.join(F.broadcast(brick_region_df), on="brickname", how="inner")

    # Output enriched bricks table (useful later: region->bricks, cutout download, etc.)
    out3a = args.output_s3.rstrip("/") + f"/phase3a/{args.variant}"
    bricks_out = out3a + "/bricks_with_region"
    bricks_q.write.mode("overwrite").parquet(bricks_out)

    # Region-level aggregations
    # Use area-weighted density as the physically meaningful density.
    agg = [
        F.countDistinct("brickname").alias("n_bricks"),
        F.sum("area_deg2").alias("area_deg2"),
        F.sum("n_gal").alias("n_gal_total"),
        F.sum(F.col("gal_density") * F.col("area_deg2")).alias("gal_density_area_sum"),
    ]

    for i in range(1, 6):
        agg.append(F.sum(F.col(f"n_lrg_v{i}")).alias(f"n_lrg_v{i}_total"))

    # Brick-quality stats
    def pct(col: str, ps: List[float]) -> F.Column:
        return F.percentile_approx(F.col(col), F.array(*[F.lit(p) for p in ps]), 10000)

    quality_cols = [
        ("psfsize_r", "psfsize_r"),
        ("psfdepth_r", "psfdepth_r"),
        ("ebv", "ebv"),
        ("brick_degree", "brick_degree"),
        ("gal_density", "gal_density"),
        ("lrg_frac_v3", "lrg_frac_v3"),
    ]
    # Optional Tier-2 columns if present
    for c in ("nexp_g", "nexp_r", "nexp_z"):
        if c in bricks_q.columns:
            quality_cols.append((c, c))

    for col, base in quality_cols:
        agg.extend([
            F.avg(col).alias(f"{base}_mean"),
            F.stddev(col).alias(f"{base}_std"),
            pct(col, [0.1, 0.5, 0.9]).alias(f"{base}_p10_p50_p90"),
        ])

    # Bounding box spans (diagnostic for compactness)
    agg.extend([
        F.min("ra").alias("ra_min"),
        F.max("ra").alias("ra_max"),
        F.min("dec").alias("dec_min"),
        F.max("dec").alias("dec_max"),
    ])

    regions = bricks_q.groupBy("region_id").agg(*agg)

    # Post-process percentile arrays into scalar columns
    for col, base in quality_cols:
        arr = f"{base}_p10_p50_p90"
        regions = (
            regions
            .withColumn(f"{base}_p10", F.col(arr)[0])
            .withColumn(f"{base}_p50", F.col(arr)[1])
            .withColumn(f"{base}_p90", F.col(arr)[2])
            .drop(arr)
        )

    # Area-weighted densities per variant
    for i in range(1, 6):
        regions = regions.withColumn(
            f"lrg_density_v{i}_areaweighted",
            F.col(f"n_lrg_v{i}_total") / F.col("area_deg2"),
        )

    # Compactness proxy: area / (bbox area * cos(dec_mid))
    dec_mid = (F.col("dec_min") + F.col("dec_max")) / F.lit(2.0)
    bbox_area = (F.col("ra_max") - F.col("ra_min")) * (F.col("dec_max") - F.col("dec_min")) * F.cos(F.radians(dec_mid))
    regions = regions.withColumn("compactness", F.when(bbox_area > 0, F.col("area_deg2") / bbox_area).otherwise(F.lit(None)))

    # Deterministic split assignment by hash(region_id, seed)
    # No Python UDF: use xxhash64 for speed and determinism.
    h = F.abs(F.xxhash64(F.col("region_id"), F.lit(int(args.split_seed))))
    u = (h % F.lit(1000000)) / F.lit(1000000.0)
    regions = regions.withColumn("split_u", u)

    regions = regions.withColumn(
        "region_split",
        F.when(F.col("split_u") < F.lit(args.test_frac), F.lit("test"))
         .when(F.col("split_u") < F.lit(args.test_frac + args.val_frac), F.lit("val"))
         .otherwise(F.lit("train"))
    )

    # Scores used by 3b selection
    # We define scores using *region-level* area-weighted v3 density and *median* quality metrics.
    # These are intended as heuristics for ranking candidate regions.
    v3_density = F.col("lrg_density_v3_areaweighted")
    w_psf = F.exp(-(F.col("psfsize_r_p50") - F.lit(args.psf_ref)) / F.lit(args.psf_sigma))
    w_psf = F.when(w_psf < 0, F.lit(0.0)).when(w_psf > 1, F.lit(1.0)).otherwise(w_psf)
    w_depth = F.lit(1.0) / (F.lit(1.0) + F.exp(-(F.col("psfdepth_r_p50") - F.lit(args.depth_ref)) / F.lit(args.depth_sigma)))
    w_depth = F.when(w_depth < 0, F.lit(0.0)).when(w_depth > 1, F.lit(1.0)).otherwise(w_depth)
    w_ebv = F.exp(-F.lit(args.ebv_k) * F.col("ebv_p50"))
    w_ebv = F.when(w_ebv < 0, F.lit(0.0)).when(w_ebv > 1, F.lit(1.0)).otherwise(w_ebv)

    regions = regions.withColumn("score_density", v3_density)
    regions = regions.withColumn("score_n_lrg", F.col("n_lrg_v3_total"))
    # area_weighted: trade off overdensity (density) with larger sky area to reduce cosmic variance and broaden training diversity
    regions = regions.withColumn("score_area_weighted", v3_density * F.pow(F.col("area_deg2"), F.lit(args.area_weighted_alpha)))
    regions = regions.withColumn("score_psf_weighted", v3_density * w_psf * w_depth * w_ebv)
    regions = regions.withColumn("w_psf", w_psf).withColumn("w_depth", w_depth).withColumn("w_ebv", w_ebv)

    # Persist region metrics
    regions_out = out3a + "/region_metrics"
    regions.write.mode("overwrite").parquet(regions_out)

    # Also write a compact CSV for quick inspection
    (regions
        .coalesce(1)
        .write.mode("overwrite").option("header", "true")
        .csv(out3a + "/region_metrics_csv")
    )


    # Metadata for provenance
    meta = {
        "phase": "3a",
        "variant": args.variant,
        "phase2_results_s3": args.phase2_results_s3,
        "bricks_fits_s3": args.bricks_fits_s3,
        "quality_cuts": {
            "max_ebv": args.max_ebv,
            "max_psf_r_arcsec": args.max_psf_r_arcsec,
            "min_psfdepth_r": args.min_psfdepth_r,
        },
        "adjacency": {
            "ra_eps": args.adj_ra_eps,
            "dec_eps": args.adj_dec_eps,
            "cell_size": args.adj_cell_size,
        },
        "splits": {
            "seed": args.split_seed,
            "test_frac": args.test_frac,
            "val_frac": args.val_frac,
        },
        "score_params": {
            "area_weighted_alpha": args.area_weighted_alpha,
            "psf_ref": args.psf_ref,
            "psf_sigma": args.psf_sigma,
            "depth_ref": args.depth_ref,
            "depth_sigma": args.depth_sigma,
            "ebv_k": args.ebv_k,
        },
        "notes": "bricks_with_region is only for bricks passing base quality cuts",
    }

    # Write metadata JSON to S3 via Spark
    meta_df = spark.createDataFrame([Row(json=json.dumps(meta))])
    meta_df.coalesce(1).write.mode("overwrite").text(out3a + "/_metadata_json")

    # Write configuration JSON for reproducibility
    write_stage_config_json(spark, out3a, "3a", args)
    print(f"[3a] Stage 3a complete. Output: {out3a}")


# ----------------------------
# Stage 3b: region selection
# ----------------------------

def stage_3b_select_regions(args: argparse.Namespace) -> None:
    # SKIP: stage 3b
    out3b = f"{args.output_s3.rstrip('/')}/phase3b/{args.variant}"
    out_sel = out3b + '/region_selections'
    if args.skip_if_exists and (not args.force) and s3_prefix_exists(out_sel):
        print(f"[SKIP] Stage 3b outputs already exist under {out3b} (use --force 1 to overwrite).")
        return

    spark = make_spark("DHS Phase3b SelectRegions", args.shuffle_partitions)

    base = args.output_s3.rstrip("/") + f"/phase3a/{args.variant}"
    regions = spark.read.parquet(base + "/region_metrics")

    # Candidate pool: optionally exclude test regions (blind holdout)
    pool = regions
    if args.exclude_splits:
        excl = [s.strip() for s in args.exclude_splits.split(",") if s.strip()]
        if excl:
            pool = pool.filter(~F.col("region_split").isin(excl))

    # Create a density quantile bucket for stratified selection
    # We stratify on v3 area-weighted density by default.
    dens = "lrg_density_v3_areaweighted"
    # Use approxQuantile on driver (small table)
    qs = [float(x) for x in args.strata_quantiles.split(",")]
    qs = sorted(set(qs))
    if qs[0] != 0.0:
        qs = [0.0] + qs
    if qs[-1] != 1.0:
        qs = qs + [1.0]

    values = pool.approxQuantile(dens, qs, 0.001)
    # Convert to bins [q0,q1), [q1,q2)...
    bins = list(zip(values[:-1], values[1:]))

    def assign_bin_expr() -> F.Column:
        expr = None
        for i, (a, b) in enumerate(bins):
            cond = (F.col(dens) >= F.lit(a)) & (F.col(dens) < F.lit(b) if i < len(bins)-1 else F.col(dens) <= F.lit(b))
            if expr is None:
                expr = F.when(cond, F.lit(i))
            else:
                expr = expr.when(cond, F.lit(i))
        return expr.otherwise(F.lit(len(bins)-1))

    pool = pool.withColumn("density_bin", assign_bin_expr())
    pool.persist()  # Cache since we use pool many times for topk and stratified selections
    print(f"[3b] Pool persisted with {pool.count()} candidate regions")

    # Build ranking configurations (per-mode k values)
    ranking_configs = build_ranking_configs_from_args(args)
    print(f"[3b] Ranking configs: {[(rc.mode, rc.k_top, rc.k_stratified) for rc in ranking_configs]}")

    # Top-K selection per ranking mode (within each region_split)
    selections = []

    def topk(df: DataFrame, score_col: str, k: int, selection_id: str, mode_label: str) -> DataFrame:
        from pyspark.sql.window import Window
        win = Window.partitionBy("region_split").orderBy(F.col(score_col).desc())
        out = df.withColumn("region_rank", F.row_number().over(win)).filter(F.col("region_rank") <= F.lit(k))
        return (
            out
            .withColumn("selection_set_id", F.lit(selection_id))
            .withColumn("ranking_mode", F.lit(mode_label))
            .withColumn("ranking_score_col", F.lit(score_col))
        )

    # Process each ranking mode with its own k_top and k_stratified
    for rc in ranking_configs:
        mode = rc.mode
        k_top = rc.k_top
        k_strat = rc.k_stratified
        score = RANKING_MODE_TO_SCORE_COL[mode]

        # Top-K selection for this mode
        if k_top > 0:
            selections.append(
                topk(pool, score, k_top, f"topk_{mode}_k{k_top}", mode)
                .withColumn("selection_strategy", F.lit("topk"))
            )
            print(f"[3b] Added topk selection: mode={mode}, k={k_top}")

        # Stratified selection for this mode (if k_strat > 0)
        if k_strat > 0:
            weights_custom = [float(x) for x in args.strata_weights.split(",") if x.strip()]
            if len(weights_custom) != len(bins):
                raise RuntimeError("strata_weights length must match number of bins (derived from strata_quantiles)")

            profiles = [("custom", weights_custom)]
            if args.emit_balanced_stratified == 1:
                profiles.append(("balanced", [1.0 / float(len(bins))] * len(bins)))

            from pyspark.sql.window import Window

            for profile_name, weights in profiles:
                parts = []
                for b, wgt in enumerate(weights):
                    kk = int(math.ceil(wgt * k_strat))
                    if kk <= 0:
                        continue
                    sub = pool.filter(F.col("density_bin") == F.lit(b))
                    win = Window.partitionBy("region_split").orderBy(F.col(score).desc())
                    sub = sub.withColumn("region_rank", F.row_number().over(win)).filter(F.col("region_rank") <= F.lit(kk))
                    parts.append(sub)
                if parts:
                    strat = parts[0]
                    for pp in parts[1:]:
                        strat = strat.unionByName(pp, allowMissingColumns=True)
                    strat = strat.withColumn("selection_set_id", F.lit(f"strat_{profile_name}_{mode}_k{k_strat}"))
                    strat = strat.withColumn("ranking_mode", F.lit(mode))
                    strat = strat.withColumn("ranking_score_col", F.lit(score))
                    strat = strat.withColumn("selection_strategy", F.lit(f"stratified_{profile_name}"))
                    selections.append(strat)
                    print(f"[3b] Added stratified selection: mode={mode}, profile={profile_name}, k={k_strat}")

    if not selections:
        raise RuntimeError("No selections generated")

    sel = selections[0]
    for s in selections[1:]:
        sel = sel.unionByName(s, allowMissingColumns=True)

    # Keep only columns needed downstream plus a few diagnostics
    keep = [
        "selection_set_id",
        "selection_strategy",
        "ranking_mode",
        "ranking_score_col",
        "region_id",
        "region_split",
        "region_rank",
        "area_deg2",
        "n_bricks",
        "n_lrg_v3_total",
        "lrg_density_v3_areaweighted",
        "score_density",
        "score_n_lrg",
        "score_area_weighted",
        "score_psf_weighted",
        "psfsize_r_p50",
        "psfdepth_r_p50",
        "ebv_p50",
        "compactness",
    ]
    keep = [c for c in keep if c in sel.columns]
    sel = sel.select(*keep)

    out3b = args.output_s3.rstrip("/") + f"/phase3b/{args.variant}"
    sel.write.mode("overwrite").parquet(out3b + "/region_selections")
    sel.coalesce(1).write.mode("overwrite").option("header", "true").csv(out3b + "/region_selections_csv")

    # Clean up persisted DataFrame
    pool.unpersist()

    # Write configuration JSON for reproducibility
    write_stage_config_json(spark, out3b, "3b", args, ranking_configs=ranking_configs)
    print(f"[3b] Stage 3b complete. Output: {out3b}")


# ----------------------------
# Stage 3c: parent catalog build
# ----------------------------

def _variant_flags_from_mags(zmag, rmag, w1mag) -> Dict[str, bool]:
    # zmag, rmag, w1mag may be numpy arrays; this is for vectorized computation in numpy.
    flags = {}
    rz = rmag - zmag
    zw1 = zmag - w1mag
    for name, v in LRG_VARIANTS.items():
        flags[name] = (zmag < v["z_max"]) & (rz > v["rz_min"]) & (zw1 > v["zw1_min"]) & np.isfinite(zmag) & np.isfinite(rz) & np.isfinite(zw1)
    return flags




def stage_3c(args: argparse.Namespace) -> None:
    """Stage 3c: Build parent LRG catalogs by scanning DR10 South sweep FITS.

    Two output modes:
      - union (default): builds ONE parent catalog for the UNION of all selected regions.
        This avoids multiplicative explosion when you generate many selection sets in Stage 3b.
        Later phases can join the parent catalog to Stage 3b's region_selections table.

      - per_selection_set: duplicates parent rows per selection_set_id (legacy behavior).
        Use only when the number of selection sets and regions is small.

    Parent selection:
      - Rows are filtered to the requested --variant (default v3_color_relaxed) to keep size
        comparable to the earlier Phase 3 scripts.
      - We compute and output v1..v5 flags for diagnostics.

    Notes on MW extinction:
      - Phase 2 uses *raw* magnitudes computed from FLUX_* nanomaggies (no MW correction).
      - Stage 3c outputs both raw mags and optional MW-corrected mags, but by default uses
        raw mags for variant flags and parent selection.
    """
    spark = make_spark("DHS Phase3c BuildParent", args.shuffle_partitions)

    variant = args.variant
    out3c = f"{args.output_s3.rstrip('/')}/phase3c/{variant}"

    mode = getattr(args, 'parent_output_mode', 'union')
    if mode not in ('union', 'per_selection_set'):
        raise ValueError(f"Invalid --parent-output-mode={mode}")

    out_parquet = f"{out3c}/parent_{mode}_parquet"
    if args.skip_if_exists and s3_prefix_exists(out_parquet) and (not args.force):
        print(f"[3c] Output exists, skipping: {out_parquet}")
        return

    if not args.sweep_index_s3:
        raise ValueError("--sweep-index-s3 is required for stage 3c")
    if not args.s3_sweep_cache_prefix:
        raise ValueError("--s3-sweep-cache-prefix is required for stage 3c")

    # Inputs from earlier stages
    bricks_with_region_path = f"{args.output_s3.rstrip('/')}/phase3a/{variant}/bricks_with_region"
    selections_path = f"{args.output_s3.rstrip('/')}/phase3b/{variant}/region_selections"

    bricks_wr = spark.read.parquet(bricks_with_region_path)
    sels = spark.read.parquet(selections_path)

    # Optional filters
    only_sets = [s.strip() for s in (args.only_selection_sets or '').split(',') if s.strip()]
    if only_sets:
        sels = sels.filter(F.col('selection_set_id').isin(only_sets))

    exclude_splits = [s.strip() for s in (args.exclude_splits or '').split(',') if s.strip()]
    if exclude_splits:
        sels = sels.filter(~F.col('region_split').isin(exclude_splits))

    if args.max_selected_regions and args.max_selected_regions > 0:
        # Deterministic truncation: sort by (region_split, region_id)
        sel_regions = sels.select('region_id', 'region_split').distinct().orderBy('region_split', 'region_id')
        sel_regions = sel_regions.limit(int(args.max_selected_regions))
    else:
        sel_regions = sels.select('region_id', 'region_split').distinct()

    # Brick list for union-of-selected regions
    bricks_needed = bricks_wr.join(sel_regions, on='region_id', how='inner').select('brickname', 'region_id', 'region_split')
    bricks_needed.persist()  # Cache since we use it for count() and collect()

    # Guardrails for accidental blowups
    if args.max_selected_bricks and args.max_selected_bricks > 0:
        n_bricks = bricks_needed.count()
        print(f"[3c] Selected {n_bricks} unique bricks for processing")
        if n_bricks > int(args.max_selected_bricks):
            raise RuntimeError(
                f"Stage 3c would process {n_bricks} bricks (> max-selected-bricks={args.max_selected_bricks}). "
                f"Reduce k-top/k-stratified, filter --only-selection-sets, or increase --max-selected-bricks if intended."
            )

    # Build mapping(s) on driver and broadcast.
    sc = spark.sparkContext

    if mode == 'union':
        # brickname -> (region_id, region_split)
        brick_map_rows = bricks_needed.collect()
        bricks_needed.unpersist()  # No longer needed after collect
        brick_to_region = {str(r['brickname']): (int(r['region_id']), str(r['region_split'])) for r in brick_map_rows}
        print(f"[3c] Broadcasting {len(brick_to_region)} unique bricks for union mode")
        bcast = sc.broadcast(brick_to_region)

        schema = T.StructType([
            T.StructField('region_id', T.IntegerType(), False),
            T.StructField('region_split', T.StringType(), False),
            T.StructField('brickname', T.StringType(), False),
            T.StructField('objid', T.LongType(), False),
            T.StructField('ra', T.DoubleType(), False),
            T.StructField('dec', T.DoubleType(), False),
            T.StructField('gmag', T.FloatType(), False),
            T.StructField('rmag', T.FloatType(), False),
            T.StructField('zmag', T.FloatType(), False),
            T.StructField('w1mag', T.FloatType(), False),
            T.StructField('rz', T.FloatType(), False),
            T.StructField('zw1', T.FloatType(), False),
            T.StructField('gmag_mw', T.FloatType(), True),
            T.StructField('rmag_mw', T.FloatType(), True),
            T.StructField('zmag_mw', T.FloatType(), True),
            T.StructField('w1mag_mw', T.FloatType(), True),
            T.StructField('rz_mw', T.FloatType(), True),
            T.StructField('zw1_mw', T.FloatType(), True),
            T.StructField('maskbits', T.LongType(), True),
            T.StructField('type', T.StringType(), True),
            T.StructField('is_v1_pure_massive', T.BooleanType(), False),
            T.StructField('is_v2_baseline_dr10', T.BooleanType(), False),
            T.StructField('is_v3_color_relaxed', T.BooleanType(), False),
            T.StructField('is_v4_mag_relaxed', T.BooleanType(), False),
            T.StructField('is_v5_very_relaxed', T.BooleanType(), False),
        ])

    else:
        # per_selection_set: brickname -> list[{selection_set_id, region_id, region_split, region_rank}]
        sel_for_join = sels.select('selection_set_id', 'region_id', 'region_split', 'region_rank')
        brick_sel = bricks_needed.join(sel_for_join, on=['region_id', 'region_split'], how='inner')
        bricks_needed.unpersist()  # No longer needed after join
        grouped = brick_sel.groupBy('brickname').agg(
            F.collect_list(F.struct('selection_set_id', 'region_id', 'region_split', 'region_rank')).alias('sels')
        )
        rows = grouped.collect()
        brick_to_sels = {str(r['brickname']): [dict(x.asDict()) for x in r['sels']] for r in rows}
        print(f"[3c] Broadcasting {len(brick_to_sels)} unique bricks for per_selection_set mode")
        bcast = sc.broadcast(brick_to_sels)

        schema = T.StructType([
            T.StructField('selection_set_id', T.StringType(), False),
            T.StructField('region_id', T.IntegerType(), False),
            T.StructField('region_split', T.StringType(), False),
            T.StructField('region_rank', T.IntegerType(), False),
            T.StructField('brickname', T.StringType(), False),
            T.StructField('objid', T.LongType(), False),
            T.StructField('ra', T.DoubleType(), False),
            T.StructField('dec', T.DoubleType(), False),
            T.StructField('gmag', T.FloatType(), False),
            T.StructField('rmag', T.FloatType(), False),
            T.StructField('zmag', T.FloatType(), False),
            T.StructField('w1mag', T.FloatType(), False),
            T.StructField('rz', T.FloatType(), False),
            T.StructField('zw1', T.FloatType(), False),
            T.StructField('gmag_mw', T.FloatType(), True),
            T.StructField('rmag_mw', T.FloatType(), True),
            T.StructField('zmag_mw', T.FloatType(), True),
            T.StructField('w1mag_mw', T.FloatType(), True),
            T.StructField('rz_mw', T.FloatType(), True),
            T.StructField('zw1_mw', T.FloatType(), True),
            T.StructField('maskbits', T.LongType(), True),
            T.StructField('type', T.StringType(), True),
            T.StructField('is_v1_pure_massive', T.BooleanType(), False),
            T.StructField('is_v2_baseline_dr10', T.BooleanType(), False),
            T.StructField('is_v3_color_relaxed', T.BooleanType(), False),
            T.StructField('is_v4_mag_relaxed', T.BooleanType(), False),
            T.StructField('is_v5_very_relaxed', T.BooleanType(), False),
        ])

    # Sweep index
    lines = spark.sparkContext.textFile(args.sweep_index_s3).filter(lambda s: s and not s.strip().startswith('#'))
    lines = lines.repartition(int(args.sweep_partitions))

    emit_mw = bool(int(getattr(args, 'emit_mw_corrected_mags', 1)))
    chunk_size = int(getattr(args, 'chunk_size', 100000))
    s3_sweep_cache_prefix = args.s3_sweep_cache_prefix  # Extract to avoid capturing entire args in closure

    def _decode_brickname(arr: np.ndarray) -> np.ndarray:
        if arr.dtype.kind in ('S', 'a'):
            return np.char.decode(arr, 'ascii', errors='ignore')
        return arr.astype(str)

    def iter_fits_rows_in_chunks(data, chunk_size: int = 100000):
        """Yield slices of FITS table data in chunks to limit memory usage."""
        n = len(data)
        for start in range(0, n, chunk_size):
            yield data[start:start + chunk_size]

    def process_partition(iter_lines: Iterable[str]) -> Iterable[Tuple]:
        import botocore
        from botocore.config import Config
        from astropy.io import fits
        import os

        client = boto3.client('s3', config=Config(retries={'max_attempts': 8, 'mode': 'standard'}))

        # Cache the brick set ONCE per partition to avoid repeated list() calls
        brick_map = bcast.value
        brick_set = frozenset(brick_map.keys())  # frozenset for faster lookup
        brick_list = list(brick_set)  # For np.isin which needs a list/array

        with tempfile.TemporaryDirectory() as td:
            for entry in iter_lines:
                url = entry.strip()
                if not url:
                    continue

                local = try_download_sweep_to_local(url, s3_sweep_cache_prefix, td, client=client)
                if local is None:
                    # Cache miss: skip
                    continue

                try:
                    with fits.open(local, memmap=True) as hdul:
                        d = hdul[1].data
                        for sub in iter_fits_rows_in_chunks(d, chunk_size=chunk_size):
                            bn = _decode_brickname(sub['BRICKNAME'])

                            # Keep only bricks that are selected (use cached brick_list)
                            mask_brick = np.isin(bn, brick_list)

                            if not np.any(mask_brick):
                                continue

                            # Slice arrays
                            bn = bn[mask_brick]
                            ra = sub['RA'][mask_brick].astype(float)
                            dec = sub['DEC'][mask_brick].astype(float)
                            objid = sub['OBJID'][mask_brick].astype('int64') if 'OBJID' in sub.dtype.names else np.arange(len(bn), dtype='int64')
                            typ = _decode_brickname(sub['TYPE'][mask_brick]) if 'TYPE' in sub.dtype.names else None

                            # Reject stars
                            if typ is not None:
                                typ_u = np.char.upper(np.char.strip(typ.astype(str)))
                                mask_gal = (typ_u != 'PSF')
                                if not np.any(mask_gal):
                                    continue
                                bn = bn[mask_gal]
                                ra = ra[mask_gal]
                                dec = dec[mask_gal]
                                objid = objid[mask_gal]
                                typ_u = typ_u[mask_gal]
                            else:
                                typ_u = None

                            fg = sub['FLUX_G'][mask_brick].astype(float)
                            fr = sub['FLUX_R'][mask_brick].astype(float)
                            fz = sub['FLUX_Z'][mask_brick].astype(float)
                            fw1 = sub['FLUX_W1'][mask_brick].astype(float)

                            if typ is not None:
                                fg = fg[mask_gal]; fr = fr[mask_gal]; fz = fz[mask_gal]; fw1 = fw1[mask_gal]

                            # Raw mags (Phase 2 consistent)
                            gmag = nanomaggies_to_mag(fg).astype(float)
                            rmag = nanomaggies_to_mag(fr).astype(float)
                            zmag = nanomaggies_to_mag(fz).astype(float)
                            w1mag = nanomaggies_to_mag(fw1).astype(float)

                            rz = (rmag - zmag).astype(float)
                            zw1 = (zmag - w1mag).astype(float)

                            flags = _variant_flags_from_mags(zmag, rmag, w1mag)

                            parent_mask = flags.get(variant)
                            if parent_mask is None:
                                raise ValueError(f"Unknown variant={variant} (expected one of {list(flags.keys())})")

                            if not np.any(parent_mask):
                                continue

                            # Slice to parent rows
                            bn = bn[parent_mask]
                            ra = ra[parent_mask]
                            dec = dec[parent_mask]
                            objid = objid[parent_mask]
                            gmag = gmag[parent_mask]
                            rmag = rmag[parent_mask]
                            zmag = zmag[parent_mask]
                            w1mag = w1mag[parent_mask]
                            rz = rz[parent_mask]
                            zw1 = zw1[parent_mask]
                            if typ_u is not None:
                                typ_u = typ_u[parent_mask]

                            # MW-corrected mags (optional output only)
                            if emit_mw and ('MW_TRANSMISSION_G' in sub.dtype.names):
                                mtg = sub['MW_TRANSMISSION_G'][mask_brick].astype(float)
                                mtr = sub['MW_TRANSMISSION_R'][mask_brick].astype(float)
                                mtz = sub['MW_TRANSMISSION_Z'][mask_brick].astype(float)
                                mtw1 = sub['MW_TRANSMISSION_W1'][mask_brick].astype(float)
                                if typ is not None:
                                    mtg = mtg[mask_gal]; mtr = mtr[mask_gal]; mtz = mtz[mask_gal]; mtw1 = mtw1[mask_gal]
                                mtg = mtg[parent_mask]; mtr = mtr[parent_mask]; mtz = mtz[parent_mask]; mtw1 = mtw1[parent_mask]

                                gmw = nanomaggies_to_mag(safe_div(fg[parent_mask], mtg)).astype(float)
                                rmw = nanomaggies_to_mag(safe_div(fr[parent_mask], mtr)).astype(float)
                                zmw = nanomaggies_to_mag(safe_div(fz[parent_mask], mtz)).astype(float)
                                wmw = nanomaggies_to_mag(safe_div(fw1[parent_mask], mtw1)).astype(float)
                                rz_mw = (rmw - zmw).astype(float)
                                zw1_mw = (zmw - wmw).astype(float)
                            else:
                                gmw = rmw = zmw = wmw = rz_mw = zw1_mw = None

                            mb = sub['MASKBITS'][mask_brick] if 'MASKBITS' in sub.dtype.names else None
                            if mb is not None and typ is not None:
                                mb = mb[mask_gal]
                            if mb is not None:
                                mb = mb[parent_mask]

                            # Emit
                            for i in range(len(bn)):
                                brick = str(bn[i])

                                if mode == 'union':
                                    region_id, region_split = brick_map.get(brick, (None, None))
                                    if region_id is None:
                                        continue
                                    yield (
                                        int(region_id),
                                        str(region_split),
                                        brick,
                                        int(objid[i]),
                                        float(ra[i]),
                                        float(dec[i]),
                                        float(gmag[i]),
                                        float(rmag[i]),
                                        float(zmag[i]),
                                        float(w1mag[i]),
                                        float(rz[i]),
                                        float(zw1[i]),
                                        (float(gmw[i]) if gmw is not None else None),
                                        (float(rmw[i]) if rmw is not None else None),
                                        (float(zmw[i]) if zmw is not None else None),
                                        (float(wmw[i]) if wmw is not None else None),
                                        (float(rz_mw[i]) if rz_mw is not None else None),
                                        (float(zw1_mw[i]) if zw1_mw is not None else None),
                                        (int(mb[i]) if mb is not None else None),
                                        (str(typ_u[i]) if typ_u is not None else None),
                                        bool(flags['v1_pure_massive'][parent_mask][i]),
                                        bool(flags['v2_baseline_dr10'][parent_mask][i]),
                                        bool(flags["v3_color_relaxed"][parent_mask][i]),
                                        bool(flags['v4_mag_relaxed'][parent_mask][i]),
                                        bool(flags['v5_very_relaxed'][parent_mask][i]),
                                    )

                                else:
                                    for srec in brick_map.get(brick, []):
                                        yield (
                                            str(srec['selection_set_id']),
                                            int(srec['region_id']),
                                            str(srec['region_split']),
                                            int(srec['region_rank']),
                                            brick,
                                            int(objid[i]),
                                            float(ra[i]),
                                            float(dec[i]),
                                            float(gmag[i]),
                                            float(rmag[i]),
                                            float(zmag[i]),
                                            float(w1mag[i]),
                                            float(rz[i]),
                                            float(zw1[i]),
                                            (float(gmw[i]) if gmw is not None else None),
                                            (float(rmw[i]) if rmw is not None else None),
                                            (float(zmw[i]) if zmw is not None else None),
                                            (float(wmw[i]) if wmw is not None else None),
                                            (float(rz_mw[i]) if rz_mw is not None else None),
                                            (float(zw1_mw[i]) if zw1_mw is not None else None),
                                            (int(mb[i]) if mb is not None else None),
                                            (str(typ_u[i]) if typ_u is not None else None),
                                            bool(flags['v1_pure_massive'][parent_mask][i]),
                                            bool(flags['v2_baseline_dr10'][parent_mask][i]),
                                            bool(flags["v3_color_relaxed"][parent_mask][i]),
                                            bool(flags['v4_mag_relaxed'][parent_mask][i]),
                                            bool(flags['v5_very_relaxed'][parent_mask][i]),
                                        )

                except Exception as e:
                    print(f"[3c][WARN] Failed sweep {url}: {e}")
                finally:
                    # Explicitly delete local file to free disk space on executor
                    if local and os.path.exists(local):
                        try:
                            os.remove(local)
                        except OSError:
                            pass
                    # Force garbage collection to reclaim memory after each sweep
                    import gc
                    gc.collect()

    rdd = lines.mapPartitions(process_partition)
    df = spark.createDataFrame(rdd, schema=schema)

    # Repartition before write
    # Use configurable partition count; default scales with sweep_partitions
    output_partitions = getattr(args, 'output_partitions', 0)
    if output_partitions <= 0:
        # Auto-scale: use sweep_partitions as a reasonable default
        output_partitions = max(200, int(args.sweep_partitions))
    print(f"[3c] Repartitioning output to {output_partitions} partitions")
    df = df.repartition(output_partitions)

    if mode == 'union':
        df.write.mode('overwrite').partitionBy('region_split', 'region_id').parquet(out_parquet)
    else:
        df.write.mode('overwrite').partitionBy('selection_set_id', 'region_split').parquet(out_parquet)

    print(f"[3c] Wrote: {out_parquet}")

    # Write configuration JSON for reproducibility
    write_stage_config_json(spark, out3c, "3c", args)
    print(f"[3c] Stage 3c complete. Output: {out3c}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["3a", "3b", "3c"], help="Which Phase 3 stage to run")

    p.add_argument("--variant", default="v3_color_relaxed", help="Baseline variant label (for output paths)")
    p.add_argument("--output-s3", required=True, help="S3 prefix for Phase 3 outputs")

    # Common Spark config
    p.add_argument("--shuffle-partitions", type=int, default=400)

    # Phase 3a
    p.add_argument("--phase2-results-s3", help="S3 path to Phase 2 per-brick results CSV (phase2_results.csv)")
    p.add_argument("--bricks-fits-s3", default=None, help="Optional S3 path to survey-bricks-dr10-south.fits.gz")
    p.add_argument(
        "--bricks-fits-columns",
        default="BRICKNAME,RA,DEC,RA1,RA2,DEC1,DEC2,EBV,PSFSIZE_G,PSFSIZE_R,PSFSIZE_Z,PSFDEPTH_G,PSFDEPTH_R,PSFDEPTH_Z,NEXP_G,NEXP_R,NEXP_Z",
        help="Comma-separated columns to keep from bricks FITS (missing columns are ignored)",
    )

    p.add_argument("--max-ebv", type=float, default=0.12)
    p.add_argument("--max-psf-r-arcsec", type=float, default=1.6)
    p.add_argument("--min-psfdepth-r", type=float, default=23.6)

    # Adjacency
    p.add_argument("--adj-ra-eps", type=float, default=0.30)
    p.add_argument("--adj-dec-eps", type=float, default=0.30)
    p.add_argument("--adj-cell-size", type=float, default=0.25)

    # Split assignment
    p.add_argument("--split-seed", type=int, default=13)
    p.add_argument("--test-frac", type=float, default=0.20)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--area-weighted-alpha", type=float, default=0.5, help="Exponent alpha for area_weighted score: density * area^alpha")

    # psf_weighted score parameters
    p.add_argument("--psf-ref", type=float, default=1.25)
    p.add_argument("--psf-sigma", type=float, default=0.25)
    p.add_argument("--depth-ref", type=float, default=23.8)
    p.add_argument("--depth-sigma", type=float, default=0.25)
    p.add_argument("--ebv-k", type=float, default=4.0)

    # Phase 3b selection knobs
    #
    # Two ways to configure ranking modes:
    #
    # Option 1 (simple): Use --ranking-modes with global --k-top and --k-stratified
    #   --ranking-modes "density,n_lrg,area_weighted,psf_weighted" --k-top 100 --k-stratified 0
    #
    # Option 2 (granular): Use --ranking-config to set per-mode k values
    #   --ranking-config "n_lrg:100:100,psf_weighted:100:100,density:30:0,area_weighted:30:0"
    #   Format: "mode:k_top:k_stratified,mode:k_top:k_stratified,..."
    #
    # If --ranking-config is provided, it takes precedence over --ranking-modes/--k-top/--k-stratified.
    #
    p.add_argument(
        "--ranking-config",
        default=None,
        help=(
            "Per-mode selection config. Format: 'mode:k_top:k_strat,mode:k_top:k_strat,...' "
            "Example: 'n_lrg:100:100,psf_weighted:100:100,density:30:0,area_weighted:30:0' "
            "If provided, overrides --ranking-modes, --k-top, --k-stratified. "
            "Available modes: density, n_lrg, area_weighted, psf_weighted."
        ),
    )
    p.add_argument("--ranking-modes", default="density,n_lrg,area_weighted,psf_weighted", help="Comma-separated list (ignored if --ranking-config is set)")
    p.add_argument("--k-top", type=int, default=100, help="Default k for top-K selection (ignored if --ranking-config is set)")
    p.add_argument("--k-stratified", type=int, default=0, help="Default k for stratified selection; 0 disables (ignored if --ranking-config is set)")
    p.add_argument("--emit-balanced-stratified", type=int, default=1, help="If 1 and k_stratified>0, also emit an equal-per-bin stratified selection")
    p.add_argument("--strata-quantiles", default="0.0,0.5,0.9,1.0", help="Quantile edges for density stratification")
    p.add_argument("--strata-weights", default="0.2,0.4,0.4", help="Fractions per density bin (must match bins)")
    p.add_argument("--exclude-splits", default=None, help="Comma-separated region_splits to exclude, e.g. 'test'")

    # Phase 3c sweep scan
    p.add_argument("--region-selection-s3", default=None, help="Override: S3 path to region selections parquet")
    p.add_argument("--only-selection-sets", default=None, help="Comma-separated selection_set_id to include (default: all)")
    p.add_argument("--parent-output-mode", default="union", choices=["union", "per_selection_set"],
                   help="Stage 3c output mode. 'union' writes one row per parent LRG in the union of selected regions (recommended). 'per_selection_set' duplicates rows for each selection_set_id.")
    p.add_argument("--max-selected-bricks", type=int, default=80000,
                   help="Safety cap: if the union of selected regions contains more than this many bricks, Stage 3c will stop. Increase intentionally if needed.")
    p.add_argument("--max-selected-regions", type=int, default=0,
                   help="Optional cap on number of selected regions (0 means no cap). Useful for trial runs.")

    p.add_argument("--sweep-index-s3", default=None, help="S3 path to sweep index text file (one URL per line)")
    p.add_argument("--s3-sweep-cache-prefix", default=None, help="S3 prefix where sweeps are cached as sweep-*.fits or sweep-*.fits.gz")
    p.add_argument("--sweep-partitions", type=int, default=600)
    p.add_argument("--chunk-size", type=int, default=100000, help="Stage 3c: sweep rows per chunk to limit executor memory")
    p.add_argument("--output-partitions", type=int, default=0, help="Stage 3c: number of output partitions before write. 0 = auto (uses sweep-partitions)")
    p.add_argument("--use-mw-correction", type=int, default=0, help="Stage 3c: if 1, apply MW_TRANSMISSION to fluxes before computing mags/colors for LRG selection. Default 0 to match Phase 2.")
    p.add_argument("--emit-mw-corrected-mags", type=int, default=1, help="Stage 3c: if 1, also emit MW-corrected mags/colors as extra columns (gmag_mw, etc)")

    # Idempotency
    p.add_argument("--skip-if-exists", type=int, default=1, help="If 1, skip the stage when its expected output prefix already has objects.")
    p.add_argument("--force", type=int, default=0, help="If 1, ignore --skip-if-exists and overwrite outputs.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.stage == "3a":
        if not args.phase2_results_s3:
            raise SystemExit("--phase2-results-s3 is required for stage 3a")
        if not args.bricks_fits_s3:
            raise SystemExit("--bricks-fits-s3 is required for stage 3a (Phase 2 output lacks ebv/psf/depth/area)")
        stage_3a_region_metrics(args)
    elif args.stage == "3b":
        stage_3b_select_regions(args)
    elif args.stage == "3c":
        if not args.sweep_index_s3:
            raise SystemExit("--sweep-index-s3 is required for stage 3c")
        if not args.s3_sweep_cache_prefix:
            raise SystemExit("--s3-sweep-cache-prefix is required for stage 3c (this pipeline reads sweeps from the S3 cache, not directly from NERSC HTTP)")
        stage_3c(args)


if __name__ == "__main__":
    main()

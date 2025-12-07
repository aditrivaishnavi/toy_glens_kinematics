from __future__ import annotations

from dataclasses import dataclass
from math import radians, sin
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

try:
    import pyvo  # TAP client; user must install this in their environment
except ImportError:
    pyvo = None  # we will guard against this in code

from .config import Phase1p5Config


@dataclass
class BrickRecord:
    brickid: int
    brickname: str
    ra: float
    dec: float
    ra1: float
    ra2: float
    dec1: float
    dec2: float
    psfsize_r: float
    psfdepth_r: float
    ebv: float
    nexp_r: int
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


def fetch_bricks(config: Phase1p5Config) -> pd.DataFrame:
    """
    Query ls_dr10.bricks_s from NOIRLab TAP and return a DataFrame with
    brick geometry and QA metrics in the scouting footprint.

    This is a pure survey QA step: no lens catalogs, no Tractor objects.
    """
    if pyvo is None:
        raise ImportError(
            "pyvo is required for TAP access. Install with: pip install pyvo"
        )

    tap = pyvo.dal.TAPService(config.tap_url)

    query = f"""
        SELECT
            brickid,
            brickname,
            ra,
            dec,
            brickra1,
            brickra2,
            brickdec1,
            brickdec2,
            psfsize_r,
            psfdepth_r,
            ebv,
            nexp_r
        FROM {config.bricks_table}
        WHERE ra BETWEEN {config.ra_min} AND {config.ra_max}
          AND dec BETWEEN {config.dec_min} AND {config.dec_max}
    """

    job = tap.submit_job(query)
    job.run()
    table = job.fetch_result()
    df = table.to_pandas()

    # If nexp_r is missing from the schema, assume 1 exposure (conservative)
    if "nexp_r" not in df.columns:
        df["nexp_r"] = 1

    # Compute brick area
    df["area_deg2"] = df.apply(
        lambda row: _compute_brick_area_deg2(
            float(row["brickra1"]),
            float(row["brickra2"]),
            float(row["brickdec1"]),
            float(row["brickdec2"]),
        ),
        axis=1,
    )

    return df


def apply_brick_quality_cuts(df: pd.DataFrame, config: Phase1p5Config) -> pd.DataFrame:
    """
    Apply hard, physically motivated quality cuts:
      - seeing (psfsize_r)
      - depth (psfdepth_r)
      - extinction (ebv)
      - number of r-band exposures (nexp_r)
    """
    mask = (
        (df["psfsize_r"] <= config.max_psfsize_r)
        & (df["psfdepth_r"] >= config.min_psfdepth_r)
        & (df["ebv"] <= config.max_ebv)
        & (df["nexp_r"] >= config.min_nexp_r)
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
    flux_cols: Dict[str, str],
    mw_trans_cols: Dict[str, str],
) -> np.ndarray:
    """
    DESI-like LRG proxy mask using extinction-corrected magnitudes.

    Inputs:
      - df: Tractor subset for one brick.
      - flux_cols: mapping band -> flux column name, e.g. {"g": "flux_g", ...}
      - mw_trans_cols: mapping band -> MW_TRANSMISSION_* column for extinction.

    Steps:
      1. Correct flux for Milky Way transmission: flux_corr = flux / transmission.
      2. Convert to AB mags (nanomaggies).
      3. Compute colors: r-z, z-W1.
      4. Apply cuts:
         - z < lrg_z_mag_max
         - r - z > lrg_min_r_minus_z
         - z - W1 > lrg_min_z_minus_w1
         - Optional: z_phot in [z_phot_min, z_phot_max]
    """
    # Corrected flux
    flux_g = df[flux_cols["g"]].values / df[mw_trans_cols["g"]].values
    flux_r = df[flux_cols["r"]].values / df[mw_trans_cols["r"]].values
    flux_z = df[flux_cols["z"]].values / df[mw_trans_cols["z"]].values
    flux_w1 = df[flux_cols["w1"]].values / df[mw_trans_cols["w1"]].values

    mag_g = _nanomaggies_to_mag(flux_g)
    mag_r = _nanomaggies_to_mag(flux_r)
    mag_z = _nanomaggies_to_mag(flux_z)
    mag_w1 = _nanomaggies_to_mag(flux_w1)

    r_minus_z = mag_r - mag_z
    z_minus_w1 = mag_z - mag_w1

    mask = (
        (mag_z < config.lrg_z_mag_max)
        & (r_minus_z > config.lrg_min_r_minus_z)
        & (z_minus_w1 > config.lrg_min_z_minus_w1)
    )

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
) -> pd.DataFrame:
    """
    For up to max_bricks_for_lrg_density bricks, query the Tractor catalog and
    estimate LRG density (number per deg^2) using a DESI-like proxy.

    This is the expensive step; we limit it to a manageable number of bricks.
    """
    if pyvo is None:
        raise ImportError(
            "pyvo is required for TAP access. Install with: pip install pyvo"
        )

    tap = pyvo.dal.TAPService(config.tap_url)

    # For reproducibility, sort bricks by brickid and take the first N
    bricks_for_density = bricks_df.sort_values("brickid").head(
        config.max_bricks_for_lrg_density
    )

    lrg_counts = []
    lrg_densities = []

    for _, row in bricks_for_density.iterrows():
        ra1 = float(row["brickra1"])
        ra2 = float(row["brickra2"])
        dec1 = float(row["brickdec1"])
        dec2 = float(row["brickdec2"])
        area_deg2 = float(row["area_deg2"])

        query = f"""
            SELECT
                ra, dec,
                flux_g, flux_r, flux_z, flux_w1,
                mw_transmission_g, mw_transmission_r,
                mw_transmission_z, mw_transmission_w1,
                photo_z
            FROM {config.tractor_table}
            WHERE ra BETWEEN {ra1} AND {ra2}
              AND dec BETWEEN {dec1} AND {dec2}
        """

        job = tap.submit_job(query)
        job.run()
        table = job.fetch_result()
        tdf = table.to_pandas()

        if tdf.empty:
            lrg_counts.append(0)
            lrg_densities.append(0.0)
            continue

        flux_cols = {
            "g": "flux_g",
            "r": "flux_r",
            "z": "flux_z",
            "w1": "flux_w1",
        }
        mw_cols = {
            "g": "mw_transmission_g",
            "r": "mw_transmission_r",
            "z": "mw_transmission_z",
            "w1": "mw_transmission_w1",
        }

        lrg_mask = _build_lrg_mask(tdf, config, flux_cols, mw_cols)
        count = int(np.count_nonzero(lrg_mask))
        density = count / area_deg2 if area_deg2 > 0 else 0.0

        lrg_counts.append(count)
        lrg_densities.append(density)

    bricks_for_density = bricks_for_density.copy()
    bricks_for_density["lrg_count"] = lrg_counts
    bricks_for_density["lrg_density"] = lrg_densities

    # Merge densities back onto the full brick table (bricks without density get 0)
    merged = bricks_df.merge(
        bricks_for_density[
            ["brickid", "lrg_count", "lrg_density"]
        ],
        on="brickid",
        how="left",
    )
    merged["lrg_count"] = merged["lrg_count"].fillna(0).astype(int)
    merged["lrg_density"] = merged["lrg_density"].fillna(0.0)

    return merged


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
                brickid=int(row["brickid"]),
                brickname=str(row["brickname"]),
                ra=float(row["ra"]),
                dec=float(row["dec"]),
                ra1=float(row["brickra1"]),
                ra2=float(row["brickra2"]),
                dec1=float(row["brickdec1"]),
                dec2=float(row["brickdec2"]),
                psfsize_r=float(row["psfsize_r"]),
                psfdepth_r=float(row["psfdepth_r"]),
                ebv=float(row["ebv"]),
                nexp_r=int(row["nexp_r"]),
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
            "brickid": [b.brickid for b in recs],
            "brickname": [b.brickname for b in recs],
            "ra": [b.ra for b in recs],
            "dec": [b.dec for b in recs],
            "brickra1": [b.ra1 for b in recs],
            "brickra2": [b.ra2 for b in recs],
            "brickdec1": [b.dec1 for b in recs],
            "brickdec2": [b.dec2 for b in recs],
            "psfsize_r": [b.psfsize_r for b in recs],
            "psfdepth_r": [b.psfdepth_r for b in recs],
            "ebv": [b.ebv for b in recs],
            "nexp_r": [b.nexp_r for b in recs],
            "area_deg2": [b.area_deg2 for b in recs],
            "lrg_count": [b.lrg_count for b in recs],
            "lrg_density": [b.lrg_density for b in recs],
        }
        return pd.DataFrame(data)

    primary_df = _component_to_df(primary)
    backup_df = _component_to_df(backup) if backup is not None else pd.DataFrame()

    return primary_df, backup_df


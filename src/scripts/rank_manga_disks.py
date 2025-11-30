#!/usr/bin/env python3
"""
rank_manga_disks.py

Select and rank MaNGA galaxies that are good candidates for
kinematic+lensing experiments, using DR17 DAPall and DRPall catalogs.

Requirements:
    pip install astropy numpy

Usage:
    python rank_manga_disks.py \
        --dapall data/dapall-v3_1_1-3.1.0.fits \
        --drpall data/drpall-v3_1_1.fits \
        --daptype HYB10-MILESHC-MASTARHC2 \
        --n_top 20 \
        --output data/manga_disk_candidates.csv

Notes:
    - This script does NOT download MAPS FITS files; it only builds
      ranked candidate list + direct URLs.
    - Ranking is *heuristic*, designed for your project needs:
        * DAPDONE == 1, DAPQUAL == 0
        * reasonably inclined disks (b/a not too face-on or edge-on)
        * non-zero Hα gas dispersion
        * higher Hα flux within 1 Re preferred
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dapall",
        type=str,
        required=True,
        help="Path to dapall-v3_1_1-3.1.0.fits",
    )
    p.add_argument(
        "--drpall",
        type=str,
        required=True,
        help="Path to drpall-v3_1_1.fits",
    )
    p.add_argument(
        "--daptype",
        type=str,
        default="HYB10-MILESHC-MASTARHC2",
        help="DAPTYPE extension name to use in DAPall "
             "(e.g. HYB10-MILESHC-MASTARHC2)",
    )
    p.add_argument(
        "--n_top",
        type=int,
        default=20,
        help="Number of top-ranked galaxies to output",
    )
    p.add_argument(
        "--output",
        type=str,
        default="manga_disk_candidates.csv",
        help="Output CSV path",
    )
    return p.parse_args()


def load_dapall(dapall_path: str, daptype: str) -> Table:
    """Load DAPall extension for a given DAPTYPE as an Astropy Table."""
    with fits.open(dapall_path) as hdul:
        if daptype not in hdul:
            raise ValueError(
                f"DAPTYPE '{daptype}' not found in {dapall_path}. "
                f"Available extensions: {[h.name for h in hdul]}"
            )
        data = hdul[daptype].data
    tab = Table(data)
    return tab


def load_drpall(drpall_path: str) -> Table:
    """Load DRPall 'MANGA' extension as an Astropy Table."""
    with fits.open(drpall_path) as hdul:
        # In DR17 the MaNGA cubes are in extension 'MANGA'
        if "MANGA" not in hdul:
            raise ValueError(
                f"Extension 'MANGA' not found in {drpall_path}. "
                f"Available extensions: {[h.name for h in hdul]}"
            )
        data = hdul["MANGA"].data
    tab = Table(data)
    return tab


def safe_log10(x):
    """Log10 with masking for non-positive values."""
    x = np.asarray(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log10(x)
    out[~np.isfinite(out)] = np.nan
    return out


def compute_inclination_deg(axis_ratio_ba):
    """
    Very rough inclination estimate from b/a, assuming thin disk:
        cos(i) ~ b/a  =>  i = arccos(b/a)
    Returns inclination in degrees, clipped to [0, 90].
    """
    ba = np.clip(axis_ratio_ba, 0.01, 1.0)
    incl_rad = np.arccos(ba)
    return np.degrees(incl_rad)


def build_candidate_table(dap: Table, drp: Table) -> Table:
    """
    Join DAPall and DRPall, apply quality cuts, and compute ranking metrics.

    Returns an Astropy Table with one row per candidate galaxy and columns:
        - plate
        - ifu
        - plateifu
        - mangaid
        - z
        - nsa_elpetro_ba (axis ratio)
        - incl_deg
        - ha_gflux_1re (if available)
        - ha_gsigma_1re
        - stellar_sigma_1re
        - score (ranking metric)
        - maps_url (direct URL to MAPS FITS)
    """
    # Basic DAP quality cuts: successful run & good quality
    if "DAPDONE" not in dap.colnames or "DAPQUAL" not in dap.colnames:
        raise ValueError("DAPall table missing DAPDONE/DAPQUAL columns.")

    good = (dap["DAPDONE"] == 1) & (dap["DAPQUAL"] == 0)
    dap_good = dap[good]

    # Extract mapping to DRPall
    if "DRPALLINDX" not in dap_good.colnames:
        raise ValueError("DAPall table missing DRPALLINDX column.")

    drp_idx = dap_good["DRPALLINDX"]
    drp_sub = drp[drp_idx]

    # Basic ID columns
    plate = dap_good["PLATE"]
    ifu = dap_good["IFUDESIGN"]
    plateifu = dap_good["PLATEIFU"]
    mangaid = dap_good["MANGAID"]

    # Axis ratio, effective radius, redshift from DRPall (NSA-based)
    # Common DRPall columns in DR17:
    #   nsa_elpetro_ba   : axis ratio (b/a)
    #   nsa_elpetro_th50_r : half-light radius in r-band [arcsec]
    #   nsa_z            : redshift
    for col in ["nsa_elpetro_ba", "nsa_elpetro_th50_r", "nsa_z"]:
        if col not in drp_sub.colnames:
            raise ValueError(f"DRPall table missing expected column '{col}'.")

    ba = drp_sub["nsa_elpetro_ba"]
    reff_r = drp_sub["nsa_elpetro_th50_r"]
    z = drp_sub["nsa_z"]

    incl_deg = compute_inclination_deg(ba)

    # Hα dispersion & stellar dispersion within 1 Re (DAPall columns)
    # Column names vary between DAP versions - try multiple patterns
    
    # Helper to find column (case-insensitive, with variants)
    def find_column(table, candidates):
        """Find first matching column from candidates list (case-insensitive)."""
        colnames_upper = {c.upper(): c for c in table.colnames}
        for cand in candidates:
            if cand.upper() in colnames_upper:
                return colnames_upper[cand.upper()]
        return None
    
    # Try to find Hα sigma column
    ha_sigma_candidates = [
        "ha_gsigma_1re", "HA_GSIGMA_1RE",
        "emline_gsigma_1re_ha", "EMLINE_GSIGMA_1RE_HA",
        "ha_sigma_1re", "HA_SIGMA_1RE",
    ]
    ha_sigma_col = find_column(dap_good, ha_sigma_candidates)
    
    # Try to find stellar sigma column
    stellar_sigma_candidates = [
        "stellar_sigma_1re", "STELLAR_SIGMA_1RE",
        "stellar_vel_sigma_1re", "STELLAR_VEL_SIGMA_1RE",
    ]
    stellar_sigma_col = find_column(dap_good, stellar_sigma_candidates)
    
    if ha_sigma_col is None or stellar_sigma_col is None:
        # Print diagnostic info
        print("\n=== DIAGNOSTIC: Available DAPall columns ===")
        sigma_cols = [c for c in dap_good.colnames if 'SIGMA' in c.upper()]
        ha_cols = [c for c in dap_good.colnames if 'HA' in c.upper() or 'HALPHA' in c.upper()]
        print(f"Columns containing 'SIGMA': {sigma_cols[:20]}")
        print(f"Columns containing 'HA': {ha_cols[:20]}")
        print(f"\nAll columns: {list(dap_good.colnames)[:50]}...")
        
        if ha_sigma_col is None:
            raise ValueError(
                f"Could not find Hα sigma column. Tried: {ha_sigma_candidates}"
            )
        if stellar_sigma_col is None:
            raise ValueError(
                f"Could not find stellar sigma column. Tried: {stellar_sigma_candidates}"
            )
    
    print(f"  Using Hα sigma column: {ha_sigma_col}")
    print(f"  Using stellar sigma column: {stellar_sigma_col}")
    
    ha_sigma_1re = dap_good[ha_sigma_col]
    stellar_sigma_1re = dap_good[stellar_sigma_col]

    # --- Hα flux within 1 Re from vector column EMLINE_GFLUX_1RE ---
    # In DAPall, emission line fluxes are stored as vector columns where each
    # row is an array of N_lines values. Hα is at index 24 (same as in MAPS).
    ha_flux_1re = None
    ha_flux_colname = "EMLINE_GFLUX_1RE[24]"
    
    # Hα emission line index (confirmed from MAPS header inspection)
    HA_LINE_INDEX = 24
    
    if "EMLINE_GFLUX_1RE" in dap_good.colnames:
        em_gflux_1re = dap_good["EMLINE_GFLUX_1RE"]  # vector column
        # Extract Hα flux (index 24) from each galaxy's emission line array
        ha_flux_1re = np.array([row[HA_LINE_INDEX] for row in em_gflux_1re])
        print(f"  Using Hα flux from: EMLINE_GFLUX_1RE[{HA_LINE_INDEX}]")
    else:
        print("  WARNING: EMLINE_GFLUX_1RE column not found in DAPall")
        print("  (will skip flux-based ranking)")

    # ------------------------------------------------------------------
    # Apply *science-motivated* filters
    # ------------------------------------------------------------------

    # 1) Axis ratio filter: avoid face-on (b/a ~1) and extreme edge-on (b/a ~0)
    ba_mask = (ba > 0.3) & (ba < 0.9)

    # 2) Non-zero Hα dispersion (avoid failed fits / no gas)
    sigma_mask = ha_sigma_1re > 10.0  # km/s; instrument ~70 km/s, but this is intrinsic

    # 3) Reasonable radius (avoid tiny / poorly-resolved systems)
    re_mask = reff_r > 2.0  # arcsec

    mask_all = ba_mask & sigma_mask & re_mask

    if ha_flux_1re is not None:
        flux_mask = ha_flux_1re > 0
        mask_all &= flux_mask

    if mask_all.sum() == 0:
        raise RuntimeError("No galaxies passed basic selection criteria.")

    plate = plate[mask_all]
    ifu = ifu[mask_all]
    plateifu = plateifu[mask_all]
    mangaid = mangaid[mask_all]
    ba = ba[mask_all]
    incl_deg = incl_deg[mask_all]
    reff_r = reff_r[mask_all]
    z = z[mask_all]
    ha_sigma_1re = ha_sigma_1re[mask_all]
    stellar_sigma_1re = stellar_sigma_1re[mask_all]

    if ha_flux_1re is not None:
        ha_flux_1re = ha_flux_1re[mask_all]

    # ------------------------------------------------------------------
    # Compute a heuristic ranking score.
    #
    # Intuition:
    #   - Prefer brighter Hα (if available) -> better SNR in velocity maps
    #   - Prefer moderate-high inclination (20–75 deg)
    #   - Prefer moderate dispersion (not completely cold, not AGN-level hot)
    # ------------------------------------------------------------------

    # Normalize components to ~[-1, +1] before combining.

    # 1) log Hα flux
    if ha_flux_1re is not None:
        log_flux = safe_log10(ha_flux_1re)
        # robust normalization
        lf_med = np.nanmedian(log_flux)
        lf_iqr = np.nanpercentile(log_flux, 75) - np.nanpercentile(log_flux, 25)
        lf_iqr = lf_iqr if lf_iqr > 0 else 1.0
        score_flux = (log_flux - lf_med) / lf_iqr
    else:
        score_flux = np.zeros_like(ha_sigma_1re, dtype=float)

    # 2) inclination preference:  0 deg = face-on, 90 = edge-on.
    # We prefer 30–70 degrees (roughly), with a soft peak around 55 deg.
    target_incl = 55.0
    width = 20.0
    score_incl = -((incl_deg - target_incl) / width) ** 2  # Gaussian-like penalty

    # 3) Hα dispersion: prefer typical disks (20–120 km/s).
    sigma = ha_sigma_1re
    # log-normal-ish scaling
    lsig = safe_log10(np.clip(sigma, 5, None))
    lsig_med = np.nanmedian(lsig)
    lsig_iqr = np.nanpercentile(lsig, 75) - np.nanpercentile(lsig, 25)
    lsig_iqr = lsig_iqr if lsig_iqr > 0 else 1.0
    score_sigma = (lsig - lsig_med) / lsig_iqr

    # Combined score (weights can be tuned)
    score = 1.0 * score_flux + 0.7 * score_incl + 0.5 * score_sigma

    # ------------------------------------------------------------------
    # Build direct MAPS URL for each candidate
    # ------------------------------------------------------------------
    base_url = (
        "https://data.sdss.org/sas/dr17/manga/spectro/analysis/"
        "v3_1_1/3.1.0/HYB10-MILESHC-MASTARHC2"
    )

    maps_url = []
    for p, i in zip(plate, ifu):
        maps_url.append(
            f"{base_url}/{p}/{i}/"
            f"manga-{p}-{i}-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz"
        )

    # ------------------------------------------------------------------
    # Assemble output table
    # ------------------------------------------------------------------
    out = Table()
    out["score"] = score
    out["plate"] = plate
    out["ifudesign"] = ifu
    out["plateifu"] = plateifu
    out["mangaid"] = mangaid
    out["z"] = z
    out["nsa_elpetro_ba"] = ba
    out["incl_deg"] = incl_deg
    out["reff_r_arcsec"] = reff_r
    out["ha_gsigma_1re"] = ha_sigma_1re
    out["stellar_sigma_1re"] = stellar_sigma_1re

    if ha_flux_1re is not None:
        out[ha_flux_colname] = ha_flux_1re

    out["maps_url"] = np.array(maps_url, dtype="U200")

    # ------------------------------------------------------------------
    # Basic sanity checks and summary statistics (diagnostics)
    # ------------------------------------------------------------------
    # Check that we have > 0 candidates
    assert len(plate) > 0, "No candidates after selection cuts."

    # Check that scores are finite
    if not np.all(np.isfinite(score)):
        raise RuntimeError("Non-finite values in ranking score.")

    print("\n=== Diagnostic summary BEFORE sorting ===")
    print(f"  Number of candidates: {len(plate)}")

    def summarize(name, arr):
        arr = np.asarray(arr)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            print(f"  {name:20s}: no finite values")
            return
        print(f"  {name:20s}: min={np.nanmin(finite):8.3f}, "
              f"median={np.nanmedian(finite):8.3f}, "
              f"max={np.nanmax(finite):8.3f}")

    summarize("incl_deg", incl_deg)
    summarize("reff_r_arcsec", reff_r)
    summarize("ha_gsigma_1re", ha_sigma_1re)
    summarize("stellar_sigma_1re", stellar_sigma_1re)
    if ha_flux_1re is not None:
        summarize("log10(Ha_flux_1re)", safe_log10(ha_flux_1re))
    summarize("score", score)

    # Sort by score descending (best first)
    out.sort("score")
    out.reverse()

    return out


def main():
    args = parse_args()

    dapall_path = Path(args.dapall)
    drpall_path = Path(args.drpall)

    if not dapall_path.exists():
        raise FileNotFoundError(dapall_path)
    if not drpall_path.exists():
        raise FileNotFoundError(drpall_path)

    print("Loading DAPall...")
    dap = load_dapall(str(dapall_path), args.daptype)
    print(f"  Rows in DAPall[{args.daptype}]: {len(dap)}")

    print("Loading DRPall...")
    drp = load_drpall(str(drpall_path))
    print(f"  Rows in DRPall[MANGA]: {len(drp)}")

    print("Building candidate table and ranking...")
    cand = build_candidate_table(dap, drp)
    print(f"  Candidates after cuts: {len(cand)}")

    # Check where known galaxy 8138-12704 lands in the ranking
    sel = (cand["plate"] == 8138) & (cand["ifudesign"] == 12704)
    if np.any(sel):
        idx = np.where(sel)[0][0]
        print("\n=== Known galaxy 8138-12704 found in candidate table ===")
        print(f"  Rank (after sorting): {idx + 1} of {len(cand)}")
        print(f"  Score: {cand['score'][idx]:.3f}")
        print(f"  Inclination: {cand['incl_deg'][idx]:.2f} deg")
        print(f"  Hα sigma: {cand['ha_gsigma_1re'][idx]:.2f} km/s")
        print(f"  Stellar sigma: {cand['stellar_sigma_1re'][idx]:.2f} km/s")
    else:
        print("\nWARNING: Galaxy 8138-12704 not found in candidate table after cuts.")

    n_top = min(args.n_top, len(cand))
    top = cand[:n_top]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top.write(out_path, format="csv", overwrite=True)

    print(f"\nWrote top {n_top} candidates to: {out_path}")

    # More detailed preview for diagnostics
    print("\n=== Top candidates (detailed preview) ===")
    preview_cols = [
        "score",
        "plate",
        "ifudesign",
        "plateifu",
        "z",
        "nsa_elpetro_ba",
        "incl_deg",
        "reff_r_arcsec",
        "ha_gsigma_1re",
        "stellar_sigma_1re",
    ]
    
    # Check if we have Hα flux column
    ha_flux_col = None
    for candidate_col in cand.colnames:
        if "EMLINE_GFLUX_1RE" in candidate_col or "ha_gflux" in candidate_col.lower():
            ha_flux_col = candidate_col
            preview_cols.append(candidate_col)
            break

    # Limit to first 5 rows for printing
    n_prev = min(5, len(top))
    for i in range(n_prev):
        row = top[i]
        print(f"\n-- Candidate {i + 1} --")
        for col in preview_cols:
            val = row[col]
            # Format based on type
            if isinstance(val, (float, np.floating)):
                print(f"  {col:22s} = {val:.4f}")
            else:
                print(f"  {col:22s} = {val}")
        print(f"  {'maps_url':22s} = {row['maps_url']}")


if __name__ == "__main__":
    main()


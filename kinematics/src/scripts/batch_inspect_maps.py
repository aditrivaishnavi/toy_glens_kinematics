#!/usr/bin/env python3
"""
batch_inspect_maps.py

Batch inspection of MaNGA MAPS FITS files to extract quality metrics
for ML training suitability and kinematic lensing studies.

Metrics extracted:
- Valid spaxel statistics
- Flux & velocity map coverage quality
- Mask contamination rate
- Velocity gradient strength (proxy for rotation clarity)
- Velocity–flux correlation (turbulence vs smooth disk)
- Dynamic range (helps detect overly flat discs)
- Flagging problematic cases (too masked / too flat / too turbulent)

Usage:
    python3 src/scripts/batch_inspect_maps.py \
        --maps_dir data/maps \
        --output data/maps_quality_summary.csv \
        --usable_index data/usable_maps_index.txt

    # Force re-inspection of all files (ignore cache):
    python3 src/scripts/batch_inspect_maps.py \
        --maps_dir data/maps --force

Features:
    - Idempotent: Re-running produces the same output
    - Incremental: Only new files are processed; cached results are reused
    - Use --force to re-inspect all files from scratch

Outputs:
    - Full CSV with all metrics for all files
    - Index file listing only usable FITS filenames (one per line)

Requirements:
    pip install pandas scikit-learn
"""

import os
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


# =============================================================================
# Configuration for MAPS inspection
# =============================================================================

HA_IDX = 24  # H-alpha line index (confirmed earlier)
MASK_BAD_VALUE = -999  # Standard mask for invalid spaxel

# Thresholds for flagging (relaxed to allow partially masked maps)
VELOCITY_GRAD_THRESHOLD = 2.0  # km/s per pixel - only truly flat fields flagged
MASK_FRACTION_THRESHOLD = 0.7  # above this = heavily masked (allows 50-70% masked outer sky)
FLUX_MAX_THRESHOLD = 5.0  # below this = low flux


# =============================================================================
# Helper: Fit simple velocity plane to check gradient strength
# =============================================================================

def compute_velocity_gradient(vel_map):
    """
    Estimate velocity gradient magnitude (km/s per pixel).
    
    Fits a plane v = a*x + b*y + c and returns sqrt(a^2 + b^2).
    Higher values indicate clearer rotation patterns.
    """
    nx, ny = vel_map.shape
    y_coords, x_coords = np.indices((nx, ny))
    
    mask = (~np.isnan(vel_map)) & (vel_map != MASK_BAD_VALUE)
    
    if np.sum(mask) < 30:
        return np.nan  # Too few valid pixels
    
    X = np.column_stack((x_coords[mask], y_coords[mask]))
    y = vel_map[mask]
    
    model = LinearRegression().fit(X, y)
    a, b = model.coef_
    
    return np.sqrt(a**2 + b**2)


def compute_fraction_valid_within_re(hdul, flux_map, re_multiplier=1.5):
    """
    Compute fraction of valid (unmasked) spaxels within 1.5 × R_eff.
    
    This metric ignores empty corners and focuses on the galaxy footprint,
    which is where the science actually happens. It directly fixes the 
    biggest weakness in simple mask fraction metrics—treating the square 
    MAPS frame as if all regions are equally meaningful, even though most 
    of the science exists only inside the galaxy footprint, not in the 
    masked sky corners.
    
    Args:
        hdul: Open FITS HDU list
        flux_map: 2D flux array (NaN where masked)
        re_multiplier: Multiple of R_eff to use (default 1.5)
        
    Returns:
        float: Fraction of valid spaxels within radius, or None if R_eff unavailable
    
    =========================================================================
    WHY 1.5 × Rₑ IS THE RIGHT CHOICE
    =========================================================================
    
    1. Physical justification:
       - 1.0 × Rₑ (half-light radius) → Inside is often bulge-dominated; 
         rotation can be weaker and less ordered.
       - 1.5 × Rₑ → Where disk galaxies usually show their STRONGEST 
         ordered rotation. This is the "sweet spot" for kinematic studies.
    
    2. Relevance for gravitational lensing:
       - If this galaxy were lensed by a foreground mass, Einstein arcs 
         would typically form at or around this radius.
       - Subhalo-induced velocity perturbations (our science target) would 
         be most detectable in this well-ordered rotation region.
    
    3. Practical data quality:
       - Inside 1.5 × Rₑ: High S/N, good spectral fits, reliable velocities.
       - Outside 1.5 × Rₑ: Often masked (low S/N), sky contamination, 
         or simply empty IFU corners that never contained galaxy light.
    
    4. Why NOT use 1.0 × Rₑ:
       - Too restrictive; misses the extended disk where rotation is clearest.
       - Many galaxies have interesting kinematics between 1.0–2.0 × Rₑ.
    
    5. Why NOT use 2.0 × Rₑ or larger:
       - Starts including noisy outer regions and masked corners.
       - Diminishing returns: flux drops, velocity errors increase.
    
    Using this metric makes the inspection:
       ✔ More physically honest
       ✔ More aligned with how astrophysical data are actually used
       ✔ More useful for final lensing + subhalo simulations
    =========================================================================
    """
    # Try to get effective radius from header
    # Check multiple possible keywords
    re_arcsec = None
    
    # Try primary header first
    primary_hdr = hdul[0].header
    for key in ['REFF', 'RE', 'R_EFF', 'NSA_ELPETRO_TH50_R', 'PETRO_TH50']:
        if key in primary_hdr:
            re_arcsec = primary_hdr[key]
            break
    
    # If not found, try to get from other extensions
    if re_arcsec is None:
        # Check if there's an elliptical coordinate extension
        if 'SPX_ELLCOO' in [h.name for h in hdul]:
            # ELLCOO contains elliptical coordinates normalized by R_eff
            # We can use this directly
            try:
                ellcoo = hdul['SPX_ELLCOO'].data
                # ellcoo[0] is typically R/R_eff, ellcoo[1] is azimuthal angle
                r_over_re = ellcoo[0]  # radius in units of R_eff
                
                ny, nx = flux_map.shape
                inside_re_mask = r_over_re <= re_multiplier
                valid_inside = inside_re_mask & (~np.isnan(flux_map))
                
                n_inside = inside_re_mask.sum()
                if n_inside == 0:
                    return None
                    
                return float(valid_inside.sum()) / float(n_inside)
            except Exception:
                pass
    
    if re_arcsec is None:
        return None  # Can't compute without R_eff
    
    # Convert R_eff (arcsec) to pixel units using MaNGA scale
    MANGA_PIXEL_SCALE = 0.5  # arcsec per spaxel
    re_pix = re_arcsec / MANGA_PIXEL_SCALE
    re_limit = re_multiplier * re_pix  # radius in pixel units
    
    # Create coordinate grid
    ny, nx = flux_map.shape
    y, x = np.indices((ny, nx))
    cx, cy = nx / 2.0, ny / 2.0  # center of map
    
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    inside_re_mask = dist <= re_limit
    
    # Valid spaxels = inside footprint AND not masked
    valid_inside = inside_re_mask & (~np.isnan(flux_map))
    
    n_inside = inside_re_mask.sum()
    if n_inside == 0:
        return None
        
    return float(valid_inside.sum()) / float(n_inside)


# =============================================================================
# Main processing function
# =============================================================================

def inspect_maps_file(filepath):
    """
    Extract key metrics for quality & usability from a single MAPS file.
    
    Returns a dict with metrics and status.
    """
    try:
        hdul = fits.open(filepath)
    except Exception as e:
        return {"file": filepath, "status": f"ERROR: {e}"}
    
    out = {"file": os.path.basename(filepath), "status": "OK"}
    
    try:
        # Extract flux & velocity
        # Use Hα gas velocity (EMLINE_GVEL index 24) for kinematic maps
        # We use gas velocity, not stellar velocity, as it shows cleaner rotation in disk galaxies
        flux_map = hdul["EMLINE_GFLUX"].data[HA_IDX]
        vel_map = hdul["EMLINE_GVEL"].data[HA_IDX]
        mask_map = hdul["EMLINE_GVEL_MASK"].data[HA_IDX]
        
        # Replace bad values with NaN
        flux = np.where(flux_map == MASK_BAD_VALUE, np.nan, flux_map.astype(float))
        vel = np.where(vel_map == MASK_BAD_VALUE, np.nan, vel_map.astype(float))
        
        # Also apply mask extension
        vel = np.where(mask_map > 0, np.nan, vel)
        flux = np.where(hdul["EMLINE_GFLUX_MASK"].data[HA_IDX] > 0, np.nan, flux)
        
        # Mask fraction
        mask_bad_frac = np.isnan(vel).sum() / vel.size
        
        # Basic flux stats
        out["flux_min"] = float(np.nanmin(flux)) if not np.all(np.isnan(flux)) else np.nan
        out["flux_max"] = float(np.nanmax(flux)) if not np.all(np.isnan(flux)) else np.nan
        out["flux_mean"] = float(np.nanmean(flux)) if not np.all(np.isnan(flux)) else np.nan
        
        # Basic velocity stats
        out["vel_min"] = float(np.nanmin(vel)) if not np.all(np.isnan(vel)) else np.nan
        out["vel_max"] = float(np.nanmax(vel)) if not np.all(np.isnan(vel)) else np.nan
        
        # Diversity measure — are spaxels all same?
        out["vel_std"] = float(np.nanstd(vel)) if not np.all(np.isnan(vel)) else np.nan
        
        # Velocity gradient strength
        out["vel_grad"] = float(compute_velocity_gradient(vel))
        
        # Correlation flux vs velocity – to detect turbulence
        valid = (~np.isnan(vel)) & (~np.isnan(flux))
        if np.sum(valid) > 30:
            corr, _ = pearsonr(vel[valid], flux[valid])
            out["flux_vel_corr"] = float(corr)
        else:
            out["flux_vel_corr"] = np.nan
        
        # Mask severity
        out["mask_fraction"] = float(mask_bad_frac)
        
        # Valid spaxel count
        out["n_valid_spaxels"] = int(np.sum(valid))
        out["total_spaxels"] = int(vel.size)
        
        # ---------------------------------------------------------------
        # Fraction valid within 1.5 × R_eff
        # ---------------------------------------------------------------
        # This is a MORE MEANINGFUL metric than total mask_fraction because:
        # - It ignores the empty corners of the square MAPS frame
        # - It focuses on the galaxy footprint where science happens
        # - 1.5 × Rₑ is where disks show strongest ordered rotation
        # - This is where lensing signatures (Einstein arcs, subhalo 
        #   perturbations) would actually appear
        # 
        # A galaxy with 60% total mask fraction but 90% valid within 
        # 1.5 × Rₑ is actually GOOD for our purposes!
        # ---------------------------------------------------------------
        frac_valid_in_re = compute_fraction_valid_within_re(hdul, flux)
        out["frac_valid_in_1.5Re"] = frac_valid_in_re
        
        # Flags (help with selection)
        out["flag_low_rotation"] = out["vel_grad"] < VELOCITY_GRAD_THRESHOLD if not np.isnan(out["vel_grad"]) else True
        out["flag_heavily_masked"] = mask_bad_frac > MASK_FRACTION_THRESHOLD
        out["flag_low_flux"] = out["flux_max"] < FLUX_MAX_THRESHOLD if not np.isnan(out["flux_max"]) else True
        
        # Overall usability flag
        out["usable"] = not (out["flag_low_rotation"] or out["flag_heavily_masked"] or out["flag_low_flux"])
        
    except Exception as e:
        out["status"] = f"ERROR extracting data: {e}"
    
    hdul.close()
    return out


# =============================================================================
# Main script entry
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch inspect MAPS FITS files for ML training suitability"
    )
    parser.add_argument(
        "--maps_dir",
        type=str,
        required=True,
        help="Directory containing MAPS FITS files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/maps_quality_summary.csv",
        help="CSV file to save summary"
    )
    parser.add_argument(
        "--usable_index",
        type=str,
        default="data/usable_maps_index.txt",
        help="Text file listing only usable FITS files (one per line)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-inspection of all files (ignore existing results)"
    )
    args = parser.parse_args()
    
    # Find all FITS files in directory
    all_files = [
        os.path.join(args.maps_dir, f)
        for f in os.listdir(args.maps_dir)
        if f.endswith(".fits.gz") or f.endswith(".fits")
    ]
    
    if not all_files:
        print(f"[ERROR] No FITS files found in {args.maps_dir}")
        return
    
    # =========================================================================
    # Idempotency & Incremental Addition
    # =========================================================================
    # Check if output CSV already exists and load previously processed files
    existing_results = []
    already_processed = set()
    
    if os.path.exists(args.output) and not args.force:
        try:
            existing_df = pd.read_csv(args.output)
            existing_results = existing_df.to_dict('records')
            already_processed = set(existing_df['file'].tolist())
            print(f"[INFO] Found existing results with {len(already_processed)} files already processed.")
        except Exception as e:
            print(f"[WARN] Could not load existing results: {e}")
            print("[WARN] Will re-process all files.")
    
    # Filter to only new files that haven't been processed
    files_to_process = [
        f for f in all_files 
        if os.path.basename(f) not in already_processed
    ]
    
    print(f"\n{'='*60}")
    print("BATCH MAPS INSPECTION")
    print(f"{'='*60}")
    print(f"[INFO] Total FITS files in directory : {len(all_files)}")
    print(f"[INFO] Already processed (cached)    : {len(already_processed)}")
    print(f"[INFO] New files to inspect          : {len(files_to_process)}")
    print(f"[INFO] Output will be saved to       : {args.output}")
    if args.force:
        print(f"[INFO] --force specified: re-inspecting ALL files")
        files_to_process = all_files
        existing_results = []
    print(f"{'='*60}\n")
    
    if len(files_to_process) == 0:
        print("[INFO] No new files to process. Results are up to date.")
        print("[INFO] Use --force to re-inspect all files.\n")
        # Still regenerate usable index from existing results
        results = existing_results
    else:
        # Process new files
        results = list(existing_results)  # Start with existing results
        new_usable_count = 0
        new_error_count = 0
        
        for i, fpath in enumerate(files_to_process, start=1):
            print(f"--- Inspecting {i}/{len(files_to_process)}: {os.path.basename(fpath)} ---")
            res = inspect_maps_file(fpath)
            
            if res["status"] == "OK":
                print(f"  Velocity gradient : {res['vel_grad']:.2f} km/s/pixel")
                print(f"  Velocity STD      : {res['vel_std']:.2f} km/s")
                print(f"  Mask fraction     : {res['mask_fraction']:.1%}")
                print(f"  Valid spaxels     : {res['n_valid_spaxels']} / {res['total_spaxels']}")
                
                # Print fraction valid within 1.5 R_eff (if available)
                frac_in_re = res.get("frac_valid_in_1.5Re")
                if frac_in_re is not None:
                    print(f"  Valid in 1.5×Rₑ   : {frac_in_re:.1%}")
                else:
                    print(f"  Valid in 1.5×Rₑ   : N/A (Rₑ not in header)")
                
                if not np.isnan(res['flux_vel_corr']):
                    print(f"  Flux-Vel Corr     : {res['flux_vel_corr']:.3f}")
                else:
                    print(f"  Flux-Vel Corr     : N/A")
                
                # Print flags
                flags = []
                if res["flag_low_rotation"]:
                    flags.append("LOW_ROTATION")
                if res["flag_heavily_masked"]:
                    flags.append("HEAVILY_MASKED")
                if res["flag_low_flux"]:
                    flags.append("LOW_FLUX")
                
                if flags:
                    print(f"  Flags             : {', '.join(flags)}")
                else:
                    print(f"  Flags             : NONE (good candidate!)")
                
                if res.get("usable", False):
                    new_usable_count += 1
                    print(f"  >>> USABLE: YES <<<")
                else:
                    print(f"  >>> USABLE: NO <<<")
            else:
                print(f"  Status: {res['status']}")
                print("  File skipped due to read/extract error.")
                new_error_count += 1
            
            print()
            results.append(res)
    
    # =========================================================================
    # Save results (merged: existing + new)
    # =========================================================================
    
    # Save full results to CSV
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    
    # Save usable files index (one filename per line)
    usable_files = [
        res["file"] for res in results 
        if res.get("status") == "OK" and res.get("usable", False)
    ]
    
    os.makedirs(os.path.dirname(args.usable_index) or ".", exist_ok=True)
    with open(args.usable_index, "w") as f:
        for fname in usable_files:
            f.write(f"{fname}\n")
    
    # Compute totals from all results (existing + new)
    total_files = len(results)
    total_errors = sum(1 for r in results if r.get("status", "").startswith("ERROR"))
    total_usable = len(usable_files)
    total_not_usable = total_files - total_errors - total_usable
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total files in results : {total_files}")
    print(f"    - Previously cached  : {len(already_processed)}")
    print(f"    - Newly processed    : {len(files_to_process)}")
    print(f"  Successfully parsed    : {total_files - total_errors}")
    print(f"  Errors                 : {total_errors}")
    print(f"  Usable for ML          : {total_usable}")
    print(f"  Not usable             : {total_not_usable}")
    print(f"\n  Full results saved to  : {args.output}")
    print(f"  Usable files index     : {args.usable_index}")
    print(f"{'='*60}\n")
    
    # Print threshold reference
    print("THRESHOLD REFERENCE:")
    print(f"  vel_grad < {VELOCITY_GRAD_THRESHOLD} km/s/pix  → flag_low_rotation")
    print(f"  mask_fraction > {MASK_FRACTION_THRESHOLD:.0%}      → flag_heavily_masked")
    print(f"  flux_max < {FLUX_MAX_THRESHOLD}             → flag_low_flux")
    print()


if __name__ == "__main__":
    main()


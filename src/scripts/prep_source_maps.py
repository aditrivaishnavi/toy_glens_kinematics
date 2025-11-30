#!/usr/bin/env python3
"""
prep_source_maps.py

Prepare 2-channel source tensors (flux, velocity) from MaNGA DR17 MAPS files.

For each selected MAPS file, this script:
- Loads an H-alpha emission-line flux map and velocity map.
- Applies masks and cleans invalid values.
- Normalizes flux to [0, 1] using robust percentiles.
- Normalizes velocity to roughly [-1, 1] using robust scaling.
- Resamples both maps to a fixed grid (e.g., 64x64).
- Stacks them into a tensor of shape (2, H, W):
    channel 0: normalized flux
    channel 1: normalized velocity
- Saves each tensor as:  source_tensor_<plateifu>.npy
- Writes a PNG preview for sanity checking.

Selection logic:
- MAPS files are collected from --maps_dir.
- If --index_file is provided (e.g., data/usable_maps_index.txt),
  only files whose basenames appear in that text file are processed.
- Optionally, --quality_csv can further refine this list based on flags
  such as 'flag_low_rotation' or 'vel_grad'.

Typical usage:
    python3 src/scripts/prep_source_maps.py \
        --maps_dir data/maps \
        --index_file data/usable_maps_index.txt \
        --outdir data/source_tensors \
        --target_size 64

    # With velocity degradation (simulates lower-res IFU than imaging):
    python3 src/scripts/prep_source_maps.py \
        --maps_dir data/maps \
        --index_file data/usable_maps_index.txt \
        --outdir data/source_tensors \
        --target_size 64 \
        --vel_degrade_factor 2.0
"""

import os
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter
import pandas as pd


# -------------------------------------------------------------------
# Configuration / defaults
# -------------------------------------------------------------------

# Confirmed: H-alpha is index 24 in EMLINE_GFLUX/EMLINE_GVEL header
HA_IDX = 24

# MaNGA nominal pixel scale (arcsec per spaxel)
MANGA_PIXEL_SCALE = 0.5

# Robust normalization percentiles for flux
FLUX_PMIN = 5.0
FLUX_PMAX = 99.5

# Robust percentile for velocity scale (of |v - v_med|)
VEL_SCALE_PCT = 95.0


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def plateifu_from_filename(fname: str) -> str:
    """
    Extract plate-IFU identifier from a MAPS filename.

    Expected pattern:
        manga-<plate>-<ifudesign>-MAPS-*.fits.gz
    Example:
        manga-11013-6101-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz
        -> plateifu = "11013-6101"
    """
    base = os.path.basename(fname)
    parts = base.split("-")
    if len(parts) < 4 or parts[0].lower() != "manga":
        # Fallback: strip extensions and return stem
        return Path(base).stem
    plate = parts[1]
    ifudesign = parts[2]
    return f"{plate}-{ifudesign}"


def load_maps_ha_flux_vel(path: Path):
    """
    Load H-alpha flux and velocity maps from a MaNGA MAPS file.

    Returns:
        flux (2D np.ndarray): H-alpha flux map
        vel  (2D np.ndarray): H-alpha velocity map
        mask (2D np.ndarray, bool): True for valid pixels, False for invalid
    """
    with fits.open(path) as hdul:
        flux_map = hdul["EMLINE_GFLUX"].data[HA_IDX]
        vel_map = hdul["EMLINE_GVEL"].data[HA_IDX]
        vel_mask = hdul["EMLINE_GVEL_MASK"].data[HA_IDX]

        # Define invalid pixels:
        # - NaNs
        # - velocity mask != 0
        # - non-positive flux
        # - sentinel -999 in either flux or velocity
        invalid = (
            np.isnan(flux_map)
            | np.isnan(vel_map)
            | (vel_mask != 0)
            | (flux_map <= 0)
            | (flux_map == -999)
            | (vel_map == -999)
        )

        mask = ~invalid

        flux = flux_map.astype(float)
        vel = vel_map.astype(float)
        flux[invalid] = np.nan
        vel[invalid] = np.nan

    return flux, vel, mask


def normalize_flux(flux: np.ndarray) -> np.ndarray:
    """
    Normalize flux map to [0, 1] using robust percentiles.

    Steps:
        - Use only positive, finite values to compute percentiles.
        - Clip to [pmin, pmax].
        - Linearly map to [0, 1].
        - NaNs -> 0.
    """
    valid = np.isfinite(flux) & (flux > 0)
    if np.sum(valid) < 20:
        fmin = np.nanmin(flux)
        fmax = np.nanmax(flux)
    else:
        fmin = np.percentile(flux[valid], FLUX_PMIN)
        fmax = np.percentile(flux[valid], FLUX_PMAX)

    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
        norm = np.zeros_like(flux, dtype=float)
        norm[valid] = 1.0
        return norm

    clipped = np.clip(flux, fmin, fmax)
    norm = (clipped - fmin) / (fmax - fmin)
    norm[~np.isfinite(norm)] = 0.0
    return norm


def normalize_velocity(vel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Normalize velocity map to roughly [-1, 1] using robust scaling.

    Steps:
        - Use only valid pixels (mask == True and finite).
        - Compute median velocity v_med.
        - Compute scale from VEL_SCALE_PCT percentile of |v - v_med|.
        - Transform: (v - v_med) / scale.
        - Clip to [-3, 3], then divide by 3 to map to [-1, 1].
        - Invalid pixels -> 0.
    """
    valid = mask & np.isfinite(vel)
    if np.sum(valid) < 20:
        return np.zeros_like(vel, dtype=float)

    v = vel[valid]
    v_med = np.median(v)
    dev = np.abs(v - v_med)
    scale = np.percentile(dev, VEL_SCALE_PCT)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.std(v)
        if not np.isfinite(scale) or scale <= 0:
            return np.zeros_like(vel, dtype=float)

    v_scaled = (vel - v_med) / scale
    v_scaled = np.clip(v_scaled, -3.0, 3.0)
    v_scaled /= 3.0
    v_scaled[~np.isfinite(v_scaled)] = 0.0
    return v_scaled


def resample_map(img: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resample a 2D map to (target_size, target_size) using bilinear interpolation.
    """
    ny, nx = img.shape
    if ny == target_size and nx == target_size:
        return img
    zoom_y = target_size / ny
    zoom_x = target_size / nx
    return zoom(img, (zoom_y, zoom_x), order=1)


def degrade_velocity_map(
    vel_norm: np.ndarray, 
    target_size: int, 
    degrade_factor: float
) -> np.ndarray:
    """
    Degrade a normalized velocity map to mimic lower resolution.

    This is useful for simulating scenarios where the IFU (velocity) has
    lower effective resolution than imaging (flux), which is common in
    multi-modal lensing studies (e.g., HST imaging + ground-based IFU).

    Args:
        vel_norm: Normalized velocity map (2D array)
        target_size: Final output size (H = W = target_size)
        degrade_factor: Degradation factor
            - If <= 1.0: no extra degradation, just resample to target_size
            - If > 1.0: downsample by this factor, blur, then upsample back

    Returns:
        Degraded velocity map of shape (target_size, target_size)

    Example:
        - degrade_factor = 1.0: No degradation (default behavior)
        - degrade_factor = 2.0: Downsample by 2×, blur, upsample back
        - degrade_factor = 3.0: More aggressive degradation

    This preserves the final shape (target_size, target_size) but removes
    small-scale structure from the velocity field.
    """
    if degrade_factor <= 1.0:
        # No extra degradation – just resample directly
        return resample_map(vel_norm, target_size)

    ny, nx = vel_norm.shape
    
    # Coarse size in each dimension; ensure at least 8x8
    coarse_y = max(8, int(ny / degrade_factor))
    coarse_x = max(8, int(nx / degrade_factor))

    # Step 1: Downsample to coarse grid
    vel_coarse = zoom(vel_norm, (coarse_y / ny, coarse_x / nx), order=1)

    # Step 2: Mild Gaussian blur on coarse grid
    # sigma ~ 0.5–1 pixel is usually enough to smooth high-frequency noise
    vel_coarse_blur = gaussian_filter(vel_coarse, sigma=0.7)

    # Step 3: Upsample back to target_size
    vel_final = zoom(
        vel_coarse_blur, 
        (target_size / coarse_y, target_size / coarse_x), 
        order=1
    )

    return vel_final


def make_preview_plot(
    flux_resampled: np.ndarray,
    vel_resampled: np.ndarray,
    out_path: Path,
    title: str,
):
    """
    Save a side-by-side preview PNG of the resampled flux and velocity maps.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax0, ax1 = axes

    im0 = ax0.imshow(flux_resampled, origin="lower", cmap="magma")
    ax0.set_title("Flux (norm)")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    im1 = ax1.imshow(vel_resampled, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_title("Velocity (norm)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def read_index_file(index_path: Path) -> set:
    """
    Read an index file listing usable MAPS filenames (one per line).

    Example file content:
        manga-8652-12703-MAPS.fits.gz
        manga-12071-12702-MAPS.fits.gz
        ...

    Returns:
        Set of filename strings.
    """
    usable = set()
    with index_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            usable.add(line)
    return usable


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare 2-channel (flux, velocity) source tensors from MaNGA MAPS."
    )
    parser.add_argument(
        "--maps_dir",
        type=str,
        required=True,
        help="Directory containing MAPS FITS files (e.g. data/maps)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/source_tensors",
        help="Directory to store output tensors and previews",
    )
    parser.add_argument(
        "--index_file",
        type=str,
        default=None,
        help=(
            "Optional text file listing usable MAPS filenames (one per line), "
            "e.g. data/usable_maps_index.txt. Only these files will be processed."
        ),
    )
    parser.add_argument(
        "--quality_csv",
        type=str,
        default=None,
        help=(
            "Optional CSV (e.g. maps_quality_summary.csv) to further filter galaxies "
            "based on flags like 'flag_low_rotation' or 'vel_grad'. "
            "This is applied after index_file filtering, if both are provided."
        ),
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=64,
        help="Target square size for resampled maps (H = W = target_size)",
    )
    parser.add_argument(
        "--vel_degrade_factor",
        type=float,
        default=1.0,
        help=(
            "Optional degradation factor for velocity maps. "
            "If >1, velocity is first downsampled by this factor and then "
            "upsampled back to target_size, mimicking lower effective resolution. "
            "If 1.0 (default), no extra degradation is applied. "
            "Use 2.0 or 3.0 to simulate IFU resolution lower than imaging."
        ),
    )
    args = parser.parse_args()

    maps_dir = Path(args.maps_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    preview_dir = outdir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Collect MAPS files present in the directory
    maps_files = sorted(
        [p for p in maps_dir.glob("*.fits.gz") if "MAPS" in p.name]
        + [p for p in maps_dir.glob("*.fits") if "MAPS" in p.name]
    )

    if not maps_files:
        print(f"[FATAL] No MAPS FITS files found in {maps_dir}")
        return

    print("============================================================")
    print("PREP SOURCE MAPS")
    print("============================================================")
    print(f"[INFO] Found {len(maps_files)} MAPS files in {maps_dir}")
    print(f"[INFO] Target size: {args.target_size}x{args.target_size}")
    if args.vel_degrade_factor > 1.0:
        print(f"[INFO] Velocity degradation factor: {args.vel_degrade_factor}x")
        print(f"       (Velocity will be downsampled, blurred, and upsampled)")
    else:
        print(f"[INFO] Velocity degradation: OFF (factor={args.vel_degrade_factor})")
    if args.index_file:
        print(f"[INFO] Using index file: {args.index_file}")
    if args.quality_csv:
        print(f"[INFO] Optional quality CSV: {args.quality_csv}")
    print("============================================================")

    # Start with all filenames present
    allowed_files = set(p.name for p in maps_files)

    # 1) Filter by index_file (primary selection)
    if args.index_file is not None:
        index_path = Path(args.index_file)
        if not index_path.exists():
            print(f"[FATAL] index_file not found: {index_path}")
            return
        usable_from_index = read_index_file(index_path)
        before = len(allowed_files)
        allowed_files = allowed_files.intersection(usable_from_index)
        print(f"[INFO] Index filter: {before} -> {len(allowed_files)} files "
              f"after intersecting with {index_path.name}")

    # 2) Optional further filtering by quality_csv
    if args.quality_csv is not None:
        qpath = Path(args.quality_csv)
        if not qpath.exists():
            print(f"[WARN] quality_csv not found: {qpath}. Ignoring filter.")
        else:
            dfq = pd.read_csv(qpath)
            if "file" in dfq.columns:
                dfq = dfq[dfq["file"].isin(allowed_files)].copy()

                if "flag_low_rotation" in dfq.columns:
                    mask_usable = (dfq["flag_low_rotation"] == False)  # noqa: E712
                elif "vel_grad" in dfq.columns:
                    mask_usable = dfq["vel_grad"] >= 2.0
                else:
                    mask_usable = np.ones(len(dfq), dtype=bool)

                usable_files = set(dfq.loc[mask_usable, "file"].values)
                before = len(allowed_files)
                allowed_files = allowed_files.intersection(usable_files)
                print(f"[INFO] Quality CSV filter: {before} -> {len(allowed_files)} "
                      "files remaining")
            else:
                print("[WARN] quality_csv has no 'file' column; ignoring filter.")

    # Now allowed_files contains the basenames to process
    if not allowed_files:
        print("[FATAL] No MAPS files remaining after filtering. Nothing to do.")
        return

    n_total = 0
    n_success = 0
    for path in maps_files:
        if path.name not in allowed_files:
            continue

        n_total += 1
        plateifu = plateifu_from_filename(path.name)
        print(f"\n[INFO] Processing {path.name} (plateifu={plateifu})")

        try:
            flux, vel, mask = load_maps_ha_flux_vel(path)
        except Exception as e:
            print(f"[ERROR] Failed to load maps from {path}: {e}")
            continue

        flux_norm = normalize_flux(flux)
        vel_norm = normalize_velocity(vel, mask)

        flux_norm[~np.isfinite(flux_norm)] = 0.0
        vel_norm[~np.isfinite(vel_norm)] = 0.0

        flux_resampled = resample_map(flux_norm, args.target_size)
        vel_resampled = degrade_velocity_map(vel_norm, args.target_size, args.vel_degrade_factor)

        tensor = np.stack([flux_resampled, vel_resampled], axis=0)

        tensor_fname = outdir / f"source_tensor_{plateifu}.npy"
        np.save(tensor_fname, tensor)
        print(f"[OK] Saved tensor: {tensor_fname}")

        preview_fname = preview_dir / f"preview_{plateifu}.png"
        make_preview_plot(
            flux_resampled,
            vel_resampled,
            preview_fname,
            title=f"{plateifu} (Hα flux + vel, norm)",
        )
        print(f"[OK] Saved preview: {preview_fname}")

        n_success += 1

    print("\n============================================================")
    print("SUMMARY")
    print("============================================================")
    print(f"  MAPS files in directory : {len(maps_files)}")
    print(f"  MAPS files after filter : {len(allowed_files)}")
    print(f"  Tensors generated       : {n_success}")
    print(f"  Output directory        : {outdir}")
    print("============================================================")


if __name__ == "__main__":
    main()


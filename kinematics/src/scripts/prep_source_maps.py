#!/usr/bin/env python3
"""
prep_source_maps.py

Prepare MaNGA MAPS files into 2-channel (flux, velocity) tensors of fixed size,
and compute simple compactness metrics on the resampled flux maps.

This script is idempotent and incremental:
- Skips galaxies that already have a tensor file AND a CSV entry.
- Merges new results with existing CSV data.
- Use --force to reprocess all galaxies.

- Reads a list of usable MAPS filenames from an index file.
- For each MAPS file:
  * Extract Hα emission-line flux (EMLINE_GFLUX, index 24).
  * Extract gas velocity (EMLINE_GVEL, index 24).
  * Apply masks.
  * Normalize flux and velocity robustly.
  * Resample both to (target_size x target_size) using bilinear interpolation.
  * Stack into a tensor (2, H, W) and save as .npy.
  * Save a preview PNG for visual inspection.
  * Compute compactness metrics on the resampled flux map:
      - total flux
      - fraction of flux in central 5x5 pixels
      - half-light radius (r50) in pixels
      - r80 radius in pixels
  * Append metrics to a CSV summary file.

Usage:
    python src/scripts/prep_source_maps.py \
        --maps_dir data/maps \
        --index_file data/usable_maps_index.txt \
        --target_size 64 \
        --vel_degrade_factor 1.0

    # Force reprocessing all galaxies:
    python src/scripts/prep_source_maps.py \
        --maps_dir data/maps \
        --index_file data/usable_maps_index.txt \
        --force
"""

import argparse
import os
import csv

import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


HA_INDEX = 24  # H-alpha index in EMLINE_GFLUX/EMLINE_GVEL cubes


def robust_flux_normalization(flux_map, mask):
    """
    Normalize flux to [0, 1] using robust percentiles on valid pixels.

    Parameters
    ----------
    flux_map : 2D np.ndarray
        Raw flux map (e.g., erg/s/cm^2/spaxel units).
    mask : 2D np.ndarray (bool)
        True where pixels are valid.

    Returns
    -------
    flux_norm : 2D np.ndarray
        Flux normalized to ~[0, 1].
    stats : dict
        Dictionary with normalization parameters (p5, p995, total_flux_raw).
    """
    valid = mask & np.isfinite(flux_map)
    if not np.any(valid):
        return np.zeros_like(flux_map), {
            "p5": 0.0,
            "p995": 0.0,
            "total_flux_raw": 0.0,
        }

    vals = flux_map[valid]
    p5 = np.percentile(vals, 5.0)
    p995 = np.percentile(vals, 99.5)

    # Avoid degenerate ranges
    if p995 <= p5:
        scale = 1.0
    else:
        scale = p995 - p5

    flux_norm = (flux_map - p5) / scale
    flux_norm[flux_norm < 0.0] = 0.0
    flux_norm[flux_norm > 1.0] = 1.0
    flux_norm[~valid] = 0.0

    return flux_norm, {
        "p5": float(p5),
        "p995": float(p995),
        "total_flux_raw": float(vals.sum()),
    }


def robust_velocity_normalization(vel_map, mask):
    """
    Center and scale velocity map to ~[-1, 1] using robust statistics.

    Parameters
    ----------
    vel_map : 2D np.ndarray
        Raw velocity map (km/s).
    mask : 2D np.ndarray (bool)
        True where pixels are valid.

    Returns
    -------
    vel_norm : 2D np.ndarray
        Velocity normalized to ~[-1, 1].
    stats : dict
        Dictionary with normalization parameters (median, scale, p95abs).
    """
    valid = mask & np.isfinite(vel_map)
    if not np.any(valid):
        return np.zeros_like(vel_map), {
            "median": 0.0,
            "scale": 1.0,
            "p95abs": 0.0,
        }

    vals = vel_map[valid]
    med = np.median(vals)
    dev = np.abs(vals - med)
    p95abs = np.percentile(dev, 95.0)
    scale = p95abs if p95abs > 0 else 1.0

    vel_norm = (vel_map - med) / scale
    vel_norm = np.clip(vel_norm, -1.0, 1.0)
    vel_norm[~valid] = 0.0

    return vel_norm, {
        "median": float(med),
        "scale": float(scale),
        "p95abs": float(p95abs),
    }


def resample_map(map2d, target_size):
    """
    Resample a 2D map to (target_size, target_size) using bilinear interpolation.

    Parameters
    ----------
    map2d : 2D np.ndarray
        Input map.
    target_size : int
        Desired size in pixels.

    Returns
    -------
    map_resampled : 2D np.ndarray
        Resampled map.
    """
    ny, nx = map2d.shape
    if ny == target_size and nx == target_size:
        return map2d.copy()

    zoom_y = target_size / ny
    zoom_x = target_size / nx
    return zoom(map2d, (zoom_y, zoom_x), order=1)


def compute_compactness_metrics(flux_64):
    """
    Compute simple compactness metrics on a normalized 64x64 flux map.

    Metrics:
      - total_flux: sum of all pixel values (already normalized, but relative is useful).
      - flux_frac_central_5x5: fraction of flux inside central 5x5 pixels.
      - r_half_pix: half-light radius in pixels.
      - r80_pix: radius enclosing 80% of the total flux.

    Assumes the galaxy is reasonably centered in the 64x64 frame.

    Parameters
    ----------
    flux_64 : 2D np.ndarray, shape (64, 64)
        Normalized flux map.

    Returns
    -------
    metrics : dict
        Dictionary of computed metrics.
    """
    H, W = flux_64.shape
    assert H == W, "Flux map is expected to be square."
    n = H

    total_flux = float(flux_64.sum())
    if total_flux <= 0:
        # Degenerate case: no flux
        return {
            "total_flux": 0.0,
            "flux_frac_central_5x5": 0.0,
            "r_half_pix": np.nan,
            "r80_pix": np.nan,
        }

    # Center coordinates (assume center pixel at (n-1)/2)
    cy = (n - 1) / 2.0
    cx = (n - 1) / 2.0

    # ----- Central 5x5 fraction -----
    half_box = 2  # 5x5 = center +/- 2
    y_min = int(max(0, cy - half_box))
    y_max = int(min(n - 1, cy + half_box))
    x_min = int(max(0, cx - half_box))
    x_max = int(min(n - 1, cx + half_box))

    central_flux = float(flux_64[y_min:y_max+1, x_min:x_max+1].sum())
    frac_central_5x5 = central_flux / total_flux

    # ----- Half-light radius and r80 (in pixels) -----
    yy, xx = np.indices((n, n))
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2).astype(np.float32)

    r_flat = r.flatten()
    f_flat = flux_64.flatten()

    # Sort pixels by radius
    order = np.argsort(r_flat)
    r_sorted = r_flat[order]
    f_sorted = f_flat[order]

    cumsum_flux = np.cumsum(f_sorted)
    frac_cumsum = cumsum_flux / total_flux

    # r_half: radius where cumulative fraction >= 0.5
    r_half = r_sorted[np.searchsorted(frac_cumsum, 0.5)]
    # r80: radius where cumulative fraction >= 0.8
    r80 = r_sorted[np.searchsorted(frac_cumsum, 0.8)]

    return {
        "total_flux": total_flux,
        "flux_frac_central_5x5": float(frac_central_5x5),
        "r_half_pix": float(r_half),
        "r80_pix": float(r80),
    }


def save_preview_png(flux_64, vel_64, out_path, title=""):
    """
    Save side-by-side preview of flux and velocity maps.

    Parameters
    ----------
    flux_64 : 2D np.ndarray
    vel_64 : 2D np.ndarray
    out_path : str
    title : str
    """
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    ax0, ax1 = axes

    im0 = ax0.imshow(flux_64, origin="lower", cmap="inferno")
    ax0.set_title("Flux (norm)")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    im1 = ax1.imshow(vel_64, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_title("Velocity (norm)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maps_dir",
        type=str,
        required=True,
        help="Directory containing MAPS FITS files.",
    )
    parser.add_argument(
        "--index_file",
        type=str,
        required=True,
        help="Text file listing MAPS filenames to process (one per line).",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=64,
        help="Resampled size (pixels). Default: 64.",
    )
    parser.add_argument(
        "--vel_degrade_factor",
        type=float,
        default=1.0,
        help=(
            "Optional factor < 1.0 to degrade velocity resolution before "
            "resampling (not used yet if =1.0)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of all galaxies, even if already done.",
    )
    args = parser.parse_args()

    maps_dir = args.maps_dir
    index_file = args.index_file
    target_size = args.target_size
    force = args.force

    print("============================================================")
    print("PREP SOURCE MAPS")
    print("============================================================")
    print(f"[INFO] MAPS directory   : {maps_dir}")
    print(f"[INFO] Index file       : {index_file}")
    print(f"[INFO] Target size      : {target_size}x{target_size}")
    print(f"[INFO] Vel degrade fact.: {args.vel_degrade_factor}")
    print(f"[INFO] Force reprocess  : {force}")
    print("============================================================")

    # Read index file
    with open(index_file, "r") as f:
        index_names = [line.strip() for line in f if line.strip()]

    # List files in maps_dir
    all_files = [
        fn for fn in os.listdir(maps_dir)
        if fn.lower().endswith(".fits") or fn.lower().endswith(".fits.gz")
    ]
    all_files_set = set(all_files)

    # Filter to intersection of index and existing files
    selected_files = [fn for fn in index_names if fn in all_files_set]

    print(f"[INFO] Found {len(all_files)} MAPS files in {maps_dir}")
    print(f"[INFO] Index file lists {len(index_names)} entries")
    print(f"[INFO] Using {len(selected_files)} files after intersection")
    print("============================================================")

    out_dir = os.path.join("data", "source_tensors")
    preview_dir = os.path.join(out_dir, "previews")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    # Prepare compactness CSV
    compactness_csv = os.path.join(out_dir, "compactness_metrics.csv")

    # Load existing CSV data if it exists (for incremental updates)
    existing_rows = {}  # plateifu -> row dict
    if os.path.exists(compactness_csv) and not force:
        with open(compactness_csv, "r", newline="") as fcsv:
            reader = csv.DictReader(fcsv)
            for row in reader:
                existing_rows[row["plateifu"]] = row
        print(f"[INFO] Loaded {len(existing_rows)} existing entries from CSV")
    elif force:
        print("[INFO] Force mode: ignoring existing CSV data")
    else:
        print("[INFO] No existing CSV found, starting fresh")

    compact_rows = []
    n_processed = 0
    n_skipped = 0

    for fname in selected_files:
        maps_path = os.path.join(maps_dir, fname)

        # Derive plateifu from filename: manga-PLATE-IFU-MAPS...
        base = os.path.basename(fname)
        # expect: manga-PLATE-IFUDESIGN-MAPS...
        parts = base.replace(".fits.gz", "").replace(".fits", "").split("-")
        if len(parts) < 3:
            print(f"[WARN] Unexpected filename format: {base}")
            plateifu = base
        else:
            plate = parts[1]
            ifu = parts[2]
            plateifu = f"{plate}-{ifu}"

        # Check if already processed (tensor exists AND CSV entry exists)
        tensor_path = os.path.join(out_dir, f"source_tensor_{plateifu}.npy")
        preview_path = os.path.join(preview_dir, f"preview_{plateifu}.png")

        if not force:
            tensor_exists = os.path.exists(tensor_path)
            csv_entry_exists = plateifu in existing_rows

            if tensor_exists and csv_entry_exists:
                print(f"[SKIP] {plateifu}: already processed (tensor + CSV entry exist)")
                n_skipped += 1
            continue

        print(f"[INFO] Processing {fname} (plateifu={plateifu})")

        with fits.open(maps_path) as hdul:
            # Get flux and velocity + masks
            gflux = hdul["EMLINE_GFLUX"].data  # shape (n_lines, ny, nx)
            gvel = hdul["EMLINE_GVEL"].data
            gflux_mask = hdul["EMLINE_GFLUX_MASK"].data
            gvel_mask = hdul["EMLINE_GVEL_MASK"].data

            # Use Hα gas velocity (EMLINE_GVEL index 24) for kinematic maps
            # We use gas velocity, not stellar velocity, as it shows cleaner rotation in disk galaxies
            ha_flux = gflux[HA_INDEX, :, :]
            ha_vel = gvel[HA_INDEX, :, :]

            # Valid pixels: mask == 0
            valid_flux_mask = (gflux_mask[HA_INDEX, :, :] == 0)
            valid_vel_mask = (gvel_mask[HA_INDEX, :, :] == 0)

            # We want velocity only where both flux and velocity are valid
            valid_mask = valid_flux_mask & valid_vel_mask

            # Normalize flux and velocity
            flux_norm, flux_stats = robust_flux_normalization(ha_flux, valid_flux_mask)
            vel_norm, vel_stats = robust_velocity_normalization(ha_vel, valid_mask)

            # Resample to target_size x target_size
            flux_res = resample_map(flux_norm, target_size)
            vel_res = resample_map(vel_norm, target_size)

            # Stack into (2, H, W)
            tensor = np.stack([flux_res, vel_res], axis=0)

        # Save tensor
        np.save(tensor_path, tensor)
        print(f"[OK] Saved tensor: {tensor_path}")

        # Save preview
        save_preview_png(flux_res, vel_res, preview_path, title=plateifu)
        print(f"[OK] Saved preview: {preview_path}")

        # Compute compactness on the resampled (64x64) flux map
        compact = compute_compactness_metrics(flux_res)
        compact_row = {
            "plateifu": plateifu,
            "maps_filename": fname,
            "total_flux_norm_sum": compact["total_flux"],
            "flux_frac_central_5x5": compact["flux_frac_central_5x5"],
            "r_half_pix": compact["r_half_pix"],
            "r80_pix": compact["r80_pix"],
            # Optionally keep normalization parameters for future reference
            "flux_p5_raw": flux_stats["p5"],
            "flux_p995_raw": flux_stats["p995"],
            "flux_total_raw": flux_stats["total_flux_raw"],
            "vel_median_raw": vel_stats["median"],
            "vel_scale_raw": vel_stats["scale"],
            "vel_p95abs_raw": vel_stats["p95abs"],
        }
        compact_rows.append(compact_row)

        print(
            f"[METRICS] {plateifu}: "
            f"total_flux={compact['total_flux']:.3f}, "
            f"frac_central_5x5={compact['flux_frac_central_5x5']:.3f}, "
            f"r_half_pix={compact['r_half_pix']:.2f}, "
            f"r80_pix={compact['r80_pix']:.2f}"
        )

        n_processed += 1
        print("------------------------------------------------------------")

    # Merge new rows with existing rows
    # New rows take precedence (overwrite existing if reprocessed)
    for row in compact_rows:
        existing_rows[row["plateifu"]] = row

    # Write merged compactness CSV
    if existing_rows:
        # Get fieldnames from a sample row
        sample_row = next(iter(existing_rows.values()))
        fieldnames = list(sample_row.keys())

        # Sort by plateifu for consistent ordering
        sorted_plateifus = sorted(existing_rows.keys())

        with open(compactness_csv, "w", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()
            for plateifu in sorted_plateifus:
                writer.writerow(existing_rows[plateifu])
        print(f"[OK] Wrote compactness metrics to: {compactness_csv}")
        print(f"[INFO] Total entries in CSV: {len(existing_rows)}")

    print("============================================================")
    print("SUMMARY")
    print("============================================================")
    print(f"  MAPS files in directory : {len(all_files)}")
    print(f"  MAPS files after filter : {len(selected_files)}")
    print(f"  Already processed (skip): {n_skipped}")
    print(f"  Newly processed         : {n_processed}")
    print(f"  Total in CSV            : {len(existing_rows)}")
    print(f"  Output directory        : {out_dir}")
    print("============================================================")


if __name__ == "__main__":
    main()

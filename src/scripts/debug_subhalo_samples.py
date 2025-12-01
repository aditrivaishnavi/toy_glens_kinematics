#!/usr/bin/env python
"""
debug_subhalo_samples.py

Visual sanity check for smooth vs subhalo-perturbed lensing, with and without
PSF + noise.

Generates a 3x2 panel:

Row 1: Smooth SIS lens (flux, velocity)
Row 2: SIS + subhalo, no PSF/noise (flux, velocity)
Row 3: SIS + subhalo, with PSF + noise (flux, velocity)

This script uses the centralized lensing utilities from src/glens/lensing_utils.py
to ensure consistency with the training pipeline in subhalo_cnn.py.

Assumptions:
- Source tensors live in:    data/source_tensors/source_tensor_{plateifu}.npy
- Each tensor has shape:     (2, H, W)
  channel 0 = normalized flux in [0, 1]
  channel 1 = normalized velocity in roughly [-1, 1]
- Coordinates are mapped to a normalized plane: [-1, 1] x [-1, 1]

TODOs for future milestones:
- Replace SIS + SIS-subhalo with proper lenstronomy macro + NFW subhalo.
- Match PSF and noise parameters exactly to those used in production subhalo
  simulations and training.
- Optionally add smooth+PSF+noise panel for more direct comparison.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.glens.lensing_utils import (
    LensSimConfig,
    load_source_tensor,
    render_sis_lens,
    render_sis_plus_subhalo,
    apply_psf_and_noise,
)


def plot_debug_triplet(smooth,
                       sub_no_psf,
                       sub_with_psf,
                       out_path: str,
                       plateifu: str,
                       config: LensSimConfig):
    """
    Create a 3x2 diagnostic plot:
    Row 1: smooth flux, smooth vel
    Row 2: subhalo (no PSF) flux, vel
    Row 3: subhalo (+PSF+noise) flux, vel
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    fig.suptitle(
        f"Debug subhalo samples for {plateifu}\n"
        f"SIS theta_E={config.theta_E_main:.2f}, "
        f"subhalo factor={config.theta_E_sub_factor:.2f}",
        fontsize=10
    )

    def imshow_panel(ax, img, title, cmap, vmin=None, vmax=None):
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=[-1, 1, -1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Choose common dynamic ranges for better comparison
    # Flux channel
    all_flux = np.concatenate([
        smooth[0].ravel(),
        sub_no_psf[0].ravel(),
        sub_with_psf[0].ravel()
    ])
    fmin, fmax = np.percentile(all_flux, [1, 99])

    # Velocity channel
    all_vel = np.concatenate([
        smooth[1].ravel(),
        sub_no_psf[1].ravel(),
        sub_with_psf[1].ravel()
    ])
    vmin, vmax = np.percentile(all_vel, [1, 99])

    # Row 1: smooth
    imshow_panel(axes[0, 0], smooth[0], "Smooth flux (SIS only)", "magma",
                 vmin=fmin, vmax=fmax)
    imshow_panel(axes[0, 1], smooth[1], "Smooth velocity (SIS only)", "RdBu_r",
                 vmin=vmin, vmax=vmax)

    # Row 2: subhalo, no PSF/noise
    imshow_panel(axes[1, 0], sub_no_psf[0], "Subhalo flux (no PSF/noise)", "magma",
                 vmin=fmin, vmax=fmax)
    imshow_panel(axes[1, 1], sub_no_psf[1], "Subhalo velocity (no PSF/noise)", "RdBu_r",
                 vmin=vmin, vmax=vmax)

    # Row 3: subhalo, with PSF+noise
    imshow_panel(axes[2, 0], sub_with_psf[0],
                 "Subhalo flux (with PSF+noise)", "magma",
                 vmin=fmin, vmax=fmax)
    imshow_panel(axes[2, 1], sub_with_psf[1],
                 "Subhalo velocity (with PSF+noise)", "RdBu_r",
                 vmin=vmin, vmax=vmax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved debug figure to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug smooth vs subhalo lensing samples (visual inspection)."
    )
    parser.add_argument(
        "--tensor_dir",
        type=str,
        default="data/source_tensors",
        help="Directory containing source_tensor_{plateifu}.npy files.",
    )
    parser.add_argument(
        "--plateifu",
        type=str,
        default="8993-12705",
        help="MaNGA plate-IFU identifier to debug (must have tensor file).",
    )
    parser.add_argument(
        "--theta_E_main",
        type=float,
        default=0.5,
        help="Einstein radius of main SIS lens (normalized units).",
    )
    parser.add_argument(
        "--theta_E_sub_factor",
        type=float,
        default=0.1,
        help="Subhalo Einstein radius factor: theta_E_sub = factor * theta_E_main.",
    )
    parser.add_argument(
        "--psf_sigma",
        type=float,
        default=1.0,
        help="Gaussian PSF sigma in pixels for subhalo-with-PSF case.",
    )
    parser.add_argument(
        "--flux_noise_sigma",
        type=float,
        default=0.08,
        help="Flux noise sigma for subhalo-with-PSF case.",
    )
    parser.add_argument(
        "--vel_noise_sigma",
        type=float,
        default=0.03,
        help="Velocity noise sigma for subhalo-with-PSF case.",
    )
    parser.add_argument(
        "--flux_mask_threshold",
        type=float,
        default=0.1,
        help="Flux threshold for velocity masking.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/debug_subhalo/debug_subhalo_triplet.png",
        help="Path to save the 3x2 diagnostic figure.",
    )
    args = parser.parse_args()

    print("============================================================")
    print("DEBUG SUBHALO SAMPLES")
    print("============================================================")
    print(f"[INFO] tensor_dir          : {args.tensor_dir}")
    print(f"[INFO] plateifu            : {args.plateifu}")
    print(f"[INFO] theta_E_main        : {args.theta_E_main}")
    print(f"[INFO] theta_E_sub_factor  : {args.theta_E_sub_factor}")
    print(f"[INFO] psf_sigma           : {args.psf_sigma}")
    print(f"[INFO] flux_noise_sigma    : {args.flux_noise_sigma}")
    print(f"[INFO] vel_noise_sigma     : {args.vel_noise_sigma}")
    print(f"[INFO] flux_mask_threshold : {args.flux_mask_threshold}")
    print(f"[INFO] output              : {args.output}")
    print("============================================================")

    # Build config from CLI args
    config = LensSimConfig(
        theta_E_main=args.theta_E_main,
        theta_E_sub_factor=args.theta_E_sub_factor,
        psf_sigma=args.psf_sigma,
        flux_noise_sigma=args.flux_noise_sigma,
        vel_noise_sigma=args.vel_noise_sigma,
        flux_mask_threshold=args.flux_mask_threshold,
    )

    # Load source tensor using centralized utility
    source_tensor = load_source_tensor(args.plateifu, args.tensor_dir)

    print(f"[INFO] Loaded source tensor shape: {source_tensor.shape}")
    print(f"[INFO] Source flux range: min={source_tensor[0].min():.3f}, "
          f"max={source_tensor[0].max():.3f}")
    print(f"[INFO] Source vel  range: min={source_tensor[1].min():.3f}, "
          f"max={source_tensor[1].max():.3f}")

    rng = np.random.default_rng(42)

    # Case 1: smooth SIS, no PSF/noise
    # Uses centralized render_sis_lens from lensing_utils
    smooth = render_sis_lens(source_tensor, config=config)

    # Case 2: SIS + subhalo, no PSF/noise
    # Uses centralized render_sis_plus_subhalo from lensing_utils
    # Fix subhalo at phi=0 (positive x-axis) for reproducible debug output
    sub_no_psf = render_sis_plus_subhalo(
        source_tensor,
        config=config,
        on_arc=True,
        subhalo_phi=0.0,  # Fixed position for debugging
        rng=rng,
    )

    # Case 3: SIS + subhalo, with PSF + noise
    # Uses centralized apply_psf_and_noise from lensing_utils
    sub_clean = render_sis_plus_subhalo(
        source_tensor,
        config=config,
        on_arc=True,
        subhalo_phi=0.0,  # Same position as case 2
        rng=rng,
    )
    sub_with_psf = apply_psf_and_noise(sub_clean, config=config, rng=rng)

    # Some quick numeric diagnostics
    def summary(name, tensor):
        f, v = tensor[0], tensor[1]
        print(f"--- {name} ---")
        print(f"  flux: min={f.min():.3f}, max={f.max():.3f}, mean={f.mean():.3f}")
        print(f"  vel : min={v.min():.3f}, max={v.max():.3f}, std={v.std():.3f}")
        print(f"  nonzero flux pixels: {(f > 0).sum()} / {f.size}")

    summary("Smooth (SIS only)", smooth)
    summary("Subhalo (no PSF/noise)", sub_no_psf)
    summary("Subhalo (with PSF+noise)", sub_with_psf)

    # Plot and save the 3x2 panel
    plot_debug_triplet(
        smooth,
        sub_no_psf,
        sub_with_psf,
        out_path=args.output,
        plateifu=args.plateifu,
        config=config,
    )

    print("============================================================")
    print("DEBUG COMPLETE")
    print("============================================================")


if __name__ == "__main__":
    main()

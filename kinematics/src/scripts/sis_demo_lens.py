#!/usr/bin/env python3
"""
sis_demo_lens.py

Simple toy strong-lensing demo for 2-channel (flux, velocity) maps.

This script uses the centralized lensing utilities from src/glens/lensing_utils.py
to ensure consistency with the training pipeline in subhalo_cnn.py.

Goals:
- Load a preprocessed MaNGA-based source tensor from:
    data/source_tensors/source_tensor_<plateifu>.npy
  where tensor shape is (2, H, W), channel 0 = normalized flux, channel 1 = normalized velocity.
- Define a Singular Isothermal Sphere (SIS) lens model in normalized coordinates.
- Apply the same lens mapping to both flux and velocity channels.
- Enforce flux-based masking on the lensed velocity:
    velocity is only trusted where there is light.
- Provide a CLI to:
    - choose a plateifu,
    - lens it with a chosen Einstein radius theta_E,
    - save a diagnostic plot.

This is a TOY model:
- It ignores cosmology, redshift, proper PSF, etc.
- It is meant to validate coordinate handling and 2-channel consistency,
  not to produce publication-ready lens models.

NOTE: This toy lens demo produces large morphological differences (blob vs ring).
It is for end-to-end plumbing, not realistic subhalo detection yet.
The task is intentionally "large-signal" to verify the pipeline works before
moving to subtle perturbations.
"""

import os
import sys
import argparse
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
)


# ---------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------

# Where prep_source_maps.py wrote the tensors
DEFAULT_TENSOR_DIR = "data/source_tensors"

# A canonical "hero" galaxy to use by default (good rotation, good coverage)
DEFAULT_CANONICAL_PLATEIFU = "8652-12703"

# Default Einstein radius in normalized [-1, 1] coordinates
DEFAULT_THETA_E = 0.7

# Flux threshold (in normalized [0, 1] flux units) below which we treat pixels as "sky"
# TODO: expose flux_mask_threshold as a dataset hyperparameter when we move to proper training.
#       Experiment with values like 0.05, 0.1, 0.2 to see effect on velocity mask coverage.
DEFAULT_FLUX_MASK_THRESHOLD = 0.1


# ---------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------

def plot_source_and_lensed(source_tensor: np.ndarray,
                           lensed_tensor: np.ndarray,
                           plateifu: str,
                           theta_E: float,
                           flux_mask_threshold: float,
                           output_path: str) -> None:
    """
    Plot source vs lensed flux and velocity in a 2x2 grid.

    Parameters
    ----------
    source_tensor : np.ndarray
        Original (2, H, W) tensor.
    lensed_tensor : np.ndarray
        Lensed (2, H, W) tensor.
    plateifu : str
        Plate-IFU identifier for title/labels.
    theta_E : float
        Einstein radius used (for title).
    flux_mask_threshold : float
        Threshold used for velocity masking.
    output_path : str
        Path to save the PNG figure.
    """
    src_flux = source_tensor[0]
    src_vel = source_tensor[1]
    img_flux = lensed_tensor[0]
    img_vel = lensed_tensor[1]

    # For plotting velocity, set masked pixels to NaN so they show as blank
    mask = img_flux >= flux_mask_threshold
    img_vel_plot = np.where(mask, img_vel, np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    (ax1, ax2), (ax3, ax4) = axes

    im1 = ax1.imshow(src_flux, origin="lower", cmap="viridis")
    ax1.set_title(f"Source Flux ({plateifu})")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(src_vel, origin="lower", cmap="RdBu_r")
    ax2.set_title("Source Velocity (normalized)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    im3 = ax3.imshow(img_flux, origin="lower", cmap="viridis")
    ax3.set_title(f"Lensed Flux (θ_E={theta_E:.2f})")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    im4 = ax4.imshow(img_vel_plot, origin="lower", cmap="RdBu_r")
    ax4.set_title("Lensed Velocity (flux-masked)")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Toy SIS Lensing: {plateifu}, θ_E={theta_E:.2f}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply a toy SIS lens to a 2-channel MaNGA-based source tensor."
    )
    parser.add_argument(
        "--tensor_dir",
        type=str,
        default=DEFAULT_TENSOR_DIR,
        help=f"Directory with source_tensor_<plateifu>.npy (default: {DEFAULT_TENSOR_DIR})",
    )
    parser.add_argument(
        "--plateifu",
        type=str,
        default=DEFAULT_CANONICAL_PLATEIFU,
        help=f"Plate-IFU to load (default: {DEFAULT_CANONICAL_PLATEIFU})",
    )
    parser.add_argument(
        "--theta_E",
        type=float,
        default=DEFAULT_THETA_E,
        help=f"Einstein radius in normalized units (default: {DEFAULT_THETA_E})",
    )
    parser.add_argument(
        "--flux_mask_threshold",
        type=float,
        default=DEFAULT_FLUX_MASK_THRESHOLD,
        help=f"Flux threshold for masking velocity (default: {DEFAULT_FLUX_MASK_THRESHOLD})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/lensing_demo/lensed_example.png",
        help="Path to save a 2x2 diagnostic plot (default: data/lensing_demo/lensed_example.png)",
    )

    args = parser.parse_args()

    print("============================================================")
    print("TOY LENS DEMO")
    print("============================================================")
    print(f"[INFO] tensor_dir           : {args.tensor_dir}")
    print(f"[INFO] plateifu             : {args.plateifu}")
    print(f"[INFO] theta_E              : {args.theta_E}")
    print(f"[INFO] flux_mask_threshold  : {args.flux_mask_threshold}")
    print(f"[INFO] output plot          : {args.output}")
    print("[INFO] Using centralized lensing from src/glens/lensing_utils.py")
    print("============================================================")

    # Load the source tensor using centralized utility
    source_tensor = load_source_tensor(args.plateifu, args.tensor_dir)
    print(f"[INFO] Loaded source tensor shape: {source_tensor.shape}")
    print(f"[INFO] Flux channel: min={source_tensor[0].min():.3f}, max={source_tensor[0].max():.3f}")
    print(f"[INFO] Vel  channel: min={source_tensor[1].min():.3f}, max={source_tensor[1].max():.3f}")

    # Build config
    config = LensSimConfig(
        theta_E_main=args.theta_E,
        flux_mask_threshold=args.flux_mask_threshold,
    )

    # Apply the toy lens using centralized utility
    lensed_tensor = render_sis_lens(source_tensor, config=config)

    # Compute mask for stats
    mask = lensed_tensor[0] >= args.flux_mask_threshold

    print(f"[INFO] Lensed flux : min={lensed_tensor[0].min():.3f}, max={lensed_tensor[0].max():.3f}")
    print(f"[INFO] Lensed vel  : min={lensed_tensor[1].min():.3f}, max={lensed_tensor[1].max():.3f}")
    print(f"[INFO] Valid vel pixels (flux >= threshold): {mask.sum()} / {mask.size} "
          f"({mask.mean()*100:.1f}%)")

    # Plot source vs lensed
    plot_source_and_lensed(
        source_tensor,
        lensed_tensor,
        plateifu=args.plateifu,
        theta_E=args.theta_E,
        flux_mask_threshold=args.flux_mask_threshold,
        output_path=args.output,
    )

    print(f"[OK] Saved diagnostic plot to: {args.output}")

    # TODO: save (source_tensor, lensed_tensor) pairs for use by toy_cnn.py.
    #       For training, we need arrays on disk, e.g.:
    #         data/lensing_demo/lensed_<plateifu>.npy  with shape (2, 64, 64)
    #       Consider a batch mode that generates many (source, lensed) pairs
    #       with varying theta_E for data augmentation.

    print("============================================================")


if __name__ == "__main__":
    main()

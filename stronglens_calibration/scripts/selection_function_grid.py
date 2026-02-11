#!/usr/bin/env python3
"""
Selection Function: Injection-Recovery Grid.

Measures the detection completeness C(theta_E, PSF, depth) by injecting
synthetic arc proxies into real DR10 negative cutouts and scoring with a
frozen trained model.

Grid axes (MNRAS_RAW_NOTES.md Section 9.3):
  - theta_E (Einstein radius): 0.5 to 3.0 arcsec
  - PSF FWHM: 0.9 to 1.8 arcsec (r-band)
  - 5-sigma depth: 22.5 to 24.5 mag (r-band)
  - Source magnitude: 22.5 to 25.5 mag (unlensed r-band)

For each grid cell:
  1. Select host galaxies (negatives) matching the PSF/depth bin
  2. Inject a synthetic arc with the specified theta_E and source flux
  3. Apply the same preprocessing as training (raw_robust, 101x101)
  4. Score with the frozen model
  5. Compute detection fraction with Bayesian binomial confidence intervals

The arc renderer is a MINIMAL PROXY. For publication-quality results,
replace with the calibrated Phase 4c injection pipeline once validated.

Supports any architecture via build_model() factory (auto-detected from
checkpoint).

Usage:
    cd /lambda/nfs/.../code
    export PYTHONPATH=.

    # Run with EfficientNetV2-S
    python scripts/selection_function_grid.py \\
        --checkpoint checkpoints/paperIV_efficientnet_v2_s/best.pt \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --host-split val \\
        --out-csv results/selection_function_efficientnet.csv \\
        --out-json results/selection_function_efficientnet_meta.json

    # Run with ResNet-18 for cross-model comparison
    python scripts/selection_function_grid.py \\
        --checkpoint checkpoints/paperIV_resnet18/best.pt \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --host-split val \\
        --out-csv results/selection_function_resnet18.csv

Author: stronglens_calibration project
Date: 2026-02-11
References:
  - MNRAS_RAW_NOTES.md Section 9.3
  - LLM conversation: "PART H: Selection function (injection-recovery)"
  - Paper IV (Inchausti et al. 2025): ensemble completeness comparison
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# dhs imports
from dhs.model import build_model
from dhs.data import load_cutout_from_file
from dhs.preprocess import preprocess_stack
from dhs.constants import CUTOUT_SIZE, STAMP_SIZE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CUTOUT_PATH_COL = "cutout_path"
LABEL_COL = "label"
SPLIT_COL = "split"

AB_ZEROPOINT_MAG = 22.5  # DESI Legacy Survey AB zeropoint
PIXEL_SCALE = 0.262      # arcsec/pixel


# ---------------------------------------------------------------------------
# Photometric utilities
# ---------------------------------------------------------------------------
def mag_to_nmgy(mag: float) -> float:
    """Convert AB magnitude to nanomaggies."""
    return 10.0 ** ((AB_ZEROPOINT_MAG - mag) / 2.5)


def psfdepth_to_mag5sig(psfdepth: float) -> float:
    """Convert psfdepth (inverse variance, nanomaggies^-2) to 5-sigma AB mag.

    DR10 convention: psfdepth_r = 1/sigma^2 for a point source.
    5-sigma depth = 22.5 - 2.5 * log10(5 / sqrt(psfdepth)).
    """
    if psfdepth <= 0:
        return float("nan")
    return float(AB_ZEROPOINT_MAG - 2.5 * np.log10(5.0 / np.sqrt(psfdepth)))


# ---------------------------------------------------------------------------
# Arc injection engine (MINIMAL PROXY)
# ---------------------------------------------------------------------------
def gaussian_blur_2d(img: np.ndarray, sigma: float) -> np.ndarray:
    """Fast 2D Gaussian blur via separable 1D convolutions."""
    if sigma <= 0:
        return img
    r = int(max(1, math.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    # Row-then-column convolution
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, img)
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, out)
    return out.astype(np.float32)


def render_arc_proxy(
    size: int,
    theta_e_arcsec: float,
    pixscale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render a simple arc-like feature at a given Einstein radius.

    This is a MINIMAL PROXY: a thin radial Gaussian ring modulated by
    angular segments, with sub-structure noise. It is sufficient for
    scaffolding the selection function pipeline but should be replaced
    with calibrated Phase 4c injections for publication.

    Returns a normalized 2D template (sums to ~1) of shape (size, size).
    """
    h = w = size
    cy = cx = size // 2
    yy, xx = np.mgrid[0:h, 0:w]
    dy = (yy - cy).astype(np.float32)
    dx = (xx - cx).astype(np.float32)
    rr = np.sqrt(dx * dx + dy * dy) + 1e-6
    ang = np.arctan2(dy, dx)

    theta_pix = theta_e_arcsec / pixscale

    # Radial profile: Gaussian ring at theta_E
    rad_sigma = rng.uniform(0.8, 1.8)  # width in pixels
    radial = np.exp(-0.5 * ((rr - theta_pix) / rad_sigma) ** 2).astype(np.float32)

    # Angular segments (1-3 arcs with varying widths)
    n_seg = int(rng.integers(1, 4))
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_seg):
        phi0 = rng.uniform(-np.pi, np.pi)
        width = rng.uniform(0.25, 0.55)  # angular width in radians
        d = (ang - phi0 + np.pi) % (2 * np.pi) - np.pi
        seg = np.exp(-0.5 * (d / width) ** 2).astype(np.float32)
        mask = np.maximum(mask, seg)

    # Sub-structure (clumpy noise in the arc)
    clumps = rng.normal(0, 1, size=(h, w)).astype(np.float32)
    clumps = gaussian_blur_2d(clumps, rng.uniform(0.6, 1.2))
    clumps = np.clip((clumps - np.percentile(clumps, 60)) / (np.std(clumps) + 1e-6), 0, None)

    arc = radial * mask * (0.6 + 0.8 * clumps)
    total = arc.sum()
    if total > 0:
        arc /= total
    return arc


def inject_arc_into_cutout(
    host_hwc: np.ndarray,
    theta_e_arcsec: float,
    source_mag_r: float,
    mu: float,
    psf_fwhm_arcsec: float,
    pixscale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inject a synthetic arc into a host galaxy cutout.

    Args:
        host_hwc: (H, W, 3) host cutout in HWC format (nanomaggies)
        theta_e_arcsec: Einstein radius in arcsec
        source_mag_r: unlensed r-band source AB magnitude
        mu: magnification factor
        psf_fwhm_arcsec: PSF FWHM for blurring
        pixscale: arcsec/pixel
        rng: numpy random generator

    Returns:
        (H, W, 3) injected cutout in HWC format
    """
    h, w, _ = host_hwc.shape

    # Total arc flux in r-band (nanomaggies)
    arc_flux_r = mag_to_nmgy(source_mag_r) * mu

    # Approximate color: blueish source (typical lensed galaxy)
    g_scale = rng.uniform(1.1, 1.6)   # g brighter than r
    z_scale = rng.uniform(0.6, 0.95)  # z dimmer than r

    # Render arc template
    arc_template = render_arc_proxy(h, theta_e_arcsec, pixscale, rng)

    # Scale to physical flux in each band
    arc_g = arc_template * arc_flux_r * g_scale
    arc_r = arc_template * arc_flux_r
    arc_z = arc_template * arc_flux_r * z_scale
    arc_grz = np.stack([arc_g, arc_r, arc_z], axis=-1).astype(np.float32)

    # Apply PSF convolution (approximate: blur the arc only)
    psf_sigma_pix = psf_fwhm_arcsec / (2.355 * pixscale)
    for band in range(3):
        arc_grz[:, :, band] = gaussian_blur_2d(arc_grz[:, :, band], max(0.0, psf_sigma_pix - 0.5))

    # Add arc to host
    injected = host_hwc.astype(np.float32) + arc_grz

    return injected


# ---------------------------------------------------------------------------
# Model loading and scoring
# ---------------------------------------------------------------------------
def load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[nn.Module, str, int]:
    """Load model, return (model, arch_name, epoch)."""
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    train_cfg = ckpt.get("train", {})
    arch = train_cfg.get("arch", "resnet18")
    pretrained = train_cfg.get("pretrained", False)
    epoch = ckpt.get("epoch", -1)

    model = build_model(arch, in_ch=3, pretrained=pretrained).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    return model, arch, epoch


@torch.no_grad()
def score_batch(model: nn.Module, batch: np.ndarray, device: torch.device) -> np.ndarray:
    """Score a batch of preprocessed images. Returns sigmoid probabilities."""
    x = torch.from_numpy(batch).float().to(device)
    logits = model(x).squeeze(1).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))


# ---------------------------------------------------------------------------
# Bayesian binomial confidence interval
# ---------------------------------------------------------------------------
def bayes_binom_ci(k: int, n: int, alpha: float = 0.32) -> Tuple[float, float]:
    """Bayesian binomial 68% CI using Jeffreys prior (Beta(0.5, 0.5)).

    Returns (lo, hi) interval for the detection fraction.
    """
    try:
        from scipy.stats import beta as beta_dist
        a = k + 0.5
        b = (n - k) + 0.5
        return (
            float(beta_dist.ppf(alpha / 2, a, b)),
            float(beta_dist.ppf(1 - alpha / 2, a, b)),
        )
    except ImportError:
        # Fallback: normal approximation
        p = k / n if n > 0 else 0
        se = np.sqrt(p * (1 - p) / max(n, 1))
        z = 1.0  # ~68% CI
        return (max(0, p - z * se), min(1, p + z * se))


# ---------------------------------------------------------------------------
# Grid bin assignment
# ---------------------------------------------------------------------------
def nearest_bin(val: float, grid: List[float]) -> float:
    """Snap value to nearest grid bin center."""
    arr = np.asarray(grid)
    return float(arr[np.argmin(np.abs(arr - val))])


# ---------------------------------------------------------------------------
# Main grid runner
# ---------------------------------------------------------------------------
def run_selection_function(
    checkpoint_path: str,
    manifest_path: str,
    host_split: str = "val",
    host_max: int = 20000,
    threshold: float = 0.5,
    injections_per_cell: int = 200,
    # Grid parameters
    theta_e_min: float = 0.5,
    theta_e_max: float = 3.0,
    theta_e_step: float = 0.25,
    psf_min: float = 0.9,
    psf_max: float = 1.8,
    psf_step: float = 0.15,
    depth_min: float = 22.5,
    depth_max: float = 24.5,
    depth_step: float = 0.5,
    source_mag_min: float = 22.5,
    source_mag_max: float = 25.5,
    mu_min: float = 5.0,
    mu_max: float = 30.0,
    # Processing
    preprocessing: str = "raw_robust",
    crop: bool = False,
    pixscale: float = PIXEL_SCALE,
    seed: int = 1337,
    device_str: str = "cuda",
    data_root: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the injection-recovery grid. Returns (results_df, metadata)."""

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Path overrides
    eff_manifest = manifest_path
    eff_ckpt = checkpoint_path
    if data_root:
        default_root = "/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration"
        eff_manifest = manifest_path.replace(default_root, data_root.rstrip("/"), 1)
        eff_ckpt = checkpoint_path.replace(default_root, data_root.rstrip("/"), 1)

    # Load model
    print(f"Loading model: {eff_ckpt}")
    model, arch, epoch = load_model_from_checkpoint(eff_ckpt, device)
    print(f"  Architecture: {arch}, Epoch: {epoch}")

    # Load manifest
    print(f"Loading manifest: {eff_manifest}")
    df = pd.read_parquet(eff_manifest)

    # Select negative hosts from the specified split
    hosts = df[(df[SPLIT_COL] == host_split) & (df[LABEL_COL] == 0)].copy()
    if hosts.empty:
        raise ValueError(f"No negative hosts found in split '{host_split}'")
    print(f"  Negative hosts in '{host_split}': {len(hosts):,}")

    # Require PSF and depth columns
    for col in ["psfsize_r", "psfdepth_r"]:
        if col not in hosts.columns:
            raise ValueError(f"Missing column '{col}' in manifest")

    # Subsample hosts if too many
    rng = np.random.default_rng(seed)
    if len(hosts) > host_max:
        hosts = hosts.sample(n=host_max, random_state=seed).reset_index(drop=True)
        print(f"  Subsampled to {len(hosts):,} hosts")

    # Build grid
    theta_es = np.round(np.arange(theta_e_min, theta_e_max + 1e-9, theta_e_step), 4).tolist()
    psf_bins = np.round(np.arange(psf_min, psf_max + 1e-9, psf_step), 4).tolist()
    depth_bins = np.round(np.arange(depth_min, depth_max + 1e-9, depth_step), 4).tolist()

    n_cells = len(theta_es) * len(psf_bins) * len(depth_bins)
    print(f"\nGrid: {len(theta_es)} theta_E x {len(psf_bins)} PSF x {len(depth_bins)} depth = {n_cells} cells")
    print(f"  theta_E: {theta_es}")
    print(f"  PSF FWHM: {psf_bins}")
    print(f"  Depth (5sig mag): {depth_bins}")
    print(f"  Injections/cell: {injections_per_cell}")
    print(f"  Total injections: {n_cells * injections_per_cell:,}")

    # Assign hosts to PSF and depth bins
    hosts["psf_bin"] = hosts["psfsize_r"].apply(lambda v: nearest_bin(float(v), psf_bins))
    hosts["depth_mag"] = hosts["psfdepth_r"].apply(lambda v: psfdepth_to_mag5sig(float(v)))
    hosts["depth_bin"] = hosts["depth_mag"].apply(lambda v: nearest_bin(float(v), depth_bins))

    # Group hosts by (psf_bin, depth_bin) for fast lookup
    host_groups: Dict[Tuple[float, float], np.ndarray] = {}
    for (pb, db), g in hosts.groupby(["psf_bin", "depth_bin"]):
        host_groups[(float(pb), float(db))] = g[CUTOUT_PATH_COL].to_numpy(object)

    # Determine input size from preprocessing config
    input_size = CUTOUT_SIZE if not crop else STAMP_SIZE  # 101 for parity, 64 for legacy

    # Run grid
    print(f"\nRunning injection-recovery grid...")
    t0 = time.time()
    rows = []
    cell_idx = 0

    for theta_e in theta_es:
        for pb in psf_bins:
            for db in depth_bins:
                cell_idx += 1
                paths = host_groups.get((pb, db))

                if paths is None or len(paths) == 0:
                    rows.append({
                        "theta_e": theta_e,
                        "psf_fwhm": pb,
                        "depth_5sig": db,
                        "n_injections": 0,
                        "n_detected": 0,
                        "completeness": float("nan"),
                        "ci68_lo": float("nan"),
                        "ci68_hi": float("nan"),
                        "mean_score": float("nan"),
                        "sufficient": False,
                    })
                    continue

                n = injections_per_cell
                sel = rng.choice(paths, size=n, replace=True)

                batch = np.zeros((n, 3, input_size, input_size), dtype=np.float32)
                for i, p in enumerate(sel):
                    try:
                        # Load host cutout (HWC)
                        with np.load(str(p)) as z:
                            host_hwc = z["cutout"].astype(np.float32)

                        # Sample source properties
                        src_mag = rng.uniform(source_mag_min, source_mag_max)
                        mu = rng.uniform(mu_min, mu_max)

                        # Inject arc
                        injected_hwc = inject_arc_into_cutout(
                            host_hwc, theta_e, src_mag, mu, pb, pixscale, rng
                        )

                        # Preprocess: HWC -> CHW, then raw_robust normalization
                        img3 = np.transpose(injected_hwc, (2, 0, 1))
                        proc = preprocess_stack(img3, mode=preprocessing, crop=crop, clip_range=10.0)
                        batch[i] = proc

                    except Exception as e:
                        # On failure, leave as zeros
                        pass

                # Score batch
                scores = score_batch(model, batch, device)
                k = int((scores >= threshold).sum())
                comp = k / n
                lo, hi = bayes_binom_ci(k, n, alpha=0.32)

                rows.append({
                    "theta_e": theta_e,
                    "psf_fwhm": pb,
                    "depth_5sig": db,
                    "n_injections": n,
                    "n_detected": k,
                    "completeness": float(comp),
                    "ci68_lo": lo,
                    "ci68_hi": hi,
                    "mean_score": float(scores.mean()),
                    "sufficient": n >= injections_per_cell,
                })

                if cell_idx % 10 == 0:
                    pct = 100 * cell_idx / n_cells
                    print(f"  Cell {cell_idx}/{n_cells} ({pct:.0f}%) "
                          f"[theta_E={theta_e}, PSF={pb}, depth={db}] "
                          f"C={comp:.3f} [{lo:.3f}, {hi:.3f}]", end="\r")

    dt = time.time() - t0
    print(f"\n  Grid complete: {n_cells} cells, {dt:.1f}s ({dt/max(n_cells,1):.2f}s/cell)")

    results_df = pd.DataFrame(rows)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_path,
        "manifest": manifest_path,
        "arch": arch,
        "epoch": epoch,
        "host_split": host_split,
        "host_max": host_max,
        "threshold": threshold,
        "injections_per_cell": injections_per_cell,
        "grid": {
            "theta_e": theta_es,
            "psf_fwhm": psf_bins,
            "depth_5sig": depth_bins,
        },
        "source_mag_range": [source_mag_min, source_mag_max],
        "mu_range": [mu_min, mu_max],
        "preprocessing": preprocessing,
        "crop": crop,
        "pixscale": pixscale,
        "seed": seed,
        "n_cells": n_cells,
        "n_sufficient_cells": int(results_df["sufficient"].sum()),
        "n_empty_cells": int((results_df["n_injections"] == 0).sum()),
        "mean_completeness": float(results_df["completeness"].dropna().mean()),
        "notes": [
            "Arc renderer is a MINIMAL PROXY (Gaussian ring + angular segments).",
            "Replace with calibrated Phase 4c injector for publication-quality realism.",
            "Preprocessing: raw_robust outer-annulus median/MAD, clip [-10, 10].",
            "Scores are sigmoid(logit) from the frozen model.",
            "CIs are Bayesian binomial (Jeffreys prior, Beta(0.5, 0.5)).",
        ],
    }

    return results_df, metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Selection Function: Injection-Recovery Grid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-csv", default="selection_function.csv")
    ap.add_argument("--out-json", default="selection_function_meta.json")
    ap.add_argument("--host-split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--host-max", type=int, default=20000)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--injections-per-cell", type=int, default=200)
    # Grid ranges
    ap.add_argument("--theta-e-min", type=float, default=0.5)
    ap.add_argument("--theta-e-max", type=float, default=3.0)
    ap.add_argument("--theta-e-step", type=float, default=0.25)
    ap.add_argument("--psf-min", type=float, default=0.9)
    ap.add_argument("--psf-max", type=float, default=1.8)
    ap.add_argument("--psf-step", type=float, default=0.15)
    ap.add_argument("--depth-min", type=float, default=22.5)
    ap.add_argument("--depth-max", type=float, default=24.5)
    ap.add_argument("--depth-step", type=float, default=0.5)
    # Source parameters
    ap.add_argument("--source-mag-min", type=float, default=22.5)
    ap.add_argument("--source-mag-max", type=float, default=25.5)
    ap.add_argument("--mu-min", type=float, default=5.0)
    ap.add_argument("--mu-max", type=float, default=30.0)
    # Processing
    ap.add_argument("--preprocessing", default="raw_robust")
    ap.add_argument("--crop", action="store_true", default=False)
    ap.add_argument("--pixscale", type=float, default=PIXEL_SCALE)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    results_df, metadata = run_selection_function(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        host_split=args.host_split,
        host_max=args.host_max,
        threshold=args.threshold,
        injections_per_cell=args.injections_per_cell,
        theta_e_min=args.theta_e_min,
        theta_e_max=args.theta_e_max,
        theta_e_step=args.theta_e_step,
        psf_min=args.psf_min,
        psf_max=args.psf_max,
        psf_step=args.psf_step,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        depth_step=args.depth_step,
        source_mag_min=args.source_mag_min,
        source_mag_max=args.source_mag_max,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
        preprocessing=args.preprocessing,
        crop=args.crop,
        pixscale=args.pixscale,
        seed=args.seed,
        device_str=args.device,
        data_root=args.data_root,
    )

    # Save results
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.out_csv, index=False)
    print(f"\nResults saved to: {args.out_csv}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to: {args.out_json}")

    # Print summary
    valid = results_df[results_df["n_injections"] > 0]
    print(f"\n{'='*60}")
    print("SELECTION FUNCTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {metadata['arch']}")
    print(f"  Grid cells: {metadata['n_cells']}")
    print(f"  Sufficient cells: {metadata['n_sufficient_cells']}")
    print(f"  Empty cells: {metadata['n_empty_cells']}")
    print(f"  Mean completeness: {metadata['mean_completeness']:.3f}")
    if len(valid) > 0:
        print(f"\n  Completeness by theta_E:")
        for te in sorted(valid["theta_e"].unique()):
            mask = valid["theta_e"] == te
            c = valid.loc[mask, "completeness"].mean()
            print(f"    theta_E={te:.2f}\": C={c:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

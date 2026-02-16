#!/usr/bin/env python3
"""
Quick visual test: generate a handful of injections and render them
side-by-side with real Tier-A lenses for manual inspection.

Outputs a single PNG grid so the user can eyeball colour, morphology,
clump artefacts, etc. before committing to a full experiment run.

Usage:
    cd /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration
    PYTHONPATH=. python scripts/test_injection_visual.py \
        --manifest manifests/training_parity_70_30_v1.parquet \
        --out test_injection_grid.png \
        --n-samples 10 --seed 99
"""
from __future__ import annotations

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dhs.injection_engine import (
    inject_sis_shear,
    sample_lens_params,
    sample_source_params,
)

AB_ZP = 22.5
PIXEL_SCALE = 0.262


def lupton_rgb(hwc: np.ndarray, crop: int = 51) -> np.ndarray:
    """Render (H,W,3) grz array as RGB using astropy Lupton stretch."""
    from astropy.visualization import make_lupton_rgb

    if crop > 0 and crop < hwc.shape[0]:
        h, w = hwc.shape[:2]
        y0, x0 = (h - crop) // 2, (w - crop) // 2
        hwc = hwc[y0:y0 + crop, x0:x0 + crop, :]
    return make_lupton_rgb(
        hwc[:, :, 2], hwc[:, :, 1], hwc[:, :, 0],
        stretch=0.5, Q=10, minimum=0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="test_injection_grid.png")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--theta-e", type=float, default=1.5)
    ap.add_argument("--target-mag", type=float, default=20.0,
                    help="Target r-band source magnitude for injections")
    ap.add_argument("--clumps-prob", type=float, default=None,
                    help="Override clumps probability (default: use engine default 0.6)")
    ap.add_argument("--crop", type=int, default=51)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load manifest, pick random Tier-A real lenses and random negatives as hosts
    df = pd.read_parquet(args.manifest)
    tier_a = df[(df["label"] == 1) & (df["tier"].str.upper() == "A")].copy()
    negatives = df[(df["label"] == 0) & (df["split"] == "val")].copy()

    real_sample = tier_a.sample(n=args.n_samples, random_state=args.seed)
    host_sample = negatives.sample(n=args.n_samples, random_state=args.seed + 1)

    # Generate injections
    theta_e = args.theta_e
    psf_fwhm = 1.2  # typical Legacy Survey r-band

    n_cols = 4  # real | host_before | injected | injection_only
    fig, axes = plt.subplots(
        args.n_samples, n_cols,
        figsize=(3.5 * n_cols, 3.5 * args.n_samples),
    )
    if args.n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(args.n_samples):
        real_row = real_sample.iloc[i]
        host_row = host_sample.iloc[i]

        # Load real Tier-A lens
        with np.load(real_row["cutout_path"]) as z:
            real_hwc = z["cutout"].astype(np.float64)

        # Load host (negative) for injection
        with np.load(host_row["cutout_path"]) as z:
            host_hwc = z["cutout"].astype(np.float32)

        host_tensor = torch.from_numpy(host_hwc).to(device)

        # Read host PSF and depth from manifest columns
        host_psf = float(host_row.get("psfsize_r", psf_fwhm))
        if np.isnan(host_psf) or host_psf <= 0:
            host_psf = psf_fwhm

        # Sample source and lens params
        clumps_kwargs = {}
        if args.clumps_prob is not None:
            clumps_kwargs["clumps_prob"] = args.clumps_prob

        source = sample_source_params(
            rng, theta_e,
            r_mag_range=(args.target_mag, args.target_mag),
            beta_frac_range=(0.10, 0.40),
            **clumps_kwargs,
        )
        lens = sample_lens_params(rng, theta_e)

        # Inject
        inj_seed = int(rng.integers(0, 2**31))
        result = inject_sis_shear(
            host_tensor, lens, source,
            pixel_scale=PIXEL_SCALE,
            psf_fwhm_r_arcsec=host_psf,
            seed=inj_seed,
            add_poisson_noise=True,
            add_sky_noise=True,
        )

        inj_hwc = result.injected[0].cpu().numpy().transpose(1, 2, 0)
        inj_only_chw = result.injection_only[0].cpu().numpy()
        inj_only_hwc = inj_only_chw.transpose(1, 2, 0)

        # Compute arc colour
        arc_g = inj_only_chw[0].sum()
        arc_r = inj_only_chw[1].sum()
        arc_z = inj_only_chw[2].sum()
        arc_gr = -2.5 * np.log10(arc_g / arc_r) if arc_g > 0 and arc_r > 0 else float("nan")
        arc_rz = -2.5 * np.log10(arc_r / arc_z) if arc_r > 0 and arc_z > 0 else float("nan")

        # Render
        crop = args.crop
        axes[i, 0].imshow(lupton_rgb(real_hwc, crop), origin="lower")
        axes[i, 0].set_title(f"Real Tier-A", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(lupton_rgb(host_hwc, crop), origin="lower")
        axes[i, 1].set_title(f"Host (before)", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(lupton_rgb(inj_hwc, crop), origin="lower")
        axes[i, 2].set_title(
            f"Injected  g-r={arc_gr:.2f} r-z={arc_rz:.2f}\n"
            f"clumps={source.n_clumps} frac={source.clump_frac:.2f}",
            fontsize=8,
        )
        axes[i, 2].axis("off")

        # Injection-only: boost stretch for visibility
        axes[i, 3].imshow(lupton_rgb(inj_only_hwc, crop), origin="lower")
        axes[i, 3].set_title(f"Arc only  mag_r={args.target_mag}", fontsize=9)
        axes[i, 3].axis("off")

    fig.suptitle(
        f"Injection Visual Test (seed={args.seed}, theta_e={theta_e}, "
        f"target_mag={args.target_mag})\n"
        f"Colour priors: g-r=N(1.15,0.30), r-z=N(0.85,0.20)"
        + (f", clumps_prob={args.clumps_prob}" if args.clumps_prob is not None else ""),
        fontsize=11, y=1.005,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

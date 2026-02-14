#!/usr/bin/env python3
"""
Bright Arc Injection Test: Injection completeness at varying source magnitudes.

SCIENTIFIC MOTIVATION
====================
Strong-lens selection-function calibration reveals a persistent sim-to-real gap:
  - Real confirmed lenses: ~73% recall at p>0.5
  - Injection-recovery completeness: 4–8% on the standard grid (source mag 23–26)

This ~70 percentage-point gap could be driven by:
  1. Brightness: Training positives and real lenses are brighter than 23–26 mag.
  2. Morphology: Real lenses may have extended arcs/clumps not well captured by
     the Sérsic injection model.
  3. Environment: Host galaxy structure, blending, color gradients.
  4. Observational systematics: Astrometric scatter, PSF variations.

This script tests hypothesis (1) by injecting sources across a wide magnitude
range (18–26 mag) and measuring detection completeness. If brightness alone
explains the gap, we should see completeness approach real-lens recall in
bright bins (e.g., 18–20 mag). If not, the gap persists and other factors
(morphology, environment) are likely contributors.

Date: 2026-02-13
References:
  - real_lens_scoring.py: real-lens recall baseline
  - selection_function_grid.py: standard injection-recovery grid
  - MNRAS_RAW_NOTES.md Section 9.3: injection-recovery methodology
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dhs.model import build_model
from dhs.preprocess import preprocess_stack
from dhs.injection_engine import (
    SourceParams,
    sample_lens_params,
    sample_source_params,
    inject_sis_shear,
    estimate_sigma_pix_from_psfdepth,
    arc_annulus_snr,
)


PIXEL_SCALE = 0.262  # arcsec/pixel (DESI Legacy Survey)

MAGNITUDE_BINS: List[Tuple[float, float]] = [
    (18.0, 19.0),
    (19.0, 20.0),
    (20.0, 21.0),
    (21.0, 22.0),
    (22.0, 23.0),
    (23.0, 24.0),
    (24.0, 25.0),
    (25.0, 26.0),
]

CUTOUT_PATH_COL = "cutout_path"
LABEL_COL = "label"
SPLIT_COL = "split"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, str, int]:
    """Load model from checkpoint using train config for architecture."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_cfg = ckpt.get("train", {})
    arch = train_cfg.get("arch", "resnet18")
    model = build_model(arch, in_ch=3, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, arch, ckpt.get("epoch", -1)


@torch.no_grad()
def score_one(model: nn.Module, img_chw: np.ndarray, device: torch.device) -> float:
    """Score a single preprocessed image. Returns sigmoid probability."""
    x = torch.from_numpy(img_chw[None]).float().to(device)
    logit = model(x).squeeze().cpu().item()
    return float(1.0 / (1.0 + np.exp(-logit)))


def scale_source_to_magnitude(
    source: SourceParams,
    target_mag: float,
) -> SourceParams:
    """Create new SourceParams with fluxes scaled to achieve target r-band magnitude."""
    target_flux = 10.0 ** ((22.5 - target_mag) / 2.5)
    if source.flux_nmgy_r <= 0:
        scale = 1.0
    else:
        scale = target_flux / source.flux_nmgy_r

    fields = {f.name: getattr(source, f.name) for f in dataclasses.fields(source)}
    fields["flux_nmgy_r"] = source.flux_nmgy_r * scale
    fields["flux_nmgy_g"] = source.flux_nmgy_g * scale
    fields["flux_nmgy_z"] = source.flux_nmgy_z * scale
    return SourceParams(**fields)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_bright_arc_test(
    checkpoint_path: str,
    manifest_path: str,
    host_split: str = "val",
    n_hosts: int = 200,
    theta_e: float = 1.5,
    out_dir: str = "results/bright_arc_injection",
    device_str: str = "cuda",
    seed: int = 42,
) -> Dict[str, Any]:
    """Run bright arc injection test across magnitude bins."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    # Load model
    print("Loading model...")
    model, arch, epoch = load_model(checkpoint_path, device)
    print(f"  Architecture: {arch}, Epoch: {epoch}")

    # Load manifest
    print(f"\nLoading manifest: {manifest_path}")
    df = pd.read_parquet(manifest_path)
    neg = df[(df[SPLIT_COL] == host_split) & (df[LABEL_COL] == 0)].copy()
    neg = neg.dropna(subset=["psfsize_r", "psfdepth_r"])
    if neg.empty:
        raise ValueError(
            f"No val negatives with valid psfsize_r and psfdepth_r in manifest"
        )
    print(f"  Val negatives (valid PSF/depth): {len(neg)}")

    # Sample hosts
    n_sample = min(n_hosts, len(neg))
    hosts = neg.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    print(f"  Sampled {n_sample} host galaxies")

    # Per-bin results
    results_by_bin: Dict[str, Dict[str, Any]] = {}

    for mag_lo, mag_hi in MAGNITUDE_BINS:
        bin_key = f"{mag_lo:.0f}-{mag_hi:.0f}"
        print(f"\n--- Magnitude bin {bin_key} ---")

        scores = []
        arc_snrs = []

        for i, (_, host_row) in enumerate(hosts.iterrows()):
            try:
                with np.load(str(host_row[CUTOUT_PATH_COL])) as z:
                    hwc = z["cutout"].astype(np.float32)
            except Exception:
                continue

            host_t = torch.from_numpy(hwc).float()
            host_psf = float(host_row["psfsize_r"])
            host_psfdepth = float(host_row["psfdepth_r"])

            # Sample lens and source
            lens = sample_lens_params(rng, theta_e)
            source = sample_source_params(rng, theta_e)

            # Override source magnitude to target bin
            target_mag = float(rng.uniform(mag_lo, mag_hi))
            source = scale_source_to_magnitude(source, target_mag)

            # Inject
            result = inject_sis_shear(
                host_nmgy_hwc=host_t,
                lens=lens,
                source=source,
                pixel_scale=PIXEL_SCALE,
                psf_fwhm_r_arcsec=host_psf,
                core_suppress_radius_pix=None,
                seed=seed + i,
            )

            inj_chw = result.injected[0].numpy()
            proc = preprocess_stack(inj_chw, mode="raw_robust", crop=False, clip_range=10.0)
            score = score_one(model, proc, device)
            scores.append(score)

            sigma_pix_r = estimate_sigma_pix_from_psfdepth(
                host_psfdepth, host_psf, PIXEL_SCALE
            )
            snr = arc_annulus_snr(result.injection_only[0], sigma_pix_r)
            arc_snrs.append(snr)

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{n_sample} scored", end="\r")

        scores_arr = np.array(scores, dtype=np.float64)
        arc_snrs_arr = np.array(arc_snrs, dtype=np.float64)
        valid = np.isfinite(scores_arr)
        n_scored = int(valid.sum())
        valid_scores = scores_arr[valid]
        valid_snrs = arc_snrs_arr[np.isfinite(arc_snrs_arr)]

        det_p03 = float((valid_scores >= 0.3).mean()) if n_scored > 0 else float("nan")
        det_p05 = float((valid_scores >= 0.5).mean()) if n_scored > 0 else float("nan")
        median_score = float(np.median(valid_scores)) if n_scored > 0 else float("nan")
        median_snr = float(np.median(valid_snrs)) if len(valid_snrs) > 0 else float("nan")

        results_by_bin[bin_key] = {
            "mag_lo": mag_lo,
            "mag_hi": mag_hi,
            "n_scored": n_scored,
            "detection_rate_p03": det_p03,
            "detection_rate_p05": det_p05,
            "median_score": median_score,
            "median_arc_snr": median_snr,
        }

        print(f"  N={n_scored}  p>0.3={det_p03:.1%}  p>0.5={det_p05:.1%}  "
              f"median_score={median_score:.4f}  median_SNR={median_snr:.1f}")

    # Build output
    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_path,
        "manifest": manifest_path,
        "host_split": host_split,
        "n_hosts": n_sample,
        "theta_e": theta_e,
        "arch": arch,
        "epoch": epoch,
        "device": str(device),
        "seed": seed,
        "magnitude_bins": [f"{lo:.0f}-{hi:.0f}" for lo, hi in MAGNITUDE_BINS],
        "results_by_bin": results_by_bin,
        "notes": [
            "Tests whether sim-to-real gap (~70 pp) is explained by brightness.",
            "Real lens recall ~73% at p>0.5; standard injection completeness 4–8%.",
            "If brightness explains gap, completeness should rise in bright bins.",
        ],
    }

    return output


def print_summary_table(output: Dict[str, Any]) -> None:
    """Print clean summary table to console."""
    print("\n" + "=" * 80)
    print("BRIGHT ARC INJECTION TEST — SUMMARY")
    print("=" * 80)
    print(f"Checkpoint: {output['checkpoint']}")
    print(f"Host split: {output['host_split']}  |  n_hosts: {output['n_hosts']}")
    print(f"theta_E: {output['theta_e']}\"  |  seed: {output['seed']}")
    print()

    header = (
        f"{'Mag bin':<10} {'N scored':>10} {'p>0.3':>10} {'p>0.5':>10} "
        f"{'median_p':>12} {'median_SNR':>12}"
    )
    print(header)
    print("-" * len(header))

    for bin_key in output["magnitude_bins"]:
        r = output["results_by_bin"][bin_key]
        n = r["n_scored"]
        p03 = r["detection_rate_p03"]
        p05 = r["detection_rate_p05"]
        med_p = r["median_score"]
        med_snr = r["median_arc_snr"]
        p03_str = f"{p03*100:.1f}%" if np.isfinite(p03) else "N/A"
        p05_str = f"{p05*100:.1f}%" if np.isfinite(p05) else "N/A"
        med_p_str = f"{med_p:.4f}" if np.isfinite(med_p) else "N/A"
        med_snr_str = f"{med_snr:.1f}" if np.isfinite(med_snr) else "N/A"
        print(f"{bin_key:<10} {n:>10} {p03_str:>10} {p05_str:>10} {med_p_str:>12} {med_snr_str:>12}")

    print("=" * 80)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bright Arc Injection Test: completeness at varying source magnitudes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    ap.add_argument("--manifest", required=True, help="Path to training manifest parquet")
    ap.add_argument("--host-split", default="val", help="Split for host negatives (default: val)")
    ap.add_argument("--n-hosts", type=int, default=200,
                    help="Number of host galaxies per magnitude bin (default: 200)")
    ap.add_argument("--theta-e", type=float, default=1.5,
                    help="Einstein radius in arcsec (default: 1.5)")
    ap.add_argument("--out-dir", required=True, help="Output directory for JSON results")
    ap.add_argument("--device", default="cuda", help="Device for inference (default: cuda)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    output = run_bright_arc_test(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        host_split=args.host_split,
        n_hosts=args.n_hosts,
        theta_e=args.theta_e,
        out_dir=args.out_dir,
        device_str=args.device,
        seed=args.seed,
    )

    out_path = os.path.join(args.out_dir, "bright_arc_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")

    print_summary_table(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

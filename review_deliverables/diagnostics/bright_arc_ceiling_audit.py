#!/usr/bin/env python3
"""Bright-arc ceiling audit.

Goal
- Diagnose whether the ~30% "bright arc" detection ceiling is caused by preprocessing (robust normalization + clipping)
  or by deeper sim-to-real mismatch.

What this script does
1) Samples host cutouts from your manifest (default: val split, label==0).
2) Injects arcs with a fixed lens configuration while sweeping source magnitude.
3) For each injection, measures:
   - arc SNR (arc-only annulus SNR in r-band)
   - shift in robust normalization statistics (median, MAD) caused by the injection
   - fraction of pixels that would be clipped at clip_range=10 after robust normalization
   - model score p
4) Aggregates detection rate vs source magnitude and prints a diagnostic summary.

Data sources (public, if you need to rebuild cutouts)
- Legacy Surveys viewer cutout service:
  https://www.legacysurvey.org/viewer/cutout
- DR10 sweeps (catalogs):
  https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/

Usage (from repo root)
  PYTHONPATH=. python diagnostics/bright_arc_ceiling_audit.py \
    --checkpoint checkpoints/.../best.pt \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --arch efficientnet_v2_s \
    --n_hosts 200

Interpretation
- If, at bright mags (e.g., 18-20), a large fraction of arc pixels are clipped (>=1-5%) AND the *pre-clip* values saturate,
  then the ceiling is plausibly a preprocessing/clipping artifact.
- If clipping is near-zero even for very bright mags, the ceiling is not caused by clipping; focus on morphology/noise/pipeline.

"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from dhs.model import build_model
from dhs.data import load_cutout_from_file
from dhs.preprocess import preprocess_stack
from dhs.utils import normalize_outer_annulus
from dhs.injection_engine import (
    inject_sie_shear,
    arc_annulus_snr,
)
from dhs.constants import PIXEL_SCALE


@dataclass
class LensParams:
    theta_E_arcsec: float = 1.5
    q: float = 0.8
    phi_deg: float = 30.0
    gamma_ext: float = 0.05
    phi_ext_deg: float = 10.0
    x0_arcsec: float = 0.0
    y0_arcsec: float = 0.0


@dataclass
class SourceParams:
    n_sersic: float = 1.0
    Re_arcsec: float = 0.25
    q: float = 0.7
    phi_deg: float = -20.0
    clump_count: int = 0
    clump_frac: float = 0.25


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def load_model(arch: str, checkpoint_path: str, device: str) -> torch.nn.Module:
    model = build_model(arch, in_ch=3, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def pick_hosts(manifest_path: str, n_hosts: int, seed: int, split: str = "val") -> pd.DataFrame:
    df = pd.read_parquet(manifest_path)
    required = {"cutout_path", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    d = df[(df["split"] == split) & (df["label"] == 0)].copy()
    if len(d) == 0:
        raise ValueError(f"No hosts found for split={split} label==0")
    rng = np.random.default_rng(seed)
    take = min(n_hosts, len(d))
    return d.sample(n=take, replace=False, random_state=int(rng.integers(0, 2**31 - 1)))


def fwhm_to_sigma_pix(fwhm_arcsec: float) -> float:
    return (fwhm_arcsec / 2.355) / PIXEL_SCALE


def inject_one(
    host_chw: np.ndarray,
    lens: LensParams,
    src: SourceParams,
    src_mag_r: float,
    colors: Tuple[float, float],
    psf_fwhm_arcsec: float,
    oversample: int,
    add_poisson: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Returns (injected_hwc, arc_only_hwc, arc_snr)."""

    # Convert to HWC
    host_hwc = host_chw.transpose(1, 2, 0).astype(np.float32)

    # Convert magnitudes to nanomaggies.
    # AB ZP=22.5 convention: flux_nmgy = 10**((22.5 - mag)/2.5)
    flux_r = 10 ** ((22.5 - src_mag_r) / 2.5)
    g_r, r_z = colors
    flux_g = flux_r * 10 ** (+g_r / 2.5)  # mag_g = mag_r + (g-r) -> flux_g = flux_r * 10**((mag_r-mag_g)/2.5)
    flux_z = flux_r * 10 ** (-r_z / 2.5)  # mag_z = mag_r - (r-z)

    thetaE_pix = lens.theta_E_arcsec / PIXEL_SCALE
    x0_pix = lens.x0_arcsec / PIXEL_SCALE
    y0_pix = lens.y0_arcsec / PIXEL_SCALE

    phi = math.radians(lens.phi_deg)
    phi_ext = math.radians(lens.phi_ext_deg)

    # Source offset: area-weighted beta_frac ~ sqrt(U)
    beta_frac = math.sqrt(rng.uniform(0.01, 1.0))
    beta = beta_frac * thetaE_pix
    ang = rng.uniform(0, 2 * math.pi)
    bx = beta * math.cos(ang)
    by = beta * math.sin(ang)

    # Build source magnitude per band
    flux_per_band = np.array([flux_g, flux_r, flux_z], dtype=np.float32)

    # PSF sigma per band (use same FWHM for all bands for this audit)
    psf_sigma = fwhm_to_sigma_pix(psf_fwhm_arcsec)
    psf_sigmas = np.array([psf_sigma, psf_sigma, psf_sigma], dtype=np.float32)

    injected_hwc, arc_only_hwc = inject_sie_shear(
        host_hwc,
        thetaE_pix=float(thetaE_pix),
        q=float(lens.q),
        phi=float(phi),
        gamma_ext=float(lens.gamma_ext),
        phi_ext=float(phi_ext),
        x0=float(x0_pix),
        y0=float(y0_pix),
        beta_x=float(bx),
        beta_y=float(by),
        n=float(src.n_sersic),
        Re_pix=float(src.Re_arcsec / PIXEL_SCALE),
        q_src=float(src.q),
        phi_src=float(math.radians(src.phi_deg)),
        flux_nmgy=flux_per_band,
        psf_sigma=psf_sigmas,
        oversample=int(oversample),
        add_clumps=bool(src.clump_count > 0),
        clump_count=int(src.clump_count),
        clump_flux_frac=float(src.clump_frac),
        rng=rng,
    )

    if add_poisson:
        # Very simple Poisson model for the arc photons.
        # Assumes nanomaggies proportional to expected counts; scale factor is unknown.
        # We only want to add *texture*; this should be treated as a diagnostic, not a physical noise model.
        # Set scale so that typical arc pixel values produce modest shot noise.
        scale = 30.0
        lam = np.clip(arc_only_hwc * scale, 0.0, None)
        noise = rng.poisson(lam).astype(np.float32) / scale - (lam.astype(np.float32) / scale)
        injected_hwc = injected_hwc + noise

    snr = float(arc_annulus_snr(arc_only_hwc, r_in=4.0, r_out=16.0, band=1))
    return injected_hwc, arc_only_hwc, snr


def score_model(model: torch.nn.Module, img_hwc_nmgy: np.ndarray, device: str, clip_range: float) -> float:
    # Convert to CHW for preprocess_stack and torch
    chw = img_hwc_nmgy.transpose(2, 0, 1).astype(np.float32)
    x = preprocess_stack(chw, mode="raw_robust", crop=False, clip_range=clip_range)
    xt = torch.from_numpy(x[None]).to(device)
    with torch.no_grad():
        logit = model(xt).squeeze()
        p = float(_sigmoid(logit).cpu().item())
    return p


def robust_stats(chw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-band (median, mad) used by normalize_outer_annulus, without clipping."""
    img = chw.astype(np.float32)
    med, mad = normalize_outer_annulus(img, clip=1e10)  # trick: returns normalized img; stats computed internally
    # The helper returns normalized image; we need the stats. Recompute here explicitly.
    H, W = img.shape[1], img.shape[2]
    yy, xx = np.ogrid[:H, :W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ann = (rr >= 20.0) & (rr <= 32.0)
    meds = []
    mads = []
    for b in range(img.shape[0]):
        v = img[b][ann]
        m = np.median(v)
        d = np.median(np.abs(v - m))
        meds.append(float(m))
        mads.append(float(max(d, 1e-8)))
    return np.array(meds, dtype=np.float32), np.array(mads, dtype=np.float32)


def clip_fraction_after_robust_norm(chw: np.ndarray, clip_range: float) -> float:
    # replicate preprocess_stack core: robust normalize then count values beyond clip
    img = chw.astype(np.float32)
    img_norm = normalize_outer_annulus(img, clip=1e10)  # no clip here
    frac = float(np.mean(np.abs(img_norm) > clip_range))
    return frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--arch", default="efficientnet_v2_s")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n_hosts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--mags", default="18,19,20,21,22,23,24,25,26")
    ap.add_argument("--psf_fwhm", type=float, default=1.2)
    ap.add_argument("--clip_range", type=float, default=10.0)
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--add_poisson", action="store_true")

    ap.add_argument("--thetaE", type=float, default=1.5)
    ap.add_argument("--q_lens", type=float, default=0.8)
    ap.add_argument("--phi_lens", type=float, default=30.0)
    ap.add_argument("--gamma_ext", type=float, default=0.05)

    ap.add_argument("--out_csv", default="")

    args = ap.parse_args()

    mags = [float(x) for x in args.mags.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    model = load_model(args.arch, args.checkpoint, args.device)
    hosts = pick_hosts(args.manifest, args.n_hosts, args.seed, split=args.split)

    lens = LensParams(
        theta_E_arcsec=args.thetaE,
        q=args.q_lens,
        phi_deg=args.phi_lens,
        gamma_ext=args.gamma_ext,
        phi_ext_deg=10.0,
    )
    src = SourceParams(n_sersic=1.0, Re_arcsec=0.25, q=0.7, phi_deg=-20.0, clump_count=0)

    rows: List[Dict] = []
    for _, row in hosts.iterrows():
        path = str(row["cutout_path"])
        if not os.path.exists(path):
            # Skip missing local cutouts
            continue
        host_chw = load_cutout_from_file(path)

        host_med, host_mad = robust_stats(host_chw)

        for mag in mags:
            # Use a fixed, lens-like blue color by default for this audit.
            # You can change this to match your empirical distribution.
            colors = (0.3, 0.2)  # (g-r, r-z)
            inj_hwc, arc_hwc, snr = inject_one(
                host_chw,
                lens=lens,
                src=src,
                src_mag_r=mag,
                colors=colors,
                psf_fwhm_arcsec=args.psf_fwhm,
                oversample=args.oversample,
                add_poisson=bool(args.add_poisson),
                rng=rng,
            )

            inj_chw = inj_hwc.transpose(2, 0, 1)
            inj_med, inj_mad = robust_stats(inj_chw)

            clip_frac = clip_fraction_after_robust_norm(inj_chw, clip_range=args.clip_range)
            p = score_model(model, inj_hwc, args.device, clip_range=args.clip_range)

            rows.append(
                dict(
                    cutout_path=path,
                    src_mag_r=mag,
                    arc_snr=snr,
                    p=p,
                    clip_frac=clip_frac,
                    d_med_r=float(inj_med[1] - host_med[1]),
                    d_mad_r=float(inj_mad[1] - host_mad[1]),
                )
            )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError("No samples scored. Check that cutout_path values exist locally.")

    # Summary
    thr = 0.3
    print("\\n=== Bright-arc ceiling audit summary ===")
    print(f"Hosts scored: {out['cutout_path'].nunique()}  (rows={len(out)})")
    print(f"clip_range={args.clip_range}  add_poisson={args.add_poisson}  psf_fwhm={args.psf_fwhm} arcsec")
    print("\\nPer-magnitude detection rates (p>0.3):")
    for mag in mags:
        d = out[out.src_mag_r == mag]
        if len(d) == 0:
            continue
        det = float(np.mean(d.p > thr))
        clip = float(np.mean(d.clip_frac))
        snr_med = float(np.median(d.arc_snr))
        print(f"  mag_r={mag:5.1f}  det={det*100:6.2f}%   median(arc_snr)={snr_med:7.2f}   mean(clip_frac)={clip*100:7.3f}%")

    print("\\nClip diagnostics:")
    print(f"  mean clip_frac over all rows: {100*np.mean(out.clip_frac):.4f}%")
    print(f"  95th percentile clip_frac:    {100*np.percentile(out.clip_frac,95):.4f}%")

    print("\\nRobust-stat shift diagnostics (r-band):")
    print(f"  mean d_med_r: {np.mean(out.d_med_r):.4g}  (nmgy)")
    print(f"  mean d_mad_r: {np.mean(out.d_mad_r):.4g}  (nmgy)")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        print(f"\\nWrote: {args.out_csv}")


if __name__ == "__main__":
    main()

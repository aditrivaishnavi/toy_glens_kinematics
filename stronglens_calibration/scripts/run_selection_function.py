#!/usr/bin/env python3
"""Selection function grid: inject arc proxy on test hosts, score with DHS model, report completeness."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common.manifest_utils import (
    CUTOUT_PATH_COL,
    LABEL_COL,
    SPLIT_COL,
    load_manifest,
    safe_json_dump,
    sigmoid,
)

try:
    from constants import AB_ZEROPOINT_MAG
except ImportError:
    AB_ZEROPOINT_MAG = 22.5

from dhs.model import build_resnet18
from dhs.preprocess import preprocess_stack


def mag_to_nmgy(mag: float) -> float:
    return 10 ** ((AB_ZEROPOINT_MAG - mag) / 2.5)


def psfdepth_to_mag(psfdepth: float, nsigma: float = 5.0) -> float:
    """Convert psfdepth (inverse variance, nanomaggies^-2) to 5-sigma depth in AB mag.

    DR10 convention: psfdepth_r is 1/sigma^2 for a point source.
    The n-sigma depth is: mag = 22.5 - 2.5*log10(nsigma / sqrt(psfdepth)).
    """
    if psfdepth <= 0:
        return float("nan")
    return float(AB_ZEROPOINT_MAG - 2.5 * np.log10(nsigma / np.sqrt(psfdepth)))


def load_npz(path: str) -> np.ndarray:
    with np.load(path) as z:
        return z["cutout"].astype(np.float32)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    r = int(max(1, math.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), 0, img)
    out = np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), 1, out)
    return out.astype(np.float32)


def render_arc_proxy(
    size: int, thetaE_arcsec: float, pixscale: float, rng: np.random.Generator
) -> np.ndarray:
    h = w = size
    cy = cx = size // 2
    yy, xx = np.mgrid[0:h, 0:w]
    dy = yy - cy
    dx = xx - cx
    rr = np.sqrt(dx * dx + dy * dy) + 1e-6
    ang = np.arctan2(dy, dx)
    theta_pix = thetaE_arcsec / pixscale
    rad_sigma = rng.uniform(0.8, 1.8)
    radial = np.exp(-0.5 * ((rr - theta_pix) / rad_sigma) ** 2).astype(np.float32)
    nseg = int(rng.integers(1, 4))
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(nseg):
        phi0 = rng.uniform(-np.pi, np.pi)
        width = rng.uniform(0.25, 0.55)
        d = (ang - phi0 + np.pi) % (2 * np.pi) - np.pi
        seg = np.exp(-0.5 * (d / width) ** 2).astype(np.float32)
        mask = np.maximum(mask, seg)
    cl = rng.normal(0, 1, size=(h, w)).astype(np.float32)
    cl = gaussian_blur(cl, rng.uniform(0.6, 1.2))
    cl = np.clip(
        (cl - np.percentile(cl, 60)) / (np.std(cl) + 1e-6), 0, None
    )
    arc = radial * mask * (0.6 + 0.8 * cl)
    arc /= arc.sum() + 1e-6
    return arc


def bayes_binom(k: int, n: int, alpha: float = 0.32) -> Tuple[float, float]:
    import scipy.stats as st

    a = k + 0.5
    b = (n - k) + 0.5
    return float(st.beta.ppf(alpha / 2, a, b)), float(
        st.beta.ppf(1 - alpha / 2, a, b)
    )


def load_model(ckpt_path: str, device: torch.device) -> nn.Module:
    """Build ResNet18(3) and load state from checkpoint (same as run_evaluation)."""
    model = build_resnet18(3).to(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def score(model: nn.Module, batch: np.ndarray, device: torch.device) -> np.ndarray:
    """batch: (N, 3, 64, 64) float32."""
    x = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        logits = model(x).squeeze(1).detach().cpu().numpy()
    return sigmoid(logits)


def nearest(val: float, grid: List[float]) -> float:
    arr = np.asarray(grid)
    return float(arr[np.argmin(np.abs(arr - val))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_csv", default="selection_function.csv")
    ap.add_argument("--out_json", default="selection_function_meta.json")
    ap.add_argument(
        "--host_split", default="test", choices=["train", "val", "test"]
    )
    ap.add_argument("--host_max", type=int, default=20000)
    ap.add_argument(
        "--depth_col",
        default="psfdepth_r",
        choices=["psfdepth_r", "galdepth_r"],
    )
    ap.add_argument("--thetaE_min", type=float, default=0.5)
    ap.add_argument("--thetaE_max", type=float, default=3.0)
    ap.add_argument("--thetaE_step", type=float, default=0.25)
    ap.add_argument("--psf_min", type=float, default=0.9)
    ap.add_argument("--psf_max", type=float, default=1.8)
    ap.add_argument("--psf_step", type=float, default=0.15)
    ap.add_argument("--depth_min", type=float, default=22.5)
    ap.add_argument("--depth_max", type=float, default=24.5)
    ap.add_argument("--depth_step", type=float, default=0.5)
    ap.add_argument("--pixscale", type=float, default=0.262)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--injections_per_cell", type=int, default=200)
    ap.add_argument("--source_mag_r_min", type=float, default=22.5)
    ap.add_argument("--source_mag_r_max", type=float, default=25.5)
    ap.add_argument("--mu_min", type=float, default=5.0)
    ap.add_argument("--mu_max", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    df = load_manifest(args.manifest)
    # Use only non-training hosts (e.g. test split, label==0).
    hosts = df[
        (df[SPLIT_COL] == args.host_split) & (df[LABEL_COL] == 0)
    ].copy()
    if hosts.empty:
        raise ValueError("No negative hosts found")

    for col in ["psfsize_r", args.depth_col]:
        if col not in hosts.columns:
            raise ValueError(f"Missing host column: {col}")

    rng = np.random.default_rng(args.seed)
    if len(hosts) > args.host_max:
        hosts = hosts.sample(n=args.host_max, random_state=args.seed)

    thetaEs = np.round(
        np.arange(args.thetaE_min, args.thetaE_max + 1e-9, args.thetaE_step),
        6,
    ).tolist()
    psfs = np.round(
        np.arange(args.psf_min, args.psf_max + 1e-9, args.psf_step), 6
    ).tolist()
    depths = np.round(
        np.arange(args.depth_min, args.depth_max + 1e-9, args.depth_step), 6
    ).tolist()

    hosts["psf_bin"] = hosts["psfsize_r"].apply(lambda v: nearest(float(v), psfs))
    # Convert psfdepth (inverse variance) to 5-sigma magnitude depth before binning
    hosts["depth_mag"] = hosts[args.depth_col].apply(
        lambda v: psfdepth_to_mag(float(v))
    )
    hosts["depth_bin"] = hosts["depth_mag"].apply(
        lambda v: nearest(float(v), depths)
    )

    groups: Dict[Tuple[float, float], np.ndarray] = {}
    for (pb, db), g in hosts.groupby(["psf_bin", "depth_bin"]):
        groups[(float(pb), float(db))] = g[CUTOUT_PATH_COL].to_numpy(object)

    model = load_model(args.checkpoint, device)

    rows = []
    for thetaE in thetaEs:
        for pb in psfs:
            for db in depths:
                paths = groups.get((float(pb), float(db)))
                if paths is None or len(paths) == 0:
                    rows.append(
                        {
                            "thetaE": thetaE,
                            "psf_fwhm": pb,
                            "depth": db,
                            "n": 0,
                            "k": 0,
                            "completeness": float("nan"),
                            "ci68_lo": float("nan"),
                            "ci68_hi": float("nan"),
                            "sufficient": False,
                        }
                    )
                    continue
                n = args.injections_per_cell
                sel = rng.choice(paths, size=n, replace=True)
                batch = np.zeros((n, 3, 64, 64), dtype=np.float32)
                for i, p in enumerate(sel):
                    host = load_npz(str(p))
                    src_mag = rng.uniform(
                        args.source_mag_r_min, args.source_mag_r_max
                    )
                    mu = rng.uniform(args.mu_min, args.mu_max)
                    arc_flux = mag_to_nmgy(src_mag) * mu
                    arc = render_arc_proxy(
                        host.shape[0], thetaE, args.pixscale, rng
                    )
                    gsc = rng.uniform(1.1, 1.6)
                    zsc = rng.uniform(0.6, 0.95)
                    arc_grz = np.stack(
                        [
                            arc * arc_flux * gsc,
                            arc * arc_flux,
                            arc * arc_flux * zsc,
                        ],
                        axis=-1,
                    ).astype(np.float32)
                    inj = host + arc_grz
                    sigma = pb / (2.355 * args.pixscale)
                    inj = gaussian_blur(inj, max(0.0, sigma - 0.5))
                    # Match training: HWC -> CHW, then dhs preprocess (crop then normalize on 64x64)
                    img3 = np.transpose(inj, (2, 0, 1))
                    proc = preprocess_stack(
                        img3, mode="raw_robust", crop=True, clip_range=10.0
                    )
                    batch[i] = proc
                scores = score(model, batch, device)
                k = int((scores >= args.threshold).sum())
                comp = k / n
                lo, hi = bayes_binom(k, n, alpha=0.32)
                rows.append(
                    {
                        "thetaE": thetaE,
                        "psf_fwhm": pb,
                        "depth": db,
                        "n": n,
                        "k": k,
                        "completeness": float(comp),
                        "ci68_lo": lo,
                        "ci68_hi": hi,
                        "sufficient": n >= args.injections_per_cell,
                    }
                )

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    meta = {
        "manifest": str(Path(args.manifest).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "host_split": args.host_split,
        "host_max": args.host_max,
        "depth_col": args.depth_col,
        "threshold": args.threshold,
        "injections_per_cell": args.injections_per_cell,
        "thetaEs": thetaEs,
        "psf_bins": psfs,
        "depth_bins": depths,
        "seed": args.seed,
        "notes": [
            "Arc renderer is minimal proxy. Replace with Phase4c injector for publication-quality realism.",
            "Preprocessing and model from dhs (preprocess_stack raw_robust crop=True, build_resnet18(3)).",
        ],
    }
    safe_json_dump(meta, args.out_json)
    print(f"Wrote {args.out_csv} and {args.out_json}")


if __name__ == "__main__":
    main()

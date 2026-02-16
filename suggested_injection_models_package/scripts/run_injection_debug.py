
#!/usr/bin/env python3
"""
Run selection function with three injection modes:

Model 1: random hosts + parametric source (baseline)
Model 2: deflector-conditioned injection (host moments set lens q, PA, and host-matched selection)
Model 3: real-source stamps (GalSim COSMOS) + deflector-conditioned injection (optional)

This script focuses on the *injection side*. It expects:
- a manifest parquet with columns:
  cutout_path, label, split, psfsize_r, psfdepth_r
- cutouts are .npz with key 'cutout' shape (101,101,3) HWC in nanomaggies

It writes outputs to either local or S3 depending on your environment.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from dhs.injection_engine import LensParams, SourceParams, inject_lensed_arcs, mag_to_nmgy
from dhs.host_matching import estimate_host_moments_rband, map_host_to_lens_params

def load_cutout_npz(path: str) -> np.ndarray:
    arr = np.load(path)["cutout"]
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC with 3 bands, got {arr.shape}")
    return arr

def sample_source_params(r_mag: float, theta_e: float, rng: np.random.Generator) -> SourceParams:
    # area-weighted beta_frac in [0.1,1.0]
    u = rng.uniform(0.1**2, 1.0**2)
    beta_frac = math.sqrt(u)
    beta = beta_frac * theta_e
    ang = rng.uniform(0.0, 2*math.pi)
    beta_x = beta * math.cos(ang)
    beta_y = beta * math.sin(ang)

    # Sersic in source plane
    re = rng.uniform(0.1, 0.5)   # arcsec
    n = rng.uniform(0.5, 4.0)
    q = rng.uniform(0.3, 1.0)
    phi = rng.uniform(0.0, math.pi)

    # colors (simple prior; replace with SED-based if you have it)
    gr = rng.uniform(0.0, 1.5)
    rz = rng.uniform(-0.3, 1.0)
    g_mag = r_mag + gr
    z_mag = r_mag - rz

    fg = mag_to_nmgy(g_mag)
    fr = mag_to_nmgy(r_mag)
    fz = mag_to_nmgy(z_mag)

    n_clumps = int(rng.integers(0, 4))
    return SourceParams(
        beta_x_arcsec=beta_x, beta_y_arcsec=beta_y,
        re_arcsec=re, n_sersic=n, q=q, phi_rad=phi,
        flux_nmgy_g=fg, flux_nmgy_r=fr, flux_nmgy_z=fz,
        n_clumps=n_clumps, clump_frac=0.25
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", choices=["1","2"], default="2")
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--theta_e", type=float, default=1.5)
    ap.add_argument("--r_mag_min", type=float, default=23.0)
    ap.add_argument("--r_mag_max", type=float, default=26.0)
    ap.add_argument("--psf_kernel", type=int, default=33)
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--out_dir", default="outputs_injection_debug")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pq.read_table(args.manifest, columns=["cutout_path","label","split","psfsize_r","psfdepth_r"]).to_pandas()
    df = df[(df["split"]==args.split) & (df["label"]==0)].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No negative hosts found for requested split")
    idxs = rng.integers(0, len(df), size=args.n_samples)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # For debug we just generate injected stamps and save metadata.
    # Integrate with your model scoring separately.
    meta_rows = []
    for i,ix in enumerate(idxs):
        row = df.iloc[int(ix)]
        host = load_cutout_npz(row["cutout_path"])
        psf = float(row["psfsize_r"]) if np.isfinite(row["psfsize_r"]) else 1.2
        psfdepth = float(row["psfdepth_r"]) if np.isfinite(row["psfdepth_r"]) else 1e4

        r_mag = float(rng.uniform(args.r_mag_min, args.r_mag_max))
        src = sample_source_params(r_mag, args.theta_e, rng)

        if args.model == "1":
            lens = LensParams(theta_e_arcsec=args.theta_e)
        else:
            hm = estimate_host_moments_rband(host)
            lp = map_host_to_lens_params(args.theta_e, hm, rng=rng)
            lens = LensParams(**lp)

        res = inject_lensed_arcs(
            torch.from_numpy(host).float(),
            lens=lens,
            src=src,
            psf_fwhm_arcsec_r=psf,
            psf_kernel_size=args.psf_kernel,
            psfdepth_r=psfdepth,
            add_noise=True,
            oversample=args.oversample,
        )

        # save a small npz for inspection
        np.savez_compressed(out/f"injected_{i:05d}.npz",
                            host=host.astype(np.float32),
                            injection_only=res.injection_only.cpu().numpy().astype(np.float32),
                            injected=res.injected.cpu().numpy().astype(np.float32),
                            meta=res.meta)

        meta_rows.append(dict(i=i, cutout_path=row["cutout_path"], psfsize_r=psf, psfdepth_r=psfdepth,
                              theta_e=args.theta_e, r_mag=r_mag, model=args.model))

        if (i+1) % 200 == 0:
            print(f"{i+1}/{args.n_samples} done")

    (out/"meta.jsonl").write_text("\n".join(json.dumps(r) for r in meta_rows))

if __name__ == "__main__":
    main()

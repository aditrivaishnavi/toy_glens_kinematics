#!/usr/bin/env python3
"""
Host Conditioning Diagnostic: LRG vs Random Host Comparison.

The key diagnostic experiment suggested by the independent LLM review:
  At matched theta_E and source brightness, inject arcs onto:
    (a) LRG-like hosts (DEV/SER morphology) — deflector-like
    (b) Random hosts (all morphology types) — Model 1 baseline

  Compare completeness directly.  This isolates the host-conditioning effect
  and answers: "does the CNN require deflector context to detect lensed arcs?"

If the CNN is truly a "lens-system detector" (not just an "arc detector"),
we expect:
  - LRG hosts: significantly higher completeness (the model "recognizes" the
    deflector context)
  - Random hosts: lower completeness (even bright arcs on disk galaxies may
    not trigger detection)

This experiment does NOT run the full 3D grid.  Instead it holds PSF/depth
at fixed values (or marginalizes) and sweeps theta_E at a few source mag bins,
running both host types at each point for direct A/B comparison.

Usage:
    cd /lambda/nfs/.../code
    export PYTHONPATH=.:stronglens_calibration/injection_model_2

    python stronglens_calibration/injection_model_2/scripts/host_conditioning_diagnostic.py \
        --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
        --manifest manifests/training_parity_70_30_v1.parquet \
        --out-dir results/host_conditioning_diagnostic

Author: stronglens_calibration project
Date: 2026-02-13
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# dhs imports
from dhs.model import build_model
from dhs.preprocess import preprocess_stack
from dhs.constants import CUTOUT_SIZE, STAMP_SIZE
from dhs.injection_engine import (
    AB_ZP,
    LensParams,
    inject_sis_shear,
    sample_lens_params,
    sample_source_params,
    estimate_sigma_pix_from_psfdepth,
    arc_annulus_snr,
)
from dhs.selection_function_utils import bayes_binomial_interval

# Model 2 imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from host_matching import estimate_host_moments_rband, map_host_to_lens_params
from host_selection import select_lrg_hosts, select_random_hosts

CUTOUT_PATH_COL = "cutout_path"
PIXEL_SCALE = 0.262


def load_model_from_checkpoint(checkpoint_path, device):
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
def score_batch(model, batch, device):
    x = torch.from_numpy(batch).float().to(device)
    logits = model(x).squeeze(1).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))


def run_diagnostic(
    model: nn.Module,
    hosts_df: pd.DataFrame,
    host_label: str,
    theta_es: List[float],
    n_per_point: int,
    preprocessing: str,
    crop: bool,
    pixscale: float,
    thresholds: List[float],
    seed: int,
    device: torch.device,
    rng: np.random.Generator,
    use_host_conditioning: bool = False,
) -> List[Dict[str, Any]]:
    """Run injection-recovery for one host type.

    Parameters
    ----------
    host_label : str
        "LRG" or "random" — for labeling output.
    use_host_conditioning : bool
        If True, use Model 2 lens params (host-conditioned q/PA).
        If False, use Model 1 lens params (independent priors).
    """
    input_size = CUTOUT_SIZE if not crop else STAMP_SIZE
    MAX_RETRIES = 5
    rows = []

    for theta_e in theta_es:
        batch_list = []
        arc_snr_list = []
        src_mag_list = []
        host_q_list = []
        n_failed = 0

        attempt = 0
        while len(batch_list) < n_per_point and attempt < n_per_point + n_per_point * MAX_RETRIES:
            attempt += 1
            host_idx = int(rng.choice(len(hosts_df)))
            host_row = hosts_df.iloc[host_idx]

            try:
                with np.load(str(host_row[CUTOUT_PATH_COL])) as z:
                    host_hwc = z["cutout"].astype(np.float32)

                host_hwc_torch = torch.from_numpy(host_hwc).float()
                host_psf = float(host_row["psfsize_r"])
                host_psfdepth = float(host_row["psfdepth_r"])

                if use_host_conditioning:
                    moments = estimate_host_moments_rband(host_hwc)
                    param_dict = map_host_to_lens_params(theta_e, moments, rng=rng)
                    x0 = float(rng.normal(0.0, 0.05))
                    y0 = float(rng.normal(0.0, 0.05))
                    lens = LensParams(
                        theta_e_arcsec=param_dict["theta_e_arcsec"],
                        shear_g1=param_dict["shear_g1"],
                        shear_g2=param_dict["shear_g2"],
                        x0_arcsec=x0, y0_arcsec=y0,
                        q_lens=param_dict["q_lens"],
                        phi_lens_rad=param_dict["phi_lens_rad"],
                    )
                    _pending_host_q = moments.q
                else:
                    lens = sample_lens_params(rng, theta_e)
                    _pending_host_q = float("nan")

                source = sample_source_params(rng, theta_e)

                src_r_mag = float("nan")
                if source.flux_nmgy_r > 0:
                    src_r_mag = AB_ZP - 2.5 * math.log10(source.flux_nmgy_r)

                inj_seed = seed + attempt
                result = inject_sis_shear(
                    host_nmgy_hwc=host_hwc_torch,
                    lens=lens, source=source,
                    pixel_scale=pixscale,
                    psf_fwhm_r_arcsec=host_psf,
                    seed=inj_seed,
                )

                sigma_pix_r = estimate_sigma_pix_from_psfdepth(host_psfdepth, host_psf, pixscale)
                snr_val = arc_annulus_snr(result.injection_only[0], sigma_pix_r)

                proc = preprocess_stack(
                    result.injected[0].numpy(), mode=preprocessing,
                    crop=crop, clip_range=10.0
                )

                batch_list.append(proc)
                arc_snr_list.append(snr_val)
                src_mag_list.append(src_r_mag)
                host_q_list.append(_pending_host_q)

            except Exception:
                n_failed += 1

        n_ok = len(batch_list)
        if n_ok == 0:
            for thr in thresholds:
                rows.append({
                    "host_type": host_label,
                    "theta_e": theta_e,
                    "threshold": thr,
                    "use_host_conditioning": use_host_conditioning,
                    "n_injections": 0,
                    "n_detected": 0,
                    "n_failed": n_failed,
                    "completeness": float("nan"),
                    "ci68_lo": float("nan"),
                    "ci68_hi": float("nan"),
                    "mean_score": float("nan"),
                    "mean_arc_snr": float("nan"),
                    "mean_host_q": float("nan"),
                })
            continue

        batch = np.stack(batch_list, axis=0)
        scores = score_batch(model, batch, device)
        arc_snrs = np.array(arc_snr_list, dtype=np.float64)
        host_qs = np.array(host_q_list, dtype=np.float64)

        valid_snrs = arc_snrs[np.isfinite(arc_snrs)]
        mean_snr = float(valid_snrs.mean()) if len(valid_snrs) > 0 else float("nan")
        valid_qs = host_qs[np.isfinite(host_qs)]
        mean_hq = float(valid_qs.mean()) if len(valid_qs) > 0 else float("nan")

        for thr in thresholds:
            detected = scores >= thr
            k = int(detected.sum())
            comp = k / n_ok
            lo, hi = bayes_binomial_interval(k, n_ok, level=0.68)
            rows.append({
                "host_type": host_label,
                "theta_e": theta_e,
                "threshold": thr,
                "use_host_conditioning": use_host_conditioning,
                "n_injections": n_ok,
                "n_detected": k,
                "n_failed": n_failed,
                "completeness": float(comp),
                "ci68_lo": lo,
                "ci68_hi": hi,
                "mean_score": float(scores.mean()),
                "mean_arc_snr": mean_snr,
                "mean_host_q": mean_hq,
            })

        print(f"    theta_E={theta_e:.2f}, {host_label}: C(0.5)={comp:.3f}, "
              f"N={n_ok}, failed={n_failed}, mean_SNR={mean_snr:.1f}")

    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Host Conditioning Diagnostic: LRG vs Random",
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="results/host_conditioning_diagnostic")
    ap.add_argument("--host-split", default="val")
    ap.add_argument("--n-per-point", type=int, default=500,
                    help="Injections per (theta_E, host_type) combination")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    ap.add_argument("--theta-es", nargs="+", type=float,
                    default=[0.75, 1.0, 1.25, 1.5, 2.0, 2.5],
                    help="theta_E values to test")
    ap.add_argument("--preprocessing", default="raw_robust")
    ap.add_argument("--crop", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model: {args.checkpoint}")
    model, arch, epoch = load_model_from_checkpoint(args.checkpoint, device)
    print(f"  Architecture: {arch}, Epoch: {epoch}")

    # Load manifest
    print(f"Loading manifest: {args.manifest}")
    df = pd.read_parquet(args.manifest)

    # Select both host populations
    print("\n--- Selecting host populations ---")
    lrg_hosts = select_lrg_hosts(df, split=args.host_split)
    random_hosts = select_random_hosts(df, split=args.host_split)
    print(f"  LRG hosts: {len(lrg_hosts):,}")
    print(f"  Random hosts: {len(random_hosts):,}")

    rng = np.random.default_rng(args.seed)

    # Run 4 conditions:
    #  1. LRG hosts + host-conditioned lens params (Model 2 full)
    #  2. LRG hosts + independent lens params (isolate host effect)
    #  3. Random hosts + independent lens params (Model 1 baseline)
    #  4. Random hosts + host-conditioned lens params (control)

    all_rows = []

    conditions = [
        ("LRG_conditioned", lrg_hosts, True),
        ("LRG_independent", lrg_hosts, False),
        ("random_independent", random_hosts, False),
        ("random_conditioned", random_hosts, True),
    ]

    for label, hosts, conditioning in conditions:
        print(f"\n--- Running: {label} ({len(hosts):,} hosts, "
              f"conditioning={conditioning}) ---")
        rows = run_diagnostic(
            model=model,
            hosts_df=hosts,
            host_label=label,
            theta_es=args.theta_es,
            n_per_point=args.n_per_point,
            preprocessing=args.preprocessing,
            crop=args.crop,
            pixscale=PIXEL_SCALE,
            thresholds=args.thresholds,
            seed=args.seed,
            device=device,
            rng=rng,
            use_host_conditioning=conditioning,
        )
        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "host_conditioning_diagnostic.csv")
    results_df.to_csv(csv_path, index=False)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "arch": arch,
        "epoch": epoch,
        "host_split": args.host_split,
        "n_per_point": args.n_per_point,
        "theta_es": args.theta_es,
        "thresholds": args.thresholds,
        "n_lrg_hosts": len(lrg_hosts),
        "n_random_hosts": len(random_hosts),
        "conditions": [c[0] for c in conditions],
        "description": (
            "4-way comparison isolating host type and lens conditioning effects. "
            "LRG_conditioned = Model 2 (full). random_independent = Model 1 (baseline). "
            "The other two are controls."
        ),
    }
    json_path = os.path.join(args.out_dir, "host_conditioning_diagnostic_meta.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nResults saved to: {csv_path}")
    print(f"Metadata saved to: {json_path}")

    # Print comparison summary
    print(f"\n{'='*70}")
    print("HOST CONDITIONING DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")
    for thr in args.thresholds:
        print(f"\n  Threshold = {thr}")
        for te in args.theta_es:
            mask = (results_df["theta_e"] == te) & (results_df["threshold"] == thr)
            sub = results_df[mask]
            parts = []
            for _, row in sub.iterrows():
                c = row["completeness"]
                n = row["n_injections"]
                label = row["host_type"]
                parts.append(f"{label}: C={c:.3f} (N={n})")
            print(f"    theta_E={te:.2f}:  " + "  |  ".join(parts))
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

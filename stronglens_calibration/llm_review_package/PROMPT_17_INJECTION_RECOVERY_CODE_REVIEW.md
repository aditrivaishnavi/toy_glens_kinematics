# Prompt 17: Code Review — Injection Cutout Recovery & Comparison Figure Pipeline

## Attached Code Package

The zip file `injection_recovery_code_package.zip` contains 11 files:

| File | Role |
|------|------|
| `recover_injection_cutouts.py` | **NEW** — Script under review (recovery/replay) |
| `find_comparison_candidates.py` | **NEW** — Script under review (figure generation) |
| `injection_engine.py` | Core injection library (`sample_lens_params`, `sample_source_params`, `inject_sis_shear`) — **read this to answer the RNG questions** |
| `selection_function_grid.py` | Original grid experiment being replayed |
| `bright_arc_injection_test.py` | Original bright-arc experiment being replayed |
| `feature_space_analysis.py` | Original linear probe experiment being replayed |
| `run_d05_full_reeval.sh` | The D05 run script — shows exact CLI args/seeds for all 10 experiments |
| `preprocess.py` | Preprocessing (no RNG, included for completeness) |
| `scoring_utils.py` | Model loading (no RNG, included for completeness) |
| `selection_function_utils.py` | Utility functions (no RNG, included for completeness) |
| `constants.py` | Constants: `CUTOUT_SIZE=101`, `STAMP_SIZE=64` |

**IMPORTANT**: The recovery script replays D05 experiments [1], [7], and [9]:
- **[1] ba_baseline** → `recover_injection_cutouts.py --mode bright_arc` (no Poisson, clip=10, bf [0.1, 0.55])
- **[7] grid_no_poisson** → `recover_injection_cutouts.py --mode grid` (no Poisson)
- **[9] linear_probe** → `recover_injection_cutouts.py --mode linear_probe` (seed=42, n=500, theta_e=1.5, mag=19.0)

Read `run_d05_full_reeval.sh` to verify parameters match.

---

## Context

We are preparing an MNRAS paper titled _"A morphological barrier: quantifying the injection realism gap for CNN strong lens finders in DESI Legacy Survey DR10."_ The paper's central finding is that CNN-based strong lens finders can distinguish injected (simulated) lenses from real lenses in embedding space (Linear Probe AUC = 0.996), implying a "morphological barrier" between sim and real.

**Problem**: During the original experiments, the raw injection cutouts (`.npz` files containing the injected images, injection-only images, and host galaxy images) were **not saved to disk** — only aggregate statistics (completeness curves, detection rates, UMAP embeddings) were retained. We now need visual examples for the paper: a figure showing brightness-matched real Tier-A lenses alongside injected lenses, demonstrating that even when brightness is matched, the CNN can distinguish them.

**Solution**: Because our injection engine is deterministic (seeded RNG, same hosts, same parameters), we can **replay** the exact same injection pipelines to regenerate identical cutouts. Two scripts have been written:

1. **`recover_injection_cutouts.py`** — Replays three original experiment pipelines to regenerate all ~112,600 injection cutouts + metadata.
2. **`find_comparison_candidates.py`** — Selects brightness-matched pairs of real Tier-A lenses and recovered injections, then renders a comparison figure.

---

## What I Need From You

Please review both scripts for **correctness**, **scientific honesty**, and **reproducibility**. I need plain, direct feedback — no rosy picture. If something is wrong or risky, say so clearly.

---

## Script 1: `recover_injection_cutouts.py`

### Purpose
Replays three deterministic injection experiments to recover the exact same cutouts that produced the paper's published results:

| Mode | Original Script | Seed | Injections | Key Parameters |
|------|----------------|------|------------|----------------|
| `grid` | `selection_function_grid.py` | 1337 | ~110,000 (500/cell × 220 cells) | θ_E ∈ [0.5, 3.0], PSF ∈ [0.9, 1.8], depth ∈ [22.5, 24.5] |
| `bright_arc` | `bright_arc_injection_test.py` | 42 | ~1,600 (200 hosts × 8 mag bins) | θ_E=1.5, β_frac ∈ [0.1, 0.55], mag bins 18-26 |
| `linear_probe` | `feature_space_analysis.py` | 42 | 1,000 (500 low_bf + 500 high_bf) | θ_E=1.5, mag=19.0, β_frac=[0.1,0.3] and [0.7,1.0] |

### How Replay Works
For each mode, the script:
1. Initializes `np.random.default_rng(seed)` with the SAME seed as the original.
2. Loads the SAME manifest and filters hosts with the SAME logic.
3. Samples hosts with the SAME `random_state` (pandas) or `rng.choice()` (numpy).
4. Calls `sample_lens_params(rng, ...)` and `sample_source_params(rng, ...)` in the SAME order.
5. Calls `inject_sis_shear(...)` with the SAME per-injection seed.
6. Saves each result as a compressed `.npz` with three arrays: `injected_hwc`, `injection_only_chw`, `host_hwc`.

### Saved Metadata (per injection)
Each row in `metadata.parquet` contains:
- `injection_id`, `experiment`, `cutout_filename`
- `host_cutout_path` (traceable to original training data)
- `theta_e`, `psf_fwhm`, `depth_5sig`
- `cnn_score` (re-scored by the same model)
- `arc_snr`, `source_r_mag`, `lensed_r_mag`
- `source_re`, `source_n_sersic`, `source_flux_r`, `lens_q`, `mag_bin`

### Code (full)

```python
#!/usr/bin/env python3
"""
Recover injection cutouts from deterministic pipeline replay.

Replays the exact injection pipelines (same seeds, same hosts, same params)
that produced the paper's results, but this time saves each injection cutout
as a .npz file with full metadata.

Three modes corresponding to three experiments:
  --mode grid         : Selection function grid (110,000 injections, seed=1337)
  --mode bright_arc   : Bright-arc test (1,600 injections, seed=42)
  --mode linear_probe : Linear probe (1,000 injections, seed=42)

IMPORTANT: The replay logic must match the original scripts EXACTLY to produce
identical injections.  Any deviation in RNG consumption order will produce
different cutouts and invalidate the correspondence with published results.

Original scripts replayed:
  grid        -> selection_function_grid.py (lines 374-570)
  bright_arc  -> bright_arc_injection_test.py (lines 155-260)
  linear_probe -> feature_space_analysis.py (lines 225-300)

Date: 2026-02-16
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dhs.model import build_model
from dhs.preprocess import preprocess_stack
from dhs.scoring_utils import load_model_and_spec
from dhs.constants import CUTOUT_SIZE, STAMP_SIZE
from dhs.injection_engine import (
    AB_ZP,
    LensParams,
    SourceParams,
    inject_sis_shear,
    sample_lens_params,
    sample_source_params,
    estimate_sigma_pix_from_psfdepth,
    arc_annulus_snr,
)
from dhs.selection_function_utils import (
    m5_from_psfdepth,
)

CUTOUT_PATH_COL = "cutout_path"
LABEL_COL = "label"
SPLIT_COL = "split"
TIER_COL = "tier"
PIXEL_SCALE = 0.262

MAGNITUDE_BINS: List[Tuple[float, float]] = [
    (18.0, 19.0), (19.0, 20.0), (20.0, 21.0), (21.0, 22.0),
    (22.0, 23.0), (23.0, 24.0), (24.0, 25.0), (25.0, 26.0),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[nn.Module, str, int, dict]:
    model, pp_kwargs = load_model_and_spec(checkpoint_path, device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    train_cfg = ckpt.get("train", {})
    arch = train_cfg.get("arch", "resnet18")
    epoch = ckpt.get("epoch", -1)
    return model, arch, epoch, pp_kwargs


@torch.no_grad()
def score_one(model: nn.Module, img_chw: np.ndarray, device: torch.device) -> float:
    x = torch.from_numpy(img_chw[None]).float().to(device)
    logit = model(x).squeeze().cpu().item()
    return float(1.0 / (1.0 + np.exp(-logit)))


@torch.no_grad()
def score_batch(model: nn.Module, batch: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(batch).float().to(device)
    logits = model(x).squeeze(1).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))


def nearest_bin(val: float, grid: List[float]) -> float:
    arr = np.asarray(grid)
    return float(arr[np.argmin(np.abs(arr - val))])


def scale_source_to_magnitude(source: SourceParams, target_mag: float) -> SourceParams:
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


def save_injection(out_dir: str, filename: str,
                   injected_chw: np.ndarray,
                   injection_only_chw: np.ndarray,
                   host_hwc: np.ndarray) -> str:
    path = os.path.join(out_dir, filename)
    np.savez_compressed(
        path,
        injected_hwc=np.transpose(injected_chw, (1, 2, 0)),
        injection_only_chw=injection_only_chw,
        host_hwc=host_hwc,
    )
    return path


def lens_params_to_dict(lens: LensParams) -> dict:
    return {f.name: getattr(lens, f.name) for f in dataclasses.fields(lens)}


def source_params_to_dict(source: SourceParams) -> dict:
    return {f.name: getattr(source, f.name) for f in dataclasses.fields(source)}


# ---------------------------------------------------------------------------
# Mode: bright_arc
# ---------------------------------------------------------------------------
def recover_bright_arc(
    checkpoint_path: str,
    manifest_path: str,
    out_dir: str,
    device: torch.device,
) -> pd.DataFrame:
    seed = 42
    theta_e = 1.5
    n_hosts = 200
    beta_frac_range = (0.1, 0.55)

    rng = np.random.default_rng(seed)
    model, arch, epoch, pp_kwargs = load_model_from_checkpoint(checkpoint_path, device)

    df = pd.read_parquet(manifest_path)
    neg = df[(df[SPLIT_COL] == "val") & (df[LABEL_COL] == 0)].copy()
    neg = neg.dropna(subset=["psfsize_r", "psfdepth_r"])
    hosts = neg.sample(n=min(n_hosts, len(neg)), random_state=seed).reset_index(drop=True)

    cutout_dir = os.path.join(out_dir, "cutouts")
    os.makedirs(cutout_dir, exist_ok=True)
    records = []
    total = 0

    for mag_lo, mag_hi in MAGNITUDE_BINS:
        bin_key = f"{mag_lo:.0f}-{mag_hi:.0f}"
        for i, (_, host_row) in enumerate(hosts.iterrows()):
            try:
                with np.load(str(host_row[CUTOUT_PATH_COL])) as z:
                    hwc = z["cutout"].astype(np.float32)
            except Exception:
                continue

            host_t = torch.from_numpy(hwc).float()
            host_psf = float(host_row["psfsize_r"])
            host_psfdepth = float(host_row["psfdepth_r"])

            lens = sample_lens_params(rng, theta_e)
            source = sample_source_params(rng, theta_e, beta_frac_range=beta_frac_range)
            target_mag = float(rng.uniform(mag_lo, mag_hi))
            source = scale_source_to_magnitude(source, target_mag)

            result = inject_sis_shear(
                host_nmgy_hwc=host_t, lens=lens, source=source,
                pixel_scale=PIXEL_SCALE, psf_fwhm_r_arcsec=host_psf,
                core_suppress_radius_pix=None, seed=seed + i,
            )

            inj_chw = result.injected[0].numpy()
            inj_only_chw = result.injection_only[0].numpy()
            proc = preprocess_stack(inj_chw, **pp_kwargs)
            cnn_score = score_one(model, proc, device)

            sigma_pix_r = estimate_sigma_pix_from_psfdepth(host_psfdepth, host_psf, PIXEL_SCALE)
            snr = arc_annulus_snr(result.injection_only[0], sigma_pix_r)
            total_lensed_flux_r = float(inj_only_chw[1].sum())
            lensed_r_mag = (AB_ZP - 2.5 * math.log10(total_lensed_flux_r)
                           if total_lensed_flux_r > 0 else float("nan"))

            filename = f"bright_arc_{bin_key}_host{i:04d}.npz"
            save_injection(cutout_dir, filename, inj_chw, inj_only_chw, hwc)
            records.append({
                "injection_id": f"bright_arc_{bin_key}_{i:04d}",
                "experiment": "bright_arc",
                "cutout_filename": filename,
                "host_cutout_path": str(host_row[CUTOUT_PATH_COL]),
                "theta_e": theta_e, "psf_fwhm": host_psf,
                "depth_5sig": float("nan"),
                "cnn_score": cnn_score, "arc_snr": float(snr),
                "source_r_mag": target_mag, "lensed_r_mag": lensed_r_mag,
                "beta_frac": float(getattr(source, "beta_x_arcsec", 0) / theta_e)
                             if theta_e > 0 else float("nan"),
                "source_re": source.re_arcsec,
                "source_n_sersic": source.n_sersic,
                "source_flux_r": source.flux_nmgy_r,
                "lens_q": lens.q_lens, "mag_bin": bin_key,
            })
            total += 1

    meta_df = pd.DataFrame(records)
    meta_df.to_parquet(os.path.join(out_dir, "metadata.parquet"), index=False)
    return meta_df


# ---------------------------------------------------------------------------
# Mode: linear_probe
# ---------------------------------------------------------------------------
def recover_linear_probe(
    checkpoint_path: str,
    manifest_path: str,
    out_dir: str,
    device: torch.device,
) -> pd.DataFrame:
    seed = 42
    theta_e = 1.5
    target_mag = 19.0
    n_samples = 500

    rng = np.random.default_rng(seed)
    model, pp_kwargs = load_model_and_spec(checkpoint_path, device)

    df = pd.read_parquet(manifest_path)
    val_df = df[df["split"] == "val"].copy()
    neg_df = val_df[val_df["label"] == 0].dropna(subset=["psfsize_r", "psfdepth_r"])
    hosts_df = neg_df.sample(n=min(2000, len(neg_df)), random_state=seed + 1)

    cutout_dir = os.path.join(out_dir, "cutouts")
    os.makedirs(cutout_dir, exist_ok=True)
    records = []
    total = 0

    for cat_name, bf_range, seed_offset in [
        ("low_bf", (0.1, 0.3), 0),
        ("high_bf", (0.7, 1.0), 10000),
    ]:
        host_indices = rng.choice(len(hosts_df), size=n_samples, replace=True)
        for i, host_idx in enumerate(host_indices):
            host_row = hosts_df.iloc[host_idx]
            try:
                with np.load(str(host_row["cutout_path"])) as z:
                    hwc = z["cutout"].astype(np.float32)
            except Exception:
                continue

            host_t = torch.from_numpy(hwc).float()
            host_psf = float(host_row["psfsize_r"])

            lens = sample_lens_params(rng, theta_e)
            source = sample_source_params(rng, theta_e, beta_frac_range=bf_range)

            target_flux = 10.0 ** ((22.5 - target_mag) / 2.5)
            if source.flux_nmgy_r > 0:
                scale = target_flux / source.flux_nmgy_r
            else:
                scale = 1.0
            fields = {f.name: getattr(source, f.name) for f in dataclasses.fields(source)}
            fields["flux_nmgy_r"] = source.flux_nmgy_r * scale
            fields["flux_nmgy_g"] = source.flux_nmgy_g * scale
            fields["flux_nmgy_z"] = source.flux_nmgy_z * scale
            source = SourceParams(**fields)

            result = inject_sis_shear(
                host_nmgy_hwc=host_t, lens=lens, source=source,
                pixel_scale=PIXEL_SCALE, psf_fwhm_r_arcsec=host_psf,
                seed=42 + seed_offset + i,
            )

            inj_chw = result.injected[0].numpy()
            inj_only_chw = result.injection_only[0].numpy()
            proc = preprocess_stack(inj_chw, **pp_kwargs)
            cnn_score = score_one(model, proc, device)
            total_lensed_flux_r = float(inj_only_chw[1].sum())
            lensed_r_mag = (AB_ZP - 2.5 * math.log10(total_lensed_flux_r)
                           if total_lensed_flux_r > 0 else float("nan"))

            filename = f"linear_probe_{cat_name}_{i:04d}.npz"
            save_injection(cutout_dir, filename, inj_chw, inj_only_chw, hwc)
            records.append({
                "injection_id": f"linear_probe_{cat_name}_{i:04d}",
                "experiment": "linear_probe", "cutout_filename": filename,
                "host_cutout_path": str(host_row["cutout_path"]),
                "theta_e": theta_e, "psf_fwhm": host_psf,
                "depth_5sig": float("nan"),
                "cnn_score": cnn_score, "arc_snr": float("nan"),
                "source_r_mag": target_mag, "lensed_r_mag": lensed_r_mag,
                "beta_frac": float("nan"),
                "source_re": source.re_arcsec,
                "source_n_sersic": source.n_sersic,
                "source_flux_r": source.flux_nmgy_r,
                "lens_q": lens.q_lens, "mag_bin": cat_name,
            })
            total += 1

    meta_df = pd.DataFrame(records)
    meta_df.to_parquet(os.path.join(out_dir, "metadata.parquet"), index=False)
    return meta_df


# ---------------------------------------------------------------------------
# Mode: grid
# ---------------------------------------------------------------------------
def recover_grid(
    checkpoint_path: str,
    manifest_path: str,
    out_dir: str,
    device: torch.device,
) -> pd.DataFrame:
    seed = 1337
    host_max = 20000
    injections_per_cell = 500
    MAX_RETRIES = 5
    preprocessing = "raw_robust"
    crop = False

    theta_e_min, theta_e_max, theta_e_step = 0.5, 3.0, 0.25
    psf_min, psf_max, psf_step = 0.9, 1.8, 0.15
    depth_min, depth_max, depth_step = 22.5, 24.5, 0.5

    model, arch, epoch, pp_kwargs = load_model_from_checkpoint(checkpoint_path, device)

    df = pd.read_parquet(manifest_path)
    hosts = df[(df[SPLIT_COL] == "val") & (df[LABEL_COL] == 0)].copy()

    rng = np.random.default_rng(seed)
    if len(hosts) > host_max:
        hosts = hosts.sample(n=host_max, random_state=seed).reset_index(drop=True)

    theta_es = np.round(np.arange(theta_e_min, theta_e_max + 1e-9, theta_e_step), 4).tolist()
    psf_bins = np.round(np.arange(psf_min, psf_max + 1e-9, psf_step), 4).tolist()
    depth_bins = np.round(np.arange(depth_min, depth_max + 1e-9, depth_step), 4).tolist()

    depth_mag_arr = m5_from_psfdepth(hosts["psfdepth_r"].to_numpy(dtype=np.float64))
    hosts["depth_mag"] = depth_mag_arr
    hosts["psf_bin"] = hosts["psfsize_r"].apply(lambda v: nearest_bin(float(v), psf_bins))
    hosts["depth_bin"] = hosts["depth_mag"].apply(lambda v: nearest_bin(float(v), depth_bins))

    host_groups: Dict[Tuple[float, float], pd.DataFrame] = {}
    for (pb, db), g in hosts.groupby(["psf_bin", "depth_bin"]):
        host_groups[(float(pb), float(db))] = g

    cutout_dir = os.path.join(out_dir, "cutouts")
    os.makedirs(cutout_dir, exist_ok=True)
    records = []
    cell_idx = 0
    total = 0
    t0 = time.time()

    for theta_e in theta_es:
        for pb in psf_bins:
            for db in depth_bins:
                cell_idx += 1
                host_df = host_groups.get((pb, db))
                if host_df is None or len(host_df) == 0:
                    continue

                n_target = injections_per_cell
                n_saved_this_cell = 0
                attempt = 0

                while n_saved_this_cell < n_target and attempt < n_target + n_target * MAX_RETRIES:
                    attempt += 1
                    host_idx = int(rng.choice(len(host_df)))
                    host_row = host_df.iloc[host_idx]
                    try:
                        with np.load(str(host_row[CUTOUT_PATH_COL])) as z:
                            host_hwc = z["cutout"].astype(np.float32)

                        host_hwc_torch = torch.from_numpy(host_hwc).float()
                        host_psf = float(host_row["psfsize_r"])
                        host_psfdepth = float(host_row["psfdepth_r"])

                        lens = sample_lens_params(rng, theta_e)
                        source = sample_source_params(rng, theta_e)

                        src_r_mag = float("nan")
                        if source.flux_nmgy_r > 0:
                            src_r_mag = AB_ZP - 2.5 * math.log10(source.flux_nmgy_r)

                        inj_seed = seed + cell_idx * (n_target + n_target * MAX_RETRIES) + attempt
                        result = inject_sis_shear(
                            host_nmgy_hwc=host_hwc_torch, lens=lens, source=source,
                            pixel_scale=PIXEL_SCALE, psf_fwhm_r_arcsec=host_psf,
                            seed=inj_seed,
                        )

                        sigma_pix_r = estimate_sigma_pix_from_psfdepth(
                            host_psfdepth, host_psf, PIXEL_SCALE)
                        injection_chw = result.injection_only[0]
                        snr_val = arc_annulus_snr(injection_chw, sigma_pix_r)
                        total_lensed_flux_r = float(injection_chw[1].sum().item())
                        lensed_r_mag = (AB_ZP - 2.5 * math.log10(total_lensed_flux_r)
                                       if total_lensed_flux_r > 0 else float("nan"))

                        inj_chw = result.injected[0].numpy()
                        inj_only_chw = injection_chw.numpy()

                        _pp_inj = dict(pp_kwargs or {})
                        _pp_inj.setdefault("mode", preprocessing)
                        _pp_inj.setdefault("crop", crop)
                        _pp_inj.setdefault("clip_range", 10.0)
                        proc = preprocess_stack(inj_chw, **_pp_inj)
                        cnn_score = score_one(model, proc, device)

                        filename = f"grid_c{cell_idx:04d}_i{n_saved_this_cell:04d}.npz"
                        save_injection(cutout_dir, filename, inj_chw, inj_only_chw, host_hwc)
                        records.append({
                            "injection_id": f"grid_c{cell_idx:04d}_i{n_saved_this_cell:04d}",
                            "experiment": "grid",
                            "cutout_filename": filename,
                            "host_cutout_path": str(host_row[CUTOUT_PATH_COL]),
                            "theta_e": theta_e, "psf_fwhm": pb,
                            "depth_5sig": db, "cnn_score": cnn_score,
                            "arc_snr": float(snr_val),
                            "source_r_mag": src_r_mag,
                            "lensed_r_mag": lensed_r_mag,
                            "beta_frac": float("nan"),
                            "source_re": source.re_arcsec,
                            "source_n_sersic": source.n_sersic,
                            "source_flux_r": source.flux_nmgy_r,
                            "lens_q": lens.q_lens,
                            "mag_bin": f"theta_e={theta_e}",
                        })
                        n_saved_this_cell += 1
                        total += 1
                    except Exception:
                        pass

    meta_df = pd.DataFrame(records)
    meta_df.to_parquet(os.path.join(out_dir, "metadata.parquet"), index=False)
    return meta_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Recover injection cutouts from deterministic replay")
    ap.add_argument("--mode", required=True, choices=["grid", "bright_arc", "linear_probe"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()

    if args.mode == "bright_arc":
        meta = recover_bright_arc(args.checkpoint, args.manifest, args.out_dir, device)
    elif args.mode == "linear_probe":
        meta = recover_linear_probe(args.checkpoint, args.manifest, args.out_dir, device)
    elif args.mode == "grid":
        meta = recover_grid(args.checkpoint, args.manifest, args.out_dir, device)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    elapsed = time.time() - t_start
    summary = {
        "mode": args.mode, "checkpoint": args.checkpoint,
        "manifest": args.manifest, "n_recovered": len(meta),
        "elapsed_seconds": elapsed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(args.out_dir, "recovery_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
```

---

## Script 2: `find_comparison_candidates.py`

### Purpose
Takes the recovered injection metadata and the real Tier-A lens catalog, selects brightness-matched pairs, and renders a comparison figure suitable for the MNRAS paper.

### Honesty Constraints
1. **No cherry-picking**: Real Tier-A lenses are selected with a fixed random seed.
2. **No fabrication**: All injected cutouts come from the recovery pipeline (traceable to original experiment parameters and seeds).
3. **Transparent matching**: Pairs are matched by host galaxy r-band integrated magnitude, binned into N brightness bins (default 5).
4. **Full audit trail**: Every selection criterion is logged to `comparison_audit.json`.
5. **Selected cutouts are copied** to a separate directory for GitHub release.

### Matching Logic
1. Compute integrated r-band magnitude for all real Tier-A and injection host galaxies.
2. Divide the magnitude range (5th–95th percentile of real Tier-A) into N bins.
3. For each bin, deterministically sample `n_per_bin` real lenses.
4. For each real lens, find the injection with the closest host magnitude in that bin.
5. Record: brightness bin, cutout paths, magnitudes, CNN score, θ_E, lensed r-mag, arc SNR, magnitude difference.

### Code (full)

```python
#!/usr/bin/env python3
"""
Find best real-vs-injection comparison candidates for the MNRAS paper figure.

Reads:
  - Real Tier-A cutouts and metadata from the training manifest.
  - Recovered injection metadata/cutouts from recover_injection_cutouts.py.

Selects top brightness-matched pairs of real Tier-A lenses and injections,
then renders a publication-quality comparison figure.

HONESTY CONSTRAINTS:
  1. Real Tier-A lenses are deterministically selected by a fixed seed.
  2. Injections are deterministically reproduced from original experiment params/seeds.
  3. The figure caption and an audit JSON log explicitly document selection criteria.
  4. All cutout files are traceable back to original pipeline runs.

Date: 2026-02-16
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CUTOUT_PATH_COL = "cutout_path"
AB_ZP = 22.5


def host_rmag_from_cutout(cutout_path: str) -> float:
    try:
        with np.load(cutout_path) as z:
            hwc = z["cutout"].astype(np.float64)
        r_flux = hwc[:, :, 1].sum()
        if r_flux > 0:
            return AB_ZP - 2.5 * np.log10(r_flux)
        return float("nan")
    except Exception:
        return float("nan")


def load_cutout_rgb(path: str, key: str = "cutout") -> np.ndarray:
    with np.load(path) as z:
        if key in z:
            hwc = z[key].astype(np.float64)
        elif "injected_hwc" in z:
            hwc = z["injected_hwc"].astype(np.float64)
        else:
            raise KeyError(f"Neither '{key}' nor 'injected_hwc' found in {path}")

    for c in range(hwc.shape[2]):
        band = hwc[:, :, c]
        med = np.median(band)
        band = band - med
        vmax = np.percentile(np.abs(band), 99.5)
        if vmax > 0:
            band = np.arcsinh(band / (0.1 * vmax)) / np.arcsinh(1.0 / 0.1)
        hwc[:, :, c] = band

    vlo, vhi = np.percentile(hwc, [0.5, 99.5])
    if vhi > vlo:
        hwc = (hwc - vlo) / (vhi - vlo)
    hwc = np.clip(hwc, 0, 1)
    return hwc


def brightness_match(
    real_df: pd.DataFrame,
    inj_df: pd.DataFrame,
    n_bins: int = 5,
    n_per_bin: int = 2,
    seed: int = 42,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    real_mags = real_df["host_rmag"].dropna()
    lo, hi = np.percentile(real_mags, [5, 95])
    bin_edges = np.linspace(lo, hi, n_bins + 1)

    pairs = []
    for i in range(n_bins):
        mag_lo, mag_hi = bin_edges[i], bin_edges[i + 1]
        bin_label = f"{mag_lo:.1f}-{mag_hi:.1f}"
        real_in_bin = real_df[
            (real_df["host_rmag"] >= mag_lo) & (real_df["host_rmag"] < mag_hi)]
        inj_in_bin = inj_df[
            (inj_df["host_rmag"] >= mag_lo) & (inj_df["host_rmag"] < mag_hi)]
        if len(real_in_bin) == 0 or len(inj_in_bin) == 0:
            continue
        real_sample = real_in_bin.sample(
            n=min(n_per_bin, len(real_in_bin)), random_state=seed + i)
        for _, real_row in real_sample.iterrows():
            mag_diffs = np.abs(inj_in_bin["host_rmag"].values - real_row["host_rmag"])
            best_idx = np.argmin(mag_diffs)
            inj_row = inj_in_bin.iloc[best_idx]
            pairs.append({
                "brightness_bin": bin_label,
                "real_cutout_path": str(real_row[CUTOUT_PATH_COL]),
                "real_host_rmag": float(real_row["host_rmag"]),
                "real_tier": str(real_row.get("tier", "A")),
                "inj_cutout_filename": str(inj_row["cutout_filename"]),
                "inj_host_rmag": float(inj_row["host_rmag"]),
                "inj_cnn_score": float(inj_row["cnn_score"]),
                "inj_theta_e": float(inj_row["theta_e"]),
                "inj_lensed_r_mag": float(inj_row.get("lensed_r_mag", float("nan"))),
                "inj_arc_snr": float(inj_row.get("arc_snr", float("nan"))),
                "mag_diff": float(mag_diffs[best_idx]),
            })
    return pairs


def render_figure(
    pairs: List[Dict], inj_cutouts_dir: str, out_path: str, n_cols: int = 2,
) -> None:
    n_pairs = len(pairs)
    if n_pairs == 0:
        return
    fig, axes = plt.subplots(n_pairs, n_cols, figsize=(4 * n_cols, 3.8 * n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    for i, pair in enumerate(pairs):
        try:
            real_rgb = load_cutout_rgb(pair["real_cutout_path"], key="cutout")
            axes[i, 0].imshow(real_rgb, origin="lower")
        except Exception as e:
            axes[i, 0].text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center",
                            transform=axes[i, 0].transAxes, fontsize=7)
        axes[i, 0].set_title(f"Real Tier-A  r={pair['real_host_rmag']:.1f}", fontsize=9)
        axes[i, 0].axis("off")

        inj_path = os.path.join(inj_cutouts_dir, pair["inj_cutout_filename"])
        try:
            inj_rgb = load_cutout_rgb(inj_path, key="injected_hwc")
            axes[i, 1].imshow(inj_rgb, origin="lower")
        except Exception as e:
            axes[i, 1].text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center",
                            transform=axes[i, 1].transAxes, fontsize=7)
        score_str = f"CNN={pair['inj_cnn_score']:.3f}" if not np.isnan(pair["inj_cnn_score"]) else ""
        axes[i, 1].set_title(
            f"Injected  r={pair['inj_host_rmag']:.1f}  "
            f"$\\theta_E$={pair['inj_theta_e']:.2f}\"  {score_str}", fontsize=9)
        axes[i, 1].axis("off")

    fig.suptitle("Brightness-Matched Real vs Injected Lenses\n(arcsinh stretch, g-r-z composite)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Find brightness-matched real-vs-injection comparison candidates")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--injection-meta", required=True)
    ap.add_argument("--injection-cutouts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-pairs", type=int, default=10)
    ap.add_argument("--n-brightness-bins", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_parquet(args.manifest)
    tier_a = df[
        (df["label"] == 1)
        & (df.get("tier", pd.Series(dtype=str)).str.upper() == "A")
    ].copy()
    if len(tier_a) == 0:
        tier_a = df[df["label"] == 1].copy()

    tier_a["host_rmag"] = tier_a[CUTOUT_PATH_COL].apply(host_rmag_from_cutout)
    tier_a = tier_a.dropna(subset=["host_rmag"])

    inj_meta = pd.read_parquet(args.injection_meta)
    if "host_rmag" not in inj_meta.columns:
        inj_meta["host_rmag"] = inj_meta["host_cutout_path"].apply(host_rmag_from_cutout)
    inj_meta = inj_meta.dropna(subset=["host_rmag"])

    n_per_bin = max(1, args.n_pairs // args.n_brightness_bins)
    pairs = brightness_match(tier_a, inj_meta, n_bins=args.n_brightness_bins,
                              n_per_bin=n_per_bin, seed=args.seed)

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": args.manifest, "injection_meta": args.injection_meta,
        "n_real_tier_a": len(tier_a), "n_injections": len(inj_meta),
        "n_brightness_bins": args.n_brightness_bins,
        "n_per_bin": n_per_bin, "seed": args.seed, "pairs": pairs,
    }
    with open(os.path.join(args.out_dir, "comparison_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    fig_path = os.path.join(args.out_dir, "comparison_real_vs_injected.pdf")
    render_figure(pairs, args.injection_cutouts, fig_path)
    fig_png = fig_path.replace(".pdf", ".png")
    render_figure(pairs, args.injection_cutouts, fig_png)

    import shutil
    selected_dir = os.path.join(args.out_dir, "selected_cutouts")
    os.makedirs(selected_dir, exist_ok=True)
    for pair in pairs:
        real_name = os.path.basename(pair["real_cutout_path"])
        dst_real = os.path.join(selected_dir, f"real_{real_name}")
        if os.path.exists(pair["real_cutout_path"]):
            shutil.copy2(pair["real_cutout_path"], dst_real)
        inj_src = os.path.join(args.injection_cutouts, pair["inj_cutout_filename"])
        dst_inj = os.path.join(selected_dir, f"inj_{pair['inj_cutout_filename']}")
        if os.path.exists(inj_src):
            shutil.copy2(inj_src, dst_inj)

if __name__ == "__main__":
    main()
```

---

## Original Scripts (for cross-reference)

The recovery script replays these three original pipelines. Key excerpts:

### 1. `selection_function_grid.py` — Grid injection loop (lines 462-570)

```python
# Original RNG consumption order (per attempt):
host_idx = int(rng.choice(len(host_df)))         # 1 draw
lens = sample_lens_params(rng, theta_e, ...)      # N draws (internal)
source = sample_source_params(rng, theta_e, ...)  # M draws (internal)
# Then inject_sis_shear with deterministic seed:
inj_seed = seed + cell_idx * (n_target + n_target * MAX_RETRIES) + attempt
```

Note: the original grid passes `q_lens_range_override=None`, `re_scale=1.0`, `gmr_shift=0.0`, `rmz_shift=0.0` (all defaults) for the D05 base run. The recovery script omits these kwargs, relying on default parameter values.

### 2. `bright_arc_injection_test.py` — Per-host injection (lines 193-230)

```python
# Original RNG consumption order (per host, per mag bin):
lens = sample_lens_params(rng, theta_e)            # N draws
source = sample_source_params(rng, theta_e, **bf_kwargs)  # M draws
target_mag = float(rng.uniform(mag_lo, mag_hi))    # 1 draw
source = scale_source_to_magnitude(source, target_mag)
# inject_sis_shear with seed = seed + i
```

### 3. `feature_space_analysis.py` — Linear probe injections (lines 250-285)

```python
# Original RNG consumption:
host_indices = rng.choice(len(hosts_df), size=n_samples, replace=True)  # bulk draw
# Then per injection:
lens = sample_lens_params(rng, theta_e)
source = sample_source_params(rng, theta_e, beta_frac_range=bf_range)
# inject_sis_shear with seed = 42 + seed_offset + i
```

---

## Detailed Questions for Review

Please answer each question directly (YES/NO first, then explanation):

### RNG Replay Fidelity

**Q1.** In `recover_grid`, the original `selection_function_grid.py` calls `sample_lens_params(rng, theta_e, q_lens_range_override=None)` and `sample_source_params(rng, theta_e, re_scale=1.0, gmr_shift=0.0, rmz_shift=0.0)` for the D05 base run (no sensitivity overrides). The recovery script calls `sample_lens_params(rng, theta_e)` and `sample_source_params(rng, theta_e)` without these kwargs. **Does omitting explicit default-valued kwargs change the RNG consumption order?** Inspect `sample_lens_params` and `sample_source_params` in the attached `injection_engine.py` — specifically verify that `q_lens_range_override=None` results in the same 6 RNG draws as omitting it, and that `re_scale=1.0`, `gmr_shift=0.0`, `rmz_shift=0.0` result in the same 10-12 RNG draws as omitting them. **Pay special attention to the conditional clump draws (#10-#12).**

**Q2.** In `recover_bright_arc`, the original passes `add_poisson_noise=False` (D05 experiment [1]) to `inject_sis_shear`. The recovery script does not pass `add_poisson_noise`. **Inspect `inject_sis_shear` in the attached `injection_engine.py`**: verify that `add_poisson_noise=False` (the default) does not consume any RNG draws from the `torch.Generator` or the global PyTorch RNG. Also check: `torch.poisson` in the Poisson branch — does it use the seeded `rng` generator, or the global RNG? (This matters for any future Poisson-mode recovery.)

**Q3.** In `recover_bright_arc`, host selection is `neg.sample(n=min(200, len(neg)), random_state=seed)`. The original uses the same call. **But what happens if the manifest has changed since the original run?** Could row ordering differ, producing different host selections? Is there a way to verify that the manifest is byte-identical?

**Q4.** In `recover_linear_probe`, the original `feature_space_analysis.py` creates an `EmbeddingExtractor(model)` and calls `extractor.register_layer_hooks()` before any injections. **Could the model hook registration change model behavior or RNG state in a way that affects `inject_sis_shear` outputs?** (Hook registration shouldn't affect weight computation, but confirm.)

**Q5.** In `recover_grid`, when a cell has no hosts (`host_df is None`), the original code writes empty rows and `continue`s without consuming the `rng`. The recovery code also `continue`s. **Is this correct? Does the original code consume any RNG draws for empty cells (e.g., in `thr_to_fpr` computation)?**

**Q5b.** The D05 run script (`run_d05_full_reeval.sh`) runs 10 experiments. The recovery script targets experiments [1] (ba_baseline), [7] (grid_no_poisson), and [9] (linear_probe). **Cross-check the exact CLI arguments in `run_d05_full_reeval.sh` against the hardcoded parameters in `recover_injection_cutouts.py`.** Are there any discrepancies in seed, beta_frac_range, theta_e, n_hosts, injections_per_cell, host_max, or any other parameter? List them all.

### Error Handling & Edge Cases

**Q6.** In `recover_bright_arc`, when `np.load()` fails (host .npz missing), the loop does `continue`, skipping that host but **still advancing the host iterator `i`**. The original code does the same. **But critically: the `rng` has already consumed draws for `sample_lens_params`, `sample_source_params`, and `rng.uniform` for this host. If the .npz load fails BEFORE those calls, the RNG state diverges.** Look at lines 204-216 of the recovery: the `rng` calls happen AFTER the `np.load()` try block. Is the `except: continue` in the right place?

**Q7.** In `recover_grid`, the `except Exception: pass` swallows all errors silently. This matches the original (which also silently retries). **But if a previously-working cutout file has been deleted from NFS since the original run, the recovery will silently skip it AND consume different RNG draws (because the rng.choice + sample_lens + sample_source still advance the RNG).** Is this a real risk? How should we verify that the NFS cutouts are intact?

**Q8.** The recovery script saves `injected_hwc` by transposing CHW -> HWC. The comparison script reads it back with key `"injected_hwc"`. **Is the transpose correct?** The original cutouts are stored as HWC — is `result.injected[0]` in CHW format? (It's a torch tensor from `inject_sis_shear`.)

### Comparison Figure Honesty

**Q9.** The brightness matching uses **host galaxy r-band integrated magnitude** (`sum of all pixels in r-band`). For real Tier-A lenses, this includes the lensing arc flux. For injection hosts, this is the non-lensed galaxy flux. **Is this an apples-to-apples comparison?** Should we use the same quantity for both (e.g., match on the injected image's total flux, not the host-only flux)?

**Q10.** The `load_cutout_rgb` function applies an arcsinh stretch with per-band median subtraction and 99.5th percentile scaling. **Is this stretch identical to what we show elsewhere in the paper (e.g., in other figures)?** If not, a reviewer could argue the visual comparison is unfairly stretched.

**Q11.** The comparison script matches real lenses to the **single closest injection by host magnitude**. This means the same injection could be matched to multiple real lenses (if multiple reals have similar magnitudes). **Is this a problem?** Should we enforce unique matching (once an injection is used, remove it from the pool)?

**Q12.** The figure currently shows only the host r-band magnitude and θ_E for injections. **Should the figure or caption also show the CNN score for the real Tier-A lenses?** This would directly demonstrate: "real lens scored X, injection with similar brightness scored Y."

### Storage & Performance

**Q13.** The grid recovery will produce ~110,000 compressed .npz files. Each contains three arrays: injected_hwc (101×101×3 float32 = 122 KB), injection_only_chw (3×101×101 float32 = 122 KB), host_hwc (101×101×3 float32 = 122 KB). **Estimated total size: ~110K × 0.1 MB (compressed) ≈ 11 GB.** Is this acceptable for the S3 budget? Could we save only the injection-only and injected arrays (skip host_hwc since it's already in the training data)?

**Q14.** The grid recovery will take significant time due to ~110K individual `inject_sis_shear` calls plus CNN scoring. **Rough estimate: at ~100 injections/second on GPU, this is ~18 minutes.** Is this estimate reasonable? Should we add checkpointing (resume from cell N if interrupted)?

### Scientific Integrity

**Q15.** The paper claims a Linear Probe AUC of 0.996 between real and injected embeddings. The recovered injections will be re-scored by the CNN. **If the re-scored CNN values differ from the original experiment's values (due to ANY of the RNG issues above), does that invalidate the figure?** Or is the figure independent of the original scores (since we're showing visual similarity, not claiming score equivalence)?

**Q16.** The comparison figure will show that visually similar images (real vs. injected) are treated differently by the CNN. **But the paper's argument is about the EMBEDDING space, not the IMAGE space.** Is a visual comparison figure actually the right thing for the paper? Would a t-SNE/UMAP figure (which we already have) be more convincing? Or do we need BOTH?

---

## Summary of What I Need

1. Direct YES/NO answers to Q1–Q16.
2. For any NO or UNSURE: specific code fixes or verification steps.
3. Any additional bugs, risks, or improvements you see that I haven't asked about.
4. Your overall assessment: **Is this recovery approach scientifically sound for an MNRAS paper?**

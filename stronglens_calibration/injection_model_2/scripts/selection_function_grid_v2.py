#!/usr/bin/env python3
"""
Selection Function Grid v2: Model 1 + Model 2 Support.

Extends the fixed selection_function_grid.py with Model 2 (deflector-conditioned)
injection support.

Model 1 (baseline):
  - Random negative hosts (all morphology types)
  - Independent lens params: q ~ U(0.5, 1.0), phi ~ U(0, pi), shear ~ N(0, 0.05)
  - Parametric Sersic source

Model 2 (deflector-conditioned):
  - LRG-like hosts only (DEV/SER Tractor morphology)
  - Lens q and PA conditioned on host light second moments
  - Small external shear ~ U(0, 0.08)
  - Same parametric Sersic source (Model 3 would use real source stamps)
  - Uses our validated SIE+shear engine (dhs.injection_engine)

Changes from v1:
  - --model {1,2} flag to select injection model
  - Model 2: LRG host selection, host-conditioned lens parameters
  - All v1 bug fixes: retry on failure, explicit failure counting, correct FPR denominators
  - Output includes injection_model column for comparison
  - host_moments_q, host_moments_phi columns for Model 2 diagnostics

Usage:
    cd /lambda/nfs/.../code
    export PYTHONPATH=.:stronglens_calibration/injection_model_2

    # Model 1 (baseline, same as v1)
    python stronglens_calibration/injection_model_2/scripts/selection_function_grid_v2.py \
        --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
        --manifest manifests/training_parity_70_30_v1.parquet \
        --model 1 \
        --out-dir results/selection_function_model1_v2

    # Model 2 (deflector-conditioned)
    python stronglens_calibration/injection_model_2/scripts/selection_function_grid_v2.py \
        --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
        --manifest manifests/training_parity_70_30_v1.parquet \
        --model 2 \
        --out-dir results/selection_function_model2

Author: stronglens_calibration project
Date: 2026-02-13
References:
  - MNRAS_RAW_NOTES.md Section 9.3
  - Paper IV (Inchausti et al. 2025)
"""
from __future__ import annotations

import argparse
import json
import logging
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

# dhs imports (core engine, model, preprocessing)
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
    bayes_binomial_interval,
)
from dhs.s3io import is_s3_uri, join_uri, write_bytes

# Model 2 imports (host matching and selection)
# These are in the injection_model_2 package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from host_matching import estimate_host_moments_rband, map_host_to_lens_params
from host_selection import select_lrg_hosts, select_random_hosts, host_selection_summary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CUTOUT_PATH_COL = "cutout_path"
LABEL_COL = "label"
SPLIT_COL = "split"
TYPE_BIN_COL = "type_bin"

PIXEL_SCALE = 0.262  # arcsec/pixel (DESI Legacy Survey)


# ---------------------------------------------------------------------------
# Model loading and scoring (identical to v1)
# ---------------------------------------------------------------------------
def load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[nn.Module, str, int, dict]:
    """Load model + preprocessing kwargs, return (model, arch, epoch, pp_kwargs).

    Uses scoring_utils.load_model_and_spec to ensure preprocessing
    parameters match what the model was trained with.
    """
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
def score_batch(model: nn.Module, batch: np.ndarray, device: torch.device) -> np.ndarray:
    """Score a batch of preprocessed images. Returns sigmoid probabilities."""
    x = torch.from_numpy(batch).float().to(device)
    logits = model(x).squeeze(1).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))


# ---------------------------------------------------------------------------
# Fixed-FPR threshold derivation (with v1 bug fixes)
# ---------------------------------------------------------------------------
@torch.no_grad()
def derive_fpr_thresholds(
    model: nn.Module,
    manifest_df: pd.DataFrame,
    split: str,
    target_fprs: List[float],
    preprocessing: str,
    crop: bool,
    device: torch.device,
    max_negatives: int = 50000,
    batch_size: int = 256,
    seed: int = 1337,
    pp_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[float, float]:
    """Derive detection thresholds from a fixed False Positive Rate.

    Bug fixes from v1:
    - Failed loads are excluded (not scored as zeros)
    - FPR rank uses N_valid, not N_attempted

    If pp_kwargs is provided, uses those instead of preprocessing/crop.
    """
    neg = manifest_df[
        (manifest_df[SPLIT_COL] == split) & (manifest_df[LABEL_COL] == 0)
    ]
    if neg.empty:
        raise ValueError(f"No negatives in split '{split}' for FPR derivation")

    if len(neg) > max_negatives:
        neg = neg.sample(n=max_negatives, random_state=seed).reset_index(drop=True)

    n_neg = len(neg)
    input_size = CUTOUT_SIZE if not crop else STAMP_SIZE
    print(f"\nFPR threshold derivation: scoring {n_neg:,} negatives from '{split}'")

    all_scores = []
    n_load_ok = 0
    n_load_failed = 0

    for start in range(0, n_neg, batch_size):
        end = min(start + batch_size, n_neg)
        batch_size_actual = end - start
        batch_arr = np.zeros((batch_size_actual, 3, input_size, input_size), dtype=np.float32)
        valid_mask = np.zeros(batch_size_actual, dtype=bool)

        for j, (_, row) in enumerate(neg.iloc[start:end].iterrows()):
            try:
                with np.load(str(row[CUTOUT_PATH_COL])) as z:
                    img = z["cutout"].astype(np.float32)
                if img.ndim == 3 and img.shape[-1] == 3:
                    img = np.transpose(img, (2, 0, 1))
                if pp_kwargs is not None:
                    proc = preprocess_stack(img, **pp_kwargs)
                else:
                    proc = preprocess_stack(img, mode=preprocessing, crop=crop, clip_range=10.0)
                batch_arr[j] = proc
                valid_mask[j] = True
                n_load_ok += 1
            except Exception as exc:
                n_load_failed += 1
                if n_load_failed <= 10:
                    print(f"  WARNING: FPR load failed ({n_load_failed}): "
                          f"{row.get(CUTOUT_PATH_COL, '?')}: {exc}")

        scores = score_batch(model, batch_arr, device)
        all_scores.append(scores[valid_mask])
        if (start // batch_size) % 10 == 0:
            print(f"  Scored {end}/{n_neg}", end="\r")

    all_scores = np.concatenate(all_scores)
    n_valid = len(all_scores)
    print(f"\n  Scored {n_valid:,} valid negatives "
          f"(loaded: {n_load_ok:,}, failed: {n_load_failed:,}). "
          f"Score range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")

    if n_load_failed > 0:
        fail_pct = 100.0 * n_load_failed / n_neg
        print(f"  WARNING: {n_load_failed:,} load failures ({fail_pct:.2f}%)")

    if n_valid == 0:
        raise RuntimeError("All negative loads failed")

    sorted_scores = np.sort(all_scores)[::-1]
    fpr_to_threshold = {}
    for fpr in target_fprs:
        k = max(1, int(np.ceil(fpr * n_valid)))
        if k >= n_valid:
            tau = 0.0
        else:
            tau = float(sorted_scores[k - 1])
        actual_fpr = float((all_scores >= tau).sum()) / n_valid
        fpr_to_threshold[fpr] = tau
        print(f"  FPR target={fpr:.1e} -> threshold={tau:.6f} "
              f"(actual FPR={actual_fpr:.4e}, N_valid={n_valid:,})")

    return fpr_to_threshold


# ---------------------------------------------------------------------------
# Host bin assignment
# ---------------------------------------------------------------------------
def nearest_bin(val: float, grid: List[float]) -> float:
    """Snap value to nearest grid bin center."""
    arr = np.asarray(grid)
    return float(arr[np.argmin(np.abs(arr - val))])


# ---------------------------------------------------------------------------
# Lens parameter sampling for Model 2
# ---------------------------------------------------------------------------
def sample_lens_params_model2(
    rng: np.random.Generator,
    theta_e_arcsec: float,
    host_hwc_nmgy: np.ndarray,
    center_sigma_arcsec: float = 0.05,
    q_floor: float = 0.5,
    q_scatter: float = 0.05,
    gamma_max: float = 0.08,
) -> Tuple[LensParams, Dict[str, float]]:
    """Sample lens parameters conditioned on host galaxy moments (Model 2).

    Returns
    -------
    lens : LensParams
        Lens parameters with q and PA aligned to host light.
    diagnostics : dict
        Host moment diagnostics (q, phi, r_half, is_fallback).
    """
    # Estimate host moments from r-band
    moments = estimate_host_moments_rband(host_hwc_nmgy)

    # Map moments to lens parameter prior
    param_dict = map_host_to_lens_params(
        theta_e_arcsec, moments,
        q_floor=q_floor,
        q_scatter=q_scatter,
        gamma_max=gamma_max,
        rng=rng,
    )

    # Add centroid jitter
    x0 = float(rng.normal(0.0, center_sigma_arcsec))
    y0 = float(rng.normal(0.0, center_sigma_arcsec))

    lens = LensParams(
        theta_e_arcsec=param_dict["theta_e_arcsec"],
        shear_g1=param_dict["shear_g1"],
        shear_g2=param_dict["shear_g2"],
        x0_arcsec=x0,
        y0_arcsec=y0,
        q_lens=param_dict["q_lens"],
        phi_lens_rad=param_dict["phi_lens_rad"],
    )

    diagnostics = {
        "host_q": moments.q,
        "host_phi_rad": moments.phi_rad,
        "host_r_half_pix": moments.r_half_pix,
        "host_is_fallback": moments.is_fallback,
        "lens_q": param_dict["q_lens"],
        "lens_phi_rad": param_dict["phi_lens_rad"],
        "gamma_ext": param_dict["gamma_ext"],
    }

    return lens, diagnostics


# ---------------------------------------------------------------------------
# Main grid runner (v2: Model 1 + Model 2)
# ---------------------------------------------------------------------------
def run_selection_function(
    checkpoint_path: str,
    manifest_path: str,
    injection_model: int = 1,
    host_split: str = "val",
    host_max: int = 20000,
    thresholds: Optional[List[float]] = None,
    fpr_targets: Optional[List[float]] = None,
    injections_per_cell: int = 500,
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
    # Processing
    preprocessing: str = "raw_robust",
    crop: bool = False,
    pixscale: float = PIXEL_SCALE,
    seed: int = 1337,
    device_str: str = "cuda",
    data_root: Optional[str] = None,
    # Sensitivity analysis overrides
    injection_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the injection-recovery grid. Returns (results_df, metadata).

    Parameters
    ----------
    injection_model : int
        1 = Model 1 (random hosts, independent lens params)
        2 = Model 2 (LRG hosts, host-conditioned lens params)
    """
    if thresholds is None:
        thresholds = [0.5]

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Sensitivity analysis overrides
    _ovr = injection_overrides or {}
    psf_scale_factor = _ovr.get("psf_scale_factor", 1.0)
    re_scale = _ovr.get("re_scale_factor", 1.0)
    gmr_shift = _ovr.get("g_minus_r_shift", 0.0)
    rmz_shift = _ovr.get("r_minus_z_shift", 0.0)
    q_lens_range_override = _ovr.get("q_lens_range", None)

    # Path overrides
    eff_manifest = manifest_path
    eff_ckpt = checkpoint_path
    if data_root:
        default_root = "/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration"
        eff_manifest = manifest_path.replace(default_root, data_root.rstrip("/"), 1)
        eff_ckpt = checkpoint_path.replace(default_root, data_root.rstrip("/"), 1)

    # Load model + preprocessing config from checkpoint
    print(f"Loading model: {eff_ckpt}")
    model, arch, epoch, pp_kwargs = load_model_from_checkpoint(eff_ckpt, device)
    print(f"  Architecture: {arch}, Epoch: {epoch}")
    print(f"  Preprocessing: {pp_kwargs}")

    # Load manifest
    print(f"Loading manifest: {eff_manifest}")
    df = pd.read_parquet(eff_manifest)

    # Host type distribution (diagnostic)
    type_summary = host_selection_summary(df, split=host_split)
    print(f"  Host type distribution in '{host_split}': {type_summary}")

    # Derive FPR-based thresholds if requested
    fpr_threshold_map: Dict[float, float] = {}
    if fpr_targets:
        fpr_threshold_map = derive_fpr_thresholds(
            model=model,
            manifest_df=df,
            split=host_split,
            target_fprs=fpr_targets,
            preprocessing=preprocessing,
            crop=crop,
            device=device,
            seed=seed,
            pp_kwargs=pp_kwargs,
        )
        for fpr, tau in fpr_threshold_map.items():
            if tau not in thresholds:
                thresholds.append(tau)

    # Select hosts based on injection model
    print(f"\n  Injection Model: {injection_model}")
    if injection_model == 2:
        print("  Host selection: LRG-like (DEV/SER morphology)")
        hosts = select_lrg_hosts(df, split=host_split)
    else:
        print("  Host selection: Random (all morphology types)")
        hosts = select_random_hosts(df, split=host_split)

    print(f"  Selected hosts: {len(hosts):,}")

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
    print(f"  Thresholds: {thresholds}")
    print(f"  Injections/cell: {injections_per_cell}")
    print(f"  Total injections: {n_cells * injections_per_cell:,}")

    # Assign hosts to PSF and depth bins
    depth_mag_arr = m5_from_psfdepth(hosts["psfdepth_r"].to_numpy(dtype=np.float64))
    hosts["depth_mag"] = depth_mag_arr
    hosts["psf_bin"] = hosts["psfsize_r"].apply(lambda v: nearest_bin(float(v), psf_bins))
    hosts["depth_bin"] = hosts["depth_mag"].apply(lambda v: nearest_bin(float(v), depth_bins))

    host_groups: Dict[Tuple[float, float], pd.DataFrame] = {}
    for (pb, db), g in hosts.groupby(["psf_bin", "depth_bin"]):
        host_groups[(float(pb), float(db))] = g

    input_size = CUTOUT_SIZE if not crop else STAMP_SIZE

    # Run grid
    MAX_RETRIES = 5
    print(f"\nRunning injection-recovery grid (Model {injection_model})...")
    t0 = time.time()
    rows = []
    cell_idx = 0
    total_inject_ok = 0
    total_inject_failed = 0
    total_fallback_moments = 0
    failure_log: List[Dict[str, Any]] = []

    for theta_e in theta_es:
        for pb in psf_bins:
            for db in depth_bins:
                cell_idx += 1
                host_df = host_groups.get((pb, db))
                thr_to_fpr = {tau: fpr for fpr, tau in fpr_threshold_map.items()}

                if host_df is None or len(host_df) == 0:
                    for thr in thresholds:
                        fpr_label = thr_to_fpr.get(thr, None)
                        threshold_type = f"FPR={fpr_label:.1e}" if fpr_label is not None else "fixed"
                        rows.append({
                            "theta_e": theta_e,
                            "psf_fwhm": pb,
                            "depth_5sig": db,
                            "threshold": thr,
                            "threshold_type": threshold_type,
                            "fpr_target": fpr_label if fpr_label is not None else float("nan"),
                            "source_mag_bin": "all",
                            "injection_model": injection_model,
                            "n_injections": 0,
                            "n_detected": 0,
                            "n_failed": 0,
                            "n_fallback_moments": 0,
                            "completeness": float("nan"),
                            "ci68_lo": float("nan"),
                            "ci68_hi": float("nan"),
                            "ci95_lo": float("nan"),
                            "ci95_hi": float("nan"),
                            "mean_score": float("nan"),
                            "mean_arc_snr": float("nan"),
                            "mean_host_q": float("nan"),
                            "sufficient": False,
                        })
                    continue

                n_target = injections_per_cell

                # Collect successful injections with retry-on-failure
                batch_list = []
                arc_snr_list = []
                src_mag_list = []
                lensed_mag_list = []  # Lensed (observed) arc magnitude
                host_q_list = []
                cell_n_failed = 0
                cell_n_fallback = 0

                attempt = 0
                while len(batch_list) < n_target and attempt < n_target + n_target * MAX_RETRIES:
                    attempt += 1
                    host_idx = int(rng.choice(len(host_df)))
                    host_row = host_df.iloc[host_idx]

                    try:
                        # Load host cutout (HWC, nanomaggies)
                        with np.load(str(host_row[CUTOUT_PATH_COL])) as z:
                            host_hwc = z["cutout"].astype(np.float32)

                        host_hwc_torch = torch.from_numpy(host_hwc).float()

                        # Per-host observing conditions
                        host_psf = float(host_row["psfsize_r"])
                        host_psfdepth = float(host_row["psfdepth_r"])

                        # Sample lens parameters based on injection model
                        if injection_model == 2:
                            lens, diag = sample_lens_params_model2(
                                rng, theta_e, host_hwc,
                            )
                            _pending_host_q = diag["host_q"]
                            _pending_is_fallback = diag["host_is_fallback"]
                        else:
                            lens = sample_lens_params(
                                rng, theta_e,
                                q_lens_range_override=q_lens_range_override,
                            )
                            _pending_host_q = float("nan")
                            _pending_is_fallback = False

                        # Sample source parameters
                        source = sample_source_params(
                            rng, theta_e,
                            re_scale=re_scale,
                            gmr_shift=gmr_shift,
                            rmz_shift=rmz_shift,
                        )

                        # Record source r-magnitude
                        src_r_mag = float("nan")
                        if source.flux_nmgy_r > 0:
                            src_r_mag = AB_ZP - 2.5 * math.log10(source.flux_nmgy_r)

                        # Inject
                        eff_psf = host_psf * psf_scale_factor
                        inj_seed = seed + cell_idx * (n_target + n_target * MAX_RETRIES) + attempt
                        result = inject_sis_shear(
                            host_nmgy_hwc=host_hwc_torch,
                            lens=lens,
                            source=source,
                            pixel_scale=pixscale,
                            psf_fwhm_r_arcsec=eff_psf,
                            seed=inj_seed,
                        )

                        # Arc SNR diagnostic
                        sigma_pix_r = estimate_sigma_pix_from_psfdepth(
                            host_psfdepth, eff_psf, pixscale
                        )
                        injection_chw = result.injection_only[0]
                        snr_val = arc_annulus_snr(injection_chw, sigma_pix_r)

                        # Lensed (observed) arc magnitude from total injected flux
                        # in r-band (index 1 in g,r,z order). This is the
                        # magnification * unlensed flux, giving the actual
                        # observed arc brightness for stratification by lensed mag.
                        total_lensed_flux_r = float(injection_chw[1].sum().item())
                        if total_lensed_flux_r > 0:
                            lensed_r_mag = AB_ZP - 2.5 * math.log10(total_lensed_flux_r)
                        else:
                            lensed_r_mag = float("nan")

                        # Preprocess for scoring (uses checkpoint's preprocessing)
                        injected_chw_np = result.injected[0].numpy()
                        proc = preprocess_stack(injected_chw_np, **pp_kwargs)

                        # SUCCESS — only now do we record diagnostics
                        batch_list.append(proc)
                        arc_snr_list.append(snr_val)
                        src_mag_list.append(src_r_mag)
                        lensed_mag_list.append(lensed_r_mag)
                        host_q_list.append(_pending_host_q)
                        if _pending_is_fallback:
                            cell_n_fallback += 1

                    except Exception as exc:
                        cell_n_failed += 1
                        total_inject_failed += 1
                        if len(failure_log) < 100:
                            failure_log.append({
                                "cell_idx": cell_idx,
                                "theta_e": theta_e,
                                "psf_bin": pb,
                                "depth_bin": db,
                                "attempt": attempt,
                                "error": str(exc),
                                "cutout_path": str(host_row.get(CUTOUT_PATH_COL, "?")),
                            })

                n_ok = len(batch_list)
                total_inject_ok += n_ok
                total_fallback_moments += cell_n_fallback

                if cell_n_failed > 0:
                    print(f"\n  WARNING: Cell {cell_idx} "
                          f"[theta_E={theta_e}, PSF={pb}, depth={db}]: "
                          f"{cell_n_failed} failures, {n_ok}/{n_target} ok")

                # Score successful injections
                if n_ok > 0:
                    batch = np.stack(batch_list, axis=0)
                    arc_snrs = np.array(arc_snr_list, dtype=np.float64)
                    source_r_mags = np.array(src_mag_list, dtype=np.float64)
                    lensed_r_mags = np.array(lensed_mag_list, dtype=np.float64)
                    host_qs = np.array(host_q_list, dtype=np.float64)
                    scores = score_batch(model, batch, device)
                else:
                    scores = np.array([], dtype=np.float64)
                    arc_snrs = np.array([], dtype=np.float64)
                    source_r_mags = np.array([], dtype=np.float64)
                    lensed_r_mags = np.array([], dtype=np.float64)
                    host_qs = np.array([], dtype=np.float64)

                # Emit results
                src_mag_bins = [(23.0, 24.0), (24.0, 25.0), (25.0, 26.0)]

                for thr in thresholds:
                    if n_ok > 0:
                        detected = scores >= thr
                        k = int(detected.sum())
                        comp = k / n_ok
                        lo, hi = bayes_binomial_interval(k, n_ok, level=0.68)
                        lo95, hi95 = bayes_binomial_interval(k, n_ok, level=0.95)
                    else:
                        detected = np.array([], dtype=bool)
                        k = 0
                        comp = float("nan")
                        lo = hi = float("nan")
                        lo95 = hi95 = float("nan")

                    valid_snrs = arc_snrs[np.isfinite(arc_snrs)] if n_ok > 0 else np.array([])
                    mean_snr = float(valid_snrs.mean()) if len(valid_snrs) > 0 else float("nan")

                    valid_qs = host_qs[np.isfinite(host_qs)] if n_ok > 0 else np.array([])
                    mean_host_q = float(valid_qs.mean()) if len(valid_qs) > 0 else float("nan")

                    fpr_label = thr_to_fpr.get(thr, None)
                    threshold_type = f"FPR={fpr_label:.1e}" if fpr_label is not None else "fixed"

                    rows.append({
                        "theta_e": theta_e,
                        "psf_fwhm": pb,
                        "depth_5sig": db,
                        "threshold": thr,
                        "threshold_type": threshold_type,
                        "fpr_target": fpr_label if fpr_label is not None else float("nan"),
                        "source_mag_bin": "all",
                        "injection_model": injection_model,
                        "n_injections": n_ok,
                        "n_detected": k,
                        "n_failed": cell_n_failed,
                        "n_fallback_moments": cell_n_fallback,
                        "completeness": float(comp),
                        "ci68_lo": lo,
                        "ci68_hi": hi,
                        "ci95_lo": lo95,
                        "ci95_hi": hi95,
                        "mean_score": float(scores.mean()) if n_ok > 0 else float("nan"),
                        "mean_arc_snr": mean_snr,
                        "mean_host_q": mean_host_q,
                        "sufficient": n_ok >= injections_per_cell,
                    })

                    # Source-magnitude stratified completeness
                    for mag_lo, mag_hi in src_mag_bins:
                        in_bin = (source_r_mags >= mag_lo) & (source_r_mags < mag_hi) if n_ok > 0 else np.array([], dtype=bool)
                        n_bin = int(in_bin.sum()) if n_ok > 0 else 0
                        if n_bin > 0:
                            k_bin = int((detected & in_bin).sum())
                            c_bin = k_bin / n_bin
                            lo_bin, hi_bin = bayes_binomial_interval(k_bin, n_bin, level=0.68)
                            lo95_bin, hi95_bin = bayes_binomial_interval(k_bin, n_bin, level=0.95)
                        else:
                            k_bin = 0
                            c_bin = float("nan")
                            lo_bin = hi_bin = float("nan")
                            lo95_bin = hi95_bin = float("nan")
                        rows.append({
                            "theta_e": theta_e,
                            "psf_fwhm": pb,
                            "depth_5sig": db,
                            "threshold": thr,
                            "threshold_type": threshold_type,
                            "fpr_target": fpr_label if fpr_label is not None else float("nan"),
                            "source_mag_bin": f"{mag_lo:.0f}-{mag_hi:.0f}",
                            "injection_model": injection_model,
                            "n_injections": n_bin,
                            "n_detected": k_bin,
                            "n_failed": 0,
                            "n_fallback_moments": 0,
                            "completeness": float(c_bin),
                            "ci68_lo": lo_bin,
                            "ci68_hi": hi_bin,
                            "ci95_lo": lo95_bin,
                            "ci95_hi": hi95_bin,
                            "mean_score": float(scores[in_bin].mean()) if n_bin > 0 else float("nan"),
                            "mean_arc_snr": float(arc_snrs[in_bin & np.isfinite(arc_snrs)].mean()) if n_ok > 0 and (in_bin & np.isfinite(arc_snrs)).sum() > 0 else float("nan"),
                            "mean_host_q": float("nan"),
                            "sufficient": n_bin >= 10,
                        })

                    # Lensed (observed) magnitude stratified completeness
                    # (LLM reviewer recommendation: report completeness by lensed
                    # magnitude for comparison with real lens surveys)
                    lensed_mag_bins = [
                        (18.0, 20.0), (20.0, 22.0), (22.0, 24.0), (24.0, 27.0),
                    ]
                    for lm_lo, lm_hi in lensed_mag_bins:
                        in_lbin = (lensed_r_mags >= lm_lo) & (lensed_r_mags < lm_hi) if n_ok > 0 else np.array([], dtype=bool)
                        n_lbin = int(in_lbin.sum()) if n_ok > 0 else 0
                        if n_lbin > 0:
                            k_lbin = int((detected & in_lbin).sum())
                            c_lbin = k_lbin / n_lbin
                            lo_lbin, hi_lbin = bayes_binomial_interval(k_lbin, n_lbin, level=0.68)
                            lo95_lbin, hi95_lbin = bayes_binomial_interval(k_lbin, n_lbin, level=0.95)
                        else:
                            k_lbin = 0
                            c_lbin = float("nan")
                            lo_lbin = hi_lbin = float("nan")
                            lo95_lbin = hi95_lbin = float("nan")
                        rows.append({
                            "theta_e": theta_e,
                            "psf_fwhm": pb,
                            "depth_5sig": db,
                            "threshold": thr,
                            "threshold_type": threshold_type,
                            "fpr_target": fpr_label if fpr_label is not None else float("nan"),
                            "source_mag_bin": f"lensed_{lm_lo:.0f}-{lm_hi:.0f}",
                            "injection_model": injection_model,
                            "n_injections": n_lbin,
                            "n_detected": k_lbin,
                            "n_failed": 0,
                            "n_fallback_moments": 0,
                            "completeness": float(c_lbin),
                            "ci68_lo": lo_lbin,
                            "ci68_hi": hi_lbin,
                            "ci95_lo": lo95_lbin,
                            "ci95_hi": hi95_lbin,
                            "mean_score": float(scores[in_lbin].mean()) if n_lbin > 0 else float("nan"),
                            "mean_arc_snr": float(arc_snrs[in_lbin & np.isfinite(arc_snrs)].mean()) if n_ok > 0 and (in_lbin & np.isfinite(arc_snrs)).sum() > 0 else float("nan"),
                            "mean_host_q": float("nan"),
                            "sufficient": n_lbin >= 10,
                        })

                if cell_idx % 10 == 0:
                    pct = 100 * cell_idx / n_cells
                    comp_display = comp if n_ok > 0 else float("nan")
                    print(f"  Cell {cell_idx}/{n_cells} ({pct:.0f}%) "
                          f"[theta_E={theta_e}, PSF={pb}, depth={db}] "
                          f"C(0.5)={comp_display:.3f}", end="\r")

    dt = time.time() - t0
    print(f"\n  Grid complete: {n_cells} cells, {dt:.1f}s ({dt / max(n_cells, 1):.2f}s/cell)")
    print(f"  Total injections: {total_inject_ok:,} ok, {total_inject_failed:,} failed "
          f"({100.0 * total_inject_failed / max(total_inject_ok + total_inject_failed, 1):.2f}%)")
    if injection_model == 2:
        print(f"  Fallback moments (host too faint for shape estimation): {total_fallback_moments:,}")

    results_df = pd.DataFrame(rows)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_path,
        "manifest": manifest_path,
        "arch": arch,
        "epoch": epoch,
        "injection_model": injection_model,
        "host_split": host_split,
        "host_max": host_max,
        "host_type_distribution": type_summary,
        "host_selection": "LRG (DEV/SER)" if injection_model == 2 else "random (all types)",
        "thresholds": thresholds,
        "injections_per_cell": injections_per_cell,
        "grid": {
            "theta_e": theta_es,
            "psf_fwhm": psf_bins,
            "depth_5sig": depth_bins,
        },
        "preprocessing": preprocessing,
        "crop": crop,
        "pixscale": pixscale,
        "seed": seed,
        "n_cells": n_cells,
        "n_sufficient_cells": int(results_df.loc[
            (results_df["threshold"] == thresholds[0]) & (results_df["source_mag_bin"] == "all"),
            "sufficient"
        ].sum()),
        "n_empty_cells": int(results_df.loc[
            (results_df["threshold"] == thresholds[0]) & (results_df["source_mag_bin"] == "all"),
            "n_injections"
        ].eq(0).sum()),
        "total_injections_ok": total_inject_ok,
        "total_injections_failed": total_inject_failed,
        "total_fallback_moments": total_fallback_moments,
        "failure_rate_pct": round(100.0 * total_inject_failed / max(total_inject_ok + total_inject_failed, 1), 4),
        "max_retries_per_injection": MAX_RETRIES,
        "failure_log_sample": failure_log[:20],
        "fpr_targets": fpr_targets if fpr_targets else [],
        "fpr_derived_thresholds": {str(k): v for k, v in fpr_threshold_map.items()},
        "injection_engine": "SIE+shear (dhs.injection_engine) — validated, 28 tests",
        "source_model": "Sersic + optional clumps",
        "lens_model_conditioning": "host-conditioned (q, PA from r-band moments)" if injection_model == 2 else "independent priors",
        "injection_overrides": injection_overrides if injection_overrides else {},
        "notes": [
            f"Injection Model {injection_model}.",
            "SIE+shear ray-shooting with Sersic source model.",
            "Flux units: nanomaggies (AB ZP=22.5).",
            "PSF blur: FFT-based Gaussian per band.",
            "Preprocessing: raw_robust outer-annulus median/MAD, clip [-10, 10].",
            "Failed injections retried with new hosts (up to MAX_RETRIES).",
            "Only successful injections in denominators.",
        ] + (
            [
                "Model 2: LRG (DEV/SER) hosts, q/PA from r-band second moments.",
                "Model 2: shear_g1/g2 from U[0,gamma_max] with random PA.",
                "Model 2: n_fallback_moments = hosts too faint for shape estimation (q=1 used).",
            ] if injection_model == 2 else []
        ),
    }

    return results_df, metadata


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def save_outputs(
    results_df: pd.DataFrame,
    metadata: Dict[str, Any],
    out_dir: str,
) -> None:
    """Save CSV and JSON to local path or S3."""
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    json_bytes = json.dumps(metadata, indent=2, default=str).encode("utf-8")

    csv_path = join_uri(out_dir, "selection_function.csv")
    json_path = join_uri(out_dir, "selection_function_meta.json")

    if is_s3_uri(out_dir):
        write_bytes(csv_path, csv_bytes, content_type="text/csv")
        write_bytes(json_path, json_bytes, content_type="application/json")
    else:
        os.makedirs(out_dir, exist_ok=True)
        with open(csv_path, "wb") as f:
            f.write(csv_bytes)
        with open(json_path, "wb") as f:
            f.write(json_bytes)

    print(f"\nResults saved to: {csv_path}")
    print(f"Metadata saved to: {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Selection Function Grid v2: Model 1 + Model 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", type=int, choices=[1, 2], default=1,
                    help="Injection model: 1=random hosts, 2=LRG deflector-conditioned")
    ap.add_argument("--out-dir", default="results/selection_function_v2",
                    help="Output directory (local path or s3:// URI)")
    ap.add_argument("--host-split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--host-max", type=int, default=20000)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    ap.add_argument("--fpr-targets", nargs="*", type=float, default=None)
    ap.add_argument("--injections-per-cell", type=int, default=500,
                    help="Injections per grid cell (default: 500). "
                         "Increased from 200 per LLM reviewer recommendation: "
                         "at n=200 and p=3.5%%, 95%% CI is [1.7%%, 7.0%%] (too wide).")
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
        injection_model=args.model,
        host_split=args.host_split,
        host_max=args.host_max,
        thresholds=args.thresholds,
        fpr_targets=args.fpr_targets,
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
        preprocessing=args.preprocessing,
        crop=args.crop,
        pixscale=args.pixscale,
        seed=args.seed,
        device_str=args.device,
        data_root=args.data_root,
    )

    save_outputs(results_df, metadata, args.out_dir)

    # Print summary
    for thr in args.thresholds:
        thr_df = results_df[
            (results_df["threshold"] == thr) & (results_df["source_mag_bin"] == "all")
        ]
        valid = thr_df[thr_df["n_injections"] > 0]
        print(f"\n{'='*70}")
        print(f"SELECTION FUNCTION SUMMARY (Model {args.model}, threshold={thr})")
        print(f"{'='*70}")
        print(f"  Architecture: {metadata['arch']}")
        print(f"  Host selection: {metadata['host_selection']}")
        print(f"  Grid cells: {metadata['n_cells']}")
        print(f"  Sufficient cells: {int(valid['sufficient'].sum())}")
        print(f"  Empty cells: {int(thr_df['n_injections'].eq(0).sum())}")
        mc = float(valid["completeness"].dropna().mean()) if len(valid) > 0 else float("nan")
        print(f"  Mean completeness: {mc:.3f}")
        if len(valid) > 0:
            print(f"\n  Completeness by theta_E:")
            for te in sorted(valid["theta_e"].unique()):
                mask = valid["theta_e"] == te
                c = valid.loc[mask, "completeness"].mean()
                snr = valid.loc[mask, "mean_arc_snr"].dropna().mean()
                print(f"    theta_E={te:.2f}: C={c:.3f}, mean_arc_SNR={snr:.1f}")
        if args.model == 2 and "mean_host_q" in valid.columns:
            mean_q = valid["mean_host_q"].dropna().mean()
            print(f"\n  Mean host q (axis ratio): {mean_q:.3f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()

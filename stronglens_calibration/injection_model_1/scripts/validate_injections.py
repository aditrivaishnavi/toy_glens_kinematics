#!/usr/bin/env python3
"""
Validate Injections: visual and quantitative QA for the injection pipeline.

Loads a manifest, samples N host cutouts, injects lensed sources with
default and core-suppressed configurations, scores both with a frozen model,
and generates diagnostic outputs:

  1. Per-injection CSV: scores, arc_annulus_snr, flux conservation, metadata
  2. Summary JSON: aggregate statistics
  3. PNG figure pages: visual examples (requires matplotlib, optional)

This script validates that the injection engine produces physically reasonable
results before running the full selection function grid.

Usage:
    cd /lambda/nfs/.../code
    export PYTHONPATH=.

    python scripts/validate_injections.py \\
        --checkpoint checkpoints/paperIV_efficientnet_v2_s/best.pt \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --host-split val \\
        --n-hosts 100 \\
        --out-dir results/injection_validation

Author: stronglens_calibration project
Date: 2026-02-11
References:
  - MNRAS_RAW_NOTES.md Section 9.3
  - dhs/injection_engine.py: SIS+shear injection API
"""
from __future__ import annotations

import argparse
import json
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
    LensParams,
    SourceParams,
    InjectionResult,
    inject_sis_shear,
    sample_lens_params,
    sample_source_params,
    estimate_sigma_pix_from_psfdepth,
    arc_annulus_snr,
)
from dhs.s3io import is_s3_uri, join_uri, write_bytes, write_json


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CUTOUT_PATH_COL = "cutout_path"
LABEL_COL = "label"
SPLIT_COL = "split"

PIXEL_SCALE = 0.262  # arcsec/pixel


# ---------------------------------------------------------------------------
# Model loading and scoring (same as selection_function_grid.py)
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
def score_single(model: nn.Module, img_chw: np.ndarray, device: torch.device) -> float:
    """Score a single preprocessed image. Returns sigmoid probability."""
    x = torch.from_numpy(img_chw[None]).float().to(device)
    logit = model(x).squeeze().cpu().item()
    return 1.0 / (1.0 + np.exp(-float(logit)))


# ---------------------------------------------------------------------------
# Flux conservation check
# ---------------------------------------------------------------------------
def check_flux_conservation(
    host_hwc: np.ndarray,
    injected_chw: np.ndarray,
    injection_only_chw: np.ndarray,
) -> Dict[str, float]:
    """Verify that injected = host + injection_only (flux conservation).

    Returns per-band max absolute error and relative error.
    """
    host_chw = np.transpose(host_hwc, (2, 0, 1)).astype(np.float32)
    reconstructed = host_chw + injection_only_chw
    diff = injected_chw - reconstructed
    band_names = ["g", "r", "z"]
    result = {}
    for b, name in enumerate(band_names):
        max_abs = float(np.max(np.abs(diff[b])))
        denom = float(np.max(np.abs(injected_chw[b])))
        max_rel = max_abs / (denom + 1e-12)
        result[f"flux_maxabs_{name}"] = max_abs
        result[f"flux_maxrel_{name}"] = max_rel
    return result


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------
def run_validation(
    checkpoint_path: str,
    manifest_path: str,
    host_split: str = "val",
    n_hosts: int = 100,
    theta_e_arcsec: float = 1.5,
    core_suppress_radius_pix: int = 5,
    preprocessing: str = "raw_robust",
    crop: bool = False,
    pixscale: float = PIXEL_SCALE,
    seed: int = 42,
    device_str: str = "cuda",
    data_root: Optional[str] = None,
    examples_per_page: int = 5,
    anchor_manifest: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], list, int]:
    """Run injection validation. Returns (per_injection_df, summary, fig_data, examples_per_page)."""

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

    # Select negative hosts
    hosts = df[(df[SPLIT_COL] == host_split) & (df[LABEL_COL] == 0)].copy()
    if hosts.empty:
        raise ValueError(f"No negative hosts found in split '{host_split}'")

    rng = np.random.default_rng(seed)
    if len(hosts) > n_hosts:
        hosts = hosts.sample(n=n_hosts, random_state=seed).reset_index(drop=True)
    print(f"  Sampled {len(hosts)} hosts for validation")

    input_size = CUTOUT_SIZE if not crop else STAMP_SIZE

    # Run injections
    print(f"\nRunning injection validation (theta_E={theta_e_arcsec}\")...")
    t0 = time.time()
    rows = []
    # Collect figure data for visual pages
    fig_data = []

    for i, (_, host_row) in enumerate(hosts.iterrows()):
        cutout_path = str(host_row[CUTOUT_PATH_COL])
        row_base: Dict[str, Any] = {"host_idx": i, "cutout_path": cutout_path}

        try:
            # Load host cutout (HWC)
            with np.load(cutout_path) as z:
                host_hwc = z["cutout"].astype(np.float32)

            host_hwc_torch = torch.from_numpy(host_hwc).float()

            # Host metadata
            host_psf = float(host_row.get("psfsize_r", 1.2))
            host_psfdepth = float(host_row.get("psfdepth_r", 100.0))

            sigma_pix_r = estimate_sigma_pix_from_psfdepth(
                host_psfdepth, host_psf, pixscale
            )

            # Sample parameters
            lens = sample_lens_params(rng, theta_e_arcsec)
            source = sample_source_params(rng, theta_e_arcsec)

            # --- Default injection (no core suppression) ---
            result_default = inject_sis_shear(
                host_nmgy_hwc=host_hwc_torch,
                lens=lens,
                source=source,
                pixel_scale=pixscale,
                psf_fwhm_r_arcsec=host_psf,
                core_suppress_radius_pix=None,
                seed=seed + i,
            )

            inj_chw_default = result_default.injected[0].numpy()
            inj_only_chw_default = result_default.injection_only[0].numpy()

            snr_default = arc_annulus_snr(result_default.injection_only[0], sigma_pix_r)

            # Flux conservation check
            flux_check = check_flux_conservation(
                host_hwc, inj_chw_default, inj_only_chw_default
            )

            # Score default injection
            proc_default = preprocess_stack(
                inj_chw_default, mode=preprocessing,
                crop=crop, clip_range=10.0
            )
            score_default = score_single(model, proc_default, device)

            # --- Core-suppressed injection ---
            result_cs = inject_sis_shear(
                host_nmgy_hwc=host_hwc_torch,
                lens=lens,
                source=source,
                pixel_scale=pixscale,
                psf_fwhm_r_arcsec=host_psf,
                core_suppress_radius_pix=core_suppress_radius_pix,
                seed=seed + i,
            )

            inj_chw_cs = result_cs.injected[0].numpy()
            inj_only_chw_cs = result_cs.injection_only[0].numpy()

            snr_cs = arc_annulus_snr(result_cs.injection_only[0], sigma_pix_r)

            proc_cs = preprocess_stack(
                inj_chw_cs, mode=preprocessing,
                crop=crop, clip_range=10.0
            )
            score_cs = score_single(model, proc_cs, device)

            # Score host alone (no injection)
            host_chw = np.transpose(host_hwc, (2, 0, 1)).astype(np.float32)
            proc_host = preprocess_stack(
                host_chw, mode=preprocessing,
                crop=crop, clip_range=10.0
            )
            score_host = score_single(model, proc_host, device)

            row = {
                **row_base,
                "psfsize_r": host_psf,
                "psfdepth_r": host_psfdepth,
                "sigma_pix_r": sigma_pix_r,
                "theta_e_arcsec": lens.theta_e_arcsec,
                "shear_g1": lens.shear_g1,
                "shear_g2": lens.shear_g2,
                "source_re": source.re_arcsec,
                "source_n": source.n_sersic,
                "flux_nmgy_r": source.flux_nmgy_r,
                "score_host": score_host,
                "score_default": score_default,
                "score_core_suppressed": score_cs,
                "arc_snr_default": snr_default,
                "arc_snr_core_suppressed": snr_cs,
                **flux_check,
                "error": "",
            }

            # Collect data for visual output
            if len(fig_data) < n_hosts:
                fig_data.append({
                    "host_hwc": host_hwc,
                    "inj_default_chw": inj_chw_default,
                    "inj_only_default_chw": inj_only_chw_default,
                    "inj_cs_chw": inj_chw_cs,
                    "inj_only_cs_chw": inj_only_chw_cs,
                    "scores": (score_host, score_default, score_cs),
                    "snrs": (snr_default, snr_cs),
                    "theta_e": lens.theta_e_arcsec,
                })

        except Exception as e:
            row = {
                **row_base,
                "error": str(e),
            }

        rows.append(row)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(hosts)} processed", end="\r")

    dt = time.time() - t0
    print(f"\n  Validation complete: {len(hosts)} hosts, {dt:.1f}s")

    results_df = pd.DataFrame(rows)

    # Compute summary statistics
    ok = results_df["error"] == ""
    ok_df = results_df[ok]

    summary: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_path,
        "manifest": manifest_path,
        "arch": arch,
        "epoch": epoch,
        "n_hosts": len(hosts),
        "n_success": int(ok.sum()),
        "n_errors": int((~ok).sum()),
        "theta_e_arcsec": theta_e_arcsec,
        "core_suppress_radius_pix": core_suppress_radius_pix,
        "preprocessing": preprocessing,
        "crop": crop,
        "seed": seed,
    }

    if len(ok_df) > 0:
        summary["score_host_mean"] = float(ok_df["score_host"].mean())
        summary["score_host_std"] = float(ok_df["score_host"].std())
        summary["score_default_mean"] = float(ok_df["score_default"].mean())
        summary["score_default_std"] = float(ok_df["score_default"].std())
        summary["score_core_suppressed_mean"] = float(ok_df["score_core_suppressed"].mean())
        summary["score_core_suppressed_std"] = float(ok_df["score_core_suppressed"].std())
        summary["arc_snr_default_mean"] = float(ok_df["arc_snr_default"].dropna().mean())
        summary["arc_snr_core_suppressed_mean"] = float(ok_df["arc_snr_core_suppressed"].dropna().mean())

        # Flux conservation (should be near zero)
        for band in ["g", "r", "z"]:
            col = f"flux_maxabs_{band}"
            if col in ok_df.columns:
                summary[f"flux_conservation_max_{band}"] = float(ok_df[col].max())

        # Detection rates at 0.5 threshold
        det_default = float((ok_df["score_default"] >= 0.5).mean())
        det_cs = float((ok_df["score_core_suppressed"] >= 0.5).mean())
        det_host = float((ok_df["score_host"] >= 0.5).mean())
        summary["detection_rate_default_0.5"] = det_default
        summary["detection_rate_core_suppressed_0.5"] = det_cs
        summary["false_positive_rate_host_0.5"] = det_host

        # Score distribution diagnostics (Issue 6b)
        # Report percentiles and flag saturation
        for score_col, label in [
            ("score_default", "default"),
            ("score_core_suppressed", "core_suppressed"),
        ]:
            vals = ok_df[score_col].dropna().values
            if len(vals) > 0:
                pctls = np.percentile(vals, [5, 25, 50, 75, 95])
                summary[f"score_{label}_p5"] = float(pctls[0])
                summary[f"score_{label}_p25"] = float(pctls[1])
                summary[f"score_{label}_p50"] = float(pctls[2])
                summary[f"score_{label}_p75"] = float(pctls[3])
                summary[f"score_{label}_p95"] = float(pctls[4])

                # Saturation warnings
                frac_high = float((vals > 0.9).mean())
                frac_low = float((vals < 0.1).mean())
                if frac_high > 0.9:
                    summary[f"WARNING_score_{label}_saturated_high"] = (
                        f"{frac_high:.1%} of scores > 0.9 — model may be trivially detecting injections"
                    )
                if frac_low > 0.9:
                    summary[f"WARNING_score_{label}_saturated_low"] = (
                        f"{frac_low:.1%} of scores < 0.1 — injections may be too faint to detect"
                    )

    # Anchor SNR comparison (Issue 6c)
    if anchor_manifest is not None:
        try:
            from scipy.stats import ks_2samp
            print(f"\n  Loading anchor manifest for SNR comparison: {anchor_manifest}")
            anchor_df = pd.read_parquet(anchor_manifest)
            # Select Tier-A positives (label=1)
            anchors = anchor_df[anchor_df.get(LABEL_COL, pd.Series(dtype=int)) == 1]
            if len(anchors) == 0:
                summary["anchor_comparison"] = "No positives in anchor manifest"
            else:
                anchor_snrs = []
                for _, arow in anchors.head(min(500, len(anchors))).iterrows():
                    try:
                        with np.load(str(arow[CUTOUT_PATH_COL])) as z:
                            a_hwc = z["cutout"].astype(np.float32)
                        a_chw = np.transpose(a_hwc, (2, 0, 1))
                        a_chw_t = torch.from_numpy(a_chw).float()
                        a_psf = float(arow.get("psfsize_r", 1.2))
                        a_psfdepth = float(arow.get("psfdepth_r", 100.0))
                        a_sigma = estimate_sigma_pix_from_psfdepth(a_psfdepth, a_psf, pixscale)
                        # For real lenses, the "injection_only" is the full image (we measure overall SNR)
                        a_snr = arc_annulus_snr(a_chw_t, a_sigma)
                        if np.isfinite(a_snr):
                            anchor_snrs.append(a_snr)
                    except Exception:
                        pass

                if len(anchor_snrs) >= 10:
                    anchor_snrs_arr = np.array(anchor_snrs)
                    inj_snrs_arr = ok_df["arc_snr_default"].dropna().values

                    ks_stat, ks_pval = ks_2samp(inj_snrs_arr, anchor_snrs_arr)
                    summary["anchor_snr_n"] = len(anchor_snrs)
                    summary["anchor_snr_median"] = float(np.median(anchor_snrs_arr))
                    summary["anchor_snr_iqr"] = [float(np.percentile(anchor_snrs_arr, 25)),
                                                  float(np.percentile(anchor_snrs_arr, 75))]
                    summary["injection_snr_median"] = float(np.median(inj_snrs_arr))
                    summary["injection_snr_iqr"] = [float(np.percentile(inj_snrs_arr, 25)),
                                                     float(np.percentile(inj_snrs_arr, 75))]
                    summary["anchor_vs_injection_ks_stat"] = float(ks_stat)
                    summary["anchor_vs_injection_ks_pval"] = float(ks_pval)
                    print(f"  Anchor SNR: median={np.median(anchor_snrs_arr):.1f}, "
                          f"IQR=[{np.percentile(anchor_snrs_arr, 25):.1f}, "
                          f"{np.percentile(anchor_snrs_arr, 75):.1f}]")
                    print(f"  Injection SNR: median={np.median(inj_snrs_arr):.1f}, "
                          f"IQR=[{np.percentile(inj_snrs_arr, 25):.1f}, "
                          f"{np.percentile(inj_snrs_arr, 75):.1f}]")
                    print(f"  KS test: stat={ks_stat:.4f}, p={ks_pval:.4f}")
                else:
                    summary["anchor_comparison"] = f"Too few anchor SNRs ({len(anchor_snrs)})"
        except Exception as e:
            summary["anchor_comparison_error"] = str(e)

    summary["notes"] = [
        "score_host: model score on bare host (should be low for negatives).",
        "score_default: score after injection without core suppression.",
        "score_core_suppressed: score after injection with core masking.",
        "arc_snr: annulus SNR proxy for r-band injection signal.",
        "flux_conservation: max absolute error between host+injection and injected.",
        "score_*_p5..p95: percentile distribution to detect score saturation (Issue 6b).",
        "anchor_*: SNR comparison with real lenses if --anchor-manifest given (Issue 6c).",
    ]

    return results_df, summary, fig_data, examples_per_page


# ---------------------------------------------------------------------------
# Plotting (optional, requires matplotlib)
# ---------------------------------------------------------------------------
def generate_figure_pages(
    fig_data: List[Dict],
    out_dir: str,
    examples_per_page: int = 5,
) -> List[str]:
    """Generate PNG figure pages. Returns list of output paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figure generation")
        return []

    def _make_rgb(chw: np.ndarray) -> np.ndarray:
        """Convert CHW nanomaggy to displayable RGB (0-1)."""
        hwc = np.transpose(chw, (1, 2, 0))
        # Use arcsinh stretch
        rgb = np.arcsinh(hwc * 10) / np.arcsinh(10)
        return np.clip(rgb, 0, 1)

    paths = []
    n_pages = (len(fig_data) + examples_per_page - 1) // examples_per_page

    for page in range(n_pages):
        start = page * examples_per_page
        end = min(start + examples_per_page, len(fig_data))
        n_rows = end - start

        fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows))
        if n_rows == 1:
            axes = axes[None, :]

        for row, idx in enumerate(range(start, end)):
            d = fig_data[idx]
            host_rgb = _make_rgb(np.transpose(d["host_hwc"], (2, 0, 1)))
            inj_def_rgb = _make_rgb(d["inj_default_chw"])
            inj_only_rgb = _make_rgb(d["inj_only_default_chw"])
            inj_cs_rgb = _make_rgb(d["inj_cs_chw"])
            inj_only_cs_rgb = _make_rgb(d["inj_only_cs_chw"])

            axes[row, 0].imshow(host_rgb)
            axes[row, 0].set_title(f"Host (score={d['scores'][0]:.3f})")
            axes[row, 1].imshow(inj_only_rgb)
            axes[row, 1].set_title(f"Injection only")
            axes[row, 2].imshow(inj_def_rgb)
            axes[row, 2].set_title(f"Default (score={d['scores'][1]:.3f}, SNR={d['snrs'][0]:.1f})")
            axes[row, 3].imshow(inj_only_cs_rgb)
            axes[row, 3].set_title(f"Core-suppressed only")
            axes[row, 4].imshow(inj_cs_rgb)
            axes[row, 4].set_title(f"Core-sup (score={d['scores'][2]:.3f}, SNR={d['snrs'][1]:.1f})")

            for ax in axes[row]:
                ax.axis("off")

        fig.suptitle(f"Injection Validation (theta_E={fig_data[start]['theta_e']:.2f}\")", fontsize=14)
        fig.tight_layout()

        fname = f"injection_validation_page{page:02d}.png"
        fpath = join_uri(out_dir, fname)

        if is_s3_uri(out_dir):
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            write_bytes(fpath, buf.getvalue(), content_type="image/png")
        else:
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(fpath, dpi=120, bbox_inches="tight")

        plt.close(fig)
        paths.append(fpath)
        print(f"  Saved {fname}")

    return paths


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def save_outputs(
    results_df: pd.DataFrame,
    summary: Dict[str, Any],
    fig_data: List[Dict],
    out_dir: str,
    examples_per_page: int = 5,
) -> None:
    """Save CSV, JSON, and PNG pages to local path or S3."""
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    csv_path = join_uri(out_dir, "injection_validation.csv")
    json_path = join_uri(out_dir, "injection_validation_summary.json")

    if is_s3_uri(out_dir):
        write_bytes(csv_path, csv_bytes, content_type="text/csv")
        write_json(json_path, summary)
    else:
        os.makedirs(out_dir, exist_ok=True)
        with open(csv_path, "wb") as f:
            f.write(csv_bytes)
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to: {csv_path}")
    print(f"Summary saved to: {json_path}")

    # Generate figure pages
    if fig_data:
        print(f"\nGenerating visual pages ({examples_per_page} examples/page)...")
        page_paths = generate_figure_pages(fig_data, out_dir, examples_per_page)
        summary["figure_pages"] = page_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Validate Injection Pipeline: visual and quantitative QA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="results/injection_validation",
                    help="Output directory (local path or s3:// URI)")
    ap.add_argument("--host-split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n-hosts", type=int, default=100)
    ap.add_argument("--theta-e", type=float, default=1.5,
                    help="Einstein radius for validation injections (arcsec)")
    ap.add_argument("--core-suppress-radius", type=int, default=5,
                    help="Core suppression radius (pixels) for ablation comparison")
    ap.add_argument("--examples-per-page", type=int, default=5,
                    help="Number of examples per figure page")
    # Processing
    ap.add_argument("--preprocessing", default="raw_robust")
    ap.add_argument("--crop", action="store_true", default=False)
    ap.add_argument("--pixscale", type=float, default=PIXEL_SCALE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--anchor-manifest", default=None,
                    help="Parquet manifest of Tier-A real lenses for SNR comparison (Issue 6c)")
    args = ap.parse_args()

    results_df, summary, fig_data, examples_per_page = run_validation(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        host_split=args.host_split,
        n_hosts=args.n_hosts,
        theta_e_arcsec=args.theta_e,
        core_suppress_radius_pix=args.core_suppress_radius,
        preprocessing=args.preprocessing,
        crop=args.crop,
        pixscale=args.pixscale,
        seed=args.seed,
        device_str=args.device,
        data_root=args.data_root,
        examples_per_page=args.examples_per_page,
        anchor_manifest=args.anchor_manifest,
    )

    # Save outputs
    save_outputs(results_df, summary, fig_data, args.out_dir, args.examples_per_page)

    # Print summary
    print(f"\n{'='*60}")
    print("INJECTION VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {summary['arch']}")
    print(f"  Hosts: {summary['n_success']} / {summary['n_hosts']} successful")
    if "score_host_mean" in summary:
        print(f"  Score (host alone):       {summary['score_host_mean']:.3f} +/- {summary['score_host_std']:.3f}")
        print(f"  Score (default inj):      {summary['score_default_mean']:.3f} +/- {summary['score_default_std']:.3f}")
        print(f"  Score (core-suppressed):  {summary['score_core_suppressed_mean']:.3f} +/- {summary['score_core_suppressed_std']:.3f}")
        print(f"  Arc SNR (default):        {summary['arc_snr_default_mean']:.1f}")
        print(f"  Arc SNR (core-sup):       {summary['arc_snr_core_suppressed_mean']:.1f}")
        print(f"  Detection rate (def@0.5): {summary['detection_rate_default_0.5']:.3f}")
        print(f"  Detection rate (cs@0.5):  {summary['detection_rate_core_suppressed_0.5']:.3f}")
        print(f"  FPR (host@0.5):           {summary['false_positive_rate_host_0.5']:.3f}")
        for band in ["g", "r", "z"]:
            key = f"flux_conservation_max_{band}"
            if key in summary:
                print(f"  Flux conservation max({band}): {summary[key]:.2e}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

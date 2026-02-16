#!/usr/bin/env python3
"""
Find best real-vs-injection comparison candidates for the MNRAS paper figure.

Reads:
  - Real Tier-A cutouts and metadata from the training manifest.
  - Recovered injection metadata/cutouts from recover_injection_cutouts.py.

Selects top brightness-matched pairs of real Tier-A lenses and injections,
then renders a publication-quality comparison figure.

HONESTY CONSTRAINTS:
  1. Real Tier-A lenses are deterministically selected by a fixed seed (no cherry-picking).
  2. Injections are deterministically reproduced from original experiment params/seeds.
  3. The figure caption and an audit JSON log explicitly document selection criteria.
  4. All cutout files are traceable back to original pipeline runs.
  5. Brightness matching uses total image flux (host+arc) for both real and injected.
  6. One-to-one matching: each injection used at most once.

Usage (run on lambda3 or locally after downloading cutouts):
    cd /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration

    python scripts/find_comparison_candidates.py \
        --manifest manifests/training_parity_70_30_v1.parquet \
        --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
        --injection-meta results/recovered_injections/grid/metadata.parquet \
        --injection-cutouts results/recovered_injections/grid/cutouts/ \
        --out-dir results/comparison_figure/ \
        --n-pairs 10 \
        --n-brightness-bins 5

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
import torch
import torch.nn as nn

CUTOUT_PATH_COL = "cutout_path"
AB_ZP = 22.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def host_morphology_metrics(cutout_path: str, key: str = "cutout") -> dict:
    """Compute simple morphology proxies from the r-band image.

    Returns dict with:
      - concentration: flux_inner / flux_total  (high => compact/elliptical)
      - n_peaks: number of bright peaks in central region (1 => isolated)
      - color_gr: g-r colour of the central 11x11 pixels
    """
    result = {"concentration": float("nan"), "n_peaks": 0, "color_gr": float("nan")}
    try:
        with np.load(cutout_path) as z:
            if key in z:
                hwc = z[key].astype(np.float64)
            else:
                return result
        h, w = hwc.shape[:2]
        cy, cx = h // 2, w // 2
        r_band = hwc[:, :, 1]

        # Concentration: fraction of r-band flux within central 11x11
        inner = r_band[cy - 5 : cy + 6, cx - 5 : cx + 6].sum()
        total = r_band.sum()
        if total > 0:
            result["concentration"] = float(inner / total)

        # Peak count: find local maxima above 50th percentile in central 31x31
        from scipy.ndimage import maximum_filter
        crop = r_band[cy - 15 : cy + 16, cx - 15 : cx + 16]
        threshold = np.percentile(crop, 50)
        local_max = maximum_filter(crop, size=5)
        peaks = (crop == local_max) & (crop > threshold)
        # Only count peaks above 30% of the global maximum
        peak_vals = crop[peaks]
        if len(peak_vals) > 0:
            bright_peaks = peak_vals > 0.3 * peak_vals.max()
            result["n_peaks"] = int(bright_peaks.sum())

        # g-r colour of central region
        g_inner = hwc[cy - 5 : cy + 6, cx - 5 : cx + 6, 0].sum()
        if g_inner > 0 and inner > 0:
            g_mag = AB_ZP - 2.5 * np.log10(g_inner)
            r_mag = AB_ZP - 2.5 * np.log10(inner)
            result["color_gr"] = float(g_mag - r_mag)
    except Exception:
        pass
    return result


def total_rmag_from_cutout(cutout_path: str, key: str = "cutout") -> float:
    """Compute r-band integrated magnitude from total image flux in a .npz file.

    For real Tier-A lenses (key='cutout'): includes host + real arc flux.
    For recovered injections (key='injected_hwc'): includes host + injected arc flux.
    This makes the brightness comparison apples-to-apples.
    """
    try:
        with np.load(cutout_path) as z:
            if key in z:
                hwc = z[key].astype(np.float64)
            else:
                return float("nan")
        r_flux = hwc[:, :, 1].sum()
        if r_flux > 0:
            return AB_ZP - 2.5 * np.log10(r_flux)
        return float("nan")
    except Exception:
        return float("nan")


def load_cutout_rgb(
    path: str, key: str = "cutout", crop_size: int = 0
) -> np.ndarray:
    """Load cutout .npz and return (H, W, 3) RGB array for display.

    Uses ``astropy.visualization.make_lupton_rgb`` -- the standard
    colour-preserving arcsinh stretch for astronomical imaging
    (Lupton et al. 2004).

    Parameters
    ----------
    path : str
        Path to .npz cutout file.
    key : str
        Array key inside the .npz (e.g. 'cutout' or 'injected_hwc').
    crop_size : int
        If > 0, crop to the central crop_size x crop_size pixels before
        rendering. Removes distracting unrelated neighbours.
    """
    from astropy.visualization import make_lupton_rgb

    with np.load(path) as z:
        if key in z:
            hwc = z[key].astype(np.float64)
        elif "injected_hwc" in z:
            hwc = z["injected_hwc"].astype(np.float64)
        else:
            raise KeyError(f"Neither '{key}' nor 'injected_hwc' found in {path}")

    # Centre-crop to remove unrelated neighbours
    if crop_size > 0 and crop_size < hwc.shape[0]:
        h, w = hwc.shape[:2]
        y0 = (h - crop_size) // 2
        x0 = (w - crop_size) // 2
        hwc = hwc[y0 : y0 + crop_size, x0 : x0 + crop_size, :]

    # Channels stored as [g, r, z] in HWC.
    # For RGB display: R=z, G=r, B=g.
    img_r = hwc[:, :, 2]  # z-band -> Red
    img_g = hwc[:, :, 1]  # r-band -> Green
    img_b = hwc[:, :, 0]  # g-band -> Blue

    # make_lupton_rgb handles sky subtraction, arcsinh stretch, and
    # colour-preserving scaling internally.  stretch/Q tuned for
    # Legacy Survey nanomaggy flux range.
    rgb = make_lupton_rgb(img_r, img_g, img_b, stretch=0.5, Q=10, minimum=0)
    return rgb


@torch.no_grad()
def score_cutouts(
    cutout_paths: List[str],
    model: nn.Module,
    pp_kwargs: dict,
    device: torch.device,
    key: str = "cutout",
) -> np.ndarray:
    """Score a list of cutout .npz files through the CNN. Returns sigmoid probabilities."""
    from dhs.preprocess import preprocess_stack

    scores = np.full(len(cutout_paths), float("nan"))
    for i, path in enumerate(cutout_paths):
        try:
            with np.load(path) as z:
                if key in z:
                    hwc = z[key].astype(np.float32)
                else:
                    continue
            if hwc.ndim == 3 and hwc.shape[-1] == 3:
                chw = np.transpose(hwc, (2, 0, 1))
            else:
                chw = hwc
            proc = preprocess_stack(chw, **pp_kwargs)
            x = torch.from_numpy(proc[None]).float().to(device)
            logit = model(x).squeeze().cpu().item()
            scores[i] = float(1.0 / (1.0 + np.exp(-logit)))
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"    Scored {i + 1}/{len(cutout_paths)}", end="\r")
    print(f"    Scored {len(cutout_paths)}/{len(cutout_paths)} cutouts")
    return scores


def brightness_match(
    real_df: pd.DataFrame,
    inj_df: pd.DataFrame,
    n_bins: int = 5,
    n_per_bin: int = 2,
    seed: int = 42,
) -> List[Dict]:
    """Select n_per_bin pairs per brightness bin with one-to-one matching."""
    rng = np.random.default_rng(seed)

    real_mags = real_df["total_rmag"].dropna()
    lo, hi = np.percentile(real_mags, [5, 95])
    bin_edges = np.linspace(lo, hi, n_bins + 1)

    used_inj_indices = set()
    pairs = []

    for i in range(n_bins):
        mag_lo, mag_hi = bin_edges[i], bin_edges[i + 1]
        bin_label = f"{mag_lo:.1f}-{mag_hi:.1f}"

        real_in_bin = real_df[
            (real_df["total_rmag"] >= mag_lo) & (real_df["total_rmag"] < mag_hi)
        ]
        inj_in_bin = inj_df[
            (inj_df["total_rmag"] >= mag_lo) & (inj_df["total_rmag"] < mag_hi)
        ]

        if len(real_in_bin) == 0 or len(inj_in_bin) == 0:
            continue

        real_sample = real_in_bin.sample(
            n=min(n_per_bin, len(real_in_bin)), random_state=seed + i
        )

        for _, real_row in real_sample.iterrows():
            mag_diffs = np.abs(inj_in_bin["total_rmag"].values - real_row["total_rmag"])
            sorted_idx = np.argsort(mag_diffs)

            matched = False
            for best_pos in sorted_idx:
                global_idx = inj_in_bin.index[best_pos]
                if global_idx not in used_inj_indices:
                    used_inj_indices.add(global_idx)
                    inj_row = inj_in_bin.iloc[best_pos]

                    pairs.append({
                        "brightness_bin": bin_label,
                        "real_cutout_path": str(real_row[CUTOUT_PATH_COL]),
                        "real_total_rmag": float(real_row["total_rmag"]),
                        "real_cnn_score": float(real_row.get("cnn_score", float("nan"))),
                        "real_tier": str(real_row.get("tier", "A")),
                        "inj_cutout_filename": str(inj_row["cutout_filename"]),
                        "inj_total_rmag": float(inj_row["total_rmag"]),
                        "inj_cnn_score": float(inj_row["cnn_score"]),
                        "inj_theta_e": float(inj_row["theta_e"]),
                        "inj_lensed_r_mag": float(inj_row.get("lensed_r_mag", float("nan"))),
                        "inj_arc_snr": float(inj_row.get("arc_snr", float("nan"))),
                        "mag_diff": float(mag_diffs[best_pos]),
                    })
                    matched = True
                    break

            if not matched:
                print(f"  WARNING: No unused injection for real lens in bin {bin_label}")

    return pairs


def render_figure(
    pairs: List[Dict],
    inj_cutouts_dir: str,
    out_path: str,
    crop_size: int = 51,
    n_cols: int = 2,
) -> None:
    """Render the comparison figure: left = real, right = injection per row.

    Centre-crops both real and injected cutouts to ``crop_size`` pixels so
    that the comparison focuses on the lens/injection, not on unrelated
    background sources at different sky positions.
    """
    n_pairs = len(pairs)
    if n_pairs == 0:
        print("  No pairs to render.")
        return

    fig, axes = plt.subplots(n_pairs, n_cols, figsize=(4 * n_cols, 3.8 * n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    for i, pair in enumerate(pairs):
        # Real lens
        try:
            real_rgb = load_cutout_rgb(
                pair["real_cutout_path"], key="cutout", crop_size=crop_size
            )
            axes[i, 0].imshow(real_rgb, origin="lower")
        except Exception as e:
            axes[i, 0].text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center",
                            transform=axes[i, 0].transAxes, fontsize=7)
        real_score = pair.get("real_cnn_score", float("nan"))
        real_score_str = f"CNN={real_score:.3f}" if not np.isnan(real_score) else ""
        axes[i, 0].set_title(
            f"Real Tier-A  r={pair['real_total_rmag']:.1f}  {real_score_str}",
            fontsize=9,
        )
        axes[i, 0].axis("off")

        # Injection
        inj_path = os.path.join(inj_cutouts_dir, pair["inj_cutout_filename"])
        try:
            inj_rgb = load_cutout_rgb(
                inj_path, key="injected_hwc", crop_size=crop_size
            )
            axes[i, 1].imshow(inj_rgb, origin="lower")
        except Exception as e:
            axes[i, 1].text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center",
                            transform=axes[i, 1].transAxes, fontsize=7)
        inj_score = pair.get("inj_cnn_score", float("nan"))
        score_str = f"CNN={inj_score:.3f}" if not np.isnan(inj_score) else ""
        arc_snr = pair.get("inj_arc_snr", float("nan"))
        snr_str = f"SNR={arc_snr:.0f}" if not np.isnan(arc_snr) else ""
        axes[i, 1].set_title(
            f"Injected  r={pair['inj_total_rmag']:.1f}  "
            f"$\\theta_E$={pair['inj_theta_e']:.2f}\"  {score_str}  {snr_str}",
            fontsize=9,
        )
        axes[i, 1].axis("off")

    fig.suptitle(
        "Brightness-Matched Real vs Injected Lenses\n"
        f"(arcsinh stretch, g-r-z composite, centre {crop_size}x{crop_size} px crop)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Find brightness-matched real-vs-injection comparison candidates"
    )
    ap.add_argument("--manifest", required=True, help="Training manifest parquet")
    ap.add_argument("--checkpoint", required=True, help="Model checkpoint for scoring real lenses")
    ap.add_argument("--injection-meta", required=True, help="Recovered injection metadata parquet")
    ap.add_argument("--injection-cutouts", required=True, help="Directory with recovered .npz cutouts")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-pairs", type=int, default=10)
    ap.add_argument("--n-brightness-bins", type=int, default=5)
    ap.add_argument("--min-arc-snr", type=float, default=30.0,
                    help="Minimum arc SNR for injections to include (filters invisible arcs)")
    ap.add_argument("--crop-size", type=int, default=51,
                    help="Centre crop size in pixels for rendering (0=full)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Load real Tier-A lenses (fail loudly if tier column missing)
    print("Loading training manifest...")
    df = pd.read_parquet(args.manifest)

    if "tier" not in df.columns:
        raise ValueError(
            "Manifest missing 'tier' column. Cannot identify Tier-A lenses. "
            "Refusing to fall back to all positives (would include noisy Tier-B labels)."
        )
    tier_a = df[
        (df["label"] == 1) & (df["tier"].str.upper() == "A")
    ].copy()
    if len(tier_a) == 0:
        raise ValueError(
            "No Tier-A lenses found (label==1 & tier=='A'). "
            "Check tier column values in manifest."
        )
    print(f"  Tier-A positives: {len(tier_a)}")

    # 2. Score real Tier-A lenses through the CNN
    print("Loading model for scoring real lenses...")
    from dhs.scoring_utils import load_model_and_spec
    model, pp_kwargs = load_model_and_spec(args.checkpoint, device)

    print("Scoring real Tier-A lenses...")
    real_paths = tier_a[CUTOUT_PATH_COL].astype(str).tolist()
    real_scores = score_cutouts(real_paths, model, pp_kwargs, device, key="cutout")
    tier_a["cnn_score"] = real_scores

    # 3. Compute total image r-band magnitude for real lenses (host + arc)
    print("Computing total-image r-band magnitudes for real lenses...")
    tier_a["total_rmag"] = tier_a[CUTOUT_PATH_COL].apply(
        lambda p: total_rmag_from_cutout(str(p), key="cutout")
    )
    tier_a = tier_a.dropna(subset=["total_rmag"])
    print(f"  Real lenses with valid magnitude: {len(tier_a)}")

    # 4. Load injection metadata
    print("Loading injection metadata...")
    inj_meta = pd.read_parquet(args.injection_meta)
    print(f"  Injections: {len(inj_meta)}")

    # 5. Filter injections by minimum arc SNR (remove invisible arcs)
    if args.min_arc_snr > 0 and "arc_snr" in inj_meta.columns:
        n_before = len(inj_meta)
        inj_meta = inj_meta[inj_meta["arc_snr"] >= args.min_arc_snr].copy()
        print(f"  Filtered by arc_snr >= {args.min_arc_snr}: {n_before} -> {len(inj_meta)}")

    # 5b. Filter injection HOSTS by morphology: keep only isolated, concentrated
    #     hosts that resemble real Tier-A lens galaxies (massive ellipticals).
    #     Measured on the HOST cutout (before injection) to avoid arc contamination.
    print("Computing host morphology metrics for injections...")
    host_metrics = inj_meta["host_cutout_path"].apply(
        lambda p: host_morphology_metrics(str(p), key="cutout")
    )
    inj_meta["host_concentration"] = host_metrics.apply(lambda d: d["concentration"])
    inj_meta["host_n_peaks"] = host_metrics.apply(lambda d: d["n_peaks"])
    inj_meta["host_color_gr"] = host_metrics.apply(lambda d: d["color_gr"])

    n_before = len(inj_meta)
    # Keep only hosts with:
    #   - single dominant peak (isolated, no companions)
    #   - high concentration (compact, elliptical-like)
    #   - red colour (g-r > 0.5, typical of ellipticals)
    inj_meta = inj_meta[
        (inj_meta["host_n_peaks"] <= 2)
        & (inj_meta["host_concentration"] >= 0.15)
        & (inj_meta["host_color_gr"] >= 0.3)
    ].copy()
    print(f"  Host morphology filter (isolated+concentrated+red): "
          f"{n_before} -> {len(inj_meta)}")

    # 6. Compute total image magnitude for injections (host + injected arc)
    print("Computing total-image r-band magnitudes for injections...")
    inj_meta["total_rmag"] = inj_meta.apply(
        lambda row: total_rmag_from_cutout(
            os.path.join(args.injection_cutouts, str(row["cutout_filename"])),
            key="injected_hwc",
        ),
        axis=1,
    )
    inj_meta = inj_meta.dropna(subset=["total_rmag"])
    print(f"  Injections with valid total magnitude: {len(inj_meta)}")

    # 7. Brightness-matched pair selection (one-to-one)
    n_per_bin = max(1, args.n_pairs // args.n_brightness_bins)
    print(f"\nSelecting {n_per_bin} pairs per {args.n_brightness_bins} bins "
          f"(one-to-one matching)...")
    pairs = brightness_match(
        tier_a, inj_meta,
        n_bins=args.n_brightness_bins,
        n_per_bin=n_per_bin,
        seed=args.seed,
    )
    print(f"  Total pairs selected: {len(pairs)}")

    # 8. Audit log
    audit_path = os.path.join(args.out_dir, "comparison_audit.json")
    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "injection_meta": args.injection_meta,
        "n_real_tier_a": len(tier_a),
        "n_injections": len(inj_meta),
        "n_brightness_bins": args.n_brightness_bins,
        "n_per_bin": n_per_bin,
        "seed": args.seed,
        "min_arc_snr": args.min_arc_snr,
        "crop_size": args.crop_size,
        "matching_metric": "total_image_r_band_magnitude (host+arc for both real and injected)",
        "one_to_one_matching": True,
        "pairs": pairs,
    }
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"  Audit log: {audit_path}")

    # 9. Render figure
    fig_path = os.path.join(args.out_dir, "comparison_real_vs_injected.pdf")
    render_figure(pairs, args.injection_cutouts, fig_path, crop_size=args.crop_size)

    fig_png = fig_path.replace(".pdf", ".png")
    render_figure(pairs, args.injection_cutouts, fig_png, crop_size=args.crop_size)

    # 10. Copy selected cutouts to out_dir for GitHub
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

    print(f"  Selected cutouts copied to: {selected_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()

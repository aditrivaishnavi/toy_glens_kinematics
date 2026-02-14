#!/usr/bin/env python3
"""
Real Lens Scoring: Score all real confirmed lenses in the validation split
using a frozen trained model and report recall at multiple detection thresholds.
Also scores a sample of negatives for FPR comparison.

This script provides a focused diagnostic for sim-to-real generalization:
  - Recall of real confirmed lenses at fixed (p>0.3, p>0.5) and FPR-derived
    thresholds (p>0.806 for FPR=0.1%, p>0.995 for FPR=0.01%)
  - Full score distribution statistics (percentiles, mean, tail fractions)
  - Negative FPR at the same thresholds for comparison

Outputs:
  - real_lens_scoring_results.json: comprehensive numerical summary
  - real_lens_scores.npz: raw score arrays (real_scores, neg_scores)
  - Console summary table

Usage:
    cd stronglens_calibration
    export PYTHONPATH=.

    python sim_to_real_validations/real_lens_scoring.py \\
        --checkpoint checkpoints/best.pt \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --out-dir results/real_lens_scoring \\
        --host-split val \\
        --n-negatives 3000

Author: stronglens_calibration project
Date: 2026-02-13
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dhs.model import build_model
from dhs.preprocess import preprocess_stack


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint using train config for architecture."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_cfg = ckpt.get("train", {})
    arch = train_cfg.get("arch", "resnet18")
    model = build_model(arch, in_ch=3, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, arch, ckpt.get("epoch", -1)


@torch.no_grad()
def score_one(model: nn.Module, cutout_path: str, device: torch.device) -> float | None:
    """Score a single cutout. Returns probability or None on error."""
    try:
        with np.load(cutout_path) as z:
            hwc = z["cutout"].astype(np.float32)
        chw = np.transpose(hwc, (2, 0, 1))
        proc = preprocess_stack(chw, mode="raw_robust", crop=False, clip_range=10.0)
        x = torch.from_numpy(proc[None]).float().to(device)
        logit = model(x).squeeze().cpu().item()
        p = 1.0 / (1.0 + np.exp(-logit))
        return float(p)
    except Exception:
        return None


def score_paths(
    model: nn.Module,
    paths: list[str],
    device: torch.device,
    progress_every: int = 200,
) -> np.ndarray:
    """Score a list of cutout paths. Returns array of probabilities (NaN on error)."""
    scores = []
    for i, path in enumerate(paths):
        p = score_one(model, path, device)
        scores.append(p if p is not None else np.nan)
        if (i + 1) % progress_every == 0:
            print(f"  Scored {i + 1}/{len(paths)}", flush=True)
    return np.array(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "p>0.3": 0.3,
    "p>0.5": 0.5,
    "p>0.806 (FPR-derived)": 0.806,
    "p>0.995 (FPR-derived)": 0.995,
}


def compute_stats(scores: np.ndarray) -> dict:
    """Compute percentiles, mean, and tail fractions."""
    valid = scores[np.isfinite(scores)]
    if len(valid) == 0:
        return {
            "p5": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "mean": np.nan,
            "frac_above_0.9": np.nan,
            "frac_below_0.1": np.nan,
            "n_valid": 0,
        }
    pctls = np.percentile(valid, [5, 25, 50, 75, 95])
    return {
        "p5": float(pctls[0]),
        "p25": float(pctls[1]),
        "median": float(pctls[2]),
        "p75": float(pctls[3]),
        "p95": float(pctls[4]),
        "mean": float(np.mean(valid)),
        "frac_above_0.9": float((valid > 0.9).mean()),
        "frac_below_0.1": float((valid < 0.1).mean()),
        "n_valid": int(len(valid)),
    }


def recall_at_thresholds(scores: np.ndarray) -> dict[str, float]:
    """Compute recall at each threshold."""
    valid = scores[np.isfinite(scores)]
    if len(valid) == 0:
        return {k: np.nan for k in THRESHOLDS}
    return {
        label: float((valid >= thr).mean())
        for label, thr in THRESHOLDS.items()
    }


def fpr_at_thresholds(scores: np.ndarray) -> dict[str, float]:
    """Compute FPR (fraction above threshold) at each threshold."""
    valid = scores[np.isfinite(scores)]
    if len(valid) == 0:
        return {k: np.nan for k in THRESHOLDS}
    return {
        label: float((valid >= thr).mean())
        for label, thr in THRESHOLDS.items()
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Score real lenses and negatives; report recall and FPR at thresholds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint (e.g. best.pt)")
    ap.add_argument("--manifest", required=True, help="Path to training manifest parquet")
    ap.add_argument(
        "--host-split",
        default="val",
        help="Split to use (default: val)",
    )
    ap.add_argument(
        "--n-negatives",
        type=int,
        default=3000,
        help="Number of random negatives to score (default: 3000)",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for JSON and NPZ results",
    )
    ap.add_argument("--device", default="cuda", help="Device for inference (default: cuda)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for negative sampling")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("REAL LENS SCORING")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Manifest:   {args.manifest}")
    print(f"Split:      {args.host_split}")
    print(f"Device:     {device}")
    print()

    # Load model
    print("Loading model...")
    model, arch, epoch = load_model(args.checkpoint, device)
    print(f"  Architecture: {arch}, Epoch: {epoch}")
    print()

    # Load manifest
    print(f"Loading manifest: {args.manifest}")
    df = pd.read_parquet(args.manifest)
    pos_rows = df[(df["split"] == args.host_split) & (df["label"] == 1)]
    neg_rows = df[(df["split"] == args.host_split) & (df["label"] == 0)]
    print(f"  Positives (label=1): {len(pos_rows)}")
    print(f"  Negatives (label=0): {len(neg_rows)}")
    print()

    if len(pos_rows) == 0:
        print("ERROR: No positives in selected split. Exiting.")
        return 1

    # Score all positives
    print(f"Scoring {len(pos_rows)} real confirmed lenses...")
    pos_paths = pos_rows["cutout_path"].astype(str).tolist()
    real_scores = score_paths(model, pos_paths, device, progress_every=200)
    n_errors = int(np.isnan(real_scores).sum())
    valid_real = real_scores[np.isfinite(real_scores)]
    print(f"  Done. Valid: {len(valid_real)}, Errors: {n_errors}")
    print()

    # Score negatives
    n_neg = min(args.n_negatives, len(neg_rows))
    neg_sample = neg_rows.sample(n=n_neg, random_state=args.seed)
    print(f"Scoring {n_neg} random negatives...")
    neg_paths = neg_sample["cutout_path"].astype(str).tolist()
    neg_scores = score_paths(model, neg_paths, device, progress_every=200)
    n_neg_errors = int(np.isnan(neg_scores).sum())
    valid_neg = neg_scores[np.isfinite(neg_scores)]
    print(f"  Done. Valid: {len(valid_neg)}, Errors: {n_neg_errors}")
    print()

    # Compute metrics
    real_recall = recall_at_thresholds(real_scores)
    real_stats = compute_stats(real_scores)
    neg_fpr = fpr_at_thresholds(neg_scores)
    neg_stats = compute_stats(neg_scores)

    # Build results dict
    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "host_split": args.host_split,
        "arch": arch,
        "epoch": epoch,
        "device": str(device),
        "seed": args.seed,
        "real_lens": {
            "n_total": int(len(real_scores)),
            "n_valid": int(len(valid_real)),
            "n_errors": n_errors,
            "recall_at_thresholds": real_recall,
            "score_stats": {
                "p5": real_stats["p5"],
                "p25": real_stats["p25"],
                "median": real_stats["median"],
                "p75": real_stats["p75"],
                "p95": real_stats["p95"],
                "mean": real_stats["mean"],
                "frac_above_0.9": real_stats["frac_above_0.9"],
                "frac_below_0.1": real_stats["frac_below_0.1"],
            },
        },
        "negatives": {
            "n_total": int(len(neg_scores)),
            "n_valid": int(len(valid_neg)),
            "n_errors": n_neg_errors,
            "fpr_at_thresholds": neg_fpr,
            "score_stats": {
                "p5": neg_stats["p5"],
                "p25": neg_stats["p25"],
                "median": neg_stats["median"],
                "p75": neg_stats["p75"],
                "p95": neg_stats["p95"],
                "mean": neg_stats["mean"],
                "frac_above_0.9": neg_stats["frac_above_0.9"],
                "frac_below_0.1": neg_stats["frac_below_0.1"],
            },
        },
    }

    # Save JSON
    json_path = os.path.join(args.out_dir, "real_lens_scoring_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {json_path}")

    # Save NPZ
    npz_path = os.path.join(args.out_dir, "real_lens_scores.npz")
    np.savez(npz_path, real_scores=real_scores, neg_scores=neg_scores)
    print(f"Scores saved:  {npz_path}")
    print()

    # Print summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Real lens recall:")
    for label, thr in THRESHOLDS.items():
        r = real_recall[label]
        pct = r * 100 if np.isfinite(r) else "N/A"
        print(f"  {label:30s}  recall = {pct:.1f}%")
    print()
    print("Real lens score distribution:")
    print(f"  p5={real_stats['p5']:.4f}  p25={real_stats['p25']:.4f}  "
          f"median={real_stats['median']:.4f}  p75={real_stats['p75']:.4f}  "
          f"p95={real_stats['p95']:.4f}")
    print(f"  mean={real_stats['mean']:.4f}  frac>0.9={real_stats['frac_above_0.9']:.4f}  "
          f"frac<0.1={real_stats['frac_below_0.1']:.4f}")
    print()
    print("Negative FPR at same thresholds:")
    for label, thr in THRESHOLDS.items():
        fpr = neg_fpr[label]
        pct = fpr * 100 if np.isfinite(fpr) else "N/A"
        print(f"  {label:30s}  FPR = {pct:.2f}%")
    print()
    print("Negative score distribution:")
    print(f"  p5={neg_stats['p5']:.4f}  p25={neg_stats['p25']:.4f}  "
          f"median={neg_stats['median']:.4f}  p75={neg_stats['p75']:.4f}  "
          f"p95={neg_stats['p95']:.4f}")
    print(f"  mean={neg_stats['mean']:.4f}  frac>0.9={neg_stats['frac_above_0.9']:.4f}  "
          f"frac<0.1={neg_stats['frac_below_0.1']:.4f}")
    print()
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

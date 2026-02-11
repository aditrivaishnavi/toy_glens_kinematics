#!/usr/bin/env python3
"""
Negative Cleaning Scorer: Score all negatives with first-pass model.

Implements the two-pass negative cleaning procedure described in
MNRAS_RAW_NOTES.md Section 7.1 and 15.1:

  1. Score all negatives with the first-pass model
  2. Flag candidates with p > 0.4 as potential contaminants
  3. Output flagged candidates for visual inspection or automatic removal
  4. Optionally rebuild the manifest without contaminated negatives

Paper IV (Inchausti et al. 2025) cleaned negatives using their Paper III
model (not publicly available). We use our own first-pass model, which is
equally defensible.

The threshold of p > 0.4 matches Paper IV's cleaning criterion.

Usage:
    cd /lambda/nfs/.../code
    export PYTHONPATH=.

    # Score all negatives in manifest
    python scripts/negative_cleaning_scorer.py \\
        --checkpoint checkpoints/paperIV_efficientnet_v2_s/best.pt \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --output results/negative_cleaning_scores.parquet \\
        --summary results/negative_cleaning_summary.json

    # Use a different threshold
    python scripts/negative_cleaning_scorer.py \\
        --checkpoint checkpoints/paperIV_resnet18/best.pt \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --threshold 0.3 \\
        --output results/negative_cleaning_scores_resnet.parquet

Author: stronglens_calibration project
Date: 2026-02-11
Aligned with: MNRAS_RAW_NOTES.md Section 7.1, 15.1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# dhs imports
from dhs.model import build_model
from dhs.data import load_cutout_from_file
from dhs.preprocess import preprocess_stack
from dhs.constants import STAMP_SIZE, CUTOUT_SIZE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CUTOUT_PATH_COL = "cutout_path"
LABEL_COL = "label"
SPLIT_COL = "split"
POOL_COL = "pool"
CONFUSER_COL = "confuser_category"
TIER_COL = "tier"
GALAXY_ID_COL = "galaxy_id"

# Paper IV cleaning threshold
DEFAULT_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Model loading (same as evaluate_parity.py)
# ---------------------------------------------------------------------------
def load_checkpoint(
    checkpoint_path: str, device: torch.device
) -> tuple[nn.Module, dict]:
    """Load model from checkpoint. Auto-detects architecture."""
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    train_cfg = ckpt.get("train", {})
    arch = train_cfg.get("arch", "resnet18")
    pretrained = train_cfg.get("pretrained", False)

    model = build_model(arch, in_ch=3, pretrained=pretrained).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    return model, ckpt


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_negatives(
    model: nn.Module,
    cutout_paths: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    preprocessing: str = "raw_robust",
    crop: bool = False,
) -> np.ndarray:
    """Score cutouts. Returns sigmoid probabilities."""
    n = len(cutout_paths)
    scores = np.zeros(n, dtype=np.float64)
    model.eval()

    # Determine expected output spatial size for fallback zeros
    fallback_size = STAMP_SIZE if crop else CUTOUT_SIZE  # 64 if crop, 101 otherwise

    n_errors = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_paths = cutout_paths[start:end]
        imgs = []
        for p in batch_paths:
            try:
                img = load_cutout_from_file(str(p))
                img = preprocess_stack(img, mode=preprocessing, crop=crop, clip_range=10.0)
                imgs.append(img)
            except Exception:
                imgs.append(np.zeros((3, fallback_size, fallback_size), dtype=np.float32))
                n_errors += 1

        batch = np.stack(imgs, axis=0)
        x = torch.from_numpy(batch).float().to(device)
        logits = model(x).squeeze(1).cpu().numpy()
        scores[start:end] = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))

        if (start // batch_size) % 100 == 0:
            print(f"  Scoring: {end}/{n} ({100*end/n:.1f}%)", end="\r")

    print(f"  Scoring: {n}/{n} (100.0%) -- done ({n_errors} load errors)")
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Score negatives and flag p > threshold for cleaning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True,
                    help="Path to first-pass model checkpoint (best.pt)")
    ap.add_argument("--manifest", required=True,
                    help="Path to training manifest parquet")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Flagging threshold (default: {DEFAULT_THRESHOLD}, matching Paper IV)")
    ap.add_argument("--output", required=True,
                    help="Output parquet with scores and flags")
    ap.add_argument("--summary", default=None,
                    help="Output JSON summary")
    ap.add_argument("--flagged-only", default=None,
                    help="Optional: output only flagged rows (parquet)")
    ap.add_argument("--clean-manifest", default=None,
                    help="Optional: output cleaned manifest (flagged rows removed)")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="Score only these splits (default: all splits)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--preprocessing", default="raw_robust")
    ap.add_argument("--crop", action="store_true", default=False)
    ap.add_argument("--data-root", default=None,
                    help="Override data root for path portability")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Path overrides
    manifest_path = args.manifest
    ckpt_path = args.checkpoint
    if args.data_root:
        default_root = "/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration"
        manifest_path = manifest_path.replace(default_root, args.data_root.rstrip("/"), 1)
        ckpt_path = ckpt_path.replace(default_root, args.data_root.rstrip("/"), 1)

    # Load model
    print(f"\nLoading model: {ckpt_path}")
    model, ckpt = load_checkpoint(ckpt_path, device)
    arch = ckpt.get("train", {}).get("arch", "unknown")
    epoch = ckpt.get("epoch", "?")
    print(f"  Architecture: {arch}, Epoch: {epoch}")

    # Load manifest
    print(f"\nLoading manifest: {manifest_path}")
    df = pd.read_parquet(manifest_path)
    print(f"  Total rows: {len(df):,}")

    # Filter to negatives only
    negatives = df[df[LABEL_COL] == 0].copy()
    print(f"  Negatives: {len(negatives):,}")

    # Optional split filtering
    if args.splits:
        negatives = negatives[negatives[SPLIT_COL].isin(args.splits)]
        print(f"  After split filter ({args.splits}): {len(negatives):,}")

    if len(negatives) == 0:
        print("ERROR: No negatives to score!")
        sys.exit(1)

    # Score all negatives
    print(f"\nScoring {len(negatives):,} negatives...")
    t0 = time.time()
    cutout_paths = negatives[CUTOUT_PATH_COL].values
    scores = score_negatives(
        model, cutout_paths, device,
        batch_size=args.batch_size,
        preprocessing=args.preprocessing,
        crop=args.crop,
    )
    dt = time.time() - t0
    print(f"  Scoring time: {dt:.1f}s ({dt/len(negatives)*1000:.1f}ms/sample)")

    # Add scores and flags
    negatives = negatives.reset_index(drop=True)
    negatives["score"] = scores
    negatives["flagged"] = scores >= args.threshold

    n_flagged = int(negatives["flagged"].sum())
    pct_flagged = 100 * n_flagged / len(negatives)
    print(f"\n{'='*60}")
    print(f"NEGATIVE CLEANING RESULTS")
    print(f"{'='*60}")
    print(f"  Total negatives scored: {len(negatives):,}")
    print(f"  Flagged (p >= {args.threshold}): {n_flagged:,} ({pct_flagged:.2f}%)")
    print(f"  Clean (p < {args.threshold}): {len(negatives) - n_flagged:,}")

    # Score distribution
    print(f"\n  Score distribution:")
    for pct in [50, 75, 90, 95, 99, 99.9]:
        val = np.percentile(scores, pct)
        print(f"    p{pct}: {val:.6f}")
    print(f"    max: {scores.max():.6f}")
    print(f"    mean: {scores.mean():.6f}")

    # Breakdown by pool (N1 vs N2)
    if POOL_COL in negatives.columns:
        print(f"\n  By pool:")
        for pool_val in sorted(negatives[POOL_COL].dropna().unique()):
            mask = negatives[POOL_COL] == pool_val
            n_pool = int(mask.sum())
            n_pool_flagged = int(negatives.loc[mask, "flagged"].sum())
            pct = 100 * n_pool_flagged / n_pool if n_pool > 0 else 0
            print(f"    {pool_val}: {n_pool_flagged}/{n_pool} flagged ({pct:.2f}%)")

    # Breakdown by confuser category
    if CONFUSER_COL in negatives.columns:
        print(f"\n  By confuser category:")
        for cat in sorted(negatives[CONFUSER_COL].dropna().unique()):
            if cat in ("none", ""):
                continue
            mask = negatives[CONFUSER_COL] == cat
            n_cat = int(mask.sum())
            n_cat_flagged = int(negatives.loc[mask, "flagged"].sum())
            pct = 100 * n_cat_flagged / n_cat if n_cat > 0 else 0
            mean_score = float(negatives.loc[mask, "score"].mean())
            print(f"    {cat}: {n_cat_flagged}/{n_cat} flagged ({pct:.2f}%), "
                  f"mean_score={mean_score:.6f}")

    # Breakdown by split
    if SPLIT_COL in negatives.columns:
        print(f"\n  By split:")
        for split_val in sorted(negatives[SPLIT_COL].dropna().unique()):
            mask = negatives[SPLIT_COL] == split_val
            n_split = int(mask.sum())
            n_split_flagged = int(negatives.loc[mask, "flagged"].sum())
            pct = 100 * n_split_flagged / n_split if n_split > 0 else 0
            print(f"    {split_val}: {n_split_flagged}/{n_split} flagged ({pct:.2f}%)")

    # Save scored negatives
    print(f"\nSaving to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    negatives.to_parquet(args.output, index=False)
    print(f"  Wrote {len(negatives):,} rows")

    # Save flagged-only subset
    if args.flagged_only:
        flagged_df = negatives[negatives["flagged"]].copy()
        os.makedirs(os.path.dirname(args.flagged_only) or ".", exist_ok=True)
        flagged_df.to_parquet(args.flagged_only, index=False)
        print(f"  Flagged-only: {len(flagged_df):,} rows -> {args.flagged_only}")

    # Save cleaned manifest (original manifest minus flagged negatives)
    if args.clean_manifest:
        flagged_paths = set(negatives.loc[negatives["flagged"], CUTOUT_PATH_COL])
        clean = df[~df[CUTOUT_PATH_COL].isin(flagged_paths)].copy()
        os.makedirs(os.path.dirname(args.clean_manifest) or ".", exist_ok=True)
        clean.to_parquet(args.clean_manifest, index=False)
        n_removed = len(df) - len(clean)
        print(f"  Cleaned manifest: {len(clean):,} rows ({n_removed} removed) -> {args.clean_manifest}")

    # Save summary JSON
    summary = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": args.checkpoint,
            "manifest": args.manifest,
            "arch": arch,
            "epoch": epoch,
            "threshold": args.threshold,
        },
        "counts": {
            "total_negatives": int(len(negatives)),
            "flagged": n_flagged,
            "clean": int(len(negatives) - n_flagged),
            "flagged_pct": round(pct_flagged, 4),
        },
        "score_distribution": {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "median": float(np.median(scores)),
            "p95": float(np.percentile(scores, 95)),
            "p99": float(np.percentile(scores, 99)),
            "max": float(scores.max()),
        },
        "note": (
            "Flagged negatives may contain unlabeled real lenses. "
            "Paper IV used a prior model (from Paper III) to score negatives "
            "and remove p > 0.4. We use our first-pass model, which is "
            "equally defensible (MNRAS_RAW_NOTES.md Section 15.1)."
        ),
    }

    if args.summary:
        os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
        with open(args.summary, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved to: {args.summary}")
    else:
        print("\n" + json.dumps(summary, indent=2, default=str))

    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"  1. Visually inspect flagged candidates (n={n_flagged})")
    print(f"  2. Decide: remove all flagged, or only visually confirmed contaminants")
    print(f"  3. Use --clean-manifest to generate a cleaned manifest for retraining")
    print(f"  4. Retrain models on cleaned manifest (second-pass models)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

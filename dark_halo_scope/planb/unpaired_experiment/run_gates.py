"""
Run shortcut gates on existing data.

This script can run on:
1. Paired data (to verify shortcut exists) - uses stamp_npz vs ctrl_stamp_npz
2. Unpaired manifest (to verify shortcut is broken)
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .constants import BANDS
from .utils import decode_npz_blob
from .preprocess import preprocess_stack
from .gates import run_shortcut_gates

logger = logging.getLogger(__name__)


def stack_from_npz(decoded: dict[str, np.ndarray]) -> np.ndarray:
    """Stack g,r,z bands into (3, H, W) array."""
    imgs = []
    for b in BANDS:
        k = f"image_{b}"
        if k not in decoded:
            raise KeyError(f"Missing {k} in npz")
        imgs.append(decoded[k].astype(np.float32))
    return np.stack(imgs, axis=0)


def load_paired_samples(
    data_root: Path,
    split: str,
    n_samples: int,
    preprocessing: str = "raw_robust",
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load paired samples (alternating pos/neg from same LRG).
    
    This is for verifying the shortcut exists in current data.
    """
    split_dir = data_root / split
    files = sorted(split_dir.glob("*.parquet"))
    
    rng = np.random.default_rng(seed)
    
    xs, ys = [], []
    n_per_class = n_samples // 2
    
    for f in files:
        if len(xs) >= n_samples:
            break
        
        df = pd.read_parquet(f)
        
        # Sample rows
        sample_rows = df.sample(n=min(100, len(df)), random_state=int(rng.integers(0, 2**31-1)))
        
        for _, row in sample_rows.iterrows():
            if len(xs) >= n_samples:
                break
            
            # Positive (stamp with arc)
            if len([y for y in ys if y == 1]) < n_per_class:
                try:
                    dec = decode_npz_blob(row["stamp_npz"])
                    img = stack_from_npz(dec)
                    img = preprocess_stack(img, mode=preprocessing)
                    xs.append(img)
                    ys.append(1)
                except Exception as e:
                    logger.warning(f"Failed to load stamp: {e}")
            
            # Negative (control without arc)
            if len([y for y in ys if y == 0]) < n_per_class:
                try:
                    dec = decode_npz_blob(row["ctrl_stamp_npz"])
                    img = stack_from_npz(dec)
                    img = preprocess_stack(img, mode=preprocessing)
                    xs.append(img)
                    ys.append(0)
                except Exception as e:
                    logger.warning(f"Failed to load ctrl: {e}")
    
    logger.info(f"Loaded {len(xs)} samples: {sum(ys)} pos, {len(ys) - sum(ys)} neg")
    return np.stack(xs, axis=0), np.array(ys)


def load_manifest_samples(
    manifest_path: str,
    split: str,
    n_samples: int,
    preprocessing: str = "raw_robust",
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Load samples from unpaired manifest."""
    df = pd.read_parquet(manifest_path)
    df = df[df["split"] == split].reset_index(drop=True)
    
    rng = np.random.default_rng(seed)
    
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=int(rng.integers(0, 2**31-1)))
    
    xs, ys = [], []
    for _, row in df.iterrows():
        try:
            dec = decode_npz_blob(row["blob"])
            img = stack_from_npz(dec)
            img = preprocess_stack(img, mode=preprocessing)
            xs.append(img)
            ys.append(int(row["label"]))
        except Exception as e:
            logger.warning(f"Failed to load: {e}")
    
    logger.info(f"Loaded {len(xs)} samples: {sum(ys)} pos, {len(ys) - sum(ys)} neg")
    return np.stack(xs, axis=0), np.array(ys)


def main():
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="Run shortcut gates")
    parser.add_argument("--data-root", help="Root directory with train/val/test subdirs (for paired)")
    parser.add_argument("--manifest", help="Unpaired manifest parquet path")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--preprocessing", default="raw_robust",
                       choices=["raw_robust", "residual_radial_profile"])
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    if args.manifest:
        xs, ys = load_manifest_samples(
            args.manifest, args.split, args.n_samples,
            preprocessing=args.preprocessing, seed=args.seed
        )
    elif args.data_root:
        xs, ys = load_paired_samples(
            Path(args.data_root), args.split, args.n_samples,
            preprocessing=args.preprocessing, seed=args.seed
        )
    else:
        parser.error("Must provide either --data-root or --manifest")
    
    results = run_shortcut_gates(xs, ys)
    
    print("\n" + "=" * 60)
    print("SHORTCUT GATE RESULTS")
    print("=" * 60)
    print(f"Core LR AUC:           {results.core_lr_auc:.4f} (target < 0.65)")
    print(f"Radial Profile AUC:    {results.radial_profile_auc:.4f} (target < 0.65)")
    print(f"Samples:               {results.n_samples}")
    print(f"Status:                {'PASS' if results.passes() else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

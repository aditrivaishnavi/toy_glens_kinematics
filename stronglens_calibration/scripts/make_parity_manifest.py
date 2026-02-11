#!/usr/bin/env python3
"""
Generate Paper IV Parity Training Manifest.

Creates a training manifest that combines positives + sampled negatives with:
- Cutout paths pointing to NFS locations
- Two split schemes: 70/15/15 (audit) and 70/30 (Paper IV parity, val+test merged)
- Unweighted labels (sample_weight=1.0) for Paper IV parity baseline
- Quality filtering on negatives

Usage:
    python scripts/make_parity_manifest.py \
        --sampled-negatives /path/to/sampled_negatives/20260211_082238/data/ \
        --positives-manifest /path/to/manifests/training_v1.parquet \
        --neg-cutout-dir-old /path/to/cutouts/negatives/20260210_025117/ \
        --neg-cutout-dir-new /path/to/cutouts/negatives/20260211_143153/ \
        --pos-cutout-dir /path/to/cutouts/positives/ \
        --output /path/to/manifests/training_parity_v1.parquet

Author: stronglens_calibration project
Date: 2026-02-11
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def build_cutout_index(cutout_dirs: list) -> dict:
    """Pre-build index of galaxy_id -> full path for all cutouts.
    
    Much faster than per-file os.path.exists() on NFS.
    """
    index = {}
    for d in cutout_dirs:
        print(f"  Indexing {d}...")
        try:
            files = os.listdir(d)
        except OSError as e:
            print(f"    WARNING: Cannot list {d}: {e}")
            continue
        n = 0
        for f in files:
            if f.endswith(".npz"):
                gid = f[:-4]  # Remove .npz
                if gid not in index:  # First directory wins
                    index[gid] = os.path.join(d, f)
                    n += 1
        print(f"    Indexed {n} cutouts from this directory")
    print(f"  Total indexed: {len(index)} unique galaxy_ids")
    return index


def main():
    parser = argparse.ArgumentParser(description="Generate Paper IV parity training manifest")
    parser.add_argument("--sampled-negatives", required=True,
                        help="Path to sampled negatives parquet directory")
    parser.add_argument("--positives-manifest", required=True,
                        help="Path to existing training_v1.parquet (for positives)")
    parser.add_argument("--neg-cutout-dirs", nargs="+", required=True,
                        help="Directories containing negative cutout .npz files (checked in order)")
    parser.add_argument("--pos-cutout-dir", required=True,
                        help="Directory containing positive cutout .npz files")
    parser.add_argument("--output", required=True,
                        help="Output parquet path")
    parser.add_argument("--split-scheme", choices=["70_15_15", "70_30"], default="70_15_15",
                        help="Split scheme: 70_15_15 (audit, default) or 70_30 (Paper IV parity)")
    args = parser.parse_args()

    print("=" * 60)
    print("Paper IV Parity Manifest Generator")
    print("=" * 60)

    # ---- Load sampled negatives ----
    print(f"\nLoading sampled negatives from: {args.sampled_negatives}")
    neg = pd.read_parquet(args.sampled_negatives)
    print(f"  {len(neg)} negatives loaded")

    # ---- Load positives from existing manifest ----
    print(f"\nLoading positives from: {args.positives_manifest}")
    existing = pd.read_parquet(args.positives_manifest)
    pos = existing[existing["label"] == 1].copy()
    print(f"  {len(pos)} positives loaded")

    # ---- Build cutout index (fast NFS-friendly approach) ----
    print(f"\nBuilding cutout index from {len(args.neg_cutout_dirs)} directories...")
    cutout_index = build_cutout_index(args.neg_cutout_dirs)

    # ---- Assign cutout paths for negatives ----
    print(f"\nMapping cutout paths for {len(neg)} negatives...")
    neg["cutout_path"] = neg["galaxy_id"].map(cutout_index)
    missing = neg["cutout_path"].isna().sum()
    print(f"  Found: {len(neg) - missing}, Missing: {missing}")

    # Drop rows without cutouts
    if missing > 0:
        print(f"  Dropping {missing} negatives without cutouts")
        neg = neg[neg["cutout_path"].notna()].reset_index(drop=True)
        print(f"  Remaining: {len(neg)} negatives")

    # ---- Verify positive cutout paths ----
    print(f"\nIndexing positive cutouts...")
    pos_index = build_cutout_index([args.pos_cutout_dir])

    # Use existing cutout_path if valid, otherwise look up from index
    def resolve_pos_cutout(row):
        if "cutout_path" in row.index and pd.notna(row.get("cutout_path")):
            return row["cutout_path"]
        gid = row.get("galaxy_id", "")
        return pos_index.get(gid, None)

    pos["cutout_path"] = pos.apply(resolve_pos_cutout, axis=1)
    pos_missing = pos["cutout_path"].isna().sum()
    print(f"  Found: {len(pos) - pos_missing}, Missing: {pos_missing}")

    if pos_missing > 0:
        pos = pos[pos["cutout_path"].notna()].reset_index(drop=True)
        print(f"  Remaining: {len(pos)} positives")

    # ---- Build unified manifest ----
    print("\nBuilding unified manifest...")

    # Standardize negative columns
    neg_manifest = neg[["galaxy_id", "cutout_path", "ra", "dec", "type_bin",
                         "nobs_z_bin", "split", "pool", "confuser_category",
                         "psfsize_r", "psfdepth_r", "ebv", "healpix_128"]].copy()
    neg_manifest["label"] = 0
    neg_manifest["tier"] = "N"
    neg_manifest["sample_weight"] = 1.0  # Unweighted for Paper IV parity

    # Standardize positive columns
    pos_manifest = pd.DataFrame({
        "galaxy_id": pos["galaxy_id"],
        "cutout_path": pos["cutout_path"],
        "ra": pos.get("ra", pos.get("match_ra", np.nan)),
        "dec": pos.get("dec", pos.get("match_dec", np.nan)),
        "type_bin": pos.get("type_bin", pos.get("match_type", "UNK")),
        "nobs_z_bin": pos.get("nobs_z_bin", "UNK"),
        "split": pos["split"],
        "pool": "POS",
        "confuser_category": "none",
        "psfsize_r": pos.get("psfsize_r", np.nan),
        "psfdepth_r": pos.get("psfdepth_r", np.nan),
        "ebv": pos.get("ebv", np.nan),
        "healpix_128": pos.get("healpix_128", np.nan),
        "label": 1,
        "tier": pos.get("tier", "A"),
        "sample_weight": 1.0,  # Unweighted for Paper IV parity
    })

    manifest = pd.concat([pos_manifest, neg_manifest], ignore_index=True)

    # ---- Handle split scheme ----
    if args.split_scheme == "70_30":
        # Paper IV parity: merge val+test into val (70/30 train/val)
        print("\nApplying 70/30 split scheme (Paper IV parity)...")
        manifest.loc[manifest["split"] == "test", "split"] = "val"
        split_counts = manifest["split"].value_counts()
        for s, c in split_counts.items():
            pct = c / len(manifest) * 100
            print(f"  {s}: {c:,} ({pct:.1f}%)")
    else:
        print("\nKeeping 70/15/15 split scheme (audit)...")
        split_counts = manifest["split"].value_counts()
        for s, c in split_counts.items():
            pct = c / len(manifest) * 100
            print(f"  {s}: {c:,} ({pct:.1f}%)")

    # ---- Summary ----
    n_pos = (manifest["label"] == 1).sum()
    n_neg = (manifest["label"] == 0).sum()
    ratio = n_neg / n_pos if n_pos > 0 else 0

    print(f"\n{'='*60}")
    print(f"Manifest Summary")
    print(f"{'='*60}")
    print(f"  Total: {len(manifest):,}")
    print(f"  Positives: {n_pos:,}")
    print(f"  Negatives: {n_neg:,}")
    print(f"  Ratio: {ratio:.1f}:1")
    print(f"  Split scheme: {args.split_scheme}")
    print(f"  All weights = 1.0: {(manifest['sample_weight'] == 1.0).all()}")

    # ---- Save ----
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    manifest.to_parquet(args.output, index=False)
    print(f"\nSaved to: {args.output}")

    # Save metadata
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_total": len(manifest),
        "n_positives": int(n_pos),
        "n_negatives": int(n_neg),
        "neg_pos_ratio": round(ratio, 1),
        "split_scheme": args.split_scheme,
        "sampled_negatives_source": args.sampled_negatives,
        "positives_source": args.positives_manifest,
        "neg_cutout_dirs": args.neg_cutout_dirs,
        "unweighted": True,
    }
    meta_path = args.output.replace(".parquet", ".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()

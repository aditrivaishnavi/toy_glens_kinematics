"""
Build unpaired training manifest from directory-based split structure.

This creates a manifest where:
- Positives come from one set of LRGs (using stamp_npz)
- Negatives come from a DIFFERENT set of LRGs (using ctrl_stamp_npz)
- Negatives are matched to positives by property bins

Adapted for our data structure:
- Splits are in separate directories (train/, val/, test/)
- Uses psf_bin and depth_bin for matching (pre-computed in data)
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .constants import MATCHING_COLUMNS, SEED_DEFAULT

logger = logging.getLogger(__name__)


def load_split_data(data_root: Path, split: str, max_files: Optional[int] = None) -> pd.DataFrame:
    """Load all parquet files from a split directory."""
    split_dir = data_root / split
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")
    
    files = sorted(split_dir.glob("*.parquet"))
    if max_files:
        files = files[:max_files]
    
    logger.info(f"Loading {len(files)} files from {split_dir}")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")
    
    if not dfs:
        raise ValueError(f"No valid parquet files in {split_dir}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined["split"] = split
    return combined


def create_lrg_id(df: pd.DataFrame, ra_col: str = "ra", dec_col: str = "dec") -> pd.DataFrame:
    """Create unique LRG identifier from ra/dec."""
    df = df.copy()
    df["_lrg_id"] = list(zip(
        df[ra_col].astype(float).round(6),
        df[dec_col].astype(float).round(6)
    ))
    return df


def split_lrgs_disjoint(
    df: pd.DataFrame,
    split: str,
    rng: np.random.Generator,
    pos_fraction: float = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split LRGs into disjoint positive and negative pools.
    
    Returns:
        (pos_df, neg_df) where LRGs in each are disjoint
    """
    split_df = df[df["split"] == split].copy()
    
    # Get unique LRGs
    lrgs = split_df["_lrg_id"].unique()
    rng.shuffle(lrgs)
    
    # Split into pos/neg pools
    mid = int(len(lrgs) * pos_fraction)
    pos_lrgs = set(lrgs[:mid])
    neg_lrgs = set(lrgs[mid:])
    
    pos_df = split_df[split_df["_lrg_id"].isin(pos_lrgs)].copy()
    neg_df = split_df[split_df["_lrg_id"].isin(neg_lrgs)].copy()
    
    logger.info(f"{split}: {len(pos_lrgs)} pos LRGs ({len(pos_df)} samples), "
                f"{len(neg_lrgs)} neg LRGs ({len(neg_df)} samples)")
    
    return pos_df, neg_df


def match_negatives_to_positives(
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    matching_cols: Sequence[str],
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Sample negatives to match positive distribution in matching_cols.
    
    For each bin defined by matching_cols, sample same number of negatives as positives.
    """
    # Check which columns exist
    available_cols = [c for c in matching_cols if c in pos_df.columns and c in neg_df.columns]
    if not available_cols:
        logger.warning(f"No matching columns available, using all negatives. "
                      f"Requested: {matching_cols}, Available in pos: {list(pos_df.columns)[:10]}")
        return neg_df
    
    logger.info(f"Matching on columns: {available_cols}")
    
    # Count positives per bin
    pos_counts = pos_df.groupby(available_cols).size().reset_index(name="n_pos")
    
    # Add bin info to negatives
    neg_with_counts = neg_df.merge(pos_counts, on=available_cols, how="left")
    neg_with_counts["n_pos"] = neg_with_counts["n_pos"].fillna(0).astype(int)
    
    # Sample negatives to match positives per bin
    sampled = []
    for _, grp in neg_with_counts.groupby(available_cols, dropna=False):
        n_target = int(grp["n_pos"].iloc[0])
        if n_target <= 0:
            continue
        
        n_available = len(grp)
        if n_available >= n_target:
            take = grp.sample(n=n_target, replace=False,
                            random_state=int(rng.integers(0, 2**31-1)))
        else:
            # Oversample if not enough negatives
            take = grp.sample(n=n_target, replace=True,
                            random_state=int(rng.integers(0, 2**31-1)))
        sampled.append(take.drop(columns=["n_pos"]))
    
    if not sampled:
        logger.warning("No matched negatives found!")
        return neg_df.head(0)
    
    matched = pd.concat(sampled, ignore_index=True)
    logger.info(f"Matched {len(matched)} negatives to {len(pos_df)} positives")
    return matched


def build_unpaired_manifest(
    data_root: str,
    output_path: str,
    matching_cols: Sequence[str] = MATCHING_COLUMNS,
    seed: int = SEED_DEFAULT,
    pos_blob_col: str = "stamp_npz",
    neg_blob_col: str = "ctrl_stamp_npz",
    max_files_per_split: Optional[int] = None,
) -> str:
    """
    Build unpaired manifest from directory-based data.
    
    Args:
        data_root: Root directory containing train/, val/, test/ subdirectories
        output_path: Where to save the manifest parquet
        matching_cols: Columns to use for matching negatives to positives
        seed: Random seed for reproducibility
        pos_blob_col: Column containing positive image blobs
        neg_blob_col: Column containing negative image blobs
        max_files_per_split: Limit files per split (for mini experiments)
    
    Returns:
        Path to created manifest
    """
    data_root = Path(data_root)
    rng = np.random.default_rng(seed)
    
    # Load all splits
    all_dfs = []
    for split in ["train", "val", "test"]:
        try:
            df = load_split_data(data_root, split, max_files=max_files_per_split)
            all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not load {split}: {e}")
    
    if not all_dfs:
        raise ValueError("No data loaded from any split")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = create_lrg_id(combined)
    
    logger.info(f"Total samples: {len(combined)}, unique LRGs: {combined['_lrg_id'].nunique()}")
    
    # Process each split
    manifest_rows = []
    
    for split in ["train", "val", "test"]:
        if split not in combined["split"].values:
            continue
        
        # Split LRGs into disjoint pools
        pos_df, neg_df = split_lrgs_disjoint(combined, split, rng)
        
        # Match negatives to positives
        matched_neg = match_negatives_to_positives(pos_df, neg_df, matching_cols, rng)
        
        # Create positive rows
        for _, row in pos_df.iterrows():
            mrow = {
                "blob": row[pos_blob_col],
                "label": 1,
                "split": split,
                "_lrg_id": row["_lrg_id"],
                "ra": row["ra"],
                "dec": row["dec"],
            }
            # Include theta_e_arcsec for stratified evaluation
            if "theta_e_arcsec" in row.index:
                mrow["theta_e_arcsec"] = row["theta_e_arcsec"]
            manifest_rows.append(mrow)
        
        # Create negative rows
        for _, row in matched_neg.iterrows():
            mrow = {
                "blob": row[neg_blob_col],
                "label": 0,
                "split": split,
                "_lrg_id": row["_lrg_id"],
                "ra": row["ra"],
                "dec": row["dec"],
            }
            # Negatives don't have theta_e, set to NaN for consistency
            mrow["theta_e_arcsec"] = float("nan")
            manifest_rows.append(mrow)
    
    manifest = pd.DataFrame(manifest_rows)
    
    # Summary
    for split in ["train", "val", "test"]:
        split_data = manifest[manifest["split"] == split]
        n_pos = (split_data["label"] == 1).sum()
        n_neg = (split_data["label"] == 0).sum()
        logger.info(f"{split}: {n_pos} positives, {n_neg} negatives")
    
    # Save
    table = pa.Table.from_pandas(manifest)
    pq.write_table(table, output_path, compression="zstd")
    logger.info(f"Saved manifest to {output_path}")
    
    return output_path


def main():
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="Build unpaired training manifest")
    parser.add_argument("--data-root", required=True, help="Root directory with train/val/test subdirs")
    parser.add_argument("--output", required=True, help="Output manifest parquet path")
    parser.add_argument("--matching-cols", default="psf_bin,depth_bin",
                       help="Comma-separated columns for matching")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--max-files", type=int, default=None,
                       help="Max files per split (for mini experiments)")
    
    args = parser.parse_args()
    
    matching_cols = [c.strip() for c in args.matching_cols.split(",") if c.strip()]
    
    build_unpaired_manifest(
        data_root=args.data_root,
        output_path=args.output,
        matching_cols=matching_cols,
        seed=args.seed,
        max_files_per_split=args.max_files,
    )


if __name__ == "__main__":
    main()

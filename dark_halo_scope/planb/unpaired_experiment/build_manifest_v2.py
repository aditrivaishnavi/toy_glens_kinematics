"""
Build unpaired training manifest V2 - METADATA ONLY.

Key changes from v1:
1. Does NOT embed image blobs in manifest
2. Stores file paths and row indices instead
3. Partitions output by split for scalability
4. Manifest size: ~10-50 MB instead of 68 GB

Data loader reads blobs on-the-fly from source parquet files.
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


def load_split_metadata(data_root: Path, split: str, max_files: Optional[int] = None) -> pd.DataFrame:
    """Load metadata from parquet files without loading blobs."""
    split_dir = data_root / split
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")
    
    files = sorted(split_dir.glob("*.parquet"))
    if max_files:
        files = files[:max_files]
    
    logger.info(f"Loading metadata from {len(files)} files in {split_dir}")
    
    # Columns to load (exclude blob columns)
    metadata_cols = ["ra", "dec", "theta_e_arcsec", "psf_bin", "depth_bin"]
    
    rows = []
    for file_idx, f in enumerate(files):
        try:
            # Read only metadata columns
            df = pd.read_parquet(f, columns=metadata_cols)
            
            # Add file reference
            df["_source_file"] = str(f)
            df["_source_file_idx"] = file_idx
            df["_row_idx"] = range(len(df))
            
            rows.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")
    
    if not rows:
        raise ValueError(f"No valid parquet files in {split_dir}")
    
    combined = pd.concat(rows, ignore_index=True)
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
    """Split LRGs into disjoint positive and negative pools."""
    split_df = df[df["split"] == split].copy()
    
    lrgs = split_df["_lrg_id"].unique()
    rng.shuffle(lrgs)
    
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
    """Sample negatives to match positive distribution."""
    available_cols = [c for c in matching_cols if c in pos_df.columns and c in neg_df.columns]
    if not available_cols:
        logger.warning(f"No matching columns available, using all negatives")
        return neg_df
    
    logger.info(f"Matching on columns: {available_cols}")
    
    pos_counts = pos_df.groupby(available_cols).size().reset_index(name="n_pos")
    neg_with_counts = neg_df.merge(pos_counts, on=available_cols, how="left")
    neg_with_counts["n_pos"] = neg_with_counts["n_pos"].fillna(0).astype(int)
    
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
            take = grp.sample(n=n_target, replace=True,
                            random_state=int(rng.integers(0, 2**31-1)))
        sampled.append(take.drop(columns=["n_pos"]))
    
    if not sampled:
        logger.warning("No matched negatives found!")
        return neg_df.head(0)
    
    matched = pd.concat(sampled, ignore_index=True)
    logger.info(f"Matched {len(matched)} negatives to {len(pos_df)} positives")
    return matched


def build_manifest_v2(
    data_root: str,
    output_dir: str,
    matching_cols: Sequence[str] = MATCHING_COLUMNS,
    seed: int = SEED_DEFAULT,
    max_files_per_split: Optional[int] = None,
) -> str:
    """
    Build metadata-only manifest partitioned by split.
    
    Output structure:
        output_dir/
            split=train/part-0000.parquet
            split=val/part-0000.parquet
            split=test/part-0000.parquet
            _metadata.json
    
    Each row contains:
        - label: 0 or 1
        - blob_type: "stamp_npz" or "ctrl_stamp_npz"
        - _source_file: path to source parquet
        - _row_idx: row index in source file
        - theta_e_arcsec: Einstein radius (NaN for negatives)
        - ra, dec, psf_bin, depth_bin, _lrg_id
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(seed)
    
    # Load metadata from all splits
    all_dfs = []
    for split in ["train", "val", "test"]:
        try:
            df = load_split_metadata(data_root, split, max_files=max_files_per_split)
            all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not load {split}: {e}")
    
    if not all_dfs:
        raise ValueError("No data loaded from any split")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = create_lrg_id(combined)
    
    logger.info(f"Total samples: {len(combined)}, unique LRGs: {combined['_lrg_id'].nunique()}")
    
    # Process each split and write partitioned output
    total_rows = {"train": 0, "val": 0, "test": 0}
    
    for split in ["train", "val", "test"]:
        if split not in combined["split"].values:
            continue
        
        pos_df, neg_df = split_lrgs_disjoint(combined, split, rng)
        matched_neg = match_negatives_to_positives(pos_df, neg_df, matching_cols, rng)
        
        # Create manifest rows (metadata only, no blobs)
        manifest_rows = []
        
        # Positives - use stamp_npz
        for _, row in pos_df.iterrows():
            manifest_rows.append({
                "label": 1,
                "blob_type": "stamp_npz",
                "_source_file": row["_source_file"],
                "_row_idx": row["_row_idx"],
                "theta_e_arcsec": row.get("theta_e_arcsec", float("nan")),
                "ra": row["ra"],
                "dec": row["dec"],
                "psf_bin": row.get("psf_bin"),
                "depth_bin": row.get("depth_bin"),
                "_lrg_id": str(row["_lrg_id"]),
            })
        
        # Negatives - use ctrl_stamp_npz
        for _, row in matched_neg.iterrows():
            manifest_rows.append({
                "label": 0,
                "blob_type": "ctrl_stamp_npz",
                "_source_file": row["_source_file"],
                "_row_idx": row["_row_idx"],
                "theta_e_arcsec": float("nan"),
                "ra": row["ra"],
                "dec": row["dec"],
                "psf_bin": row.get("psf_bin"),
                "depth_bin": row.get("depth_bin"),
                "_lrg_id": str(row["_lrg_id"]),
            })
        
        # Write partitioned parquet
        split_df = pd.DataFrame(manifest_rows)
        split_dir = output_dir / f"split={split}"
        split_dir.mkdir(exist_ok=True)
        
        # Write in chunks for very large splits
        chunk_size = 500_000
        for i, start in enumerate(range(0, len(split_df), chunk_size)):
            chunk = split_df.iloc[start:start + chunk_size]
            chunk_path = split_dir / f"part-{i:04d}.parquet"
            chunk.to_parquet(chunk_path, index=False, compression="zstd")
        
        n_pos = (split_df["label"] == 1).sum()
        n_neg = (split_df["label"] == 0).sum()
        total_rows[split] = len(split_df)
        logger.info(f"{split}: {n_pos} positives, {n_neg} negatives, written to {split_dir}")
    
    # Write metadata
    import json
    metadata = {
        "version": "v2",
        "format": "partitioned_parquet",
        "created_date": pd.Timestamp.now().isoformat(),
        "seed": seed,
        "matching_columns": list(matching_cols),
        "source_data": str(data_root),
        "splits": total_rows,
        "columns": ["label", "blob_type", "_source_file", "_row_idx", 
                   "theta_e_arcsec", "ra", "dec", "psf_bin", "depth_bin", "_lrg_id"],
        "note": "Blobs not embedded. Data loader reads from _source_file at _row_idx.",
    }
    
    with open(output_dir / "_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Manifest saved to {output_dir}")
    logger.info(f"Total: {sum(total_rows.values())} rows")
    
    return str(output_dir)


def main():
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="Build metadata-only manifest (v2)")
    parser.add_argument("--data-root", required=True, help="Root directory with train/val/test subdirs")
    parser.add_argument("--output-dir", required=True, help="Output directory for partitioned manifest")
    parser.add_argument("--matching-cols", default="psf_bin,depth_bin",
                       help="Comma-separated columns for matching")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--max-files", type=int, default=None,
                       help="Max files per split (for mini experiments)")
    
    args = parser.parse_args()
    
    matching_cols = [c.strip() for c in args.matching_cols.split(",") if c.strip()]
    
    build_manifest_v2(
        data_root=args.data_root,
        output_dir=args.output_dir,
        matching_cols=matching_cols,
        seed=args.seed,
        max_files_per_split=args.max_files,
    )


if __name__ == "__main__":
    main()

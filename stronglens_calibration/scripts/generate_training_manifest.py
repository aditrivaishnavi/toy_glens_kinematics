#!/usr/bin/env python3
"""
Generate Training Manifest from Cutouts

Bridges the gap between cutout generation (.npz files) and training (parquet manifest).
Creates a parquet manifest with columns expected by dhs/data.py:
  - s3_path: Full S3 path to the .npz file
  - label: 1 for positives, 0 for negatives
  - split: train/val/test
  - galaxy_id: Unique identifier
  - ra, dec: Coordinates
  - Other metadata columns

Usage:
    python scripts/generate_training_manifest.py \
        --positives s3://darkhaloscope/stronglens_calibration/cutouts/positives/TIMESTAMP/ \
        --negatives s3://darkhaloscope/stronglens_calibration/cutouts/negatives/TIMESTAMP/ \
        --output s3://darkhaloscope/stronglens_calibration/training_manifests/TIMESTAMP/

Author: Generated for stronglens_calibration project
Date: 2026-02-05
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import boto3
import numpy as np
import pandas as pd

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import AWS_REGION, S3_BUCKET

# =============================================================================
# CONSTANTS
# =============================================================================

PIPELINE_VERSION = "1.0.0"

# Split ratios (standard 80/10/10)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TrainingManifest] %(levelname)s: %(message)s",
)
logger = logging.getLogger("TrainingManifest")


# =============================================================================
# S3 UTILITIES
# =============================================================================

def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    if s3_uri.startswith("s3://"):
        path = s3_uri[5:]
        bucket, _, key = path.partition("/")
        return bucket, key
    raise ValueError(f"Invalid S3 URI: {s3_uri}")


def list_npz_files(s3_client, bucket: str, prefix: str) -> List[Dict]:
    """List all .npz files under a prefix, extracting metadata from filenames."""
    files = []
    paginator = s3_client.get_paginator("list_objects_v2")
    
    # Ensure prefix ends correctly
    if not prefix.endswith("/"):
        prefix += "/"
    
    # Look in data/ subdirectory
    data_prefix = prefix + "data/"
    
    for page in paginator.paginate(Bucket=bucket, Prefix=data_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".npz"):
                # Extract galaxy_id from filename (e.g., "galaxy_12345.npz" -> "12345")
                filename = os.path.basename(key)
                galaxy_id = filename.replace(".npz", "").replace("galaxy_", "").replace("neg_", "").replace("pos_", "")
                
                files.append({
                    "s3_key": key,
                    "s3_path": f"s3://{bucket}/{key}",
                    "galaxy_id": galaxy_id,
                    "filename": filename,
                })
    
    return files


def extract_metadata_from_npz(s3_client, bucket: str, key: str) -> Optional[Dict]:
    """Extract metadata from NPZ file header (stored as 'metadata' array)."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        data = np.load(response["Body"], allow_pickle=True)
        
        # Get coordinates if stored
        metadata = {}
        if "ra" in data.files:
            metadata["ra"] = float(data["ra"])
        if "dec" in data.files:
            metadata["dec"] = float(data["dec"])
        if "metadata" in data.files:
            # Metadata stored as dictionary
            stored = data["metadata"].item() if data["metadata"].ndim == 0 else dict(data["metadata"])
            metadata.update(stored)
            
        return metadata
    except Exception as e:
        logger.warning(f"Could not extract metadata from {key}: {e}")
        return None


def assign_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Assign train/val/test splits stratified by label."""
    np.random.seed(seed)
    
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Assign splits within each label group
    splits = []
    for label in [0, 1]:
        subset = df[df["label"] == label].copy()
        n = len(subset)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        
        subset_splits = (
            ["train"] * n_train +
            ["val"] * n_val +
            ["test"] * (n - n_train - n_val)
        )
        subset["split"] = subset_splits
        splits.append(subset)
    
    return pd.concat(splits, ignore_index=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate training manifest from cutouts")
    parser.add_argument("--positives", required=True, help="S3 path to positive cutouts")
    parser.add_argument("--negatives", required=True, help="S3 path to negative cutouts")
    parser.add_argument("--output", required=True, help="S3 path for output manifest")
    parser.add_argument("--sample-metadata", type=int, default=100,
                        help="Number of files to sample for metadata extraction (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    args = parser.parse_args()
    
    start_time = datetime.now(timezone.utc)
    logger.info(f"Starting manifest generation v{PIPELINE_VERSION}")
    logger.info(f"Positives: {args.positives}")
    logger.info(f"Negatives: {args.negatives}")
    
    s3 = boto3.client("s3", region_name=AWS_REGION)
    
    # Parse paths
    pos_bucket, pos_prefix = parse_s3_uri(args.positives)
    neg_bucket, neg_prefix = parse_s3_uri(args.negatives)
    out_bucket, out_prefix = parse_s3_uri(args.output)
    
    # List all cutout files
    logger.info("Listing positive cutouts...")
    pos_files = list_npz_files(s3, pos_bucket, pos_prefix)
    logger.info(f"  Found {len(pos_files)} positive cutouts")
    
    logger.info("Listing negative cutouts...")
    neg_files = list_npz_files(s3, neg_bucket, neg_prefix)
    logger.info(f"  Found {len(neg_files)} negative cutouts")
    
    if len(pos_files) == 0:
        logger.error("No positive cutouts found!")
        sys.exit(1)
    if len(neg_files) == 0:
        logger.error("No negative cutouts found!")
        sys.exit(1)
    
    # Build dataframes
    pos_df = pd.DataFrame(pos_files)
    pos_df["label"] = 1
    pos_df["cutout_type"] = "positive"
    
    neg_df = pd.DataFrame(neg_files)
    neg_df["label"] = 0
    neg_df["cutout_type"] = "negative"
    
    # Combine
    df = pd.concat([pos_df, neg_df], ignore_index=True)
    logger.info(f"Total samples: {len(df)} (pos={len(pos_df)}, neg={len(neg_df)})")
    
    # Sample metadata from a few files to get ra/dec
    # (Skip if too slow - coordinates are in the .npz files anyway)
    if args.sample_metadata > 0:
        logger.info(f"Sampling metadata from {min(args.sample_metadata, len(df))} files...")
        sample_indices = np.random.choice(len(df), min(args.sample_metadata, len(df)), replace=False)
        for idx in sample_indices:
            row = df.iloc[idx]
            bucket = pos_bucket if row["label"] == 1 else neg_bucket
            metadata = extract_metadata_from_npz(s3, bucket, row["s3_key"])
            if metadata:
                for k, v in metadata.items():
                    if k not in df.columns:
                        df[k] = None
                    df.loc[idx, k] = v
    
    # Assign splits
    logger.info("Assigning train/val/test splits...")
    df = assign_splits(df, seed=args.seed)
    
    # Log split distribution
    split_counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    logger.info(f"Split distribution:\n{split_counts}")
    
    # Prepare output
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    if not out_prefix.endswith("/"):
        out_prefix += "/"
    out_prefix += f"{timestamp}/"
    
    # Save manifest as parquet
    manifest_key = out_prefix + "manifest.parquet"
    logger.info(f"Saving manifest to s3://{out_bucket}/{manifest_key}")
    
    # Write to buffer and upload
    import io
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3.put_object(Bucket=out_bucket, Key=manifest_key, Body=buffer.getvalue())
    
    # Save summary JSON
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "created_at": start_time.isoformat(),
        "total_samples": len(df),
        "positives": len(pos_df),
        "negatives": len(neg_df),
        "ratio": round(len(neg_df) / len(pos_df), 1) if len(pos_df) > 0 else 0,
        "splits": {
            "train": int((df["split"] == "train").sum()),
            "val": int((df["split"] == "val").sum()),
            "test": int((df["split"] == "test").sum()),
        },
        "source_paths": {
            "positives": args.positives,
            "negatives": args.negatives,
        },
        "output_path": f"s3://{out_bucket}/{out_prefix}",
    }
    
    summary_key = out_prefix + "summary.json"
    s3.put_object(
        Bucket=out_bucket,
        Key=summary_key,
        Body=json.dumps(summary, indent=2),
        ContentType="application/json",
    )
    
    # Print summary
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info("=" * 60)
    logger.info("MANIFEST GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"  Positives: {len(pos_df)}")
    logger.info(f"  Negatives: {len(neg_df)}")
    logger.info(f"  Ratio: {len(neg_df) / len(pos_df):.1f}:1")
    logger.info(f"Output: s3://{out_bucket}/{out_prefix}")
    logger.info(f"Elapsed: {elapsed:.1f}s")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

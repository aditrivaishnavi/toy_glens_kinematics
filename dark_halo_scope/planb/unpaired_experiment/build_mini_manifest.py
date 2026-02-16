"""
Build mini manifest for quick validation experiments.

Creates a smaller balanced subset from the full unpaired manifest.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def build_mini_manifest(
    full_manifest_path: str,
    output_path: str,
    train_samples: int = 5000,
    val_samples: int = 2000,
    seed: int = 42,
) -> None:
    """Build mini manifest from full manifest."""
    print(f"Loading full manifest: {full_manifest_path}")
    df = pd.read_parquet(full_manifest_path)
    
    print(f"Full manifest: {len(df)} rows")
    print(f"  Train: {len(df[df['split'] == 'train'])}")
    print(f"  Val: {len(df[df['split'] == 'val'])}")
    print(f"  Test: {len(df[df['split'] == 'test'])}")
    
    rng = np.random.default_rng(seed)
    
    mini_dfs = []
    
    for split, n_samples in [("train", train_samples), ("val", val_samples), ("test", val_samples)]:
        split_df = df[df["split"] == split].copy()
        
        # Balance positive and negative
        pos_df = split_df[split_df["label"] == 1]
        neg_df = split_df[split_df["label"] == 0]
        
        n_each = n_samples // 2
        
        if len(pos_df) < n_each:
            print(f"  Warning: {split} has only {len(pos_df)} positives, using all")
            pos_sample = pos_df
        else:
            pos_sample = pos_df.sample(n=n_each, random_state=seed)
        
        if len(neg_df) < n_each:
            print(f"  Warning: {split} has only {len(neg_df)} negatives, using all")
            neg_sample = neg_df
        else:
            neg_sample = neg_df.sample(n=n_each, random_state=seed)
        
        split_mini = pd.concat([pos_sample, neg_sample], ignore_index=True)
        split_mini = split_mini.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        mini_dfs.append(split_mini)
        print(f"  {split}: {len(split_mini)} samples ({len(pos_sample)} pos, {len(neg_sample)} neg)")
    
    mini_df = pd.concat(mini_dfs, ignore_index=True)
    
    print(f"\nMini manifest: {len(mini_df)} total rows")
    
    # Preserve theta_e column if present for stratified evaluation
    if "theta_e_arcsec" in mini_df.columns:
        print(f"  θ_E range: {mini_df['theta_e_arcsec'].min():.2f} - {mini_df['theta_e_arcsec'].max():.2f}")
    
    mini_df.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build mini manifest")
    parser.add_argument("--input", required=True, help="Full manifest path")
    parser.add_argument("--output", required=True, help="Output mini manifest path")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    build_mini_manifest(
        args.input,
        args.output,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

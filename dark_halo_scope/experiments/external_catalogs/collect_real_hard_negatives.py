"""
Collect real hard negatives for training data augmentation.

This script collects hard negatives from multiple sources:
1. High-scoring false positives from model inference
2. Known ring galaxy catalogs
3. Known merger catalogs
4. SIMBAD queries for lens-mimic objects

Usage:
    python collect_real_hard_negatives.py \
        --model-scores /path/to/inference_scores.parquet \
        --ring-catalog /path/to/rings.parquet \
        --merger-catalog /path/to/mergers.parquet \
        --output-dir ./hard_negatives/ \
        --min-score 0.8 \
        --top-k 5000
"""

import argparse
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class HardNegativeConfig:
    """Configuration for hard negative collection."""
    # Score-based mining
    min_score_threshold: float = 0.8
    top_k_per_source: int = 5000
    
    # Known lens exclusion
    known_lens_catalog: Optional[str] = None
    exclusion_radius_arcsec: float = 5.0
    
    # Output format
    include_cutouts: bool = False
    stamp_size: int = 64
    
    # Deduplication
    dedup_radius_arcsec: float = 2.0
    
    # Weighting for training
    default_weight: float = 1.0


def load_model_scores(
    scores_path: str,
    min_score: float = 0.8,
    top_k: int = 5000,
    label_col: str = 'label'
) -> pd.DataFrame:
    """
    Load high-scoring negatives from model inference.
    
    Parameters
    ----------
    scores_path : str
        Path to parquet with (ra, dec, score, label) columns
    min_score : float
        Minimum score threshold
    top_k : int
        Maximum number to keep
    label_col : str
        Column name for label (0=negative)
    
    Returns
    -------
    pd.DataFrame
        High-scoring negatives
    """
    logger.info(f"Loading model scores from {scores_path}")
    
    df = pd.read_parquet(scores_path)
    
    # Filter to negatives (label=0) with high scores
    if label_col in df.columns:
        df = df[df[label_col] == 0]
    
    # Filter by score
    score_col = 'score' if 'score' in df.columns else 'p_lens'
    df = df[df[score_col] >= min_score]
    
    # Sort and take top-k
    df = df.sort_values(score_col, ascending=False).head(top_k)
    
    # Standardize columns
    result = pd.DataFrame({
        'ra': df['ra'].values,
        'dec': df['dec'].values,
        'score': df[score_col].values,
        'source': 'model_fp',
        'weight': 1.0
    })
    
    logger.info(f"Selected {len(result)} high-scoring false positives")
    return result


def load_catalog_hard_negatives(
    catalog_path: str,
    source_name: str,
    ra_col: str = 'ra',
    dec_col: str = 'dec',
    max_count: int = 5000
) -> pd.DataFrame:
    """
    Load hard negatives from a catalog file.
    
    Parameters
    ----------
    catalog_path : str
        Path to catalog parquet
    source_name : str
        Name for this source (e.g., 'ring_galaxy', 'merger')
    ra_col, dec_col : str
        Column names for coordinates
    max_count : int
        Maximum number to include
    
    Returns
    -------
    pd.DataFrame
        Catalog entries
    """
    logger.info(f"Loading {source_name} catalog from {catalog_path}")
    
    df = pd.read_parquet(catalog_path)
    
    # Take random subset if too many
    if len(df) > max_count:
        df = df.sample(n=max_count, random_state=42)
    
    result = pd.DataFrame({
        'ra': df[ra_col].values,
        'dec': df[dec_col].values,
        'score': np.nan,  # No model score for catalog entries
        'source': source_name,
        'weight': 1.0
    })
    
    logger.info(f"Loaded {len(result)} entries from {source_name}")
    return result


def exclude_known_lenses(
    hard_negatives: pd.DataFrame,
    known_lens_catalog: str,
    exclusion_radius_arcsec: float = 5.0
) -> pd.DataFrame:
    """
    Remove entries that match known lens positions.
    
    Parameters
    ----------
    hard_negatives : pd.DataFrame
        Hard negative candidates
    known_lens_catalog : str
        Path to known lens catalog
    exclusion_radius_arcsec : float
        Exclusion radius for matching
    
    Returns
    -------
    pd.DataFrame
        Filtered hard negatives
    """
    logger.info(f"Excluding known lenses from {known_lens_catalog}")
    
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    
    known = pd.read_parquet(known_lens_catalog)
    
    # Build sky coordinates
    hn_coords = SkyCoord(
        ra=hard_negatives['ra'].values * u.degree,
        dec=hard_negatives['dec'].values * u.degree
    )
    
    known_coords = SkyCoord(
        ra=known['ra'].values * u.degree,
        dec=known['dec'].values * u.degree
    )
    
    # Find matches
    idx, sep, _ = hn_coords.match_to_catalog_sky(known_coords)
    is_match = sep < exclusion_radius_arcsec * u.arcsec
    
    n_excluded = is_match.sum()
    result = hard_negatives[~is_match].copy()
    
    logger.info(f"Excluded {n_excluded} entries matching known lenses")
    return result


def deduplicate_by_position(
    df: pd.DataFrame,
    radius_arcsec: float = 2.0
) -> pd.DataFrame:
    """
    Remove duplicate entries based on sky position.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ra, dec columns
    radius_arcsec : float
        Deduplication radius
    
    Returns
    -------
    pd.DataFrame
        Deduplicated entries (keeps first occurrence)
    """
    from scipy.spatial import cKDTree
    
    # Simple approach: round to nearest 2 arcsec
    df = df.copy()
    precision = radius_arcsec / 3600.0  # Convert to degrees
    
    df['ra_round'] = (df['ra'] / precision).round() * precision
    df['dec_round'] = (df['dec'] / precision).round() * precision
    
    df = df.drop_duplicates(subset=['ra_round', 'dec_round'], keep='first')
    df = df.drop(columns=['ra_round', 'dec_round'])
    
    return df


def assign_weights(
    df: pd.DataFrame,
    weight_by_source: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Assign training weights based on source.
    
    Parameters
    ----------
    df : pd.DataFrame
        Hard negatives with 'source' column
    weight_by_source : dict
        Mapping from source name to weight
    
    Returns
    -------
    pd.DataFrame
        DataFrame with updated weights
    """
    if weight_by_source is None:
        weight_by_source = {
            'model_fp': 3.0,  # High-scoring FPs are most valuable
            'ring_galaxy': 2.0,
            'merger': 1.5,
            'simbad_ring': 1.0
        }
    
    df = df.copy()
    
    for source, weight in weight_by_source.items():
        mask = df['source'] == source
        df.loc[mask, 'weight'] = weight
    
    return df


def collect_hard_negatives(
    model_scores_path: Optional[str] = None,
    ring_catalog_path: Optional[str] = None,
    merger_catalog_path: Optional[str] = None,
    known_lens_path: Optional[str] = None,
    output_dir: str = './hard_negatives/',
    config: HardNegativeConfig = None
) -> pd.DataFrame:
    """
    Collect hard negatives from all sources.
    
    Parameters
    ----------
    model_scores_path : str, optional
        Path to model inference scores
    ring_catalog_path : str, optional
        Path to ring galaxy catalog
    merger_catalog_path : str, optional
        Path to merger catalog
    known_lens_path : str, optional
        Path to known lens catalog for exclusion
    output_dir : str
        Output directory
    config : HardNegativeConfig
        Collection configuration
    
    Returns
    -------
    pd.DataFrame
        Combined hard negative catalog
    """
    if config is None:
        config = HardNegativeConfig()
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_hard_negs = []
    
    # Collect from model scores
    if model_scores_path and os.path.exists(model_scores_path):
        df = load_model_scores(
            model_scores_path,
            min_score=config.min_score_threshold,
            top_k=config.top_k_per_source
        )
        all_hard_negs.append(df)
    
    # Collect from ring catalog
    if ring_catalog_path and os.path.exists(ring_catalog_path):
        df = load_catalog_hard_negatives(
            ring_catalog_path,
            source_name='ring_galaxy',
            max_count=config.top_k_per_source
        )
        all_hard_negs.append(df)
    
    # Collect from merger catalog
    if merger_catalog_path and os.path.exists(merger_catalog_path):
        df = load_catalog_hard_negatives(
            merger_catalog_path,
            source_name='merger',
            max_count=config.top_k_per_source
        )
        all_hard_negs.append(df)
    
    if not all_hard_negs:
        logger.warning("No hard negatives collected from any source")
        return pd.DataFrame()
    
    # Combine
    combined = pd.concat(all_hard_negs, ignore_index=True)
    logger.info(f"Combined {len(combined)} hard negatives from {len(all_hard_negs)} sources")
    
    # Exclude known lenses
    if known_lens_path and os.path.exists(known_lens_path):
        combined = exclude_known_lenses(
            combined,
            known_lens_path,
            exclusion_radius_arcsec=config.exclusion_radius_arcsec
        )
    
    # Deduplicate
    combined = deduplicate_by_position(combined, radius_arcsec=config.dedup_radius_arcsec)
    logger.info(f"After deduplication: {len(combined)} hard negatives")
    
    # Assign weights
    combined = assign_weights(combined)
    
    # Save
    output_path = os.path.join(output_dir, 'hard_negatives_combined.parquet')
    combined.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(combined)} hard negatives to {output_path}")
    
    # Save summary
    summary = combined.groupby('source').agg({
        'ra': 'count',
        'weight': 'mean'
    }).rename(columns={'ra': 'count', 'weight': 'avg_weight'})
    
    summary_path = os.path.join(output_dir, 'hard_negatives_summary.csv')
    summary.to_csv(summary_path)
    logger.info(f"Summary saved to {summary_path}")
    
    return combined


def create_training_lookup(
    hard_negatives_path: str,
    output_path: str,
    position_precision_arcsec: float = 1.0
) -> None:
    """
    Create a lookup table for fast hard negative identification during training.
    
    Parameters
    ----------
    hard_negatives_path : str
        Path to hard negatives parquet
    output_path : str
        Output parquet path
    position_precision_arcsec : float
        Position precision for hashing
    """
    logger.info(f"Creating training lookup from {hard_negatives_path}")
    
    df = pd.read_parquet(hard_negatives_path)
    
    # Create position hash for fast lookup
    precision = position_precision_arcsec / 3600.0
    df['ra_hash'] = (df['ra'] / precision).round().astype(int)
    df['dec_hash'] = (df['dec'] / precision).round().astype(int)
    df['pos_hash'] = df['ra_hash'].astype(str) + '_' + df['dec_hash'].astype(str)
    
    # Keep only lookup columns
    lookup = df[['pos_hash', 'weight', 'source']].copy()
    
    lookup.to_parquet(output_path, index=False)
    logger.info(f"Saved lookup table with {len(lookup)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect real hard negatives for training"
    )
    parser.add_argument(
        "--model-scores",
        type=str,
        help="Path to model inference scores parquet"
    )
    parser.add_argument(
        "--ring-catalog",
        type=str,
        help="Path to ring galaxy catalog"
    )
    parser.add_argument(
        "--merger-catalog",
        type=str,
        help="Path to merger catalog"
    )
    parser.add_argument(
        "--known-lenses",
        type=str,
        help="Path to known lens catalog for exclusion"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.8,
        help="Minimum model score threshold for FP mining"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5000,
        help="Maximum entries per source"
    )
    parser.add_argument(
        "--create-lookup",
        action="store_true",
        help="Also create training lookup table"
    )
    
    args = parser.parse_args()
    
    config = HardNegativeConfig(
        min_score_threshold=args.min_score,
        top_k_per_source=args.top_k
    )
    
    df = collect_hard_negatives(
        model_scores_path=args.model_scores,
        ring_catalog_path=args.ring_catalog,
        merger_catalog_path=args.merger_catalog,
        known_lens_path=args.known_lenses,
        output_dir=args.output_dir,
        config=config
    )
    
    if args.create_lookup and len(df) > 0:
        create_training_lookup(
            os.path.join(args.output_dir, 'hard_negatives_combined.parquet'),
            os.path.join(args.output_dir, 'hard_neg_lookup.parquet')
        )


if __name__ == "__main__":
    main()


"""
Deflector-matched (LRG-like) host selection from training manifest.

For Model 2, we inject arcs onto hosts that resemble real strong-lens deflectors:
massive red elliptical galaxies.  This module provides the selection logic.

Selection criteria:
    1. Tractor morphology type in {DEV, SER}: de Vaucouleurs and general Sersic
       profiles, which correspond to elliptical and early-type galaxies.
       REX (round exponential) is excluded as it represents disk-dominated systems.
    2. Label == 0 (negative): confirmed non-lenses to avoid contamination.
    3. Quality OK: quality_ok == True (if column exists).
    4. Optionally filter by PSF/depth bin (same as Model 1 grid).

Why DEV/SER?
    Real strong-lens deflectors are overwhelmingly massive early-type galaxies.
    The DESI Legacy Survey Tractor pipeline classifies these as DEV (de Vaucouleurs
    n=4 profile) or SER (general Sersic with n as free parameter, typically n > 2
    for early types).  By restricting hosts to these types, we ensure the injected
    arc+deflector systems look similar to what the CNN sees in real lenses.

    REX objects (round exponential, n=1) are disk-dominated and rarely act as
    strong-lens deflectors.

References:
    - Tractor morphological types: Lang et al. (2016)
    - SLACS deflector demographics: Bolton et al. (2006)
    - MNRAS_RAW_NOTES.md Section 9.3

Author: stronglens_calibration project
Date: 2026-02-13
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default LRG-like morphology types (Tractor)
LRG_TYPES: Set[str] = {"DEV", "SER"}


def select_lrg_hosts(
    manifest_df: pd.DataFrame,
    split: str = "val",
    lrg_types: Optional[Set[str]] = None,
    require_quality_ok: bool = True,
    label_col: str = "label",
    split_col: str = "split",
    type_col: str = "type_bin",
    quality_col: str = "quality_ok",
) -> pd.DataFrame:
    """Select LRG-like (deflector-matched) hosts from the manifest.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Full training manifest with at least columns: label, split, type_bin.
    split : str
        Which split to select from ("train", "val", "test").
    lrg_types : set of str or None
        Tractor morphology types to accept.  Default: {"DEV", "SER"}.
    require_quality_ok : bool
        If True and ``quality_col`` exists, additionally filter quality_ok == True.
    label_col, split_col, type_col, quality_col : str
        Column name overrides.

    Returns
    -------
    pd.DataFrame
        Filtered manifest rows (LRG-like negatives from the requested split).

    Raises
    ------
    ValueError
        If required columns are missing or no hosts match.
    """
    if lrg_types is None:
        lrg_types = LRG_TYPES

    # Validate required columns
    for col in [label_col, split_col, type_col]:
        if col not in manifest_df.columns:
            raise ValueError(
                f"Required column '{col}' not found in manifest. "
                f"Available columns: {list(manifest_df.columns)[:20]}..."
            )

    # Base filter: negatives from the requested split
    mask = (manifest_df[split_col] == split) & (manifest_df[label_col] == 0)

    # Morphology filter: LRG-like types
    mask = mask & manifest_df[type_col].isin(lrg_types)

    # Optional quality filter
    if require_quality_ok and quality_col in manifest_df.columns:
        mask = mask & (manifest_df[quality_col] == True)  # noqa: E712

    result = manifest_df[mask].copy().reset_index(drop=True)

    n_total_neg = int(
        ((manifest_df[split_col] == split) & (manifest_df[label_col] == 0)).sum()
    )
    n_lrg = len(result)

    logger.info(
        "LRG host selection: %d/%d negatives in '%s' are LRG-like (%s) = %.1f%%",
        n_lrg,
        n_total_neg,
        split,
        ", ".join(sorted(lrg_types)),
        100.0 * n_lrg / max(n_total_neg, 1),
    )

    if n_lrg == 0:
        # Log available type_bin distribution for debugging
        neg_split = manifest_df[
            (manifest_df[split_col] == split) & (manifest_df[label_col] == 0)
        ]
        if type_col in neg_split.columns:
            type_dist = neg_split[type_col].value_counts().to_dict()
            logger.warning(
                "No LRG hosts found! type_bin distribution in '%s' negatives: %s",
                split,
                type_dist,
            )
        raise ValueError(
            f"No LRG-like hosts found in split '{split}' with types {lrg_types}. "
            f"Check that the manifest has type_bin column with DEV/SER values."
        )

    return result


def select_random_hosts(
    manifest_df: pd.DataFrame,
    split: str = "val",
    require_quality_ok: bool = True,
    label_col: str = "label",
    split_col: str = "split",
    quality_col: str = "quality_ok",
) -> pd.DataFrame:
    """Select random (unmatched) hosts — Model 1 baseline selection.

    Same as the original grid runner: all negatives in the split, no
    morphology filter.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Full training manifest.
    split : str
        Which split.
    require_quality_ok : bool
        If True and quality_ok exists, filter on it.

    Returns
    -------
    pd.DataFrame
        All negative hosts from the split.
    """
    for col in [label_col, split_col]:
        if col not in manifest_df.columns:
            raise ValueError(f"Required column '{col}' not found in manifest")

    mask = (manifest_df[split_col] == split) & (manifest_df[label_col] == 0)

    if require_quality_ok and quality_col in manifest_df.columns:
        mask = mask & (manifest_df[quality_col] == True)  # noqa: E712

    result = manifest_df[mask].copy().reset_index(drop=True)

    if len(result) == 0:
        raise ValueError(f"No negative hosts found in split '{split}'")

    logger.info(
        "Random host selection: %d negatives in '%s'", len(result), split
    )
    return result


def host_selection_summary(
    manifest_df: pd.DataFrame,
    split: str = "val",
    label_col: str = "label",
    split_col: str = "split",
    type_col: str = "type_bin",
) -> Dict[str, int]:
    """Return a summary of host type distribution in the requested split.

    Useful for sanity-checking before running experiments.

    Returns
    -------
    dict
        Mapping from type_bin value to count of negatives.
    """
    neg = manifest_df[
        (manifest_df[split_col] == split) & (manifest_df[label_col] == 0)
    ]
    if type_col not in neg.columns:
        return {"_total": len(neg), "_no_type_col": True}

    counts = neg[type_col].value_counts().to_dict()
    counts["_total"] = len(neg)
    return counts

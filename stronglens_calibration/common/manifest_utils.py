"""
Manifest utilities for training_v1.parquet and schema-aligned scripts.

Column names match MANIFEST_SCHEMA_TRAINING_V1.md (verified on Lambda).
"""
from __future__ import annotations

import os
import json
from typing import Tuple

import numpy as np
import pandas as pd

# Manifest schema defaults (VERIFIED for training_v1.parquet)
CUTOUT_PATH_COL = "cutout_path"
LABEL_COL = "label"
SPLIT_COL = "split"
SAMPLE_WEIGHT_COL = "sample_weight"
TIER_COL = "tier"
GALAXY_ID_COL = "galaxy_id"

POOL_COL = "pool"
CONFUSER_COL = "confuser_category"


def load_manifest(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    for col in [CUTOUT_PATH_COL, LABEL_COL, SPLIT_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in manifest.")
    return df


def ensure_split_values(
    df: pd.DataFrame, allowed: Tuple[str, ...] = ("train", "val", "test")
) -> None:
    bad = sorted(set(df[SPLIT_COL].dropna().unique()) - set(allowed))
    if bad:
        raise ValueError(f"Unexpected split values: {bad}. Allowed={allowed}")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_json_dump(obj, path: str) -> None:
    """Write JSON to path, creating parent directory if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

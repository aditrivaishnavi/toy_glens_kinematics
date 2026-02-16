"""
Data loader for unpaired experiment.

Loads data from:
1. Unpaired manifest (mode="unpaired") - LRG-disjoint pos/neg
2. Paired data (mode="paired") - same LRG pos/neg (for baseline comparison)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

from .constants import BANDS, SEED_DEFAULT
from .utils import decode_npz_blob
from .preprocess import preprocess_stack


@dataclass
class AugmentConfig:
    """Augmentation configuration."""
    hflip: bool = True
    vflip: bool = True
    rot90: bool = True
    core_dropout_prob: float = 0.0
    core_radius: float = 5.0
    az_shuffle_prob: float = 0.0


def stack_from_npz(decoded: dict[str, np.ndarray]) -> np.ndarray:
    """Stack g,r,z bands into (3, H, W) array."""
    imgs = []
    for b in BANDS:
        k = f"image_{b}"
        if k not in decoded:
            raise KeyError(f"Missing {k} in npz")
        imgs.append(decoded[k].astype(np.float32))
    return np.stack(imgs, axis=0)


def apply_augmentations(img: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    """Apply augmentations to image."""
    x = img.copy()
    
    if cfg.hflip and rng.random() < 0.5:
        x = x[..., :, ::-1]
    
    if cfg.vflip and rng.random() < 0.5:
        x = x[..., ::-1, :]
    
    if cfg.rot90:
        k = int(rng.integers(0, 4))
        x = np.rot90(x, k=k, axes=(-2, -1))
    
    if cfg.core_dropout_prob > 0 and rng.random() < cfg.core_dropout_prob:
        from .utils import radial_rmap
        H, W = x.shape[-2], x.shape[-1]
        r = radial_rmap(H, W)
        m = r < cfg.core_radius
        x[:, m] = 0.0
    
    return x.astype(np.float32).copy()


class UnpairedDataset:
    """Dataset for unpaired training from manifest.
    
    Supports two manifest formats:
    1. V1 (embedded blobs): Single parquet with 'blob' column
    2. V2 (metadata-only): Partitioned directory with '_source_file' and '_row_idx'
    """
    
    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        preprocessing: str = "raw_robust",
        aug_config: Optional[AugmentConfig] = None,
        seed: int = SEED_DEFAULT,
    ):
        self.preprocessing = preprocessing
        self.aug_config = aug_config or AugmentConfig()
        self.seed = seed
        
        manifest_path = Path(manifest_path)
        
        # Detect manifest format
        if manifest_path.is_dir():
            # V2: Partitioned directory
            split_dir = manifest_path / f"split={split}"
            if not split_dir.exists():
                raise ValueError(f"Split directory not found: {split_dir}")
            files = sorted(split_dir.glob("*.parquet"))
            dfs = [pd.read_parquet(f) for f in files]
            self.df = pd.concat(dfs, ignore_index=True)
            self._manifest_v2 = True
            self._source_cache = {}  # Cache loaded source files
        else:
            # V1: Single parquet with embedded blobs
            df = pd.read_parquet(manifest_path)
            self.df = df[df["split"] == split].reset_index(drop=True)
            self._manifest_v2 = False
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        row = self.df.iloc[idx]
        
        # Get blob based on manifest version
        if self._manifest_v2:
            blob = self._load_blob_v2(row)
        else:
            blob = row["blob"]
        
        # Decode and stack
        dec = decode_npz_blob(blob)
        img = stack_from_npz(dec)
        
        # Preprocess
        img = preprocess_stack(img, mode=self.preprocessing)
        
        # Augment
        rng = np.random.default_rng((self.seed * 1000003 + idx) & 0x7fffffff)
        img = apply_augmentations(img, self.aug_config, rng)
        
        return img, int(row["label"])
    
    def _load_blob_v2(self, row) -> bytes:
        """Load blob from source file (V2 manifest)."""
        source_file = row["_source_file"]
        row_idx = int(row["_row_idx"])
        blob_type = row["blob_type"]  # "stamp_npz" or "ctrl_stamp_npz"
        
        # Cache source DataFrames for efficiency
        if source_file not in self._source_cache:
            # Only load blob columns to save memory
            self._source_cache[source_file] = pd.read_parquet(
                source_file, 
                columns=["stamp_npz", "ctrl_stamp_npz"]
            )
            # Limit cache size
            if len(self._source_cache) > 50:
                # Remove oldest entry
                oldest = next(iter(self._source_cache))
                del self._source_cache[oldest]
        
        source_df = self._source_cache[source_file]
        return source_df.iloc[row_idx][blob_type]
    
    def get_theta_e(self, idx: int) -> float:
        """Get theta_E for sample at idx. Returns NaN for negatives."""
        row = self.df.iloc[idx]
        if row["label"] == 0:
            return float("nan")
        if "theta_e_arcsec" in self.df.columns:
            return float(row["theta_e_arcsec"])
        return float("nan")
    
    def get_all_theta_e(self) -> np.ndarray:
        """Get theta_E array for all samples. NaN for negatives."""
        if "theta_e_arcsec" in self.df.columns:
            te = self.df["theta_e_arcsec"].values.copy()
            te[self.df["label"] == 0] = np.nan
            return te.astype(np.float32)
        return np.full(len(self.df), np.nan, dtype=np.float32)


class PairedDataset:
    """Dataset for paired training (baseline comparison)."""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        preprocessing: str = "raw_robust",
        aug_config: Optional[AugmentConfig] = None,
        seed: int = SEED_DEFAULT,
        max_files: Optional[int] = None,
        file_list: Optional[str] = None,
    ):
        self.preprocessing = preprocessing
        self.aug_config = aug_config or AugmentConfig()
        self.seed = seed
        
        # Load parquet files
        if file_list:
            with open(file_list) as f:
                files = [Path(line.strip()) for line in f if line.strip()]
        else:
            split_dir = Path(data_root) / split
            files = sorted(split_dir.glob("*.parquet"))
            if max_files:
                files = files[:max_files]
        
        dfs = [pd.read_parquet(f) for f in files]
        self.df = pd.concat(dfs, ignore_index=True)
        
        # Each row will be used twice: once for stamp (pos), once for ctrl (neg)
        self._length = len(self.df) * 2
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        row_idx = idx // 2
        is_positive = (idx % 2) == 0
        
        row = self.df.iloc[row_idx]
        
        # Select blob column
        blob_col = "stamp_npz" if is_positive else "ctrl_stamp_npz"
        
        # Decode and stack
        dec = decode_npz_blob(row[blob_col])
        img = stack_from_npz(dec)
        
        # Preprocess
        img = preprocess_stack(img, mode=self.preprocessing)
        
        # Augment
        rng = np.random.default_rng((self.seed * 1000003 + idx) & 0x7fffffff)
        img = apply_augmentations(img, self.aug_config, rng)
        
        label = 1 if is_positive else 0
        return img, label
    
    def get_theta_e(self, idx: int) -> float:
        """Get theta_E for sample at idx. Returns NaN for negatives (odd indices)."""
        row_idx = idx // 2
        is_positive = (idx % 2) == 0
        if not is_positive:
            return float("nan")
        row = self.df.iloc[row_idx]
        if "theta_e_arcsec" in self.df.columns:
            return float(row["theta_e_arcsec"])
        return float("nan")
    
    def get_all_theta_e(self) -> np.ndarray:
        """Get theta_E array for all samples. NaN for negatives (odd indices)."""
        te = np.full(self._length, np.nan, dtype=np.float32)
        if "theta_e_arcsec" in self.df.columns:
            for i in range(len(self.df)):
                te[i * 2] = self.df.iloc[i]["theta_e_arcsec"]
        return te

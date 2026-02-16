from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .io_npz import load_npz_cutout
from .preprocess import preprocess, PreprocessConfig

def _rot90(img: np.ndarray, k: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(img, k, axes=(0,1)))

@dataclass
class AugmentConfig:
    hflip: bool = True
    vflip: bool = True
    rot90: bool = True

class ManifestNPZDataset(Dataset):
    """Dataset backed by training_v1.parquet manifest."""
    def __init__(
        self,
        manifest_path: str,
        split: str,
        preprocess_cfg: PreprocessConfig,
        augment_cfg: Optional[AugmentConfig] = None,
    ):
        self.df = pd.read_parquet(manifest_path)
        if "split" not in self.df.columns:
            raise KeyError("manifest missing 'split'")
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        required = ["cutout_path", "label", "sample_weight"]
        for col in required:
            if col not in self.df.columns:
                raise KeyError(f"manifest missing required column '{col}'")
        self.preprocess_cfg = preprocess_cfg
        self.augment_cfg = augment_cfg

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        cut = load_npz_cutout(row["cutout_path"])
        img = preprocess(cut.image, self.preprocess_cfg)
        if self.augment_cfg is not None:
            img = self._augment(img)
        # (H,W,C) -> (C,H,W)
        x = torch.from_numpy(np.transpose(img, (2,0,1))).float()
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        w = torch.tensor(float(row["sample_weight"]), dtype=torch.float32)
        tier = row["tier"] if "tier" in row and pd.notna(row["tier"]) else ""
        return {"x": x, "y": y, "w": w, "tier": tier, "path": row["cutout_path"]}

    def _augment(self, img: np.ndarray) -> np.ndarray:
        cfg = self.augment_cfg
        if cfg.hflip and np.random.rand() < 0.5:
            img = img[:, ::-1, :]
        if cfg.vflip and np.random.rand() < 0.5:
            img = img[::-1, :, :]
        if cfg.rot90:
            k = np.random.randint(0, 4)
            if k:
                img = _rot90(img, k)
        return np.ascontiguousarray(img)

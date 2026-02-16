from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from data.preprocessing import load_fits_grz, Preprocessor, PreprocessConfig

@dataclass
class DatasetConfig:
    data_root: str
    split: str = "train"
    use_asinh: bool = False
    asinh_a: float = 3.0
    augment: bool = True
    seed: int = 0
    metadata_csv: str = ""
    metadata_cols: str = ""

class DR10CutoutDataset(Dataset):
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        root = Path(cfg.data_root)
        manifest = pd.read_json(root / "manifest.json", typ="series").to_dict()
        self.pos_dir = Path(manifest["pos_dir"])
        self.neg_dir = Path(manifest["neg_dir"])
        self.df = pd.read_csv(Path(manifest["splits"][cfg.split]))
        self.pre = Preprocessor(PreprocessConfig(use_asinh=cfg.use_asinh, asinh_a=cfg.asinh_a, do_augment=cfg.augment, seed=cfg.seed))
        self.meta = None
        self.meta_cols = []
        if cfg.metadata_csv:
            mdf = pd.read_csv(cfg.metadata_csv)
            if "id" not in mdf.columns:
                raise ValueError("metadata_csv must include id")
            self.meta_cols = [c.strip() for c in cfg.metadata_cols.split(",") if c.strip()]
            self.meta = mdf.set_index("id")[self.meta_cols]

    def __len__(self): return len(self.df)

    def _path(self, sid: str, label: int) -> Path:
        return (self.pos_dir if label == 1 else self.neg_dir) / f"{sid}.fits"

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sid = str(row["id"]); y = int(row["label"])
        img = self.pre(load_fits_grz(str(self._path(sid, y))))
        x = torch.from_numpy(img)
        target = torch.tensor(float(y), dtype=torch.float32)
        if self.meta is not None:
            if sid in self.meta.index:
                meta = torch.tensor(self.meta.loc[sid].to_numpy(dtype=np.float32), dtype=torch.float32)
            else:
                meta = torch.zeros(len(self.meta_cols), dtype=torch.float32)
            return x, meta, target
        return x, target

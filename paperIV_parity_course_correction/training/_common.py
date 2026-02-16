from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Any, Tuple
import os, json, time, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from stronglens.dataset import ManifestNPZDataset, AugmentConfig
from stronglens.preprocess import PreprocessConfig
from stronglens.metrics import safe_auc, pr_at_thresh
from stronglens.train_utils import weighted_bce_loss, save_run_info, RunInfo, predict_logits

def seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_loaders(manifest: str, micro_batch: int, num_workers: int, crop_size: int|None):
    # Paper IV parity: crop_size=None keeps 101x101
    pp = PreprocessConfig(mode="raw_robust", clip=10.0, crop_size=crop_size)
    aug = AugmentConfig(hflip=True, vflip=True, rot90=True)
    ds_train = ManifestNPZDataset(manifest, "train", pp, aug)
    ds_val = ManifestNPZDataset(manifest, "val", pp, None)
    train_loader = DataLoader(ds_train, batch_size=micro_batch, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=micro_batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    return ds_train, ds_val, train_loader, val_loader

def effective_accum_steps(effective_batch: int, micro_batch: int) -> int:
    if effective_batch % micro_batch != 0:
        raise ValueError(f"effective_batch {effective_batch} must be divisible by micro_batch {micro_batch}")
    return effective_batch // micro_batch

def save_preds_parquet(out_path: str, y_true, logits, tiers):
    import numpy as np
    import pandas as pd
    scores = 1/(1+np.exp(-np.asarray(logits)))
    df = pd.DataFrame({"y": y_true, "logit": logits, "score": scores, "tier": tiers})
    df.to_parquet(out_path, index=False)

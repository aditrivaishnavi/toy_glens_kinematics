#!/usr/bin/env python3
"""
Phase 5: Production-grade lens-finder training.

Features:
- Mixed precision (AMP) for 2x speedup on V100
- Early stopping with patience
- Cosine annealing LR with warmup
- AUROC as primary metric (not accuracy)
- Per-class metrics (control vs injection)
- Data augmentation (astronomy-safe: flips, 90° rotations)
- Gradient clipping
- Comprehensive TensorBoard logging
- Checkpoint management with best model tracking
- Training resumption from checkpoint

Usage (single GPU):
  python phase5_train_production.py \
    --data s3://darkhaloscope/phase4_pipeline/phase4c/.../stamps/train_stamp64... \
    --contract_json dark_halo_scope/model/phase5_required_columns_contract.json \
    --out_dir /data/phase5/models/resnet18_v1 \
    --epochs 20 --early_stopping_patience 3

Usage (multi-GPU with torchrun):
  torchrun --standalone --nproc_per_node=4 phase5_train_production.py ...
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

try:
    import fsspec
except ImportError:
    raise RuntimeError("pip install fsspec s3fs")

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    raise RuntimeError("pip install pyarrow")

try:
    from torchvision import models, transforms
except ImportError:
    raise RuntimeError("pip install torchvision")

try:
    from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] sklearn not available, AUROC will be computed manually")

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# Cache import
try:
    from .data_cache import DataCache
    HAS_CACHE = True
except ImportError:
    try:
        from dark_halo_scope.model.data_cache import DataCache
        HAS_CACHE = True
    except ImportError:
        try:
            from data_cache import DataCache
            HAS_CACHE = True
        except ImportError:
            HAS_CACHE = False


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class TrainingConfig:
    """All hyperparameters in one place for reproducibility."""
    # Data
    data_path: str = ""
    stamp_size: int = 64
    batch_size: int = 256
    num_workers: int = 4
    
    # Model
    arch: str = "resnet18"
    
    # Training
    epochs: int = 20
    steps_per_epoch: int = 5000
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    min_lr: float = 1e-6
    
    # Regularization
    label_smoothing: float = 0.0
    dropout: float = 0.0
    gradient_clip: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    
    # Augmentation
    augment: bool = True
    
    # Mixed precision
    use_amp: bool = True
    
    # Logging
    log_every: int = 100
    eval_every_epoch: int = 1
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
    
    @classmethod
    def from_args(cls, args) -> "TrainingConfig":
        cfg = cls()
        for k in cfg.__dataclass_fields__:
            if hasattr(args, k):
                setattr(cfg, k, getattr(args, k))
        return cfg


# ===========================================================================
# Data Augmentation (Astronomy-safe)
# ===========================================================================

class AstronomyAugment:
    """
    Augmentations that preserve astronomy semantics:
    - Random 90° rotations (sky orientation is arbitrary)
    - Random horizontal/vertical flips (no preferred direction)
    - NO color jitter (magnitudes are physical)
    - NO random crops (stamp centering is meaningful)
    """
    
    def __init__(self, p_flip: float = 0.5, p_rot90: float = 0.5):
        self.p_flip = p_flip
        self.p_rot90 = p_rot90
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (C, H, W) tensor"""
        # Random 90° rotations
        if random.random() < self.p_rot90:
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            x = torch.rot90(x, k, dims=(-2, -1))
        
        # Random horizontal flip
        if random.random() < self.p_flip:
            x = torch.flip(x, dims=[-1])
        
        # Random vertical flip
        if random.random() < self.p_flip:
            x = torch.flip(x, dims=[-2])
        
        return x


# ===========================================================================
# Dataset
# ===========================================================================

def _decode_stamp_npz(npz_bytes: bytes) -> Dict[str, np.ndarray]:
    """Decode NPZ from stamp_npz column."""
    if npz_bytes is None:
        raise ValueError("stamp_npz is None")
    bio = io.BytesIO(npz_bytes)
    with np.load(bio) as npz:
        return {f"image_{b}": npz[f"image_{b}"] for b in ["g", "r", "z"]}


def _robust_normalize(img: np.ndarray) -> np.ndarray:
    """Per-stamp robust normalization: (x - median) / (1.4826*MAD)."""
    x = img.astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = float(np.std(x) + 1e-6)
    return np.clip((x - med) / scale, -10.0, 10.0)


def _list_parquet_files(path: str) -> List[str]:
    """List all parquet files under a path."""
    if path.startswith("s3://"):
        fs = fsspec.open(path).fs
        files = fs.glob(path.rstrip("/") + "/**/*.parquet")
        return sorted(["s3://" + f if not f.startswith("s3://") else f for f in files])
    else:
        import glob
        return sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))


def _load_contract_cols(contract_json_path: str) -> List[str]:
    """Load required columns from contract JSON file."""
    if contract_json_path.startswith("sandbox:"):
        contract_json_path = contract_json_path.replace("sandbox:", "")
    with open(contract_json_path, "r") as f:
        doc = json.load(f)
    cols = doc.get("phase5_required_columns", doc.get("required_columns", []))
    return list(cols)


class Phase4cDataset(IterableDataset):
    """
    Streams Phase 4c unified parquet with proper sharding for DDP + DataLoader workers.
    """
    
    def __init__(
        self,
        parquet_files: List[str],
        split: str,
        stamp_size: int = 64,
        augment: bool = False,
        seed: int = 0,
        max_rows_per_file: Optional[int] = None,
    ):
        super().__init__()
        self.parquet_files = parquet_files
        self.split = split
        self.stamp_size = stamp_size
        self.augment = AstronomyAugment() if augment else None
        self.seed = seed
        self.max_rows = max_rows_per_file

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Dict]]:
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        
        # Shuffle files deterministically
        rng = random.Random(self.seed + rank)
        files = list(self.parquet_files)
        rng.shuffle(files)
        
        # Assign files to this (rank, worker)
        assigned = []
        for i, f in enumerate(files):
            if (i % world) != rank:
                continue
            if (len(assigned) % num_workers) != worker_id:
                assigned.append(f)  # Count but don't yield
                continue
            assigned.append(f)
        
        # Filter to just this worker's files
        my_files = [f for i, f in enumerate(files) 
                    if (i % world) == rank and ((i // world) % num_workers) == worker_id]
        
        for pf_path in my_files:
            yield from self._iter_file(pf_path)

    def _iter_file(self, path: str):
        """Iterate rows in a single parquet file."""
        try:
            if path.startswith("s3://"):
                fs = fsspec.open(path).fs
                f = fs.open(path, "rb")
                pf = pq.ParquetFile(f)
            else:
                pf = pq.ParquetFile(path)
        except Exception as e:
            print(f"[WARN] Failed to open {path}: {e}")
            return
        
        cols = ["stamp_npz", "lens_model", "region_split", "cutout_ok",
                "theta_e_arcsec", "psf_fwhm_used_r", "arc_snr"]
        cols = [c for c in cols if c in pf.schema.names]
        
        rows_yielded = 0
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=cols)
            
            # Filter by split
            if "region_split" in table.column_names:
                splits = table["region_split"].to_pylist()
                keep = [i for i, s in enumerate(splits) if s == self.split]
            else:
                keep = list(range(table.num_rows))
            
            if not keep:
                continue
            
            stamp_col = table["stamp_npz"].to_pylist()
            lens_col = table["lens_model"].to_pylist() if "lens_model" in table.column_names else ["UNK"] * table.num_rows
            cutout_col = table["cutout_ok"].to_pylist() if "cutout_ok" in table.column_names else [1] * table.num_rows
            
            # Optional metadata
            theta_col = table["theta_e_arcsec"].to_pylist() if "theta_e_arcsec" in table.column_names else [None] * table.num_rows
            psf_col = table["psf_fwhm_used_r"].to_pylist() if "psf_fwhm_used_r" in table.column_names else [None] * table.num_rows
            snr_col = table["arc_snr"].to_pylist() if "arc_snr" in table.column_names else [None] * table.num_rows
            
            for i in keep:
                if self.max_rows and rows_yielded >= self.max_rows:
                    return
                
                try:
                    if cutout_col[i] != 1:
                        continue
                    if stamp_col[i] is None:
                        continue
                    
                    imgs = _decode_stamp_npz(stamp_col[i])
                    ig = _robust_normalize(imgs["image_g"])
                    ir = _robust_normalize(imgs["image_r"])
                    iz = _robust_normalize(imgs["image_z"])
                    
                    x = torch.from_numpy(np.stack([ig, ir, iz], axis=0)).float()
                    
                    # Apply augmentation
                    if self.augment:
                        x = self.augment(x)
                    
                    # Label: 0=control, 1=injection
                    y = 0.0 if lens_col[i] == "CONTROL" else 1.0
                    y_t = torch.tensor([y], dtype=torch.float32)
                    
                    meta = {
                        "theta_e": theta_col[i],
                        "psf_fwhm": psf_col[i],
                        "arc_snr": snr_col[i],
                        "lens_model": lens_col[i],
                    }
                    
                    yield x, y_t, meta
                    rows_yielded += 1
                    
                except Exception as e:
                    if random.random() < 1e-5:
                        print(f"[WARN] Row parse error: {e}")
                    continue


# ===========================================================================
# Model
# ===========================================================================

def build_model(arch: str, dropout: float = 0.0) -> nn.Module:
    """Build model adapted for 64x64 input."""
    arch = arch.lower()
    
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        if dropout > 0:
            m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 1))
        else:
            m.fc = nn.Linear(512, 1)
        return m
    
    if arch == "resnet34":
        m = models.resnet34(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        if dropout > 0:
            m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 1))
        else:
            m.fc = nn.Linear(512, 1)
        return m
    
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
        return m
    
    if arch == "small_cnn":
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(256, 1),
        )
    
    raise ValueError(f"Unknown arch: {arch}")


# ===========================================================================
# Learning Rate Scheduler
# ===========================================================================

class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        lr = self.get_lr()
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = lr
        return lr
    
    def get_lr(self) -> float:
        if self.step_count < self.warmup_steps:
            # Linear warmup
            return self.base_lrs[0] * (self.step_count / max(1, self.warmup_steps))
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (1 + math.cos(math.pi * progress))


# ===========================================================================
# Metrics
# ===========================================================================

@dataclass
class Metrics:
    """Training/validation metrics."""
    loss: float = 0.0
    n_samples: int = 0
    n_correct: int = 0
    n_pos: int = 0  # Actual positives (injections)
    n_neg: int = 0  # Actual negatives (controls)
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    
    # For AUROC
    all_probs: List[float] = None
    all_labels: List[int] = None
    
    def __post_init__(self):
        if self.all_probs is None:
            self.all_probs = []
        if self.all_labels is None:
            self.all_labels = []
    
    def update(self, loss: float, probs: torch.Tensor, labels: torch.Tensor):
        """Update metrics with a batch."""
        batch_size = probs.size(0)
        self.loss += loss * batch_size
        self.n_samples += batch_size
        
        preds = (probs >= 0.5).float()
        labels_flat = labels.view(-1)
        preds_flat = preds.view(-1)
        
        self.n_correct += int((preds_flat == labels_flat).sum().item())
        self.n_pos += int(labels_flat.sum().item())
        self.n_neg += int((1 - labels_flat).sum().item())
        
        self.tp += int(((preds_flat == 1) & (labels_flat == 1)).sum().item())
        self.fp += int(((preds_flat == 1) & (labels_flat == 0)).sum().item())
        self.tn += int(((preds_flat == 0) & (labels_flat == 0)).sum().item())
        self.fn += int(((preds_flat == 0) & (labels_flat == 1)).sum().item())
        
        # Store for AUROC
        self.all_probs.extend(probs.view(-1).cpu().tolist())
        self.all_labels.extend(labels.view(-1).int().cpu().tolist())
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        if self.n_samples == 0:
            return {"loss": float("nan"), "acc": float("nan"), "auroc": float("nan")}
        
        result = {
            "loss": self.loss / self.n_samples,
            "acc": self.n_correct / self.n_samples,
            "n_samples": self.n_samples,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
        }
        
        # Precision, Recall, F1
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        
        result["precision"] = precision
        result["recall"] = recall
        result["f1"] = f1
        
        # Specificity (true negative rate)
        result["specificity"] = self.tn / max(1, self.tn + self.fp)
        
        # AUROC
        if len(self.all_probs) > 10 and len(set(self.all_labels)) > 1:
            if HAS_SKLEARN:
                result["auroc"] = roc_auc_score(self.all_labels, self.all_probs)
                result["avg_precision"] = average_precision_score(self.all_labels, self.all_probs)
            else:
                # Manual AUROC (approximate)
                result["auroc"] = self._manual_auroc()
        else:
            result["auroc"] = float("nan")
            result["avg_precision"] = float("nan")
        
        return result
    
    def _manual_auroc(self) -> float:
        """Simple AUROC without sklearn."""
        pairs = list(zip(self.all_probs, self.all_labels))
        pairs.sort(key=lambda x: -x[0])
        
        n_pos = sum(1 for _, l in pairs if l == 1)
        n_neg = sum(1 for _, l in pairs if l == 0)
        
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        
        # Count concordant pairs
        concordant = 0
        pos_seen = 0
        for prob, label in pairs:
            if label == 1:
                pos_seen += 1
            else:
                concordant += pos_seen
        
        return concordant / (n_pos * n_neg)


# ===========================================================================
# Early Stopping
# ===========================================================================

class EarlyStopping:
    """Early stopping with patience."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # "max" for AUROC, "min" for loss
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """Returns True if should stop."""
        if not math.isfinite(score):
            return False
        
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False


# ===========================================================================
# Training Loop
# ===========================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    scaler: GradScaler,
    device: torch.device,
    cfg: TrainingConfig,
    epoch: int,
    global_step: int,
    writer: Optional[SummaryWriter],
    rank: int,
) -> Tuple[Dict[str, float], int]:
    """Train for one epoch."""
    model.train()
    metrics = Metrics()
    
    bce = nn.BCEWithLogitsLoss()
    label_smooth_eps = cfg.label_smoothing
    
    it = iter(loader)
    step_times = []
    
    for step in range(cfg.steps_per_epoch):
        t0 = time.time()
        
        try:
            x, y, _meta = next(it)
        except StopIteration:
            it = iter(loader)
            x, y, _meta = next(it)
        
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # Apply label smoothing: y' = y*(1-ε) + 0.5*ε
        if label_smooth_eps > 0:
            y_smooth = y * (1 - label_smooth_eps) + 0.5 * label_smooth_eps
        else:
            y_smooth = y
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=cfg.use_amp):
            logit = model(x)
            loss = bce(logit, y_smooth)
        
        scaler.scale(loss).backward()
        
        if cfg.gradient_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        lr = scheduler.step()
        
        # Update metrics
        with torch.no_grad():
            probs = torch.sigmoid(logit)
            metrics.update(loss.item(), probs, y)
        
        global_step += 1
        step_times.append(time.time() - t0)
        
        # Logging
        if rank == 0 and (step + 1) % cfg.log_every == 0:
            avg_time = sum(step_times[-cfg.log_every:]) / len(step_times[-cfg.log_every:])
            samples_per_sec = cfg.batch_size / avg_time
            
            m = metrics.compute()
            print(f"[TRAIN] epoch={epoch} step={step+1}/{cfg.steps_per_epoch} "
                  f"loss={m['loss']:.4f} acc={m['acc']:.4f} lr={lr:.2e} "
                  f"samples/s={samples_per_sec:.0f}")
            
            if writer:
                writer.add_scalar("train/loss", m["loss"], global_step)
                writer.add_scalar("train/acc", m["acc"], global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/samples_per_sec", samples_per_sec, global_step)
    
    return metrics.compute(), global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainingConfig,
    max_batches: int = 500,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    metrics = Metrics()
    bce = nn.BCEWithLogitsLoss()
    
    for batch_idx, (x, y, _meta) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        with autocast(enabled=cfg.use_amp):
            logit = model(x)
            loss = bce(logit, y)
        
        probs = torch.sigmoid(logit)
        metrics.update(loss.item(), probs, y)
    
    return metrics.compute()


# ===========================================================================
# Checkpointing
# ===========================================================================

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    best_auroc: float,
    cfg: TrainingConfig,
):
    """Save training checkpoint."""
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_auroc": best_auroc,
        "model_state_dict": (model.module if hasattr(model, "module") else model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_step_count": scheduler.step_count,
        "scaler_state_dict": scaler.state_dict(),
        "config": cfg.to_dict(),
    }
    
    if path.startswith("s3://"):
        buf = io.BytesIO()
        torch.save(state, buf)
        with fsspec.open(path, "wb") as f:
            f.write(buf.getvalue())
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer = None,
                    scheduler: CosineWarmupScheduler = None, scaler: GradScaler = None) -> Dict:
    """Load checkpoint."""
    if path.startswith("s3://"):
        with fsspec.open(path, "rb") as f:
            state = torch.load(f, map_location="cpu")
    else:
        state = torch.load(path, map_location="cpu")
    
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(state["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler and "scheduler_step_count" in state:
        scheduler.step_count = state["scheduler_step_count"]
    if scaler and "scaler_state_dict" in state:
        scaler.load_state_dict(state["scaler_state_dict"])
    
    return state


# ===========================================================================
# Main
# ===========================================================================

def setup_ddp() -> Tuple[int, int, int]:
    """Setup distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank
    return 0, 1, 0


def main():
    ap = argparse.ArgumentParser(description="Phase 5: Production lens-finder training")
    
    # Required
    ap.add_argument("--data", required=True, help="Data path (local or S3)")
    ap.add_argument("--contract_json", required=True, help="Required columns contract")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    
    # Model
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34", "efficientnet_b0", "small_cnn"])
    ap.add_argument("--dropout", type=float, default=0.0)
    
    # Training
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps_per_epoch", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_epochs", type=float, default=1.0)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--gradient_clip", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    
    # Early stopping
    ap.add_argument("--early_stopping_patience", type=int, default=5)
    ap.add_argument("--early_stopping_min_delta", type=float, default=0.001)
    
    # Data
    ap.add_argument("--stamp_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--augment", type=int, default=1, help="Enable augmentation (0/1)")
    ap.add_argument("--max_rows_per_file", type=int, default=0, help="Dev cap; 0=unlimited")
    
    # Mixed precision
    ap.add_argument("--use_amp", type=int, default=1, help="Use mixed precision (0/1)")
    
    # Logging
    ap.add_argument("--log_every", type=int, default=100)
    
    # Cache
    ap.add_argument("--cache_root", default="/data/cache")
    ap.add_argument("--force_cache_refresh", action="store_true")
    
    # Resume
    ap.add_argument("--resume", default="", help="Path to checkpoint to resume from")
    
    # Reproducibility
    ap.add_argument("--seed", type=int, default=42)
    
    args = ap.parse_args()
    
    # Setup
    rank, world, local_rank = setup_ddp()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    
    cfg = TrainingConfig.from_args(args)
    cfg.augment = bool(args.augment)
    cfg.use_amp = bool(args.use_amp)
    
    # Load contract columns
    required_cols = _load_contract_cols(args.contract_json)
    if rank == 0:
        print(f"[INFO] Loaded {len(required_cols)} required columns from contract")
    
    # Seed
    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    random.seed(cfg.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed + rank)
    
    # Resolve data path
    data_path = args.data
    if data_path.startswith("s3://") and HAS_CACHE:
        if rank == 0:
            print(f"[INFO] Using data cache...")
        cache = DataCache(cache_root=args.cache_root)
        data_path = cache.get(data_path, force_refresh=args.force_cache_refresh)
    
    if rank == 0:
        print(f"[INFO] Data path: {data_path}")
        print(f"[INFO] Config: {json.dumps(cfg.to_dict(), indent=2)}")
    
    # List files
    parquet_files = _list_parquet_files(data_path)
    if not parquet_files:
        raise RuntimeError(f"No parquet files found at {data_path}")
    if rank == 0:
        print(f"[INFO] Found {len(parquet_files)} parquet files")
    
    # Validate contract columns exist (fail-fast)
    if rank == 0:
        first_pf = pq.ParquetFile(parquet_files[0])
        schema_names = set(first_pf.schema.names)
        
        # Required columns for training
        required_for_training = ["stamp_npz", "lens_model", "region_split", "cutout_ok"]
        missing = [c for c in required_for_training if c not in schema_names]
        if missing:
            raise RuntimeError(f"CONTRACT VIOLATION: Missing required columns: {missing}")
        
        # Validate contract columns exist for downstream completeness
        missing_contract = [c for c in required_cols if c not in schema_names]
        if missing_contract:
            print(f"[WARN] Missing {len(missing_contract)} contract columns (may affect Phase 5 completeness):")
            print(f"  {missing_contract[:10]}{'...' if len(missing_contract) > 10 else ''}")
        else:
            print(f"[INFO] Contract validation passed - all {len(required_cols)} columns present")
    
    max_rows = args.max_rows_per_file if args.max_rows_per_file > 0 else None
    
    # Datasets
    train_ds = Phase4cDataset(parquet_files, "train", cfg.stamp_size, augment=cfg.augment, 
                               seed=cfg.seed, max_rows_per_file=max_rows)
    val_ds = Phase4cDataset(parquet_files, "val", cfg.stamp_size, augment=False,
                             seed=cfg.seed + 999, max_rows_per_file=5000)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=max(1, cfg.num_workers // 2), pin_memory=True)
    
    # Model
    model = build_model(cfg.arch, dropout=cfg.dropout).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Model: {cfg.arch}, params: {n_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Scheduler
    warmup_steps = int(cfg.warmup_epochs * cfg.steps_per_epoch)
    total_steps = cfg.epochs * cfg.steps_per_epoch
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps, cfg.min_lr)
    
    # Scaler for AMP
    scaler = GradScaler(enabled=cfg.use_amp)
    
    # Early stopping
    early_stop = EarlyStopping(patience=cfg.early_stopping_patience, 
                               min_delta=cfg.early_stopping_min_delta, mode="max")
    
    # Checkpoint directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # TensorBoard
    writer = None
    if rank == 0 and SummaryWriter and not args.out_dir.startswith("s3://"):
        writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tensorboard"))
    
    # Resume
    start_epoch = 0
    global_step = 0
    best_auroc = 0.0
    
    if args.resume:
        if rank == 0:
            print(f"[INFO] Resuming from {args.resume}")
        state = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best_auroc = state.get("best_auroc", 0.0)
    
    # Training loop
    if rank == 0:
        print(f"[INFO] Starting training from epoch {start_epoch}")
        print(f"[INFO] Device: {device}, World: {world}, AMP: {cfg.use_amp}")
    
    for epoch in range(start_epoch, cfg.epochs):
        if world > 1:
            dist.barrier()
        
        # Train
        train_metrics, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, cfg, epoch, global_step, writer, rank
        )
        
        # Evaluate (only rank 0, but sync stop signal to all ranks)
        should_stop = torch.tensor([0], dtype=torch.int32, device=device)
        
        if rank == 0:
            val_metrics = evaluate(model, val_loader, device, cfg)
            
            print(f"\n[EPOCH {epoch}] Train: loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f}")
            print(f"[EPOCH {epoch}] Val: loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} "
                  f"auroc={val_metrics['auroc']:.4f} precision={val_metrics['precision']:.4f} "
                  f"recall={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f}")
            
            if writer:
                for k, v in val_metrics.items():
                    if isinstance(v, float) and math.isfinite(v):
                        writer.add_scalar(f"val/{k}", v, global_step)
            
            # Save checkpoint
            ckpt_path = os.path.join(args.out_dir, "checkpoint_last.pt")
            save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, 
                           epoch, global_step, best_auroc, cfg)
            
            # Best model
            if math.isfinite(val_metrics["auroc"]) and val_metrics["auroc"] > best_auroc:
                best_auroc = val_metrics["auroc"]
                best_path = os.path.join(args.out_dir, "checkpoint_best.pt")
                save_checkpoint(best_path, model, optimizer, scheduler, scaler,
                               epoch, global_step, best_auroc, cfg)
                print(f"[INFO] New best AUROC: {best_auroc:.4f}")
            
            # Early stopping check
            if early_stop(val_metrics["auroc"]):
                print(f"[INFO] Early stopping at epoch {epoch} (patience={cfg.early_stopping_patience})")
                should_stop[0] = 1
            
            print()
        
        # Broadcast stop signal to all ranks (prevents DDP deadlock)
        if world > 1:
            dist.broadcast(should_stop, src=0)
            dist.barrier()
        
        # All ranks check stop signal and break together
        if should_stop[0].item() == 1:
            break
    
    # Cleanup
    if world > 1:
        dist.destroy_process_group()
    
    if rank == 0:
        print(f"[DONE] Training complete. Best AUROC: {best_auroc:.4f}")
        print(f"[DONE] Best model: {os.path.join(args.out_dir, 'checkpoint_best.pt')}")


if __name__ == "__main__":
    main()


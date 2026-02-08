
#!/usr/bin/env python3
"""
Phase 5 full-scale training for GH200/H100-class GPUs.

Key features:
- PyArrow dataset streaming over Parquet (local or S3 via pyarrow.fs)
- Decodes `stamp_npz` bytes (NPZ with image_g/image_r/image_z keys)
- Uses bf16 AMP by default (H100-friendly)
- Supports ResNet18/ConvNeXt-Tiny/EfficientNet-B0
- Optional scalar metadata fusion (MLP) into classifier head
- Optional schema contract enforcement (Phase 5 required columns)
- Optional filtering of positives by resolvability (theta_e/PSF) and/or arc SNR

Scientific defaults:
- Augmentations limited to flips and 90-degree rotations.
- Reports ROC-derived FPR at fixed completeness targets and completeness at fixed FPR targets.

Run examples:
torchrun --standalone --nproc_per_node=1 phase5_train_fullscale_gh200_v2.py \
  --data s3://.../phase4c_unified/ --out_dir s3://.../phase5_runs/run1 \
  --arch convnext_tiny --epochs 8 --batch_size 512 --use_bf16 --augment \
  --min_theta_over_psf 0.5
"""

import argparse
import io
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader

import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.fs as pafs

try:
    import torchvision
    from torchvision.models import resnet18, convnext_tiny, efficientnet_b0
except ImportError as e:
    raise RuntimeError("torchvision is required (pip install torchvision)") from e


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def rank() -> int:
    return dist.get_rank() if is_dist() else 0

def world() -> int:
    return dist.get_world_size() if is_dist() else 1

def seed_all(seed: int):
    random.seed(seed + rank())
    np.random.seed(seed + rank())
    torch.manual_seed(seed + rank())
    torch.cuda.manual_seed_all(seed + rank())

def barrier():
    if is_dist():
        dist.barrier()

def ddp_init(backend: str = "nccl"):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

def ddp_cleanup():
    if is_dist():
        dist.destroy_process_group()


def decode_stamp_npz(npz_bytes: bytes) -> np.ndarray:
    """Decode stamp NPZ bytes to (3, H, W) array.
    
    Expects image_g, image_r, image_z keys for grz bandset.
    Raises ValueError if any band is missing (caller should filter by bandset).
    """
    if npz_bytes is None:
        raise ValueError("stamp_npz is None")
    bio = io.BytesIO(npz_bytes)
    with np.load(bio) as npz:
        keys = list(npz.keys())
        # Validate all 3 bands exist
        required = {"image_g", "image_r", "image_z"}
        missing = required - set(keys)
        if missing:
            raise ValueError(f"stamp_npz missing bands: {missing} (has: {keys})")
        g = npz["image_g"].astype(np.float32)
        r = npz["image_r"].astype(np.float32)
        z = npz["image_z"].astype(np.float32)
    return np.stack([g, r, z], axis=0)


def robust_mad_norm(x: np.ndarray, clip: float = 10.0, eps: float = 1e-6) -> np.ndarray:
    """Normalize using median/MAD of full image (legacy method)."""
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[0]):
        v = x[c]
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        scale = 1.4826 * mad + eps
        vv = (v - med) / scale
        if clip is not None:
            vv = np.clip(vv, -clip, clip)
        out[c] = vv.astype(np.float32)
    return out


def robust_mad_norm_outer(x: np.ndarray, clip: float = 10.0, eps: float = 1e-6,
                          inner_frac: float = 0.5) -> np.ndarray:
    """Normalize using outer annulus only to avoid leaking injection strength.
    
    FIX D1: The full-image normalization can encode injection strength as a global cue
    since injected flux changes the tail of the pixel distribution. By computing
    median/MAD from the outer region only (where arcs are less likely), we reduce
    this information leakage.
    
    Args:
        x: Image array of shape (C, H, W)
        clip: Clip normalized values to [-clip, clip]
        eps: Small constant for numerical stability
        inner_frac: Fraction of image to exclude from center (0.5 = exclude inner half)
    """
    out = np.empty_like(x, dtype=np.float32)
    h, w = x.shape[-2:]
    cy, cx = h // 2, w // 2
    ri = int(min(h, w) * inner_frac / 2)
    
    # Create circular mask for outer region
    yy, xx = np.ogrid[:h, :w]
    outer_mask = ((yy - cy)**2 + (xx - cx)**2) > ri**2
    
    for c in range(x.shape[0]):
        v = x[c]
        outer_v = v[outer_mask]
        med = np.median(outer_v)
        mad = np.median(np.abs(outer_v - med))
        scale = 1.4826 * mad + eps
        vv = (v - med) / scale
        if clip is not None:
            vv = np.clip(vv, -clip, clip)
        out[c] = vv.astype(np.float32)
    return out


def aug_rot_flip(x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    k = int(rng.randint(0, 4))
    if k:
        x = np.rot90(x, k=k, axes=(1, 2)).copy()
    if rng.rand() < 0.5:
        x = x[:, :, ::-1].copy()
    if rng.rand() < 0.5:
        x = x[:, ::-1, :].copy()
    return x


def compute_theta_over_psf(theta_e: Optional[float], psf_used: Optional[float], psf_manifest: Optional[float]) -> Optional[float]:
    psf = None
    if psf_used is not None and np.isfinite(psf_used) and psf_used > 0:
        psf = psf_used
    elif psf_manifest is not None and np.isfinite(psf_manifest) and psf_manifest > 0:
        psf = psf_manifest
    if psf is None or theta_e is None or not np.isfinite(theta_e):
        return None
    return float(theta_e / psf)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, prob, 1 - prob)
        w = torch.where(targets > 0.5, self.alpha, 1 - self.alpha)
        loss = -w * (1 - pt).pow(self.gamma) * torch.log(pt.clamp_min(1e-8))
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def roc_curve_np(scores: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve with proper tie handling.
    
    FIXES (per LLM1/LLM2 analysis):
    1. Keep LAST element of each tied-score run (not first)
    2. Prepend (0,0) origin point
    3. Handle edge cases for empty/single-class data
    """
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    yy = y[order].astype(np.int64)
    P = int(yy.sum())
    N = len(yy) - P
    
    if P == 0 or N == 0:
        # Return minimal valid ROC curve
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])
    
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    
    # FIX: Keep LAST element of each tied-score run (where score changes OR is last element)
    # np.diff(s) != 0 finds where score changes; we add True at end to always include last
    distinct = np.r_[np.diff(s) != 0, True]
    
    tp = tp[distinct]
    fp = fp[distinct]
    thr = s[distinct]
    
    tpr = tp / P
    fpr = fp / N
    
    # FIX: Prepend (0, 0) origin point for proper ROC curve
    fpr = np.r_[0.0, fpr]
    tpr = np.r_[0.0, tpr]
    thr = np.r_[np.inf, thr]  # threshold above max score for origin
    
    return fpr, tpr, thr


def fpr_at_tpr(scores: np.ndarray, y: np.ndarray, tpr_targets: Sequence[float]) -> Dict[str, float]:
    fpr, tpr, _ = roc_curve_np(scores, y)
    out = {}
    if len(fpr) == 0:
        for tt in tpr_targets:
            out[f"fpr@tpr{tt:.2f}"] = math.nan
        return out
    for tt in tpr_targets:
        idx = np.searchsorted(tpr, tt, side="left")
        out[f"fpr@tpr{tt:.2f}"] = float(fpr[idx]) if idx < len(fpr) else math.nan
    return out


def tpr_at_fpr(scores: np.ndarray, y: np.ndarray, fpr_targets: Sequence[float]) -> Dict[str, float]:
    fpr, tpr, _ = roc_curve_np(scores, y)
    out = {}
    if len(fpr) == 0:
        for ft in fpr_targets:
            out[f"tpr@fpr{ft:g}"] = math.nan
        return out
    for ft in fpr_targets:
        idx = np.searchsorted(fpr, ft, side="right") - 1
        out[f"tpr@fpr{ft:g}"] = float(tpr[idx]) if idx >= 0 else 0.0
    return out


@dataclass
class StreamConfig:
    data: str
    split: str
    seed: int
    columns: List[str]
    meta_cols: List[str]
    augment: bool
    mad_clip: float
    max_rows: int
    min_theta_over_psf: float
    min_arc_snr: float
    norm_method: str = "full"  # FIX D1: "full" or "outer"
    epoch: int = 0  # FIX A6: Added for epoch-dependent shuffle


class ParquetStreamDataset(IterableDataset):
    def __init__(self, cfg: StreamConfig, filesystem: Optional[pafs.FileSystem] = None):
        super().__init__()
        self.cfg = cfg
        self.fs = filesystem
        self._epoch = 0  # Mutable epoch for epoch-dependent shuffle
    
    def set_epoch(self, epoch: int):
        """Set epoch for epoch-dependent shuffling (call before each epoch)."""
        self._epoch = epoch

    def _iter_fragments(self) -> List[ds.Fragment]:
        """Get fragments sharded by BOTH DDP rank AND DataLoader worker id.
        
        FIX (per LLM1 Q2): Without worker sharding, all N workers process same fragments,
        causing N*duplication per epoch and accelerated overfitting.
        """
        # Use partitioning="hive" to auto-detect region_split from directory structure
        dataset = ds.dataset(self.cfg.data, format="parquet", filesystem=self.fs, partitioning="hive")
        frags = list(dataset.get_fragments(filter=ds.field("region_split") == self.cfg.split))
        
        # CRITICAL: Shard by BOTH DDP rank AND DataLoader worker id
        wi = torch.utils.data.get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1
        
        shard = rank() * num_workers + worker_id
        nshard = world() * num_workers
        return frags[shard::nshard]

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float, Optional[np.ndarray]]]:
        cfg = self.cfg
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        # FIX A6: Include epoch in seed so shuffle varies each epoch
        rng = np.random.RandomState(cfg.seed + 997 * rank() + 131 * worker_id + 7919 * self._epoch)
        logger = logging.getLogger(__name__)

        frags = self._iter_fragments()
        if not frags:
            logger.warning(f"No fragments found for split='{cfg.split}' in {cfg.data}")
            return
        rng.shuffle(frags)

        yielded = 0
        skip_count = 0
        total_processed = 0
        
        for frag in frags:
            if cfg.max_rows and yielded >= cfg.max_rows:
                break
            
            # Validate required columns exist in this fragment
            frag_schema_names = set(frag.physical_schema.names)
            required_cols = ["stamp_npz", "is_control"]
            missing = [c for c in required_cols if c not in frag_schema_names]
            if missing:
                raise RuntimeError(f"Fragment missing required columns {missing}. Schema: {frag.physical_schema.names}")
            
            scanner = ds.Scanner.from_fragment(frag, columns=cfg.columns + cfg.meta_cols, batch_size=2048)
            for rb in scanner.to_batches():
                colmap = {name: rb.column(i) for i, name in enumerate(rb.schema.names)}
                n = rb.num_rows
                for i in range(n):
                    if cfg.max_rows and yielded >= cfg.max_rows:
                        break
                    
                    total_processed += 1

                    # quality gate
                    if "cutout_ok" in colmap:
                        ok = colmap["cutout_ok"][i].as_py()
                        if ok is None or int(ok) != 1:
                            continue
                    
                    # bandset gate: only process grz stamps (3-band)
                    if "bandset" in colmap:
                        bandset = colmap["bandset"][i].as_py()
                        if bandset != "grz":
                            skip_count += 1
                            continue

                    is_ctrl = int(colmap["is_control"][i].as_py())
                    if is_ctrl not in (0, 1):
                        continue

                    # filter positives
                    if is_ctrl == 0 and (cfg.min_theta_over_psf > 0 or cfg.min_arc_snr > 0):
                        theta = colmap.get("theta_e_arcsec", None)
                        psf_used = colmap.get("psf_fwhm_used_r", None)
                        psf_man = colmap.get("psfsize_r", None)
                        arc_snr = colmap.get("arc_snr", None)

                        theta_v = theta[i].as_py() if theta is not None else None
                        psf_used_v = psf_used[i].as_py() if psf_used is not None else None
                        psf_man_v = psf_man[i].as_py() if psf_man is not None else None
                        res = compute_theta_over_psf(theta_v, psf_used_v, psf_man_v)
                        if cfg.min_theta_over_psf > 0 and (res is None or res < cfg.min_theta_over_psf):
                            continue

                        if cfg.min_arc_snr > 0:
                            snr_v = arc_snr[i].as_py() if arc_snr is not None else None
                            if snr_v is None or float(snr_v) < cfg.min_arc_snr:
                                continue

                    # Decode stamp - narrow exception scope to expected errors only
                    try:
                        x = decode_stamp_npz(colmap["stamp_npz"][i].as_py())
                        # FIX D1: Use outer annulus normalization to reduce injection leakage
                        if cfg.norm_method == "outer":
                            x = robust_mad_norm_outer(x, clip=cfg.mad_clip)
                        else:
                            x = robust_mad_norm(x, clip=cfg.mad_clip)
                    except (ValueError, KeyError, IOError) as e:
                        skip_count += 1
                        if skip_count <= 10:
                            logger.warning(f"Skipping row due to decode error: {e}")
                        continue

                    if cfg.augment:
                        x = aug_rot_flip(x, rng)

                    y = 0.0 if is_ctrl == 1 else 1.0

                    meta = None
                    if cfg.meta_cols:
                        vals = []
                        for c in cfg.meta_cols:
                            v = colmap[c][i].as_py()
                            vals.append(float(v) if v is not None else 0.0)
                        meta = np.asarray(vals, dtype=np.float32)

                    yielded += 1
                    yield x, y, meta
        
        # Log skip statistics at end of iteration
        if total_processed > 0 and skip_count > 0:
            skip_ratio = skip_count / total_processed
            logger.info(f"Data iteration complete: yielded={yielded}, skipped={skip_count}/{total_processed} ({skip_ratio:.1%})")
            if skip_ratio > 0.10:
                logger.error(f"High skip rate ({skip_ratio:.1%}) - check data integrity!")
                # Don't raise here as it would interrupt training mid-epoch
                # The warning is sufficient for investigation


def collate(batch):
    xs, ys, metas = zip(*batch)
    x = torch.from_numpy(np.stack(xs, axis=0))
    y = torch.tensor(ys, dtype=torch.float32)
    if metas[0] is None:
        return x, y, None
    m = torch.from_numpy(np.stack(metas, axis=0))
    return x, y, m


class MetaFusionHead(nn.Module):
    def __init__(self, feat_dim: int, meta_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        m = self.meta_mlp(meta)
        x = torch.cat([feats, m], dim=1)
        return self.classifier(x).squeeze(1)


def build_model(arch: str, meta_dim: int = 0, dropout: float = 0.1) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        backbone = m
    elif arch == "convnext_tiny":
        m = convnext_tiny(weights=None)
        feat_dim = m.classifier[2].in_features
        m.classifier = nn.Identity()
        backbone = m
    elif arch == "efficientnet_b0":
        m = efficientnet_b0(weights=None)
        feat_dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
        backbone = m
    else:
        raise ValueError(f"Unknown arch: {arch}")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.meta_dim = meta_dim
            if meta_dim > 0:
                self.head = MetaFusionHead(feat_dim, meta_dim, hidden=256, dropout=dropout)
            else:
                self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, 1))

        def forward(self, x: torch.Tensor, meta: Optional[torch.Tensor] = None) -> torch.Tensor:
            feats = self.backbone(x)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if feats.ndim > 2:
                feats = torch.flatten(feats, 1)
            if self.meta_dim > 0:
                if meta is None:
                    raise ValueError("meta required but missing")
                return self.head(feats, meta)
            return self.head(feats).squeeze(1)

    return Model()


def load_contract(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "required_columns" not in obj:
        raise ValueError("Contract JSON missing required_columns")
    return obj


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> Dict[str, float]:
    model.eval()
    scores = []
    labels = []
    batches = 0
    for x, y, meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if meta is not None:
            meta = meta.to(device, non_blocking=True)
        logits = model(x, meta)
        prob = torch.sigmoid(logits)
        scores.append(prob.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        batches += 1
        if max_batches and batches >= max_batches:
            break
    s = np.concatenate(scores).astype(np.float64)
    yy = np.concatenate(labels).astype(np.int64)

    out = {"n_eval": int(len(yy)), "pos_eval": int(yy.sum()), "neg_eval": int(len(yy) - yy.sum())}
    out.update(fpr_at_tpr(s, yy, [0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50]))
    out.update(tpr_at_fpr(s, yy, [1e-2, 1e-3, 1e-4, 1e-5]))
    fpr, tpr, _ = roc_curve_np(s, yy)
    out["auroc"] = float(np.trapz(tpr, fpr)) if len(fpr) > 1 else math.nan
    
    # FIX C1/C4: Calibration monitoring - detect binary score collapse
    binary_low = (s < 0.01).sum()
    binary_high = (s > 0.99).sum()
    binary_frac = (binary_low + binary_high) / max(len(s), 1)
    out["binary_score_frac"] = float(binary_frac)
    if binary_frac > 0.5:
        logging.warning(f"CALIBRATION COLLAPSE: {binary_frac:.1%} of scores are <0.01 or >0.99")
    
    # FIX B5: Log top-50 negative and bottom-50 positive scores for diagnostics
    neg_mask = (yy == 0)
    pos_mask = (yy == 1)
    if neg_mask.sum() > 0:
        neg_scores = s[neg_mask]
        top_neg_idx = np.argsort(-neg_scores)[:50]
        out["top_neg_scores"] = neg_scores[top_neg_idx].tolist()
    if pos_mask.sum() > 0:
        pos_scores = s[pos_mask]
        bottom_pos_idx = np.argsort(pos_scores)[:50]
        out["bottom_pos_scores"] = pos_scores[bottom_pos_idx].tolist()
    
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Phase 5 training with scientifically-validated defaults for lens finding."
    )
    ap.add_argument("--data", required=True, help="Path to Phase 4c stamps (local or s3://)")
    ap.add_argument("--out_dir", required=True, help="Output directory for checkpoints and logs")
    
    # Model architecture
    ap.add_argument("--arch", default="convnext_tiny", choices=["resnet18", "convnext_tiny", "efficientnet_b0"],
                    help="CNN backbone architecture")
    
    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    ap.add_argument("--batch_size", type=int, default=512, help="Batch size per GPU")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW")
    ap.add_argument("--dropout", type=float, default=0.1, help="Dropout rate in classifier head")
    ap.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed")
    
    # Precision - bf16 is DEFAULT for H100/GH200
    ap.add_argument("--use_bf16", action="store_true", default=True,
                    help="Use bfloat16 mixed precision (default: True)")
    ap.add_argument("--no_bf16", action="store_false", dest="use_bf16",
                    help="Disable bfloat16, use fp32")
    ap.add_argument("--use_fp16", action="store_true", default=False,
                    help="Use float16 mixed precision instead of bf16")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    
    # Data limits
    ap.add_argument("--max_train_rows_per_epoch", type=int, default=0, help="Max train rows per epoch (0=all)")
    ap.add_argument("--max_val_rows", type=int, default=2_000_000, help="Max validation rows")
    ap.add_argument("--val_batches", type=int, default=250, help="Max validation batches per epoch")
    
    # Augmentation - DEFAULT ON
    ap.add_argument("--augment", action="store_true", default=True,
                    help="Enable augmentation (rot90 + flips) - default: True")
    ap.add_argument("--no_augment", action="store_false", dest="augment",
                    help="Disable augmentation")
    
    # Normalization - OUTER ANNULUS is scientifically preferred to avoid injection leakage
    ap.add_argument("--mad_clip", type=float, default=10.0, help="MAD clipping threshold")
    ap.add_argument("--norm_method", default="outer", choices=["full", "outer"],
                    help="Normalization method: 'outer' (annulus, preferred) or 'full' (legacy)")
    
    # Resolvability filtering - DEFAULT 0.5 to exclude unresolved lenses
    ap.add_argument("--min_theta_over_psf", type=float, default=0.5,
                    help="Minimum theta_e / PSF_FWHM ratio for positives (0.5 = resolved only)")
    ap.add_argument("--min_arc_snr", type=float, default=0.0,
                    help="Minimum arc SNR for positives (0 = no filter)")
    
    # Metadata fusion - DEFAULT includes PSF and depth for conditioning
    ap.add_argument("--meta_cols", default="psfsize_r,psfdepth_r",
                    help="Comma-separated scalar columns to fuse (default: psfsize_r,psfdepth_r)")
    ap.add_argument("--contract_json", default="", help="Path to schema contract JSON")
    
    # Loss function - FOCAL LOSS is preferred for imbalanced/hard example focus
    ap.add_argument("--loss", default="focal", choices=["bce", "focal"],
                    help="Loss function: 'focal' (preferred for low-FPR) or 'bce'")
    ap.add_argument("--focal_alpha", type=float, default=0.25, help="Focal loss alpha")
    ap.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    
    # Early stopping
    ap.add_argument("--early_stopping_patience", type=int, default=3, 
                    help="Stop if no improvement for N epochs. 0=disable")
    ap.add_argument("--resume", default="", help="Path to checkpoint to resume from")
    args = ap.parse_args()

    ddp_init()
    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    fs = None
    if args.data.startswith("s3://") or args.out_dir.startswith("s3://"):
        fs = pafs.S3FileSystem(region=os.environ.get("AWS_REGION"))

    # Use partitioning="hive" to auto-detect region_split from directory structure
    dataset = ds.dataset(args.data, format="parquet", filesystem=fs, partitioning="hive")

    if args.contract_json:
        contract = load_contract(args.contract_json)
        missing = [c for c in contract["required_columns"] if c not in dataset.schema.names]
        if missing:
            raise ValueError(f"Dataset missing contract columns: {missing}")

    # Note: region_split is a virtual Hive partition column, checked separately
    required_min = ["stamp_npz", "is_control", "cutout_ok"]
    for c in required_min:
        if c not in dataset.schema.names:
            raise ValueError(f"Dataset missing required column: {c}")
    # Verify region_split is available via partitioning
    if "region_split" not in dataset.schema.names:
        raise ValueError("Dataset missing region_split partitioning. Ensure data is Hive-partitioned by region_split.")

    meta_cols = [c.strip() for c in args.meta_cols.split(",") if c.strip()]
    for c in meta_cols:
        if c not in dataset.schema.names:
            raise ValueError(f"Meta column '{c}' not found in dataset")
    
    # FIX E1: Forbidden metadata guard - block columns that leak labels
    FORBIDDEN_META = frozenset({
        # Injection parameters (direct label leakage)
        "theta_e_arcsec", "theta_e", "src_dmag", "src_reff_arcsec", "src_n",
        "src_e", "src_phi_deg", "src_x_arcsec", "src_y_arcsec",
        "lens_e", "lens_phi_deg", "shear_gamma", "shear_phi_deg", "shear",
        # Derived from injection (indirect leakage)
        "arc_snr", "magnification", "tangential_stretch", "radial_stretch",
        # Labels themselves
        "is_control", "label", "cutout_ok",
    })
    for c in meta_cols:
        if c in FORBIDDEN_META:
            raise ValueError(f"REFUSING meta_cols that leak labels: '{c}'. "
                            f"Forbidden columns: {sorted(FORBIDDEN_META)}")

    # Note: region_split is a virtual Hive partition column - don't include in fragment reads
    # Include bandset to filter for grz-only (3-band stamps required for decode_stamp_npz)
    base_cols = ["stamp_npz", "is_control", "cutout_ok", "bandset", "theta_e_arcsec", "psf_fwhm_used_r", "psfsize_r", "arc_snr"]
    cols = [c for c in base_cols if c in dataset.schema.names]

    train_cfg = StreamConfig(
        data=args.data, split="train", seed=args.seed,
        columns=cols, meta_cols=meta_cols,
        augment=args.augment, mad_clip=args.mad_clip,
        max_rows=args.max_train_rows_per_epoch,
        min_theta_over_psf=args.min_theta_over_psf, min_arc_snr=args.min_arc_snr,
        norm_method=args.norm_method,
    )
    val_cfg = StreamConfig(
        data=args.data, split="val", seed=args.seed + 99,
        columns=cols, meta_cols=meta_cols,
        augment=False, mad_clip=args.mad_clip,
        norm_method=args.norm_method,
        max_rows=args.max_val_rows,
        min_theta_over_psf=args.min_theta_over_psf, min_arc_snr=args.min_arc_snr,
    )

    train_dataset = ParquetStreamDataset(train_cfg, filesystem=fs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        ParquetStreamDataset(val_cfg, filesystem=fs),
        batch_size=args.batch_size, num_workers=max(1, args.num_workers // 2),
        pin_memory=True, persistent_workers=(args.num_workers > 0),
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.arch, meta_dim=len(meta_cols), dropout=args.dropout).to(device)
    if is_dist():
        model = DDP(model, device_ids=[int(os.environ.get("LOCAL_RANK", "0"))])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=1e-6)

    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma) if args.loss == "focal" else nn.BCEWithLogitsLoss()

    use_amp = args.use_bf16 or args.use_fp16
    amp_dtype = torch.bfloat16 if args.use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler() if (use_amp and amp_dtype == torch.float16) else None

    start_epoch = 0
    best_metric = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        (model.module if isinstance(model, DDP) else model).load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("best_metric", -1.0))

    history = []
    no_improve_count = 0
    early_stop = False
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training: epochs={args.epochs}, batch_size={args.batch_size}, arch={args.arch}")
    logger.info(f"Device: {device}, AMP: {use_amp} (dtype={amp_dtype if use_amp else 'N/A'})")
    
    for epoch in range(start_epoch, args.epochs):
        if early_stop:
            break
        
        # FIX A6: Update epoch for epoch-dependent shuffle
        train_dataset.set_epoch(epoch)
        
        model.train()
        total_loss = 0.0
        n_seen = 0
        batch_idx = 0
        log_interval = 100  # Log every 100 batches

        logger.info(f"Epoch {epoch + 1}/{args.epochs} - Starting training...")
        
        for x, y, meta in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if meta is not None:
                meta = meta.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(x, meta)
                    loss = criterion(logits, y)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()
            else:
                logits = model(x, meta)
                loss = criterion(logits, y)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

            total_loss += float(loss.detach().cpu().item()) * x.size(0)
            n_seen += x.size(0)
            batch_idx += 1
            
            if batch_idx % log_interval == 0:
                avg_loss = total_loss / max(1, n_seen)
                logger.info(f"  Epoch {epoch+1} - Batch {batch_idx}: samples={n_seen}, loss={avg_loss:.4f}")

        logger.info(f"Epoch {epoch + 1} - Training complete: {n_seen} samples, avg_loss={total_loss/max(1,n_seen):.4f}")
        tl = torch.tensor([total_loss, n_seen], device=device, dtype=torch.float64)
        if is_dist():
            dist.all_reduce(tl, op=dist.ReduceOp.SUM)
        total_loss = float(tl[0].item())
        n_seen = int(tl[1].item())
        sched.step()

        if rank() == 0:
            metrics = evaluate(model.module if isinstance(model, DDP) else model, val_loader, device=device, max_batches=args.val_batches)
            metrics.update({
                "train_loss": total_loss / max(1, n_seen),
                "epoch": epoch,
                "lr": float(opt.param_groups[0]["lr"]),
            })
            history.append(metrics)

            key = "tpr@fpr0.0001"
            score = metrics.get(key, math.nan)
            if not np.isfinite(score):
                score = metrics.get("auroc", -1.0)

            ckpt = {
                "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "epoch": epoch,
                "best_metric": best_metric,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.out_dir, "ckpt_last.pt"))
            
            # Save per-epoch checkpoint for reproducibility
            torch.save(ckpt, os.path.join(args.out_dir, f"ckpt_epoch_{epoch}.pt"))

            if score > best_metric:
                best_metric = float(score)
                ckpt["best_metric"] = best_metric
                torch.save(ckpt, os.path.join(args.out_dir, "ckpt_best.pt"))
                no_improve_count = 0
            else:
                no_improve_count += 1
                if args.early_stopping_patience > 0 and no_improve_count >= args.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}: no improvement for {no_improve_count} epochs")
                    early_stop = True

            with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, sort_keys=True)

            print(json.dumps(metrics, indent=2, sort_keys=True))
        
        # Broadcast early_stop signal to all ranks in DDP
        if is_dist():
            stop_tensor = torch.tensor([1 if early_stop else 0], device=device)
            dist.broadcast(stop_tensor, src=0)
            early_stop = bool(stop_tensor.item())

        barrier()

    if rank() == 0:
        print(f"Done. Best metric={best_metric:.6g}. Outputs in {args.out_dir}")

    ddp_cleanup()


if __name__ == "__main__":
    main()

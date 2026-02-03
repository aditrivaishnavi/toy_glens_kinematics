#!/usr/bin/env python3
"""
Gen4 Training Script with Hard Negative Mining Support

This script extends the base Phase 5 training with:
- Hard negative identification via (ra, dec) position matching
- Weighted sampling of hard negatives (upsampling factor)
- All fixes from Gen3 (worker sharding, epoch shuffle, outer norm, etc.)

Key differences from base training:
- --hard_neg_path: Path to parquet with hard negative positions
- --hard_neg_weight: How many times to yield each hard negative (default 5)
"""

import argparse
import io
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

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
import pyarrow.parquet as pq

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
    if npz_bytes is None:
        raise ValueError("stamp_npz is None")
    bio = io.BytesIO(npz_bytes)
    with np.load(bio) as npz:
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
    """Normalize using outer annulus only to avoid leaking injection strength."""
    out = np.empty_like(x, dtype=np.float32)
    h, w = x.shape[-2:]
    cy, cx = h // 2, w // 2
    ri = int(min(h, w) * inner_frac / 2)
    
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


def load_hard_negative_lookup(path: str) -> Set[Tuple[float, float]]:
    """Load hard negative positions as a set of (ra_key, dec_key) tuples for fast lookup."""
    if not path or not os.path.exists(path):
        return set()
    
    table = pq.read_table(path)
    lookup = set()
    
    # Check for ra_key/dec_key or ra/dec columns
    if "ra_key" in table.column_names and "dec_key" in table.column_names:
        ra_col = table["ra_key"].to_pylist()
        dec_col = table["dec_key"].to_pylist()
    elif "ra" in table.column_names and "dec" in table.column_names:
        ra_col = [round(v, 6) for v in table["ra"].to_pylist()]
        dec_col = [round(v, 6) for v in table["dec"].to_pylist()]
    else:
        logging.warning(f"Hard negative file missing ra/dec columns: {table.column_names}")
        return set()
    
    for ra, dec in zip(ra_col, dec_col):
        if ra is not None and dec is not None:
            lookup.add((ra, dec))
    
    logging.info(f"Loaded {len(lookup)} hard negative positions from {path}")
    return lookup


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
    """Compute ROC curve with proper tie handling."""
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    yy = y[order].astype(np.int64)
    P = int(yy.sum())
    N = len(yy) - P
    
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])
    
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    
    distinct = np.r_[np.diff(s) != 0, True]
    
    tp = tp[distinct]
    fp = fp[distinct]
    thr = s[distinct]
    
    tpr = tp / P
    fpr = fp / N
    
    fpr = np.r_[0.0, fpr]
    tpr = np.r_[0.0, tpr]
    thr = np.r_[np.inf, thr]
    
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
    norm_method: str = "outer"
    epoch: int = 0
    # Gen4 hard negative fields
    hard_neg_lookup: Set[Tuple[float, float]] = None
    hard_neg_weight: int = 1


class ParquetStreamDatasetWithHardNeg(IterableDataset):
    """Extended dataset that supports hard negative upsampling."""
    
    def __init__(self, cfg: StreamConfig, filesystem: Optional[pafs.FileSystem] = None):
        super().__init__()
        self.cfg = cfg
        self.fs = filesystem
        self._epoch = 0
    
    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _iter_fragments(self) -> List[ds.Fragment]:
        dataset = ds.dataset(self.cfg.data, format="parquet", filesystem=self.fs, partitioning="hive")
        frags = list(dataset.get_fragments(filter=ds.field("region_split") == self.cfg.split))
        
        wi = torch.utils.data.get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1
        
        shard = rank() * num_workers + worker_id
        nshard = world() * num_workers
        
        return frags[shard::nshard]

    def _is_hard_negative(self, ra: float, dec: float) -> bool:
        """Check if position matches a hard negative."""
        if self.cfg.hard_neg_lookup is None or len(self.cfg.hard_neg_lookup) == 0:
            return False
        ra_key = round(ra, 6)
        dec_key = round(dec, 6)
        return (ra_key, dec_key) in self.cfg.hard_neg_lookup

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float, Optional[np.ndarray]]]:
        cfg = self.cfg
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = np.random.RandomState(cfg.seed + 997 * rank() + 131 * worker_id + 7919 * self._epoch)
        logger = logging.getLogger(__name__)

        frags = self._iter_fragments()
        if not frags:
            logger.warning(f"No fragments found for split='{cfg.split}' in {cfg.data}")
            return
        rng.shuffle(frags)

        yielded = 0
        hard_neg_yielded = 0
        skip_count = 0
        total_processed = 0
        
        for frag in frags:
            if cfg.max_rows and yielded >= cfg.max_rows:
                break
            
            frag_schema_names = set(frag.physical_schema.names)
            required_cols = ["stamp_npz", "is_control"]
            missing = [c for c in required_cols if c not in frag_schema_names]
            if missing:
                raise RuntimeError(f"Fragment missing required columns {missing}")
            
            # Include ra/dec for hard negative matching
            read_cols = cfg.columns + cfg.meta_cols
            if "ra" not in read_cols:
                read_cols = read_cols + ["ra"]
            if "dec" not in read_cols:
                read_cols = read_cols + ["dec"]
            
            scanner = ds.Scanner.from_fragment(frag, columns=read_cols, batch_size=2048)
            for rb in scanner.to_batches():
                colmap = {name: rb.column(i) for i, name in enumerate(rb.schema.names)}
                n = rb.num_rows
                for i in range(n):
                    if cfg.max_rows and yielded >= cfg.max_rows:
                        break
                    
                    total_processed += 1

                    if "cutout_ok" in colmap:
                        ok = colmap["cutout_ok"][i].as_py()
                        if ok is None or int(ok) != 1:
                            continue

                    is_ctrl = int(colmap["is_control"][i].as_py())
                    if is_ctrl not in (0, 1):
                        continue

                    # Filter positives by resolvability
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

                    # Decode stamp
                    try:
                        x = decode_stamp_npz(colmap["stamp_npz"][i].as_py())
                        if cfg.norm_method == "outer":
                            x = robust_mad_norm_outer(x, clip=cfg.mad_clip)
                        else:
                            x = robust_mad_norm(x, clip=cfg.mad_clip)
                    except (ValueError, KeyError, IOError) as e:
                        skip_count += 1
                        if skip_count <= 10:
                            logger.warning(f"Skipping row due to decode error: {e}")
                        continue

                    y = 0.0 if is_ctrl == 1 else 1.0

                    meta = None
                    if cfg.meta_cols:
                        vals = []
                        for c in cfg.meta_cols:
                            v = colmap[c][i].as_py()
                            vals.append(float(v) if v is not None else 0.0)
                        meta = np.asarray(vals, dtype=np.float32)

                    # Check if this is a hard negative (control that model scored high)
                    ra = colmap["ra"][i].as_py()
                    dec = colmap["dec"][i].as_py()
                    is_hard_neg = is_ctrl == 1 and self._is_hard_negative(ra, dec)
                    
                    # Determine how many times to yield this sample
                    repeat_count = cfg.hard_neg_weight if is_hard_neg else 1
                    
                    for _ in range(repeat_count):
                        if cfg.augment:
                            x_aug = aug_rot_flip(x.copy(), rng)
                        else:
                            x_aug = x
                        
                        yielded += 1
                        if is_hard_neg:
                            hard_neg_yielded += 1
                        yield x_aug, y, meta
        
        if total_processed > 0:
            logger.info(f"Data iteration: yielded={yielded}, hard_neg={hard_neg_yielded}, "
                       f"skipped={skip_count}/{total_processed}")


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
    
    # Calibration monitoring
    binary_low = (s < 0.01).sum()
    binary_high = (s > 0.99).sum()
    binary_frac = (binary_low + binary_high) / max(len(s), 1)
    out["binary_score_frac"] = float(binary_frac)
    if binary_frac > 0.5:
        logging.warning(f"CALIBRATION COLLAPSE: {binary_frac:.1%} of scores are <0.01 or >0.99")
    
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Gen4 training with hard negative mining support."
    )
    ap.add_argument("--data", required=True, help="Path to Phase 4c stamps")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    
    # Gen4-specific: Hard negative mining
    ap.add_argument("--hard_neg_path", default="", help="Path to hard negative lookup parquet")
    ap.add_argument("--hard_neg_weight", type=int, default=5, 
                    help="How many times to yield each hard negative (upsampling factor)")
    
    # Model
    ap.add_argument("--arch", default="convnext_tiny", choices=["resnet18", "convnext_tiny", "efficientnet_b0"])
    
    # Training
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    
    # Precision
    ap.add_argument("--use_bf16", action="store_true", default=True)
    ap.add_argument("--no_bf16", action="store_false", dest="use_bf16")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    
    # Data
    ap.add_argument("--max_rows", type=int, default=0, help="Max rows per worker per epoch (0=all)")
    ap.add_argument("--val_max_batches", type=int, default=500)
    ap.add_argument("--mad_clip", type=float, default=10.0)
    ap.add_argument("--norm_method", default="outer", choices=["full", "outer"])
    
    # Filtering
    ap.add_argument("--min_theta_over_psf", type=float, default=0.5)
    ap.add_argument("--min_arc_snr", type=float, default=0.0)
    
    # Metadata
    ap.add_argument("--meta_cols", default="psfsize_r,psfdepth_r")
    
    # Augmentation
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--no_augment", action="store_false", dest="augment")
    
    # Loss
    ap.add_argument("--loss", default="focal", choices=["focal", "bce"])
    ap.add_argument("--focal_alpha", type=float, default=0.25)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    
    # Early stopping
    ap.add_argument("--early_stopping_patience", type=int, default=5)
    ap.add_argument("--resume", default="", help="Path to checkpoint to resume from")
    args = ap.parse_args()

    ddp_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load hard negative lookup
    hard_neg_lookup = load_hard_negative_lookup(args.hard_neg_path)
    
    # Parse metadata columns
    meta_cols = [c.strip() for c in args.meta_cols.split(",") if c.strip()] if args.meta_cols else []
    
    # Forbidden metadata check
    FORBIDDEN_META = frozenset({
        "theta_e_arcsec", "theta_e", "src_dmag", "src_reff_arcsec", "src_n",
        "src_e", "src_phi_deg", "src_x_arcsec", "src_y_arcsec",
        "lens_e", "lens_phi_deg", "shear_gamma", "shear_phi_deg", "shear",
        "arc_snr", "magnification", "tangential_stretch", "radial_stretch",
        "is_control", "label", "cutout_ok",
    })
    for c in meta_cols:
        if c in FORBIDDEN_META:
            raise ValueError(f"REFUSING meta_cols that leak labels: '{c}'")

    base_cols = ["stamp_npz", "is_control", "cutout_ok", "theta_e_arcsec", "psf_fwhm_used_r", "psfsize_r", "arc_snr"]
    
    # Build datasets
    train_cfg = StreamConfig(
        data=args.data, split="train", seed=args.seed,
        columns=base_cols, meta_cols=meta_cols,
        augment=args.augment, mad_clip=args.mad_clip,
        max_rows=args.max_rows,
        min_theta_over_psf=args.min_theta_over_psf,
        min_arc_snr=args.min_arc_snr,
        norm_method=args.norm_method,
        hard_neg_lookup=hard_neg_lookup,
        hard_neg_weight=args.hard_neg_weight,
    )
    val_cfg = StreamConfig(
        data=args.data, split="val", seed=args.seed + 1,
        columns=base_cols, meta_cols=meta_cols,
        augment=False, mad_clip=args.mad_clip,
        max_rows=0, min_theta_over_psf=0.0, min_arc_snr=0.0,
        norm_method=args.norm_method,
        hard_neg_lookup=None,  # Don't weight hard negatives in validation
        hard_neg_weight=1,
    )

    train_dataset = ParquetStreamDatasetWithHardNeg(train_cfg)
    val_dataset = ParquetStreamDatasetWithHardNeg(val_cfg)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             num_workers=args.num_workers, collate_fn=collate,
                             pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           num_workers=args.num_workers, collate_fn=collate,
                           pin_memory=True)

    # Build model
    model = build_model(args.arch, meta_dim=len(meta_cols), dropout=args.dropout)
    model = model.to(device)
    if is_dist():
        model = DDP(model, device_ids=[int(os.environ.get("LOCAL_RANK", "0"))])
    
    # Loss
    if args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # AMP
    use_amp = args.use_bf16 and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if args.use_bf16 else torch.float16
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    # Resume
    start_epoch = 0
    best_metric = -1.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        if is_dist():
            model.module.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("best_metric", -1.0))
        logging.info(f"Resumed from {args.resume}, epoch {start_epoch}")

    history = []
    no_improve_count = 0
    early_stop = False
    
    logger = logging.getLogger(__name__)
    logger.info(f"Gen4 Training: epochs={args.epochs}, batch={args.batch_size}, arch={args.arch}")
    logger.info(f"Hard negatives: {len(hard_neg_lookup)} positions, weight={args.hard_neg_weight}x")
    
    for epoch in range(start_epoch, args.epochs):
        if early_stop:
            break
        
        train_dataset.set_epoch(epoch)
        model.train()
        
        total_loss = 0.0
        n_batches = 0
        
        for x, y, meta in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if meta is not None:
                meta = meta.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(x, meta)
                loss = criterion(logits, y)
            
            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            
            total_loss += float(loss.item())
            n_batches += 1
            
            if n_batches % 500 == 0:
                logger.info(f"Epoch {epoch}, batch {n_batches}, loss={total_loss/n_batches:.4f}")
        
        avg_loss = total_loss / max(n_batches, 1)
        
        # Validation
        if rank() == 0:
            metrics = evaluate(model.module if is_dist() else model, val_loader, device, args.val_max_batches)
            metrics["epoch"] = epoch
            metrics["train_loss"] = avg_loss
            metrics["train_batches"] = n_batches
            history.append(metrics)
            
            score = metrics.get("tpr@fpr0.0001", 0.0)
            
            # Save checkpoints
            ckpt = {
                "epoch": epoch,
                "model_state_dict": (model.module if is_dist() else model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "metrics": metrics,
            }
            
            # Save every 5th epoch
            if epoch % 5 == 0:
                torch.save(ckpt, os.path.join(args.out_dir, f"ckpt_epoch_{epoch}.pt"))
            
            torch.save(ckpt, os.path.join(args.out_dir, "ckpt_last.pt"))

            if score > best_metric:
                best_metric = float(score)
                ckpt["best_metric"] = best_metric
                torch.save(ckpt, os.path.join(args.out_dir, "ckpt_best.pt"))
                no_improve_count = 0
                logger.info(f"Epoch {epoch}: NEW BEST tpr@fpr1e-4 = {score:.4f}")
            else:
                no_improve_count += 1
                if args.early_stopping_patience > 0 and no_improve_count >= args.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}: no improvement for {no_improve_count} epochs")
                    early_stop = True

            with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, sort_keys=True)

            print(json.dumps(metrics, indent=2, sort_keys=True))
        
        # Broadcast early_stop to all ranks
        if is_dist():
            stop_tensor = torch.tensor([1 if early_stop else 0], device=device)
            dist.broadcast(stop_tensor, src=0)
            early_stop = bool(stop_tensor.item())

        barrier()

    if rank() == 0:
        logger.info(f"Done. Best tpr@fpr1e-4={best_metric:.4f}. Outputs in {args.out_dir}")

    ddp_cleanup()


if __name__ == "__main__":
    main()


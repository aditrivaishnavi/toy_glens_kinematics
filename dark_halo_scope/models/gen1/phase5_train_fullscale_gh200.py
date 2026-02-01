#!/usr/bin/env python3
"""
phase5_train_fullscale_gh200.py

Full-scale training script for Dark Halo Scope Phase 5 (lens-finder CNN) optimized for GH200/H100-class GPUs.

Key features:
- Streams Parquet via PyArrow Dataset fragments (local paths or s3:// URIs).
- Decodes `stamp_npz` (bytes) into 3-band 64x64 images (g,r,z).
- Uses bf16 AMP by default (recommended for H100).
- Supports ResNet18 / ConvNeXt-Tiny / EfficientNet-B0 backbones.
- Optional scalar-metadata fusion head.
- Optional Phase 5 schema contract enforcement (static required columns list + assertions).
- Optional distributed training (DDP) via torchrun.

Assumptions (consistent with your Phase 4c schema):
- Image bytes are in column `stamp_npz` and contain keys: image_g, image_r, image_z.
- Label column is `is_control` where 1 = CONTROL (negative), 0 = INJECTION (positive).
- Split column is `region_split` with values in {"train","val","test"}.
- Validity column `cutout_ok` exists and should be 1 for usable rows.

Example:
torchrun --standalone --nproc_per_node=1 phase5_train_fullscale_gh200.py \
  --data s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/stamps/train_stamp64_bandsgrz_gridgrid_small/ \
  --out_dir ./runs/gh200_resnet18 \
  --model resnet18 \
  --amp_dtype bf16 \
  --batch_size 512 \
  --epochs 2 \
  --val_steps 800 \
  --contract_json /path/to/phase5_required_columns_contract.json
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs
except Exception as e:  # pragma: no cover
    raise RuntimeError("pyarrow is required (pip install pyarrow)") from e

try:
    import torchvision
    import torchvision.transforms.functional as TF
except Exception as e:  # pragma: no cover
    raise RuntimeError("torchvision is required (pip install torchvision)") from e

try:
    from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
except Exception as e:  # pragma: no cover
    raise RuntimeError("torchmetrics is required (pip install torchmetrics)") from e


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def dist_rank() -> int:
    return torch.distributed.get_rank() if is_dist() else 0


def dist_world_size() -> int:
    return torch.distributed.get_world_size() if is_dist() else 1


def dist_barrier() -> None:
    if is_dist():
        torch.distributed.barrier()


def setup_distributed_from_env() -> None:
    """
    Initialize DDP if launched with torchrun.
    Safe to call even for single-process runs.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def human(n: float) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.2f}K"
    return f"{n:.0f}"


# -----------------------------
# Contract enforcement
# -----------------------------
def load_contract_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_schema_has_columns(dataset: ds.Dataset, required_cols: Sequence[str]) -> None:
    schema = dataset.schema
    present = set(schema.names)
    missing = [c for c in required_cols if c not in present]
    if missing:
        raise RuntimeError(f"Dataset is missing required columns: {missing}")


# -----------------------------
# Data decoding + preprocessing
# -----------------------------
BANDS = ("g", "r", "z")
NPZ_KEYS = ("image_g", "image_r", "image_z")


def decode_stamp_npz(npz_bytes: bytes) -> np.ndarray:
    """
    Decode compressed NPZ (bytes) with keys image_g/image_r/image_z.
    Returns: float32 array shaped (3, H, W).
    """
    if npz_bytes is None:
        raise ValueError("stamp_npz is None")
    bio = io.BytesIO(npz_bytes)
    with np.load(bio) as z:
        imgs = []
        for k in NPZ_KEYS:
            if k not in z:
                raise KeyError(f"NPZ missing key: {k}")
            arr = z[k]
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            imgs.append(arr)
    x = np.stack(imgs, axis=0)  # (3,H,W)
    return x


def robust_norm(x: np.ndarray, clip: float = 10.0, per_channel: bool = False, eps: float = 1e-6) -> np.ndarray:
    """
    Robust normalize using median/MAD:
      z = (x - median) / (1.4826 * MAD)
    """
    if per_channel:
        # compute per-channel stats
        med = np.median(x, axis=(1, 2), keepdims=True)
        mad = np.median(np.abs(x - med), axis=(1, 2), keepdims=True)
    else:
        med = np.median(x)
        mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad + eps
    z = (x - med) / scale
    if clip is not None and clip > 0:
        z = np.clip(z, -clip, clip)
    return z.astype(np.float32, copy=False)


def aug_astro_safe(t: torch.Tensor, do_aug: bool) -> torch.Tensor:
    """
    Astronomy-safe augmentations: flips + 90-degree rotations.
    Input: tensor (C,H,W)
    """
    if not do_aug:
        return t
    # Random 90-degree rotations
    k = random.randint(0, 3)
    if k:
        t = torch.rot90(t, k, dims=(1, 2))
    # Flips
    if random.random() < 0.5:
        t = torch.flip(t, dims=(2,))
    if random.random() < 0.5:
        t = torch.flip(t, dims=(1,))
    return t


# -----------------------------
# PyArrow streaming dataset
# -----------------------------
@dataclass
class StreamConfig:
    data_uri: str
    split: str
    columns: List[str]
    batch_rows: int
    shuffle_fragments: bool
    seed: int


class ParquetStream:
    """
    Streams rows from a parquet dataset using PyArrow dataset fragments.
    Provides rank-aware sharding across fragments for DDP.

    Yields Python dict rows (column -> value) in a memory-light way.
    """
    def __init__(self, cfg: StreamConfig, filesystem: Optional[pafs.FileSystem] = None):
        self.cfg = cfg
        self.filesystem = filesystem
        
        # Check if data_uri points to a partitioned directory (e.g., region_split=train/)
        # If so, try to discover partitions; otherwise treat as flat dataset
        data_uri = cfg.data_uri.rstrip("/")
        
        # Check if we should use partitioning (directory contains region_split=* subdirs)
        try:
            self.dataset = ds.dataset(
                data_uri, 
                format="parquet", 
                filesystem=filesystem,
                partitioning="hive"  # Auto-detect Hive-style partitioning
            )
        except Exception:
            # Fallback to non-partitioned
            self.dataset = ds.dataset(data_uri, format="parquet", filesystem=filesystem)
        
        # Build filter expression - only filter by region_split if it exists in schema
        schema_names = set(self.dataset.schema.names)
        if "region_split" in schema_names:
            self.filter_expr = (ds.field("region_split") == cfg.split) & (ds.field("cutout_ok") == 1)
        elif "cutout_ok" in schema_names:
            # No region_split in schema - assume we're pointing directly at correct partition
            self.filter_expr = ds.field("cutout_ok") == 1
        else:
            self.filter_expr = None
        
        if self.filter_expr is not None:
            self._fragments = list(self.dataset.get_fragments(filter=self.filter_expr))
        else:
            self._fragments = list(self.dataset.get_fragments())
            
        if cfg.shuffle_fragments:
            rng = random.Random(cfg.seed + dist_rank())
            rng.shuffle(self._fragments)
        # Shard fragments by rank
        ws = dist_world_size()
        rk = dist_rank()
        self._fragments = [f for i, f in enumerate(self._fragments) if (i % ws) == rk]

    def __iter__(self):
        # Use a scanner per fragment to control memory and allow interleaving.
        for frag in self._fragments:
            scanner = ds.Scanner.from_fragment(
                frag,
                columns=self.cfg.columns,
                filter=None,  # already filtered at fragment selection
                batch_size=self.cfg.batch_rows,
                use_threads=True,
            )
            for rb in scanner.to_batches():
                # Convert recordbatch columns to Python lists, row-wise
                cols = {name: rb.column(i) for i, name in enumerate(rb.schema.names)}
                n = rb.num_rows
                for j in range(n):
                    row = {}
                    for name, arr in cols.items():
                        # stamp_npz is binary, others are scalars
                        row[name] = arr[j].as_py()
                    yield row


def make_filesystem_for_uri(uri: str) -> Optional[pafs.FileSystem]:
    if uri.startswith("s3://"):
        # Uses AWS credentials from env/instance role.
        return pafs.S3FileSystem(region=os.environ.get("AWS_REGION"))
    return None


# -----------------------------
# Model definitions
# -----------------------------
class ImageBackbone(nn.Module):
    def __init__(self, name: str, dropout: float = 0.0):
        super().__init__()
        self.name = name.lower()
        self.dropout = dropout

        if self.name == "resnet18":
            m = torchvision.models.resnet18(weights=None)
            # Adapt for 64x64: 3x3 stride1, remove maxpool
            m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            m.maxpool = nn.Identity()
            self.feature_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
        elif self.name == "convnext_tiny":
            m = torchvision.models.convnext_tiny(weights=None)
            # classifier: (LayerNorm2d, Flatten, Linear). Keep LayerNorm2d and Flatten, remove Linear.
            self.feature_dim = m.classifier[2].in_features  # 768
            m.classifier = nn.Sequential(
                m.classifier[0],  # LayerNorm2d
                m.classifier[1],  # Flatten
            )
            self.backbone = m
        elif self.name == "efficientnet_b0":
            m = torchvision.models.efficientnet_b0(weights=None)
            self.feature_dim = m.classifier[1].in_features  # 1280
            # classifier: (Dropout, Linear). Keep nothing but add AdaptiveAvgPool2d + Flatten
            m.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.backbone = m
        else:
            raise ValueError(f"Unknown model: {name}")

        self.drop = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.drop(feat)
        return feat


class FusedHead(nn.Module):
    def __init__(self, img_dim: int, meta_dim: int = 0, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        in_dim = img_dim + (meta_dim if meta_dim > 0 else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, feat: torch.Tensor, meta: Optional[torch.Tensor] = None) -> torch.Tensor:
        if meta is not None:
            x = torch.cat([feat, meta], dim=1)
        else:
            x = feat
        return self.net(x).squeeze(1)


class LensFinderModel(nn.Module):
    def __init__(self, backbone: str, use_metadata: bool, meta_dim: int, dropout: float):
        super().__init__()
        self.backbone = ImageBackbone(backbone, dropout=dropout)
        self.use_metadata = use_metadata
        self.head = FusedHead(self.backbone.feature_dim, meta_dim=meta_dim if use_metadata else 0)

    def forward(self, x: torch.Tensor, meta: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.backbone(x)
        if self.use_metadata:
            if meta is None:
                raise ValueError("use_metadata=True but meta is None")
            return self.head(feat, meta)
        return self.head(feat, None)


# -----------------------------
# Metadata processing
# -----------------------------
DEFAULT_META_COLS = [
    "theta_e_arcsec",
    "src_dmag",
    "src_reff_arcsec",
    "psf_fwhm_used_r",
    "psfdepth_r",
    "bad_pixel_frac",
    "wise_brightmask_frac",
    "arc_snr",
]


def compute_meta_stats(
    uri: str,
    split: str,
    meta_cols: Sequence[str],
    max_rows: int,
    batch_rows: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """
    Compute mean/std for metadata features on a sample of rows.
    (Avoids having to precompute offline; can be replaced with a fixed JSON later.)
    """
    fs = make_filesystem_for_uri(uri)
    dataset = ds.dataset(uri, format="parquet", filesystem=fs)
    filt = (ds.field("region_split") == split) & (ds.field("cutout_ok") == 1)
    cols = list(meta_cols)
    scanner = dataset.scanner(columns=cols, filter=filt, batch_size=batch_rows, use_threads=True)
    rng = random.Random(seed)
    # Reservoir sample rows to avoid reading everything
    samples = {c: [] for c in cols}
    seen = 0
    for rb in scanner.to_batches():
        n = rb.num_rows
        for i in range(n):
            seen += 1
            for ci, c in enumerate(cols):
                v = rb.column(ci)[i].as_py()
                if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    continue
                if len(samples[c]) < max_rows:
                    samples[c].append(float(v))
                else:
                    j = rng.randint(0, seen - 1)
                    if j < max_rows:
                        samples[c][j] = float(v)
        # Early stop if we've filled all lists and scanned enough
        if seen >= max_rows * 5 and all(len(samples[c]) >= max_rows for c in cols):
            break

    stats = {}
    for c in cols:
        arr = np.asarray(samples[c], dtype=np.float64)
        if arr.size == 0:
            stats[c] = {"mean": 0.0, "std": 1.0}
            continue
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        if std <= 1e-9:
            std = 1.0
        stats[c] = {"mean": mean, "std": std}
    return stats


def normalize_meta(row: Dict, meta_cols: Sequence[str], stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    feats = []
    for c in meta_cols:
        v = row.get(c, None)
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            v = stats[c]["mean"]
        v = (float(v) - stats[c]["mean"]) / stats[c]["std"]
        feats.append(v)
    return np.asarray(feats, dtype=np.float32)


# -----------------------------
# Training loops
# -----------------------------
@dataclass
class TrainCfg:
    data: str
    out_dir: str
    model: str
    amp_dtype: str
    epochs: int
    batch_size: int
    grad_accum: int
    lr: float
    weight_decay: float
    dropout: float
    seed: int
    batch_rows: int
    shuffle_fragments: bool
    aug: bool
    log_every: int
    val_steps: int
    max_train_steps: int
    use_metadata: bool
    meta_cols: List[str]
    meta_stats_json: Optional[str]
    compute_meta_stats: bool
    contract_json: Optional[str]
    save_every: int
    num_workers_hint: int  # not used for DataLoader here; kept for parity


def make_amp_autocast(dtype_name: str):
    dtype_name = dtype_name.lower()
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp16", "float16"):
        return torch.float16
    if dtype_name in ("fp32", "float32", "none"):
        return None
    raise ValueError(f"Unknown amp dtype: {dtype_name}")


def iter_batches(
    stream: Iterable[Dict],
    batch_size: int,
    aug: bool,
    meta_cols: Optional[Sequence[str]] = None,
    meta_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    """
    Turn a row-stream into torch batches.
    """
    xs, ys, ms = [], [], []
    for row in stream:
        try:
            x = decode_stamp_npz(row["stamp_npz"])
            x = robust_norm(x, clip=10.0, per_channel=False)
            xt = torch.from_numpy(x)  # (3,H,W)
            xt = aug_astro_safe(xt, do_aug=aug)
            xs.append(xt)

            is_control = row["is_control"]
            if is_control is None:
                raise ValueError("is_control is None")
            y = 0.0 if int(is_control) == 1 else 1.0
            ys.append(y)

            if meta_cols is not None and meta_stats is not None:
                m = normalize_meta(row, meta_cols, meta_stats)
                ms.append(torch.from_numpy(m))
        except Exception:
            # If decode fails, skip row (but keep the batch consistent)
            continue

        if len(xs) >= batch_size:
            xb = torch.stack(xs, dim=0)  # (B,3,H,W)
            yb = torch.tensor(ys, dtype=torch.float32)
            mb = torch.stack(ms, dim=0) if ms else None
            yield xb, yb, mb
            xs, ys, ms = [], [], []

    # last partial
    if xs:
        xb = torch.stack(xs, dim=0)
        yb = torch.tensor(ys, dtype=torch.float32)
        mb = torch.stack(ms, dim=0) if ms else None
        yield xb, yb, mb


@torch.no_grad()
def evaluate(
    model: nn.Module,
    stream: Iterable[Dict],
    device: torch.device,
    cfg: TrainCfg,
    meta_stats: Optional[Dict[str, Dict[str, float]]],
) -> Dict[str, float]:
    model.eval()
    auroc = BinaryAUROC().to(device)
    ap = BinaryAveragePrecision().to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    steps = 0
    total_loss = 0.0
    for xb, yb, mb in iter_batches(
        stream,
        batch_size=cfg.batch_size,
        aug=False,
        meta_cols=cfg.meta_cols if cfg.use_metadata else None,
        meta_stats=meta_stats if cfg.use_metadata else None,
    ):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True) if mb is not None else None

        logits = model(xb, mb)
        loss = loss_fn(logits, yb)
        total_loss += float(loss.item())

        probs = torch.sigmoid(logits)
        auroc.update(probs, yb.int())
        ap.update(probs, yb.int())

        steps += 1
        if cfg.val_steps > 0 and steps >= cfg.val_steps:
            break

    if steps == 0:
        return {"val_loss": float("nan"), "val_auroc": float("nan"), "val_ap": float("nan")}
    return {
        "val_loss": total_loss / steps,
        "val_auroc": float(auroc.compute().item()),
        "val_ap": float(ap.compute().item()),
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    step: int,
    best_metric: float,
    meta_stats: Optional[Dict],
    cfg: TrainCfg,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "meta_stats": meta_stats,
        "cfg": dataclasses.asdict(cfg),
    }
    torch.save(ckpt, path)


def train_main(cfg: TrainCfg) -> None:
    setup_distributed_from_env()
    device = get_device()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    is_main = (dist_rank() == 0)

    set_seed(cfg.seed + dist_rank())

    os.makedirs(cfg.out_dir, exist_ok=True)
    if is_main:
        with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(cfg), f, indent=2)

    fs = make_filesystem_for_uri(cfg.data)
    dataset = ds.dataset(cfg.data, format="parquet", filesystem=fs)

    # Optional contract enforcement
    if cfg.contract_json:
        contract = load_contract_json(cfg.contract_json)
        required_cols = contract.get("required_columns", [])
        assert_schema_has_columns(dataset, required_cols)

    # Columns needed for training stream
    # Only include region_split if it's in the schema (may be partitioned out)
    schema_names = set(dataset.schema.names)
    base_cols = ["stamp_npz", "is_control", "cutout_ok"]
    if "region_split" in schema_names:
        base_cols.append("region_split")
    cols = base_cols.copy()
    if cfg.use_metadata:
        cols += cfg.meta_cols
    # De-duplicate while preserving order, only keep columns that exist
    seen = set()
    cols = [c for c in cols if c in schema_names and not (c in seen or seen.add(c))]

    # Meta stats
    meta_stats = None
    if cfg.use_metadata:
        if cfg.meta_stats_json and os.path.exists(cfg.meta_stats_json):
            with open(cfg.meta_stats_json, "r", encoding="utf-8") as f:
                meta_stats = json.load(f)
        elif cfg.compute_meta_stats:
            if is_main:
                meta_stats = compute_meta_stats(
                    uri=cfg.data,
                    split="train",
                    meta_cols=cfg.meta_cols,
                    max_rows=200_000,
                    batch_rows=cfg.batch_rows,
                    seed=cfg.seed,
                )
                with open(os.path.join(cfg.out_dir, "meta_stats.json"), "w", encoding="utf-8") as f:
                    json.dump(meta_stats, f, indent=2)
            dist_barrier()
            if not is_main:
                with open(os.path.join(cfg.out_dir, "meta_stats.json"), "r", encoding="utf-8") as f:
                    meta_stats = json.load(f)
        else:
            # Fallback: standard normal
            meta_stats = {c: {"mean": 0.0, "std": 1.0} for c in cfg.meta_cols}

    # Model
    model = LensFinderModel(
        backbone=cfg.model,
        use_metadata=cfg.use_metadata,
        meta_dim=len(cfg.meta_cols) if cfg.use_metadata else 0,
        dropout=cfg.dropout,
    ).to(device)

    if is_dist():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Simple cosine schedule across total steps (approx)
    # If max_train_steps is set, cosine over that, else approximate from epochs * (unknown) => use epoch-local cosine.
    loss_fn = nn.BCEWithLogitsLoss()

    amp_dtype = make_amp_autocast(cfg.amp_dtype)
    use_amp = (amp_dtype is not None and device.type == "cuda")

    if cfg.amp_dtype.lower() in ("fp16", "float16"):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None  # bf16 and fp32 do not need scaling

    writer = SummaryWriter(cfg.out_dir) if is_main else None

    best_auroc = -1.0
    global_step = 0

    def make_stream(split: str, shuffle: bool) -> ParquetStream:
        scfg = StreamConfig(
            data_uri=cfg.data,
            split=split,
            columns=cols,
            batch_rows=cfg.batch_rows,
            shuffle_fragments=shuffle,
            seed=cfg.seed,
        )
        return ParquetStream(scfg, filesystem=fs)

    # Train epochs
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        steps_in_epoch = 0

        # For streaming: we stop after max_train_steps (global) if configured
        train_stream = make_stream("train", shuffle=cfg.shuffle_fragments)

        for xb, yb, mb in iter_batches(
            train_stream,
            batch_size=cfg.batch_size,
            aug=cfg.aug,
            meta_cols=cfg.meta_cols if cfg.use_metadata else None,
            meta_stats=meta_stats if cfg.use_metadata else None,
        ):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True) if mb is not None else None

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(xb, mb)
                loss = loss_fn(logits, yb) / max(1, cfg.grad_accum)

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % cfg.grad_accum == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item()) * max(1, cfg.grad_accum)
            steps_in_epoch += 1
            global_step += 1

            if is_main and cfg.log_every > 0 and (global_step % cfg.log_every == 0):
                lr = optimizer.param_groups[0]["lr"]
                msg = f"epoch={epoch} step={global_step} loss={running_loss/steps_in_epoch:.4f} lr={lr:.2e}"
                print(msg, flush=True)
                if writer:
                    writer.add_scalar("train/loss", running_loss / steps_in_epoch, global_step)
                    writer.add_scalar("train/lr", lr, global_step)

            if cfg.max_train_steps > 0 and global_step >= cfg.max_train_steps:
                break

        # Validation
        val_stream = make_stream("val", shuffle=False)
        val_metrics = evaluate(model, val_stream, device, cfg, meta_stats)

        if is_main:
            dt = time.time() - t0
            print(
                f"Epoch {epoch} done in {dt/60:.1f} min | "
                f"train_loss={running_loss/max(1,steps_in_epoch):.4f} "
                f"val_loss={val_metrics['val_loss']:.4f} "
                f"val_auroc={val_metrics['val_auroc']:.4f} "
                f"val_ap={val_metrics['val_ap']:.4f}",
                flush=True,
            )
            if writer:
                writer.add_scalar("val/loss", val_metrics["val_loss"], epoch)
                writer.add_scalar("val/auroc", val_metrics["val_auroc"], epoch)
                writer.add_scalar("val/ap", val_metrics["val_ap"], epoch)

        # Checkpoints (main process only)
        if is_main and (epoch % cfg.save_every == 0):
            save_checkpoint(
                path=os.path.join(cfg.out_dir, "checkpoints", f"epoch_{epoch}.pt"),
                model=model.module if hasattr(model, "module") else model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                best_metric=best_auroc,
                meta_stats=meta_stats,
                cfg=cfg,
            )

        # Best model
        if is_main and not math.isnan(val_metrics["val_auroc"]) and val_metrics["val_auroc"] > best_auroc:
            best_auroc = val_metrics["val_auroc"]
            save_checkpoint(
                path=os.path.join(cfg.out_dir, "checkpoints", "best.pt"),
                model=model.module if hasattr(model, "module") else model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                best_metric=best_auroc,
                meta_stats=meta_stats,
                cfg=cfg,
            )

        if cfg.max_train_steps > 0 and global_step >= cfg.max_train_steps:
            break

    if is_main:
        if writer:
            writer.flush()
            writer.close()
        print(f"Best val AUROC: {best_auroc:.4f}", flush=True)


def parse_args() -> TrainCfg:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Parquet dataset path (local dir or s3://...)")
    p.add_argument("--out_dir", required=True, help="Output directory for logs/checkpoints")
    p.add_argument("--model", default="resnet18", choices=["resnet18", "convnext_tiny", "efficientnet_b0"])
    p.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--batch_rows", type=int, default=2048, help="Rows per PyArrow record batch")
    p.add_argument("--shuffle_fragments", action="store_true", help="Shuffle fragments each epoch (rank-local)")
    p.add_argument("--no_aug", action="store_true", help="Disable astronomy-safe augmentation")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_steps", type=int, default=800, help="Val batches per epoch (0 = full val)")
    p.add_argument("--max_train_steps", type=int, default=0, help="Optional cap on total train steps")
    p.add_argument("--use_metadata", action="store_true", help="Concatenate scalar metadata to image embedding")
    p.add_argument("--meta_cols", type=str, default=",".join(DEFAULT_META_COLS), help="Comma-separated metadata cols")
    p.add_argument("--meta_stats_json", type=str, default="", help="JSON with mean/std per meta col")
    p.add_argument("--compute_meta_stats", action="store_true", help="Compute meta stats from train split")
    p.add_argument("--contract_json", type=str, default="", help="Phase 5 contract json to enforce required columns")
    p.add_argument("--save_every", type=int, default=1, help="Save epoch checkpoints every N epochs")
    p.add_argument("--num_workers_hint", type=int, default=0)
    args = p.parse_args()

    return TrainCfg(
        data=args.data,
        out_dir=args.out_dir,
        model=args.model,
        amp_dtype=args.amp_dtype,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=max(1, args.grad_accum),
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        seed=args.seed,
        batch_rows=args.batch_rows,
        shuffle_fragments=bool(args.shuffle_fragments),
        aug=(not args.no_aug),
        log_every=args.log_every,
        val_steps=args.val_steps,
        max_train_steps=args.max_train_steps,
        use_metadata=bool(args.use_metadata),
        meta_cols=[c.strip() for c in args.meta_cols.split(",") if c.strip()],
        meta_stats_json=args.meta_stats_json if args.meta_stats_json else None,
        compute_meta_stats=bool(args.compute_meta_stats),
        contract_json=args.contract_json if args.contract_json else None,
        save_every=max(1, args.save_every),
        num_workers_hint=args.num_workers_hint,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_main(cfg)

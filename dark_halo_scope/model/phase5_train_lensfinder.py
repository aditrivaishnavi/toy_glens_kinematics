#!/usr/bin/env python3
"""
Phase 5: Train an image-based lens-finder (CNN) on Phase 4c stamps.

Key properties:
- PyTorch training with optional Distributed Data Parallel (DDP) via torchrun.
- Reads unified Phase 4c parquet (stamps + metrics in same rows).
- Decodes stamp_npz binary column containing compressed NPZ with image_g/r/z.
- Enforces a "required columns" contract so Phase 5 cannot silently drop
  any column needed later for completeness binning and provenance.

Typical single-node multi-GPU launch (4 GPUs):
  torchrun --standalone --nproc_per_node=4 phase5_train_lensfinder.py \
    --data "/local/phase4c/stamps/train_stamp64_bandsgrz_gridgrid_small" \
    --contract_json "dark_halo_scope/model/phase5_required_columns_contract.json" \
    --split train \
    --arch resnet18 \
    --out_dir "/local/phase5/models/resnet18_v1" \
    --epochs 5 --steps_per_epoch 5000 --batch_size 256

Notes:
- Phase 4c stores images in stamp_npz as compressed NPZ with keys image_g, image_r, image_z
- The label is binary: injected lens vs control, using lens_model == "CONTROL".
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import fsspec
except Exception as e:
    raise RuntimeError("Missing dependency fsspec. Install: pip install fsspec s3fs") from e

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    raise RuntimeError("Missing dependency pyarrow. Install: pip install pyarrow") from e

try:
    from torchvision import models
except Exception as e:
    raise RuntimeError("Missing dependency torchvision. Install: pip install torchvision") from e

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None  # type: ignore


KEY_COLS_DEFAULT = ["experiment_id", "task_id"]


def _is_s3_path(path: str) -> bool:
    return path.startswith("s3://")


def _fs_for(path: str):
    return fsspec.open(path).fs


def _list_parquet_files(path: str) -> List[str]:
    """List all parquet files under a path (local or S3)."""
    if _is_s3_path(path):
        fs = _fs_for(path)
        glob_pat = path.rstrip("/") + "/**/*.parquet"
        files = fs.glob(glob_pat)
        return sorted(["s3://" + f if not f.startswith("s3://") else f for f in files])
    else:
        import glob
        files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
        return sorted(files)


def _open_parquet(path: str):
    if _is_s3_path(path):
        fs = _fs_for(path)
        f = fs.open(path, "rb")
        return pq.ParquetFile(f)
    return pq.ParquetFile(path)


def _load_contract_required_cols(contract_json_path: str) -> List[str]:
    """Load required columns from contract JSON."""
    p = contract_json_path
    if p.startswith("sandbox:"):
        p = p.replace("sandbox:", "")
    with open(p, "r") as f:
        doc = json.load(f)
    # Try both key names for compatibility
    cols = doc.get("phase5_required_columns", doc.get("required_columns", []))
    return list(cols)


def _decode_stamp_npz(npz_bytes: bytes, bands: Tuple[str, str, str] = ("g", "r", "z")) -> Dict[str, np.ndarray]:
    """Decode compressed NPZ from stamp_npz column into per-band image arrays."""
    if npz_bytes is None:
        raise ValueError("stamp_npz is None")
    bio = io.BytesIO(npz_bytes)
    with np.load(bio) as npz:
        result = {}
        for band in bands:
            key = f"image_{band}"
            if key in npz.files:
                result[band] = npz[key]
            else:
                raise ValueError(f"Missing {key} in stamp_npz")
        return result


def _robust_normalize(img: np.ndarray) -> np.ndarray:
    """Per-stamp robust normalization: (x - median) / (1.4826*MAD)."""
    x = img.astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = float(np.std(x) + 1e-6)
    y = (x - med) / scale
    return np.clip(y, -10.0, 10.0)


class UnifiedParquetDataset(IterableDataset):
    """
    Streams Phase 4c unified parquet (stamps + metrics in same rows).
    Decodes stamp_npz binary column to extract g/r/z images.
    """
    
    def __init__(
        self,
        parquet_files: List[str],
        split: str,
        required_cols: Sequence[str],
        image_hw: Tuple[int, int] = (64, 64),
        seed: int = 0,
        max_rows_per_shard: Optional[int] = None,
    ):
        super().__init__()
        self.parquet_files = parquet_files
        self.split = split
        self.required_cols = list(required_cols)
        self.h, self.w = image_hw
        self.seed = seed
        self.max_rows_per_shard = max_rows_per_shard

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]]:
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # DDP rank sharding
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))

        # Deterministic shard ordering per epoch
        rng = random.Random(self.seed + rank)
        files = list(self.parquet_files)
        rng.shuffle(files)

        # Distribute shards across (rank, worker)
        assigned: List[str] = []
        for i, f in enumerate(files):
            if (i % world) != rank:
                continue
            j = len(assigned)
            if (j % num_workers) != worker_id:
                continue
            assigned.append(f)

        if not assigned:
            return iter(())

        for parquet_file in assigned:
            yield from self._iter_one_file(parquet_file)

    def _iter_one_file(self, parquet_file: str):
        """Iterate over rows in a single parquet file."""
        pf = _open_parquet(parquet_file)
        schema_names = pf.schema.names

        # Verify required columns exist
        missing = [c for c in self.required_cols if c not in schema_names and c != "stamp_npz"]
        if missing:
            # Only warn, don't fail - some columns may be optional
            pass

        # Columns to read
        cols_to_read = ["stamp_npz", "lens_model", "region_split", "cutout_ok"]
        cols_to_read += [c for c in ["theta_e_arcsec", "psf_fwhm_used_r", "psfdepth_r", 
                                      "bad_pixel_frac", "wise_brightmask_frac", 
                                      "experiment_id", "task_id"] if c in schema_names]
        cols_to_read = list(set(cols_to_read))

        n_row_groups = pf.num_row_groups
        for rg in range(n_row_groups):
            table = pf.read_row_group(rg, columns=cols_to_read)
            
            # Filter by split
            if "region_split" in table.column_names:
                split_arr = table["region_split"].to_pylist()
                keep_idx = [i for i, s in enumerate(split_arr) if s == self.split]
            else:
                keep_idx = list(range(table.num_rows))

            if not keep_idx:
                continue

            # Cap if needed
            if self.max_rows_per_shard is not None:
                keep_idx = keep_idx[:self.max_rows_per_shard]

            # Get columns as lists
            stamp_npz_col = table["stamp_npz"].to_pylist()
            lens_model_col = table["lens_model"].to_pylist() if "lens_model" in table.column_names else [None] * table.num_rows
            cutout_ok_col = table["cutout_ok"].to_pylist() if "cutout_ok" in table.column_names else [1] * table.num_rows

            # Metadata for debugging
            meta_cols = ["experiment_id", "task_id", "theta_e_arcsec", "psf_fwhm_used_r", 
                        "psfdepth_r", "bad_pixel_frac", "wise_brightmask_frac"]
            meta_arrays = {}
            for c in meta_cols:
                if c in table.column_names:
                    meta_arrays[c] = table[c].to_pylist()

            for i in keep_idx:
                try:
                    # Skip failed cutouts
                    if cutout_ok_col[i] != 1:
                        continue
                    
                    # Skip if stamp_npz is None (metrics_only mode)
                    npz_bytes = stamp_npz_col[i]
                    if npz_bytes is None:
                        continue

                    # Decode NPZ
                    imgs = _decode_stamp_npz(npz_bytes)
                    
                    # Stack and normalize
                    ig = _robust_normalize(imgs["g"])
                    ir = _robust_normalize(imgs["r"])
                    iz = _robust_normalize(imgs["z"])
                    
                    x = np.stack([ig, ir, iz], axis=0)
                    x_t = torch.from_numpy(x).float()

                    # Label: 0 for control, 1 for injection
                    lens = lens_model_col[i]
                    y = 0.0 if (lens == "CONTROL") else 1.0
                    y_t = torch.tensor([y], dtype=torch.float32)

                    # Metadata dict
                    meta = {c: meta_arrays[c][i] for c in meta_arrays}
                    yield x_t, y_t, meta

                except Exception as e:
                    # Skip pathological rows; log rarely
                    if random.random() < 1e-4:
                        print(f"[WARN] Skipping row due to parse error: {e}")
                    continue


def build_model(arch: str, in_ch: int = 3) -> nn.Module:
    """Build CNN model for binary classification."""
    arch = arch.lower()
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        # Adapt for 64x64 input (smaller stride, no maxpool)
        m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m
    if arch == "resnet34":
        m = models.resnet34(weights=None)
        m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        if in_ch != 3:
            first = m.features[0][0]
            m.features[0][0] = nn.Conv2d(in_ch, first.out_channels, kernel_size=first.kernel_size,
                                         stride=first.stride, padding=first.padding, bias=False)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
        return m
    if arch == "convnext_tiny":
        m = models.convnext_tiny(weights=None)
        if in_ch != 3:
            first = m.features[0][0]
            m.features[0][0] = nn.Conv2d(in_ch, first.out_channels, kernel_size=first.kernel_size,
                                         stride=first.stride, padding=first.padding, bias=False)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
        return m
    if arch == "small_cnn":
        return nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1),
        )
    raise ValueError(f"Unknown arch: {arch}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 200) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    n = 0
    loss_sum = 0.0
    correct = 0
    bce = nn.BCEWithLogitsLoss()
    for batch_idx, (x, y, _meta) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logit = model(x)
        loss = bce(logit, y)
        loss_sum += float(loss.item()) * x.size(0)
        pred = (torch.sigmoid(logit) >= 0.5).float()
        correct += int((pred == y).sum().item())
        n += x.size(0)
        if batch_idx + 1 >= max_batches:
            break
    if n == 0:
        return {"val_loss": float("nan"), "val_acc": float("nan")}
    return {"val_loss": loss_sum / n, "val_acc": correct / n}


def setup_ddp() -> Tuple[int, int, int]:
    """
    Returns (rank, world_size, local_rank).
    Uses env vars set by torchrun.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank
    return 0, 1, 0


def is_main_process(rank: int) -> bool:
    return rank == 0


def _mkdir(path: str):
    if path.startswith("s3://"):
        return
    os.makedirs(path, exist_ok=True)


def _write_bytes(path: str, data: bytes):
    if path.startswith("s3://"):
        with fsspec.open(path, "wb") as f:
            f.write(data)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, "wb") as f:
            f.write(data)


def _torch_save_to_bytes(obj: Dict) -> bytes:
    b = io.BytesIO()
    torch.save(obj, b)
    return b.getvalue()


def main():
    ap = argparse.ArgumentParser(description="Phase 5: Train lens-finder CNN on Phase 4c stamps")
    ap.add_argument("--data", required=True, help="Root path to Phase 4c unified parquet (stamps + metrics)")
    ap.add_argument("--contract_json", required=True, help="phase5_required_columns_contract.json path")
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--arch", default="resnet18", help="resnet18|resnet34|efficientnet_b0|convnext_tiny|small_cnn")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps_per_epoch", type=int, default=5000, help="Training steps per epoch (streaming)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_rows_per_shard", type=int, default=0, help="Dev mode cap; 0 means no cap")
    ap.add_argument("--out_dir", required=True, help="Output directory for checkpoints and logs")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--stamp_size", type=int, default=64, help="Stamp size (64 or 96)")
    args = ap.parse_args()

    rank, world, local_rank = setup_ddp()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    # Load contract
    required_cols = _load_contract_required_cols(args.contract_json)
    if is_main_process(rank):
        print(f"[INFO] Loaded {len(required_cols)} required columns from contract")

    # List parquet files
    parquet_files = _list_parquet_files(args.data)
    if not parquet_files:
        raise RuntimeError(f"No parquet files found at {args.data}")
    if is_main_process(rank):
        print(f"[INFO] Found {len(parquet_files)} parquet files")

    max_rows = args.max_rows_per_shard if args.max_rows_per_shard > 0 else None

    # Training dataset
    train_ds = UnifiedParquetDataset(
        parquet_files=parquet_files,
        split=args.split,
        required_cols=required_cols,
        image_hw=(args.stamp_size, args.stamp_size),
        seed=args.seed,
        max_rows_per_shard=max_rows,
    )

    # Validation dataset (use smaller cap)
    val_ds = UnifiedParquetDataset(
        parquet_files=parquet_files,
        split="val",
        required_cols=required_cols,
        image_hw=(args.stamp_size, args.stamp_size),
        seed=args.seed + 999,
        max_rows_per_shard=2000,
    )

    loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=max(1, args.num_workers // 2), pin_memory=True)

    model = build_model(args.arch).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    bce = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    _mkdir(args.out_dir)
    writer = None
    if is_main_process(rank) and SummaryWriter is not None and not args.out_dir.startswith("s3://"):
        writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb"))

    global_step = 0
    best_val_loss = float("inf")

    if is_main_process(rank):
        print(f"[INFO] arch={args.arch} split={args.split} world={world} batch={args.batch_size} steps_per_epoch={args.steps_per_epoch}")
        print(f"[INFO] device={device} stamp_size={args.stamp_size}")

    for epoch in range(args.epochs):
        if world > 1:
            dist.barrier()

        model.train()
        running = {"loss": 0.0, "n": 0}

        it = iter(loader)
        for step in range(args.steps_per_epoch):
            try:
                x, y, _meta = next(it)
            except StopIteration:
                it = iter(loader)
                x, y, _meta = next(it)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logit = model(x)
            loss = bce(logit, y)
            loss.backward()
            opt.step()

            running["loss"] += float(loss.item()) * x.size(0)
            running["n"] += x.size(0)
            global_step += 1

            if is_main_process(rank) and (global_step % args.log_every == 0):
                avg = running["loss"] / max(1, running["n"])
                img_per_sec = running["n"] / (args.log_every * (1.0 / args.log_every))  # Approximate
                print(f"[TRAIN] epoch={epoch} step={global_step} loss={avg:.5f} n={running['n']}")
                if writer:
                    writer.add_scalar("train/loss", avg, global_step)
                running = {"loss": 0.0, "n": 0}

        # Evaluate
        if is_main_process(rank):
            val = evaluate(model.module if hasattr(model, "module") else model, val_loader, device=device, max_batches=200)
            print(f"[VAL] epoch={epoch} val_loss={val['val_loss']:.5f} val_acc={val['val_acc']:.4f}")
            if writer:
                writer.add_scalar("val/loss", val["val_loss"], global_step)
                writer.add_scalar("val/acc", val["val_acc"], global_step)

            # Save checkpoints
            state = {
                "epoch": epoch,
                "global_step": global_step,
                "arch": args.arch,
                "state_dict": (model.module if hasattr(model, "module") else model).state_dict(),
                "optimizer": opt.state_dict(),
                "args": vars(args),
            }

            # Always save a rolling "last"
            last_path = os.path.join(args.out_dir, "checkpoint_last.pt")
            _write_bytes(last_path, _torch_save_to_bytes(state))

            # Save best by val_loss
            if math.isfinite(val["val_loss"]) and val["val_loss"] < best_val_loss:
                best_val_loss = float(val["val_loss"])
                best_path = os.path.join(args.out_dir, "checkpoint_best.pt")
                _write_bytes(best_path, _torch_save_to_bytes(state))
                print(f"[INFO] New best checkpoint saved: {best_path}")

        if world > 1:
            dist.barrier()

    if world > 1:
        dist.destroy_process_group()

    if is_main_process(rank):
        print(f"[DONE] Training complete. Best val_loss={best_val_loss:.5f}")


if __name__ == "__main__":
    main()

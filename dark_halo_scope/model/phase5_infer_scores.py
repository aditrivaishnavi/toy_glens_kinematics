#!/usr/bin/env python3
"""
Phase 5: Run GPU inference for a trained lens-finder and write per-row scores to parquet.

Inputs:
- Phase 4c unified parquet (stamps + metrics in same rows)
- Trained checkpoint (checkpoint_best.pt or checkpoint_last.pt)

Outputs:
- A parquet dataset with one row per stamp, carrying:
  - keys (experiment_id, task_id)
  - split, injection/control label, and all contract columns needed for
    Phase 5 completeness binning
  - model_logit and model_score (sigmoid(logit))

Typical run:
  python phase5_infer_scores.py \
    --data "/local/phase4c/stamps/train_stamp64_bandsgrz_gridgrid_small" \
    --contract_json "dark_halo_scope/model/phase5_required_columns_contract.json" \
    --checkpoint "/local/phase5/models/resnet18_v1/checkpoint_best.pt" \
    --arch resnet18 \
    --split test \
    --out "/local/phase5/scores/resnet18_v1/test_scores"

Notes:
- Phase 4c stores images in stamp_npz as compressed NPZ with keys image_g, image_r, image_z
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

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
    # Try relative import first (when running as module)
    from .data_cache import DataCache
    HAS_CACHE = True
except ImportError:
    try:
        # Try absolute import (when running directly)
        from dark_halo_scope.model.data_cache import DataCache
        HAS_CACHE = True
    except ImportError:
        try:
            # Try same-directory import (when running from model dir)
            from data_cache import DataCache
            HAS_CACHE = True
        except ImportError:
            HAS_CACHE = False


KEY_COLS_DEFAULT = ["experiment_id", "task_id"]


def _is_s3(path: str) -> bool:
    return path.startswith("s3://")


def _fs_for(path: str):
    return fsspec.open(path).fs


def _list_parquet_files(root: str) -> List[str]:
    if _is_s3(root):
        fs = _fs_for(root)
        files = fs.glob(root.rstrip("/") + "/**/*.parquet")
        out = ["s3://" + f if not f.startswith("s3://") else f for f in files]
        return sorted(out)
    else:
        import glob
        files = glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)
        return sorted(files)


def _open_parquet(path: str):
    if _is_s3(path):
        fs = _fs_for(path)
        f = fs.open(path, "rb")
        return pq.ParquetFile(f)
    return pq.ParquetFile(path)


def _load_contract_cols(contract_json_path: str) -> List[str]:
    p = contract_json_path
    if p.startswith("sandbox:"):
        p = p.replace("sandbox:", "")
    with open(p, "r") as f:
        doc = json.load(f)
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
    x = img.astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = float(np.std(x) + 1e-6)
    y = (x - med) / scale
    return np.clip(y, -10.0, 10.0)


def build_model(arch: str, in_ch: int = 3) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        m = models.resnet18(weights=None)
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


def _write_parquet(out_path: str, table: pa.Table):
    if _is_s3(out_path):
        fs = _fs_for(out_path)
        with fs.open(out_path, "wb") as f:
            pq.write_table(table, f, compression="zstd")
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pq.write_table(table, out_path, compression="zstd")


def main():
    ap = argparse.ArgumentParser(description="Phase 5: Run inference and write scores")
    ap.add_argument("--data", required=True, help="Root path to Phase 4c unified parquet (local or s3://...)")
    ap.add_argument("--contract_json", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    ap.add_argument("--out", required=True, help="Output directory for score parquet files")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_row_groups", type=int, default=0, help="Dev mode cap; 0 means no cap")
    ap.add_argument("--stamp_size", type=int, default=64)
    ap.add_argument("--cache_root", default="/data/cache", help="Local cache directory for S3 data")
    ap.add_argument("--force_cache_refresh", action="store_true", help="Force re-download from S3")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[INFO] Using device: {device}")

    # Resolve data path: use cache if S3 URI
    data_path = args.data
    if data_path.startswith("s3://"):
        if HAS_CACHE:
            print(f"[INFO] Data is S3 URI, using cache...")
            cache = DataCache(cache_root=args.cache_root)
            data_path = cache.get(data_path, force_refresh=args.force_cache_refresh)
        else:
            print(f"[WARN] S3 URI provided but data_cache module not available. Streaming from S3.")
    print(f"[INFO] Data path: {data_path}")

    required_cols = _load_contract_cols(args.contract_json)
    print(f"[INFO] Contract requires {len(required_cols)} columns")
    
    # Validate contract columns exist in first parquet file (fail-fast)
    parquet_files = _list_parquet_files(data_path)
    if not parquet_files:
        raise RuntimeError(f"No parquet files found at {data_path}")
    
    first_pf = _open_parquet(parquet_files[0])
    schema_names = set(first_pf.schema.names)
    missing_contract = [c for c in required_cols if c not in schema_names]
    if missing_contract:
        raise RuntimeError(
            f"CONTRACT VIOLATION: Missing {len(missing_contract)} required columns in data.\n"
            f"Missing: {missing_contract[:10]}{'...' if len(missing_contract) > 10 else ''}\n"
            f"Available: {sorted(schema_names)[:20]}..."
        )
    print(f"[INFO] Contract validation passed - all {len(required_cols)} columns present")

    # Load model
    model = build_model(args.arch).to(device)
    ckpt_path = args.checkpoint
    if ckpt_path.startswith("sandbox:"):
        ckpt_path = ckpt_path.replace("sandbox:", "")
    if _is_s3(ckpt_path):
        with fsspec.open(ckpt_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu")
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[INFO] Loaded checkpoint from {ckpt_path}")
    print(f"[INFO] Found {len(parquet_files)} parquet files")

    out_root = args.out.rstrip("/")
    os.makedirs(out_root, exist_ok=True) if not _is_s3(out_root) else None

    h, w = args.stamp_size, args.stamp_size
    total_rows = 0
    skipped_counts = {"cutout_ok_0": 0, "stamp_npz_null": 0, "decode_error": 0}

    for shard_idx, parquet_file in enumerate(parquet_files):
        pf = _open_parquet(parquet_file)
        schema_names = pf.schema.names

        # Columns to read: contract columns + stamp_npz + metadata
        cols_to_read = ["stamp_npz", "lens_model", "region_split", "cutout_ok"]
        cols_to_read += [c for c in required_cols if c in schema_names]
        cols_to_read = list(set(cols_to_read))

        n_row_groups = pf.num_row_groups
        if args.max_row_groups and args.max_row_groups > 0:
            n_row_groups = min(n_row_groups, args.max_row_groups)

        for rg in range(n_row_groups):
            table = pf.read_row_group(rg, columns=cols_to_read)

            # Filter split
            if args.split != "all" and "region_split" in table.column_names:
                split_arr = table["region_split"].to_pylist()
                keep = [i for i, s in enumerate(split_arr) if s == args.split]
            else:
                keep = list(range(table.num_rows))
            if not keep:
                continue

            # Get columns
            stamp_npz_col = table["stamp_npz"].to_pylist()
            lens_model_col = table["lens_model"].to_pylist() if "lens_model" in table.column_names else [None] * table.num_rows
            cutout_ok_col = table["cutout_ok"].to_pylist() if "cutout_ok" in table.column_names else [1] * table.num_rows

            # Read all contract columns for output
            meta_cols = {}
            for c in required_cols:
                if c in table.column_names:
                    meta_cols[c] = table[c].to_pylist()

            # Prepare output lists
            logit_out = [float('nan')] * len(keep)
            score_out = [float('nan')] * len(keep)
            label_keep = []
            valid_mask = []

            for j, i in enumerate(keep):
                lens = lens_model_col[i]
                label_keep.append(0 if lens == "CONTROL" else 1)
                is_valid = cutout_ok_col[i] == 1 and stamp_npz_col[i] is not None
                valid_mask.append(is_valid)
                if cutout_ok_col[i] != 1:
                    skipped_counts["cutout_ok_0"] += 1
                elif stamp_npz_col[i] is None:
                    skipped_counts["stamp_npz_null"] += 1

            # Batch inference
            def flush_batch(batch_x: List[np.ndarray], batch_pos: List[int]):
                if not batch_x:
                    return
                xb = torch.from_numpy(np.stack(batch_x, axis=0)).float().to(device)
                with torch.no_grad():
                    logit = model(xb).squeeze(1).detach().cpu().numpy()
                    prob = 1.0 / (1.0 + np.exp(-logit))
                for pos, l, s in zip(batch_pos, logit.tolist(), prob.tolist()):
                    logit_out[pos] = float(l)
                    score_out[pos] = float(s)

            batch_x: List[np.ndarray] = []
            batch_pos: List[int] = []
            
            for j, i in enumerate(keep):
                if not valid_mask[j]:
                    continue
                try:
                    npz_bytes = stamp_npz_col[i]
                    imgs = _decode_stamp_npz(npz_bytes)
                    ig = _robust_normalize(imgs["g"])
                    ir = _robust_normalize(imgs["r"])
                    iz = _robust_normalize(imgs["z"])
                    x = np.stack([ig, ir, iz], axis=0)
                    batch_x.append(x)
                    batch_pos.append(j)
                    if len(batch_x) >= args.batch_size:
                        flush_batch(batch_x, batch_pos)
                        batch_x, batch_pos = [], []
                except Exception:
                    skipped_counts["decode_error"] += 1
                    continue
            flush_batch(batch_x, batch_pos)

            # Build output table
            out_cols = {}
            for c in required_cols:
                if c in meta_cols:
                    out_cols[c] = pa.array([meta_cols[c][i] for i in keep])
                else:
                    out_cols[c] = pa.nulls(len(keep))

            out_cols["lens_model"] = pa.array([lens_model_col[i] for i in keep])
            if "region_split" in table.column_names:
                out_cols["region_split"] = pa.array([table["region_split"].to_pylist()[i] for i in keep])
            out_cols["y_true"] = pa.array(label_keep, type=pa.int8())
            out_cols["model_logit"] = pa.array(logit_out, type=pa.float32())
            out_cols["model_score"] = pa.array(score_out, type=pa.float32())

            out_table = pa.table(out_cols)

            # Write output
            part = f"part-shard{shard_idx:05d}-rg{rg:03d}.parquet"
            out_path = out_root + "/" + part
            _write_parquet(out_path, out_table)
            total_rows += out_table.num_rows

        if shard_idx % 50 == 0:
            print(f"[INFO] processed shard={shard_idx}/{len(parquet_files)} total_rows_written={total_rows}")

    print(f"[DONE] total_rows_written={total_rows} out={args.out}")
    print(f"[INFO] Skipped rows: {skipped_counts}")


if __name__ == "__main__":
    main()

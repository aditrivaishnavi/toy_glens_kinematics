#!/usr/bin/env python3
"""
Phase 5: Run GPU inference for a trained lens-finder and write per-row scores to parquet.

Inputs:
- Phase 4c stamps parquet (pixels)
- Phase 4c metrics parquet (metadata + provenance)
- Trained checkpoint (checkpoint_best.pt or checkpoint_last.pt)

Outputs:
- A parquet dataset with one row per stamp, carrying:
  - keys (experiment_id, task_id)
  - split, injection/control label, and all contract columns needed for
    Phase 5 completeness binning
  - model_logit and model_score (sigmoid(logit))

Typical run:
  python phase5_infer_scores.py \
    --stamps "s3://.../phase4c/.../stamps/train_stamp64_bandsgrz_gridgrid_small" \
    --metrics "s3://.../phase4c/.../metrics/train_stamp64_bandsgrz_gridgrid_small" \
    --contract_json "sandbox:/mnt/data/phase5_required_columns_contract.json" \
    --checkpoint "s3://.../phase5/models/resnet18_v1/checkpoint_best.pt" \
    --arch resnet18 \
    --split test \
    --out "s3://.../phase5/scores/resnet18_v1/test_scores"

Notes:
- This script assumes stamps and metrics parquet shards are row-aligned as produced by Phase 4c.
- If experiment_id/task_id exist in both tables, alignment is verified for a small sample.
"""

from __future__ import annotations

import argparse
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


IMAGE_COL_CANDIDATES: List[Tuple[str, str, str]] = [
    ("image_g", "image_r", "image_z"),
    ("stamp_g", "stamp_r", "stamp_z"),
    ("img_g", "img_r", "img_z"),
    ("g", "r", "z"),
]
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
    return list(doc.get("required_columns", []))


def _detect_image_cols(schema_names: Sequence[str]) -> Tuple[str, str, str]:
    s = set(schema_names)
    for cg, cr, cz in IMAGE_COL_CANDIDATES:
        if cg in s and cr in s and cz in s:
            return cg, cr, cz
    raise ValueError(f"Could not find image columns for g/r/z. Tried: {IMAGE_COL_CANDIDATES}")


def _to_2d_array(x, h: int, w: int) -> np.ndarray:
    if x is None:
        raise ValueError("Image cell is None")
    if isinstance(x, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(x, dtype=np.float32)
        if arr.size != h * w:
            arr = np.frombuffer(x, dtype=np.float64).astype(np.float32)
        if arr.size != h * w:
            raise ValueError(f"Bytes image has {arr.size} elements, expected {h*w}")
        return arr.reshape(h, w)
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape == (h, w):
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 1 and arr.size == h * w:
        return arr.astype(np.float32, copy=False).reshape(h, w)
    if arr.ndim == 1 and arr.size == h:
        try:
            arr2 = np.stack([np.asarray(r, dtype=np.float32) for r in arr], axis=0)
            if arr2.shape == (h, w):
                return arr2
        except Exception:
            pass
    raise ValueError(f"Unsupported image cell shape {arr.shape}")


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


def _pair_shards(stamps_root: str, metrics_root: str) -> List[Tuple[str, str]]:
    stamps = _list_parquet_files(stamps_root)
    metrics = _list_parquet_files(metrics_root)

    def rel(p: str, root: str) -> str:
        rp = p[5:] if p.startswith("s3://") else p
        rr = root[5:] if root.startswith("s3://") else root
        rr = rr.rstrip("/") + "/"
        if rp.startswith(rr):
            return rp[len(rr):]
        return os.path.basename(rp)

    m_map = {rel(p, metrics_root): p for p in metrics}
    pairs: List[Tuple[str, str]] = []
    for s in stamps:
        r = rel(s, stamps_root)
        m = m_map.get(r) or m_map.get(os.path.basename(r))
        if m:
            pairs.append((s, m))
    if not pairs:
        raise RuntimeError("Failed to pair stamps and metrics shards. Check inputs.")
    return pairs


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


def _read_table(pf: pq.ParquetFile, row_group: int, columns: List[str]) -> pa.Table:
    return pf.read_row_group(row_group, columns=columns)


def _write_parquet(out_path: str, table: pa.Table):
    if _is_s3(out_path):
        # write to a single file on S3; for large runs prefer partitioned output
        fs = _fs_for(out_path)
        with fs.open(out_path, "wb") as f:
            pq.write_table(table, f, compression="zstd")
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pq.write_table(table, out_path, compression="zstd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamps", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--contract_json", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    ap.add_argument("--out", required=True, help="Output directory (dataset) or a .parquet file path")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_row_groups", type=int, default=0, help="Dev mode cap; 0 means no cap")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    required_cols = _load_contract_cols(args.contract_json)

    # Load model
    model = build_model(args.arch).to(device)
    ckpt_path = args.checkpoint
    if ckpt_path.startswith("sandbox:"):
        ckpt_path = ckpt_path.replace("sandbox:", "")
    if _is_s3(ckpt_path):
        with fsspec.open(ckpt_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Handle different checkpoint key formats (state_dict, model, model_state_dict)
    state_dict = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("model_state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint missing model weights. Keys found: {list(ckpt.keys())}")
    model.load_state_dict(state_dict)
    model.eval()

    pairs = _pair_shards(args.stamps, args.metrics)

    out_root = args.out.rstrip("/")
    write_as_dataset = not out_root.endswith(".parquet")

    h, w = 64, 64
    total_rows = 0

    for shard_idx, (stamp_file, metric_file) in enumerate(pairs):
        pf_s = _open_parquet(stamp_file)
        pf_m = _open_parquet(metric_file)

        schema_s = pf_s.schema.names
        schema_m = pf_m.schema.names

        cg, cr, cz = _detect_image_cols(schema_s)

        missing = [c for c in required_cols if c not in schema_m]
        if missing:
            raise RuntimeError(f"Metrics shard missing required columns: {missing}\nFile: {metric_file}")

        # columns to read from metrics, including split filter and label
        m_cols = list(set(required_cols + ["lens_model", "region_split"]))
        s_cols = [cg, cr, cz] + [c for c in KEY_COLS_DEFAULT if c in schema_s]

        n_row_groups = pf_m.num_row_groups
        if args.max_row_groups and args.max_row_groups > 0:
            n_row_groups = min(n_row_groups, args.max_row_groups)

        for rg in range(n_row_groups):
            tm = _read_table(pf_m, rg, m_cols)
            if args.split != "all" and "region_split" in tm.column_names:
                split_arr = tm["region_split"].to_pylist()
                keep = [i for i, s in enumerate(split_arr) if s == args.split]
            else:
                keep = list(range(tm.num_rows))
            if not keep:
                continue

            ts = _read_table(pf_s, rg, s_cols)

            # Verify key alignment if possible
            verify_keys = all(k in ts.column_names for k in KEY_COLS_DEFAULT)
            if verify_keys:
                km = {k: tm[k].to_pylist() for k in KEY_COLS_DEFAULT}
                ks = {k: ts[k].to_pylist() for k in KEY_COLS_DEFAULT}
                for i in keep[:5]:
                    for k in KEY_COLS_DEFAULT:
                        if km[k][i] != ks[k][i]:
                            raise RuntimeError("Key mismatch between stamps and metrics; row alignment violated.")

            # Build inference batches
            # y_true is computed directly for kept rows to avoid alignment bugs
            meta_rows: Dict[str, List] = {c: tm[c].to_pylist() for c in required_cols if c in tm.column_names}
            lens_model = tm["lens_model"].to_pylist()
            # y_true is computed directly for kept rows to avoid alignment bugs
            label_keep = [0 if lens_model[i] == "CONTROL" else 1 for i in keep]

            logit_out = [float('nan')] * len(keep)
            score_out = [float('nan')] * len(keep)
            
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
                try:
                    ig = _to_2d_array(ts[cg][i].as_py() if hasattr(ts[cg][i], 'as_py') else ts[cg][i], h, w)
                    ir = _to_2d_array(ts[cr][i].as_py() if hasattr(ts[cr][i], 'as_py') else ts[cr][i], h, w)
                    iz = _to_2d_array(ts[cz][i].as_py() if hasattr(ts[cz][i], 'as_py') else ts[cz][i], h, w)
                    x = np.stack([_robust_normalize(ig), _robust_normalize(ir), _robust_normalize(iz)], axis=0)
                    batch_x.append(x)
                    batch_pos.append(j)
                    if len(batch_x) >= args.batch_size:
                        flush_batch(batch_x, batch_pos)
                        batch_x, batch_pos = [], []
                except Exception:
                    # keep NaNs in outputs for this row
                    continue
            flush_batch(batch_x, batch_pos)

            # Build output table for kept rows
            out_cols = {}

            # Required contract cols
            for c in required_cols:
                if c in tm.column_names:
                    vals = meta_rows[c]
                    out_cols[c] = pa.array([vals[i] for i in keep])
                else:
                    out_cols[c] = pa.nulls(len(keep))

            out_cols["lens_model"] = pa.array([lens_model[i] for i in keep])
            if "region_split" in tm.column_names:
                out_cols["region_split"] = pa.array([tm["region_split"].to_pylist()[i] for i in keep])
            out_cols["y_true"] = pa.array(label_keep, type=pa.int8())
            out_cols["model_logit"] = pa.array(logit_out, type=pa.float32())
            out_cols["model_score"] = pa.array(score_out, type=pa.float32())

            table = pa.table(out_cols)

            # Write one file per shard-rowgroup to avoid huge single file
            if write_as_dataset:
                part = f"part-shard{shard_idx:05d}-rg{rg:03d}.parquet"
                out_path = out_root + "/" + part
            else:
                # If user requested a single file, append shard/rg suffix to avoid overwriting
                root_dir = os.path.dirname(out_root)
                base = os.path.basename(out_root).replace(".parquet", "")
                out_path = os.path.join(root_dir, f"{base}-shard{shard_idx:05d}-rg{rg:03d}.parquet")

            _write_parquet(out_path, table)
            total_rows += table.num_rows

        if shard_idx % 50 == 0:
            print(f"[INFO] processed shard={shard_idx}/{len(pairs)} total_rows_written={total_rows}")

    print(f"[DONE] total_rows_written={total_rows} out={args.out}")


if __name__ == "__main__":
    main()
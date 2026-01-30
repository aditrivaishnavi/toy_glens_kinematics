
#!/usr/bin/env python3
"""
Phase 5 inference (score generation) for unified Phase 4c Parquet dataset.

- Reads Parquet rows containing `stamp_npz` and metadata.
- Loads a checkpoint produced by phase5_train_fullscale_gh200_v2.py (ckpt_best.pt or ckpt_last.pt).
- Writes a score table as Parquet, suitable for:
    - phase5_eval_stratified_fpr.py (paper-style metrics)
    - hard-negative mining (phase5_mine_hard_negatives.py)
    - Spark aggregation / completeness maps

This script is intentionally conservative:
- Applies the SAME robust per-stamp median/MAD normalization used in training.
- No augmentation.
- Supports metadata fusion if model was trained with --meta_cols.
"""

import argparse
import io
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.fs as pafs

try:
    from torchvision.models import resnet18, convnext_tiny, efficientnet_b0
except ImportError as e:
    raise RuntimeError("torchvision is required (pip install torchvision)") from e

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def decode_stamp_npz(npz_bytes: bytes) -> np.ndarray:
    bio = io.BytesIO(npz_bytes)
    with np.load(bio) as npz:
        g = npz["image_g"].astype(np.float32)
        r = npz["image_r"].astype(np.float32)
        z = npz["image_z"].astype(np.float32)
    return np.stack([g, r, z], axis=0)


def robust_mad_norm(x: np.ndarray, clip: float = 10.0, eps: float = 1e-6) -> np.ndarray:
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


class MetaFusionHead(nn.Module):
    """Metadata fusion head - must match training script exactly."""
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


def build_model_wrapped(arch: str, meta_dim: int = 0, dropout: float = 0.1) -> nn.Module:
    """Build model with backbone wrapper - for newer checkpoints."""
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


def build_model_simple(arch: str) -> nn.Module:
    """Build simple model without wrapper - for older checkpoints (e.g., train_lambda.py)."""
    arch = arch.lower()
    if arch == "resnet18":
        m = resnet18(weights=None)
        # Modify conv1 for 64x64 input (smaller kernel)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        # Binary classification output
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m
    else:
        raise ValueError(f"Simple model not supported for arch: {arch}")


def build_model(arch: str, meta_dim: int = 0, dropout: float = 0.1, state_dict_keys: Optional[set] = None):
    """
    Build model, auto-detecting wrapper vs simple based on state dict keys.
    Returns (model, is_simple_model) tuple.
    """
    # Check if state dict has 'backbone.' prefix
    if state_dict_keys:
        has_backbone_prefix = any(k.startswith("backbone.") for k in state_dict_keys)
        has_fc = any(k.startswith("fc.") for k in state_dict_keys)
        
        if has_fc and not has_backbone_prefix:
            # Simple model (train_lambda.py style)
            logger.info("Detected simple model structure (no backbone wrapper)")
            return build_model_simple(arch), True
    
    # Default: wrapped model
    return build_model_wrapped(arch, meta_dim=meta_dim, dropout=dropout), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Input unified parquet path (local or s3://...)")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path (.pt)")
    ap.add_argument("--out", required=True, help="Output score dataset directory")
    ap.add_argument("--arch", default="", help="Override arch (otherwise uses ckpt args)")
    ap.add_argument("--meta_cols", default="", help="Override meta_cols (otherwise uses ckpt args)")
    ap.add_argument("--split", default="", help="If set, filter region_split == split (train/val/test/real)")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--mad_clip", type=float, default=10.0)
    ap.add_argument("--use_bf16", action="store_true")
    ap.add_argument("--use_fp16", action="store_true")
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--write_batch_rows", type=int, default=200_000)
    args = ap.parse_args()

    # FS selection
    fs = None
    if args.data.startswith("s3://") or args.out.startswith("s3://") or args.ckpt.startswith("s3://"):
        fs = pafs.S3FileSystem(region=os.environ.get("AWS_REGION"))

    # Load ckpt
    # If ckpt is on S3, download to local temp via pyarrow fs.
    ckpt_path = args.ckpt
    if ckpt_path.startswith("s3://"):
        # read bytes to buffer and torch.load from BytesIO
        with fs.open_input_file(ckpt_path) as f:
            buf = f.read()
        ckpt = torch.load(io.BytesIO(buf), map_location="cpu", weights_only=False)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Support multiple checkpoint formats: args, cfg, or config
    ckpt_args = ckpt.get("args") or ckpt.get("cfg") or ckpt.get("config") or {}
    
    # Get arch from checkpoint - try multiple keys
    arch = args.arch or ckpt_args.get("arch") or ckpt_args.get("model") or "convnext_tiny"
    
    # Get meta_cols from checkpoint if not overridden
    if args.meta_cols:
        meta_cols = [c.strip() for c in args.meta_cols.split(",") if c.strip()]
    else:
        # Try multiple formats for meta_cols
        meta_cols_val = ckpt_args.get("meta_cols", "")
        if isinstance(meta_cols_val, list):
            meta_cols = meta_cols_val
        elif isinstance(meta_cols_val, str) and meta_cols_val:
            meta_cols = [c.strip() for c in meta_cols_val.split(",") if c.strip()]
        else:
            meta_cols = []
    
    # Check if model was trained with metadata
    use_meta = ckpt_args.get("use_metadata", False) or ckpt_args.get("use_meta", False)
    if not use_meta:
        meta_cols = []  # Model was trained without metadata fusion
    
    # Support multiple checkpoint state dict keys
    state_dict = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state_dict is None:
        raise ValueError(f"Could not find model state dict in checkpoint. Keys: {list(ckpt.keys())}")
    
    logger.info(f"Using arch={arch}, meta_cols={meta_cols}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, is_simple_model = build_model(arch, meta_dim=len(meta_cols), state_dict_keys=set(state_dict.keys()))
    model = model.to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # Simple models don't support metadata - force empty
    if is_simple_model:
        meta_cols = []
        logger.info("Simple model detected - disabling metadata fusion")

    use_amp = args.use_bf16 or args.use_fp16
    amp_dtype = torch.bfloat16 if args.use_bf16 else torch.float16

    # Read dataset with Hive partitioning support
    dataset = ds.dataset(args.data, format="parquet", filesystem=fs, partitioning="hive")
    filt = None
    if args.split:
        if "region_split" not in dataset.schema.names:
            raise ValueError("split filtering requested but region_split not present")
        filt = (ds.field("region_split") == args.split)

    # Columns to carry through. Keep as many as possible but avoid huge columns.
    carry = []
    for c in ["injection_id","ra","dec","brickname","objid","region_id","region_split","is_control","theta_e_arcsec","psf_fwhm_used_r","psfsize_r","arc_snr","cutout_ok","lens_model","grid_name","control_kind"]:
        if c in dataset.schema.names:
            carry.append(c)
    
    # Validate required columns
    required = ["stamp_npz", "is_control"]
    missing = [c for c in required if c not in dataset.schema.names]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    
    # Validate meta columns if using metadata fusion
    for c in meta_cols:
        if c not in dataset.schema.names:
            raise ValueError(f"Meta column '{c}' not found in dataset")
    
    cols = carry + ["stamp_npz"] + [c for c in meta_cols if c not in carry]

    if not args.out.startswith("s3://"):
        os.makedirs(args.out, exist_ok=True)

    total = 0
    skip_count = 0
    out_tables = []

    def flush(tables: List[pa.Table]):
        if not tables:
            return
        tab = pa.concat_tables(tables, promote=True)
        pq.write_to_dataset(tab, root_path=args.out, compression="zstd", filesystem=fs)
        tables.clear()

    for batch in dataset.to_batches(columns=cols, batch_size=args.write_batch_rows, filter=filt):
        if args.max_rows and total >= args.max_rows:
            break
        if args.max_rows:
            batch = batch.slice(0, min(len(batch), args.max_rows - total))
        n = len(batch)
        total += n

        # Decode + normalize in a loop (NPZ is per-row)
        xs = []
        metas = []
        valid_idx = []
        for i in range(n):
            ok = True
            if "cutout_ok" in batch.schema.names:
                v = batch["cutout_ok"][i].as_py()
                ok = (v is not None and int(v) == 1)
            if not ok:
                continue
            
            # Narrow exception scope to specific expected errors
            try:
                x = decode_stamp_npz(batch["stamp_npz"][i].as_py())
                x = robust_mad_norm(x, clip=args.mad_clip)
            except (ValueError, KeyError, IOError) as e:
                skip_count += 1
                if skip_count <= 10:
                    logger.warning(f"Skipping row due to decode error: {e}")
                continue
            
            xs.append(x)
            valid_idx.append(i)
            
            # Collect metadata if needed
            if meta_cols:
                meta_vals = []
                for c in meta_cols:
                    v = batch[c][i].as_py()
                    meta_vals.append(float(v) if v is not None else 0.0)
                metas.append(np.asarray(meta_vals, dtype=np.float32))

        if not xs:
            continue

        x = torch.from_numpy(np.stack(xs, axis=0)).to(device)
        meta_tensor = None
        if meta_cols:
            meta_tensor = torch.from_numpy(np.stack(metas, axis=0)).to(device)
        
        with torch.no_grad():
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    if is_simple_model:
                        logits = model(x)
                    else:
                        logits = model(x, meta_tensor)
            else:
                if is_simple_model:
                    logits = model(x)
                else:
                    logits = model(x, meta_tensor)
            # Handle output shape - squeeze if needed
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            # Convert to float32 before numpy (bf16 not supported)
            score = torch.sigmoid(logits).float().detach().cpu().numpy().astype(np.float32)

        # Build output batch using valid_idx rows
        out = {}
        for c in carry:
            arr = batch[c].take(pa.array(valid_idx))
            out[c] = arr
        out["score"] = pa.array(score)
        out_tables.append(pa.table(out))

        if len(out_tables) >= 4:
            flush(out_tables)

    flush(out_tables)
    
    if skip_count > 0:
        skip_ratio = skip_count / total if total > 0 else 0
        logger.info(f"Skipped {skip_count}/{total} rows ({skip_ratio:.1%}) due to decode errors")
        if skip_ratio > 0.10:
            logger.error(f"High skip rate ({skip_ratio:.1%}) - check data integrity!")
    
    logger.info(f"Wrote scores to {args.out}. Rows processed: {total}")


if __name__ == "__main__":
    main()

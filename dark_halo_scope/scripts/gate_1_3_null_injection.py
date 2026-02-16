#!/usr/bin/env python3
"""
Gate 1.3: Null-injection test.
Verifies model correctly identifies controls as non-lenses.
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
import pyarrow.dataset as ds
import io
import json
from datetime import datetime, timezone

RESULTS = {"gate": "1.3", "timestamp": datetime.now(timezone.utc).isoformat()}

MODEL_PATH = "/lambda/nfs/darkhaloscope-training-dc/models/gen5_cosmos/ckpt_best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MetaFusionHead(nn.Module):
    def __init__(self, feat_dim, meta_dim, hidden=256, dropout=0.1):
        super().__init__()
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
    def forward(self, feats, meta):
        m = self.meta_mlp(meta)
        x = torch.cat([feats, m], dim=1)
        return self.classifier(x).squeeze(1)

m = convnext_tiny(weights=None)
feat_dim = m.classifier[2].in_features
m.classifier = nn.Identity()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = m
        self.head = MetaFusionHead(feat_dim, 2, hidden=256, dropout=0.1)
    def forward(self, x, meta=None):
        feats = self.backbone(x)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        if meta is None:
            meta = torch.zeros(feats.shape[0], 2, device=feats.device)
        return self.head(feats, meta)

model = Model()
ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model"])
model = model.to(device)
model.eval()

RESULTS["model_epoch"] = int(ckpt["epoch"])

def robust_mad_norm_outer(x, clip=10.0, eps=1e-6, inner_frac=0.5):
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

def decode_stamp(blob):
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        g = z["image_g"].astype(np.float32)
        r = z["image_r"].astype(np.float32)
        zb = z["image_z"].astype(np.float32)
    return np.stack([g, r, zb], axis=0)

data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")
filt = (ds.field("region_split") == "test") & (ds.field("cutout_ok") == 1) & (ds.field("is_control") == 1)
table = dataset.to_table(filter=filt, columns=["stamp_npz", "psfsize_r", "psfdepth_r"])

n_samples = min(1000, table.num_rows)
predictions = []
for i in range(n_samples):
    blob = table["stamp_npz"][i].as_py()
    if blob is None:
        continue
    try:
        img = decode_stamp(blob)
        img_norm = robust_mad_norm_outer(img)
        if not np.isfinite(img_norm).all():
            continue
        psfsize = table["psfsize_r"][i].as_py() or 1.0
        psfdepth = table["psfdepth_r"][i].as_py() or 0.0
        x = torch.tensor(img_norm[np.newaxis], dtype=torch.float32).to(device)
        meta = torch.tensor([[psfsize, psfdepth]], dtype=torch.float32).to(device)
        with torch.no_grad():
            logit = model(x, meta)
            prob = torch.sigmoid(logit).item()
        predictions.append(prob)
    except:
        continue

predictions = np.array(predictions)
RESULTS["n_evaluated"] = int(len(predictions))
RESULTS["mean_p_lens"] = float(predictions.mean())
RESULTS["std_p_lens"] = float(predictions.std())
RESULTS["frac_gt_0.5"] = float((predictions > 0.5).mean())
RESULTS["frac_gt_0.9"] = float((predictions > 0.9).mean())
RESULTS["overall_passed"] = bool(predictions.mean() < 0.2 and (predictions > 0.5).mean() < 0.1)

with open("gate_1_3_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print(json.dumps(RESULTS, indent=2))
print(f"\nGATE 1.3: {'PASS' if RESULTS['overall_passed'] else 'FAIL'}")

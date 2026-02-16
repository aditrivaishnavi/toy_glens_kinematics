#!/usr/bin/env python3
"""
Phase 2 Extended: Center-Masked Diagnostic with Controls Comparison

Extends the original Phase 2 diagnostic to:
1. Run masking on BOTH positives and controls
2. Report per-sample delta distributions
3. Compare class-specific behavior

Per LLM recommendation: "Run the same masking on controls and report the 
mean shift in p for controls. If masking reduces positives a lot but barely 
affects controls, the center is being used as a positive cue."
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
import pyarrow.dataset as ds
import io
import json
from datetime import datetime, timezone

RESULTS = {
    "phase": "2_extended",
    "name": "Center-Masked Diagnostic with Controls",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "description": "Tests masking effect on both positives and controls"
}

print("=" * 70)
print("PHASE 2 EXTENDED: CENTER-MASKED WITH CONTROLS")
print("=" * 70)

# Configuration
MASK_RADIUS = 10  # pixels (2.62 arcsec at 0.262"/pix)
N_SAMPLES = 500  # per class

# Load model
MODEL_PATH = "/lambda/nfs/darkhaloscope-training-dc/models/gen5_cosmos/ckpt_best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
print(f"Loaded model from epoch {ckpt['epoch']}")

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

def mask_center_outer_fill(img, r_mask=MASK_RADIUS, seed=None):
    """Mask center with pixels resampled from outer annulus."""
    if seed is not None:
        np.random.seed(seed)
    
    C, H, W = img.shape
    out = img.copy()
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r2 = (yy - cy)**2 + (xx - cx)**2
    
    center_mask = r2 < r_mask**2
    outer_mask = r2 >= (r_mask * 2)**2
    
    for c in range(C):
        outer_vals = img[c][outer_mask]
        if len(outer_vals) == 0:
            continue
        n_center = center_mask.sum()
        fill_vals = np.random.choice(outer_vals, size=n_center, replace=True)
        out[c][center_mask] = fill_vals
    
    return out

def decode_stamp(blob):
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        g = z["image_g"].astype(np.float32)
        r = z["image_r"].astype(np.float32)
        zb = z["image_z"].astype(np.float32)
    return np.stack([g, r, zb], axis=0)

# Load data
data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

# Load both positives and controls from test set
filt_pos = (ds.field("region_split") == "test") & (ds.field("cutout_ok") == 1) & (ds.field("is_control") == 0)
filt_ctrl = (ds.field("region_split") == "test") & (ds.field("cutout_ok") == 1) & (ds.field("is_control") == 1)

table_pos = dataset.to_table(filter=filt_pos, columns=["stamp_npz", "psfsize_r", "psfdepth_r"])
table_ctrl = dataset.to_table(filter=filt_ctrl, columns=["stamp_npz", "psfsize_r", "psfdepth_r"])

print(f"Positives: {table_pos.num_rows}, Controls: {table_ctrl.num_rows}")

# Sample
np.random.seed(42)
pos_indices = np.random.choice(table_pos.num_rows, min(N_SAMPLES, table_pos.num_rows), replace=False)
ctrl_indices = np.random.choice(table_ctrl.num_rows, min(N_SAMPLES, table_ctrl.num_rows), replace=False)

def evaluate_class(table, indices, class_name):
    """Evaluate original and masked predictions for a class."""
    original_probs = []
    masked_probs = []
    
    for i, idx in enumerate(indices):
        if i % 100 == 0:
            print(f"  {class_name}: {i}/{len(indices)}")
        
        blob = table["stamp_npz"][int(idx)].as_py()
        if blob is None:
            continue
        
        try:
            img = decode_stamp(blob)
            if not np.isfinite(img).all():
                continue
            
            psfsize = table["psfsize_r"][int(idx)].as_py() or 1.0
            psfdepth = table["psfdepth_r"][int(idx)].as_py() or 0.0
            
            # Original
            img_norm = robust_mad_norm_outer(img)
            if not np.isfinite(img_norm).all():
                continue
            
            x = torch.tensor(img_norm[np.newaxis], dtype=torch.float32).to(device)
            meta = torch.tensor([[psfsize, psfdepth]], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                logit = model(x, meta)
                prob_orig = torch.sigmoid(logit).item()
            
            # Masked
            img_masked = mask_center_outer_fill(img, r_mask=MASK_RADIUS, seed=i)
            img_masked_norm = robust_mad_norm_outer(img_masked)
            if not np.isfinite(img_masked_norm).all():
                continue
            
            x_masked = torch.tensor(img_masked_norm[np.newaxis], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                logit_masked = model(x_masked, meta)
                prob_masked = torch.sigmoid(logit_masked).item()
            
            original_probs.append(prob_orig)
            masked_probs.append(prob_masked)
            
        except Exception as e:
            continue
    
    return np.array(original_probs), np.array(masked_probs)

print("\nEvaluating positives...")
pos_orig, pos_masked = evaluate_class(table_pos, pos_indices, "Positives")

print("\nEvaluating controls...")
ctrl_orig, ctrl_masked = evaluate_class(table_ctrl, ctrl_indices, "Controls")

# Compute per-sample deltas
pos_delta = pos_orig - pos_masked  # Positive delta means masking reduced confidence
ctrl_delta = ctrl_orig - ctrl_masked

print("\n" + "=" * 70)
print("RESULTS:")
print("=" * 70)

print(f"\nPOSITIVES (n={len(pos_orig)}):")
print(f"  Original: mean p = {pos_orig.mean():.4f}")
print(f"  Masked:   mean p = {pos_masked.mean():.4f}")
print(f"  Mean drop: {pos_delta.mean():.4f} ({pos_delta.mean()/pos_orig.mean()*100:.1f}%)")
print(f"  Median drop: {np.median(pos_delta):.4f}")
print(f"  Drop > 0.3: {(pos_delta > 0.3).mean():.1%}")
print(f"  Drop > 0.5: {(pos_delta > 0.5).mean():.1%}")

print(f"\nCONTROLS (n={len(ctrl_orig)}):")
print(f"  Original: mean p = {ctrl_orig.mean():.4f}")
print(f"  Masked:   mean p = {ctrl_masked.mean():.4f}")
print(f"  Mean drop: {ctrl_delta.mean():.4f}")
print(f"  Median drop: {np.median(ctrl_delta):.4f}")

print(f"\nCOMPARISON:")
pos_drop_pct = pos_delta.mean() / pos_orig.mean() * 100 if pos_orig.mean() > 0 else 0
ctrl_drop_pct = ctrl_delta.mean() / ctrl_orig.mean() * 100 if ctrl_orig.mean() > 0 else 0
print(f"  Positives drop: {pos_drop_pct:.1f}%")
print(f"  Controls drop:  {ctrl_drop_pct:.1f}%")
print(f"  Differential:   {pos_drop_pct - ctrl_drop_pct:.1f} percentage points")

# Store results
RESULTS["mask_radius_pixels"] = MASK_RADIUS
RESULTS["mask_radius_arcsec"] = MASK_RADIUS * 0.262

RESULTS["positives"] = {
    "n": int(len(pos_orig)),
    "original_mean": float(pos_orig.mean()),
    "masked_mean": float(pos_masked.mean()),
    "mean_drop": float(pos_delta.mean()),
    "median_drop": float(np.median(pos_delta)),
    "drop_percent": float(pos_drop_pct),
    "frac_drop_gt_0.3": float((pos_delta > 0.3).mean()),
    "frac_drop_gt_0.5": float((pos_delta > 0.5).mean()),
    "delta_percentiles": {
        "p10": float(np.percentile(pos_delta, 10)),
        "p25": float(np.percentile(pos_delta, 25)),
        "p50": float(np.percentile(pos_delta, 50)),
        "p75": float(np.percentile(pos_delta, 75)),
        "p90": float(np.percentile(pos_delta, 90))
    }
}

RESULTS["controls"] = {
    "n": int(len(ctrl_orig)),
    "original_mean": float(ctrl_orig.mean()),
    "masked_mean": float(ctrl_masked.mean()),
    "mean_drop": float(ctrl_delta.mean()),
    "median_drop": float(np.median(ctrl_delta)),
    "delta_percentiles": {
        "p10": float(np.percentile(ctrl_delta, 10)),
        "p25": float(np.percentile(ctrl_delta, 25)),
        "p50": float(np.percentile(ctrl_delta, 50)),
        "p75": float(np.percentile(ctrl_delta, 75)),
        "p90": float(np.percentile(ctrl_delta, 90))
    }
}

RESULTS["differential_drop_pct"] = float(pos_drop_pct - ctrl_drop_pct)

# Interpretation
print("\n" + "=" * 70)
print("INTERPRETATION:")
print("=" * 70)

if pos_drop_pct > 15 and ctrl_drop_pct < 5:
    interpretation = "CENTER IS A POSITIVE CUE"
    explanation = "Masking reduces positive confidence significantly but barely affects controls. The model uses center features to identify lenses."
elif pos_drop_pct > 15 and ctrl_drop_pct > 15:
    interpretation = "CENTER AFFECTS BOTH CLASSES SIMILARLY"
    explanation = "Masking affects both classes. Center information is used for general prediction, not specifically for lens detection."
else:
    interpretation = "CENTER NOT A STRONG CUE"
    explanation = "Masking has limited effect on predictions. Model relies more on arc/outer features."

print(f"Interpretation: {interpretation}")
print(f"Explanation: {explanation}")

RESULTS["interpretation"] = interpretation
RESULTS["explanation"] = explanation

# Save results
output_path = "/lambda/nfs/darkhaloscope-training-dc/phase2_extended_results.json"
with open(output_path, "w") as f:
    json.dump(RESULTS, f, indent=2)

print(f"\nResults saved to {output_path}")

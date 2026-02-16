#!/usr/bin/env python3
"""
Deep Investigation: What's the model actually seeing?
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from astropy.io import fits
import io
import pyarrow.dataset as ds
from torchvision.models import convnext_tiny

print("=" * 70)
print("DEEP INVESTIGATION: WHAT DOES THE MODEL SEE?")
print("=" * 70)

# Load model
MODEL_PATH = "/lambda/nfs/darkhaloscope-training-dc/models/gen5_cosmos/ckpt_best.pt"
device = torch.device("cuda")

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
backbone = m

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = backbone
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
print(f"Model loaded (epoch {ckpt['epoch']}, best_metric={ckpt['best_metric']:.4f})")

def robust_mad_norm_outer(x, clip=10.0, eps=1e-6, inner_frac=0.5):
    """Normalize using outer annulus - MUST MATCH TRAINING."""
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

# ============================================================
# Test 1: Check model predictions on training data
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: MODEL ON TRAINING DATA")
print("=" * 70)

data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

# Get positives and negatives
pos_filter = (ds.field("region_split") == "test") & (ds.field("is_control") == 0) & (ds.field("cutout_ok") == 1)
neg_filter = (ds.field("region_split") == "test") & (ds.field("is_control") == 1) & (ds.field("cutout_ok") == 1)

pos_table = dataset.to_table(filter=pos_filter, columns=["stamp_npz", "arc_snr"])
pos_table = pos_table.slice(0, 20)
neg_table = dataset.to_table(filter=neg_filter, columns=["stamp_npz"])
neg_table = neg_table.slice(0, 20)

def decode_training_stamp(blob):
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        g = z["image_g"].astype(np.float32)
        r = z["image_r"].astype(np.float32)
        zb = z["image_z"].astype(np.float32)
    return np.stack([g, r, zb], axis=0)

print("\nTraining POSITIVES (injected lenses):")
pos_probs = []
for i in range(min(10, pos_table.num_rows)):
    blob = pos_table["stamp_npz"][i].as_py()
    snr = pos_table["arc_snr"][i].as_py()
    img = decode_training_stamp(blob)
    img_norm = robust_mad_norm_outer(img)
    
    with torch.no_grad():
        x = torch.tensor(img_norm[None], dtype=torch.float32).to(device)
        logit = model(x)
        p = torch.sigmoid(logit).item()
    pos_probs.append(p)
    if i < 5:
        print(f"  Sample {i}: arc_snr={snr:.1f}, p_lens={p:.4f}")

print(f"\n  Mean p_lens for positives: {np.mean(pos_probs):.4f}")

print("\nTraining NEGATIVES (controls):")
neg_probs = []
for i in range(min(10, neg_table.num_rows)):
    blob = neg_table["stamp_npz"][i].as_py()
    img = decode_training_stamp(blob)
    img_norm = robust_mad_norm_outer(img)
    
    with torch.no_grad():
        x = torch.tensor(img_norm[None], dtype=torch.float32).to(device)
        logit = model(x)
        p = torch.sigmoid(logit).item()
    neg_probs.append(p)
    if i < 5:
        print(f"  Sample {i}: p_lens={p:.4f}")

print(f"\n  Mean p_lens for negatives: {np.mean(neg_probs):.4f}")

# ============================================================
# Test 2: Check model on anchor data with proper normalization
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: ANCHOR DATA - BEFORE AND AFTER NORMALIZATION")
print("=" * 70)

anchor_dir = Path("/lambda/nfs/darkhaloscope-training-dc/anchor_cutouts/known_lenses")

# Test a few anchors
test_anchors = ["SDSSJ1205+4910", "SDSSJ2300+0022", "BELLSJ0847+2348", "SDSSJ2321-0939"]

for name in test_anchors:
    fits_file = anchor_dir / f"{name}.fits"
    if fits_file.exists():
        with fits.open(fits_file) as hdu:
            img = hdu[0].data.astype(np.float32)
        
        # Before normalization
        print(f"\n{name}:")
        print(f"  Before norm: r-band range=[{img[1].min():.4f}, {img[1].max():.4f}]")
        
        # After normalization
        img_norm = robust_mad_norm_outer(img)
        print(f"  After norm:  r-band range=[{img_norm[1].min():.4f}, {img_norm[1].max():.4f}]")
        
        # Run inference
        with torch.no_grad():
            x = torch.tensor(img_norm[None], dtype=torch.float32).to(device)
            logit = model(x)
            p = torch.sigmoid(logit).item()
        print(f"  p_lens = {p:.4f}")

# ============================================================
# Test 3: Compare normalized images
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: COMPARE NORMALIZED IMAGES (Training vs Anchor)")
print("=" * 70)

# Get one training positive
blob = pos_table["stamp_npz"][0].as_py()
train_img = decode_training_stamp(blob)
train_norm = robust_mad_norm_outer(train_img)

# Get one anchor
with fits.open(anchor_dir / "SDSSJ1205+4910.fits") as hdu:
    anchor_img = hdu[0].data.astype(np.float32)
anchor_norm = robust_mad_norm_outer(anchor_img)

print("\nTraining positive (after normalization):")
print(f"  r-band: min={train_norm[1].min():.2f}, max={train_norm[1].max():.2f}, mean={train_norm[1].mean():.2f}, std={train_norm[1].std():.2f}")

print("\nAnchor SDSSJ1205+4910 (after normalization):")
print(f"  r-band: min={anchor_norm[1].min():.2f}, max={anchor_norm[1].max():.2f}, mean={anchor_norm[1].mean():.2f}, std={anchor_norm[1].std():.2f}")

print("\n=== CONCLUSION ===")
print("""
If the model is working correctly on training data but not on anchors:
1. The normalization might not match exactly
2. The anchor images might look fundamentally different
3. The arcs in anchors might not be visible (below noise)

If the model gives random/low predictions on both training and anchor:
1. Something is wrong with the evaluation code
2. The model is not properly loaded
""")

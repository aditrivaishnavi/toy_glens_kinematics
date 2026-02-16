#!/usr/bin/env python3
"""
Check if model is overfitting to training artifacts.
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
print("OVERFITTING CHECK")
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

def decode_training_stamp(blob):
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        g = z["image_g"].astype(np.float32)
        r = z["image_r"].astype(np.float32)
        zb = z["image_z"].astype(np.float32)
    return np.stack([g, r, zb], axis=0)

# ============================================================
# Test 1: What if we swap the image content?
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: SWAP EXPERIMENT")
print("=" * 70)

# Get a training positive and negative
data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

pos_filter = (ds.field("region_split") == "test") & (ds.field("is_control") == 0) & (ds.field("cutout_ok") == 1)
neg_filter = (ds.field("region_split") == "test") & (ds.field("is_control") == 1) & (ds.field("cutout_ok") == 1)

pos_table = dataset.to_table(filter=pos_filter, columns=["stamp_npz"])
neg_table = dataset.to_table(filter=neg_filter, columns=["stamp_npz"])

pos_img = decode_training_stamp(pos_table["stamp_npz"][0].as_py())
neg_img = decode_training_stamp(neg_table["stamp_npz"][0].as_py())

pos_norm = robust_mad_norm_outer(pos_img)
neg_norm = robust_mad_norm_outer(neg_img)

with torch.no_grad():
    pos_x = torch.tensor(pos_norm[None], dtype=torch.float32).to(device)
    neg_x = torch.tensor(neg_norm[None], dtype=torch.float32).to(device)
    p_pos = torch.sigmoid(model(pos_x)).item()
    p_neg = torch.sigmoid(model(neg_x)).item()

print(f"Original positive (lens injection): p_lens = {p_pos:.4f}")
print(f"Original negative (control):        p_lens = {p_neg:.4f}")

# Now create a "fake" positive by adding random noise
fake_pos = neg_img.copy()
# Add some Gaussian noise to simulate arc (but wrong morphology)
cy, cx = 32, 32
for c in range(3):
    # Add a ring-like pattern
    y, x = np.ogrid[:64, :64]
    r = np.sqrt((y - cy)**2 + (x - cx)**2)
    ring = np.exp(-((r - 15)**2) / 20) * 0.01  # Faint ring at radius 15
    fake_pos[c] += ring

fake_norm = robust_mad_norm_outer(fake_pos)
with torch.no_grad():
    fake_x = torch.tensor(fake_norm[None], dtype=torch.float32).to(device)
    p_fake = torch.sigmoid(model(fake_x)).item()

print(f"Fake positive (control + synthetic ring): p_lens = {p_fake:.4f}")

# ============================================================
# Test 2: Check if model responds to specific patterns
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: WHAT FEATURES DOES MODEL RESPOND TO?")
print("=" * 70)

# Get an anchor
anchor_dir = Path("/lambda/nfs/darkhaloscope-training-dc/anchor_cutouts/known_lenses")
with fits.open(anchor_dir / "SDSSJ2321-0939.fits") as hdu:
    low_anchor = hdu[0].data.astype(np.float32)

low_norm = robust_mad_norm_outer(low_anchor)
with torch.no_grad():
    x = torch.tensor(low_norm[None], dtype=torch.float32).to(device)
    p_orig = torch.sigmoid(model(x)).item()

print(f"Low-scoring anchor (SDSSJ2321-0939) original: p_lens = {p_orig:.4f}")

# Add the same fake ring
low_with_ring = low_anchor.copy()
for c in range(3):
    y, x = np.ogrid[:64, :64]
    r = np.sqrt((y - 32)**2 + (x - 32)**2)
    ring = np.exp(-((r - 15)**2) / 20) * 0.01
    low_with_ring[c] += ring

low_ring_norm = robust_mad_norm_outer(low_with_ring)
with torch.no_grad():
    x = torch.tensor(low_ring_norm[None], dtype=torch.float32).to(device)
    p_ring = torch.sigmoid(model(x)).item()

print(f"Low anchor + synthetic ring: p_lens = {p_ring:.4f}")

# ============================================================
# Test 3: Check training positive vs negative pixel difference
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: POSITIVE vs NEGATIVE PIXEL ANALYSIS")
print("=" * 70)

# Get multiple positives and negatives
pos_imgs = []
neg_imgs = []
for i in range(min(20, pos_table.num_rows)):
    img = decode_training_stamp(pos_table["stamp_npz"][i].as_py())
    pos_imgs.append(robust_mad_norm_outer(img))
for i in range(min(20, neg_table.num_rows)):
    img = decode_training_stamp(neg_table["stamp_npz"][i].as_py())
    neg_imgs.append(robust_mad_norm_outer(img))

pos_stack = np.stack(pos_imgs)
neg_stack = np.stack(neg_imgs)

print("Average normalized image statistics:")
print(f"  POSITIVES: mean={pos_stack.mean():.4f}, std={pos_stack.std():.4f}")
print(f"  NEGATIVES: mean={neg_stack.mean():.4f}, std={neg_stack.std():.4f}")

# Check per-channel
for c, name in enumerate(["g", "r", "z"]):
    pos_mean = pos_stack[:, c].mean()
    neg_mean = neg_stack[:, c].mean()
    print(f"  {name}-band: pos_mean={pos_mean:.4f}, neg_mean={neg_mean:.4f}, diff={pos_mean-neg_mean:.4f}")

# Check center vs outer
center_mask = np.zeros((64, 64), dtype=bool)
center_mask[24:40, 24:40] = True

pos_center = np.mean([img[1, center_mask].mean() for img in pos_imgs])
neg_center = np.mean([img[1, center_mask].mean() for img in neg_imgs])
print(f"\n  Center region (r-band): pos={pos_center:.4f}, neg={neg_center:.4f}, diff={pos_center-neg_center:.4f}")

# Check annulus (where arcs typically are)
y, x = np.ogrid[:64, :64]
r = np.sqrt((y - 32)**2 + (x - 32)**2)
annulus_mask = (r >= 10) & (r <= 20)

pos_annulus = np.mean([img[1, annulus_mask].mean() for img in pos_imgs])
neg_annulus = np.mean([img[1, annulus_mask].mean() for img in neg_imgs])
print(f"  Annulus region (r-band): pos={pos_annulus:.4f}, neg={neg_annulus:.4f}, diff={pos_annulus-neg_annulus:.4f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

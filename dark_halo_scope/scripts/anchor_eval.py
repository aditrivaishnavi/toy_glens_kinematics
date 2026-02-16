#!/usr/bin/env python3
"""Evaluate Gen5 model on real SLACS/BELLS anchor lenses."""
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from astropy.io import fits
from torchvision.models import convnext_tiny

MODEL_PATH = "/lambda/nfs/darkhaloscope-training-dc/models/gen5_cosmos/ckpt_best.pt"
ANCHOR_PATH = "/lambda/nfs/darkhaloscope-training-dc/anchor_cutouts"

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
        # Flatten if needed (convnext returns (B, C, 1, 1) or (B, C))
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

def robust_mad_norm_outer(x: np.ndarray, clip: float = 10.0, eps: float = 1e-6,
                          inner_frac: float = 0.5) -> np.ndarray:
    """Normalize using outer annulus only - MATCHES TRAINING EXACTLY."""
    out = np.empty_like(x, dtype=np.float32)
    h, w = x.shape[-2:]
    cy, cx = h // 2, w // 2
    ri = int(min(h, w) * inner_frac / 2)
    
    # Create circular mask for outer region
    yy, xx = np.ogrid[:h, :w]
    outer_mask = ((yy - cy)**2 + (xx - cx)**2) > ri**2
    
    for c in range(x.shape[0]):
        v = x[c]
        outer_v = v[outer_mask]
        med = np.median(outer_v)
        mad = np.median(np.abs(outer_v - med))
        scale = 1.4826 * mad + eps  # MAD to std conversion factor
        vv = (v - med) / scale
        if clip is not None:
            vv = np.clip(vv, -clip, clip)
        out[c] = vv.astype(np.float32)
    return out

def load_fits_stamps(anchor_dir):
    stamps = []
    names = []
    for fits_file in sorted(Path(anchor_dir).glob("*.fits")):
        try:
            with fits.open(fits_file) as hdu:
                data = hdu[0].data.astype(np.float32)
                if data.ndim == 2:
                    img = np.stack([data, data, data], axis=0)
                elif data.ndim == 3:
                    if data.shape[0] == 3:
                        img = data
                    else:
                        img = np.stack([data[0], data[0], data[0]], axis=0)
                else:
                    continue
                if img.shape[1] != 64 or img.shape[2] != 64:
                    from scipy.ndimage import zoom
                    new_img = np.zeros((3, 64, 64), dtype=np.float32)
                    for c in range(3):
                        factor_y = 64 / img.shape[1]
                        factor_x = 64 / img.shape[2]
                        new_img[c] = zoom(img[c], (factor_y, factor_x), order=1)
                    img = new_img
                img = robust_mad_norm_outer(img)
                if np.isfinite(img).all():
                    stamps.append(img)
                    names.append(fits_file.stem)
        except Exception as e:
            print(f"  Error loading {fits_file.name}: {e}")
    return stamps, names

print()
print("=" * 60)
print("ANCHOR LENS EVALUATION (SLACS/BELLS)")
print("=" * 60)

known_lens_dir = Path(ANCHOR_PATH) / "known_lenses"
hard_neg_dir = Path(ANCHOR_PATH) / "hard_negatives"

print(f"Loading known lenses from {known_lens_dir}...")
known_stamps, known_names = load_fits_stamps(known_lens_dir)
print(f"  Loaded {len(known_stamps)} known lenses")

print(f"Loading hard negatives from {hard_neg_dir}...")
neg_stamps, neg_names = load_fits_stamps(hard_neg_dir)
print(f"  Loaded {len(neg_stamps)} hard negatives")

if len(known_stamps) > 0:
    print()
    print("Running inference on known lenses...")
    with torch.no_grad():
        x = torch.tensor(np.array(known_stamps), dtype=torch.float32).to(device)
        logits = model(x)
        known_probs = torch.sigmoid(logits).cpu().numpy()
    
    print()
    print("Top 20 Known Lenses by p_lens:")
    print("-" * 60)
    sorted_idx = np.argsort(known_probs)[::-1]
    for i, idx in enumerate(sorted_idx[:20]):
        print(f"  {i+1:2d}. {known_names[idx]:35s} p_lens = {known_probs[idx]:.4f}")
    
    print()
    print("Recall at Thresholds:")
    for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        recall = (known_probs >= thresh).mean()
        n = (known_probs >= thresh).sum()
        print(f"  @{thresh}: {recall*100:.1f}% ({n}/{len(known_probs)})")
    
    print()
    print("Statistics:")
    print(f"  Mean p_lens: {np.mean(known_probs):.4f}")
    print(f"  Median p_lens: {np.median(known_probs):.4f}")
    print(f"  Std p_lens: {np.std(known_probs):.4f}")
    print(f"  Min p_lens: {np.min(known_probs):.4f}")
    print(f"  Max p_lens: {np.max(known_probs):.4f}")

if len(neg_stamps) > 0:
    print()
    print("Running inference on hard negatives...")
    with torch.no_grad():
        x = torch.tensor(np.array(neg_stamps), dtype=torch.float32).to(device)
        logits = model(x)
        neg_probs = torch.sigmoid(logits).cpu().numpy()
    
    print()
    print("Hard Negatives by p_lens:")
    sorted_idx = np.argsort(neg_probs)[::-1]
    for i, idx in enumerate(sorted_idx):
        print(f"  {i+1:2d}. {neg_names[idx]:35s} p_lens = {neg_probs[idx]:.4f}")

print()
print("=" * 60)
print("DONE")
print("=" * 60)

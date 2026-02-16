#!/usr/bin/env python3
"""
Gate 1.7: Arc-Suppressed Positive Test

Zero out the annulus where arcs should live in positive samples (10-25px).
If the model still predicts high lens probability, it's NOT using arc features.

Per LLM recommendation: "Arc-suppressed positives: take positive samples and 
zero out the annulus where arcs live. If the model still predicts lens with 
high confidence, the dataset is core-driven."
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
    "gate": "1.7",
    "name": "Arc-Suppressed Positive Test",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "description": "Tests if model uses arc features by zeroing arc region"
}

print("=" * 70)
print("GATE 1.7: ARC-SUPPRESSED POSITIVE TEST")
print("=" * 70)

# Configuration
ARC_INNER_RADIUS = 10  # pixels - inner edge of arc annulus
ARC_OUTER_RADIUS = 25  # pixels - outer edge of arc annulus
N_SAMPLES = 1000

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

RESULTS["model_epoch"] = int(ckpt["epoch"])

def robust_mad_norm_outer(x, clip=10.0, eps=1e-6, inner_frac=0.5):
    """Normalize using outer annulus MAD (matching training)."""
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

def suppress_arc_annulus(img, r_inner=ARC_INNER_RADIUS, r_outer=ARC_OUTER_RADIUS):
    """
    Zero out the annulus where arcs live.
    Fills with outer-region random samples to maintain noise statistics.
    """
    C, H, W = img.shape
    out = img.copy()
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r2 = (yy - cy)**2 + (xx - cx)**2
    
    # Arc annulus mask
    arc_mask = (r2 >= r_inner**2) & (r2 < r_outer**2)
    
    # Sample from outer region (beyond arc annulus)
    outer_mask = r2 >= r_outer**2
    
    for c in range(C):
        outer_vals = img[c][outer_mask]
        if len(outer_vals) == 0:
            continue
        n_arc = arc_mask.sum()
        fill_vals = np.random.choice(outer_vals, size=n_arc, replace=True)
        out[c][arc_mask] = fill_vals
    
    return out

def decode_stamp(blob):
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        g = z["image_g"].astype(np.float32)
        r = z["image_r"].astype(np.float32)
        zb = z["image_z"].astype(np.float32)
    return np.stack([g, r, zb], axis=0)

# Load positive samples
data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

# Use test split positives
filt = (ds.field("region_split") == "test") & (ds.field("cutout_ok") == 1) & (ds.field("is_control") == 0)
table = dataset.to_table(filter=filt, columns=["stamp_npz", "psfsize_r", "psfdepth_r", "arc_snr"])

print(f"Total positive samples: {table.num_rows}")

# Sample for testing
np.random.seed(42)
indices = np.random.choice(table.num_rows, min(N_SAMPLES, table.num_rows), replace=False)

original_probs = []
suppressed_probs = []
arc_snrs = []

print(f"\nEvaluating {len(indices)} samples...")

for i, idx in enumerate(indices):
    if i % 200 == 0:
        print(f"  Progress: {i}/{len(indices)}")
    
    blob = table["stamp_npz"][int(idx)].as_py()
    if blob is None:
        continue
    
    try:
        img = decode_stamp(blob)
        if not np.isfinite(img).all():
            continue
        
        arc_snr = table["arc_snr"][int(idx)].as_py() or 0.0
        psfsize = table["psfsize_r"][int(idx)].as_py() or 1.0
        psfdepth = table["psfdepth_r"][int(idx)].as_py() or 0.0
        
        # Original prediction
        img_norm = robust_mad_norm_outer(img)
        if not np.isfinite(img_norm).all():
            continue
        
        x = torch.tensor(img_norm[np.newaxis], dtype=torch.float32).to(device)
        meta = torch.tensor([[psfsize, psfdepth]], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logit = model(x, meta)
            prob_orig = torch.sigmoid(logit).item()
        
        # Arc-suppressed prediction
        img_suppressed = suppress_arc_annulus(img, r_inner=ARC_INNER_RADIUS, r_outer=ARC_OUTER_RADIUS)
        img_supp_norm = robust_mad_norm_outer(img_suppressed)
        if not np.isfinite(img_supp_norm).all():
            continue
        
        x_supp = torch.tensor(img_supp_norm[np.newaxis], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logit_supp = model(x_supp, meta)
            prob_supp = torch.sigmoid(logit_supp).item()
        
        original_probs.append(prob_orig)
        suppressed_probs.append(prob_supp)
        arc_snrs.append(arc_snr)
        
    except Exception as e:
        continue

original_probs = np.array(original_probs)
suppressed_probs = np.array(suppressed_probs)
arc_snrs = np.array(arc_snrs)

print(f"\nEvaluated {len(original_probs)} samples successfully")

# Analyze results
print("\n" + "=" * 70)
print("RESULTS:")
print("=" * 70)

print(f"\nOriginal predictions:")
print(f"  Mean p_lens: {original_probs.mean():.4f}")
print(f"  Recall@0.5: {(original_probs > 0.5).mean():.1%}")
print(f"  Recall@0.9: {(original_probs > 0.9).mean():.1%}")

print(f"\nArc-suppressed predictions:")
print(f"  Mean p_lens: {suppressed_probs.mean():.4f}")
print(f"  Recall@0.5: {(suppressed_probs > 0.5).mean():.1%}")
print(f"  Recall@0.9: {(suppressed_probs > 0.9).mean():.1%}")

drop = original_probs.mean() - suppressed_probs.mean()
drop_pct = drop / original_probs.mean() * 100 if original_probs.mean() > 0 else 0

print(f"\nDrop in mean p_lens: {drop:.4f} ({drop_pct:.1f}%)")

# Per-sample analysis
per_sample_drop = original_probs - suppressed_probs
print(f"\nPer-sample drop statistics:")
print(f"  Mean drop: {per_sample_drop.mean():.4f}")
print(f"  Median drop: {np.median(per_sample_drop):.4f}")
print(f"  Samples with drop > 0.3: {(per_sample_drop > 0.3).mean():.1%}")
print(f"  Samples with drop > 0.5: {(per_sample_drop > 0.5).mean():.1%}")

# Stratify by arc_snr
print("\nBy arc_snr bin:")
for snr_min, snr_max in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 100)]:
    mask = (arc_snrs >= snr_min) & (arc_snrs < snr_max)
    if mask.sum() < 10:
        continue
    orig_bin = original_probs[mask]
    supp_bin = suppressed_probs[mask]
    drop_bin = orig_bin.mean() - supp_bin.mean()
    print(f"  SNR [{snr_min}-{snr_max}): n={mask.sum()}, orig={orig_bin.mean():.3f}, supp={supp_bin.mean():.3f}, drop={drop_bin:.3f}")

# Store results
RESULTS["n_evaluated"] = int(len(original_probs))
RESULTS["arc_inner_radius"] = ARC_INNER_RADIUS
RESULTS["arc_outer_radius"] = ARC_OUTER_RADIUS
RESULTS["original_mean_p"] = float(original_probs.mean())
RESULTS["original_recall_at_0.5"] = float((original_probs > 0.5).mean())
RESULTS["suppressed_mean_p"] = float(suppressed_probs.mean())
RESULTS["suppressed_recall_at_0.5"] = float((suppressed_probs > 0.5).mean())
RESULTS["drop_mean_p"] = float(drop)
RESULTS["drop_percent"] = float(drop_pct)
RESULTS["per_sample_drop_median"] = float(np.median(per_sample_drop))
RESULTS["frac_samples_drop_gt_0.3"] = float((per_sample_drop > 0.3).mean())

# Pass/Fail
# If suppressed mean is still high (>0.5), model isn't using arcs
THRESHOLD = 0.5
passed = suppressed_probs.mean() < THRESHOLD

RESULTS["threshold"] = THRESHOLD
RESULTS["overall_passed"] = passed

print("\n" + "=" * 70)
print("GATE 1.7 CONCLUSION:")
print("=" * 70)

if passed:
    print(f"PASS: Arc-suppressed mean p_lens = {suppressed_probs.mean():.4f} < {THRESHOLD}")
    print("When arcs are removed, model no longer predicts 'lens' with high confidence.")
    print("This suggests the model IS using arc features.")
else:
    print(f"FAIL: Arc-suppressed mean p_lens = {suppressed_probs.mean():.4f} >= {THRESHOLD}")
    print("When arcs are removed, model STILL predicts 'lens' with high confidence!")
    print("This proves the model is NOT using arc features - it's using shortcuts.")

# Save results
output_path = "/lambda/nfs/darkhaloscope-training-dc/gate_1_7_results.json"
with open(output_path, "w") as f:
    json.dump(RESULTS, f, indent=2)

print(f"\nResults saved to {output_path}")

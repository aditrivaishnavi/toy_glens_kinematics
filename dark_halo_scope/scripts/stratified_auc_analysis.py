#!/usr/bin/env python3
"""
Stratified AUC Analysis: Compare val/test AUC within matched difficulty bins.

This verifies whether the AUC(val)=0.8945 vs AUC(test)=0.9945 gap is due to
difficulty stratification (different distributions of arc_snr, theta_e/psf).
"""
import numpy as np
import pyarrow.dataset as ds
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
import io
from collections import Counter

print("=" * 70)
print("PHASE 0.2: STRATIFIED AUC ANALYSIS")
print("=" * 70)

# ============================================================
# Load Model
# ============================================================
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
print(f"Model loaded from epoch {ckpt['epoch']}")

# ============================================================
# Normalization (must match training)
# ============================================================
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

# ============================================================
# Load Data and Compute Predictions
# ============================================================
def load_split_data(split, max_samples=10000):
    """Load samples from a split with all needed columns."""
    data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
    dataset = ds.dataset(data_path, format="parquet", partitioning="hive")
    
    filt = (ds.field("region_split") == split) & (ds.field("cutout_ok") == 1)
    cols = ["stamp_npz", "is_control", "arc_snr", "theta_e_arcsec", 
            "psfsize_r", "psfdepth_r", "bad_pixel_frac", "bandset"]
    
    table = dataset.to_table(filter=filt, columns=cols)
    table = table.slice(0, min(max_samples, table.num_rows))
    
    print(f"  Loaded {table.num_rows} samples from {split}")
    return table

def compute_predictions(table, batch_size=64):
    """Run inference on all samples."""
    n = table.num_rows
    predictions = []
    labels = []
    meta_data = []
    
    for i in range(0, n, batch_size):
        batch_end = min(i + batch_size, n)
        batch_imgs = []
        batch_meta = []
        
        for j in range(i, batch_end):
            blob = table["stamp_npz"][j].as_py()
            if blob is None:
                continue
            try:
                img = decode_stamp(blob)
                img_norm = robust_mad_norm_outer(img)
                if not np.isfinite(img_norm).all():
                    continue
                batch_imgs.append(img_norm)
                
                psfsize = table["psfsize_r"][j].as_py() or 1.0
                psfdepth = table["psfdepth_r"][j].as_py() or 0.0
                batch_meta.append([psfsize, psfdepth])
                
                labels.append(0 if table["is_control"][j].as_py() == 1 else 1)
                
                arc_snr = table["arc_snr"][j].as_py() or 0.0
                theta_e = table["theta_e_arcsec"][j].as_py() or 0.0
                bad_pix = table["bad_pixel_frac"][j].as_py() or 0.0
                meta_data.append({
                    "arc_snr": arc_snr,
                    "theta_e": theta_e,
                    "psfsize_r": psfsize,
                    "psfdepth_r": psfdepth,
                    "bad_pixel_frac": bad_pix,
                    "theta_over_psf": theta_e / psfsize if psfsize > 0 else 0,
                })
            except Exception as e:
                continue
        
        if len(batch_imgs) == 0:
            continue
            
        x = torch.tensor(np.stack(batch_imgs), dtype=torch.float32).to(device)
        meta_tensor = torch.tensor(np.array(batch_meta), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logits = model(x, meta_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        predictions.extend(probs.tolist())
    
    return np.array(predictions), np.array(labels), meta_data

# ============================================================
# Stratified AUC Computation
# ============================================================
def compute_stratified_auc(predictions, labels, meta_data, bins_config, bin_col):
    """Compute AUC within each bin."""
    results = {}
    values = np.array([m[bin_col] for m in meta_data])
    
    for bin_name, (low, high) in bins_config.items():
        mask = (values >= low) & (values < high)
        y_true = labels[mask]
        y_pred = predictions[mask]
        
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        if n_pos < 10 or n_neg < 10:
            results[bin_name] = {"auc": None, "n": len(y_true), "n_pos": n_pos, "n_neg": n_neg}
        else:
            auc = roc_auc_score(y_true, y_pred)
            results[bin_name] = {"auc": auc, "n": len(y_true), "n_pos": n_pos, "n_neg": n_neg}
    
    return results

# ============================================================
# Main Analysis
# ============================================================
print("\nLoading validation set...")
val_table = load_split_data("val", max_samples=10000)
print("Computing predictions on val...")
val_preds, val_labels, val_meta = compute_predictions(val_table)

print("\nLoading test set...")
test_table = load_split_data("test", max_samples=10000)
print("Computing predictions on test...")
test_preds, test_labels, test_meta = compute_predictions(test_table)

# Global AUC
val_auc_global = roc_auc_score(val_labels, val_preds)
test_auc_global = roc_auc_score(test_labels, test_preds)

print("\n" + "=" * 70)
print("GLOBAL AUC:")
print("=" * 70)
print(f"  Val AUC:  {val_auc_global:.4f} (n={len(val_labels)})")
print(f"  Test AUC: {test_auc_global:.4f} (n={len(test_labels)})")
print(f"  Gap:      {test_auc_global - val_auc_global:.4f}")

# Define bins
THETA_PSF_BINS = {
    "0.5-0.75": (0.5, 0.75),
    "0.75-1.0": (0.75, 1.0),
    "1.0-1.5": (1.0, 1.5),
    "1.5-2.5": (1.5, 2.5),
    "2.5+": (2.5, float('inf')),
}

ARC_SNR_BINS = {
    "0-2": (0, 2),
    "2-5": (2, 5),
    "5-10": (5, 10),
    "10-20": (10, 20),
    "20+": (20, float('inf')),
}

# Stratified AUC by theta_e/psf
print("\n" + "=" * 70)
print("STRATIFIED AUC BY theta_e / psfsize_r:")
print("=" * 70)
val_by_theta = compute_stratified_auc(val_preds, val_labels, val_meta, THETA_PSF_BINS, "theta_over_psf")
test_by_theta = compute_stratified_auc(test_preds, test_labels, test_meta, THETA_PSF_BINS, "theta_over_psf")

print(f"{'Bin':<15} {'Val AUC':<12} {'Test AUC':<12} {'Gap':<10} {'Val n':<10} {'Test n':<10}")
print("-" * 70)
for bin_name in THETA_PSF_BINS:
    v = val_by_theta[bin_name]
    t = test_by_theta[bin_name]
    v_auc = f"{v['auc']:.4f}" if v['auc'] else "N/A"
    t_auc = f"{t['auc']:.4f}" if t['auc'] else "N/A"
    gap = f"{t['auc'] - v['auc']:.4f}" if (v['auc'] and t['auc']) else "N/A"
    print(f"{bin_name:<15} {v_auc:<12} {t_auc:<12} {gap:<10} {v['n']:<10} {t['n']:<10}")

# Stratified AUC by arc_snr
print("\n" + "=" * 70)
print("STRATIFIED AUC BY arc_snr:")
print("=" * 70)
val_by_snr = compute_stratified_auc(val_preds, val_labels, val_meta, ARC_SNR_BINS, "arc_snr")
test_by_snr = compute_stratified_auc(test_preds, test_labels, test_meta, ARC_SNR_BINS, "arc_snr")

print(f"{'Bin':<15} {'Val AUC':<12} {'Test AUC':<12} {'Gap':<10} {'Val n':<10} {'Test n':<10}")
print("-" * 70)
for bin_name in ARC_SNR_BINS:
    v = val_by_snr[bin_name]
    t = test_by_snr[bin_name]
    v_auc = f"{v['auc']:.4f}" if v['auc'] else "N/A"
    t_auc = f"{t['auc']:.4f}" if t['auc'] else "N/A"
    gap = f"{t['auc'] - v['auc']:.4f}" if (v['auc'] and t['auc']) else "N/A"
    print(f"{bin_name:<15} {v_auc:<12} {t_auc:<12} {gap:<10} {v['n']:<10} {t['n']:<10}")

# ============================================================
# Phase 0.3: Distribution Comparison
# ============================================================
print("\n" + "=" * 70)
print("PHASE 0.3: DISTRIBUTION COMPARISON (VAL vs TEST)")
print("=" * 70)

from scipy.stats import ks_2samp

COMPARE_VARS = ["arc_snr", "theta_e", "psfsize_r", "psfdepth_r", "bad_pixel_frac", "theta_over_psf"]

print(f"{'Variable':<20} {'Val Mean':<12} {'Test Mean':<12} {'Ratio':<10} {'KS stat':<10} {'KS p-val':<10}")
print("-" * 80)

for var in COMPARE_VARS:
    val_vals = np.array([m[var] for m in val_meta])
    test_vals = np.array([m[var] for m in test_meta])
    
    # Remove NaN/Inf
    val_vals = val_vals[np.isfinite(val_vals)]
    test_vals = test_vals[np.isfinite(test_vals)]
    
    if len(val_vals) == 0 or len(test_vals) == 0:
        print(f"{var:<20} {'N/A':<12} {'N/A':<12}")
        continue
    
    stat, pval = ks_2samp(val_vals, test_vals)
    val_mean = np.mean(val_vals)
    test_mean = np.mean(test_vals)
    ratio = test_mean / val_mean if val_mean != 0 else float('inf')
    
    # Flag significant differences
    flag = " ***" if pval < 0.001 else " **" if pval < 0.01 else " *" if pval < 0.05 else ""
    print(f"{var:<20} {val_mean:<12.4f} {test_mean:<12.4f} {ratio:<10.3f} {stat:<10.4f} {pval:<10.4f}{flag}")

# ============================================================
# Conclusion
# ============================================================
print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)

# Check if stratified gaps are small
all_theta_gaps = []
for bn in THETA_PSF_BINS:
    v, t = val_by_theta[bn], test_by_theta[bn]
    if v['auc'] and t['auc']:
        all_theta_gaps.append(abs(t['auc'] - v['auc']))

all_snr_gaps = []
for bn in ARC_SNR_BINS:
    v, t = val_by_snr[bn], test_by_snr[bn]
    if v['auc'] and t['auc']:
        all_snr_gaps.append(abs(t['auc'] - v['auc']))

max_theta_gap = max(all_theta_gaps) if all_theta_gaps else 0
max_snr_gap = max(all_snr_gaps) if all_snr_gaps else 0
mean_theta_gap = np.mean(all_theta_gaps) if all_theta_gaps else 0
mean_snr_gap = np.mean(all_snr_gaps) if all_snr_gaps else 0

print(f"Global AUC gap: {test_auc_global - val_auc_global:.4f}")
print(f"Max stratified gap (theta/psf bins): {max_theta_gap:.4f}")
print(f"Max stratified gap (arc_snr bins): {max_snr_gap:.4f}")
print(f"Mean stratified gap (theta/psf bins): {mean_theta_gap:.4f}")
print(f"Mean stratified gap (arc_snr bins): {mean_snr_gap:.4f}")

if max_theta_gap < 0.05 and max_snr_gap < 0.05:
    print("\n✓ PASS: Stratified gaps are small (<0.05)")
    print("  The global AUC gap is explained by difficulty stratification.")
    print("  PROCEED to Phase 1 (Pipeline Parity Check).")
    exit(0)
elif max_theta_gap < 0.10 and max_snr_gap < 0.10:
    print("\n⚠ MARGINAL: Stratified gaps are moderate (0.05-0.10)")
    print("  Likely explained by stratification, but monitor.")
    print("  PROCEED to Phase 1 with caution.")
    exit(0)
else:
    print("\n✗ FAIL: Stratified gaps are large (>0.10)")
    print("  Something other than difficulty stratification differs between val/test.")
    print("  INVESTIGATE before proceeding.")
    exit(1)

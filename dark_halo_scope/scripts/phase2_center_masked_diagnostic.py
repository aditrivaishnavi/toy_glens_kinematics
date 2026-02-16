#!/usr/bin/env python3
"""
Phase 2: Center-masked diagnostic training.
Tests if model relies on lens-galaxy core for classification.

This script provides:
1. Center masking augmentation function
2. Diagnostic training configuration
3. Evaluation of masking impact on anchor recall

The diagnostic tests whether masking the central region (where the lens galaxy
lives) during training forces the model to learn arc features instead of
galaxy-core shortcuts.
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
import pyarrow.dataset as ds
import io
import json
from datetime import datetime, timezone

# ============================================================
# Center Masking Augmentation Function
# ============================================================

def mask_center_outer_fill(img, r_mask=10, seed=None):
    """
    Mask center with pixels resampled from outer annulus.
    
    This removes the central lens galaxy signal while maintaining
    realistic noise statistics. The model must then rely on arc
    features in the outer region.
    
    Args:
        img: (C, H, W) array - 3-channel image
        r_mask: mask radius in pixels (default 10 = 2.62 arcsec at 0.262"/pix)
        seed: random seed for reproducibility
    
    Returns:
        Masked image with center filled from outer annulus samples
    
    Example:
        >>> img = np.random.randn(3, 64, 64).astype(np.float32)
        >>> masked = mask_center_outer_fill(img, r_mask=10)
        >>> assert masked.shape == img.shape
    """
    if seed is not None:
        np.random.seed(seed)
    
    C, H, W = img.shape
    out = img.copy()
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r2 = (yy - cy)**2 + (xx - cx)**2
    
    # Mask center (r < r_mask)
    center_mask = r2 < r_mask**2
    
    # Sample from outer annulus (r >= 2*r_mask)
    outer_mask = r2 >= (r_mask * 2)**2
    
    for c in range(C):
        outer_vals = img[c][outer_mask]
        if len(outer_vals) == 0:
            continue
        n_center = center_mask.sum()
        fill_vals = np.random.choice(outer_vals, size=n_center, replace=True)
        out[c][center_mask] = fill_vals
    
    return out


def apply_center_mask_augmentation(batch_imgs, r_mask=10, p=0.5):
    """
    Apply center masking to a batch with probability p.
    
    Args:
        batch_imgs: (B, C, H, W) tensor
        r_mask: mask radius in pixels
        p: probability of applying mask
    
    Returns:
        (B, C, H, W) tensor with some images masked
    """
    B, C, H, W = batch_imgs.shape
    out = batch_imgs.clone()
    
    for i in range(B):
        if np.random.random() < p:
            img_np = batch_imgs[i].cpu().numpy()
            masked = mask_center_outer_fill(img_np, r_mask=r_mask)
            out[i] = torch.from_numpy(masked)
    
    return out


# ============================================================
# Diagnostic Configuration
# ============================================================

DIAGNOSTIC_CONFIG = {
    "experiment_name": "gen5_center_masked_diagnostic",
    "description": "Test if masking center during training improves anchor recall",
    "mask_radii_to_test": [8, 10, 12],  # pixels (2.1", 2.6", 3.1" at 0.262"/pix)
    "fill_policy": "outer_annulus_resample",
    "apply_at_training": True,
    "apply_at_inference": False,  # Never mask at inference
    "mask_probability": 0.5,  # 50% of training samples get masked
    "epochs": 3,  # Quick diagnostic, not full training
    "batch_size": 64,
    "learning_rate": 1e-4,
    "metrics_to_track": [
        "synthetic_auc",
        "synthetic_tpr_at_fpr_1pct",
        "anchor_recall_at_0.5",
        "anchor_mean_p_lens"
    ],
    "expected_outcomes": {
        "if_shortcut": "AUC drops significantly when center masked",
        "if_no_shortcut": "AUC stays similar, anchor recall may improve"
    }
}

# ============================================================
# Run Diagnostic Test (Quick Version)
# ============================================================

def run_quick_diagnostic():
    """
    Quick diagnostic: Check how much model predictions change
    when we mask the center of positives.
    
    This is a fast proxy for the full diagnostic training.
    If predictions drop significantly, the model relies on center.
    """
    print("=" * 70)
    print("PHASE 2: CENTER-MASKED DIAGNOSTIC (QUICK VERSION)")
    print("=" * 70)
    
    MODEL_PATH = "/lambda/nfs/darkhaloscope-training-dc/models/gen5_cosmos/ckpt_best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
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
    
    # Normalization function
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
    
    # Load positive samples from test set
    data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
    dataset = ds.dataset(data_path, format="parquet", partitioning="hive")
    filt = (ds.field("region_split") == "test") & (ds.field("cutout_ok") == 1) & (ds.field("is_control") == 0)
    table = dataset.to_table(filter=filt, columns=["stamp_npz", "psfsize_r", "psfdepth_r", "arc_snr"])
    
    print(f"Total positive samples in test: {table.num_rows}")
    
    n_samples = min(500, table.num_rows)
    results = {
        "r_mask": [],
        "original_p": [],
        "masked_p": []
    }
    
    # Test different mask radii
    for r_mask in DIAGNOSTIC_CONFIG["mask_radii_to_test"]:
        print(f"\nTesting r_mask = {r_mask} pixels ({r_mask * 0.262:.2f} arcsec)...")
        
        original_probs = []
        masked_probs = []
        
        for i in range(n_samples):
            blob = table["stamp_npz"][i].as_py()
            if blob is None:
                continue
            try:
                img = decode_stamp(blob)
                
                # Original prediction
                img_norm = robust_mad_norm_outer(img)
                if not np.isfinite(img_norm).all():
                    continue
                
                psfsize = table["psfsize_r"][i].as_py() or 1.0
                psfdepth = table["psfdepth_r"][i].as_py() or 0.0
                
                x = torch.tensor(img_norm[np.newaxis], dtype=torch.float32).to(device)
                meta = torch.tensor([[psfsize, psfdepth]], dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    logit = model(x, meta)
                    prob_orig = torch.sigmoid(logit).item()
                
                # Masked prediction
                img_masked = mask_center_outer_fill(img, r_mask=r_mask, seed=i)
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
        
        original_probs = np.array(original_probs)
        masked_probs = np.array(masked_probs)
        
        print(f"  Evaluated {len(original_probs)} samples")
        print(f"  Original: mean p = {original_probs.mean():.3f}, recall@0.5 = {(original_probs > 0.5).mean():.1%}")
        print(f"  Masked:   mean p = {masked_probs.mean():.3f}, recall@0.5 = {(masked_probs > 0.5).mean():.1%}")
        print(f"  Drop: {(original_probs.mean() - masked_probs.mean()):.3f}")
        
        results["r_mask"].append(r_mask)
        results["original_p"].append(float(original_probs.mean()))
        results["masked_p"].append(float(masked_probs.mean()))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    RESULTS = {
        "phase": "2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": "Quick diagnostic: How much do predictions drop when center is masked?",
        "config": DIAGNOSTIC_CONFIG,
        "results_by_radius": []
    }
    
    for i, r in enumerate(results["r_mask"]):
        entry = {
            "r_mask_pixels": r,
            "r_mask_arcsec": r * 0.262,
            "mean_p_original": results["original_p"][i],
            "mean_p_masked": results["masked_p"][i],
            "drop": results["original_p"][i] - results["masked_p"][i],
            "drop_percent": (results["original_p"][i] - results["masked_p"][i]) / results["original_p"][i] * 100 if results["original_p"][i] > 0 else 0
        }
        RESULTS["results_by_radius"].append(entry)
        print(f"r_mask={r}px: drop = {entry['drop']:.3f} ({entry['drop_percent']:.1f}%)")
    
    # Interpretation
    max_drop = max(r["drop_percent"] for r in RESULTS["results_by_radius"])
    
    if max_drop > 30:
        interpretation = "HIGH RELIANCE ON CENTER - Model likely uses galaxy-core shortcuts"
        recommendation = "Full center-masked retraining recommended"
    elif max_drop > 15:
        interpretation = "MODERATE RELIANCE ON CENTER - Model uses mix of center and arc features"
        recommendation = "Center-masked training may improve anchor recall"
    else:
        interpretation = "LOW RELIANCE ON CENTER - Model primarily uses arc features"
        recommendation = "Center-masked training unlikely to help significantly"
    
    RESULTS["interpretation"] = interpretation
    RESULTS["recommendation"] = recommendation
    RESULTS["overall_max_drop_percent"] = max_drop
    
    print(f"\nInterpretation: {interpretation}")
    print(f"Recommendation: {recommendation}")
    
    # Save results
    with open("/lambda/nfs/darkhaloscope-training-dc/phase2_diagnostic_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    
    print("\nResults saved to phase2_diagnostic_results.json")
    return RESULTS


if __name__ == "__main__":
    print("Configuration:")
    print(json.dumps(DIAGNOSTIC_CONFIG, indent=2))
    print("\n")
    run_quick_diagnostic()

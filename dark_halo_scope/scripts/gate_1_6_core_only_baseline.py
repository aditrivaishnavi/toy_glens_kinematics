#!/usr/bin/env python3
"""
Gate 1.6: Core-Only Baseline Classifier Test

Trains a simple classifier using ONLY the central r < 10px region.
If this achieves high AUC (>0.6), it proves there's a shortcut in the
core region that the model can exploit without learning arc features.

Per LLM recommendation: "Core-only baseline: train a classifier on only 
r < 10 px. If it achieves high AUC, you have a shortcut."
"""
import numpy as np
import pyarrow.dataset as ds
import io
import json
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

RESULTS = {
    "gate": "1.6",
    "name": "Core-Only Baseline Classifier",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "description": "Tests if core-only features can separate classes (shortcut detection)"
}

print("=" * 70)
print("GATE 1.6: CORE-ONLY BASELINE CLASSIFIER")
print("=" * 70)

# Configuration
CORE_RADIUS = 10  # pixels
N_SAMPLES = 10000  # per class

def decode_stamp(blob):
    """Decode NPZ blob to (3, H, W) array."""
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        g = z["image_g"].astype(np.float32)
        r = z["image_r"].astype(np.float32)
        zb = z["image_z"].astype(np.float32)
    return np.stack([g, r, zb], axis=0)

def extract_core_features(img, r_core=CORE_RADIUS):
    """
    Extract features from only the central region.
    Returns flattened core pixels plus summary stats.
    """
    C, H, W = img.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    core_mask = ((yy - cy)**2 + (xx - cx)**2) < r_core**2
    
    features = []
    for c in range(C):
        core_pixels = img[c][core_mask]
        # Summary statistics
        features.extend([
            np.mean(core_pixels),
            np.std(core_pixels),
            np.median(core_pixels),
            np.max(core_pixels),
            np.min(core_pixels),
            np.percentile(core_pixels, 25),
            np.percentile(core_pixels, 75),
        ])
    
    return np.array(features)

# Load data
print("Loading data...")
data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

# Use train split for training, val for testing
train_filt = (ds.field("region_split") == "train") & (ds.field("cutout_ok") == 1)
val_filt = (ds.field("region_split") == "val") & (ds.field("cutout_ok") == 1)

train_table = dataset.to_table(filter=train_filt, columns=["stamp_npz", "is_control"])
val_table = dataset.to_table(filter=val_filt, columns=["stamp_npz", "is_control"])

print(f"Train samples: {train_table.num_rows}")
print(f"Val samples: {val_table.num_rows}")

# Sample indices
np.random.seed(42)

def sample_indices_by_class(table, n_per_class):
    """Sample balanced indices from table."""
    control_idx = [i for i in range(table.num_rows) if table["is_control"][i].as_py() == 1]
    positive_idx = [i for i in range(table.num_rows) if table["is_control"][i].as_py() == 0]
    
    ctrl_sample = np.random.choice(control_idx, min(n_per_class, len(control_idx)), replace=False)
    pos_sample = np.random.choice(positive_idx, min(n_per_class, len(positive_idx)), replace=False)
    
    return ctrl_sample, pos_sample

train_ctrl, train_pos = sample_indices_by_class(train_table, N_SAMPLES)
val_ctrl, val_pos = sample_indices_by_class(val_table, N_SAMPLES // 2)

print(f"Train: {len(train_ctrl)} controls, {len(train_pos)} positives")
print(f"Val: {len(val_ctrl)} controls, {len(val_pos)} positives")

# Extract features
def extract_features_from_table(table, indices, label):
    """Extract core features from table at given indices."""
    X, y = [], []
    for idx in indices:
        blob = table["stamp_npz"][int(idx)].as_py()
        if blob is None:
            continue
        try:
            img = decode_stamp(blob)
            if not np.isfinite(img).all():
                continue
            feats = extract_core_features(img)
            if np.isfinite(feats).all():
                X.append(feats)
                y.append(label)
        except Exception:
            continue
    return np.array(X), np.array(y)

print("\nExtracting core features from train controls...")
X_train_ctrl, y_train_ctrl = extract_features_from_table(train_table, train_ctrl, 0)
print(f"  {len(X_train_ctrl)} samples extracted")

print("Extracting core features from train positives...")
X_train_pos, y_train_pos = extract_features_from_table(train_table, train_pos, 1)
print(f"  {len(X_train_pos)} samples extracted")

print("Extracting core features from val controls...")
X_val_ctrl, y_val_ctrl = extract_features_from_table(val_table, val_ctrl, 0)
print(f"  {len(X_val_ctrl)} samples extracted")

print("Extracting core features from val positives...")
X_val_pos, y_val_pos = extract_features_from_table(val_table, val_pos, 1)
print(f"  {len(X_val_pos)} samples extracted")

# Combine
X_train = np.vstack([X_train_ctrl, X_train_pos])
y_train = np.concatenate([y_train_ctrl, y_train_pos])
X_val = np.vstack([X_val_ctrl, X_val_pos])
y_val = np.concatenate([y_val_ctrl, y_val_pos])

print(f"\nTotal train: {len(X_train)}, Total val: {len(X_val)}")

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train logistic regression
print("\nTraining logistic regression on core-only features...")
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)

# Evaluate
train_probs = clf.predict_proba(X_train_scaled)[:, 1]
val_probs = clf.predict_proba(X_val_scaled)[:, 1]

train_auc = roc_auc_score(y_train, train_probs)
val_auc = roc_auc_score(y_val, val_probs)

print(f"\nTrain AUC: {train_auc:.4f}")
print(f"Val AUC: {val_auc:.4f}")

# Feature importance (coefficient magnitudes)
feature_names = []
for band in ['g', 'r', 'z']:
    feature_names.extend([
        f'{band}_mean', f'{band}_std', f'{band}_median', 
        f'{band}_max', f'{band}_min', f'{band}_q25', f'{band}_q75'
    ])

coef_importance = list(zip(feature_names, np.abs(clf.coef_[0])))
coef_importance.sort(key=lambda x: x[1], reverse=True)

print("\nTop 5 most important features:")
for name, imp in coef_importance[:5]:
    print(f"  {name}: {imp:.4f}")

# Store results
RESULTS["core_radius_pixels"] = CORE_RADIUS
RESULTS["n_train"] = int(len(X_train))
RESULTS["n_val"] = int(len(X_val))
RESULTS["train_auc"] = float(train_auc)
RESULTS["val_auc"] = float(val_auc)
RESULTS["top_features"] = [{"name": n, "importance": float(i)} for n, i in coef_importance[:10]]

# Pass/Fail determination
# AUC > 0.6 indicates a shortcut (should be ~0.5 for random)
AUC_THRESHOLD = 0.6
passed = val_auc < AUC_THRESHOLD

RESULTS["threshold"] = AUC_THRESHOLD
RESULTS["overall_passed"] = passed

print("\n" + "=" * 70)
print("GATE 1.6 CONCLUSION:")
print("=" * 70)

if passed:
    print(f"PASS: Core-only AUC = {val_auc:.4f} < {AUC_THRESHOLD}")
    print("Core features alone cannot separate classes well.")
    print("No evidence of core-based shortcut.")
else:
    print(f"FAIL: Core-only AUC = {val_auc:.4f} >= {AUC_THRESHOLD}")
    print("Core features can separate classes!")
    print("This indicates a shortcut in the central region.")
    print("Possible causes:")
    print("  - Core brightness differs between classes")
    print("  - Lens galaxy morphology differs")
    print("  - Normalization artifacts")

# Save results
output_path = "/lambda/nfs/darkhaloscope-training-dc/gate_1_6_results.json"
with open(output_path, "w") as f:
    json.dump(RESULTS, f, indent=2)

print(f"\nResults saved to {output_path}")

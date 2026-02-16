#!/usr/bin/env python3
"""
Gate 1.5: Normalization-Stat Leakage Gate

Checks if the normalization statistics (median, MAD, clip fraction) differ
between positives and controls. If they do, the model could exploit these
differences as shortcuts without learning arc morphology.

Per LLM recommendation: "Compute and compare by class: per-band outer median,
per-band outer MAD, per-band clipping fraction."
"""
import numpy as np
import pyarrow.dataset as ds
import io
import json
from datetime import datetime, timezone
from scipy.stats import ks_2samp

RESULTS = {
    "gate": "1.5",
    "name": "Normalization-Stat Leakage Gate",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "description": "Checks if normalization stats differ between classes"
}

print("=" * 70)
print("GATE 1.5: NORMALIZATION-STAT LEAKAGE GATE")
print("=" * 70)

# Configuration (matching training)
CLIP_THRESHOLD = 10.0
INNER_FRAC = 0.5
EPS = 1e-6

def compute_norm_stats(img, clip=CLIP_THRESHOLD, inner_frac=INNER_FRAC, eps=EPS):
    """
    Compute normalization statistics for a 3-channel image.
    Returns: dict with per-band median, MAD, clip_fraction
    """
    C, H, W = img.shape
    cy, cx = H // 2, W // 2
    ri = int(min(H, W) * inner_frac / 2)
    yy, xx = np.ogrid[:H, :W]
    outer_mask = ((yy - cy)**2 + (xx - cx)**2) > ri**2
    
    stats = {}
    for c, band in enumerate(['g', 'r', 'z']):
        v = img[c]
        outer_v = v[outer_mask]
        
        med = np.median(outer_v)
        mad = np.median(np.abs(outer_v - med))
        scale = 1.4826 * mad + eps
        
        # Compute normalized values to check clipping
        normed = (v - med) / scale
        clip_frac = np.mean(np.abs(normed) >= clip)
        
        stats[f'outer_median_{band}'] = float(med)
        stats[f'outer_mad_{band}'] = float(mad)
        stats[f'clip_frac_{band}'] = float(clip_frac)
    
    return stats

def decode_stamp(blob):
    """Decode NPZ blob to (3, H, W) array."""
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        g = z["image_g"].astype(np.float32)
        r = z["image_r"].astype(np.float32)
        zb = z["image_z"].astype(np.float32)
    return np.stack([g, r, zb], axis=0)

# Load data
data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

# Filter to train split
filt = (ds.field("region_split") == "train") & (ds.field("cutout_ok") == 1)
table = dataset.to_table(filter=filt, columns=["stamp_npz", "is_control"])

print(f"Total samples: {table.num_rows}")

# Sample for efficiency (5000 per class should be sufficient)
n_per_class = 5000
np.random.seed(42)

# Separate indices by class
control_indices = [i for i in range(table.num_rows) if table["is_control"][i].as_py() == 1]
positive_indices = [i for i in range(table.num_rows) if table["is_control"][i].as_py() == 0]

print(f"Controls: {len(control_indices)}, Positives: {len(positive_indices)}")

# Sample
control_sample = np.random.choice(control_indices, min(n_per_class, len(control_indices)), replace=False)
positive_sample = np.random.choice(positive_indices, min(n_per_class, len(positive_indices)), replace=False)

print(f"Sampling {len(control_sample)} controls and {len(positive_sample)} positives")

# Compute stats for each class
control_stats = []
positive_stats = []

print("\nComputing normalization stats for controls...")
for idx in control_sample:
    blob = table["stamp_npz"][int(idx)].as_py()
    if blob is None:
        continue
    try:
        img = decode_stamp(blob)
        if not np.isfinite(img).all():
            continue
        stats = compute_norm_stats(img)
        control_stats.append(stats)
    except Exception as e:
        continue

print(f"Computed {len(control_stats)} control stats")

print("\nComputing normalization stats for positives...")
for idx in positive_sample:
    blob = table["stamp_npz"][int(idx)].as_py()
    if blob is None:
        continue
    try:
        img = decode_stamp(blob)
        if not np.isfinite(img).all():
            continue
        stats = compute_norm_stats(img)
        positive_stats.append(stats)
    except Exception as e:
        continue

print(f"Computed {len(positive_stats)} positive stats")

# Compare distributions
STAT_KEYS = [
    'outer_median_r', 'outer_mad_r', 'clip_frac_r',
    'outer_median_g', 'outer_mad_g', 'clip_frac_g',
    'outer_median_z', 'outer_mad_z', 'clip_frac_z'
]

comparisons = []
any_failed = False

print("\n" + "=" * 70)
print("COMPARISON RESULTS:")
print("=" * 70)
print(f"{'Stat':<20} {'Ctrl Mean':<12} {'Pos Mean':<12} {'Diff':<10} {'KS stat':<10} {'p-val':<10} {'Status'}")
print("-" * 85)

for key in STAT_KEYS:
    ctrl_vals = np.array([s[key] for s in control_stats if key in s])
    pos_vals = np.array([s[key] for s in positive_stats if key in s])
    
    if len(ctrl_vals) < 100 or len(pos_vals) < 100:
        print(f"{key:<20} INSUFFICIENT DATA")
        continue
    
    ctrl_mean = np.mean(ctrl_vals)
    pos_mean = np.mean(pos_vals)
    diff = pos_mean - ctrl_mean
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(ctrl_vals) + np.var(pos_vals)) / 2)
    cohens_d = abs(diff) / (pooled_std + 1e-10)
    
    # KS test
    ks_stat, ks_pval = ks_2samp(ctrl_vals, pos_vals)
    
    # Pass criteria: effect size < 0.1 (small effect)
    passed = cohens_d < 0.1
    if not passed:
        any_failed = True
    
    status = "PASS" if passed else "FAIL"
    print(f"{key:<20} {ctrl_mean:<12.6f} {pos_mean:<12.6f} {diff:<10.6f} {ks_stat:<10.4f} {ks_pval:<10.4f} {status} (d={cohens_d:.3f})")
    
    comparisons.append({
        "stat": key,
        "ctrl_mean": float(ctrl_mean),
        "ctrl_std": float(np.std(ctrl_vals)),
        "pos_mean": float(pos_mean),
        "pos_std": float(np.std(pos_vals)),
        "diff": float(diff),
        "cohens_d": float(cohens_d),
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval),
        "passed": passed
    })

RESULTS["comparisons"] = comparisons
RESULTS["n_controls"] = len(control_stats)
RESULTS["n_positives"] = len(positive_stats)
RESULTS["overall_passed"] = not any_failed

print("\n" + "=" * 70)
print("GATE 1.5 CONCLUSION:")
print("=" * 70)

if any_failed:
    failed_stats = [c["stat"] for c in comparisons if not c["passed"]]
    print(f"FAIL: {len(failed_stats)} normalization stats differ significantly between classes")
    print(f"Failed stats: {failed_stats}")
    print("This indicates potential leakage through normalization artifacts.")
    RESULTS["failed_stats"] = failed_stats
else:
    print("PASS: All normalization stats are matched between classes (Cohen's d < 0.1)")
    print("No evidence of normalization-based leakage.")

# Save results
output_path = "/lambda/nfs/darkhaloscope-training-dc/gate_1_5_results.json"
with open(output_path, "w") as f:
    json.dump(RESULTS, f, indent=2)

print(f"\nResults saved to {output_path}")

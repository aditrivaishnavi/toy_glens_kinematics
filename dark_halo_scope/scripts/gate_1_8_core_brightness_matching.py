#!/usr/bin/env python3
"""
Gate 1.8: Core Brightness Matching Check

Verifies that positives and controls have matched core brightness distributions.
Since we inject onto the same LRGs, core brightness SHOULD match perfectly.
If it doesn't, there's a data generation bug.

Per LLM recommendation: "Core brightness matching by class (central aperture 
flux distribution for positives vs controls)."
"""
import numpy as np
import pyarrow.dataset as ds
import io
import json
from datetime import datetime, timezone
from scipy.stats import ks_2samp

RESULTS = {
    "gate": "1.8",
    "name": "Core Brightness Matching Check",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "description": "Verifies positives and controls have matched core brightness"
}

print("=" * 70)
print("GATE 1.8: CORE BRIGHTNESS MATCHING CHECK")
print("=" * 70)

# Configuration
CORE_RADIUS = 8  # pixels (same as our central aperture metric)
N_SAMPLES = 10000  # per class

def decode_stamp(blob):
    bio = io.BytesIO(blob)
    with np.load(bio) as z:
        g = z["image_g"].astype(np.float32)
        r = z["image_r"].astype(np.float32)
        zb = z["image_z"].astype(np.float32)
    return np.stack([g, r, zb], axis=0)

def compute_core_brightness(img, r_core=CORE_RADIUS):
    """Compute brightness metrics for central region."""
    C, H, W = img.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    core_mask = ((yy - cy)**2 + (xx - cx)**2) < r_core**2
    
    # r-band is index 1
    r_band = img[1]
    core_pixels = r_band[core_mask]
    
    return {
        'core_mean_r': float(np.mean(core_pixels)),
        'core_max_r': float(np.max(core_pixels)),
        'core_median_r': float(np.median(core_pixels)),
        'core_sum_r': float(np.sum(core_pixels))
    }

# Load data
data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

filt = (ds.field("region_split") == "train") & (ds.field("cutout_ok") == 1)
table = dataset.to_table(filter=filt, columns=["stamp_npz", "is_control"])

print(f"Total samples: {table.num_rows}")

# Sample by class
np.random.seed(42)
control_idx = [i for i in range(table.num_rows) if table["is_control"][i].as_py() == 1]
positive_idx = [i for i in range(table.num_rows) if table["is_control"][i].as_py() == 0]

ctrl_sample = np.random.choice(control_idx, min(N_SAMPLES, len(control_idx)), replace=False)
pos_sample = np.random.choice(positive_idx, min(N_SAMPLES, len(positive_idx)), replace=False)

print(f"Sampling {len(ctrl_sample)} controls and {len(pos_sample)} positives")

# Compute brightness for each class
def extract_brightness(table, indices, label):
    brightness = []
    for idx in indices:
        blob = table["stamp_npz"][int(idx)].as_py()
        if blob is None:
            continue
        try:
            img = decode_stamp(blob)
            if not np.isfinite(img).all():
                continue
            b = compute_core_brightness(img)
            brightness.append(b)
        except:
            continue
    return brightness

print("\nExtracting core brightness from controls...")
ctrl_brightness = extract_brightness(table, ctrl_sample, 0)
print(f"  {len(ctrl_brightness)} samples extracted")

print("Extracting core brightness from positives...")
pos_brightness = extract_brightness(table, pos_sample, 1)
print(f"  {len(pos_brightness)} samples extracted")

# Compare distributions
METRICS = ['core_mean_r', 'core_max_r', 'core_median_r', 'core_sum_r']

comparisons = []
any_failed = False

print("\n" + "=" * 70)
print("COMPARISON RESULTS:")
print("=" * 70)
print(f"{'Metric':<20} {'Ctrl Mean':<12} {'Pos Mean':<12} {'Ratio':<10} {'KS stat':<10} {'Status'}")
print("-" * 75)

for metric in METRICS:
    ctrl_vals = np.array([b[metric] for b in ctrl_brightness])
    pos_vals = np.array([b[metric] for b in pos_brightness])
    
    ctrl_mean = np.mean(ctrl_vals)
    pos_mean = np.mean(pos_vals)
    ratio = pos_mean / ctrl_mean if ctrl_mean != 0 else float('inf')
    
    # Effect size
    pooled_std = np.sqrt((np.var(ctrl_vals) + np.var(pos_vals)) / 2)
    cohens_d = abs(pos_mean - ctrl_mean) / (pooled_std + 1e-10)
    
    # KS test
    ks_stat, ks_pval = ks_2samp(ctrl_vals, pos_vals)
    
    # Pass if ratio is close to 1.0 (within 5%)
    passed = 0.95 <= ratio <= 1.05 and cohens_d < 0.1
    if not passed:
        any_failed = True
    
    status = "PASS" if passed else "FAIL"
    print(f"{metric:<20} {ctrl_mean:<12.6f} {pos_mean:<12.6f} {ratio:<10.3f} {ks_stat:<10.4f} {status}")
    
    comparisons.append({
        "metric": metric,
        "ctrl_mean": float(ctrl_mean),
        "ctrl_std": float(np.std(ctrl_vals)),
        "pos_mean": float(pos_mean),
        "pos_std": float(np.std(pos_vals)),
        "ratio": float(ratio),
        "cohens_d": float(cohens_d),
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval),
        "passed": passed
    })

# Also check distribution shapes
print("\n" + "=" * 70)
print("DISTRIBUTION PERCENTILES (core_mean_r):")
print("=" * 70)

ctrl_vals = np.array([b['core_mean_r'] for b in ctrl_brightness])
pos_vals = np.array([b['core_mean_r'] for b in pos_brightness])

percentiles = [5, 25, 50, 75, 95]
print(f"{'Percentile':<15} {'Controls':<15} {'Positives':<15} {'Ratio':<10}")
print("-" * 55)
for p in percentiles:
    ctrl_p = np.percentile(ctrl_vals, p)
    pos_p = np.percentile(pos_vals, p)
    ratio = pos_p / ctrl_p if ctrl_p != 0 else float('inf')
    print(f"{p}th%{'':<10} {ctrl_p:<15.6f} {pos_p:<15.6f} {ratio:<10.3f}")

RESULTS["comparisons"] = comparisons
RESULTS["n_controls"] = len(ctrl_brightness)
RESULTS["n_positives"] = len(pos_brightness)
RESULTS["overall_passed"] = not any_failed

print("\n" + "=" * 70)
print("GATE 1.8 CONCLUSION:")
print("=" * 70)

if any_failed:
    failed_metrics = [c["metric"] for c in comparisons if not c["passed"]]
    print(f"FAIL: Core brightness differs between classes!")
    print(f"Failed metrics: {failed_metrics}")
    print("This could be a data generation bug or indicate injection artifacts.")
    RESULTS["failed_metrics"] = failed_metrics
else:
    print("PASS: Core brightness matches between positives and controls")
    print("Injections do not alter the central galaxy brightness.")

# Save results
output_path = "/lambda/nfs/darkhaloscope-training-dc/gate_1_8_results.json"
with open(output_path, "w") as f:
    json.dump(RESULTS, f, indent=2)

print(f"\nResults saved to {output_path}")

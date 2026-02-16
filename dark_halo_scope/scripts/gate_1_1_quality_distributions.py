#!/usr/bin/env python3
"""
Gate 1.1: Class-conditional quality distribution check.
Verifies positives and controls are matched in data-quality space.
"""
import pyarrow.dataset as ds
import numpy as np
import json
from scipy.stats import ks_2samp
from datetime import datetime, timezone

RESULTS = {"gate": "1.1", "timestamp": datetime.now(timezone.utc).isoformat(), "checks": []}

data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

QUALITY_COLS = ["bad_pixel_frac", "cutout_ok", "physics_valid"]
cols_to_read = ["is_control", "region_split"] + QUALITY_COLS + ["arc_snr", "bandset"]

filt = (ds.field("region_split") == "train") & (ds.field("cutout_ok") == 1)
table = dataset.to_table(filter=filt, columns=cols_to_read)
df = table.to_pandas()

RESULTS["total_samples"] = len(df)
RESULTS["n_controls"] = int((df['is_control']==1).sum())
RESULTS["n_positives"] = int((df['is_control']==0).sum())

for col in QUALITY_COLS:
    if col not in df.columns:
        RESULTS["checks"].append({"column": col, "status": "MISSING"})
        continue
    
    ctrl = df[df['is_control']==1][col].dropna()
    pos = df[df['is_control']==0][col].dropna()
    
    if len(ctrl) < 2 or len(pos) < 2:
        RESULTS["checks"].append({
            "column": col,
            "status": "INSUFFICIENT_DATA",
            "ctrl_count": int(len(ctrl)),
            "pos_count": int(len(pos))
        })
        continue
    
    stat, pval = ks_2samp(ctrl, pos)
    passed = bool(pval > 0.01)
    
    RESULTS["checks"].append({
        "column": col,
        "ctrl_mean": float(ctrl.mean()),
        "ctrl_std": float(ctrl.std()),
        "pos_mean": float(pos.mean()),
        "pos_std": float(pos.std()),
        "ks_stat": float(stat),
        "ks_pval": float(pval),
        "passed": passed
    })

# arc_snr distribution for positives
pos_snr = df[df['is_control']==0]['arc_snr'].dropna()
RESULTS["arc_snr_distribution"] = {
    "mean": float(pos_snr.mean()),
    "median": float(pos_snr.median()),
    "frac_lt_2": float((pos_snr < 2).mean()),
    "frac_lt_5": float((pos_snr < 5).mean()),
    "frac_gt_20": float((pos_snr > 20).mean())
}

# Overall pass/fail
valid_checks = [c for c in RESULTS["checks"] if c.get("passed") is not None]
all_passed = bool(all(c.get("passed", False) for c in valid_checks)) if valid_checks else False
RESULTS["overall_passed"] = all_passed
RESULTS["total_samples"] = int(len(df))

# Save results
with open("gate_1_1_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print(json.dumps(RESULTS, indent=2))
print(f"\nGATE 1.1: {'PASS' if all_passed else 'FAIL'}")

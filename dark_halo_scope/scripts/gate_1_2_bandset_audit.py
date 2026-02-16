#!/usr/bin/env python3
"""
Gate 1.2: Bandset consistency audit.
Verifies all samples have consistent band coverage.
"""
import pyarrow.dataset as ds
import json
from datetime import datetime, timezone

RESULTS = {"gate": "1.2", "timestamp": datetime.now(timezone.utc).isoformat()}

data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

table = dataset.to_table(columns=["bandset", "is_control", "region_split"])
df = table.to_pandas()

RESULTS["total_samples"] = int(len(df))
bandset_counts = df['bandset'].value_counts().to_dict()
RESULTS["bandset_counts"] = {str(k): int(v) for k, v in bandset_counts.items()}

ctrl_bandset = df[df['is_control']==1]['bandset'].value_counts().to_dict()
pos_bandset = df[df['is_control']==0]['bandset'].value_counts().to_dict()
RESULTS["bandset_by_class"] = {
    "controls": {str(k): int(v) for k, v in ctrl_bandset.items()},
    "positives": {str(k): int(v) for k, v in pos_bandset.items()}
}

non_grz = df[df['bandset'] != 'grz']
RESULTS["non_grz_count"] = int(len(non_grz))
RESULTS["non_grz_by_class"] = {str(k): int(v) for k, v in non_grz.groupby('is_control').size().to_dict().items()} if len(non_grz) > 0 else {}

RESULTS["overall_passed"] = len(non_grz) == 0

with open("gate_1_2_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print(json.dumps(RESULTS, indent=2))
print(f"\nGATE 1.2: {'PASS' if RESULTS['overall_passed'] else 'FAIL'}")

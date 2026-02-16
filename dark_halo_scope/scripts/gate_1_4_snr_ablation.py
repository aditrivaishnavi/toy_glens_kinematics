#!/usr/bin/env python3
"""
Gate 1.4: Check if invvar is available for per-pixel SNR representation.
"""
import pyarrow.dataset as ds
import json
from datetime import datetime, timezone

RESULTS = {"gate": "1.4", "timestamp": datetime.now(timezone.utc).isoformat()}

data_path = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
dataset = ds.dataset(data_path, format="parquet", partitioning="hive")
schema_names = dataset.schema.names

RESULTS["schema_columns"] = schema_names
RESULTS["has_invvar_npz"] = "invvar_npz" in schema_names
RESULTS["has_stamp_invvar"] = "stamp_invvar" in schema_names

if RESULTS["has_invvar_npz"] or RESULTS["has_stamp_invvar"]:
    RESULTS["status"] = "AVAILABLE - implement SNR ablation"
    RESULTS["overall_passed"] = None  # Needs further testing
else:
    RESULTS["status"] = "DEFERRED - invvar not stored"
    RESULTS["recommendation"] = "Add invvar to Phase 4c for future runs"
    RESULTS["overall_passed"] = "DEFERRED"

with open("gate_1_4_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print(json.dumps(RESULTS, indent=2))

#!/usr/bin/env python3
"""Generate Stage 4c schema list for the Phase 5 contract from spark_phase4_pipeline_v3.py.

Usage:
  python generate_phase5_contract_from_pipeline.py \
    --pipeline spark_phase4_pipeline_v3.py \
    --out_json phase5_required_columns_contract.json \
    --out_py stage4c_schema_autogen.py

Notes:
- This script extracts the first yield Row(...) inside stage_4c_inject_cutouts().
- It updates only stage4c_metrics_schema_columns in the JSON file.
"""

import argparse, json, re
from pathlib import Path

def extract_stage4c_row_fields(py_text: str) -> list[str]:
    m = re.search(r"def\s+stage_4c_inject_cutouts\s*\(", py_text)
    if not m:
        raise RuntimeError("Could not find stage_4c_inject_cutouts()")
    start = m.start()
    m2 = re.search(r"\byield\s+Row\s*\(", py_text[start:])
    if not m2:
        raise RuntimeError("Could not find 'yield Row(' within stage_4c_inject_cutouts()")
    j = start + m2.end()
    depth = 1
    k = j
    while k < len(py_text) and depth > 0:
        ch = py_text[k]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        k += 1
    body = py_text[j:k-1]
    keys = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=", body)
    out, seen = [], set()
    for x in keys:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_py", required=True)
    args = ap.parse_args()

    txt = Path(args.pipeline).read_text()
    cols = extract_stage4c_row_fields(txt)

    out_json = Path(args.out_json)
    contract = json.loads(out_json.read_text()) if out_json.exists() else {}
    contract["source_pipeline_file"] = Path(args.pipeline).name
    contract["stage4c_metrics_schema_columns"] = cols
    out_json.write_text(json.dumps(contract, indent=2))
    print(f"Wrote: {out_json} (columns={len(cols)})")

    out_py = Path(args.out_py)
    out_py.write_text("# Auto-generated. Do not edit by hand.\nSTAGE4C_METRICS_SCHEMA_COLUMNS = " + repr(cols) + "\n")
    print(f"Wrote: {out_py}")

if __name__ == "__main__":
    main()

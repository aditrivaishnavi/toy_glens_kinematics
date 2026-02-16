from __future__ import annotations
import argparse, json, os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--id-col", default="galaxy_id", help="Use galaxy_id or cutout_path")
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest)
    required = ["cutout_path", "label", "split", "sample_weight"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    splits = {s: df[df["split"] == s] for s in ["train", "val", "test"] if s in df["split"].unique()}
    stats = {"rows": int(len(df)), "splits": {k: int(len(v)) for k,v in splits.items()}}
    stats["pos"] = int((df["label"] == 1).sum())
    stats["neg"] = int((df["label"] == 0).sum())

    id_col = args.id_col
    if id_col not in df.columns:
        raise SystemExit(f"id-col {id_col} not in manifest columns")
    ids = {k: set(v[id_col].astype(str).tolist()) for k,v in splits.items()}
    overlaps = {}
    keys = list(ids.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a, b = keys[i], keys[j]
            inter = ids[a].intersection(ids[b])
            overlaps[f"{a}∩{b}"] = int(len(inter))
    stats["overlaps"] = overlaps

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

if __name__ == "__main__":
    main()

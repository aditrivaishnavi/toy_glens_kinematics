from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def bootstrap_auc_and_recall(y: np.ndarray, s: np.ndarray, thresh: float, nboot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = y.size
    aucs = []
    recalls = []
    for _ in range(nboot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        ss = s[idx]
        # AUC requires both classes
        if np.unique(yy).size == 2:
            aucs.append(roc_auc_score(yy, ss))
        # recall
        tp = np.sum((ss >= thresh) & (yy == 1))
        fn = np.sum((ss < thresh) & (yy == 1))
        recalls.append(tp / max(tp + fn, 1))
    return np.asarray(aucs), np.asarray(recalls)

def ci(arr: np.ndarray, alpha: float):
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    lo = np.quantile(arr, alpha/2)
    hi = np.quantile(arr, 1-alpha/2)
    return (float(lo), float(np.median(arr)), float(hi))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Parquet with columns: y, score, tier")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--alpha", type=float, default=0.05, help="0.05 for 95% CI")
    args = ap.parse_args()

    df = pd.read_parquet(args.preds)
    if "y" not in df or "score" not in df:
        raise KeyError("preds parquet must contain y and score")
    y = df["y"].to_numpy().astype(int)
    s = df["score"].to_numpy().astype(float)

    aucs, recalls = bootstrap_auc_and_recall(y, s, args.thresh, args.nboot, args.seed)
    out = {
        "overall": {
            "auc_ci": ci(aucs, args.alpha),
            "recall_ci": ci(recalls, args.alpha),
            "n": int(y.size),
            "pos": int(np.sum(y==1)),
        }
    }
    if "tier" in df:
        for tier in sorted(df["tier"].fillna("").unique().tolist()):
            if tier == "": 
                continue
            dft = df[df["tier"] == tier]
            yt = dft["y"].to_numpy().astype(int)
            st = dft["score"].to_numpy().astype(float)
            aucs_t, rec_t = bootstrap_auc_and_recall(yt, st, args.thresh, args.nboot, args.seed+1)
            out[tier] = {"auc_ci": ci(aucs_t, args.alpha), "recall_ci": ci(rec_t, args.alpha), "n": int(yt.size), "pos": int(np.sum(yt==1))}
    import json, os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

if __name__ == "__main__":
    main()

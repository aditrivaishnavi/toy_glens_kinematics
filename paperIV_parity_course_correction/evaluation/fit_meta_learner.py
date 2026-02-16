from __future__ import annotations
import argparse, os, json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resnet-preds", required=True)
    ap.add_argument("--effnet-preds", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    r = pd.read_parquet(args.resnet_preds)
    e = pd.read_parquet(args.effnet_preds)

    # Align by row order assumption: both were produced from the same val manifest ordering.
    if len(r) != len(e):
        raise SystemExit("Prediction files have different lengths; need a join key to align.")
    y = r["y"].to_numpy().astype(int)
    X = np.vstack([r["score"].to_numpy(), e["score"].to_numpy()]).T

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    p = clf.predict_proba(X)[:,1]
    auc = roc_auc_score(y, p) if np.unique(y).size==2 else float("nan")

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "meta_learner.json"), "w") as f:
        json.dump({"coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(), "val_auc": float(auc)}, f, indent=2)
    print("meta-learner val AUC:", auc)

if __name__ == "__main__":
    main()

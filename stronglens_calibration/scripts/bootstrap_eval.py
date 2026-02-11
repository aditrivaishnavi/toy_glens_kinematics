#!/usr/bin/env python3
"""Bootstrap CIs for AUC, recall, precision on a split; optional by-tier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.manifest_utils import (
    CUTOUT_PATH_COL,
    LABEL_COL,
    SPLIT_COL,
    TIER_COL,
    load_manifest,
    safe_json_dump,
    sigmoid,
)


def load_preds(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if CUTOUT_PATH_COL not in df.columns:
        raise ValueError(f"Preds must contain {CUTOUT_PATH_COL}")
    if "score" not in df.columns and "logit" not in df.columns:
        raise ValueError("Preds must contain score or logit")
    if "score" not in df.columns:
        df["score"] = sigmoid(df["logit"].to_numpy(dtype=np.float64))
    # Deduplicate by cutout_path so merge does not inflate rows (keep last)
    n_before = len(df)
    df = df.drop_duplicates(subset=[CUTOUT_PATH_COL], keep="last")
    if len(df) < n_before:
        import warnings
        warnings.warn(
            f"Preds had {n_before - len(df)} duplicate cutout_path rows; kept last per path."
        )
    return df[[CUTOUT_PATH_COL, "score"]]


def metrics(y: np.ndarray, s: np.ndarray, thr: float) -> Dict:
    y = y.astype(int)
    auc = float(roc_auc_score(y, s)) if (y.min() != y.max()) else float("nan")
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    return {
        "auc": auc,
        "recall": float(recall),
        "precision": float(precision),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": int(len(y)),
        "pos": int((y == 1).sum()),
        "neg": int((y == 0).sum()),
    }


def bootstrap_ci(
    y: np.ndarray, s: np.ndarray, thr: float, iters: int, seed: int
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    rec = []
    prec = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        m = metrics(y[idx], s[idx], thr)
        if not np.isnan(m["auc"]):
            aucs.append(m["auc"])
        if not np.isnan(m["recall"]):
            rec.append(m["recall"])
        if not np.isnan(m["precision"]):
            prec.append(m["precision"])

    def q(arr, lo, hi):
        if len(arr) < 20:
            return (float("nan"), float("nan"))
        a = np.asarray(arr)
        return (float(np.quantile(a, lo)), float(np.quantile(a, hi)))

    return {
        "auc_68": q(aucs, 0.16, 0.84),
        "auc_95": q(aucs, 0.025, 0.975),
        "recall_68": q(rec, 0.16, 0.84),
        "recall_95": q(rec, 0.025, 0.975),
        "precision_68": q(prec, 0.16, 0.84),
        "precision_95": q(prec, 0.025, 0.975),
        "iters": iters,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--bootstrap_iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out_json", default="bootstrap_eval.json")
    args = ap.parse_args()

    man = load_manifest(args.manifest)
    preds = load_preds(args.preds)
    df = man.merge(preds, on=CUTOUT_PATH_COL, how="inner")
    df = df[df[SPLIT_COL] == args.split].copy()
    if df.empty:
        raise ValueError("No rows after merge for split")
    y = df[LABEL_COL].to_numpy(np.int64)
    s = df["score"].to_numpy(np.float64)

    out = {
        "manifest": str(Path(args.manifest).resolve()),
        "preds": str(Path(args.preds).resolve()),
        "split": args.split,
        "threshold": args.threshold,
        "overall": metrics(y, s, args.threshold),
        "overall_bootstrap": bootstrap_ci(
            y, s, args.threshold, args.bootstrap_iters, args.seed
        ),
        "n_merged": int(len(df)),
        "n_manifest": int(len(man)),
    }
    if TIER_COL in df.columns:
        out["by_tier"] = {}
        for t in sorted(df[TIER_COL].dropna().unique().tolist()):
            dft = df[df[TIER_COL] == t]
            yt = dft[LABEL_COL].to_numpy(np.int64)
            st = dft["score"].to_numpy(np.float64)
            out["by_tier"][str(t)] = {
                "metrics": metrics(yt, st, args.threshold),
                "bootstrap": bootstrap_ci(
                    yt, st, args.threshold, min(args.bootstrap_iters, 5000), args.seed + 17
                ),
            }
    safe_json_dump(out, args.out_json)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

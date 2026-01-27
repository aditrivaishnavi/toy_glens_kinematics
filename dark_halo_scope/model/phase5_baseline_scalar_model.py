#!/usr/bin/env python3
"""
Phase 5: Metadata-only baseline model (logistic regression).

Purpose:
- Provide a simple, interpretable baseline that predicts whether a stamp will be
  flagged by the trained CNN, using only scalar metadata (no pixels).
- This baseline is useful for:
  - sanity checks (if the baseline matches the CNN too well, the CNN may be
    exploiting a metadata leak)
  - interpretability and ablation discussion in a paper

Inputs:
- A Phase 5 score table produced by phase5_infer_scores.py (parquet dataset).
  This table carries both metadata (Phase 4c columns) and model_score.

Output:
- A trained sklearn LogisticRegression model (joblib)
- A metrics JSON file (AUROC, accuracy, calibration summary)

Example:
  python phase5_baseline_scalar_model.py \
    --scores "s3://.../phase5/scores/resnet18_v1/all_scores" \
    --out_dir "s3://.../phase5/baselines/resnet18_v1" \
    --label_from_scores \
    --score_threshold 0.5 \
    --features theta_e_arcsec,src_dmag,src_reff_arcsec,psf_fwhm_used_r,psfdepth_r,bad_pixel_frac,wise_brightmask_frac
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import numpy as np

try:
    import fsspec
except Exception as e:
    raise RuntimeError("Missing dependency fsspec. Install: pip install fsspec s3fs") from e

try:
    import pyarrow.dataset as ds
except Exception as e:
    raise RuntimeError("Missing dependency pyarrow. Install: pip install pyarrow") from e

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except Exception as e:
    raise RuntimeError("Missing dependency scikit-learn. Install: pip install scikit-learn") from e

try:
    import joblib
except Exception as e:
    raise RuntimeError("Missing dependency joblib. Install: pip install joblib") from e


def _is_s3(path: str) -> bool:
    return path.startswith("s3://")


def _read_dataset(path: str, columns: List[str], max_rows: Optional[int] = None):
    if _is_s3(path):
        filesystem = fsspec.filesystem("s3")
        dset = ds.dataset(path, format="parquet", filesystem=filesystem)
    else:
        dset = ds.dataset(path, format="parquet")
    table = dset.to_table(columns=columns)
    if max_rows is not None and table.num_rows > max_rows:
        table = table.slice(0, max_rows)
    return table


def _write_bytes(path: str, data: bytes):
    if _is_s3(path):
        with fsspec.open(path, "wb") as f:
            f.write(data)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="Parquet dataset from phase5_infer_scores.py")
    ap.add_argument("--out_dir", required=True, help="Output directory (local or s3://)")
    ap.add_argument("--features", default="", help="Comma-separated feature list. If empty, uses a safe default.")
    ap.add_argument("--max_rows", type=int, default=0, help="Dev cap; 0 means no cap")
    ap.add_argument("--label_from_scores", action="store_true", help="Label is (model_score >= score_threshold)")
    ap.add_argument("--score_threshold", type=float, default=0.5)
    ap.add_argument("--label_from_proxy", action="store_true", help="Label is proxy (arc_snr>=snr_th AND theta_over_psf>=sep_th)")
    ap.add_argument("--snr_th", type=float, default=5.0)
    ap.add_argument("--sep_th", type=float, default=0.8)
    ap.add_argument("--train_split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--test_split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    if args.label_from_scores == args.label_from_proxy:
        raise SystemExit("Choose exactly one label source: --label_from_scores OR --label_from_proxy")

    default_feats = [
        "theta_e_arcsec",
        "src_dmag",
        "src_reff_arcsec",
        "psf_fwhm_used_r",
        "psfsize_r",
        "psfdepth_r",
        "bad_pixel_frac",
        "wise_brightmask_frac",
    ]
    feats = [f.strip() for f in args.features.split(",") if f.strip()] if args.features else default_feats

    # Columns required for labeling and splits
    cols = list(set(feats + ["region_split", "model_score", "arc_snr", "theta_over_psf", "lens_model", "y_true"]))
    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None
    tab = _read_dataset(args.scores, columns=cols, max_rows=max_rows)

    df = tab.to_pandas()

    # Focus on injections if available, for "completeness-like" modeling
    if "y_true" in df.columns and df["y_true"].notnull().any():
        df = df[df["y_true"] == 1]
    elif "lens_model" in df.columns and df["lens_model"].notnull().any():
        df = df[df["lens_model"] != "CONTROL"]

    # Label
    if args.label_from_scores:
        if "model_score" not in df.columns:
            raise SystemExit("model_score column not found in input.")
        y = (df["model_score"].astype(float) >= args.score_threshold).astype(int).values
    else:
        if "arc_snr" not in df.columns or "theta_over_psf" not in df.columns:
            raise SystemExit("arc_snr and theta_over_psf required for proxy labeling.")
        y = ((df["arc_snr"].astype(float) >= args.snr_th) & (df["theta_over_psf"].astype(float) >= args.sep_th)).astype(int).values

    # Features
    X = df[feats].copy()
    for c in feats:
        X[c] = X[c].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    # Splits
    if "region_split" not in df.columns:
        raise SystemExit("region_split column not found; cannot split.")
    split = df["region_split"].astype(str).values
    tr = split == args.train_split
    te = split == args.test_split

    if tr.sum() < 1000 or te.sum() < 1000:
        print(f"[WARN] small split sizes: train={tr.sum()} test={te.sum()}")

    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]

    # Model
    clf = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            max_iter=200,
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=None,
        )),
    ])
    clf.fit(X_train, y_train)

    # Evaluate
    p_test = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else float("nan")
    acc = accuracy_score(y_test, (p_test >= 0.5).astype(int))
    brier = brier_score_loss(y_test, p_test)

    metrics = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "features": feats,
        "label_from_scores": bool(args.label_from_scores),
        "label_from_proxy": bool(args.label_from_proxy),
        "score_threshold": float(args.score_threshold),
        "snr_th": float(args.snr_th),
        "sep_th": float(args.sep_th),
        "test_auc": float(auc),
        "test_accuracy": float(acc),
        "test_brier": float(brier),
    }

    # Serialize
    out_dir = args.out_dir.rstrip("/")
    model_path = out_dir + "/baseline_logreg.joblib"
    metrics_path = out_dir + "/baseline_metrics.json"

    import io
    bio = io.BytesIO()
    joblib.dump(clf, bio)
    _write_bytes(model_path, bio.getvalue())
    _write_bytes(metrics_path, json.dumps(metrics, indent=2).encode("utf-8"))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

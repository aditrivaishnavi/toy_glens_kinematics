
#!/usr/bin/env python3
"""
Evaluate FPR vs completeness (TPR) overall and stratified by bins from a Phase 5 score table.

Input: parquet(s) with at least
- y_true (0/1) or is_control (1 for control, 0 for injection)
- score (float, higher = more lens-like)
- theta_e_arcsec (for injections; controls often 0)
- psf_fwhm_used_r and/or psfsize_r

Outputs:
- A CSV summary printed to stdout (and optionally written to --out_csv)
- For each stratum, reports FPR at specified completeness levels (TPR targets)
- Also reports completeness at specified FPR levels (useful for "how much TPR at 1e-4").

This is designed to support paper-quality figures and tables.
"""

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _as_numpy(arr):
    if hasattr(arr, "to_numpy"):
        return arr.to_numpy(zero_copy_only=False)
    return np.asarray(arr)


def _roc_points(scores: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return arrays (fpr, tpr, thr) for all unique thresholds.
    Uses a stable, vectorized implementation without sklearn.
    """
    # Sort by score desc
    order = np.argsort(-scores, kind="mergesort")
    scores_sorted = scores[order]
    y_sorted = y[order].astype(np.int64)

    P = y_sorted.sum()
    N = len(y_sorted) - P
    if P == 0 or N == 0:
        return np.array([]), np.array([]), np.array([])

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)

    # Take points at score changes
    distinct = np.r_[True, scores_sorted[1:] != scores_sorted[:-1]]
    tp = tp[distinct]
    fp = fp[distinct]
    thr = scores_sorted[distinct]

    tpr = tp / P
    fpr = fp / N
    return fpr, tpr, thr


def _fpr_at_tpr(fpr, tpr, thr, tpr_targets: List[float]) -> List[Tuple[float, float]]:
    out = []
    for tt in tpr_targets:
        # Find first index where tpr >= tt (as threshold decreases).
        idx = np.searchsorted(tpr, tt, side="left")
        if idx >= len(tpr):
            out.append((math.nan, math.nan))
        else:
            out.append((float(fpr[idx]), float(thr[idx])))
    return out


def _tpr_at_fpr(fpr, tpr, thr, fpr_targets: List[float]) -> List[Tuple[float, float]]:
    out = []
    for ft in fpr_targets:
        idx = np.searchsorted(fpr, ft, side="right") - 1
        if idx < 0:
            out.append((0.0, float(thr[0]) if len(thr) else math.nan))
        else:
            out.append((float(tpr[idx]), float(thr[idx])))
    return out


def _bin_edges(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _compute_resolution(theta, psf):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = theta / psf
    r[~np.isfinite(r)] = np.nan
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="Parquet directory/file with per-row scores")
    ap.add_argument("--score_col", default="score", help="Score column name")
    ap.add_argument("--label_col", default="y_true", help="Label column (0/1). If absent, will derive from is_control.")
    ap.add_argument("--is_control_col", default="is_control", help="Fallback label column: 1=control, 0=injection")
    ap.add_argument("--theta_col", default="theta_e_arcsec")
    ap.add_argument("--psf_used_col", default="psf_fwhm_used_r")
    ap.add_argument("--psf_manifest_col", default="psfsize_r")
    ap.add_argument("--tpr_targets", default="0.99,0.95,0.90,0.85,0.80,0.70,0.50")
    ap.add_argument("--fpr_targets", default="1e-1,1e-2,1e-3,1e-4,1e-5")
    ap.add_argument("--theta_bins", default="0.0,0.5,0.75,1.0,1.25,1.5,2.0,3.0")
    ap.add_argument("--res_bins", default="0.0,0.4,0.6,0.8,1.0,1.5,2.0,99")
    ap.add_argument("--max_rows", type=int, default=0, help="Optional cap for quick runs (0 = no cap)")
    ap.add_argument("--out_csv", default="", help="Optional output CSV path")
    args = ap.parse_args()

    tpr_targets = _bin_edges(args.tpr_targets)
    fpr_targets = _bin_edges(args.fpr_targets)
    theta_bins = _bin_edges(args.theta_bins)
    res_bins = _bin_edges(args.res_bins)

    dataset = ds.dataset(args.scores, format="parquet")
    cols = [args.score_col, args.theta_col, args.psf_used_col, args.psf_manifest_col, args.label_col, args.is_control_col]
    # Only keep existing columns
    cols = [c for c in cols if c in dataset.schema.names]
    if args.score_col not in cols:
        raise ValueError(f"Missing score column: {args.score_col}")

    rows = []
    seen = 0

    # Load in one pass (for full dataset you may want to pre-aggregate; this is for review runs).
    # If the dataset is huge, use --max_rows or sample.
    for batch in dataset.to_batches(columns=cols, batch_size=200_000):
        if args.max_rows and seen >= args.max_rows:
            break
        take = len(batch)
        if args.max_rows:
            take = min(take, args.max_rows - seen)
            batch = batch.slice(0, take)
        seen += take

        scores = _as_numpy(batch[args.score_col]).astype(np.float64)

        if args.label_col in batch.schema.names:
            y = _as_numpy(batch[args.label_col]).astype(np.int64)
        elif args.is_control_col in batch.schema.names:
            is_ctrl = _as_numpy(batch[args.is_control_col]).astype(np.int64)
            y = (1 - is_ctrl).astype(np.int64)
        else:
            raise ValueError("Need either label_col or is_control_col")

        theta = _as_numpy(batch[args.theta_col]).astype(np.float64) if args.theta_col in batch.schema.names else np.full_like(scores, np.nan)
        psf_used = _as_numpy(batch[args.psf_used_col]).astype(np.float64) if args.psf_used_col in batch.schema.names else np.full_like(scores, np.nan)
        psf_man = _as_numpy(batch[args.psf_manifest_col]).astype(np.float64) if args.psf_manifest_col in batch.schema.names else np.full_like(scores, np.nan)
        psf = np.where(np.isfinite(psf_used) & (psf_used > 0), psf_used, psf_man)
        res = _compute_resolution(theta, psf)

        rows.append((scores, y, theta, res))

    if not rows:
        raise RuntimeError("No data loaded")

    scores = np.concatenate([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    theta = np.concatenate([r[2] for r in rows])
    res = np.concatenate([r[3] for r in rows])

    def strata_masks():
        # overall first
        yield ("ALL", np.ones_like(y, dtype=bool))
        # theta bins (only meaningful for injections; keep controls in each stratum as negatives)
        for i in range(len(theta_bins) - 1):
            lo, hi = theta_bins[i], theta_bins[i+1]
            m = (theta >= lo) & (theta < hi)
            yield (f"theta[{lo},{hi})", m | (y == 0)  # include all controls as negatives
        for i in range(len(res_bins) - 1):
            lo, hi = res_bins[i], res_bins[i+1]
            m = (res >= lo) & (res < hi)
            yield (f"res[{lo},{hi})", m | (y == 0)

    lines = []
    header = ["stratum", "n", "P", "N"]
    for tt in tpr_targets:
        header += [f"FPR@TPR{tt:.2f}", f"thr@TPR{tt:.2f}"]
    for ft in fpr_targets:
        header += [f"TPR@FPR{ft:g}", f"thr@FPR{ft:g}"]
    lines.append(",".join(header))

    for name, m in strata_masks():
        idx = np.where(m)[0]
        if len(idx) < 1000:
            continue
        s = scores[idx]
        yy = y[idx]
        P = int(yy.sum())
        N = int(len(yy) - P)
        if P == 0 or N == 0:
            continue

        fpr, tpr, thr = _roc_points(s, yy)
        if len(fpr) == 0:
            continue

        # Ensure monotonic for searchsorted (fpr and tpr both non-decreasing with descending threshold)
        # Our construction already yields monotonic.

        fpr_at = _fpr_at_tpr(fpr, tpr, thr, tpr_targets)
        tpr_at = _tpr_at_fpr(fpr, tpr, thr, fpr_targets)

        row = [name, str(len(idx)), str(P), str(N)]
        for (v, th) in fpr_at:
            row += [f"{v:.6g}", f"{th:.6g}"]
        for (v, th) in tpr_at:
            row += [f"{v:.6g}", f"{th:.6g}"]
        lines.append(",".join(row))

    out = "\n".join(lines)
    print(out)

    if args.out_csv:
        with open(args.out_csv, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()

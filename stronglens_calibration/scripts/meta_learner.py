#!/usr/bin/env python3
"""
Meta-Learner: Feature-Weighted Stacking (Coscrato et al. 2020).

Implements the 1-layer neural network with 300 hidden nodes used by
Inchausti et al. (2025, Paper IV) to combine ResNet + EfficientNetV2
predictions into an ensemble score.

Paper IV achieves:
  - ResNet AUC:       0.9984 (val)
  - EfficientNet AUC: 0.9987 (val)
  - Ensemble AUC:     0.9989 (val, meta-learner)

Architecture (per Coscrato et al. 2020):
  Input(2) -> Linear(2, 300) -> ReLU -> Linear(300, 1)

The meta-learner is trained on the *training split* predictions from both
base models, and evaluated on the *validation* (70/30) or *test* (70/15/15)
split. This avoids data leakage: the base model predictions on training data
are obtained via the model trained on that same data, so they may be
overconfident, but the meta-learner learns to calibrate and combine them.

For a more rigorous approach (not used in Paper IV), one could use
cross-validated predictions on the training set (k-fold stacking).

Usage:
    cd /lambda/nfs/.../code
    export PYTHONPATH=.

    # 1. First, generate predictions from both models using evaluate_parity.py:
    python scripts/evaluate_parity.py \\
        --checkpoint checkpoints/paperIV_resnet18/best.pt \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --split train --export-predictions results/preds_resnet18_train.parquet
    python scripts/evaluate_parity.py \\
        --checkpoint checkpoints/paperIV_resnet18/best.pt \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --split val --export-predictions results/preds_resnet18_val.parquet
    python scripts/evaluate_parity.py \\
        --checkpoint checkpoints/paperIV_efficientnet_v2_s/best.pt \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --split train --export-predictions results/preds_efficientnet_train.parquet
    python scripts/evaluate_parity.py \\
        --checkpoint checkpoints/paperIV_efficientnet_v2_s/best.pt \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --split val --export-predictions results/preds_efficientnet_val.parquet

    # 2. Train the meta-learner:
    python scripts/meta_learner.py \\
        --preds-a-train results/preds_resnet18_train.parquet \\
        --preds-b-train results/preds_efficientnet_train.parquet \\
        --preds-a-eval results/preds_resnet18_val.parquet \\
        --preds-b-eval results/preds_efficientnet_val.parquet \\
        --output results/meta_learner_results.json \\
        --export-predictions results/preds_ensemble_val.parquet \\
        --save-model results/meta_learner.pt

Author: stronglens_calibration project
Date: 2026-02-11
References:
  - Coscrato et al. 2020, "Feature-weighted stacking"
  - Inchausti et al. 2025, Section 3.3 (Paper IV)
  - MNRAS_RAW_NOTES.md Section 7.4 and 15.2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CUTOUT_PATH_COL = "cutout_path"
LABEL_COL = "label"
SPLIT_COL = "split"
TIER_COL = "tier"


# ---------------------------------------------------------------------------
# Meta-learner architecture
# ---------------------------------------------------------------------------
class MetaLearner(nn.Module):
    """1-layer neural network for combining base model predictions.

    Architecture (Coscrato et al. 2020 / Paper IV):
        Input(n_models) -> Linear(n_models, hidden) -> ReLU -> Linear(hidden, 1)

    Default: n_models=2, hidden=300 (Paper IV spec).
    """

    def __init__(self, n_models: int = 2, hidden: int = 300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_models, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, n_models) base model scores. Returns (N, 1) logits."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading and merging
# ---------------------------------------------------------------------------
def load_predictions(path: str) -> pd.DataFrame:
    """Load predictions from parquet or CSV."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if CUTOUT_PATH_COL not in df.columns:
        raise ValueError(f"Predictions must contain '{CUTOUT_PATH_COL}'")
    if "score" not in df.columns:
        raise ValueError("Predictions must contain 'score'")

    return df


def merge_predictions(
    preds_a: pd.DataFrame,
    preds_b: pd.DataFrame,
    name_a: str = "score_a",
    name_b: str = "score_b",
) -> pd.DataFrame:
    """Merge two prediction DataFrames on cutout_path.

    Both must have: cutout_path, score, label.
    Returns DataFrame with: cutout_path, score_a, score_b, label, and any
    shared metadata columns (tier, pool, split, etc.).
    """
    # Rename score columns
    a = preds_a.rename(columns={"score": name_a})
    b = preds_b.rename(columns={"score": name_b})

    # Merge on cutout_path
    cols_a = [CUTOUT_PATH_COL, name_a]
    cols_b = [CUTOUT_PATH_COL, name_b]

    # Keep label and metadata from preds_a
    meta_cols = [c for c in [LABEL_COL, TIER_COL, SPLIT_COL, "pool", "confuser_category"]
                 if c in a.columns]
    cols_a.extend(meta_cols)

    merged = a[cols_a].merge(b[cols_b], on=CUTOUT_PATH_COL, how="inner")

    if len(merged) < min(len(a), len(b)):
        print(f"  WARNING: Merged {len(merged)} rows from {len(a)} (A) x {len(b)} (B). "
              f"Some cutout_paths don't match.", file=sys.stderr)

    return merged


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_meta_learner(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_models: int = 2,
    hidden: int = 300,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 512,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device_str: str = "cpu",
) -> Tuple[MetaLearner, Dict[str, Any]]:
    """Train the meta-learner on base model predictions.

    Args:
        X_train: (N, n_models) base model scores
        y_train: (N,) binary labels
        n_models: number of input features (base models)
        hidden: hidden layer size
        epochs: training epochs
        lr: learning rate
        batch_size: mini-batch size
        weight_decay: L2 regularization
        seed: random seed
        device_str: device

    Returns:
        (trained model, training history dict)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device_str)
    model = MetaLearner(n_models=n_models, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    X = torch.from_numpy(X_train).float()
    y = torch.from_numpy(y_train).float().unsqueeze(1)

    n = len(X)
    history = {"train_loss": [], "train_auc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            xb = X[idx].to(device)
            yb = y[idx].to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Compute training AUC
        model.eval()
        with torch.no_grad():
            logits_all = model(X.to(device)).squeeze(1).cpu().numpy()
            scores_all = 1.0 / (1.0 + np.exp(-logits_all))
            train_auc = float(roc_auc_score(y_train, scores_all))

        history["train_loss"].append(avg_loss)
        history["train_auc"].append(train_auc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [epoch {epoch:3d}] loss={avg_loss:.6f} train_auc={train_auc:.6f}")

    model.eval()
    return model, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_meta_learner(
    model: MetaLearner, X: np.ndarray, device_str: str = "cpu"
) -> np.ndarray:
    """Score samples with the meta-learner. Returns sigmoid scores."""
    device = torch.device(device_str)
    model.eval()
    xt = torch.from_numpy(X).float().to(device)
    logits = model(xt).squeeze(1).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))


def bootstrap_auc(
    y_true: np.ndarray, y_score: np.ndarray, n_bootstrap: int = 10000, seed: int = 42
) -> Dict[str, Any]:
    """Bootstrap CI for AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.min() == yt.max():
            continue
        aucs.append(float(roc_auc_score(yt, ys)))
    a = np.asarray(aucs)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "ci68": (float(np.quantile(a, 0.16)), float(np.quantile(a, 0.84))),
        "ci95": (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))),
        "n_valid": len(aucs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Meta-Learner: 1-layer NN (300 hidden nodes) per Coscrato et al. 2020",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--preds-a-train", required=True,
                    help="Model A (e.g. ResNet-18) training split predictions (parquet/csv)")
    ap.add_argument("--preds-b-train", required=True,
                    help="Model B (e.g. EfficientNetV2-S) training split predictions (parquet/csv)")
    ap.add_argument("--preds-a-eval", required=True,
                    help="Model A eval split predictions (parquet/csv)")
    ap.add_argument("--preds-b-eval", required=True,
                    help="Model B eval split predictions (parquet/csv)")
    ap.add_argument("--hidden", type=int, default=300,
                    help="Hidden layer size (default: 300, per Paper IV)")
    ap.add_argument("--epochs", type=int, default=100,
                    help="Meta-learner training epochs (default: 100)")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate (default: 1e-3)")
    ap.add_argument("--batch-size", type=int, default=512,
                    help="Mini-batch size (default: 512)")
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="L2 regularization (default: 1e-4)")
    ap.add_argument("--n-bootstrap", type=int, default=10000,
                    help="Bootstrap resamples for AUC CI (default: 10000)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--device", default="cpu",
                    help="Device (default: cpu; meta-learner is tiny)")
    ap.add_argument("--output", default=None,
                    help="Output JSON path (default: stdout)")
    ap.add_argument("--export-predictions", default=None,
                    help="Export ensemble predictions (parquet/csv)")
    ap.add_argument("--save-model", default=None,
                    help="Save meta-learner state dict (.pt)")
    args = ap.parse_args()

    print("=" * 60)
    print("Meta-Learner: Feature-Weighted Stacking")
    print("  Coscrato et al. 2020 / Paper IV (Inchausti et al. 2025)")
    print(f"  Hidden: {args.hidden}, Epochs: {args.epochs}, LR: {args.lr}")
    print("=" * 60)

    # Load and merge training predictions
    print("\nLoading training predictions...")
    preds_a_train = load_predictions(args.preds_a_train)
    preds_b_train = load_predictions(args.preds_b_train)
    train_merged = merge_predictions(preds_a_train, preds_b_train)
    print(f"  Training samples: {len(train_merged):,}")

    # Load and merge eval predictions
    print("Loading eval predictions...")
    preds_a_eval = load_predictions(args.preds_a_eval)
    preds_b_eval = load_predictions(args.preds_b_eval)
    eval_merged = merge_predictions(preds_a_eval, preds_b_eval)
    print(f"  Eval samples: {len(eval_merged):,}")

    # Prepare arrays
    X_train = train_merged[["score_a", "score_b"]].values.astype(np.float32)
    y_train = train_merged[LABEL_COL].values.astype(np.float32)
    X_eval = eval_merged[["score_a", "score_b"]].values.astype(np.float32)
    y_eval = eval_merged[LABEL_COL].values.astype(np.float32)

    # Individual model AUCs on eval set
    auc_a = float(roc_auc_score(y_eval, eval_merged["score_a"].values))
    auc_b = float(roc_auc_score(y_eval, eval_merged["score_b"].values))
    print(f"\nIndividual model AUCs (eval):")
    print(f"  Model A: {auc_a:.6f}")
    print(f"  Model B: {auc_b:.6f}")

    # Simple average baseline
    avg_scores = 0.5 * (eval_merged["score_a"].values + eval_merged["score_b"].values)
    auc_avg = float(roc_auc_score(y_eval, avg_scores))
    print(f"  Simple average: {auc_avg:.6f}")

    # Train meta-learner
    print(f"\nTraining meta-learner ({args.epochs} epochs)...")
    t0 = time.time()
    model, history = train_meta_learner(
        X_train, y_train,
        n_models=2,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device_str=args.device,
    )
    dt = time.time() - t0
    print(f"  Training time: {dt:.1f}s")

    # Evaluate meta-learner
    print("\nEvaluating meta-learner on eval set...")
    ensemble_scores = score_meta_learner(model, X_eval, args.device)
    auc_ensemble = float(roc_auc_score(y_eval, ensemble_scores))
    print(f"  Ensemble AUC: {auc_ensemble:.6f}")

    # Bootstrap CIs
    print(f"\nBootstrap CIs ({args.n_bootstrap:,} resamples)...")
    boot_a = bootstrap_auc(y_eval, eval_merged["score_a"].values, args.n_bootstrap, args.seed)
    boot_b = bootstrap_auc(y_eval, eval_merged["score_b"].values, args.n_bootstrap, args.seed + 1)
    boot_avg = bootstrap_auc(y_eval, avg_scores, args.n_bootstrap, args.seed + 2)
    boot_ens = bootstrap_auc(y_eval, ensemble_scores, args.n_bootstrap, args.seed + 3)

    # Assemble results
    results = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hidden": args.hidden,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "n_train": int(len(train_merged)),
            "n_eval": int(len(eval_merged)),
            "n_bootstrap": args.n_bootstrap,
            "preds_a_train": args.preds_a_train,
            "preds_b_train": args.preds_b_train,
            "preds_a_eval": args.preds_a_eval,
            "preds_b_eval": args.preds_b_eval,
            "reference": "Coscrato et al. 2020; Inchausti et al. 2025 Section 3.3",
        },
        "eval_auc": {
            "model_a": auc_a,
            "model_b": auc_b,
            "simple_average": auc_avg,
            "meta_learner": auc_ensemble,
        },
        "eval_bootstrap": {
            "model_a": boot_a,
            "model_b": boot_b,
            "simple_average": boot_avg,
            "meta_learner": boot_ens,
        },
        "paper_iv_comparison": {
            "paper_iv_resnet_auc": 0.9984,
            "paper_iv_efficientnet_auc": 0.9987,
            "paper_iv_ensemble_auc": 0.9989,
            "our_model_a_auc": auc_a,
            "our_model_b_auc": auc_b,
            "our_ensemble_auc": auc_ensemble,
            "ensemble_gain_over_best_individual": auc_ensemble - max(auc_a, auc_b),
        },
        "training_history": {
            "final_train_loss": history["train_loss"][-1],
            "final_train_auc": history["train_auc"][-1],
        },
        "model_weights": {
            "layer_0_weight_shape": list(model.net[0].weight.shape),
            "layer_0_bias_shape": list(model.net[0].bias.shape),
            "layer_2_weight_shape": list(model.net[2].weight.shape),
            "n_params": sum(p.numel() for p in model.parameters()),
        },
    }

    # Per-tier AUC (if available)
    if TIER_COL in eval_merged.columns:
        tier_results = {}
        for tier_val in sorted(eval_merged[TIER_COL].dropna().unique()):
            mask = eval_merged[TIER_COL] == tier_val
            yt = y_eval[mask.values]
            if yt.min() == yt.max() or mask.sum() < 10:
                continue
            tier_results[f"tier_{tier_val}"] = {
                "n": int(mask.sum()),
                "model_a_auc": float(roc_auc_score(yt, eval_merged.loc[mask, "score_a"].values)),
                "model_b_auc": float(roc_auc_score(yt, eval_merged.loc[mask, "score_b"].values)),
                "ensemble_auc": float(roc_auc_score(yt, ensemble_scores[mask.values])),
            }
        results["by_tier"] = tier_results

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n" + json.dumps(results, indent=2, default=str))

    # Export ensemble predictions
    if args.export_predictions:
        pred_df = pd.DataFrame({
            CUTOUT_PATH_COL: eval_merged[CUTOUT_PATH_COL].values,
            "score_a": eval_merged["score_a"].values,
            "score_b": eval_merged["score_b"].values,
            "score_ensemble": ensemble_scores,
            LABEL_COL: y_eval.astype(int),
        })
        for col in [TIER_COL, SPLIT_COL, "pool", "confuser_category"]:
            if col in eval_merged.columns:
                pred_df[col] = eval_merged[col].values

        os.makedirs(os.path.dirname(args.export_predictions) or ".", exist_ok=True)
        if args.export_predictions.endswith(".parquet"):
            pred_df.to_parquet(args.export_predictions, index=False)
        else:
            pred_df.to_csv(args.export_predictions, index=False)
        print(f"Exported {len(pred_df):,} ensemble predictions to: {args.export_predictions}")

    # Save model weights
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "config": {
                "n_models": 2,
                "hidden": args.hidden,
            },
            "training": {
                "epochs": args.epochs,
                "lr": args.lr,
                "seed": args.seed,
                "final_train_auc": history["train_auc"][-1],
            },
            "eval_auc": auc_ensemble,
        }, args.save_model)
        print(f"Model saved to: {args.save_model}")

    # Summary
    print("\n" + "=" * 60)
    print("META-LEARNER SUMMARY")
    print("=" * 60)
    print(f"  Architecture: Input(2) -> Linear(2, {args.hidden}) -> ReLU -> Linear({args.hidden}, 1)")
    print(f"  Parameters:   {sum(p.numel() for p in model.parameters()):,}")
    print()
    print(f"  Model A AUC:       {auc_a:.6f}  95% CI [{boot_a['ci95'][0]:.6f}, {boot_a['ci95'][1]:.6f}]")
    print(f"  Model B AUC:       {auc_b:.6f}  95% CI [{boot_b['ci95'][0]:.6f}, {boot_b['ci95'][1]:.6f}]")
    print(f"  Simple avg AUC:    {auc_avg:.6f}  95% CI [{boot_avg['ci95'][0]:.6f}, {boot_avg['ci95'][1]:.6f}]")
    print(f"  Meta-learner AUC:  {auc_ensemble:.6f}  95% CI [{boot_ens['ci95'][0]:.6f}, {boot_ens['ci95'][1]:.6f}]")
    print()
    print(f"  Ensemble gain over best individual: {auc_ensemble - max(auc_a, auc_b):+.6f}")
    print()
    print(f"  Paper IV ensemble AUC: 0.9989")
    print("=" * 60)


if __name__ == "__main__":
    main()

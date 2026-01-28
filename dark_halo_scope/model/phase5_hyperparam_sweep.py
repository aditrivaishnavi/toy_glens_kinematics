#!/usr/bin/env python3
"""
Phase 5: Hyperparameter sweep using Optuna.

Performs Bayesian optimization over:
- Learning rate
- Weight decay
- Dropout
- Architecture
- Batch size

Usage:
  python phase5_hyperparam_sweep.py \
    --data /data/phase4c/stamps/train_stamp64... \
    --contract_json dark_halo_scope/model/phase5_required_columns_contract.json \
    --out_dir /data/phase5/sweeps/sweep_v1 \
    --n_trials 20 \
    --epochs_per_trial 3 \
    --steps_per_epoch 1000

This runs quickly because:
- Fewer epochs per trial (3 vs 20)
- Fewer steps per epoch (1000 vs 5000)
- Optuna prunes unpromising trials early
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import optuna
    from optuna.trial import Trial
except ImportError:
    raise RuntimeError("pip install optuna")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Import from production training script
from phase5_train_production import (
    Phase4cDataset, build_model, TrainingConfig, Metrics,
    CosineWarmupScheduler, _list_parquet_files
)

# Cache
try:
    from .data_cache import DataCache
    HAS_CACHE = True
except ImportError:
    try:
        from dark_halo_scope.model.data_cache import DataCache
        HAS_CACHE = True
    except ImportError:
        try:
            from data_cache import DataCache
            HAS_CACHE = True
        except ImportError:
            HAS_CACHE = False


def objective(trial: Trial, args, parquet_files: List[str]) -> float:
    """Optuna objective function - returns validation AUROC to maximize."""
    
    # Sample hyperparameters
    arch = trial.suggest_categorical("arch", ["resnet18", "small_cnn"])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    warmup_epochs = trial.suggest_float("warmup_epochs", 0.5, 2.0)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Datasets
    train_ds = Phase4cDataset(
        parquet_files, "train", stamp_size=args.stamp_size,
        augment=True, seed=args.seed, max_rows_per_file=args.max_rows_per_file or None
    )
    val_ds = Phase4cDataset(
        parquet_files, "val", stamp_size=args.stamp_size,
        augment=False, seed=args.seed + 999, max_rows_per_file=2000
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=True)
    
    # Model
    model = build_model(arch, dropout=dropout).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler
    warmup_steps = int(warmup_epochs * args.steps_per_epoch)
    total_steps = args.epochs_per_trial * args.steps_per_epoch
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps, min_lr=1e-6)
    
    # Scaler
    scaler = GradScaler(enabled=True)
    
    # Loss
    bce = nn.BCEWithLogitsLoss()
    
    best_auroc = 0.0
    
    for epoch in range(args.epochs_per_trial):
        # Training
        model.train()
        it = iter(train_loader)
        
        for step in range(args.steps_per_epoch):
            try:
                x, y, _ = next(it)
            except StopIteration:
                it = iter(train_loader)
                x, y, _ = next(it)
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                logit = model(x)
                loss = bce(logit, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        
        # Validation
        model.eval()
        metrics = Metrics()
        
        with torch.no_grad():
            for batch_idx, (x, y, _) in enumerate(val_loader):
                if batch_idx >= 200:
                    break
                
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                with autocast():
                    logit = model(x)
                    loss = bce(logit, y)
                
                probs = torch.sigmoid(logit)
                metrics.update(loss.item(), probs, y)
        
        val_metrics = metrics.compute()
        auroc = val_metrics.get("auroc", 0.0)
        
        if auroc > best_auroc:
            best_auroc = auroc
        
        # Report for pruning
        trial.report(auroc, epoch)
        
        # Prune if unpromising
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        print(f"  [Trial {trial.number}] epoch={epoch} auroc={auroc:.4f} loss={val_metrics['loss']:.4f}")
    
    return best_auroc


def main():
    ap = argparse.ArgumentParser(description="Phase 5 hyperparameter sweep")
    ap.add_argument("--data", required=True, help="Data path")
    ap.add_argument("--contract_json", required=True)
    ap.add_argument("--out_dir", required=True, help="Output directory for study results")
    ap.add_argument("--n_trials", type=int, default=20, help="Number of trials")
    ap.add_argument("--epochs_per_trial", type=int, default=3, help="Epochs per trial")
    ap.add_argument("--steps_per_epoch", type=int, default=1000, help="Steps per epoch (smaller for speed)")
    ap.add_argument("--stamp_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_rows_per_file", type=int, default=0)
    ap.add_argument("--cache_root", default="/data/cache")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--study_name", default="phase5_sweep")
    args = ap.parse_args()
    
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Resolve data
    data_path = args.data
    if data_path.startswith("s3://") and HAS_CACHE:
        print("[INFO] Using data cache...")
        cache = DataCache(cache_root=args.cache_root)
        data_path = cache.get(data_path)
    
    print(f"[INFO] Data path: {data_path}")
    
    parquet_files = _list_parquet_files(data_path)
    print(f"[INFO] Found {len(parquet_files)} parquet files")
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Create Optuna study
    storage_path = os.path.join(args.out_dir, "optuna_study.db")
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{storage_path}",
        direction="maximize",  # Maximize AUROC
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    
    print(f"[INFO] Starting sweep with {args.n_trials} trials")
    print(f"[INFO] Each trial: {args.epochs_per_trial} epochs x {args.steps_per_epoch} steps")
    
    def objective_wrapper(trial):
        return objective(trial, args, parquet_files)
    
    study.optimize(objective_wrapper, n_trials=args.n_trials, show_progress_bar=True)
    
    # Results
    print("\n" + "="*60)
    print("SWEEP COMPLETE")
    print("="*60)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best AUROC: {study.best_trial.value:.4f}")
    print("\nBest hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
    
    # Save results
    results = {
        "best_trial": study.best_trial.number,
        "best_auroc": study.best_trial.value,
        "best_params": study.best_trial.params,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
        "sweep_config": vars(args),
        "timestamp": datetime.now().isoformat(),
    }
    
    results_path = os.path.join(args.out_dir, "sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Optuna DB: {storage_path}")
    
    # Generate command for best params
    bp = study.best_trial.params
    print("\n" + "="*60)
    print("RECOMMENDED TRAINING COMMAND:")
    print("="*60)
    print(f"""
python phase5_train_production.py \\
  --data {args.data} \\
  --contract_json {args.contract_json} \\
  --out_dir /data/phase5/models/best_from_sweep \\
  --arch {bp['arch']} \\
  --lr {bp['lr']:.6f} \\
  --weight_decay {bp['weight_decay']:.6f} \\
  --dropout {bp['dropout']:.2f} \\
  --batch_size {bp['batch_size']} \\
  --warmup_epochs {bp['warmup_epochs']:.2f} \\
  --label_smoothing {bp['label_smoothing']:.3f} \\
  --epochs 20 \\
  --steps_per_epoch 5000 \\
  --early_stopping_patience 5
""")


if __name__ == "__main__":
    main()


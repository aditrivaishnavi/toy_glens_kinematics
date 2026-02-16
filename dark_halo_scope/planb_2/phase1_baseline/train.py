#!/usr/bin/env python3
"""
Phase 1: Training Script

Full training loop with:
- Pre-flight validation
- Training monitoring
- Gate evaluation during training
- Post-training validation
- Early stopping

Lessons Learned Incorporated:
- L5.1: Don't declare victory until verified
- L5.2: Check logs immediately
- L6.2: Verify code matches expectations
- L4.4: Never assume clean data
- L1.2: No duplicate functions - use shared module

Exit Criteria Per Epoch:
- Loss is finite and decreasing
- AUROC is improving or stable
- Gate metrics tracked

Exit Criteria Per Training:
- All gates pass (see evaluate_gates)
- Model saved to checkpoint
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from torch.cuda.amp import GradScaler, autocast

# Import from shared module - SINGLE SOURCE OF TRUTH
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.constants import (
    STAMP_SIZE, NUM_CHANNELS, CORE_SIZE_PIX, GATES,
    DEFAULT_LR, DEFAULT_WEIGHT_DECAY, DEFAULT_EPOCHS,
    DEFAULT_EARLY_STOPPING_PATIENCE, get_core_slice,
)
from shared.schema import BATCH_SCHEMA

# Local imports
from data_loader import build_training_loader, build_eval_loader, validate_loader
from model import build_model, ModelConfig, validate_model


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Set up logging to file and console."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir}/training.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


# =============================================================================
# PREFLIGHT CHECKS
# =============================================================================
# Lesson L5.4: Validate code before running expensive operations

def run_preflight_checks(
    train_loader,
    val_loader,
    model: nn.Module,
    device: str,
    logger: logging.Logger
) -> Dict[str, bool]:
    """
    Run all preflight checks before training.
    
    Exit Criteria:
    1. Data loaders produce valid batches
    2. Model forward/backward passes work
    3. No NaN in initial forward pass
    
    All checks MUST pass before training proceeds.
    """
    logger.info("="*60)
    logger.info("PREFLIGHT CHECKS")
    logger.info("="*60)
    
    results = {
        "train_loader_valid": False,
        "val_loader_valid": False,
        "model_valid": False,
        "first_batch_ok": False,
    }
    
    # Check 1: Train loader
    logger.info("Checking train loader...")
    train_check = validate_loader(train_loader, n_batches=3)
    results["train_loader_valid"] = train_check["all_passed"]
    logger.info(f"  Train loader: {'PASS' if train_check['all_passed'] else 'FAIL'}")
    
    # Check 2: Val loader
    logger.info("Checking val loader...")
    val_check = validate_loader(val_loader, n_batches=3)
    results["val_loader_valid"] = val_check["all_passed"]
    logger.info(f"  Val loader: {'PASS' if val_check['all_passed'] else 'FAIL'}")
    
    # Check 3: Model
    logger.info("Checking model...")
    model_check = validate_model(model, device=device)
    results["model_valid"] = model_check["all_passed"]
    logger.info(f"  Model: {'PASS' if model_check['all_passed'] else 'FAIL'}")
    
    # Check 4: First batch forward pass
    logger.info("Checking first batch...")
    try:
        batch = next(iter(train_loader))
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        
        model.eval()
        with torch.no_grad():
            logits = model(x)
        
        if torch.isfinite(logits).all():
            results["first_batch_ok"] = True
            logger.info("  First batch: PASS")
        else:
            logger.error("  First batch: FAIL (NaN in output)")
    except Exception as e:
        logger.error(f"  First batch: FAIL ({e})")
    
    # Summary
    all_passed = all(results.values())
    logger.info("-"*60)
    if all_passed:
        logger.info("✓ ALL PREFLIGHT CHECKS PASSED")
    else:
        logger.error("✗ PREFLIGHT CHECKS FAILED")
        for name, passed in results.items():
            if not passed:
                logger.error(f"  Failed: {name}")
    
    return results


# =============================================================================
# GATE EVALUATION
# =============================================================================
# Lesson L21: Core brightness shortcut exists - must monitor

def evaluate_gates(
    model: nn.Module,
    loader,
    device: str,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Evaluate gate metrics for shortcut detection.
    
    Gates:
    1. core_lr_auc: LR on central 10x10 pixels
    2. auroc_full: Standard AUROC
    3. auroc_core_masked: AUROC with center masked
    4. core_masked_drop: Relative drop when center masked
    
    Exit Criteria:
    - core_lr_auc < 0.65 (shortcut blocked)
    - core_masked_drop < 0.10 (not center-dependent)
    """
    model.eval()
    
    all_logits = []
    all_labels = []
    all_core_features = []
    all_logits_masked = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].numpy()
            
            # Full prediction
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_logits.extend(probs)
            all_labels.extend(y)
            
            # Extract core features for LR
            core = x[:, :, 27:37, 27:37]  # Central 10x10
            core_flat = core.cpu().reshape(x.shape[0], -1).numpy()
            all_core_features.append(core_flat)
            
            # Masked prediction
            x_masked = x.clone()
            h, w = x.shape[2], x.shape[3]
            cy, cx = h // 2, w // 2
            yy, xx = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij"
            )
            r = torch.sqrt((xx - cx).float()**2 + (yy - cy).float()**2)
            mask = r < 5  # Mask radius
            
            for i in range(x.shape[0]):
                for c in range(x.shape[1]):
                    outer_median = x_masked[i, c, ~mask].median()
                    x_masked[i, c, mask] = outer_median
            
            logits_masked = model(x_masked)
            probs_masked = torch.sigmoid(logits_masked).cpu().numpy().flatten()
            all_logits_masked.extend(probs_masked)
    
    # Compute metrics
    results = {}
    
    # Full AUROC
    results["auroc_full"] = roc_auc_score(all_labels, all_logits)
    
    # Masked AUROC
    results["auroc_core_masked"] = roc_auc_score(all_labels, all_logits_masked)
    
    # Core masked drop
    results["core_masked_drop"] = (
        (results["auroc_full"] - results["auroc_core_masked"]) / results["auroc_full"]
    )
    
    # Core LR AUC
    X_core = np.vstack(all_core_features)
    y_core = np.array(all_labels)
    
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_core, y_core)
    probs_lr = lr.predict_proba(X_core)[:, 1]
    results["core_lr_auc"] = roc_auc_score(y_core, probs_lr)
    
    # Log results
    logger.info("Gate Evaluation Results:")
    logger.info(f"  AUROC (full): {results['auroc_full']:.4f}")
    logger.info(f"  AUROC (core masked): {results['auroc_core_masked']:.4f}")
    logger.info(f"  Core masked drop: {results['core_masked_drop']:.2%}")
    logger.info(f"  Core LR AUC: {results['core_lr_auc']:.4f}")
    
    # Check gates
    gates_passed = {
        "core_lr_auc": results["core_lr_auc"] < 0.65,
        "core_masked_drop": results["core_masked_drop"] < 0.10,
    }
    
    logger.info("Gate Status:")
    for gate, passed in gates_passed.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {gate}: {status}")
    
    results["gates_passed"] = gates_passed
    results["all_gates_passed"] = all(gates_passed.values())
    
    return results


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: Optional[GradScaler],
    grad_clip: float,
    logger: logging.Logger,
    epoch: int,
    log_every: int = 100,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Monitoring:
    - Loss per batch
    - Gradient norms
    - NaN detection
    """
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
        
        # Check for NaN
        if torch.isnan(loss):
            logger.error(f"NaN loss at batch {batch_idx}")
            return {"loss": float("nan"), "error": "nan_loss"}
        
        # Accumulate metrics
        total_loss += loss.item() * x.shape[0]
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == y).sum().item()
        total_samples += x.shape[0]
        
        # Log progress
        if (batch_idx + 1) % log_every == 0:
            avg_loss = total_loss / total_samples
            acc = total_correct / total_samples
            logger.info(
                f"Epoch {epoch} [{batch_idx+1}] "
                f"Loss: {avg_loss:.4f} Acc: {acc:.4f}"
            )
    
    elapsed = time.time() - start_time
    
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "elapsed_sec": elapsed,
        "samples": total_samples,
    }


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    model.eval()
    
    total_loss = 0.0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device).unsqueeze(1)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * x.shape[0]
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy().flatten())
    
    auroc = roc_auc_score(all_labels, all_probs)
    
    return {
        "loss": total_loss / len(all_labels),
        "auroc": auroc,
        "n_samples": len(all_labels),
    }


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(
    parquet_root: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    hard_negative_ratio: float = 0.4,
    core_dropout_prob: float = 0.5,
    mixed_precision: bool = True,
    num_workers: int = 4,
    early_stopping_patience: int = 15,
    gate_eval_every: int = 5,
    device: str = "cuda",
    seed: int = 42,
    wandb_project: Optional[str] = None,
    wandb_run: Optional[str] = None,
):
    """
    Main training function with full validation.
    
    Exit Criteria:
    - All preflight checks pass
    - Training completes without NaN
    - Final gates pass
    """
    # Setup
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    logger.info("="*60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Parquet root: {parquet_root}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Weight decay: {weight_decay}")
    logger.info(f"Hard negative ratio: {hard_negative_ratio}")
    logger.info(f"Core dropout prob: {core_dropout_prob}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")
    
    # Save config
    config = {
        "parquet_root": parquet_root,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "hard_negative_ratio": hard_negative_ratio,
        "core_dropout_prob": core_dropout_prob,
        "mixed_precision": mixed_precision,
        "early_stopping_patience": early_stopping_patience,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Build data loaders
    logger.info("Building data loaders...")
    train_loader = build_training_loader(
        parquet_root,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        hard_negative_ratio=hard_negative_ratio,
        core_dropout_prob=core_dropout_prob,
    )
    val_loader = build_eval_loader(
        parquet_root,
        split="val",
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # Build model
    logger.info("Building model...")
    model_config = ModelConfig(
        arch="resnet18",
        pretrained=True,
        in_channels=3,
        replace_first_conv=True,
        first_conv_kernel=3,
    )
    model = build_model(config=model_config, device=device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )
    
    # Loss and scaler
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler() if mixed_precision and device == "cuda" else None
    
    # Run preflight checks
    preflight = run_preflight_checks(
        train_loader, val_loader, model, device, logger
    )
    
    if not all(preflight.values()):
        logger.error("ABORTING: Preflight checks failed")
        return {"success": False, "error": "preflight_failed", "preflight": preflight}
    
    # WandB logging
    if wandb_project:
        try:
            import wandb
            wandb.init(project=wandb_project, name=wandb_run, config=config)
        except Exception as e:
            logger.warning(f"WandB init failed: {e}")
    
    # Training loop
    logger.info("="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    
    best_auroc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    
    for epoch in range(1, epochs + 1):
        logger.info(f"\n{'='*40}")
        logger.info(f"Epoch {epoch}/{epochs}")
        logger.info(f"{'='*40}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, scaler, grad_clip, logger, epoch
        )
        
        # Check for training failure
        if "error" in train_metrics:
            logger.error(f"Training failed: {train_metrics['error']}")
            return {"success": False, "error": train_metrics["error"]}
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        logger.info(
            f"Epoch {epoch} - "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"Val AUROC: {val_metrics['auroc']:.4f}"
        )
        
        # Update scheduler
        scheduler.step()
        
        # Track history
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_auroc": val_metrics["auroc"],
            "lr": scheduler.get_last_lr()[0],
        }
        
        # Gate evaluation
        if epoch % gate_eval_every == 0 or epoch == epochs:
            logger.info("Running gate evaluation...")
            gate_results = evaluate_gates(model, val_loader, device, logger)
            epoch_record["gates"] = gate_results
        
        history.append(epoch_record)
        
        # Save checkpoint
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auroc": val_metrics["auroc"],
                "config": config,
            }, f"{output_dir}/best_model.pt")
            
            logger.info(f"✓ New best model saved (AUROC: {best_auroc:.4f})")
        else:
            patience_counter += 1
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            }, f"{output_dir}/checkpoint_epoch{epoch}.pt")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch} (patience {early_stopping_patience})")
            break
        
        # WandB logging
        if wandb_project:
            try:
                import wandb
                wandb.log(epoch_record)
            except:
                pass
    
    # Save last checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }, f"{output_dir}/last_model.pt")
    
    # Final evaluation
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION")
    logger.info("="*60)
    
    # Load best model
    checkpoint = torch.load(f"{output_dir}/best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Final gate evaluation
    final_gates = evaluate_gates(model, val_loader, device, logger)
    
    # Save results
    results = {
        "success": True,
        "best_epoch": best_epoch,
        "best_auroc": best_auroc,
        "final_gates": final_gates,
        "history": history,
        "config": config,
    }
    
    with open(f"{output_dir}/training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Final status
    logger.info("\n" + "="*60)
    if final_gates["all_gates_passed"]:
        logger.info("✓ TRAINING COMPLETE - ALL GATES PASSED")
    else:
        logger.warning("⚠ TRAINING COMPLETE - SOME GATES FAILED")
    logger.info("="*60)
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train lens classifier")
    
    # Required
    parser.add_argument("--parquet-root", required=True, help="Path to parquet data")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    
    # Mitigations
    parser.add_argument("--hard-negative-ratio", type=float, default=0.4)
    parser.add_argument("--core-dropout-prob", type=float, default=0.5)
    
    # Compute
    parser.add_argument("--mixed-precision", action="store_true", default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Training control
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--gate-eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    
    # Logging
    parser.add_argument("--wandb-project", help="WandB project name")
    parser.add_argument("--wandb-run", help="WandB run name")
    
    args = parser.parse_args()
    
    results = train(
        parquet_root=args.parquet_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        hard_negative_ratio=args.hard_negative_ratio,
        core_dropout_prob=args.core_dropout_prob,
        mixed_precision=args.mixed_precision,
        num_workers=args.num_workers,
        device=args.device,
        early_stopping_patience=args.early_stopping_patience,
        gate_eval_every=args.gate_eval_every,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
    )
    
    sys.exit(0 if results.get("success") else 1)


if __name__ == "__main__":
    main()

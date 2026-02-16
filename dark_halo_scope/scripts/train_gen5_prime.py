#!/usr/bin/env python3
"""
Gen5-Prime Training Script

Shortcut-resistant lens finder training with:
- 6-channel input (raw + residual)
- Paired positive/control samples
- On-the-fly hard negative generation
- Curriculum learning (50% -> 20% hard-neg)
- Periodic gate validation

Usage:
    python train_gen5_prime.py --config config.yaml
    python train_gen5_prime.py  # Uses defaults

Author: DarkHaloScope Team
Date: 2026-02-05
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.paired_training_v2 import (
    build_training_loader,
    run_gates_quick,
    PairedParquetDataset,
    CoaddCutoutProvider,
    DropOnErrorWrapper,
    Preprocess6CH,
    PairedMixCollate,
    make_outer_mask,
    DEFAULT_CLIP,
    DEFAULT_RESID_SIGMA_PIX,
    DEFAULT_OUTER_R_PIX,
    STAMP_SIZE,
)
from training.convnext_6ch import LensFinder6CH


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration with defaults."""
    
    # Data paths
    parquet_root: str = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
    coadd_cache: str = "/lambda/nfs/darkhaloscope-training-dc/dr10/coadd_cache"
    output_dir: str = "/lambda/nfs/darkhaloscope-training-dc/models/gen5_prime"
    disk_cache_dir: str = "/lambda/nfs/darkhaloscope-training-dc/cutout_cache"
    
    # API fallback for when coadd cache is unavailable
    use_api_fallback: bool = True
    
    # Model
    arch: str = "tiny"
    pretrained: bool = True
    init: str = "copy_or_zero"
    meta_dim: int = 2
    hidden: int = 256
    dropout: float = 0.1
    
    # Training
    epochs: int = 30
    batch_pairs: int = 64
    num_workers: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 3
    
    # Mixing probabilities
    pos_prob: float = 0.4
    ctrl_prob: float = 0.4
    hardneg_prob: float = 0.2
    
    # Curriculum
    curriculum_enabled: bool = True
    start_hardneg_prob: float = 0.5
    end_hardneg_prob: float = 0.2
    anneal_epochs: int = 18
    
    # Pilot mode for testing (small subset)
    pilot_mode: bool = False
    pilot_samples: int = 1000
    
    # Preprocessing
    clip: float = 10.0
    resid_sigma_pix: float = 3.0
    
    # Checkpointing
    checkpoint_every: int = 5
    run_gates_every: int = 5
    gate_max_pairs: int = 1000
    
    # Mixed precision
    use_amp: bool = True
    
    # Logging
    log_every: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    
    logger = logging.getLogger("train_gen5_prime")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"))
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# =============================================================================
# CURRICULUM LEARNING
# =============================================================================

def get_curriculum_hardneg_prob(epoch: int, cfg: TrainingConfig) -> float:
    """Get hard negative probability for current epoch."""
    if not cfg.curriculum_enabled:
        return cfg.hardneg_prob
    
    if epoch >= cfg.anneal_epochs:
        return cfg.end_hardneg_prob
    
    # Linear interpolation
    frac = epoch / cfg.anneal_epochs
    prob = cfg.start_hardneg_prob - frac * (cfg.start_hardneg_prob - cfg.end_hardneg_prob)
    return prob


def update_collate_probs(collate: PairedMixCollate, hardneg_prob: float):
    """Update collate function mixing probabilities."""
    # Keep pos_prob fixed, adjust ctrl_prob to maintain sum = 1
    collate.hardneg_prob = hardneg_prob
    collate.ctrl_prob = 1.0 - collate.pos_prob - hardneg_prob


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: Optional[GradScaler],
    device: torch.device,
    epoch: int,
    cfg: TrainingConfig,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(loader):
        x = batch.x6.to(device, non_blocking=True)
        y = batch.y.to(device, non_blocking=True)
        
        # Build meta tensor [psfsize_r (arcsec), psfdepth_r]
        meta = torch.stack([
            batch.meta["psf_fwhm_arcsec"],
            batch.meta["psfdepth_r"],
        ], dim=1).to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                logits = model(x, meta)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x, meta)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        # Metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct = (preds == y).sum().item()
            
            total_loss += loss.item() * x.shape[0]
            total_correct += correct
            total_samples += x.shape[0]
            
            all_preds.extend(probs.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
        
        # Logging
        if (batch_idx + 1) % cfg.log_every == 0:
            elapsed = time.time() - start_time
            samples_per_sec = total_samples / elapsed
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | Acc: {correct/x.shape[0]:.3f} | "
                f"Speed: {samples_per_sec:.1f} samples/s"
            )
    
    # Epoch metrics
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    # Compute AUC
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "auc": auc,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        x = batch.x6.to(device, non_blocking=True)
        y = batch.y.to(device, non_blocking=True)
        
        meta = torch.stack([
            batch.meta["psf_fwhm_arcsec"],
            batch.meta["psfdepth_r"],
        ], dim=1).to(device, non_blocking=True)
        
        logits = model(x, meta)
        loss = criterion(logits, y)
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct = (preds == y).sum().item()
        
        total_loss += loss.item() * x.shape[0]
        total_correct += correct
        total_samples += x.shape[0]
        
        all_preds.extend(probs.cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "auc": auc,
    }


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(cfg: TrainingConfig):
    """Main training function."""
    
    # Setup
    logger = setup_logging(cfg.output_dir)
    logger.info("=" * 70)
    logger.info("Gen5-Prime Training")
    logger.info("=" * 70)
    logger.info(f"Config: {json.dumps(cfg.to_dict(), indent=2)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Save config
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    
    # Model
    logger.info("Creating model...")
    model = LensFinder6CH(
        arch=cfg.arch,
        pretrained=cfg.pretrained,
        init=cfg.init,
        meta_dim=cfg.meta_dim,
        hidden=cfg.hidden,
        dropout=cfg.dropout,
    )
    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data loaders
    logger.info("Creating data loaders...")
    
    # Training loader
    max_pairs = cfg.pilot_samples if cfg.pilot_mode else None
    train_loader = build_training_loader(
        parquet_root=cfg.parquet_root,
        coadd_cache_root=cfg.coadd_cache,
        split="train",
        batch_pairs=cfg.batch_pairs,
        num_workers=cfg.num_workers if not cfg.pilot_mode else 0,  # No workers in pilot mode
        max_pairs_index=max_pairs,
        pos_prob=cfg.pos_prob,
        ctrl_prob=cfg.ctrl_prob,
        hardneg_prob=cfg.hardneg_prob,
        use_api_fallback=cfg.use_api_fallback,
        disk_cache_dir=cfg.disk_cache_dir,
    )
    logger.info(f"Train loader: {len(train_loader)} batches" + (" [PILOT MODE]" if cfg.pilot_mode else ""))
    
    # Validation loader (with fixed probabilities)
    val_max_pairs = min(500, cfg.pilot_samples) if cfg.pilot_mode else None
    val_loader = build_training_loader(
        parquet_root=cfg.parquet_root,
        coadd_cache_root=cfg.coadd_cache,
        split="val",
        batch_pairs=cfg.batch_pairs,
        num_workers=cfg.num_workers if not cfg.pilot_mode else 0,
        max_pairs_index=val_max_pairs,
        pos_prob=0.5,  # Balanced for validation
        ctrl_prob=0.5,
        hardneg_prob=0.0,  # No hard negatives in validation
        use_api_fallback=cfg.use_api_fallback,
        disk_cache_dir=cfg.disk_cache_dir,
    )
    logger.info(f"Val loader: {len(val_loader)} batches")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr / 100)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Mixed precision scaler
    scaler = GradScaler() if cfg.use_amp and device.type == "cuda" else None
    
    # Training history
    history = {
        "train_loss": [],
        "train_auc": [],
        "val_loss": [],
        "val_auc": [],
        "hardneg_prob": [],
        "gate_results": [],
    }
    
    best_val_auc = 0.0
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        
        # Update curriculum
        hardneg_prob = get_curriculum_hardneg_prob(epoch, cfg)
        update_collate_probs(train_loader.collate_fn, hardneg_prob)
        logger.info(f"\nEpoch {epoch}/{cfg.epochs} | Hard-neg prob: {hardneg_prob:.2f}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch, cfg, logger
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch} complete | "
            f"Train Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["train_auc"].append(train_metrics["auc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])
        history["hardneg_prob"].append(hardneg_prob)
        
        # Save best model
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_auc": val_metrics["auc"],
                "config": cfg.to_dict(),
            }, os.path.join(cfg.output_dir, "ckpt_best.pt"))
            logger.info(f"  New best model saved (AUC: {best_val_auc:.4f})")
        
        # Periodic checkpoint
        if (epoch + 1) % cfg.checkpoint_every == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_auc": val_metrics["auc"],
                "config": cfg.to_dict(),
            }, os.path.join(cfg.output_dir, f"ckpt_epoch{epoch}.pt"))
        
        # Run gates
        if (epoch + 1) % cfg.run_gates_every == 0:
            logger.info("Running validation gates...")
            try:
                gate_results = run_gates_quick(
                    parquet_root=cfg.parquet_root,
                    coadd_cache_root=cfg.coadd_cache,
                    split="val",
                    model=model,
                    device=str(device),
                    max_pairs=cfg.gate_max_pairs if not cfg.pilot_mode else min(200, cfg.pilot_samples),
                    use_api_fallback=cfg.use_api_fallback,
                    disk_cache_dir=cfg.disk_cache_dir,
                )
                history["gate_results"].append({
                    "epoch": epoch,
                    "results": gate_results,
                })
                
                # Log key gate metrics
                if "strata" in gate_results and "x_ge_1p0" in gate_results["strata"]:
                    x_ge_1 = gate_results["strata"]["x_ge_1p0"]
                    if "core_auc_lr" in x_ge_1:
                        logger.info(f"  Gate (x>=1): Core AUC={x_ge_1['core_auc_lr']:.3f}, "
                                    f"Annulus AUC={x_ge_1['annulus_auc_lr']:.3f}")
                
                if "model_gates" in gate_results:
                    mg = gate_results["model_gates"]
                    logger.info(f"  Model gates: Hard-neg p={mg['hardneg_mean_p']:.3f}, "
                                f"Arc-occ drop={mg['arc_occlusion_drop_abs']:.3f}")
            except Exception as e:
                logger.warning(f"Gate validation failed: {e}")
        
        # Save history
        with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
    
    # Final checkpoint
    torch.save({
        "epoch": cfg.epochs - 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "val_auc": val_metrics["auc"],
        "config": cfg.to_dict(),
    }, os.path.join(cfg.output_dir, "ckpt_last.pt"))
    
    # Final gates on test set
    logger.info("\n" + "=" * 70)
    logger.info("Running final gates on test set...")
    try:
        final_gates = run_gates_quick(
            parquet_root=cfg.parquet_root,
            coadd_cache_root=cfg.coadd_cache,
            split="test",
            model=model,
            device=str(device),
            max_pairs=2000 if not cfg.pilot_mode else min(500, cfg.pilot_samples),
            use_api_fallback=cfg.use_api_fallback,
            disk_cache_dir=cfg.disk_cache_dir,
        )
        with open(os.path.join(cfg.output_dir, "final_gate_results.json"), "w") as f:
            json.dump(final_gates, f, indent=2)
        logger.info("Final gate results saved.")
        
        # Summary
        if "strata" in final_gates and "x_ge_1p0" in final_gates["strata"]:
            x_ge_1 = final_gates["strata"]["x_ge_1p0"]
            logger.info("\n" + "=" * 70)
            logger.info("FINAL GATE SUMMARY (x >= 1.0)")
            logger.info("=" * 70)
            core_auc = x_ge_1.get('core_auc_lr', 'N/A')
            radial_auc = x_ge_1.get('radial_profile_auc_lr', 'N/A')
            annulus_auc = x_ge_1.get('annulus_auc_lr', 'N/A')
            logger.info(f"Core AUC:    {core_auc:.3f if isinstance(core_auc, (int, float)) else core_auc} (target <= 0.60)")
            logger.info(f"Radial AUC:  {radial_auc:.3f if isinstance(radial_auc, (int, float)) else radial_auc} (target <= 0.60)")
            logger.info(f"Annulus AUC: {annulus_auc:.3f if isinstance(annulus_auc, (int, float)) else annulus_auc} (target >= 0.75)")
        
        if "model_gates" in final_gates:
            mg = final_gates["model_gates"]
            hardneg_p = mg.get('hardneg_mean_p', 'N/A')
            arc_drop = mg.get('arc_occlusion_drop_abs', 'N/A')
            logger.info(f"Hard-neg p:  {hardneg_p:.3f if isinstance(hardneg_p, (int, float)) else hardneg_p} (target <= 0.05)")
            logger.info(f"Arc-occ drop: {arc_drop:.3f if isinstance(arc_drop, (int, float)) else arc_drop} (target >= 0.30)")
    except Exception as e:
        logger.error(f"Final gate validation failed: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info(f"Training complete! Best val AUC: {best_val_auc:.4f}")
    logger.info(f"Output: {cfg.output_dir}")
    logger.info("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Gen5-Prime lens finder")
    parser.add_argument("--parquet-root", type=str, default=None)
    parser.add_argument("--coadd-cache", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--disk-cache-dir", type=str, default=None, help="Directory to cache API-fetched cutouts")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-pairs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--arch", type=str, default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Run in pilot mode with small subset")
    parser.add_argument("--pilot-samples", type=int, default=1000, help="Number of samples for pilot mode")
    parser.add_argument("--no-api-fallback", action="store_true", help="Disable Legacy Survey API fallback")
    args = parser.parse_args()
    
    cfg = TrainingConfig()
    
    if args.parquet_root:
        cfg.parquet_root = args.parquet_root
    if args.coadd_cache:
        cfg.coadd_cache = args.coadd_cache
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_pairs:
        cfg.batch_pairs = args.batch_pairs
    if args.lr:
        cfg.lr = args.lr
    if args.num_workers:
        cfg.num_workers = args.num_workers
    if args.arch:
        cfg.arch = args.arch
    if args.no_curriculum:
        cfg.curriculum_enabled = False
    if args.pilot:
        cfg.pilot_mode = True
    if args.pilot_samples:
        cfg.pilot_samples = args.pilot_samples
    if args.no_api_fallback:
        cfg.use_api_fallback = False
    if args.disk_cache_dir:
        cfg.disk_cache_dir = args.disk_cache_dir
    if args.no_amp:
        cfg.use_amp = False
    
    train(cfg)


if __name__ == "__main__":
    main()

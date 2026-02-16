"""
Training script for unpaired experiment.

Supports:
1. Unpaired training (from manifest)
2. Paired training (for baseline comparison)
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from .data_loader import UnpairedDataset, PairedDataset, AugmentConfig
from .gates import run_shortcut_gates
from .constants import SEED_DEFAULT, EARLY_STOPPING_PATIENCE
from .scheduled_mask import ScheduledCoreMask, apply_deterministic_mask
from .thetae_stratification import auroc_by_thetae, ThetaEBinResult

# θ_E bins for stratified evaluation (in arcsec)
THETA_E_BINS = [
    (0.5, 1.0),
    (1.0, 1.5),
    (1.5, 2.0),
    (2.0, 3.0),
    (3.0, 5.0),
]

logger = logging.getLogger(__name__)


def build_resnet18(in_ch: int = 3) -> nn.Module:
    """Build ResNet18 for binary classification."""
    import torchvision.models as models
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(512, 1)
    return m


def collate_fn(batch):
    """Collate batch of (img, label) tuples."""
    xs, ys = zip(*batch)
    x = torch.from_numpy(np.stack(xs, axis=0)).float()
    y = torch.from_numpy(np.array(ys)).float().view(-1, 1)
    
    # Final safety: replace any NaN/Inf with 0
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    return x, y


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate model and return AUROC."""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.sigmoid(logits).cpu().numpy().ravel()
        ys.append(y.numpy().ravel())
        ps.append(p)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return float(roc_auc_score(y, p))


@torch.no_grad()
def evaluate_with_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    """Evaluate model and return (AUROC, y_true, y_pred)."""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.sigmoid(logits).cpu().numpy().ravel()
        ys.append(y.numpy().ravel())
        ps.append(p)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    auc = float(roc_auc_score(y, p))
    return auc, y, p


def parse_schedule(schedule_str: str) -> list:
    """Parse schedule string to list of (epoch, radius, prob) tuples."""
    entries = []
    for part in schedule_str.split(","):
        epoch, radius, prob = part.split(":")
        entries.append((int(epoch), int(radius), float(prob)))
    return entries


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int = 0,
    core_mask: Optional[ScheduledCoreMask] = None,
) -> float:
    """Train one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        # Apply scheduled core masking if enabled
        if core_mask is not None:
            x = core_mask(x, epoch, deterministic=False)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            logits = model(x)
            loss = loss_fn(logits, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def run_gates_on_loader(loader: DataLoader, n_samples: int = 1000) -> dict:
    """Run shortcut gates on samples from loader."""
    xs, ys = [], []
    for x, y in loader:
        xs.extend(x.numpy())
        ys.extend(y.numpy().ravel().astype(int))
        if len(xs) >= n_samples:
            break
    
    xs = np.stack(xs[:n_samples], axis=0)
    ys = np.array(ys[:n_samples])
    
    results = run_shortcut_gates(xs, ys)
    return {"core_lr_auc": results.core_lr_auc, "radial_profile_auc": results.radial_profile_auc}


def main():
    """Main training entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="Train lens detector")
    
    # Data args
    parser.add_argument("--manifest", help="Unpaired manifest path (for unpaired mode)")
    parser.add_argument("--data-root", help="Data root path (for paired mode)")
    parser.add_argument("--file-list", help="File list for paired mode (subset of files)")
    parser.add_argument("--preprocessing", default="raw_robust",
                       choices=["raw_robust", "residual_radial_profile"])
    
    # Augmentation args
    parser.add_argument("--core-dropout-prob", type=float, default=0.0,
                       help="Legacy: constant core dropout probability (use --scheduled-masking instead)")
    parser.add_argument("--az-shuffle-prob", type=float, default=0.0)
    
    # Scheduled masking (overrides --core-dropout-prob if set)
    parser.add_argument("--scheduled-masking", action="store_true",
                       help="Use scheduled core masking (0-10: r=7 p=0.7, 10-30: r=5 p=0.5, 30+: r=3 p=0.3)")
    parser.add_argument("--schedule", type=str, default="0:7:0.7,10:5:0.5,30:3:0.3",
                       help="Custom schedule in format 'epoch:radius:prob,epoch:radius:prob,...'")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    
    # Output args
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate-eval-every", type=int, default=5,
                       help="Run gates every N epochs")
    parser.add_argument("--fresh", action="store_true",
                       help="Force fresh start, ignore existing checkpoint")
    
    args = parser.parse_args()
    
    # Set seeds for full reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup augmentation
    aug_train = AugmentConfig(
        hflip=True, vflip=True, rot90=True,
        core_dropout_prob=args.core_dropout_prob,
        az_shuffle_prob=args.az_shuffle_prob,
    )
    aug_val = AugmentConfig(hflip=False, vflip=False, rot90=False)
    
    # Create datasets
    if args.manifest:
        logger.info(f"Loading unpaired data from {args.manifest}")
        ds_train = UnpairedDataset(args.manifest, "train", args.preprocessing, aug_train, args.seed)
        ds_val = UnpairedDataset(args.manifest, "val", args.preprocessing, aug_val, args.seed)
        ds_test = UnpairedDataset(args.manifest, "test", args.preprocessing, aug_val, args.seed)
    elif args.data_root:
        logger.info(f"Loading paired data from {args.data_root}")
        file_list = args.file_list if args.file_list else None
        ds_train = PairedDataset(args.data_root, "train", args.preprocessing, aug_train, args.seed, file_list=file_list)
        ds_val = PairedDataset(args.data_root, "val", args.preprocessing, aug_val, args.seed)
        ds_test = PairedDataset(args.data_root, "test", args.preprocessing, aug_val, args.seed)
    else:
        parser.error("Must provide either --manifest or --data-root")
    
    logger.info(f"Train samples: {len(ds_train)}, Val samples: {len(ds_val)}")
    
    # Create loaders
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(3).to(device)
    logger.info(f"Model on {device}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision and device.type == "cuda"))
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Auto-resume from checkpoint if exists (unless --fresh)
    start_epoch = 1
    best_auc = -1.0
    checkpoint_path = os.path.join(args.output_dir, "last.pt")
    if os.path.exists(checkpoint_path) and not args.fresh:
        logger.info(f"Found checkpoint, auto-resuming from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt.get("val_auc", -1.0)
        logger.info(f"Resumed from epoch {ckpt['epoch']}, best_auc={best_auc:.4f}")
    elif args.fresh and os.path.exists(checkpoint_path):
        logger.info(f"--fresh specified, ignoring existing checkpoint")
    
    # Setup scheduled core masking
    core_mask = None
    if args.scheduled_masking:
        schedule = parse_schedule(args.schedule)
        core_mask = ScheduledCoreMask(schedule, image_size=63, fill_value=0.0)
        logger.info(f"Scheduled masking enabled: {schedule}")
    
    # Training loop
    bad_epochs = 0
    
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        
        # Log current mask params if using scheduled masking
        if core_mask is not None:
            radius, prob = core_mask.get_current_params(epoch - 1)  # 0-indexed for schedule
            logger.info(f"Epoch {epoch}: core mask r={radius}, p={prob:.2f}")
        
        # Train
        train_loss = train_epoch(model, dl_train, optimizer, loss_fn, device, scaler, 
                                 epoch=epoch-1, core_mask=core_mask)
        scheduler.step()
        
        # Validate
        val_auc = evaluate(model, dl_val, device)
        
        # Checkpoints
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_auc": val_auc,
            "train_loss": train_loss,
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last.pt"))
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))
            bad_epochs = 0
        else:
            bad_epochs += 1
        
        dt = time.time() - t0
        logger.info(f"Epoch {epoch:03d}: loss={train_loss:.4f} val_auc={val_auc:.4f} best={best_auc:.4f} dt={dt:.1f}s")
        
        # Run gates periodically
        if epoch % args.gate_eval_every == 0:
            gate_results = run_gates_on_loader(dl_val, n_samples=1000)
            logger.info(f"  Gates: core_lr_auc={gate_results['core_lr_auc']:.4f}, "
                       f"radial_auc={gate_results['radial_profile_auc']:.4f}")
        
        # Early stopping
        if bad_epochs >= args.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    logger.info(f"Training complete. Best val AUC: {best_auc:.4f}")
    
    # Initialize metrics dict for JSON output
    metrics = {
        "experiment_config": {
            "manifest": args.manifest,
            "data_root": args.data_root,
            "preprocessing": args.preprocessing,
            "core_dropout_prob": args.core_dropout_prob,
            "scheduled_masking": args.scheduled_masking,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
        "training": {
            "best_val_auroc": best_auc,
            "final_epoch": epoch,
        },
    }
    
    # Final gate evaluation
    logger.info("Running final gate evaluation...")
    final_gates = run_gates_on_loader(dl_val, n_samples=2000)
    logger.info(f"Final gates: core_lr_auc={final_gates['core_lr_auc']:.4f}, "
               f"radial_auc={final_gates['radial_profile_auc']:.4f}")
    
    gate_status = "PASS" if (final_gates['core_lr_auc'] < 0.65 and final_gates['radial_profile_auc'] < 0.65) else "FAIL"
    logger.info(f"Gate status: {gate_status}")
    
    metrics["val_gates"] = {
        "core_lr_auc": final_gates["core_lr_auc"],
        "radial_profile_auc": final_gates["radial_profile_auc"],
        "status": gate_status,
    }
    
    # Run core sensitivity stress test
    logger.info("Running core sensitivity stress test on validation...")
    core_sens = run_core_sensitivity_test(model, dl_val, device)
    metrics["val_core_sensitivity"] = core_sens
    
    # θ_E stratified evaluation on validation
    logger.info("Running θ_E stratified evaluation on validation...")
    theta_e_results = run_thetae_stratified_eval(model, ds_val, dl_val, device)
    metrics["val_thetae_stratified"] = theta_e_results
    
    # ==================== TEST SET EVALUATION ====================
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("=" * 60)
    
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    
    # Test AUROC
    test_auc, test_y, test_p = evaluate_with_predictions(model, dl_test, device)
    logger.info(f"Test AUROC: {test_auc:.4f}")
    metrics["test"] = {"auroc": test_auc}
    
    # Test gates
    test_gates = run_gates_on_loader(dl_test, n_samples=2000)
    logger.info(f"Test gates: core_lr_auc={test_gates['core_lr_auc']:.4f}, "
               f"radial_auc={test_gates['radial_profile_auc']:.4f}")
    metrics["test"]["gates"] = test_gates
    
    # Test core sensitivity
    logger.info("Running core sensitivity stress test on test...")
    test_core_sens = run_core_sensitivity_test(model, dl_test, device)
    metrics["test"]["core_sensitivity"] = test_core_sens
    
    # Test θ_E stratified
    logger.info("Running θ_E stratified evaluation on test...")
    test_theta_e_results = run_thetae_stratified_eval(model, ds_test, dl_test, device)
    metrics["test"]["thetae_stratified"] = test_theta_e_results
    
    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Best Val AUROC:    {best_auc:.4f}")
    logger.info(f"Test AUROC:        {test_auc:.4f}")
    logger.info(f"Val Core LR AUC:   {final_gates['core_lr_auc']:.4f}")
    logger.info(f"Val Radial AUC:    {final_gates['radial_profile_auc']:.4f}")
    logger.info(f"Core Reliance:     {core_sens['core_reliance']:.4f}")
    logger.info(f"Gate Status:       {gate_status}")
    logger.info("=" * 60)


@torch.no_grad()
def run_core_sensitivity_test(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    """Run core sensitivity curve at deterministic mask radii. Returns results dict."""
    model.eval()
    
    radii_to_test = [0, 3, 5, 7, 10, 15]
    results = {}
    
    for radius in radii_to_test:
        ys, ps = [], []
        for x, y in loader:
            x = x.to(device)
            if radius > 0:
                x = apply_deterministic_mask(x, radius)
            logits = model(x)
            p = torch.sigmoid(logits).cpu().numpy().ravel()
            ys.append(y.numpy().ravel())
            ps.append(p)
        
        y = np.concatenate(ys)
        p = np.concatenate(ps)
        auc = float(roc_auc_score(y, p))
        results[f"r{radius}"] = auc
    
    logger.info("Core Sensitivity Curve (AUROC vs mask radius):")
    for r in radii_to_test:
        logger.info(f"  r={r:2d}: AUROC={results[f'r{r}']:.4f}")
    
    # Compute sensitivity: how much does AUROC drop as mask radius increases?
    auc_no_mask = results["r0"]
    auc_r7 = results["r7"]
    core_reliance = auc_no_mask - auc_r7
    results["core_reliance"] = core_reliance
    
    logger.info(f"Core reliance (AUROC drop from r=0 to r=7): {core_reliance:.4f}")
    if core_reliance > 0.05:
        logger.warning(f"Model shows significant core reliance ({core_reliance:.4f} > 0.05)")
        results["status"] = "FAIL"
    else:
        logger.info(f"Core reliance is acceptable ({core_reliance:.4f} <= 0.05)")
        results["status"] = "PASS"
    
    return results


def run_thetae_stratified_eval(
    model: nn.Module,
    dataset,  # UnpairedDataset or PairedDataset
    loader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """Run θ_E stratified AUROC evaluation."""
    # Get predictions
    _, y_true, y_score = evaluate_with_predictions(model, loader, device)
    
    # Get theta_E values
    if hasattr(dataset, 'get_all_theta_e'):
        theta_e = dataset.get_all_theta_e()
    else:
        logger.warning("Dataset doesn't support get_all_theta_e, skipping stratification")
        return {"error": "no_theta_e_support"}
    
    # Check if we have theta_E data
    valid_theta_e = ~np.isnan(theta_e)
    n_valid = np.sum(valid_theta_e)
    if n_valid < 100:
        logger.warning(f"Only {n_valid} samples have theta_E, skipping stratification")
        return {"error": "insufficient_theta_e_data", "n_valid": int(n_valid)}
    
    # Run stratified evaluation
    bin_results = auroc_by_thetae(y_true, y_score, theta_e, THETA_E_BINS, min_count=50)
    
    results = {"bins": []}
    for br in bin_results:
        bin_dict = {
            "lo": br.lo,
            "hi": br.hi,
            "n_pos": br.n_pos,
            "n_neg": br.n_neg,
            "auc": br.auc if not np.isnan(br.auc) else None,
        }
        results["bins"].append(bin_dict)
        if not np.isnan(br.auc):
            logger.info(f"  θ_E [{br.lo:.1f}, {br.hi:.1f}): n_pos={br.n_pos}, n_neg={br.n_neg}, AUC={br.auc:.4f}")
        else:
            logger.info(f"  θ_E [{br.lo:.1f}, {br.hi:.1f}): n_pos={br.n_pos} (insufficient)")
    
    # Compute summary stats
    valid_aucs = [b["auc"] for b in results["bins"] if b["auc"] is not None]
    if valid_aucs:
        results["min_auc"] = min(valid_aucs)
        results["max_auc"] = max(valid_aucs)
        results["auc_spread"] = max(valid_aucs) - min(valid_aucs)
        logger.info(f"  θ_E AUC range: [{results['min_auc']:.4f}, {results['max_auc']:.4f}], spread={results['auc_spread']:.4f}")
    
    return results


if __name__ == "__main__":
    main()

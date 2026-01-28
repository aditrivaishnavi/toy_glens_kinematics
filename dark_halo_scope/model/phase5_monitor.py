#!/usr/bin/env python3
"""
Phase 5: Training Monitor

Real-time monitoring of training progress with:
- Convergence detection
- Overfitting detection
- Time-to-completion estimates
- Learning rate analysis
- Best model tracking

Usage:
  # Monitor a running or completed training
  python phase5_monitor.py --run_dir /data/phase5/models/resnet18_v1

  # Monitor with auto-refresh (watch mode)
  python phase5_monitor.py --run_dir /data/phase5/models/resnet18_v1 --watch --interval 30

  # Analyze and recommend stopping
  python phase5_monitor.py --run_dir /data/phase5/models/resnet18_v1 --analyze
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None


@dataclass
class TrainingMetrics:
    """Metrics from a training run."""
    epoch: int
    global_step: int
    train_loss: float
    val_loss: float
    val_acc: float
    val_auroc: float
    val_precision: float
    val_recall: float
    val_f1: float
    lr: float
    timestamp: str = ""


def load_tensorboard_logs(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """Load metrics from TensorBoard logs."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[WARN] tensorboard not installed, cannot read TB logs")
        return {}
    
    tb_dir = os.path.join(log_dir, "tensorboard")
    if not os.path.exists(tb_dir):
        return {}
    
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    
    metrics = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        metrics[tag] = [(e.step, e.value) for e in events]
    
    return metrics


def load_checkpoints(run_dir: str) -> Dict[str, Dict]:
    """Load checkpoint metadata."""
    checkpoints = {}
    
    for name in ["checkpoint_last.pt", "checkpoint_best.pt"]:
        path = os.path.join(run_dir, name)
        if os.path.exists(path) and torch:
            try:
                ckpt = torch.load(path, map_location="cpu")
                checkpoints[name] = {
                    "epoch": ckpt.get("epoch", -1),
                    "global_step": ckpt.get("global_step", -1),
                    "best_auroc": ckpt.get("best_auroc", -1),
                    "config": ckpt.get("config", {}),
                }
            except Exception as e:
                print(f"[WARN] Failed to load {path}: {e}")
    
    return checkpoints


def analyze_convergence(metrics: Dict[str, List[Tuple[int, float]]]) -> Dict:
    """Analyze training convergence."""
    analysis = {}
    
    # Get validation loss history
    val_loss = metrics.get("val/loss", [])
    val_auroc = metrics.get("val/auroc", [])
    train_loss = metrics.get("train/loss", [])
    
    if len(val_loss) < 2:
        return {"status": "insufficient_data", "epochs": len(val_loss)}
    
    # Best metrics
    if val_auroc:
        best_auroc_idx = max(range(len(val_auroc)), key=lambda i: val_auroc[i][1])
        analysis["best_auroc"] = val_auroc[best_auroc_idx][1]
        analysis["best_auroc_epoch"] = val_auroc[best_auroc_idx][0]
    
    if val_loss:
        best_loss_idx = min(range(len(val_loss)), key=lambda i: val_loss[i][1])
        analysis["best_val_loss"] = val_loss[best_loss_idx][1]
        analysis["best_val_loss_epoch"] = val_loss[best_loss_idx][0]
    
    # Recent trend (last 3 epochs)
    if len(val_loss) >= 3:
        recent = [v for _, v in val_loss[-3:]]
        trend = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-8)
        
        if trend > 0.05:
            analysis["trend"] = "degrading"
            analysis["trend_pct"] = trend * 100
        elif trend < -0.02:
            analysis["trend"] = "improving"
            analysis["trend_pct"] = trend * 100
        else:
            analysis["trend"] = "plateau"
            analysis["trend_pct"] = trend * 100
    
    # Overfitting detection
    if train_loss and val_loss and len(train_loss) >= 3 and len(val_loss) >= 3:
        # Compare recent train vs val loss ratio
        recent_train = [v for _, v in train_loss[-3:]]
        recent_val = [v for _, v in val_loss[-3:]]
        
        train_trend = (recent_train[-1] - recent_train[0]) / max(abs(recent_train[0]), 1e-8)
        val_trend = (recent_val[-1] - recent_val[0]) / max(abs(recent_val[0]), 1e-8)
        
        if train_trend < -0.02 and val_trend > 0.02:
            analysis["overfitting"] = True
            analysis["overfitting_signal"] = "train improving but val degrading"
        else:
            analysis["overfitting"] = False
    
    # Epochs since improvement
    if val_auroc:
        current_epoch = val_auroc[-1][0]
        best_epoch = analysis.get("best_auroc_epoch", current_epoch)
        analysis["epochs_since_improvement"] = current_epoch - best_epoch
    
    return analysis


def estimate_time_remaining(metrics: Dict[str, List], config: Dict) -> Dict:
    """Estimate time to completion."""
    train_loss = metrics.get("train/loss", [])
    
    if len(train_loss) < 2:
        return {"status": "insufficient_data"}
    
    # Estimate time per step from timestamps (if available)
    # Fallback: assume constant step rate
    total_epochs = config.get("epochs", 20)
    steps_per_epoch = config.get("steps_per_epoch", 5000)
    
    current_step = train_loss[-1][0] if train_loss else 0
    total_steps = total_epochs * steps_per_epoch
    
    # Rough time estimate (assuming 100 steps/sec on V100)
    steps_remaining = total_steps - current_step
    estimated_seconds = steps_remaining / 100  # Rough estimate
    
    return {
        "current_step": current_step,
        "total_steps": total_steps,
        "progress_pct": current_step / max(total_steps, 1) * 100,
        "steps_remaining": steps_remaining,
        "estimated_hours_remaining": estimated_seconds / 3600,
    }


def recommend_action(analysis: Dict, config: Dict) -> str:
    """Recommend whether to continue, stop, or adjust training."""
    recommendations = []
    
    # Check overfitting
    if analysis.get("overfitting", False):
        recommendations.append("⚠️  OVERFITTING detected. Consider stopping or reducing learning rate.")
    
    # Check plateau
    if analysis.get("trend") == "plateau":
        epochs_since = analysis.get("epochs_since_improvement", 0)
        patience = config.get("early_stopping_patience", 5)
        if epochs_since >= patience:
            recommendations.append(f"🛑 PLATEAU for {epochs_since} epochs. Early stopping should trigger.")
        elif epochs_since >= patience // 2:
            recommendations.append(f"⏸️  No improvement for {epochs_since} epochs. Monitor closely.")
    
    # Check degrading
    if analysis.get("trend") == "degrading":
        recommendations.append("📉 Val loss DEGRADING. Consider stopping if this continues.")
    
    # Check improving
    if analysis.get("trend") == "improving":
        recommendations.append("📈 Still improving. Continue training.")
    
    # Check AUROC
    best_auroc = analysis.get("best_auroc", 0)
    if best_auroc > 0.95:
        recommendations.append("🎯 AUROC > 0.95 - Excellent performance!")
    elif best_auroc > 0.90:
        recommendations.append("✅ AUROC > 0.90 - Good performance.")
    elif best_auroc > 0.80:
        recommendations.append("⚠️  AUROC 0.80-0.90 - Moderate. May need more epochs or tuning.")
    elif best_auroc > 0:
        recommendations.append("❌ AUROC < 0.80 - Poor. Review data/model/hyperparams.")
    
    if not recommendations:
        recommendations.append("🔄 Training in progress. Continue monitoring.")
    
    return "\n".join(recommendations)


def print_report(run_dir: str, watch: bool = False, interval: int = 30):
    """Print monitoring report."""
    while True:
        os.system("clear" if os.name != "nt" else "cls")
        
        print("="*70)
        print(f"PHASE 5 TRAINING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Run: {run_dir}")
        print("="*70)
        
        # Load data
        metrics = load_tensorboard_logs(run_dir)
        checkpoints = load_checkpoints(run_dir)
        
        # Config
        config = {}
        if "checkpoint_last.pt" in checkpoints:
            config = checkpoints["checkpoint_last.pt"].get("config", {})
        
        # Basic stats
        print("\n📊 PROGRESS")
        print("-"*40)
        
        if "checkpoint_last.pt" in checkpoints:
            ckpt = checkpoints["checkpoint_last.pt"]
            print(f"  Current epoch: {ckpt['epoch']}")
            print(f"  Global step: {ckpt['global_step']:,}")
            print(f"  Best AUROC: {ckpt['best_auroc']:.4f}")
        
        if "checkpoint_best.pt" in checkpoints:
            best_ckpt = checkpoints["checkpoint_best.pt"]
            print(f"  Best epoch: {best_ckpt['epoch']}")
        
        # Time estimate
        time_est = estimate_time_remaining(metrics, config)
        if time_est.get("status") != "insufficient_data":
            print(f"\n⏱️  TIME")
            print("-"*40)
            print(f"  Progress: {time_est['progress_pct']:.1f}%")
            print(f"  Steps: {time_est['current_step']:,} / {time_est['total_steps']:,}")
            print(f"  Est. remaining: {time_est['estimated_hours_remaining']:.1f} hours")
        
        # Metrics
        print(f"\n📈 METRICS")
        print("-"*40)
        
        for key in ["train/loss", "val/loss", "val/acc", "val/auroc", "val/precision", "val/recall"]:
            if key in metrics and metrics[key]:
                latest = metrics[key][-1][1]
                print(f"  {key}: {latest:.4f}")
        
        # Analysis
        analysis = analyze_convergence(metrics)
        
        print(f"\n🔍 ANALYSIS")
        print("-"*40)
        
        if analysis.get("status") == "insufficient_data":
            print("  Waiting for more data...")
        else:
            print(f"  Trend: {analysis.get('trend', 'unknown')} ({analysis.get('trend_pct', 0):.1f}%)")
            print(f"  Overfitting: {'Yes ⚠️' if analysis.get('overfitting') else 'No ✅'}")
            print(f"  Epochs since best: {analysis.get('epochs_since_improvement', 0)}")
            print(f"  Best AUROC: {analysis.get('best_auroc', 0):.4f} (epoch {analysis.get('best_auroc_epoch', 0)})")
        
        # Recommendation
        print(f"\n💡 RECOMMENDATION")
        print("-"*40)
        print(recommend_action(analysis, config))
        
        # Config summary
        if config:
            print(f"\n⚙️  CONFIG")
            print("-"*40)
            for key in ["arch", "lr", "batch_size", "epochs", "early_stopping_patience"]:
                if key in config:
                    print(f"  {key}: {config[key]}")
        
        if not watch:
            break
        
        print(f"\n[Refreshing in {interval}s... Ctrl+C to stop]")
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser(description="Monitor Phase 5 training")
    ap.add_argument("--run_dir", required=True, help="Training run directory")
    ap.add_argument("--watch", action="store_true", help="Auto-refresh mode")
    ap.add_argument("--interval", type=int, default=30, help="Refresh interval (seconds)")
    ap.add_argument("--analyze", action="store_true", help="Deep analysis and stop")
    args = ap.parse_args()
    
    if not os.path.exists(args.run_dir):
        print(f"[ERROR] Run directory not found: {args.run_dir}")
        return
    
    print_report(args.run_dir, watch=args.watch, interval=args.interval)


if __name__ == "__main__":
    main()


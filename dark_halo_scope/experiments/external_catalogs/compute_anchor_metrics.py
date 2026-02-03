"""
Compute anchor baseline metrics for real-data validation.

This script:
1. Runs inference on known lens cutouts (SLACS, BELLS)
2. Runs inference on hard negative cutouts (ring galaxies, mergers)
3. Computes key metrics:
   - Known lens recovery rate at various thresholds
   - Hard negative contamination rate
   - Score separation between positives and negatives
4. Generates diagnostic plots and reports

Usage:
    python compute_anchor_metrics.py \
        --known-lens-dir ./cutouts/known_lenses/ \
        --hard-neg-dir ./cutouts/hard_negatives/ \
        --checkpoint /path/to/model.pt \
        --output-dir ./results/anchor_baseline/
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, asdict
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Metric Definitions
# =============================================================================

@dataclass
class AnchorMetrics:
    """
    Anchor baseline metrics for real-data validation.
    
    These metrics indicate whether the model is ready for deployment.
    """
    # Known lens recovery
    n_known_lenses: int
    n_recovered_at_0p5: int  # Threshold = 0.5
    n_recovered_at_0p9: int  # Threshold = 0.9
    recovery_rate_0p5: float
    recovery_rate_0p9: float
    
    # Hard negative contamination
    n_hard_negatives: int
    n_contaminated_at_0p5: int
    n_contaminated_at_0p9: int
    contamination_rate_0p5: float
    contamination_rate_0p9: float
    
    # Score distributions
    known_lens_score_median: float
    known_lens_score_mean: float
    known_lens_score_std: float
    hard_neg_score_median: float
    hard_neg_score_mean: float
    hard_neg_score_std: float
    
    # Separation metrics
    score_separation: float  # median_pos - median_neg
    auroc_anchor: float  # AUROC on anchor data
    
    # Decision gate
    passes_anchor_gate: bool
    gate_criteria: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.gate_criteria is None:
            self.gate_criteria = self.evaluate_gate_criteria()
    
    def evaluate_gate_criteria(self) -> Dict[str, bool]:
        """Evaluate whether metrics pass decision gate criteria."""
        criteria = {
            'recovery_0p5_gt_70': self.recovery_rate_0p5 >= 0.70,
            'contamination_0p5_lt_15': self.contamination_rate_0p5 <= 0.15,
            'score_separation_gt_0p4': self.score_separation >= 0.4,
            'auroc_gt_0p8': self.auroc_anchor >= 0.8
        }
        return criteria
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: str) -> 'AnchorMetrics':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# Inference Functions
# =============================================================================

def load_cutouts(cutout_dir: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load cutouts from directory.
    
    Parameters
    ----------
    cutout_dir : str
        Directory containing cutout npz files
    
    Returns
    -------
    stamps : np.ndarray
        Array of shape (N, 3, H, W)
    metadata : pd.DataFrame
        Metadata for each cutout
    """
    cutout_dir = Path(cutout_dir)
    
    # Check for cutouts subdirectory
    if (cutout_dir / "cutouts").exists():
        cutout_dir = cutout_dir / "cutouts"
    
    npz_files = sorted(cutout_dir.glob("*.npz"))
    
    if len(npz_files) == 0:
        logger.warning(f"No cutout files found in {cutout_dir}")
        return np.array([]), pd.DataFrame()
    
    stamps = []
    metadata = []
    
    for f in npz_files:
        try:
            data = np.load(f)
            stamps.append(data['stamp'])
            metadata.append({
                'file': str(f),
                'ra': float(data['ra']) if 'ra' in data else np.nan,
                'dec': float(data['dec']) if 'dec' in data else np.nan
            })
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")
    
    if len(stamps) == 0:
        return np.array([]), pd.DataFrame()
    
    stamps = np.stack(stamps, axis=0)
    metadata = pd.DataFrame(metadata)
    
    logger.info(f"Loaded {len(stamps)} cutouts from {cutout_dir}")
    
    return stamps, metadata


def run_inference(
    stamps: np.ndarray,
    checkpoint_path: str,
    batch_size: int = 64,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Run inference on stamps using trained model.
    
    Parameters
    ----------
    stamps : np.ndarray
        Input stamps of shape (N, 3, H, W)
    checkpoint_path : str
        Path to model checkpoint
    batch_size : int
        Batch size for inference
    device : str
        Device to use ('cpu' or 'cuda')
    
    Returns
    -------
    np.ndarray
        Probability scores of shape (N,)
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        logger.error("PyTorch not available")
        raise
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine model architecture from checkpoint
    if 'arch' in checkpoint:
        arch = checkpoint['arch']
    else:
        arch = 'convnext_tiny'  # default
    
    # Build model
    model = _build_model(arch, in_channels=stamps.shape[1])
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Run inference
    scores = []
    
    with torch.no_grad():
        for i in range(0, len(stamps), batch_size):
            batch = stamps[i:i+batch_size]
            batch = torch.from_numpy(batch).float().to(device)
            
            # Normalize (match training normalization)
            batch = _normalize_batch(batch)
            
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            scores.extend(probs.tolist())
    
    return np.array(scores)


def _build_model(arch: str, in_channels: int = 3):
    """Build model architecture."""
    import torch.nn as nn
    
    if arch == 'convnext_tiny':
        try:
            from torchvision.models import convnext_tiny
            model = convnext_tiny(weights=None)
            # Modify first conv for different input channels
            model.features[0][0] = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)
            # Modify classifier for binary output
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        except ImportError:
            # Fallback to simple CNN
            model = _simple_cnn(in_channels)
    elif arch == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        model = _simple_cnn(in_channels)
    
    return model


def _simple_cnn(in_channels: int):
    """Simple CNN fallback."""
    import torch.nn as nn
    
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 1)
    )


def _normalize_batch(batch):
    """Normalize batch to match training normalization."""
    import torch
    
    # Percentile normalization
    for i in range(batch.shape[0]):
        for c in range(batch.shape[1]):
            channel = batch[i, c]
            p2, p98 = torch.quantile(channel, torch.tensor([0.02, 0.98]).to(channel.device))
            batch[i, c] = (channel - p2) / (p98 - p2 + 1e-6)
            batch[i, c] = torch.clamp(batch[i, c], -1, 3)
    
    return batch


# =============================================================================
# Metric Computation
# =============================================================================

def compute_metrics(
    known_lens_scores: np.ndarray,
    hard_neg_scores: np.ndarray,
    thresholds: List[float] = [0.5, 0.9]
) -> AnchorMetrics:
    """
    Compute anchor baseline metrics.
    
    Parameters
    ----------
    known_lens_scores : np.ndarray
        Scores for known lenses (should be high)
    hard_neg_scores : np.ndarray
        Scores for hard negatives (should be low)
    thresholds : list
        Thresholds to evaluate at
    
    Returns
    -------
    AnchorMetrics
        Computed metrics
    """
    from sklearn.metrics import roc_auc_score
    
    # Recovery rates
    n_known = len(known_lens_scores)
    n_hard_neg = len(hard_neg_scores)
    
    recovered_0p5 = np.sum(known_lens_scores >= 0.5)
    recovered_0p9 = np.sum(known_lens_scores >= 0.9)
    
    contaminated_0p5 = np.sum(hard_neg_scores >= 0.5)
    contaminated_0p9 = np.sum(hard_neg_scores >= 0.9)
    
    # Score statistics
    known_median = float(np.median(known_lens_scores)) if n_known > 0 else 0.0
    known_mean = float(np.mean(known_lens_scores)) if n_known > 0 else 0.0
    known_std = float(np.std(known_lens_scores)) if n_known > 0 else 0.0
    
    hard_neg_median = float(np.median(hard_neg_scores)) if n_hard_neg > 0 else 1.0
    hard_neg_mean = float(np.mean(hard_neg_scores)) if n_hard_neg > 0 else 1.0
    hard_neg_std = float(np.std(hard_neg_scores)) if n_hard_neg > 0 else 0.0
    
    # Separation
    score_separation = known_median - hard_neg_median
    
    # AUROC
    if n_known > 0 and n_hard_neg > 0:
        labels = np.concatenate([np.ones(n_known), np.zeros(n_hard_neg)])
        scores = np.concatenate([known_lens_scores, hard_neg_scores])
        auroc = float(roc_auc_score(labels, scores))
    else:
        auroc = 0.5
    
    # Build metrics object
    metrics = AnchorMetrics(
        n_known_lenses=n_known,
        n_recovered_at_0p5=int(recovered_0p5),
        n_recovered_at_0p9=int(recovered_0p9),
        recovery_rate_0p5=float(recovered_0p5 / n_known) if n_known > 0 else 0.0,
        recovery_rate_0p9=float(recovered_0p9 / n_known) if n_known > 0 else 0.0,
        n_hard_negatives=n_hard_neg,
        n_contaminated_at_0p5=int(contaminated_0p5),
        n_contaminated_at_0p9=int(contaminated_0p9),
        contamination_rate_0p5=float(contaminated_0p5 / n_hard_neg) if n_hard_neg > 0 else 0.0,
        contamination_rate_0p9=float(contaminated_0p9 / n_hard_neg) if n_hard_neg > 0 else 0.0,
        known_lens_score_median=known_median,
        known_lens_score_mean=known_mean,
        known_lens_score_std=known_std,
        hard_neg_score_median=hard_neg_median,
        hard_neg_score_mean=hard_neg_mean,
        hard_neg_score_std=hard_neg_std,
        score_separation=score_separation,
        auroc_anchor=auroc,
        passes_anchor_gate=False  # Will be evaluated in __post_init__
    )
    
    # Evaluate gate
    metrics.gate_criteria = metrics.evaluate_gate_criteria()
    metrics.passes_anchor_gate = all(metrics.gate_criteria.values())
    
    return metrics


# =============================================================================
# Reporting
# =============================================================================

def generate_report(
    metrics: AnchorMetrics,
    output_dir: str,
    known_lens_scores: np.ndarray = None,
    hard_neg_scores: np.ndarray = None
) -> str:
    """
    Generate anchor baseline report.
    
    Parameters
    ----------
    metrics : AnchorMetrics
        Computed metrics
    output_dir : str
        Output directory
    known_lens_scores, hard_neg_scores : np.ndarray
        Score arrays for histograms
    
    Returns
    -------
    str
        Path to report file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "anchor_metrics.json")
    metrics.to_json(metrics_path)
    
    # Generate markdown report
    report_lines = [
        "# Anchor Baseline Validation Report",
        "",
        f"**Date**: {pd.Timestamp.now().isoformat()}",
        "",
        "## Summary",
        "",
        f"**Passes Anchor Gate**: {'YES' if metrics.passes_anchor_gate else 'NO'}",
        "",
        "### Gate Criteria",
        "",
        "| Criterion | Value | Threshold | Pass |",
        "|-----------|-------|-----------|------|",
        f"| Known lens recovery @ 0.5 | {metrics.recovery_rate_0p5:.1%} | ≥70% | {'✓' if metrics.gate_criteria['recovery_0p5_gt_70'] else '✗'} |",
        f"| Hard negative contamination @ 0.5 | {metrics.contamination_rate_0p5:.1%} | ≤15% | {'✓' if metrics.gate_criteria['contamination_0p5_lt_15'] else '✗'} |",
        f"| Score separation | {metrics.score_separation:.3f} | ≥0.4 | {'✓' if metrics.gate_criteria['score_separation_gt_0p4'] else '✗'} |",
        f"| AUROC | {metrics.auroc_anchor:.3f} | ≥0.8 | {'✓' if metrics.gate_criteria['auroc_gt_0p8'] else '✗'} |",
        "",
        "## Detailed Metrics",
        "",
        "### Known Lens Recovery",
        "",
        f"- Total known lenses: {metrics.n_known_lenses}",
        f"- Recovered @ threshold 0.5: {metrics.n_recovered_at_0p5} ({metrics.recovery_rate_0p5:.1%})",
        f"- Recovered @ threshold 0.9: {metrics.n_recovered_at_0p9} ({metrics.recovery_rate_0p9:.1%})",
        "",
        "### Hard Negative Contamination",
        "",
        f"- Total hard negatives: {metrics.n_hard_negatives}",
        f"- Contaminated @ threshold 0.5: {metrics.n_contaminated_at_0p5} ({metrics.contamination_rate_0p5:.1%})",
        f"- Contaminated @ threshold 0.9: {metrics.n_contaminated_at_0p9} ({metrics.contamination_rate_0p9:.1%})",
        "",
        "### Score Distributions",
        "",
        "| Metric | Known Lenses | Hard Negatives |",
        "|--------|--------------|----------------|",
        f"| Median | {metrics.known_lens_score_median:.3f} | {metrics.hard_neg_score_median:.3f} |",
        f"| Mean | {metrics.known_lens_score_mean:.3f} | {metrics.hard_neg_score_mean:.3f} |",
        f"| Std | {metrics.known_lens_score_std:.3f} | {metrics.hard_neg_score_std:.3f} |",
        "",
        "## Interpretation",
        "",
    ]
    
    if metrics.passes_anchor_gate:
        report_lines.extend([
            "The model **passes** the anchor baseline gate and is ready for:",
            "- SOTA benchmark comparisons",
            "- Survey-scale candidate ranking",
            "- Selection function estimation",
        ])
    else:
        report_lines.extend([
            "The model **fails** the anchor baseline gate. Before proceeding:",
            "",
        ])
        if not metrics.gate_criteria['recovery_0p5_gt_70']:
            report_lines.append("- **Improve recovery**: Model may need better sim-to-real transfer")
        if not metrics.gate_criteria['contamination_0p5_lt_15']:
            report_lines.append("- **Reduce contamination**: Add more hard negatives to training")
        if not metrics.gate_criteria['score_separation_gt_0p4']:
            report_lines.append("- **Improve separation**: Model may be overfit to synthetic features")
    
    # Write report
    report_path = os.path.join(output_dir, "anchor_baseline_report.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Report saved to {report_path}")
    
    # Generate histograms if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        if known_lens_scores is not None and hard_neg_scores is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(known_lens_scores, bins=50, alpha=0.6, label=f'Known Lenses (n={len(known_lens_scores)})', color='green')
            ax.hist(hard_neg_scores, bins=50, alpha=0.6, label=f'Hard Negatives (n={len(hard_neg_scores)})', color='red')
            
            ax.axvline(0.5, color='k', linestyle='--', label='Threshold 0.5')
            ax.axvline(metrics.known_lens_score_median, color='green', linestyle=':', label=f'Lens median: {metrics.known_lens_score_median:.2f}')
            ax.axvline(metrics.hard_neg_score_median, color='red', linestyle=':', label=f'HN median: {metrics.hard_neg_score_median:.2f}')
            
            ax.set_xlabel('Model Score')
            ax.set_ylabel('Count')
            ax.set_title('Anchor Baseline Score Distributions')
            ax.legend()
            
            hist_path = os.path.join(output_dir, "score_histograms.png")
            fig.savefig(hist_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Histogram saved to {hist_path}")
            
    except ImportError:
        logger.warning("matplotlib not available, skipping histograms")
    
    return report_path


# =============================================================================
# Main Pipeline
# =============================================================================

def run_anchor_baseline(
    known_lens_dir: str,
    hard_neg_dir: str,
    checkpoint_path: str,
    output_dir: str,
    device: str = 'cpu',
    batch_size: int = 64
) -> AnchorMetrics:
    """
    Run complete anchor baseline validation pipeline.
    
    Parameters
    ----------
    known_lens_dir : str
        Directory with known lens cutouts
    hard_neg_dir : str
        Directory with hard negative cutouts
    checkpoint_path : str
        Path to model checkpoint
    output_dir : str
        Output directory for results
    device : str
        Device for inference
    batch_size : int
        Batch size
    
    Returns
    -------
    AnchorMetrics
        Computed metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load cutouts
    known_stamps, known_meta = load_cutouts(known_lens_dir)
    hard_neg_stamps, hard_neg_meta = load_cutouts(hard_neg_dir)
    
    if len(known_stamps) == 0:
        logger.error("No known lens cutouts found")
        raise ValueError("No known lens cutouts found")
    
    if len(hard_neg_stamps) == 0:
        logger.error("No hard negative cutouts found")
        raise ValueError("No hard negative cutouts found")
    
    # Run inference
    logger.info("Running inference on known lenses...")
    known_scores = run_inference(known_stamps, checkpoint_path, batch_size, device)
    
    logger.info("Running inference on hard negatives...")
    hard_neg_scores = run_inference(hard_neg_stamps, checkpoint_path, batch_size, device)
    
    # Save scores
    known_meta['score'] = known_scores
    hard_neg_meta['score'] = hard_neg_scores
    
    known_meta.to_parquet(os.path.join(output_dir, "known_lens_scores.parquet"), index=False)
    hard_neg_meta.to_parquet(os.path.join(output_dir, "hard_neg_scores.parquet"), index=False)
    
    # Compute metrics
    metrics = compute_metrics(known_scores, hard_neg_scores)
    
    # Generate report
    generate_report(metrics, output_dir, known_scores, hard_neg_scores)
    
    # Log summary
    logger.info("=" * 60)
    logger.info("ANCHOR BASELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Known lens recovery @ 0.5: {metrics.recovery_rate_0p5:.1%}")
    logger.info(f"Hard neg contamination @ 0.5: {metrics.contamination_rate_0p5:.1%}")
    logger.info(f"Score separation: {metrics.score_separation:.3f}")
    logger.info(f"AUROC: {metrics.auroc_anchor:.3f}")
    logger.info(f"PASSES GATE: {metrics.passes_anchor_gate}")
    logger.info("=" * 60)
    
    return metrics


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute anchor baseline metrics for real-data validation"
    )
    parser.add_argument(
        "--known-lens-dir",
        type=str,
        required=True,
        help="Directory with known lens cutouts"
    )
    parser.add_argument(
        "--hard-neg-dir",
        type=str,
        required=True,
        help="Directory with hard negative cutouts"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (cpu or cuda)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference"
    )
    
    args = parser.parse_args()
    
    run_anchor_baseline(
        known_lens_dir=args.known_lens_dir,
        hard_neg_dir=args.hard_neg_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()


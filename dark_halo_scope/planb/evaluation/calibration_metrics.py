"""
Calibration and Evaluation Metrics for Lens Finding.

Key insight: A score of 0.99 should mean "99% likely to be a lens" 
under realistic prevalence. This module provides:

1. Calibration curves (reliability diagrams)
2. Expected Calibration Error (ECE)
3. Prevalence-adjusted metrics
4. Selection function surfaces

Reference: Huang et al. (2508.20087) - they use top-0.01% threshold
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class CalibrationResult:
    """Results from calibration analysis."""
    # Binned calibration
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    bin_accuracy: np.ndarray  # Actual positive rate per bin
    bin_confidence: np.ndarray  # Mean predicted probability per bin
    bin_counts: np.ndarray  # Number of samples per bin
    
    # Scalar metrics
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float  # Brier score (MSE of probabilities)
    
    # Reliability at key thresholds
    reliability_at_99: float = None  # Actual accuracy when model says 0.99
    reliability_at_95: float = None
    reliability_at_90: float = None


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'uniform'
) -> CalibrationResult:
    """
    Compute calibration metrics.
    
    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_prob: Predicted probabilities
        n_bins: Number of calibration bins
        strategy: 'uniform' for equal-width bins, 'quantile' for equal-count
    
    Returns:
        CalibrationResult with binned calibration and scalar metrics
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    n = len(y_true)
    
    # Define bins
    if strategy == 'uniform':
        bin_edges = np.linspace(0, 1, n_bins + 1)
    else:  # quantile
        bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)
    
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_accuracy = np.zeros(n_bins)
    bin_confidence = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    # Compute binned statistics
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i+1])
        else:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
        
        bin_counts[i] = mask.sum()
        
        if bin_counts[i] > 0:
            bin_accuracy[i] = y_true[mask].mean()
            bin_confidence[i] = y_prob[mask].mean()
    
    # Expected Calibration Error (weighted by bin count)
    weights = bin_counts / n
    ece = np.sum(weights * np.abs(bin_accuracy - bin_confidence))
    
    # Maximum Calibration Error
    mce = np.max(np.abs(bin_accuracy - bin_confidence)[bin_counts > 0])
    
    # Brier score
    brier_score = np.mean((y_prob - y_true) ** 2)
    
    result = CalibrationResult(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        bin_accuracy=bin_accuracy,
        bin_confidence=bin_confidence,
        bin_counts=bin_counts,
        ece=ece,
        mce=mce,
        brier_score=brier_score,
    )
    
    # Reliability at key thresholds
    for thresh, attr in [(0.99, 'reliability_at_99'), 
                         (0.95, 'reliability_at_95'),
                         (0.90, 'reliability_at_90')]:
        mask = y_prob >= thresh
        if mask.sum() > 0:
            setattr(result, attr, y_true[mask].mean())
    
    return result


def adjust_for_prevalence(
    y_prob: np.ndarray,
    train_prevalence: float,
    target_prevalence: float,
) -> np.ndarray:
    """
    Adjust predicted probabilities for different prevalence.
    
    If model was trained with train_prevalence but deployed with 
    target_prevalence, the posterior probabilities need adjustment.
    
    Uses Bayes' theorem:
    P(lens|score, π_target) = P(score|lens) * π_target / P(score)
    
    Args:
        y_prob: Predicted probabilities from model
        train_prevalence: Prevalence in training data (e.g., 0.01 for 1%)
        target_prevalence: Prevalence in deployment (e.g., 1e-5 for DR10)
    
    Returns:
        Adjusted probabilities
    """
    # Avoid division by zero
    epsilon = 1e-10
    
    # Convert to odds
    train_odds = train_prevalence / (1 - train_prevalence + epsilon)
    target_odds = target_prevalence / (1 - target_prevalence + epsilon)
    
    # Likelihood ratio from model output
    # P(score|lens) / P(score|non-lens) ∝ y_prob / (1 - y_prob)
    likelihood_ratio = y_prob / (1 - y_prob + epsilon)
    
    # Adjust for prevalence shift
    adjustment = target_odds / (train_odds + epsilon)
    adjusted_odds = likelihood_ratio * adjustment
    
    # Convert back to probability
    adjusted_prob = adjusted_odds / (1 + adjusted_odds)
    
    return np.clip(adjusted_prob, 0, 1)


@dataclass 
class SelectionFunctionResult:
    """Selection function analysis results."""
    # Stratification variables and bins
    strata_names: List[str]
    strata_bins: Dict[str, np.ndarray]
    
    # Recall (completeness) by stratum
    recall_by_stratum: Dict[Tuple, float]
    recall_uncertainty: Dict[Tuple, float]  # Bootstrap CI
    
    # FPR by contaminant category
    fpr_by_category: Dict[str, float]
    fpr_uncertainty: Dict[str, float]
    
    # Summary statistics
    mean_recall: float
    min_recall: float
    max_recall: float
    recall_spread: float  # max - min


def compute_selection_function(
    positives: List[Dict],  # Must have 'score', 'is_detected', and strata info
    negatives: List[Dict],  # Same structure
    threshold: float = 0.5,
    strata: List[str] = None,  # e.g., ['exp_bin', 'psf_bin', 'depth_bin']
    n_bootstrap: int = 100,
) -> SelectionFunctionResult:
    """
    Compute selection function (recall) as function of observing conditions.
    
    This is the core of Option 1: What is the model's sensitivity across
    the parameter space of the survey?
    
    Args:
        positives: List of positive samples with scores and strata info
        negatives: List of negative samples with scores and strata info
        threshold: Detection threshold
        strata: List of keys to use for stratification
        n_bootstrap: Number of bootstrap samples for uncertainty
    
    Returns:
        SelectionFunctionResult with recall surfaces
    """
    if strata is None:
        strata = ['exp_bin', 'psf_bin']
    
    # Group positives by strata
    from collections import defaultdict
    pos_by_strata = defaultdict(list)
    
    for p in positives:
        key = tuple(p.get(s, 0) for s in strata)
        pos_by_strata[key].append(p)
    
    # Compute recall per stratum
    recall_by_stratum = {}
    recall_uncertainty = {}
    
    rng = np.random.default_rng(42)
    
    for key, samples in pos_by_strata.items():
        n = len(samples)
        if n == 0:
            continue
        
        # Detected if score >= threshold
        detected = [s.get('score', 0) >= threshold for s in samples]
        recall = np.mean(detected)
        recall_by_stratum[key] = recall
        
        # Bootstrap CI
        if n >= 5:
            boot_recalls = []
            for _ in range(n_bootstrap):
                boot_idx = rng.choice(n, size=n, replace=True)
                boot_detected = [detected[i] for i in boot_idx]
                boot_recalls.append(np.mean(boot_detected))
            recall_uncertainty[key] = np.std(boot_recalls)
        else:
            # Wilson score interval for small n
            recall_uncertainty[key] = np.sqrt(recall * (1 - recall) / (n + 1))
    
    # Group negatives by category for FPR
    neg_by_cat = defaultdict(list)
    for n_sample in negatives:
        cat = n_sample.get('category', 'unknown')
        neg_by_cat[cat].append(n_sample)
    
    fpr_by_category = {}
    fpr_uncertainty = {}
    
    for cat, samples in neg_by_cat.items():
        n = len(samples)
        if n == 0:
            continue
        
        false_pos = [s.get('score', 0) >= threshold for s in samples]
        fpr = np.mean(false_pos)
        fpr_by_category[cat] = fpr
        
        if n >= 5:
            boot_fprs = []
            for _ in range(n_bootstrap):
                boot_idx = rng.choice(n, size=n, replace=True)
                boot_fp = [false_pos[i] for i in boot_idx]
                boot_fprs.append(np.mean(boot_fp))
            fpr_uncertainty[cat] = np.std(boot_fprs)
        else:
            fpr_uncertainty[cat] = np.sqrt(fpr * (1 - fpr) / (n + 1))
    
    # Summary stats
    recall_values = list(recall_by_stratum.values())
    mean_recall = np.mean(recall_values) if recall_values else 0
    min_recall = np.min(recall_values) if recall_values else 0
    max_recall = np.max(recall_values) if recall_values else 0
    
    return SelectionFunctionResult(
        strata_names=strata,
        strata_bins={},  # Filled by caller
        recall_by_stratum=recall_by_stratum,
        recall_uncertainty=recall_uncertainty,
        fpr_by_category=fpr_by_category,
        fpr_uncertainty=fpr_uncertainty,
        mean_recall=mean_recall,
        min_recall=min_recall,
        max_recall=max_recall,
        recall_spread=max_recall - min_recall,
    )


def compute_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_thresholds: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Compute ROC and PR curves.
    
    Returns:
        Dict with 'fpr', 'tpr', 'precision', 'recall', 'thresholds'
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    thresholds = np.linspace(0, 1, n_thresholds)
    
    tpr = np.zeros(n_thresholds)  # True positive rate = recall
    fpr = np.zeros(n_thresholds)  # False positive rate
    precision = np.zeros(n_thresholds)
    
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    
    for i, thresh in enumerate(thresholds):
        pred_pos = y_prob >= thresh
        
        tp = (pred_pos & (y_true == 1)).sum()
        fp = (pred_pos & (y_true == 0)).sum()
        
        tpr[i] = tp / n_pos if n_pos > 0 else 0
        fpr[i] = fp / n_neg if n_neg > 0 else 0
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 1
    
    # AUC-ROC (trapezoidal)
    sorted_idx = np.argsort(fpr)
    auc_roc = np.trapz(tpr[sorted_idx], fpr[sorted_idx])
    
    # AUC-PR
    sorted_idx = np.argsort(tpr)
    auc_pr = np.trapz(precision[sorted_idx], tpr[sorted_idx])
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': tpr,  # Same as TPR
        'thresholds': thresholds,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
    }


class EvaluationFramework:
    """
    Complete evaluation framework for lens finding.
    
    Produces:
    1. Calibration curves + ECE
    2. ROC/PR curves + AUC
    3. Selection function surfaces
    4. FPR by contaminant category
    5. Prevalence-adjusted reliability
    """
    
    def __init__(
        self,
        name: str = "lens_finder_eval",
        output_dir: str = None,
    ):
        self.name = name
        self.output_dir = Path(output_dir) if output_dir else None
        
        self.results = {}
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metadata: List[Dict] = None,
        strata: List[str] = None,
        train_prevalence: float = 0.01,
        target_prevalence: float = 1e-5,
    ) -> Dict:
        """
        Run full evaluation.
        
        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities
            metadata: Per-sample metadata (for stratification)
            strata: Keys to stratify by
            train_prevalence: Prevalence in training data
            target_prevalence: Prevalence in deployment
        
        Returns:
            Dict with all evaluation results
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        
        # 1. Calibration
        calibration = compute_calibration(y_prob, y_true)
        
        # 2. ROC/PR
        curves = compute_roc_pr_curves(y_true, y_prob)
        
        # 3. Prevalence adjustment
        adjusted_prob = adjust_for_prevalence(
            y_prob, train_prevalence, target_prevalence
        )
        calibration_adjusted = compute_calibration(adjusted_prob, y_true)
        
        # 4. Selection function (if metadata provided)
        selection_func = None
        if metadata:
            positives = [m for m, y in zip(metadata, y_true) if y == 1]
            negatives = [m for m, y in zip(metadata, y_true) if y == 0]
            
            # Add scores
            for m, p in zip(metadata, y_prob):
                m['score'] = p
            
            selection_func = compute_selection_function(
                positives, negatives, threshold=0.5, strata=strata
            )
        
        # Compile results
        self.results = {
            'calibration': {
                'ece': calibration.ece,
                'mce': calibration.mce,
                'brier_score': calibration.brier_score,
                'reliability_at_99': calibration.reliability_at_99,
                'reliability_at_95': calibration.reliability_at_95,
            },
            'calibration_adjusted': {
                'ece': calibration_adjusted.ece,
                'reliability_at_99': calibration_adjusted.reliability_at_99,
            },
            'roc_pr': {
                'auc_roc': curves['auc_roc'],
                'auc_pr': curves['auc_pr'],
            },
            'prevalence': {
                'train': train_prevalence,
                'target': target_prevalence,
            },
        }
        
        if selection_func:
            self.results['selection_function'] = {
                'mean_recall': selection_func.mean_recall,
                'min_recall': selection_func.min_recall,
                'max_recall': selection_func.max_recall,
                'recall_spread': selection_func.recall_spread,
                'fpr_by_category': selection_func.fpr_by_category,
            }
        
        # Save if output dir provided
        if self.output_dir:
            self.save()
        
        return self.results
    
    def save(self):
        """Save results to disk."""
        if not self.output_dir:
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON summary
        summary_path = self.output_dir / f"{self.name}_summary.json"
        
        # Convert numpy to python for JSON
        def to_python(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {str(k): to_python(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [to_python(v) for v in obj]
            return obj
        
        with open(summary_path, 'w') as f:
            json.dump(to_python(self.results), f, indent=2)
    
    def print_summary(self):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print(f"EVALUATION SUMMARY: {self.name}")
        print("="*60)
        
        if 'calibration' in self.results:
            c = self.results['calibration']
            print("\nCalibration:")
            print(f"  ECE: {c['ece']:.4f}")
            print(f"  MCE: {c['mce']:.4f}")
            print(f"  Brier Score: {c['brier_score']:.4f}")
            if c.get('reliability_at_99'):
                print(f"  Reliability at 0.99: {c['reliability_at_99']:.4f}")
            if c.get('reliability_at_95'):
                print(f"  Reliability at 0.95: {c['reliability_at_95']:.4f}")
        
        if 'roc_pr' in self.results:
            r = self.results['roc_pr']
            print("\nROC/PR:")
            print(f"  AUC-ROC: {r['auc_roc']:.4f}")
            print(f"  AUC-PR: {r['auc_pr']:.4f}")
        
        if 'selection_function' in self.results:
            s = self.results['selection_function']
            print("\nSelection Function:")
            print(f"  Mean Recall: {s['mean_recall']:.4f}")
            print(f"  Min Recall: {s['min_recall']:.4f}")
            print(f"  Max Recall: {s['max_recall']:.4f}")
            print(f"  Recall Spread: {s['recall_spread']:.4f}")
            
            if s.get('fpr_by_category'):
                print("\n  FPR by Category:")
                for cat, fpr in s['fpr_by_category'].items():
                    print(f"    {cat}: {fpr:.4f}")
        
        print("="*60)


def main():
    """Test evaluation framework with synthetic data."""
    np.random.seed(42)
    
    # Generate synthetic predictions
    n_pos = 100
    n_neg = 10000
    
    # Positives: higher scores
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    
    # Simulate imperfect classifier
    pos_scores = np.random.beta(3, 1, n_pos)  # Skewed high
    neg_scores = np.random.beta(1, 3, n_neg)  # Skewed low
    y_prob = np.concatenate([pos_scores, neg_scores])
    
    # Create metadata
    metadata = []
    for i in range(len(y_true)):
        metadata.append({
            'exp_bin': np.random.randint(0, 4),
            'psf_bin': np.random.randint(0, 4),
            'category': np.random.choice(['ring', 'spiral', 'elliptical', 'merger'])
        })
    
    # Run evaluation
    framework = EvaluationFramework(
        name="synthetic_test",
        output_dir="dark_halo_scope/planb/evaluation/test_results"
    )
    
    results = framework.evaluate(
        y_true, y_prob,
        metadata=metadata,
        strata=['exp_bin', 'psf_bin'],
        train_prevalence=0.01,
        target_prevalence=1e-5,
    )
    
    framework.print_summary()


if __name__ == "__main__":
    main()

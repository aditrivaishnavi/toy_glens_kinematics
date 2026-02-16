"""thetae_stratification.py

Helper to compute per-bin AUROC by theta_E.

IMPORTANT: theta_E is only defined for positives (lensed images).
Negatives have NaN theta_E. For θ_E stratified evaluation:
- Filter positives by θ_E bin
- Sample equal number of negatives (random from all negatives)
- Compute AUC on this balanced subset
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score

@dataclass
class ThetaEBinResult:
    lo: float
    hi: float
    n_pos: int
    n_neg: int
    auc: float

def auroc_by_thetae(
    y_true: np.ndarray,
    y_score: np.ndarray,
    theta_e: np.ndarray,
    bins: List[Tuple[float, float]],
    min_count: int = 100,
    seed: int = 42,
) -> List[ThetaEBinResult]:
    """
    Compute AUROC for each theta_E bin.
    
    For each bin:
    1. Select positives with theta_E in [lo, hi)
    2. Randomly sample equal number of negatives
    3. Compute AUROC on this balanced subset
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_score: Model prediction scores
        theta_e: theta_E values (NaN for negatives)
        bins: List of (lo, hi) tuples defining bins
        min_count: Minimum positives per bin to compute AUC
        seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    
    # Identify all negatives (for sampling)
    neg_mask = y_true == 0
    neg_indices = np.where(neg_mask)[0]
    
    out: List[ThetaEBinResult] = []
    for lo, hi in bins:
        # Find positives in this theta_E bin
        pos_in_bin = (y_true == 1) & (theta_e >= lo) & (theta_e < hi)
        pos_indices = np.where(pos_in_bin)[0]
        n_pos = len(pos_indices)
        
        if n_pos < min_count:
            out.append(ThetaEBinResult(lo, hi, n_pos, 0, float("nan")))
            continue
        
        # Sample equal number of negatives
        n_neg_sample = min(n_pos, len(neg_indices))
        if n_neg_sample == 0:
            out.append(ThetaEBinResult(lo, hi, n_pos, 0, float("nan")))
            continue
        
        neg_sample = rng.choice(neg_indices, size=n_neg_sample, replace=False)
        
        # Combine
        all_indices = np.concatenate([pos_indices, neg_sample])
        yt = y_true[all_indices]
        ys = y_score[all_indices]
        
        if len(np.unique(yt)) < 2:
            out.append(ThetaEBinResult(lo, hi, n_pos, n_neg_sample, float("nan")))
            continue
        
        auc = float(roc_auc_score(yt, ys))
        out.append(ThetaEBinResult(lo, hi, n_pos, n_neg_sample, auc))
    
    return out

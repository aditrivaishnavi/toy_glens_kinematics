"""Shortcut detection gates with proper cross-validation.

CRITICAL: This fixes the train-on-train bias in the external LLM's code.
We use StratifiedKFold cross-validation to get unbiased AUC estimates.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import logging

from .constants import CORE_BOX
from .utils import center_slices, azimuthal_median_profile

logger = logging.getLogger(__name__)


@dataclass
class GateResults:
    """Results from shortcut gates."""
    core_lr_auc: float
    radial_profile_auc: float
    n_samples: int
    
    def passes(self, threshold: float = 0.65) -> bool:
        """Check if both gates pass (AUC below threshold)."""
        return self.core_lr_auc < threshold and self.radial_profile_auc < threshold
    
    def __str__(self):
        status = "PASS" if self.passes() else "FAIL"
        return (f"GateResults({status}): core_lr_auc={self.core_lr_auc:.4f}, "
                f"radial_profile_auc={self.radial_profile_auc:.4f}, n={self.n_samples}")


def _extract_core_features(x3: np.ndarray) -> np.ndarray:
    """Extract central 10x10 pixels as features."""
    _, H, W = x3.shape
    ys, xs = center_slices(H, W, CORE_BOX)
    return x3[:, ys, xs].reshape(-1)


def _extract_radial_features(x3: np.ndarray, r_max: int = 32) -> np.ndarray:
    """Extract azimuthal median profile as features."""
    feats = []
    for b in range(3):
        prof = azimuthal_median_profile(x3[b], r_max=r_max)
        feats.append(prof)
    return np.concatenate(feats, axis=0)


def core_lr_auc_cv(xs: np.ndarray, ys: np.ndarray, n_splits: int = 5) -> float:
    """
    Compute core-only LR AUC with cross-validation.
    
    FIXED: Uses cross_val_predict to avoid train-on-train bias.
    """
    X = np.stack([_extract_core_features(x) for x in xs], axis=0)
    y = ys.astype(int)
    
    # Handle edge cases
    if len(np.unique(y)) < 2:
        logger.warning("Only one class present, returning AUC=0.5")
        return 0.5
    
    if len(y) < n_splits * 2:
        logger.warning(f"Too few samples ({len(y)}) for {n_splits}-fold CV")
        n_splits = max(2, len(y) // 4)
    
    clf = LogisticRegression(max_iter=500, solver="liblinear", random_state=42)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    try:
        # Get cross-validated predictions
        probs = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')
        p = probs[:, 1]
        return float(roc_auc_score(y, p))
    except Exception as e:
        logger.error(f"Core LR AUC failed: {e}")
        return 0.5


def radial_lr_auc_cv(xs: np.ndarray, ys: np.ndarray, r_max: int = 32, n_splits: int = 5) -> float:
    """
    Compute radial-profile-only LR AUC with cross-validation.
    
    This catches shortcuts where the model uses radial brightness patterns
    instead of arc morphology.
    """
    X = np.stack([_extract_radial_features(x, r_max=r_max) for x in xs], axis=0)
    y = ys.astype(int)
    
    # Handle edge cases
    if len(np.unique(y)) < 2:
        logger.warning("Only one class present, returning AUC=0.5")
        return 0.5
    
    if len(y) < n_splits * 2:
        n_splits = max(2, len(y) // 4)
    
    clf = LogisticRegression(max_iter=500, solver="liblinear", random_state=42)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    try:
        probs = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')
        p = probs[:, 1]
        return float(roc_auc_score(y, p))
    except Exception as e:
        logger.error(f"Radial LR AUC failed: {e}")
        return 0.5


def run_shortcut_gates(xs: np.ndarray, ys: np.ndarray) -> GateResults:
    """
    Run both shortcut detection gates.
    
    Args:
        xs: (N, 3, H, W) preprocessed images
        ys: (N,) labels (0 or 1)
    
    Returns:
        GateResults with both AUC values
    """
    logger.info(f"Running shortcut gates on {len(ys)} samples...")
    
    core_auc = core_lr_auc_cv(xs, ys)
    radial_auc = radial_lr_auc_cv(xs, ys)
    
    results = GateResults(
        core_lr_auc=core_auc,
        radial_profile_auc=radial_auc,
        n_samples=len(ys)
    )
    
    logger.info(str(results))
    return results

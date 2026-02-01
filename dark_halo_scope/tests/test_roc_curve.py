"""Unit tests for ROC curve implementation.

These tests verify the fixes from LLM1/LLM2 analysis:
- A2: Keep LAST element of tied-score run (not first)
- A3: Prepend (0,0) origin point
- B3: Curve should reach (1,1) even with tied scores
"""

import numpy as np
import pytest
from typing import Tuple, Dict, Sequence
import math


# Inline implementation of the fixed ROC functions for testing
# (avoids importing torch-dependent training script)

def roc_curve_np(scores: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve with proper tie handling."""
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    yy = y[order].astype(np.int64)
    P = int(yy.sum())
    N = len(yy) - P
    
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])
    
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    
    # FIX: Keep LAST element of each tied-score run
    distinct = np.r_[np.diff(s) != 0, True]
    
    tp = tp[distinct]
    fp = fp[distinct]
    thr = s[distinct]
    
    tpr = tp / P
    fpr = fp / N
    
    # FIX: Prepend (0, 0) origin point
    fpr = np.r_[0.0, fpr]
    tpr = np.r_[0.0, tpr]
    thr = np.r_[np.inf, thr]
    
    return fpr, tpr, thr


def tpr_at_fpr(scores: np.ndarray, y: np.ndarray, fpr_targets: Sequence[float]) -> Dict[str, float]:
    fpr, tpr, _ = roc_curve_np(scores, y)
    out = {}
    if len(fpr) == 0:
        for ft in fpr_targets:
            out[f"tpr@fpr{ft:g}"] = math.nan
        return out
    for ft in fpr_targets:
        idx = np.searchsorted(fpr, ft, side="right") - 1
        out[f"tpr@fpr{ft:g}"] = float(tpr[idx]) if idx >= 0 else 0.0
    return out


def fpr_at_tpr(scores: np.ndarray, y: np.ndarray, tpr_targets: Sequence[float]) -> Dict[str, float]:
    fpr, tpr, _ = roc_curve_np(scores, y)
    out = {}
    if len(fpr) == 0:
        for tt in tpr_targets:
            out[f"fpr@tpr{tt:.2f}"] = math.nan
        return out
    for tt in tpr_targets:
        idx = np.searchsorted(tpr, tt, side="left")
        out[f"fpr@tpr{tt:.2f}"] = float(fpr[idx]) if idx < len(fpr) else math.nan
    return out


class TestROCCurveTieHandling:
    """Test that ROC curve handles tied scores correctly."""
    
    def test_roc_includes_origin(self):
        """ROC curve should start at (0, 0)."""
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        y = np.array([1, 1, 0, 1, 0])
        
        fpr, tpr, _ = roc_curve_np(scores, y)
        
        assert fpr[0] == 0.0, "FPR should start at 0"
        assert tpr[0] == 0.0, "TPR should start at 0"
    
    def test_roc_reaches_one_one(self):
        """ROC curve should end at (1, 1)."""
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        y = np.array([1, 1, 0, 1, 0])
        
        fpr, tpr, _ = roc_curve_np(scores, y)
        
        assert fpr[-1] == 1.0, "FPR should end at 1"
        assert tpr[-1] == 1.0, "TPR should end at 1"
    
    def test_roc_with_tied_scores_at_end(self):
        """ROC should handle tied scores at the end correctly.
        
        This was the main bug: when many samples have the same score at the end,
        the curve would truncate instead of reaching (1, 1).
        """
        # 3 high-scoring positives, 5 tied low-scoring samples (mix of pos/neg)
        scores = np.array([0.9, 0.8, 0.7, 0.01, 0.01, 0.01, 0.01, 0.01])
        y = np.array([1, 1, 0, 1, 0, 0, 1, 0])
        
        fpr, tpr, _ = roc_curve_np(scores, y)
        
        # Must include origin
        assert fpr[0] == 0.0, "FPR should start at 0"
        assert tpr[0] == 0.0, "TPR should start at 0"
        
        # Must reach (1, 1) even with ties
        assert fpr[-1] == 1.0, f"FPR should end at 1, got {fpr[-1]}"
        assert tpr[-1] == 1.0, f"TPR should end at 1, got {tpr[-1]}"
    
    def test_roc_with_binary_scores(self):
        """ROC should handle extreme binary-like scores (calibration collapse scenario)."""
        # Simulate calibration collapse: scores are 0.001 or 0.999
        scores = np.array([0.999, 0.999, 0.999, 0.001, 0.001, 0.001, 0.001, 0.001])
        y = np.array([1, 1, 0, 1, 0, 0, 0, 0])  # 3 pos, 5 neg
        
        fpr, tpr, _ = roc_curve_np(scores, y)
        
        assert fpr[0] == 0.0
        assert tpr[0] == 0.0
        assert fpr[-1] == 1.0
        assert tpr[-1] == 1.0
    
    def test_roc_single_class_returns_valid_curve(self):
        """ROC should return valid curve even for single-class data."""
        # All positives
        scores = np.array([0.9, 0.8, 0.7])
        y = np.array([1, 1, 1])
        
        fpr, tpr, _ = roc_curve_np(scores, y)
        
        # Should return minimal valid curve
        assert len(fpr) >= 2
        assert fpr[0] == 0.0
        assert fpr[-1] == 1.0


class TestTPRAtFPR:
    """Test tpr_at_fpr function at extreme thresholds."""
    
    def test_tpr_at_fpr_extreme_threshold(self):
        """Test tpr_at_fpr at very low FPR (1e-4)."""
        # 1000 negatives, 100 positives
        np.random.seed(42)
        n_neg, n_pos = 1000, 100
        neg_scores = np.random.uniform(0, 0.5, n_neg)
        pos_scores = np.random.uniform(0.3, 1.0, n_pos)
        
        scores = np.concatenate([neg_scores, pos_scores])
        y = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
        
        result = tpr_at_fpr(scores, y, [1e-4, 1e-3, 1e-2])
        
        # At FPR=1e-2 (10 FPs allowed out of 1000), we should have reasonable TPR
        assert result["tpr@fpr0.01"] >= 0.0
        assert result["tpr@fpr0.01"] <= 1.0
    
    def test_tpr_at_fpr_with_tied_scores(self):
        """Test that tpr_at_fpr works with tied scores."""
        scores = np.array([0.9, 0.8, 0.8, 0.8, 0.1, 0.1, 0.1])
        y = np.array([1, 1, 0, 1, 0, 0, 0])  # 3 pos, 4 neg
        
        result = tpr_at_fpr(scores, y, [1e-2, 0.25, 0.5])
        
        for key, val in result.items():
            assert 0.0 <= val <= 1.0 or np.isnan(val), f"{key} = {val} is invalid"


class TestFPRAtTPR:
    """Test fpr_at_tpr function."""
    
    def test_fpr_at_tpr_basic(self):
        """Test fpr_at_tpr returns valid values."""
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        y = np.array([1, 1, 1, 0, 0, 1, 0, 0])  # 4 pos, 4 neg
        
        result = fpr_at_tpr(scores, y, [0.5, 0.75, 0.90])
        
        for key, val in result.items():
            assert 0.0 <= val <= 1.0 or np.isnan(val), f"{key} = {val} is invalid"


class TestROCVsSklearn:
    """Compare our implementation to sklearn."""
    
    def test_auroc_matches_sklearn(self):
        """Our AUROC should match sklearn's."""
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            pytest.skip("sklearn not available")
        
        np.random.seed(42)
        scores = np.random.rand(500)
        y = np.random.randint(0, 2, 500)
        
        fpr, tpr, _ = roc_curve_np(scores, y)
        our_auroc = np.trapz(tpr, fpr)
        sklearn_auroc = roc_auc_score(y, scores)
        
        assert abs(our_auroc - sklearn_auroc) < 0.01, \
            f"AUROC mismatch: ours={our_auroc:.4f}, sklearn={sklearn_auroc:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


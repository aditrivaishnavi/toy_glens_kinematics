from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))

def pr_at_thresh(y_true: np.ndarray, y_score: np.ndarray, thresh: float = 0.5) -> Dict[str, float]:
    y_pred = (y_score >= thresh).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}

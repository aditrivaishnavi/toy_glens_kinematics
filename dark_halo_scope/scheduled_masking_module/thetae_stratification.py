
"""thetae_stratification.py

Helper to compute per-bin AUROC by theta_E.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score

@dataclass
class ThetaEBinResult:
    lo: float
    hi: float
    n: int
    auc: float

def auroc_by_thetae(
    y_true: np.ndarray,
    y_score: np.ndarray,
    theta_e: np.ndarray,
    bins: List[Tuple[float, float]],
    min_count: int = 200,
) -> List[ThetaEBinResult]:
    out: List[ThetaEBinResult] = []
    for lo, hi in bins:
        m = (theta_e >= lo) & (theta_e < hi)
        n = int(np.sum(m))
        if n < min_count:
            out.append(ThetaEBinResult(lo, hi, n, float("nan")))
            continue
        yt = y_true[m]
        ys = y_score[m]
        if len(np.unique(yt)) < 2:
            out.append(ThetaEBinResult(lo, hi, n, float("nan")))
            continue
        out.append(ThetaEBinResult(lo, hi, n, float(roc_auc_score(yt, ys))))
    return out

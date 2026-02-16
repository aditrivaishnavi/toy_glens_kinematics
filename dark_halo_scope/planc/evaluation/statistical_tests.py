from __future__ import annotations
import numpy as np

def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, m+1) / m)
    passed = p[order] <= thresh
    if not np.any(passed):
        return np.zeros(m, dtype=bool)
    k = np.max(np.where(passed)[0])
    cutoff = p[order][k]
    return p <= cutoff

"""
Injection recovery analysis.

Compute completeness and purity as functions of lens parameters
from simulated injection tests.
"""

import numpy as np
from typing import Tuple, Dict, Optional


def compute_completeness(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute completeness (true positive rate).
    
    Completeness = (detected lenses) / (all lenses)
    """
    mask = labels > 0.5
    if mask.sum() == 0:
        return np.nan
    
    detected = predictions[mask] > threshold
    return detected.mean()


def compute_purity(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute purity (precision).
    
    Purity = (true lenses) / (all detections)
    """
    detected = predictions > threshold
    if detected.sum() == 0:
        return np.nan
    
    true_positives = (labels[detected] > 0.5)
    return true_positives.mean()


def bin_completeness(
    predictions: np.ndarray,
    labels: np.ndarray,
    param1: np.ndarray,
    param2: np.ndarray,
    bins1: np.ndarray,
    bins2: np.ndarray,
    threshold: float = 0.5,
    min_samples: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute completeness in 2D bins.
    
    Parameters
    ----------
    predictions : np.ndarray
        P_lens scores
    labels : np.ndarray
        True labels (0 or 1)
    param1, param2 : np.ndarray
        Parameters to bin by (e.g., theta_E, m_source)
    bins1, bins2 : np.ndarray
        Bin edges for each parameter
    threshold : float
        Detection threshold
    min_samples : int
        Minimum samples per bin for valid estimate
    
    Returns
    -------
    C : np.ndarray
        Shape (len(bins1)-1, len(bins2)-1) completeness map
    counts : np.ndarray
        Number of true lenses per bin
    bin_centers : tuple
        (centers1, centers2) bin center arrays
    """
    n_bins1 = len(bins1) - 1
    n_bins2 = len(bins2) - 1
    
    C = np.full((n_bins1, n_bins2), np.nan)
    counts = np.zeros((n_bins1, n_bins2), dtype=int)
    
    # Only consider positive samples (true lenses)
    lens_mask = labels > 0.5
    
    for i in range(n_bins1):
        for j in range(n_bins2):
            # Find samples in this bin
            in_bin = (
                (param1 >= bins1[i]) & (param1 < bins1[i+1]) &
                (param2 >= bins2[j]) & (param2 < bins2[j+1]) &
                lens_mask
            )
            
            n_in_bin = in_bin.sum()
            counts[i, j] = n_in_bin
            
            if n_in_bin >= min_samples:
                C[i, j] = (predictions[in_bin] > threshold).mean()
    
    # Bin centers
    centers1 = 0.5 * (bins1[:-1] + bins1[1:])
    centers2 = 0.5 * (bins2[:-1] + bins2[1:])
    
    return C, counts, (centers1, centers2)


def completeness_vs_theta_E(
    predictions: np.ndarray,
    labels: np.ndarray,
    theta_E: np.ndarray,
    bins: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute completeness as function of θ_E.
    
    Returns
    -------
    bin_centers : np.ndarray
    completeness : np.ndarray
    counts : np.ndarray
    """
    if bins is None:
        bins = np.linspace(0.5, 3.0, 11)
    
    n_bins = len(bins) - 1
    completeness = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    
    lens_mask = labels > 0.5
    
    for i in range(n_bins):
        in_bin = (theta_E >= bins[i]) & (theta_E < bins[i+1]) & lens_mask
        counts[i] = in_bin.sum()
        
        if counts[i] >= 5:
            completeness[i] = (predictions[in_bin] > threshold).mean()
    
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, completeness, counts


def completeness_vs_magnitude(
    predictions: np.ndarray,
    labels: np.ndarray,
    m_source: np.ndarray,
    bins: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute completeness as function of source magnitude.
    """
    if bins is None:
        bins = np.linspace(21, 26, 11)
    
    n_bins = len(bins) - 1
    completeness = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    
    lens_mask = labels > 0.5
    
    for i in range(n_bins):
        in_bin = (m_source >= bins[i]) & (m_source < bins[i+1]) & lens_mask
        counts[i] = in_bin.sum()
        
        if counts[i] >= 5:
            completeness[i] = (predictions[in_bin] > threshold).mean()
    
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, completeness, counts


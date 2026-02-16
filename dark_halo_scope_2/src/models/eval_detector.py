"""
Evaluation utilities for lens detector.

Provides functions for computing metrics, generating ROC curves,
and analyzing detector performance.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import h5py
from tqdm import tqdm

from .detector import LensDetector


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    theta_E_pred: Optional[np.ndarray] = None,
    theta_E_true: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Returns
    -------
    dict
        Dictionary containing accuracy, AUC, precision, recall, etc.
    """
    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall
    precision, recall, pr_thresholds = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    
    # At threshold = 0.5
    preds = (probs > 0.5).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    
    accuracy = (tp + tn) / len(labels)
    precision_05 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_05 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_05 = 2 * precision_05 * recall_05 / (precision_05 + recall_05) if (precision_05 + recall_05) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'precision_05': precision_05,
        'recall_05': recall_05,
        'f1_05': f1_05,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': thresholds,
        'precision_curve': precision,
        'recall_curve': recall
    }
    
    # θ_E regression metrics (for positive samples)
    if theta_E_pred is not None and theta_E_true is not None:
        pos_mask = labels > 0.5
        if pos_mask.sum() > 0:
            theta_E_mae = np.abs(theta_E_pred[pos_mask] - theta_E_true[pos_mask]).mean()
            theta_E_rmse = np.sqrt(((theta_E_pred[pos_mask] - theta_E_true[pos_mask])**2).mean())
            theta_E_bias = (theta_E_pred[pos_mask] - theta_E_true[pos_mask]).mean()
            metrics['theta_E_mae'] = theta_E_mae
            metrics['theta_E_rmse'] = theta_E_rmse
            metrics['theta_E_bias'] = theta_E_bias
    
    return metrics


def recall_at_fpr(
    probs: np.ndarray,
    labels: np.ndarray,
    target_fpr: float = 0.05
) -> Tuple[float, float]:
    """
    Compute recall at a fixed false positive rate.
    
    Parameters
    ----------
    probs, labels : np.ndarray
        Predictions and ground truth
    target_fpr : float
        Target FPR (e.g., 0.05 for 5%)
    
    Returns
    -------
    recall : float
        Recall at the target FPR
    threshold : float
        Threshold achieving target FPR
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    
    # Find threshold closest to target FPR
    idx = np.argmin(np.abs(fpr - target_fpr))
    
    return tpr[idx], thresholds[idx]


@torch.no_grad()
def evaluate_on_hdf5(
    model: LensDetector,
    h5_path: str,
    device: torch.device,
    batch_size: int = 64
) -> Dict:
    """
    Evaluate model on an HDF5 dataset.
    
    Returns
    -------
    dict
        Full evaluation results including predictions
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    all_theta_E_pred = []
    all_theta_E_true = []
    
    with h5py.File(h5_path, 'r') as f:
        n_samples = f['labels'].shape[0]
        image_key = 'images_preproc' if 'images_preproc' in f else 'images_raw'
        
        for start in tqdm(range(0, n_samples, batch_size), desc="Evaluating"):
            end = min(start + batch_size, n_samples)
            
            images = torch.from_numpy(f[image_key][start:end]).to(device)
            labels = f['labels'][start:end]
            theta_E_true = f['theta_E'][start:end]
            
            p_lens, theta_E_pred = model(images)
            
            all_probs.append(p_lens.cpu().numpy())
            all_labels.append(labels)
            all_theta_E_pred.append(theta_E_pred.cpu().numpy())
            all_theta_E_true.append(theta_E_true)
    
    probs = np.concatenate(all_probs).flatten()
    labels = np.concatenate(all_labels).flatten()
    theta_E_pred = np.concatenate(all_theta_E_pred).flatten()
    theta_E_true = np.concatenate(all_theta_E_true).flatten()
    
    metrics = compute_metrics(probs, labels, theta_E_pred, theta_E_true)
    metrics['probs'] = probs
    metrics['labels'] = labels
    metrics['theta_E_pred'] = theta_E_pred
    metrics['theta_E_true'] = theta_E_true
    
    return metrics


def get_top_candidates(
    model: LensDetector,
    h5_path: str,
    device: torch.device,
    n_top: int = 1000,
    batch_size: int = 64
) -> Dict:
    """
    Get top lens candidates sorted by probability.
    
    Useful for active learning: select high P_lens candidates
    from real backgrounds for manual inspection.
    """
    model.eval()
    
    all_probs = []
    all_indices = []
    
    with h5py.File(h5_path, 'r') as f:
        n_samples = f['images_raw'].shape[0]
        image_key = 'images_preproc' if 'images_preproc' in f else 'images_raw'
        
        for start in tqdm(range(0, n_samples, batch_size), desc="Scanning"):
            end = min(start + batch_size, n_samples)
            
            images = torch.from_numpy(f[image_key][start:end]).to(device)
            p_lens, _ = model(images)
            
            all_probs.append(p_lens.cpu().numpy().flatten())
            all_indices.extend(range(start, end))
    
    probs = np.concatenate(all_probs)
    indices = np.array(all_indices)
    
    # Sort by probability
    sorted_idx = np.argsort(probs)[::-1]
    top_idx = sorted_idx[:n_top]
    
    return {
        'indices': indices[top_idx],
        'probs': probs[top_idx]
    }


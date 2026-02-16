"""
Training pipeline for lens detector.

This module provides functions to train the LensDetector model
on simulated lens datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Dict, Tuple
from tqdm import tqdm

from .detector import LensDetector


class LensDataset(Dataset):
    """PyTorch dataset for lens detection training."""
    
    def __init__(
        self,
        h5_path: str,
        preprocess: bool = True,
        use_metadata: bool = False
    ):
        self.h5_path = h5_path
        self.preprocess = preprocess
        self.use_metadata = use_metadata
        
        with h5py.File(h5_path, 'r') as f:
            self.n_samples = f['labels'].shape[0]
            # Check if preprocessed
            self.image_key = 'images_preproc' if 'images_preproc' in f else 'images_raw'
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            image = f[self.image_key][idx]
            label = f['labels'][idx]
            theta_E = f['theta_E'][idx]
            
            if self.use_metadata and 'meta' in f:
                meta = f['meta'][idx]
            else:
                meta = np.zeros(1, dtype=np.float32)
        
        return {
            'image': torch.from_numpy(image),
            'label': torch.tensor(label, dtype=torch.float32),
            'theta_E': torch.tensor(theta_E, dtype=torch.float32),
            'meta': torch.from_numpy(meta)
        }


def train_epoch(
    model: LensDetector,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    class_weight: float = 1.0,
    theta_E_weight: float = 0.1
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_class_loss = 0
    total_reg_loss = 0
    correct = 0
    total = 0
    
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch['image'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)
        theta_E_true = batch['theta_E'].to(device).unsqueeze(1)
        meta = batch['meta'].to(device) if model.n_meta_features > 0 else None
        
        optimizer.zero_grad()
        
        p_lens, theta_E_pred = model(images, meta)
        
        # Classification loss
        class_loss = bce_loss(p_lens, labels)
        
        # Regression loss (only for positive samples)
        mask = labels > 0.5
        if mask.sum() > 0:
            reg_loss = mse_loss(theta_E_pred[mask], theta_E_true[mask])
        else:
            reg_loss = torch.tensor(0.0, device=device)
        
        # Combined loss
        loss = class_weight * class_loss + theta_E_weight * reg_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_class_loss += class_loss.item()
        total_reg_loss += reg_loss.item()
        
        # Accuracy
        pred = (p_lens > 0.5).float()
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'class_loss': total_class_loss / n_batches,
        'reg_loss': total_reg_loss / n_batches,
        'accuracy': correct / total
    }


@torch.no_grad()
def evaluate(
    model: LensDetector,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    
    all_probs = []
    all_labels = []
    all_theta_E_pred = []
    all_theta_E_true = []
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch['image'].to(device)
        labels = batch['label'].numpy()
        theta_E_true = batch['theta_E'].numpy()
        meta = batch['meta'].to(device) if model.n_meta_features > 0 else None
        
        p_lens, theta_E_pred = model(images, meta)
        
        all_probs.append(p_lens.cpu().numpy())
        all_labels.append(labels)
        all_theta_E_pred.append(theta_E_pred.cpu().numpy())
        all_theta_E_true.append(theta_E_true)
    
    probs = np.concatenate(all_probs).flatten()
    labels = np.concatenate(all_labels).flatten()
    theta_E_pred = np.concatenate(all_theta_E_pred).flatten()
    theta_E_true = np.concatenate(all_theta_E_true).flatten()
    
    # Metrics
    preds = (probs > 0.5).astype(int)
    accuracy = (preds == labels).mean()
    
    # For positive samples only
    pos_mask = labels > 0.5
    if pos_mask.sum() > 0:
        theta_E_mae = np.abs(theta_E_pred[pos_mask] - theta_E_true[pos_mask]).mean()
    else:
        theta_E_mae = 0.0
    
    return {
        'accuracy': accuracy,
        'theta_E_mae': theta_E_mae,
        'probs': probs,
        'labels': labels
    }


def train_detector(
    train_h5: str,
    val_h5: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[str] = None
) -> LensDetector:
    """
    Train lens detector on simulated data.
    
    Parameters
    ----------
    train_h5, val_h5 : str
        Paths to training and validation HDF5 files
    output_dir : str
        Directory to save checkpoints
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    lr : float
        Learning rate
    device : str, optional
        Device to train on (auto-detected if None)
    
    Returns
    -------
    LensDetector
        Trained model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Datasets
    train_dataset = LensDataset(train_h5)
    val_dataset = LensDataset(val_h5)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = LensDetector(n_meta_features=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f}")
        print(f"  Val:   acc={val_metrics['accuracy']:.4f}, θ_E MAE={val_metrics['theta_E_mae']:.3f}")
        
        scheduler.step(1 - val_metrics['accuracy'])
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
            print(f"  → New best model (acc={best_val_acc:.4f})")
    
    # Load best model
    model.load_state_dict(torch.load(f"{output_dir}/best_model.pt"))
    
    return model


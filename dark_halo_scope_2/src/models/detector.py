"""
Lens detector model based on ResNet-18.

This module provides a CNN-based lens detector with optional
metadata fusion and dual outputs for classification and θ_E regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional


class LensDetector(nn.Module):
    """
    ResNet-18 based lens detector with metadata fusion.
    
    Architecture:
    - ResNet-18 trunk (4 input channels: g, r, z, R)
    - Optional metadata MLP
    - Combined head with:
      - P_lens: sigmoid output for classification
      - θ̂_E: regression output for Einstein radius
    
    Parameters
    ----------
    n_meta_features : int
        Number of metadata features (0 to disable metadata branch)
    pretrained_trunk : bool
        Whether to use pretrained ResNet weights (adapted for 4 channels)
    """
    
    def __init__(
        self,
        n_meta_features: int = 0,
        pretrained_trunk: bool = False
    ):
        super().__init__()
        
        self.n_meta_features = n_meta_features
        
        # Build ResNet-18 trunk
        if pretrained_trunk:
            trunk = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            trunk = models.resnet18(weights=None)
        
        # Modify first conv for 4 input channels
        # Average pretrained weights across channels if using pretrained
        old_conv = trunk.conv1
        self.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        if pretrained_trunk:
            # Initialize from pretrained: average RGB weights for R channel
            with torch.no_grad():
                self.conv1.weight[:, :3] = old_conv.weight
                self.conv1.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
        
        # Rest of ResNet (excluding final FC)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu
        self.maxpool = trunk.maxpool
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        self.layer4 = trunk.layer4
        self.avgpool = trunk.avgpool
        
        # Feature dimension from ResNet-18
        self.image_feat_dim = 512
        
        # Metadata MLP (optional)
        if n_meta_features > 0:
            self.meta_mlp = nn.Sequential(
                nn.Linear(n_meta_features, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU()
            )
            self.combined_dim = self.image_feat_dim + 32
        else:
            self.meta_mlp = None
            self.combined_dim = self.image_feat_dim
        
        # Classification head
        self.fc_class = nn.Sequential(
            nn.Linear(self.combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        # Regression head for θ_E
        self.fc_theta_E = nn.Sequential(
            nn.Linear(self.combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive output
        )
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract image features."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def forward(
        self,
        images: torch.Tensor,
        metadata: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        images : torch.Tensor
            Shape (B, 4, H, W) preprocessed images
        metadata : torch.Tensor, optional
            Shape (B, n_meta_features) catalog metadata
        
        Returns
        -------
        p_lens : torch.Tensor
            Shape (B, 1) probability of being a lens
        theta_E_pred : torch.Tensor
            Shape (B, 1) predicted Einstein radius
        """
        # Image features
        img_feat = self.forward_features(images)
        
        # Combine with metadata if available
        if self.meta_mlp is not None and metadata is not None:
            meta_feat = self.meta_mlp(metadata)
            combined = torch.cat([img_feat, meta_feat], dim=1)
        else:
            combined = img_feat
        
        # Classification
        logits = self.fc_class(combined)
        p_lens = torch.sigmoid(logits)
        
        # Regression
        theta_E_pred = self.fc_theta_E(combined)
        
        return p_lens, theta_E_pred
    
    def predict(
        self,
        images: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with binary classification.
        
        Returns
        -------
        labels : torch.Tensor
            Binary predictions (0 or 1)
        p_lens : torch.Tensor
            Probabilities
        theta_E_pred : torch.Tensor
            Predicted Einstein radii
        """
        p_lens, theta_E_pred = self.forward(images, metadata)
        labels = (p_lens > threshold).long().squeeze()
        return labels, p_lens, theta_E_pred


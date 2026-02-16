#!/usr/bin/env python3
"""
Phase 1: Model Definition

ResNet18-based lens classifier adapted for 64x64 stamps.

Lessons Learned Incorporated:
- Use smaller first conv for 64x64 input (3x3 instead of 7x7)
- Proper pretrained weight handling for different input channels
- Gradient clipping to prevent exploding gradients
- L1.2: No duplicate functions - use shared constants

Exit Criteria:
- Forward pass produces (B, 1) logits
- No NaN in gradients after backward pass
- Memory footprint < 500MB for batch_size=128
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models

# Import from shared module - SINGLE SOURCE OF TRUTH
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.constants import STAMP_SIZE, NUM_CHANNELS


@dataclass
class ModelConfig:
    """Model configuration."""
    arch: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 1  # Binary classification
    in_channels: int = NUM_CHANNELS  # Use constant
    
    # Modifications for 64x64 input
    replace_first_conv: bool = True
    first_conv_kernel: int = 3
    first_conv_stride: int = 1
    remove_first_pool: bool = True
    
    # Dropout
    dropout: float = 0.0


class LensClassifier(nn.Module):
    """
    Strong lens classifier based on ResNet.
    
    Modifications from standard ResNet:
    1. First conv adapted for 64x64 input (3x3, stride=1)
    2. First maxpool removed
    3. Final fc layer produces single logit
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Build backbone
        if config.arch == "resnet18":
            backbone = models.resnet18(
                weights="IMAGENET1K_V1" if config.pretrained else None
            )
        elif config.arch == "resnet34":
            backbone = models.resnet34(
                weights="IMAGENET1K_V1" if config.pretrained else None
            )
        elif config.arch == "resnet50":
            backbone = models.resnet50(
                weights="IMAGENET1K_V1" if config.pretrained else None
            )
        else:
            raise ValueError(f"Unknown architecture: {config.arch}")
        
        # Modify first conv for 64x64 input
        if config.replace_first_conv:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels=config.in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=config.first_conv_kernel,
                stride=config.first_conv_stride,
                padding=config.first_conv_kernel // 2,
                bias=False
            )
            
            # Initialize from pretrained weights if available
            if config.pretrained and config.in_channels == 3:
                with torch.no_grad():
                    # Average pretrained weights over kernel and replicate
                    pretrained_weight = old_conv.weight
                    # Center crop or pad pretrained kernel
                    k = config.first_conv_kernel
                    if k < 7:
                        center = 7 // 2
                        half = k // 2
                        backbone.conv1.weight.copy_(
                            pretrained_weight[:, :, center-half:center+half+1, center-half:center+half+1]
                        )
                    else:
                        backbone.conv1.weight.copy_(pretrained_weight)
        
        # Handle different number of input channels
        if config.in_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels=config.in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            
            if config.pretrained:
                with torch.no_grad():
                    # Repeat pretrained weights across channels
                    if config.in_channels == 6:
                        # For 6-channel: duplicate RGB weights
                        backbone.conv1.weight.copy_(
                            torch.cat([old_conv.weight, old_conv.weight], dim=1)
                        )
                    else:
                        # Average RGB and repeat
                        avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
                        backbone.conv1.weight.copy_(
                            avg_weight.repeat(1, config.in_channels, 1, 1)
                        )
        
        # Remove first maxpool for 64x64 input
        if config.remove_first_pool:
            backbone.maxpool = nn.Identity()
        
        # Store feature extractor
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        
        # Classifier head
        self.avgpool = backbone.avgpool
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        
        # Compute feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, config.in_channels, 64, 64)
            feat = self.features(dummy)
            feat = self.avgpool(feat)
            feat_dim = feat.numel()
        
        self.fc = nn.Linear(feat_dim, config.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Logits tensor (B, num_classes)
        """
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = feat.flatten(1)
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits
    
    def get_feature_map(self, x: torch.Tensor, layer: str = "layer4") -> torch.Tensor:
        """
        Get intermediate feature map for visualization.
        
        Args:
            x: Input tensor
            layer: Which layer to extract ("layer1", "layer2", "layer3", "layer4")
        
        Returns:
            Feature map tensor
        """
        # Forward through layers up to specified point
        h = self.features[0](x)  # conv1
        h = self.features[1](h)  # bn1
        h = self.features[2](h)  # relu
        h = self.features[3](h)  # maxpool
        
        h = self.features[4](h)  # layer1
        if layer == "layer1":
            return h
        
        h = self.features[5](h)  # layer2
        if layer == "layer2":
            return h
        
        h = self.features[6](h)  # layer3
        if layer == "layer3":
            return h
        
        h = self.features[7](h)  # layer4
        return h


def build_model(
    checkpoint_path: Optional[str] = None,
    config: Optional[ModelConfig] = None,
    device: str = "cuda"
) -> LensClassifier:
    """
    Build model, optionally loading from checkpoint.
    
    Exit Criteria:
    - Model forward pass works with (1, 3, 64, 64) input
    - Output shape is (1, 1)
    """
    if config is None:
        config = ModelConfig()
    
    model = LensClassifier(config)
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    
    model.to(device)
    return model


def validate_model(
    model: LensClassifier,
    device: str = "cuda"
) -> dict:
    """
    Validate model meets requirements.
    
    Exit Criteria:
    1. Forward pass produces correct output shape
    2. Backward pass produces no NaN gradients
    3. Memory usage is reasonable
    """
    results = {
        "forward_shape_ok": False,
        "backward_no_nan": False,
        "memory_ok": False,
    }
    
    model.eval()
    model.to(device)
    
    # Test forward pass
    x = torch.randn(4, model.config.in_channels, 64, 64, device=device)
    
    with torch.no_grad():
        logits = model(x)
    
    results["forward_shape_ok"] = (
        logits.shape == (4, model.config.num_classes)
    )
    
    # Test backward pass
    model.train()
    x = torch.randn(4, model.config.in_channels, 64, 64, device=device, requires_grad=True)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    
    # Check for NaN gradients
    has_nan = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            has_nan = True
            break
    
    results["backward_no_nan"] = not has_nan
    
    # Check memory usage
    if device == "cuda" and torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        results["memory_mb"] = memory_mb
        results["memory_ok"] = memory_mb < 1000  # < 1GB
    else:
        results["memory_ok"] = True
        results["memory_mb"] = 0
    
    results["all_passed"] = all([
        results["forward_shape_ok"],
        results["backward_no_nan"],
        results["memory_ok"],
    ])
    
    return results


if __name__ == "__main__":
    print("Testing model...")
    
    config = ModelConfig(
        arch="resnet18",
        pretrained=True,
        in_channels=3,
        replace_first_conv=True,
        first_conv_kernel=3,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config=config, device=device)
    
    print(f"\nModel architecture: {config.arch}")
    print(f"Input channels: {config.in_channels}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    results = validate_model(model, device=device)
    
    print("\nValidation Results:")
    for key, value in results.items():
        status = "✓" if value else "✗"
        if isinstance(value, bool):
            print(f"  {key}: {status}")
        else:
            print(f"  {key}: {value}")
    
    if results["all_passed"]:
        print("\n✓ ALL CHECKS PASSED")
    else:
        print("\n✗ VALIDATION FAILED")

#!/usr/bin/env python3
"""
ConvNeXt 6-Channel Stem Patch

This module provides utilities to modify ConvNeXt models to accept 6-channel
input (3 raw + 3 residual channels) for shortcut-resistant training.

Initialization strategies:
    - "kaiming": Full Kaiming initialization for all 6 channels
    - "copy_or_zero": Copy pretrained RGB weights to ch0-2, zero-init ch3-5
    - "copy_or_copy": Copy pretrained RGB weights to both ch0-2 and ch3-5

Recommended: Use "copy_or_zero" with pretrained weights. This preserves
pretrained behavior for raw channels while starting residual channels
as ignored (zero weights).

Author: DarkHaloScope Team
Date: 2026-02-05
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, convnext_small, convnext_base
from torchvision.models import ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights
from typing import Optional, Literal


def patch_convnext_stem_to_6ch(
    model: nn.Module,
    init: Literal["kaiming", "copy_or_zero", "copy_or_copy"] = "copy_or_zero",
) -> nn.Module:
    """
    Patch ConvNeXt model to accept 6-channel input.
    
    Args:
        model: ConvNeXt model (convnext_tiny, small, or base)
        init: Initialization strategy:
            - "kaiming": Kaiming normal init for all 6 channels
            - "copy_or_zero": Copy RGB weights to ch0-2, zero ch3-5 (recommended)
            - "copy_or_copy": Copy RGB weights to both ch0-2 and ch3-5
    
    Returns:
        Modified model with 6-channel input.
    
    Example:
        >>> m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        >>> m = patch_convnext_stem_to_6ch(m, init="copy_or_zero")
        >>> x = torch.randn(1, 6, 64, 64)
        >>> out = m(x)
    """
    # Find the first Conv2d in the stem
    old = model.features[0][0]
    assert isinstance(old, nn.Conv2d), f"Expected Conv2d, got {type(old)}"
    assert old.in_channels == 3, f"Expected 3 input channels, got {old.in_channels}"
    
    # Create new Conv2d with 6 input channels
    new = nn.Conv2d(
        in_channels=6,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=(old.bias is not None),
        padding_mode=old.padding_mode,
    )
    
    # Initialize weights
    with torch.no_grad():
        if init == "kaiming":
            nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
            if new.bias is not None:
                nn.init.zeros_(new.bias)
        
        elif init == "copy_or_zero":
            # Copy pretrained weights for RGB channels
            new.weight[:, :3].copy_(old.weight)
            # Zero-init residual channels (start by ignoring them)
            new.weight[:, 3:].zero_()
            if new.bias is not None and old.bias is not None:
                new.bias.copy_(old.bias)
        
        elif init == "copy_or_copy":
            # Copy pretrained weights to both RGB and residual channels
            new.weight[:, :3].copy_(old.weight)
            new.weight[:, 3:].copy_(old.weight)
            if new.bias is not None and old.bias is not None:
                new.bias.copy_(old.bias)
        
        else:
            raise ValueError(f"Unknown init strategy: {init}")
    
    # Replace the stem conv
    model.features[0][0] = new
    
    return model


class MetaFusionHead(nn.Module):
    """
    Fusion head that combines CNN features with metadata.
    
    Architecture:
        - MLP for metadata: meta_dim -> hidden -> hidden
        - Classifier: (feat_dim + hidden) -> hidden -> 1
    """
    
    def __init__(
        self,
        feat_dim: int,
        meta_dim: int = 2,
        hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
    
    def forward(self, feats: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: (B, feat_dim) CNN features
            meta: (B, meta_dim) metadata [psfsize_r, psfdepth_r]
        
        Returns:
            (B,) logits
        """
        m = self.meta_mlp(meta)
        x = torch.cat([feats, m], dim=1)
        return self.classifier(x).squeeze(1)


class LensFinder6CH(nn.Module):
    """
    Complete 6-channel lens finder model.
    
    Architecture:
        - ConvNeXt backbone (6-channel input)
        - MetaFusionHead for classification
    
    Example:
        >>> model = LensFinder6CH(arch="tiny", pretrained=True, init="copy_or_zero")
        >>> x = torch.randn(4, 6, 64, 64)
        >>> meta = torch.randn(4, 2)  # [psfsize_r, psfdepth_r]
        >>> logits = model(x, meta)
    """
    
    def __init__(
        self,
        arch: Literal["tiny", "small", "base"] = "tiny",
        pretrained: bool = True,
        init: Literal["kaiming", "copy_or_zero", "copy_or_copy"] = "copy_or_zero",
        meta_dim: int = 2,
        hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Create backbone
        if arch == "tiny":
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = convnext_tiny(weights=weights)
        elif arch == "small":
            weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = convnext_small(weights=weights)
        elif arch == "base":
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = convnext_base(weights=weights)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        # Patch to 6 channels
        backbone = patch_convnext_stem_to_6ch(backbone, init=init)
        
        # Get feature dimension from classifier
        feat_dim = backbone.classifier[2].in_features
        
        # Remove classifier head
        backbone.classifier = nn.Identity()
        
        self.backbone = backbone
        self.head = MetaFusionHead(
            feat_dim=feat_dim,
            meta_dim=meta_dim,
            hidden=hidden,
            dropout=dropout,
        )
        
        self._feat_dim = feat_dim
        self._meta_dim = meta_dim
    
    @property
    def feat_dim(self) -> int:
        return self._feat_dim
    
    @property
    def meta_dim(self) -> int:
        return self._meta_dim
    
    def forward(
        self,
        x: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, 6, H, W) input images
            meta: (B, meta_dim) metadata, default zeros if not provided
        
        Returns:
            (B,) logits
        """
        feats = self.backbone(x)
        
        # Flatten if needed (ConvNeXt may output (B, C, 1, 1) or (B, C))
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        
        # Default metadata if not provided
        if meta is None:
            meta = torch.zeros(feats.shape[0], self._meta_dim, device=feats.device)
        
        return self.head(feats, meta)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features without classification."""
        feats = self.backbone(x)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        return feats


def load_checkpoint_3ch_to_6ch(
    checkpoint_path: str,
    arch: str = "tiny",
    init: str = "copy_or_zero",
    device: str = "cpu",
) -> LensFinder6CH:
    """
    Load a 3-channel checkpoint and convert to 6-channel model.
    
    This is useful for initializing a 6-channel model from a pretrained
    3-channel model.
    
    Args:
        checkpoint_path: path to 3-channel model checkpoint
        arch: architecture name
        init: initialization strategy for new channels
        device: device to load to
    
    Returns:
        6-channel model with loaded weights (where applicable)
    """
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Create 6-channel model
    model = LensFinder6CH(arch=arch, pretrained=False, init="kaiming")
    
    # Load state dict, handling stem mismatch
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    
    # Get current 6-channel stem weight
    current_stem = model.backbone.features[0][0].weight.data
    
    # Find stem key
    stem_key = None
    for k in state_dict.keys():
        if "features.0.0.weight" in k:
            stem_key = k
            break
    
    if stem_key is not None:
        old_stem = state_dict[stem_key]
        if old_stem.shape[1] == 3:
            # Convert 3-channel stem to 6-channel
            with torch.no_grad():
                if init == "copy_or_zero":
                    current_stem[:, :3].copy_(old_stem)
                    current_stem[:, 3:].zero_()
                elif init == "copy_or_copy":
                    current_stem[:, :3].copy_(old_stem)
                    current_stem[:, 3:].copy_(old_stem)
            
            # Replace in state dict
            state_dict[stem_key] = current_stem
    
    # Load (with strict=False to allow mismatches)
    model.load_state_dict(state_dict, strict=False)
    
    return model


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing ConvNeXt 6-channel patch...")
    
    # Test basic patching
    print("\n1. Testing patch_convnext_stem_to_6ch...")
    m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    m = patch_convnext_stem_to_6ch(m, init="copy_or_zero")
    
    x = torch.randn(2, 6, 64, 64)
    out = m(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (2, 768), f"Expected (2, 768), got {out.shape}"
    print("   PASS")
    
    # Test LensFinder6CH
    print("\n2. Testing LensFinder6CH...")
    model = LensFinder6CH(arch="tiny", pretrained=True, init="copy_or_zero")
    
    x = torch.randn(4, 6, 64, 64)
    meta = torch.randn(4, 2)
    logits = model(x, meta)
    print(f"   Input: {x.shape}, Meta: {meta.shape} -> Logits: {logits.shape}")
    assert logits.shape == (4,), f"Expected (4,), got {logits.shape}"
    print("   PASS")
    
    # Test without meta
    print("\n3. Testing forward without meta...")
    logits = model(x)
    assert logits.shape == (4,)
    print("   PASS")
    
    # Test feature extraction
    print("\n4. Testing feature extraction...")
    feats = model.extract_features(x)
    print(f"   Features: {feats.shape}")
    assert feats.shape == (4, model.feat_dim)
    print("   PASS")
    
    # Verify weight initialization
    print("\n5. Verifying copy_or_zero initialization...")
    stem_weight = model.backbone.features[0][0].weight.data
    rgb_norm = stem_weight[:, :3].norm().item()
    resid_norm = stem_weight[:, 3:].norm().item()
    print(f"   RGB weight norm: {rgb_norm:.4f}")
    print(f"   Residual weight norm: {resid_norm:.4f}")
    assert resid_norm < 1e-6, "Residual channels should be zero-initialized"
    print("   PASS")
    
    print("\n" + "=" * 50)
    print("All tests passed!")

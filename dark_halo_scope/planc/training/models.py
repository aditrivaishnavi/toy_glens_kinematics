from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as tvm

@dataclass
class ModelConfig:
    name: str = "resnet18"
    pretrained: bool = True
    in_channels: int = 3
    num_classes: int = 1
    metadata_dim: int = 0

class MetadataMLP(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class LensNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone, feat_dim = self._make_backbone(cfg.name, cfg.pretrained, cfg.in_channels)
        self.use_meta = cfg.metadata_dim > 0
        if self.use_meta:
            self.meta = MetadataMLP(cfg.metadata_dim)
            self.head = nn.Linear(feat_dim + 64, cfg.num_classes)
        else:
            self.head = nn.Linear(feat_dim, cfg.num_classes)

    def _make_backbone(self, name: str, pretrained: bool, in_channels: int):
        if name.startswith("resnet"):
            ctor = getattr(tvm, name)
            m = ctor(weights="IMAGENET1K_V1" if pretrained else None)
            if in_channels != 3:
                m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()
            return m, feat_dim
        if name.startswith("efficientnet"):
            ctor = getattr(tvm, name)
            m = ctor(weights="IMAGENET1K_V1" if pretrained else None)
            if in_channels != 3:
                first = m.features[0][0]
                m.features[0][0] = nn.Conv2d(in_channels, first.out_channels, kernel_size=first.kernel_size,
                                             stride=first.stride, padding=first.padding, bias=False)
            feat_dim = m.classifier[1].in_features
            m.classifier = nn.Identity()
            return m, feat_dim
        raise ValueError(f"Unknown model: {name}")

    def forward(self, x: torch.Tensor, meta: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = self.backbone(x)
        if f.ndim > 2:
            f = torch.flatten(f, 1)
        if self.use_meta:
            if meta is None:
                raise ValueError("metadata_dim>0 but meta is None")
            m = self.meta(meta)
            f = torch.cat([f, m], dim=1)
        return self.head(f).squeeze(1)

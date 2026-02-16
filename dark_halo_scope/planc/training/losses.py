from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class LossConfig:
    name: str = "bce"
    pos_weight: float = 1.0
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight.to(logits.device))
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        w = (1 - pt).pow(self.gamma)
        return (w * bce).mean()

def make_loss(cfg: LossConfig) -> nn.Module:
    if cfg.name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.pos_weight], dtype=torch.float32))
    if cfg.name == "focal":
        return FocalLoss(gamma=cfg.focal_gamma, pos_weight=cfg.pos_weight)
    raise ValueError(cfg.name)

def apply_label_smoothing(y: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return y
    return y * (1 - smoothing) + 0.5 * smoothing

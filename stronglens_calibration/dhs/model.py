from __future__ import annotations
import torch.nn as nn
import torchvision.models as models
import torch

def build_resnet18(in_ch: int = 3) -> nn.Module:
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(512, 1)
    return m

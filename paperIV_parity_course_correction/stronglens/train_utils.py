from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import os, json, time
import torch
from torch import nn

@dataclass
class RunInfo:
    command: str
    config: Dict[str, Any]
    git_commit: Optional[str] = None
    timestamp_utc: Optional[str] = None

def save_run_info(outdir: str, info: RunInfo) -> None:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "run_info.json")
    with open(path, "w") as f:
        json.dump(info.__dict__, f, indent=2, sort_keys=True)

def weighted_bce_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # targets in {0,1} long
    targets_f = targets.float()
    loss = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(1), targets_f, reduction="none")
    return (loss * weights).mean()

@torch.no_grad()
def predict_logits(model: nn.Module, loader, device: torch.device) -> Tuple[list, list, list]:
    model.eval()
    ys, ps, tiers = [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].cpu().numpy().tolist()
        t = batch["tier"]
        logits = model(x).detach().cpu().numpy().reshape(-1).tolist()
        ys.extend(y)
        ps.extend(logits)
        tiers.extend(list(t))
    return ys, ps, tiers

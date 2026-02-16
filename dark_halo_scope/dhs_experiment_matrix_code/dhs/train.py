from __future__ import annotations
import os, time
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from .data import LensDataset, DatasetConfig, SplitConfig
from .transforms import AugmentConfig
from .model import build_resnet18

@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = "cuda"
    early_stopping_patience: int = 10
    out_dir: str = "./checkpoints"
    mixed_precision: bool = True

def _collate(batch):
    xs, ys = zip(*batch)
    x = torch.from_numpy(np.stack(xs, axis=0)).float()
    y = torch.from_numpy(np.array(ys)).float().view(-1,1)
    return x, y

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.sigmoid(logits).cpu().numpy().ravel()
        ys.append(y.numpy().ravel())
        ps.append(p)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return float(roc_auc_score(y, p))

def train_one(tcfg: TrainConfig, dcfg: DatasetConfig, aug: AugmentConfig):
    os.makedirs(tcfg.out_dir, exist_ok=True)
    ds_tr = LensDataset(dcfg, SplitConfig(split_value="train"), aug)
    ds_va = LensDataset(dcfg, SplitConfig(split_value="val"), AugmentConfig(hflip=False, vflip=False, rot90=False))

    dl_tr = DataLoader(ds_tr, batch_size=tcfg.batch_size, shuffle=True, num_workers=tcfg.num_workers,
                       pin_memory=True, collate_fn=_collate)
    dl_va = DataLoader(ds_va, batch_size=tcfg.batch_size, shuffle=False, num_workers=tcfg.num_workers,
                       pin_memory=True, collate_fn=_collate)

    device = torch.device(tcfg.device if torch.cuda.is_available() else "cpu")
    model = build_resnet18(3).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(tcfg.mixed_precision and device.type=="cuda"))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_auc, best_path = -1.0, None
    bad = 0

    for epoch in range(1, tcfg.epochs+1):
        model.train()
        t0 = time.time()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        auc = evaluate(model, dl_va, device)
        ckpt = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "best_auc": best_auc, "dataset": dcfg.__dict__, "train": tcfg.__dict__}
        torch.save(ckpt, os.path.join(tcfg.out_dir, "last.pt"))
        if auc > best_auc:
            best_auc = auc
            best_path = os.path.join(tcfg.out_dir, "best.pt")
            torch.save(ckpt, best_path)
            bad = 0
        else:
            bad += 1

        print(f"[epoch {epoch:03d}] val_auc={auc:.4f} best={best_auc:.4f} dt={time.time()-t0:.1f}s")
        if bad >= tcfg.early_stopping_patience:
            break
    return best_path, best_auc

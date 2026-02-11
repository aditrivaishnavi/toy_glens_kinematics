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
    """Standard collate for paired/unpaired_manifest modes (x, y)."""
    xs, ys = zip(*batch)
    x = torch.from_numpy(np.stack(xs, axis=0)).float()
    y = torch.from_numpy(np.array(ys)).float().view(-1,1)
    return x, y


def _collate_weighted(batch):
    """Weighted collate for file_manifest mode (x, y, weight)."""
    xs, ys, ws = zip(*batch)
    x = torch.from_numpy(np.stack(xs, axis=0)).float()
    y = torch.from_numpy(np.array(ys)).float().view(-1,1)
    w = torch.from_numpy(np.array(ws)).float().view(-1,1)
    return x, y, w

@torch.no_grad()
def evaluate(model, loader, device, weighted=False):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        if weighted:
            x, y, _ = batch  # Ignore weights during eval
        else:
            x, y = batch
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

    # Use weighted collate for file_manifest mode
    weighted = dcfg.mode == "file_manifest"
    collate_fn = _collate_weighted if weighted else _collate

    dl_tr = DataLoader(ds_tr, batch_size=tcfg.batch_size, shuffle=True, num_workers=tcfg.num_workers,
                       pin_memory=True, collate_fn=collate_fn)
    dl_va = DataLoader(ds_va, batch_size=tcfg.batch_size, shuffle=False, num_workers=tcfg.num_workers,
                       pin_memory=True, collate_fn=collate_fn)

    device = torch.device(tcfg.device if torch.cuda.is_available() else "cpu")
    model = build_resnet18(3).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(tcfg.mixed_precision and device.type=="cuda"))
    # Use reduction='none' for weighted loss
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none' if weighted else 'mean')

    best_auc, best_path = -1.0, None
    bad = 0

    for epoch in range(1, tcfg.epochs+1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for batch in dl_tr:
            if weighted:
                x, y, w = batch
                x, y, w = x.to(device), y.to(device), w.to(device)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                w = None

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                if weighted and w is not None:
                    # Weighted BCE loss
                    per_sample_loss = loss_fn(logits, y)
                    loss = (per_sample_loss * w).mean()
                else:
                    loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1
        sched.step()

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        auc = evaluate(model, dl_va, device, weighted=weighted)
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

        print(f"[epoch {epoch:03d}] loss={avg_loss:.4f} val_auc={auc:.4f} best={best_auc:.4f} dt={time.time()-t0:.1f}s")
        if bad >= tcfg.early_stopping_patience:
            break
    return best_path, best_auc

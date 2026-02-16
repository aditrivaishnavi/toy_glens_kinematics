from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from common.logging_utils import setup_logging
from training.dataset import DR10CutoutDataset, DatasetConfig
from training.models import LensNet, ModelConfig
from training.losses import make_loss, LossConfig, apply_label_smoothing
import logging
logger = logging.getLogger(__name__)

def eval_auc(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch)==3:
                x, meta, y = batch
                logits = model(x.to(device), meta.to(device))
            else:
                x, y = batch
                logits = model(x.to(device))
            ys.append(y.numpy()); ps.append(torch.sigmoid(logits).cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return float(roc_auc_score(y, p))

def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="resnet18", choices=["resnet18","resnet34","resnet50","efficientnet_b0","efficientnet_b1"])
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--pos_weight", type=float, default=5.0)
    ap.add_argument("--loss", type=str, default="bce", choices=["bce","focal"])
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--use_asinh", action="store_true")
    ap.add_argument("--asinh_a", type=float, default=3.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metadata_csv", type=str, default="")
    ap.add_argument("--metadata_cols", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DR10CutoutDataset(DatasetConfig(args.data_root,"train",args.use_asinh,args.asinh_a,True,args.seed,args.metadata_csv,args.metadata_cols))
    val_ds = DR10CutoutDataset(DatasetConfig(args.data_root,"val",args.use_asinh,args.asinh_a,False,args.seed+1,args.metadata_csv,args.metadata_cols))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=max(1,args.num_workers//2), pin_memory=True)

    meta_dim = len([c for c in args.metadata_cols.split(",") if c.strip()]) if args.metadata_csv else 0
    model = LensNet(ModelConfig(name=args.model, pretrained=args.pretrained, metadata_dim=meta_dim))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    loss_fn = make_loss(LossConfig(name=args.loss, pos_weight=args.pos_weight, label_smoothing=args.label_smoothing))
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    best_auc = -1.0
    hist = []
    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            if len(batch)==3:
                x, meta, y = batch
                x=x.to(device); meta=meta.to(device); y=y.to(device)
                y_sm = apply_label_smoothing(y, args.label_smoothing)
                logits = model(x, meta)
            else:
                x, y = batch
                x=x.to(device); y=y.to(device)
                y_sm = apply_label_smoothing(y, args.label_smoothing)
                logits = model(x)
            loss = loss_fn(logits, y_sm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item())*len(y); n += len(y)
        sched.step()
        val_auc = eval_auc(model, val_loader, device)
        hist.append({"epoch":epoch,"train_loss":total/max(1,n),"val_auc":val_auc,"lr":opt.param_groups[0]["lr"]})
        logger.info("epoch=%d loss=%.4f val_auc=%.4f", epoch, total/max(1,n), val_auc)
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(), "args": vars(args), "best_auc": best_auc}, out_dir/"best.pt")
    (out_dir/"history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    logger.info("Best val AUC %.4f", best_auc)

if __name__ == "__main__":
    main()

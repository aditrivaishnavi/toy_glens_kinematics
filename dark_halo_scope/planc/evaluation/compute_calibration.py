from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from training.dataset import DR10CutoutDataset, DatasetConfig
from training.models import LensNet, ModelConfig
from common.logging_utils import setup_logging

def load_model(ckpt_path: Path, device: str):
    ck = torch.load(ckpt_path, map_location=device)
    args = ck.get("args", {})
    m = LensNet(ModelConfig(name=args.get("model","resnet18"), pretrained=False))
    m.load_state_dict(ck["model"], strict=True)
    m.to(device); m.eval()
    return m

def score(m, dl, device):
    ys, ps = [], []
    with torch.no_grad():
        for batch in dl:
            if len(batch)==3:
                x, meta, y = batch
                logits = m(x.to(device), meta.to(device))
            else:
                x, y = batch
                logits = m(x.to(device))
            ys.append(y.numpy()); ps.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)

def ece(y, p, n_bins=15):
    bins = np.linspace(0,1,n_bins+1)
    idx = np.digitize(p, bins) - 1
    e = 0.0
    for b in range(n_bins):
        m = idx==b
        if not np.any(m): 
            continue
        acc = float(np.mean(y[m]))
        conf = float(np.mean(p[m]))
        e += float(m.mean()) * abs(acc-conf)
    return float(e)

def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_bins", type=int, default=15)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_model(Path(args.ckpt), device)
    ds = DR10CutoutDataset(DatasetConfig(args.data_root,"val",augment=False))
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    y, p = score(m, dl, device)

    bins = np.linspace(0,1,args.n_bins+1)
    bid = np.digitize(p, bins) - 1
    acc, conf, n = [], [], []
    for b in range(args.n_bins):
        mask = bid==b
        if not np.any(mask):
            acc.append(None); conf.append(None); n.append(0)
        else:
            acc.append(float(np.mean(y[mask])))
            conf.append(float(np.mean(p[mask])))
            n.append(int(mask.sum()))
    out = {"bins": bins.tolist(), "acc": acc, "conf": conf, "n": n, "ece": ece(y,p,args.n_bins)}
    (out_dir/"calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()

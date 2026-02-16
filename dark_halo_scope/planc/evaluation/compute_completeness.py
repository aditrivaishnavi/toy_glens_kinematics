from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from training.dataset import DR10CutoutDataset, DatasetConfig
from training.models import LensNet, ModelConfig
from common.logging_utils import setup_logging
import logging
logger = logging.getLogger(__name__)

def load_model(ckpt_path: Path, device: str):
    ck = torch.load(ckpt_path, map_location=device)
    args = ck.get("args", {})
    model = LensNet(ModelConfig(name=args.get("model","resnet18"), pretrained=False))
    model.load_state_dict(ck["model"], strict=True)
    model.to(device); model.eval()
    return model

def score(model, loader, device):
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
    return y, p

def bootstrap_recall(y, p, thr, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.where(y==1)[0]
    if len(pos)==0:
        return {"recall": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    recalls = []
    for _ in range(n_boot):
        samp = rng.choice(pos, size=len(pos), replace=True)
        recalls.append(float(np.mean(p[samp] >= thr)))
    recalls = np.array(recalls)
    return {"recall": float(np.mean(p[pos] >= thr)), "ci_low": float(np.quantile(recalls,0.025)), "ci_high": float(np.quantile(recalls,0.975))}

def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(args.ckpt), device)

    ds = DR10CutoutDataset(DatasetConfig(args.data_root,"test",augment=False,seed=args.seed))
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    y, p = score(model, dl, device)

    out = {"threshold": args.threshold, "test_auc": float(roc_auc_score(y,p)), "test_recall": bootstrap_recall(y,p,args.threshold,args.n_boot,args.seed),
           "n": int(len(y)), "n_pos": int((y==1).sum()), "n_neg": int((y==0).sum())}
    (out_dir/"summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_dir/"summary.json")

if __name__ == "__main__":
    main()

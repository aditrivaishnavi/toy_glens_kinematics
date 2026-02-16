from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
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

def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--contaminants_csv", type=str, default="")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_model(Path(args.ckpt), device)

    ds = DR10CutoutDataset(DatasetConfig(args.data_root,"test",augment=False))
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    y, p = score(m, dl, device)
    yhat = (p >= args.threshold).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0,1]).tolist()
    (out_dir/"failure_summary.json").write_text(json.dumps({"threshold":args.threshold,"confusion_matrix":cm}, indent=2), encoding="utf-8")

    if args.contaminants_csv:
        contam = pd.read_csv(args.contaminants_csv)
        test = pd.read_csv(Path(args.data_root)/"test.csv")
        test["id"] = test["id"].astype(str); contam["id"]=contam["id"].astype(str)
        merged = test.merge(contam, on="id", how="inner")
        if len(merged)>0 and "category" in merged.columns:
            pred_map = {str(test.iloc[i]["id"]): float(p[i]) for i in range(len(test))}
            merged["p"] = merged["id"].map(pred_map)
            merged["fp"] = (merged["p"] >= args.threshold).astype(int)
            merged.groupby("category")["fp"].mean().reset_index(name="fpr").to_csv(out_dir/"fpr_by_category.csv", index=False)

if __name__ == "__main__":
    main()

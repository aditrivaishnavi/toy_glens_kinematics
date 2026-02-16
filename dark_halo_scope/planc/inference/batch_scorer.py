from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from training.models import LensNet, ModelConfig
from data.preprocessing import load_fits_grz, robust_norm
from common.logging_utils import setup_logging
import logging
logger = logging.getLogger(__name__)

class FitsDataset(torch.utils.data.Dataset):
    def __init__(self, paths): self.paths = list(paths)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        x = robust_norm(load_fits_grz(p))
        return torch.from_numpy(x), p

def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    args = ck.get("args", {})
    m = LensNet(ModelConfig(name=args.get("model","resnet18"), pretrained=False))
    m.load_state_dict(ck["model"], strict=True)
    m.to(device); m.eval()
    return m

def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fits_glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)
    paths = sorted([str(p) for p in Path('.').glob(args.fits_glob)])
    if not paths:
        raise SystemExit("No FITS matched")

    dl = DataLoader(FitsDataset(paths), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    rows = []
    with torch.no_grad():
        for x, ps in dl:
            logits = model(x.to(device))
            prob = torch.sigmoid(logits).cpu().numpy()
            for pth, pr in zip(ps, prob):
                rows.append({"path": pth, "p": float(pr)})

    df = pd.DataFrame(rows).sort_values("p", ascending=False)
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    logger.info("Wrote %s", outp)

if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration_json", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cal = json.loads(Path(args.calibration_json).read_text())
    acc = np.array([a if a is not None else np.nan for a in cal["acc"]], dtype=float)
    conf = np.array([c if c is not None else np.nan for c in cal["conf"]], dtype=float)
    m = ~np.isnan(acc) & ~np.isnan(conf)
    plt.figure()
    plt.plot(conf[m], acc[m])
    plt.plot([0,1],[0,1])
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical positive fraction")
    plt.savefig(out_dir/"reliability.png", dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse, yaml
from dhs.data import DatasetConfig
from dhs.transforms import AugmentConfig
from dhs.train import TrainConfig, train_one

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    dcfg = DatasetConfig(**cfg["dataset"])
    aug = AugmentConfig(**cfg.get("augment", {}))
    tcfg = TrainConfig(**cfg["train"])
    best_path, best_auc = train_one(tcfg, dcfg, aug)
    print(f"BEST_AUC={best_auc:.4f} BEST_CKPT={best_path}")

if __name__ == "__main__":
    main()

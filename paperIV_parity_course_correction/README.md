# Paper IV Parity Course Correction Pack (DESI DR10 Strong Lens)

This package provides:
- Manifest validation & split-disjointness checks using **training_v1.parquet** schema (label/split/cutout_path/sample_weight/tier).
- Training scripts that **match Paper IV's explicit training protocol** as closely as practical on single-GPU:
  - **101×101** inputs (no 64×64 crop)
  - **160 epochs**
  - **Step LR halving at fixed epoch**
  - Large effective batch sizes via **gradient accumulation**
  - ResNet and EfficientNet baselines, plus optional meta-learner

## What Paper IV explicitly states (from arXiv:2508.20087v1)
- Cutouts: **101×101 pixels** per band
- Split: **70/30 train/validation** (no separate test described)
- ResNet: batch size **2048**, initial LR **5e-4**, LR halved at **epoch 80**, selected epoch around **126**.
- EfficientNet: batch size **512**, initial LR **3.88e-4**, LR halved at **epoch 130**, trained to **160 epochs** (best around epoch **50**).
- Loss: cross-entropy (binary classification), Adam optimizer, horizontal/vertical flips.

Because their exact ResNet implementation details and pixel-value normalization are not fully specified in the paper text,
this pack implements:
- ResNet: torchvision **resnet18** (closest named analogue); you can optionally swap in your custom small-ResNet if/when you have their code.
- EfficientNet: torchvision **efficientnet_b0** with ImageNet weights.

## Quickstart

1) Verify manifest & splits:
```bash
python tools/verify_manifest_and_splits.py \
  --manifest /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/manifests/training_v1.parquet \
  --out out/verify.json
```

2) Train ResNet parity run (effective batch 2048 via grad accum):
```bash
python training/train_paperIV_resnet.py \
  --manifest /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/manifests/training_v1.parquet \
  --epochs 160 --base-lr 5e-4 --lr-step-epoch 80 \
  --effective-batch 2048 --micro-batch 128 \
  --outdir out/resnet18_paperIV_parity
```

3) Train EfficientNet parity run (effective batch 512 via grad accum):
```bash
python training/train_paperIV_efficientnet.py \
  --manifest /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/manifests/training_v1.parquet \
  --epochs 160 --base-lr 3.88e-4 --lr-step-epoch 130 \
  --effective-batch 512 --micro-batch 64 \
  --outdir out/effnetb0_paperIV_parity
```

4) Optional: fit meta-learner on validation predictions (stacking):
```bash
python evaluation/fit_meta_learner.py \
  --resnet-preds out/resnet18_paperIV_parity/preds_val.parquet \
  --effnet-preds out/effnetb0_paperIV_parity/preds_val.parquet \
  --outdir out/meta_learner
```

## Notes
- These scripts intentionally **do not early-stop** by default, to match Paper IV training length.
- They still save the **best validation AUC** checkpoint for deployment/selection-function work.

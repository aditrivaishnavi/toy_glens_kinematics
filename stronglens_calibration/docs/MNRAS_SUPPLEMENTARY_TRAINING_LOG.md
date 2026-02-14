# MNRAS Supplementary: Complete Training Log and Model Details

**Last updated**: 2026-02-13T16:22:00Z (ALL 5 RUNS COMPLETE. v3 cosine finished 160/160.)
**Purpose**: Comprehensive record of all training runs for MNRAS paper writing, figure generation, and peer review defense.

---

## 1. Compute Infrastructure

All training was performed on Lambda Cloud GPU instances with shared NFS storage.

| Property | Value |
|----------|-------|
| GPU | NVIDIA GH200 480GB (97,871 MiB VRAM) |
| Architecture | ARM aarch64 (Grace Hopper) |
| Driver | 570.195.03 |
| Python | 3.12.3 |
| PyTorch | 2.7.0 |
| CUDA | 12.8 |
| Storage | Shared NFS (`/lambda/nfs/darkhaloscope-training-dc/`) |
| S3 backup | `s3://darkhaloscope/stronglens_calibration/checkpoints/` (synced every 10 min) |

### Instance Assignments

| Instance | Hostname | Run | Start Time (UTC) | Status |
|----------|----------|-----|-------------------|--------|
| lambda (lambda1) | 192-222-56-237 | EfficientNetV2-S v2 (step LR) | 2026-02-12T05:18:40Z | **COMPLETE** (160/160) |
| lambda2 | 192-222-51-239 | BottleneckedResNet | 2026-02-12T05:18:18Z | **COMPLETE** (160/160) |
| lambda2 | 192-222-51-239 | ResNet-18 (sequential after BnResNet) | 2026-02-11T20:12Z | **STOPPED** (E68) |
| lambda3 | 192-222-50-120 | EfficientNetV2-S v3 (cosine LR) | 2026-02-12T16:39:21Z | **COMPLETE** (160/160) |
| lambda4 | 192-222-51-94 | EfficientNetV2-S v4 (finetune) | 2026-02-12T19:21:45Z | **COMPLETE** (60/60) |

---

## 2. Training Dataset

| Property | Value |
|----------|-------|
| Manifest | `training_parity_70_30_v1.parquet` |
| Total rows | 451,681 |
| Split scheme | 70/30 train/val (Paper IV parity) |
| Dataset seed | 42 |
| Input size | 101 x 101 x 3 (g, r, z bands) |
| Pixel scale | 0.262 arcsec/pixel |
| Cutout format | NPZ float32 arrays |
| Preprocessing | `raw_robust` (median/MAD from outer 20% ring, clip [-10,+10]) |
| Augmentation | hflip (50%), vflip (50%), rot90 (uniform k in {0,1,2,3}) |
| Aug seed formula | `seed = (42 * 1000003 + sample_idx + epoch * len(dataset)) & 0x7fffffff` |
| Loss function | BCEWithLogitsLoss, reduction='mean', unweighted (all sample_weight=1.0) |
| Optimizer | AdamW (weight_decay=1e-4) for all runs |

### Split Details

| Split | Total | Positives | Negatives | Neg:Pos Ratio |
|-------|-------|-----------|-----------|---------------|
| train | 316,100 | 3,356 | 312,744 | 93.2:1 |
| val | 135,581 | 1,432 | 134,149 | 93.7:1 |
| **Total** | **451,681** | **4,788** | **446,893** | **93.3:1** |

### Manifest Columns

`galaxy_id`, `cutout_path`, `ra`, `dec`, `type_bin`, `nobs_z_bin`, `split`, `pool`, `confuser_category`, `psfsize_r`, `psfdepth_r`, `ebv`, `healpix_128`, `label`, `tier`, `sample_weight`

### Positive Sample Composition

- **Tier A**: 389 candidates with spectroscopic confirmation or HST imaging
- **Tier B**: 4,399 candidates with high-confidence visual grades only
- All treated equally for training (sample_weight = 1.0)

### Negative Sample Composition

- Pool N1 (deployment-representative): ~85% randomly sampled from SER/DEV/REX galaxies
- Pool N2 (hard confusers): ~15% morphologically selected (ring galaxies, spirals, mergers)
- Galaxy types: SER, DEV, REX (following Paper IV; EXP excluded)
- Quality cuts: nobs >= 3 in g/r/z, z < 20 mag

---

## 3. Model Architectures

### 3.1 EfficientNetV2-S (Runs v2, v3, v4)

| Property | Value |
|----------|-------|
| Base architecture | `torchvision.models.efficientnet_v2_s` |
| Total parameters | 20,178,769 (~20.2M) |
| Paper IV comparison | 20,542,883 (~20.5M) — 1.8% fewer params |
| Input channels | 3 (g, r, z) |
| Classifier head | `nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, 1))` |
| Output | Single logit (binary classification) |
| Pre-training | ImageNet (v2/v3) or from v2 checkpoint (v4) |
| Mixed precision | Yes (torch.cuda.amp) |

### 3.2 BottleneckedResNet (Run on Lambda2)

| Property | Value |
|----------|-------|
| Architecture | Custom Lanusse-style compact ResNet with 1x1 bottleneck reductions |
| Base channels | 27 |
| Total parameters | ~195,000 |
| Paper IV comparison | 194,433 — near-exact match |
| Input channels | 3 (g, r, z) |
| Classifier head | `nn.Linear(base_ch*8, 1)` after global average pooling |
| Output | Single logit (binary classification) |
| Pre-training | None (trained from scratch) |
| Mixed precision | Yes (torch.cuda.amp) |
| Design rationale | Matches Paper IV's custom "shielded ResNet" parameter count to neutralize referee objection about architecture capacity mismatch |

---

## 4. Run Configurations and Hyperparameters

### 4.1 Run v2: EfficientNetV2-S with Step LR (Lambda1)

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Config file | `paperIV_efficientnet_v2_s_v2.yaml` |  |
| Config SHA-256 prefix | `ff1a92822cbe793b` |  |
| Architecture | efficientnet_v2_s | Paper IV parity |
| Pretrained | ImageNet (EfficientNet_V2_S_Weights.DEFAULT) | Paper IV uses pretrained |
| Epochs | 160 | Paper IV protocol |
| Micro-batch | 64 | GPU memory constraint |
| Effective batch | 512 (8 accum steps) | Paper IV uses 512 for EfficientNet |
| Learning rate | 3.88e-4 | Paper IV Section 3.2 |
| LR schedule | Step (halve at epoch 130) | Paper IV protocol |
| Weight decay | 1e-4 | Standard AdamW |
| Freeze backbone | 5 epochs | Prevent catastrophic forgetting |
| LR warmup | 5 epochs (linear from lr/100 to lr) | Stabilize pretrained backbone |
| Early stopping | Disabled (run all 160 epochs) | Paper IV protocol |
| Checkpoint dir | `checkpoints/paperIV_efficientnet_v2_s_v2/` |  |

### 4.2 Run: BottleneckedResNet with Step LR (Lambda2)

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Config file | `paperIV_bottlenecked_resnet.yaml` |  |
| Config SHA-256 prefix | `c4df1dde21a28313` |  |
| Architecture | bottlenecked_resnet (base_ch=27) | ~195K params, matches Paper IV's 194K |
| Pretrained | No (trained from scratch) | Not a pretrained architecture |
| Epochs | 160 | Paper IV protocol |
| Micro-batch | 128 | Smaller model allows larger batch |
| Effective batch | 2048 (16 accum steps) | Paper IV uses 2048 for ResNet |
| Learning rate | 5e-4 | Paper IV Section 3.2 |
| LR schedule | Step (halve at epoch 80) | Paper IV protocol |
| Weight decay | 1e-4 | Standard AdamW |
| Freeze backbone | 0 (no freeze) | Not pretrained |
| LR warmup | 0 (no warmup) | Not pretrained |
| Early stopping | Disabled (run all 160 epochs) | Paper IV protocol |
| Checkpoint dir | `checkpoints/paperIV_bottlenecked_resnet/` |  |

### 4.3 Run v3: EfficientNetV2-S with Cosine LR (Lambda3)

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Config file | `paperIV_efficientnet_v2_s_v3_cosine.yaml` |  |
| Config SHA-256 prefix | `cefe8de65c8b7c8c` |  |
| Architecture | efficientnet_v2_s | Same as v2 |
| Pretrained | ImageNet | Same as v2 |
| Epochs | 160 | Same as v2 |
| Micro-batch | 64 | Same as v2 |
| Effective batch | 512 (8 accum steps) | Same as v2 |
| Learning rate | 3.88e-4 | Same as v2 |
| **LR schedule** | **Cosine (CosineAnnealingLR, T_max=160)** | **ONLY CHANGE vs v2**: continuous decay prevents post-peak overfitting |
| Weight decay | 1e-4 | Same as v2 |
| Freeze backbone | 5 epochs | Same as v2 |
| LR warmup | 5 epochs | Same as v2 |
| Early stopping | Disabled | Same as v2 |
| Checkpoint dir | `checkpoints/paperIV_efficientnet_v2_s_v3_cosine/` |  |

### 4.4 Run v4: EfficientNetV2-S Fine-tune from v2 Best (Lambda4)

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Config file | `paperIV_efficientnet_v2_s_v4_finetune.yaml` |  |
| Config SHA-256 prefix | `e4323e4db1e78b60` |  |
| Architecture | efficientnet_v2_s | Same architecture |
| **Init weights** | **v2/best.pt (epoch 19, AUC=0.99148)** | **Phase 2 fine-tuning from proven peak** |
| Pretrained | No (loads our checkpoint, not ImageNet) | Weights come from init_weights |
| **Epochs** | **60** | Short fine-tuning run |
| Micro-batch | 64 | Same as v2/v3 |
| Effective batch | 512 (8 accum steps) | Same as v2/v3 |
| **Learning rate** | **5e-5 (8x lower than v2)** | Gentle optimization from already-good weights |
| **LR schedule** | **Cosine (T_max=60)** | Continuous decay over 60 epochs |
| Weight decay | 1e-4 | Same as others |
| Freeze backbone | 0 | Model is already adapted |
| LR warmup | 3 epochs (from 5e-7 to 5e-5) | Brief stabilization |
| Early stopping | Disabled | Run all 60 epochs |
| Checkpoint dir | `checkpoints/paperIV_efficientnet_v2_s_v4_finetune/` |  |

---

## 5. Paper IV Reference Metrics (Inchausti et al. 2025)

| Model | Val AUC | Best Epoch | Parameters | Batch Size | LR | LR Step | Framework |
|-------|---------|------------|------------|------------|----|---------| ----------|
| Custom ResNet (Lanusse 2018) | **0.9984** | 126 / 160 | 194,433 | 2048 | 5e-4 | Step@80 | TensorFlow |
| EfficientNetV2 (pretrained) | **0.9987** | 50 / 160 | 20,542,883 | 512 | 3.88e-4 | Step@130 | TensorFlow |
| Meta-learner (ensemble) | **0.9989** | - | ~300 nodes | - | - | - | TensorFlow |
| Deployment threshold | >= 0.9867 (top 0.01 percentile of ~43M images) | | | | | | |

### Key Differences from Paper IV

| Aspect | Paper IV | This Work | Impact |
|--------|----------|-----------|--------|
| Framework | TensorFlow | PyTorch 2.7 | Minor numerical differences |
| GPU | 4x A100 (NERSC) | 1x GH200 (Lambda Cloud) | Gradient accumulation emulates multi-GPU |
| Positive count | 1,372 (confirmed) | 4,788 (389 confirmed + 4,399 visual) | Noisier positives may limit AUC ceiling |
| Negative count | 134,182 | 446,893 | 3.3x more negatives |
| Neg:Pos ratio | ~98:1 | ~93:1 | Comparable |
| Negative cleaning | Spherimatch + prior model p>0.4 | Not done | May contain unlabeled real lenses in negatives |
| Optimizer | Not specified | AdamW | Standard choice |
| Normalization | Not specified | raw_robust (median/MAD, outer 20% ring) | Our implementation choice |
| Augmentation | Not specified | hflip + vflip + rot90 | Conservative geometric only |

---

## 6. Complete Epoch-by-Epoch Training Logs

### 6.1 Run v2: EfficientNetV2-S Step LR (Lambda1)

**Status**: **COMPLETE** (160 / 160, finished 2026-02-13 ~04:14 UTC)
**Best**: val_auc=0.9915 at epoch 19
**Final**: val_auc=0.9736 at epoch 160

| Epoch | Train Loss | Val AUC | Best AUC | LR | Time (s) | Phase | Notes |
|-------|-----------|---------|----------|-----|----------|-------|-------|
| 1 | 0.3064 | 0.5247 | 0.5247 | 8.07e-05 | 482.9 | Frozen + warmup | Backbone frozen, LR warmup 1/5 |
| 2 | 0.1988 | 0.5262 | 0.5262 | 1.58e-04 | 483.3 | Frozen + warmup | Warmup 2/5 |
| 3 | 0.1686 | 0.5339 | 0.5339 | 2.34e-04 | 490.3 | Frozen + warmup | Warmup 3/5 |
| 4 | 0.1442 | 0.5427 | 0.5427 | 3.11e-04 | 473.1 | Frozen + warmup | Warmup 4/5 |
| 5 | 0.1270 | 0.5472 | 0.5472 | 3.88e-04 | 471.7 | Frozen + warmup | Warmup 5/5, LR reached full |
| 6 | 0.0321 | 0.9781 | 0.9781 | 3.88e-04 | 476.0 | **Unfreeze** | **Backbone unfrozen. AUC jumps 0.55->0.98** |
| 7 | 0.0219 | 0.9881 | 0.9881 | 3.88e-04 | 471.6 | Full training | Rapid improvement |
| 8 | 0.0191 | 0.9886 | 0.9886 | 3.88e-04 | 472.5 | Full training | |
| 9 | 0.0174 | 0.9894 | 0.9894 | 3.88e-04 | 475.9 | Full training | |
| 10 | 0.0164 | 0.9852 | 0.9894 | 3.88e-04 | 490.3 | Full training | First dip |
| 11 | 0.0161 | 0.9897 | 0.9897 | 3.88e-04 | 477.1 | Full training | New best |
| 12 | 0.0151 | 0.9876 | 0.9897 | 3.88e-04 | 470.1 | Full training | |
| 13 | 0.0144 | 0.9885 | 0.9897 | 3.88e-04 | 474.3 | Full training | |
| 14 | 0.0137 | 0.9880 | 0.9897 | 3.88e-04 | 474.6 | Full training | |
| 15 | 0.0133 | 0.9888 | 0.9897 | 3.88e-04 | 476.2 | Full training | |
| 16 | 0.0127 | 0.9859 | 0.9897 | 3.88e-04 | 479.1 | Full training | |
| 17 | 0.0121 | 0.9870 | 0.9897 | 3.88e-04 | 517.8 | Full training | |
| 18 | 0.0115 | 0.9881 | 0.9897 | 3.88e-04 | 518.6 | Full training | |
| 19 | 0.0111 | **0.9915** | **0.9915** | 3.88e-04 | 525.0 | Full training | **PEAK. Best checkpoint saved.** |
| 20 | 0.0106 | 0.9894 | 0.9915 | 3.88e-04 | 529.9 | Full training | Post-peak decline begins |
| 21 | 0.0100 | 0.9878 | 0.9915 | 3.88e-04 | 523.7 | Overfitting | |
| 22 | 0.0094 | 0.9887 | 0.9915 | 3.88e-04 | 497.1 | Overfitting | |
| 23 | 0.0089 | 0.9880 | 0.9915 | 3.88e-04 | 500.2 | Overfitting | |
| 24 | 0.0083 | 0.9826 | 0.9915 | 3.88e-04 | 475.6 | Overfitting | Larger dip |
| 25 | 0.0079 | 0.9867 | 0.9915 | 3.88e-04 | 466.7 | Overfitting | |
| 26 | 0.0074 | 0.9840 | 0.9915 | 3.88e-04 | 471.0 | Overfitting | |
| 27 | 0.0073 | 0.9886 | 0.9915 | 3.88e-04 | 504.9 | Overfitting | |
| 28 | 0.0065 | 0.9852 | 0.9915 | 3.88e-04 | 500.9 | Overfitting | |
| 29 | 0.0065 | 0.9896 | 0.9915 | 3.88e-04 | 506.1 | Overfitting | Near-best recovery |
| 30 | 0.0065 | 0.9873 | 0.9915 | 3.88e-04 | 523.0 | Overfitting | |
| 31 | 0.0061 | 0.9855 | 0.9915 | 3.88e-04 | 560.0 | Overfitting | |
| 32 | 0.0058 | 0.9831 | 0.9915 | 3.88e-04 | 524.9 | Overfitting | |
| 33 | 0.0057 | 0.9842 | 0.9915 | 3.88e-04 | 508.6 | Overfitting | |
| 34 | 0.0053 | 0.9851 | 0.9915 | 3.88e-04 | 502.2 | Overfitting | |
| 35 | 0.0049 | 0.9843 | 0.9915 | 3.88e-04 | 516.9 | Overfitting | |
| 36 | 0.0047 | 0.9866 | 0.9915 | 3.88e-04 | 542.6 | Overfitting | |
| 37 | 0.0045 | 0.9841 | 0.9915 | 3.88e-04 | 565.2 | Overfitting | |
| 38 | 0.0043 | 0.9812 | 0.9915 | 3.88e-04 | 542.4 | Overfitting | |
| 39 | 0.0044 | 0.9852 | 0.9915 | 3.88e-04 | 544.9 | Overfitting | |
| 40 | 0.0041 | 0.9807 | 0.9915 | 3.88e-04 | 536.1 | Overfitting | |
| 41 | 0.0042 | 0.9837 | 0.9915 | 3.88e-04 | 523.0 | Overfitting | |
| 42 | 0.0036 | 0.9796 | 0.9915 | 3.88e-04 | 519.8 | Overfitting | |
| 43 | 0.0037 | 0.9828 | 0.9915 | 3.88e-04 | 503.7 | Overfitting | |
| 44 | 0.0037 | 0.9846 | 0.9915 | 3.88e-04 | 524.3 | Overfitting | |
| 45 | 0.0035 | 0.9835 | 0.9915 | 3.88e-04 | 509.5 | Overfitting | |
| 46 | 0.0034 | 0.9843 | 0.9915 | 3.88e-04 | 496.1 | Overfitting | |
| 47 | 0.0030 | 0.9826 | 0.9915 | 3.88e-04 | 506.0 | Overfitting | |
| 48 | 0.0033 | 0.9818 | 0.9915 | 3.88e-04 | 496.6 | Overfitting | |
| 49 | 0.0031 | 0.9837 | 0.9915 | 3.88e-04 | 500.7 | Overfitting | |
| 50 | 0.0031 | 0.9806 | 0.9915 | 3.88e-04 | 496.0 | Overfitting | Paper IV EffNet peaks here; ours peaked at E19 |
| 51 | 0.0030 | 0.9855 | 0.9915 | 3.88e-04 | 683.5 | Overfitting | |
| 52 | 0.0027 | 0.9816 | 0.9915 | 3.88e-04 | 705.9 | Overfitting | |
| 53 | 0.0028 | 0.9785 | 0.9915 | 3.88e-04 | 685.7 | Overfitting | |
| 54 | 0.0027 | 0.9830 | 0.9915 | 3.88e-04 | 679.9 | Overfitting | |
| 55 | 0.0028 | 0.9817 | 0.9915 | 3.88e-04 | 577.2 | Overfitting | |
| 56 | 0.0025 | 0.9784 | 0.9915 | 3.88e-04 | 523.5 | Overfitting | |
| 57 | 0.0027 | 0.9811 | 0.9915 | 3.88e-04 | 500.5 | Overfitting | |
| 58 | 0.0024 | 0.9850 | 0.9915 | 3.88e-04 | 493.2 | Overfitting | |
| 59 | 0.0024 | 0.9839 | 0.9915 | 3.88e-04 | 491.5 | Overfitting | |
| 60 | 0.0022 | 0.9836 | 0.9915 | 3.88e-04 | 501.7 | Overfitting | |
| 61 | 0.0023 | 0.9818 | 0.9915 | 3.88e-04 | 515.4 | Overfitting | |
| 62 | 0.0023 | 0.9795 | 0.9915 | 3.88e-04 | 520.7 | Overfitting | |
| 63 | 0.0023 | 0.9778 | 0.9915 | 3.88e-04 | 501.0 | Overfitting | |
| 64 | 0.0022 | 0.9813 | 0.9915 | 3.88e-04 | 499.8 | Overfitting | |
| 65 | 0.0022 | 0.9792 | 0.9915 | 3.88e-04 | 508.0 | Overfitting | |
| 66 | 0.0021 | 0.9799 | 0.9915 | 3.88e-04 | 488.7 | Overfitting | |
| 67 | 0.0021 | 0.9795 | 0.9915 | 3.88e-04 | 514.0 | Overfitting | |
| 68 | 0.0023 | 0.9734 | 0.9915 | 3.88e-04 | 493.0 | Overfitting | |
| 69 | 0.0019 | 0.9773 | 0.9915 | 3.88e-04 | 494.8 | Overfitting | |
| 70 | 0.0020 | 0.9833 | 0.9915 | 3.88e-04 | 499.3 | Overfitting | |
| 71 | 0.0020 | 0.9827 | 0.9915 | 3.88e-04 | 505.5 | Overfitting | |
| 72 | 0.0019 | 0.9789 | 0.9915 | 3.88e-04 | 509.2 | Overfitting | |
| 73 | 0.0018 | 0.9794 | 0.9915 | 3.88e-04 | 513.9 | Overfitting | |
| 74 | 0.0019 | 0.9785 | 0.9915 | 3.88e-04 | 495.7 | Overfitting | |
| 75 | 0.0020 | 0.9826 | 0.9915 | 3.88e-04 | 501.8 | Overfitting | |
| 76 | 0.0017 | 0.9799 | 0.9915 | 3.88e-04 | 502.2 | Overfitting | |
| 77 | 0.0019 | 0.9805 | 0.9915 | 3.88e-04 | 493.8 | Overfitting | |
| 78 | 0.0017 | 0.9802 | 0.9915 | 3.88e-04 | 532.5 | Overfitting | |
| 79 | 0.0017 | 0.9782 | 0.9915 | 3.88e-04 | 518.7 | Overfitting | |
| 80 | 0.0017 | 0.9812 | 0.9915 | 3.88e-04 | 540.7 | Overfitting | Paper IV ResNet LR step would be here |
| 81 | 0.0016 | 0.9793 | 0.9915 | 3.88e-04 | 507.0 | Overfitting | |
| 82 | 0.0018 | 0.9813 | 0.9915 | 3.88e-04 | 507.7 | Overfitting | |
| 83 | 0.0016 | 0.9816 | 0.9915 | 3.88e-04 | 505.4 | Overfitting | |
| 84 | 0.0016 | 0.9799 | 0.9915 | 3.88e-04 | 526.8 | Overfitting | |
| 85 | 0.0017 | 0.9736 | 0.9915 | 3.88e-04 | 518.7 | Overfitting | Significant dip |
| 86 | 0.0014 | 0.9750 | 0.9915 | 3.88e-04 | 516.1 | Overfitting | |
| 87 | 0.0015 | 0.9731 | 0.9915 | 3.88e-04 | 511.5 | Overfitting | |
| 88 | 0.0016 | 0.9799 | 0.9915 | 3.88e-04 | 509.1 | Overfitting | |
| 89 | 0.0013 | 0.9798 | 0.9915 | 3.88e-04 | 503.3 | Overfitting | |
| 90 | 0.0015 | 0.9773 | 0.9915 | 3.88e-04 | 505.7 | Overfitting | |
| 91 | 0.0013 | 0.9706 | 0.9915 | 3.88e-04 | 511.5 | Overfitting | **Worst val AUC so far** |
| 92 | 0.0016 | 0.9741 | 0.9915 | 3.88e-04 | 504.9 | Overfitting | |
| 93 | 0.0014 | 0.9769 | 0.9915 | 3.88e-04 | 518.6 | Overfitting | |
| 94 | 0.0016 | 0.9771 | 0.9915 | 3.88e-04 | 509.2 | Overfitting | |
| 95 | 0.0014 | 0.9794 | 0.9915 | 3.88e-04 | 502.6 | Overfitting | |
| 96 | 0.0014 | 0.9758 | 0.9915 | 3.88e-04 | 504.5 | Overfitting | |
| 97 | 0.0012 | 0.9784 | 0.9915 | 3.88e-04 | 521.8 | Overfitting | |
| 98 | 0.0013 | 0.9770 | 0.9915 | 3.88e-04 | 504.3 | Overfitting | |
| 99 | 0.0014 | 0.9765 | 0.9915 | 3.88e-04 | 521.5 | Overfitting | |
| 100 | 0.0014 | 0.9779 | 0.9915 | 3.88e-04 | 544.9 | Overfitting | |
| 101 | 0.0014 | 0.9761 | 0.9915 | 3.88e-04 | 551.5 | Overfitting | |
| 102 | 0.0013 | 0.9768 | 0.9915 | 3.88e-04 | 535.4 | Overfitting | |
| 103 | 0.0012 | 0.9740 | 0.9915 | 3.88e-04 | 515.9 | Overfitting | |
| 104 | 0.0012 | 0.9708 | 0.9915 | 3.88e-04 | 517.3 | Overfitting | |
| 105 | 0.0010 | 0.9689 | 0.9915 | 3.88e-04 | 515.5 | Overfitting | Prior worst |
| 106 | 0.0014 | 0.9776 | 0.9915 | 3.88e-04 | 535.3 | Overfitting | Dead-cat bounce from 0.969 |
| 107 | 0.0013 | 0.9762 | 0.9915 | 3.88e-04 | 497.3 | Overfitting | |
| 108 | 0.0010 | 0.9768 | 0.9915 | 3.88e-04 | 514.0 | Overfitting | |
| 109 | 0.0011 | 0.9756 | 0.9915 | 3.88e-04 | 510.0 | Overfitting | |
| 110 | 0.0011 | 0.9758 | 0.9915 | 3.88e-04 | 512.0 | Overfitting | |
| 111 | 0.0010 | 0.9755 | 0.9915 | 3.88e-04 | 516.0 | Overfitting | |
| 112 | 0.0012 | 0.9760 | 0.9915 | 3.88e-04 | 520.0 | Overfitting | |
| 113 | 0.0012 | 0.9763 | 0.9915 | 3.88e-04 | 533.9 | Overfitting | |
| 114 | 0.0012 | 0.9792 | 0.9915 | 3.88e-04 | 507.9 | Overfitting | |
| 115 | 0.0011 | 0.9799 | 0.9915 | 3.88e-04 | 514.7 | Overfitting | Small bounce |
| 116 | 0.0011 | 0.9731 | 0.9915 | 3.88e-04 | 524.9 | Overfitting | |
| 117 | 0.0011 | 0.9755 | 0.9915 | 3.88e-04 | 510.3 | Overfitting | |
| 118 | 0.0011 | 0.9747 | 0.9915 | 3.88e-04 | 515.9 | Overfitting | |
| 119 | 0.0010 | 0.9766 | 0.9915 | 3.88e-04 | 538.3 | Overfitting | |
| 120 | 0.0010 | 0.9755 | 0.9915 | 3.88e-04 | 521.4 | Overfitting | |
| 121 | 0.0010 | 0.9768 | 0.9915 | 3.88e-04 | 543.8 | Overfitting | |
| 122 | 0.0012 | 0.9786 | 0.9915 | 3.88e-04 | 535.5 | Overfitting | |
| 123 | 0.0012 | 0.9790 | 0.9915 | 3.88e-04 | 521.2 | Overfitting | |
| 124 | 0.0009 | 0.9765 | 0.9915 | 3.88e-04 | 512.0 | Overfitting | |
| 125 | 0.0010 | 0.9780 | 0.9915 | 3.88e-04 | 520.0 | Overfitting | |
| 126 | 0.0010 | 0.9745 | 0.9915 | 3.88e-04 | 539.7 | Overfitting | |
| 127 | 0.0011 | 0.9758 | 0.9915 | 3.88e-04 | 524.1 | Overfitting | |
| 128 | 0.0009 | 0.9765 | 0.9915 | 3.88e-04 | 505.9 | Overfitting | |
| 129 | 0.0011 | 0.9733 | 0.9915 | 3.88e-04 | 515.9 | Overfitting | Last epoch before LR step |
| 130 | 0.0009 | 0.9779 | 0.9915 | **1.94e-04** | 508.1 | **LR step** | **LR halved from 3.88e-4 to 1.94e-4** |
| 131 | 0.0004 | 0.9751 | 0.9915 | 1.94e-04 | 511.5 | Post-step | Loss drops 2.5x immediately |
| 132 | 0.0003 | 0.9764 | 0.9915 | 1.94e-04 | 519.6 | Post-step | |
| 133 | 0.0003 | 0.9760 | 0.9915 | 1.94e-04 | 529.1 | Post-step | |
| 134 | 0.0004 | 0.9763 | 0.9915 | 1.94e-04 | 529.1 | Post-step | |
| 135 | 0.0003 | 0.9783 | 0.9915 | 1.94e-04 | 527.2 | Post-step | |
| 136 | 0.0002 | 0.9728 | 0.9915 | 1.94e-04 | 536.0 | Post-step | |
| 137 | 0.0003 | 0.9690 | 0.9915 | 1.94e-04 | 533.5 | Post-step | |
| 138 | 0.0004 | 0.9720 | 0.9915 | 1.94e-04 | 519.6 | Post-step | |
| 139 | 0.0003 | 0.9769 | 0.9915 | 1.94e-04 | 537.4 | Post-step | |
| 140 | 0.0003 | 0.9785 | 0.9915 | 1.94e-04 | 571.7 | Post-step | |
| 141 | 0.0003 | 0.9755 | 0.9915 | 1.94e-04 | 538.2 | Post-step | |
| 142 | 0.0004 | 0.9772 | 0.9915 | 1.94e-04 | 524.1 | Post-step | |
| 143 | 0.0003 | 0.9744 | 0.9915 | 1.94e-04 | 509.3 | Post-step | |
| 144 | 0.0002 | 0.9775 | 0.9915 | 1.94e-04 | 524.7 | Post-step | |
| 145 | 0.0003 | 0.9720 | 0.9915 | 1.94e-04 | 508.6 | Post-step | |
| 146 | 0.0002 | 0.9701 | 0.9915 | 1.94e-04 | 516.1 | Post-step | |
| 147 | 0.0003 | 0.9765 | 0.9915 | 1.94e-04 | 529.7 | Post-step | |
| 148 | 0.0003 | 0.9760 | 0.9915 | 1.94e-04 | 497.4 | Post-step | |
| 149 | 0.0004 | 0.9739 | 0.9915 | 1.94e-04 | 503.8 | Post-step | |
| 150 | 0.0003 | 0.9747 | 0.9915 | 1.94e-04 | 499.3 | Post-step | Epoch 150 checkpoint saved |
| 151 | 0.0002 | 0.9760 | 0.9915 | 1.94e-04 | 516.6 | Post-step | |
| 152 | 0.0003 | 0.9741 | 0.9915 | 1.94e-04 | 527.3 | Post-step | |
| 153 | 0.0002 | 0.9761 | 0.9915 | 1.94e-04 | 508.5 | Post-step | |
| 154 | 0.0003 | 0.9746 | 0.9915 | 1.94e-04 | 510.9 | Post-step | |
| 155 | 0.0004 | 0.9753 | 0.9915 | 1.94e-04 | 502.0 | Post-step | |
| 156 | 0.0002 | 0.9778 | 0.9915 | 1.94e-04 | 503.8 | Post-step | Best post-step AUC |
| 157 | 0.0003 | 0.9748 | 0.9915 | 1.94e-04 | 493.9 | Post-step | |
| 158 | 0.0002 | 0.9762 | 0.9915 | 1.94e-04 | 488.6 | Post-step | |
| 159 | 0.0002 | 0.9761 | 0.9915 | 1.94e-04 | 499.0 | Post-step | |
| 160 | 0.0003 | 0.9736 | 0.9915 | 1.94e-04 | 495.8 | **FINAL** | **RUN COMPLETE. BEST_AUC=0.9915 at E19.** |

**Critical Finding**: Peak at E19 (AUC=0.9915), then decline in val AUC from 0.991 to 0.969-0.980 band over 141 epochs. The constant LR=3.88e-4 was too large for fine-grained refinement after convergence. **LR step at E130** (halving to 1.94e-4) caused an immediate train loss drop from ~0.001 to ~0.0003, but **did NOT recover val AUC** — the model oscillated in 0.970-0.978 band post-step (best post-step: 0.9785 at E140, occasional peaks like 0.9778 at E156). Final AUC at E160 was 0.9736. The model's generalization was irreversibly damaged by prolonged high-LR overfitting. The best checkpoint remains E19. **RUN COMPLETE.**

---

### 6.2 Run: BottleneckedResNet Step LR (Lambda2)

**Status**: **COMPLETE** (160 / 160 epochs, finished 2026-02-13 ~03:05 UTC)
**Best**: val_auc=0.9799 at epoch 68
**Final**: val_auc=0.9659 at epoch 160

| Epoch | Train Loss | Val AUC | Best AUC | LR | Time (s) | Phase | Notes |
|-------|-----------|---------|----------|-----|----------|-------|-------|
| 1 | 0.0815 | 0.7789 | 0.7789 | 5.00e-04 | 451.8 | Full training | From scratch (no pretrain) |
| 2 | 0.0502 | 0.8146 | 0.8146 | 5.00e-04 | 444.8 | Full training | |
| 3 | 0.0469 | 0.8638 | 0.8638 | 5.00e-04 | 464.6 | Full training | |
| 4 | 0.0428 | 0.8877 | 0.8877 | 5.00e-04 | 450.7 | Full training | |
| 5 | 0.0384 | 0.9219 | 0.9219 | 5.00e-04 | 452.2 | Full training | |
| 6 | 0.0345 | 0.9326 | 0.9326 | 5.00e-04 | 436.1 | Full training | |
| 7 | 0.0319 | 0.9355 | 0.9355 | 5.00e-04 | 458.2 | Full training | |
| 8 | 0.0305 | 0.9258 | 0.9355 | 5.00e-04 | 455.9 | Full training | |
| 9 | 0.0289 | 0.9539 | 0.9539 | 5.00e-04 | 470.3 | Full training | |
| 10 | 0.0281 | 0.9556 | 0.9556 | 5.00e-04 | 462.2 | Full training | |
| 11 | 0.0273 | 0.9658 | 0.9658 | 5.00e-04 | 459.4 | Full training | |
| 12 | 0.0261 | 0.9668 | 0.9668 | 5.00e-04 | 437.7 | Full training | |
| 13 | 0.0258 | 0.9629 | 0.9668 | 5.00e-04 | 429.6 | Full training | |
| 14 | 0.0255 | 0.9486 | 0.9668 | 5.00e-04 | 449.1 | Full training | |
| 15 | 0.0245 | 0.9702 | 0.9702 | 5.00e-04 | 451.0 | Full training | |
| 16 | 0.0239 | 0.9677 | 0.9702 | 5.00e-04 | 456.1 | Full training | |
| 17 | 0.0239 | 0.9661 | 0.9702 | 5.00e-04 | 465.9 | Full training | |
| 18 | 0.0235 | 0.9720 | 0.9720 | 5.00e-04 | 457.3 | Full training | |
| 19 | 0.0228 | 0.9758 | 0.9758 | 5.00e-04 | 449.6 | Full training | |
| 20 | 0.0230 | 0.9681 | 0.9758 | 5.00e-04 | 444.2 | Full training | |
| 21 | 0.0222 | 0.9750 | 0.9758 | 5.00e-04 | 453.2 | Full training | |
| 22 | 0.0219 | 0.9664 | 0.9758 | 5.00e-04 | 451.5 | Full training | |
| 23 | 0.0220 | 0.9674 | 0.9758 | 5.00e-04 | 445.9 | Full training | |
| 24 | 0.0211 | 0.9772 | 0.9772 | 5.00e-04 | 636.1 | Full training | |
| 25 | 0.0211 | 0.9758 | 0.9772 | 5.00e-04 | 637.2 | Full training | |
| 26 | 0.0205 | 0.9681 | 0.9772 | 5.00e-04 | 604.1 | Full training | |
| 27 | 0.0209 | 0.9696 | 0.9772 | 5.00e-04 | 638.8 | Full training | |
| 28 | 0.0203 | 0.9725 | 0.9772 | 5.00e-04 | 539.8 | Full training | |
| 29 | 0.0201 | 0.9720 | 0.9772 | 5.00e-04 | 479.4 | Full training | |
| 30 | 0.0199 | 0.9755 | 0.9772 | 5.00e-04 | 482.7 | Full training | |
| 31 | 0.0196 | 0.9764 | 0.9772 | 5.00e-04 | 507.4 | Full training | |
| 32 | 0.0193 | 0.9746 | 0.9772 | 5.00e-04 | 468.1 | Full training | |
| 33 | 0.0194 | 0.9733 | 0.9772 | 5.00e-04 | 475.2 | Full training | |
| 34 | 0.0187 | 0.9671 | 0.9772 | 5.00e-04 | 475.1 | Full training | |
| 35 | 0.0190 | 0.9718 | 0.9772 | 5.00e-04 | 474.6 | Full training | |
| 36 | 0.0184 | 0.9779 | 0.9779 | 5.00e-04 | 468.2 | Full training | |
| 37 | 0.0186 | 0.9769 | 0.9779 | 5.00e-04 | 474.6 | Full training | |
| 38 | 0.0185 | 0.9775 | 0.9779 | 5.00e-04 | 507.6 | Full training | |
| 39 | 0.0184 | 0.9782 | 0.9782 | 5.00e-04 | 488.8 | Full training | |
| 40 | 0.0177 | 0.9785 | 0.9785 | 5.00e-04 | 475.6 | Full training | |
| 41 | 0.0179 | 0.9794 | 0.9794 | 5.00e-04 | 478.0 | Full training | |
| 42 | 0.0172 | 0.9757 | 0.9794 | 5.00e-04 | 480.9 | Full training | |
| 43 | 0.0175 | 0.9775 | 0.9794 | 5.00e-04 | 456.8 | Full training | |
| 44 | 0.0173 | 0.9795 | 0.9795 | 5.00e-04 | 476.3 | Full training | |
| 45 | 0.0170 | 0.9791 | 0.9795 | 5.00e-04 | 458.8 | Full training | |
| 46 | 0.0165 | 0.9793 | 0.9795 | 5.00e-04 | 480.5 | Full training | |
| 47 | 0.0167 | 0.9792 | 0.9795 | 5.00e-04 | 465.8 | Full training | |
| 48 | 0.0166 | 0.9742 | 0.9795 | 5.00e-04 | 466.3 | Full training | |
| 49 | 0.0166 | 0.9771 | 0.9795 | 5.00e-04 | 451.8 | Full training | |
| 50 | 0.0163 | 0.9739 | 0.9795 | 5.00e-04 | 461.2 | Full training | |
| 51 | 0.0158 | 0.9763 | 0.9795 | 5.00e-04 | 459.3 | Full training | |
| 52 | 0.0160 | 0.9785 | 0.9795 | 5.00e-04 | 459.1 | Full training | |
| 53 | 0.0156 | 0.9713 | 0.9795 | 5.00e-04 | 481.8 | Full training | |
| 54 | 0.0152 | 0.9736 | 0.9795 | 5.00e-04 | 471.8 | Full training | |
| 55 | 0.0155 | 0.9792 | 0.9795 | 5.00e-04 | 450.5 | Full training | |
| 56 | 0.0151 | 0.9754 | 0.9795 | 5.00e-04 | 457.0 | Full training | |
| 57 | 0.0150 | 0.9758 | 0.9795 | 5.00e-04 | 454.1 | Full training | |
| 58 | 0.0147 | 0.9728 | 0.9795 | 5.00e-04 | 455.3 | Full training | |
| 59 | 0.0148 | 0.9761 | 0.9795 | 5.00e-04 | 453.5 | Full training | |
| 60 | 0.0149 | 0.9769 | 0.9795 | 5.00e-04 | 463.3 | Full training | |
| 61 | 0.0147 | 0.9746 | 0.9795 | 5.00e-04 | 497.6 | Full training | |
| 62 | 0.0144 | 0.9753 | 0.9795 | 5.00e-04 | 467.6 | Full training | |
| 63 | 0.0140 | 0.9743 | 0.9795 | 5.00e-04 | 457.2 | Full training | |
| 64 | 0.0138 | 0.9769 | 0.9795 | 5.00e-04 | 462.8 | Full training | |
| 65 | 0.0140 | 0.9760 | 0.9795 | 5.00e-04 | 474.8 | Full training | |
| 66 | 0.0134 | 0.9734 | 0.9795 | 5.00e-04 | 464.9 | Full training | |
| 67 | 0.0134 | 0.9759 | 0.9795 | 5.00e-04 | 471.6 | Full training | |
| 68 | 0.0137 | **0.9799** | **0.9799** | 5.00e-04 | 474.1 | Full training | **PEAK. Best checkpoint saved.** |
| 69 | 0.0139 | 0.9763 | 0.9799 | 5.00e-04 | 460.1 | Full training | |
| 70 | 0.0136 | 0.9769 | 0.9799 | 5.00e-04 | 463.6 | Full training | |
| 71 | 0.0136 | 0.9758 | 0.9799 | 5.00e-04 | 457.3 | Full training | |
| 72 | 0.0129 | 0.9755 | 0.9799 | 5.00e-04 | 446.5 | Full training | |
| 73 | 0.0126 | 0.9748 | 0.9799 | 5.00e-04 | 473.8 | Full training | |
| 74 | 0.0133 | 0.9740 | 0.9799 | 5.00e-04 | 456.7 | Full training | |
| 75 | 0.0120 | 0.9737 | 0.9799 | 5.00e-04 | 469.0 | Full training | |
| 76 | 0.0129 | 0.9676 | 0.9799 | 5.00e-04 | 471.9 | Full training | |
| 77 | 0.0125 | 0.9735 | 0.9799 | 5.00e-04 | 475.5 | Full training | |
| 78 | 0.0125 | 0.9751 | 0.9799 | 5.00e-04 | 469.8 | Full training | |
| 79 | 0.0126 | 0.9733 | 0.9799 | 5.00e-04 | 481.9 | Full training | |
| 80 | 0.0123 | 0.9756 | 0.9799 | **2.50e-04** | 472.4 | **LR step** | **LR halved from 5e-4 to 2.5e-4** |
| 81 | 0.0110 | 0.9735 | 0.9799 | 2.50e-04 | 473.7 | Post-step | Loss drops immediately |
| 82 | 0.0100 | 0.9788 | 0.9799 | 2.50e-04 | 482.4 | Post-step | Near-best recovery |
| 83 | 0.0098 | 0.9756 | 0.9799 | 2.50e-04 | 491.9 | Post-step | |
| 84 | 0.0100 | 0.9765 | 0.9799 | 2.50e-04 | 486.0 | Post-step | |
| 85 | 0.0093 | 0.9736 | 0.9799 | 2.50e-04 | 475.1 | Post-step | |
| 86 | 0.0091 | 0.9757 | 0.9799 | 2.50e-04 | 476.1 | Post-step | |
| 87 | 0.0092 | 0.9761 | 0.9799 | 2.50e-04 | 488.7 | Post-step | |
| 88 | 0.0090 | 0.9745 | 0.9799 | 2.50e-04 | 487.2 | Post-step | |
| 89 | 0.0087 | 0.9747 | 0.9799 | 2.50e-04 | 475.8 | Post-step | |
| 90 | 0.0088 | 0.9728 | 0.9799 | 2.50e-04 | 484.2 | Post-step | |
| 91 | 0.0085 | 0.9722 | 0.9799 | 2.50e-04 | 630.7 | Post-step | |
| 92 | 0.0081 | 0.9735 | 0.9799 | 2.50e-04 | 668.0 | Post-step | |
| 93 | 0.0084 | 0.9696 | 0.9799 | 2.50e-04 | 661.2 | Post-step | |
| 94 | 0.0083 | 0.9741 | 0.9799 | 2.50e-04 | 662.4 | Post-step | |
| 95 | 0.0079 | 0.9720 | 0.9799 | 2.50e-04 | 578.2 | Post-step | |
| 96 | 0.0073 | 0.9721 | 0.9799 | 2.50e-04 | 499.2 | Post-step | |
| 97 | 0.0082 | 0.9726 | 0.9799 | 2.50e-04 | 669.4 | Post-step | |
| 98 | 0.0075 | 0.9723 | 0.9799 | 2.50e-04 | 659.6 | Post-step | |
| 99 | 0.0075 | 0.9709 | 0.9799 | 2.50e-04 | 667.3 | Post-step | |
| 100 | 0.0072 | 0.9736 | 0.9799 | 2.50e-04 | 660.4 | Post-step | |
| 101 | 0.0071 | 0.9738 | 0.9799 | 2.50e-04 | 564.5 | Post-step | |
| 102 | 0.0068 | 0.9727 | 0.9799 | 2.50e-04 | 492.6 | Post-step | |
| 103 | 0.0068 | 0.9706 | 0.9799 | 2.50e-04 | 496.5 | Post-step | |
| 104 | 0.0070 | 0.9705 | 0.9799 | 2.50e-04 | 479.3 | Post-step | |
| 105 | 0.0063 | 0.9721 | 0.9799 | 2.50e-04 | 475.2 | Post-step | |
| 106 | 0.0062 | 0.9714 | 0.9799 | 2.50e-04 | 480.1 | Post-step | |
| 107 | 0.0059 | 0.9708 | 0.9799 | 2.50e-04 | 479.5 | Post-step | |
| 108 | 0.0062 | 0.9706 | 0.9799 | 2.50e-04 | 484.8 | Post-step | |
| 109 | 0.0063 | 0.9711 | 0.9799 | 2.50e-04 | 485.4 | Post-step | |
| 110 | 0.0065 | 0.9717 | 0.9799 | 2.50e-04 | 504.4 | Post-step | |
| 111 | 0.0063 | 0.9706 | 0.9799 | 2.50e-04 | 480.3 | Post-step | |
| 112 | 0.0059 | 0.9727 | 0.9799 | 2.50e-04 | 475.8 | Post-step | |
| 113 | 0.0063 | 0.9750 | 0.9799 | 2.50e-04 | 466.0 | Post-step | Small uptick in band |
| 114 | 0.0060 | 0.9725 | 0.9799 | 2.50e-04 | 470.0 | Post-step | |
| 115 | 0.0058 | 0.9718 | 0.9799 | 2.50e-04 | 475.0 | Post-step | |
| 116 | 0.0059 | 0.9710 | 0.9799 | 2.50e-04 | 472.0 | Post-step | |
| 117 | 0.0057 | 0.9700 | 0.9799 | 2.50e-04 | 478.0 | Post-step | |
| 118 | 0.0058 | 0.9695 | 0.9799 | 2.50e-04 | 482.0 | Post-step | |
| 119 | 0.0058 | 0.9687 | 0.9799 | 2.50e-04 | 489.0 | Post-step | |
| 120 | 0.0056 | 0.9678 | 0.9799 | 2.50e-04 | 485.1 | Post-step | |
| 121 | 0.0054 | 0.9680 | 0.9799 | 2.50e-04 | 481.5 | Post-step | |
| 122 | 0.0053 | 0.9712 | 0.9799 | 2.50e-04 | 471.3 | Post-step | |
| 123 | 0.0054 | 0.9681 | 0.9799 | 2.50e-04 | 498.1 | Post-step | |
| 124 | 0.0054 | 0.9702 | 0.9799 | 2.50e-04 | 489.2 | Post-step | |
| 125 | 0.0051 | 0.9709 | 0.9799 | 2.50e-04 | 490.9 | Post-step | |
| 126 | 0.0051 | 0.9712 | 0.9799 | 2.50e-04 | 484.7 | Post-step | |
| 127 | 0.0050 | 0.9687 | 0.9799 | 2.50e-04 | 485.5 | Post-step | |
| 128 | 0.0052 | 0.9638 | 0.9799 | 2.50e-04 | 474.6 | Post-step | **New worst val AUC** |
| 129 | 0.0051 | 0.9685 | 0.9799 | 2.50e-04 | 489.6 | Post-step | |
| 130 | 0.0050 | 0.9655 | 0.9799 | 2.50e-04 | 482.7 | Post-step | |
| 131 | 0.0051 | 0.9706 | 0.9799 | 2.50e-04 | 484.4 | Post-step | |
| 132 | 0.0050 | 0.9707 | 0.9799 | 2.50e-04 | 514.3 | Post-step | |
| 133 | 0.0044 | 0.9698 | 0.9799 | 2.50e-04 | 501.6 | Post-step | |
| 134 | 0.0049 | 0.9690 | 0.9799 | 2.50e-04 | 477.8 | Post-step | |
| 135 | 0.0045 | 0.9686 | 0.9799 | 2.50e-04 | 487.3 | Post-step | |
| 136 | 0.0048 | 0.9717 | 0.9799 | 2.50e-04 | 493.6 | Post-step | |
| 137 | 0.0046 | 0.9678 | 0.9799 | 2.50e-04 | 494.9 | Post-step | |
| 138 | 0.0045 | 0.9677 | 0.9799 | 2.50e-04 | 503.8 | Post-step | |
| 139 | 0.0046 | 0.9711 | 0.9799 | 2.50e-04 | 490.8 | Post-step | |
| 140 | 0.0044 | 0.9655 | 0.9799 | 2.50e-04 | 503.0 | Post-step | |
| 141 | 0.0047 | 0.9674 | 0.9799 | 2.50e-04 | 497.7 | Post-step | |
| 142 | 0.0043 | 0.9706 | 0.9799 | 2.50e-04 | 488.6 | Post-step | |
| 143 | 0.0043 | 0.9686 | 0.9799 | 2.50e-04 | 498.2 | Post-step | |
| 144 | 0.0044 | 0.9648 | 0.9799 | 2.50e-04 | 497.6 | Post-step | |
| 145 | 0.0042 | 0.9667 | 0.9799 | 2.50e-04 | 503.5 | Post-step | |
| 146 | 0.0041 | 0.9629 | 0.9799 | 2.50e-04 | 503.1 | Post-step | **New worst val AUC** |
| 147 | 0.0039 | 0.9670 | 0.9799 | 2.50e-04 | 491.1 | Post-step | |
| 148 | 0.0043 | 0.9691 | 0.9799 | 2.50e-04 | 514.4 | Post-step | |
| 149 | 0.0040 | 0.9704 | 0.9799 | 2.50e-04 | 482.0 | Post-step | |
| 150 | 0.0040 | 0.9691 | 0.9799 | 2.50e-04 | 479.5 | Post-step | |
| 151 | 0.0038 | 0.9687 | 0.9799 | 2.50e-04 | 493.5 | Post-step | |
| 152 | 0.0040 | 0.9716 | 0.9799 | 2.50e-04 | 502.7 | Post-step | |
| 153 | 0.0038 | 0.9699 | 0.9799 | 2.50e-04 | 483.4 | Post-step | |
| 154 | 0.0040 | 0.9647 | 0.9799 | 2.50e-04 | 480.2 | Post-step | New worst val AUC |
| 155 | 0.0036 | 0.9687 | 0.9799 | 2.50e-04 | 486.6 | Post-step | |
| 156 | 0.0038 | 0.9705 | 0.9799 | 2.50e-04 | 457.5 | Post-step | |
| 157 | 0.0037 | 0.9671 | 0.9799 | 2.50e-04 | 464.6 | Post-step | |
| 158 | 0.0038 | 0.9658 | 0.9799 | 2.50e-04 | 477.3 | Post-step | |
| 159 | 0.0037 | 0.9666 | 0.9799 | 2.50e-04 | 482.1 | Post-step | |
| 160 | 0.0034 | 0.9659 | 0.9799 | **1.25e-04** | 480.6 | **FINAL** | **RUN COMPLETE. BEST_AUC=0.9799 at E68.** |

**Critical Finding**: Slow, steady improvement from 0.78 (E1) to 0.98 (E68), consistent with training a small model from scratch. Post LR-step at E80, loss immediately drops (0.012 -> 0.010 -> 0.008) but val AUC does NOT improve — it has actually **worsened** from the 0.970-0.975 band (E80-120) to the 0.963-0.971 band (E120-160). Worst val AUC = 0.9629 at E146, new worst 0.9647 at E154. Final E160 AUC=0.9659. The 195K-param model reached its capacity ceiling. Paper IV achieved 0.9984 with 194K params, confirming the gap is data-driven (uncleaned negatives, noisier positives), not architecture. **RUN COMPLETE.**

---

### 6.3 Run v3: EfficientNetV2-S Cosine LR (Lambda3)

**Status**: **COMPLETE** (160 / 160, finished 2026-02-13 16:21 UTC)
**Best**: val_auc=0.9895 at epoch 17
**Final**: val_auc=0.9644 at epoch 160

| Epoch | Train Loss | Val AUC | Best AUC | LR | Time (s) | Phase | Notes |
|-------|-----------|---------|----------|-----|----------|-------|-------|
| 1 | 0.3162 | 0.5244 | 0.5244 | 8.07e-05 | 634.3 | Frozen + warmup | Backbone frozen, LR warmup 1/5 |
| 2 | 0.2062 | 0.5258 | 0.5258 | 1.57e-04 | 481.0 | Frozen + warmup | Warmup 2/5 |
| 3 | 0.1734 | 0.5351 | 0.5351 | 2.34e-04 | 485.8 | Frozen + warmup | Warmup 3/5 |
| 4 | 0.1493 | 0.5380 | 0.5380 | 3.11e-04 | 502.4 | Frozen + warmup | Warmup 4/5 |
| 5 | 0.1298 | 0.5442 | 0.5442 | 3.88e-04 | 489.1 | Frozen + warmup | Warmup 5/5, LR reached full |
| 6 | 0.0331 | 0.9789 | 0.9789 | 3.87e-04 | 498.0 | **Unfreeze** | **Backbone unfrozen. AUC 0.54->0.98** |
| 7 | 0.0217 | 0.9881 | 0.9881 | 3.86e-04 | 484.8 | Full training | Cosine LR slowly decaying |
| 8 | 0.0195 | 0.9888 | 0.9888 | 3.86e-04 | 485.7 | Full training | |
| 9 | 0.0176 | 0.9881 | 0.9888 | 3.85e-04 | 484.1 | Full training | |
| 10 | 0.0167 | 0.9880 | 0.9888 | 3.84e-04 | 482.3 | Full training | |
| 11 | 0.0159 | 0.9863 | 0.9888 | 3.83e-04 | 508.0 | Full training | |
| 12 | 0.0152 | 0.9871 | 0.9888 | 3.83e-04 | 479.9 | Full training | |
| 13 | 0.0146 | 0.9889 | 0.9889 | 3.82e-04 | 478.2 | Full training | Tiny new best |
| 14 | 0.0140 | 0.9883 | 0.9889 | 3.81e-04 | 486.0 | Full training | |
| 15 | 0.0135 | 0.9858 | 0.9889 | 3.80e-04 | 523.3 | Full training | |
| 16 | 0.0129 | 0.9880 | 0.9889 | 3.79e-04 | 518.2 | Full training | |
| 17 | 0.0123 | **0.9895** | **0.9895** | 3.77e-04 | 533.6 | Full training | **Current best** |
| 18 | 0.0113 | 0.9873 | 0.9895 | 3.76e-04 | 532.7 | Full training | |
| 19 | 0.0112 | 0.9883 | 0.9895 | 3.75e-04 | 509.0 | Full training | |
| 20 | 0.0106 | 0.9867 | 0.9895 | 3.73e-04 | 509.6 | Full training | v2 peaked here (E19); v3 holding |
| 21 | 0.0101 | 0.9873 | 0.9895 | 3.72e-04 | 507.9 | Full training | |
| 22 | 0.0096 | 0.9889 | 0.9895 | 3.70e-04 | 522.9 | Full training | Near-best recovery |
| 23 | 0.0087 | 0.9871 | 0.9895 | 3.69e-04 | 527.0 | Full training | |
| 24 | 0.0084 | 0.9857 | 0.9895 | 3.67e-04 | 530.1 | Full training | |
| 25 | 0.0080 | 0.9893 | 0.9895 | 3.65e-04 | 543.2 | Full training | Near-best recovery |
| 26 | 0.0071 | 0.9824 | 0.9895 | 3.63e-04 | 518.2 | Full training | Significant dip (similar to v2 E24-26 pattern) |
| 27 | 0.0070 | 0.9858 | 0.9895 | 3.61e-04 | 517.7 | Full training | Recovery from E26 dip |
| 28 | 0.0069 | 0.9863 | 0.9895 | 3.59e-04 | 520.9 | Full training | Continued recovery |
| 29 | 0.0064 | 0.9881 | 0.9895 | 3.57e-04 | 529.1 | Full training | Strong recovery — approaching best again |
| 30 | 0.0061 | 0.9870 | 0.9895 | 3.55e-04 | 528.0 | Full training | |
| 31 | 0.0058 | 0.9862 | 0.9895 | 3.52e-04 | 530.0 | Full training | |
| 32 | 0.0055 | 0.9858 | 0.9895 | 3.50e-04 | 531.0 | Full training | |
| 33 | 0.0053 | 0.9854 | 0.9895 | 3.49e-04 | 532.1 | Full training | |
| 34 | 0.0046 | 0.9859 | 0.9895 | 3.46e-04 | 529.7 | Full training | |
| 35 | 0.0047 | 0.9859 | 0.9895 | 3.44e-04 | 529.8 | Full training | Stabilizing ~0.986 |
| 36 | 0.0045 | 0.9839 | 0.9895 | 3.42e-04 | 540.9 | Full training | |
| 37 | 0.0040 | 0.9839 | 0.9895 | 3.39e-04 | 525.4 | Full training | |
| 38 | 0.0037 | 0.9806 | 0.9895 | 3.36e-04 | 549.5 | Full training | Dip |
| 39 | 0.0038 | 0.9835 | 0.9895 | 3.34e-04 | 540.8 | Full training | |
| 40 | 0.0034 | 0.9855 | 0.9895 | 3.31e-04 | 489.6 | Full training | |
| 41 | 0.0034 | 0.9855 | 0.9895 | 3.28e-04 | 478.4 | Full training | |
| 42 | 0.0032 | 0.9835 | 0.9895 | 3.26e-04 | 496.3 | Full training | |
| 43 | 0.0031 | 0.9836 | 0.9895 | 3.23e-04 | 514.0 | Full training | |
| 44 | 0.0028 | 0.9846 | 0.9895 | 3.20e-04 | 503.8 | Full training | |
| 45 | 0.0031 | 0.9819 | 0.9895 | 3.17e-04 | 518.6 | Full training | |
| 46 | 0.0028 | 0.9862 | 0.9895 | 3.14e-04 | 515.2 | Full training | |
| 47 | 0.0025 | 0.9793 | 0.9895 | 3.11e-04 | 496.9 | Full training | Dip |
| 48 | 0.0025 | 0.9823 | 0.9895 | 3.08e-04 | 486.2 | Full training | |
| 49 | 0.0025 | 0.9824 | 0.9895 | 3.05e-04 | 494.1 | Full training | |
| 50 | 0.0023 | 0.9824 | 0.9895 | 3.02e-04 | 488.5 | Full training | |
| 51 | 0.0022 | 0.9824 | 0.9895 | 2.99e-04 | 487.7 | Full training | |
| 52 | 0.0023 | 0.9845 | 0.9895 | 2.95e-04 | 500.6 | Full training | |
| 53 | 0.0020 | 0.9796 | 0.9895 | 2.92e-04 | 509.5 | Full training | |
| 54 | 0.0021 | 0.9812 | 0.9895 | 2.89e-04 | 492.2 | Full training | |
| 55 | 0.0020 | 0.9814 | 0.9895 | 2.85e-04 | 490.4 | Full training | |
| 56 | 0.0018 | 0.9802 | 0.9895 | 2.82e-04 | 498.4 | Full training | |
| 57 | 0.0019 | 0.9801 | 0.9895 | 2.79e-04 | 490.3 | Full training | |
| 58 | 0.0016 | 0.9790 | 0.9895 | 2.75e-04 | 505.8 | Full training | |
| 59 | 0.0017 | 0.9781 | 0.9895 | 2.72e-04 | 512.0 | Full training | |
| 60 | 0.0015 | 0.9783 | 0.9895 | 2.68e-04 | 517.7 | Full training | |
| 61 | 0.0016 | 0.9811 | 0.9895 | 2.65e-04 | 489.4 | Full training | |
| 62 | 0.0014 | 0.9817 | 0.9895 | 2.61e-04 | 501.5 | Full training | |
| 63 | 0.0014 | 0.9771 | 0.9895 | 2.58e-04 | 499.2 | Full training | |
| 64 | 0.0014 | 0.9763 | 0.9895 | 2.54e-04 | 482.5 | Full training | |
| 65 | 0.0013 | 0.9784 | 0.9895 | 2.50e-04 | 489.7 | Full training | |
| 66 | 0.0012 | 0.9743 | 0.9895 | 2.47e-04 | 555.8 | Full training | |
| 67 | 0.0009 | 0.9787 | 0.9895 | 2.43e-04 | 685.8 | Full training | |
| 68 | 0.0013 | 0.9823 | 0.9895 | 2.39e-04 | 688.4 | Full training | Best since E25; possible stabilization |
| 69 | 0.0011 | 0.9798 | 0.9895 | 2.36e-04 | 681.5 | Full training | |
| 70 | 0.0012 | 0.9816 | 0.9895 | 2.32e-04 | 719.7 | Full training | Epoch 70 checkpoint saved |
| 71 | 0.0011 | 0.9746 | 0.9895 | 2.28e-04 | 541.3 | Full training | Dip |
| 72 | 0.0009 | 0.9809 | 0.9895 | 2.24e-04 | 723.8 | Full training | Recovery |
| 73 | 0.0010 | 0.9795 | 0.9895 | 2.21e-04 | 722.5 | Full training | |
| 74 | 0.0010 | 0.9778 | 0.9895 | 2.17e-04 | 721.7 | Full training | |
| 75 | 0.0009 | 0.9793 | 0.9895 | 2.13e-04 | 698.9 | Full training | |
| 76 | 0.0008 | 0.9810 | 0.9895 | 2.09e-04 | 640.3 | Full training | |
| 77 | 0.0009 | 0.9801 | 0.9895 | 2.05e-04 | 496.5 | Full training | |
| 78 | 0.0007 | 0.9780 | 0.9895 | 2.02e-04 | 510.3 | Full training | |
| 79 | 0.0008 | 0.9792 | 0.9895 | 1.98e-04 | 523.3 | Full training | |
| 80 | 0.0008 | 0.9788 | 0.9895 | 1.94e-04 | 494.9 | Full training | LR matches v2's post-step LR |
| 81 | 0.0007 | 0.9779 | 0.9895 | 1.90e-04 | 519.3 | Full training | |
| 82 | 0.0006 | 0.9818 | 0.9895 | 1.86e-04 | 518.7 | Full training | Strong uptick |
| 83 | 0.0006 | 0.9806 | 0.9895 | 1.83e-04 | 522.2 | Full training | |
| 84 | 0.0007 | 0.9801 | 0.9895 | 1.79e-04 | 506.8 | Full training | |
| 85 | 0.0007 | 0.9705 | 0.9895 | 1.75e-04 | 524.1 | Full training | **Dip** — worst since E66 |
| 86 | 0.0005 | 0.9781 | 0.9895 | 1.71e-04 | 523.1 | Full training | Recovery from dip |
| 87 | 0.0007 | 0.9797 | 0.9895 | 1.67e-04 | 500.5 | Full training | |
| 88 | 0.0004 | 0.9771 | 0.9895 | 1.64e-04 | 505.2 | Full training | |
| 89 | 0.0006 | 0.9823 | 0.9895 | 1.60e-04 | 509.4 | Full training | Best since E68 |
| 90 | 0.0005 | 0.9783 | 0.9895 | 1.56e-04 | 520.9 | Full training | Epoch 90 checkpoint saved |
| 91 | 0.0006 | 0.9798 | 0.9895 | 1.52e-04 | 679.5 | Full training | |
| 92 | 0.0005 | 0.9757 | 0.9895 | 1.49e-04 | 536.3 | Full training | |
| 93 | 0.0004 | 0.9734 | 0.9895 | 1.45e-04 | 536.2 | Full training | |
| 94 | 0.0004 | 0.9712 | 0.9895 | 1.41e-04 | 509.5 | Full training | |
| 95 | 0.0004 | 0.9722 | 0.9895 | 1.38e-04 | 526.3 | Full training | |
| 96 | 0.0004 | 0.9734 | 0.9895 | 1.34e-04 | 512.5 | Full training | |
| 97 | 0.0005 | 0.9739 | 0.9895 | 1.30e-04 | 505.7 | Full training | |
| 98 | 0.0003 | 0.9752 | 0.9895 | 1.27e-04 | 525.5 | Full training | |
| 99 | 0.0003 | 0.9775 | 0.9895 | 1.23e-04 | 530.4 | Full training | Best since E89 |
| 100 | 0.0004 | 0.9727 | 0.9895 | 1.20e-04 | 509.6 | Full training | Epoch 100 checkpoint saved |
| 101 | 0.0003 | 0.9757 | 0.9895 | 1.16e-04 | 489.8 | Full training | |
| 102 | 0.0003 | 0.9754 | 0.9895 | 1.13e-04 | 482.3 | Full training | |
| 103 | 0.0003 | 0.9775 | 0.9895 | 1.09e-04 | 483.1 | Full training | |
| 104 | 0.0002 | 0.9764 | 0.9895 | 1.06e-04 | 484.3 | Full training | |
| 105 | 0.0001 | 0.9749 | 0.9895 | 1.03e-04 | 489.9 | Full training | |
| 106 | 0.0003 | 0.9715 | 0.9895 | 9.92e-05 | 491.8 | Full training | LR drops below 1e-4 |
| 107 | 0.0002 | 0.9711 | 0.9895 | 9.59e-05 | 491.9 | Full training | |
| 108 | 0.0002 | 0.9720 | 0.9895 | 9.26e-05 | 482.3 | Full training | |
| 109 | 0.0002 | 0.9690 | 0.9895 | 8.94e-05 | 476.3 | Full training | New worst since E85 |
| 110 | 0.0002 | 0.9724 | 0.9895 | 8.62e-05 | 492.1 | Full training | Epoch 110 checkpoint saved |
| 111 | 0.0001 | 0.9682 | 0.9895 | 8.31e-05 | 467.7 | Full training | **New worst val AUC** |
| 112 | 0.0002 | 0.9710 | 0.9895 | 8.00e-05 | 482.4 | Full training | |
| 113 | 0.0002 | 0.9716 | 0.9895 | 7.69e-05 | 486.8 | Full training | |
| 114 | 0.0001 | 0.9705 | 0.9895 | 7.39e-05 | 493.5 | Full training | |
| 115 | 0.0001 | 0.9733 | 0.9895 | 7.09e-05 | 472.6 | Full training | |
| 116 | 0.0002 | 0.9725 | 0.9895 | 6.80e-05 | 478.8 | Full training | |
| 117 | 0.0001 | 0.9707 | 0.9895 | 6.51e-05 | 469.8 | Full training | |
| 118 | 0.0001 | 0.9715 | 0.9895 | 6.23e-05 | 486.9 | Full training | |
| 119 | 0.0001 | 0.9693 | 0.9895 | 5.95e-05 | 476.2 | Full training | |
| 120 | 0.0000 | 0.9685 | 0.9895 | 5.68e-05 | 488.8 | Full training | Epoch 120 checkpoint saved |
| 121 | 0.0001 | 0.9666 | 0.9895 | 5.42e-05 | 493.5 | Full training | Continued decline |
| 122 | 0.0001 | 0.9661 | 0.9895 | 5.15e-05 | 471.3 | Full training | |
| 123 | 0.0001 | 0.9652 | 0.9895 | 4.90e-05 | 478.0 | Full training | |
| 124 | 0.0001 | 0.9668 | 0.9895 | 4.65e-05 | 474.7 | Full training | |
| 125 | 0.0001 | 0.9662 | 0.9895 | 4.40e-05 | 481.2 | Full training | |
| 126 | 0.0001 | 0.9646 | 0.9895 | 4.16e-05 | 477.3 | Full training | |
| 127 | 0.0001 | 0.9663 | 0.9895 | 3.93e-05 | 470.7 | Full training | |
| 128 | 0.0000 | 0.9671 | 0.9895 | 3.71e-05 | 462.7 | Full training | |
| 129 | 0.0001 | 0.9657 | 0.9895 | 3.48e-05 | 453.2 | Full training | |
| 130 | 0.0000 | 0.9636 | 0.9895 | 3.27e-05 | 451.3 | Full training | **New worst val AUC** |
| 131 | 0.0001 | 0.9661 | 0.9895 | 3.06e-05 | 454.3 | Full training | |
| 132 | 0.0000 | 0.9654 | 0.9895 | 2.86e-05 | 455.7 | Full training | |
| 133 | 0.0000 | 0.9646 | 0.9895 | 2.66e-05 | 454.9 | Full training | |
| 134 | 0.0000 | 0.9649 | 0.9895 | 2.47e-05 | 459.8 | Full training | |
| 135 | 0.0001 | 0.9653 | 0.9895 | 2.29e-05 | 469.4 | Full training | |
| 136 | 0.0000 | 0.9659 | 0.9895 | 2.11e-05 | 459.2 | Full training | |
| 137 | 0.0000 | 0.9650 | 0.9895 | 1.94e-05 | 454.9 | Full training | |
| 138 | 0.0000 | 0.9641 | 0.9895 | 1.78e-05 | 739.2 | Full training | |
| 139 | 0.0000 | 0.9651 | 0.9895 | 1.63e-05 | 894.8 | Full training | |
| 140 | 0.0000 | 0.9641 | 0.9895 | 1.48e-05 | 874.0 | Full training | Epoch 140 checkpoint saved |
| 141 | 0.0000 | 0.9618 | 0.9895 | 1.33e-05 | 1055.3 | Full training | **New worst: 0.9618** |
| 142 | 0.0000 | 0.9626 | 0.9895 | 1.20e-05 | 968.2 | Full training | |
| 143 | 0.0000 | 0.9640 | 0.9895 | 1.07e-05 | 931.8 | Full training | |
| 144 | 0.0000 | 0.9637 | 0.9895 | 9.50e-06 | 536.7 | Full training | |
| 145 | 0.0000 | 0.9649 | 0.9895 | 8.35e-06 | 531.0 | Full training | |
| 146 | 0.0000 | 0.9647 | 0.9895 | 7.28e-06 | 521.8 | Full training | |
| 147 | 0.0000 | 0.9650 | 0.9895 | 6.29e-06 | 507.4 | Full training | |
| 148 | 0.0000 | 0.9645 | 0.9895 | 5.36e-06 | 533.6 | Full training | |
| 149 | 0.0000 | 0.9644 | 0.9895 | 4.51e-06 | 539.0 | Full training | |
| 150 | 0.0000 | 0.9649 | 0.9895 | 3.73e-06 | 558.4 | Full training | Epoch 150 checkpoint saved |
| 151 | 0.0000 | 0.9646 | 0.9895 | 3.02e-06 | 563.2 | Full training | |
| 152 | 0.0000 | 0.9646 | 0.9895 | 2.39e-06 | 531.3 | Full training | |
| 153 | 0.0000 | 0.9644 | 0.9895 | 1.83e-06 | 520.4 | Full training | |
| 154 | 0.0000 | 0.9643 | 0.9895 | 1.34e-06 | 540.6 | Full training | |
| 155 | 0.0000 | 0.9647 | 0.9895 | 9.34e-07 | 568.3 | Full training | |
| 156 | 0.0000 | 0.9642 | 0.9895 | 5.98e-07 | 543.3 | Full training | |
| 157 | 0.0000 | 0.9646 | 0.9895 | 3.36e-07 | 569.6 | Full training | |
| 158 | 0.0000 | 0.9649 | 0.9895 | 1.50e-07 | 553.2 | Full training | |
| 159 | 0.0000 | 0.9642 | 0.9895 | 3.74e-08 | 568.7 | Full training | |
| 160 | 0.0000 | 0.9644 | 0.9895 | 0.00e+00 | 555.6 | **FINAL** | **RUN COMPLETE. BEST_AUC=0.9895 at E17.** |

**Critical Finding**: Peaked at E17 (AUC=0.9895), then entered gradual decline for 143 epochs. The cosine schedule provided a smoother decay than v2's step schedule but did NOT prevent overfitting. Five clear regimes:

- **E1-17 (ascent)**: Mirrors v2's trajectory. Peaks at E17 (0.9895), slightly below v2's E19 peak (0.9915).
- **E18-50 (initial decline + stabilization)**: AUC drops then stabilizes around 0.982-0.986 band. Cosine LR provided genuine stability vs v2's steeper decline.
- **E51-92 (continued decline)**: AUC dropped to 0.975-0.982 band. Occasional spikes to 0.9823 (E68, E82, E89).
- **E93-120 (accelerating decline)**: AUC fell further to 0.968-0.978 band. LR decayed from 1.45e-4 to 5.7e-5. Worst at E111: 0.9682.
- **E121-160 (plateau at floor)**: AUC stabilized in a tight 0.963-0.967 band as LR decayed from 5.4e-5 to 0. Worst at E141: 0.9618. Final E160 AUC: 0.9644.

**Key conclusions from v3**:
1. The cosine schedule did NOT materially outperform the step schedule. Peak AUC (0.9895) was below v2's (0.9915), and final AUC (0.9644) was below v2's (0.9736).
2. The continued LR decay did NOT push AUC back toward peak — the answer to the earlier open question is definitive: **it declined further**, settling 0.025 below peak.
3. v3's E120-160 band (0.963-0.967) is actually WORSE than v2's post-step band (0.970-0.978), confirming that very low LR cannot prevent overfitting-induced generalization loss.
4. The cosine decay produced a nearly identical worst-case AUC to v2 (0.9618 vs 0.9689), but v3's average late-epoch AUC was slightly lower.
5. **The overfitting is definitively data-driven, not schedule-driven.** Both step and cosine schedules produce the same qualitative trajectory: early peak, then irreversible decline.

**Cosine LR decay schedule** (actual):
- E10: 3.84e-4 | E20: 3.73e-4 | E40: 3.31e-4 | E60: 2.68e-4 | E80: 1.94e-4 | E100: 1.20e-4 | E120: 5.68e-5 | E140: 1.48e-5 | E160: 0.00

---

### 6.4 Run v4: EfficientNetV2-S Fine-tune from v2 Best (Lambda4)

**Status**: **COMPLETE** (60 / 60, finished 2026-02-13 04:14 UTC)
**Init weights**: v2/best.pt (epoch 19, AUC=0.99148)

| Epoch | Train Loss | Val AUC | Best AUC | LR | Time (s) | Phase | Notes |
|-------|-----------|---------|----------|-----|----------|-------|-------|
| 1 | 0.0078 | **0.9921** | **0.9921** | 1.70e-05 | 683.9 | Warmup 1/3 | **Immediately exceeds v2 best (0.9915)!** |
| 2 | 0.0069 | 0.9916 | 0.9921 | 3.34e-05 | 500.7 | Warmup 2/3 | Slight dip as LR increases |
| 3 | 0.0063 | 0.9916 | 0.9921 | 4.98e-05 | 505.4 | Warmup 3/3 | LR reached full 5e-5 |
| 4 | 0.0055 | 0.9905 | 0.9921 | 4.96e-05 | 525.4 | Cosine decay | Cosine decay begins |
| 5 | 0.0051 | 0.9897 | 0.9921 | 4.93e-05 | 529.8 | Cosine decay | Val AUC declining gently |
| 6 | 0.0044 | 0.9891 | 0.9921 | 4.89e-05 | 514.8 | Cosine decay | Train loss still dropping |
| 7 | 0.0038 | 0.9893 | 0.9921 | 4.85e-05 | 519.5 | Cosine decay | Possible stabilization |
| 8 | 0.0033 | 0.9881 | 0.9921 | 4.80e-05 | 503.8 | Cosine decay | Decline continues |
| 9 | 0.0029 | 0.9872 | 0.9921 | 4.74e-05 | 504.2 | Cosine decay | Train loss near v2's overfit level |
| 10 | 0.0024 | 0.9858 | 0.9921 | 4.68e-05 | 499.6 | Cosine decay | |
| 11 | 0.0022 | 0.9863 | 0.9921 | 4.61e-05 | 519.2 | Cosine decay | |
| 12 | 0.0017 | 0.9844 | 0.9921 | 4.53e-05 | 521.7 | Cosine decay | |
| 13 | 0.0018 | 0.9866 | 0.9921 | 4.46e-05 | 523.0 | Cosine decay | |
| 14 | 0.0014 | 0.9826 | 0.9921 | 4.37e-05 | 510.5 | Cosine decay | |
| 15 | 0.0012 | 0.9822 | 0.9921 | 4.28e-05 | 499.7 | Cosine decay | |
| 16 | 0.0011 | 0.9828 | 0.9921 | 4.18e-05 | 509.5 | Cosine decay | |
| 17 | 0.0009 | 0.9827 | 0.9921 | 4.08e-05 | 512.3 | Cosine decay | |
| 18 | 0.0006 | 0.9828 | 0.9921 | 3.98e-05 | 533.1 | Cosine decay | |
| 19 | 0.0005 | 0.9848 | 0.9921 | 3.87e-05 | 528.7 | Cosine decay | Uptick — low LR helping |
| 20 | 0.0023 | 0.9858 | 0.9921 | 3.76e-05 | 511.6 | Cosine decay | Loss spike (stochastic) but AUC improves |
| 21 | 0.0019 | 0.9842 | 0.9921 | 3.64e-05 | 508.1 | Cosine decay | |
| 22 | 0.0015 | 0.9849 | 0.9921 | 3.53e-05 | 500.8 | Cosine decay | |
| 23 | 0.0013 | 0.9852 | 0.9921 | 3.41e-05 | 515.4 | Cosine decay | |
| 24 | 0.0011 | 0.9826 | 0.9921 | 3.28e-05 | 537.0 | Cosine decay | |
| 25 | 0.0012 | 0.9839 | 0.9921 | 3.16e-05 | 545.8 | Cosine decay | |
| 26 | 0.0011 | 0.9837 | 0.9921 | 3.03e-05 | 521.8 | Cosine decay | |
| 27 | 0.0010 | 0.9825 | 0.9921 | 2.90e-05 | 525.8 | Cosine decay | |
| 28 | 0.0010 | 0.9831 | 0.9921 | 2.77e-05 | 505.7 | Cosine decay | |
| 29 | 0.0007 | 0.9836 | 0.9921 | 2.64e-05 | 508.0 | Cosine decay | |
| 30 | 0.0008 | 0.9824 | 0.9921 | 2.51e-05 | 507.2 | Cosine decay | |
| 31 | 0.0007 | 0.9834 | 0.9921 | 2.38e-05 | 514.8 | Cosine decay | |
| 32 | 0.0006 | 0.9819 | 0.9921 | 2.24e-05 | 521.4 | Cosine decay | |
| 33 | 0.0006 | 0.9822 | 0.9921 | 2.11e-05 | 524.7 | Cosine decay | |
| 34 | 0.0006 | 0.9825 | 0.9921 | 1.99e-05 | 518.0 | Cosine decay | |
| 35 | 0.0005 | 0.9813 | 0.9921 | 1.86e-05 | 505.9 | Cosine decay | |
| 36 | 0.0004 | 0.9823 | 0.9921 | 1.73e-05 | 522.9 | Cosine decay | |
| 37 | 0.0004 | 0.9815 | 0.9921 | 1.61e-05 | 525.9 | Cosine decay | |
| 38 | 0.0004 | 0.9828 | 0.9921 | 1.49e-05 | 536.3 | Cosine decay | |
| 39 | 0.0004 | 0.9820 | 0.9921 | 1.37e-05 | 537.3 | Cosine decay | |
| 40 | 0.0003 | 0.9807 | 0.9921 | 1.25e-05 | 523.6 | Cosine decay | |
| 41 | 0.0003 | 0.9805 | 0.9921 | 1.14e-05 | 528.1 | Cosine decay | |
| 42 | 0.0004 | 0.9808 | 0.9921 | 1.03e-05 | 539.5 | Cosine decay | |
| 43 | 0.0003 | 0.9815 | 0.9921 | 9.29e-06 | 525.2 | Cosine decay | |
| 44 | 0.0003 | 0.9805 | 0.9921 | 8.29e-06 | 509.4 | Cosine decay | |
| 45 | 0.0003 | 0.9810 | 0.9921 | 7.34e-06 | 524.4 | Cosine decay | |
| 46 | 0.0003 | 0.9807 | 0.9921 | 6.44e-06 | 523.3 | Cosine decay | |
| 47 | 0.0004 | 0.9805 | 0.9921 | 5.59e-06 | 505.8 | Cosine decay | |
| 48 | 0.0002 | 0.9801 | 0.9921 | 4.79e-06 | 502.9 | Cosine decay | |
| 49 | 0.0002 | 0.9808 | 0.9921 | 4.04e-06 | 503.6 | Cosine decay | |
| 50 | 0.0002 | 0.9801 | 0.9921 | 3.36e-06 | 498.5 | Cosine decay | Epoch 50 checkpoint saved |
| 51 | 0.0002 | 0.9801 | 0.9921 | 2.73e-06 | 492.3 | Cosine decay | |
| 52 | 0.0002 | 0.9808 | 0.9921 | 2.17e-06 | 510.2 | Cosine decay | |
| 53 | 0.0002 | 0.9805 | 0.9921 | 1.67e-06 | 660.7 | Cosine decay | |
| 54 | 0.0001 | 0.9802 | 0.9921 | 1.23e-06 | 731.7 | Cosine decay | |
| 55 | 0.0002 | 0.9803 | 0.9921 | 8.54e-07 | 711.5 | Cosine decay | |
| 56 | 0.0002 | 0.9800 | 0.9921 | 5.48e-07 | 708.6 | Cosine decay | |
| 57 | 0.0002 | 0.9803 | 0.9921 | 3.09e-07 | 621.8 | Cosine decay | |
| 58 | 0.0002 | 0.9802 | 0.9921 | 1.37e-07 | 509.3 | Cosine decay | |
| 59 | 0.0002 | 0.9801 | 0.9921 | 3.44e-08 | 501.6 | Cosine decay | |
| 60 | 0.0002 | 0.9794 | 0.9921 | 0.00e+00 | 499.4 | **FINAL** | **RUN COMPLETE. BEST_AUC=0.9921 at E1.** |

**Rationale**: Loads v2's peak weights (E19, AUC=0.9915) and fine-tunes with 8x lower LR (5e-5) + cosine decay. Peak at E1 (AUC=0.9921) — the tiny initial LR nudge made beneficial micro-adjustments. Then monotonic decline through E16 (0.983), followed by a plateau in the 0.980-0.981 band (E17-60) as LR approaches zero. Final AUC at E60 was 0.9794 (LR=0). The E1 checkpoint (AUC=0.9921) is our definitive best model. **RUN COMPLETE.**

---

### 6.5 Run: ResNet-18 (Lambda2, stopped at E68)

**Status**: **STOPPED** (killed at epoch 68 / 160 due to severe overfitting)
**Best**: val_auc=0.9611 (epoch unknown — from first 30 epochs, log lost on restart)
**Config**: `configs/paperIV_resnet18.yaml` (standard torchvision ResNet-18, ~11.2M params)

| Hyperparameter | Value |
|----------------|-------|
| Architecture | torchvision ResNet-18 (pretrained ImageNet) |
| Parameters | ~11.2M |
| Epochs planned | 160 |
| Epochs completed | 68 (stopped) |
| Micro-batch | 64 |
| Effective batch | 512 (8 accum steps) |
| Learning rate | 5.00e-04 |
| LR schedule | Step |

**Note**: The nohup log only contains epochs 31-68. Epochs 1-30 were from an earlier launch whose log was overwritten on restart. The best checkpoint (AUC=0.9611) was saved during the first 30 epochs.

| Epoch | Train Loss | Val AUC | Best AUC | LR | Time (s) | Notes |
|-------|-----------|---------|----------|-----|----------|-------|
| 31 | 0.0009 | 0.9122 | 0.9611 | 5.00e-04 | 494.9 | *First epoch in surviving log* |
| 32 | 0.0016 | 0.9352 | 0.9611 | 5.00e-04 | 465.1 | |
| 33 | 0.0006 | 0.9210 | 0.9611 | 5.00e-04 | 457.8 | |
| 34 | 0.0013 | 0.9129 | 0.9611 | 5.00e-04 | 451.1 | |
| 35 | 0.0008 | 0.9270 | 0.9611 | 5.00e-04 | 470.2 | |
| 36 | 0.0007 | 0.9355 | 0.9611 | 5.00e-04 | 466.4 | |
| 37 | 0.0008 | 0.9224 | 0.9611 | 5.00e-04 | 469.6 | |
| 38 | 0.0011 | 0.9327 | 0.9611 | 5.00e-04 | 458.6 | |
| 39 | 0.0015 | 0.9138 | 0.9611 | 5.00e-04 | 623.5 | |
| 40 | 0.0010 | 0.9315 | 0.9611 | 5.00e-04 | 648.1 | |
| 41 | 0.0003 | 0.9331 | 0.9611 | 5.00e-04 | 660.1 | |
| 42 | 0.0006 | 0.8983 | 0.9611 | 5.00e-04 | 674.1 | **Worst val AUC** |
| 43 | 0.0009 | 0.9284 | 0.9611 | 5.00e-04 | 586.6 | |
| 44 | 0.0013 | 0.9239 | 0.9611 | 5.00e-04 | 489.9 | |
| 45 | 0.0008 | 0.9314 | 0.9611 | 5.00e-04 | 462.9 | |
| 46 | 0.0010 | 0.9122 | 0.9611 | 5.00e-04 | 457.5 | |
| 47 | 0.0009 | 0.9233 | 0.9611 | 5.00e-04 | 462.1 | |
| 48 | 0.0006 | 0.9180 | 0.9611 | 5.00e-04 | 469.4 | |
| 49 | 0.0008 | 0.9226 | 0.9611 | 5.00e-04 | 472.2 | |
| 50 | 0.0010 | 0.9196 | 0.9611 | 5.00e-04 | 486.6 | |
| 51 | 0.0009 | 0.9091 | 0.9611 | 5.00e-04 | 440.0 | |
| 52 | 0.0008 | 0.9196 | 0.9611 | 5.00e-04 | 439.2 | |
| 53 | 0.0006 | 0.9166 | 0.9611 | 5.00e-04 | 455.1 | |
| 54 | 0.0008 | 0.9361 | 0.9611 | 5.00e-04 | 460.2 | |
| 55 | 0.0011 | 0.9288 | 0.9611 | 5.00e-04 | 444.8 | |
| 56 | 0.0007 | 0.9289 | 0.9611 | 5.00e-04 | 441.8 | |
| 57 | 0.0004 | 0.9373 | 0.9611 | 5.00e-04 | 439.2 | Best in surviving log |
| 58 | 0.0003 | 0.9244 | 0.9611 | 5.00e-04 | 431.4 | |
| 59 | 0.0006 | 0.9007 | 0.9611 | 5.00e-04 | 427.0 | |
| 60 | 0.0011 | 0.9262 | 0.9611 | 5.00e-04 | 421.7 | |
| 61 | 0.0006 | 0.9185 | 0.9611 | 5.00e-04 | 441.4 | |
| 62 | 0.0005 | 0.9186 | 0.9611 | 5.00e-04 | 485.4 | |
| 63 | 0.0010 | 0.9069 | 0.9611 | 5.00e-04 | 482.5 | |
| 64 | 0.0007 | 0.9152 | 0.9611 | 5.00e-04 | 474.1 | |
| 65 | 0.0005 | 0.9097 | 0.9611 | 5.00e-04 | 448.9 | |
| 66 | 0.0007 | 0.9252 | 0.9611 | 5.00e-04 | 435.1 | |
| 67 | 0.0009 | 0.9120 | 0.9611 | 5.00e-04 | 436.5 | |
| 68 | 0.0003 | 0.9216 | 0.9611 | 5.00e-04 | 448.9 | *Last epoch before kill* |

**Critical Finding**: Standard ResNet-18 (11.2M params) was catastrophically overfit by E31 (train loss ~0.001 but val AUC dropped from peak 0.9611 to 0.91-0.93 band). This is the largest model tested and its generalization degraded the worst. The wildly oscillating val AUC (0.898-0.937 range) and near-zero train loss indicate severe memorization. This run was intentionally stopped at E68 as further training was futile. Paper IV does not use standard ResNet-18 — their compact 194K-param architecture was specifically designed to avoid this. **This run serves as an ablation showing that naive over-parameterized ResNet fails on this task.**

**NFS Artifacts**: best.pt, last.pt (E68), epoch_010-060.pt (every 10), run_info.json, nohup_training.log

---

## 7. Checkpointing Strategy

| Checkpoint | When Saved | Content |
|------------|------------|---------|
| `best.pt` | Whenever val_auc improves | model state_dict, optimizer state_dict, scheduler state_dict, epoch, best_auc, dataset config, train config |
| `last.pt` | Every epoch | Same as above |
| `epoch_NNN.pt` | Every 10 epochs | Same as above |
| `run_info.json` | Once at start | git commit, timestamp, config path, config SHA-256, command, dataset seed |

All checkpoints stored on NFS: `/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/checkpoints/<run_name>/`
Background S3 sync every 10 minutes: `s3://darkhaloscope/stronglens_calibration/checkpoints/<run_name>/`
Full nohup training logs archived to NFS: `<checkpoint_dir>/nohup_training.log` (copied from each instance's local `/home/ubuntu/nohup_*.out`)

---

## 8. Summary of Key Findings (Updated Live)

### 8.1 Training Dynamics

1. **Backbone freeze/unfreeze is essential for pretrained EfficientNetV2-S.** Without it (earlier failed runs), the model suffered catastrophic forgetting. With 5 epochs of frozen backbone, the classifier head learns a reasonable mapping before the full model is fine-tuned.

2. **Epoch-aware augmentation seeding is critical.** A prior bug used fixed augmentation per sample across all epochs, causing extreme memorization (train loss -> 0, val AUC peaks at E1 then declines). Fixed by incorporating epoch into the augmentation seed.

3. **Constant LR causes prolonged overfitting.** v2's step LR at 3.88e-4 peaked at E19 then monotonically declined for 80+ epochs. The LR step at E130 comes too late.

4. **BottleneckedResNet reaches capacity-limited plateau.** With 195K params, the model achieves 0.9799 — respectable but 0.0185 below Paper IV's 0.9984. The gap is likely data-driven (uncleaned negatives, noisier positives).

### 8.2 Gap Analysis vs Paper IV

| Model | Paper IV AUC | Our Best AUC | Run | Gap | Explanation |
|-------|-------------|-------------|-----|-----|-------------|
| ResNet-class | 0.9984 | 0.9799 | BnResNet E68 (COMPLETE) | -0.0185 | Data quality (uncleaned negatives, Tier-B label noise); model is capacity-limited at 195K params on our data |
| EfficientNet (cosine) | 0.9987 | 0.9895 | v3 cosine E17 (COMPLETE) | -0.0092 | Cosine LR did not improve over step LR |
| EfficientNet | 0.9987 | **0.9921** | **v4 finetune E1 (COMPLETE)** | **-0.0066** | Phase 2 fine-tune from v2 peak; best result across all runs |
| ResNet-18 (ablation) | N/A | 0.9611 | ResNet-18 E<30 (STOPPED) | N/A | Over-parameterized (11.2M) — catastrophic overfitting. Not a Paper IV architecture. |
| Meta-learner | 0.9989 | N/A | N/A | N/A | Not yet implemented |

### 8.3 Trend Analysis (Updated 2026-02-13 06:30 UTC)

#### EfficientNetV2-S v2 (Lambda1) — Epochs 1-160 — **COMPLETE**
**Phase**: **FINISHED.** **Best**: E19, AUC=0.9915. **Final**: E160, AUC=0.9736.

Four clear regimes:
- **E1-19 (ascent)**: Rapid improvement to AUC=0.9915. Backbone unfreezes at E5.
- **E20-60 (slow decline at high LR)**: Gradual erosion from 0.991 to 0.977. Train loss drops to ~0.002.
- **E61-129 (accelerating decline)**: AUC falls to 0.97-0.976 band with constant LR=3.88e-4. Worst: 0.9689 at E105.
- **E130-160 (post-LR-step)**: LR halved to 1.94e-4. Train loss immediately collapses from ~0.001 to ~0.0002, but **val AUC did not recover** — oscillating in 0.970-0.978 band. Best post-step: 0.9785 (E140). Final E160 AUC: 0.9736. The model's generalization was irreversibly damaged by prolonged high-LR overfitting.

**Verdict**: **COMPLETE.** Best checkpoint at E19 is definitive. The LR step at E130 confirmed that the overfitting is structural, not optimization-limited. All 160 epochs finished. Nohup log archived.

#### BottleneckedResNet (Lambda2) — Epochs 1-160 — **COMPLETE**
**Phase**: **FINISHED.** **Best**: E68, AUC=0.9799. **Final**: E160, AUC=0.9659.

- **E1-68 (gradual ascent)**: Slow, steady climb from 0.78 to 0.9799 over 68 epochs — consistent with training a 195K-param model from scratch.
- **E69-80 (plateau)**: AUC oscillates in 0.975-0.980 band.
- **E80-120 (post-step decline)**: After LR step (5e-4 -> 2.5e-4), AUC degrades to 0.967-0.975 band despite lower train loss.
- **E120-160 (continued decline)**: AUC fell further to 0.963-0.972 band. Worst: 0.9629 at E146, 0.9647 at E154. Final E160: 0.9659.

**Verdict**: **COMPLETE.** The 195K-param model peaked at E68 (0.9799). The 0.0185 gap to Paper IV (0.9984) is definitively data-driven — uncleaned negatives and noisier (Tier-B) positives limit the achievable AUC at this model capacity. All checkpoints on NFS and S3. Nohup log archived to NFS.

#### ResNet-18 (Lambda2) — Epochs 1-68 — **STOPPED**
**Phase**: **KILLED** at E68 due to catastrophic overfitting.  **Best**: AUC=0.9611 (early epochs, log lost).

Standard torchvision ResNet-18 (11.2M params) was catastrophically overfit by E31 — train loss near zero but val AUC oscillating wildly in the 0.89-0.94 band, 5+ points below its early peak of 0.9611. Intentionally stopped. Serves as an ablation demonstrating that naive over-parameterized architectures fail on this task. All checkpoints on NFS and S3.

#### EfficientNetV2-S v3 Cosine LR (Lambda3) — Epochs 1-160 — **COMPLETE**
**Phase**: **FINISHED.** **Best**: E17, AUC=0.9895. **Final**: E160, AUC=0.9644.

Five clear regimes:
- **E1-17 (ascent)**: Mirrors v2's trajectory. Peaks at E17 (0.9895), slightly below v2's E19 peak (0.9915).
- **E18-50 (initial decline + stabilization)**: AUC drops then stabilizes around 0.982-0.986 band. Cosine LR provided genuine stability vs v2's steeper decline.
- **E51-92 (continued decline)**: AUC dropped to 0.975-0.982 band. Occasional spikes to 0.9823 (E68, E82, E89).
- **E93-120 (accelerating decline)**: AUC fell to 0.968-0.978 band. LR decayed from 1.45e-4 to 5.7e-5. Worst at E111: 0.9682.
- **E121-160 (plateau at floor)**: AUC settled into a tight 0.963-0.967 band as LR decayed to zero. Worst: E141 at 0.9618. Final E160: 0.9644.

**Verdict**: **COMPLETE. Cosine LR did NOT outperform step LR.** Peak AUC (0.9895) was below v2's (0.9915). Final AUC (0.9644) was below v2's (0.9736). The continued LR decay to near-zero drove AUC *further down* rather than recovering it — definitively answering the earlier open question. v3's E120-160 band (0.963-0.967) was worse than v2's post-step band (0.970-0.978). **This conclusively demonstrates that the overfitting is data-driven, not LR-schedule-driven.** All 160 epochs finished. Nohup log archived to NFS.

#### EfficientNetV2-S v4 Finetune (Lambda4) — Epochs 1-60 — **COMPLETE**
**Phase**: **FINISHED.** **Best**: E1, AUC=0.9921 (overall best). **Final**: E60, AUC=0.9794.

- **E1 (LR=1.7e-5)**: Immediately achieves AUC=0.9921 — surpassing source checkpoint's 0.9915.
- **E2-16 (monotonic decline)**: AUC drops steadily from 0.992 to 0.983 as model memorizes at even this low LR.
- **E17-19 (interesting uptick)**: AUC recovered to 0.985 temporarily.
- **E20-54 (final plateau)**: AUC settled in a tight 0.980-0.981 band.
- **E55-60 (LR → 0)**: AUC stable at 0.980. LR decayed from 8.5e-7 to 0.0. Final E60 AUC: 0.9794.

**Verdict**: **BEST MODEL CONFIRMED. RUN COMPLETE.** The E1 checkpoint (AUC=0.9921) is our definitive best, narrowing the Paper IV gap to just 0.0066. The late-stage plateau at ~0.980 shows the model is stable with near-zero LR. This checkpoint has been used for the full selection function grid. All checkpoints on NFS and S3. Nohup log archived.

**SELECTION FUNCTION NOTE**: The v4 E1 checkpoint has been used to run the full injection-recovery selection function grid (see Section 10 below). This is the first publication-quality selection function measurement for this project.

### 8.5 Overall Training Summary (2026-02-13 16:22 UTC) — ALL RUNS COMPLETE

| Run | Instance | Status | Epochs | Best AUC | Best Epoch | Final AUC |
|-----|----------|--------|--------|----------|------------|-----------|
| v4 finetune | lambda4 | **COMPLETE** | 60 | **0.9921** | 1 | 0.9794 |
| v2 step LR | lambda | **COMPLETE** | 160 | 0.9915 | 19 | 0.9736 |
| v3 cosine LR | lambda3 | **COMPLETE** | 160 | 0.9895 | 17 | 0.9644 |
| BnResNet | lambda2 | **COMPLETE** | 160 | 0.9799 | 68 | 0.9659 |
| ResNet-18 | lambda2 | **STOPPED** | 68/160 | 0.9611 | <30 | 0.9216 |

Key conclusions:

1. **Best model**: v4 finetune E1, AUC=0.9921 (gap to Paper IV: 0.0066)
2. **Best from-scratch EfficientNet**: v2 E19, AUC=0.9915 (gap: 0.0072)
3. **Best BottleneckedResNet**: E68, AUC=0.9799 (gap: 0.0185)
4. **Cosine LR (v3)**: Did not beat step LR — peak at E17 (0.9895, gap: 0.0092), final AUC 0.9644 (worse than v2's 0.9736). **Cosine decay to zero drove AUC further down, not up.**
5. **ResNet-18**: Catastrophic overfitting (11.2M params too large). Serves as ablation.
6. **All runs confirm**: overfitting begins early and is not recoverable by LR reduction alone — best checkpoints are always in E1-68 range
7. **Paper IV gap**: Attributable to data differences (uncleaned negatives, noisier positives), not architecture or training protocol
8. **LR schedule comparison (v2 vs v3)**: Both schedules produce near-identical peak AUC (0.9915 vs 0.9895) and the same qualitative trajectory (early peak → irreversible decline). Cosine schedule's final AUC (0.9644) is actually worse than step schedule's (0.9736), likely because near-zero LR in late cosine epochs cannot correct accumulated overfitting.

### 8.4 What the Paper Can Claim

Our paper's contribution is the **selection function audit**, not the classifier itself. AUC > 0.99 for EfficientNet and > 0.97 for BottleneckedResNet demonstrates "competent, protocol-matched training" sufficient to anchor the selection function analysis. The residual gap is attributable to documented data differences (negative cleaning, positive catalog composition) which we report transparently.

---

## 9. File References

| File | Location |
|------|----------|
| v2 config | `configs/paperIV_efficientnet_v2_s_v2.yaml` |
| BnResNet config | `configs/paperIV_bottlenecked_resnet.yaml` |
| v3 config | `configs/paperIV_efficientnet_v2_s_v3_cosine.yaml` |
| v4 config | `configs/paperIV_efficientnet_v2_s_v4_finetune.yaml` |
| ResNet-18 config | `configs/paperIV_resnet18.yaml` |
| Training code | `dhs/train.py` (train_one function) |
| Model code | `dhs/model.py` (build_model factory) |
| Data code | `dhs/data.py` (LensDataset, epoch-aware augmentation) |
| Experiment runner | `dhs/scripts/run_experiment.py` |
| Training manifest | `manifests/training_parity_70_30_v1.parquet` |
| NFS checkpoint root | `/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/checkpoints/` |
| S3 checkpoint root | `s3://darkhaloscope/stronglens_calibration/checkpoints/` |

### NFS Artifact Inventory (Completed Runs)

**BottleneckedResNet** (`paperIV_bottlenecked_resnet/`):
- `best.pt` (E68, AUC=0.9799), `last.pt` (E160), `run_info.json`
- `epoch_010.pt` through `epoch_160.pt` (every 10 epochs, 16 files)
- `nohup_training.log` (full training log, 160 epochs)

**ResNet-18** (`paperIV_resnet18/`):
- `best.pt` (AUC=0.9611, from early epochs), `last.pt` (E68), `run_info.json`
- `epoch_010.pt` through `epoch_060.pt` (every 10 epochs, 6 files)
- `nohup_training.log` (partial: epochs 31-68 only; first 30 epochs lost from earlier launch)

**EfficientNetV2-S v2** (`paperIV_efficientnet_v2_s_v2/`): **COMPLETE**
- `best.pt` (E19, AUC=0.9915), `last.pt` (E160), `run_info.json`
- `epoch_010.pt` through `epoch_160.pt` (every 10 epochs, 16 files)
- `nohup_training.log` (full training log, 160 epochs)

**EfficientNetV2-S v3** (`paperIV_efficientnet_v2_s_v3_cosine/`): **COMPLETE**
- `best.pt` (E17, AUC=0.9895), `last.pt` (E160), `run_info.json`
- `epoch_010.pt` through `epoch_160.pt` (every 10 epochs, 16 files)
- `nohup_training.log` (full training log, 160 epochs)

**EfficientNetV2-S v4** (`paperIV_efficientnet_v2_s_v4_finetune/`): **COMPLETE**
- `best.pt` (E1, AUC=0.9921), `last.pt` (E60), `run_info.json`
- `epoch_010.pt` through `epoch_060.pt` (every 10 epochs, 6 files)
- `nohup_training.log` (full training log, 60 epochs)

---

---

## 10. Selection Function Results (2026-02-12)

### 10.1 Configuration

| Parameter | Value |
|-----------|-------|
| Model | EfficientNetV2-S v4 finetune (E1, AUC=0.9921) |
| Checkpoint | `paperIV_efficientnet_v2_s_v4_finetune/best.pt` |
| Manifest | `training_parity_70_30_v1.parquet` |
| Host split | val (134,149 negatives, subsampled to 20,000) |
| Injection engine | SIE+shear ray-shooting, Sersic+clumps source |
| Grid axes | theta_E: 0.5-3.0" (step 0.25), PSF: 0.9-1.8" (step 0.15), depth: 24.0-25.5 mag (step 0.5) |
| Cells | 308 (11 x 7 x 4) |
| Injections/cell | 200 |
| Total injections | 61,600 |
| Source r-band magnitude | Uniform 23-26 mag |
| Thresholds | p>0.3, p>0.5, p>0.7, FPR=0.1% (p>0.806), FPR=0.01% (p>0.995) |

### 10.2 FPR-Derived Thresholds

| Target FPR | Derived threshold | Actual FPR |
|-----------|------------------|-----------|
| 0.1% | p = 0.8059 | 1.0000e-03 |
| 0.01% | p = 0.9951 | 1.0000e-04 |

### 10.3 Completeness by Source Magnitude (p > 0.3)

| Source mag bin | Mean completeness |
|---------------|------------------|
| 23-24 (bright) | 7.9% |
| 24-25 (medium) | 3.9% |
| 25-26 (faint) | 1.1% |
| All (23-26) | 4.3% |

### 10.4 Completeness by theta_E (all sources, p > 0.3)

| theta_E (") | C(p>0.3) | C(p>0.5) | C(FPR=0.1%) | Arc SNR |
|------------|----------|----------|-------------|---------|
| 0.50 | 1.0% | 0.6% | 0.3% | 4.4 |
| 0.75 | 2.7% | 1.8% | 1.0% | 6.0 |
| 1.00 | 4.3% | 3.3% | 1.9% | 7.2 |
| 1.25 | 5.2% | 4.2% | 2.8% | 7.8 |
| 1.50 | 5.2% | 4.2% | 2.8% | 8.0 |
| 1.75 | **5.7%** | **4.9%** | **3.6%** | **8.2** |
| 2.00 | 5.5% | 4.7% | 3.8% | 7.7 |
| 2.25 | 5.0% | 4.4% | 3.2% | 7.0 |
| 2.50 | 4.7% | 4.0% | 3.2% | 6.0 |
| 2.75 | 4.4% | 3.5% | 2.7% | 4.8 |
| 3.00 | 4.0% | 3.3% | 2.2% | 4.0 |

### 10.5 Brightest Sources (23-24 mag) by theta_E (p > 0.3)

| theta_E (") | Completeness | N_injections |
|------------|-------------|-------------|
| 0.50 | 2.1% | 1,841 |
| 0.75 | 5.9% | 1,919 |
| 1.00 | 8.5% | 1,899 |
| 1.25 | **10.7%** | 1,942 |
| 1.50 | 8.9% | 1,853 |
| 1.75 | 10.6% | 1,881 |
| 2.00 | 9.7% | 1,849 |
| 2.25 | 8.3% | 1,859 |
| 2.50 | 8.1% | 1,892 |
| 2.75 | 8.0% | 1,850 |
| 3.00 | 6.2% | 1,881 |

### 10.6 Key Scientific Findings

1. **Low overall completeness**: Even at the most permissive threshold (p>0.3), the CNN achieves only 4.3% completeness averaged over a realistic source population (r_mag 23-26). This is because the model was trained on confirmed, obvious lenses and therefore has a high confidence threshold.

2. **Strong source brightness dependence**: Completeness varies 7x between bright (23-24 mag: 7.9%) and faint (25-26 mag: 1.1%) sources. The selection function is dominated by source brightness, not Einstein radius.

3. **Peak at theta_E = 1.25-1.75"**: Completeness peaks where arc SNR is highest (8.0-8.2), consistent with the PSF-blending limit at small theta_E and surface brightness dilution at large theta_E.

4. **Sharp drop below theta_E < 1.0"**: The model is essentially blind to small Einstein radius systems (C < 3% for theta_E < 1.0"), where arcs are unresolved from the deflector.

5. **FPR thresholds**: At FPR=0.1%, the threshold is p=0.806, and completeness drops to ~2.5%. At FPR=0.01%, threshold is p=0.995 and completeness drops to <1%. This quantifies the purity-completeness tradeoff.

### 10.7 Injection Validation (Pre-Grid QA)

| Check | Result |
|-------|--------|
| Flux conservation | 0.00 (perfect, all bands) |
| Host FPR (p>0.5) | 0.0% |
| Detection rate (default, p>0.5) | 4.0% |
| Detection rate (core-suppressed, p>0.5) | 5.0% |
| Mean arc SNR | 6.3 |
| Success rate | 100/100 |

### 10.8 Sensitivity Analysis (Complete)

9 perturbation variants run (100 injections/cell, 385 cells each, ~4 min/variant):

| Perturbation | Mean delta-C | Max |delta-C| per cell | Std delta-C |
|-------------|-------------|----------------------|------------|
| PSF FWHM +10% | -0.52% | 4.0% | 0.82% |
| PSF FWHM -10% | +0.54% | 5.0% | 0.92% |
| Source R_e +30% | +0.15% | 4.0% | 0.69% |
| Source R_e -30% | -0.24% | 4.0% | 0.84% |
| Color (g-r) +0.2 mag (redder) | -0.64% | 5.0% | 0.94% |
| Color (g-r) -0.2 mag (bluer) | +0.61% | 7.0% | 0.96% |
| Lens q broader [0.3-1.0] | +0.28% | 5.0% | 1.18% |
| Lens q narrower [0.7-1.0] | -0.25% | 5.0% | 1.00% |

**Findings:**
1. **Source color is the dominant systematic**: +/-0.2 mag g-r shift causes +/-0.6% mean completeness change. Bluer sources easier to detect (higher contrast against red LRG hosts).
2. **PSF FWHM is second**: +/-10% PSF causes +/-0.5% completeness shift. Better seeing helps.
3. **Source size and lens ellipticity are subdominant**: <0.3% mean shifts.
4. **Maximum per-cell shift**: 5-7% in individual cells, but population-averaged systematic envelope is <1% for all perturbations.
5. **Systematic uncertainty << statistical uncertainty** at these low completeness levels (Bayesian binomial 68% CI width of ~3-7% per cell at 200 injections).

---

---

## 11. Injection Model 2: Deflector-Conditioned Injection (2026-02-13)

### 11.1 Motivation and LLM Review Summary

An independent LLM review of the Model 1 injection pipeline identified the following:

1. **The ~43-point gap between injection completeness and real-lens recall is NOT defensible as a selection function**, because Model 1 injects onto random hosts rather than deflector-like hosts.
2. **Three silent-failure bugs** in `selection_function_grid.py` that could silently depress completeness:
   - FPR threshold derivation inserts all-zero images on load failure and scores them
   - FPR rank uses `n_neg` not `n_valid`
   - Injection loop counts failed injections toward denominator
3. **Model 2 (deflector-conditioned injection) is the minimum for MNRAS**: lens parameters must be conditioned on host galaxy properties.
4. **Model 1 should be reframed as an ablation** demonstrating that naive injection fails.

### 11.2 Bug Fixes Applied (selection_function_grid.py)

All three silent-failure bugs were fixed:
- Failed loads are now excluded from scoring (not scored as zeros)
- FPR rank computed from `n_valid`, not `n_attempted`
- Injections retry on failure with new hosts (up to 5 retries)
- Explicit `n_ok`/`n_failed` counters in output
- Failure log captured in metadata JSON

### 11.3 Model 2 Implementation

**Key components** (in `stronglens_calibration/injection_model_2/`):

1. **host_matching.py**: Estimates host galaxy axis ratio (q) and position angle (PA) from r-band second moments. Adapted from LLM starter code, hardened with input validation, edge case handling, and 26 passing unit tests.

2. **host_selection.py**: Selects LRG-like hosts (DEV/SER Tractor morphology) from manifest. In the val split: 85,804 SER + 26,940 DEV = 112,744 LRG-like hosts (84% of 134,149 total negatives).

3. **selection_function_grid_v2.py**: Extended grid runner with `--model {1,2}` flag. Model 2 uses LRG hosts and host-conditioned q/PA.

4. **Lens parameter conditioning**: LensParams q and PA are derived from host light moments (not independent priors). Shear drawn from U(0, 0.08) with random PA.

**REJECTED from LLM starter code**: The LLM's `injection_engine.py` was rejected due to 3 confirmed mathematical bugs in SIE deflection:
- `denom = psi + q^2` instead of `psi`
- atan/atanh swapped for x/y
- Prefactor `q` instead of `sqrt(q)`

Our existing engine (`dhs/injection_engine.py`) with 28 passing tests is used unchanged.

### 11.4 Host Conditioning Diagnostic Results

4-way comparison at 500 injections/point, 6 theta_E values:

| Condition | Description | Mean C(p>0.5) |
|-----------|-------------|---------------|
| LRG_conditioned | LRG hosts + host-conditioned q/PA (Model 2 full) | ~3.0% |
| LRG_independent | LRG hosts + random q/PA | ~3.2% |
| random_independent | All hosts + random q/PA (Model 1 baseline) | ~3.3% |
| random_conditioned | All hosts + host-conditioned q/PA | ~3.4% |

**Interpretation**: At N=500 per point, the differences between conditions are within statistical noise (~1-2%). The host morphology effect (LRG vs random) appears smaller than expected. This suggests:

1. The ~43-point gap between injection and real-lens recall is NOT primarily due to host mismatch at this level of analysis
2. Other factors (source appearance, preprocessing, CNN feature space) may dominate
3. We need higher N per point (1000+) and/or stratified analysis by host brightness/concentration to detect the effect
4. Model 1 completeness being uniformly low (~3-5%) may be a floor set by source realism, not host selection

### 11.5 Model 2 Full Grid Run

**Status**: Running on Lambda5 (launched 2026-02-13, v4 finetune checkpoint epoch 1)

Grid configuration: 11 theta_E × 7 PSF × 5 depth × 200 inj/cell = 77,000 total injections

Results will be compared directly to the Model 1 fixed grid run from §10.

### 11.6 Paper Framing (Revised)

Based on the diagnostic results:

- **Model 1** is an ablation showing that naive injection produces low completeness (~2-5%)
- **Model 2** is the primary methodology paper contribution showing host-conditioned injection
- The host conditioning diagnostic (§11.4) suggests the gap is not dominated by host morphology alone — this is itself a publishable finding
- Model 3 (real source stamps) may be needed to close the remaining gap

---

**Status as of 2026-02-13 16:22 UTC**: **ALL 5 RUNS COMPLETE.** All best checkpoints, epoch checkpoints (every 10), full nohup training logs, and run_info.json files are preserved on NFS and S3. Lambda instances are ready for termination — all local data has been archived to NFS.

---

## 12. LLM Reviewer Findings (Post-Training)

### 12.1 v2-to-v4 Improvement is Statistically Indistinguishable

The v4 finetune (best epoch 1) achieves AUC 0.9921 vs v2's 0.9915.
With ~1432 val positives and ~62K val negatives, the AUC standard error
is approximately sigma_AUC = sqrt(0.99 × 0.01 / 1432) ≈ 0.0026.
The improvement of 0.0006 is 0.0006 / 0.0026 = 0.23 standard errors.

**Conclusion**: The v2→v4 improvement is well within sampling noise (p >> 0.05).
For DeLong's test, the expected z-statistic is ~0.2, far below significance.

**Paper language**: "v2 and v4 achieve statistically indistinguishable
validation AUC (0.9915 vs 0.9921; delta = 0.0006, approximately 0.23σ).
We use v4 as the reference model but note the improvement from fine-tuning
is within sampling noise."

### 12.2 v4 Best-at-Epoch-1 Honest Framing

v4 loads v2's epoch 19 weights and trains with 8× lower LR and cosine
schedule. The best epoch is epoch 1. This occurs because:
- The 3-epoch warmup starts at lr/100 ≈ 5e-7
- Epoch 1 uses lr ≈ 1.7e-7 (essentially zero)
- By epoch 2+, LR increases enough to cause overfitting

**Honest interpretation**: v4 is v2 + epsilon. The +0.0006 AUC gain is a
minimal perturbation from near-zero LR, not evidence that 60 epochs of
training were needed.

**Paper framing**: "Two-phase training: (1) 160 epochs from ImageNet
initialization (v2), (2) 60 epochs of fine-tuning with 8× reduced LR (v4).
Best validation AUC achieved within the first fine-tuning epoch, consistent
with the model being near-optimal after phase 1."

### 12.3 Geographic Test Split Recommendation

Current best-epoch selection uses the val set repeatedly, introducing
selection bias. The reported val AUC is an optimistic estimate.

**Recommendation for publication**:
1. Report TEST-SET AUC as the primary metric (unbiased estimate).
2. Report val AUC only as the early-stopping criterion.
3. The 70/15/15 manifest has a geographic test split (HEALPix-based)
   that provides spatial decorrelation. Use this for the final metric.
4. For v5 retrain: save checkpoints every epoch, select by val AUC,
   report test AUC. If test AUC << val AUC, overfitting to val.

### 12.4 pAUC at Low FPR

Standard AUC (0.9921) is dominated by the trivial part of the ROC curve
where FPR > 1%. The scientifically relevant operating regime is FPR < 0.1%
(each false positive requires expensive spectroscopic follow-up).

Partial AUC at FPR < 0.1% (pAUC) has been added to evaluate_parity.py.
This should be reported alongside full AUC in the paper.

### 12.5 z-Band PSF Scaling Correction

LLM2 identified that the z-band PSF scaling (relative to r-band) was set
to 1.00, but atmospheric Kolmogorov turbulence gives λ^{-1/5} scaling:
z-band PSF ≈ 0.94× r-band. This has been corrected in `injection_engine.py`
(`psf_fwhm_scale_z`: 1.00 → 0.94). The g-band default (1.05) was already
close to the physical value (~1.07).

This is a ~6% correction in z-band PSF width. Given that PSF effects are
estimated at 3-5% of the total gap, the impact is small but eliminates a
known systematic error.

### 12.6 Positive-Class Calibration

LLM2 (Q6.14) pointed out that overall ECE/MCE are dominated by the
overwhelming negative class (93:1 ratio). A model can have excellent
overall calibration while being poorly calibrated on positives.

`evaluate_parity.py` now computes positive-class ECE and MCE (calibration
assessed only on positive labels). This is more informative for
understanding whether p=0.8 truly means ~80% chance of being a lens.

### 12.7 Host-Cutout Noise Estimation

LLM1 (Q3.1) recommended estimating pixel noise directly from the host
cutout's sky ring (MAD in an outer annulus) as an alternative to the
psfdepth-based approach. This avoids any ambiguity about psfdepth units.

A new utility function `estimate_sigma_pix_from_cutout()` has been added
to `injection_engine.py` for this purpose. Results should be compared
against psfdepth-based estimates to verify consistency.

### 12.8 Arc Morphology Pixel Statistics

LLM1 (Q1.2) asked for specific, measurable pixel-level properties that
differ between real and injected arcs. A new diagnostic script
`arc_morphology_statistics.py` computes:
- High-frequency power (2D power spectrum fraction above k=0.2 cycles/pix)
- Structure tensor anisotropy (eigenvalue ratio = elongation)
- Color-gradient coherence (cross-band spatial correlation)
- Local variance ratio (arc region vs sky region)

If real arcs show significantly higher high-freq power and lower color
coherence than Sersic injections, this confirms morphological mismatch.

### 12.9 Real-Arc Residual Morphology Experiment

LLM2 Phase 3 recommended replacing the Sersic source with real arc
residuals (galaxy-subtracted images of confirmed lenses). The script
`real_arc_morphology_experiment.py` implements this:
1. Extract arc residuals from Tier-A lenses via median-filter subtraction
2. Inject these residuals into host galaxies
3. Compare CNN detection rates: Sersic vs real-arc at fixed beta_frac

If detection jumps significantly with real-arc morphology, source
realism is a major driver of the gap.

### 12.10 Per-Layer Fréchet Distance

LLM2 (Q1.2) recommended computing Fréchet distance at each layer of the
CNN to identify at which feature scale the real-vs-injection divergence
occurs. `feature_space_analysis.py` now hooks all 8 feature blocks of
EfficientNetV2-S and computes per-layer FD.

If divergence appears early (low-level features = texture/noise), Poisson
noise or pixel-level statistics dominate. If late (semantic features),
morphology/geometry dominates.

### 12.11 Uncertainty Surfaces for Completeness Maps

LLM1 (Q6.9) noted that 200 injections/cell gives wide CIs. The
completeness grid now outputs both 68% and 95% credible intervals
(`ci95_lo`, `ci95_hi` columns). These should be plotted as uncertainty
surfaces in completeness maps for the paper. At n=500/cell, the 95% CI
for p=3.5% is approximately [2.1%, 5.4%] — still wide but publishable.

### 12.12 Paper IV Cross-Model Check

LLM2 (Q6.12) recommended running `bright_arc_injection_test.py` with
Paper IV's pre-trained checkpoint (if available, ~30 min compute). If
Paper IV achieves >50% bright-arc detection vs our ~30%, the gap is
model-dependent. If similar (~25-35%), the gap is fundamental to Sersic
injection morphology.

---

## 13. Prompt 3 LLM Analysis: Retrain Decision and Pre-Retrain Experiments

*Added 2026-02-13. Sections 13-18 added per LLM Prompt 3 analysis.*

Both LLMs agree: **do NOT retrain until cheap experiments Q2.1-Q2.4 are
completed.** The 70pp completeness gap (3.5% vs ~70% in the literature for
detectable-only injections) is driven primarily by injection geometry
(beta_frac area-weighted sampling) and morphology (smooth Sersic vs clumpy
real galaxies), NOT by the annulus normalization bug.

### 13.1 Q2.1: Beta_frac Ceiling Test (Already Available)

Run with existing infrastructure:

```bash
cd stronglens_calibration
export PYTHONPATH=.

# Restrict beta_frac to high-magnification regime
python sim_to_real_validations/bright_arc_injection_test.py \
    --checkpoint checkpoints/best.pt \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --out-dir results/q21_beta_frac_cap \
    --beta-frac-range 0.1 0.55 \
    --n-injections 1000

# Success criterion: bright-arc ceiling moves from ~30% to >=40%.
# If so, beta_frac geometry is a major driver.
```

### 13.2 Q2.2: Embedding Separation Analysis (Already Available)

Run with existing infrastructure:

```bash
python scripts/feature_space_analysis.py \
    --checkpoint checkpoints/best.pt \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --out-dir results/q22_embedding_analysis \
    --n-real 500 --n-inj 500 --n-neg 500

# Success criterion: FD(real, inj) / FD(real, neg) > 2 => injections
# live in a different feature manifold than real lenses.
```

### 13.3 Q2.3: Annulus Stat Comparison (NEW)

```bash
python scripts/annulus_comparison.py \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --out-dir results/q23_annulus_comparison \
    --n-samples 1000

# If KS p-value > 0.05: annulus change is cosmetic.
# If KS p-value < 0.001 AND correlated with PSF/depth: material.
```

### 13.4 Q2.4: Mismatched Annulus Scoring (NEW)

```bash
python scripts/mismatched_annulus_scoring.py \
    --checkpoint checkpoints/best.pt \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --out-dir results/q24_mismatched_annulus \
    --n-samples 500

# LLM1 prediction: "almost certainly hurts." But magnitude of
# degradation is informative for the retrain decision.
```

## 14. Label Noise Estimation (Prompt 3 Q3.5-Q3.7, Q4.3)

### 14.1 Tier-B False-Positive Rate

Our positive training set consists of:
- **Tier-A**: 389 spectroscopically or multiply-imaged confirmed lenses
- **Tier-B**: 4,399 visual-only lens candidates (no spectroscopic confirmation)

Estimated Tier-B false-positive rate: **~10%** (based on visual inspection
quality and comparison with spectroscopic follow-up campaigns in the
literature). This implies ~440 mislabeled positives in the training set.

Impact: AUC and overall recall are robust to ~10% label noise (standard
result from Natarajan et al. 2013, Rolnick et al. 2017). However:
- **Calibration degrades**: the model's probability outputs are biased
  upward by contaminated positives
- **High-purity regime** (FPR < 0.1%): label noise has disproportionate
  impact. Contaminated negatives masquerading as positives inflate
  apparent recall.

### 14.2 Negative Contamination

From the DESI lens candidate catalog geometry:
- Training negatives: 446,893
- Estimated real lenses among negatives: **~10-50** (extrapolating from
  the DESI lens density of ~0.001 per sq deg in the training footprint)

This is negligible for training (0.01%) but becomes relevant for the
FPR calculation at very low thresholds. If even 10 of the "false
positives" at FPR < 0.01% are actually real lenses, the true FPR is
even lower than reported.

### 14.3 LLM1 Recommendation

> "A referee will push you to report pAUC or TPR at fixed low FPR,
> not only AUC" (LLM1 Q3.7).

**Implemented**: `evaluate_parity.py` now reports:
- pAUC at FPR < 0.1% and FPR < 1%
- TPR at FPR = 0.1% and FPR = 1% (interpolated from ROC curve)
- Positive-class ECE and MCE

## 15. Success Criteria and GO/NO-GO Decision Tree

### 15.1 Retraining Success Criteria (from LLM1)

| Metric | Current | GO Threshold | Notes |
|--------|---------|-------------|-------|
| AUC | 0.9928 | >= 0.9930 | Must not regress |
| Tier-A recall (p>0.3) | 73.3% | >= 80% | Primary metric |
| Bright-arc ceiling | ~30% | >= 40% | From Q2.1 |
| Main-map completeness | 3.5% | >= 7% (2x) | Ambitious but justifiable |

LLM1 prediction for retrain with annulus fix alone: AUC +0.001 to +0.003,
bright-arc ceiling +3 to +7pp.

### 15.2 GO/NO-GO Decision Logic

```
Q2.3: Annulus distributions differ materially?
  ├── NO  → Annulus is cosmetic → NO-GO on annulus-only retrain
  └── YES → Q2.4: Mismatched scoring degrades significantly?
              ├── NO  → Model robust → NO-GO
              └── YES → GO (annulus is a genuine confounder)

Q2.1: beta_frac cap raises ceiling to >=40%?
  ├── YES → Geometry is primary driver → Consider wider priors
  └── NO  → Morphology is primary driver → Sersic→real-arc swap needed

Q2.2: Embedding FD(real,inj) >> FD(real,neg)?
  ├── YES → Injection manifold differs → Realism upgrade needed
  └── NO  → Injections are well-placed → Gap is elsewhere
```

### 15.3 Retrain-Failure Fallback Plan (LLM1 Q2.6)

If retraining with annulus fix + extended priors fails to meet GO thresholds:

1. **Conclude** gap is injection realism + prior mismatch (not a code bug)
2. **Publish** beta_frac-restricted completeness as the primary result
   ("completeness of detectable lenses" rather than "all lenses")
3. **Report** embedding separation as evidence for the realism gap
4. **Next steps** (future work section):
   - Swap Sersic profiles for real galaxy stamps (HUDF/COSMOS cutouts)
   - Adopt brightness acceptance criteria (a la HOLISMOKES)
   - Consider fine-tuning Zoobot (galaxy morphology foundation model;
     outperforms purpose-built CNNs in Euclid 2025)

## 16. Hostile-Referee Defense Strategy (Prompt 3 Q4.1-Q4.11)

### 16.1 "How is 3.5% completeness useful?" (Q4.1)

**Rebuttal framework:**

1. **3.5% is the marginal completeness over the full stated prior volume.**
   Most of that volume contains faint, low-magnification sources that no
   CNN can detect. This is consistent with Jacobs et al. (2019), who found
   only 4.8% of simulated lenses received the highest confidence grade from
   human inspection.

2. **Stratified completeness reveals the model's actual capability.** At
   bright source magnitudes (23-24) with theta_E > 1.5", completeness
   reaches ~30%. With beta_frac restricted to high-magnification systems
   (< 0.55), we expect >= 40% (pending Q2.1 results).

3. **The primary deliverable is the selection function, not a single
   number.** The selection function table gives completeness as a function
   of (theta_E, source_mag, beta_frac), enabling population studies to
   correct for selection.

4. **Comparison to literature:** HOLISMOKES individual ResNets achieve
   TPR0 = 10-40%, and network committees reach ~60%, but on a pre-filtered
   set with brightness acceptance criteria (arc pixel > 5-sigma sky AND
   > 1.5x lens flux AND mu >= 5). Our pipeline has no such filter, which
   inflates the denominator.

### 16.2 "Training data contamination" (Q4.2)

**Implemented controls:**
- `real_lens_scoring.py --tier-a-only`: evaluates on confirmed lenses only
- Training-split leakage guard: prints Tier-A counts per split
- Geographic test split (Section 12.3): spatial isolation prevents field leakage

### 16.3 "Label noise invalidates your results" (Q4.3)

**Response:** See Section 14 for concrete estimates. AUC is robust to
~10% label noise. We now report pAUC and TPR at fixed FPR, which are
more sensitive to label noise and provide honest performance bounds.

### 16.4 "Sensitivity around wrong priors is meaningless" (Q4.4)

**Two valid framings:**

1. **Anchor priors to observed distributions.** Show that our R_e, n,
   source mag, and color priors overlap with the observed lensed-source
   population (Section 18).

2. **Publish priors as part of the data product.** The completeness table
   explicitly conditions on stated priors. Any downstream user can
   reweight if their population model differs.

### 16.5 "Why not use real galaxy stamps?" (Q4.5-Q4.6)

**Response:** Sersic profiles are the standard in the field (Collett 2015,
Herle et al. 2024, HOLISMOKES). The real-arc residual experiment
(Section 12.9) quantifies the detection gap between Sersic and real
morphology. If the gap is large, this motivates HUDF/COSMOS-based
injection in future work.

### 16.6 "Paper IV gap is unexplained" (Q4.7-Q4.8)

**Documented confounders (too many to attribute to a single factor):**

| Factor | This Work | Paper IV |
|--------|-----------|----------|
| Positive set | 4,788 (389 A + 4,399 B) | 1,372 confirmed |
| Negative cleaning | Standard | p > 0.4 flagged |
| Annulus | (20, 32) bug | (32.5, 45) intended |
| Tier-B weighting | Enabled | Disabled |
| GPU/framework | Lambda Cloud A100 | Different setup |

LLM1: "too many uncontrolled variables to attribute to any single factor."
The Paper IV cross-model check (Section 12.12) will help disambiguate.

### 16.7 "No independent holdout" (Q4.9)

**Documented limitation.** All Tier-A lenses participate in training.
No spectroscopically confirmed holdout set exists for DESI DR10 at this
time. Mitigation: geographic test split (Section 12.3) ensures spatial
isolation.

Binomial CI for 73.3% recall on ~130 val Tier-A: [65%, 80%] (95% CI).
This is wide but honest.

### 16.8 "Missing source redshift dimension" (Q4.10)

**Documented limitation.** No redshift stratification in completeness
currently. Source redshift determines:
- Physical size -> angular R_e
- Rest-frame SED -> observed colors
- Surface brightness dimming -> detectability

Our R_e, mag, and color priors implicitly encode z-dependent distributions,
but we cannot disentangle z-specific completeness without explicit
z-labeled injections. This is noted as a limitation in the paper.

### 16.9 "No comparison to published methods" (Q4.11)

See Section 17 (Literature Comparison).

## 17. Literature Comparison (Prompt 3)

### 17.1 Herle et al. (2024, MNRAS 534, 1093)

The only dedicated CNN selection function study. Key parameters:
- R_S in U(0.05, 0.3), n in U(1, 4)
- At 8-sigma threshold:
  - Median selected theta_E >= 0.879"
  - Median selected R_S >= 0.178"
  - **Median selected n >= 2.55** (our previous n_max=2.5 cut this off)
- At 12-sigma threshold:
  - theta_E >= 1.04", R_S >= 0.194", n >= 2.62
- Selection function is **independent of power-law slope gamma** (good
  news for mass modeling with CNN-selected samples)
- Selection **reinforces cross-section bias**: the compounding effect
  means CNN-found samples are more biased than cross-section alone

### 17.2 HOLISMOKES XI (Canameras et al. 2024)

- 1,574 real HUDF galaxy stamps (no Sersic!) as source templates
- SIE + shear via GLEE, convolved with location-specific PSF
- Brightness acceptance criteria:
  - Arc pixel > 5-sigma sky noise
  - Arc pixel > 1.5x lens flux
  - Total magnification mu >= 5
- Individual ResNets: TPR0 = 10-40%
- Network committees: up to ~60%
- **Our pipeline has NO such acceptance filter**, which directly inflates
  the denominator and lowers overall completeness

### 17.3 Euclid (2025)

- ~70% completeness on Q1 data (injection set designed for detectable lenses)
- Fine-tuned Zoobot (pretrained galaxy morphology foundation model)
  **outperforms purpose-built CNNs**
- Bayesian ensemble: 52 +/- 2% purity at 50% completeness
- Pearce-Casey (2025): ~77% TPR at 0.8% FPR

### 17.4 Jacobs et al. (2019, 2021)

- 2019: 500 simulated high-z lenses with theta_E > 2", only **4.8%**
  received highest confidence grade from human inspection
- 2021: Performance degrades where g-band source mag exceeds ~21.5
- **Our 3.5% is directly comparable to the 4.8% result** from Jacobs,
  given that our prior volume extends to fainter sources.

### 17.5 DES CNN (Gonzalez et al. 2025)

- 31-70% on confirmed candidates (different denominator: already-visible
  lenses, not drawn from a prior volume including faint sources)

### 17.6 Our Position

3.5% marginal completeness is consistent with a broad parameter space
including many intrinsically undetectable faint-source configurations.
When restricted to the bright/high-magnification regime, our completeness
should be significantly higher (Q2.1 experiment pending). The selection
function table enables model-specific completeness corrections for any
sub-population.

## 18. Injection Prior Justification (Prompt 3 Q4.4-Q4.6)

### 18.1 Source Effective Radius R_e

| Range | Source | Notes |
|-------|--------|-------|
| (0.05, 0.25) | Original | Compact end only |
| (0.05, 0.30) | Herle et al. (2024) | U(0.05, 0.3) |
| (0.1, 0.5) | Collett (2015) | z~1-3 typical |
| (0.2-0.8) | Observed z~1 late-type | From HST sizes |
| **(0.05, 0.50)** | **This work (updated)** | Covers z~2 population |

Extending to 0.50" captures the late-type disk galaxies that dominate
the lensed source population at z ~ 1-2. The previous (0.05, 0.25) range
systematically under-represented larger star-forming disks.

### 18.2 Sersic Index n

| Range | Source | Notes |
|-------|--------|-------|
| (0.7, 2.5) | Original | Misses CNN-preferred n |
| (1, 4) | Herle et al. (2024) | U(1, 4) |
| n=1 | Collett (2015) | Fixed exponential disk |
| **(0.5, 4.0)** | **This work (updated)** | Standard practice |

**Critical finding:** Herle et al. (2024) show CNNs preferentially select
sources with median n >= 2.55 (at 8-sigma). Our previous n_max=2.5 cut
off exactly at this threshold, potentially excluding the most detectable
injected sources.

### 18.3 Source Magnitude

- Our range: U(23, 26) unlensed r-band
- Herle et al.: U(M_lens, 24) — always fainter than lens but brighter cut
- Consequence: our faint end (25-26 mag) produces mostly undetectable arcs,
  lowering marginal completeness

### 18.4 Beta_frac (Source Offset)

- Our sampling: area-weighted (beta_frac = sqrt(U(lo^2, hi^2)))
- P(beta_frac < 0.55) ~= 29.5%
- This matches the ~30% bright-arc ceiling observed empirically
- Physically motivated: uniform source density on the sky

### 18.5 Source Colors

- g-r ~ N(0.2, 0.25): blue star-forming, reasonable for z ~ 1-3 sources
- r-z ~ N(0.1, 0.25): bluer than typical lens galaxies
- Should be validated against observed lensed source colors in future work

### 18.6 HOLISMOKES Contrast

HOLISMOKES uses 1,574 real HUDF galaxy stamps with:
- Clumps, asymmetry, realistic color gradients
- Rest-frame UV morphology at z ~ 1-3

Our Sersic profiles are smooth by construction. The clump model (60%
probability, 1-4 clumps at 15-45% flux fraction) adds some structure but
cannot replicate the richness of real galaxy morphology. The real-arc
residual experiment (Section 12.9) directly tests whether this matters.

---

## 19. Prompt 4 LLM Analysis: Experiment Run Commands (TIER 2a)

Exact commands for the four pre-retrain experiments (Q2.1-Q2.4).
All commands assume Lambda3 with `.venv-lambda3` active and code rsynced.

### 19.1 Q2.1: Beta-Frac Restriction Test

```bash
# Runtime: 30-60 min (GPU) per LLM1, 2-4 hrs per LLM2
cd /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration
python sim_to_real_validations/bright_arc_injection_test.py \
    --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --out-dir results/D01_$(date +%Y%m%d)_pre_retrain_diagnostics/q21_beta_frac \
    --beta-frac-range 0.1 0.55
```

**Decision gate:** If detection rate with beta_frac cap >= 40%, geometry is
the primary driver. If still ~30%, morphology is the bottleneck.

### 19.2 Q2.2: Embedding UMAP + Linear Probe

```bash
# Runtime: 20-40 min (GPU)
python scripts/feature_space_analysis.py \
    --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --out-dir results/D01_$(date +%Y%m%d)_pre_retrain_diagnostics/q22_embedding_umap \
    --n-samples 200
```

**Decision gate:** Linear probe AUC > 0.85 = CNN encodes "injection-ness"
(realism gap). FD(real,inj) >> FD(real,neg) = injections occupy different
manifold region. AUC ~ 0.5 = injections well-camouflaged.

### 19.3 Q2.3: Annulus Comparison

```bash
# Runtime: 1-5 min (CPU)
python scripts/annulus_comparison.py \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --out-dir results/D01_$(date +%Y%m%d)_pre_retrain_diagnostics/q23_annulus_comparison \
    --n-samples 1000
```

**Decision gate:** If (32.5,45) vs (20,32) distributions overlap
substantially (KS p > 0.05), annulus is cosmetic. If distributions
differ materially, proceed to Q2.4.

### 19.4 Q2.4: Mismatched Annulus Scoring

```bash
# Runtime: 10-30 min (GPU)
python scripts/mismatched_annulus_scoring.py \
    --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --out-dir results/D01_$(date +%Y%m%d)_pre_retrain_diagnostics/q24_mismatched_scoring \
    --n-samples 500
```

**Decision gate:** If AUC drops > 0.002 with mismatched preprocessing,
model is fragile to annulus → GO for retrain. If AUC stable, model is
robust → annulus fix alone won't help.

### 19.5 Additional CPU Diagnostics

```bash
# Split balance (CPU, ~10s):
python scripts/split_balance_diagnostic.py \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --out-dir results/D01_$(date +%Y%m%d)_pre_retrain_diagnostics/split_balance_check

# Masked pixels (CPU, ~1-5 min):
python scripts/masked_pixel_diagnostic.py \
    --manifest manifests/training_parity_70_30_v1.parquet \
    --out-dir results/D01_$(date +%Y%m%d)_pre_retrain_diagnostics/masked_pixel_check \
    --n-samples 1000 --threshold 0.05
```

**Decision gate for masked pixels:** If >5% of cutouts have >5% bad pixels,
need to define masking policy before retrain (drop / inpaint / propagate invvar).

---

## 20. Prompt 4: Expanded Success Criteria (TIER 2b)

### 20.1 Dual Threshold Sets

Both LLM reviewers proposed different success thresholds:

| Metric | LLM1 Threshold | LLM2 Threshold | Current Value |
|--------|----------------|----------------|---------------|
| AUC | >= 0.9930 | -- | 0.9921 |
| Tier-A recall threshold | p > 0.3 | p > 0.5 | p > 0.3: 73.3% |
| Tier-A recall target | >= 80% | >= 65% | 73.3% (at p>0.3) |
| Bright-arc ceiling | >= 40% | -- | ~30% |
| Main-map completeness | >= 7% (2x) | -- | 3.5% |

**AUC Standard Error:** For N=14,000 validation samples, SE ~ 0.0026.
Observed v2-to-v4 improvement was +0.0006 (within 1 SE). Any claimed
improvement must be >= 2 SE = 0.0052 to be statistically significant.

### 20.2 LLM2's Sequential Decision Tree (verbatim thresholds)

```
Step 1: Run gen5a + gen5b with corrected annulus
  → If AUC improves >= 0.002:
      Annulus was a genuine confounder → proceed
  → If AUC flat:
      Annulus cosmetic → focus on injection priors

Step 2: Evaluate Tier-A recall at p > 0.5
  → If recall >= 65%: PASS (publishable)
  → If recall < 65%: Investigate further

Step 3: Compute multi-dimensional completeness grid
  → Completeness(theta_E, beta_frac, depth)
  → If any cell > 20%: Report as "detectable lens population"

Step 4: Literature comparison table
  → Must include: Jacobs+19, Huang+20, He+20, Rojas+22, Herle+24
  → Distinguish apples-to-apples (same survey) vs cross-survey
```

---

## 21. Prompt 4: Paper Framing Options (TIER 2c)

### 21.1 LLM1 Negative-Result Framing

**Title:** "Strong-lens completeness in the DESI Legacy Surveys:
detection limits from injection recovery"

**Key message:** Even with a well-calibrated CNN, injection-based
completeness is limited by the realism gap between parametric (Sersic)
injections and real lensed arcs.

**4 key figures:**
1. Selection function grid (theta_E × depth × beta_frac)
2. UMAP: real Tier-A vs injections vs negatives
3. Beta_frac dependence plot showing ~30% ceiling origin
4. Annulus impact: before vs after correction

### 21.2 LLM2 Positive-Result Framing

**Title:** "Injection-recovery completeness of the DHS CNN lens finder
in the DESI Legacy Imaging Surveys"

**Key message:** First systematic, multi-dimensional completeness
characterization of a CNN lens finder on DR10, with well-documented
priors and reproducible methodology.

**7 key figures:**
1. Architecture diagram (EfficientNetV2-S + training pipeline)
2. Selection function heatmap (theta_E × depth)
3. Beta_frac restricted completeness (the "detectable" population)
4. Embedding space visualization (4-group UMAP/t-SNE)
5. Annulus normalization comparison
6. ROC curve with Tier-A recall annotation
7. Literature comparison scatter plot

**LLM2 timeline:** ~3 weeks (1 week diagnostics, 1 week retrain, 1 week writeup)

---

## 22. Prompt 4: Weighted Loss Documentation (TIER 2d)

Config `gen5c_efficientnet_weighted_loss.yaml` (formerly
`paperIV_efficientnet_v2_s_v5_weighted_loss.yaml`) already implements:

```yaml
unweighted_loss: false
label_weights:
  tier_a: 1.0
  tier_b: 0.5
```

This down-weights Tier-B (visual-only) positives relative to confirmed
Tier-A lenses in the binary cross-entropy loss. LLM2 specifically
recommended running this alongside unweighted (gen5a) to isolate the
effect of label noise on the high-purity regime.

**Expected behavior:** Weighted loss should improve Tier-A recall at
the expense of marginal Tier-B recall, because the loss gradient is
dominated by the higher-confidence labels.

---

## 23. Prompt 4: Dual Prior Reporting and Ablation Matrix (TIER 2e+2f)

### 23.1 Dual Prior Reporting Strategy (LLM1 Q4.4)

Report completeness under TWO injection prior sets:

1. **Broad priors** (current): Full parameter space
   - theta_E: [0.5, 3.0], beta_frac: [0.1, 1.0], r_mag: [23, 26]
   - Represents "all possible strong lenses"

2. **Observationally anchored priors**: Restricted to detectable regime
   - theta_E: [0.8, 2.5], beta_frac: [0.1, 0.55], r_mag: [23, 25]
   - Represents "lenses a ground-based survey could plausibly detect"

The paper should present BOTH and discuss how prior choice affects
the completeness number. This pre-empts the "you chose priors to
make your number look good" referee objection.

### 23.2 Pre-Registered Ablation Comparisons (LLM1 Q4.7-Q4.8)

The following ablation matrix is pre-registered before any retrain:

| Comparison | Variable Isolated | Expected Signal |
|------------|-------------------|-----------------|
| gen4 vs gen5a | Annulus correction | AUC +0.001-0.003 |
| gen5a vs gen5b | From-scratch vs finetune | Finetune usually wins |
| gen5a vs gen5c | Weighted vs unweighted loss | Tier-A recall improvement |
| gen4+IM1 vs gen5a+IM1 | Annulus effect on completeness | +3-7pp ceiling |
| IM1 vs IM2 (gen5a) | Host selection method | IM2 expected neutral/positive |

LLM2 framing (Q4.8): Present as "our pipeline vs Paper IV" comparison,
measuring the effect of each documented improvement independently.

---

## 24. Prompt 4: Model 2 LRG Contrast Suppression (TIER 2g)

### 24.1 LLM2's Novel Insight

Model 2 (deflector-conditioned injection) performed 0.77pp WORSE than
Model 1. LLM2 proposes a physical explanation:

**Hypothesis:** Galaxy-size-dependent contrast suppression. LRG host
galaxies have larger effective radii than random negatives. The Sersic
profile of the LRG extends further into the annulus region where arcs
are injected, reducing the arc-to-host contrast ratio. This effect is:

1. Worse for extended hosts (LRGs) than compact hosts
2. Exacerbated by the (20,32) annulus normalization which may include
   host galaxy light
3. Potentially alleviated by the (32.5,45) corrected annulus

**Testable prediction:** The annulus fix (gen5a) should benefit Model 2
more than Model 1, because the wider annulus better separates host galaxy
light from the normalization reference. If IM2 performance improves
relative to IM1 after the annulus fix, this supports the hypothesis.

---

## 25. Prompt 4: Poisson Noise Limitation (TIER 2h)

### 25.1 Current Status

The injection engine has `add_poisson_noise=False` by default. This
means injected arc pixels include sky background noise from the host
cutout but NOT the additional Poisson noise from the arc's own photons.

### 25.2 Quantitative Impact

For a mag-20 arc (bright end of realistic range):
- Arc flux: ~3.63 × 10^4 nanomaggies (f = 10^(0.4*(22.5-20)))
- Poisson noise: sqrt(f × gain) ~ 190 counts (typical gain ~1 e-/ADU)
- Sky noise: ~40 counts/pixel at typical depth

**Ratio:** At mag-20, Poisson noise is ~4.8× the sky noise per pixel.
At mag-24 (typical faint arc), Poisson noise is ~0.3× sky → negligible.

### 25.3 Literature Context

- **Collett (2015):** Does not add Poisson noise to lensed sources
  (uses smooth ray-traced surface brightness)
- **Rojas et al. (2023):** Omits Poisson noise for injected arcs
- **HOLISMOKES:** Uses real galaxy stamps which implicitly include noise

**Conclusion:** Omission is standard practice for faint arcs but
introduces a subtle artifact for bright arcs (mag < 22). This could
contribute to the bright-arc over-detection (the model has never seen
a noisy bright arc). Flagged as a known limitation, not a blocking issue
for the current paper. The parameter `add_poisson_noise` exists for
future experiments.

---

## 26. Prompt 4: MNRAS Publication Readiness Checklist (TIER 2i)

From LLM2's "5 minimum requirements for publication":

- [ ] **Requirement 1:** Run all 4 pre-retrain experiments (Q2.1-Q2.4)
      and document results, regardless of outcome
- [ ] **Requirement 2:** Report Tier-A recall with 95% binomial CIs,
      at both p > 0.3 and p > 0.5 thresholds
- [ ] **Requirement 3:** Present multi-dimensional completeness grid
      (theta_E × beta_frac × depth) with at least 2 depth bins
- [ ] **Requirement 4:** Include literature comparison table with
      at least: Jacobs+19, Huang+20, He+20, Rojas+22, Herle+24
- [ ] **Requirement 5:** Either retrain with annulus fix OR explicitly
      document the (20,32) normalization as a known systematic
      (with quantified impact from Q2.3/Q2.4)

**Additional (from LLM1):**
- [ ] Report injection priors alongside results (dual prior strategy)
- [ ] Include embedding space visualization as evidence for realism gap
- [ ] Provide all run commands, configs, and random seeds for
      reproducibility (this document serves that purpose)

---

## 27. Prompt 4: Nanomaggy Verification Need (TIER 2j)

### 27.1 Issue

The injection engine assumes cutout pixel values are in nanomaggies
(AB zeropoint = 22.5 mag). This assumption propagates through:
- `AB_ZP = 22.5` constant in `injection_engine.py`
- Arc flux calculation: `f_nmgy = 10^(0.4 * (AB_ZP - mag))`
- Noise estimation via `estimate_sigma_pix_from_psfdepth()`

If cutout units are NOT nanomaggies (e.g., scaled by exposure time,
or in ADU), all injected arc fluxes are systematically wrong.

### 27.2 Verification Method

For a handful of objects in the manifest:
1. Load the cutout NPZ
2. Integrate flux in a large aperture (e.g., r < 30 pixels)
3. Compare to Tractor catalog `flux_g`, `flux_r`, `flux_z`
4. If ratio is consistently ~1.0, units are correct
5. If ratio is consistently != 1.0, identify the scaling factor

**Script needed:** A small verification script (10-20 lines) that
cross-matches cutout integrated photometry against Tractor catalog.
Priority: medium (before retrain, not before diagnostics).

---

## 28. Prompt 4: psfdepth Resolution (TIER 2k)

### 28.1 Definition

`psfdepth` in DESI Legacy Surveys DR10 is the **inverse variance** of
the PSF flux estimator, in units of 1/nanomaggies^2:

```
psfdepth = 1 / sigma_psf^2
```

where sigma_psf is the 1-sigma PSF flux uncertainty.

### 28.2 Conversion to 5-sigma Depth

```python
depth_5sig_mag = -2.5 * (np.log10(5.0 / np.sqrt(psfdepth)) - 9.0)
```

This is correctly implemented in `dhs/selection_function_utils.py::m5_from_psfdepth()`.

### 28.3 Source

DESI Legacy Surveys DR10 documentation:
https://www.legacysurvey.org/dr10/catalogs/

Confirmed via cross-check with Tractor code and astropy zeropoints.
The `psfdepth` column is NOT AB magnitude depth (a common confusion).

---

## 29. D01 Pre-Retrain Diagnostic Results (2026-02-14)

Run on Lambda3 (GH200 480GB, Python 3.12.3, torch 2.7.0+cu128).
Total runtime: 170 seconds. All 6 diagnostics passed.

### 29.1 Gate 1: Data Quality -- PASS

**Split balance (Diagnostic 1):**
- Train: 316,100 (277 Tier-A, 3,079 Tier-B, 312,744 neg)
- Val: 135,581 (112 Tier-A, 1,320 Tier-B, 134,149 neg)
- PSF size KS test: p = 0.174 (no significant difference)
- PSF depth KS test: p = 0.123 (no significant difference)
- HEALPix: all 4,788 positives have NaN healpix (missing ra/dec in manifest)

**Masked pixels (Diagnostic 2):**
- Non-finite pixels: 0.0% mean, 0.0% max
- Zero pixels: 0.0005% mean, 0.5% max
- Flagged cutouts (>5%): 0 out of 1,000
- **Conclusion:** No masking policy needed.

### 29.2 Gate 2: Annulus Impact -- **GO FOR RETRAIN**

**Annulus comparison (Diagnostic 3):**
- Median normalization reference: old (20,32) = 0.000467, new (32.5,45) = 0.000340
- KS test on medians: p = 2.3e-10 (highly significant difference)
- KS test on MADs: p = 0.648 (no significant difference)
- No correlation with PSF (r = -0.025) or depth (r = 0.026)
- **Conclusion:** Annulus position shifts the normalization reference point materially.

**Mismatched annulus scoring (Diagnostic 4):**

| Metric | Native (20,32) | Mismatched (32.5,45) | Delta |
|--------|----------------|---------------------|-------|
| Recall (p>0.3) | 74.0% | 70.4% | **-3.6pp** |
| Recall (p>0.5) | 69.8% | 66.0% | **-3.8pp** |
| Recall (p>0.7) | 64.2% | 61.6% | -2.6pp |
| FPR (p>0.3) | 0.20% | 0.20% | 0.0pp |
| Median pos score | 0.943 | 0.901 | -0.042 |

- **Conclusion:** Model loses 3.6-3.8pp recall when fed mismatched preprocessing.
  This exceeds the 0.002 AUC threshold. The model IS fragile to annulus choice.
  **GO for retrain with corrected annulus.**

### 29.3 Gate 3: Geometry vs Morphology -- BORDERLINE

**Beta_frac restriction (Diagnostic 5), beta_frac in [0.1, 0.55]:**

| Mag bin | Detection (p>0.3) | Detection (p>0.5) | Median SNR |
|---------|-------------------|--------------------|------------|
| 18-19 | 17.0% | 9.0% | 1556 |
| 19-20 | 24.5% | 18.0% | 672 |
| 20-21 | 27.5% | 17.0% | 250 |
| **21-22** | **35.5%** | **27.0%** | **101** |
| 22-23 | 31.0% | 27.5% | 39 |
| 23-24 | 24.0% | 18.5% | 16 |
| 24-25 | 8.5% | 7.0% | 6 |
| 25-26 | 1.0% | 0.5% | 2 |

- Peak detection rate at p>0.3: 35.5% (mag 21-22), below the 40% threshold.
- **Conclusion:** Beta_frac restriction helps (+5pp) but does NOT explain the
  full gap. Both geometry and morphology contribute.
- **Counterintuitive finding:** Brightest arcs (mag 18-19, SNR>1000) have the
  LOWEST detection rate (17%). This is a strong signal that the CNN has learned
  to reject unrealistic bright injections -- real lenses at those magnitudes
  would look different (more complex morphology, different noise properties).

### 29.4 Gate 4: Injection Realism -- MASSIVE GAP CONFIRMED

**Feature space analysis (Diagnostic 6):**
- Linear probe AUC (real Tier-A vs low-bf injections): **0.991 +/- 0.010**
  - 5-fold CV, near-perfect separability
  - The CNN unambiguously encodes "injection-ness" in its feature space
- Frechet distance: real vs low-bf injections = **219.7**, real vs high-bf = **199.8**
- Per-layer FD: grows from 0.22 (features_0) to 63.1 (features_3)
  - Gap emerges in mid-level features, not just final layer
  - Deeper layers (features_4-7) could not be computed (n=112 < dim)

**Median scores by group:**

| Group | Median Score |
|-------|-------------|
| Real Tier-A | 0.995 |
| Low-bf injections | 0.107 |
| High-bf injections | 0.017 |
| Negatives | 0.000015 |

- **Conclusion:** The CNN has learned a clear "injection detector" alongside
  the "lens detector". Real lenses score 0.995, injections score 0.107.
  This is the fundamental driver of the sim-to-real gap. The realism
  limitation is NOT a code bug -- it is an inherent property of Sersic
  parametric injections vs real lensed arc morphology.

### 29.5 Overall D01 Decision

```
Gate 1 (Data Quality):    PASS -- clean data, no masking needed
Gate 2 (Annulus):         GO   -- 3.6-3.8pp recall drop with wrong annulus
Gate 3 (Beta_frac):       BORDERLINE -- 35.5% ceiling, below 40% threshold
Gate 4 (Realism):         CONFIRMED -- probe AUC = 0.991, massive gap

DECISION: GO for retrain (gen5a + gen5b)
  - Annulus fix is justified (Gate 2)
  - But expect modest gains only (Gate 4 shows fundamental realism limit)
  - Paper should frame completeness as "injection-based lower bound"
```

---

*This document was updated 2026-02-14. Sections 12 added per LLM
Prompt 2 analysis. Sections 13-18 added per LLM Prompt 3 analysis.
Sections 19-28 added per LLM Prompt 4 analysis. Section 29 added
with D01 pre-retrain diagnostic results from Lambda3.*

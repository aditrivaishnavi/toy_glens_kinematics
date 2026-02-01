# Gen 1: ResNet18 on v3_color_relaxed

**Status**: Completed (Failed - identified critical data issues)  
**Training Date**: 2026-01-28 to 2026-01-29  
**Platform**: Lambda Labs GH200 (96GB)

## Overview

| Property | Value |
|----------|-------|
| Model Architecture | ResNet18 (modified for 64×64 input) |
| Data Version | v3_color_relaxed |
| Grid | grid_small |
| Control Type | **Paired** (same galaxy, no injection) |
| PSF Model | Gaussian |
| Final tpr@fpr1e-4 | **0.4%** (FAILED) |
| Final fpr@tpr0.85 | **6.2%** |

## Critical Issues Identified

1. **Paired Controls**: Controls used the SAME galaxy as positives (just without injection). Model learned "is there extra flux?" rather than lens morphology.
2. **Unresolved Injections**: With theta_e = [0.3, 0.6, 1.0] and median PSF ~1.3", ~60% of injections had theta_e/PSF < 0.5 (unresolved).
3. **Gaussian PSF**: Real DECaLS PSFs have extended wings; Gaussian underestimates this.

## Data Preparation

### Phase 3: Parent Sample Selection (v3_color_relaxed)

```sql
-- Galaxy selection criteria
TYPE != 'PSF'           -- Extended sources only
flux_r > 0 AND flux_z > 0 AND flux_w1 > 0
z < 20.4                -- Magnitude cut
r - z > 0.4             -- LRG color cuts
z - W1 > 0.8
```

**Output**: ~145,000 LRG targets

### Phase 4a: Manifest Generation

**Grid: grid_small**
| Parameter | Values |
|-----------|--------|
| theta_e_arcsec | [0.3, 0.6, 1.0] |
| src_dmag | [1.0, 2.0] |
| src_reff_arcsec | [0.08, 0.15] |
| src_e | [0.0, 0.3] |
| shear | [0.0, 0.03] |

**Total configs**: 48

```bash
# Phase 4a command (EMR)
spark-submit \
  --deploy-mode cluster \
  spark_phase4_pipeline.py \
  --stage 4a \
  --output-s3 s3://darkhaloscope/phase4_pipeline \
  --variant v3_color_relaxed \
  --grid-train grid_small \
  --control-frac-train 0.5
```

### Phase 4c: Lens Injection

```bash
# Phase 4c command (EMR)
spark-submit \
  --deploy-mode cluster \
  spark_phase4_pipeline.py \
  --stage 4c \
  --output-s3 s3://darkhaloscope/phase4_pipeline \
  --variant v3_color_relaxed \
  --experiment-id train_stamp64_bandsgrz_gridgrid_small
```

## Data Transfer to Lambda

```bash
# On emr-launcher
aws s3 sync s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/stamps/train_stamp64_bandsgrz_gridgrid_small/ \
  /tmp/phase4c_v3/

# rclone to Lambda filesystem
rclone sync /tmp/phase4c_v3/ lambda-ohio:darkhaloscope-training/phase4c/
```

## Training

**Script**: `phase5_train_fullscale_gh200.py`

**Hyperparameters**:
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Batch size | 512 |
| Epochs | 10 |
| Loss | BCEWithLogitsLoss |
| Mixed precision | bf16 |
| Augmentation | Flip + Rot90 |

**Training Command**:
```bash
PYTHONUNBUFFERED=1 nohup python3 -u phase5_train_fullscale_gh200.py \
    --data /lambda/nfs/darkhaloscope-training/phase4c \
    --out_dir /lambda/nfs/darkhaloscope-training/phase5/models/resnet18_v1 \
    --model resnet18 \
    --epochs 10 \
    --batch_size 512 \
    --lr 3e-4 \
    --amp_dtype bf16 \
    > /tmp/resnet18_train.log 2>&1 &
```

## Evaluation Results

**Training Progression**:
| Epoch | Train Loss | Val Loss | Val AUROC |
|-------|-----------|----------|-----------|
| 1 | 0.2831 | 0.1824 | 0.9762 |
| 5 | 0.0892 | 0.0734 | 0.9941 |
| 10 | 0.0612 | 0.0589 | 0.9963 |

**FPR vs Completeness (Final Model)**:
| Completeness (TPR) | FPR | log₁₀(FPR) |
|--------------------|-----|------------|
| 99.0% | 4.14e-01 | -0.38 |
| 95.0% | 1.88e-01 | -0.73 |
| 90.0% | 1.02e-01 | -0.99 |
| **85.0%** | **6.17e-02** | **-1.21** |
| 80.0% | 3.97e-02 | -1.40 |
| 70.0% | 1.79e-02 | -1.75 |

**Completeness at Fixed FPR**:
| Target FPR | Actual Completeness (TPR) |
|------------|---------------------------|
| 1e-2 | 74.7% |
| 1e-3 | 35.2% |
| **1e-4** | **0.4%** |
| 1e-5 | ~0% |

## S3 Locations

```
Data: s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/stamps/train_stamp64_bandsgrz_gridgrid_small/
Checkpoints: s3://darkhaloscope/phase5/models/colab/
```

## Lessons Learned

1. **High AUROC (0.996) is misleading** - dominated by easy cases
2. **Paired controls allow trivial shortcuts** - must use different galaxies
3. **Most injections unresolved** - need theta_e >= 0.5" for morphology learning
4. **Always report FPR at fixed completeness** - this is the operational metric


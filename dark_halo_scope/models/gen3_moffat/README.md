# Gen 3: ConvNeXt-Tiny on v4_sota_moffat (Full Patch)

**Status**: Preparing  
**Target Training Date**: 2026-01-31 (after Gen2 completes + data ready)  
**Platform**: Lambda Labs GH200 (96GB) - Washington DC

## Overview

| Property | Value |
|----------|-------|
| Model Architecture | ConvNeXt-Tiny |
| Data Version | v4_sota_moffat |
| Grid | grid_sota |
| Control Type | Unpaired |
| PSF Model | **Moffat** (beta=3.5) |
| Loss Function | **Focal Loss** |
| Training Strategy | **Curriculum** (strict resolved → full) |
| Hard Negatives | Yes (from Gen2) |
| Target tpr@fpr1e-4 | >85% |

## Key Improvements Over Gen2

| Aspect | Gen2 | Gen3 | Rationale |
|--------|------|------|-----------|
| PSF Model | Gaussian | **Moffat** | Better sim-to-real match |
| Loss | BCE | **Focal** | Focuses on hard examples |
| Training | Single stage | **Curriculum** | Learn resolved first |
| Hard Negatives | No | **Yes** | Push FPR down |
| Worker Sharding | Bug present | **Fixed** | No sample duplication |
| Metadata Guard | None | **Active** | Prevent label leakage |

## Patched Code Features

### Training Script (`phase5_train_fullscale_gh200_v2.py`)
- Multi-worker sharding fix: shards by (rank, worker_id)
- Forbidden metadata guard: blocks arc_snr, injection params
- Focal loss support: `--loss focal --focal_alpha 0.25 --focal_gamma 2.0`
- Curriculum support: `--min_theta_over_psf`, `--min_arc_snr`
- Resume capability: `--resume /path/to/checkpoint.pt`

### Inference Script (`phase5_infer_scores_v2.py`)
- Metadata fusion support (matches training)
- Streaming parquet with PyArrow

### Evaluation Script (`phase5_eval_stratified_fpr.py`)
- Stratified FPR by theta_e bins and theta_e/PSF bins
- Syntax errors fixed

### Hard Negative Mining (`phase5_mine_hard_negatives.py`)
- Streaming + BallTree for efficient mining
- Excludes known lens positions

## Data Preparation

### Phase 3: Parent Sample Selection

Same as Gen1/Gen2 - uses v3_color_relaxed parent sample (~145K LRGs)

### Phase 4a: Manifest Generation

Reuses v4_sota manifests (already has extended grid + unpaired controls)

### Phase 4c: Lens Injection with Moffat PSF

```bash
# Phase 4c command (EMR - with Moffat PSF)
spark-submit \
  --deploy-mode cluster \
  --driver-memory 8g \
  --executor-memory 18g \
  --executor-cores 4 \
  --num-executors 100 \
  --conf spark.sql.parquet.compression.codec=gzip \
  spark_phase4_pipeline.py \
  --stage 4c \
  --output-s3 s3://darkhaloscope/phase4_pipeline \
  --variant v4_sota_moffat \
  --manifests-subdir manifests \
  --psf-model moffat \
  --moffat-beta 3.5 \
  --experiment-id train_stamp64_bandsgrz_gridgrid_sota
```

**Key Change**: `--psf-model moffat --moffat-beta 3.5`

## Data Transfer to Lambda

```bash
# On emr-launcher: Download v4_sota_moffat stamps
aws s3 sync s3://darkhaloscope/phase4_pipeline/phase4c/v4_sota_moffat/stamps/train_stamp64_bandsgrz_gridgrid_sota/ \
  /data/staging/v4_sota_moffat/

# rclone to Lambda DC filesystem
rclone sync /data/staging/v4_sota_moffat/ lambda-dc:darkhaloscope-training-dc/phase4c_v4_sota_moffat/ \
  --transfers 16 --checkers 8 --progress
```

## Training

### Pre-Training: Hard Negative Mining from Gen2

```bash
# Run inference with Gen2 model on parent sample
python3 phase5_infer_scores_v2.py \
    --data /lambda/nfs/darkhaloscope-training-dc/phase4c_v4_sota \
    --ckpt /lambda/nfs/darkhaloscope-training-dc/runs/pathb_v4sota_convnext/best.pt \
    --out /lambda/nfs/darkhaloscope-training-dc/scores/gen2_parent_scores.parquet \
    --split train \
    --batch_size 1024 \
    --use_bf16

# Mine top-50K hard negatives
python3 phase5_mine_hard_negatives.py \
    --scores /lambda/nfs/darkhaloscope-training-dc/scores/gen2_parent_scores.parquet \
    --out /lambda/nfs/darkhaloscope-training-dc/hard_negs/gen2_top50k.parquet \
    --topk 50000 \
    --min_score 0.90
```

### Stage A1: Strict Resolved Curriculum

**Hyperparameters**:
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-2 |
| Dropout | 0.10 |
| Batch size | 256 |
| Epochs | 6 |
| Loss | **Focal (alpha=0.25, gamma=2.0)** |
| Mixed precision | bf16 |
| Augmentation | Flip + Rot90 |
| min_theta_over_psf | **0.80** |
| min_arc_snr | **7.0** |
| Metadata | psfsize_r, psfdepth_r |

**Training Command (Stage A1)**:
```bash
PYTHONUNBUFFERED=1 nohup python3 -u phase5_train_fullscale_gh200_v2.py \
    --data /lambda/nfs/darkhaloscope-training-dc/phase4c_v4_sota_moffat \
    --out_dir /lambda/nfs/darkhaloscope-training-dc/runs/gen3_stageA1 \
    --arch convnext_tiny \
    --epochs 6 \
    --batch_size 256 \
    --lr 3e-4 \
    --weight_decay 1e-2 \
    --dropout 0.10 \
    --use_bf16 \
    --augment \
    --loss focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --min_theta_over_psf 0.80 \
    --min_arc_snr 7.0 \
    --meta_cols psfsize_r,psfdepth_r \
    > /tmp/gen3_stageA1.log 2>&1 &
```

### Stage A2: Relaxed (Resume from A1)

**Changes from A1**:
| Parameter | A1 Value | A2 Value |
|-----------|----------|----------|
| Learning rate | 3e-4 | **1e-4** |
| min_theta_over_psf | 0.80 | **0.50** |
| min_arc_snr | 7.0 | **3.0** |
| Resume | - | **A1 best.pt** |

**Training Command (Stage A2)**:
```bash
PYTHONUNBUFFERED=1 nohup python3 -u phase5_train_fullscale_gh200_v2.py \
    --data /lambda/nfs/darkhaloscope-training-dc/phase4c_v4_sota_moffat \
    --out_dir /lambda/nfs/darkhaloscope-training-dc/runs/gen3_stageA2 \
    --arch convnext_tiny \
    --epochs 6 \
    --batch_size 256 \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --dropout 0.10 \
    --use_bf16 \
    --augment \
    --loss focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --min_theta_over_psf 0.50 \
    --min_arc_snr 3.0 \
    --meta_cols psfsize_r,psfdepth_r \
    --resume /lambda/nfs/darkhaloscope-training-dc/runs/gen3_stageA1/best.pt \
    > /tmp/gen3_stageA2.log 2>&1 &
```

## Evaluation

### Stratified FPR Evaluation

```bash
python3 phase5_eval_stratified_fpr.py \
    --scores /lambda/nfs/darkhaloscope-training-dc/scores/gen3_test_scores.parquet \
    --out_csv /lambda/nfs/darkhaloscope-training-dc/results/gen3_stratified_fpr.csv \
    --theta_bins 0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5 \
    --res_bins 0.5,0.7,0.9,1.1,1.4,2.0 \
    --tpr_targets 0.80,0.85,0.90,0.95 \
    --fpr_targets 1e-5,1e-4,1e-3,1e-2
```

## S3 Locations

```
Data: s3://darkhaloscope/phase4_pipeline/phase4c/v4_sota_moffat/stamps/train_stamp64_bandsgrz_gridgrid_sota/
Checkpoints: /lambda/nfs/darkhaloscope-training-dc/runs/gen3_stageA1/
             /lambda/nfs/darkhaloscope-training-dc/runs/gen3_stageA2/
```

## Expected Performance

Based on prior improvements:
| Metric | Gen1 | Gen2 (Epoch 2) | Gen3 (Target) |
|--------|------|----------------|---------------|
| tpr@fpr1e-4 | 0.4% | 77.8% | **>85%** |
| fpr@tpr0.85 | 6.2% | 0.17% | **<0.1%** |

## Files in This Directory

| File | Purpose |
|------|---------|
| `phase5_train_fullscale_gh200_v2.py` | Patched training script with all fixes |
| `phase5_infer_scores_v2.py` | Inference script with metadata fusion |
| `phase5_eval_stratified_fpr.py` | Stratified FPR evaluation |
| `phase5_mine_hard_negatives.py` | Hard negative mining with BallTree |


# Gen 2: ConvNeXt-Tiny on v4_sota

**Status**: RERUN PENDING (all bugs fixed)  
**Original Training Date**: 2026-01-31  
**Bug Fix Date**: 2026-02-01  
**Platform**: Lambda Labs GH200 (96GB) - Washington DC

## Overview

| Property | Value |
|----------|-------|
| Model Architecture | ConvNeXt-Tiny |
| Data Version | v4_sota |
| Grid | grid_sota |
| Control Type | **Unpaired** (different galaxies) |
| PSF Model | Gaussian |
| Original tpr@fpr1e-4 (peak) | **79.8%** (epoch 3) |
| Original tpr@fpr1e-4 (final) | **0.0%** (collapsed) |

## Bug Analysis Summary

The original Gen2 training collapsed due to multiple bugs identified by LLM1 and LLM2:

### Confirmed Bugs (Now Fixed)

| ID | Bug | Impact | Status |
|----|-----|--------|--------|
| A1 | Worker sharding ignores worker_id | **8x sample duplication** per epoch | FIXED |
| A2 | ROC tie bug: keeps FIRST of tied run | Metric truncation, false 0% TPR | FIXED |
| A3 | ROC missing (0,0) origin | Edge case failures | FIXED |
| A4 | No early stopping | 8 wasted epochs of overfitting | FIXED |
| A6 | Shuffle repeats every epoch | Reduced stochasticity | FIXED |
| E1 | No forbidden metadata guard | Risk of label leakage | FIXED |
| C1/C4 | No calibration collapse detection | Silent failure | FIXED |
| B5 | No top-k score logging | Hard to diagnose issues | FIXED |
| D1 | Full-image normalization leaks injection | Shortcut learning | FIXED (optional outer annulus) |

### Key Finding: "0.0%" Was Rounding

LLM1 identified that `tpr@fpr1e-4 = 0.0%` was actually **1/62910 = 0.00159%** rounded down.
This means exactly 1 positive consistently scored above the threshold - not a complete failure.

## Key Improvements Over Gen1

| Aspect | Gen1 | Gen2 | Improvement |
|--------|------|------|-------------|
| Control Type | Paired | **Unpaired** | Prevents trivial shortcut |
| theta_e range | 0.3-1.0" | **0.5-2.5"** | 80%+ resolved |
| Total configs | 48 | **1,008** | Better coverage |
| Compression | Snappy | **Gzip** | ~40% smaller |

## Data Preparation

### Phase 3: Parent Sample Selection

Same as Gen1 - uses v3_color_relaxed parent sample (~145K LRGs)

### Phase 4a: Manifest Generation

**Grid: grid_sota** (Extended for Resolved Lenses)
| Parameter | Values |
|-----------|--------|
| theta_e_arcsec | [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5] |
| src_dmag | [0.5, 1.0, 1.5, 2.0] |
| src_reff_arcsec | [0.06, 0.10, 0.15, 0.20] |
| src_e | [0.0, 0.2, 0.4] |
| shear | [0.0, 0.02, 0.04] |

**Total configs**: 1,008

```bash
# Phase 4a command (EMR)
spark-submit \
  --deploy-mode cluster \
  --driver-memory 8g \
  --executor-memory 18g \
  spark_phase4_pipeline.py \
  --stage 4a \
  --output-s3 s3://darkhaloscope/phase4_pipeline \
  --variant v4_sota \
  --grid-train grid_sota \
  --control-frac-train 0.5
```

### Phase 4c: Lens Injection

```bash
# Phase 4c command (EMR - with gzip compression)
spark-submit \
  --deploy-mode cluster \
  --driver-memory 8g \
  --executor-memory 18g \
  --executor-cores 4 \
  --num-executors 50 \
  --conf spark.sql.parquet.compression.codec=gzip \
  spark_phase4_pipeline.py \
  --stage 4c \
  --output-s3 s3://darkhaloscope/phase4_pipeline \
  --variant v4_sota \
  --manifests-subdir manifests \
  --experiment-id train_stamp64_bandsgrz_gridgrid_sota
```

## Training (RERUN WITH FIXES)

**Script**: `phase5_train_fullscale_gh200_v2.py` (patched version)

### Fixes Applied

| Fix ID | Description |
|--------|-------------|
| A1 | Worker sharding: `shard = rank * num_workers + worker_id` |
| A2+A3 | ROC curve: keep LAST of tied run, prepend (0,0) origin |
| A4 | Early stopping: `--early_stopping_patience 3` |
| A6 | Epoch-dependent shuffle via `set_epoch(epoch)` |
| E1 | Forbidden metadata guard blocks label-leaking columns |
| C1/C4 | Calibration collapse warning when >50% scores are binary |
| B5 | Log top-50 neg scores and bottom-50 pos scores |
| D1 | Optional `--norm_method outer` for outer annulus normalization |

### Hyperparameters (Rerun)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-2 |
| Batch size | 512 |
| Epochs | 8 (was 12) |
| Early stopping patience | 3 |
| Loss | BCEWithLogitsLoss |
| Mixed precision | bf16 |
| Augmentation | Flip + Rot90 |
| min_theta_over_psf | 0.0 (no filtering) |
| norm_method | full (legacy, consider outer) |

### Training Command (Rerun)

```bash
PYTHONUNBUFFERED=1 nohup python3 -u phase5_train_fullscale_gh200_v2.py \
    --data /lambda/nfs/darkhaloscope-training-dc/phase4c_v4_sota \
    --out_dir /lambda/nfs/darkhaloscope-training-dc/runs/gen2_fixed \
    --arch convnext_tiny \
    --epochs 8 \
    --batch_size 512 \
    --lr 3e-4 \
    --weight_decay 1e-2 \
    --use_bf16 \
    --augment \
    --early_stopping_patience 3 \
    > /tmp/gen2_fixed.log 2>&1 &
```

## Original Training History (Buggy)

| Epoch | tpr@fpr1e-4 | tpr@fpr0.001 | fpr@tpr0.85 | train_loss |
|-------|-------------|--------------|-------------|------------|
| 0 | 69.8% | 83.1% | 0.11% | 0.083 |
| 1 | 78.8% | 81.5% | 0.18% | 0.023 |
| 2 | 77.8% | 82.2% | 0.17% | 0.011 |
| **3** | **79.8%** | **85.4%** | **0.08%** | 0.007 |
| 4 | 77.7% | 86.1% | 0.06% | 0.005 |
| 5 | 0.0%* | 85.2% | 0.10% | 0.003 |
| 6 | 0.0%* | 85.5% | 0.09% | 0.002 |
| 7 | 0.0%* | 0.0%* | 0.22% | 0.001 |
| 8-11 | 0.0%* | 0.0%* | ~0.2% | <0.001 |

*Note: "0.0%" is actually 1/62910 = 0.00159% due to rounding

## Root Cause Analysis

1. **Worker Sharding Bug (A1)**: 8x sample duplication per epoch caused rapid overfitting
2. **ROC Tie Bug (A2)**: Metric truncation made tpr@fpr1e-4 appear as 0%
3. **No Early Stopping (A4)**: Training continued 8 epochs past optimal point
4. **Shuffle Repetition (A6)**: Same shuffle order every epoch reduced diversity

## S3 Locations

```
Data: s3://darkhaloscope/phase4_pipeline/phase4c/v4_sota/stamps/train_stamp64_bandsgrz_gridgrid_sota/
Original Checkpoints: /lambda/nfs/darkhaloscope-training-dc/runs/pathb_v4sota_convnext/
Rerun Checkpoints: /lambda/nfs/darkhaloscope-training-dc/runs/gen2_fixed/
```

## Verification: Unpaired Controls Confirmed

```python
# Analysis of v4_sota data:
Total rows analyzed: 48,690
Unique control galaxy positions: 2,715
Unique positive galaxy positions: 2,770
Overlapping positions: 0

# CONFIRMED: Different galaxies for controls vs positives
# LLM1 clarified: This is random assignment, not 1:1 pairing
```

## Lessons Learned

1. **Worker sharding is critical** - Missing worker_id causes N*duplication
2. **ROC tie handling matters** - Binary scores break naive implementations
3. **Early stopping is essential** - Prevents wasted compute and overfitting
4. **Epoch-dependent shuffle** - With persistent_workers, seed must vary
5. **"0%" often means rounding** - 1/62910 = 0.00159% rounds to 0.0%
6. **tpr@fpr1e-4 is controlled by ~7 negatives** - Very sensitive metric

## Files in This Directory

| File | Purpose |
|------|---------|
| `phase5_train_fullscale_gh200_v2.py` | **Patched** training script with all fixes |
| `README.md` | This documentation |

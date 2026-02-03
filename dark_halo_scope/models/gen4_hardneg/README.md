# Gen 4: ConvNeXt-Tiny with Hard Negative Mining

**Status**: Ready for Training  
**Platform**: Lambda Labs GH200 (96GB) - Washington DC

## Overview

| Property | Value |
|----------|-------|
| Model Architecture | ConvNeXt-Tiny |
| Data Version | v4_sota_moffat (from Gen3) |
| Control Type | Unpaired |
| PSF Model | Moffat (beta=3.5) |
| Loss Function | Focal Loss |
| **Key Innovation** | Hard Negative Mining from Gen2/Gen3 |
| Target tpr@fpr1e-4 | >90% |

## Key Improvements Over Gen3

| Aspect | Gen3 | Gen4 | Rationale |
|--------|------|------|-----------|
| Hard Negatives | No | **Yes (40k)** | Focus learning on difficult cases |
| Hard Neg Weight | N/A | **5x-10x** | Increase importance of hard cases |
| Training Data | Base only | **Base + Weighted HN** | Reduce FPR at operating point |

## Hard Negative Strategy

Hard negatives are false positives from Gen2/Gen3 inference (controls scored >0.9):
- **Source**: Gen2 (20k) + Gen3 (20k) = 40k candidates
- **Deduplication**: By (ra, dec) position → ~30-35k unique
- **Weighting**: 5x weight during training

The training script:
1. Loads the base training data (v4_sota_moffat)
2. Creates a lookup set of hard negative (ra, dec) positions
3. During iteration, checks if each sample matches a hard negative
4. Hard negatives are yielded multiple times (weighted sampling)

## Training Script

The Gen4-specific training script is `phase5_train_gen4_hardneg.py` which:
- Extends the base training script
- Adds `--hard_neg_path` argument for hard negative parquet
- Adds `--hard_neg_weight` argument for upsampling factor
- Implements position-based matching for hard negative identification

## Training Command

```bash
PYTHONUNBUFFERED=1 nohup python3 -u phase5_train_gen4_hardneg.py \
    --data /lambda/nfs/darkhaloscope-training-dc/phase4c_v4_sota_moffat \
    --out_dir /lambda/nfs/darkhaloscope-training-dc/models/gen4_hardneg \
    --hard_neg_path /lambda/nfs/darkhaloscope-training-dc/hard_negatives/merged/hard_neg_lookup.parquet \
    --hard_neg_weight 5 \
    --arch convnext_tiny \
    --epochs 50 \
    --batch_size 256 \
    --lr 3e-4 \
    --weight_decay 1e-2 \
    --dropout 0.1 \
    --use_bf16 \
    --augment \
    --loss focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --min_theta_over_psf 0.5 \
    --norm_method outer \
    --meta_cols psfsize_r,psfdepth_r \
    --early_stopping_patience 5 \
    > /tmp/gen4_train.log 2>&1 &
```

## Evaluation

After training, run:
```bash
# Inference
python3 phase5_infer_scores_v2.py \
    --data /lambda/nfs/darkhaloscope-training-dc/phase4c_v4_sota_moffat \
    --ckpt /lambda/nfs/darkhaloscope-training-dc/models/gen4_hardneg/ckpt_best.pt \
    --out /lambda/nfs/darkhaloscope-training-dc/scores/gen4_round0_test_scores.parquet \
    --split test \
    --arch convnext_tiny \
    --batch_size 1024 \
    --use_bf16

# Stratified FPR evaluation
python3 phase5_eval_stratified_fpr.py \
    --scores /lambda/nfs/darkhaloscope-training-dc/scores/gen4_round0_test_scores.parquet \
    --out_csv /lambda/nfs/darkhaloscope-training-dc/eval/gen4_round0_stratified_fpr.csv
```

## Expected Performance

Based on hard negative mining literature:
| Metric | Gen3 | Gen4 (Target) |
|--------|------|---------------|
| tpr@fpr1e-4 | 69.7% | **>85%** |
| tpr@fpr1e-3 | 88.8% | **>95%** |
| fpr@tpr0.90 | 0.15% | **<0.05%** |

## Files

| File | Purpose |
|------|---------|
| `phase5_train_gen4_hardneg.py` | Training script with hard neg support |
| `README.md` | This file |


# Phase 5 Smoke Test Review Request

## Context
We are building a CNN-based lens finder for the Dark Halo Scope project, targeting MNRAS/ApJ/AAS submission. This is our first training run to validate the pipeline before full training.

## Training Environment
- Platform: Google Colab Pro (T4 GPU, 16GB VRAM)
- Framework: PyTorch 2.x
- Data source: S3 bucket (darkhaloscope)

---

## Critical Issue Discovered

### Debug Tier Class Distribution
| Metric | Debug Tier | Train Tier |
|--------|------------|------------|
| is_control=0 (injections) | 100% | 48.6% |
| is_control=1 (controls) | **0%** | 51.4% |

**The debug tier was generated with ONLY injections (no controls).** This is a data generation bug in Phase 4c.

### Implication
The smoke test trained on data with only ONE class, meaning:
- The model learned to predict 1 (lens) for everything
- AUROC is undefined (nan) because AUROC requires both classes
- The decreasing loss is meaningless - it's just the model getting better at predicting 1

---

## Raw Training Output

```
Using device: cuda
Train files: 45, Val files: 5
Train samples: 16512, Val samples: 768
Train class balance (sample): 1000.0/1000 positives  ← ALL POSITIVES!
Val class balance (sample): 500.0/500 positives      ← ALL POSITIVES!
⚠️ WARNING: Validation set has only ONE class! AUROC will be undefined.

Epoch 1: Train Loss=0.1561, Val Loss=0.0075, AUROC=nan
Epoch 2: Train Loss=0.0037, Val Loss=0.0018, AUROC=nan
Epoch 3: Train Loss=0.0012, Val Loss=0.0008, AUROC=nan
...
Epoch 10: Train Loss=0.0001, Val Loss=0.0001, AUROC=nan

🎉 Training complete!
   Best Val Loss: {best_loss:.4f}  ← Also has a formatting bug
```

---

## S3 Verification Results

### Debug Tier (what we used for smoke test)
```
Path: s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/stamps/debug_stamp64_bandsgrz_gridgrid_small/
Files: 51 parquet files
Size: ~800 MB
Sample file analysis:
  Total rows: 96
  is_control=0 (injections): 96
  is_control=1 (controls): 0
  Class balance: 100.0% injections
```

### Train Tier (what we should use for full training)
```
Path: s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/stamps/train_stamp64_bandsgrz_gridgrid_small/
Files: 6,001 parquet files
Size: ~470 GB
Sample file analysis:
  Total rows: 2470
  is_control=0 (injections): 1200
  is_control=1 (controls): 1270
  Class balance: 48.6% injections, 51.4% controls  ← CORRECT!
```

---

## Training Code Summary

### Model Architecture
- ResNet-18 style CNN
- Input: 3 channels (g, r, z bands), 64x64 pixels
- Output: Binary classification (lens vs control)
- Loss: BCEWithLogitsLoss
- Optimizer: AdamW (lr=3e-4, weight_decay=0.01)
- Scheduler: CosineAnnealingLR

### Data Pipeline
- Parquet files with `stamp_npz` (compressed 3-band images) and `is_control` column
- Normalization: Per-stamp median/MAD robust normalization
- Augmentation: Random flips (H/V) and 90° rotations
- Label mapping: `is_control=1 → label=0` (control), `is_control=0 → label=1` (injection)

### Key Code Snippet (Label Logic)
```python
# Label: is_control=1 means no lens (negative), is_control=0 means injection (positive)
is_control = table['is_control'][row_idx].as_py()
label = 0.0 if is_control == 1 else 1.0
```

---

## Questions for Review

### 1. Data Generation Bug
The debug tier has 0 controls. Is this:
- A bug in Phase 4c that only affected the debug tier?
- Expected behavior (debug tier intentionally only has injections)?
- A filtering issue in the S3 path we used?

**Verified fact**: The train tier has correct 50/50 split.

### 2. Can We Proceed to Full Training?
Given that:
- The train tier has correct class balance (48.6% / 51.4%)
- The training code appears correct (validated via loss decreasing, checkpoint saving)
- The data loading works (17k+ samples loaded successfully)

Is it safe to proceed to full training on the train tier?

### 3. Training Metrics Interpretation
With only one class:
- Loss decreased from 0.1561 → 0.0001 (expected for trivial prediction)
- AUROC = nan (expected - undefined for one class)
- Model saved based on val_loss (fallback when AUROC unavailable)

Does this indicate any bugs in the training code itself, or is the behavior correct given the broken data?

### 4. Full Training Concerns

For full training on ~470GB / ~10M rows:
1. **Colab limitations**: Can Colab Pro handle streaming from S3 or do we need to download all data first?
2. **Training time estimate**: With ~2 min/epoch on 17k samples, what's the expected time for 10M samples?
3. **Memory concerns**: Is the current per-file parquet loading efficient enough?
4. **Checkpointing**: Current code saves locally then uploads to S3 - is this robust for long runs?

### 5. Model Architecture
Is ResNet-18 appropriate for:
- 64x64 input size (small images)?
- Binary classification task?
- ~5M injections + ~5M controls?

Should we consider smaller/larger architectures?

### 6. Publication Readiness
For MNRAS/ApJ submission:
1. What AUROC should we target?
2. What additional metrics should we track (precision, recall, F1, ROC curves)?
3. Should we implement cross-validation or is train/val/test split sufficient?
4. What visualizations do we need (sample predictions, attention maps, error analysis)?

---

## Code Bug Found

The final print statement has a formatting error:
```python
print(f"   Best AUROC: {best_auroc:.4f}" if best_auroc > 0 else "   Best Val Loss: {best_loss:.4f}")
```

Should be:
```python
print(f"   Best AUROC: {best_auroc:.4f}" if best_auroc > 0 else f"   Best Val Loss: {best_loss:.4f}")
```

---

## GO/NO-GO Decision Request

Based on the above information:
1. **Should we proceed to full training on the train tier?**
2. **Are there any blocking issues in the training code?**
3. **What fixes (if any) should be applied before full training?**

Please provide a clear GO or NO-GO decision with justification.


# Gen 2 Training Debug Package

## How to Use This Package

1. Read this README first for context
2. Review `history.json` for training metrics
3. Review `phase5_train_fullscale_gh200_v2.py` for the training code
4. Review `gen2_training_bug_analysis.md` for our current analysis

---

## Project Background

We are building a **CNN-based gravitational lens finder** for the Dark Energy Spectroscopic Instrument (DESI) Legacy Imaging Survey (DECaLS DR10). 

**Goal**: Detect strong gravitational lenses in galaxy images for publication in MNRAS/ApJ/AAS.

**Pipeline**:
```
Phase 3: Select LRG parent sample from DECaLS (~2M galaxies)
    ↓
Phase 4a: Create injection manifest (which galaxies get synthetic lenses)
    ↓
Phase 4c: Inject synthetic lensed arcs into real galaxy cutouts
    ↓
Phase 5: Train CNN to classify lens vs non-lens
```

---

## Data Description

**Dataset**: v4_sota (stored in S3 as Parquet)
- ~1.3M training samples
- 50% positives (galaxies WITH injected synthetic lenses)
- 50% controls (galaxies WITHOUT injection)
- 64x64 pixel stamps in g, r, z bands (3-channel input like RGB)
- Each sample stored as `stamp_npz` (compressed numpy array in Parquet)

**CRITICAL DATA ISSUE - PAIRED CONTROLS**:
- Controls are the SAME galaxy as positives, just without the lens injection
- This means for each positive (galaxy + lens), there's a control (same galaxy, no lens)
- The model might learn "is there extra flux?" instead of "is there an arc?"

**Validation Set**:
```
n_eval: 128,000 samples
pos_eval: 62,910 (positives with injected lenses)
neg_eval: 65,090 (controls without injection)
```

---

## Model Architecture

**ConvNeXt-Tiny** with modifications:
- Input: 64x64x3 (g, r, z bands)
- First conv layer kernel adapted from 4x4 stride 4 to smaller stride for 64x64 input
- Backbone outputs 768-dim features
- Classification head: Linear(768, 1) for binary output
- Optional metadata fusion (PSF size, depth) via MLP

---

## Training Configuration

```python
arch = "convnext_tiny"
epochs = 12
batch_size = 256
lr = 3e-4
weight_decay = 1e-2
dropout = 0.1
num_workers = 8  # BUG: causes 8x sample duplication per epoch
optimizer = AdamW
scheduler = CosineAnnealingLR(T_max=12, eta_min=1e-6)
loss = FocalLoss(alpha=0.25, gamma=2.0)
amp_dtype = bfloat16
augmentation = dihedral (rot90 + flips)
```

---

## The Problem

Training shows **severe performance collapse** after epoch 4.

### Metrics Over Time (see history.json for full data)

| Epoch | tpr@fpr1e-4 | tpr@fpr0.001 | fpr@tpr0.50 | fpr@tpr0.85 | train_loss |
|-------|-------------|--------------|-------------|-------------|------------|
| 0 | 69.8% | 83.1% | 0.00% | 0.11% | 0.083 |
| 1 | 78.8% | 81.5% | 0.00% | 0.18% | 0.023 |
| 3 | **79.8%** | 85.4% | 0.003% | 0.08% | 0.007 |
| 4 | 77.7% | 86.1% | 0.003% | 0.06% | 0.005 |
| 5 | **0.0%** | 85.2% | **0.07%** | 0.10% | 0.003 |
| 6 | 0.0% | 85.5% | 0.08% | 0.09% | 0.002 |
| 7 | 0.0% | **0.0%** | **0.22%** | 0.22% | 0.001 |
| 11 | 0.0% | 0.0% | 0.26% | 0.26% | 0.0001 |

### Key Observations

1. **tpr@fpr1e-4 cliff**: Drops from 79.8% → 0% between epoch 4 and 5
2. **tpr@fpr0.001 cliff**: Drops from 85% → 0% between epoch 6 and 7
3. **fpr@tpr0.50 spike**: Jumps from 0.003% → 0.07% at epoch 5
4. **Binary score distribution**: At epoch 7, fpr@tpr0.50 = fpr@tpr0.70 = fpr@tpr0.85 = 0.22%
5. **Train loss → 0**: Drops to 0.0001, indicating severe overfitting

### What the Metrics Mean

- `tpr@fpr1e-4`: True Positive Rate when False Positive Rate = 0.01% (only ~6 false positives allowed out of 65,090 negatives)
- `fpr@tpr0.50`: False Positive Rate needed to capture 50% of positives
- When `fpr@tpr0.50 = fpr@tpr0.85`, it means the score distribution is nearly binary

---

## Our Current Hypotheses

### 1. PAIRED CONTROLS [Suspected PRIMARY Cause]

**Evidence**:
- `fpr@tpr0.50` jumps from 0% to 0.07% at epoch 5
- This means ~46 negatives now score as high as positives
- Paired controls let model learn shortcuts ("extra flux") not morphology ("arc shape")

**Mechanism**:
- Model memorizes training pairs
- On validation, some control galaxies have features associated with "lens" in training
- These controls get high scores, pushing threshold up
- At FPR=1e-4, threshold becomes so high only 1 positive exceeds it

### 2. WORKER SHARDING BUG [Suspected ACCELERATOR]

**The Bug** (line 221 in training script):
```python
def _iter_fragments(self) -> List[ds.Fragment]:
    ...
    return frags[rank()::world()]  # BUG: doesn't account for DataLoader workers
```

**Problem**:
- With `num_workers=8`, each worker calls `_iter_fragments()` independently
- All 8 workers return the SAME fragments
- Result: Each sample seen 8x per epoch instead of 1x

**Effect**:
- 12 epochs = 96 effective epochs
- Overfitting accelerated by 8x

### 3. NO EARLY STOPPING [ENABLER]

- Best model was at epoch 3 (tpr@fpr1e-4 = 79.8%)
- Training continued for 8 more epochs
- No mechanism to stop when metrics plateaued

---

## Files in This Package

| File | Description |
|------|-------------|
| `README.md` | This file - full context |
| `history.json` | Complete training metrics for all 12 epochs |
| `phase5_train_fullscale_gh200_v2.py` | Training script with suspected bugs |
| `gen2_training_bug_analysis.md` | Our detailed analysis document |

---

## What We Need From You

1. **Confirm or refute** our hypotheses with evidence from the code and metrics

2. **Explain the exact mechanism** of the tpr@fpr1e-4 cliff drop (why epoch 4→5 specifically?)

3. **Identify any bugs** in `phase5_train_fullscale_gh200_v2.py` we haven't found

4. **Is the evaluation code correct?** Check `roc_curve_np()` and `tpr_at_fpr()` functions

5. **Recommend fixes** beyond what we've identified

---

## Key Code Sections to Review

### Evaluation Functions (lines 152-194)
```python
def roc_curve_np(scores, y):
    # Custom ROC curve implementation
    
def tpr_at_fpr(scores, y, fpr_targets):
    # Finds TPR at specific FPR thresholds
```

### Data Loading (lines 211-230)
```python
class ParquetStreamDataset(IterableDataset):
    def _iter_fragments(self):
        # BUG: Worker sharding issue here
```

### Training Loop (lines 570-660)
```python
for epoch in range(start_epoch, args.epochs):
    # No early stopping mechanism
```


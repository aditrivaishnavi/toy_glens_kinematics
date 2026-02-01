# Debug Request: Gen 2 Training Performance Collapse

## Project Context

We are building a CNN-based gravitational lens finder for the Dark Energy Spectroscopic Instrument (DESI) Legacy Imaging Survey (DECaLS DR10). The goal is to detect strong gravitational lenses in galaxy images for submission to MNRAS/ApJ/AAS.

**Data Pipeline**:
- Phase 3: Select LRG (Luminous Red Galaxy) parent sample from DECaLS
- Phase 4a: Create injection manifest (which galaxies get synthetic lenses)
- Phase 4c: Inject synthetic lensed arcs into real galaxy cutouts, produce 64x64 stamps
- Phase 5: Train CNN to classify lens vs non-lens

**Current Data (v4_sota)**:
- ~1.3M training samples (50% positives with injected lenses, 50% controls)
- 64x64 pixel stamps in g, r, z bands (3-channel input)
- Stored as Parquet with `stamp_npz` column containing compressed numpy arrays

## Model Architecture

ConvNeXt-Tiny with modifications:
- Input: 64x64x3 (modified from ImageNet 224x224)
- First conv layer adapted for 64x64 input
- Output: 1 logit for binary classification
- Optional metadata fusion (PSF size, depth) via MLP head

## The Problem

Training shows severe performance collapse after epoch 4:

| Epoch | tpr@fpr1e-4 | tpr@fpr0.001 | fpr@tpr0.85 | train_loss |
|-------|-------------|--------------|-------------|------------|
| 0 | 69.8% | 83.1% | 0.11% | 0.083 |
| 1 | 78.8% | 81.5% | 0.18% | 0.023 |
| 3 | **79.8%** | 85.4% | 0.08% | 0.007 |
| 5 | **0.0%** | 85.2% | 0.10% | 0.003 |
| 7 | 0.0% | **0.0%** | 0.22% | 0.001 |
| 11 | 0.0% | 0.0% | 0.26% | 0.0001 |

**Key Observations**:
1. `tpr@fpr1e-4` drops from 79.8% to 0% between epoch 4 and 5
2. `tpr@fpr0.001` drops from 85% to 0% between epoch 6 and 7
3. Train loss drops to 0.0001 indicating severe overfitting
4. `fpr@tpr0.50` increases from 0% to 0.07% at epoch 5, meaning some negatives now score as high as positives

## What We Think Is Happening

**Evidence from metrics**:
- At epoch 7: `fpr@tpr0.50 = fpr@tpr0.70 = fpr@tpr0.85 = 0.22%`
- This means the score distribution is nearly binary
- ~85% of positives score above some threshold, ~15% score below ALL negatives

**Our Hypothesis (ordered by contribution)**:

1. **PAIRED CONTROLS [PRIMARY]**
   - Controls are the SAME galaxy as positives, just without injection
   - Model may learn "is there extra flux?" instead of "is there an arc?"
   - Evidence: Some negatives score as high as positives after overfitting

2. **WORKER SHARDING BUG [ACCELERATOR]**
   - Found bug: `_iter_fragments()` returns `frags[rank::world]` regardless of DataLoader worker
   - With `num_workers=8`, each sample is seen 8x per epoch
   - Effect: 12 epochs = 96 effective epochs, accelerating overfitting 8x

3. **NO EARLY STOPPING [ENABLER]**
   - Best model was at epoch 3, training continued to epoch 11
   - No mechanism to stop when performance plateaued

## Training Configuration

```python
epochs = 12
batch_size = 256
lr = 3e-4
weight_decay = 1e-2
dropout = 0.1
num_workers = 8  # BUG: causes 8x duplication
optimizer = AdamW
scheduler = CosineAnnealingLR(T_max=12, eta_min=1e-6)
loss = FocalLoss(alpha=0.25, gamma=2.0)
amp_dtype = bfloat16
```

## Validation Set Stats

```
n_eval: 128,000
pos_eval: 62,910 (positives with injected lenses)
neg_eval: 65,090 (controls without injection)
```

## Questions for Debug

1. **Is our paired controls hypothesis correct?** How can we verify this from the code/data?

2. **What explains the sudden cliff-like drop** from 79.8% to 0% between epoch 4 and 5? This isn't gradual degradation.

3. **Is there a bug in the evaluation code** (`roc_curve_np`, `tpr_at_fpr` functions)?

4. **Should we expect this behavior** with focal loss + cosine annealing + overfitting?

5. **What's the relationship** between train_loss approaching 0 and the metric collapse?

6. **Are there other bugs** in the training script we haven't identified?

## Files Attached

- `phase5_train_fullscale_gh200_v2.py` - Training script
- `history.json` - Full training history with all metrics
- `gen2_training_bug_analysis.md` - Our analysis document

## What We Need

1. Confirm or refute our hypotheses with evidence
2. Identify any bugs we missed in the training code
3. Explain the exact mechanism of the tpr@fpr1e-4 collapse
4. Recommend fixes (beyond what we've already done)


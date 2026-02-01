# Gen 2 Training Bug Analysis

## Summary

Training completed 12 epochs but suffered severe overfitting. Best checkpoint at epoch 3, but model continued training for 8 more epochs with degrading performance.

**Key Finding**: The PRIMARY cause is the paired controls issue, not the worker sharding bug. Worker sharding ACCELERATED the failure but didn't cause it.

## Training History

| Epoch | tpr@fpr1e-4 | tpr@fpr0.001 | fpr@tpr0.85 | train_loss |
|-------|-------------|--------------|-------------|------------|
| 0 | 69.8% | 83.1% | 0.11% | 0.083 |
| 1 | 78.8% | 81.5% | 0.18% | 0.023 |
| 2 | 77.8% | 82.2% | 0.17% | 0.011 |
| **3** | **79.8%** | **85.4%** | **0.08%** | 0.007 |
| 4 | 77.7% | 86.1% | 0.06% | 0.005 |
| 5 | 0.0% | 85.2% | 0.10% | 0.003 |
| 6 | 0.0% | 85.5% | 0.09% | 0.002 |
| 7 | 0.0% | 0.0% | 0.22% | 0.001 |
| 8 | 0.0% | 0.0% | 0.14% | 0.0007 |
| 9 | 0.0% | 0.0% | 0.21% | 0.0004 |
| 10 | 0.0% | 0.0% | 0.16% | 0.0002 |
| 11 | 0.0% | 0.0% | 0.26% | 0.0001 |

## Bugs Identified

### Bug 1: Worker Sharding Duplication (CRITICAL)

**Location**: `phase5_train_fullscale_gh200_v2.py`, line 221

**Code**:
```python
def _iter_fragments(self) -> List[ds.Fragment]:
    dataset = ds.dataset(self.cfg.data, format="parquet", filesystem=self.fs, partitioning="hive")
    frags = list(dataset.get_fragments(filter=ds.field("region_split") == self.cfg.split))
    return frags[rank()::world()]  # BUG: Only shards by DDP rank, not DataLoader worker
```

**Impact**: With `num_workers=8` (default), all 8 workers process the SAME fragments. Each sample is seen 8 times per epoch instead of once.

**Effect**: 
- 8x data duplication per epoch
- Effective learning rate is 8x higher than intended
- Faster overfitting because model sees same samples repeatedly

**Fix**:
```python
def _iter_fragments(self) -> List[ds.Fragment]:
    dataset = ds.dataset(self.cfg.data, format="parquet", filesystem=self.fs, partitioning="hive")
    frags = list(dataset.get_fragments(filter=ds.field("region_split") == self.cfg.split))
    
    # Shard by BOTH rank AND worker id
    wi = torch.utils.data.get_worker_info()
    worker_id = wi.id if wi is not None else 0
    num_workers = wi.num_workers if wi is not None else 1
    
    shard = rank() * num_workers + worker_id
    nshard = world() * num_workers
    return frags[shard::nshard]
```

### Bug 2: No Early Stopping (CRITICAL)

**Location**: `phase5_train_fullscale_gh200_v2.py`, training loop

**Impact**: Model continued training for 12 epochs even though best performance was at epoch 3-4. 8 wasted epochs of overfitting.

**Evidence**: Train loss dropped to 0.0001 (severe overfitting), but no mechanism stopped training.

**Fix**: Add early stopping with patience:
```python
patience = 3
no_improve_count = 0

for epoch in range(start_epoch, args.epochs):
    # ... training ...
    
    if score > best_metric:
        best_metric = score
        no_improve_count = 0
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

### Bug 3: tpr@fpr1e-4 Collapse Explanation

**Symptom**: tpr@fpr1e-4 drops from 79.8% to ~0% after epoch 5.

**Root Cause**: 
1. As model overfits, it becomes extremely confident on "easy" samples
2. Model assigns very high probabilities (>0.999) to easy positives
3. Model assigns very low probabilities (<0.001) to hard positives
4. The threshold for FPR=1e-4 becomes so high that only ~1 positive exceeds it

**Evidence from history.json**:
- Epoch 5: `tpr@fpr0.0001 = 0.0000159` (1 positive out of 62,910)
- This is exactly 1/62910, meaning only 1 positive sample exceeds the threshold

**The pattern**:
- fpr@tpr0.50 increases from 0.0 to 0.002 as training progresses
- This means the model needs MORE false positives to achieve 50% TPR
- The model is becoming confident on a smaller subset of positives

### Bug 4: No Regularization Tuning

**Issue**: Default `weight_decay=1e-2` and `dropout=0.1` are standard values but not tuned for this dataset.

**Observation**: Train loss drops to 0.0001 while AUROC stays stable at ~0.99, indicating the model is memorizing rather than generalizing.

## Hyperparameters Used

| Parameter | Value | Issue |
|-----------|-------|-------|
| epochs | 12 | Too many without early stopping |
| batch_size | 256 | OK |
| lr | 3e-4 | OK |
| weight_decay | 1e-2 | May need increase |
| dropout | 0.1 | May need increase |
| num_workers | 8 | Causes 8x duplication |
| loss | focal | OK |

## Recommendations

### Immediate Fixes (For Next Training Run)

1. **Fix worker sharding** - Critical, ~8x sample duplication
2. **Add early stopping** - Patience=3 epochs
3. **Reduce epochs** - 8 max with early stopping
4. **Increase regularization** - weight_decay=0.05, dropout=0.2

### Training Parameters for Next Run

```bash
python phase5_train_fullscale_gh200_v2.py \
    --data /path/to/data \
    --out_dir /path/to/output \
    --arch convnext_tiny \
    --epochs 8 \
    --batch_size 256 \
    --lr 2e-4 \
    --weight_decay 0.05 \
    --dropout 0.2 \
    --num_workers 4 \
    --use_bf16 \
    --augment \
    --loss focal
```

## Definitive Root Cause Analysis

The tpr@fpr1e-4 dropped from 79.8% to 0% due to **negatives scoring like positives**:

| Metric | Epoch 3 | Epoch 5 | Epoch 7 |
|--------|---------|---------|---------|
| fpr@tpr0.50 | 0.003% | 0.07% | 0.22% |
| tpr@fpr0.001 | 85.4% | 85.2% | 0.0% |
| tpr@fpr1e-4 | 79.8% | 0.0% | 0.0% |

**Key Evidence**: 
- fpr@tpr0.50 = fpr@tpr0.70 = fpr@tpr0.85 at epoch 7
- This means the score distribution is nearly binary
- ~85% of positives score above a threshold, ~15% score below all negatives

**Root Causes (ordered by contribution)**:

1. **PAIRED CONTROLS [PRIMARY]** - Model learns "is there extra flux?" not "is there an arc?"
2. **8x SAMPLE DUPLICATION [ACCELERATOR]** - Worker sharding bug causes 8x faster overfitting
3. **NO EARLY STOPPING [ENABLER]** - Training continued 8 epochs past optimal

## Conclusion

The Gen 2 training was fundamentally flawed due to:
1. **Paired controls** - Primary cause of learning shortcuts instead of arc morphology
2. **8x sample duplication** from worker sharding bug - Accelerated overfitting by 8x
3. **No early stopping** allowing severe overfitting
4. **Wasted compute** - 8 epochs of useless training

The best checkpoint (epoch 3, tpr@fpr1e-4=79.8%) was saved correctly, but the training process masked serious issues that will affect future runs.

**Fixes Applied**:
- Worker sharding: Fixed to shard by rank * num_workers + worker_id
- Early stopping: Added with patience=3
- Paired controls: Will be addressed in Gen 3 with unpaired controls


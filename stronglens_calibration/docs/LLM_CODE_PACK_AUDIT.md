# Audit: LLM-Provided Paper IV Parity Code Pack

**Date:** 2026-02-11
**Source:** `paperIV_parity_course_correction/` (13 files)
**Reviewer:** Automated audit against Paper IV (arXiv:2508.20087v1) and LLM recommendations

---

## Files Reviewed

| File | Purpose | Verdict |
|------|---------|---------|
| `README.md` | Docs | Has wrong EfficientNet name (b0 vs v2_s) |
| `requirements.txt` | Deps | OK |
| `stronglens/io_npz.py` | Cutout loading | OK |
| `stronglens/preprocess.py` | Normalization + crop | MAD factor mismatch (1.4826x) |
| `stronglens/dataset.py` | DataLoader | Non-reproducible augmentation |
| `stronglens/train_utils.py` | Loss, predict, run_info | OK |
| `stronglens/metrics.py` | AUC, precision/recall | OK |
| `training/_common.py` | Shared training utils | OK |
| `training/train_paperIV_resnet.py` | ResNet training | Uses weighted loss, Adam not AdamW, no AMP |
| `training/train_paperIV_efficientnet.py` | EfficientNet training | WRONG MODEL (V1 not V2), same issues as ResNet |
| `tools/verify_manifest_and_splits.py` | Split verification | OK |
| `evaluation/bootstrap_metrics.py` | Bootstrap CIs | OK |
| `evaluation/fit_meta_learner.py` | Meta-learner | Wrong architecture, trains on eval data |

---

## CRITICAL Issues (3)

### C1. EfficientNet is V1, not V2

**File:** `training/train_paperIV_efficientnet.py` line 8

```python
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
```

**Problem:** Paper IV Section 3.2.2 states: "we integrate the EfficientNetV2" with 20,542,883 parameters. The code uses `efficientnet_b0` (EfficientNet V1, ~5.3M params). The LLM's own latest recommendation was `efficientnet_v2_s` (~21.5M params).

**Fix:** Change to:
```python
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
```

### C2. Loss is weighted even for parity

**Files:** `training/train_paperIV_resnet.py` line 67, `training/train_paperIV_efficientnet.py` line 68

```python
loss = weighted_bce_loss(logits, y, w) / accum
```

**Problem:** Paper IV Equation 1 is standard (unweighted) cross-entropy. Our manifest has `sample_weight` values of 0.5 for Tier-B positives. Using weighted loss changes the effective training signal.

**Fix:** Add `--unweighted` flag. When set, use `reduction='mean'` BCE and ignore `w`. Alternatively, set all weights to 1.0 in a parity manifest.

### C3. No 70/30 manifest generation

**Problem:** Paper IV uses 70/30 train/validation. Our manifest has 70/15/15. The code loads `split=="train"` (70%) and `split=="val"` (15%), leaving test (15%) unused. Training validates on only half the intended validation set.

**Fix:** Create `scripts/make_paperIV_baseline_manifest.py` that merges val+test into "val" and sets all weights to 1.0.

---

## IMPORTANT Issues (6)

### I1. Preprocessing MAD scaling factor mismatch

**File:** `stronglens/preprocess.py` line 32

```python
band = (band - med) / (1.4826 * mad)
```

**Our existing code** (`dhs/utils.py` line 40):
```python
return (img - med) / mad
```

**Problem:** The 1.4826 factor converts MAD to approximate Gaussian sigma. Our existing training used raw MAD. Using different normalization means the parity model sees different input distributions than our baseline.

**Fix:** Remove the 1.4826 factor to match our existing preprocessing, OR document it as a deliberate change and retrain all models consistently.

### I2. No AMP/mixed precision

**Problem:** Our existing `train.py` uses `torch.cuda.amp.GradScaler` and `autocast`. The LLM code does not. For 101x101 inputs with effective batch 2048 via accumulation on single GPU, this means slower training and higher memory usage.

**Fix:** Add AMP support with `torch.cuda.amp.autocast` and `GradScaler`.

### I3. Meta-learner architecture is wrong

**File:** `evaluation/fit_meta_learner.py`

```python
clf = LogisticRegression(max_iter=1000)
```

**Problem:** Paper IV Section 3.2.3: "a simple one-layer neural network with 300 nodes." LogisticRegression is a linear model (3 parameters for 2 inputs). Paper IV's meta-learner has ~1,200 parameters (2->300->1 with biases).

**Fix:** Replace with `torch.nn.Sequential(nn.Linear(2, 300), nn.ReLU(), nn.Linear(300, 1))` trained with BCE loss.

### I4. Meta-learner trains and evaluates on same data

**File:** `evaluation/fit_meta_learner.py` lines 26-27

```python
clf.fit(X, y)
p = clf.predict_proba(X)[:,1]
```

**Problem:** Paper IV trains meta-learner on training set predictions and evaluates on validation. This code fits and evaluates on the same val set (overfitting).

**Fix:** Accept separate `--train-preds` and `--val-preds` arguments. Fit on train, evaluate on val.

### I5. Adam without weight decay

**Files:** Both training scripts

```python
optimizer = Adam(model.parameters(), lr=args.base_lr)
```

**Problem:** No weight_decay specified. Our existing code uses `AdamW` with `weight_decay=1e-4`.

**Fix:** Change to `AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-4)`.

### I6. Missing scripts

- No `negative_cleaning_highscore_filter.py` (recommended by LLM for negative pool cleaning)
- No `make_paperIV_baseline_manifest.py` (recommended for 70/30 parity manifest)
- No bottlenecked ResNet model (recommended to address 58x parameter gap)

---

## MINOR Issues (4)

### M1. StepLR possible off-by-one

The `LambdaLR` with `scheduler.step()` after each epoch means the LR halves at training epoch 81 (1-indexed), not epoch 80. Off by one epoch. Unlikely to affect results meaningfully.

### M2. Non-reproducible augmentation

`dataset.py` uses `np.random.rand()` (global state). With multi-worker DataLoader, augmentation order isn't deterministic across runs. Our existing code uses `np.random.default_rng(seed)` per sample.

### M3. No cutout_path in prediction outputs

`save_preds_parquet` saves y/logit/score/tier but not `cutout_path`. Meta-learner alignment relies on row order. Should include a join key for robustness.

### M4. Final-epoch predictions saved, not best-epoch

Lines 98-99 of `train_paperIV_resnet.py` predict from the current (final epoch) model, not the best-AUC checkpoint. For meta-learner input, should load `best.pt` and predict from that.

---

## What IS Correct

- Overall gradient accumulation structure is sound
- 160 epochs with no early stopping matches Paper IV
- StepLR halving approach is correct in principle
- 101x101 input by default (crop_size=0 -> None) is correct
- Seed setting covers random, numpy, torch, cuda
- History logging to JSONL per epoch
- Best-AUC checkpoint saving logic
- Bootstrap evaluation with CI computation
- Split verification script

---

## Integration Plan

All issues will be fixed during integration into `stronglens_calibration/dhs/`. The LLM code serves as a useful scaffold for the training loop, gradient accumulation, and evaluation pipeline, but requires significant corrections before it can produce Paper IV-comparable results.

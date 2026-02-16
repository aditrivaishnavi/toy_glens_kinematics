# Phase 1 Baseline Training Worklog

## Overview

This worklog documents the execution of Phase 1 baseline training for the Gen5-Prime lens classifier.
All metrics, decisions, and findings are recorded here for audit purposes.

---

## Session: 2026-02-06

### Environment Setup

| Parameter | Value |
|-----------|-------|
| **Machine** | Lambda (192.222.56.237) |
| **GPU** | NVIDIA GH200 480GB |
| **GPU Memory** | 97,871 MiB total |
| **CUDA Version** | 12.8 |
| **PyTorch Version** | 2.7.0 |
| **Python Version** | 3.12.3 |
| **Disk Space** | 3.9 TB available |

### Data Configuration

| Parameter | Value |
|-----------|-------|
| **S3 Path** | `s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_paired/` |
| **Local Path** | `/home/ubuntu/data/v5_cosmos_paired/` |
| **Train Files** | 10 (subset; full dataset has ~1000 files) |
| **Train Samples** | 14,480 |
| **Val Files** | 10 |
| **Val Samples** | 99,700 |
| **Test Files** | 10 |
| **Test Samples** | 88,626 |

**Decision**: Started training with subset (10 files per split) while full train data downloads in background. This allows us to validate the training pipeline before committing to full-scale training.

---

## Phase 0: Foundation Verification

**Timestamp**: 2026-02-06 09:15 UTC

### Split Integrity Check

| Split Pair | Overlapping Bricks | Status |
|------------|-------------------|--------|
| train vs val | 0 | ✅ PASS |
| train vs test | 0 | ✅ PASS |
| val vs test | 0 | ✅ PASS |

**Result**: No brick overlap between splits. Data leakage prevention verified.

### Paired Data Integrity Check

| Check | Passed | Failed | Status |
|-------|--------|--------|--------|
| shape_match | 100 | 0 | ✅ PASS |
| not_identical | 100 | 0 | ✅ PASS |
| no_nan_inf | 100 | 0 | ✅ PASS |
| value_range | 100 | 0 | ✅ PASS |
| expected_shape | 100 | 0 | ✅ PASS |

**Result**: All 100 sampled rows passed all integrity checks.

---

## Phase 1: Environment Validation

**Timestamp**: 2026-02-06 09:20 UTC

### GPU Health Check
- GPU visible: ✅
- Memory free: 97,871 MiB
- Temperature: 34°C
- CUDA available: ✅

### Python Environment Check
- PyTorch: 2.7.0 ✅
- scikit-learn: 1.4.1 ✅
- torchvision: 0.22.0 ✅
- All imports: ✅

---

## Phase 2: Code Verification

**Timestamp**: 2026-02-06 09:21 UTC

### Syntax Check
All Phase 1 files passed syntax verification:
- `phase1_baseline/data_loader.py` ✅
- `phase1_baseline/model.py` ✅
- `phase1_baseline/train.py` ✅
- `phase1_baseline/validate_baseline.py` ✅

### Unit Tests
| Test Class | Passed | Failed | Status |
|------------|--------|--------|--------|
| TestDecodeStampNpz | 3 | 0 | ✅ |
| TestNormalization | 3 | 0 | ✅ |
| TestAzimuthalShuffle | 3 | 0 | ✅ |
| TestCoreDropout | 2 | 0 | ✅ |
| TestModelForward | 3 | 0 | ✅ |
| TestValidationFunctions | 3 | 0 | ✅ |
| **TOTAL** | **17** | **0** | ✅ |

**Fix Applied**: Updated `test_azimuthal_shuffle_preserves_total_flux` to use relative tolerance instead of absolute tolerance for float32 precision.

### Contract Tests
| Test Class | Passed | Failed | Status |
|------------|--------|--------|--------|
| TestConstantsContract | 5 | 0 | ✅ |
| TestSchemaContract | 5 | 0 | ✅ |
| TestCrossPhaseConsistency | 3 | 0 | ✅ |
| TestSchemaValidation | 6 | 0 | ✅ |
| **TOTAL** | **19** | **0** | ✅ |

---

## Phase 3: Training Preflight

**Timestamp**: 2026-02-06 09:22 UTC

### Data Loader Validation
| Check | Result |
|-------|--------|
| shape_correct | ✅ True |
| no_nan_inf | ✅ True |
| labels_binary | ✅ True |
| value_range_ok | ✅ True |
| batches_checked | 3 |

### Model Validation
| Check | Result |
|-------|--------|
| forward_shape_ok | ✅ True |
| backward_no_nan | ✅ True |
| memory_ok | ✅ True |
| memory_mb | 225.27 |

### Gradient Flow Test
| Metric | Value |
|--------|-------|
| Input shape | (32, 3, 64, 64) |
| Label shape | (32, 1) |
| Initial loss | 0.7025 |
| NaN gradients | 0 |

---

## Phase 4: Baseline Training Execution

**Timestamp**: 2026-02-06 09:23 UTC

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Architecture** | ResNet18 (pretrained) |
| **Input Channels** | 3 (g, r, z bands) |
| **Stamp Size** | 64x64 |
| **Epochs** | 50 |
| **Batch Size** | 128 |
| **Learning Rate** | 1e-4 |
| **Weight Decay** | 1e-4 |
| **Hard Negative Ratio** | 0.4 |
| **Core Dropout Prob** | 0.5 |
| **Mixed Precision** | True |
| **Early Stopping Patience** | 15 |
| **Gate Eval Frequency** | Every 5 epochs |
| **Seed** | 42 |

### Preflight Results (During Training)
- Train loader: ✅ PASS
- Val loader: ✅ PASS
- Model: ✅ PASS
- First batch: ✅ PASS

### Training Progress

| Epoch | Train Loss | Val Loss | Val AUROC | Status |
|-------|------------|----------|-----------|--------|
| 1 | 0.0558 | 0.1653 | 0.9890 | ✅ Best model saved |
| 2 | 0.0187 | 0.1999 | 0.9891 | ✅ Best model saved |
| 3 | 0.0093 | 0.2589 | 0.9884 | ⚠️ Val loss increasing |
| 4 | 0.0084 | 0.1959 | 0.9905 | ✅ Best model saved |
| 5 | 0.0059 | 0.2349 | 0.9893 | 🔴 Gate eval - SHORTCUT DETECTED |
| 6 | 0.0051 | 0.2842 | 0.9866 | Val loss spike |
| 7 | 0.0050 | 0.2394 | 0.9899 | Recovered |
| 8 | 0.0037 | 0.2433 | 0.9904 | Slight improvement |
| 9 | 0.0040 | 0.3268 | 0.9888 | Val loss spike |
| 10 | 0.0033 | 0.2406 | 0.9913 | 🔴 Gate eval - SHORTCUT PERSISTS |
| 15 | 0.0026 | 0.2776 | 0.9898 | Gate eval |
| 20 | 0.0015 | 0.2530 | 0.9906 | Gate eval |
| 23 | 0.0010 | 0.2496 | **0.9913** | ✅ **FINAL BEST MODEL** |
| 25 | 0.0012 | 0.2978 | 0.9893 | Gate eval |
| 30 | 0.0006 | 0.2809 | 0.9900 | Gate eval |
| 35 | 0.0004 | 0.3714 | 0.9886 | Gate eval |
| 38 | 0.0002 | 0.3242 | 0.9902 | 🛑 **EARLY STOPPING** (patience 15) |

### Epoch 5 Gate Evaluation (CRITICAL)

| Gate | Value | Threshold | Direction | Status |
|------|-------|-----------|-----------|--------|
| **AUROC (full)** | 0.9893 | 0.85 | > | ✅ PASS |
| **AUROC (core masked)** | 0.9880 | - | - | (info) |
| **Core masked drop** | 0.13% | 10% | < | ✅ PASS |
| **Core LR AUC** | **0.9497** | 0.65 | < | 🔴 **FAIL** |

### Epoch 10 Gate Evaluation

| Gate | Value | Threshold | Direction | Status |
|------|-------|-----------|-----------|--------|
| **AUROC (full)** | 0.9913 | 0.85 | > | ✅ PASS (best) |
| **AUROC (core masked)** | 0.9903 | - | - | (info) |
| **Core masked drop** | 0.10% | 10% | < | ✅ PASS (borderline!) |
| **Core LR AUC** | **0.9497** | 0.65 | < | 🔴 **FAIL** (unchanged) |

**Observation**: Core LR AUC is identical at epochs 5 and 10 (0.9497). The shortcut is baked into the training data and core dropout augmentation is not breaking it.

### ⚠️ CRITICAL FINDING: CORE BRIGHTNESS SHORTCUT DETECTED

**Problem**: Core LR AUC = 0.9497 means a simple logistic regression on the central 10x10 pixels achieves 95% AUC.

**Implication**: The model may be classifying based on central region brightness (lens galaxy + arc overlap) rather than arc morphology.

**Why this matters**:
1. Real lenses have bright cores (lens galaxy)
2. Arc injection adds signal that overlaps with the core
3. Model learns "bright core = lens" instead of "arc structure = lens"

**Why core_masked_drop passed (0.13%) but core_lr_auc failed (0.95)**:
- The model uses the full image, so masking the center doesn't hurt much (it learned arc features too)
- BUT the core alone is highly predictive, indicating the signal is concentrated there
- The model may be using a combination of core brightness AND arc features

**Next Steps**:
1. Continue training to see if this changes with more epochs
2. Consider increasing core_dropout_prob from 0.5 to 0.7 or 0.8
3. May need to investigate the training data for core brightness imbalance
4. Ablation without core_dropout should show even higher Core LR AUC

### Epoch 1 Detail
| Batch | Loss | Accuracy |
|-------|------|----------|
| 100 | 0.1094 | 95.68% |
| 200 | 0.0858 | 96.83% |
| 300 | 0.0711 | 97.40% |
| 400 | 0.0632 | 97.73% |
| 500 | 0.0573 | 97.97% |

**Observation**: Very high initial AUROC (0.989) and Core LR AUC (0.95) confirms the model is exploiting the core brightness shortcut despite 50% core dropout augmentation.

---

## Decisions Log

| Decision | Rationale | Date |
|----------|-----------|------|
| Started with data subset (10 files) | Validate pipeline before full-scale training | 2026-02-06 |
| Used local data instead of S3 streaming | Faster I/O, pandas S3 read works but Path.glob() doesn't | 2026-02-06 |
| Fixed test tolerance to use relative | Float32 precision causes absolute tolerance failures | 2026-02-06 |
| Background download of full train data | Training can proceed while data downloads | 2026-02-06 |
| Continued training despite Core LR FAIL | Document the shortcut, complete baseline for comparison | 2026-02-06 |
| Fixed torch.load for PyTorch 2.7 | Backward compatible fix for weights_only default change | 2026-02-06 |
| Early stopping triggered at epoch 38 | No improvement after epoch 23 for 15 epochs | 2026-02-06 |

---

## Issues and Fixes

### Issue 1: Test Tolerance Too Strict
- **Problem**: `test_azimuthal_shuffle_preserves_total_flux` failed with matching values (121.804 vs 121.804)
- **Root Cause**: Absolute tolerance (1e-5) too strict for float32 sums
- **Fix**: Changed to relative tolerance: `rel_diff = abs(orig - shuf) / (abs(orig) + 1e-10) < 1e-5`

### Issue 2: Tilde Expansion in Python
- **Problem**: `~/data/...` path not expanded in Python's `pathlib.Path`
- **Root Cause**: Python doesn't expand `~` automatically
- **Fix**: Use full path `/home/ubuntu/data/...`

### Issue 3: PyTorch 2.7 weights_only Default Change
- **Problem**: Final evaluation failed with `_pickle.UnpicklingError: Weights only load failed`
- **Root Cause**: PyTorch 2.7 changed `torch.load()` default from `weights_only=False` to `weights_only=True`
- **Impact**: Training completed successfully, but final eval script crashed
- **Fix**: Added `weights_only=False` to all `torch.load()` calls in planb codebase

---

## Files Modified This Session

| File | Change |
|------|--------|
| `planb/tests/test_core_functions.py` | Fixed flux conservation test tolerance |
| `planb/phase1_baseline/train.py` | Fixed torch.load() for PyTorch 2.7 (weights_only=False) |
| `planb/phase1_baseline/model.py` | Fixed torch.load() for PyTorch 2.7 (weights_only=False) |
| `planb/emr/jobs/spark_score_all.py` | Fixed torch.load() for PyTorch 2.7 (weights_only=False) |

---

## Phase 4 COMPLETED: Final Results

**Training Duration**: 09:23 - 11:15 UTC (1h 52m)
**Total Epochs**: 38 (early stopped at patience 15)
**Best Epoch**: 23

### Final Model Metrics

| Metric | Value | Gate Threshold | Status |
|--------|-------|----------------|--------|
| **Best Val AUROC** | 0.9913 | > 0.85 | ✅ PASS |
| **Core LR AUC** | 0.9497 | < 0.65 | 🔴 **FAIL** |
| **Core Masked Drop** | 0.08% | < 10% | ✅ PASS |
| **Final Train Loss** | 0.0002 | - | - |
| **Final Val Loss** | 0.3242 | - | - |

### Gate Evaluation History

| Epoch | AUROC | Core Masked | Core LR AUC | Status |
|-------|-------|-------------|-------------|--------|
| 5 | 0.9893 | 0.9880 (0.13%) | 0.9497 | FAIL |
| 10 | 0.9913 | 0.9903 (0.10%) | 0.9497 | FAIL |
| 15 | 0.9898 | 0.9878 (0.20%) | 0.9497 | FAIL |
| 20 | 0.9906 | 0.9895 (0.11%) | 0.9497 | FAIL |
| 25 | 0.9893 | 0.9878 (0.15%) | 0.9498 | FAIL |
| 30 | 0.9900 | 0.9894 (0.06%) | 0.9497 | FAIL |
| 35 | 0.9886 | 0.9879 (0.08%) | 0.9497 | FAIL |

**Key Observation**: Core LR AUC remained constant at 0.9497 throughout training. The shortcut is fundamental to the data, not a training artifact.

### Files Generated

| File | Path | Size |
|------|------|------|
| Best Model | `checkpoints/gen5_prime_baseline/best_model.pt` | 134 MB |
| Last Model | `checkpoints/gen5_prime_baseline/last_model.pt` | 45 MB |
| Training Log | `checkpoints/gen5_prime_baseline/training.log` | 30 KB |
| Config | `checkpoints/gen5_prime_baseline/config.json` | 334 B |

---

## Phase 5: Post-Training Analysis

### Root Cause Analysis: Core Brightness Shortcut

The Core LR AUC = 0.95 indicates that a simple logistic regression trained ONLY on the central 10x10 pixels (100 features) can classify lens vs non-lens with 95% accuracy. This is a severe shortcut.

**Hypothesis**: The arc injection process adds flux to the core region (where the lens galaxy is), making the core systematically brighter in lens images compared to non-lens images.

**Evidence**:
1. Core LR AUC is constant across all epochs (0.9497)
2. Core masked drop is very low (0.08-0.20%) - masking the core barely hurts the CNN
3. This suggests the CNN learned BOTH core brightness AND arc structure
4. The data itself contains the shortcut - it's not a model learning issue

**Implications for Real Data**:
- If the model relies on core brightness, it will fail on real lenses where brightness varies
- Real galaxy cores have varying brightness unrelated to lensing
- The model may have low precision on real data (many false positives from bright non-lens galaxies)

### Recommended Next Steps

1. **Investigate Training Data**: Analyze core brightness distribution for lens vs non-lens
2. **Increase Core Dropout**: Try 0.7 or 0.8 instead of 0.5
3. **Core Normalization**: Normalize core region independently before training
4. **Ablation Studies**: Complete no_hardneg, no_coredrop, minimal runs for comparison
5. **Data Augmentation**: Add random core brightness perturbation

---

## Important Metrics Gap (Identified 2026-02-06)

### Issue: Limited Checkpoint Retention

**Current behavior:**
- Only `best_model.pt` (best AUROC) saved
- Periodic checkpoints every 10 epochs (`checkpoint_epoch10.pt`, etc.)
- `last_model.pt` at training end

**Problem:** We may want to analyze models at different training stages, especially if:
- Core LR AUC changes over training (it didn't - stayed at 0.9497)
- AUROC is not the only metric we care about

**Recommendation:** Add `--save-all-epochs` flag for ablation studies.

### Issue: AUROC is Not Sufficient Metric

From `core_leakage_raw_data.md`, arc flux in core is θ_E dependent:
| θ_E Range | % Arc Flux in Core |
|-----------|-------------------|
| < 0.75" | 41% |
| 0.75-1" | 30% |
| 1-1.5" | 21% |
| 1.5-2" | 15% |
| >= 2" | 7.7% |

**Data confirms `theta_e_arcsec` is available in parquet (0.5" - 2.5").**

**Metrics we SHOULD be tracking but are NOT:**
1. **Stratified AUROC by θ_E bin** - critical for understanding shortcut impact
2. **Precision @ 0.5, 0.8, 0.9 thresholds** - for operational deployment
3. **Recall @ high precision (0.95)** - science use case
4. **Core LR AUC by θ_E bin** - to see if shortcut is θ_E dependent
5. **Hard negative AUROC** - performance on azimuthally shuffled controls

**Recommendation:** Add `--extended-metrics` flag that computes stratified performance.

### Why This Matters for Real Data

- Real lens θ_E distribution differs from training
- If model relies on core brightness for small-θ_E detection, it will fail on:
  - Bright galaxies without lenses (false positives)
  - Faint lenses with small θ_E (false negatives)
- Stratified metrics would reveal if model generalizes across θ_E range

---

## Phase 6: Ablation Studies (In Progress)

**Timestamp**: 2026-02-06 16:38 UTC

### Infrastructure Setup

| Instance | GPU | Purpose | Output Directory |
|----------|-----|---------|------------------|
| lambda | NVIDIA GH200 480GB | no_hardneg ablation | `~/checkpoints/ablation_no_hardneg/` |
| lambda2 | NVIDIA GH200 480GB | no_coredrop ablation | `~/darkhaloscope-training-dc/planb_outputs/ablation_no_coredrop/` |
| lambda3 | NVIDIA GH200 480GB | minimal ablation | `~/darkhaloscope-training-dc/planb_outputs/ablation_minimal/` |

### Data Setup

**Decision**: Copied training data from lambda local storage to NFS, then to lambda2/lambda3 local storage for optimal I/O performance.

| Path | Size | File Counts |
|------|------|-------------|
| NFS: `~/darkhaloscope-training-dc/planb_training_data/v5_cosmos_paired/` | 128 GB | 1000/10/10 (train/val/test) |
| lambda2 local: `~/data/v5_cosmos_paired/` | 128 GB | 1000/10/10 |
| lambda3 local: `~/data/v5_cosmos_paired/` | 128 GB | 1000/10/10 |

**Rationale for Local Storage**: Local SSD provides faster I/O than NFS during training. ~2.5 min to copy 128GB to NFS, then parallel copy to both instances.

### Ablation Configurations

| Ablation | hard_negative_ratio | core_dropout_prob | Purpose |
|----------|---------------------|-------------------|---------|
| **baseline** | 0.4 | 0.5 | Full mitigation stack |
| **no_hardneg** | 0.0 | 0.5 | Test hard negative contribution |
| **no_coredrop** | 0.3 | 0.0 | Test core dropout contribution |
| **minimal** | 0.0 | 0.0 | No mitigations (pure shortcut exposure) |

### Ablation Status (2026-02-06 16:40 UTC)

| Ablation | Instance | Status | Current Epoch | Best AUROC | Core LR AUC |
|----------|----------|--------|---------------|------------|-------------|
| no_hardneg | lambda | ⏳ Running | 19/50 | 0.9961 (E8) | 0.9497 |
| no_coredrop | lambda2 | ⏳ Running | 1/50 | - | - |
| minimal | lambda3 | ⏳ Running | 1/50 | - | - |

### Code Changes for Ablations

Modified `train.py` to:
1. Add `--save-every-epoch` flag for comprehensive checkpoint retention
2. Change default periodic checkpoint from every 10 epochs to every 5 epochs
3. Enhanced checkpoint data (includes optimizer state, val_auroc, train_loss)
4. Fixed PyTorch 2.7 `torch.load()` compatibility

### Missing Dependency Fix

**Issue**: lambda2 and lambda3 were missing `pyarrow` (required for parquet reading)
**Fix**: `pip install pyarrow` on both instances (pyarrow-23.0.0)

---

## Next Steps (Updated)

1. ✅ ~~Monitor training to epoch 5 for first gate evaluation~~ DONE
2. ✅ ~~Check Core LR AUC gate (target: < 0.65)~~ DONE - FAILED (0.95)
3. ✅ ~~Check Core Masked Drop gate (target: < 0.10)~~ DONE - PASSED (0.08%)
4. ✅ ~~Continue to epoch 50 or until early stopping~~ DONE - Stopped at epoch 38
5. ⏳ Run ablation experiments (no_hardneg, no_coredrop, minimal) - **IN PROGRESS**
6. ⏳ Aggregate and compare results - **WAITING**

---

## Data Fix: Core-Neutral Injection (2026-02-06 17:25 UTC)

### Problem
Core LR AUC = 0.9497 indicates severe core brightness shortcut baked into training data.

### Solution Implemented
Added `--core-mask-radius` argument to `spark_phase4_pipeline_gen5.py` that zeros out central pixels of injected arc before adding to background galaxy.

### Code Changes

| File | Change |
|------|--------|
| `emr/gen5/spark_phase4_pipeline_gen5.py` | Added `--core-mask-radius` argument |
| `emr/gen5/spark_phase4_pipeline_gen5.py` | Added `mask_core_flux()` helper function |
| `emr/gen5/spark_phase4_pipeline_gen5.py` | Applied mask at COSMOS injection site (line ~2668) |
| `emr/gen5/spark_phase4_pipeline_gen5.py` | Applied mask at Sersic injection site (line ~2719) |

### Next Steps
1. Generate pilot dataset (20k pairs) with `--core-mask-radius 5`
2. Compute Core LR AUC on pilot data
3. If < 0.65: regenerate full dataset
4. If >= 0.65: try larger radius (7 or 10)

### Documentation
- `planb/DATA_FIX_PLAN.md` - Detailed fix plan and rationale
- `planb/EXTERNAL_REVIEW_PROMPT.md` - External review request with full context

---

## BREAKTHROUGH: Residual Preprocessing Breaks Shortcuts (2026-02-06 19:10 UTC)

### Summary

After comprehensive testing, we discovered that **residual radial profile preprocessing** is the key to breaking the core brightness shortcut, NOT unpaired training alone.

### Gate Results Comparison (Cross-Validated, 5 seeds each)

| Configuration | Core AUC | Radial AUC | Status |
|---------------|----------|------------|--------|
| paired + raw | 0.9513 ± 0.02 | 0.9794 ± 0.01 | **FAIL** |
| unpaired + raw | 0.9488 ± 0.02 | 0.9782 ± 0.01 | **FAIL** |
| **paired + residual** | **0.5537 ± 0.02** | **0.4998 ± 0.01** | **PASS** |
| unpaired + residual | 0.6824 ± 0.02 | 0.4986 ± 0.01 | FAIL (borderline) |
| **unpaired + residual + core_dropout(r=7)** | **0.5113 ± 0.02** | **~0.50** | **PASS** |

### Key Insights

1. **Residual preprocessing is the key**: Subtracting the azimuthally-averaged radial profile removes the inner image brightness signal while preserving arc morphology.

2. **Paired + residual is actually better**: With paired data (same LRG), the residual subtraction perfectly cancels LRG-intrinsic variations. Different LRGs in unpaired data have residual profile differences.

3. **Unpaired + residual is borderline**: Mean Core AUC of 0.68 is close to 0.65 threshold. Adding core_dropout(r=7) during training pushes it below threshold.

4. **Three viable configurations**:
   - `paired + residual_radial_profile` - simplest, passes both gates
   - `unpaired + residual + core_dropout(r=7)` - more defensible, passes both gates
   - `unpaired + residual + core_dropout(r=10)` - most conservative, Core AUC ~0.50

### Residual Preprocessing Algorithm

```python
def residual_radial_profile(img):
    # 1. Normalize by outer annulus (r=20-32) median/MAD
    img_normalized = (img - median(outer)) / mad(outer)
    
    # 2. Compute azimuthal median at each radius
    for r in range(32):
        profile[r] = median(pixels_at_radius_r)
    
    # 3. Subtract radial profile model
    residual = img_normalized - profile_model
    return residual
```

### Implementation

New code created in `planb/unpaired_experiment/`:
- `preprocess.py` - `residual_radial_profile` and `raw_robust` modes
- `gates.py` - Cross-validated gates (fixes external LLM's train-on-train bug)
- `build_manifest.py` - LRG-disjoint unpaired manifest builder
- `data_loader.py` - Support for both paired and unpaired training
- `train.py` - Training script with gate evaluation

### Next Steps

1. Decide between paired+residual vs unpaired+residual+core_dropout
2. Run full training with chosen configuration
3. Evaluate on validation set with all gates

---

## Ablation Results: no_hardneg (2026-02-06 19:05 UTC)

The `no_hardneg` ablation finished with early stopping at epoch 28:
- **Val AUROC**: 0.9950
- **Best Epoch**: 13
- **Early Stopping Reason**: No improvement for 15 epochs

Note: This ablation used `raw` preprocessing, so the model still exploited the shortcut.

---

## Appendix: Gate Thresholds

| Gate | Threshold | Direction | Purpose |
|------|-----------|-----------|---------|
| AUROC_synth | 0.85 | > | Model must exceed |
| Core_LR_AUC | 0.65 | < | Shortcut detection (must be below) |
| Core_masked_drop | 0.10 | < | Center dependency (must be below) |
| Hardneg_AUROC | 0.70 | > | Hard negative performance |

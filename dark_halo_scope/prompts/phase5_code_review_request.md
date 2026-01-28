# Phase 5 Code Review Request: Training Pipeline Validation

## Context

We are building a lens-finder CNN for the Dark Halo Scope project, which searches for gravitational lenses in the DECaLS DR10 South survey. This is intended for publication in MNRAS/ApJ/AAS journals.

**Goal**: Validate that our Phase 5 training pipeline follows world-class model training best practices and will produce scientifically defensible results.

---

## Background: What Was Fixed From Previous LLM's Code

A previous LLM provided Phase 5 code that had several critical bugs. We fixed these issues:

### Bug 1: Image Data Format Incompatibility (CRITICAL)

**Problem**: The LLM's code expected separate parquet columns for image bands (`image_g`, `image_r`, `image_z`). However, our Phase 4c pipeline stores images in a single `stamp_npz` binary column containing a compressed NPZ with keys `image_g`, `image_r`, `image_z`.

**Fix Applied**: Added `_decode_stamp_npz()` function to extract images from NPZ:

```python
def _decode_stamp_npz(npz_bytes: bytes) -> Dict[str, np.ndarray]:
    """Decode compressed NPZ from stamp_npz column into per-band image arrays."""
    if npz_bytes is None:
        raise ValueError("stamp_npz is None")
    bio = io.BytesIO(npz_bytes)
    with np.load(bio) as npz:
        return {f"image_{b}": npz[f"image_{b}"] for b in ["g", "r", "z"]}
```

### Bug 2: Data Path Assumption (MAJOR)

**Problem**: The LLM's code assumed stamps and metrics were in separate parquet directories, requiring `--stamps` and `--metrics` arguments. Our Phase 4c pipeline produces a **unified parquet** with both stamps and metrics in the same rows.

**Fix Applied**: Changed to single `--data` argument pointing to unified Phase 4c output.

### Bug 3: Import Path Issues

**Problem**: The cache import used `from data_cache import DataCache` which fails depending on how the script is invoked.

**Fix Applied**: Try multiple import paths:

```python
try:
    from .data_cache import DataCache  # Relative
except ImportError:
    try:
        from dark_halo_scope.model.data_cache import DataCache  # Absolute
    except ImportError:
        from data_cache import DataCache  # Same-directory
```

### Bug 4: Contract File Reference

**Problem**: Referenced `spark_phase4_pipeline_v3.py` which doesn't exist.

**Fix Applied**: Changed to `spark_phase4_pipeline.py`.

---

## New Production Training Suite

We created three new scripts for production-grade training:

### 1. `phase5_train_production.py` - Main Training Script

**Features implemented:**
- Mixed precision (AMP) for 2x speedup on V100
- Early stopping with configurable patience
- Cosine annealing LR with linear warmup
- AUROC as primary metric (not accuracy)
- Per-class metrics (precision, recall, F1, specificity)
- Astronomy-safe data augmentation (flips, 90° rotations only)
- Gradient clipping (default 1.0)
- Label smoothing support
- Comprehensive TensorBoard logging
- Checkpoint management (last + best by AUROC)
- Training resumption from checkpoint
- Data caching for S3 data

### 2. `phase5_hyperparam_sweep.py` - Hyperparameter Optimization

**Features:**
- Optuna-based Bayesian optimization
- Searches over: LR, weight_decay, dropout, architecture, batch_size, warmup, label_smoothing
- MedianPruner for early trial termination
- SQLite storage for resumable sweeps
- Auto-generates recommended training command

### 3. `phase5_monitor.py` - Training Monitor

**Features:**
- Real-time convergence detection
- Overfitting detection
- Time-to-completion estimates
- Watch mode with auto-refresh
- Actionable recommendations

### 4. `data_cache.py` - S3 Data Caching

**Strategy:**
- Cache up to 900GB on local NVMe
- If cache full, return S3 URI for streaming (no eviction thrashing)
- Cached data stays cached forever
- Disk space safety checks before download

---

## Phase 4c Data Schema

The training data comes from Phase 4c with this schema:

**Key columns used for training:**
- `stamp_npz`: Binary NPZ containing `image_g`, `image_r`, `image_z` (64x64 float32)
- `lens_model`: "CONTROL" or "SIE" or "SIS" - determines label (0 vs 1)
- `region_split`: "train", "val", or "test" - region-based holdout
- `cutout_ok`: 1 if stamp is valid

**Key columns preserved for completeness analysis:**
- `theta_e_arcsec`: Einstein radius
- `psf_fwhm_used_r`: Per-stamp PSF FWHM
- `arc_snr`: Peak SNR of injected arc
- `bad_pixel_frac`, `wise_brightmask_frac`: Quality metrics

---

## Dataset Statistics

- Total samples: ~10.6 million
- Train/val/test split: ~80/10/10 by region (region-disjoint)
- Class balance: 50% controls, 50% injections
- Stamp size: 64x64 pixels, 3 channels (g, r, z)
- Bands: DECaLS g, r, z

---

## Target Hardware

- Instance: AWS p3.2xlarge (1x V100 16GB, 8 vCPU, 61GB RAM)
- Storage: 1TB NVMe
- Estimated training time: 10-20 hours

---

## Questions for Review

### A. Data Loading and Preprocessing

1. **Is the `_decode_stamp_npz` approach correct?** We load the binary `stamp_npz` column, wrap in BytesIO, and use `np.load()`. Are there any edge cases or performance concerns?

2. **Is the robust normalization correct?** We use `(x - median) / (1.4826 * MAD)` with clipping to [-10, 10]. Is this standard for astronomical images?

3. **Is the augmentation astronomy-safe?** We only apply:
   - Random 90° rotations (k=1,2,3)
   - Random horizontal flip
   - Random vertical flip
   
   We do NOT apply color jitter, crops, or other augmentations. Is this correct for gravitational lens detection?

4. **Is the label assignment correct?** `y = 0 if lens_model == "CONTROL" else 1`. Should we handle any edge cases?

### B. Model Architecture

5. **Is the ResNet18 adaptation for 64x64 correct?** We:
   - Replace conv1: 7x7 stride 2 → 3x3 stride 1
   - Remove maxpool
   - Keep rest of architecture
   
   Is this the right approach for small images?

6. **Should we use pretrained weights?** Currently using `weights=None`. Would ImageNet pretraining help for astronomy?

7. **Is the output layer correct?** Single output with BCEWithLogitsLoss for binary classification?

### C. Training Configuration

8. **Is the learning rate schedule appropriate?**
   - Linear warmup for 1 epoch
   - Cosine decay to 1e-6
   - Base LR: 3e-4
   
   Is this reasonable for ResNet18 on this data?

9. **Is early stopping on AUROC correct?** We stop when AUROC doesn't improve for 5 epochs with min_delta=0.001. Is this appropriate?

10. **Is gradient clipping at 1.0 appropriate?** Should we use a different value?

11. **Are the hyperparameter search ranges reasonable?**
    - LR: 1e-5 to 1e-3 (log scale)
    - Weight decay: 1e-6 to 1e-3 (log scale)
    - Dropout: 0.0 to 0.5
    - Batch size: 128, 256, 512

### D. Metrics and Evaluation

12. **Is AUROC the right primary metric?** For balanced binary classification, should we also track:
    - Average Precision (AP)?
    - F1 at optimal threshold?
    - Calibration metrics?

13. **Is the validation strategy sound?** We:
    - Use region-disjoint splits (no spatial leakage)
    - Evaluate on 500 batches per epoch
    - Track separate metrics for controls vs injections
    
    Is anything missing?

14. **How should we select the operating threshold for deployment?** We train with BCE but need a threshold for binary predictions. Should we:
    - Use 0.5?
    - Optimize for F1?
    - Set based on target false positive rate?

### E. Distributed Training and Performance

15. **Is the DDP setup correct?** We use:
    - `torchrun --nproc_per_node=N`
    - DistributedDataParallel with device_ids=[local_rank]
    - Data sharding across (rank, worker)
    
    Any issues?

16. **Is mixed precision (AMP) correctly implemented?** We use:
    - `autocast(enabled=cfg.use_amp)` for forward pass
    - `GradScaler` for backward pass
    - Unscale before gradient clipping

17. **Any performance optimizations we're missing?**
    - pin_memory=True ✓
    - non_blocking=True ✓
    - set_to_none=True for zero_grad ✓

### F. Scientific Rigor

18. **Is the control sample handling correct?** Controls have:
    - `theta_e = 0`
    - `lens_model = "CONTROL"`
    - NULL for arc_snr, magnification, etc.
    
    We label them as 0 and train normally. Is this correct?

19. **What validation should we run before publishing results?**
    - Calibration curves?
    - Per-theta_e/PSF performance breakdown?
    - Known lens recovery test?

20. **How do we ensure the model generalizes beyond our injection parameters?** Our injections have specific:
    - theta_e ∈ {0.3, 0.6, 1.0} arcsec
    - src_dmag ∈ {-2, -1, 0, 1, 2}
    - Specific lens models (SIE, SIS)
    
    How do we validate generalization?

### G. Code Quality

21. **Are there any bugs or issues in the attached code?** Please review for:
    - Off-by-one errors
    - Memory leaks
    - Race conditions in distributed training
    - Incorrect tensor operations

22. **Is the checkpoint format appropriate?** We save:
    - model_state_dict
    - optimizer_state_dict
    - scheduler step count
    - scaler state
    - config
    - epoch, global_step, best_auroc

23. **Is error handling sufficient?** We silently skip problematic rows with low probability logging. Is this appropriate?

### H. Recommendations

24. **What additional features would you recommend for publication-grade training?**

25. **What are the biggest risks in our current implementation?**

26. **Are there any astronomy-specific best practices we're missing?**

---

## Files to Attach

Please attach the following Python files for review:

1. `dark_halo_scope/model/phase5_train_production.py` - Main training script
2. `dark_halo_scope/model/phase5_hyperparam_sweep.py` - Hyperparameter sweep
3. `dark_halo_scope/model/phase5_monitor.py` - Training monitor
4. `dark_halo_scope/model/data_cache.py` - Data caching layer
5. `dark_halo_scope/model/phase5_infer_scores.py` - Inference script
6. `dark_halo_scope/model/phase5_required_columns_contract.py` - Schema contract

---

## Expected Output

Please provide:

1. **GO / NO-GO assessment** for using this training pipeline
2. **Critical issues** that must be fixed before training
3. **Recommendations** for improvements
4. **Answers to each numbered question** above
5. **Code patches** for any bugs found

We want to ensure this training produces a model that:
- Achieves state-of-the-art performance on lens detection
- Follows ML best practices
- Produces scientifically defensible results
- Can withstand peer review scrutiny


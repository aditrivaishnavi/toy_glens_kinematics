# Gen5-Prime Training Configuration Review

## Executive Summary

I'm preparing to train Gen5-Prime, a gravitational lens detection model using paired 6-channel input (positive stamp + control stamp). This document details the training configuration, rationale for each hyperparameter, verification strategy, and requests your review before proceeding.

---

## 1. Training Configuration

```python
# ============================================================================
# GEN5-PRIME TRAINING CONFIGURATION
# ============================================================================

config = {
    # -------------------------------------------------------------------------
    # MODEL ARCHITECTURE
    # -------------------------------------------------------------------------
    
    "model_name": "resnet18",
    # Why: ResNet18 is the baseline used in original research (Lanusse et al., 
    # Jacobs et al.). Using same architecture ensures fair comparison.
    # Implication: ~11M parameters, proven effective for lens detection.
    # Alternative considered: EfficientNet-B0 (5M params), but less comparable.
    
    "input_channels": 6,
    # Why: 3 channels (g,r,z) from stamp + 3 channels from ctrl_stamp.
    # Implication: Model can learn differential features (stamp - ctrl).
    # This is the KEY CHANGE from Gen5 (3-channel) to Gen5-Prime.
    # Consistency: First conv layer modified: Conv2d(6, 64, ...) vs (3, 64, ...).
    
    "num_classes": 1,
    # Why: Binary classification (lens vs no-lens).
    # Output: Single logit, sigmoid applied for probability.
    
    "pretrained": False,
    # Why: ImageNet pretrained weights expect 3-channel RGB input.
    # With 6-channel input, pretrained weights don't transfer meaningfully.
    # We train from scratch on our domain-specific data.
    
    # -------------------------------------------------------------------------
    # DATA CONFIGURATION
    # -------------------------------------------------------------------------
    
    "train_path": "s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_paired/train/",
    "val_path": "s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_paired/val/",
    "test_path": "s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_paired/test/",
    
    "exclude_bricks": ["0460m800", "3252p267"],
    # Why: These bricks have arc_snr=0 anomalies (per data quality review).
    # Implication: Removes ~216 samples from training (0.003% of data).
    # Split integrity verified: These bricks don't appear in val/test.
    
    "input_keys": {
        "positive": ["stamp_npz.image_g", "stamp_npz.image_r", "stamp_npz.image_z"],
        "control": ["ctrl_stamp_npz.image_g", "ctrl_stamp_npz.image_r", "ctrl_stamp_npz.image_z"],
    },
    # Why: Explicit key mapping to ensure correct data loading.
    # Order matters: g, r, z bands must be consistent between stamp and ctrl.
    
    "stamp_size": 64,
    # Why: Fixed 64x64 pixel stamps as created in phase4c pipeline.
    # At 0.262 arcsec/pixel, this covers ~17 arcsec field of view.
    # Sufficient for Einstein radii up to ~5 arcsec.
    
    # -------------------------------------------------------------------------
    # NORMALIZATION
    # -------------------------------------------------------------------------
    
    "normalization": "per_sample_robust",
    # Why: Each sample normalized independently using robust statistics.
    # Method: (x - median) / (1.4826 * MAD) to handle outliers.
    # Alternative: Global normalization (fixed mean/std across dataset).
    # Chosen because: Lens brightness varies significantly across samples.
    
    "clip_range": (-5, 20),
    # Why: After normalization, clip to [-5, 20] sigma to prevent extreme values.
    # Upper bound asymmetric because galaxy cores are bright (positive outliers).
    # Implication: Preserves dynamic range while limiting numerical instability.
    
    # -------------------------------------------------------------------------
    # AUGMENTATION
    # -------------------------------------------------------------------------
    
    "augmentation": {
        "random_rotation": True,  # 0, 90, 180, 270 degrees
        "random_flip": True,      # Horizontal and vertical
        "random_transpose": True, # Swap axes
    },
    # Why: Gravitational lensing has no preferred orientation.
    # These augmentations are physically valid (sky has no "up").
    # Note: We apply SAME augmentation to stamp and ctrl_stamp (paired).
    
    "noise_augmentation": False,
    # Why: Input data already has realistic noise from real observations.
    # Adding synthetic noise may degrade sim-to-real transfer.
    
    # -------------------------------------------------------------------------
    # TRAINING HYPERPARAMETERS
    # -------------------------------------------------------------------------
    
    "batch_size": 256,
    # Why: Larger batches stabilize gradient estimates.
    # Fits in GPU memory for 6-channel 64x64 input on A100 (40GB).
    # Effective batch size = 256 * num_gpus if using DDP.
    
    "epochs": 50,
    # Why: Based on convergence analysis of Gen4/Gen5 training curves.
    # Gen4 converged around epoch 30-40, adding 20% buffer.
    # Early stopping will trigger if val loss plateaus for 10 epochs.
    # Total samples: ~6.5M * 50 = 325M sample iterations.
    
    "learning_rate": 1e-3,
    # Why: Standard starting LR for Adam with ResNet.
    # Will be reduced by scheduler during training.
    
    "optimizer": "AdamW",
    # Why: Adam with decoupled weight decay.
    # Better generalization than vanilla Adam (Loshchilov & Hutter, 2019).
    
    "weight_decay": 1e-4,
    # Why: L2 regularization to prevent overfitting.
    # Standard value for image classification tasks.
    
    "scheduler": {
        "type": "CosineAnnealingWarmRestarts",
        "T_0": 10,        # First restart after 10 epochs
        "T_mult": 2,      # Double period after each restart
        "eta_min": 1e-6,  # Minimum LR
    },
    # Why: Cosine annealing allows exploration of loss landscape.
    # Warm restarts help escape local minima.
    # Schedule: LR cycles at epochs 10, 30, 70... (we stop at 50).
    
    "early_stopping": {
        "patience": 10,
        "min_delta": 1e-4,
        "monitor": "val_loss",
    },
    # Why: Stop if validation loss doesn't improve for 10 epochs.
    # Prevents overfitting and saves compute.
    
    # -------------------------------------------------------------------------
    # LOSS FUNCTION
    # -------------------------------------------------------------------------
    
    "loss": "BCEWithLogitsLoss",
    # Why: Binary cross-entropy for binary classification.
    # Numerically stable (log-sum-exp trick built in).
    
    "label_smoothing": 0.0,
    # Why: No label smoothing for now.
    # May add 0.05-0.1 if model becomes overconfident.
    
    "class_weights": None,
    # Why: Dataset is balanced (positives = negatives after pairing).
    # Each positive stamp is paired with its own control.
    # No need for class weighting.
    
    # -------------------------------------------------------------------------
    # EVALUATION METRICS
    # -------------------------------------------------------------------------
    
    "metrics": [
        "auroc",           # Primary metric: Area Under ROC Curve
        "auprc",           # Area Under Precision-Recall Curve
        "accuracy",        # Overall accuracy at threshold=0.5
        "precision@0.95",  # Precision when recall=0.95
        "recall@0.95",     # Recall when precision=0.95
    ],
    # Why: Multiple metrics capture different aspects of performance.
    # AUROC is primary for comparison with prior work.
    # Precision@recall=0.95 important for practical candidate selection.
    
    "diagnostic_metrics": [
        "arc_snr_0_accuracy",     # Performance on arc_snr=0 samples
        "per_brick_auroc",        # Detect brick-specific bias
        "theta_e_binned_auroc",   # Performance vs Einstein radius
    ],
    # Why: These catch data quality issues and physics-dependent biases.
    
    # -------------------------------------------------------------------------
    # CHECKPOINTING & LOGGING
    # -------------------------------------------------------------------------
    
    "checkpoint_dir": "s3://darkhaloscope/models/gen5_prime/",
    "save_every_n_epochs": 5,
    "save_best": True,
    "log_to_wandb": True,
    "wandb_project": "dark_halo_scope",
    "wandb_run_name": "gen5_prime_6ch_v1",
}
```

---

## 2. Consistency and Correctness Verification

### 2.1 Input Pipeline Verification
```python
# Before training, I will verify:

# 1. Channel ordering is consistent
assert stamp.shape == (6, 64, 64)  # [g,r,z,ctrl_g,ctrl_r,ctrl_z]

# 2. Same augmentation applied to both stamp and ctrl
augmented_stamp, augmented_ctrl = augment(stamp, ctrl, seed=same_seed)

# 3. Normalization doesn't break physics
# Check: stamp[0:3] - ctrl[0:3] should still show arc pattern after norm

# 4. Excluded bricks are actually excluded
assert "0460m800" not in train_bricks
assert "3252p267" not in train_bricks
```

### 2.2 Label Consistency
```python
# Positive samples (is_control=0 in original data):
#   - stamp_npz: LRG + injected arc (label=1, has lens)
#   - ctrl_stamp_npz: LRG only (shown as negative example)

# During training:
#   - Input: concat(stamp, ctrl) → 6 channels
#   - Label: 1 (lens present in stamp portion)

# The model learns: "If channels 0-2 differ from channels 3-5 in arc-like way → lens"
```

### 2.3 Comparison with Prior Work

| Setting | Gen4 | Gen5 | Gen5-Prime | Justification |
|---------|------|------|------------|---------------|
| Input channels | 3 | 3 | **6** | Paired learning |
| Batch size | 128 | 256 | 256 | Same as Gen5 |
| Epochs | 40 | 50 | 50 | Convergence observed |
| LR | 1e-3 | 1e-3 | 1e-3 | Standard |
| Optimizer | Adam | AdamW | AdamW | Same as Gen5 |
| Architecture | ResNet18 | ResNet18 | ResNet18 | Fair comparison |

---

## 3. Training Quality Verification Plan

### 3.1 During Training (Automated Checks)

| Check | Frequency | Pass Criteria |
|-------|-----------|---------------|
| Val loss decreasing | Every epoch | Lower than previous best within 10 epochs |
| Train/Val gap | Every epoch | < 0.1 difference (no severe overfitting) |
| Gradient norm | Every step | < 10.0 (no explosion) |
| Learning rate | Every epoch | Following expected schedule |
| AUROC on val | Every epoch | Increasing, > 0.9 by epoch 20 |

### 3.2 Diagnostic Checks (Every 10 Epochs)

| Check | Purpose | Pass Criteria |
|-------|---------|---------------|
| arc_snr=0 accuracy | Paired-consistency | Should behave like negatives (acc < 60%) |
| Per-brick AUROC variance | Detect brick bias | Std < 0.05 across bricks |
| Theta_E binned AUROC | Physics consistency | AUROC increases with theta_E |
| Confusion matrix | Balance check | FP/FN ratio reasonable |

### 3.3 Post-Training Validation

1. **Holdout test set evaluation** (never seen during training)
2. **Sim-to-real gap test**: Apply to real lens candidates from literature
3. **Failure case analysis**: Manual inspection of high-confidence errors
4. **Calibration plot**: Predicted probability vs actual frequency

---

## 4. Questions for You

### Q1: Proactive Checks During Training

What can I proactively monitor to ensure the model learns correctly?

**My current plan:**
- [ ] Watch for train/val divergence (overfitting signal)
- [ ] Monitor per-class accuracy (shouldn't be 100% on either class)
- [ ] Check gradient flow to early layers (shouldn't vanish)
- [ ] Visualize attention maps at epoch 10, 25, 50 (should focus on arc region)

**Are there additional checks you recommend?**

### Q2: Epoch Count Justification

I chose **50 epochs** based on:
- Gen4 converged at ~30-40 epochs
- 6-channel input is more complex, may need more iterations
- Early stopping will prevent wasted compute if convergence is faster
- ~325M sample iterations total

**Is 50 epochs appropriate, or should I increase/decrease?**

### Q3: Comparability to SOTA

To ensure comparability with prior work (Jacobs et al., Lanusse et al., Huang et al.):

| Metric | Prior SOTA | Our Target | Notes |
|--------|------------|------------|-------|
| AUROC | 0.95-0.98 | > 0.97 | On simulated test set |
| Precision@90% recall | ~0.85 | > 0.90 | Key for candidate selection |
| Sim-to-real purity | 50-70% | > 75% | Critical for Gen5-Prime |

**Are these targets reasonable?**

### Q4: Additional Requirements

Is there anything else I should:
- Add to the configuration?
- Monitor during training?
- Validate before deployment?

---

## 5. Self-Generated Questions (You Should Also Ask These)

### Q5: Handling of Paired Samples

How exactly are positive and negative examples constructed during training?

**Current design:**
- Each batch contains positive samples only (from the paired dataset)
- The "negative" signal comes from the ctrl portion of the 6-channel input
- Model learns: stamp ≠ ctrl → lens present

**Concern:** Should we also include explicit negative samples (random LRGs without lens injection)? The current design may not teach the model what a true negative looks like.

### Q6: Control Stamp Usage

Should the control stamp be:
- (A) Just an input feature (current plan)
- (B) Also used as explicit negative example (double the effective training data)
- (C) Used in a contrastive loss (stamp embedding vs ctrl embedding)

**Current choice:** (A) - Control as input feature only.
**Trade-off:** (B) would double data but may confuse the model about what "negative" means.

### Q7: Generalization to Real Data

The paired dataset is simulated (injected arcs on real LRGs). The real test scenario is:
- Real lens candidates from surveys
- No ground truth "control" available

**Question:** During inference on real data, what do we use as ctrl_stamp?
- Option 1: Same image duplicated (stamp = ctrl, 6 identical channels)
- Option 2: Train a separate 3-channel model for real deployment
- Option 3: Use median LRG template as ctrl

**This is a critical deployment question I need your input on.**

### Q8: Shortcut Learning Risk

With 6-channel input, the model might learn shortcuts like:
- "If channel variance is high → lens" (ignoring actual arc pattern)
- "If channels 0-2 have higher brightness → lens" (trivial flux comparison)

**Mitigation:**
- Augmentation applies same transform to both stamp and ctrl
- Normalization is per-channel, not per-sample
- Monitor attention maps to ensure arc-localized focus

**Is this sufficient, or should we add explicit shortcut-resistant training?**

### Q9: Calibration

For practical use, we need well-calibrated probabilities, not just ranking.

**Plan:** 
- After training, fit Platt scaling on validation set
- Report Expected Calibration Error (ECE)

**Should calibration be a blocking criterion for deployment?**

---

## 6. Go/No-Go Decision

### Go Criteria Met:
- [x] Data quality validated (arc localization, split integrity)
- [x] Configuration consistent with prior work
- [x] Verification plan in place
- [x] Diagnostic metrics defined

### Pending Items:
- [ ] Your confirmation on epoch count
- [ ] Your input on Q7 (deployment inference without ctrl)
- [ ] Your guidance on Q6 (ctrl usage strategy)

### My Recommendation: **Conditional GO**

Proceed with training if you confirm:
1. Epoch count of 50 is appropriate
2. Inference strategy for real data (Q7) is decided
3. No additional blocking concerns

---

## Next Steps (Upon Approval)

1. Create filtered training dataset (exclude 2 bricks)
2. Implement 6-channel data loader with paired augmentation
3. Modify ResNet18 input layer for 6 channels
4. Run training with above configuration
5. Report progress at epochs 10, 25, 50
6. Evaluate on test set and real lens candidates

**Please confirm GO or provide feedback on the configuration.**

# Comprehensive Experiment Plan v2

**Date**: 2026-02-06
**Author**: AI Assistant (reviewed by user)
**Status**: DRAFT - Awaiting user approval before execution

---

## 1. Objective

Systematically evaluate training configurations to find the most scientifically defensible approach for lens detection that:
1. Does NOT rely on shortcuts (core brightness, radial profile)
2. Generalizes to real survey data
3. Maintains sensitivity across all θ_E ranges

---

## 2. Dataset Specification

### 2.1 Full Dataset (from unpaired manifest)

| Split | Total | Positive | Negative |
|-------|-------|----------|----------|
| Train | 29,108 | 14,554 | 14,554 |
| Val | 100,240 | 50,120 | 50,120 |
| Test | 88,528 | 44,264 | 44,264 |

### 2.2 10% Sample Dataset (for faster iteration)

| Split | Total | Positive | Negative |
|-------|-------|----------|----------|
| Train | 2,910 | 1,455 | 1,455 |
| Val | 10,024 | 5,012 | 5,012 |
| Test | 8,852 | 4,426 | 4,426 |

**Sampling strategy**: Stratified random sampling preserving:
- Label balance (50/50)
- θ_E distribution (proportional sampling across bins)
- PSF bin distribution
- Depth bin distribution

### 2.3 Paired Dataset (for baseline comparison)

Use existing paired data at `/home/ubuntu/data/v5_cosmos_paired/` with same 10% sampling strategy.

---

## 3. Experiment Matrix

### 3.1 Phase 1: Baseline Comparison (Paired vs Unpaired)

| ID | Data Mode | Preprocessing | Core Masking | Purpose |
|----|-----------|---------------|--------------|---------|
| **A1** | Paired | Residual | None | Diagnostic baseline |
| **B1** | Unpaired | Residual | None | Deployment baseline |

### 3.2 Phase 2: Masking Variants (if B1 gate fails)

| ID | Data Mode | Preprocessing | Core Masking | Purpose |
|----|-----------|---------------|--------------|---------|
| **B2** | Unpaired | Residual | Stochastic r=5, p=0.5 | Conservative variant |
| **B3** | Unpaired | Residual | Scheduled masking | External LLM recommended |
| **B4** | Unpaired | Residual | Stochastic r=3, p=0.3 | Minimal masking |

### 3.3 Scheduled Masking Schedule (for B3)

| Epoch Range | Radius | Probability |
|-------------|--------|-------------|
| 0-10 | 7 | 0.7 |
| 10-30 | 5 | 0.5 |
| 30+ | 3 | 0.3 |

---

## 4. Training Configuration

### 4.1 Model Architecture

- **Architecture**: ResNet18
- **Input**: 3 channels (g, r, z), 63×63 pixels
- **Output**: Binary classification (lens / no lens)

### 4.2 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch size | 128 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Early stopping patience | 15 epochs |
| Mixed precision | Yes |

### 4.3 Augmentations

| Augmentation | Probability |
|--------------|-------------|
| Horizontal flip | 0.5 |
| Vertical flip | 0.5 |
| 90° rotation | 0.25 each |

---

## 5. Metrics to Capture

### 5.1 Per-Epoch Metrics (logged every epoch)

| Metric | Description |
|--------|-------------|
| `train_loss` | BCE loss on training set |
| `val_auroc` | AUROC on validation set |
| `val_loss` | BCE loss on validation set |
| `learning_rate` | Current learning rate |
| `epoch_time_sec` | Time per epoch |

### 5.2 Gate Metrics (every 5 epochs)

| Metric | Description | Target |
|--------|-------------|--------|
| `core_lr_auc` | Core LR AUC (5-fold CV) | < 0.65 |
| `radial_lr_auc` | Radial profile LR AUC (5-fold CV) | < 0.55 |
| `gate_n_samples` | Number of samples used | 2000 |

### 5.3 θ_E Stratified Metrics (every 5 epochs)

| θ_E Bin (arcsec) | Metrics |
|------------------|---------|
| [0.50, 0.75) | AUROC, count |
| [0.75, 1.00) | AUROC, count |
| [1.00, 1.50) | AUROC, count |
| [1.50, 2.00) | AUROC, count |
| [2.00, 2.50) | AUROC, count |
| [2.50, 3.00) | AUROC, count |

### 5.4 End-of-Training Metrics

| Metric | Description |
|--------|-------------|
| `best_val_auroc` | Best validation AUROC achieved |
| `best_epoch` | Epoch with best val AUROC |
| `final_core_lr_auc` | Core LR AUC on full val set |
| `final_radial_lr_auc` | Radial LR AUC on full val set |

### 5.5 Core Sensitivity Curve (end of training)

| Mask Radius | Metric |
|-------------|--------|
| r=0 | Val AUROC (no mask) |
| r=3 | Val AUROC |
| r=5 | Val AUROC |
| r=7 | Val AUROC |
| r=10 | Val AUROC |
| r=15 | Val AUROC |

**Derived metric**: `core_reliance` = AUROC(r=0) - AUROC(r=7)
- Target: < 0.05 (model not heavily relying on core)

### 5.6 Test Set Metrics (final evaluation only)

| Metric | Description |
|--------|-------------|
| `test_auroc` | AUROC on held-out test set |
| `test_auroc_by_thetae` | Per-θ_E bin AUROC on test |
| `test_core_sensitivity` | Core sensitivity curve on test |

---

## 6. Success Criteria

### 6.1 Gate Thresholds

| Gate | PASS | WARN | FAIL |
|------|------|------|------|
| Core LR AUC | < 0.60 | 0.60-0.65 | > 0.65 |
| Radial LR AUC | < 0.55 | 0.55-0.60 | > 0.60 |

### 6.2 CNN Stress Test Thresholds

| Metric | PASS | WARN | FAIL |
|--------|------|------|------|
| Core reliance (r=7 drop) | < 0.03 | 0.03-0.05 | > 0.05 |

### 6.3 θ_E Stratification

| Criterion | PASS | FAIL |
|-----------|------|------|
| Min bin AUROC | > 0.80 | < 0.70 |
| Max AUROC spread | < 0.15 | > 0.20 |

### 6.4 Overall Model Quality

| Metric | Target |
|--------|--------|
| Val AUROC | > 0.90 |
| Test AUROC | > 0.88 |

---

## 7. Execution Plan

### 7.1 Phase 1: Setup (Before any training)

| Step | Task | Status |
|------|------|--------|
| 1.1 | Create 10% stratified sample manifest | Pending |
| 1.2 | Create paired 10% sample manifest | Pending |
| 1.3 | Verify data loading works for both | Pending |
| 1.4 | Update training script for all metrics | Pending |
| 1.5 | Sync code to all Lambda instances | Pending |

### 7.2 Phase 2: Baseline Experiments (A1, B1)

| Step | Task | Instance | Status |
|------|------|----------|--------|
| 2.1 | Run A1 (paired + residual) | lambda | Pending |
| 2.2 | Run B1 (unpaired + residual) | lambda | Pending |
| 2.3 | Analyze A1 vs B1 results | - | Pending |
| 2.4 | Decision: proceed to Phase 3 or done | - | Pending |

### 7.3 Phase 3: Masking Variants (if needed)

| Step | Task | Instance | Status |
|------|------|----------|--------|
| 3.1 | Run B2 (stochastic r=5) | lambda | Pending |
| 3.2 | Run B3 (scheduled masking) | lambda | Pending |
| 3.3 | Run B4 (stochastic r=3) | lambda | Pending |
| 3.4 | Compare all B variants | - | Pending |

### 7.4 Phase 4: Full Training (after selecting best config)

| Step | Task | Status |
|------|------|--------|
| 4.1 | Train selected config on full data | Pending |
| 4.2 | Final test set evaluation | Pending |
| 4.3 | Document results | Pending |

---

## 8. Resource Allocation

### 8.1 Current GPU Status

| Instance | Current Job | Est. Completion |
|----------|-------------|-----------------|
| lambda | Ablation (no_coredrop) | ~2-3 hours |
| lambda2 | Ablation (no_coredrop) | ~2-3 hours |
| lambda3 | Ablation (minimal) | ~2-3 hours |

### 8.2 Experiment Duration Estimates (10% sample)

| Experiment | Est. Time |
|------------|-----------|
| A1, B1, B2, B3, B4 | ~30-45 min each |
| Total Phase 2-3 | ~3-4 hours |

### 8.3 Experiment Duration Estimates (Full data)

| Experiment | Est. Time |
|------------|-----------|
| Single experiment | ~2-3 hours |
| Full training | ~2-3 hours |

---

## 9. Output Artifacts

### 9.1 Per-Experiment

| Artifact | Path |
|----------|------|
| Checkpoints | `/home/ubuntu/checkpoints/{exp_id}/` |
| Best model | `best.pt` |
| Last model | `last.pt` |
| Training log | `stdout.log` |
| Metrics JSON | `metrics.json` |

### 9.2 Summary

| Artifact | Path |
|----------|------|
| Results comparison | `planb/EXPERIMENT_RESULTS.md` |
| Metrics CSV | `planb/experiment_metrics.csv` |

---

## 10. Decision Tree

```
After A1 and B1 complete:
│
├── B1 Core AUC < 0.60 AND core_reliance < 0.03
│   └── SUCCESS: Use B1 config, proceed to full training
│
├── B1 Core AUC 0.60-0.65 AND core_reliance < 0.05
│   └── ACCEPTABLE: Use B1 config with monitoring
│
├── B1 Core AUC > 0.65 BUT core_reliance < 0.05
│   └── TEST MASKING VARIANTS (B2, B3, B4)
│   │
│   ├── B3 (scheduled) passes all criteria
│   │   └── Use B3 config
│   │
│   ├── B2 or B4 passes with better θ_E spread
│   │   └── Use that config
│   │
│   └── All fail core criteria
│       └── STOP - need data redesign
│
└── B1 Core AUC > 0.65 AND core_reliance > 0.05
    └── STOP - model is exploiting shortcut
```

---

## 11. Approval Checklist

Before execution, confirm:

- [ ] Dataset sampling strategy approved
- [ ] Experiment matrix approved
- [ ] Metrics list complete
- [ ] Success criteria agreed
- [ ] Resource allocation acceptable
- [ ] Decision tree logic agreed

---

## 12. Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-06 | Initial draft | AI Assistant |

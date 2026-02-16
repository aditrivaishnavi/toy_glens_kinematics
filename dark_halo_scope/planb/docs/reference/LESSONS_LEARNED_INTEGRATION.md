# Lessons Learned Integration

This document maps each lesson learned from previous failures to specific code implementations in the planb directory.

---

## Code Bugs → Prevention

| Lesson | Implementation |
|--------|----------------|
| **L1.1** boto3 import at module level | Not applicable (no EMR in training phase) |
| **L1.2** Duplicate function definitions | Single-file modules with clear structure |
| **L1.3** Variable referenced before assignment | All variables initialized before conditionals |
| **L1.4** Undefined variables in processing | Explicit row extraction in data_loader.py |
| **L1.5** Surface brightness units wrong | Not applicable (using pre-generated data) |
| **L1.6** PSF kernel exceeding stamp size | Checked in azimuthal_shuffle radial binning |
| **L1.7** _parse_s3 not handling s3a:// | Not applicable (local paths) |

---

## Data Quality Issues → Validation

| Lesson | Implementation | File |
|--------|----------------|------|
| **L7.1** NaN in raw stamps | `validate_stamp()` checks every sample | `data_loader.py:100-120` |
| **L7.2** Missing bands | `decode_stamp_npz()` handles missing keys | `data_loader.py:50-90` |
| **L7.3** Inverted train/val/test splits | `verify_split_integrity.py` checks brick overlap | `phase0_foundation/` |

---

## Incorrect Assumptions → Explicit Checks

| Lesson | Implementation | File |
|--------|----------------|------|
| **L4.3** "Smoke test validates pipeline" | `run_preflight_checks()` runs actual loaders | `train.py:80-130` |
| **L4.4** "Data has no NaN" | Every sample validated before use | `data_loader.py` |
| **L4.5** "arc_snr is integrated SNR" | Document units in config | `configs/*.yaml` |
| **L4.6** "Train/val/test is 70/15/15" | Verify actual proportions | `phase0_foundation/` |

---

## AI Mistakes → Process Fixes

| Lesson | Implementation | File |
|--------|----------------|------|
| **L5.1** Declaring victory prematurely | Explicit exit criteria per phase | `EXIT_CRITERIA.md` |
| **L5.2** Not checking logs immediately | `setup_logging()` with file + console | `train.py` |
| **L5.3** Using wrong argument names | ArgParser with explicit help | `train.py` |
| **L5.4** Not validating code before upload | `test_core_functions.py` unit tests | `tests/` |
| **L5.5** Not reading error messages | Detailed error logging | `train.py` |

---

## Process Failures → Validation Steps

| Lesson | Implementation | File |
|--------|----------------|------|
| **L6.1** No local testing before EMR | All training runs locally first | N/A |
| **L6.2** Not verifying code matches | Git-tracked, no S3 code upload | N/A |
| **L6.3** Not tracking code versions | Config saved with training | `train.py` |

---

## Shortcut Detection → Gate System

| Lesson | Implementation | File |
|--------|----------------|------|
| **L21** Core brightness shortcut | `evaluate_gates()` with Core_LR_AUC | `train.py:150-220` |
| **L22** Arc overlap dominates core | `apply_core_dropout()` augmentation | `data_loader.py:160-200` |
| **L1119-1185** Shortcut detection gates | Full gate suite in training | `validate_baseline.py` |

---

## Specific Code Implementations

### 1. NaN/Inf Detection (Lesson L7.1, L1.4)

```python
# data_loader.py:100-120
def validate_stamp(stamp: np.ndarray, bandset: str) -> Dict[str, bool]:
    results = {
        "no_nan": not np.isnan(stamp).any(),
        "no_inf": not np.isinf(stamp).any(),
        ...
    }
    results["valid"] = all(results.values())
    return results
```

### 2. Core Shortcut Detection (Lesson L21)

```python
# train.py:150-200 (evaluate_gates)
def evaluate_gates(model, loader, device, logger):
    # Core LR AUC - must be < 0.65
    core = x[:, :, 27:37, 27:37]  # Central 10x10
    lr = LogisticRegression()
    lr.fit(core_flat, y)
    core_lr_auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])
```

### 3. Core Dropout (Lesson L22)

```python
# data_loader.py:160-200
def apply_core_dropout(img, radius=5, fill_mode="outer_median"):
    """Force model to learn from outer regions."""
    core_mask = r < radius
    result[core_mask] = outer_median
```

### 4. Preflight Validation (Lesson L4.3, L5.1)

```python
# train.py:80-130
def run_preflight_checks(train_loader, val_loader, model, device, logger):
    """Run ALL preflight checks before training."""
    results = {
        "train_loader_valid": validate_loader(train_loader)["all_passed"],
        "val_loader_valid": validate_loader(val_loader)["all_passed"],
        "model_valid": validate_model(model)["all_passed"],
        "first_batch_ok": check_first_batch(model, train_loader),
    }
    if not all(results.values()):
        logger.error("ABORTING: Preflight checks failed")
        return results
```

### 5. Exit Criteria Tracking (Lesson L5.1)

```python
# train.py:280-320
# Save results with explicit pass/fail
results = {
    "success": True,
    "best_epoch": best_epoch,
    "best_auroc": best_auroc,
    "final_gates": final_gates,
}
with open(f"{output_dir}/training_results.json", "w") as f:
    json.dump(results, f)
```

---

## Checklist for Each Training Run

Before training:
- [ ] Run `pytest tests/test_core_functions.py`
- [ ] Run `python phase0_foundation/run_all_phase0.py`
- [ ] Verify all Phase 0 checks pass

During training:
- [ ] Monitor loss for NaN
- [ ] Check expected ranges (see EXIT_CRITERIA.md)
- [ ] Watch for divergence

After training:
- [ ] Verify `training_results.json` exists
- [ ] Check all gates pass
- [ ] Compare to expected values
- [ ] Run `validate_baseline.py` independently

---

## Red Flags to Watch For

| Symptom | Likely Cause | Action |
|---------|--------------|--------|
| Loss → NaN | Data has NaN or exploding grads | Check data validation, reduce LR |
| AUROC = 1.0 | Data leakage or trivial task | Check split integrity, core LR gate |
| Core_LR_AUC > 0.80 | Shortcut not mitigated | Increase core_dropout_prob |
| Training loss → 0 | Memorization | Add regularization, early stop |
| Val/train gap > 0.1 | Overfitting | More augmentation, less capacity |

---

## Long-Running Training Jobs → Process Management

| Lesson | Implementation | File |
|--------|----------------|------|
| **L8.1** Always use nohup/tmux | Run training with `nohup ... &` or inside `tmux` | N/A |
| **L8.2** Checkpoint resumption must be DEFAULT | Auto-resume from `last.pt` unless `--fresh` | `train.py:254-268` |
| **L8.3** Never pkill without understanding workers | DataLoader workers have same pattern - killing them crashes training | N/A |
| **L8.4** Verify process status before killing | Always `ps aux` to understand parent/child before `pkill` | N/A |

### Incident: 2026-02-07

**What happened:** Training jobs B3 and B4 were running but had duplicate processes from earlier restart attempts. Used `pkill -f` to "clean up" but killed the original working processes and their DataLoader workers, losing ~8 epochs of progress.

**Root causes:**
1. Training script had no auto-resume - a basic feature for long-running jobs
2. Used `pkill` without understanding that DataLoader workers are child processes with similar patterns
3. Didn't verify which processes were the originals vs duplicates before killing

**Fixes implemented:**
1. Added auto-resume as DEFAULT behavior (line 254-268 in train.py)
2. Use `--fresh` flag to explicitly start over
3. Document: NEVER use `pkill -f` on training - always `kill <specific_pid>`

---

*This document should be reviewed before each training run.*

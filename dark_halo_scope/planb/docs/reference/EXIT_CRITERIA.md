# Exit Criteria for Each Phase

This document defines explicit PASS/FAIL criteria for each phase of the build plan.
No phase proceeds until all exit criteria of the previous phase are met.

---

## Phase 0: Foundation Lock

### 0.1 Anchor Validation
| Criterion | Threshold | Check |
|-----------|-----------|-------|
| Anchor count | ≥ 30 | `validate_anchors.py` |
| Required columns | All present | `validate_anchors.py` |
| No duplicate names | 0 duplicates | `validate_anchors.py` |
| Theta_e range | All ≥ 0.5" | `validate_anchors.py` |
| DR10 coverage | 100% of sample | `validate_anchors.py` |

### 0.2 Contaminant Validation
| Criterion | Threshold | Check |
|-----------|-----------|-------|
| Ring galaxies | ≥ 50 | `validate_contaminants.py` |
| Spiral galaxies | ≥ 50 | `validate_contaminants.py` |
| Mergers | ≥ 30 | `validate_contaminants.py` |
| No anchor overlap | 0 overlap | `validate_contaminants.py` |

### 0.3 Split Integrity
| Criterion | Threshold | Check |
|-----------|-----------|-------|
| Train/val overlap | 0 bricks | `verify_split_integrity.py` |
| Train/test overlap | 0 bricks | `verify_split_integrity.py` |
| Val/test overlap | 0 bricks | `verify_split_integrity.py` |

### 0.4 Paired Data Integrity
| Criterion | Threshold | Check |
|-----------|-----------|-------|
| Shape match (stamp=ctrl) | 100% | `verify_paired_data.py` |
| Stamp ≠ ctrl | 100% | `verify_paired_data.py` |
| No NaN/Inf | 100% | `verify_paired_data.py` |
| Expected shape (3,64,64) | 100% | `verify_paired_data.py` |

### 0.5 Code Quality
| Criterion | Threshold | Check |
|-----------|-----------|-------|
| Syntax check | 0 errors | `python -m py_compile *.py` |
| Unit tests | 100% pass | `pytest tests/` |
| Data loading test | PASS | `test_data_loading.py` |

### 0.6 Evaluation Protocol
| Criterion | Threshold | Check |
|-----------|-----------|-------|
| Protocol locked | Yes | `evaluation_protocol.yaml` exists |
| Metrics defined | All required | Manual check |
| Thresholds set | All gates | Manual check |

**Phase 0 Exit Gate:**
- ALL checks must pass
- Output: `phase0_summary.json` with `all_passed: true`

---

## Phase 1: Baseline Training

### 1.1 Preflight Checks
| Criterion | Threshold | Check |
|-----------|-----------|-------|
| Train loader valid | PASS | `train.py` preflight |
| Val loader valid | PASS | `train.py` preflight |
| Model forward pass | PASS | `train.py` preflight |
| No NaN in first batch | PASS | `train.py` preflight |

### 1.2 Training Progress (per epoch)
| Criterion | Threshold | Check |
|-----------|-----------|-------|
| Loss finite | No NaN | `train.py` monitoring |
| Loss decreasing | Trend down | `train.py` monitoring |
| Val AUROC improving | Trend up or stable | `train.py` monitoring |

### 1.3 Expected Ranges
| Metric | Epoch 1 | Epoch 5 | Epoch 10 | Check |
|--------|---------|---------|----------|-------|
| Train loss | [0.3, 1.0] | [0.15, 0.5] | [0.1, 0.3] | `train.py` |
| Val AUROC | [0.55, 0.70] | [0.70, 0.85] | [0.80, 0.92] | `train.py` |

### 1.4 Post-Training Gates
| Gate | Threshold | Direction | Check |
|------|-----------|-----------|-------|
| AUROC_synth | 0.85 | > | `validate_baseline.py` |
| Core_LR_AUC | 0.65 | < | `validate_baseline.py` |
| Core_masked_drop | 10% | < | `validate_baseline.py` |
| Hard_neg_AUROC | 0.70 | > | `validate_baseline.py` |

**Phase 1 Exit Gate:**
- Training completes without NaN
- Best AUROC > 0.85
- All 4 gates pass
- Output: `training_results.json`, `best_model.pt`

---

## Phase 1 Ablations

### 1.5 Ablation: No Hard Negatives
| Metric | Expected Change | Check |
|--------|-----------------|-------|
| Hard_neg_AUROC | ↓ (< 0.70) | Should worsen |
| Other gates | May worsen | Record |

### 1.6 Ablation: No Core Dropout
| Metric | Expected Change | Check |
|--------|-----------------|-------|
| Core_masked_drop | ↑ (> 10%) | Should worsen |
| Core_LR_AUC | ↑ (> 0.65) | Should worsen |

### 1.7 Ablation: Minimal
| Metric | Expected Change | Check |
|--------|-----------------|-------|
| All gates | May fail | Record |
| AUROC_synth | Should still be high | Shortcut expected |

**Ablation Exit Gate:**
- All 4 ablation experiments complete
- Each produces `training_results.json`
- Comparison table generated

---

## Phase 2: Gen7 Ablation

### 2.1 Code Fixes
| Fix | Verification | Check |
|-----|--------------|-------|
| clump_flux_frac semantics | Peak amplitude, not integrated | Unit test |
| Color gradient vs spatial | Spatial modulation | Unit test |

### 2.2 Parameter Validation
| Parameter | Valid Range | Check |
|-----------|-------------|-------|
| n_clumps | [2, 6] | Random sample |
| clump_sigma_pix | [0.8, 2.0] | Random sample |
| gradient_strength | [0.0, 0.3] | Random sample |

### 2.3 Training
| Gate | Threshold | Check |
|------|-----------|-------|
| All Phase 1 gates | Same thresholds | `validate_baseline.py` |
| Delta vs baseline | Record | Comparison |

**Phase 2 Exit Gate:**
- Code fixes verified with unit tests
- Training completes
- Comparison with baseline computed (with 95% CI)

---

## Phase 3: Gen8 Ablation

### 3.1 Artifact Calibration
| Artifact | DR10 Rate | Check |
|----------|-----------|-------|
| Cosmic rays | ~X per stamp | Literature/empirical |
| PSF anisotropy | ~Y ellipticity | Literature/empirical |
| Astrometric jitter | ~Z pixels | Literature/empirical |

### 3.2 Double PSF Prevention
| Check | Expected | Verification |
|-------|----------|--------------|
| PSF applied once | Single convolution | Code review + visual |

### 3.3 Training
| Gate | Threshold | Check |
|------|-----------|-------|
| All Phase 1 gates | Same thresholds | `validate_baseline.py` |
| Delta vs baseline | Record | Comparison |

**Phase 3 Exit Gate:**
- Artifact rates documented with sources
- Double PSF verified absent
- Training completes
- Comparison with baseline computed (with 95% CI)

---

## Phase 4: Final Evaluation

### 4.1 Aggregation
| Check | Criterion | Verification |
|-------|-----------|--------------|
| All experiments complete | 6 variants | `aggregate_results.py` |
| Master comparison table | Generated | `master_comparison.csv` |

### 4.2 Selection Function
| Check | Criterion | Verification |
|-------|-----------|--------------|
| Stratified by θ_E | 4 bins computed | `selection_function.py` |
| Stratified by PSF | 3 bins computed | `selection_function.py` |
| Stratified by arc_snr | 4 bins computed | `selection_function.py` |
| Bootstrap CIs | 1000 samples | `selection_function.py` |

### 4.3 Reproducibility
| Check | Criterion | Verification |
|-------|-----------|--------------|
| Multiple seeds | ≥ 3 seeds | Training logs |
| Variance reported | Std across seeds | Results JSON |

### 4.4 Figures
| Figure | Content | Check |
|--------|---------|-------|
| Ablation comparison | Bar chart + CIs | `generate_figures.py` |
| Selection function | 2D heatmap or curves | `generate_figures.py` |
| Training curves | Loss/AUROC vs epoch | `generate_figures.py` |

**Phase 4 Exit Gate:**
- All 6 ablations complete
- Selection function computed with CIs
- At least 3 seeds for final baseline
- All figures generated
- Master comparison table exported

---

## Summary: GO/NO-GO Criteria

### GREEN LIGHT (Proceed)
All of:
- Phase 0: `all_passed: true`
- Phase 1 baseline: All 4 gates pass
- At least one ablation variant trained

### YELLOW LIGHT (Investigate)
Any of:
- Some gates marginal (within 5% of threshold)
- Ablation shows unexpected pattern
- Training converges slowly

### RED LIGHT (Stop)
Any of:
- Phase 0 fails
- Training produces NaN
- All Phase 1 gates fail
- Severe regression from baseline
- Evidence of data leakage

---

## Lesson Learned Integration

These exit criteria directly address past failures:

| Lesson | Exit Criterion |
|--------|----------------|
| L5.1: Premature victory declaration | Explicit verification at each gate |
| L4.4: Assumed clean data | NaN/Inf validation |
| L21: Core brightness shortcut | Core_LR_AUC and Core_masked gates |
| L4.3: Mock vs real code path | Preflight checks run actual code |
| L1.4: NaN in raw stamps | Data validation in loader |
| L6.2: S3 code mismatch | No S3 in this local-first flow |

---

*Last updated: 2026-02-05*

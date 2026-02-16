# Build Plan Review Request

## Context

We are developing a strong gravitational lens detection system using deep learning. After extensive debugging and analysis, we have converged on a research direction that focuses on:

1. **Shortcut mitigation** via paired counterfactual controls, azimuthal-shuffle hard negatives, and core dropout
2. **Ablation study** of source realism (Gen7: procedural Sersic+clumps) and domain randomization (Gen8: imaging artifacts)
3. **Selection function measurement** as the key scientific output

This document requests your review of our comprehensive build plan before we begin execution.

---

## Research Thesis

> "Which aspects of source realism and imaging nuisance variation most affect sim-to-real transfer and the inferred selection function for lens finding in seeing-limited survey imaging?"

This thesis aligns with what the lens-finding literature explicitly identifies as limiting factors (Springer Space Science Reviews 2024).

---

## What We Are NOT Claiming

- We are **not** claiming "shortcut-aware training" as a novel invention
- We are **not** claiming "hard negatives" as new
- We are **not** claiming "paired controls" as new

---

## What We ARE Claiming (Proposed)

1. **Diagnostic contribution:** We identify and quantify a specific shortcut failure mode in lens-finding CNNs (core leakage, AUC=0.90 on central 10x10 pixels)

2. **Methodological contribution:** We propose a combined mitigation strategy:
   - Paired counterfactual controls (same LRG, same sky, ±arc)
   - Azimuthal-shuffle hard negatives (preserve radial profile, destroy morphology)
   - Core dropout during training

3. **Evaluation contribution:** We define a task-specific gate suite:
   - Core-only LR AUC (threshold < 0.65)
   - Core-masked AUROC drop (threshold < 10%)
   - Hard-negative AUROC (threshold > 0.70)

4. **Ablation contribution:** We isolate the effect of:
   - Procedural source complexity (Gen7: Sersic + clumps)
   - Imaging artifacts (Gen8: cosmic rays, PSF anisotropy, jitter)

5. **Selection function contribution:** We measure completeness as a function of θ_E, PSF FWHM, arc_snr with bootstrap confidence intervals

---

## Build Plan Summary

### Phase 0: Foundation Lock (Days 1-2)
- Lock evaluation protocol BEFORE training (no metric shopping)
- Validate anchor set (≥30 real lenses)
- Validate contaminant set (rings, spirals, mergers)
- Verify train/val/test split integrity (no brick overlap)
- Verify paired data quality
- Run all unit and integration tests

### Phase 1: Baseline Training (Days 3-5)
- Train Gen5-Prime baseline with ALL mitigations
- Ablation grid: full, no-hardneg, no-coredrop, minimal
- Post-training gate validation
- Compare ablation variants

### Phase 2: Gen7 Ablation (Days 6-8)
- Fix known code issues (clump_flux_frac semantics)
- Train Gen7 variant
- Compare to baseline with bootstrap CIs

### Phase 3: Gen8 Ablation (Days 9-11)
- Calibrate artifact rates to DR10 statistics
- Train Gen8 variant
- Compare to baseline

### Phase 4: Final Evaluation (Days 12-14)
- Aggregate all results
- Compute selection function
- Generate paper figures
- Run 3 seeds for reproducibility

---

## Gate Criteria

### Phase 0 Exit Gates
| Gate | Criterion | Purpose |
|------|-----------|---------|
| Anchor validation | ≥30 anchors, all checks pass | Ensure real-lens evaluation is valid |
| Contaminant validation | Categories populated, no overlap | Ensure FPR measurement is valid |
| Split integrity | No brick overlap | Prevent data leakage |
| Paired data | Shapes correct, no NaN | Data quality |
| Tests pass | All unit/integration tests | Code quality |

### Phase 1 Exit Gates
| Gate | Threshold | Interpretation |
|------|-----------|----------------|
| AUROC_synth | > 0.85 | Model learns something |
| Core_LR_AUC | < 0.65 | Shortcut blocked |
| Core_masked_drop | < 10% | Not core-dependent |
| Hard_neg_AUROC | > 0.70 | Morphology learned |

---

## Code Structure

```
planb/
├── configs/
│   ├── evaluation_protocol.yaml   # LOCKED metrics
│   └── gen5_prime_baseline.yaml   # Training config
├── phase0_foundation/
│   ├── run_all_phase0.py          # Run all checks
│   ├── validate_anchors.py
│   ├── validate_contaminants.py
│   ├── verify_split_integrity.py
│   └── verify_paired_data.py
├── phase1_baseline/
│   └── validate_baseline.py       # Post-training gates
├── phase4_evaluation/
│   └── aggregate_results.py       # Comparison table
└── tests/
    └── test_data_loading.py
```

---

## Specific Questions for Review

### Q1: Is the build plan rigorous enough for PhD-level research?

We have attempted to include:
- Pre-specified metrics (no post-hoc shopping)
- Explicit gate criteria at each phase
- Validation before proceeding
- Reproducibility requirements (3 seeds, bootstrap CIs)

**Is anything missing?**

### Q2: Are the gate thresholds appropriate?

| Gate | Threshold | Concern |
|------|-----------|---------|
| Core_LR_AUC < 0.65 | May be too strict if physics leakage is real | Should we allow 0.70? |
| Core_masked_drop < 10% | May penalize models that legitimately use center | Should we stratify by θ_E? |
| Hard_neg_AUROC > 0.70 | May be too loose if shuffling is easy | Should we require 0.75? |

**Should any thresholds be adjusted?**

### Q3: Is the ablation matrix sufficient?

Current matrix:
- baseline_full (all mitigations)
- ablate_no_hardneg (paired + core dropout only)
- ablate_no_coredrop (paired + hard neg only)
- ablate_minimal (paired only)
- gen7_hybrid (procedural sources)
- gen8_domain_rand (imaging artifacts)

**Should we add any variants?** For example:
- 6-channel (raw + residual) vs 3-channel?
- Different hard negative ratios (20%, 40%, 60%)?

### Q4: Is the selection function analysis adequate?

We plan to stratify by:
- θ_E bins: [0.5-1", 1-1.5", 1.5-2", 2-3"]
- PSF FWHM bins: [0.8-1.2", 1.2-1.5", 1.5-2"]
- arc_snr bins: [3-10, 10-20, 20-50, 50+]

**Should we add other stratifications?** For example:
- Source redshift?
- LRG brightness?
- Sky background level?

### Q5: What could make this paper rejected?

Anticipated criticisms:
1. "This is just engineering hygiene, not research"
2. "No real-lens validation" (we have limited anchors)
3. "Simulation-to-real gap not closed"
4. "Effect sizes are too small"

**How should we address these preemptively?**

### Q6: Is there any critical missing component?

We have:
- Paired data (done)
- Training code (partial, needs completion)
- Validation code (done)
- Evaluation code (done)
- Configuration (done)

**What else is needed?**

---

## Timeline

| Phase | Days | Compute | Risk |
|-------|------|---------|------|
| Phase 0 | 1-2 | Minimal | Low |
| Phase 1 | 3-5 | ~$200 GPU | Medium (training may fail) |
| Phase 2 | 6-8 | ~$100 GPU | Low |
| Phase 3 | 9-11 | ~$100 GPU | Medium (artifact tuning) |
| Phase 4 | 12-14 | Minimal | Low |

**Total: 4-6 weeks, ~$400-600 compute**

---

## Decision Requested

Please provide:

1. **GO / CONDITIONAL-GO / NO-GO** on the build plan
2. Any **critical blockers** that must be addressed
3. **Recommended changes** to thresholds, ablations, or analysis
4. **Anticipated reviewer objections** and how to address them
5. Any **missing components** in the code structure

---

## Attached Files

The following files are available for detailed review:

1. `BUILD_PLAN_OPTION_B.md` - Full 950-line build plan
2. `configs/evaluation_protocol.yaml` - Locked evaluation metrics
3. `configs/gen5_prime_baseline.yaml` - Baseline training config
4. `phase0_foundation/*.py` - All Phase 0 validation scripts
5. `phase1_baseline/validate_baseline.py` - Gate validation
6. `phase4_evaluation/aggregate_results.py` - Results aggregation
7. `tests/test_data_loading.py` - Integration tests

---

## Summary

We believe this build plan represents a rigorous, scientifically defensible approach to producing original research on strong lens detection. The key innovations are not the individual components (which are established) but their specific combination and the measured outcomes:

1. **Shortcut diagnosis and mitigation** specific to lens finding
2. **Controlled ablation** of realism knobs
3. **Selection function measurement** as primary output

We request your critical review before proceeding with expensive training runs.

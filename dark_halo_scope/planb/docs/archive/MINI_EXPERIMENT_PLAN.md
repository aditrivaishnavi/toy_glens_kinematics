# Mini Experiment Plan

**Date**: 2026-02-06
**Purpose**: Validate pipeline and preprocessing before committing to full training

## Rationale (from External LLM)

- Start with mini-dataset for high-confidence experimentation
- Validate gates pass BEFORE full training
- Check θ_E-stratified performance to ensure small-θ_E arcs aren't being sacrificed
- Only proceed to full training if mini experiments look good

---

## Mini Dataset Specification

| Parameter | Value |
|-----------|-------|
| Train samples | 5,000 (2,500 pos, 2,500 neg) |
| Val samples | 2,000 (1,000 pos, 1,000 neg) |
| Epochs | 15 |
| Gate evaluation | Every 5 epochs |
| θ_E bins | [0.5, 0.75), [0.75, 1.0), [1.0, 1.5), [1.5, 2.0), [2.0, 2.5), [2.5, 3.0) |

---

## Configurations to Test

### Phase 1: Baseline validation (run first)

| ID | Config | Purpose |
|----|--------|---------|
| M-A1 | Paired + residual | Diagnostic baseline - should have Core AUC ~0.55 |
| M-B1 | Unpaired + residual | Deployment baseline - expect Core AUC ~0.69 |

### Phase 2: Masking variants (if Phase 1 looks good)

| ID | Config | Purpose |
|----|--------|---------|
| M-B2 | Unpaired + residual + stochastic r=5 p=0.5 | Conservative variant |
| M-B3 | Unpaired + residual + scheduled masking | External LLM recommended |

---

## Success Criteria (from External LLM)

### Gate thresholds
- Core LR AUC < 0.65 (ideally < 0.60)
- Radial LR AUC < 0.55 (should be ~0.50 with residual)

### θ_E stratification
- No severe drop in small-θ_E bins ([0.5, 0.75), [0.75, 1.0))
- AUROC should be reasonably consistent across bins

### Model performance
- Val AUROC > 0.90 (sanity check that model learns something)

---

## Evaluation Protocol

1. **After each gate-eval epoch (5, 10, 15)**:
   - Core LR AUC (5-fold CV)
   - Radial LR AUC (5-fold CV)
   - Val AUROC overall
   - Val AUROC per θ_E bin

2. **End of training**:
   - Core sensitivity curve (r=0, 3, 5, 7, 10, 15)
   - Final gate assessment

---

## Decision Tree

```
Mini M-B1 (unpaired+residual) results:
├── Core AUC < 0.65 AND θ_E bins balanced
│   └── PROCEED to full B1 training
├── Core AUC 0.65-0.70 AND θ_E bins balanced
│   └── Test M-B2/M-B3 with masking
│       ├── Masking helps → PROCEED with masking config
│       └── Masking hurts small-θ_E → Reduce masking, re-test
└── Core AUC > 0.70 OR θ_E bins severely imbalanced
    └── STOP - investigate further before proceeding
```

---

## Implementation Steps

1. Create mini manifest (subsample from full manifest)
2. Run M-A1 and M-B1 in sequence (or parallel if GPU allows)
3. Analyze results
4. Decide on Phase 2 or proceed to full training

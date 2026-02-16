# External LLM Review Archive

**Date:** 2026-02-06
**Status:** ARCHIVED - Review completed, findings integrated

---

## Summary

During Phase 1 baseline training, we discovered a severe core brightness shortcut (Core LR AUC = 0.95). We requested external LLM review to validate our analysis and proposed solution.

### Key Outcomes

1. **Root cause confirmed**: Inner image physics is the primary driver of core leakage
2. **Unpaired training validated**: LRG-disjoint split is the standard approach in literature
3. **Residual preprocessing discovered**: Subtracting azimuthal profile breaks the shortcut
4. **Two gates established**: Core LR AUC < 0.65 and Radial LR AUC < 0.55

### Decision Made

- Adopted unpaired training with residual preprocessing
- Mini experiments validated approach (B1 passed all gates)
- Proceeding with 5 full-scale experiments

---

## Files Consolidated Here

The following files have been merged into this archive:

| Original File | Content |
|---------------|---------|
| `EXTERNAL_REVIEW_PROMPT.md` | Prompt sent to external LLM |
| `EXTERNAL_LLM_REPLY.md` | First external response |
| `EXTERNAL_LLM_GATE_REVIEW.md` | Gate-focused review |
| `EXTERNAL_LLM_FINAL_REPLY.md` | Final recommendations |
| `EXTERNAL_LLM_COMBINED_REVIEW.md` | Combined analysis document |
| `INDEPENDENT_REVIEW_OF_EXTERNAL_LLM.md` | Our review of their recommendations |
| `LLM_REVIEW_BUILD_PLAN.md` | Build plan derived from review |

---

## Key Findings from External Review

### 1. Root Cause Validation

The external LLM confirmed our analysis:
- Inner (counter) image of lensed source is physical, not a bug
- Arc flux in core is θ_E dependent (41% for small θ_E, 8% for large)
- This is fundamental physics, not correctable at data level

### 2. Solution Recommendations

1. **Unpaired Training**: Standard practice in lens finding literature
2. **Residual Preprocessing**: Subtract azimuthal median profile to remove radial structure
3. **Core Masking**: Optional additional robustness through core dropout
4. **Gate System**: Two gates to validate shortcut removal

### 3. Gate Thresholds

| Gate | Threshold | Purpose |
|------|-----------|---------|
| Core LR AUC | < 0.65 | Ensure model doesn't rely on core brightness |
| Radial LR AUC | < 0.55 | Ensure model doesn't rely on radial profile |
| AUROC | > 0.85 | Ensure model still performs well |

### 4. Mini Experiment Results

| Config | Core AUC | Radial AUC | Gates |
|--------|----------|------------|-------|
| Paired + Residual | 0.93 | ~0.50 | FAIL |
| **Unpaired + Residual** | **0.50** | ~0.50 | **PASS** |

---

## Integration Actions Taken

1. Created `planb/unpaired_experiment/` module with:
   - `preprocess.py`: residual_radial_profile implementation
   - `gates.py`: Cross-validated gate evaluation
   - `build_manifest.py`: LRG-disjoint manifest builder
   - `train.py`: Full training script with gate evaluation

2. Built V2 manifests (metadata-only, partitioned)

3. Started 5 full-scale experiments:
   - A1: Paired + Residual
   - B1: Unpaired + Residual
   - B2: Unpaired + Residual + dropout r=5
   - B3: Unpaired + Residual + scheduled masking
   - B4: Unpaired + Residual + dropout r=3

---

*Original files available in git history if detailed context needed.*

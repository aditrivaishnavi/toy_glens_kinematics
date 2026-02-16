# Gen6/7/8/Uber Priority Plan for Publication

**Date:** 2026-02-05  
**Status:** Gen5 training complete (AUC=0.895), EMR 4c Corrected running

---

## Executive Summary

Based on the LLM's research recommendations and our code review, here is the prioritized plan for achieving publication-quality results in MNRAS/ApJ/AAS.

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| Gen5 Training | ✅ **COMPLETE** | AUC=0.895, TPR@FPR1%=95.8% |
| Gen5 4c Corrected | 🔄 Running | Adds arc_snr_sum, lensed_hlr_arcsec |
| Gen6/7/8 Code | ✅ Fixed | All 8 tests pass |
| Compaction/Relabel | ⏳ Pending | After 4c completes |

---

## Priority Order (Recommended)

### Phase 1: Complete Current Work (This Week)
**Goal:** Finish Gen5 and validate

| Task | Time Est. | Dependencies | Parallelizable? |
|------|-----------|--------------|-----------------|
| 1.1 Wait for EMR 4c to complete | 3-4 hrs | Running | N/A |
| 1.2 Run compaction with split relabeling | 1-2 hrs | 1.1 | No |
| 1.3 Validate Gen5 model on SLACS/BELLS anchors | 2-3 hrs | Gen5 done | Yes (can start now) |
| 1.4 Generate selection function C(θ_E, z_l) curves | 4-6 hrs | 1.2 | No |

### Phase 2: Quick Ablations (1 Week)
**Goal:** Establish which generations help

| Task | Time Est. | Dependencies | Parallelizable? |
|------|-----------|--------------|-----------------|
| 2.1 Prepare deep source bank (Gen6) | 4-8 hrs | Deep cutouts needed | Yes |
| 2.2 Generate Gen7 hybrid manifest | 2-3 hrs | Manifest ready | Yes |
| 2.3 Generate Gen8 domain-randomized manifest | 2-3 hrs | Manifest ready | Yes |
| 2.4 Pilot training: Gen6 (5 epochs) | 3-4 hrs | 2.1 | **Yes - parallel** |
| 2.5 Pilot training: Gen7 (5 epochs) | 3-4 hrs | 2.2 | **Yes - parallel** |
| 2.6 Pilot training: Gen8 (5 epochs) | 3-4 hrs | 2.3 | **Yes - parallel** |
| 2.7 Evaluate pilots on anchor sets | 2-3 hrs | 2.4-2.6 | No |

### Phase 3: Full Ablation Training (1-2 Weeks)
**Goal:** Full training for generations that showed improvement

| Task | Time Est. | Dependencies | Parallelizable? |
|------|-----------|--------------|-----------------|
| 3.1 Full Gen6 training (50 epochs) | 12-24 hrs | 2.7 shows gain | **Yes** |
| 3.2 Full Gen7 training (50 epochs) | 12-24 hrs | 2.7 shows gain | **Yes** |
| 3.3 Full Gen8 training (50 epochs) | 12-24 hrs | 2.7 shows gain | **Yes** |
| 3.4 Uber mix generation | 4-6 hrs | 3.1-3.3 | No |
| 3.5 Uber training (50 epochs) | 12-24 hrs | 3.4 | No |

### Phase 4: Ensemble & Analysis (1 Week)
**Goal:** Build final model and analyze

| Task | Time Est. | Dependencies | Parallelizable? |
|------|-----------|--------------|-----------------|
| 4.1 Temperature calibration for each model | 2-3 hrs | 3.1-3.5 | Yes |
| 4.2 Build calibrated ensemble | 2-3 hrs | 4.1 | No |
| 4.3 Selection function analysis per model | 4-6 hrs | 3.1-3.5 | Yes |
| 4.4 Region-bootstrap confidence intervals | 4-6 hrs | 4.2 | No |
| 4.5 Generate figures and tables | 4-8 hrs | 4.1-4.4 | No |

### Phase 5: Paper Writing (2-3 Weeks)
**Goal:** MNRAS/ApJ submission

| Task | Time Est. | Dependencies | Parallelizable? |
|------|-----------|--------------|-----------------|
| 5.1 Methods section | 1 week | 4.5 | Yes |
| 5.2 Results section | 1 week | 4.5 | Yes |
| 5.3 Discussion + conclusions | 3-5 days | 5.1-5.2 | No |
| 5.4 Code/data release prep | 1 week | All | Yes |

---

## Parallelization Strategy

### What Can Run in Parallel

```
                    ┌─────────────────────┐
                    │  Gen5 Validation    │
                    │  (SLACS/BELLS)      │
                    └─────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Gen6 Pilot    │   │ Gen7 Pilot    │   │ Gen8 Pilot    │
│ (Lambda GPU)  │   │ (Lambda GPU)  │   │ (Lambda GPU)  │
└───────────────┘   └───────────────┘   └───────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ▼
                    ┌─────────────────────┐
                    │  Compare Pilots     │
                    │  Decide Full Runs   │
                    └─────────────────────┘
```

### Requirements for Parallelization

1. **Multiple GPUs or Sequential GPU Time:**
   - If Lambda has 1 GPU: Run pilots sequentially (12-15 hrs total)
   - If multiple GPUs: Run all 3 pilots in parallel (4 hrs)

2. **EMR for Data Generation:**
   - Gen6/7/8 manifests can be generated in parallel on EMR
   - Each needs separate cluster OR sequential steps

3. **Deep Source Bank (Gen6):**
   - Requires access to deep imaging cutouts (HSC or Legacy deep fields)
   - **BLOCKER:** Do we have deep cutouts? If not, Gen6 is delayed.

---

## Critical Decisions Needed

### Decision 1: Deep Source Data
**Question:** Do we have access to deep ground-based cutouts for Gen6?
- If YES: Proceed with Gen6
- If NO: Skip Gen6, focus on Gen7+Gen8

### Decision 2: Publication Strategy
**Options:**
1. **Quick Paper (Gen5 only):** 2-3 weeks
   - Pros: Fast, defensible baseline
   - Cons: Limited novelty

2. **Ablation Paper (Gen5 + best of Gen7/8):** 4-6 weeks
   - Pros: Shows methodology rigor
   - Cons: More work

3. **Comprehensive Paper (Gen5-8 + Uber):** 6-8 weeks
   - Pros: Maximum impact
   - Cons: Most effort, risk of scope creep

**Recommendation:** Option 2 (Ablation Paper) - Best effort/impact ratio

### Decision 3: Split Strategy
**Question:** Implement split relabeling now or later?
- **Now:** More training data, better model
- **Later:** Faster to first result

**Recommendation:** Implement now (already have script ready)

---

## Resource Requirements

### Compute
| Resource | Usage | Est. Cost |
|----------|-------|-----------|
| Lambda GPU (GH200) | Training (5 models × 24 hrs) | ~$600-1000 |
| EMR (34 × m5.2xlarge) | Data generation (3 runs × 6 hrs) | ~$500-800 |
| S3 Storage | ~2TB additional | ~$50/month |

### Data
| Data | Status | Action Needed |
|------|--------|---------------|
| COSMOS bank | ✅ Ready | None |
| Deep cutouts (Gen6) | ❓ Unknown | Need to source |
| SLACS/BELLS anchors | ❓ Need | Download/prepare |
| DR10 hard negatives | ✅ Available | None |

### Time
| Phase | Calendar Days |
|-------|---------------|
| Phase 1 | 2-3 days |
| Phase 2 | 5-7 days |
| Phase 3 | 7-14 days |
| Phase 4 | 5-7 days |
| Phase 5 | 14-21 days |
| **Total** | **5-7 weeks** |

---

## Recommended Immediate Actions

### Today (Priority Order)

1. **Check EMR 4c status** - Is it still running?
2. **Validate Gen5 model exists** - Check Lambda for model files
3. **Start SLACS/BELLS download** - Don't wait, get anchors ready
4. **Confirm deep cutout availability** - Critical for Gen6 decision

### This Week

1. Complete EMR 4c → Compaction → Split relabeling
2. Run Gen5 inference on SLACS/BELLS
3. Decide on Gen6 (based on data availability)
4. Prepare Gen7/Gen8 manifests
5. Start pilot trainings

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Gen6 data unavailable | Medium | High | Focus on Gen7+Gen8 |
| Gen7/8 don't improve | Medium | Medium | Still publish Gen5 + selection function |
| EMR failures | Low | Medium | Code is tested, have retries |
| Reviewer rejection | Medium | High | Pre-empt with thorough ablations |
| Scope creep | High | Medium | Stick to plan, time-box phases |

---

## Success Criteria

### Minimum Viable Paper
- Gen5 model with AUC > 0.85 ✅
- Selection function C(θ_E) for DR10
- Validation on SLACS/BELLS (recall > 50%)
- Comparison to at least one prior method

### Strong Paper
- All of above, PLUS:
- At least one ablation (Gen7 or Gen8) showing measurable improvement
- Region-bootstrap confidence intervals
- Code release

### Outstanding Paper
- All of above, PLUS:
- Full Gen5-8 ablation suite
- Uber ensemble outperforming individuals
- Novel insights on sim-to-real gap
- Large real-lens discovery catalog

---

## LLM Recommendations Summary

From the LLM's detailed response:

1. **Do ablations first, then Uber** - Scientifically defensible
2. **Re-split to ~80/20 train/val** - More training signal
3. **High-value additions:** Lens model variability, substructure, color gradients
4. **Evaluation:** TPR at fixed FPR, bootstrap over regions, 3 seeds minimum
5. **Ensemble:** Calibrate each model, average logits, re-calibrate ensemble
6. **Paper 1 strategy:** Gen5 + selection function + real-anchor validation
7. **Anticipated criticisms:** Sim-real gap, prior dependence, leakage, reproducibility

---

*This plan will be updated as work progresses. See `lessons_learned_and_common_mistakes.md` for error prevention.*

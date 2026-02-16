# Plan C: Full Pipeline Status (v2 - LLM-Aligned)

**Goal:** MNRAS paper on "Selection Functions and Failure Modes of DR10 Lens Finders"

**Last Updated:** 2026-02-05  
**Audit Status:** Fully aligned with LLM blueprint

---

## Phase 1: Data Preparation (Week 1) — IN PROGRESS

### 1.1 Positive Lens Data
| Task | Status | Notes |
|------|--------|-------|
| Download 5,104 DESI candidates (JPG) | ✅ Done | planb/data/desi_lenses_full/ |
| Enrich with Tractor metadata | 🔄 Running | nobs_z, psfsize_z, psfdepth_z, type |
| **Download FITS cutouts (101×101)** | ⏳ Pending | **Need 3-channel grz FITS** |
| Separate Tier-A vs Tier-B | ✅ Done | grading: confident (435) vs probable (4669) |

### 1.2 Analyze Positive Distribution ⏳ PENDING
- [ ] Distribution of nobs_z (z-band exposures)
- [ ] Distribution of psfsize_z (seeing)
- [ ] Distribution of psfdepth_z (depth)
- [ ] Distribution of Tractor type (SER/DEV/REX/EXP)
- [ ] Generate stratification bins matching Huang et al.

### 1.3 Negative Galaxy Sampling ⏳ PENDING
- [ ] Query DataLab for galaxies matching positive distribution
- [ ] **Stratify by (nobs_z, psfsize_z, psfdepth_z, type)** — per LLM
- [ ] **Maintain 100:1 negative:positive ratio per bin** — per LLM
- [ ] Target: ~500,000 negatives
- [ ] Download FITS cutouts (101×101 grz)
- [ ] Verify cutout size exactly 101×101

### 1.4 Contaminant Catalog ⏳ PENDING
| Contaminant Type | Source | Target Count |
|------------------|--------|--------------|
| Ring galaxies | Galaxy Zoo DR2 | ~2,000 |
| Face-on spirals | Galaxy Zoo morphology | ~2,000 |
| Mergers/interactions | Galaxy Zoo mergers | ~2,000 |
| Edge-on disks | Galaxy Zoo edge-on | ~2,000 |
| Bright star artifacts | Gaia + DR10 cross-match | ~2,000 |
| **Total** | | **~10,000** |

**NEW: Contaminant sources identified** (Gap #9 fixed)

---

## Phase 2: Model Training (Week 2)

### 2.1 Data Preparation ⏳ PENDING
- [ ] Create train/val/test splits (70/15/15)
- [ ] Maintain stratification in splits
- [ ] Implement data augmentation:
  - **Rotations (0/90/180/270)** — per LLM
  - **Flips (horizontal, vertical)**
  - **Mild Gaussian noise**
  - **Mild intensity jitter**
- [ ] Build PyTorch DataLoader

### 2.2 Baseline Model ⏳ PENDING

**Architecture Options:**
| Model | Use Case | Notes |
|-------|----------|-------|
| **ResNet-18** | Primary baseline | Start here (LLM recommendation) |
| **EfficientNet-B0** | Alternative | ImageNet pretrained (Gap #6 fixed) |
| ResNet-34 | Only if underfitting | "Deeper nets mostly add capacity you don't have label-quality to exploit" |

**Input Format:**
- Primary: 3-channel (g,r,z) at 101×101
- **Ablation: z-only** (Gap #7 fixed)

**NEW: Metadata Branch (Optional)** (Gap #1 fixed)
```
Image branch: ResNet-18 → 512-dim
Metadata branch: [nexp_z, psfsize_z, psfdepth_z, type_onehot] → MLP → 32-dim
Concat → FC → sigmoid
```
- Only if metadata correctness guaranteed
- Do NOT include sky coordinates or brick IDs (leakage risk)

### 2.3 Training Protocol ⏳ PENDING

**Loss Function:**
- Primary: `BCEWithLogitsLoss` + `pos_weight`
- **Fallback: Focal Loss** if collapse on rare positives (Gap #8 fixed)

**NEW: Label Handling for Tier-B** (Gap #2 fixed)
```python
# Tier-A (confirmed): target = 1.0
# Tier-B (probable): target = 0.8 (label smoothing)
# Or use grade weights: confident=1.0, probable=0.7
```

**Optimizer:**
- AdamW, cosine LR schedule
- Early stopping on validation AUC
- Save best checkpoint

### 2.4 Baseline Evaluation ⏳ PENDING
- [ ] ROC curve, AUC
- [ ] PR curve
- [ ] FPR at various thresholds
- [ ] **Sanity check: Top-K predictions not dominated by artifacts** — per LLM

---

## Phase 3: Selection Function Analysis (Week 3)

### 3.1 Completeness Measurement ⏳ PENDING
- [ ] **Recall on Tier-A anchors ONLY (n=435)** — per LLM
- [ ] Stratify by: nobs_z, psfsize_z, psfdepth_z, type
- [ ] Bootstrap confidence intervals

**NEW: Small-N Uncertainty** (Gap #4 fixed)
```python
# When stratum has N < 30:
# - Report exact binomial interval
# - OR Bayesian beta posterior: Beta(successes+1, failures+1)
# - Flag as "insufficient data" in tables
```

- [ ] Generate completeness heatmaps (nobs vs PSF, nobs vs depth)

### 3.2 Calibration Analysis ⏳ PENDING

**Prevalence-Free Metrics:**
- [ ] ROC/PR curves (no prevalence assumption)
- [ ] Reliability diagrams on labeled sets

**NEW: Scenario-Weighted Calibration** (Gap #3 fixed)
```python
# Deployment prior: assume 1 lens per 10,000 galaxies
deployment_prior = 1 / 10000

# Adjusted precision at threshold t:
# P(lens|score>t) = (recall * prior) / (recall * prior + FPR * (1-prior))
```

- [ ] Expected Calibration Error (ECE)
- [ ] Explicit caveat: "label = training label, not ground truth"

### 3.3 Failure Mode Analysis ⏳ PENDING
- [ ] FPR by contaminant category (rings, spirals, mergers, artifacts)
- [ ] Identify systematic failure patterns
- [ ] GradCAM visualizations (qualitative only, do not oversell as causal)
- [ ] Hard negative mining from high-score false positives

### 3.4 NEW: Spatial Analysis (Gap #5 fixed)
- [ ] **Region holdout cross-validation**
  - Split by sky region (e.g., RA quadrants)
  - Train on 3 regions, test on 1
  - Assess spatial correlation in errors
- [ ] Check for brick-level or footprint-level biases
- [ ] Report any significant spatial variation in completeness

---

## Phase 4: Ensemble Diversification (Week 3-4)

### 4.1 Domain-Split Training ⏳ PENDING

**Split by known confounds** (per LLM):
| Model | Domain | Rationale |
|-------|--------|-----------|
| Model A | Good seeing (psfsize_z < 1.2") | PSF confound |
| Model B | Poor seeing (psfsize_z > 1.4") | PSF confound |
| Model C | Low exposures (nobs_z ≤ 2) | Exposure confound |
| Model D | High exposures (nobs_z ≥ 4) | Exposure confound |
| Model E | SER/DEV types only | Morphology |
| Model F | EXP/REX types only | Morphology |

**Pick ONE axis for 4-week timeline** (LLM advice: don't do all)

### 4.2 Diversity Analysis ⏳ PENDING
- [ ] Prediction correlation on large unlabeled set
- [ ] Disagreement rate near threshold (most relevant for human review)
- [ ] Ensemble entropy / variance

### 4.3 Ensemble Evaluation ⏳ PENDING
- [ ] **Simple averaging first** — per LLM
- [ ] Meta-learner only if averaging beaten on held-out validation
- [ ] Recovery improvement in weak strata
- [ ] Selection function broadening

---

## Phase 5: Paper Writing (Week 4)

### 5.1 Figures ⏳ PENDING
| Figure | Content |
|--------|---------|
| Fig 1 | Example lens cutouts (Tier-A, Tier-B, different quality regimes) |
| Fig 2 | Positive metadata distributions (nobs, PSF, depth, type) |
| Fig 3 | ROC/PR curves with confidence bands |
| Fig 4 | Completeness heatmaps (nobs vs PSF, nobs vs depth) |
| Fig 5 | Reliability diagram + ECE |
| Fig 6 | FPR by contaminant type (bar chart) |
| Fig 7 | Ensemble diversity / improvement |
| Fig 8 | GradCAM failure mode gallery |

### 5.2 Tables ⏳ PENDING
| Table | Content |
|-------|---------|
| Table 1 | Data summary (Tier-A, Tier-B, negatives, contaminants) |
| Table 2 | Model architecture and training details |
| Table 3 | Completeness by stratum (with bootstrap CIs + binomial where N small) |
| Table 4 | FPR by contaminant category |
| Table 5 | Calibration metrics (ECE, scenario-weighted precision) |
| Table 6 | Ensemble vs baseline comparison |

### 5.3 Sections ⏳ PENDING
- [ ] Abstract
- [ ] Introduction (lens finding, selection functions, motivation)
- [ ] Data (Tier-A/Tier-B, negatives, contaminants, stratification)
- [ ] Methods (model, metadata branch, training, label handling)
- [ ] Results (selection function, calibration, failures, spatial)
- [ ] Discussion (implications, limitations, what is/isn't detectable)
- [ ] Conclusions

### 5.4 Reviewer Preemptions (per LLM)
| Objection | Response |
|-----------|----------|
| "Trained on candidates from similar models" | Completeness evaluated on independent Tier-A anchors |
| "Negative sampling not representative" | Stratified by z_nexp; FPR by contaminant reported |
| "Selection function is model-dependent" | Yes; provided for specified baseline + sensitivity via ensemble |
| "Small-N anchors" | Flagged; binomial/beta CIs; spatial correlation assessed |

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: DATA PREPARATION                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  lenscat ──► 5,104 DESI candidates ──► Tractor metadata              │
│                     │                         │                       │
│              Tier-A (435)            Distribution analysis            │
│              Tier-B (4669)           (nobs, PSF, depth, type)        │
│                     │                         │                       │
│                     ▼                         ▼                       │
│              FITS cutouts ◄───── Stratification bins                 │
│              (3×101×101 grz)              │                           │
│                                           ▼                           │
│  DataLab ◄────────────────────────────────┘                          │
│     │                                                                 │
│     ▼                                                                 │
│  500K negatives ──► FITS cutouts (matched distribution)              │
│                                                                       │
│  Galaxy Zoo ──► 10K contaminants ──► FITS cutouts                    │
│  + Gaia                                                               │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        PHASE 2: TRAINING                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Train (70%) ──► ResNet-18 ──► Baseline                              │
│                  + optional metadata branch                           │
│                  + Tier-B label smoothing (0.8)                      │
│                                                                       │
│  Val (15%) ──► Early stopping (AUC)                                  │
│  Test (15%) ──► Final evaluation                                     │
│                                                                       │
│  Ablations: EfficientNet-B0, z-only, focal loss                      │
│                                                                       │
│  Domain splits ──► Specialized models ──► Ensemble (avg)             │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        PHASE 3: EVALUATION                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Tier-A (435) ──► Completeness by stratum                            │
│                   + Bootstrap CI                                      │
│                   + Binomial/beta for small N                        │
│                   + Spatial holdout CV                               │
│                          │                                            │
│                          ▼                                            │
│                   Selection function: P(detect | nobs, PSF, depth)   │
│                                                                       │
│  Contaminants ──► FPR by category ──► Failure modes                  │
│                                                                       │
│  All predictions ──► Calibration                                     │
│                      + Reliability diagram                           │
│                      + ECE                                           │
│                      + Scenario-weighted (1:10000 prior)             │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        PHASE 5: PAPER                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Key claim: "Detection probability = f(nobs, PSF, depth, type)"      │
│                                                                       │
│  Novel contributions:                                                 │
│    1. Rigorous selection function for DR10 lens finding              │
│    2. Bias audit tied to operational choices                         │
│    3. Controlled ensemble diversification study                      │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Gap Resolution Summary

| Gap | Issue | Resolution | Status |
|-----|-------|------------|--------|
| #1 | Metadata branch missing | Added optional metadata branch to 2.2 | ✅ Fixed |
| #2 | Label smoothing for Tier-B | Added to 2.3 Training Protocol | ✅ Fixed |
| #3 | Scenario-weighted calibration | Added to 3.2 with 1:10000 prior | ✅ Fixed |
| #4 | Binomial/Bayesian CI | Added to 3.1 for small-N strata | ✅ Fixed |
| #5 | Spatial holdout | Added new section 3.4 | ✅ Fixed |
| #6 | EfficientNet-B0 option | Added to 2.2 architecture options | ✅ Fixed |
| #7 | z-only ablation | Added to 2.2 input format | ✅ Fixed |
| #8 | Focal loss fallback | Added to 2.3 loss function | ✅ Fixed |
| #9 | Contaminant sources | Galaxy Zoo + Gaia identified in 1.4 | ✅ Fixed |

---

## Current Status

| Phase | Duration | Progress |
|-------|----------|----------|
| Phase 1: Data Prep | 3-4 days | 40% |
| Phase 2: Training | 2-3 days | 0% |
| Phase 3: Evaluation | 2-3 days | 0% |
| Phase 4: Ensemble | 2-3 days | 0% |
| Phase 5: Paper | 3-5 days | 0% |
| **Total** | **~2-3 weeks** | **~15%** |

**Current blocker:** Metadata enrichment running (51% complete, ~7 min remaining)

---

## Immediate Next Steps

1. ⏳ Wait for metadata enrichment to complete
2. Download FITS cutouts for all 5,104 candidates
3. Analyze positive distribution
4. Design stratification bins
5. Query negatives from DataLab (Galaxy Zoo for contaminants)
6. Begin training pipeline

---

## Week-by-Week Checkpoints (per LLM)

### Week 1 Checkpoints
- [ ] Tier-A/Tier-B separated (confirmed = 435, probable = 4669)
- [ ] Tractor metadata validated on random sample
- [ ] Stratified negative catalog built (matched z_nexp per type)
- [ ] Cutout size verified = 101×101

### Week 2 Checkpoints
- [ ] ResNet-18 baseline trained
- [ ] Held-out test AUC stable
- [ ] Train/val curves stable (no collapse)
- [ ] Top-K predictions not dominated by artifacts

### Week 3 Checkpoints
- [ ] Recall vs (PSF, depth, nexp) on Tier-A only
- [ ] Bootstrapped CIs computed
- [ ] FPR by contaminant category
- [ ] In best strata, Tier-A recall > worst strata

### Week 4 Checkpoints
- [ ] Domain-specialized models (one axis)
- [ ] Diversity metrics + performance delta
- [ ] Paper-quality figures ready

# Audit: My Pipeline vs LLM Blueprint

Cross-checking FULL_PIPELINE_STATUS.md against LLM_BLUEPRINT_RESPONSE.md

---

## ✅ Correct Alignments

### Data Preparation
| LLM Said | My Status | Match? |
|----------|-----------|--------|
| Tier-A and Tier-B separated | Listed as done | ✅ |
| Tractor metadata match | Running now | ✅ |
| Stratified negatives by z_nexp | Listed | ✅ |
| 100:1 negative:lens ratio per bin | Listed | ✅ |
| 101×101 pixel cutouts | Implicit in download | ✅ |

### Architecture
| LLM Said | My Status | Match? |
|----------|-----------|--------|
| ResNet-18 first | Listed | ✅ |
| BCEWithLogits + pos_weight | Listed | ✅ |
| AdamW, cosine LR | Listed | ✅ |
| Early stopping on val AUC | Listed | ✅ |
| 3-channel (g,r,z) input | Implicit | ✅ |

### Evaluation
| LLM Said | My Status | Match? |
|----------|-----------|--------|
| Recall on Tier-A anchors only | Listed | ✅ |
| Bootstrap CIs | Listed | ✅ |
| FPR by contaminant category | Listed | ✅ |
| Reliability diagrams | Listed | ✅ |
| GradCAM (qualitative) | Listed | ✅ |

### Ensemble
| LLM Said | My Status | Match? |
|----------|-----------|--------|
| Domain splits by confounds | Listed (PSF, nexp) | ✅ |
| Prediction correlation | Listed | ✅ |
| Simple averaging first | Listed | ✅ |

---

## ⚠️ DISCREPANCIES / MISSING ITEMS

### 1. Train/Val Split Ratio
| LLM Said | My Status | Issue |
|----------|-----------|-------|
| Train/val 70/30 | 70/15/15 (train/val/test) | **Different!** |

**LLM explicitly said "Train/validation split 70/30"** - no separate test set mentioned. But my status has 70/15/15. This may be fine (test set is good practice), but needs verification.

### 2. Metadata Branch (MISSING)
| LLM Said | My Status | Issue |
|----------|-----------|-------|
| Optional metadata branch: nexp_z, psfsize_r, psfdepth_r, tractor_type | **NOT MENTIONED** | ❌ Missing |

**The LLM suggested a metadata branch** for the classifier (if metadata correctness guaranteed). I didn't include this option anywhere.

### 3. Specific Stratification Variables
| LLM Said | My Status | Issue |
|----------|-----------|-------|
| z_nexp (z-band exposure count) | nobs_z | ✅ Same |
| psfsize_r (or z) | psfsize_z | ✅ OK |
| psfdepth_r (or z) | psfdepth_z | ✅ OK |
| Tractor type | type | ✅ OK |

### 4. Label Smoothing for Tier-B (MISSING)
| LLM Said | My Status | Issue |
|----------|-----------|-------|
| Tier-B with label smoothing, grade weights | **NOT MENTIONED** | ❌ Missing |

**LLM said:** "Tier-B is used for training (with label smoothing, and/or grade weights)"
I didn't include this in training protocol.

### 5. Scenario-Weighted Calibration (MISSING)
| LLM Said | My Status | Issue |
|----------|-----------|-------|
| Scenario-weighted calibration for deployment prior (10,000:1) | **NOT MENTIONED** | ❌ Missing |

**LLM said:** Report "scenario-weighted calibration for a claimed deployment prior (example: 10,000:1)"
My status only mentions "Prevalence-adjusted metrics" vaguely.

### 6. Augmentations Details (INCOMPLETE)
| LLM Said | My Status | Issue |
|----------|-----------|-------|
| Rotations (0/90/180/270), flips; mild Gaussian noise; mild intensity jitter | "rotations, flips, noise" | ⚠️ Incomplete |

I had the right idea but less specific.

### 7. Top-K Sanity Check (PRESENT)
| LLM Said | My Status | Match? |
|----------|-----------|--------|
| Top-K predictions not dominated by artifacts | Listed | ✅ |

### 8. Strata with "Insufficient Data" Flagging (PRESENT)
| LLM Said | My Status | Match? |
|----------|-----------|--------|
| Mark strata with insufficient data | Listed | ✅ |

### 9. Exact Binomial / Bayesian Beta Posterior (MISSING)
| LLM Said | My Status | Issue |
|----------|-----------|-------|
| "When N is tiny, also report exact binomial interval or Bayesian beta posterior" | Only "Bootstrap CIs" | ⚠️ Incomplete |

### 10. Spatial Correlation / Region Holdouts (MISSING)
| LLM Said | My Status | Issue |
|----------|-----------|-------|
| "spatial correlation is assessed via region holdouts" | **NOT MENTIONED** | ❌ Missing |

This is mentioned in reviewer objection #4.

### 11. EfficientNet Option (MISSING)
| LLM Said | My Status | Issue |
|----------|-----------|-------|
| EfficientNet B0/B1 as option | **NOT MENTIONED** | ⚠️ Could add |

LLM said: "EfficientNet: B0 (or B1 if you have enough GPU RAM). Use ImageNet pretraining for faster convergence."

### 12. z-only Ablation (MISSING)
| LLM Said | My Status | Issue |
|----------|-----------|-------|
| "z-only is a useful ablation" | **NOT MENTIONED** | ⚠️ Could add |

### 13. Focal Loss Alternative (MISSING)
| LLM Said | My Status | Issue |
|----------|-----------|-------|
| "focal loss if you see collapse on rare positives" | **NOT MENTIONED** | ⚠️ Could add |

---

## ❌ CRITICAL GAPS

### Gap 1: Tier-A Anchor List Not Defined
**LLM checkpoint test:** "Are your negatives matched by z-exposure bin?"
**LLM evaluation:** "Recall on Tier-A anchors only"

**Problem:** I claimed "Separate Tier-A (confirmed) vs Tier-B (candidates)" as done, but we haven't actually verified which lenses in the DESI catalog are confirmed (Tier-A) vs candidates (Tier-B). The lenscat data has a `confirmed` flag, but we need to verify this is properly used.

### Gap 2: No Contaminant Source Defined
**LLM said:** "Build contaminant sets explicitly"

**My status:** Listed ring galaxies, spirals, mergers, etc. but no source identified.
**Problem:** Where do we get these? Galaxy Zoo? Need specific catalog sources.

### Gap 3: Cutout Size Verification
**LLM checkpoint:** "Confirm cutout size exactly 101×101"

**My status:** Listed as implicit, but not verified.
**Problem:** Need to verify the downloaded cutouts are 101×101.

---

## 🔧 FIXES NEEDED TO FULL_PIPELINE_STATUS.md

1. **Add metadata branch option** to Phase 2
2. **Add label smoothing/grade weighting** for Tier-B to Phase 2
3. **Add scenario-weighted calibration (10,000:1 prior)** to Phase 3
4. **Add binomial/Bayesian posterior** for small-N strata to Phase 3
5. **Add spatial holdout / region-based CV** to Phase 3
6. **Add EfficientNet B0 option** to Phase 2
7. **Add z-only ablation** to Phase 2
8. **Add focal loss fallback** to Phase 2
9. **Add specific contaminant sources** to Phase 1
10. **Add cutout size verification** to Phase 1
11. **Verify Tier-A/Tier-B separation logic** in Phase 1

---

## Week-by-Week Checkpoint Cross-Check

### Week 1 Checkpoints (LLM)
- [ ] Positive catalog ingested (Tier-A and Tier-B separated) — **Need to verify separation logic**
- [ ] Local Tractor metadata match validated on random sample — **In progress**
- [ ] Stratified negative catalog built — **Pending**
- [ ] Negatives matched by z-exposure bin — **Pending**
- [ ] Cutout size exactly 101×101 — **Need to verify**

### Week 2 Checkpoints (LLM)
- [ ] ResNet-18 baseline trained — **Pending**
- [ ] Held-out test set with stable AUC — **Pending**
- [ ] Train/val curves stable (no collapse) — **Pending**
- [ ] Top-K predictions not dominated by artifacts — **Pending**

### Week 3 Checkpoints (LLM)
- [ ] Recall vs (PSF, depth, nexp) using Tier-A anchors only — **Pending**
- [ ] Bootstrapped CIs — **Pending**
- [ ] FPR by contaminant category — **Pending**
- [ ] In best strata, Tier-A recall higher than worst — **Pending**

### Week 4 Checkpoints (LLM)
- [ ] Domain-specialized models (one split axis) — **Pending**
- [ ] Diversity metrics + performance delta — **Pending**
- [ ] Paper-quality figures — **Pending**

---

## Summary

| Category | Items Correct | Items Missing/Wrong |
|----------|---------------|---------------------|
| Data Prep | 5/7 | 2 (contaminant source, cutout verify) |
| Training | 5/9 | 4 (metadata branch, label smooth, focal, EfficientNet) |
| Evaluation | 6/9 | 3 (binomial, scenario-weighted, spatial holdout) |
| Ensemble | 3/3 | 0 |
| **Total** | **19/28** | **9 gaps** |

**Alignment score: ~68%**

Most core elements are correct, but several important details from the LLM blueprint are missing.

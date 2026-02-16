# Phase 1 Validation Results: Paired Controls Analysis

**Date:** 2026-02-05  
**Purpose:** Verify if paired controls fix the shortcut learning problem

---

## Executive Summary

**KEY FINDING:** The core brightness shortcut is due to **physical arc overlap**, NOT LRG selection bias. Paired controls alone will NOT fix the problem.

| Metric | Unpaired Data | Paired Data | Interpretation |
|--------|---------------|-------------|----------------|
| Central brightness ratio | 1.64x | 1.67x | Nearly identical! |
| LRG selection bias | - | -3.2% | Negligible |
| Injection contribution | - | +67.1% | Arc flux in center |
| Central-Only AUC | 0.98 | 0.76 | Still predictive |

**Conclusion:** Need center degradation augmentation, not just paired controls.

---

## 1. Cutout Determinism Test

**Purpose:** Verify if we can salvage existing dataset by re-fetching paired controls.

**Result:** ✓ PASS

| Metric | Value |
|--------|-------|
| Samples tested | 10 |
| Match rate | 100% |
| Max pixel difference | 0.0 |
| Correlation | 1.0 |

**Decision:** Cutout re-fetching is perfectly deterministic. Salvage approach is viable.

---

## 2. Theta-E and PSF Distribution

**Purpose:** Understand the physical constraints of the data.

| Parameter | Min | Max | Mean |
|-----------|-----|-----|------|
| theta_e (arcsec) | 0.50" | 2.50" | 1.35" |
| PSF size (arcsec) | 0.97" | 1.60" | 1.32" |
| Core radius (theta_e - 1.5*PSF) | 0.0 px | 3.6 px | 0.3 px |

**Key insight:** For 92.3% of samples, the core radius is < 2 pixels. The Einstein ring is smaller than or comparable to the PSF, meaning the arc flux is smeared into the central region.

---

## 3. Arc-Annulus vs Core Analysis (Large theta_e = 1.3-2.5")

**Purpose:** Verify that injection adds flux to arc region, not core.

| Region | Injection Contribution | As % of Control |
|--------|----------------------|-----------------|
| Core (theta-aware mask) | +0.0046 | +8.0% |
| Arc Annulus | +0.0063 | +145.9% |

**Conclusion:** Arc annulus has 18x more injection flux than core. Injection is working correctly - arc is at Einstein radius.

---

## 4. Fixed Central Mask Analysis (r < 8 pixels)

**Purpose:** Compare paired vs unpaired data using the same mask as original shortcut analysis.

### Unpaired Gen5 Data (original problem)
| Metric | Positives | Controls | Ratio |
|--------|-----------|----------|-------|
| Central mean (r-band) | 0.0336 | 0.0205 | 1.64x |

### Paired Data (same LRG)
| Metric | Positives | Controls | Ratio |
|--------|-----------|----------|-------|
| Central mean (r-band) | 0.0290 | 0.0174 | 1.67x |

### Attribution of Brightness Difference

| Source | Contribution |
|--------|--------------|
| LRG selection bias | -3.2% |
| Arc overlap with center | +67.1% |
| **Total** | ~64% |

**Critical finding:** The 64% brightness difference in unpaired data is almost entirely due to **physical arc overlap** (67.1%), with essentially zero contribution from LRG selection bias (-3.2%).

---

## 5. Central-Only AUC Comparison

| Dataset | Central-Only AUC | Interpretation |
|---------|-----------------|----------------|
| Unpaired Gen5 | 0.98 | Trivial shortcut |
| Paired Data | 0.76 | Still predictive! |

**Why is paired AUC still 0.76?** Because the injection itself adds 67% more flux to the central region. Even with the same base LRG, positives are brighter in the center due to arc overlap.

---

## 6. Implications for Remediation

### What Works
- ✓ Cutout re-fetching is deterministic (salvage viable)
- ✓ Injection physics is correct (arc at Einstein radius)
- ✓ Theta-aware masking works for large theta_e

### What Doesn't Work
- ✗ Paired controls alone don't fix the shortcut
- ✗ Fixed r<8 mask includes arc for most samples

### Required Next Steps

1. **Center Degradation Augmentation**
   - Blur or mask central region during training
   - Force model to learn arc morphology in outer regions
   - LLM recommendation: Gaussian blur with sigma ~ PSF before normalization

2. **Photometric Jitter**
   - Random gain variations to break absolute flux shortcuts
   - Already in LLM's `shortcut_resistant_augmentation` code

3. **Hard Negatives**
   - Add spirals, rings, mergers to teach contaminant rejection
   - 20% of training data from Galaxy Zoo DECaLS

---

## 7. Code and Data Artifacts

| Artifact | Location |
|----------|----------|
| Determinism test results | `/lambda/nfs/darkhaloscope-training-dc/phase1_determinism_results.json` |
| Paired pilot data | `/data/paired_export/` on emr-launcher |
| Validation results | `/data/paired_export/simplified_validation_results.json` |
| Theta-aware validation script | `dark_halo_scope/scripts/theta_aware_validation.py` |

---

## 8. Questions for LLM Review

1. **Is center degradation the right solution?** Given that arc overlap is physical, should we blur the center to force outer-region learning, or is there a better approach?

2. **What about different theta_e regimes?** For large theta_e (> 2"), the arc IS separable from core. Should we stratify training by theta_e?

3. **Model architecture changes?** Would attention mechanisms that explicitly attend to the arc annulus help?

4. **Expected performance impact?** After center degradation, what AUC should we expect on:
   - Synthetic test set
   - Real anchors (Tier-A vs Tier-B)

5. **Is 0.76 AUC on paired data acceptable?** Or does this still indicate a problem that center degradation should fix?

---

## Appendix: Raw Validation Output

```
======================================================================
SIMPLIFIED VALIDATION: Fixed r<8px Central Region
======================================================================
Loaded 1000 paired samples

======================================================================
GATE 1: Central-Only AUC (r < 8px)
======================================================================
Central-Only AUC: 0.7601
⚠ Central features still predictive (expected due to arc overlap)

======================================================================
GATE 3: Core Brightness Match
======================================================================
Positives central mean: 0.029028
Controls central mean:  0.017369
Ratio: 1.6713

Injection contribution to center: 0.011659
As % of control: 67.1%

======================================================================
KEY COMPARISON: PAIRED vs UNPAIRED
======================================================================
In UNPAIRED Gen5 data:
  Positives core: 0.0336
  Controls core:  0.0205
  Ratio: 1.64x (64% brighter)

In PAIRED data:
  Positives core: 0.0290
  Controls core:  0.0174
  Injection contribution: 67.1%

The 64% difference in unpaired data was due to:
  - Base LRG selection bias: -3.2%
  - Arc overlap with center: 67.1%
```

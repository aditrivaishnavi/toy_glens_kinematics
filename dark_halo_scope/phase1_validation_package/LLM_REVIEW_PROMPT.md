# LLM Review Request: Strong Lens Detection Shortcut Analysis

**Date:** 2026-02-05  
**Project:** DarkHaloScope - Strong Gravitational Lens Finder for DESI DR10  
**Target Venues:** MNRAS, ApJ

---

## 1. Executive Summary

We are building a CNN-based strong gravitational lens finder for ground-based imaging (DESI Legacy DR10). During validation, we discovered that our model achieves **AUC = 0.98** on synthetic data but **4.4% recall** on real known lenses.

Investigation revealed the model learned a **brightness shortcut** instead of arc morphology. We implemented **paired controls** (same LRG with and without injection) to fix this, but discovered the shortcut **persists** even with paired data.

**We need your expert review of our analysis and recommendations for next steps.**

---

## 2. The Problem: Shortcut Learning

### 2.1 Original Discovery (Unpaired Data)

We trained a lens finder on synthetic data:
- **Positives:** LRG cutouts + injected lensed arc
- **Controls:** Different LRG cutouts (no injection)

A simple logistic regression on ONLY the central 10x10 pixels achieved:
```
Central-Only AUC: 0.98 (should be ~0.50 if no shortcut)
```

Core brightness comparison:
```
Positives central mean (r-band): 0.0336
Controls central mean (r-band):  0.0205
Ratio: 1.64x (positives 64% brighter)
```

**Hypothesis:** The model learned "bright center = lens" instead of detecting arc morphology.

### 2.2 Paired Controls Experiment

To test if this was due to LRG selection bias (positives from different/brighter LRGs than controls), we generated **paired controls**:
- For each positive, re-fetch the **same LRG cutout** from DR10 without injection
- Same sky coordinates, same base galaxy

**Paired Data Results:**
```
Positives central mean (r-band): 0.0290
Controls central mean (r-band):  0.0174
Ratio: 1.67x (STILL 67% brighter!)

Central-Only AUC: 0.76 (still predictive!)
```

### 2.3 Attribution Analysis

We decomposed the 64% brightness difference:
```
Source                      Contribution
─────────────────────────────────────────
LRG selection bias:         -3.2%  (negligible!)
Arc overlap with center:    +67.1% (dominant)
─────────────────────────────────────────
Total:                      ~64%
```

**Key Finding:** The shortcut is due to **physical arc overlap with the central region**, NOT data selection bias.

---

## 3. Physical Context

### 3.1 Ground-Based Imaging Constraints

| Parameter | Value | Implication |
|-----------|-------|-------------|
| Pixel scale | 0.262"/pixel | 64px stamp = 16.8" |
| Mean PSF FWHM | 1.32" | ~5 pixels |
| Mean Einstein radius | 1.35" | ~5 pixels |
| Typical arc radius | 0.5-2.5" | 2-10 pixels |

### 3.2 The Core Mask Problem

Using theta-aware masking (core = r < θ_E - 1.5×PSF):
```
Core radius distribution:
  Min: 0.0 px
  Max: 3.6 px
  Mean: 0.3 px
  Samples with core_r > 2px: 7.7%
```

**For 92.3% of samples, the Einstein ring is smaller than or comparable to the PSF, so arc flux is smeared into the central region.**

### 3.3 Arc-Annulus Analysis (Large θ_E = 1.3-2.5")

For samples where we CAN separate core from arc:
```
Region              Injection Contribution   As % of Control
───────────────────────────────────────────────────────────
Core (r < θ-1.5*PSF)    +0.0046              +8.0%
Arc Annulus (|r-θ|<1.5*PSF)  +0.0063         +145.9%
```

**The injection IS working correctly - arc is at Einstein radius, not center. The issue is PSF smearing.**

---

## 4. Validation Code and Results

### 4.1 Cutout Determinism Test

**Purpose:** Verify we can salvage existing data by re-fetching paired controls.

**Result:** ✓ PASS (100% match, max_diff=0.0, corr=1.0)

```python
# verify_cutout_determinism.py (included in zip)
# Key result:
for i in range(10):
    stored = load_from_lambda(sample_i)
    refetched = fetch_from_s3(ra, dec, brickname)
    diff = stored - refetched
    print(f"max_diff={np.max(np.abs(diff))}")  # All 0.0
```

### 4.2 Paired Controls Generation

**Code:** `generate_paired_pilot.py` (included in zip)

```python
# For each positive sample:
pos_stamp = load_positive(task_id)  # Base LRG + injected arc
ctrl_stamp = fetch_base_cutout(ra, dec, brickname)  # Same LRG, no arc

# Core brightness comparison
h, w = 64, 64
cy, cx = h // 2, w // 2
yy, xx = np.ogrid[:h, :w]
core_mask = ((yy - cy)**2 + (xx - cx)**2) < 8**2  # r < 8 pixels

pos_core = pos_stamp[1][core_mask].mean()  # r-band
ctrl_core = ctrl_stamp[1][core_mask].mean()
injection_contribution = pos_core - ctrl_core
```

### 4.3 Full Validation Results

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

### 4.4 Arc-Annulus AUC (Positive Result)

```
======================================================================
GATE 2: Arc-Annulus Baseline AUC
======================================================================
Samples: 1000 (500 pos, 500 ctrl)
Arc-Annulus AUC: 0.7062
✓ PASS: Arc features are predictive (AUC ≥ 0.70)
```

---

## 5. Our Proposed Solution

Based on our analysis, we propose **center degradation augmentation**:

```python
def center_degradation(img, sigma_pix=4.0, prob=0.5):
    """
    Randomly blur central region during training.
    Forces model to learn arc morphology in outer regions.
    """
    if np.random.random() > prob:
        return img
    
    h, w = img.shape[-2:]
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    center_mask = r < 10  # 10 pixel radius
    
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(img, sigma_pix)
    out = img.copy()
    out[center_mask] = blurred[center_mask]
    return out
```

**Expected outcome:**
- Central-Only AUC drops from 0.76 → ~0.55 (near random)
- Model learns arc morphology instead of brightness
- Real anchor recall improves

---

## 6. Direct Questions for Review

### Q1: Is Our Analysis Correct?

We concluded that the 64% core brightness difference is due to:
- **3.2% from LRG selection bias** (negligible)
- **67.1% from arc overlap with center** (physical)

**Is this attribution methodology sound? Are there confounds we missed?**

### Q2: Is Center Degradation the Right Solution?

Given that arc overlap is physical, we propose blurring the center during training. 

**Alternatives we considered:**
- Mask center with noise instead of blur
- Only use large-θ_E samples where arc is separable
- Attention mechanisms to focus on outer regions

**Which approach would you recommend and why?**

### Q3: What Central-Only AUC Should We Target?

Currently:
- Unpaired: 0.98
- Paired: 0.76

**After center degradation, what AUC indicates the shortcut is fixed?**
- a) < 0.55 (near random)
- b) < 0.60 (weak predictability acceptable)
- c) Other threshold?

### Q4: Is Arc-Annulus AUC = 0.71 Sufficient?

The arc annulus alone achieves AUC = 0.71. 

**Is this strong enough to build a robust lens finder? What arc-annulus AUC would indicate:**
- a) Insufficient signal
- b) Marginal signal  
- c) Strong signal

### Q5: What About Real Anchor Recall?

Original Gen5 achieved 4.4% recall on SLACS/BELLS anchors. We attributed this to:
1. SLACS/BELLS lenses were discovered via spectroscopy + HST, not ground-based
2. Their arcs are often invisible in DR10 imaging (arc_visibility_snr < 2.0)

**After fixing the shortcut, what recall should we expect on:**
- Tier-A anchors (ground-visible, SNR > 2.0)
- Tier-B anchors (HST-only visible)

### Q6: How Do SOTA Papers Handle This?

We found references to:
- Petrillo 2018: Normalize by peak brightness
- Jacobs 2019: Real non-lenses + contaminants

**What specific techniques do modern lens finder papers use to prevent brightness shortcuts? Can you cite specific papers and their approaches?**

### Q7: Is Our Parameter Space Too Narrow?

Current injection parameters:
```
θ_E: [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5] arcsec
src_dmag: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] mag fainter than lens
```

**Should we expand to include:**
- Larger θ_E (3.0", 4.0") where arc is clearly separable?
- Fainter sources (src_dmag > 3.0) for harder examples?

### Q8: Center Degradation Implementation Details

**Which is better:**
- a) Blur center (preserves some structure)
- b) Replace center with noise (complete removal)
- c) Blend (50% original + 50% blur)

**What blur sigma should we use?**
- PSF-matched (1.3" = 5 pixels)
- Larger (2× PSF = 10 pixels)
- Adaptive based on θ_E

### Q9: Training Strategy

**Should we:**
- a) Apply center degradation to 100% of samples
- b) Apply randomly (50% probability)
- c) Apply only to positives (not controls)
- d) Curriculum: start with degradation, anneal off

### Q10: Validation Protocol After Fix

**What gates should we run to confirm the fix worked?**

Our current plan:
1. Central-Only AUC < 0.55
2. Arc-Annulus AUC > 0.70
3. Full-image AUC > 0.90
4. Tier-A anchor recall > 50%

**Is this sufficient? What else should we check?**

---

## 7. Files Included in This Package

| File | Description |
|------|-------------|
| `verify_cutout_determinism.py` | Tests if re-fetching cutouts is deterministic |
| `generate_paired_pilot.py` | Generates paired positive/control samples |
| `theta_aware_validation.py` | Runs theta-aware validation gates |
| `phase1_validation_results.md` | Full validation report |
| `lessons_learned_and_common_mistakes.md` | All mistakes and lessons from project |
| `LLM_REVIEW_PROMPT.md` | This document |

---

## 8. Summary

**What we know:**
1. ✓ Cutout re-fetching is deterministic (salvage viable)
2. ✓ Arc injection is correct (arc at Einstein radius)
3. ✓ Arc-annulus features ARE predictive (AUC = 0.71)
4. ✗ Core brightness shortcut persists with paired data
5. ✗ The cause is physical arc overlap, not data bias

**What we need:**
1. Validation that our analysis is correct
2. Confirmation that center degradation is the right approach
3. Specific implementation recommendations
4. Expected performance targets after fix

**Timeline:** We aim to implement the fix and retrain within 1 week.

---

*Thank you for your review. Please be direct and critical - we need honest feedback to ensure scientific rigor.*

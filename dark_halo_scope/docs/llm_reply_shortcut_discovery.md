# LLM Reply: Shortcut Discovery and Root Cause Analysis

**Date:** 2026-02-05  
**Context:** Following up on your review of our Gen5 validation work

---

## Executive Summary

Your recommended shortcut gates revealed **CATASTROPHIC FAILURES**:

| Gate | Name | Result | Key Finding |
|------|------|--------|-------------|
| 1.5 | Normalization-Stat Leakage | **FAIL** | Clipping fraction differs by d=0.66 in g-band |
| 1.6 | Core-Only Baseline | **FAIL** | Logistic regression on r<10px achieves **AUC=0.98** |
| 1.7 | Arc-Suppressed Test | **FAIL** | Model still predicts p=0.62 with arcs removed |
| 1.8 | Core Brightness Matching | **FAIL** | Positives are **64% brighter** in core than controls |
| Phase 2 Extended | Center Masking | Confirms | Controls go UP 10x when masked |

**Root Cause:** Positives and controls have dramatically different core brightness distributions. The model learned "bright center = lens" instead of "arc morphology = lens".

---

## Gate 1.5: Normalization-Stat Leakage

### Code

```python
#!/usr/bin/env python3
"""Gate 1.5: Normalization-Stat Leakage Gate"""
import numpy as np
import pyarrow.dataset as ds
import io
from scipy.stats import ks_2samp

CLIP_THRESHOLD = 10.0
INNER_FRAC = 0.5

def compute_norm_stats(img, clip=CLIP_THRESHOLD, inner_frac=INNER_FRAC, eps=1e-6):
    C, H, W = img.shape
    cy, cx = H // 2, W // 2
    ri = int(min(H, W) * inner_frac / 2)
    yy, xx = np.ogrid[:H, :W]
    outer_mask = ((yy - cy)**2 + (xx - cx)**2) > ri**2
    
    stats = {}
    for c, band in enumerate(['g', 'r', 'z']):
        v = img[c]
        outer_v = v[outer_mask]
        med = np.median(outer_v)
        mad = np.median(np.abs(outer_v - med))
        scale = 1.4826 * mad + eps
        normed = (v - med) / scale
        clip_frac = np.mean(np.abs(normed) >= clip)
        
        stats[f'outer_median_{band}'] = float(med)
        stats[f'outer_mad_{band}'] = float(mad)
        stats[f'clip_frac_{band}'] = float(clip_frac)
    return stats

# Sample 5000 per class, compute stats, compare with KS test and Cohen's d
```

### Results

| Stat | Ctrl Mean | Pos Mean | Cohen's d | Status |
|------|-----------|----------|-----------|--------|
| outer_median_r | 0.000666 | 0.000769 | 0.092 | PASS |
| outer_mad_r | 0.002769 | 0.002784 | 0.011 | PASS |
| **clip_frac_r** | 0.0158 | 0.0253 | **0.343** | FAIL |
| **outer_median_g** | 0.000338 | 0.000449 | **0.158** | FAIL |
| outer_mad_g | 0.001570 | 0.001631 | 0.080 | PASS |
| **clip_frac_g** | 0.0088 | 0.0305 | **0.658** | FAIL |
| outer_median_z | 0.001694 | 0.001800 | 0.045 | PASS |
| outer_mad_z | 0.007226 | 0.007272 | 0.016 | PASS |
| clip_frac_z | 0.013915 | 0.015621 | 0.079 | PASS |

**Interpretation:** Positives have 3.5x higher g-band clipping fraction than controls. The injected arcs push more pixels past the clip threshold.

---

## Gate 1.6: Core-Only Baseline Classifier

### Code

```python
#!/usr/bin/env python3
"""Gate 1.6: Core-Only Baseline Classifier"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CORE_RADIUS = 10  # pixels

def extract_core_features(img, r_core=CORE_RADIUS):
    C, H, W = img.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    core_mask = ((yy - cy)**2 + (xx - cx)**2) < r_core**2
    
    features = []
    for c in range(C):
        core_pixels = img[c][core_mask]
        features.extend([
            np.mean(core_pixels),
            np.std(core_pixels),
            np.median(core_pixels),
            np.max(core_pixels),
            np.min(core_pixels),
            np.percentile(core_pixels, 25),
            np.percentile(core_pixels, 75),
        ])
    return np.array(features)

# Train logistic regression on 20k samples, evaluate on 10k
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)
val_auc = roc_auc_score(y_val, clf.predict_proba(X_val_scaled)[:, 1])
```

### Results

| Metric | Value |
|--------|-------|
| Train AUC | 0.9909 |
| **Val AUC** | **0.9800** |
| Top Feature | g_q25 (importance=17.19) |
| 2nd Feature | g_median (importance=10.83) |
| 3rd Feature | r_median (importance=7.86) |

**Interpretation:** A simple logistic regression on ONLY the central 10 pixels achieves 98% AUC! This is definitive proof of a core-based shortcut. The model doesn't need to see arcs at all.

---

## Gate 1.7: Arc-Suppressed Positive Test

### Code

```python
#!/usr/bin/env python3
"""Gate 1.7: Arc-Suppressed Positive Test"""

ARC_INNER_RADIUS = 10  # pixels
ARC_OUTER_RADIUS = 25  # pixels

def suppress_arc_annulus(img, r_inner=10, r_outer=25):
    C, H, W = img.shape
    out = img.copy()
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r2 = (yy - cy)**2 + (xx - cx)**2
    
    arc_mask = (r2 >= r_inner**2) & (r2 < r_outer**2)
    outer_mask = r2 >= r_outer**2
    
    for c in range(C):
        outer_vals = img[c][outer_mask]
        n_arc = arc_mask.sum()
        fill_vals = np.random.choice(outer_vals, size=n_arc, replace=True)
        out[c][arc_mask] = fill_vals
    return out

# Run inference on 1000 positives with and without arc suppression
```

### Results

| Metric | Original | Arc-Suppressed | Drop |
|--------|----------|----------------|------|
| Mean p_lens | 0.9265 | **0.6217** | 32.9% |
| Recall@0.5 | 93.3% | 62.9% | |
| Recall@0.9 | 88.1% | 41.9% | |

**By Arc SNR:**

| SNR Bin | n | Original p | Suppressed p | Drop |
|---------|---|------------|--------------|------|
| 0-2 | 214 | 0.835 | 0.615 | 0.220 |
| 2-5 | 309 | 0.918 | 0.696 | 0.222 |
| 5-10 | 242 | 0.973 | 0.699 | 0.274 |
| 10-20 | 153 | 0.967 | 0.526 | 0.441 |
| 20+ | 79 | 0.987 | 0.298 | **0.689** |

**Interpretation:** When arcs are removed, the model STILL predicts lens with p=0.62 (above threshold). For faint arcs (SNR<2), the drop is only 0.22 - the model barely uses them. For bright arcs (SNR>20), the drop is 0.69 - the model does use arcs when highly visible. Overall, the model heavily relies on core shortcuts.

---

## Gate 1.8: Core Brightness Matching (ROOT CAUSE)

### Code

```python
#!/usr/bin/env python3
"""Gate 1.8: Core Brightness Matching Check"""

CORE_RADIUS = 8  # pixels

def compute_core_brightness(img, r_core=8):
    C, H, W = img.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    core_mask = ((yy - cy)**2 + (xx - cx)**2) < r_core**2
    r_band = img[1]  # r-band
    core_pixels = r_band[core_mask]
    
    return {
        'core_mean_r': float(np.mean(core_pixels)),
        'core_max_r': float(np.max(core_pixels)),
        'core_median_r': float(np.median(core_pixels)),
        'core_sum_r': float(np.sum(core_pixels))
    }

# Compare 10k controls vs 10k positives
```

### Results

| Metric | Controls | Positives | Ratio |
|--------|----------|-----------|-------|
| core_mean_r | 0.0205 | 0.0336 | **1.64x** |
| core_max_r | 0.102 | 0.125 | 1.23x |
| core_median_r | 0.0133 | 0.0256 | **1.92x** |
| core_sum_r | 3.96 | 6.48 | **1.64x** |

**Percentile Comparison (core_mean_r):**

| Percentile | Controls | Positives | Ratio |
|------------|----------|-----------|-------|
| 5th | 0.0060 | 0.0094 | 1.56x |
| 25th | 0.0106 | 0.0168 | 1.59x |
| 50th | 0.0149 | 0.0240 | 1.61x |
| 75th | 0.0210 | 0.0357 | 1.70x |
| 95th | 0.0403 | 0.0713 | 1.77x |

**ROOT CAUSE IDENTIFIED:** Positives are 64% brighter in the core than controls! This is consistent across ALL percentiles (56-77% brighter). This creates a trivial shortcut - the model just detects brighter cores.

---

## Phase 2 Extended: Center Masking with Controls

### Results

| Class | Original Mean p | Masked Mean p | Change |
|-------|-----------------|---------------|--------|
| Positives | 0.9199 | 0.7867 | -14.5% |
| Controls | 0.0122 | **0.1270** | **+940%** |

**Interpretation:** When the center is masked:
- Positives go down (as expected - removing the bright center)
- **Controls go UP 10x** (critical insight - without the dim center cue, they look more like positives!)

This confirms the model learned: "bright center = lens, dim center = not lens".

---

## Summary of Root Cause

1. **Positives and controls have different LRG populations:**
   - Positives appear to be selected from brighter LRGs
   - OR the injection process adds significant flux to the core

2. **The model learned a trivial shortcut:**
   - "If central r<10px is bright → predict lens"
   - This achieves 98% AUC without ever looking at arcs

3. **The shortcut fails on real data:**
   - Real SLACS/BELLS lenses don't follow this brightness pattern
   - The model correctly detects brighter synthetic lenses but fails on real ones

---

## Questions for LLM Review

### Q1: Why would positives have 64% brighter cores?

Our injection pipeline injects a lensed COSMOS source onto an LRG cutout. The lensed source adds flux. But shouldn't the COSMOS source flux be concentrated in the arc region (10-25 px), not the core (<10 px)?

Possible explanations:
1. The lensed source has some flux in the core (for small theta_e, the Einstein ring overlaps the center)
2. The positives and controls are sampled from different LRG populations
3. There's a selection bias (only keeping "successful" injections that are brighter)

**Which do you think is most likely?**

### Q2: Is this a paired/unpaired control issue?

Our pipeline uses "unpaired controls" - random LRGs without injection. If the LRGs used for injection are systematically different from the control LRGs, this would explain the brightness difference.

**Should we switch to paired controls** (same LRG cutout, with and without injection)?

### Q3: Center-masked training - will it work?

Given that the core shortcut is so strong (98% AUC), would center-masked training (masking r<10px during training) be sufficient to force arc learning? Or is the dataset fundamentally broken?

### Q4: Should we regenerate the entire dataset?

The core brightness mismatch is severe. Options:
1. **Center-masked training** - Mask r<10px during training to force arc learning
2. **Paired controls** - Use same LRG for positive and control
3. **Brightness matching** - Post-hoc filter to match core brightness distributions
4. **Full regeneration** - Fix the data pipeline and regenerate 10M+ samples

**What is the most efficient path forward?**

### Q5: Arc-suppressed results show arc usage at high SNR

For arc_snr > 20, the model's prediction drops by 0.69 when arcs are removed. This suggests the model DOES use arcs when highly visible.

**Does this mean the model is partially correct**, and we just need to reduce reliance on core shortcuts?

### Q6: Priority of fixes

Given limited time and compute, what should be the priority order?
1. Fix the core brightness mismatch (data-level fix)
2. Center-masked training (training-level fix)
3. Expand Tier-A anchors (evaluation-level fix)
4. Something else?

---

## Full Code Repository

All scripts are available in `dark_halo_scope/scripts/`:
- `gate_1_5_normalization_stats.py`
- `gate_1_6_core_only_baseline.py`
- `gate_1_7_arc_suppressed.py`
- `gate_1_8_core_brightness_matching.py`
- `phase2_center_masked_extended.py`

All results are saved as JSON in `/lambda/nfs/darkhaloscope-training-dc/`:
- `gate_1_5_results.json`
- `gate_1_6_results.json`
- `gate_1_7_results.json`
- `gate_1_8_results.json`
- `phase2_extended_results.json`

---

## Decision Gate

Based on these findings, we **CANNOT proceed with Gen5 training** until the core brightness mismatch is resolved. The current model is fundamentally learning the wrong features.

**Recommended next steps:**
1. Investigate why positives have brighter cores
2. Implement one of the fixes (paired controls, brightness matching, or center-masked training)
3. Re-run all gates to verify the fix worked
4. Only then proceed with retraining

We await your guidance on the most efficient path forward.

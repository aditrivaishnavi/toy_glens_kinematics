# Comprehensive LLM Review: Core Brightness Shortcut Discovery

**Project:** Dark Halo Scope - Strong Gravitational Lens Finder for DESI DR10
**Target Venue:** MNRAS / ApJ / AAS quality publication
**Date:** 2026-02-05
**Status:** CRITICAL - Training pipeline fundamentally flawed

---

## Executive Summary

We are building a machine learning lens finder for DESI Legacy DR10 ground-based imaging. After following your previous recommendations to add shortcut-detection gates, we discovered **catastrophic failures** that invalidate our entire Gen5 training run.

**Key Finding:** A simple logistic regression trained on ONLY the central 10 pixels of our images achieves **98% AUC**. This proves our model learned to detect "bright centers" instead of "arc morphology".

**Root Cause:** Positives (lensed images) have 64% brighter cores than controls (non-lensed images).

We need your expertise to:
1. Understand how SOTA papers handled similar issues
2. Determine the most efficient fix
3. Get concrete code to implement the solution
4. Ensure our methodology is publication-ready

---

## Problem Statement

### What We Built

1. **Training Data Pipeline:**
   - Take ~145,000 Luminous Red Galaxies (LRGs) from DESI DR10
   - For positives: Inject lensed COSMOS source galaxies using lenstronomy
   - For controls: Use random LRGs without injection (unpaired)
   - Total: ~10 million samples (50% positive, 50% control)

2. **Model:**
   - ConvNeXt-Tiny backbone with metadata fusion head
   - Input: 64x64 pixel grz cutouts + PSF size + depth metadata
   - Normalization: Outer-annulus robust MAD normalization

3. **Results:**
   - Synthetic test AUC: 0.9945 (excellent!)
   - Real SLACS/BELLS anchor recall: 4.4% (terrible!)

### What Went Wrong

Following your recommendation, we added shortcut-detection gates:

| Gate | What It Tests | Result | Key Metric |
|------|---------------|--------|------------|
| 1.5 | Normalization stats differ? | **FAIL** | g-band clip frac: d=0.658 |
| 1.6 | Core-only classifier works? | **FAIL** | AUC=0.98 (should be ~0.5) |
| 1.7 | Model uses arcs? | **FAIL** | p=0.62 with arcs removed |
| 1.8 | Core brightness matched? | **FAIL** | Positives 64% brighter |

### The Root Cause

**Core brightness distributions differ dramatically between classes:**

| Metric | Controls | Positives | Ratio |
|--------|----------|-----------|-------|
| core_mean_r | 0.0205 | 0.0336 | **1.64x** |
| core_median_r | 0.0133 | 0.0256 | **1.92x** |
| core_max_r | 0.102 | 0.125 | 1.23x |

The model learned: **"bright center = lens"** instead of **"arc morphology = lens"**.

---

## Detailed Gate Results

### Gate 1.5: Normalization-Stat Leakage

We checked if the normalization statistics (computed from outer annulus) differ between classes.

```
| Stat           | Ctrl Mean | Pos Mean  | Cohen's d | Status |
|----------------|-----------|-----------|-----------|--------|
| clip_frac_r    | 0.0158    | 0.0253    | 0.343     | FAIL   |
| outer_median_g | 0.000338  | 0.000449  | 0.158     | FAIL   |
| clip_frac_g    | 0.0088    | 0.0305    | 0.658     | FAIL   |
```

**Interpretation:** Positives have 3.5x higher g-band clipping fraction. The injected arcs push more pixels past the normalization clip threshold.

### Gate 1.6: Core-Only Baseline (CRITICAL)

We trained a logistic regression using ONLY features from the central 10-pixel radius.

**Features extracted (per band):**
- mean, std, median, max, min, q25, q75

**Results:**
- Train AUC: 0.9909
- **Val AUC: 0.9800** (should be ~0.50 if no shortcut!)

**Top Features:**
1. g_q25 (importance: 17.19)
2. g_median (importance: 10.83)
3. r_median (importance: 7.86)

**Interpretation:** A trivial linear classifier achieves 98% AUC using only the center. The model doesn't need to see arcs at all.

### Gate 1.7: Arc-Suppressed Positive Test

We zeroed out the arc region (10-25 pixel radius) in positive samples and ran inference.

```
| Metric         | Original | Arc-Suppressed | Drop   |
|----------------|----------|----------------|--------|
| Mean p_lens    | 0.9265   | 0.6217         | 32.9%  |
| Recall@0.5     | 93.3%    | 62.9%          |        |
```

**By Arc SNR:**
```
| SNR Bin | n   | Original | Suppressed | Drop  |
|---------|-----|----------|------------|-------|
| 0-2     | 214 | 0.835    | 0.615      | 0.220 |
| 2-5     | 309 | 0.918    | 0.696      | 0.222 |
| 5-10    | 242 | 0.973    | 0.699      | 0.274 |
| 10-20   | 153 | 0.967    | 0.526      | 0.441 |
| 20+     | 79  | 0.987    | 0.298      | 0.689 |
```

**Interpretation:** When arcs are removed, the model STILL predicts lens with p=0.62. For faint arcs (SNR<2), the model barely uses them (drop=0.22). For bright arcs (SNR>20), the model does use them (drop=0.69).

### Gate 1.8: Core Brightness Matching (ROOT CAUSE)

We compared central aperture (r<8 pixels) brightness between classes.

```
| Percentile | Controls | Positives | Ratio |
|------------|----------|-----------|-------|
| 5th        | 0.0060   | 0.0094    | 1.56x |
| 25th       | 0.0106   | 0.0168    | 1.59x |
| 50th       | 0.0149   | 0.0240    | 1.61x |
| 75th       | 0.0210   | 0.0357    | 1.70x |
| 95th       | 0.0403   | 0.0713    | 1.77x |
```

**ROOT CAUSE:** Positives are consistently 56-77% brighter in the core across ALL percentiles.

### Phase 2 Extended: Center Masking with Controls

We masked the center (r<10px) and compared effects on both classes.

```
| Class     | Original Mean p | Masked Mean p | Change   |
|-----------|-----------------|---------------|----------|
| Positives | 0.9199          | 0.7867        | -14.5%   |
| Controls  | 0.0122          | 0.1270        | +940%    |
```

**Critical Insight:** When center is masked, controls go UP 10x! The model was using dim centers to identify controls as "not lens". This confirms the shortcut: "bright center = lens, dim center = not lens".

---

## Why Does This Happen?

### Hypothesis 1: Unpaired Controls

Our pipeline uses "unpaired controls" - random LRGs without injection. If the LRGs used for injection are systematically different from control LRGs, this would explain the brightness difference.

### Hypothesis 2: Injection Adds Core Flux

The lensed COSMOS source may contribute flux to the central region:
- For small theta_e (~0.3"), the Einstein ring overlaps the center
- The COSMOS source itself may have central flux before lensing

### Hypothesis 3: Selection Bias

We filter for "successful" injections (physics_valid=True, arc_snr>threshold). This may preferentially select brighter LRGs.

---

## Questions for LLM Review

### Q1: How do SOTA papers handle the paired/unpaired control problem?

**Context:** Papers like Lanusse et al. (2018), Jacobs et al. (2019), Huang et al. (2020), and more recently Rojas et al. (2022) and O'Riordan et al. (2023) have trained CNNs to find strong lenses.

**Specific Questions:**
- Do these papers use paired controls (same galaxy with and without injection)?
- Or unpaired controls (random non-lens galaxies)?
- How do they ensure the model doesn't learn shortcuts from brightness differences?
- **Please cite specific methodological choices from these papers.**

### Q2: How do SOTA papers validate that models learn morphology, not shortcuts?

**Context:** We were confident in our model until we ran shortcut gates. Other papers report high precision/recall - did they run similar validation?

**Specific Questions:**
- Do any papers perform "core-only baseline" tests or equivalent?
- Do any papers report arc-suppressed or center-masked ablations?
- What sanity checks do they recommend for injection-based training?
- **Please cite any papers that discuss shortcut learning in lens finding.**

### Q3: What is the correct way to generate matched controls?

**Our Current Approach:**
```python
# Unpaired controls: random LRGs
control_df = lrg_catalog.sample(n=n_controls)
```

**Options We're Considering:**
1. **Paired controls:** Same LRG cutout, with and without injection
2. **Brightness-matched controls:** Post-hoc filter to match distributions
3. **Core-subtracted injection:** Remove lens light before adding arc

**Specific Questions:**
- Which approach is most scientifically rigorous?
- Which approach is most computationally efficient?
- Are there papers that explicitly compare these approaches?
- **Please provide concrete code for the recommended approach.**

### Q4: Should we use center-masked training?

**Context:** Our Phase 2 Extended test shows that masking the center:
- Reduces positive predictions by 14.5%
- Increases control predictions by 940%

**Options:**
1. **Masked training:** Mask r<10px with probability p during training
2. **Two-view learning:** Process both masked and unmasked views
3. **Gradient penalty:** Penalize gradients in the central region

**Specific Questions:**
- Will masked training force the model to learn arcs?
- Or will it just learn a different shortcut (e.g., normalization stats)?
- What mask radius and probability are optimal?
- **Please provide concrete training code with recommended hyperparameters.**

### Q5: Can we salvage the existing 10M sample dataset?

**Context:** Regenerating 10M samples is expensive (~$500 EMR cost, ~12 hours).

**Options:**
1. **Post-hoc filtering:** Remove samples where core brightness differs
2. **Reweighting:** Weight samples inversely to core brightness imbalance
3. **Brightness augmentation:** Randomly scale core brightness during training
4. **Center masking:** Mask center during training (don't regenerate)

**Specific Questions:**
- Which approach would salvage the dataset without regeneration?
- Is any of these scientifically defensible for a publication?
- Or must we regenerate with paired controls?
- **Please provide concrete code for the most efficient salvage approach.**

### Q6: What would make this paper publishable in MNRAS/ApJ?

**Context:** Our goal is a selection-function paper: "What fraction of strong lenses are detectable in DR10 as a function of Einstein radius, lens redshift, and source properties?"

**Current Contributions:**
1. Novel COSMOS-based injection pipeline (realistic source morphologies)
2. Quantitative selection function C(theta_E, z_l)
3. Recovery of known lenses at predicted rates

**Problem:** If our model learned shortcuts, the selection function is wrong.

**Specific Questions:**
- What validation is required for a credible selection-function paper?
- How should we present the shortcut discovery as a methodological contribution?
- What comparisons to previous selection-function papers (e.g., Collett 2015) are needed?
- **Please outline the minimum viable paper structure with these findings.**

### Q7: Concrete Fix - Step by Step

**Request:** Please provide a complete, actionable remediation plan with:

1. **Data-level fix:** How to generate matched positives and controls
2. **Training-level fix:** How to prevent shortcut learning during training
3. **Validation-level fix:** What gates must pass before publication
4. **Code:** Working Python code for each step

**Specifically, we need:**

```python
# 1. Function to generate paired controls
def generate_paired_sample(lrg_cutout, cosmos_source, lens_params):
    """
    Returns:
        positive: lrg_cutout + injected lensed source
        control: lrg_cutout (same cutout, no injection)
    """
    # YOUR CODE HERE

# 2. Training augmentation to prevent shortcuts
def shortcut_resistant_augmentation(batch_images, batch_labels):
    """
    Apply augmentations that prevent brightness-based shortcuts.
    """
    # YOUR CODE HERE

# 3. Validation function to verify no shortcuts
def validate_no_shortcuts(model, dataset):
    """
    Run all shortcut gates and return pass/fail with metrics.
    """
    # YOUR CODE HERE
```

### Q8: Are there alternative injection strategies?

**Current Approach:**
1. Start with LRG cutout (lens galaxy + background)
2. Inject lensed COSMOS source additively
3. Result: lens_galaxy + lensed_arc + background

**Problem:** This adds flux to the core region (especially for small theta_e).

**Possible Alternatives:**
1. **Lens-light subtraction first:** Subtract lens galaxy, add arc, add lens galaxy back
2. **Annulus-only injection:** Only inject into r > 10px (but this is unphysical)
3. **Core normalization:** Normalize core brightness post-injection

**Specific Questions:**
- How do SOTA papers handle the fact that arcs add flux to images?
- Is there a physically correct way to inject that preserves core brightness?
- **Please describe the injection methodology of papers with successful real-lens recovery.**

### Q9: Quantify the expected impact

**Current State:**
- Core-only AUC: 0.98 (model uses shortcuts)
- Arc-suppressed p_lens: 0.62 (model barely uses arcs)

**After Fix:**
- Core-only AUC should be: < 0.55 (near random)
- Arc-suppressed p_lens should be: < 0.30 (significant drop)

**Questions:**
- What core-only AUC is acceptable? Is 0.55 the right threshold?
- What arc-suppressed p_lens is acceptable?
- How do we know when the fix is "good enough"?
- **Please provide quantitative pass/fail criteria for each gate.**

### Q10: Timeline and Priority

**Constraints:**
- GPU compute: 8x H100 available for 1 week
- EMR budget: ~$1000 remaining
- Timeline: Need results in 2 weeks

**Options (not mutually exclusive):**
1. Regenerate data with paired controls (~$500, 12 hours)
2. Retrain with center masking (~$0, 6 hours training)
3. Post-hoc brightness matching (~$0, 2 hours)
4. Expand Tier-A anchors (~$100, 4 hours)

**Question:**
- **What is the optimal sequence of actions given our constraints?**
- Which can be done in parallel?
- What is the minimum viable fix for a credible paper?

---

## Attached Files

This package contains:

### Scripts (Python)
1. `gate_1_5_normalization_stats.py` - Normalization leakage check
2. `gate_1_6_core_only_baseline.py` - Core-only classifier test
3. `gate_1_7_arc_suppressed.py` - Arc suppression test
4. `gate_1_8_core_brightness_matching.py` - Core brightness comparison
5. `phase2_center_masked_extended.py` - Center masking with controls

### Results (JSON)
1. `gate_1_5_results.json`
2. `gate_1_6_results.json`
3. `gate_1_7_results.json`
4. `gate_1_8_results.json`
5. `phase2_extended_results.json`

---

## Summary

We have discovered that our strong lens finder learned to detect "bright centers" instead of "arc morphology". This invalidates our current training run and selection function analysis.

We need your expert guidance to:
1. Understand how this problem is typically handled in the literature
2. Determine the most efficient fix within our constraints
3. Get concrete, working code to implement the solution
4. Ensure our methodology meets publication standards

**Please provide specific, actionable recommendations with code examples where possible.**

Thank you for your thorough review.

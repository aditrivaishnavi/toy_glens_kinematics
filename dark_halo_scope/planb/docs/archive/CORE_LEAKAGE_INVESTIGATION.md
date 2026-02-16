# Core Leakage Investigation - Structured Analysis

**Date:** 2026-02-06
**Status:** ROOT CAUSE IDENTIFIED
**Goal:** Understand the root cause with high confidence before proposing any fix

---

## ROOT CAUSE IDENTIFIED: Inner (Counter) Image Physics

### The Finding

The "core leakage" is **NOT**:
- ❌ A calibration bug
- ❌ A PSF convolution error
- ❌ An artifact of extended source profiles

The "core leakage" **IS**:
- ✅ **The physical inner (counter) image of the lensed source**

### Strong Lensing Basics: Two Images from Offset Sources

For an SIS/SIE lens with Einstein radius θ_E and source offset β from the optical axis:

1. **Outer image** at θ = β + θ_E (beyond Einstein ring, magnified)
2. **Inner image** at θ = θ_E - β (inside Einstein ring, demagnified)

When β < θ_E (source inside the caustic), BOTH images form.

### Our Training Data Statistics

| Metric | Value |
|--------|-------|
| Median source offset | 0.55 × θ_E |
| Median inner image position | 2.1 pixels from center |
| Inner images in core (r < 5 pix) | **91.8%** |
| Inner images in r < 3 pix | **70.7%** |

### By Einstein Radius

| θ_E Range | Median Inner Image | % in Core |
|-----------|-------------------|-----------|
| 0.50-0.75" | 0.9 pix | 100% |
| 0.75-1.00" | 1.4 pix | 100% |
| 1.00-1.50" | 1.9 pix | 100% |
| 1.50-2.00" | 2.4 pix | 98.2% |
| 2.00-2.50" | 3.4 pix | 84.5% |

### Core Flux Fraction Measurements

| θ_E Range | Core Flux Fraction | Inner Image Position |
|-----------|-------------------|---------------------|
| 0.50-0.75" | 39.5% | ~1 pixel from center |
| 0.75-1.00" | 33.1% | ~1.4 pixels |
| 1.00-1.50" | 23.6% | ~1.9 pixels |
| 1.50-2.00" | 15.4% | ~2.4 pixels |
| 2.00-2.50" | 10.0% | ~3.4 pixels |

**The core flux fraction tracks the inner image position exactly.**

---

## Why This Matters

### This is Correct Physics

The inner image is a **real feature** of strong gravitational lensing:
- It's predicted by the lens equation
- It's observed in real lens systems (though often faint)
- Our simulation is correctly reproducing it

### But It Creates a Shortcut

A classifier can learn to detect the **added central flux** without understanding:
- Arc morphology
- Ring-like structure
- Tangential stretch patterns

This is problematic because:
1. The inner image is often too faint to detect in real observations
2. It overlaps with the bright lens galaxy (LRG) center
3. Real detection should focus on the visible outer arc

---

## Prior Analysis Summary (for reference)

| Finding | Evidence | Implication |
|---------|----------|-------------|
| Core leakage is NOT calibration mismatch | Outer annulus offset ≈ 0 | Not a systematic background error |
| Core offset scales with arc properties | R²=0.52 with arc flux | Consistent with physics |
| Core offset decreases with larger θ_E | Negative slope | Consistent with physics |
| Lensed image is diffuse, not ring-like | Radial profile is flat | Inner image + PSF + outer arc |
| Leakage exceeds simple PSF expectation | 7.7% vs 0.4% expected | Inner image, not just PSF |

**Prior conclusion was partially correct** - it IS physics, but the specific mechanism is the inner image, not just "extended source wings."

---

## Critical Question: Is This a Problem?

### Arguments FOR "This is Fine"

1. **Physics is correct**: Inner images exist in real lensed systems
2. **Detection is valid**: If the model learns to detect inner images, that's legitimate
3. **No domain shift in principle**: Real lenses have the same physics

### Arguments FOR "This is a Problem"

1. **Observational reality**: Inner images are often invisible in real data because:
   - They're highly demagnified
   - They're overwhelmed by lens galaxy (LRG) light
   - Noise floor hides them
   
2. **Detection shortcut**: The model learns "core brightness" not "arc morphology"
   - May not generalize to lenses with visible arcs but hidden inner images
   - May trigger false positives from any central brightening

3. **Training vs reality mismatch**: 
   - Training: clean synthetic inner image added to LRG
   - Reality: inner image buried in LRG light and noise

### What We Need to Determine

**Question 1:** In real observed strong lenses, how often is the inner image actually visible above the noise and lens galaxy background?

**Question 2:** Are published lens simulators (CMU, Bologna, etc.) also including inner images?

**Question 3:** Does masking the core during training hurt performance on real lenses that DO have visible inner images?

---

## What We Still Don't Know

### 1. Is the Source Profile Realistic?

The injection uses Sersic n=1 (exponential) with source effective radius (reff) typically 0.2-0.5 arcsec.

**Questions:**
- What is the actual distribution of reff in the training data?
- How does lensed morphology depend on reff?
- Are real lensed sources this extended?
- What do published lens simulations use?

### 2. Why is Leakage 7.7% vs Expected 0.4%?

For θ_E ≥ 2" (arc at ~8.6 pix), simple PSF spreading predicts ~0.4% leakage. We observe 7.7%.

**Possible explanations:**
- Extended source wings contributing to core
- Lens magnification near caustic mapping source center to image center
- Source position offset (±0.4" from lens center) placing some source light near center
- Something else we haven't considered

**What we need:** A controlled experiment varying source parameters to understand the contribution of each factor.

### 3. What Do Published Papers Do?

We should consult the literature on:
- How lens simulation papers handle this issue
- What source profiles they use
- Whether they mask the core or use other mitigations
- Best practices for avoiding training shortcuts

### 4. Is the PSF Convolution Correct?

Test 1 showed the PSF peak was at (9,9) instead of (32,32) for impulse at (32,32). An alternative implementation gave correct results. 

**Status:** It's unclear if this was fixed in production or if the buggy code generated our training data.

---

## Structured Investigation Plan

### Phase 1: Understand the Current Data (TODAY)

#### 1.1 Distribution Analysis

Compute and document the distribution of key injection parameters:
```python
# What we need to extract from training data:
- theta_e_arcsec: distribution, min, max, median
- src_reff_arcsec: distribution (THIS IS CRITICAL)
- src_x_arcsec, src_y_arcsec: source position offsets
- psf_fwhm_pix: PSF size distribution
- arc_snr: signal-to-noise distribution
```

**Goal:** Understand what parameter space we're training on.

#### 1.2 Controlled Experiments

Run controlled experiments varying one parameter at a time:

**Experiment A: Source reff impact**
```python
# Fix: theta_e=2", PSF FWHM=5 pix, source at center
# Vary: src_reff from 0.1" to 1.0" in steps of 0.1"
# Measure: core flux fraction for each reff
```

**Experiment B: Source position impact**
```python
# Fix: theta_e=2", PSF FWHM=5 pix, reff=0.3"
# Vary: source offset from -0.5" to +0.5" from lens center
# Measure: core flux fraction for each offset
```

**Experiment C: θ_E impact (verification)**
```python
# Fix: PSF FWHM=5 pix, reff=0.3", source at center
# Vary: theta_e from 0.5" to 3.0"
# Measure: core flux fraction
```

### Phase 2: Verify PSF Convolution (TODAY)

#### 2.1 Check Production Code

Verify whether the PSF bug identified in Test 1 was fixed before generating the training data.

```python
# In spark_phase4_pipeline_gen5.py, check _fft_convolve2d()
# Does it use ifftshift correctly?
# Compare to scipy.signal.fftconvolve as ground truth
```

#### 2.2 Re-render a Sample

Take a sample from training data, re-render the arc with known-correct PSF convolution, and compare:
- Does the core flux match what's in the training data?
- If not, how much of the discrepancy is from PSF bug?

### Phase 3: Literature Review (TOMORROW)

#### 3.1 Published Lens Simulation Papers

Review how these papers handle lens injection:
- Lanusse et al. (CMU DeepLens) - What source profiles do they use?
- Metcalf et al. (Bologna CNN lens finder) - Any discussion of shortcuts?
- Jacobs et al. (Galaxy Zoo lens search) - Training data methodology?
- Petrillo et al. (KiDS lens search) - Simulation approach?

#### 3.2 Standard Practice Questions

- What source profiles are commonly used?
- Do researchers mask the core region?
- Is "paired training" (positive/negative from same galaxy) standard?
- Any published discussions of shortcut risks?

### Phase 4: Synthesize Findings (TOMORROW)

Once we have:
1. Parameter distributions from our data
2. Controlled experiment results
3. PSF verification
4. Literature context

We can:
- Identify the **specific cause** with high confidence
- Propose a **targeted fix** (not a band-aid)
- Validate the fix is **scientifically correct**

---

## Key Hypotheses to Test

### H1: Extended Source Morphology

**Hypothesis:** Sersic n=1 with reff=0.3" creates lensed images that are too diffuse, putting significant flux in the core.

**Test:** Controlled experiment varying reff. If core flux drops dramatically with smaller reff, this is the cause.

**Implication:** Use more compact sources (smaller reff or higher Sersic n).

### H2: Source Position Near Caustic

**Hypothesis:** When source is placed near caustic (within caustic), lensed image includes a central "demagnified" component.

**Test:** Controlled experiment varying source position. Check if on-caustic sources produce more core flux.

**Implication:** Avoid placing sources too close to caustic.

### H3: PSF Convolution Bug

**Hypothesis:** The PSF bug identified in Test 1 affects production data, artificially spreading flux to unexpected locations.

**Test:** Re-render samples with correct PSF, compare to training data.

**Implication:** Fix PSF code and regenerate data.

### H4: This is Correct Physics We Must Design Around

**Hypothesis:** Real lensed galaxies DO have diffuse morphology that extends to the core. The injection is physically correct.

**Test:** Compare to high-resolution HST lens images - do real lenses show core flux?

**Implication:** The shortcut is unavoidable; must use training techniques (core dropout, hard negatives) to mitigate.

---

## Immediate Actions

### Action 1: Extract Parameter Distributions

```bash
# On Lambda, analyze the training data
python -c "
import pandas as pd
import numpy as np
from pathlib import Path

# Load sample of training data
files = list(Path('/home/ubuntu/data/v5_cosmos_paired/train').glob('*.parquet'))[:10]
df = pd.concat([pd.read_parquet(f) for f in files])

# Key parameters
print('theta_e_arcsec:', df['theta_e_arcsec'].describe())
print('src_reff_arcsec:', df['src_reff_arcsec'].describe() if 'src_reff_arcsec' in df.columns else 'NOT IN DATA')
print('arc_snr:', df['arc_snr'].describe() if 'arc_snr' in df.columns else 'NOT IN DATA')
"
```

### Action 2: Check PSF Convolution Code

Review `_fft_convolve2d()` in `spark_phase4_pipeline_gen5.py` and compare to correct implementation.

### Action 3: Design Controlled Experiment Script

Create a script that:
1. Renders arcs with controlled parameters
2. Measures core flux fraction
3. Outputs results for analysis

---

## Success Criteria

Before proposing any fix, we must:

1. **Identify the dominant cause** with >90% confidence
2. **Understand how published papers handle this** (or confirm we're doing something novel)
3. **Have a quantitative model** of how parameters affect core flux
4. **Validate any proposed fix** on synthetic data before regenerating

---

## What We Will NOT Do

1. ❌ Implement ad-hoc fixes without understanding the cause
2. ❌ Assume the prior analysis is complete (verify it)
3. ❌ Rush to regenerate data without high confidence in the fix
4. ❌ Ignore the literature on this problem

---

## Open Questions for Discussion

1. Do we have access to the exact code version that generated the training data? (To verify PSF bug status)

2. Is there a parameter file or config that documents the injection parameters used?

3. Can we access HST images of known lenses to compare morphology?

4. Should we pause training until this is resolved, or continue to document the failure mode?

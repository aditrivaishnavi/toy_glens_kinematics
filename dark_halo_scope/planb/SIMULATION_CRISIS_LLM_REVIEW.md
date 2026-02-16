# Critical Finding: Simulation Realism Crisis

**Date:** 2026-02-07  
**Status:** URGENT - Fundamental training data problem discovered  
**Request:** Strategic guidance on path forward

---

## Executive Summary

We discovered that our simulated training data for strong gravitational lens detection is **fundamentally unrealistic**. Simulated arcs are ~100x brighter than real arcs in DESI Legacy Survey DR10. This likely explains why models trained on simulations may fail on real data.

Meanwhile, we found that the state-of-the-art lens finder (Huang et al., arXiv:2508.20087) trains on **real confirmed lenses**, not simulations, and has successfully discovered 3,868 new lens candidates across 4 papers.

---

## Part 1: Our Simulation Problem

### Quantitative Evidence

We measured arc properties in real confirmed lenses vs. our simulations:

| Metric | Real DR10 Lenses | Our Simulations | Gap |
|--------|------------------|-----------------|-----|
| Arc SNR (signal-to-noise) | **~0** (at noise level) | **30-80** | **~100x too bright** |
| Core/Arc brightness ratio | 0.3-10 | 6-16 | Variable |
| Visual appearance | Barely visible, subtle | Very prominent, obvious | Completely different |

### What This Means

1. **Model learns wrong features**: If training arcs are 100x brighter, the model learns to detect features that don't exist in real data
2. **No sim-to-real transfer**: Model will fail on real lenses where arcs are subtle
3. **High training AUC is meaningless**: The model is solving a much easier problem than the real task

### Root Cause

Our arc injection uses arbitrary flux scaling:
```python
arc_flux_total = random(100, 500)  # Arbitrary ADU - not calibrated to reality
```

Real lensed sources:
- Source galaxies: r ~ 22-24 mag (very faint)
- After lensing magnification (μ ~ 10-30): r ~ 20-22 mag
- In DR10 with ~1" seeing: arcs are blurred into galaxy light, often invisible to human eye

---

## Part 2: The Huang et al. Approach (What Works)

### Paper Series

The Huang et al. group has published 4 papers discovering lenses in DESI Legacy Surveys:

| Paper | Year | New Candidates | Cumulative |
|-------|------|----------------|------------|
| Paper I (Huang et al. 2020) | 2020 | 335 | 335 |
| Paper II (Huang et al. 2021) | 2021 | 1,210 | 1,545 |
| Paper III (Storfer et al. 2024) | 2024 | 1,512 | 3,057 |
| Paper IV (Inchausti et al. 2025) | 2025 | 811 | 3,868 |

**Reference:** arXiv:2508.20087 (Paper IV)

### Their Key Insight: Train on Real Data

From Paper IV, Section 3.1:
> "As in Papers I, II and III, we continue to use **observed images** of both lenses and nonlenses for training."

Their training set:
- **Positives:** 1,372 real confirmed lenses from literature (SLACS, BELLS, SL2S, etc.)
- **Negatives:** 134,182 real non-lens galaxies from DR10
- **Ratio:** ~100:1 negative to positive
- **Cutout size:** 101 × 101 pixels (~26" × 26")

### Their Architecture

- ResNet (194,433 parameters) + EfficientNet (20.5M parameters)
- Ensemble with meta-learner
- Validation AUC: ~0.999

### Why It Works

1. Training and test distributions are identical (both real DR10 images)
2. No sim-to-real gap to bridge
3. Model learns actual features that distinguish real lenses

---

## Part 3: Available Resources

### Confirmed Lenses We Can Access

From `lenscat` (community catalog):
- **Total lenses:** 32,838 from all surveys
- **DESI-prefixed (Huang et al. discoveries):** 5,104 lens candidates
- All have coordinates; we can download DR10 cutouts

From literature (spectroscopically confirmed):
- SLACS: ~50 lenses
- BELLS: ~20 lenses  
- SL2S: ~10 lenses
- Others: ~20 lenses

**Total available:** ~5,200 lens images (candidates + confirmed)

### What We've Already Downloaded

- 100 confirmed lenses (SLACS, BELLS, SL2S, etc.) with FITS + JPEG
- 100 DESI candidate lenses (test batch)
- All from DR10, 101×101 pixels, g/r/z bands

---

## Part 4: Options We're Considering

### Option A: Train on Real Lenses (Like Huang et al.)

**Approach:**
- Use ~5,000 DESI lens candidates as positives
- Sample ~500,000 random galaxies as negatives
- Train ResNet/EfficientNet

**Pros:**
- Proven to work (Huang et al. achieved 0.999 AUC)
- No sim-to-real gap
- Can reproduce their results

**Cons:**
- We're not discovering anything new
- Limited to what's already known
- No control over θ_E distribution

### Option B: Heavy Augmentation of Real Lenses

**Approach:**
- Take ~5,000 real lenses
- Apply aggressive augmentation:
  - Rotations (every 15°)
  - Flips (horizontal, vertical)
  - Random crops/shifts
  - Noise injection
  - Color jitter
  - Cutout/dropout
  - MixUp with other lenses
- Generate 100K+ effective training samples

**Pros:**
- More data variety
- Can still use real lens features
- Might improve generalization

**Cons:**
- Augmented lenses may not represent real variety
- Risk of overfitting to augmentation artifacts

### Option C: Fix Simulation Realism

**Approach:**
- Calibrate arc flux using survey zeropoints
- Match source magnitude distribution (r ~ 22-24)
- Use realistic PSF from DR10
- Add proper noise from real sky regions
- Validate: simulated arc SNR should be ~0-5, not 30-80

**Pros:**
- Can generate unlimited training data
- Control over θ_E, source properties, etc.
- Scientific understanding of what we're doing

**Cons:**
- Complex to get right
- May still miss domain-specific features
- Need to validate carefully

### Option D: Domain Adaptation / Transfer Learning

**Approach:**
- Pre-train on simulations (as we did)
- Fine-tune on real lenses
- Or use domain adversarial training

**Pros:**
- Leverages both simulated and real data
- May bridge the gap

**Cons:**
- Unclear if simulation features are useful at all
- Added complexity

### Option E: Semi-Supervised Learning

**Approach:**
- Train initial model on real lenses
- Apply to DR10, get high-confidence predictions
- Use predictions as pseudo-labels
- Retrain with expanded dataset

**Pros:**
- Self-improving loop
- Can discover new lenses

**Cons:**
- Error propagation risk
- Need careful confidence calibration

---

## Part 5: Direct Questions for LLM Review

### Q1: Is our simulation approach salvageable?

Given that real arcs have SNR ~ 0 (essentially invisible at survey depth), is it even possible to create realistic simulations? Or should we abandon simulations entirely?

### Q2: How should we use the ~5,000 DESI lens candidates?

Huang et al. trained on 1,372 lenses. We have access to ~5,000. What's the best way to leverage this larger dataset?

Specific sub-questions:
- Should we filter by grading (confident vs. probable)?
- How important is the 100:1 negative ratio they used?
- What augmentation strategy would maximize effective training data?

### Q3: Can augmentation compensate for limited real data?

If we have 5,000 real lens images, what augmentation strategy would best expand this to 100K+ effective samples while preserving realistic features?

### Q4: What's the minimum viable approach to get a working lens finder?

Given our resources (5,000 lens candidates, DR10 access, compute), what's the fastest path to a model that actually works on real data?

### Q5: Should we pursue a hybrid approach?

Is there value in combining:
- Real lenses for positive examples
- Calibrated simulations for exploring parameter space (e.g., rare high-θ_E cases)

### Q6: What validation should we use?

How do we know if our model will actually work before running on the full survey? What holdout strategy makes sense?

---

## Part 6: Our Current State

### What We Have

1. **5 training runs** (A1, B1, B2, B3, B4) on simulated data - likely invalid due to realism issue
2. **100 confirmed lenses** downloaded with metadata
3. **100 DESI candidates** downloaded (test batch)
4. **Scripts** to download all 5,104 DESI candidates
5. **Contaminant catalog** (176 non-lenses for FPR testing)

### Compute Available

- 5 Lambda Cloud GPUs running (but on simulation-trained models)
- AWS EMR available for data generation

### Timeline Pressure

- Current training runs will complete in ~24 hours
- But if simulation data is fundamentally flawed, those runs may be worthless

---

## Requested Output

Please provide:

1. **Assessment**: Is our simulation approach fundamentally flawed, or can it be salvaged?

2. **Recommendation**: Which option (A-E) should we pursue, or a combination?

3. **Concrete Next Steps**: What should we do in the next 24 hours?

4. **Augmentation Strategy**: If using real lenses, what specific augmentations?

5. **Architecture Advice**: Should we stick with ResNet18 or use their ResNet+EfficientNet ensemble?

6. **Sample Size Analysis**: Is 5,000 lenses enough? What's the minimum viable?

7. **Risk Assessment**: What could go wrong with each approach?

---

## References

- Huang et al. 2020: Paper I (335 candidates)
- Huang et al. 2021: Paper II (1,210 candidates)  
- Storfer et al. 2024: Paper III (1,512 candidates)
- Inchausti et al. 2025: Paper IV (811 candidates) - arXiv:2508.20087
- lenscat: https://github.com/lenscat/lenscat (community lens catalog)
- Bolton et al. 2008: SLACS Survey
- Brownstein et al. 2012: BELLS Survey

---

*This document prepared for external LLM review to get strategic guidance on training data approach.*


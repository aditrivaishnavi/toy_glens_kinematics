# External Review Request: Unpaired Training Design

**Date:** 2026-02-06
**Context:** Strong gravitational lens detection
**Request:** Critical review of proposed data design change

---

## Background (Read Carefully)

We are training a CNN to detect strong gravitational lenses in astronomical images. The training data consists of:

- **Positives:** Real LRG (luminous red galaxy) images with synthetically injected lensed arcs
- **Negatives:** The same LRG images without the injected arc

### The Problem We Discovered

A logistic regression using ONLY the central 10×10 pixels achieves **AUC = 0.95** distinguishing positives from negatives.

After investigation, we identified the root cause:

**For SIS/SIE lenses with source offset β from the optical axis:**
- Outer arc forms at θ = β + θ_E (beyond Einstein ring)
- **Inner image forms at θ = θ_E - β (inside Einstein ring)**

Our data statistics:
- Median source offset: β ≈ 0.5 × θ_E
- Median inner image position: **2.1 pixels from center**
- **91.8% of samples have inner image within 5 pixels of center**

The inner image adds detectable flux to the core. Since we're comparing the same LRG with/without arc, this difference is a trivial shortcut.

### Why This Matters

The real detection task is: "Given an image of an LRG, determine if there's a lens."

We DON'T have a "control" version to compare against. Different LRGs have vastly different intrinsic core brightnesses. A model trained on paired data may fail in the wild.

---

## Proposed Solution: Unpaired Training

### Current Design (PAIRED)

```
For each sample:
  Positive: LRG_A + arc (stamp_npz)
  Negative: LRG_A without arc (ctrl_stamp_npz)
```

### Proposed Design (UNPAIRED)

```
Positives: stamp_npz from LRG pool P
Negatives: ctrl_stamp_npz from DIFFERENT LRG pool N
Where: P ∩ N = ∅ (no LRG appears in both)
```

### Implementation

1. Group samples by unique LRG (identified by ra, dec coordinates)
2. Randomly split LRGs into two disjoint pools
3. Pool P provides positives (LRG + arc)
4. Pool N provides negatives (LRG without arc)
5. No LRG appears in both pools

---

## Mini Experiment Protocol

We propose to validate on a small dataset before committing:

### Dataset Sizes

| Dataset | Positives | Negatives | Unique LRGs |
|---------|-----------|-----------|-------------|
| Mini unpaired train | 2000 | 2000 | ~300 |
| Mini unpaired val | 500 | 500 | ~75 |
| Mini paired test | 500 | 500 | ~40 |

### Experiments

| ID | Training | Test | Purpose |
|----|----------|------|---------|
| E1 | Paired (baseline) | Paired | Current performance |
| E2 | Unpaired | Unpaired | New approach |
| E3 | Unpaired | Paired | Cross-generalization |
| E4 | E2 model | Core-only | Shortcut check |

### Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| E2 AUROC | > 0.80 | Model can still detect lenses |
| E4 Core LR AUC | < 0.65 | Shortcut is broken |
| E3 vs E2 gap | < 0.05 | Generalizes across designs |

---

## Questions for Review

### Q1: Is the Unpaired Design Sound?

Does splitting LRGs into disjoint pools correctly break the pairing shortcut?

**Concern:** There could be other correlations we haven't considered.

### Q2: What Property Balancing is Critical?

We need to ensure positives and negatives have similar LRG property distributions:
- Redshift
- Magnitude/brightness
- PSF size
- Sky background depth

**What else should we check?** Are there properties that could create new shortcuts?

### Q3: Is Our Sample Size Sufficient?

With ~300 unique LRGs in mini train, is this enough to:
- Avoid overfitting to LRG identity?
- Get statistically meaningful results?
- Extrapolate to full dataset behavior?

### Q4: What Additional Gates Should We Run?

Besides Core LR AUC, what other shortcut detectors would you recommend?

Ideas:
- Arc brightness only (does model use arc or just brightness?)
- Masked core test (performance with core zeroed out)
- LRG morphology classifier (is model detecting LRG shape?)

### Q5: Alternative Approaches?

Are there better ways to create unpaired training data?

Options we considered:
1. **Shuffle existing controls:** Use ctrl_stamp_npz from different rows (proposed)
2. **Generate new negatives:** Sample fresh LRGs without any injection
3. **Hybrid:** Some paired (same LRG), some unpaired (different LRG)
4. **Hard negative mining:** Use non-lens LRGs that "look like" lenses

### Q6: What Could Go Wrong?

What failure modes should we monitor?

Potential issues:
- LRG property imbalance creating new shortcuts
- Task becomes too hard (AUROC drops below useful threshold)
- Model memorizes LRG identities instead of learning arc features
- New shortcut we haven't anticipated

### Q7: Literature Precedent?

Are there published lens-finding papers that use unpaired training? What do they do?

---

## Data Available

Our existing dataset contains:

| Field | Description |
|-------|-------------|
| stamp_npz | 64×64×3 image (g,r,z bands) with arc |
| ctrl_stamp_npz | Same LRG without arc |
| ra, dec | LRG position (identifies unique galaxy) |
| theta_e_arcsec | Einstein radius (0.5-2.5") |
| brickname | Sky region identifier |
| psf_fwhm_used_r | PSF size |
| arc_snr | Arc signal-to-noise ratio |

Full dataset: ~1M samples, ~10k unique LRGs
Mini experiment: 4k samples, ~300 unique LRGs

---

## Timeline

- **Day 1:** Create mini dataset, run E1 and E2 (need ~2-3 hours GPU each)
- **Day 2:** Run E3, E4, analyze results, iterate if needed
- **Day 3:** Decision on full-scale implementation

---

## Current Training Status

We have 3 ablation experiments running on paired data:

| Instance | Experiment | Epoch | Status |
|----------|------------|-------|--------|
| lambda | no_hardneg | 24/50 | Keep running |
| lambda2 | no_coredrop | 6/50 | Keep running (important) |
| lambda3 | minimal | 6/50 | **Can stop for unpaired experiments** |

---

## Request

Please provide:

1. **Critical review** of the unpaired design
2. **Answers to Q1-Q7** above
3. **Alternative suggestions** if you see a better approach
4. **Risk assessment** of what we might be missing
5. **Recommendation** on whether to proceed with mini experiment

We want to get the data design RIGHT before investing more compute. High confidence is essential.

---

*End of review request*

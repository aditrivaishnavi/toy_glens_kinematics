# Unpaired Training Proposal - For External Review

**Date:** 2026-02-06
**Status:** Draft for review
**Request:** Critical review of proposed approach before implementation

---

## Executive Summary

We've identified that our current paired training design (same LRG with/without arc) creates a shortcut where a logistic regression on the central 10×10 pixels achieves AUC = 0.95. This is because the inner (counter) image of the lens adds detectable flux to the core.

**Proposed solution:** Unpaired training where positives and negatives come from DIFFERENT LRGs.

---

## Problem Statement

### Current Design (PAIRED)

```
For each sample:
  Positive: LRG_A + injected arc (stamp_npz)
  Negative: LRG_A without arc (ctrl_stamp_npz)
```

**Why this fails:**
1. The inner image adds ~10-40% extra flux to the core
2. A simple classifier can exploit this difference
3. This doesn't match real detection (you don't have a "control" version)

### Proposed Design (UNPAIRED)

```
Positives: stamp_npz from various LRGs
Negatives: ctrl_stamp_npz from DIFFERENT LRGs
```

**Why this should work:**
1. Different LRGs have natural brightness variation
2. Model must learn "what arcs look like" not "brightness difference"
3. Matches real detection task

---

## Detailed Proposal

### Phase 1: Mini Dataset Experiment

#### 1.1 Data Preparation

Create a mini unpaired dataset from existing data:

```python
# Pseudocode for unpaired data creation
def create_unpaired_dataset(df, n_samples=2000):
    # Group by unique LRG (ra, dec)
    lrg_groups = df.groupby(['ra', 'dec'])
    lrg_ids = list(lrg_groups.groups.keys())
    
    # Shuffle LRGs
    np.random.shuffle(lrg_ids)
    
    # Split: first half for positives, second half for negatives
    mid = len(lrg_ids) // 2
    pos_lrgs = set(lrg_ids[:mid])
    neg_lrgs = set(lrg_ids[mid:])
    
    # Positives: stamp_npz from pos_lrgs
    pos_samples = df[df[['ra','dec']].apply(tuple, axis=1).isin(pos_lrgs)]
    positives = [(row['stamp_npz'], 1) for _, row in pos_samples.iterrows()]
    
    # Negatives: ctrl_stamp_npz from neg_lrgs (no arc)
    neg_samples = df[df[['ra','dec']].apply(tuple, axis=1).isin(neg_lrgs)]
    negatives = [(row['ctrl_stamp_npz'], 0) for _, row in neg_samples.iterrows()]
    
    return positives[:n_samples//2] + negatives[:n_samples//2]
```

**Key property:** No LRG appears in both positive and negative sets.

#### 1.2 Mini Experiment Protocol

| Experiment | Training Data | Test Data | Purpose |
|------------|---------------|-----------|---------|
| E1: Paired (baseline) | Same LRG pos/neg | Same LRG pos/neg | Current performance |
| E2: Unpaired train | Different LRG pos/neg | Different LRG pos/neg | New approach |
| E3: Cross-test | Unpaired train | Paired test | Generalization check |
| E4: Core LR gate | Unpaired train | Core-only test | Shortcut check |

#### 1.3 Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| E2 AUROC | > 0.80 | Model can still detect lenses |
| E4 Core LR AUC | < 0.65 | Core shortcut is broken |
| E3 vs E2 gap | < 0.05 | Generalizes to paired test |

### Phase 2: Validation Questions

#### Q1: Does LRG Splitting Maintain Balance?

Check that positive and negative LRGs have similar property distributions:
- Redshift distribution
- Magnitude distribution
- PSF size distribution
- Depth distribution

**Concern:** If pos/neg LRGs are systematically different, we create a new shortcut.

**Mitigation:** Stratified splitting by key properties.

#### Q2: How Many Unique LRGs Do We Need?

Current data has ~1287 unique LRGs in 10 files. Full dataset may have 10,000+.

**Question:** Is this enough for unpaired training without overfitting to LRG identity?

**Test:** Monitor train/val gap during mini experiment.

#### Q3: What About Multiple Injections Per LRG?

Current data has 2-18 samples per LRG with different arc configurations.

**For positives:** Use all variations (different theta_E, source position, etc.)
**For negatives:** Use only one ctrl per LRG (they're identical)

**Concern:** Data imbalance (many positive variations, one negative per LRG)

**Mitigation:** Sample positives to match negative count, or weight loss.

---

## Implementation Plan

### Step 1: Create Mini Dataset Script (30 min)

```bash
# Create 4000-sample mini dataset for experiments
python create_unpaired_mini_dataset.py \
  --input /home/ubuntu/data/v5_cosmos_paired/train \
  --output /home/ubuntu/data/unpaired_mini \
  --n-positives 2000 \
  --n-negatives 2000 \
  --seed 42
```

### Step 2: Run Experiments (2-3 hours each)

```bash
# E1: Paired baseline
python train.py --data paired_mini --epochs 30 --output exp_e1_paired

# E2: Unpaired
python train.py --data unpaired_mini --epochs 30 --output exp_e2_unpaired

# E4: Core LR gate on E2
python evaluate_core_lr.py --model exp_e2_unpaired/best_model.pt
```

### Step 3: Analyze Results (30 min)

Compare metrics across experiments, document findings.

---

## Potential Issues and Mitigations

### Issue 1: LRG Property Imbalance

**Risk:** Positive LRGs systematically differ from negative LRGs (e.g., brighter, different redshift).

**Detection:** Plot property distributions for pos vs neg LRGs.

**Mitigation:** Stratified splitting, or explicit balancing.

### Issue 2: LRG Memorization

**Risk:** Model memorizes specific LRGs rather than learning arc features.

**Detection:** Monitor train/val gap; test on held-out LRGs.

**Mitigation:** Ensure no LRG overlap between train/val/test.

### Issue 3: Harder Task

**Risk:** Unpaired task is genuinely harder and performance drops significantly.

**Detection:** Compare E1 vs E2 AUROC.

**Interpretation:** Some drop is expected and acceptable if Core LR gate passes.

### Issue 4: New Shortcut

**Risk:** Model finds a different shortcut (e.g., arc brightness, not core).

**Detection:** Additional gates (arc-masked test, etc.)

**Mitigation:** Hard negative mining, diverse augmentation.

---

## Resource Requirements

### Compute

- Mini experiments: 1 GPU, ~2-3 hours each
- Can use one of the currently running Lambda instances

### Data

- No new data generation needed
- Use existing v5_cosmos_paired data with shuffled assignments

### Time

- Day 1: Create mini dataset, run E1 and E2
- Day 2: Run E3, E4, analyze results, iterate

---

## Questions for External Reviewer

1. **Is the unpaired design sound?** Does splitting LRGs into pos/neg pools correctly break the pairing shortcut?

2. **What property balance checks are critical?** Which LRG properties should we verify are balanced between pos/neg?

3. **Sample size for mini experiment?** Is 4000 samples (2000 pos, 2000 neg) sufficient for meaningful results?

4. **Additional gates?** Besides Core LR AUC, what other shortcut detectors should we run?

5. **Alternative approaches?** Are there better ways to create unpaired training data from our existing pool?

6. **Risk assessment:** What could go wrong with this approach that we haven't considered?

---

## Appendix: Current Data Statistics

From existing training data:

| Metric | Value |
|--------|-------|
| Total parquet files | 1000 |
| Samples per file (approx) | 1200 |
| Unique LRGs (10 files) | 1287 |
| Samples per LRG | 2-18 (median 12) |
| theta_E range | 0.5-2.5 arcsec |
| Source reff | 0.06-0.20 arcsec |
| PSF FWHM (r-band) | 1.0-1.6 arcsec |

---

## Appendix: Root Cause Analysis

### Why Core LR AUC = 0.95 in Current Training

1. **Inner image physics:** For SIS/SIE lens with source offset β, inner image forms at θ = θ_E - β
2. **Typical source offset:** β ≈ 0.5 × θ_E (source placed inside Einstein radius)
3. **Inner image position:** θ_inner ≈ 0.5 × θ_E ≈ 1-3 pixels from center
4. **91.8% of samples** have inner image within r < 5 pixels (core region)
5. **This is correct physics** but creates a detectable shortcut

### Why Unpaired Breaks the Shortcut

In unpaired training:
- Positive LRG_A has inner image flux → core is brighter
- Negative LRG_B has no inner image → but LRG_B may have different intrinsic brightness
- The core brightness is NOT predictive because LRG variation dominates

---

## Decision Points

After mini experiment, we need to decide:

1. **If E2 AUROC > 0.80 and E4 Core LR < 0.65:** Proceed with full unpaired training
2. **If E2 AUROC < 0.70:** Task may be too hard; consider hybrid approach
3. **If E4 Core LR > 0.70:** Unpaired didn't break shortcut; investigate why

---

*End of proposal. Awaiting external review.*

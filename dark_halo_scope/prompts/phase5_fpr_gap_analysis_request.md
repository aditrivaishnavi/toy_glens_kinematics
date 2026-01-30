# Phase 5 FPR Gap Analysis - Request for Critical Review

## Context

We are developing a CNN-based gravitational lens finder for MNRAS/ApJ publication. We have completed training and inference on our Phase 4c dataset. **We need state-of-the-art performance to be publishable.**

**Your task**: Critically review our findings, tell us if you agree with our conclusions, identify what we're missing, and provide corrected code for each phase requiring changes.

---

## CRITICAL DISCLOSURE: Unverified SOTA Claim

**I (the reviewing AI) was told that SOTA lens finders achieve FPR ~10^-4 at 85% completeness. I have NOT independently verified this claim.** My web searches for lens finding papers with specific FPR metrics did not return relevant results.

**I am asking YOU to:**
1. **Search for and cite actual published papers** on CNN/ML-based gravitational lens finding
2. **Report what FPR and completeness values those papers actually achieve**
3. **Provide the paper titles, arxiv IDs, and specific table/figure references**
4. **Tell us what is the realistic SOTA benchmark we should be comparing against**

Do NOT accept the 10^-4 claim at face value. We need evidence-based targets.

---

## 1. Inference Results

### Model Performance Summary

| Model | Val AUROC | Val AP | Training Time |
|-------|-----------|--------|---------------|
| ResNet18 (10 epochs) | 0.9963 | 0.9964 | ~8 hours |
| ConvNeXt-Tiny (6 epochs) | 0.9976 | 0.9978 | ~8.5 hours |

### FPR vs Completeness Analysis (ResNet18)

```
======================================================================
FPR vs TPR (Completeness) Analysis
======================================================================
Completeness     FPR             Threshold       Log10(FPR)
----------------------------------------------------------------------
      99.0%     4.14e-01         0.0105         -0.38
      95.0%     1.88e-01         0.0699         -0.73
      90.0%     1.02e-01         0.1622         -0.99
      85.0%     6.17e-02         0.2540         -1.21
      80.0%     3.97e-02         0.3380         -1.40
      70.0%     1.79e-02         0.4879         -1.75
      50.0%     5.58e-03         0.6870         -2.25
```

### Comparison to CLAIMED SOTA (UNVERIFIED)

```
======================================================================
COMPARISON TO CLAIMED SOTA (FPR=10^-4 @ 85% completeness) - UNVERIFIED
======================================================================
Our FPR at 85% completeness: 6.17e-02 (log10: -1.21)
CLAIMED SOTA target:         1.00e-04 (log10: -4.00)  <-- NOT INDEPENDENTLY VERIFIED
Gap (if claim is accurate):  2.79 orders of magnitude

Our completeness at FPR=10^-4: 0.4%
CLAIMED SOTA target:           85.0%  <-- NOT INDEPENDENTLY VERIFIED
```

**WARNING**: The "10^-4 FPR at 85% completeness" figure was mentioned in conversation but I could NOT find papers to verify this. **Please search for and cite actual papers with their reported metrics.**

**If this SOTA claim is accurate**, the gap is approximately 3 orders of magnitude. But we need to verify what SOTA actually is before drawing conclusions.

---

## 2. Our Analysis of the Gap

### Data Preparation Pipeline Flowchart

```
Phase 3: LRG Selection
├── Selection: TYPE != 'PSF', positive fluxes in r/z/w1
├── v3_color_relaxed: z < 20.4, r-z > 0.4, z-w1 > 0.8
└── Output: ~145k LRG targets across footprint

Phase 4a: Injection Manifest Generation
├── Injection grid (grid_small):
│   ├── theta_e: [0.3, 0.6, 1.0] arcsec
│   ├── src_dmag: [1.0, 2.0] (source 1-2 mag fainter than lens)
│   ├── src_reff: [0.08, 0.15] arcsec
│   ├── src_e: [0.0, 0.3]
│   └── shear: [0.0, 0.03]
├── Control fraction: 50%
└── Control strategy: SAME LRG cutouts with theta_e=0 (no injection)

Phase 4b: Coadd Cache
└── Download DR10 coadd images for target bricks

Phase 4c: Injection + Stamps
├── Lens model: SIE (lenstronomy) + Sersic source profile
├── PSF convolution: Per-band Gaussian approximation
├── Stamp size: 64×64 pixels (16.8 arcsec)
└── Output: ~10.6M samples (50% controls, 50% injections)

Phase 5: Training
├── Architecture: ResNet18 (modified for 64×64 input)
├── Normalization: Per-channel median/MAD robust normalization
├── Augmentation: Random flips + 90° rotations
└── Loss: BCE with logits
```

### Identified Critical Gaps

#### Gap #1: Controls Are Trivially Different From Positives

**What we did**: Controls are the exact same LRG cutouts used for injections, but without any flux added.

**The problem**: The CNN learns to detect "is there extra flux in this image?" rather than "does this image contain arc-like morphology consistent with gravitational lensing?"

**What SOTA papers do**: Include hard negatives that visually resemble lenses but are not:
- Ring galaxies
- Spiral galaxies with prominent arms
- Edge-on disk galaxies
- Merger systems with tidal features
- AGN with jets or extended structure

#### Gap #2: Most Injections Are Unresolved

**Statistics from our data**:
```
theta_e range: 0.3-1.0 arcsec
Median PSF:    1.3 arcsec

Resolution distribution (theta_e / PSF):
- Unresolved (< 0.5):   59.6%
- Marginal (0.5-1.0):   40.3%
- Resolved (>= 1.0):     0.1%
```

**The problem**: 60% of our "lenses" have Einstein radii smaller than half the PSF FWHM. The arc morphology is completely blurred out - the only detectable signal is "there's extra flux here."

**What SOTA papers do**: Focus on theta_e > PSF or theta_e > 0.8*PSF where arc morphology is actually detectable.

#### Gap #3: Parametric Source Profiles

**What we did**: Sersic elliptical profiles for source galaxies.

**What SOTA papers do**: Use real galaxy cutouts from COSMOS/HST with realistic clumpy, irregular morphology.

#### Gap #4: No Hard Negative Mining

**What we did**: Fixed training set, no iteration.

**What SOTA papers do**: Iterative process:
1. Train initial model
2. Run on survey data
3. Collect false positives
4. Add to training as hard negatives
5. Retrain, repeat

---

## 3. Our Perspective on What To Do

### Immediate Fixes (Can do now)

1. **Extend theta_e range to [0.5, 2.5] arcsec** - Ensures most injections are resolved
2. **Filter training to theta_e/PSF > 0.5** - Only train on resolvable lenses
3. **Increase source brightness** - src_dmag in [0.5, 1.5] instead of [1.0, 2.0]

### Medium-Term Fixes (Require new data)

1. **Add hard negatives from Galaxy Zoo ring galaxy catalog**
2. **Add high-ellipticity objects (edge-on disks)**
3. **Query for known non-lens arc-like objects**

### Long-Term Fixes (Significant pipeline changes)

1. **Replace Sersic sources with real COSMOS cutouts**
2. **Implement hard negative mining loop**
3. **Larger stamps (128×128) for better context**

---

## 4. Questions for Your Review

### FIRST: Establish the Actual SOTA Benchmark

**Before any other analysis, please answer these questions with citations:**

0. **What are the actual published SOTA results for CNN/ML gravitational lens finders?**
   - Cite specific papers (arxiv ID, title, authors, year)
   - Quote the exact FPR and completeness/TPR values from those papers
   - Specify which table/figure contains these numbers
   - Describe what dataset they evaluated on (simulated? real survey? which survey?)

### Agreement Check (After Establishing SOTA)

1. **Do you agree that the FPR gap is primarily due to trivially-easy controls rather than model architecture or training procedure?**

2. **Do you agree that training on 60% unresolved lenses is a fundamental flaw that no amount of architectural improvement can fix?**

3. **Do you agree that our current results, while showing the pipeline works, are NOT publishable without addressing the negative sample quality?**

### What Are We Missing?

4. **What aspects of our analysis are incorrect or incomplete?**

5. **Are there other critical gaps we haven't identified?**

6. **Is there a faster path to SOTA performance that we're overlooking?**

7. **How do SOTA papers actually construct their training sets? Please cite specific papers and their methodology.**

---

## 5. Requested Deliverables

### For Each Phase Requiring Changes, Please Provide:

#### A. Clear Summary
- What the change is trying to accomplish
- Why it addresses the identified gap
- Expected impact on FPR/completeness

#### B. Self-Reviewed Code
- Complete, runnable code (not snippets)
- Tested logic (no obvious bugs)
- Clear comments explaining key decisions

#### C. Integration Instructions
- How to run the code
- Required inputs/outputs
- How it connects to other phases

### Specific Code Requests:

1. **Phase 4a modification**: New injection grid with extended theta_e range [0.5, 2.5]

2. **Phase 4a modification**: Hard negative sampling strategy (if implementable without external catalogs)

3. **Phase 5 modification**: Training data filtering to exclude unresolved samples (theta_e/PSF < 0.5)

4. **Phase 5 modification**: Any architectural changes you recommend

5. **Inference script**: Updated to compute stratified FPR by theta_e bin and resolution bin

---

## 6. Current Code Locations

For reference, our current implementations are at:

```
dark_halo_scope/emr/spark_phase4_pipeline.py          # Phase 4a/4b/4c
dark_halo_scope/model/phase5_train_fullscale_gh200.py # Training
dark_halo_scope/model/phase5_infer_scores.py          # Inference
dark_halo_scope/scripts/train_lambda.py               # Simpler training script
```

### Key Functions in spark_phase4_pipeline.py:

- `build_grid(name)` at line 176 - Defines injection parameter grids
- `inject_sie_stamp()` at line 725 - SIE lens injection using lenstronomy
- `stage_4a_build_manifests()` at line 1004 - Manifest generation with control assignment

---

## 7. Validation Data

### Dataset Statistics (Train Tier)

```
Total samples:     10,627,158
Controls:           5,299,324 (49.9%)
Injections:         5,327,834 (50.1%)

Injection parameters:
- theta_e_arcsec:  min=0.30, max=1.00, median=0.60
- src_dmag:        min=1.00, max=2.00, median=2.00
- arc_snr:         min=0.00, median=22.86, max=9154.72

Observing conditions:
- psfsize_r:       median=1.305 arcsec
- psfdepth_r:      median=24.55 mag
```

---

## 8. Summary of What We Need

1. **Validation of our gap analysis** - Are we correctly identifying the problems?

2. **Corrected code for each phase** - Self-reviewed, complete, runnable

3. **Clear summaries** - So we can verify the code does what you intend

4. **Explicit next steps** - Prioritized list of actions to reach SOTA performance

5. **Honest assessment** - If SOTA is not achievable with reasonable effort, tell us what performance IS achievable and how to position the paper accordingly.

---

**We are committed to scientific rigor and need your honest, critical assessment. Please do not hedge or give vague suggestions - we need specific, actionable guidance with working code.**


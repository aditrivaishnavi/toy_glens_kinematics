# Request: Detailed Implementation Blueprint for DR10 Lens Finding Study

**Date:** 2026-02-07  
**Request Type:** Technical implementation guidance with code  
**Target Output:** MNRAS-quality research

---

## Context

We are pivoting from simulation-based training (which failed due to unrealistic arc brightness—100× too bright) to a real-data approach following Huang et al. We need a **concrete, detailed implementation blueprint** for:

1. **Option 1:** Selection Functions and Failure Modes paper
2. **Option 2:** Ensemble Diversification via Domain-Specialized Training

---

## Required Reading

Please thoroughly review these papers before providing guidance:

### Paper I: Huang et al. 2020
- **Title:** "Finding Strong Gravitational Lenses in the DESI DECam Legacy Survey"
- **arXiv:** 2005.04730
- **Key content:** Initial methodology, ResNet architecture, training on real lenses

### Paper II: Huang et al. 2021  
- **Title:** "Discovering New Strong Gravitational Lenses in the DESI Legacy Imaging Surveys"
- **arXiv:** 2110.09488
- **Key content:** The z-band exposures confound lesson, improved sampling

### Paper III: Storfer et al. 2024
- **Title:** "New Strong Gravitational Lenses from the DESI Legacy Imaging Surveys Data Release 9"
- **arXiv:** (find in references)
- **Key content:** Grading criteria (A/B/C), EfficientNet addition, visual inspection protocol

### Paper IV: Inchausti et al. 2025
- **Title:** "Detecting strong satisfying lenses in DESI Legacy Survey imaging using foundation models and custom CNNs"
- **arXiv:** 2508.20087
- **Key content:** 134,182 nonlenses, 100:1 ratio, meta-learner, top 0.01% thresholding, Tractor type stratification

---

## Our Current Data

### Downloading Now (5,104 DESI Lens Candidates)

```
Source: lenscat (https://github.com/lenscat/lenscat)
Filter: All entries with "DESI" in name (Huang et al. discoveries)
Count: 5,104 candidates
Grading: Mix of "confident" and "probable"

For each lens:
- FITS cutout (g, r, z bands): 101×101 pixels (~26.4")
- JPEG RGB image
- Metadata: RA, Dec, zlens, grading, reference

Output: dark_halo_scope/planc/data/positives/
```

### Available Anchor Lenses (Tier-A: Spectroscopically Confirmed)

```
Sources: SLACS, BELLS, SL2S, SWELLS, GALLERY
Count: ~100 with visible arcs in DR10
Use: Ground truth for completeness measurement
```

### Negative Sampling Plan

```
Strategy: Stratified sampling matching Huang et al. methodology
Bins:
  - z-band exposures: [1, 2, 3, 4+]
  - PSF FWHM: [<1.0", 1.0-1.3", 1.3-1.6", >1.6"]
  - Depth: [<22.5, 22.5-23.0, 23.0-23.5, >23.5 mag]
  - Tractor type: [SER, DEV, REX, EXP]

Ratio: 100:1 negatives:positives per bin
Target: ~500,000 total negatives
```

### Contaminant Categories

```
- Ring galaxies (morphological confusers)
- Face-on spirals (ring-like structure)
- Mergers/interactions (arc-like tidal features)
- Edge-on disks (extended linear features)
- Bright star artifacts (spikes, halos, ghosts)
```

---

## Project Code Structure

```
dark_halo_scope/planc/
├── README.md                    # Project goal (written)
├── data/
│   ├── positives/               # Real lens cutouts (downloading)
│   │   ├── tier_a/              # Spectroscopically confirmed
│   │   └── tier_b/              # Huang et al. candidates
│   ├── negatives/               # Stratified non-lens galaxies
│   │   ├── by_stratum/          # Organized by (exp, psf, depth) bin
│   │   └── catalog.csv
│   └── contaminants/            # Morphological confusers
│       ├── rings/
│       ├── spirals/
│       ├── mergers/
│       └── artifacts/
├── training/
│   ├── baseline/                # Single model
│   └── ensemble/                # Domain-split variants
├── evaluation/
│   ├── completeness/            # Selection function surfaces
│   ├── calibration/             # Reliability diagrams
│   └── failure_modes/           # FPR by contaminant type
└── paper/
    ├── figures/
    └── draft/
```

---

## Specific Questions Requiring Your Guidance

### 1. Model Architecture

What specific models should we implement?

- **ResNet variant?** Which depth (18, 34, 50)? Pre-trained on ImageNet or train from scratch?
- **EfficientNet variant?** B0, B1, B2? Pre-trained?
- **Metadata branch?** Should we include Tractor catalog features (z_nexp, PSF, depth, morphology type)?
- **Input format?** 3-channel (g,r,z) or single-channel (z-band only)? Cutout size?

### 2. Selection Function Methodology

Should the selection function be computed using:

- **A single model** evaluated across strata?
- **Multiple specialized models** (one per stratum)?
- **An ensemble** where we measure individual model sensitivities?

How do we properly compute:
- Completeness as f(θ_E, depth, PSF)?
- Uncertainty on completeness estimates (bootstrap? binomial CI?)?
- Calibration curves with limited positive samples?

### 3. Ensemble Diversification (Option 2)

Their paper notes the meta-learner gains are modest because base models are trained on the same data. They suggest training on different subsets.

Concrete questions:
- How should we define the domain splits? By:
  - Seeing quality (good vs poor)?
  - Depth (shallow vs deep)?
  - Morphology (SER/DEV vs EXP/late-type)?
  - Geographic region (to test spatial generalization)?
  
- How do we measure "diversity"?
  - Prediction correlation?
  - Disagreement rate on edge cases?
  - Ensemble variance?

- What ensemble method?
  - Simple averaging?
  - Learned meta-classifier?
  - Stacking?

### 4. Training Protocol

Please specify:
- Loss function (BCE? Focal loss for class imbalance?)
- Optimizer and learning rate schedule
- Data augmentation (flips, rotations, noise injection?)
- Batch size and epochs
- Early stopping criteria
- Handling class imbalance (weighting? oversampling?)

### 5. Evaluation Protocol

For the selection function paper, we need:

**Completeness measurement:**
- How to handle the fact that Tier-B positives are only "candidates" not confirmed?
- Should we weight by grading (confident vs probable)?
- How many bootstrap iterations for CI?

**Calibration:**
- How to estimate true prevalence for reliability curves?
- Should we report both prevalence-free (ROC/PR) and scenario-weighted metrics?

**Failure mode analysis:**
- How to systematically identify what causes false positives?
- Should we use SHAP/GradCAM for interpretability?

### 6. Paper Structure

What sections and figures will reviewers expect?

---

## Deeper Questions We Need Answered

### 7. Data Quality and Preprocessing

- **Bad pixels and artifacts:** How should we handle cosmic rays, bad columns, edge effects in FITS cutouts? Mask them? Interpolate? Flag and exclude?
- **Normalization:** Per-image normalization vs global statistics from DR10? Asinh stretch? Linear scaling? What do Huang et al. use?
- **Missing data:** Some cutouts may have incomplete band coverage. How to handle? Exclude entirely or impute?
- **Cutout centering:** DR10 cutouts are centered on RA/Dec. If the lens is offset from catalog position, does this bias the model?

### 8. The Label Noise Problem

**Critical issue:** We are training on Tier-B candidates discovered by Huang et al.'s model. This creates circularity:

- Their model found these candidates
- We train on these candidates  
- Our model will inherit their model's biases

How do we:
- Quantify how much bias this introduces?
- Use semi-supervised techniques to reduce label noise impact?
- Weight confident vs probable candidates appropriately?
- Should we use label smoothing (e.g., 0.9 instead of 1.0 for candidates)?

**What fraction of Tier-B candidates are actually lenses?** Huang et al. report purity, but it varies by grade. How should we incorporate this uncertainty?

### 9. Reproducibility and Fair Comparison

- **We cannot exactly replicate Huang et al.** because we don't have their exact train/test splits. How do we make fair comparisons?
- **What baselines should we include?**
  - Random classifier
  - Morphology-only classifier (Tractor type)
  - Classical methods (ring detection, arc fitting)
  - Their reported numbers (with appropriate caveats)
- **How do we ensure our code is reproducible?** Fixed seeds, logged hyperparameters, version-controlled data splits?

### 10. Statistical Rigor

- **Multiple testing:** If we report completeness across 50+ strata, we're doing many hypothesis tests. Do we need Bonferroni or FDR correction?
- **Low-N problem:** With only ~100 Tier-A lenses, some strata will have 0-5 lenses. How do we report completeness with such small samples? Bayesian binomial? Just report "insufficient data"?
- **Correlation structure:** Lenses in the same region share observing conditions. Does this violate independence assumptions? Do we need spatial cross-validation?
- **How do we distinguish "model failure" from "lens not detectable in this data"?** (i.e., some lenses may be genuinely invisible at DR10 depth)

### 11. Generalization and Domain Shift

- **Will models trained on DR10 generalize to Rubin LSST or Euclid?** Should we discuss this in the paper?
- **Are we overfitting to DR10-specific artifacts?** (e.g., specific telescope optics, processing pipeline quirks)
- **Geographic variation:** Does the model perform differently in different parts of the sky? How do we test this?
- **Temporal variation:** DR10 includes data from multiple epochs. Does observing date matter?

### 12. Practical Deployment Considerations

- **Inference speed:** To scan 43 million DR10 galaxies at 0.1s each takes 50 GPU-days. Is this acceptable? Do we need distillation or pruning?
- **Memory constraints:** How do we batch process millions of cutouts? Stream from disk? Pre-load to GPU memory?
- **Prioritization for visual inspection:** With top 0.01% yielding ~4,300 candidates, how do we rank them for human review? Confidence? Rarity? Interestingness?
- **Should we release a public inference endpoint or just the model weights?**

### 13. Alternative Approaches We Should Consider

Please evaluate whether these would strengthen or complicate the paper:

- **Foundation models:** DINOv2 or CLIP as frozen feature extractors + small classifier head. Faster to train, potentially more robust. Trade-off?
- **Self-supervised pretraining:** Pretrain on unlabeled DR10 galaxies (SimCLR, MAE), then fine-tune. Worth the extra complexity?
- **Active learning:** Instead of random negative sampling, iteratively sample hard negatives based on model uncertainty. Feasible in 4 weeks?
- **Multi-task learning:** Predict lens vs non-lens AND Tractor type simultaneously. Does shared representation help?

### 14. Failure Modes of Our Own Approach

Help us anticipate what could go wrong:

- **Selection bias in Tier-B:** If Huang et al.'s model preferentially found "easy" lenses (bright arcs, good seeing), our completeness estimates will be optimistic. How do we test for this?
- **Stratification bin choices:** If our exposure/PSF/depth bins are wrong, the selection function will be misleading. How do we validate bin choices?
- **Contaminant catalog incompleteness:** If we miss important contaminant categories (e.g., galaxy clusters, AGN hosts), our FPR estimates will be wrong. How do we ensure completeness?
- **Overfitting to evaluation set:** If we tune hyperparameters on Tier-A lenses, we'll overfit to those specific systems. How do we avoid this with so few anchors?

### 15. What Makes This Paper High-Impact?

Beyond "just publishable," what would make this paper widely cited?

- **Is selection function the right framing?** Or should we emphasize something else (e.g., "audit of ML lens finders," "quantifying the human-ML gap")?
- **What figures will people cite?** (e.g., completeness heatmap, failure mode gallery)
- **Should we release artifacts beyond the paper?** Trained models, candidate catalog, selection function lookup table?
- **What communities care about this?** Just strong lensing? Broader ML-for-astronomy? Survey design?

### 16. Technical Details We Need

Please provide or point us to:

- **How to query DR10 for z-band exposure counts, PSF FWHM, depth at a given RA/Dec.** API? Catalog files? SQL?
- **How to match cutout positions to Tractor catalog entries.** Cone search? Direct lookup?
- **What preprocessing did Huang et al. actually use?** (normalization, cropping, augmentation) — cite specific paper sections.
- **Standard train/val/test split ratios for this problem.** 80/10/10? 70/15/15?

### 17. Reviewer Objections to Preempt

What will MNRAS reviewers criticize? Help us address these proactively:

- "Your completeness is only measured on lenses found by the same type of model"
- "You have no confirmed labels, only candidates"
- "Your negative sampling may not represent the true galaxy population"
- "The selection function depends on your specific model; another model would give different results"
- "You haven't shown this matters for any downstream science"

For each likely objection, what is the honest response and/or additional experiment we should include?

---

## Request Format

Please provide:

### A. Detailed Implementation Plan
- Step-by-step with dependencies
- Estimated compute requirements
- Potential failure modes and mitigations

### B. Complete Code Files

For each component, provide **complete, runnable Python code** that I can save directly:

1. **`data/download_negatives.py`** — Stratified negative sampling from DR10
2. **`data/query_dr10_metadata.py`** — Get exposure counts, PSF, depth for any RA/Dec
3. **`data/prepare_dataset.py`** — Create train/val/test splits maintaining stratification
4. **`data/preprocessing.py`** — FITS loading, normalization, augmentation transforms
5. **`training/models.py`** — ResNet, EfficientNet, optional metadata branch
6. **`training/train_baseline.py`** — Single model training loop
7. **`training/train_ensemble.py`** — Domain-split ensemble training
8. **`training/losses.py`** — BCE, focal loss, label smoothing options
9. **`evaluation/compute_completeness.py`** — Selection function surfaces with bootstrap CI
10. **`evaluation/compute_calibration.py`** — Reliability diagrams, ECE, prevalence adjustment
11. **`evaluation/analyze_failures.py`** — FPR by category, GradCAM visualizations
12. **`evaluation/statistical_tests.py`** — Multiple testing corrections, spatial CV
13. **`inference/batch_scorer.py`** — Efficient inference on millions of cutouts
14. **`paper/generate_figures.py`** — Publication-ready plots (completeness heatmaps, calibration curves, failure galleries)
15. **`paper/generate_tables.py`** — LaTeX tables for results

### C. Validation Checkpoints

What intermediate results should we verify before proceeding?
- After data preparation: what statistics should we check?
- After training: what metrics indicate success/failure?
- Before paper submission: what claims need additional support?

---

## Constraints

- **Framework:** PyTorch preferred (we have existing infrastructure)
- **Compute:** Single GPU (V100/A100) for training; can scale for hyperparameter search
- **Timeline:** 4 weeks to paper draft
- **Honesty:** Do not suggest approaches that sound good but don't work in practice. Be critical.

---

## Critical Requirements

1. **Be thorough.** Read all four papers carefully. Cite specific sections when relevant.

2. **Be honest.** If something is uncertain or there are multiple valid approaches, say so. Don't pretend there's one "correct" answer when the literature is ambiguous.

3. **Be scientifically rigorous.** Every metric, every claim should be defensible to an MNRAS referee. Anticipate objections.

4. **Provide complete code.** Not pseudocode, not fragments. Complete, runnable Python files with all imports, proper error handling, and comments explaining non-obvious choices.

5. **Cite your sources.** When recommending a specific approach (e.g., focal loss, specific augmentations), cite where this was shown to work for similar problems.

---

## Summary

We are building a paper titled:

> **"Selection Functions and Failure Modes of Real-Image Lens Finders in DESI Legacy Survey DR10"**

with a secondary contribution:

> **"Ensemble Diversification via Domain-Specialized Training"**

Please provide the detailed blueprint and code to execute this research in 4 weeks, ensuring the methodology will survive peer review at MNRAS.

---

## Science Impact Questions

### Why Does Selection Function Matter?

Help us articulate the downstream science impact:

1. **For cosmology:** If we use lens counts to constrain dark matter, how does incomplete/biased selection affect the inference? Can you quantify this?

2. **For dark matter substructure:** If the selection function depends on arc smoothness, are we biased against lenses that probe small-scale structure?

3. **For time-delay cosmography:** If we preferentially find quads vs doubles, does this bias H₀ measurements?

4. **For survey design:** How should Rubin LSST or Roman design their lens searches given what we learn about DR10 selection?

### What Question Should This Paper Answer?

Frame the paper around a compelling question:

- Option A: "What fraction of strong lenses in DR10 can current methods detect?"
- Option B: "Where do ML lens finders fail, and why?"
- Option C: "How should we design lens searches to minimize selection bias?"
- Option D: "Are published lens catalogs representative of the true lens population?"

Which framing is most impactful? Most publishable? Most honest given our data?

---

## Final Meta-Questions

1. **Is 4 weeks realistic?** If not, what is the minimum viable paper we could write in 4 weeks, with a clear path to a fuller version later?

2. **Are we solving the right problem?** Given our resources (5K candidates, 100 anchors, single GPU), is selection function the best use of our time? What else could we do?

3. **What would you do differently?** If you were starting this project from scratch with our constraints, what approach would you take?

4. **What are we missing?** What obvious considerations have we overlooked that a domain expert would immediately notice?

---

*We value honest assessment over optimistic promises. If parts of this plan are infeasible or misguided, say so and suggest corrections.*

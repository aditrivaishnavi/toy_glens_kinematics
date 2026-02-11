# MNRAS Paper Plan: Selection Functions and Failure Modes

**Date:** 2026-02-07 (original); Execution Status updated 2026-02-11  
**Status:** Strategic pivot based on LLM review. **NOTE**: The core plan sections (Options 1-3, 4-week blueprint) predate the Paper IV parity course correction. The current ground truth for training details is `stronglens_calibration/docs/MNRAS_RAW_NOTES.md`.  
**Target:** MNRAS-level publication  
**Current focus:** Selection function audit (Option 1) with Paper IV parity baseline models

---

## Key Insight

> "You can still produce original, defensible MNRAS-level work without pretending your simulation path was 'the' answer. What changed is that you measured a specific failure mode (arc SNR realism), and that measurement should drive the next paper."

**Our contribution is NOT**: "We trained a lens finder"  
**Our contribution IS**: "We quantified what lens finders can and cannot detect in DR10"

---

## Chosen Path: Option 1 + Option 2 Combined

### Paper Title (working)

**"Selection Functions and Failure Modes of Real-Image Lens Finders in DESI Legacy Survey DR10"**

### Core Contributions

1. **Rigorous, transparent selection function** for DR10 lens finding
2. **Bias audit** tied to operational choices (exposures binning, morphology types, PSF/depth)
3. **Controlled ensemble diversification study** motivated by Huang et al.'s own limitation statement

---

## What Huang et al. (2508.20087) Establishes as Baseline

We build on their methodology (cite explicitly):

| Their Practice | Our Extension |
|----------------|---------------|
| Train on real DR10 cutouts | Same, but with stratification study |
| 134,182 nonlenses, ~100:1 ratio | Match this, add depth/PSF bins |
| Bin nonlenses by z-band exposures | Extend to PSF FWHM, depth, EBV |
| ResNet + EfficientNet + meta-learner | Add domain-split variants |
| Top 0.01 percentile threshold | Same, plus calibration curves |
| Purity by Tractor type | Extend to full selection function |

---

## Research Design

### Data

**Positives (lenses):**
- Tier-A: Spectroscopically confirmed (SLACS, BELLS, SL2S) with visible arcs
- Tier-B: Huang et al. candidates (5,104 from lenscat) as "silver positives"

**Negatives (non-lenses):**
- DR10 galaxies matched to search population
- Stratified by:
  - z-band exposures (like they do)
  - PSF FWHM bins
  - Depth bins
  - Tractor type (SER/DEV/REX) — EXP excluded for Paper IV parity per LLM recommendation

**Contaminants (for FPR):**
- Morphological confusers: rings, spirals, mergers, edge-on
- Artifacts: bright-star spikes, ghosts
- Hard negatives: high-score rejects from their pipeline (if accessible)

### Measurements

**1. Completeness (recall) as function of:**
- z-band exposures/passes
- PSF FWHM
- Depth (5σ magnitude limit)
- Galactic extinction (EBV)
- Tractor type (SER/DEV/REX/EXP)
- Lens scale (θ_E where available)

**2. False positive rate by contaminant family:**
- Rings, spirals, mergers, edge-on
- Artifacts (spikes, ghosts)
- Category-conditional FPR with uncertainty

**3. Calibration:**
- Reliability curves
- Expected calibration error
- How "0.99 score" maps to actual probability under realistic prevalence

### Key Figures

1. **Completeness surfaces**: Recall vs (depth, PSF) heatmaps
2. **FPR by category**: Bar chart with bootstrap CIs
3. **Calibration curve**: Predicted probability vs actual frequency
4. **Selection function summary**: "Detectable if X; not interpretable outside X"

---

## Option 2: Ensemble Diversification Study

### Their Limitation (from paper)

> "The meta-learner cannot hugely outperform averaging because the two models are trained on the same data."

They suggest training on different subsets could increase diversity.

### Our Experiment

Train domain-specialized base models:
- **Model A**: Good seeing + deep exposures
- **Model B**: Worse seeing + shallow exposures  
- **Model C**: SER/DEV morphologies only
- **Model D**: EXP/late-type heavy

**Measure:**
1. Prediction correlation drops
2. Ensemble discovery set changes
3. Selection function becomes broader

**Deliverable:**
> "Ensemble diversity improves yield at fixed human-inspection budget"

---

## Option 3: Simulations as Evaluation Tool

### Reframe Our Simulation Work

**NOT**: "We trained on fake arcs and it worked"  
**BUT**: "Simulations are controlled probes for response characterization"

### How to Use

1. Calibrate injections so arc-annulus SNR matches real DR10 (our measurement = calibration target)
2. Inject into real DR10 galaxies
3. Measure detection probability vs (θ_E, source mag, PSF, depth) under real-trained model
4. Report selection function combining:
   - Empirical performance on real anchors
   - Injection-recovery for controlled extrapolation

---

## 4-Week Blueprint

### Week 1: Data Preparation

- [ ] Freeze evaluation sets (anchors Tier-A/Tier-B + contaminants)
- [ ] Download all 5,104 DESI candidates from lenscat
- [ ] Build stratified negative sampling:
  - z-band exposures bins
  - PSF FWHM bins
  - Depth bins
- [ ] Implement calibration + uncertainty reporting framework

### Week 2: Baseline Training

- [ ] Train baseline real-image model(s)
- [ ] Produce metrics:
  - Recall on Tier-A anchors
  - FPR by contaminant type
  - Calibration curves
- [ ] Document baseline performance

### Week 3: Ensemble Experiments

- [ ] Train domain-split ensemble variants (Option 2)
- [ ] Measure:
  - Prediction correlation
  - Yield at fixed budget
  - Selection function broadening
- [ ] Compare to single-model baseline

### Week 4: Paper Writing

- [ ] Methods: Data, confound controls, stratification
- [ ] Results: Selection function surfaces, failure modes
- [ ] Discussion: What is/isn't detectable in DR10; implications for search strategies
- [ ] Figures: Completeness heatmaps, calibration curves, FPR bars

---

## Framing for Reviewers

### We Claim

1. A **rigorous, transparent selection function** for DR10 lens finding
2. A **bias audit** tied to operational choices already known to matter
3. A **controlled ensemble diversification study** motivated by prior limitation

### We Do NOT Claim

- "We invented lens finding"
- "We beat Huang et al.'s discovery count"
- "Simulations are the answer"

### Why This Is Publishable

- Astrophysics-methods work with direct survey relevance
- Does not require beating their discovery count
- Addresses gaps they explicitly acknowledge
- Honest about what simulations can/cannot do

---

## Execution Status (Updated 2026-02-11)

### Completed

| Step | Date | Details |
|------|------|---------|
| Positive catalog | 2026-02-09 | 4,788 candidates matched to DR10 (389 Tier-A, 4,399 Tier-B; from 5,104 in lenscat, 316 outside DR10) |
| Negative pool extraction (EMR) | 2026-02-10 | 26.7M galaxies from DR10 sweeps (SER/DEV/REX) |
| Stratified sampling (EMR) | 2026-02-11 | 453,100 negatives, 100:1 per stratum, 85:15 N1:N2 |
| Cutout generation batch 1 (EMR) | 2026-02-10 | 411,662 negative cutouts |
| Cutout generation batch 2 (EMR) | 2026-02-11 | 35,233 additional negative cutouts |
| Cutout sync (batch 2 to NFS) | 2026-02-11 | Synced to Lambda NFS for full coverage |
| Manifest generation | 2026-02-11 | Two manifests: 70/30 (parity) + 70/15/15 (audit) |
| HEALPix split verification | 2026-02-11 | Zero overlap confirmed between train/val/test |

### In Progress

| Step | Started | Status |
|------|---------|--------|
| EfficientNetV2-S parity training | 2026-02-11 | 160 epochs, StepLR@130, eff batch 512, 101x101, unweighted CE |
| ResNet-18 parity training | Queued | Starts after EfficientNetV2-S completes |

### Model Training Configuration (Paper IV Parity)

Both models use the corrected manifest with **446,893 negatives** (batch 1: 411,662 + batch 2: 35,233 cutouts, minus 6,207 missing) achieving **93.3:1** ratio (Paper IV: ~98:1).

**Manifest rebuild**: Initial training (epochs 1–~20) used batch 1 cutouts only (86:1). Training was stopped, batch 2 synced from S3, manifests rebuilt, and training restarted from scratch.

| Parameter | ResNet-18 | EfficientNetV2-S | Paper IV |
|-----------|-----------|-------------------|----------|
| Params | 11.2M | 20.2M | 194K / 20.5M |
| Input | 101x101x3 | 101x101x3 | 101x101x3 |
| Epochs | 160 | 160 | 160 |
| Eff batch | 2048 | 512 | 2048 / 512 |
| LR | 5e-4 | 3.88e-4 | 5e-4 / 3.88e-4 |
| StepLR | @80 | @130 | @80 / @130 |
| Loss | Unweighted BCE | Unweighted BCE | Unweighted CE |
| Split | 70/30 | 70/30 | 70/30 |

### Known Limitations (to be addressed in paper)

1. **No negative cleaning**: Paper IV used Spherimatch + prior model p>0.4 filtering. We did not replicate this (prior model not available). Disclosed as limitation.
2. **ResNet capacity mismatch**: Our ResNet-18 has 58x more params than Paper IV's custom 194K-param Lanusse-style ResNet. Explicitly reported.
3. **Normalization unspecified**: Paper IV doesn't describe their normalization. Our raw_robust (outer-annulus median/MAD) is defensible but may differ.

### Pending

- Evaluation metrics (AUC, ECE, MCE, FPR by confuser)
- Meta-learner (1-layer NN, 300 nodes, Coscrato et al. 2020)
- Selection function injection grid
- Tier-weighted audit training runs
- Paper draft

---

## References to Cite

- Inchausti et al. 2025 (Paper IV): arXiv:2508.20087 - Main methodological anchor
- Huang et al. 2020 (Paper I): arXiv:2005.04730 - Initial methodology
- Huang et al. 2021 (Paper II): arXiv:2206.02764 - DR9 extension, exposures binning
- Storfer et al. 2024 (Paper III): arXiv:2509.18086 - Keck spectroscopy, grading criteria
- Lanusse et al. 2018: arXiv:1703.02642 - Shielded ResNet architecture basis
- Tan & Le 2021: EfficientNetV2 architecture
- Coscrato et al. 2020: Feature-weighted stacking (meta-learner)
- Dey et al. 2019: arXiv:1804.08657 - DESI Legacy Imaging Surveys
- Górski et al. 2005: HEALPix spatial indexing
- Bolton et al. 2008: SLACS (anchor lenses)
- Brownstein et al. 2012: BELLS (anchor lenses)

---

## Success Criteria

A successful paper will:

1. Show completeness varies predictably with observing conditions
2. Identify specific failure modes (which contaminants fool the model)
3. Provide calibration so scores can be interpreted probabilistically
4. Demonstrate ensemble diversity improves detection or characterization
5. Give clear guidance: "Use this model for X; do not trust it for Y"

---

*This plan incorporates strategic guidance from external LLM review (2026-02-07)*  
*Updated 2026-02-11: Added execution status, model training config, and limitations*


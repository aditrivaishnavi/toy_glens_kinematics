# Paper Framing: The Injection Realism Gap in CNN Lens Selection Functions

**Target:** MNRAS  
**Working Title:** "The injection realism gap in CNN strong lens selection functions: quantifying parametric source limitations with DESI Legacy Survey DR10"  
**Date:** 2026-02-14  
**Status:** Post-D03 reinvestigation, D04 matched comparison running

---

## 1. Central Thesis

CNN lens finders trained on real survey cutouts learn a "real-lens manifold" that naive
parametric forward-model injections do not reproduce. We quantify this mismatch directly
in CNN feature space and show it dominates selection-function inference unless injection
realism is validated and enforced.

---

## 2. Abstract (Draft)

We present a comprehensive analysis of the selection function for a CNN strong
gravitational lens finder applied to DESI Legacy Imaging Survey DR10. Our
EfficientNetV2-S classifier, trained on 451,681 cutouts (101x101 pixels, g/r/z bands),
achieves 89.3% recall (95% CI: [82.6%, 94.0%]) on 112 spectroscopically confirmed
(Tier-A) lenses held out from training, with zero spatial overlap between training and
validation sets.

Standard injection-recovery using parametric Sersic source profiles yields a marginal
completeness of [D04 baseline number pending]% over the full parameter space. We
demonstrate this gap between real-lens recall and injection completeness is dominated by
injection model limitations, not classifier performance. A linear probe in the CNN's
penultimate feature space achieves AUC = 0.991 +/- 0.010 separating parametric
injections from real lenses, establishing that injection morphology --- not model
capacity --- is the binding constraint.

We identify two specific contributors to the realism gap. First, adding physically
correct Poisson (shot) noise to injected arcs [D04 Poisson results pending], confirming
that source texture is a contributor. Second, restricting the injection geometry to
favorable lensing configurations (low impact parameter) roughly doubles detection at
moderate magnitudes, isolating the role of magnification geometry. A UMAP visualization
of the CNN's 1280-dimensional embedding space reveals that real lenses and parametric
injections occupy distinct, barely overlapping manifolds.

We provide the multi-dimensional completeness map C(theta_E, PSF, depth) as a
conservative lower bound on the true selection function and propose the linear probe AUC
as a quantitative realism gate for future injection pipelines. Our results demonstrate
that injection-based selection functions must be validated against the real-lens feature
distribution before being used for population-level inference.

---

## 3. Paper Outline

### 3.1 Introduction
- Importance of selection functions for strong lens population studies
- Standard approach: injection-recovery with parametric sources
- The open question: how realistic must injections be?
- Our contribution: first quantitative measurement of the sim-to-real gap in CNN feature space

### 3.2 Data and Model
- DESI Legacy Survey DR10 imaging (g/r/z, 0.262"/pixel)
- Training set: 451,681 cutouts (277 Tier-A + 3,079 Tier-B + 312,744 negatives + mirrors)
- EfficientNetV2-S architecture (pretrained ImageNet, finetuned)
- Preprocessing: raw_robust normalization (annulus median/MAD, clip [-10,+10])
- Train/val split: HEALPix-based spatial splitting (zero Tier-A overlap confirmed)

### 3.3 Injection Pipeline
- SIE+shear ray-tracing with Sersic source profiles
- Parameter ranges (theta_E, beta_frac, source magnitude, Sersic n, etc.)
- Poisson noise option (gain ~ 150 e-/nmgy for DR10 coadd)
- No correlated noise, no band-dependent PSF (limitation)

### 3.4 Results

#### 3.4.1 Real Lens Performance
- Tier-A recall: 89.3% [82.6%, 94.0%] at p > 0.3 (100/112 confirmed lenses)
- Tier-B recall: 72.0% [69.5%, 74.3%] (visual candidates, ~10% label noise)
- AUC: 0.9921

**Table: Tier-A recall at multiple thresholds with Wilson 95% CIs**

#### 3.4.2 Injection-Recovery Completeness
- Multi-dimensional grid: C(theta_E, PSF, depth)
- D04 matched comparison: baseline vs fixed-Poisson on identical grid parameters
- Completeness map as a conservative lower bound
- NOTE: Earlier D03 numbers (2.6%) were invalidated due to Poisson clamp bug and reporting bug

**Figure: Completeness heatmap (theta_E vs depth) at fixed PSF**

#### 3.4.3 The Sim-to-Real Gap
- Linear probe AUC = 0.991: CNN trivially separates injections from real lenses
- UMAP visualization: distinct manifolds for real vs injected
- Per-layer Frechet distance: separation emerges at mid-level features (texture/shape)
- Score gap: real Tier-A median score 0.995 vs low-bf injections 0.107

**Figure: UMAP embedding colored by category (real, low-bf injection, high-bf injection, negative)**
**Figure: UMAP embedding colored by CNN score**

#### 3.4.4 Diagnosing the Gap
- Poisson noise: D04 will provide correct numbers (D03 was invalidated by clamp bug)
- Clip-range analysis: bright arcs clipped at clip_range=10, clip=20 recovers morphology
- Beta_frac restriction: restricting to [0.1, 0.55] doubles detection at mag 21-22
- Combined Poisson + clip_range=20: D04 re-run with fixed Poisson implementation
- Band-dependent PSF: known limitation (single-band PSF used for all bands)

**Figure: Detection rate vs source magnitude for different injection configurations**

#### 3.4.5 Data Quality and Split Validation
- Zero spatial leakage for Tier-A (recomputed HEALPix analysis)
- PSF/depth balanced across train/val splits
- Annulus normalization characterization (Appendix)

### 3.5 Discussion
- Comparison with published CNN lens finders and injection strategies
- What the completeness lower bound means for population studies
- Pathway to improved injection realism (real galaxy stamps, correlated noise)
- Linear probe AUC as a proposed standard realism metric

### 3.6 Conclusion
- 89.3% recall on confirmed lenses demonstrates the CNN is an effective finder
- Injection-based completeness is a conservative lower bound due to source model fidelity
- Linear probe AUC = 0.991 is the first quantitative measurement of this gap
- Injection realism validation is mandatory before selection functions are trusted

---

## 4. Key Numbers for Paper

| Metric | Value | 95% CI | Note |
|--------|-------|--------|------|
| Tier-A recall (p>0.3) | 89.3% | [82.6%, 94.0%] | Headline number, 100/112 |
| Tier-A recall (p>0.5) | 83.9% | [76.3%, 89.8%] | |
| Tier-B recall (p>0.3) | 72.0% | [69.5%, 74.3%] | Visual candidates |
| AUC (gen4) | 0.9921 | | All positives |
| Linear probe AUC (real vs inj) | 0.991 | +/- 0.010 (CV) | Sim-to-real gap metric |
| Marginal completeness (no Poisson, D04) | **PENDING** | | Matched grid, depth 22.5-24.5 |
| Marginal completeness (Poisson, D04) | **PENDING** | | Fixed torch.poisson, matched grid |
| Peak combined (Poisson+clip20, D04) | **PENDING** | | Fixed Poisson + clip_range=20 |
| Poisson noise improvement (D04) | **PENDING** | | D03 numbers invalidated by clamp bug |
| Beta_frac restriction improvement | +15.5pp | | At mag 21-22, restricted vs unrestricted |
| Tier-A spatial leakage | 0 pixels | | Zero overlap train/val |
| Real Tier-A median score | 0.995 | | |
| Low-bf injection median score | 0.107 | | |

---

## 5. Key Figures

1. **UMAP embedding space** (two-panel: category-colored + score-colored)
2. **Detection rate vs source magnitude** (multiple injection configurations on one plot)
3. **Completeness heatmap** C(theta_E, depth) with and without Poisson noise
4. **Score distributions** for real Tier-A, injections, and negatives
5. **Per-layer Frechet distance** showing where separation emerges

---

## 6. Claims We Can Make

- The CNN achieves 89.3% recall on spectroscopically confirmed lenses with zero spatial leakage
- Injection-recovery completeness is a conservative lower bound, limited by source model fidelity
- Linear probe AUC = 0.991 is the first quantitative measurement of the injection realism gap for ground-based survey CNN lens finders
- Adding Poisson noise increases detection by up to +17.5pp, identifying source texture as a major contributor
- Restricting to favorable lensing geometry roughly doubles detection at moderate magnitudes

## 6.5 Known Limitations (to acknowledge explicitly)

- Band-dependent PSF: The injection pipeline uses r-band PSF FWHM for all three bands.
  Real observations have different PSF sizes in g, r, and z, creating band-dependent
  morphological variations that parametric injections do not reproduce. This is a
  known limitation shared by most published injection-recovery analyses.

## 7. Claims We Should NOT Make

- "The selection function completeness is 3.5%" as an unbiased estimate
- "The CNN misses 97% of lensed sources" (conflates injection gap with model performance)
- Any direct comparison of absolute completeness with other surveys without noting injection methodology differences
- "Retraining with corrected annulus would improve performance" (evidence does not support this)

---

## 8. Referee Preemption Strategy

### Criticism 1: "Completeness 3.5% is meaningless given probe AUC 0.991"
**Response:** We agree and explicitly present completeness as a lower bound. The linear probe result IS our contribution --- it quantifies why parametric injection completeness should not be trusted at face value.

### Criticism 2: "Only 112 Tier-A lenses; CI spans 17pp"
**Response:** Report Wilson CIs explicitly. Supplement with Tier-B metrics (with caveats). Note forthcoming spectroscopic campaigns (DESI, 4MOST) will expand the sample by an order of magnitude.

### Criticism 3: "Known preprocessing bug (annulus radii)"
**Response:** Appendix documents four diagnostic experiments characterizing the annulus effect. Median shifts by 0.15 normalized units; MAD unchanged (KS p=0.648). No PSF/depth correlation. Mismatched scoring: 3.6pp drop, 1.3 sigma, not significant. Conclude cosmetic for model performance.

### Criticism 4: "No independent holdout set"
**Response:** Tier-A train and val occupy 274 and 112 unique HEALPix pixels with zero overlap. The split is spatially disjoint. While both sets were identified through the same campaigns, the zero spatial overlap ensures the model has not seen similar sky regions during training.

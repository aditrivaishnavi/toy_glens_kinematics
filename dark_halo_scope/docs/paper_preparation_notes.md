# Paper Preparation Notes: Strong Lens Selection Function for DESI DR10

**Target Venues:** MNRAS, ApJ, AAS Journals  
**Working Title:** "Quantifying the Detectability of Strong Galaxy-Galaxy Lenses in Ground-Based Imaging: A Selection Function for DESI Legacy DR10"  
**Last Updated:** 2026-02-05

---

## 1. Core Scientific Contribution

### 1.1 Primary Goal

Quantify **which dark matter halos are actually visible as strong galaxy-galaxy lenses** in DESI Legacy Imaging DR10, given its real image quality, and how this observational "window" maps onto halo mass and redshift.

### 1.2 Key Outputs

1. **Selection function C(θ_E, z_l, source properties)** - Completeness as a function of Einstein radius, lens redshift, and source brightness/size
2. **Observability map** - Which (mass, redshift) combinations are detectable at DR10 depth/seeing
3. **Validated lens finder** - With rigorous shortcut-free validation

### 1.3 Novel Contributions

1. **COSMOS-based injection pipeline** - Using real HST galaxy morphologies instead of parametric Sérsic profiles
2. **Paired θ-aware validation framework** - Novel gates to detect and prevent shortcut learning
3. **Quantitative detectability thresholds** - arc_visibility_snr criterion for ground-based detection

---

## 2. Methodological Contributions (KEY FOR REVIEWERS)

### 2.1 The Shortcut Problem (Discovery)

**What we found:**
- A simple logistic regression on ONLY central 10 pixels achieved **AUC = 0.98**
- Positives were **64% brighter** in the core than controls
- Model learned "bright center = lens" instead of "arc morphology = lens"

**Why this matters:**
- Standard synthetic metrics (AUC on test set) can be dominated by photometric shortcuts
- High synthetic performance does NOT guarantee real-world transfer
- Most published lens-finder papers don't report these validation gates

**Key numbers to cite:**
```
Gate 1.6 (Core-Only Baseline):
  - Train AUC: 0.9909
  - Val AUC: 0.9800
  - Top feature: g_q25 (25th percentile of g-band core pixels)

Gate 1.8 (Core Brightness):
  - Controls core_mean_r: 0.0205
  - Positives core_mean_r: 0.0336
  - Ratio: 1.64x (64% brighter)
```

### 2.2 The Solution: Paired Controls + θ-Aware Gates

**Paired controls:**
- Same LRG cutout used for both positive (with injection) and control (without)
- Eliminates selection bias between classes
- Core brightness is identical by construction

**θ-aware validation gates:**
- Separate "lens-galaxy-only" region (r < θ_px - 1.5*PSF_px) from "arc-annulus" region
- Core-only AUC should be ≤ 0.55 (near random)
- Arc-annulus AUC should be > 0.7 (arcs are predictive)

**Key equation:**
```
Lens-galaxy-only mask: r < max(0, θ_E/pixscale - 1.5 * PSF_FWHM/pixscale)
Arc-annulus mask: |r - θ_E/pixscale| < 1.5 * PSF_FWHM/pixscale
```

### 2.3 Hard Negatives

**Why needed:**
- Paired controls eliminate shortcuts but don't teach contaminant rejection
- Real surveys contain spirals, ring galaxies, mergers that mimic arcs

**Composition:**
```
Training Data:
├── 40% Paired positives (LRG + injected arc)
├── 40% Paired controls (same LRG, no arc)
└── 20% Hard negatives (magnitude-matched spirals, rings, mergers)
```

**Source:** Galaxy Zoo DECaLS classifications (ring galaxies, mergers)

---

## 3. Data and Pipeline Details

### 3.1 Parent Sample

**Source:** DESI Legacy DR10 South sweeps  
**Selection:** v3_color_relaxed LRG cuts  
**Size:** ~145,000 unique LRG targets  
**Manifest:** ~12M rows across 133K bricks

### 3.2 Source Galaxy Library

**Source:** HST/COSMOS via GalSim RealGalaxyCatalog  
**Resolution:** 0.03"/pixel (HST) → 0.262"/pixel (DR10)  
**Morphology:** Real clumpy, star-forming galaxies (not parametric Sérsic)

**Key advantage over previous work:**
> "Unlike parametric Sérsic sources used in prior work (Jacobs et al. 2019, Petrillo et al. 2018), our COSMOS-based sources preserve the clumpy, irregular morphologies of real high-redshift galaxies, enabling more realistic arc structure."

### 3.3 Injection Pipeline

**Physics:**
- Lens model: SIE + external shear
- Ray-tracing: lenstronomy INTERPOL light model
- PSF: Moffat (β=3.5) matched to DR10 per-object PSF size
- Flux units: nanomaggies (consistent with DR10)

**Parameter grid:**
```
θ_E: [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5] arcsec
src_dmag: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] mag fainter than lens
src_reff: [0.05, 0.08, 0.12, 0.18, 0.25] arcsec
Total: 180 configurations (expanded from original 12)
```

### 3.4 Normalization

**Method:** Outer-annulus robust MAD normalization
```python
# Compute stats from outer region (r > 0.5 * stamp_size)
median = np.median(outer_pixels)
mad = np.median(np.abs(outer_pixels - median))
scale = 1.4826 * mad  # Convert MAD to sigma

# Normalize
normalized = (image - median) / scale

# Smooth compression (NOT hard clipping)
normalized = np.arcsinh(normalized / 10.0) * 10.0
```

**Why outer-annulus:**
- Prevents injection strength from leaking through normalization
- Outer region is background-dominated, not affected by arc

---

## 4. Key Metrics and Results (To Be Updated)

### 4.1 Shortcut Detection Gates

| Gate | Metric | Unpaired Result | Target (Paired) |
|------|--------|-----------------|-----------------|
| 1.5 | Clip fraction Cohen's d | 0.658 (FAIL) | < 0.1 |
| 1.6 | Core-only AUC | 0.98 (FAIL) | ≤ 0.55 |
| 1.7 | Arc-suppressed mean p | 0.62 (FAIL) | < 0.30 |
| 1.8 | Core brightness ratio | 1.64x (FAIL) | 0.95-1.05 |

### 4.2 Anchor Evaluation

**Tier-A (ground-visible, arc_visibility_snr > 2.0):**
- SDSSJ0029-0055 (SNR=3.51)
- SDSSJ0252+0039 (SNR=3.16)
- SDSSJ0959+0410 (SNR=3.91)
- SDSSJ0832+0404 (SNR=7.95)

**Tier-B (below DR10 detectability):**
- 11 additional SLACS/BELLS lenses with SNR < 2.0

**Key finding:**
> "Most SLACS/BELLS lenses were discovered via spectroscopy and HST follow-up; their arcs are often invisible in ground-based DR10 imaging. Only 4 of 15 (27%) pass our arc_visibility_snr > 2.0 threshold."

### 4.3 Selection Function (To Be Computed)

**Planned outputs:**
- C(θ_E) at fixed z_l, source properties
- C(z_l) at fixed θ_E, source properties
- 2D completeness map C(θ_E, z_l)
- Mass-redshift observability contours

---

## 5. Comparison to Prior Work

### 5.1 Jacobs et al. (2019) - DECaLS Lens Search

**Their approach:**
- Simulated lenses + real non-lenses/contaminants
- Explicitly warn about depth-driven bias

**Our improvement:**
- Paired controls eliminate photometric shortcuts
- θ-aware validation gates quantify shortcut risk

**Citation:** arXiv:1906.00970

### 5.2 Petrillo et al. (2018) - KiDS CNN

**Their approach:**
- Normalize by galaxy peak brightness
- Negatives from LRGs + contaminants (spirals)

**Our improvement:**
- Outer-annulus normalization (arc-invariant)
- COSMOS sources (realistic morphology)

**Citation:** arXiv:1807.04764

### 5.3 Collett (2015) - Selection Function

**Their approach:**
- Analytic selection function based on detectability arguments

**Our improvement:**
- Empirical selection function from injection-recovery
- Accounts for CNN detectability, not just theoretical visibility

**Citation:** ApJ 811, 20

### 5.4 Huang et al. (2020, 2021) - DR8/DR9 Visual Search

**Their approach:**
- Visual inspection of CNN candidates
- Published catalogs of lens candidates

**Relevance:**
- Use their confirmed lenses for Tier-A anchor expansion

**Citation:** arXiv:2005.04730

---

## 6. Figures to Prepare

### 6.1 Methodology Figures

1. **Pipeline schematic** - LRG selection → COSMOS injection → CNN training → Selection function
2. **Paired control illustration** - Same cutout with/without arc
3. **θ-aware mask diagram** - Showing lens-galaxy-only vs arc-annulus regions

### 6.2 Shortcut Discovery Figures

4. **Core brightness distributions** - Histogram of positives vs controls (1.64x ratio)
5. **Core-only AUC** - ROC curve for logistic regression on central pixels
6. **Arc-suppressed predictions** - Mean p_lens vs arc_snr for original vs suppressed

### 6.3 Results Figures

7. **Completeness vs θ_E** - C(θ_E) curves for different z_l
8. **Completeness vs arc_snr** - C(arc_snr) showing detection threshold
9. **Mass-redshift observability map** - 2D contour of C(M_halo, z_l)
10. **Anchor recovery** - Tier-A vs Tier-B performance

---

## 7. Tables to Prepare

### 7.1 Data Summary Tables

| Table | Content |
|-------|---------|
| 1 | LRG parent sample statistics |
| 2 | COSMOS source library properties |
| 3 | Injection parameter grid |
| 4 | Training data composition |

### 7.2 Results Tables

| Table | Content |
|-------|---------|
| 5 | Shortcut gate results (before/after fix) |
| 6 | Tier-A anchor list with arc_visibility_snr |
| 7 | Model performance metrics by θ_E bin |
| 8 | Selection function C(θ_E, z_l) grid |

---

## 8. Key Statements for Abstract/Introduction

### 8.1 Problem Statement

> "Strong gravitational lensing provides a powerful probe of dark matter structure, but the detectability of lenses in ground-based surveys depends critically on observational conditions. Previous lens-finder training has relied on synthetic metrics that can be dominated by photometric shortcuts, leading to overestimated completeness."

### 8.2 Our Contribution

> "We present a rigorous framework for training and validating strong lens finders that eliminates photometric shortcuts through paired controls and θ-aware validation gates. We apply this framework to DESI Legacy DR10, producing a selection function that quantifies which Einstein radii and lens redshifts are detectable at DR10 depth and seeing."

### 8.3 Key Finding

> "We find that standard unpaired synthetic training produces classifiers with 98% AUC on synthetic data but fails on real lenses due to photometric shortcuts. After implementing paired controls, our model achieves [X]% recall on ground-visible anchors while maintaining [Y] purity at [Z] false positive rate."

---

## 9. Reviewer Concerns to Address

### 9.1 "Why not use existing lens catalogs for training?"

**Response:** Existing catalogs are biased by discovery methods (spectroscopic pre-selection, HST follow-up). Training on them would bake in selection effects. Injection-recovery with controlled parameters enables unbiased completeness estimation.

### 9.2 "How do you know your COSMOS sources are representative?"

**Response:** COSMOS galaxies span the expected source population (z ~ 1-2, star-forming, r ~ 0.1-0.3"). We verify flux conservation and morphology preservation through Gate 1 (flux conservation test).

### 9.3 "Why trust CNN over visual inspection?"

**Response:** Visual inspection cannot be calibrated for selection function estimation. CNNs with paired validation provide quantifiable completeness with uncertainty bounds.

---

## 10. Timeline for Paper Writing

| Phase | Duration | Tasks |
|-------|----------|-------|
| Data fixes | 1 week | Paired controls, hard negatives, retrain |
| Validation | 3 days | θ-aware gates, anchor evaluation |
| Selection function | 1 week | Compute C(θ_E, z_l), uncertainty |
| Writing | 2 weeks | Draft, figures, tables |
| Internal review | 1 week | Co-author feedback |
| Submission | - | Target: 6 weeks from now |

---

## 11. Notes and Updates Log

### 2026-02-05: Shortcut Discovery

- Discovered core brightness mismatch (1.64x) between positives and controls
- Core-only AUC = 0.98 proves shortcut learning
- LLM review recommended paired controls + θ-aware gates
- Created comprehensive remediation plan

### 2026-02-05: Cutout Determinism Verified

- Tested 10 samples: re-fetched cutouts match stored stamps exactly (max_diff=0.0, corr=1.0)
- **Decision: SALVAGE** - Can generate paired controls by re-fetching same cutouts
- Cost saved: ~$500 EMR + 12 hours vs full regeneration

### 2026-02-05: Paired Controls Pilot Analysis

- Generated 1000 paired samples (positive + re-fetched control from same LRG)
- **Fixed r<8 mask analysis:** Core ratio = 1.67x (positives brighter)
- **Theta-aware mask analysis (large theta_e = 1.3-2.5"):**
  - Core (r < theta_e - 1.5*PSF): +8% injection flux
  - Arc annulus (|r - theta_e| < 1.5*PSF): +146% injection flux
- Arc is at Einstein radius, not center - injection is correct

### 2026-02-05: Critical Finding - Arc Overlap, Not LRG Bias

**Key discovery that changes remediation strategy:**
- In UNPAIRED data: 64% central brightness difference
- In PAIRED data: 67.1% central brightness difference (from injection alone!)
- **LRG selection bias: -3.2%** (essentially zero)

**Implications:**
1. The "shortcut" is NOT due to unpaired LRG selection bias
2. The "shortcut" is due to **physical arc overlap** with the central region
3. For ground-based surveys with PSF ~1.3", most arcs overlap the lens galaxy core
4. **Paired controls alone will NOT fix this** - need center degradation

**Solution:** Implement center blur/mask augmentation during training to force model to learn arc morphology in outer regions, not just "bright center = lens"

### 2026-02-04: Gen5 Training Complete

- Training completed with AUC = 0.9945 on synthetic test
- Real anchor recall only 4.4% - triggered investigation
- Identified sim-to-real gap in brightness distributions

### 2026-02-03: COSMOS Integration

- Completed COSMOS source bank (1000 galaxies)
- Verified flux conservation through lenstronomy INTERPOL
- Fixed surface brightness vs total flux unit confusion

---

*This document should be updated throughout the project. Add new findings, metrics, and insights as they emerge.*

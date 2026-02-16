# Prompt 11: Generate Full MNRAS Paper Draft

**Date:** 2026-02-14
**Request:** Generate a complete, submission-ready first draft of an MNRAS paper based on the verified experimental results from D01–D05. Include all raw tables. Figures will be generated separately — include figure placeholders with detailed captions describing what each figure should show.

---

## Context Summary

We have built a CNN (EfficientNetV2-S) strong gravitational lens finder for the DESI Legacy Imaging Survey DR10 (g/r/z bands, 101x101 pixel cutouts at 0.262"/pixel). The model achieves 89.3% recall on 112 spectroscopically confirmed (Tier-A) lenses.

We attempted to calibrate the model's selection function using standard injection-recovery with parametric Sersic source profiles (SIS+shear lensing). The injection completeness is only 3.41% over the full parameter space — an 86-percentage-point gap compared to real-lens recall.

**Our key finding:** This gap is morphological, not textural. We prove this through a controlled Poisson noise experiment: adding physically correct shot noise to injections DEGRADES detection (from 3.41% to 2.37%), falsifying the hypothesis that the gap is caused by missing noise texture. A gain sweep control experiment (gain=1e12, effectively zero Poisson noise) recovers the no-Poisson baseline exactly, proving the result is not a code bug.

A linear probe (logistic regression on CNN penultimate features) separates real lenses from injections with AUC = 0.996, confirming the CNN has learned to distinguish them in feature space.

**The paper's contribution:** (1) The first quantitative measurement of the injection realism gap in CNN feature space for ground-based surveys; (2) experimental falsification of the "missing texture" hypothesis via a controlled Poisson noise + gain sweep experiment; (3) the linear probe AUC as a proposed realism gate for injection pipelines; (4) the completeness map as a rigorously characterized conservative lower bound.

---

## Questions for You (In Addition to Generating the Draft)

### Q1: Journal strategy
Given our results and framing, which journal is the best fit? Consider:
- **MNRAS** — no page charges, Herle et al. (2024) published a similar (characterization of injection biases) paper there
- **A&A** — slightly higher impact factor (~5.4), HOLISMOKES lens-finder series published there
- **ApJ** — top prestige but ~$150-200/page charges
- **AJ** — strong home for methods/technical papers, slightly lower prestige

What would maximize acceptance probability for a methods-validation paper with this framing?

### Q2: Paper title
Suggest 2-3 title options. LLM2 (Prompt 9) proposed: "The morphological barrier in parametric injection-recovery for CNN strong lens finders: evidence from DESI Legacy Survey DR10"

### Q3: Does the paper stand without a real galaxy stamp (HUDF) pilot experiment?
A reasonable reviewer will ask: "You've shown Sersic injections are unrealistic. So what? Show me realistic ones." Both previous LLM reviewers suggested a HUDF pilot. We have chosen NOT to include it in this submission. Is that defensible? How should we frame future work to preempt this criticism?

### Q4: What is the single biggest weakness a referee will attack?
Be specific. Tell us the exact critique and your recommended defense.

---

## Verified Experimental Results (D05)

All numbers below are from the D05 verified re-evaluation (2026-02-14), independently reproducing D01–D04 with verified code. Seeds are fixed; results are reproducible.

### Tier-A Recall

| Threshold | Recall | n_detected / 112 | 95% Wilson CI |
|-----------|--------|-------------------|---------------|
| p > 0.3 | **89.3%** | 100/112 | [82.6%, 94.0%] |
| p > 0.5 | 83.9% | 94/112 | [76.3%, 89.8%] |
| FPR=1e-3 (p>0.806) | 79.5% | 89/112 | [71.3%, 86.1%] |
| FPR=1e-4 (p>0.995) | 48.2% | 54/112 | [39.1%, 57.4%] |

12 Tier-A lenses missed (score < 0.3). Characterization pending.

### Bright-Arc Detection Rates (p > 0.3)

All: theta_E=1.5", N=200 per mag bin, seed=42, beta_frac [0.1, 0.55] unless noted.

| Mag bin | Baseline | Poisson (g=150) | clip=20 | Poiss+clip20 | Unrestricted* | Gain=1e12 |
|---------|----------|-----------------|---------|--------------|--------------|-----------|
| 18–19 | 17.0% | 14.5% | 30.5% | 31.0% | 17.0% | **17.0%** |
| 19–20 | 24.5% | 18.0% | 32.0% | 26.5% | 21.5% | **24.5%** |
| 20–21 | 27.5% | 25.5% | 37.0% | 25.5% | 28.0% | **27.5%** |
| 21–22 | **35.5%** | 33.5% | **40.5%** | 24.0% | 20.0% | **35.5%** |
| 22–23 | 31.0% | 29.5% | 35.0% | 27.5% | 17.5% | **31.0%** |
| 23–24 | 24.0% | 17.5% | 14.5% | 8.5% | 7.0% | **24.0%** |
| 24–25 | 8.5% | 6.0% | 4.5% | 1.5% | 4.5% | **8.5%** |
| 25–26 | 1.0% | 1.0% | 0.0% | 0.0% | 0.0% | **1.0%** |

*Unrestricted uses beta_frac [0.1, 1.0].

Key: Gain=1e12 column matches Baseline exactly — proves Poisson code is correct; the degradation at gain=150 is real physics.

### Selection Function Grid (p > 0.3)

Grid: 11 theta_E x 7 PSF x 5 depth = 385 cells, 220 non-empty, 165 empty. Depth: 22.5–24.5 mag. 500 injections/cell. Seed: 1337.

| Condition | Marginal C | Detected | Total |
|-----------|-----------|----------|-------|
| No Poisson (baseline) | **3.41%** | 3,755 | 110,000 |
| Poisson (gain=150) | **2.37%** | 2,610 | 110,000 |

Completeness at all thresholds:

| Threshold | No-Poisson | Poisson | Deficit |
|-----------|-----------|---------|---------|
| p > 0.3 | 3.41% | 2.37% | -1.04pp |
| p > 0.5 | 2.75% | 1.80% | -0.95pp |
| p > 0.7 | 2.26% | 1.37% | -0.89pp |
| FPR=1e-3 | 1.98% | 1.18% | -0.80pp |
| FPR=1e-4 | 0.55% | 0.25% | -0.30pp |

Completeness by theta_E (no-Poisson, p>0.3):

| theta_E | C(p>0.3) | n_det/n_inj |
|---------|----------|-------------|
| 0.50" | 0.44% | 44/10,000 |
| 0.75" | 1.22% | 122/10,000 |
| 1.00" | 2.57% | 257/10,000 |
| 1.25" | 3.61% | 361/10,000 |
| 1.50" | 4.33% | 433/10,000 |
| 1.75" | 4.58% | 458/10,000 |
| 2.00" | 4.66% | 466/10,000 |
| 2.25" | 4.44% | 444/10,000 |
| 2.50" | 4.32% | 432/10,000 |
| 2.75" | 4.10% | 410/10,000 |
| 3.00" | 3.28% | 328/10,000 |

Completeness by lensed apparent magnitude (no-Poisson, p>0.3):

| Lensed mag | C(p>0.3) | n_det/n_inj |
|-----------|----------|-------------|
| 18–20 | 48.8% | 20/41 |
| 20–22 | 20.7% | 2,559/12,361 |
| 22–24 | 1.55% | 1,082/70,016 |
| 24–27 | 0.34% | 94/27,582 |

### Linear Probe

| Metric | Value |
|--------|-------|
| Probe AUC (real Tier-A vs low-bf injections) | **0.996 ± 0.004** (5-fold CV) |
| Frechet distance (features_0, early layers) | 0.21 |
| Frechet distance (features_3, mid layers) | 63.58 |
| Median score: real Tier-A | 0.995 |
| Median score: injections (low-bf, mag 19) | 0.110 |
| Median score: negatives | 1.5e-5 |

### Gain Sweep Result

At gain=1e12 (Poisson ON but noise negligible), all bright-arc detection rates match the no-Poisson baseline EXACTLY at every magnitude bin. This proves:
1. The Poisson code is correct (no bugs)
2. At gain=150, Poisson noise is physically large enough to degrade detection
3. Per-pixel analysis: a mag-21 arc spread over ~80 pixels has ~1.05 photoelectrons/pixel at gain=150; Poisson noise std equals the signal

---

## Injection Pipeline Details

### Lens Model
- Singular Isothermal Ellipsoid (SIE) + external shear
- theta_E: 0.5"–3.0" (grid); 1.5" (bright-arc tests)
- Shear: |gamma| ~ half-normal(sigma=0.05)
- Lens axis ratio: q_lens ~ U[0.5, 1.0]

### Source Model
- Sersic profile + optional clumps (60% probability, 1–4 clumps)
- r-band magnitude: 23–26 (grid); 18–26 (bright-arc tests)
- Sersic index: n ~ U[0.5, 4.0]
- Effective radius: R_e ~ U[0.05", 0.50"]
- Axis ratio: q_src ~ U[0.3, 1.0]
- Colors: g-r ~ N(0.2, 0.25), r-z ~ N(0.1, 0.25)

### Source Position
- beta_frac = beta/theta_E, area-weighted: beta_frac = sqrt(U[lo^2, hi^2])
- Default range: [0.1, 1.0]; restricted tests use [0.1, 0.55]

### Preprocessing
- Mode: raw_robust (outer-annulus median subtraction, MAD normalization)
- Annulus: r_in=20, r_out=32 pixels (known suboptimal for 101x101; does not affect conclusions)
- Clip range: [-10, +10] (default); [-20, +20] for clip_range=20 tests
- PSF: Gaussian per band, r-band FWHM from manifest, g scaled x1.05, z scaled x0.94

### Poisson Noise Implementation
```python
arc_electrons = injection.clamp(min=0.0) * gain_e_per_nmgy
noisy_electrons = torch.poisson(arc_electrons)
noise_electrons = noisy_electrons - arc_electrons
injection = injection + noise_electrons / gain_e_per_nmgy
```
- gain=150 e-/nmgy (approximate DR10 coadd)
- Zero-flux pixels: torch.poisson(0) = 0 (no noise injected into sky)

---

## Training Details

- Architecture: EfficientNetV2-S (pretrained ImageNet)
- Training: gen4 checkpoint — finetune from gen2 epoch 19 at LR=5e-5 for 60 epochs
- Loss: unweighted binary cross-entropy
- Dataset: 451,681 cutouts total
  - 277 Tier-A (spectroscopically confirmed) + 3,079 Tier-B (visual candidates) positives
  - ~135,000 negatives in training, ~135,000 in validation
  - Mirrored (horizontal + vertical flips) for augmentation
- Split: 70/30 train/val by HEALPix, zero Tier-A spatial overlap
- Best validation AUC: 0.9921

---

## Comparison with Published Results

| Paper | Survey | Method | Completeness / Recall | Injection source | Notes |
|-------|--------|--------|----------------------|-----------------|-------|
| This work | DESI DR10 | EfficientNetV2-S | 89.3% Tier-A recall; 3.41% injection completeness | Sersic + clumps | Linear probe AUC 0.996 proves gap is morphological |
| Herle et al. (2024) | Euclid sim | ResNet-like | Selection bias characterized | Sersic (parametric) | No real lenses; bias by n, R_e documented |
| HOLISMOKES XI (2024) | HSC | CNN ensemble | TPR_0 = 10-40% on 189 confirmed | **Real HUDF stamps** | Explicitly notes Sersic inadequacy |
| Huang et al. (2020) | DECaLS | ResNet | Catalog published, no formal completeness | Sersic | No injection-recovery completeness map |
| Jacobs et al. (2019) | DES | CNN | ~50% for bright arcs (brief test) | Sersic | No multi-dimensional completeness |

---

## Instructions for the Draft

1. **Write the full paper text** in MNRAS style (LaTeX-compatible markdown). Include all sections from Introduction through Conclusion + one Appendix on the annulus characterization.

2. **Include all tables as raw formatted tables** with proper captions, referencing the exact numbers above. Do not round or modify any number.

3. **For figures, include placeholders** with detailed captions specifying:
   - What data to plot
   - Axis labels and ranges
   - Which data columns/files to use
   - Color scheme and legend

4. **Frame the paper around the "falsification ladder"** narrative:
   - Establish the gap (probe AUC, score distributions)
   - Hypothesize texture is the cause
   - Falsify with Poisson experiment
   - Validate with gain sweep control
   - Conclude the barrier is morphological
   - Provide the selection function as a conservative lower bound with the diagnostic framework

5. **Address the reviewer concern head-on:** "How do you know your injections are realistic?" Our answer: we don't claim they are. We're the first to rigorously show they aren't, quantify the gap, diagnose its cause, and provide a framework (linear probe) for the community to measure it themselves.

6. **Include a "Comparison with published work" subsection** in the Discussion, referencing the table above.

7. **Future work section** should be concrete and specific: real galaxy templates (HUDF/GOODS-CANDELS), correlated noise, band-dependent PSF. Frame as "the natural next step that our diagnostic framework enables."

---

## Attached Files

### All D05 Results
- `results/D05_20260214_full_reeval/ba_baseline/bright_arc_results_bf0.10_0.55.json`
- `results/D05_20260214_full_reeval/ba_poisson_fixed/bright_arc_results_bf0.10_0.55.json`
- `results/D05_20260214_full_reeval/ba_clip20/bright_arc_results_bf0.10_0.55.json`
- `results/D05_20260214_full_reeval/ba_poisson_clip20/bright_arc_results_bf0.10_0.55.json`
- `results/D05_20260214_full_reeval/ba_unrestricted/bright_arc_results_bf0.10_1.00.json`
- `results/D05_20260214_full_reeval/ba_gain_1e12/bright_arc_results_bf0.10_0.55.json`
- `results/D05_20260214_full_reeval/grid_no_poisson/selection_function.csv`
- `results/D05_20260214_full_reeval/grid_no_poisson/selection_function_meta.json`
- `results/D05_20260214_full_reeval/grid_poisson_fixed/selection_function.csv`
- `results/D05_20260214_full_reeval/grid_poisson_fixed/selection_function_meta.json`
- `results/D05_20260214_full_reeval/linear_probe/feature_space_results.json`
- `results/D05_20260214_full_reeval/tier_a_scoring/real_lens_scoring_results.json`
- `results/D05_20260214_full_reeval/d05_log.txt`

### Core Code (for methodology description)
- `dhs/injection_engine.py` — injection pipeline (Poisson lines 549–567)
- `dhs/preprocess.py` — preprocessing (annulus normalization)
- `dhs/data.py` — data loading
- `dhs/utils.py` — model loading, preprocessing spec
- `scripts/selection_function_grid.py` — grid runner
- `scripts/feature_space_analysis.py` — linear probe and embedding analysis
- `sim_to_real_validations/bright_arc_injection_test.py` — bright-arc tests
- `sim_to_real_validations/real_lens_scoring.py` — Tier-A scoring

### Configuration
- `configs/paperIV_efficientnet_v2_s_v4_finetune.yaml` — training config
- `configs/injection_priors.yaml` — injection parameter ranges (single source of truth)

### Documentation
- `EXPERIMENT_REGISTRY.md` — full experiment history D01–D05
- `docs/PAPER_FRAMING_SIM_TO_REAL_GAP.md` — paper outline and framing notes
- `docs/MNRAS_SUPPLEMENTARY_TRAINING_LOG.md` — training and diagnostic history
- `docs/APPENDIX_ANNULUS_CHARACTERIZATION.md` — annulus diagnostic results

### Previous LLM Reviews (for context on reviewer concerns)
- `llm_review_package/PROMPT_9_RESCUE_THIS_PAPER.md`
- `llm_review_package/PROMPT_10_VERIFIED_REEVAL.md`

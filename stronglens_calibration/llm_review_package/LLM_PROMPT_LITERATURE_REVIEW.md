# LLM Prompt: Rigorous Literature Review for Injection-Recovery Selection Function

**Date:** 2026-02-12
**Context:** MNRAS paper — "Calibrating CNN-Based Strong Gravitational Lens Finders in DESI Legacy Survey DR10"

---

## Why we are writing this prompt

In a previous prompt, we asked you to review the following papers and compare their injection-recovery and selection function approaches to ours. Your response was shallow — you mentioned "challenge datasets" generically and cited Huang et al. in passing, but **did not actually review any of the specific papers** we listed. You did not compare injection models, source populations, validation approaches, flux calibration methods, detection thresholds, or completeness estimation procedures across papers.

We need this done properly. Our injection-recovery selection function is the **core novelty claim** of the paper. Positioning it in the literature is not optional — it's how we defend the contribution against referees. We need to know, for each paper:

1. What they actually did for injection/selection function calibration
2. How their approach differs from ours
3. What they did better or worse
4. What a referee who has read those papers would expect from us

---

## Papers to review (read each one carefully)

Please read and review each of the following papers. For each, answer the structured questions below.

### Required papers

| # | Citation | arXiv / DOI | Key reason we need this reviewed |
|---|----------|-------------|----------------------------------|
| 1 | Collett 2015, "The population of galaxy–galaxy strong lenses in forthcoming optical imaging surveys" | 1507.02657 | Theoretical strong lens population model. Predicts lens counts as a function of survey depth. We may use this for a population cross-check (convolving our C(θ_E,...) with their predicted population). |
| 2 | Collett & Auger 2014, "Cosmological constraints from strong gravitational lensing in clusters of galaxies" | (check ref) | Selection function methodology for lensing surveys. May not be the right paper — please verify and find the correct Collett paper on selection functions if this is wrong. |
| 3 | Jacobs et al. 2019, "Finding strong lenses in CFHTLS using convolutional neural networks" | 1811.03786 | CNN lens finder with injection tests. One of the earliest to inject simulated lenses into real survey images for completeness estimation. |
| 4 | Metcalf et al. 2019, "The strong gravitational lens finding challenge" | 1802.03609 | Bologna Lens Challenge. Standardized injection-recovery benchmark for lens finders. Establishes community practice for how injection realism is evaluated. |
| 5 | Huang et al. 2020, "Discovering New Strong Gravitational Lenses in the DESI Legacy Imaging Surveys" | 2005.04730 | First DESI lens-finder paper. How did they characterize completeness? |
| 6 | Huang et al. 2021, "New Strong Gravitational Lenses from the DESI Legacy Imaging Surveys Data Release 9" | 2206.02764 | DR9 extension. Any selection function improvements? |
| 7 | Rojas et al. 2022, "Strong lens systems search in the Dark Energy Survey using Convolutional Neural Networks" | (find arXiv) | DES lens search with selection function. How did they handle injection? |
| 8 | Cañameras et al. 2021, "HOLISMOKES – VI. Candidate multiply lensed quasars selected using a neural network" | (find arXiv) | Systematic lens search with quantified completeness. |
| 9 | Cañameras et al. 2024 or latest HOLISMOKES paper | (find latest) | Updated pipeline. Has the completeness methodology improved? |
| 10 | Inchausti et al. 2025, "Strong Lens Discoveries in DESI Legacy Imaging Surveys DR10 with Two Deep Learning Architectures" | 2508.20087 | Paper IV — our primary comparison. Do they report ANY form of injection-recovery completeness? If not, that confirms our unique contribution. |
| 11 | Huang et al. 2025 (if different from #10) | Check if 2508.20087 is Huang or Inchausti lead | Clarify authorship and whether there are multiple DR10 papers. |
| 12 | Sonnenfeld 2022 or any recent paper on strong lens selection functions specifically | (find) | Any dedicated selection function paper we're missing? |
| 13 | Gavazzi et al. 2014 or Marshall et al. 2009 | (find) | Older selection function work for comparison. |

If any of these citations are incorrect or you cannot find the paper, say so explicitly and suggest the correct reference.

---

## For EACH paper, answer these structured questions

### A. Injection / simulation approach

1. **Did this paper perform injection-recovery?** (Yes/No/Partial)
2. **What lens model did they use for injection?** (SIS, SIE, NFW, ray-traced from N-body, other?)
3. **What source model did they use?** (Point source, Sersic, real galaxy images from HST, GalSim, other?)
4. **How did they handle flux calibration?** (AB magnitudes, survey-specific zeropoint, arbitrary units?)
5. **How did they handle PSF?** (Per-image PSF model, Gaussian with survey FWHM, Moffat, no PSF?)
6. **How did they handle noise?** (No added noise, Gaussian noise from depth, empirical noise, full forward model?)
7. **How did they handle magnification?** Specifically: did they normalize source flux as unlensed or lensed? Did magnification emerge from ray-tracing or was it applied as a multiplicative factor?
8. **What source offset / impact parameter distribution did they use?** (Uniform, area-weighted, physical caustic crossing rate, other?)
9. **Did they inject into real survey images?** Or simulated backgrounds?
10. **What morphological diversity was covered?** (Arcs only, doubles, quads, rings, extended+compact?)

### B. Selection function methodology

11. **How did they define "detection"?** (Fixed probability threshold, fixed FPR, visual inspection, other?)
12. **Did they report completeness as a function of what parameters?** (θ_E, source mag, PSF, depth, magnification, source size, other?)
13. **How did they estimate uncertainty on completeness?** (Binomial CI, bootstrap, Bayesian, none?)
14. **Did they report a multi-model comparison** of selection functions? (If ensemble, did they show per-model vs ensemble completeness?)
15. **Did they perform a sensitivity analysis?** (Varying injection parameters to show robustness?)

### C. Validation

16. **How did they validate that injections were realistic?** (SNR comparison with real lenses, visual panels, distributional tests, none?)
17. **Did they compare injection-based completeness to real-lens recall?** (Matched known-lens recovery?)
18. **Did they perform a population-level cross-check?** (Predicting observed counts from selection function × population model?)

### D. What can we learn?

19. **What did they do that we should replicate or cite?**
20. **What limitation do they acknowledge that applies to us too?**
21. **What would a referee who knows this paper expect to see in ours?**

---

## Additional questions given our current situation

We have now built an injection engine (SIS + external shear, Sersic source, nanomaggy flux, FFT Gaussian PSF, per-host conditioning on psfsize_r and psfdepth_r). During review, we identified several issues. Please address these in light of the literature:

### Magnification and flux normalization

22. **How should flux normalization work in injection-recovery?** We discovered that our injection engine normalizes the Sersic profile by its IMAGE-PLANE integral, then scales to the specified "unlensed flux." This means total_image_flux = flux_unlensed, and the magnification amplification is completely lost. The arcs are 5-30× too faint. What is the correct procedure? How do published injection pipelines handle this?

23. **Should `flux_nmgy_r` represent unlensed source flux or total lensed (observed) flux?** If unlensed: we need to fix the normalization to let magnification amplify the total flux naturally. If lensed: the r_mag range (23-26) needs to be reconsidered for DR10 detectability. What convention do published papers use?

### Lens model choice

24. **Is SIS + shear sufficient, or do we need SIE?** Our code uses SIS (spherical), not SIE (elliptical). The LLM recommended SIE but delivered SIS. What do published selection function papers use? Is SIS defensible if stated as a limitation? What is the expected bias of SIS-only completeness vs SIE-based completeness?

### Detection threshold

25. **Should we use fixed-FPR or fixed-probability thresholds?** The LLM recommended fixed-FPR as the primary operating point. What do published papers actually use? How do we implement this: score the validation negatives, find the threshold at FPR=1e-3, then apply to injections?

### Source population

26. **What source offset distribution is standard?** We sample `beta_frac = uniform(0.1, 1.0)` giving `beta = beta_frac * theta_E`. For an area-weighted prior, P(β) ∝ β, so we should sample `beta_frac = sqrt(uniform(0.01, 1.0))`. What do published papers use? Does this choice significantly affect completeness?

27. **What r-band magnitude range is appropriate for injected sources?** For unlensed source magnitudes, what range produces detectable lensed arcs in DR10 after magnification? For lensed (observed) arc magnitudes, what range spans the detection boundary?

### Minimum viable vs aspirational

28. **Given that Paper IV does NOT report injection-recovery completeness** (confirmed by our review of 2508.20087): what is the minimum we need to publish a credible selection function that clearly differentiates our paper? Rank the following in priority order:
    - a) Correct flux normalization (magnification)
    - b) SIE instead of SIS
    - c) Fixed-FPR thresholds
    - d) Sensitivity analysis
    - e) Matched known-lens recovery cross-check
    - f) Population count prediction
    - g) Multi-model comparison (ResNet vs EfficientNet vs ensemble)
    - h) Source magnitude as explicit grid axis

29. **What is an honest 1-paragraph "Limitations" statement** we can write for the selection function section, acknowledging SIS-only, Gaussian PSF, no SED templates, etc., that a referee would find acceptable?

30. **Are there any recent (2023-2025) papers on CNN lens-finder selection functions** that we're missing entirely? The field is moving fast. Any preprints on arXiv that supersede older approaches?

---

## What we expect back

1. **A paper-by-paper structured review** answering questions A-D for each of the ~13 papers
2. **A comparison table** summarizing injection model, source model, PSF treatment, noise treatment, magnification handling, detection definition, and validation approach across all papers
3. **Direct answers to questions 22-30** with citations to specific papers where relevant
4. **A priority-ordered action plan** for what to fix first in our injection pipeline, informed by what the literature considers standard practice
5. **An honest assessment** of where our approach sits relative to the field: are we above, at, or below the bar for MNRAS?

**Be specific. Cite page numbers, section numbers, and figure numbers. Do not give vague summaries. We will use your review to write the Related Work section of the paper and to defend our methodology against referees.**

# LLM Prompt: Injection-Recovery Pipeline for Selection Function Calibration

**Date:** 2026-02-11
**Context:** MNRAS paper — "Calibrating CNN-Based Strong Gravitational Lens Finders in DESI Legacy Survey DR10"
**Attached:** `llm_review_injection_pipeline.zip` (full pipeline codebase)

---

## Who we are

We are building a CNN-based strong gravitational lens finder for DESI Legacy Survey DR10 images. The paper aims for **Level 2 Comparability** with Inchausti et al. (2025, "Paper IV"). Our **main original contribution** is a rigorous, quantitative selection function — i.e., measuring the detection completeness C(θ_E, PSF, depth, source_mag) of the classifier by injecting synthetic lensed arcs into real DR10 galaxy cutouts and measuring recovery rates.

This selection function is **the core novelty claim of the paper** and must be publishable in MNRAS or equivalent journals. Everything else (training, evaluation, ensemble) follows Paper IV closely. The injection-recovery pipeline is where we differentiate ourselves.

---

## The problem: We have wasted enormous time and need working code

We have attempted to build this injection pipeline **multiple times across 4+ generations** (Gen5, Gen6/7/8, Plan B, Plan C) over several months, and every attempt has been derailed by cascading failures. We are providing the full history below so you understand exactly what went wrong and can avoid repeating these mistakes. **We need you to provide complete, working, correct code** for the injection-recovery pipeline that addresses all of these issues.

---

## Historical failures (read carefully — these are real failures we encountered)

### Failure 1: Simulated arcs 100× too bright (Simulation Crisis)

In our Gen5/Plan B pipeline, injected arcs had SNR 30–80, while real DR10 arcs are at or near the noise level (SNR ~ 0–3). Root cause: arc flux was set as `rng.uniform(100, 500)` in arbitrary ADU units, not tied to AB magnitudes, survey zeropoints, or real source luminosities.

**Impact:** Model trained on hyper-bright arcs learned to detect obvious features that don't exist in real data. Synthetic AUC was 0.99+, but real-lens recall was 4.4%.

### Failure 2: Core brightness shortcut (Physical, not a bug)

The PSF-convolved inner (counter) image of the lens adds ~10–40% flux to the central pixels. A logistic regression on the central 10×10 pixels alone reached AUC ≈ 0.95–0.98. The model learned "brighter core = lens" instead of "arc morphology = lens."

**Impact:** Four separate shortcut gates (1.5–1.8) all failed. Residual preprocessing and core dropout were needed to mitigate.

### Failure 3: Surface brightness unit errors (lenstronomy)

lenstronomy expects flux in units of flux/arcsec², but we passed flux/pixel. This caused arcs to be ~1000× too faint or too bright depending on the direction of the error.

**Impact:** Multiple reruns of data generation.

### Failure 4: PSF kernel size exceeding stamp size

For large PSF FWHM values, the convolution kernel exceeded the 64×64 stamp size, causing silent failures.

### Failure 5: Anchor/real-lens evaluation mismatch

Training LRGs were ~44× brighter in central aperture than SLACS/BELLS anchor lenses. SLACS/BELLS are spectroscopically discovered and many arcs aren't visible in DR10 ground-based imaging at all. We were evaluating against the wrong benchmark.

### Failure 6: Source morphology mismatch

Source effective radius was 3–8 pixels (0.8–2.1 arcsec), while real lensed sources are ~0.1–0.5 arcsec. This produces unrealistically extended arcs.

### Failure 7: Worker sharding duplication

DataLoader workers all saw the same data fragments, causing ~8× sample duplication and rapid overfitting.

### Failure 8: Code bugs in generated pipelines

- Duplicate function definitions in 3000+ line files (second `render_cosmos_lensed_source` silently overrode the first)
- `to_surface_brightness` imported but never defined
- `max_kernel_size` used but never set
- `__init__.py` importing non-existent classes
- `_parse_s3` regex failing on `s3a://` URLs

### Failure 9: NaN/Inf propagation

~0.08% of cutouts had NaN pixels, which propagated through training loss and caused NaN gradients. No NaN guards existed.

### Failure 10: Wrong train/val/test split proportions

Split was inverted (26/39/35 instead of 70/15/15). Caught late.

### Failure 11: Inner image dominates core flux

For θ_E < 0.75 arcsec, ~40% of total flux lands in the core from the inner image alone (91.8% of core flux). Even at larger θ_E, inner image contributes 7.7% of core flux (vs 0.4% expected from PSF alone). This is physical and cannot be avoided — it must be modeled correctly.

### Failure 12: Arc SNR distribution biased toward easy cases

The injection parameter grid had most samples at high SNR; very few near the detection threshold (SNR 0.8–2 range). This means the selection function was poorly constrained exactly where it matters most.

---

## What we have now (attached code)

The attached zip contains our **current working pipeline** (training, evaluation, meta-learner, negative cleaning, selection function scaffolding). The key file is:

- **`scripts/selection_function_grid.py`** — Contains a **minimal proxy** arc renderer: a Gaussian ring at θ_E modulated by angular segments, with clumpy noise. This is explicitly marked as "NOT publication quality" and must be replaced.

The minimal proxy renderer:
1. Draws a thin radial Gaussian ring at the Einstein radius
2. Modulates it with 1–3 angular segments
3. Adds sub-structure via smoothed Gaussian noise
4. Scales to physical flux using AB magnitudes and nanomaggies
5. Applies approximate PSF convolution
6. Adds to host galaxy cutout

This is adequate for scaffolding and pipeline testing, but **lacks the physical realism required for a refereed journal paper**.

---

## What we need from you

### Part A: Strategy and principles

1. **What is the best sim-to-real implementation approach** for injection-recovery in DR10? Consider:
   - We are injecting into **real DR10 galaxy cutouts** (101×101, g/r/z, nanomaggies, 0.262 arcsec/pixel)
   - The goal is measuring **detection completeness**, not training the model
   - The model is already trained on real lenses (Paper IV parity approach)
   - We need arcs that are photometrically calibrated to DR10 conditions

2. **What should our north star principles be?** What invariants must hold for the injection to be scientifically defensible? For example:
   - Flux conservation
   - PSF matching to per-cutout DR10 conditions
   - Noise model matching
   - Source population realism
   - Inner image handling
   - What else?

3. **What is the right objective function** for measuring how close our injections are to real lensed systems? How do we quantify the sim-to-real gap for the injection specifically (not for training data)?

4. **What are the validation gates and metrics** that MNRAS reviewers would expect for a selection function paper? What must we demonstrate to make the injection calibration acceptable? Consider:
   - Photometric consistency checks
   - Comparison with known lenses
   - Arc SNR distribution realism
   - Parameter coverage
   - Statistical rigor of completeness estimates

5. **What goal should we aim for?** Given that:
   - We are NOT using injections for training (model is trained on real lenses)
   - We ARE using injections only for the selection function (measuring completeness)
   - The injections need to be "realistic enough" to probe the model's detection boundary
   - Perfect realism is not required — what IS required?

### Part B: Literature cross-check

6. **Cross-check our approach against related calibration papers.** Specifically:
   - Collett & Auger (2014) — selection function for lens surveys
   - Jacobs et al. (2019) — CNN lens finder with injection tests
   - Metcalf et al. (2019) — strong lens finding challenge
   - Huang et al. (2020, 2021) — DECaLS lens finding
   - Rojas et al. (2022) — selection function for lens searches
   - Cañameras et al. (2021, 2024) — systematic lens finding surveys
   - Inchausti et al. (2025) — Paper IV, our primary comparison
   - Huang et al. (2025, arXiv:2508.20087) — latest DR10 lens finding
   - Any other relevant papers on injection-recovery for lens finding selection functions

7. **Is our approach (real-image trained model + injection-recovery selection function) research-journal-worthy?** What would make it a strong vs. weak contribution? What are the potential reviewer objections and how do we pre-empt them?

### Part C: Complete working code

8. **Provide the full, corrected, updated code** for the injection-recovery pipeline. This must include:

   a. **`injection_engine.py`** — The core injection module:
      - SIE lens model (or simpler if justified — explain your choice)
      - Realistic source population (Sérsic profiles with physically motivated parameters)
      - Proper flux calibration in nanomaggies (DR10 zeropoint = 22.5 AB mag)
      - Per-cutout PSF handling (using `psfsize_r` from the manifest, or explain alternative)
      - Proper inner image handling (document how it affects the core)
      - Multi-band (g, r, z) color model for lensed sources
      - Noise-aware injection (adding Poisson/Gaussian noise consistent with DR10 depth)
      - NaN/Inf guards throughout

   b. **`selection_function_grid.py`** (updated) — The grid runner:
      - Uses the new injection engine
      - Grid over: θ_E (0.5–3.0 arcsec), PSF FWHM (0.9–1.8 arcsec), 5σ depth (22.5–24.5 mag)
      - Also varies: source magnitude, magnification
      - Bayesian binomial CIs for completeness per cell
      - Exports results as CSV + JSON metadata
      - Proper logging and progress reporting

   c. **`validate_injections.py`** — Injection validation script:
      - Photometric consistency checks (flux conservation before/after)
      - Arc SNR distribution analysis
      - Comparison with real anchor lenses (visual and statistical)
      - Parameter distribution coverage check
      - Any other validation you recommend

   d. **Any supporting modules** needed (utilities, constants, etc.)

9. **Explain every fix and design choice.** For each component, explain:
   - What was wrong in our historical attempts
   - What approach you took and why
   - What assumptions you made
   - What limitations remain
   - What a reviewer might question

### Part D: Critical clarifications we need you to address

These are questions we realized we should have asked earlier. Please address each explicitly.

#### Data reality (what we actually have and don't have)

10. **Inverse-variance planes**: Our `.npz` cutouts store only the image array (`"cutout"` key, shape `(101, 101, 3)` HWC, nanomaggies). We do **NOT** have per-pixel inverse-variance (invvar) maps stored alongside the cutouts. We DO have `psfdepth_r` in the manifest (inverse-variance for point sources, nanomaggies^-2). How should we model per-pixel noise for injection without invvar planes? Is `psfdepth_r` sufficient to approximate the noise? What is the correct formula?

11. **Band-specific PSF**: Our manifest contains `psfsize_r` (r-band PSF FWHM in arcsec) but NOT `psfsize_g` or `psfsize_z`. The PSF differs across bands (g is typically broader than r, z can be broader or narrower depending on conditions). How should we handle multi-band PSF convolution with only r-band PSF size? What are typical g/r and z/r PSF ratios for DECam?

12. **PSF shape model**: `psfsize_r` is a scalar FWHM. What functional form should we assume? Gaussian, Moffat (and what β?), or something else? DR10 uses per-brick spatially varying PSF models, but we only have the scalar summary. Is Gaussian adequate for injection-recovery, or does the Moffat wing structure matter for arc detection?

13. **Correlated noise in coadds**: DR10 images are coadded from multiple single-epoch exposures with Lanczos resampling, which introduces pixel-to-pixel noise correlations. This means the effective noise is NOT simply `sigma = 1/sqrt(psfdepth_r)` per independent pixel. How should we handle this? Does it significantly affect arc SNR calculations? What correction factor is appropriate?

#### Physics (image morphology coverage)

14. **Multiple image morphologies**: Our current renderer only produces arc-like features (partial Einstein rings). But real strong lensing produces a variety of image configurations:
    - **Doubles** (most common at small θ_E, especially for point-like sources)
    - **Quads/crosses** (for elliptical lenses with source near caustic)
    - **Complete Einstein rings** (source exactly behind lens)
    - **Partial arcs** (extended source, large θ_E)
    
    For the selection function, should we cover all morphologies? Or is it acceptable to focus on arcs/rings (which is what a CNN trained on DR10 imaging would most likely detect)? How do published selection function papers handle this?

15. **Source color/SED model**: Currently we assign random g/r/z color ratios. For a refereed paper, do we need a physically motivated color model? Options:
    - Draw source redshifts from a distribution and assign SEDs (e.g., star-forming blue galaxy templates)?
    - Use empirical color distributions of known lensed sources?
    - Show that the selection function is insensitive to color choice (and therefore random is fine)?
    
    What approach do published papers take? We suspect color has a small effect on detection (morphology dominates), but we need to justify this.

16. **Magnification distribution**: We currently sample magnification uniformly in [5, 30]. The physical magnification distribution for a SIS depends on source position relative to the caustic and follows a steep power law (most systems are near μ ~ 2–5, few are highly magnified). Should we use the physical distribution, or is uniform acceptable for a grid that explicitly conditions on θ_E? Is magnification even the right parameter, or should we parameterize by source position offset from the caustic?

17. **Lens ellipticity and shear**: SIS produces azimuthally symmetric arcs. Real lenses have ellipticity (ε ~ 0.2–0.5) and external shear (γ ~ 0.01–0.1). SIE + shear produces doubles, quads, and asymmetric arcs. How important is this for the selection function? Does a SIS-only selection function have a known bias?

#### Methodology (statistical rigor)

18. **Detection threshold dependence**: Our selection function C(θ_E, PSF, depth) currently uses a fixed threshold of p > 0.5. Should we instead:
    - Report completeness at multiple thresholds (0.3, 0.5, 0.7)?
    - Report at a fixed false-positive rate (e.g., FPR = 1%)?
    - Report the full completeness-vs-threshold curve per cell?
    - Use a threshold-independent metric?
    
    What do published papers use? What would MNRAS reviewers prefer?

19. **Statistical power**: With 200 injections per cell, the binomial standard error at 50% completeness is ~3.5% (95% CI width ≈ ±7%). Is this precision sufficient for the claims we want to make? Should we do a formal power analysis to determine the minimum injections/cell needed? Some cells may have very few host galaxies — what is the minimum for a credible completeness estimate?

20. **Source property marginalization**: Within each (θ_E, PSF, depth) cell, we randomly draw source_mag and μ from a prior. This means C(θ_E, PSF, depth) is an average over the source population prior. Should we:
    - Make source_mag an explicit grid axis (making it a 4D grid)?
    - Document the assumed prior and show a sensitivity analysis?
    - Report both the marginalized and conditional completeness?

21. **Multi-model selection function**: Should we compute the selection function separately for:
    - ResNet-18
    - EfficientNetV2-S
    - The meta-learner ensemble
    - Simple average ensemble
    
    And show how the detection boundary differs between models? This could be a strong result: "the ensemble recovers X% more lenses at small θ_E than either individual model."

22. **Does Paper IV report a selection function?** If Inchausti et al. (2025) do NOT report injection-recovery completeness, this is our clear unique contribution. If they do, we need to compare and show improvement. Please check and advise on positioning.

#### Validation (what convinces a referee)

23. **Visual validation**: Should we produce side-by-side figure panels showing:
    - Real DR10 confirmed lenses (from our positive catalog)
    - Injections at matched θ_E, brightness, and host properties
    - And demonstrate they are qualitatively indistinguishable?
    
    Is this standard practice? How many examples should we show?

24. **Known-lens recovery cross-check**: The gold-standard calibration is: inject arcs with properties **matched to known real lenses** (Tier-A anchors with measured θ_E, source brightness) into host galaxies at matched PSF/depth, then verify that the model's recovery rate on these injections matches its actual recall on the real lenses. If they match, the injection is calibrated. If they don't, the injection is biased. Should we implement this check? How precisely can we expect agreement?

25. **Sensitivity analysis**: If we perturb injection parameters (source size ±30%, PSF ±10%, color ±0.2 mag, Sérsic index ±0.5), how much should the completeness change? If it's very sensitive to uncertain parameters, the selection function is unreliable. Should we include a formal sensitivity/robustness analysis? What perturbations should we test?

26. **Population prediction cross-check**: Can we use the selection function to predict the expected number of detectable lenses in DR10 (by convolving C(θ_E, ...) with a theoretical lens population model from, e.g., Collett 2015), and compare with observed lens counts? If the prediction is within a factor of ~2, that validates the selection function at the population level.

#### Paper integration

27. **What figures and tables should this produce?** Please recommend the specific figures for the MNRAS paper, e.g.:
    - 2D completeness heatmap C(θ_E, depth) marginalized over PSF
    - Completeness vs θ_E at different depth bins
    - Example injection panels (input host → injected → model score)
    - Sensitivity analysis results
    - Comparison with real-lens recall
    
    What would be a typical set for a lens-finding selection function paper?

28. **Narrative framing**: What should the selection function REVEAL that goes beyond Paper IV? What is the scientific story? For example:
    - "Completeness drops sharply below θ_E < X arcsec due to PSF blending"
    - "Shallow regions (depth < Y) miss Z% of lenses detectable in deep regions"
    - "The ensemble improves completeness by W% over individual models at the detection boundary"
    
    What findings would make this a compelling paper contribution?

### Part E: Constraints

- **Python 3.11**, PyTorch 2.7+, NumPy 1.26+
- **No lenstronomy dependency** if possible (it has been a source of unit confusion and bugs for us). If lenstronomy is truly necessary, explain why and provide clear unit documentation. If a simpler analytical lens model suffices for the selection function purpose, prefer that.
- Must work with **101×101 pixel cutouts** in g, r, z bands, stored as `.npz` files with key `"cutout"` in shape `(101, 101, 3)` HWC format, in nanomaggies.
- Preprocessing: `raw_robust` mode (outer-annulus median/MAD normalization, clip to [-10, +10])
- Must integrate with our existing `build_model()` factory and checkpoint loading
- Must handle the manifest schema (see attached `MANIFEST_SCHEMA_TRAINING_V1.md`): columns include `cutout_path`, `label`, `split`, `tier`, `pool`, `confuser_category`, `psfsize_r`, `psfdepth_r`
- Pixel scale: 0.262 arcsec/pixel
- DR10 AB zeropoint: 22.5 mag (flux in nanomaggies: `nmgy = 10^((22.5 - mag)/2.5)`)
- Target runtime: selection function grid should complete in < 2 hours on a single GPU (NVIDIA GH200 or A100)

---

## Summary of what we expect back

1. **Strategy document**: North star principles, objective function for sim-to-real gap, validation gates for MNRAS acceptance, what "good enough" means for injection-recovery (not training)
2. **Literature review**: How our approach compares, is it journal-worthy, what are the reviewer risks, what did each comparison paper do for their injection/selection function
3. **Answers to all 28 questions above** (Parts A–D), especially the data reality questions (10–13) and the "minimum viable injection" scoping (Q5, Q11, Q14)
4. **Complete working code**: injection_engine.py, updated selection_function_grid.py, validate_injections.py, any supporting modules — with all answers incorporated into the design
5. **Explanation of every design choice and fix**, referencing our historical failures
6. **Recommended paper figures and narrative** (Q27, Q28)
7. **Honest assessment**: What limitations remain, what claims we can and cannot make, what a skeptical referee would push back on

**This is the most critical deliverable for our paper. We have spent months on failed attempts. Please be thorough, correct, and practical. We would rather have a simpler injection that is defensible and correct than a complex one that has subtle bugs.**

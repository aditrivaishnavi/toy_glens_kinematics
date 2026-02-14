# LLM Prompt: Independent Review of Injection Model 1 — Results and Code Correctness

**Date:** 2026-02-13
**Context:** MNRAS paper — "Calibrating CNN-Based Strong Gravitational Lens Finders in DESI Legacy Survey DR10"
**Attached:** `injection_model_1_code_package.zip` (complete injection engine, selection function pipeline, sim-to-real validation suite, 28-test physics test suite)

---

## What We Need From You

We need two things:

1. **An independent scientific assessment** of whether our Injection Model 1 results are publishable in MNRAS, given the current state of the art in the field.
2. **A rigorous code review** of the attached code package for correctness, bugs, and subtle errors.

We are asking you to be **brutally honest**. We have a history of being told things are fine when they are not. Please treat this as a hostile peer review.

---

## Part 1: What We Built (Injection Model 1)

### 1.1 Injection Pipeline Architecture

We built a self-contained injection-recovery pipeline for measuring the CNN selection function (detection completeness) of a strong gravitational lens finder. The key design:

- **Purpose:** Measure completeness C(θ_E, PSF, depth, source_mag) by injecting synthetic lensed arcs into real DR10 galaxy cutouts and measuring recovery rates.
- **NOT used for training.** The CNN was trained on real confirmed lenses from DESI Legacy Survey DR10 (Paper IV parity approach). The injection is ONLY for calibrating the selection function.

### 1.2 Specific Technical Choices

| Component | Our Choice | Implementation Detail |
|-----------|-----------|----------------------|
| **Lens mass model** | SIE + external shear | Kormann et al. 1994; q_lens sampled U[0.5, 1.0]; phi_lens U[0, pi]; gamma U[0, 0.1]; epsilon-softened at origin |
| **Source light model** | Single Sersic + optional Gaussian clumps | n_sersic U[0.5, 4.0]; R_e U[0.1, 0.5]"; q_source U[0.3, 1.0]; 0-3 clumps at 25% flux fraction |
| **Source magnitude** | r-band AB | Main grid: 23-26; Bright test: 18-26 |
| **Source colors** | Random (not SED-based) | g-r U[0.0, 1.5]; r-z U[-0.3, 1.0]; converted to nanomaggies |
| **Source offset** | Area-weighted | P(β_frac) ∝ β_frac; β = β_frac * θ_E; β_frac in [0.1, 1.0] |
| **Magnification** | Physical from ray-tracing | NOT a free parameter; emerges from SIE lens equation |
| **Flux normalization** | Total unlensed source flux | Sersic integral (Graham & Driver 2005) normalizes profile; flux_observed = ∫ magnified_profile * pixel_area |
| **Flux units** | Nanomaggies throughout | AB ZP = 22.5; no unit conversions at boundaries |
| **PSF** | Gaussian per band | sigma = psfsize_r / 2.355; FFT convolution on full stamp |
| **Noise estimation** | Gaussian from psfdepth_r | sigma_pix = 1/sqrt(psfdepth_r * N_pix_per_PSF); N_pix = pi * (FWHM/2/pixscale)^2 |
| **Sub-pixel oversampling** | 4x default | Verified convergent to <0.5% vs 8x |
| **Core suppression** | Optional radial mask (3-pixel radius) | Tested but NOT used in main grid |
| **Host galaxies** | **Random non-lens galaxies from val split** | Matched to PSF/depth bin; NOT deflector-matched |
| **Injection method** | Additive: injected_image = host + lensed_arc | Verified: max flux conservation error = 0.0 nmgy |
| **Pixel scale** | 0.262"/pixel | DR10 Legacy Survey standard |
| **Stamp size** | 101×101×3 (g, r, z) | HWC, nanomaggies |
| **Preprocessing** | raw_robust | Outer-annulus (r > 40 pixels) median/MAD normalization, clip [-10, 10] |

### 1.3 Selection Function Grid

- **Grid:** 11 θ_E (0.5-3.0") × 7 PSF (0.9-1.8") × 4 depth (24.0-25.5 mag) = 308 cells
- **Injections/cell:** 200 (total: 61,600)
- **Source magnitude bins:** 23-24, 24-25, 25-26 (uniformly sampled)
- **Detection thresholds:** p>0.3, p>0.5, FPR=0.1% (p>0.806), FPR=0.01% (p>0.995)
- **Confidence intervals:** Bayesian binomial, Jeffreys prior Beta(0.5, 0.5)

---

## Part 2: Our Results

### 2.1 Physics Validation (28 unit tests, ALL PASS, 0 skipped)

We ran 28 unit and physics tests covering 8 categories. All pass, including 4 lenstronomy cross-validation tests run against lenstronomy==1.13.5 on the same machine:

| Category | # Tests | Key tolerance | Status |
|----------|---------|---------------|--------|
| Magnification physics | 5 | on-axis mu >> 30; known-offset matches theory | ALL PASS |
| Sersic integral | 4 | Matches closed-form for n=1,q=1; correct R_e^2 and q scaling | ALL PASS |
| SIE lens model | 4 | SIE(q=1) = SIS to <1e-6"; different morphology at q=0.7 | ALL PASS |
| **lenstronomy cross-validation** | **4** | **SIS/SIE deflection <0.1% relative; lensed flux <1% relative** | **ALL PASS** |
| Sub-pixel oversampling | 2 | 4x vs 8x <0.5% | ALL PASS |
| Area-weighted sampling | 2 | KS test p > 0.01 | ALL PASS |
| Lens parameter sampling | 5 | q_lens in range; meta keys present | ALL PASS |
| PSF and core suppression | 2 | PSF preserves flux <1%; core masking zeros center | ALL PASS |

### 2.2 Selection Function Completeness

**Overall completeness at p > 0.3:**

| Source mag bin | N injections | N detected | Completeness |
|---------------|-------------|-----------|-------------|
| All           | 61,600      | 2,673     | **4.3%**    |
| 23-24         | 20,666      | 1,633     | **7.9%**    |
| 24-25         | 20,657      | 816       | **4.0%**    |
| 25-26         | 20,277      | 224       | **1.1%**    |

**Completeness by θ_E (all mag, p > 0.3):**
- Peaks at θ_E = 1.75" (5.7%)
- Drops to 0.9% at θ_E = 0.5"
- Drops to 4.0% at θ_E = 3.0"

### 2.3 Sensitivity Analysis (8 perturbations)

| Perturbation | Mean dC | Max |dC| |
|-------------|---------|----------|
| PSF ±10% | ±0.004 | 0.050 |
| Source R_e ±30% | ±0.002 | 0.040 |
| Color g-r ±0.2 mag | ±0.005 | 0.050 |
| q_lens [0.3,1] vs [0.7,1] | ±0.003 | 0.050 |

All perturbations produce max |dC| < 5 percentage points.

### 2.4 Sim-to-Real Validation Results

**Real lens recall:**
| Threshold | Recall |
|-----------|--------|
| p > 0.3 | **73.3%** |
| p > 0.5 | **68.7%** |
| FPR = 0.1% | **59.7%** |
| FPR = 0.01% | **24.8%** |

**Confuser morphology test** (is the model detecting galaxy shape instead of arcs?):
| Category | N | Frac > 0.3 |
|----------|---|-----------|
| ring_proxy | 200 | 1.0% |
| edge_on_proxy | 200 | 0.5% |
| large_galaxy | 200 | 0.0% |
| blue_clumpy | 200 | 0.0% |
| Random negatives | 200 | 0.0% |

All confuser categories score at baseline FPR levels. Model is NOT exploiting morphology shortcuts.

**Bright arc injection test** (does brightness explain the gap?):
| Source mag | Det (p>0.3) | Arc SNR |
|-----------|-----------|---------|
| 18-19 | **30.5%** | 945 |
| 19-20 | 27.0% | 405 |
| 20-21 | 22.5% | 157 |
| 23-24 | 9.0% | 9.3 |
| 25-26 | 0.5% | 1.4 |

Even at blindingly bright source magnitudes (mag 18-19, arc SNR ~900), injection completeness plateaus at ~30%.

### 2.5 The Gap

| Metric | Value |
|--------|-------|
| Real lens recall (p>0.3) | 73.3% |
| Best injection completeness (bright arcs, p>0.3) | 30.5% |
| **Residual gap** | **~43 percentage points** |
| Standard grid completeness (mag 23-26) | 4.3% |
| **Full gap vs real recall** | **~69 percentage points** |

**Our interpretation:** The model learned a joint host+arc feature during training on real lens systems (which have both arcs AND characteristic massive elliptical deflector galaxies). Our injections put arcs onto random non-lens hosts that lack the deflector galaxy context. The gap is primarily caused by the host galaxy mismatch, not by injection physics errors.

---

## Part 3: What the Literature Does (State of the Art)

Through our literature review, we identified the following as the current state of the art:

### 3.1 HOLISMOKES XI (Cañameras et al. 2024, A&A 692, A72)

The most rigorous published lens-finding methodology:
- **Source galaxies:** 1,574 real galaxies from the Hubble Ultra Deep Field (HUDF) with spectroscopic redshifts from MUSE. NOT parametric profiles.
- **Deflector galaxies:** Real LRGs from SDSS with measured spectroscopic redshifts AND velocity dispersions (50,220 LRGs).
- **Mass model:** SIE with axis ratio and PA inferred from i-band light profile, perturbed following SLACS mass-to-light offsets. External shear U[0, 0.1].
- **Injection method:** Lensed HUDF source convolved with per-position, per-band survey PSF model, then coadded onto the deflector LRG's actual survey cutout.
- **Key differences from us:** (1) Real galaxy sources vs our parametric Sersic. (2) Deflector-matched hosts vs our random non-lens hosts. (3) Per-position PSF vs our Gaussian approximation.

### 3.2 Herle, O'Riordan & Vegetti (2024, MNRAS 534, 1093)

The only paper specifically focused on quantifying CNN selection functions:
- Showed CNNs are biased toward larger θ_E, larger sources, more concentrated Sersic profiles.
- Selection function is independent of mass profile slope.
- Used three training datasets with different simulation approaches.
- Their selection function shows CNN biases reinforce the natural lensing cross-section bias.

### 3.3 Euclid (2025)

- SIE + lenstronomy simulations
- Real galaxy morphologies for sources
- Per-position survey PSF models
- Deflector-matched injection onto observed galaxies

### 3.4 Metcalf et al. (2019) — Bologna Lens Challenge

- Standardized injection-recovery benchmark
- Fully simulated images (not injection into real images)
- Demonstrated CNN superiority over traditional methods
- Noted: "the degree to which the efficiency and biases of lens finders can be quantified largely depends on the realism of the simulated data"

### 3.5 Summary: Where We Stand

| Dimension | Our Model 1 | State of the Art | Assessment |
|-----------|------------|-----------------|------------|
| Lens model | SIE + shear | SIE + shear | **At parity** |
| Magnification | Physical ray-traced | Physical ray-traced | **At parity** |
| Flux calibration | Nanomaggies, AB ZP 22.5 | Survey-matched | **At parity** |
| Source morphology | Single Sersic + clumps | Real HUDF galaxies | **Below** |
| Host galaxy | Random non-lens | Deflector-matched LRGs with σ_v | **Below** |
| PSF | Gaussian from psfsize_r | Per-position survey PSF | **Below (minor)** |
| Noise | Gaussian from psfdepth_r | Per-pixel or empirical | **Below (minor)** |
| Cross-validation | lenstronomy verified | lenstronomy used natively | **At parity** |
| Sensitivity analysis | 8 perturbations, <5% | Herle+2024 varies params | **At parity** |

---

## Part 4: Specific Questions for Independent Assessment

We need you to answer each of the following questions with specific, justified reasoning. Do not give generic or diplomatic answers.

### On the results

**Q1.** Our injection completeness is 4.3% (p>0.3) while real lens recall is 73.3%. The bright arc test shows a 30% ceiling even at mag 18-19. **Is a ~43% residual gap between bright-injection completeness and real-lens recall scientifically defensible, or does it indicate a fundamental flaw in our injection pipeline?** Specifically: is it physically plausible that the model requires host galaxy context (deflector morphology) to that degree, or should we suspect that something else is wrong with our injections (e.g., incorrect arc morphology, wrong spatial distribution, preprocessing artifact)?

**Q2.** The confuser test shows all non-lens galaxy categories (ring, edge-on, large, blue clumpy) score at baseline FPR levels (<1%). **Does this conclusively rule out morphology shortcuts?** Or could there be a more subtle shortcut (e.g., the model detecting the absence of a typical deflector galaxy rather than the presence of an arc)?

**Q3.** The sensitivity analysis shows all 8 perturbations produce max |dC| < 5 percentage points. **Is this sufficient to claim "robustness"?** What additional perturbations should we test? Specifically, did we miss any perturbation that could explain the 43% gap?

**Q4.** Our completeness peaks at θ_E = 1.75" and drops to 0.9% at θ_E = 0.5". Herle et al. (2024) show CNN bias toward larger Einstein radii with 50% of selected systems having θ_E ≥ 0.879". **Are our completeness-vs-θ_E numbers consistent with the Herle et al. selection function shape?** Or do our numbers look anomalously low?

**Q5.** We report FPR = 0.43% at p>0.3 on 3,000 scored negatives. **Is this FPR acceptable for MNRAS publication?** What FPR would a referee expect, and should we use a stricter operating point?

### On publishability

**Q6.** Given that our injection model uses parametric Sersic sources on random hosts (below state of the art), while the state of the art uses real HUDF sources on deflector-matched LRGs: **Is Model 1 alone publishable in MNRAS as a selection function paper?** Or would it be rejected on the grounds that the injection realism is insufficient?

**Q7.** If we frame the 4.3% completeness as a "conservative lower bound" and present it alongside the 73.3% real-lens recall, with the 43% gap honestly attributed to host galaxy context: **Would MNRAS referees accept this framing?** What specific objections would they raise?

**Q8.** Paper IV (Inchausti et al. 2025) does NOT report an injection-recovery selection function. Our selection function is the claimed novel contribution. **Is the contribution still novel and valuable even at Model 1's level of realism?** Or has the bar moved past this?

**Q9.** Herle et al. (2024) quantified CNN selection functions. **How does our work differ from theirs in a way that adds value?** We use a different survey (DR10 vs their simulations), a different model (EfficientNetV2-S vs their generic CNN), and we include sim-to-real validation that they do not.

### On the physics

**Q10.** Our flux normalization: the Sersic profile is normalized by its total unlensed analytical integral, so `flux_nmgy_r` represents the total unlensed source flux. Magnification then amplifies the observed flux naturally through ray-tracing. **Is this the correct convention?** Or should `flux_nmgy_r` represent the observed (lensed) flux?

**Q11.** Our noise model uses sigma_pix = 1/sqrt(psfdepth_r * N_pix_per_PSF) where N_pix_per_PSF = pi * (FWHM/2/pixscale)^2. **Is this formula correct for estimating per-pixel noise from psfdepth?** DR10 psfdepth is the inverse variance of PSF flux, so sigma_PSF_flux = 1/sqrt(psfdepth). To get per-pixel sigma, we divide by sqrt(N_effective_pixels). Is our N_pix calculation right?

**Q12.** We use a Gaussian PSF approximation. DR10 actually has Moffat-like PSFs with extended wings. **Could the Gaussian PSF be systematically biasing our injection completeness?** Specifically: Gaussian PSFs concentrate more light near the core and less in the wings compared to Moffat. Would this make injected arcs appear more or less detectable than they should be?

### On code correctness

**Q13.** Please review the attached code package carefully for:
- (a) Any flux unit errors (nanomaggies vs magnitudes vs surface brightness)
- (b) Any coordinate system errors (arcsec vs pixels, x vs y, sign conventions)
- (c) Any normalization errors in the Sersic profile or lens equation
- (d) Any bugs in the SIE deflection implementation (especially the q→1 branch and the atanh clamping)
- (e) Any errors in the area-weighted sampling or source placement
- (f) Any errors in the arc_annulus_snr calculation
- (g) Any errors in the selection function grid logic (binning, threshold application, CI calculation)
- (h) Any errors in the sensitivity analysis (are the perturbations correctly applied?)
- (i) Any off-by-one or indexing errors
- (j) Any numerical stability issues (NaN propagation, division by zero, overflow)

---

## Part 5: What We Plan Next

We plan two upgrades:

**Model 2 (deflector-matched injection):** Inject synthetic Sersic arcs onto deflector-matched LRG hosts (from our positive pool), using the host galaxy's properties to assign the SIE mass model parameters.

**Model 3 (deflector-matched + real HST sources):** Replace Sersic source profiles with real galaxy images from HST COSMOS/HUDF, ray-traced through the SIE model and painted onto deflector-matched hosts. This would bring us to full state-of-the-art parity with HOLISMOKES.

**Q14.** Given Models 1, 2, and 3: **What is the minimum we need for MNRAS?** Is Model 2 sufficient, or do we need Model 3? What would you recommend as the priority ordering?

**Q15.** If we present all three models (parametric-on-random, parametric-on-matched, real-source-on-matched) as a progression, **does this strengthen the paper** by showing how each improvement changes the selection function? Or does presenting the weaker models hurt us?

---

## Attached Code Package

The attached `injection_model_1_code_package.zip` contains:

```
injection_model_1_code_package/
├── README.md                              # This prompt
├── engine/
│   ├── injection_engine.py                # Core physics engine (694 lines)
│   └── selection_function_utils.py        # Bayesian CI, depth conversion (68 lines)
├── scripts/
│   ├── selection_function_grid.py         # Main 308-cell grid runner (722 lines)
│   ├── sensitivity_analysis.py            # 8-perturbation runs (297 lines)
│   ├── validate_injections.py             # Runtime QA with visual output (650 lines)
│   └── sim_to_real_validation.py          # Comprehensive diagnostic (615 lines)
├── validation/
│   ├── real_lens_scoring.py               # Score real lenses (337 lines)
│   ├── confuser_morphology_test.py        # Morphology shortcut test (330 lines)
│   ├── bright_arc_injection_test.py       # Brightness-dependent completeness (330 lines)
│   └── sim_to_real_validation.py          # Sim-to-real diagnostic (615 lines)
├── tests/
│   └── test_injection_engine.py           # 28 unit/physics tests (713 lines)
└── docs/
    └── INJECTION_MODEL_1_VALIDATION.md    # Full validation report with all results
```

**Please read every file and report any bugs, correctness issues, or concerns.** We have been burned by subtle errors before (wrong flux units, wrong surface brightness convention, coordinate flips, etc.). We need you to verify the physics is correct, not just that the code runs.

---

## Summary of What We Expect Back

1. **Answers to all 15 questions** with specific, justified reasoning
2. **An honest MNRAS-publishability assessment** for Model 1 alone
3. **A priority recommendation** for Model 2 vs Model 3
4. **A code review report** identifying any bugs, errors, or concerns in the attached code
5. **Specific suggestions** for what to add, fix, or change before submission

**Be direct. Be specific. Cite code locations (file, function, line) when reporting issues. Do not soften your assessment to be encouraging.**

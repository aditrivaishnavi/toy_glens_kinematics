# LLM Prompt: Follow-Up — Independent Review of Model 1 and Model 2 Results, Next Steps

**Date:** 2026-02-13
**Context:** MNRAS paper — "Calibrating CNN-Based Strong Gravitational Lens Finders in DESI Legacy Survey DR10"
**Attached:** `full_code_package_model1_model2.zip` — **complete, self-contained** codebase for both Injection Model 1 and Injection Model 2, including the full `dhs` package (model, preprocessing, constants, data loading, training), all scripts, all tests.

---

## Why We Are Writing This Follow-Up

In your previous response (Q1–Q15 + Q13 code review), you raised several valid concerns. We have addressed them. But more importantly, **we have now run Model 2 and the results contradict your central prediction.** We need you to independently assess what happened, what it means, and what we should do next.

We are asking you to be **honest and sincere**. Do not tell us things are fine if they are not. Do not be diplomatic. We have a history of spending weeks on approaches that turned out to be wrong because we accepted plausible-sounding explanations without stress-testing them. We need your genuine assessment.

### What you said last time that we addressed:

1. **"The zip is not self-contained"** — Fixed. The attached zip now includes the **complete `dhs/` package** with `model.py`, `preprocess.py`, `constants.py`, `data.py`, `train.py`, `transforms.py`, `utils.py`, `gates.py`, `calibration.py`, `s3io.py`, `injection_engine.py`, `selection_function_utils.py`, and `__init__.py`. Every import resolves. You can now audit the full scoring and preprocessing pipeline.

2. **"Silent failure modes: FPR threshold derivation inserts blank samples, injection loop counts failures in denominator"** — Fixed. The v2 grid runner (`injection_model_2/scripts/selection_function_grid_v2.py`) now:
   - Retries failed injections with new hosts (up to `MAX_RETRIES=5`)
   - Only counts successful injections in denominators
   - Logs `total_injections_ok`, `total_injections_failed`, `failure_rate_pct`, and a `failure_log_sample` in metadata
   - Our runs show **0 injection failures** across all 44,000 (Model 1) and 41,800 (Model 2) injections

3. **"Prove preprocessing identity"** — The preprocessing code is now in the zip (`dhs/preprocess.py`). Both real-lens scoring and injection scoring use `preprocess_stack(img, mode="raw_robust", crop=False)` which applies outer-annulus median/MAD normalization, clips to [-10, 10], no cropping (101×101 throughout). You can verify this yourself.

4. **"Run Model 2 and show the gap closes"** — We ran it. **The gap did not close. It got slightly worse.** Details below.

5. **"Repackage code for reproducibility"** — Done. Complete zip attached.

---

## Part 1: What We Built for Model 2

Model 2 extends Model 1 with two changes:
1. **Host selection:** Only DEV/SER (LRG-like) hosts instead of random hosts of all types.
2. **Lens parameter conditioning:** SIE axis ratio (q_lens) and position angle (phi_lens) are derived from the host galaxy's r-band second moments, instead of being sampled independently.

### Model 2 Implementation Details

**Host moment estimation** (`injection_model_2/host_matching.py`):
- Extracts r-band channel from the 101×101×3 HWC cutout
- Computes second moments (I_xx, I_yy, I_xy) using intensity-weighted pixel coordinates
- Derives axis ratio q = sqrt(min_eigenvalue / max_eigenvalue) of the moment matrix
- Derives position angle phi = 0.5 * atan2(2 * I_xy, I_xx - I_yy)
- Falls back to q=1.0 (round) if the host is too faint (sum < 1e-6 nanomaggies) or moments are degenerate
- **26 unit tests, all passing** (see `injection_model_2/tests/test_host_matching.py`)

**Lens parameter mapping** (`map_host_to_lens_params` in `host_matching.py`):
- q_lens = clip(q_host * scale, 0.3, 1.0) where scale ~ U[0.8, 1.2] (mild scatter)
- phi_lens = phi_host + small perturbation
- Shear: gamma ~ U[0, gamma_max], random PA (NOT conditioned on host)
- This follows the approach used in HOLISMOKES (Cañameras et al. 2024) and your suggestion in Q14

**Host selection** (`injection_model_2/host_selection.py`):
- Filters manifest for `type_r in ("SER", "DEV")` — the LRG-like morphology types
- 112,744 LRG hosts available (out of 134,149 total)

### Code you provided that we used and rejected:

- **`host_matching.py` (moment-based q/PA):** We adopted the concept, hardened the implementation with robust edge-case handling, and added 26 unit tests. We did NOT use your code verbatim.
- **`injection_engine.py` (your suggested engine):** We **rejected** this entirely. We found **3 confirmed mathematical errors** in your SIE deflection implementation:
  1. Wrong denominator: used `psi + q^2` instead of `psi` (where `psi = sqrt(q^2 * x^2 + y^2)`)
  2. Swapped `atan` and `atanh` in the deflection components
  3. Wrong prefactor: used `q` instead of `sqrt(q)`
  We kept our original validated engine (`dhs/injection_engine.py`) which passes 28 tests including 4 lenstronomy cross-validation tests.

---

## Part 2: Complete Results

### 2.1 Model 1 Results (Rerun — Bugfixed Code)

**Grid:** 11 θ_E × 7 PSF × 5 depth = 385 cells, 200 injections/cell
**Total successful injections:** 44,000 (0 failures)
**Host type:** Random (all types: SER 64%, DEV 20%, REX 16%)

| Threshold | Mean Completeness | N populated cells |
|-----------|------------------|-------------------|
| p > 0.3   | **3.54%**        | 220               |
| p > 0.5   | **2.88%**        | 220               |
| p > 0.7   | **2.37%**        | 220               |
| FPR=0.1% (p>0.806) | **2.05%** | 220            |
| FPR=0.01% (p>0.995) | **0.59%** | 220           |

**By source magnitude bin (p > 0.3):**

| Mag bin | N injections | N detected | Mean Completeness |
|---------|-------------|-----------|------------------|
| All     | 44,000      | 1,557     | **3.54%**        |
| 23-24   | 14,572      | 1,052     | **7.21%**        |
| 24-25   | 14,775      | 402       | **2.72%**        |
| 25-26   | 14,653      | 103       | **0.70%**        |

**By θ_E (p > 0.3, all mag):**

| θ_E (") | Mean Completeness | Mean Arc SNR |
|---------|------------------|--------------|
| 0.50    | 0.65%            | 2.6          |
| 0.75    | 1.45%            | 3.5          |
| 1.00    | 3.02%            | 4.2          |
| 1.25    | 4.35%            | 4.7          |
| 1.50    | 4.65%            | 4.8          |
| 1.75    | 5.12%            | 4.7          |
| 2.00    | 4.75%            | 4.5          |
| 2.25    | 4.08%            | 3.9          |
| 2.50    | 3.85%            | 3.4          |
| 2.75    | 3.78%            | 2.9          |
| 3.00    | 3.23%            | 2.5          |

### 2.2 Model 2 Results (Rerun — Bugfixed Code)

**Grid:** 11 θ_E × 7 PSF × 5 depth = 385 cells, 200 injections/cell
**Total successful injections:** 41,800 (0 failures)
**Host type:** LRG only (DEV/SER)
**Lens conditioning:** q_lens and phi_lens from host r-band second moments
**Fallback moments (host too faint):** 0 out of 41,800

| Threshold | Mean Completeness | N populated cells |
|-----------|------------------|-------------------|
| p > 0.3   | **2.77%**        | 209               |
| p > 0.5   | **2.25%**        | 209               |
| p > 0.7   | **1.80%**        | 209               |
| FPR=0.1% (p>0.806) | **1.55%** | 209            |
| FPR=0.01% (p>0.995) | **0.38%** | 209           |

**By source magnitude bin (p > 0.3):**

| Mag bin | N injections | N detected | Mean Completeness |
|---------|-------------|-----------|------------------|
| All     | 41,800      | 1,156     | **2.77%**        |
| 23-24   | 14,018      | 805       | **5.77%**        |
| 24-25   | 13,956      | 300       | **2.17%**        |
| 25-26   | 13,826      | 51        | **0.37%**        |

**By θ_E (p > 0.3, all mag):**

| θ_E (") | Mean Completeness | Mean Arc SNR | Mean Host q |
|---------|------------------|--------------|-------------|
| 0.50    | 0.87%            | 2.6          | 0.830       |
| 0.75    | 1.24%            | 3.5          | 0.828       |
| 1.00    | 2.34%            | 4.1          | 0.830       |
| 1.25    | 3.34%            | 4.4          | 0.827       |
| 1.50    | 3.79%            | 4.9          | 0.829       |
| 1.75    | 4.05%            | 5.0          | 0.828       |
| 2.00    | 3.42%            | 4.4          | 0.829       |
| 2.25    | 3.47%            | 4.1          | 0.829       |
| 2.50    | 2.74%            | 3.4          | 0.831       |
| 2.75    | 2.95%            | 2.8          | 0.832       |
| 3.00    | 2.21%            | 2.3          | 0.829       |

### 2.3 Head-to-Head Comparison: Model 1 vs Model 2

| Threshold | Model 1 | Model 2 | Difference |
|-----------|---------|---------|------------|
| p > 0.3   | 3.54%  | 2.77%   | **-0.77pp (Model 2 is WORSE)** |
| p > 0.5   | 2.88%  | 2.25%   | **-0.63pp** |
| p > 0.7   | 2.37%  | 1.80%   | **-0.57pp** |
| FPR=0.1%  | 2.05%  | 1.55%   | **-0.50pp** |
| FPR=0.01% | 0.59%  | 0.38%   | **-0.21pp** |

**Model 2 is consistently worse than Model 1 across ALL thresholds.**

### 2.4 Host Conditioning Diagnostic (4-Way Comparison)

We ran a controlled 4-way experiment to isolate whether the effect comes from host type or lens parameter conditioning:

| Condition | Description | Mean C (p>0.3) |
|-----------|------------|-----------------|
| LRG_conditioned | LRG hosts + q/PA from moments (= Model 2) | **4.90%** |
| LRG_independent | LRG hosts + independent q/PA | **4.90%** |
| random_independent | Random hosts + independent q/PA (= Model 1) | **4.77%** |
| random_conditioned | Random hosts + q/PA from moments | **4.63%** |

(Note: These are from a smaller diagnostic run with 500 injections/point at 6 θ_E values, not the full 385-cell grid, so absolute numbers differ slightly from the full runs.)

**Result: The four conditions are statistically indistinguishable.** Neither host type (LRG vs random) nor lens parameter conditioning (conditioned vs independent) makes a meaningful difference.

### 2.5 The Gap (Updated)

| Metric | Value |
|--------|-------|
| Real lens recall (p>0.3) | **73.3%** |
| Model 1 injection completeness (p>0.3) | 3.5% |
| Model 2 injection completeness (p>0.3) | 2.8% |
| Best bright-arc injection (mag 18-19, p>0.3) | 30.5% |
| **Gap: real recall − best injection** | **~43 percentage points** |

---

## Part 3: Bugs We Found and Fixed in the Grid Runner

In addition to the 3 bugs you identified (silent failures in FPR derivation, injection denominator, rank calculation), we found and fixed:

**Bug 1 (Medium — data integrity): `host_q_list` / `batch_list` length mismatch**
In `selection_function_grid_v2.py`, `host_q_list.append()` was called BEFORE the try/except block for injection, while `batch_list.append()` was inside the try block. If any injection failed, `host_q_list` would have an extra entry, causing misaligned `mean_host_q` statistics. Fixed by deferring `host_q_list.append()` to the success path. (Had no numerical impact since failure count = 0, but the code was incorrect.)

**Bug 2 (Low — metadata): `n_sufficient_cells` and `n_empty_cells` overcounting**
The metadata calculation did not filter for `source_mag_bin == "all"`, so each per-magnitude-bin row was counted as a separate "cell." This inflated the metadata numbers ~4× (e.g., 880 instead of 220). Fixed by adding the filter. Cosmetic only; did not affect completeness.

**Bug 3 (Low — documentation): Notes said "SIS+shear" instead of "SIE+shear"**
The engine uses SIE (ellipsoidal). Fixed.

---

## Part 4: Questions for Independent Assessment

We need clear, specific answers to each question. Not vague summaries.

### On the Central Result: Model 2 Did Not Help

**Q1.** You predicted in Q14 that Model 2 (deflector-conditioned injection) should "show that the injection–real gap collapses substantially at fixed thresholds." The data shows Model 2 is actually **0.77 percentage points worse** than Model 1. The 4-way diagnostic shows zero significant effect of host type or conditioning.

- Was your prediction wrong? If so, why?
- You also said in Q14: "If Model 2 does not improve completeness meaningfully, your story is not 'host mismatch.' It is 'injections do not resemble training positives in feature space.'" **Do you stand by this fallback diagnosis?**
- What specific aspects of "feature space" do you think are mismatched? Can you be concrete — not just "morphology" but specifically what feature statistics would differ between our Sersic injections and real lensed arcs as seen by EfficientNetV2?

**Q2.** The mean host q for Model 2 is ~0.83 (nearly round). This seems high for LRG hosts.
- Is 0.83 a physically reasonable mean axis ratio for DEV/SER galaxies in DR10?
- Could our moment estimation be biased toward round shapes? (The code is in `injection_model_2/host_matching.py` — please review.)
- If the hosts are genuinely round (q~0.83), then conditioning the lens q on the host q gives q_lens ~ 0.83 * U[0.8, 1.2] ≈ U[0.66, 1.0]. Model 1 uses q_lens ~ U[0.5, 1.0]. Could this narrower, rounder q_lens range in Model 2 explain the lower completeness (rounder lenses produce less elongated, harder-to-detect arcs)?

**Q3.** The bright arc test showed a 30% ceiling even at mag 18-19 (arc SNR ~900). Now we know LRG hosts don't help either. **What is the 30% ceiling caused by?** Please give a concrete, testable hypothesis — not just "source morphology." Specifically:
- What fraction of the ceiling do you attribute to: (a) source morphology (Sersic too smooth), (b) preprocessing mismatch, (c) color/SED mismatch, (d) arc spatial distribution mismatch, (e) something else we haven't considered?
- How would you design an experiment to determine which factor dominates? Provide specific code/methodology.

**Q4.** We notice that Model 2 has fewer populated cells (209 vs 220) than Model 1. This is because LRG hosts are concentrated at certain PSF/depth combinations (they are bright, preferentially in good-seeing fields). Could the cell population difference bias the comparison? Should we restrict both models to the same set of populated cells for a fair comparison?

### On the Physics

**Q5.** Our mean arc SNR ranges from 2.3 to 5.0 across the θ_E grid. **Are these arc SNR values physically reasonable for detectable lenses?** Real lens candidates in DR10 presumably have higher arc SNR to be visible. If our typical injection has arc SNR of ~4, is it even reasonable to expect the CNN to detect it? What arc SNR do real detected lenses have in DR10?

**Q6.** Model 2 shows mean_host_q = 0.829 ± 0.002 across ALL θ_E bins — essentially constant. Is this physically suspicious? Should the host morphology vary with θ_E (larger Einstein radii come from more massive, potentially rounder galaxies)?

**Q7.** We do NOT add noise to the injected arc signal (because we inject into real noisy cutouts, so the real noise is already present). You flagged this in your previous review (Q13.C.3). **Is this actually correct?** The injected arc sits ON TOP of the existing noise floor. In a real lens, the arc photons would contribute Poisson noise. Are we underestimating the noise on the arc by not adding Poisson noise proportional to the arc signal? Could this make injected arcs look slightly "too clean" compared to real arcs?

**Q8.** We use a Gaussian PSF with sigma = psfsize_r / 2.355. You said the Gaussian PSF effect is "a few percent to maybe 10%" and "fix it in Model 3 parity, but do not expect it to explain the main gap." Given that the main gap is ~70 percentage points: is this assessment still your view? Should we deprioritize PSF improvements?

### On Publishability and Next Steps

**Q9.** Given that:
- Model 1 completeness is 3.5% (p>0.3) while real lens recall is 73.3%
- Model 2 (your recommended next step) did NOT close the gap
- The 4-way diagnostic shows host type and conditioning are irrelevant
- The bright-arc test shows a 30% ceiling
- The primary remaining hypothesis is source morphology mismatch

**Is this paper still publishable in MNRAS?** Be honest. If the answer is "not in its current form," say so clearly and tell us what is minimally needed.

**Q10.** You previously suggested "Model 2.5" — use real DR10 blue galaxies as source-plane images, lensed through SIE. **We need you to provide complete, working code for this.** Specifically:
- How do we select the source galaxies from DR10? What color cuts, magnitude ranges, and morphology types?
- How do we extract source-plane images from DR10 cutouts? Do we need to deconvolve the PSF? Handle the sky background?
- How do we place these in the source plane and lens them? The current engine expects an analytic Sersic profile — what changes are needed?
- What are the expected failure modes?
- **Please provide working code** (not pseudocode, not stubs) with clear instructions and data source URLs.

**Q11.** You also suggested "Model B" — arc transplant from real lenses (fit smooth model, extract residual arc, transplant). **Is this practical with ~200 real lenses?** How would you handle the small sample size? Would this introduce a new form of label circularity (arcs from training lenses injected for testing)?

**Q12.** You suggested "Model C" — feature-space calibration / importance weighting using model embeddings. **Can you explain this more concretely?** What specific steps would we take? What embeddings would we compare? How would the correction factor be applied to the selection function? Provide mathematical formulation.

### On the Training and Model

**Q13.** Our EfficientNetV2-S model was trained as follows:
- Architecture: EfficientNetV2-S (20.2M params)
- Training data: 134,149 negatives + ~4,800 positives (28:1 ratio), 70/30 train/val split
- Input: 101×101×3 (g, r, z), nanomaggies, raw_robust preprocessing
- Training protocol: Two-phase — (1) 19 epochs at LR=3.88e-4 with step schedule, peaked at val_AUC=0.9915; (2) 60-epoch fine-tuning from peak checkpoint at LR=5e-5 with cosine decay
- Best val_AUC: 0.9915 (epoch 19 of phase 1)
- Sample weighting: unweighted loss (balanced by label)

**Is there anything about this training setup that could cause the model to reject synthetic injections while accepting real lenses?** For example: if the training positives (real lens candidates) have specific statistical properties (central galaxy light profile, arc color, arc position relative to center) that Sersic injections don't match, the model would learn to discriminate between "real positive" and "synthetic injection" even at high SNR. Could this explain the 30% ceiling?

**Q14.** The training config shows `preprocessing: raw_robust` with `crop: false` (101×101). The injection pipeline also uses `raw_robust` without cropping. **Can you verify from the attached `dhs/preprocess.py` that the preprocessing applied during scoring of injections is identical to the preprocessing applied during training?** This is your previous Priority #2 recommendation. We need you to audit this from the actual code, not from our description.

### On Code Review

**Q15.** The full `dhs/` package is now in the zip. In your previous review, you could not audit the scoring pipeline because `dhs.model`, `dhs.preprocess`, `dhs.constants` were missing. **Please now audit:**
- (a) `dhs/preprocess.py`: Is `preprocess_stack()` correctly implementing outer-annulus median/MAD normalization? Are there any edge cases that could produce different normalizations for real cutouts vs injected cutouts?
- (b) `dhs/model.py`: Is the model architecture correct? Any issues with the inference path?
- (c) `dhs/data.py`: Is the data loading pipeline correct for scoring? Any preprocessing differences between training and inference?
- (d) `dhs/transforms.py`: Are the augmentations applied during training only, or could they leak into inference?
- (e) `injection_model_2/host_matching.py`: Are the moment calculations correct? Is the q/PA estimation robust? Review the 26 test cases.
- (f) `injection_model_2/scripts/selection_function_grid_v2.py`: Is the grid logic correct now? Are the silent failure fixes adequate?

**Q16.** In our injection engine (`dhs/injection_engine.py`), the `inject_sis_shear()` function injects the lensed arc into the cutout by simple addition: `cutout + arc_image`. The arc is in nanomaggies, the cutout is in nanomaggies. Then the combined image goes through `preprocess_stack(raw_robust)`.

Consider: for a real lens in the training data, the "arc" light is part of the cutout from the start — it went through the same photometric pipeline (sky subtraction, flat fielding, etc.) as the host galaxy. For our injections, the arc is a synthetic addition that never went through that pipeline. **Could this difference in how the arc was "observed" cause a detectable statistical signature** that the CNN picks up? For example:
- Sky subtraction in DR10 might partially subtract extended arc light in real lenses, making them appear different from our additive injections
- Flat-fielding patterns, scattered light, or other instrumental effects would be present in real arcs but not in our synthetic ones
- The noise statistics in the arc region would be different (real arc: Poisson + read noise, our injection: only the host's noise, no additional photon noise from the arc)

### On Concrete Next Steps

**Q17.** Given everything above, what is your recommended **priority ordering** of next steps? Please rank these and for the top 3, provide complete implementation details:

1. Model 2.5 (real DR10 source galaxies)
2. Preprocessing audit (detailed comparison of real lens vs injection feature statistics)
3. Feature-space embedding comparison (what does the CNN "see" differently?)
4. Arc transplant from real lenses (Model B)
5. PSF improvement (per-position survey PSF instead of Gaussian)
6. Noise model improvement (add Poisson noise for arc photons)
7. Feature-space calibration / importance weighting (Model C)
8. Something else we haven't considered

**For your top recommendation: provide working code, specific data sources with URLs or download instructions, expected runtime, and expected outcome** (what completeness improvement would you predict, and what would you conclude if it doesn't materialize?).

**Q18.** If the paper is salvageable in its current state (Model 1 + Model 2 negative result), **what would the paper structure look like?** Give us a concrete outline:
- Title
- Abstract (1-paragraph draft)
- Section structure
- Key figures (describe what each figure shows)
- Main claims and how they are supported by the data

If the paper requires Model 2.5 or further work, say so, and estimate the timeline (assuming single GPU, existing data infrastructure).

---

## Part 5: What We Need From You

1. **Honest assessment** of whether we are on the right track or wasting time.
2. **Concrete diagnosis** of why Model 2 failed to improve completeness.
3. **Working code** for the highest-priority next step (not stubs, not pseudocode — code that runs).
4. **Clear data source instructions** — URLs, download commands, expected file formats.
5. **Specific predictions** — what completeness improvement should we expect from each proposed model, so we can tell if it's working or not.
6. **Red flags** — anything in our results or code that looks wrong, suspicious, or inconsistent.

**Please do not hold back.** We would rather hear "this approach is fundamentally flawed and here's why" than "looks good, maybe try X." If there is a basic error in our methodology that explains the 70-point gap, we need to know now, not after another month of work.

---

## Attached Code Package Structure

```
full_code_package_model1_model2/
├── dhs/                                    # Complete scoring/training/preprocessing package
│   ├── __init__.py
│   ├── calibration.py                      # Calibration utilities
│   ├── constants.py                        # STAMP_SIZE, CUTOUT_SIZE, etc.
│   ├── data.py                             # Dataset class for training/inference
│   ├── gates.py                            # Quality gates
│   ├── injection_engine.py                 # Core physics engine (SIE+shear, Sersic, PSF)
│   ├── model.py                            # ResNet18, BottleneckedResNet, EfficientNetV2-S
│   ├── preprocess.py                       # preprocess_stack (raw_robust, residual)
│   ├── s3io.py                             # S3 I/O utilities
│   ├── selection_function_utils.py         # Bayesian CI, depth conversion
│   ├── train.py                            # Training loop
│   ├── transforms.py                       # Data augmentations
│   └── utils.py                            # Normalize, radial profile, etc.
├── injection_model_1/                      # Model 1: Parametric Sersic on random hosts
│   ├── README.md
│   ├── engine/
│   │   ├── injection_engine.py             # Same engine, local copy
│   │   └── selection_function_utils.py
│   ├── scripts/
│   │   ├── selection_function_grid.py      # Original v1 grid runner
│   │   ├── sensitivity_analysis.py
│   │   ├── sim_to_real_validation.py
│   │   └── validate_injections.py
│   ├── tests/
│   │   └── test_injection_engine.py        # 28 tests (all pass)
│   └── validation/
│       ├── bright_arc_injection_test.py
│       ├── confuser_morphology_test.py
│       ├── real_lens_scoring.py
│       └── sim_to_real_validation.py
├── injection_model_2/                      # Model 2: Sersic on LRG hosts, conditioned q/PA
│   ├── README.md
│   ├── host_matching.py                    # Moment-based q/PA estimation
│   ├── host_selection.py                   # LRG host selection
│   ├── scripts/
│   │   ├── selection_function_grid_v2.py   # Bugfixed grid runner (supports Model 1 & 2)
│   │   └── host_conditioning_diagnostic.py # 4-way diagnostic
│   └── tests/
│       └── test_host_matching.py           # 26 tests (all pass)
└── results/                                # (empty in zip — results shown in this prompt)
```

### Test Status

| Test Suite | N Tests | Status |
|-----------|---------|--------|
| injection_model_1/tests/test_injection_engine.py | 28 | **ALL PASS** |
| injection_model_2/tests/test_host_matching.py | 26 | **ALL PASS** |

### Run Metadata

**Model 1 rerun:**
```json
{
  "injection_model": 1,
  "host_selection": "random (all types)",
  "lens_model_conditioning": "independent priors",
  "total_injections_ok": 44000,
  "total_injections_failed": 0,
  "n_cells": 385,
  "n_sufficient_cells": 220,
  "n_empty_cells": 165,
  "preprocessing": "raw_robust",
  "crop": false,
  "seed": 1337
}
```

**Model 2 rerun:**
```json
{
  "injection_model": 2,
  "host_selection": "LRG (DEV/SER)",
  "lens_model_conditioning": "host-conditioned (q, PA from r-band moments)",
  "total_injections_ok": 41800,
  "total_injections_failed": 0,
  "total_fallback_moments": 0,
  "n_cells": 385,
  "n_sufficient_cells": 209,
  "n_empty_cells": 176,
  "preprocessing": "raw_robust",
  "crop": false,
  "seed": 1337
}
```

**Host conditioning diagnostic:**
```json
{
  "n_per_point": 500,
  "theta_es": [0.75, 1.0, 1.25, 1.5, 2.0, 2.5],
  "conditions": ["LRG_conditioned", "LRG_independent", "random_independent", "random_conditioned"],
  "n_lrg_hosts": 112744,
  "n_random_hosts": 134149
}
```

---

**Please treat this as a hostile peer review. We need the truth, not reassurance.**

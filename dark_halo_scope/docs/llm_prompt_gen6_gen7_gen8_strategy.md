# Prompt: Gen6/7/8 Code Fixes + Research Strategy for Publication

**Context:** We are developing a strong gravitational lensing detection system with two complementary goals:
1. **Selection Function Calibration:** A completeness-calibrated lens finder using injection-recovery methodology to quantify `C(θ_E, z_l)` - the probability of detecting a lens as a function of Einstein radius and redshift.
2. **Practical Lens Finder:** A high-performance classifier for discovering new strong lenses in survey data (DESI Legacy DR10).

**Target Venues:** MNRAS, ApJ, AAS journals (peer-reviewed astrophysics)

**Current State:** Gen5 model using COSMOS real galaxy sources is running. We have code for Gen6/7/8 but it needs fixes.

---

## Part 1: Code Review Findings - Please Fix These Issues

We reviewed the Gen6/7/8/Uber code bundle and found the following issues. **Please regenerate the code with these fixes applied:**

### 🔴 CRITICAL ISSUES (Must Fix)

#### Issue 1: `decode_stamp_npz` Format Incompatibility
**File:** `dhs_gen/validation/quality_checks.py`

**Problem:** Your code assumes stamps have key `"img"`:
```python
arr = z["img"] if "img" in z else z[list(z.keys())[0]]
```

**Our actual format uses separate band keys:**
```python
# Actual stamp format
z["image_g"]  # g-band 64x64
z["image_r"]  # r-band 64x64  
z["image_z"]  # z-band 64x64
```

**Required Fix:** Detect and handle both formats.

---

#### Issue 2: Surface Brightness Units Not Documented
**File:** `dhs_gen/deep_sources/deep_source_bank.py`

**Critical Context:** In Gen5, we had a catastrophic bug where lenstronomy's `INTERPOL` light model expects **surface brightness** (flux/arcsec²) but we passed **flux/pixel**. This caused ~1000x flux errors.

**Required Fix:**
1. Document explicitly that templates are stored in flux/pixel units
2. Add a prominent warning in docstrings about the conversion needed:
```python
# When using with lenstronomy INTERPOL:
# surface_brightness = template * flux_nmgy / (pixel_scale_arcsec ** 2)
```
3. Consider adding a helper function for the conversion

---

#### Issue 3: float16 Precision Loss
**File:** `dhs_gen/deep_sources/deep_source_bank.py` (line 107)

**Problem:**
```python
images.append(img_rs.astype(np.float16))  # Only ~3 decimal digits!
```

**Impact:** float16 causes quantization noise and can produce NaN for extreme values.

**Required Fix:** Use float32:
```python
images.append(img_rs.astype(np.float32))
```

---

### 🟠 IMPORTANT ISSUES (Should Fix)

#### Issue 4: No NaN/Inf Validation
**Files:** `sersic_clumps.py`, `deep_source_bank.py`, `artifacts.py`

**Context:** Our Gen5 training crashed due to NaN values in input data.

**Required Fix:** Add validation at the end of each generator:
```python
if not np.isfinite(output_image).all():
    raise ValueError(f"NaN/Inf detected in generated image for key={key}")
```

---

#### Issue 5: Uber Mixer Creates Duplicate Columns
**File:** `dhs_gen/uber/mixer.py`

**Problem:** `append_column` doesn't check if column exists.

**Required Fix:**
```python
for col in ["source_mode", "artifact_profile", "gen_variant"]:
    if col in tbl.column_names:
        tbl = tbl.drop(col)
```

---

#### Issue 6: PSF Kernel Size Validation Missing
**Files:** `utils.py` kernel functions

**Context:** In Gen5, PSF kernel (145x145) exceeded stamp size (64x64) causing crashes.

**Required Fix:** Add `max_size` parameter with validation:
```python
def moffat_kernel(fwhm_pix, beta=3.5, size=33, max_size=None):
    if max_size is not None:
        size = min(size, max_size)
    # Ensure odd size
    size = size if size % 2 == 1 else size - 1
```

---

#### Issue 7: Missing `bandset` Handling
**File:** `quality_checks.py`

**Context:** Some stamps have only r-band (`bandset="r"`) instead of all three.

**Required Fix:** Filter or handle gracefully:
```python
if "bandset" in row.column_names:
    bandset = row["bandset"][0].as_py()
    if bandset != "grz":
        # Handle single-band stamps differently
```

---

#### Issue 8: Relative Imports May Fail on Spark Executors
**Problem:** `from ..utils import ...` requires proper package structure.

**Required Fix:** Either:
1. Add setup.py/pyproject.toml for proper installation, OR
2. Document that module must be installed via `pip install -e .`

---

### 🟡 MINOR ISSUES

- Add `__all__` exports to `__init__.py` files
- Add unit test for `deep_source_bank.py`
- Verify `bilinear_resample` flux conservation with test
- Use config files instead of hardcoded paths in examples

---

## Part 2: Strategic Research Questions

Given our two-pronged research goals, please advise on the following:

### Question 1: Additional Injection Mechanisms

Beyond Gen6 (deep sources), Gen7 (hybrid Sersic+clumps), and Gen8 (domain randomization), **what other injection mechanisms should we consider for original research?**

Specifically:
- Are there other source morphology models used in the literature we're missing?
- Should we consider variable lensing configurations (multipoles, external shear, substructure)?
- What about time-variable sources (for time-domain surveys)?
- Are there injection strategies specifically important for selection function calibration?

---

### Question 2: Ablation Methodology

**What ablation studies would reviewers expect to see in MNRAS/ApJ?**

Consider:
- Which hyperparameters are most scientifically meaningful to vary?
- How should we isolate the effect of each generation's contribution?
- What statistical tests are appropriate for comparing model variants?
- How many random seeds / training runs are needed for robust conclusions?

---

### Question 3: Model Fusion and Ensembles

**After training Gen5-Gen8 models, how should we combine them?**

Consider:
- Simple averaging vs. learned weighting
- Stacking (meta-learner on model outputs)
- Uncertainty-weighted ensembles
- Should ensemble weights vary by image properties (depth, seeing)?
- For selection function calibration, how do we propagate ensemble uncertainty?

---

### Question 4: Beyond Gen8 - What's Missing?

**What innovations would make a real impact beyond these 8 model generations?**

Consider:
- Self-supervised pretraining on unlabeled survey data
- Contrastive learning approaches
- Physics-informed architectures or loss functions
- Active learning / hard negative mining strategies
- Transfer learning from other astronomical domains
- Multi-scale or attention-based architectures

---

### Question 5: Selection Function vs. Lens Finder Trade-offs

**How should we balance the two research goals?**

Our goals have different requirements:
- **Selection Function:** Needs calibrated completeness, controlled injections, well-characterized biases
- **Lens Finder:** Needs high recall on real lenses, low false positive rate, practical usability

Questions:
- Should we optimize for one goal first, then adapt?
- Are there model design choices that help both equally?
- How do we validate selection function calibration on real data?
- What anchor datasets (SLACS, BELLS, etc.) are essential?

---

### Question 6: Publication Priority Order

**Given limited resources, what's the optimal priority order for publishing defensible original research?**

Our current progress:
- ✅ Gen1-4: Sersic sources, various PSF models
- ✅ Gen5: COSMOS real galaxy sources (currently running)
- 🔧 Gen6-8: Code ready but needs fixes

Options to prioritize:
1. Complete Gen5 → Publish single strong paper
2. Complete Gen5 + one ablation (e.g., Gen7) → Publish comparative paper
3. Complete all Gen5-8 → Publish comprehensive methodology paper
4. Focus on selection function calibration → Publish completeness-focused paper
5. Focus on real lens discovery → Publish catalog/discovery paper

Please recommend:
- Which publication strategy maximizes impact vs. effort?
- What's the minimum viable paper for each strategy?
- Which ablations are "must-have" vs. "nice-to-have"?

---

### Question 7: Reviewer Concerns Anticipation

**What criticisms should we anticipate from MNRAS/ApJ reviewers?**

Consider:
- Simulation-to-real gap concerns
- Selection function validation
- Comparison to existing methods (CMU DeepLens, Holloway+, HOLISMOKES)
- Statistical rigor of performance claims
- Reproducibility requirements
- Data and code availability expectations

---

### Question 8: Novel Contributions

**What aspects of our approach are genuinely novel vs. incremental improvements?**

Our claimed novelties:
1. COSMOS galaxy sources for injection (vs. Sersic)
2. Injection-recovery selection function calibration
3. DR10 coverage at scale (>10M injections)
4. Hard negative mining from real survey

Are these sufficient for a top-tier publication? What would strengthen the novelty claim?

---

## Part 3: Deliverables Requested

Please provide:

1. **Fixed Code:** Regenerated Gen6/7/8/Uber code with all issues addressed
2. **Research Roadmap:** Priority-ordered list of experiments for publication
3. **Ablation Design:** Specific experiments with expected outcomes
4. **Ensemble Strategy:** Recommended approach with implementation sketch
5. **Reviewer Response Prep:** Anticipated criticisms and pre-emptive responses

---

## Technical Context (For Reference)

### Current Pipeline Architecture
```
Phase 4a: Manifest generation (lens parameters, source params)
     ↓
Phase 4c: Injection + stamp generation (COSMOS/deep/hybrid sources)
     ↓
Phase 4p5: Compaction + split relabeling
     ↓
Phase 5: Model training (ConvNeXt-Tiny, focal loss)
     ↓
Phase 5b: Inference on real survey data
```

### Current Model Configuration (Gen5)
- Architecture: ConvNeXt-Tiny
- Input: 64×64×3 (g, r, z bands)
- Loss: Focal loss (α=0.25, γ=2.0)
- Normalization: Outer annulus
- Augmentation: Random flips, rotations
- Meta features: psfsize_r, psfdepth_r

### Dataset Size
- Training: ~2.5M stamps per epoch
- Validation: ~1M stamps
- Test: ~1M stamps (locked)

### Computational Resources
- EMR: 34 × m5.2xlarge (272 vCores)
- Lambda GPU: GH200 (97GB VRAM)

---

*Please be thorough and specific. We need actionable advice for publication-quality research.*

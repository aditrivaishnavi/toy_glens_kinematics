# LLM Review Request: Gen6/7/8 Code + Priority Plan

**Context:** We are building a strong gravitational lensing detection system with two goals:
1. **Selection Function Calibration:** Quantify lens detectability as C(θ_E, z_l)
2. **Practical Lens Finder:** High-performance classifier for DESI Legacy DR10

**Current Status:**
- Gen5 training COMPLETE (AUC=0.895, TPR@FPR1%=95.8%)
- Gen6/7/8 code reviewed, bugs fixed, integrated into codebase
- EMR 4c Corrected job running (generating training data with new metrics)

---

## What I Need You to Review

### 1. Code Quality and Correctness

The attached `dhs_gen_v2_fixed.zip` contains:

```
dhs_gen/
├── deep_sources/          # Gen6: Ground-based deep cutouts
│   ├── deep_source_bank.py
│   └── deep_source_sampler.py
├── hybrid_sources/        # Gen7: Sersic + clumps
│   └── sersic_clumps.py
├── domain_randomization/  # Gen8: Artifacts/augmentation
│   └── artifacts.py
├── uber/                  # Uber mixer
│   └── mixer.py
├── validation/            # QA checks
│   └── quality_checks.py
├── utils.py               # Shared utilities
├── tests/                 # Unit tests
└── INDEPENDENT_REVIEW.md  # My review findings
```

### 2. Priority Plan Review

See `docs/gen6_gen7_gen8_priority_plan.md` for the detailed plan.

---

## Previous Issues Found and Fixed

When I independently reviewed the code that was claimed to be "fixed", I found these critical bugs:

| Issue | Problem | Status |
|-------|---------|--------|
| `decode_stamp_npz` | Returned `ndarray` not `Tuple[ndarray, str]`; still checked for "img" key | ✅ Fixed |
| `to_surface_brightness` | Imported but function didn't exist | ✅ Fixed |
| `max_kernel_size` | Used as variable but never defined | ✅ Fixed |
| `__init__.py` imports | 3 files imported non-existent classes | ✅ Fixed |

All 8 tests now pass.

---

## Direct Questions for You

### Code Questions

**Q1: Is the `to_surface_brightness` implementation correct?**

```python
def to_surface_brightness(flux_per_pixel: np.ndarray, pixscale_arcsec: float) -> np.ndarray:
    """Convert flux/pixel to surface brightness (flux/arcsec²)."""
    pixel_area_arcsec2 = pixscale_arcsec ** 2
    return flux_per_pixel / pixel_area_arcsec2
```

Is dividing by pixel_area the correct conversion for lenstronomy INTERPOL?

---

**Q2: Is the `decode_stamp_npz` band ordering guaranteed?**

```python
for band_key, band_char in [("image_g", "g"), ("image_r", "r"), ("image_z", "z")]:
    if band_key in z:
        bands.append(np.asarray(z[band_key], dtype=np.float32))
        bandset_chars += band_char
arr = np.stack(bands, axis=0)  # (C, H, W)
```

If only `image_r` and `image_z` exist (no `image_g`), will the resulting array have r at index 0 and z at index 1? Is this correct or should we pad with zeros for missing bands?

---

**Q3: Does the domain randomization apply artifacts in the right order?**

Current order in `apply_domain_randomization`:
1. Background plane
2. PSF anisotropy (convolution)
3. Cosmic rays
4. Saturation wings
5. Astrometric jitter

Is this physically realistic? Should PSF convolution happen before or after cosmic rays?

---

**Q4: Is the `bilinear_resample` flux conservation correct?**

```python
out = out / max(1e-12, (scale_y * scale_x))
```

This divides by the scale factor to conserve total flux. But bilinear interpolation doesn't perfectly conserve flux. Is this approximation acceptable for our use case (< 2% error acceptable)?

---

**Q5: Are there any race conditions or thread-safety issues for Spark?**

The code uses hash-based deterministic RNG seeded by task_id. Is there any shared state that could cause issues when running on multiple Spark executors simultaneously?

---

### Priority Plan Questions

**Q6: Is the phased approach correct?**

My proposed phases:
1. Complete Gen5 validation (3-5 days)
2. Quick pilot trainings for Gen6/7/8 (1 week)
3. Full training for winning generations (1-2 weeks)
4. Ensemble + analysis (1 week)
5. Paper writing (2-3 weeks)

**Total: 5-7 weeks**

Is this timeline realistic? What typically gets underestimated?

---

**Q7: Should we skip Gen6 if we don't have deep cutouts?**

Gen6 requires ground-based deep imaging cutouts (HSC deep fields or similar). If we don't have these readily available, should we:
- A) Spend time acquiring them (delay 1-2 weeks)
- B) Skip Gen6 entirely and focus on Gen7+Gen8
- C) Use a proxy dataset (what would work?)

---

**Q8: Is the 80/20 train/val re-split strategy correct?**

Current situation:
- train: 26% (3.2M)
- val: 39% (4.8M)
- test: 35% (4.3M)

Proposed fix: Keep test locked, reassign most of val to train → 70-80% train, 10-15% val, 35% test

Is this the right approach? Any concerns about changing splits mid-project?

---

**Q9: For the ensemble, should we use logit averaging or probability averaging?**

Proposed approach:
1. Temperature-scale each model on validation
2. Average logits (not probabilities)
3. Re-calibrate ensemble

Is logit averaging better than probability averaging for this use case? Why?

---

**Q10: What's the minimum viable paper for MNRAS/ApJ?**

Given our current progress:
- ✅ Gen5 model with AUC=0.895
- ✅ 12M+ injections with selection function metadata
- ⏳ Gen6/7/8 ablations (not yet done)

Could we publish Gen5 alone as a "methods + selection function" paper? Or do reviewers expect ablations?

---

### Scientific Questions

**Q11: Is the hybrid source (Sersic + clumps) scientifically defensible?**

Gen7 generates sources as:
- Base Sersic profile (n=1.0 ± 0.15, re=6 pixels)
- 2-6 Gaussian clumps placed around the center
- Optional gradient

Does this represent realistic high-z star-forming galaxies? What parameters should we tune?

---

**Q12: Are the domain randomization artifact amplitudes realistic?**

Current defaults:
- Cosmic ray amplitude: 8× MAD
- Saturation wing amplitude: 12× MAD
- Background plane amplitude: 2% of MAD
- PSF ellipticity sigma: 0.05

Are these physically motivated or arbitrary? How should we calibrate them against real DR10 data?

---

**Q13: What's the biggest risk to publication?**

Of these potential criticisms, which is most likely to sink the paper?
- A) Simulation-to-real gap (models don't transfer)
- B) Selection function prior dependence
- C) Train/test leakage
- D) Insufficient ablations
- E) Lack of real lens validation

---

**Q14: Should we attempt real lens discovery before publication?**

We have a trained Gen5 model. Should we:
- A) Run inference on DR10 and attempt to find new lenses (high risk, high reward)
- B) Focus purely on methodology and selection function (safer)
- C) Do both but report discovery as "future work"

---

**Q15: What additional injection mechanisms should we prioritize?**

Beyond Gen6/7/8, the LLM suggested:
- Lens mass model variability (power-law slope, multipoles)
- Substructure and satellites
- Multi-band color gradients

Which of these would have the highest impact for the least implementation effort?

---

## Code Snippets for Reference

### Surface Brightness Conversion (utils.py)

```python
def to_surface_brightness(flux_per_pixel: np.ndarray, pixscale_arcsec: float) -> np.ndarray:
    """Convert flux/pixel to surface brightness (flux/arcsec²).
    
    IMPORTANT: lenstronomy INTERPOL expects surface brightness, not flux/pixel.
    This was the source of the Gen5 ~1000x flux bug.
    """
    pixel_area_arcsec2 = pixscale_arcsec ** 2
    return flux_per_pixel / pixel_area_arcsec2
```

### Decode Stamp NPZ (quality_checks.py)

```python
def decode_stamp_npz(blob: bytes) -> Tuple[np.ndarray, str]:
    def _decode(z):
        if "image_r" in z or "image_g" in z or "image_z" in z:
            bands = []
            bandset_chars = ""
            for band_key, band_char in [("image_g", "g"), ("image_r", "r"), ("image_z", "z")]:
                if band_key in z:
                    bands.append(np.asarray(z[band_key], dtype=np.float32))
                    bandset_chars += band_char
            if len(bands) == 1:
                arr = bands[0][None, :, :]
            else:
                arr = np.stack(bands, axis=0)
            return arr, bandset_chars
        # ... fallback for legacy format
```

### Hybrid Source Generator (sersic_clumps.py)

```python
def generate_hybrid_source(key: str, H: int = 96, W: int = 96, ...):
    rng = rng_from_hash(key, salt=salt)
    base = _sersic_2d(H, W, re_pix=re0, n=n0, q=q0, phi=phi, x0=x0, y0=y0)
    
    for _ in range(n_clumps):
        r = float(abs(rng.normal(loc=re0, scale=0.6 * re0)))
        ang = float(rng.uniform(0, 2 * math.pi))
        cx = x0 + r * math.cos(ang)
        cy = y0 + r * math.sin(ang)
        _add_gaussian_clump(cl, amp=frac, sigma_pix=sig, x0=cx, y0=cy)
    
    if not np.isfinite(img).all():
        raise ValueError(f"NaN/Inf detected in generated hybrid source for key={key}")
    return {"img": img.astype(np.float32), "meta": meta}
```

---

## Priority Plan Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| 1. Gen5 Validation | 3-5 days | SLACS/BELLS recall, selection function curves |
| 2. Quick Pilots | 5-7 days | Gen6/7/8 pilot models (5 epochs each) |
| 3. Full Training | 7-14 days | Full models for winning generations |
| 4. Ensemble | 5-7 days | Calibrated ensemble, analysis |
| 5. Paper | 14-21 days | MNRAS/ApJ submission |

**Total: 5-7 weeks to submission**

---

## Attached Files

1. `dhs_gen_v2_fixed.zip` - Complete code package (35KB)
2. `docs/gen6_gen7_gen8_priority_plan.md` - Detailed priority plan
3. `docs/lessons_learned_and_common_mistakes.md` - Error patterns to avoid

---

Please review carefully and answer the 15 questions above. Be direct and specific. If you see issues, say so clearly.

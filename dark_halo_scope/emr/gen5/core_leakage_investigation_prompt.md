# Core Leakage Investigation - Need Expert Input

## Context

We ran the core leakage LR gate you recommended. **It failed decisively.**

### Results

| Stratum | Core-only AUC | Expected | Status |
|---------|---------------|----------|--------|
| Overall | **0.90** | < 0.60 | FAIL |
| theta < 1" | 0.88 | Higher OK (arc at core) | Marginal |
| 1" ≤ theta < 2" | 0.85 | Should be lower | FAIL |
| theta ≥ 2" | **0.74** | Should be ~0.50 | FAIL |

### Key Finding

**Stamp core is ALWAYS brighter than ctrl core** - 100% of samples across all bricks.

```
Aggregate statistics (20 samples with theta_E >= 2"):
  Mean(stamp_core): 0.023639
  Mean(ctrl_core):  0.020565
  Mean(diff):       0.003074  (~15% offset)
  
Correlation(core_diff, arc_snr): 0.29  # LOW - not from arc flux
```

Even for large Einstein radii where the arc should be 10+ pixels from center, the stamp core is systematically brighter than ctrl core.

---

## The Question

**Is this core brightness offset:**

### (A) A Bug in the Injection Pipeline?

If the injection code incorrectly adds a constant background or doesn't properly normalize, we should fix it. Signs of a bug:
- Constant offset regardless of arc properties
- Offset not correlated with arc_snr
- Offset inconsistent with physics expectations

### (B) Physical Reality (PSF Spreading)?

If the injected arc is convolved with the PSF, light spreads from the arc into the core region. This is physically correct but creates a detection shortcut. Signs of physics:
- Offset should correlate with arc_snr (more flux → more spreading)
- Offset should decrease with larger theta_E (arc further from core)
- Can compute expected PSF contribution

### (C) Design Limitation We Must Accept?

Maybe the injection method inherently can't preserve background perfectly. We should design around it:
- Use 3-channel model (stamp only)
- Use ctrl as hard negative, not input
- Accept model will see different brightness distributions

---

## Our Data

The arc injection pipeline (from upstream Gen5):
1. Starts with real LRG image (ctrl)
2. Generates synthetic arc using lensing equation
3. Adds arc to LRG image to create stamp

**What we observed:**
- ctrl_stamp_npz = base LRG cutout from DESI coadds (fetched by us)
- stamp_npz = LRG + injected arc (from upstream injection pipeline)
- Core of stamp is ~15% brighter than core of ctrl, REGARDLESS of arc position

---

## Specific Questions

### Q1: Is 15% Core Flux Increase Physical for Arc Injection?

For a typical arc at theta_E = 2" (about 8 pixels at 0.262"/pix), PSF FWHM ~ 1.5":
- How much arc light should spread into the central 10x10 pixels?
- Is 15% increase plausible, or does this indicate a bug?

### Q2: Should Core Brightness Correlate with Arc SNR?

We see low correlation (0.29) between core brightness offset and arc_snr. 
- If PSF spreading is the cause, shouldn't brighter arcs (higher SNR) cause more core leakage?
- Does low correlation suggest a systematic (non-arc) offset?

### Q3: Should We Fix the Injection Pipeline?

Options:
- **(Fix)** Investigate and correct the injection code
- **(Normalize)** Post-hoc normalize stamp to match ctrl background
- **(Accept)** Design around it with 3-channel approach

What do you recommend and why?

### Q4: What's the Standard Practice in Lens Simulation Literature?

When injecting synthetic arcs into real images for training:
- How do researchers typically handle background consistency?
- Is this a known issue in the field?
- Any references we should consult?

---

## Our Current Plan (Pending Your Input)

1. **Pivot to 3-channel model** (stamp only, ctrl as hard negative)
2. **Per-sample normalization** to reduce brightness shortcuts
3. **Defer pipeline fix** until we understand if it's physics or bug

**Do you agree with this approach, or should we prioritize fixing the injection first?**

---

## Actual Injection Code (from spark_phase4_pipeline_gen5.py)

### Main Injection Call (line 2676):
```python
# For each band (g, r, z):
add_b = render_lensed_source(
    stamp_size=size,
    pixscale_arcsec=PIX_SCALE_ARCSEC,  # 0.262
    lens_model=lens_model_str,          # "SIS" or "SIE"
    theta_e_arcsec=theta_e,
    lens_e=lens_e_val,
    lens_phi_rad=lens_phi_rad,
    shear=shear,
    shear_phi_rad=shear_phi_rad,
    src_total_flux_nmgy=src_flux_b,
    src_reff_arcsec=src_reff,
    src_e=src_e,
    src_phi_rad=src_phi_rad,
    src_x_arcsec=src_x_arcsec,          # Random offset within ±0.4"
    src_y_arcsec=src_y_arcsec,          # Random offset within ±0.4"
    psf_fwhm_pix=psf_sigma_b * 2.355,   # Typically ~4-6 pixels
    psf_model="gaussian" or "moffat",
    moffat_beta=3.5,
    psf_apply=True,
)

# Add arc to LRG base image:
imgs[b] = (imgs[b] + add_b).astype(np.float32)
```

### render_lensed_source Function (lines 652-746):
```python
def render_lensed_source(...) -> np.ndarray:
    """Returns Float32 array of shape (stamp_size, stamp_size) in nMgy/pixel"""
    
    half = stamp_size // 2
    coords = (np.arange(stamp_size) - half + 0.5) * pixscale_arcsec
    x, y = np.meshgrid(coords, coords)
    
    # Compute deflection based on lens model
    if lens_model == "CONTROL" or theta_e_arcsec <= 0:
        return np.zeros((stamp_size, stamp_size), dtype=np.float32)  # NO ARC
    
    # SIE or SIS deflection
    if lens_model == "SIE":
        ax_l, ay_l = deflection_sie(x, y, theta_e_arcsec, lens_e, lens_phi_rad)
    else:  # SIS
        ax_l, ay_l = deflection_sis(x, y, theta_e_arcsec)
    
    # Lens equation: beta = theta - alpha
    beta_x = x - ax_l - ax_s - src_x_arcsec
    beta_y = y - ay_l - ay_s - src_y_arcsec
    
    # Evaluate Sersic profile at SOURCE PLANE coordinates
    base = sersic_profile_Ie1(beta_x, beta_y, src_reff_arcsec, q_src, src_phi_rad, n=1.0)
    
    # Normalize to intrinsic source flux
    unit_flux = sersic_unit_total_flux(src_reff_arcsec, q_src, n=1.0)
    amp_Ie = src_total_flux_nmgy / unit_flux
    
    img = base * amp_Ie * pix_area
    
    # PSF convolution (this SPREADS flux!)
    if psf_apply and psf_fwhm_pix > 0:
        img = _convolve_psf(img, psf_fwhm_pix, psf_model=psf_model, moffat_beta=moffat_beta)
    
    return img.astype(np.float32)
```

### Key Observations:
1. **Arc is rendered on source-plane coordinates** - The Sersic profile is evaluated at ray-traced positions
2. **Source is offset by ±0.4"** - But this is from the lens center, not the stamp center
3. **PSF convolution applied** - This SPREADS arc flux into neighboring pixels, including toward center
4. **No explicit background subtraction** - The arc image should theoretically be zero where no source light lands

### Potential Sources of Core Leakage:
1. **Extended Sersic wings** - n=1 Sersic (exponential) has infinite extent, may have non-zero flux at center
2. **PSF spreading** - Even if arc is at theta_E, PSF convolution spreads light everywhere
3. **Lens magnification at center** - The lens equation may map some source light to the core region
4. **Numerical precision** - Small non-zero values accumulating

### What We DON'T See:
- No explicit background pedestal added
- No sky level added to arc
- CONTROL samples return zeros (line 707)

**The question remains:** Is the ~15% core brightness increase from PSF spreading (physics) or from something else (bug)?

---

## Q5: Can You Devise Specific Tests to Distinguish Bug vs Physics?

We have access to the data and can run diagnostic scripts. **What specific tests would definitively distinguish:**

### Hypothesis A: Bug (Constant Background Offset)

Expected signatures:
- Core offset is CONSTANT regardless of arc properties
- No dependence on theta_E, arc_snr, or arc flux
- Offset might be band-dependent (different calibration per band)

### Hypothesis B: Physics (PSF Spreading)

Expected signatures:
- Core offset scales with total arc flux
- Core offset decreases with larger theta_E (arc further from center)
- Offset follows PSF profile (decays as 1/r² or Moffat profile)

### What We Can Measure

We have for each sample:
- `stamp_npz`: 3-band image with arc
- `ctrl_stamp_npz`: 3-band base LRG (no arc)
- `theta_e_arcsec`: Einstein radius
- `arc_snr`: arc signal-to-noise ratio
- `brickname`, `ra`, `dec`: location metadata

**Please suggest 2-3 specific diagnostic tests with:**
1. What to compute
2. What result indicates BUG
3. What result indicates PHYSICS

For example:
- "Compute core_offset vs theta_E regression. If slope ≈ 0, it's a bug. If negative slope, it's physics."
- "Compute core_offset vs arc_total_flux. If R² < 0.1, it's a bug. If R² > 0.5, it's physics."

---

## Summary

We need to know:
1. Is this **physics** (PSF spreading) or **bug** (background offset)?
2. If physics, is 15% core increase plausible for typical arc parameters?
3. Should we fix the pipeline, or design around it?
4. **What specific tests can we run to make the distinction?**

Your expertise on lens simulation best practices would be very valuable here.

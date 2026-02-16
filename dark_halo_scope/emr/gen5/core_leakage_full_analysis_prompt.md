# Core Leakage Analysis - Complete Context for Decision

## Source Files (for verification)

All code is in the repository and can be inspected:

**Phase 4 Pipeline (injection code):**
- `dark_halo_scope/emr/gen5/spark_phase4_pipeline_gen5.py` (3272 lines)
  - `render_lensed_source()`: lines 652-746
  - `sersic_profile_Ie1()`: lines 535-570
  - Arc addition to LRG: lines 2655-2676

**Diagnostic Scripts (run on emr-launcher):**
- `dark_halo_scope/emr/gen5/diagnostic_test1_outer_annulus.py`
- `dark_halo_scope/emr/gen5/diagnostic_test2_regression.py`
- `dark_halo_scope/emr/gen5/diagnostic_test3_stratified.py`
- `dark_halo_scope/emr/gen5/diagnostic_test4_radial_profile.py`

**Data Source:**
- S3: `s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_paired/train/`

---

## Background

We're building a gravitational lens detector. The training data consists of:
- **stamp_npz**: Real LRG image + synthetically injected lensed arc
- **ctrl_stamp_npz**: Base LRG image (no arc) - fetched from DESI coadd cache

We ran a **core leakage LR gate**: train a logistic regression on only the central 10×10 pixels to distinguish stamp from ctrl. **It achieved AUC = 0.90**, indicating the core region alone can distinguish positive from control.

**The question:** Is this physics (PSF spreading of arc light) or a bug (systematic mismatch between stamp and ctrl)?

---

## Phase 4 Injection Pipeline Code

### Main Injection Function: `render_lensed_source()`

```python
def render_lensed_source(
    stamp_size: int,
    pixscale_arcsec: float,
    lens_model: str,  # "SIS", "SIE", or "CONTROL"
    theta_e_arcsec: float,
    lens_e: float,
    lens_phi_rad: float,
    shear: float,
    shear_phi_rad: float,
    src_total_flux_nmgy: float,
    src_reff_arcsec: float,
    src_e: float,
    src_phi_rad: float,
    src_x_arcsec: float,
    src_y_arcsec: float,
    psf_fwhm_pix: float,
    psf_model: str = "gaussian",
    moffat_beta: float = 3.5,
    psf_apply: bool = True,
) -> np.ndarray:
    """
    Render a lensed source image with correct magnification behavior.
    
    CRITICAL: This function normalizes the UNLENSED source analytically,
    then applies lensing. The observed (lensed) flux will be HIGHER than
    the intrinsic source flux due to gravitational magnification.
    
    Returns:
        Float32 array of shape (stamp_size, stamp_size) in nMgy/pixel
    """
    half = stamp_size // 2
    coords = (np.arange(stamp_size) - half + 0.5) * pixscale_arcsec
    x, y = np.meshgrid(coords, coords)
    
    # Compute deflection based on lens model
    if lens_model == "CONTROL" or theta_e_arcsec <= 0:
        # No lensing - return zeros (control sample)
        return np.zeros((stamp_size, stamp_size), dtype=np.float32)
    
    if lens_model == "SIE":
        ax_l, ay_l = deflection_sie(x, y, theta_e_arcsec, lens_e, lens_phi_rad)
    else:  # SIS
        ax_l, ay_l = deflection_sis(x, y, theta_e_arcsec)
    
    # Add external shear deflection
    if abs(shear) > 1e-6:
        ax_s, ay_s = deflection_shear(x, y, shear, shear_phi_rad)
    else:
        ax_s, ay_s = 0.0, 0.0
    
    # Lens equation: beta = theta - alpha
    beta_x = x - ax_l - ax_s - src_x_arcsec
    beta_y = y - ay_l - ay_s - src_y_arcsec
    
    # Source axis ratio from ellipticity
    q_src = float(np.clip((1.0 - src_e) / (1.0 + src_e + 1e-6), 0.2, 0.999))
    
    # Evaluate Sersic profile (I(Re) = 1) at SOURCE PLANE coordinates
    base = sersic_profile_Ie1(beta_x, beta_y, src_reff_arcsec, q_src, src_phi_rad, n=1.0)
    
    # Compute amplitude to give correct INTRINSIC total flux
    unit_flux = sersic_unit_total_flux(src_reff_arcsec, q_src, n=1.0) + 1e-30
    amp_Ie = src_total_flux_nmgy / unit_flux
    
    # Pixel area for conversion to flux/pixel
    pix_area = pixscale_arcsec ** 2
    
    # Surface brightness in nMgy/pixel
    img = base * amp_Ie * pix_area
    
    # PSF convolution
    if psf_apply and psf_fwhm_pix > 0:
        img = _convolve_psf(img, psf_fwhm_pix, psf_model=psf_model, moffat_beta=moffat_beta)
    
    return img.astype(np.float32)
```

### Sersic Profile Function

```python
def sersic_profile_Ie1(beta_x: np.ndarray, beta_y: np.ndarray, 
                       reff_arcsec: float, q: float, phi_rad: float, 
                       n: float = 1.0) -> np.ndarray:
    """
    Evaluate Sersic surface brightness profile with I(Re) = 1.
    
    The profile is evaluated in the source plane (beta coordinates).
    n=1.0 = exponential profile (default used in injection)
    """
    # Rotate to align with major axis
    c = math.cos(phi_rad)
    s = math.sin(phi_rad)
    xp = c * beta_x + s * beta_y
    yp = -s * beta_x + c * beta_y
    
    # Elliptical radius
    q = float(np.clip(q, 0.1, 1.0))
    R = np.sqrt(xp**2 + (yp / q)**2 + 1e-18)
    
    # Sersic profile: I(R) = I_e * exp(-b_n * [(R/R_e)^(1/n) - 1])
    b = sersic_bn(n)
    return np.exp(-b * ((R / reff_arcsec) ** (1.0 / n) - 1.0))
```

### How Arc Gets Added to LRG (lines 2655-2676)

```python
# For each band (g, r, z):
add_b = render_lensed_source(
    stamp_size=size,
    pixscale_arcsec=PIX_SCALE_ARCSEC,  # 0.262 arcsec/pixel
    lens_model=lens_model_str,          # "SIS" or "SIE"
    theta_e_arcsec=theta_e,
    lens_e=lens_e_val,
    lens_phi_rad=lens_phi_rad,
    shear=shear,
    shear_phi_rad=shear_phi_rad,
    src_total_flux_nmgy=src_flux_b,
    src_reff_arcsec=src_reff,           # Source effective radius
    src_e=src_e,
    src_phi_rad=src_phi_rad,
    src_x_arcsec=src_x_arcsec,          # Random offset ±0.4"
    src_y_arcsec=src_y_arcsec,
    psf_fwhm_pix=psf_sigma_b * 2.355,   # Typically 4-6 pixels
    psf_model=args.psf_model,           # "gaussian" or "moffat"
    moffat_beta=args.moffat_beta,
    psf_apply=True,
)

# Simple addition - no background adjustment
imgs[b] = (imgs[b] + add_b).astype(np.float32)
```

---

## Diagnostic Tests We Ran

### Test 1: Outer Annulus Pedestal Check

**Purpose:** If there's a systematic calibration mismatch, the outer annulus (far from any arc) should also be offset.

```python
# Compute for each sample:
diff = stamp - ctrl

# Outer annulus: r >= 20 pixels (far from any arc at theta_E < 3")
outer_mask = radial_mask(diff.shape, 20, 32)
outer_mean = np.mean(diff[outer_mask])

# Core: r < 5 pixels
core_mask = radial_mask(diff.shape, 0, 5)
core_mean = np.mean(diff[core_mask])
```

**Results:**

| Region | Mean offset | Interpretation |
|--------|-------------|----------------|
| Outer (r≥20 pix) | **0.00004** | ~zero |
| Core (r<5 pix) | **0.015** | significant |
| Ratio | 363x | offset concentrated in core |

**Conclusion:** Outer annulus is essentially zero. This is NOT a systematic calibration mismatch.

---

### Test 2: Core Offset Regression Analysis

**Purpose:** Check if core offset scales with arc properties (physics) or is constant (bug).

```python
from scipy import stats

# For each sample, compute:
theta_e = row['theta_e_arcsec']
arc_snr = row['arc_snr']
total_arc_flux = np.sum(stamp - ctrl)
core_offset = np.mean(stamp_core) - np.mean(ctrl_core)

# Test correlations:
slope_theta, _, r_theta, _, _ = stats.linregress(theta_e, core_offset)
slope_flux, _, r_flux, _, _ = stats.linregress(total_arc_flux, core_offset)
slope_snr, _, r_snr, _, _ = stats.linregress(arc_snr, core_offset)
```

**Results:**

| Predictor | Slope | R² | p-value | Interpretation |
|-----------|-------|-----|---------|----------------|
| theta_E | -0.006 | 0.10 | <0.0001 | Negative (larger θE → less core offset) |
| Total arc flux | +0.0008 | **0.52** | <0.0001 | Strong positive correlation |
| arc_snr | +0.0005 | **0.40** | <0.0001 | Positive correlation |

**Conclusion:** Core offset scales with arc properties, not constant. This suggests physics.

---

### Test 3: Stratified Analysis by theta_E

**Purpose:** Check if core leakage decreases with larger Einstein radius.

```python
# Group samples by theta_E bins:
for low, high in [(0, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 5.0)]:
    subset = [d for d in data if low <= d['theta_e'] < high]
    
    # For each subset, compute:
    core_diff_pct = 100 * np.sum(diff[core_mask]) / np.sum(ctrl[core_mask])
    leakage_pct = 100 * np.sum(diff[core_mask]) / np.sum(diff)
```

**Results:**

| theta_E bin | Arc radius (pix) | Core diff (% of ctrl) | Arc flux in core |
|-------------|------------------|----------------------|------------------|
| < 0.75" | 1.9 | 69% | 41% |
| 0.75-1" | 2.9 | 57% | 33% |
| 1-1.5" | 4.3 | 51% | 23% |
| 1.5-2" | 5.7 | 43% | 16% |
| ≥ 2" | 8.6 | 22% | **7.7%** |

**Conclusion:** Clear trend - larger theta_E → less core leakage. This is physics.

---

### Test 4: Radial Profile Analysis (theta_E ≥ 2" only)

**Purpose:** For samples where arc should be at r~8.6 pix, what does the radial profile look like?

```python
# For samples with theta_E >= 2":
diff = stamp_r - ctrl_r

# Compute azimuthal radial profile:
for ri in range(25):
    mask = (r >= ri) & (r < ri + 1)
    profile[ri] = np.mean(diff[mask])
```

**Results (mean over 90 samples with theta_E ≥ 2"):**

```
Radius | Mean Diff | Expected for ring
-------|-----------|------------------
  0    | 0.0049    | ~0 (no arc here)
  1    | 0.0050    | ~0
  2    | 0.0053    | ~0
  3    | 0.0056    | ~0
  4    | 0.0060    | ~0
  5    | 0.0064    | ~0
  6    | 0.0066    | rising
  7    | 0.0067    | PEAK (arc at r=8.6)
  8    | 0.0066    | PEAK
  9    | 0.0064    | PEAK
 10    | 0.0061    | declining
 11    | 0.0057    | 
 12    | 0.0052    | 
 15    | 0.0032    | ~0
 17    | 0.0019    | ~0
```

**Critical observation:** The profile is **surprisingly flat**, not peaked at theta_E.
- Core (r=0-4) has ~75% of the peak flux
- This is NOT a thin ring at r=8.6 pixels
- The light distribution is broad and extends to the center

---

## Your Earlier Analysis vs Our Data

### Your Gaussian PSF Estimate

You calculated:
- PSF FWHM ≈ 1.5" → σ ≈ 2.43 pix
- Arc at r₀ = 8 pix
- exp(-r₀²/(2σ²)) ≈ exp(-5.6) ≈ **0.4%** leakage

### What We Observe for theta_E ≥ 2"

- Arc flux in core: **7.7%** (not 0.4%)
- Core/peak ratio: **75%** (profile is flat, not peaked)

### Discrepancy Explanation

The injection uses **Sersic n=1 (exponential) source profile**, which:
1. Has infinite extent in the source plane (extended wings)
2. When ray-traced through the lens equation, source wings map to various image-plane radii
3. The lensed image is NOT a thin ring, but a **diffuse extended structure**

The radial profile confirms this: the "arc" is not concentrated at theta_E, it's spread across the entire stamp.

---

## Key Question

Is the core leakage:

### (A) A Bug - Systematic Mismatch

Evidence AGAINST this:
- Outer annulus is ~zero (0.00004 vs 0.015 in core)
- Core offset scales with arc flux (R² = 0.52)
- Core offset decreases with larger theta_E (slope = -0.006)

### (B) Simple PSF Spreading

Evidence AGAINST this:
- 7.7% leakage at r=8.6 pix >> 0.4% expected for Gaussian PSF
- Radial profile is flat, not peaked at theta_E

### (C) Extended Source + Lens Physics

Evidence FOR this:
- Sersic n=1 has extended wings in source plane
- Lens equation maps source wings to various image radii
- Radial profile is diffuse, not ring-like
- Leakage scales with arc properties (not constant)

---

## Summary of Evidence

| Test | Result | Bug? | Simple PSF? | Extended Source? |
|------|--------|------|-------------|------------------|
| Outer annulus offset | ~zero | ✗ | ? | ✓ |
| Core offset vs flux | R²=0.52 | ✗ | ✓ | ✓ |
| Core offset vs theta_E | negative slope | ✗ | ✓ | ✓ |
| Leakage at theta_E=2" | 7.7% | ? | ✗ | ✓ |
| Radial profile | flat, diffuse | ✗ | ✗ | ✓ |

**Our conclusion:** The core leakage is **real physics** from lensing an extended Sersic source, NOT a pipeline bug. However, it still creates a training shortcut.

---

## Questions for You

1. **Do you agree with our analysis?** Specifically:
   - Does the near-zero outer annulus rule out systematic mismatch?
   - Does the flat radial profile indicate extended source physics?

2. **Is the injection physics correct?** 
   - Should a Sersic n=1 source produce such diffuse lensed images?
   - Is this realistic for lensed galaxies, or are we using too extended a source profile?

3. **Should we proceed with 3-channel approach?**
   - Even if it's physics, the shortcut is real (model can exploit core brightness)
   - Using ctrl as hard negative (not input) avoids this

4. **Alternative solutions?**
   - Use point sources or more compact profiles?
   - Mask the core region during training?
   - Add source morphology diversity?

---

## Data Context

- **Pixel scale:** 0.262 arcsec/pixel
- **Stamp size:** 64×64 pixels
- **PSF FWHM:** ~1.2-1.5 arcsec (4-6 pixels)
- **PSF model:** Gaussian or Moffat (β=3.5)
- **Source profile:** Sersic n=1 (exponential)
- **Source reff:** typically 0.2-0.5 arcsec
- **theta_E range:** 0.5-2.5 arcsec

Please help us make the GO/NO-GO decision for training with current data vs investigating further.

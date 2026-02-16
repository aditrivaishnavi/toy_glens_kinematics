# Test A Results: Arc-Only Core Fraction

Per your recommendation, I implemented the decisive test to separate physics vs mismatch.

## Methodology

For 300 samples:
1. Re-render the arc (add_b) from stored injection parameters using CORRECT PSF convolution
2. Compute core_frac_arc_only = sum(arc_only[core]) / sum(arc_only[all])
3. Compute core_frac_diff = sum((stamp-ctrl)[core]) / sum((stamp-ctrl)[all])
4. Compare

Core defined as r < 5 pixels from center.

## Code

```python
def render_arc_only(stamp_size, theta_e_arcsec, src_reff_arcsec, src_e, 
                    src_phi_rad, src_x_arcsec, src_y_arcsec, psf_fwhm_arcsec):
    """Re-render lensed arc from stored parameters."""
    half = stamp_size // 2
    pix_idx = np.arange(stamp_size) - half + 0.5
    y_grid, x_grid = np.meshgrid(pix_idx, pix_idx, indexing="ij")
    x = x_grid * PIX_SCALE_ARCSEC
    y = y_grid * PIX_SCALE_ARCSEC
    
    ax, ay = deflection_sis(x, y, theta_e_arcsec)
    beta_x = x - ax - src_x_arcsec
    beta_y = y - ay - src_y_arcsec
    
    q_src = (1.0 - src_e) / (1.0 + src_e + 1e-6)
    base = sersic_profile_Ie1(beta_x, beta_y, src_reff_arcsec, q_src, src_phi_rad, n=1.0)
    img = base * (PIX_SCALE_ARCSEC ** 2)
    
    if psf_fwhm_arcsec > 0:
        psf_fwhm_pix = psf_fwhm_arcsec / PIX_SCALE_ARCSEC
        sigma_pix = psf_fwhm_pix / 2.355
        k = _gaussian_kernel2d(sigma_pix)
        img = _fft_convolve2d_correct(img, k)  # Using FIXED convolution
    
    return img

def compute_core_fraction(img, core_radius=5):
    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    core_mask = r < core_radius
    total_flux = np.sum(np.abs(img))
    core_flux = np.sum(np.abs(img[core_mask]))
    return core_flux / total_flux if total_flux > 0 else 0.0
```

## Results

### Overall (N=300)

| Metric | Arc-Only (re-rendered) | Stamp - Ctrl |
|--------|------------------------|--------------|
| Mean   | 0.329 | 0.233 |
| Median | 0.240 | 0.214 |

### Stratified by θ_E

| θ_E bin | N | Arc-Only | Diff |
|---------|---|----------|------|
| < 0.75" | 57 | 0.716 | 0.411 |
| 0.75-1" | 42 | 0.518 | 0.332 |
| 1-1.5" | 82 | 0.293 | 0.237 |
| 1.5-2" | 41 | 0.148 | 0.168 |
| ≥ 2" | 78 | 0.079 | 0.080 |

---

## PSF Bug Fix

Applied the fix you recommended:

```python
# BEFORE (buggy):
pad = np.fft.ifftshift(pad)

# AFTER (fixed):
oy, ox = kh // 2, kw // 2
pad = np.roll(np.roll(pad, -oy, axis=0), -ox, axis=1)
```

### Verification

```
Impulse at: (32, 32)
Peak after FIXED convolution: (32, 32)
PASS: Fix verified
```

---

## Questions

1. For θ_E ≥ 2", arc_only = 0.079 and diff = 0.080 are nearly identical. What does this indicate?

2. For θ_E < 1", arc_only (0.52-0.72) is higher than diff (0.33-0.41). What could explain this difference?

3. Based on these Test A results, what is your assessment: physics vs mismatch?

4. Should we proceed with training? If so, what additional gates or ablations do you recommend?

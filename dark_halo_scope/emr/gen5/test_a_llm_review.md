# Test A Results - Request for Interpretation

You previously recommended "Test A: Arc-only core fraction" as the decisive test to separate physics vs mismatch. I implemented and ran it.

## Methodology

For 300 samples:
1. Re-render the lensed arc from stored injection parameters (theta_e, src_reff, src_e, src_phi, src_x, src_y, psf_fwhm)
2. Use SIS lens model and Sersic n=1 source profile
3. Apply PSF convolution using the FIXED (roll-based) implementation
4. Compute core_frac = sum(image[r < 5]) / sum(image[all])
5. Compare arc-only to (stamp - ctrl)

## Code Used

```python
def render_arc_only(stamp_size, theta_e_arcsec, src_reff_arcsec, src_e, 
                    src_phi_rad, src_x_arcsec, src_y_arcsec, psf_fwhm_arcsec):
    half = stamp_size // 2
    pix_idx = np.arange(stamp_size) - half + 0.5
    y_grid, x_grid = np.meshgrid(pix_idx, pix_idx, indexing="ij")
    x = x_grid * PIX_SCALE_ARCSEC  # 0.262
    y = y_grid * PIX_SCALE_ARCSEC
    
    # SIS deflection
    ax, ay = deflection_sis(x, y, theta_e_arcsec)
    beta_x = x - ax - src_x_arcsec
    beta_y = y - ay - src_y_arcsec
    
    # Sersic n=1 source
    q_src = (1.0 - src_e) / (1.0 + src_e + 1e-6)
    base = sersic_profile_Ie1(beta_x, beta_y, src_reff_arcsec, q_src, src_phi_rad, n=1.0)
    img = base * (PIX_SCALE_ARCSEC ** 2)
    
    # PSF convolution (FIXED version)
    if psf_fwhm_arcsec > 0:
        psf_fwhm_pix = psf_fwhm_arcsec / PIX_SCALE_ARCSEC
        sigma_pix = psf_fwhm_pix / 2.355
        k = _gaussian_kernel2d(sigma_pix)
        img = _fft_convolve2d_correct(img, k)
    
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

## Raw Results

### Overall (N=300)

| Metric | Arc-Only (re-rendered) | Stamp - Ctrl |
|--------|------------------------|--------------|
| Mean   | 0.329 | 0.233 |
| Median | 0.240 | 0.214 |
| Std    | 0.256 | 0.131 |

### Stratified by θ_E

| θ_E bin | N | Arc-Only | Diff |
|---------|---|----------|------|
| < 0.75" | 57 | 0.716 | 0.411 |
| 0.75-1" | 42 | 0.518 | 0.332 |
| 1-1.5" | 82 | 0.293 | 0.237 |
| 1.5-2" | 41 | 0.148 | 0.168 |
| ≥ 2" | 78 | 0.079 | 0.080 |

---

## PSF Bug Status

The bug you identified in `_fft_convolve2d` has been fixed:

```python
# BEFORE (buggy):
pad = np.fft.ifftshift(pad)

# AFTER (fixed):
oy, ox = kh // 2, kw // 2
pad = np.roll(np.roll(pad, -oy, axis=0), -ox, axis=1)
```

Verification:
```
Impulse at: (32, 32)
Peak after FIXED convolution: (32, 32)
```

The re-rendering for Test A used the FIXED convolution.

---

## Questions

1. **For θ_E ≥ 2"**: Arc-only = 0.079, Diff = 0.080. These are nearly identical. What is your interpretation?

2. **For θ_E < 1"**: Arc-only (0.52-0.72) is significantly higher than Diff (0.33-0.41). What could explain this discrepancy?

3. **Overall**: Arc-only mean (0.329) is higher than Diff mean (0.233). Is this expected or concerning?

4. **Does this test resolve physics vs mismatch?** Or do I need additional tests?

5. **What is your recommendation for training?** Given these results, should I:
   - Proceed with 3-channel (stamp only)?
   - Run additional diagnostics first?
   - Something else?

6. **Are there flaws in my Test A implementation** that could explain the discrepancy at small θ_E?

---

## Attached Files

- `test_arc_only_core_fraction.py` - Full implementation
- `spark_phase4_pipeline_gen5.py` - Pipeline with PSF fix applied

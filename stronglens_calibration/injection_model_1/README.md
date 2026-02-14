# Injection Model 1: Parametric Sersic Source on Random Non-Lens Hosts

**Status:** Complete. Validated. Archived.
**Date:** 2026-02-13
**Superseded by:** Injection Model 2 (deflector-matched hosts) and Model 3 (deflector-matched + real HST sources) — in development.

---

## Summary of Approach

This was our first-generation injection pipeline for measuring the CNN selection function. It is a self-contained, analytically correct implementation, but uses **simplified source morphology** and **random (non-deflector-matched) host galaxies**, which places it below the current state of the art set by HOLISMOKES (Cañameras et al. 2024, A&A 692, A72) and Euclid (2025).

### Key Parameters and Mechanisms

| Parameter | Value / Mechanism | Notes |
|-----------|-------------------|-------|
| **Lens model** | SIE (Kormann et al. 1994) + external shear | q_lens in [0.5, 1.0], gamma in [0, 0.1] |
| **Source model** | Single Sersic profile + optional Gaussian clumps | n_sersic in [0.5, 4.0], R_e in [0.1, 0.5] arcsec |
| **Source flux** | Total unlensed flux in nanomaggies | AB ZP = 22.5 throughout |
| **Source magnitudes** | r-band mag 23-26 (main grid), 18-26 (bright arc test) | Uniformly sampled per mag bin |
| **Source colors** | g-r in [0.0, 1.5], r-z in [-0.3, 1.0] | Random, not SED-based |
| **Source offset** | Area-weighted: P(beta_frac) prop to beta_frac | beta = beta_frac * theta_E, beta_frac in [0.1, 1.0] |
| **Magnification** | Physical, from SIE ray-tracing | Not a free parameter; emerges from lensing |
| **PSF** | Gaussian, per-band, from psfsize_r | sigma = FWHM / 2.355 |
| **PSF convolution** | FFT (full-stamp) | Verified to preserve total flux to <1% |
| **Noise estimation** | Gaussian from psfdepth_r | sigma_pix = 1/sqrt(psfdepth * n_pix_per_psf) |
| **Sub-pixel oversampling** | 4x (default) | Convergent to <0.5% vs 8x |
| **Core suppression** | Optional radial masking of central pixels | Tested but not used in main grid |
| **Host galaxies** | Random non-lens galaxies from validation split | NOT deflector-matched |
| **Host conditioning** | Matched to PSF/depth bin from manifest | psfsize_r and psfdepth_r |
| **Grid** | 11 theta_E x 7 PSF x 4 depth = 308 cells | 200 injections/cell = 61,600 total |
| **Detection thresholds** | p>0.3, p>0.5, FPR=0.1%, FPR=0.01% | FPR-derived preferred |
| **Confidence intervals** | Bayesian binomial, Jeffreys prior | Beta(0.5, 0.5) |
| **Pixel scale** | 0.262 arcsec/pixel | DR10 Legacy Survey |
| **Stamp size** | 101 x 101 pixels | g, r, z bands |
| **Preprocessing** | raw_robust (outer-annulus median/MAD, clip [-10,10]) | Identical to training |

### Known Limitations (motivating Model 2 and Model 3)

1. **Random hosts, not deflector-matched.** The model learned a joint host+arc feature during training on real lenses. Injecting onto random non-lens hosts breaks this correlation, producing a conservative (low) completeness estimate. State of the art (HOLISMOKES) injects onto actual LRGs with known velocity dispersions.

2. **Parametric Sersic source.** Real lensed sources have complex morphology (clumps, mergers, irregular structure). State of the art (HOLISMOKES) uses real HST/HUDF galaxy images ray-traced through the lens model.

3. **Gaussian PSF approximation.** DR10 uses per-brick spatially-varying PSF models. We approximate with a Gaussian from the scalar psfsize_r. State of the art uses per-position survey PSF models.

4. **No per-pixel noise model.** We estimate noise from psfdepth_r assuming independent pixels. Real coadd images have correlated noise.

5. **Random source colors.** Not drawn from a physical SED model. State of the art draws source redshifts and applies color corrections.

---

## Directory Structure

```
injection_model_1/
├── README.md                              # This file
├── INJECTION_MODEL_1_VALIDATION.md        # Full validation report with all results
├── engine/
│   ├── injection_engine.py                # SIE + Sersic ray-tracing, FFT PSF, flux calibration (694 lines)
│   └── selection_function_utils.py        # Bayesian binomial CI, depth conversion (68 lines)
├── scripts/
│   ├── selection_function_grid.py         # Main 308-cell injection-recovery grid (722 lines)
│   ├── sensitivity_analysis.py            # 8-perturbation systematic uncertainty (297 lines)
│   ├── validate_injections.py             # Runtime QA: flux conservation, visual pages (650 lines)
│   └── sim_to_real_validation.py          # Comprehensive sim-to-real diagnostic (615 lines)
├── validation/
│   ├── README.md                          # Validation suite overview
│   ├── real_lens_scoring.py               # Score real lenses for recall/FPR (337 lines)
│   ├── confuser_morphology_test.py        # Morphology shortcut test (330 lines)
│   ├── bright_arc_injection_test.py       # Brightness-dependent completeness (330 lines)
│   └── sim_to_real_validation.py          # Comprehensive sim-to-real diagnostic (615 lines)
└── tests/
    └── test_injection_engine.py           # 28 unit/physics tests, lenstronomy cross-validated (713 lines)
```

## Key Results

| Metric | Value |
|--------|-------|
| Overall completeness (p>0.3, all mag) | **4.3%** |
| Peak completeness (theta_E=1.75") | **5.7%** |
| Real lens recall (p>0.3) | **73.3%** |
| Completeness gap (recall - injection) | **~69 pp** |
| Bright-arc ceiling (mag 18-19, p>0.3) | **30.5%** |
| Residual gap above brightness | **~43 pp** (host context) |
| Sensitivity max |dC| | **<5 pp** across all 8 perturbations |
| Confuser FPR | **<1%** all categories (baseline-consistent) |
| Physics tests | **28/28 pass, 0 skipped** |
| lenstronomy cross-validation | **<0.1% deflection, <1% flux** |

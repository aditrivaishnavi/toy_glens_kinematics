# Phase 1 Parameters: Analytic Observability Window

*Documentation for the physical and survey parameters used in Phase 1 of Dark Halo Scope.*

---

## Overview

Phase 1 builds an **analytic observability window** for galaxy-scale strong lenses in DESI Legacy DR10. This document describes all parameters used, their sources, and their justification.

---

## 1. Survey / Instrument Parameters

These are **standard values** from DECam and DR10 documentation.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `pixel_scale_arcsec` | 0.262 | DECam pixel scale in arcsec/pixel |
| `seeing_fwhm_g` | 1.35" | Typical g-band PSF FWHM |
| `seeing_fwhm_r` | 1.25" | Typical r-band PSF FWHM (primary lensing band) |
| `seeing_fwhm_z` | 1.30" | Typical z-band PSF FWHM |
| `depth_g_5sigma` | 24.5 mag | 5σ point-source depth in g |
| `depth_r_5sigma` | 23.8 mag | 5σ point-source depth in r |
| `depth_z_5sigma` | 23.0 mag | 5σ point-source depth in z |

### Discussion

- **Pixel scale**: DECam's pixel scale is ≈0.262–0.263 arcsec/pixel, which is what the Legacy Surveys use in their coadds.

- **Seeing**: Dey et al. (2019) and subsequent Legacy Survey characterizations give median PSF FWHM values around 1.1–1.4 arcsec in the optical bands, so 1.25" in r is a reasonable representative value.

- **Depths**: Values like g≈24.5, r≈23.8, z≈23.0 mag are consistent with the DESI Legacy Imaging Surveys performance in DECaLS fields.

### References

- Dey, A., et al. (2019). "Overview of the DESI Legacy Imaging Surveys." *AJ*, 157, 168. [arXiv:1804.08657](https://arxiv.org/abs/1804.08657)
- DESI Legacy Imaging Surveys DR10 Documentation: https://www.legacysurvey.org/dr10/

---

## 2. Cosmology

| Parameter | Value | Description |
|-----------|-------|-------------|
| `h0` | **67.4** km/s/Mpc | Hubble constant (Planck 2018) |
| `omega_m` | **0.315** | Matter density parameter (Planck 2018) |

### Discussion

We adopt the **Planck 2018** flat ΛCDM cosmology with H₀ = 67.4 km s⁻¹ Mpc⁻¹ and Ωₘ = 0.315.

This is the current standard cosmology from CMB observations and ensures our mass estimates are directly comparable to other recent lensing studies.

### References

- Planck Collaboration (2018). "Planck 2018 results. VI. Cosmological parameters." *A&A*, 641, A6. [arXiv:1807.06209](https://arxiv.org/abs/1807.06209)

---

## 3. Detection Thresholds

| Parameter | Value | Resulting Threshold |
|-----------|-------|---------------------|
| `k_blind` | 0.5 | θ_E < 0.625" → Blind |
| `k_good` | 1.0 | θ_E ≥ 1.25" → Good |

### Classification Scheme

```
Detectability Class    Condition                  Physical Meaning
─────────────────────────────────────────────────────────────────────
Blind                  θ_E < k_blind × FWHM       Arc structure destroyed in PSF core
Low Trust              k_blind × FWHM ≤ θ_E       Marginal detection, blended with lens
                       < k_good × FWHM
Good                   θ_E ≥ k_good × FWHM        Clear tangential distortions visible
```

### Discussion

These are **not universal constants**; they are physically motivated modeling choices:

- **θ_E < 0.5 × FWHM**: Essentially unresolved in the PSF core, so any arc structure is destroyed.

- **θ_E ≈ 1 × FWHM**: You start to see clearly tangential distortions; above this, lens morphology is visually and algorithmically detectable.

Different groups pick slightly different numbers (some use 0.5×FWHM as a hard lower bound and 1.5×FWHM as a comfortable detection regime), but our values are firmly within the "reasonable" range.

### Key Points

1. Present them explicitly as **analytic classification thresholds**
2. Later test them against injection–recovery results (Phase 4–6) to see whether, for example, "completeness drops sharply below ~1.1–1.3 arcsec," which would support or refine k_blind and k_good

**Status**: Not industry standards, but **defensible starting assumptions** that we will empirically calibrate.

---

## 4. Redshift Assumptions

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lens_z_range` | (0.1, 1.0) | Lens redshift grid range |
| `source_z_min` | 1.5 | Minimum source redshift considered |
| `source_z_max` | 2.5 | Maximum source redshift considered |
| `representative_source_z` | 2.0 | Fixed z_s for θ_E → Mass conversion |

### Discussion

**Lens redshift grid (0.1–1.0)**:
- Galaxy–galaxy lenses in DESI/Legacy tend to sit around z_l ≈ 0.2–0.8
- Exploring 0.1–1.0 is perfectly reasonable and captures the full range

**Source redshift fixed at z_s = 2.0**:
- For galaxy–galaxy lenses in ground-based surveys, z_s ~ 1–3 is typical
- Median around z_s = 2 for star-forming sources
- Using z_s = 2 as a representative value for the analytic map is **standard practice**

### Important Caveats

We must clearly state that:
1. Mass estimates depend on z_s
2. In the analytic phase we are mapping θ_E and a representative mass range, not precise halo masses for specific objects

Later, for real lenses, we will either work in θ_E-space only or fold in a realistic prior for P(z_s).

---

## 5. Grid Resolution

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_z_grid` | 100 | Number of lens redshift grid points |
| `n_theta_grid` | 150 | Number of Einstein radius grid points |

### Discussion

These are purely **numerical choices**. They are already fine for smooth maps; we are not doing precision parameter estimation here, just exploring the functional dependence.

The resulting grids have:
- z_l resolution: Δz ≈ 0.009
- θ_E resolution: Δθ_E ≈ 0.019"

This is more than sufficient for the diagnostic plots we produce.

---

## 6. Interpreting Phase 1 Plots

### θ_E vs z_l Detectability Plot (`phase1_thetaE_window.png`)

As expected from the way we defined k_blind/k_good, the "blind / low-trust / good" regions are **horizontal bands in θ_E**, independent of z_l. This is correct: seeing-limited resolvability is set by angular size, not by redshift directly.

Key features:
- **Blind zone**: Below ≈0.625" (θ_E < 0.5 × 1.25")
- **Low trust zone**: 0.625" – 1.25"
- **Good zone**: Above ≈1.25" (θ_E ≥ 1.0 × 1.25")

These look physically reasonable for DR10-quality ground-based imaging.

### M(<θ_E) vs z_l Observability Window (`phase1_mass_window.png`)

The transition from "blind" to "good" around log₁₀ M ≈ 11.1–11.4 at moderate redshift is consistent with expectations:
- Galaxy-scale lenses with θ_E ≈ 1–2" have enclosed masses of order a few ×10¹¹ M⊙ inside θ_E
- This applies for z_l ~ 0.3–0.7 and z_s ~ 2

**Physical interpretation**: The "observable" region is essentially "high-mass halos at a range of z_l" — this is exactly the physics story we want. DR10 is a window onto the **massive end of the halo population**.

### Visibility Ratio Plot (`phase1_visibility_ratio.png`)

A 1D diagram showing detectability as a function of θ_E/FWHM with the analytic thresholds clearly marked:
- k_blind = 0.5: Below this, arcs are unresolved in the PSF core
- k_good = 1.0: Above this, ring/arc structure is clearly resolvable

This plot emphasizes that these are **analytic thresholds to be calibrated** with injection–recovery tests.

### Seeing Sensitivity Plot (`phase1_seeing_sensitivity.png`)

Shows how detection thresholds shift with seeing FWHM:

| FWHM | θ_blind | θ_good |
|------|---------|--------|
| 1.0" | 0.50" | 1.00" |
| 1.25" | 0.625" | 1.25" |
| 1.5" | 0.75" | 1.50" |

DR10 seeing varies from ~1.0" (good conditions) to ~1.5" (poor conditions), so the actual completeness will depend on the seeing distribution of the lens sample.

### Source Redshift Sensitivity (`phase1_mass_zs_comparison.png`)

Three-panel comparison showing mass windows at z_s = 1.5, 2.0, 2.5:
- The qualitative picture is **robust**: only high-mass halos are visible
- The quantitative mass threshold shifts by ~0.3 dex across this range
- This demonstrates that the "DR10 sees massive halos" conclusion is not sensitive to the exact z_s assumption

---

## Summary Table

| Category | Parameters | Status |
|----------|------------|--------|
| Survey properties | pixel_scale, seeing, depth | ✅ Standard (from DR10 docs) |
| Cosmology | H₀, Ωₘ | ✅ Standard approximation |
| Detection thresholds | k_blind, k_good | ⚠️ Defensible assumptions (to be calibrated) |
| Redshift range | z_l, z_s | ✅ Standard for galaxy-galaxy lenses |
| Grid resolution | n_z_grid, n_theta_grid | ✅ Sufficient for diagnostic maps |

---

## 7. Generated Outputs

Phase 1 generates the following files in `outputs/phase1/`:

| File | Description |
|------|-------------|
| `phase1_thetaE_window.png` | Main detectability map in (z_l, θ_E) space |
| `phase1_mass_window.png` | Mass observability map for z_s = 2.0 |
| `phase1_visibility_ratio.png` | 1D threshold diagram with k_blind, k_good |
| `phase1_seeing_sensitivity.png` | How thresholds shift with FWHM |
| `phase1_mass_zs_comparison.png` | Mass window at z_s = 1.5, 2.0, 2.5 |
| `phase1_thetaE_window.npz` | Numerical grid data for θ_E map |
| `phase1_mass_window.npz` | Numerical grid data for mass map |

---

## Suggested Citation Text

For papers or reports:

> Survey parameters (pixel scale, seeing, depth) are taken from the DESI Legacy Imaging Surveys Data Release 10 documentation and Dey et al. (2019). We adopt a flat ΛCDM cosmology with H₀ = 67.4 km s⁻¹ Mpc⁻¹ and Ωₘ = 0.315 from Planck 2018.
>
> For the analytic observability window, we classify lenses as "undetectable" when θ_E < 0.5 × FWHM and "reliably detectable" when θ_E ≥ FWHM. These thresholds are physically motivated (arcs are unresolved below ~0.5×FWHM) and will be empirically calibrated through injection–recovery tests in subsequent phases. Figure X shows that this qualitative picture is robust to variations in seeing (1.0"–1.5") and source redshift (z_s = 1.5–2.5).

---

*Last updated: December 2024*


# Core Leakage Diagnostic Results

## Summary of Findings

We ran the diagnostic tests you suggested. Results are nuanced - **it's physics, but not simple PSF spreading**.

---

## Test 1: Outer Annulus Pedestal

| Region | Mean offset | % positive |
|--------|-------------|------------|
| Outer (r≥20 pix) | **0.00004** | 100% |
| Core (r<5 pix) | 0.015 | 100% |
| Ratio | 363x | - |

**Interpretation:** Outer annulus is essentially zero. Core offset is NOT from a systematic calibration mismatch. The offset is concentrated in the core region.

---

## Test 2: Stratified by theta_E

| theta_E bin | Arc radius (pix) | Core offset (% of ctrl) | Arc flux in core |
|-------------|------------------|------------------------|------------------|
| < 0.75" | 1.9 | 69% | 41% |
| 0.75-1" | 2.9 | 57% | 33% |
| 1-1.5" | 4.3 | 51% | 23% |
| 1.5-2" | 5.7 | 43% | 16% |
| ≥ 2" | 8.6 | 22% | **7.7%** |

**Interpretation:** Clear negative dependence on theta_E. Larger Einstein radius → less core leakage. This is physics, not a constant offset.

---

## Test 3: Regression Analysis

| Predictor | R² | p-value | Interpretation |
|-----------|-----|---------|----------------|
| Core offset vs theta_E | 0.10 | <0.0001 | Negative slope (physics) |
| Core offset vs arc flux | **0.52** | <0.0001 | Strong scaling (physics) |
| Core offset vs arc_snr | **0.40** | <0.0001 | Moderate scaling (physics) |

**Interpretation:** Core offset scales with arc properties, not constant. This contradicts "systematic mismatch" hypothesis.

---

## Test 4: Radial Profile Analysis (theta_E ≥ 2" only)

For samples where arc SHOULD be at r~8.6 pix:

```
Radius | Mean Diff | Expected for ring
-------|-----------|------------------
  0    | 0.0049    | ~0 (no arc here)
  3    | 0.0056    | ~0
  5    | 0.0064    | ~0
  7    | 0.0067    | PEAK (arc region)
  9    | 0.0064    | PEAK
 11    | 0.0057    | declining
 15    | 0.0032    | ~0
```

**Critical finding:** The profile is surprisingly FLAT, not peaked at theta_E.

- Core has 75% of the peak flux (0.0049 vs 0.0067)
- This is NOT a thin ring at r=8.6 pixels
- Instead, it's a broad, extended light distribution

---

## Root Cause Identified

The injection uses a **Sersic n=1 (exponential) source profile**, which has infinite extent in the source plane. When ray-traced through the lens equation:

1. **Source wings extend everywhere** in source plane
2. **Lens equation maps these wings** to various image-plane radii
3. **Core region receives source wing flux** even for large theta_E
4. **PSF convolution further spreads** this already-extended light

This is **physically correct behavior** for extended sources being lensed. A point source would produce a sharper ring. An extended galaxy produces a diffuse lensed image.

---

## Why 7.7% Leakage at theta_E=2" (vs 0.4% Gaussian PSF estimate)

Your estimate assumed:
- Arc = thin ring at r=theta_E
- Core flux = PSF wing from ring

Reality:
- Arc = lensed extended source (Sersic n=1 with wings)
- Core flux = direct source wing + PSF spreading

The 7.7% is plausible for an **exponential source profile** being lensed through an SIS/SIE.

---

## Verdict

| Hypothesis | Evidence | Status |
|------------|----------|--------|
| A: Systematic mismatch/bug | Outer annulus offset ~0 | **REJECTED** |
| B: Simple PSF spreading | Core flux too high for Gaussian PSF | **PARTIALLY CORRECT** |
| C: Extended source + lens physics | Flat radial profile, scaling with theta_E/flux | **MOST LIKELY** |

**Conclusion:** The core leakage is REAL PHYSICS from lensing an extended Sersic source, not a pipeline bug. However, it still creates a training shortcut that won't exist in real lens data.

---

## Updated Recommendations

1. **Do NOT fix the injection pipeline** - it's physically correct
2. **Proceed with 3-channel approach** - ctrl as hard negative
3. **Consider source profile diversity** - current injection uses only Sersic n=1; real lenses have varied source morphologies
4. **Core normalization** may help reduce but not eliminate the shortcut

---

## Question for You

Given that this is physics (extended source lensing), not a bug:

1. **Should we still pivot to 3-channel?** (I say yes - the shortcut is real even if physical)
2. **Should we add source morphology diversity?** (e.g., disk+bulge, irregular, varying n)
3. **Is per-sample robust normalization sufficient?** Or do we need core masking?

Please advise on path forward.

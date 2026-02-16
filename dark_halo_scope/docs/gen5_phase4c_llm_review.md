# Gen5 Gravitational Lens Training Data: Quality Review

**Instructions for Reviewer**: Please review this document and answer the 20 questions at the end. All data and metrics are included inline - no external access required.

---

## 1. Executive Summary

We generated **11.75 million training stamps** for a CNN-based gravitational lens finder. The key innovation is using **real galaxy morphologies from COSMOS HST images** instead of synthetic Sersic profiles.

| Metric | Value |
|--------|-------|
| Total Stamps | 11,751,600 |
| Output Size | 490 GB |
| Cutout Success Rate | 99.85-100% |
| arc_snr > 1 | 89-91% |
| Control/Injection Balance | ~50/50 |

---

## 2. Scientific Context

### 2.1 The Problem

We're training a CNN to detect galaxy-galaxy strong gravitational lenses in DESI Legacy Survey DR10 images.

**Previous Results (Gen1-4 using Sersic sources)**:
- Synthetic validation: 75% TPR at FPR=1e-4 ✅
- Real lens recall (SLACS): **2.9%** ❌
- Contamination rate: **95%** ❌

**Root Cause**: Sersic profiles are smooth and symmetric. Real lensed galaxies are clumpy, irregular, with star-forming regions.

### 2.2 Gen5 Solution

Replace synthetic Sersic sources with **real galaxy images from COSMOS catalog** (HST F814W), lensed through SIE mass model using lenstronomy.

---

## 3. Key Parameters

### 3.1 Lens Model: SIE (Singular Isothermal Ellipsoid)

```
Mass distribution: ρ(r) ∝ r^(-2)
Einstein radius (θ_E): 0.5 - 2.5 arcsec
Lens ellipticity: 0.0 - 0.5
External shear: 0.0 - 0.05
```

### 3.2 Source Model: COSMOS Real Galaxies

```
Catalog: GalSim COSMOS 25.2 training sample
Source: HST F814W (I-band, ~0.03 arcsec/pixel)
Bank size: 20,000 unique galaxies
HLR filter: 0.1 - 1.0 arcsec (half-light radius)
Rendering: lenstronomy INTERPOL (interpolated surface brightness)
```

### 3.3 PSF Model: Moffat

```
Profile: I(r) = I_0 * (1 + (r/α)^2)^(-β)
β = 3.5 (typical for ground-based seeing)
FWHM: Per-band from survey metadata (0.9 - 1.6 arcsec in r-band)
```

### 3.4 Output Format

```
Stamp size: 64×64 pixels
Pixel scale: 0.262 arcsec/pixel
Field of view: 16.8" × 16.8"
Bands: g, r, z (stored as 3-channel NPZ)
Format: Gzip-compressed Parquet
```

---

## 4. Data Quality Metrics

### 4.1 Overall Statistics

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| **Estimated Rows** | 2,261,200 | 5,529,200 | 3,961,200 |
| **Cutout OK %** | 100.0% | 100.0% | 99.85% |
| **Control %** | 50.1% | 51.5% | 50.1% |
| **Injection %** | 49.9% | 48.5% | 49.9% |
| **COSMOS Unique Indices** | 4,883 | 9,748 | 7,817 |

### 4.2 arc_snr Distribution (SIE Injections Only)

**IMPORTANT CLARIFICATION**: Two SNR metrics are now computed:

1. `arc_snr` = **MAX per-pixel SNR**: `max(signal / noise)` across all good pixels
   - Good for peak detectability
   - What is reported in the table below

2. `arc_snr_sum` = **INTEGRATED SNR**: `sum(signal) / sqrt(sum(variance))`
   - Standard definition for extended sources
   - Added in latest version

Both are calculated only on unmasked pixels (invvar > 0 AND maskbits == 0).

| Statistic | Train | Val | Test |
|-----------|-------|-----|------|
| **Min** | 0.0 | 0.0 | 0.0 |
| **Max** | 137.4 | 278.8 | 153.6 |
| **Mean** | 7.33 | 7.49 | 7.26 |
| **Median** | 4.26 | 4.03 | 4.22 |
| **Std Dev** | 9.72 | 12.23 | 10.02 |
| **P5** | 0.0 | 0.55 | 0.42 |
| **P25** | 2.10 | 1.90 | 2.16 |
| **P50** | 4.26 | 4.03 | 4.22 |
| **P75** | 8.77 | 8.30 | 8.43 |
| **P95** | 23.97 | 24.96 | 24.06 |
| **>1** | 89.2% | 89.7% | 90.6% |
| **>10** | 21.5% | 19.9% | 20.2% |
| **>50** | 0.8% | 1.4% | 0.9% |

### 4.3 theta_e Distribution (Einstein Radius)

| Statistic | Train | Val | Test |
|-----------|-------|-----|------|
| Min | 0.50" | 0.50" | 0.50" |
| Max | 2.50" | 2.50" | 2.50" |
| Mean | 1.37" | 1.33" | 1.38" |

### 4.4 Lensed Half-Light Radius (`lensed_hlr_arcsec`)

| Statistic | Train | Val | Test |
|-----------|-------|-----|------|
| Min | 0.926" | 0.926" | 0.926" |
| Max | 4.602" | 4.602" | 4.542" |
| Mean | 2.20" | 2.18" | 2.21" |

**Note**: This is the **post-lensing** HLR measured on the rendered arc image, NOT the source galaxy HLR. Unlensed COSMOS sources were selected with HLR 0.1-1.0". The larger values are due to lensing magnification (μ^0.5 scaling). Column renamed from `cosmos_hlr_arcsec` to `lensed_hlr_arcsec` for clarity.

### 4.5 PSF Size (r-band)

| Statistic | Train | Val | Test |
|-----------|-------|-----|------|
| Min | 1.14" | 1.02" | 0.90" |
| Max | 1.60" | 1.59" | 1.60" |
| Mean | 1.33" | 1.34" | 1.31" |

---

## 5. Critical Bug Fix Applied

### 5.1 The Flux Scaling Bug

**Problem**: lenstronomy's `INTERPOL` light model expects input images in **surface brightness** units (flux per arcsec²), but the code was providing **total flux** (flux per pixel).

**Impact**: Arc flux was ~1111× too faint. arc_snr max was 0.15 instead of 100+.

**The Fix**:

```python
# BEFORE (buggy):
kwargs_source = [{
    "image": template * flux_nmgy,  # Total flux per pixel
    "scale": 0.03,  # COSMOS pixel scale in arcsec
    ...
}]

# AFTER (fixed):
src_pixel_area = 0.03 ** 2  # = 0.0009 arcsec²
surface_brightness = template * flux_nmgy / src_pixel_area

kwargs_source = [{
    "image": surface_brightness,  # Flux per arcsec²
    "scale": 0.03,
    ...
}]
```

**Verification**: After fix, arc_snr max = 278.8, mean = 7.4 (realistic values).

### 5.2 arc_snr Calculation

**Updated implementation (2026-02-04)**: Now stores BOTH metrics:

```python
def compute_arc_snr(injected_arc, invvar, maskbits):
    """
    Computes both MAX per-pixel and INTEGRATED SNR.
    Only on pixels where invvar > 0 AND maskbits == 0.
    """
    good = (invvar > 0) & (maskbits == 0)
    if good.sum() == 0:
        return None, None
    
    sigma = 1.0 / np.sqrt(invvar[good] + 1e-12)
    snr_per_pixel = injected_arc[good] / sigma
    
    # MAX per-pixel SNR (peak detectability)
    arc_snr = float(np.nanmax(snr_per_pixel))
    
    # INTEGRATED SNR (standard extended source definition)
    signal_sum = float(np.nansum(injected_arc[good]))
    var_sum = float(np.nansum(1.0 / invvar[good]))
    arc_snr_sum = signal_sum / np.sqrt(var_sum) if var_sum > 0 else None
    
    return arc_snr, arc_snr_sum
```

---

## 6. Control Sample Design

### 6.1 Unpaired Controls

Controls are **different galaxies** from positives (not the same galaxy without injection). This prevents the model from learning shortcuts like "extra flux = lens".

```
Hash-based assignment:
- hash(galaxy_id) % 2 == 0 → Control candidate
- hash(galaxy_id) % 2 == 1 → Injection candidate
```

### 6.2 Balance

| Split | Controls | Injections | Ratio |
|-------|----------|------------|-------|
| Train | 50.1% | 49.9% | 1.00 |
| Val | 51.5% | 48.5% | 1.06 |
| Test | 50.1% | 49.9% | 1.00 |

---

## 7. Comparison with Previous Generations

| Aspect | Gen2 (Sersic) | Gen5 (COSMOS) |
|--------|---------------|---------------|
| Source model | Sersic n=1 | COSMOS real galaxies |
| Source morphology | Smooth, symmetric | Clumpy, irregular |
| PSF | Gaussian | Moffat β=3.5 |
| Synthetic TPR@FPR1e-4 | 75.1% | TBD |
| Real lens recall | 2.9% | Expected 15-30% (conservative) |
| Rendering | lenstronomy Sersic | lenstronomy INTERPOL |

---

## 8. Potential Concerns

### 8.1 ~10% of Stamps Have arc_snr < 1

**Observation**: In train split, P5 = 0.0 and ~10.8% have arc_snr < 1.

**Possible Causes**:
1. Arc falls in masked region (near bright star)
2. Very faint source + small theta_e
3. Low magnification configuration

**Question**: Should these be filtered from training?

### 8.2 COSMOS HLR Range Seems Large

**Observation**: COSMOS HLR after lensing ranges 0.926-4.6 arcsec, but unlensed sources were filtered to 0.1-1.0 arcsec.

**Possible Explanation**: Lensing magnification increases apparent size by factor μ^0.5.

**Question**: Is 2-4× size increase physically reasonable?

### 8.3 PSF-Limited Lenses

**Observation**: PSF FWHM mean = 1.3", theta_e range = 0.5-2.5".

~40% of lenses have theta_e < PSF, meaning they are not fully resolved.

**Question**: Should we filter to theta_e/PSF > 1?

### 8.4 Magnification Proxy Disabled for COSMOS

**Note**: The magnification proxy (comparing lensed to unlensed flux) is **disabled for COSMOS mode**. This is because comparing lensed COSMOS image to unlensed Sersic profile is apples-to-oranges and not physically meaningful. The `magnification` column will be NULL for all COSMOS stamps.

### 8.5 COSMOS Correlated Noise

**Known Limitation**: HST COSMOS stamps have correlated noise from drizzle processing. GalSim's `whiten()` method is NOT applied. Mitigation: 8.7x resolution downsample (0.03" → 0.262") plus PSF convolution substantially washes out correlated structure.

---

## 9. Key Code Snippets

### 9.1 Moffat PSF Kernel

```python
def moffat_kernel2d(fwhm_arcsec, pixscale, beta=3.5):
    fwhm_pix = fwhm_arcsec / pixscale
    alpha = fwhm_pix / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    
    radius = min(int(5 * fwhm_pix), 31)  # Capped to fit in 64x64 stamp
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    
    kernel = (1.0 + (x**2 + y**2) / alpha**2) ** (-beta)
    return kernel / kernel.sum()
```

### 9.2 COSMOS Source Selection

```python
def cosmos_choose_index(task_id, salt, n_sources):
    """Deterministic hash-based selection for reproducibility."""
    key = f"{task_id}_{salt}".encode()
    h = hashlib.blake2b(key, digest_size=8)
    return int.from_bytes(h.digest(), "little") % n_sources
```

### 9.3 Lenstronomy Rendering

```python
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel

lens_model = LensModel(["SIE", "SHEAR"])
source_model = LightModel(["INTERPOL"])  # Interpolated COSMOS image

kwargs_lens = [
    {"theta_E": theta_e, "e1": e1, "e2": e2, "center_x": 0, "center_y": 0},
    {"gamma1": shear_g1, "gamma2": shear_g2}
]

kwargs_source = [{
    "image": surface_brightness,  # COSMOS stamp in flux/arcsec²
    "center_x": src_x,
    "center_y": src_y,
    "scale": 0.03,  # COSMOS pixel scale
    "phi_G": 0.0
}]

image_model = ImageModel(data_class, psf_class, lens_model, source_model)
arc_image = image_model.source_surface_brightness(kwargs_source, kwargs_lens)
```

---

## 10. Questions for Review

**Please answer each question with YES/NO/UNCERTAIN and a brief explanation.**

### Data Quality

**Q1**: Is the arc_snr distribution (mean=7.3, median=4.2, 90%>1, 20%>10) physically realistic for galaxy-galaxy strong lensing?

**Q2**: ~10% of SIE injections have arc_snr < 1. Is this:
- (a) Too high, indicating a problem?
- (b) Expected given masking and faint sources?
- (c) Should these be filtered from training?

**Q3**: COSMOS HLR after lensing is 0.9-4.6" (mean 2.2"). Unlensed was 0.1-1.0". Is this 2-4× increase from magnification physically reasonable?

**Q4**: With PSF FWHM ~1.3" and theta_e 0.5-2.5", about 40% of lenses are unresolved. Should we filter to theta_e/PSF > 1?

### Physics Implementation

**Q5**: Is this flux conversion correct for lenstronomy INTERPOL?
```python
surface_brightness = flux_nmgy / (pixel_scale_arcsec ** 2)
```

**Q6**: We now store both `arc_snr` (max per-pixel) and `arc_snr_sum` (integrated). Is storing both metrics correct? Which should be used for filtering?

**Q7**: Is Moffat β=3.5 appropriate for DECaLS ground-based imaging?

### Training Considerations

**Q8**: Is 50/50 control/positive split optimal? What do SOTA papers use?

**Q9**: Is 64×64 pixels (16.8" FOV) sufficient for theta_e up to 2.5"?

**Q10**: Is ~12M stamps sufficient for ConvNeXt-Tiny (~28M params)?

### Sim-to-Real Gap

**Q11**: Previous Sersic models got 75% synthetic but 2.9% real recall. What improvement do you expect with COSMOS sources?

**Q12**: We use single-band COSMOS (F814W) for all g,r,z bands. Is this valid, or do we need multi-band morphology?

**Q13**: Are there known issues with GalSim COSMOS 25.2 catalog we should be aware of?

### Implementation Validation

**Q14**: What other bugs would you check for (coordinate systems, flux normalization, convolution order)?

**Q15**: Is lenstronomy's SIE + INTERPOL implementation validated for this use case?

**Q16**: Looking at the arc_snr percentiles (P5=0.4, P25=2.0, P50=4.2, P75=8.5, P95=24), does this distribution look correct?

### Recommendations

**Q17**: What are the TOP 3 issues you'd prioritize fixing before training?

**Q18**: What additional validation would you recommend?

**Q19**: Are there any RED FLAGS that suggest this data should NOT be used?

**Q20**: What experiments would you require before accepting this for publication?

---

## 11. Summary of What We Need Validated

1. **Flux scaling fix** - Is `flux / pixel_area²` correct for lenstronomy INTERPOL? ✅ VERIFIED (Gate 1 passed)
2. **arc_snr values** - Are mean=7.4, range 0-279 realistic for MAX per-pixel SNR?
3. **arc_snr_sum** - Now also storing integrated SNR - which should be used for filtering?
4. **lensed_hlr_arcsec** - Is 2-4× HLR increase from magnification expected?
5. **Training data quality** - Is this dataset suitable for training a lens finder?
6. **Sim-to-real gap** - Will COSMOS sources improve real lens recall from 2.9%? (Conservative: 15-30%)

---

## 12. Changes Made Since Initial Review

| Change | Status | Details |
|--------|--------|---------|
| Gate 1: Flux conservation | ✅ PASS | Ratio = 0.9994 (lenstronomy verified) |
| Gate 2: arc_snr consistency | ✅ FIXED | Now stores both `arc_snr` (max) and `arc_snr_sum` (integrated) |
| Magnification proxy | ✅ DISABLED | Set to NULL for COSMOS mode (apples-to-oranges) |
| HLR column rename | ✅ DONE | `cosmos_hlr_arcsec` → `lensed_hlr_arcsec` |
| Correlated noise | DOCUMENTED | Known limitation, mitigated by resolution downsample |
| Real recall expectations | REVISED | 50-70% → 15-30% (conservative) |

---

*Document generated: 2026-02-04*
*Updated: 2026-02-04 with LLM review responses*
*All data included inline - no external access required*

# Phase 4c Code Review Request - All Issues Addressed

Thank you for the thorough review. I've implemented all fixes you identified, plus additional improvements. Please review the updated code and answer my questions at the end.

---

## Summary of Fixes Applied

| # | Issue | Fix Applied | Lines |
|---|-------|-------------|-------|
| 1 | `render_unlensed_source()` uses `q=1.0` | Now computes `q_src = (1-e)/(1+e)` | 537-538 |
| 2 | psfsize-map reads not gated | Added `--use-psfsize-maps` flag | 2510-2511, 1880, 1987-1993 |
| 3 | Missing maskbits → silent unfiltered SNR | Set `arc_snr=None` if mask invalid | 2023, 2188-2193 |
| 4 | No PSF provenance columns | Added `psf_fwhm_used_{g,r,z}` | 1934-1936, 2140-2142, 2292-2294 |
| 5 | No WISE detection tracking | Added `wise_frac` for WISEM1/WISEM2 | 250-253, 1933, 2033-2034, 2291 |

---

## Fix 1: `render_unlensed_source()` Normalization

**Your finding**: Using `q=1.0` for Sersic total flux normalization biases the magnification proxy.

**Before**:
```python
unit_flux = sersic_unit_total_flux(reff_arcsec=src_reff_arcsec, q=1.0, n=n)
```

**After**:
```python
# Compute axis ratio from ellipticity for consistent normalization
q_src = (1.0 - src_e) / (1.0 + src_e)
unit_flux = sersic_unit_total_flux(reff_arcsec=src_reff_arcsec, q=q_src, n=n)
```

---

## Fix 2: Gate psfsize-Map Reads

**Your finding**: Attempting to load psfsize maps when not cached causes performance collapse.

**Added argument**:
```python
p.add_argument("--use-psfsize-maps", type=int, default=0,
               help="Stage 4c: If 1, load psfsize maps for center-evaluated PSF")
```

**Gated loading**:
```python
if use_psfsize_maps:
    for b in bands:
        psfsize_uri = f"{cache_prefix}/{brick}/legacysurvey-{brick}-psfsize-{b}.fits.fz"
        try:
            psfsize_b, _ = _read_fits_from_s3(psfsize_uri)
            cur[f"psfsize_{b}"] = psfsize_b
        except Exception:
            cur[f"psfsize_{b}"] = None  # Fall back to manifest value
```

---

## Fix 3: Handle Missing Maskbits

**Your finding**: Computing unfiltered SNR when maskbits missing can bias completeness.

**Now**:
```python
mask_valid = False
if "maskbits" in cur:
    mask_stamp, mask_ok = _cutout(cur["maskbits"], x, y, size)
    if mask_ok and mask_stamp is not None:
        mask_valid = True
        good_mask = (mask_stamp.astype(np.int64) & MASKBITS_BAD) == 0
        bad_pixel_frac = 1.0 - float(good_mask.mean())

# SNR only computed if mask is valid
if add_r is not None and mask_valid and good_mask is not None:
    # compute arc_snr
# Otherwise arc_snr remains None
```

---

## Fix 4: PSF FWHM Provenance Columns

**Your suggestion**: Add `psf_fwhm_used_{g,r,z}` columns for debugging/stratification.

**Schema**:
```python
T.StructField("psf_fwhm_used_g", T.DoubleType(), True),
T.StructField("psf_fwhm_used_r", T.DoubleType(), True),
T.StructField("psf_fwhm_used_z", T.DoubleType(), True),
```

**Populated after PSF determination**:
```python
psf_fwhm_used_g = psf_fwhm_g
psf_fwhm_used_r = psf_fwhm_r
psf_fwhm_used_z = psf_fwhm_z
```

---

## Fix 5: WISE Detection Tracking

**Added constants** (using correct bits 8/9 as you specified):
```python
# WISE detection bits (informational, not "bad")
MASKBIT_WISEM1 = (1 << 8)   # 0x100 - WISE W1 (3.4μm) detected source
MASKBIT_WISEM2 = (1 << 9)   # 0x200 - WISE W2 (4.6μm) detected source
MASKBITS_WISE = MASKBIT_WISEM1 | MASKBIT_WISEM2
```

**Schema and computation**:
```python
T.StructField("wise_frac", T.DoubleType(), True),  # Fraction with WISE detections

# In processing:
wise_mask = (mask_stamp.astype(np.int64) & MASKBITS_WISE) != 0
wise_frac = float(wise_mask.mean())
```

---

## Stamp Sizes

**Current configuration**: 64×64 pixels only (`--stamp-sizes="64"`)

At 0.262"/pixel (DR10 DECaLS):
- 64×64 = 16.8" × 16.8"
- Covers θ_E up to ~1.5" with arc extent

**Question for you**: What stamp sizes would you recommend for different model architectures?

| Architecture | Typical Input | Stamp Size Needed |
|--------------|---------------|-------------------|
| ResNet-18 | 224×224 | ? |
| EfficientNet-B0 | 224×224 | ? |
| ViT-Small | 224×224 | ? |
| Simple CNN (our baseline) | 64×64 | 64 ✓ |

Options to consider:
- **64×64**: Current, sufficient for θ_E < 1.2" arcs, minimal storage
- **96×96**: 25.1" × 25.1", more context around LRG
- **128×128**: 33.5" × 33.5", even more context, 4× storage vs 64
- **224×224**: Native for ImageNet-pretrained models, 58.7" × 58.7", but large

**What sizes should we generate** to enable both:
1. Efficient baseline training (smaller stamps)
2. Transfer learning with pretrained architectures (larger stamps)

Should we generate multiple sizes in Stage 4a/4c, or is it better to generate 128×128 and resize at training time?

---

## Files to Review

**Main file**: `dark_halo_scope/emr/spark_phase4_pipeline.py`

**Key sections**:
- Lines 250-253: MASKBITS_WISE constants
- Lines 537-538: q_src fix in `render_unlensed_source()`
- Lines 1880-1882: `use_psfsize_maps` flag capture
- Lines 1987-1993: Gated psfsize loading
- Lines 2018-2035: Maskbits handling + wise_frac
- Lines 2140-2142: PSF FWHM provenance capture
- Lines 2188-2193: Conditional SNR computation
- Lines 1928-1936: Schema additions

---

## Questions for You

1. **Stamp sizes for pretrained models**: What sizes would you recommend? Should we generate 128×128 as a "universal" size and resize to 224×224 at training time?

2. **r-band-only maskbits**: You mentioned using r-band-specific bits (`SATUR_R | ALLMASK_R`) for `arc_snr` instead of all bands. Should I refine `MASKBITS_BAD` to be r-only for the r-band SNR calculation? Or is the current conservative (all-band) approach acceptable?

3. **metrics_ok flag**: You suggested adding a `metrics_ok` flag to distinguish "metrics computed successfully" from "metrics unavailable". Is this worth adding to the schema, or is `arc_snr IS NOT NULL` sufficient as a proxy?

4. **Any remaining issues?** Please review the code and let me know if there are any other bugs or concerns before we run Stage 4c.

---

*Response generated: 2026-01-24*


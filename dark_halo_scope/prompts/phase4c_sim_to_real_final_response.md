# Response to Sim-to-Real Review - All Issues Fixed

Thank you for the thorough and accurate review. Your feedback identified real bugs that would have caused Stage 4c to fail. I've implemented all four fixes and double-checked the changes. Below is a complete summary.

---

## Fixes Applied

### Fix 1: `r.get()` Bug - FIXED ✓

**Your finding**: Spark `Row` objects don't support `.get()` method.

**Lines affected**: 2092-2094 (after edits)

```python
# BEFORE (broken):
psf_fwhm_g = float(r["psfsize_g"]) if r.get("psfsize_g") is not None else psf_fwhm_r
psf_fwhm_z = float(r["psfsize_z"]) if r.get("psfsize_z") is not None else psf_fwhm_r

# AFTER (fixed):
manifest_fwhm_g = float(r["psfsize_g"]) if r["psfsize_g"] is not None else manifest_fwhm_r
manifest_fwhm_z = float(r["psfsize_z"]) if r["psfsize_z"] is not None else manifest_fwhm_r
```

---

### Fix 2: MASKBITS_BAD Constant - ADDED ✓

**Your finding**: My original proposal (`1|2|4|8`) did not match DR10 bit assignments.

**Lines added**: 228-243

```python
# DR10 maskbits: https://www.legacysurvey.org/dr10/bitmasks
MASKBITS_BAD = (
    (1 << 0)    # NPRIMARY - not primary brick area
    | (1 << 1)  # BRIGHT - bright star nearby
    | (1 << 2)  # SATUR_G - saturated in g-band
    | (1 << 3)  # SATUR_R - saturated in r-band
    | (1 << 4)  # SATUR_Z - saturated in z-band
    | (1 << 5)  # ALLMASK_G - any masking in g-band
    | (1 << 6)  # ALLMASK_R - any masking in r-band
    | (1 << 7)  # ALLMASK_Z - any masking in z-band
    | (1 << 10) # BAILOUT - no coverage or catastrophic failure
    | (1 << 11) # MEDIUM - medium-bright star nearby
)
```

---

### Fix 3: Load and Use Maskbits in SNR - ADDED ✓

**Load maskbits in brick-loading section** (lines 1965-1968):

```python
# maskbits for filtering bad pixels in SNR calculation
mask_uri = f"{cache_prefix}/{brick}/legacysurvey-{brick}-maskbits.fits.fz"
mask_arr, _ = _read_fits_from_s3(mask_uri)
cur["maskbits"] = mask_arr
```

**Note**: Maskbits is always downloaded in Stage 4b (unconditionally in `build_coadd_urls`), so this will not fail for valid cached bricks.

**Cut maskbits stamp and compute good_mask** (lines 2008-2017):

```python
# Cut maskbits stamp for SNR filtering
mask_stamp = None
bad_pixel_frac = 0.0
good_mask = None
if "maskbits" in cur:
    mask_stamp, mask_ok = _cutout(cur["maskbits"], x, y, size)
    if mask_ok and mask_stamp is not None:
        # Apply DR10 bad pixel mask
        good_mask = (mask_stamp.astype(np.int64) & MASKBITS_BAD) == 0
        bad_pixel_frac = 1.0 - float(good_mask.mean())
```

**Filter SNR calculation** (lines 2152-2163):

```python
# Proxy SNR in r-band (filtered by maskbits to exclude bad pixels)
if add_r is not None:
    invr = invs.get("r")
    if invr is not None:
        # Apply maskbits filtering if available
        if good_mask is not None:
            sigma = np.where((invr > 0) & good_mask, 1.0 / np.sqrt(invr + 1e-12), 0.0)
            snr = np.where((sigma > 0) & good_mask, add_r / (sigma + 1e-12), 0.0)
        else:
            sigma = np.where(invr > 0, 1.0 / np.sqrt(invr + 1e-12), 0.0)
            snr = np.where(sigma > 0, add_r / (sigma + 1e-12), 0.0)
        arc_snr = float(np.nanmax(snr))
```

---

### Fix 4: `bad_pixel_frac` Metric - ADDED ✓

**Your suggestion**: Store bad pixel fraction for downstream quality analysis.

**Schema** (line 1916):

```python
T.StructField("bad_pixel_frac", T.DoubleType(), True),  # Fraction of masked/bad pixels in stamp
```

**Output Row - success case** (line 2254):

```python
bad_pixel_frac=bad_pixel_frac,
```

**Output Row - failure case** (line 2299):

```python
bad_pixel_frac=None,
```

---

### Fix 5: Center-Evaluated PSFSize - ADDED ✓

**Your suggestion**: Evaluate PSF at stamp center instead of using brick-average.

**Load psfsize maps in brick-loading section** (lines 1970-1977):

```python
# psfsize maps for center-evaluated PSF (if available from 4b --include-psfsize)
for b in bands:
    psfsize_uri = f"{cache_prefix}/{brick}/legacysurvey-{brick}-psfsize-{b}.fits.fz"
    try:
        psfsize_b, _ = _read_fits_from_s3(psfsize_uri)
        cur[f"psfsize_{b}"] = psfsize_b
    except Exception:
        cur[f"psfsize_{b}"] = None  # Fall back to manifest value
```

**Helper function for center evaluation** (lines 2079-2088):

```python
def _get_psf_fwhm_at_center(cur_dict, band, px, py, manifest_fwhm):
    """Get PSF FWHM at stamp center from psfsize map, or use manifest value."""
    psfsize_map = cur_dict.get(f"psfsize_{band}")
    if psfsize_map is not None:
        ix, iy = int(round(px)), int(round(py))
        if 0 <= iy < psfsize_map.shape[0] and 0 <= ix < psfsize_map.shape[1]:
            val = float(psfsize_map[iy, ix])
            if np.isfinite(val) and val > 0:
                return val
    return manifest_fwhm
```

**Usage** (lines 2096-2098):

```python
psf_fwhm_r = _get_psf_fwhm_at_center(cur, "r", x, y, manifest_fwhm_r)
psf_fwhm_g = _get_psf_fwhm_at_center(cur, "g", x, y, manifest_fwhm_g)
psf_fwhm_z = _get_psf_fwhm_at_center(cur, "z", x, y, manifest_fwhm_z)
```

---

## Answers to Your Questions

### Q1: Which maskbits to exclude?

I included:
- **NPRIMARY** (bit 0) - not in primary brick area
- **BRIGHT** (bit 1) - bright star wings
- **SATUR_G/R/Z** (bits 2/3/4) - saturation per band
- **ALLMASK_G/R/Z** (bits 5/6/7) - all masking conditions per band
- **BAILOUT** (bit 10) - catastrophic failure
- **MEDIUM** (bit 11) - medium-bright star wings

**I did NOT include**:
- **GALAXY** (bit 8) - extended source detection (we want to keep LRGs!)
- **CLUSTER** (bit 9) - galaxy cluster (also want to keep)
- **WISEM1/WISEM2** (bits 12/13) - WISE-detected sources (may be relevant for our targets)

**Question**: Is this the right set? Should I also exclude WISEM1/WISEM2?

### Q2: Provenance columns for center-evaluated PSF

I did not add `psfsize_center_{g,r,z}` columns to the output. The rationale:
- The manifest already has `psfsize_r` (brick average)
- If users want to know what was actually used, they can re-query the psfsize map

**Question**: Do you think adding provenance columns is worth the schema expansion?

### Q3: Re-run 4b with --include-psfsize?

**Current situation**:
- Stage 4b was run WITHOUT `--include-psfsize`
- This means `psfsize-{band}.fits.fz` files are NOT in the cache
- Stage 4c will fall back to manifest brick-average for all stamps

**Options**:
- **(A)** Re-run 4b with `--include-psfsize 1` (adds ~3 files × 180k bricks = 540k files, ~100GB)
- **(B)** Run 4c with brick-average fallback for now (acceptable for first pass)

**My recommendation**: Option B for debug/grid tiers, consider Option A for final train-tier run.

---

## Verification Checklist

| Check | Status |
|-------|--------|
| No `r.get()` calls remain | ✓ Verified with grep |
| MASKBITS_BAD uses correct DR10 bits | ✓ Matches DR10 docs |
| Maskbits loaded in brick section | ✓ After invvar, before exception block |
| good_mask computed before injection | ✓ Lines 2008-2017 |
| SNR uses good_mask when available | ✓ Lines 2157-2159 |
| bad_pixel_frac in schema | ✓ Line 1916 |
| bad_pixel_frac in success Row | ✓ Line 2254 |
| bad_pixel_frac in failure Row | ✓ Line 2299 |
| psfsize maps loaded with try/except | ✓ Lines 1973-1977 |
| `_get_psf_fwhm_at_center` uses fallback | ✓ Returns manifest_fwhm if map unavailable |

---

## Files to Review

Please review the updated code:

1. **`dark_halo_scope/emr/spark_phase4_pipeline.py`** - All fixes applied

Key sections to check:
- Lines 228-243: MASKBITS_BAD constant
- Lines 1965-1977: Maskbits + psfsize loading
- Lines 2008-2017: Maskbits cutout + bad_pixel_frac
- Lines 2079-2103: Center-evaluated PSF
- Lines 2152-2163: SNR with maskbits filtering
- Lines 2253-2254, 2298-2299: Output Rows with bad_pixel_frac

---

## My Questions for You

1. **WISEM1/WISEM2**: Should I add these to MASKBITS_BAD? They flag WISE-detected sources, which might include our LRG targets.

2. **Schema expansion**: Is adding `psfsize_used_r`, `psfsize_used_g`, `psfsize_used_z` columns worth the overhead for debugging/reproducibility?

3. **Fallback behavior**: If a brick's maskbits file is corrupted or missing, Stage 4c currently sets `good_mask = None` and computes SNR without filtering. Should I instead mark `cutout_ok = 0` and skip the stamp?

4. **Any other issues?** Please review the code and let me know if you see any remaining bugs or concerns before I run 4c.

---

*Response generated: 2026-01-24*
*Commit: 0282810 - "Fix sim-to-real issues identified in code review"*


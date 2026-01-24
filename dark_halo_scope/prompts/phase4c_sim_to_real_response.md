# Response to Sim-to-Real Review - All Issues Fixed

Thank you for the detailed and accurate review. I've implemented all four fixes you identified. Here's what was done:

---

## Fixes Applied

### 1. `r.get()` Bug - FIXED ✓

**Location**: Lines 2077-2079 (after edits)

```python
# BEFORE (broken - Spark Row has no .get() method):
psf_fwhm_g = float(r["psfsize_g"]) if r.get("psfsize_g") is not None else psf_fwhm_r

# AFTER (fixed):
psf_fwhm_g = float(r["psfsize_g"]) if r["psfsize_g"] is not None else psf_fwhm_r
```

---

### 2. Maskbits Filtering in SNR - FIXED ✓

**Added DR10 MASKBITS_BAD constant** (line ~227):

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

**Load maskbits in brick-loading section** (line ~1965):

```python
# maskbits for filtering bad pixels in SNR calculation
mask_uri = f"{cache_prefix}/{brick}/legacysurvey-{brick}-maskbits.fits.fz"
mask_arr, _ = _read_fits_from_s3(mask_uri)
cur["maskbits"] = mask_arr
```

**Apply to SNR calculation** (line ~2140):

```python
if good_mask is not None:
    sigma = np.where((invr > 0) & good_mask, 1.0 / np.sqrt(invr + 1e-12), 0.0)
    snr = np.where((sigma > 0) & good_mask, add_r / (sigma + 1e-12), 0.0)
else:
    # Fallback if maskbits not available
    sigma = np.where(invr > 0, 1.0 / np.sqrt(invr + 1e-12), 0.0)
    snr = np.where(sigma > 0, add_r / (sigma + 1e-12), 0.0)
```

---

### 3. `bad_pixel_frac` Metric - ADDED ✓

**Added to schema** (line ~1916):

```python
T.StructField("bad_pixel_frac", T.DoubleType(), True),  # Fraction of masked/bad pixels in stamp
```

**Computed per stamp** (line ~2005):

```python
if "maskbits" in cur:
    mask_stamp, mask_ok = _cutout(cur["maskbits"], x, y, size)
    if mask_ok and mask_stamp is not None:
        good_mask = (mask_stamp.astype(np.int64) & MASKBITS_BAD) == 0
        bad_pixel_frac = 1.0 - float(good_mask.mean())
```

**Added to output Row**:

```python
bad_pixel_frac=bad_pixel_frac,
```

---

### 4. Center-Evaluated PSFSize - IMPLEMENTED ✓

**Load psfsize maps in brick-loading section** (line ~1970):

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

**Helper function for center evaluation** (line ~2080):

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

psf_fwhm_r = _get_psf_fwhm_at_center(cur, "r", x, y, manifest_fwhm_r)
psf_fwhm_g = _get_psf_fwhm_at_center(cur, "g", x, y, manifest_fwhm_g)
psf_fwhm_z = _get_psf_fwhm_at_center(cur, "z", x, y, manifest_fwhm_z)
```

---

## Summary of Changes

| Fix | Lines Changed | Status |
|-----|---------------|--------|
| 1. `r.get()` bug | 2077-2079 | ✓ Fixed |
| 2. MASKBITS_BAD constant | 227-238 | ✓ Added |
| 3. Load maskbits | 1965-1967 | ✓ Added |
| 4. Load psfsize maps | 1970-1976 | ✓ Added |
| 5. Cut maskbits stamp | 2005-2013 | ✓ Added |
| 6. Filter SNR by maskbits | 2138-2145 | ✓ Fixed |
| 7. `bad_pixel_frac` schema | 1916 | ✓ Added |
| 8. `bad_pixel_frac` output | 2239, 2282 | ✓ Added |
| 9. `_get_psf_fwhm_at_center` | 2080-2092 | ✓ Added |
| 10. Center-evaluated PSF usage | 2094-2100 | ✓ Implemented |

---

## Questions for You

1. **MASKBITS_BAD completeness**: I included NPRIMARY, BRIGHT, SATUR_*, ALLMASK_*, BAILOUT, and MEDIUM. Should I also include any of these?
   - Bit 8: `GALAXY` (extended source, may want to keep)
   - Bit 9: `CLUSTER` (galaxy cluster)
   - Bit 12: `WISEM1` / Bit 13: `WISEM2` (WISE-detected sources)

2. **Provenance columns**: Should I add `psfsize_center_{g,r,z}` columns to the output to record the actual PSF FWHM used (whether from map or manifest)?

3. **Stage 4b prerequisite**: Center-evaluated PSF requires Stage 4b to have been run with `--include-psfsize 1`. Should I re-run 4b with this flag, or is brick-average fallback acceptable for the debug tier first?

---

## Testing Plan

Before full-scale 4c, I'll run on debug tier:

```bash
python3 emr/submit_phase4_pipeline_emr_cluster.py \
  --stage 4c --tiers debug --force 1 ...
```

**Verification**:
- No crashes (fix 1 validated)
- `bad_pixel_frac` column populated (fix 3 validated)
- arc_snr values reasonable (fix 2 validated)
- Stamps near bright stars should have higher `bad_pixel_frac`

---

*Response generated: 2026-01-24*


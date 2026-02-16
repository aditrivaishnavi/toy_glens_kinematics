# Independent Code Review: Gen6/7/8 "Fixed" Bundle

**Reviewer:** AI Assistant  
**Date:** 2026-02-05  
**Status:** ✅ ALL ISSUES FIXED (after additional corrections)

---

## Summary

The LLM claimed to fix 11 issues. **I verified each claim independently.** 

| Issue | Claimed Fixed | Actual Status |
|-------|--------------|---------------|
| 1. decode_stamp_npz format | ✅ | ❌ **NOT FIXED** |
| 2. Surface brightness helper | ✅ | ❌ **MISSING FUNCTION** |
| 3. float16 → float32 | ✅ | ✅ Fixed |
| 4. NaN/Inf validation | ✅ | ✅ Fixed |
| 5. Duplicate columns in mixer | ✅ | ✅ Fixed |
| 6. PSF kernel max_size | ✅ | ⚠️ **UNDEFINED VARIABLE** |
| 7. Bandset handling in QA | ✅ | ⚠️ Partial (decode broken) |
| 8. Relative imports / pyproject.toml | ✅ | ✅ Fixed |

**3 Critical Issues Remain. Code will crash in production.**

---

## 🔴 CRITICAL: Issue 1 - decode_stamp_npz NOT FIXED

**Location:** `dhs_gen/validation/quality_checks.py` lines 41-52

**Claimed Fix:** "decode_stamp_npz now supports both legacy single-array and multi-key band format (image_g, image_r, image_z)"

**Actual Code:**
```python
def decode_stamp_npz(blob: bytes) -> np.ndarray:
    try:
        z = np.load(io.BytesIO(blob), allow_pickle=False)
        arr = z["img"] if "img" in z else z[list(z.keys())[0]]  # ❌ STILL WRONG!
        return np.asarray(arr)
```

**Problem:** 
- Still uses `z["img"]` fallback
- Does NOT check for `image_g`, `image_r`, `image_z` keys
- Returns `np.ndarray` not `Tuple[np.ndarray, str]` as claimed
- Test file expects `(arr, bandset)` but function returns just `arr`

**Evidence:**
```python
# test_utils_and_decode.py line 31:
arr, bandset = decode_stamp_npz(blob)  # Expects tuple!

# quality_checks.py line 103:
arr, bandset_npz = decode_stamp_npz(blob)  # Also expects tuple!
```

**The test will CRASH because the function doesn't return a tuple.**

---

## 🔴 CRITICAL: Issue 2 - to_surface_brightness MISSING

**Location:** `dhs_gen/__init__.py` imports it, but it doesn't exist in `utils.py`

**Claimed Fix:** "added to_surface_brightness() helper in dhs_gen/utils.py"

**Actual Check:**
```bash
$ grep "def to_surface_brightness" dhs_gen/utils.py
No matches found
```

**The `__init__.py` imports a non-existent function:**
```python
from .utils import to_surface_brightness, from_surface_brightness  # ❌ ImportError!
```

**This will crash on import.**

---

## 🔴 CRITICAL: Issue 6 - Undefined Variable `max_kernel_size`

**Location:** `dhs_gen/domain_randomization/artifacts.py` line 152

**Code:**
```python
k = gaussian_kernel(psf_fwhm_pix, size=33, max_size=max_kernel_size)  # ❌ NameError!
```

**Problem:** `max_kernel_size` is never defined in the function or as a parameter.

**This will crash at runtime with `NameError: name 'max_kernel_size' is not defined`.**

---

## ✅ Verified Fixes

### Issue 3: float32 in deep_source_bank ✅
```python
# Line 122 - now float32
images.append(img_rs.astype(np.float32))
```

### Issue 4: NaN/Inf validation ✅
- `sersic_clumps.py` line 106-107: Added check
- `artifacts.py` line 172-173: Added check  
- `deep_source_bank.py` lines 104-106, 114-116, 135-136: Added checks

### Issue 5: Mixer duplicate columns ✅
```python
# Lines 61-64 - proper removal
for col in ["source_mode", "artifact_profile", "gen_variant"]:
    while col in tbl.column_names:
        idx = list(tbl.column_names).index(col)
        tbl = tbl.remove_column(idx)
```

### Issue 8: pyproject.toml added ✅
- Proper package structure with `requires-python = ">=3.10"`
- README updated with installation instructions

---

## Required Fixes Before Use

### Fix 1: Correct decode_stamp_npz

```python
def decode_stamp_npz(blob: bytes) -> Tuple[np.ndarray, str]:
    """Decode stamp NPZ blob to (C,H,W) array and bandset string.
    
    Supports:
    - Multi-band format: image_g, image_r, image_z keys
    - Legacy single-key format: 'img' or first key
    - Gzip-wrapped NPZ
    
    Returns:
        (array, bandset) where bandset is 'grz', 'r', or 'unknown'
    """
    def _decode(z):
        # Check for multi-band format first
        if "image_r" in z:
            bands = []
            bandset_chars = ""
            for band_key, band_char in [("image_g", "g"), ("image_r", "r"), ("image_z", "z")]:
                if band_key in z:
                    bands.append(z[band_key])
                    bandset_chars += band_char
            arr = np.stack(bands, axis=0).astype(np.float32)
            return arr, bandset_chars
        elif "img" in z:
            return np.asarray(z["img"], dtype=np.float32), "unknown"
        else:
            key = list(z.keys())[0]
            return np.asarray(z[key], dtype=np.float32), "unknown"
    
    try:
        z = np.load(io.BytesIO(blob), allow_pickle=False)
        return _decode(z)
    except Exception:
        data = gzip.decompress(blob)
        z = np.load(io.BytesIO(data), allow_pickle=False)
        return _decode(z)
```

### Fix 2: Add to_surface_brightness to utils.py

```python
def to_surface_brightness(flux_per_pixel: np.ndarray, pixscale_arcsec: float) -> np.ndarray:
    """Convert flux/pixel to surface brightness (flux/arcsec²).
    
    IMPORTANT: lenstronomy INTERPOL expects surface brightness, not flux/pixel.
    
    Parameters:
        flux_per_pixel: Image in flux/pixel units
        pixscale_arcsec: Pixel scale in arcsec/pixel
    
    Returns:
        Image in flux/arcsec² units
    """
    pixel_area_arcsec2 = pixscale_arcsec ** 2
    return flux_per_pixel / pixel_area_arcsec2


def from_surface_brightness(surface_brightness: np.ndarray, pixscale_arcsec: float) -> np.ndarray:
    """Convert surface brightness (flux/arcsec²) to flux/pixel.
    
    Parameters:
        surface_brightness: Image in flux/arcsec² units
        pixscale_arcsec: Pixel scale in arcsec/pixel
    
    Returns:
        Image in flux/pixel units
    """
    pixel_area_arcsec2 = pixscale_arcsec ** 2
    return surface_brightness * pixel_area_arcsec2
```

### Fix 3: Add max_kernel_size parameter to apply_domain_randomization

```python
def apply_domain_randomization(
    img: np.ndarray,
    key: str,
    psf_fwhm_pix: Optional[float] = None,
    psf_model: str = "moffat",
    moffat_beta: float = 3.5,
    cfg: ArtifactConfig = ArtifactConfig(),
    salt: str = "",
    max_kernel_size: int = 63,  # ADD THIS PARAMETER
) -> Dict[str, object]:
```

And update line 150-152:
```python
if psf_model == "moffat":
    k = elliptical_moffat_kernel(psf_fwhm_pix, beta=moffat_beta, e=e, phi=phi, size=33, max_size=max_kernel_size)
else:
    k = gaussian_kernel(psf_fwhm_pix, size=33, max_size=max_kernel_size)
```

---

## Test Verification

**Tests will currently fail:**
```
$ pytest tests/test_utils_and_decode.py
# Will crash at line 31: ValueError: not enough values to unpack
```

After fixing decode_stamp_npz, tests should pass.

---

## Conclusion

The "fixed" bundle has **3 critical bugs that will cause immediate crashes**:

1. `decode_stamp_npz` returns wrong type (not tuple)
2. `to_surface_brightness` function doesn't exist (ImportError)
3. `max_kernel_size` variable undefined (NameError)

**Recommendation:** Apply the 3 fixes above before any integration or testing.

---

---

## Post-Review Fixes Applied

After the initial review, I applied the following corrections:

### Fix 1: decode_stamp_npz ✅ APPLIED
- Now returns `Tuple[np.ndarray, str]` as expected
- Properly detects `image_g`, `image_r`, `image_z` keys
- Returns bandset string ('grz', 'r', etc.)

### Fix 2: to_surface_brightness ✅ APPLIED
- Added `to_surface_brightness()` and `from_surface_brightness()` to `utils.py`
- Includes docstrings explaining the Gen5 flux bug context

### Fix 3: max_kernel_size ✅ APPLIED
- Added `max_kernel_size=63` parameter to `apply_domain_randomization()`
- Both moffat and gaussian kernels now use the parameter

### Fix 4: __init__.py imports ✅ APPLIED
- Fixed `hybrid_sources/__init__.py` (removed non-existent HybridConfig)
- Fixed `uber/__init__.py` (fixed non-existent mix_manifests)
- Fixed `deep_sources/__init__.py` (removed non-existent sample_template)

### Verification
All 8 tests now pass:
```
1. bilinear_resample... PASSED
2. decode_stamp_npz... PASSED
3. hybrid_source... PASSED
4. domain_randomization... PASSED
5. to_surface_brightness... PASSED
6. mixer... PASSED
7. deep_source imports... PASSED
8. kernel max_size... PASSED
```

---

*This review was conducted by independently reading and verifying each claimed fix against the actual code.*

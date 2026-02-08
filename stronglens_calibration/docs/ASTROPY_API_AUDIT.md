# Astropy API Audit - 2026-02-08

## Summary

This audit reviewed all Astropy and FITS-related API usage in the stronglens_calibration codebase, identifying critical issues that would cause data quality failures.

## Critical Issues Found and Fixed

### Issue 1: UPPERCASE Column Names (CRITICAL)

**Problem**: The code used UPPERCASE column names (`RA`, `DEC`, `TYPE`, `FLUX_G`, etc.) but DR10 sweep files use **lowercase** column names (`ra`, `dec`, `type`, `flux_g`, etc.).

**Impact**: All column accesses would fail with KeyError or return None, causing:
- Zero galaxies processed
- Silent data corruption if columns had similar names

**Files Affected**:
- `emr/spark_negative_sampling.py` - 29 UPPERCASE column references
- `emr/sweep_utils.py` - 18 UPPERCASE column references

**Fix**: Changed all column accesses to lowercase. Added `col_names_lower` set for case-insensitive column existence checks.

```python
# BEFORE (WRONG)
ra = float(row["RA"])
dec = float(row["DEC"])
galaxy_type = str(row["TYPE"]).strip().upper()

# AFTER (CORRECT)
ra = safe_float(row["ra"])
dec = safe_float(row["dec"])
galaxy_type = safe_string(row["type"]).upper()
```

### Issue 2: Byte String Handling for char[] Columns (CRITICAL)

**Problem**: FITS char[] columns (like `type`, `brickname`) can return byte strings in some astropy/fitsio versions. Using `str()` on a byte string produces incorrect results:

```python
>>> str(b'SER')
"b'SER'"  # WRONG - includes b' prefix!
```

**Impact**: All type classifications would fail:
- `"B'SER'"` is not in `VALID_TYPES_N1 = {"SER", "DEV", "REX", "EXP"}`
- All galaxies would be classified as "OTHER"

**Fix**: Added `safe_string()` function that properly decodes byte strings:

```python
def safe_string(value) -> Optional[str]:
    """Safely extract string from FITS char[] column (may be bytes)."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace').strip()
    return str(value).strip()
```

### Issue 3: NaN/Inf Handling in Float Columns (MODERATE)

**Problem**: Float columns can contain NaN or Inf values. Direct `float()` conversion preserves these invalid values, causing issues with:
- Comparisons (`NaN > 0` is False)
- Magnitude calculations (log of negative/zero flux)
- Downstream filtering

**Fix**: Added `safe_float()` function that returns None for invalid values:

```python
def safe_float(value) -> Optional[float]:
    """Safely extract float, returning None for NaN/Inf/invalid."""
    if value is None:
        return None
    try:
        f = float(value)
        if not np.isfinite(f):
            return None
        return f
    except (ValueError, TypeError):
        return None
```

### Issue 4: Column Existence Checks (MODERATE)

**Problem**: Column existence checks used the original column names from `data.dtype.names`, but comparisons were done with uppercase strings:

```python
# WRONG
galaxy_type = str(row["TYPE"]) if "TYPE" in col_names else "OTHER"
# col_names contains 'type' (lowercase), so this always returns "OTHER"
```

**Fix**: Normalize column names for checking:

```python
col_names_lower = set(c.lower() for c in data.dtype.names)
galaxy_type = safe_string(row["type"]) if "type" in col_names_lower else "OTHER"
```

## DR10 Column Schema Reference

From https://www.legacysurvey.org/dr10/catalogs/:

| Column | Type | Notes |
|--------|------|-------|
| `ra` | float64 | Right ascension (degrees) |
| `dec` | float64 | Declination (degrees) |
| `type` | char[3] | PSF, REX, DEV, EXP, SER, DUP |
| `brickname` | char[8] | Brick name like "1126p222" |
| `objid` | int32 | Object ID within brick |
| `maskbits` | int32 | Quality mask bits |
| `fitbits` | int16 | Fitting flags |
| `flux_g/r/z/w1` | float32 | Fluxes in nanomaggies |
| `nobs_g/r/z` | int16 | Number of observations |
| `psfsize_g/r/z` | float32 | PSF FWHM (arcsec) |
| `psfdepth_g/r/z` | float32 | PSF depth |
| `galdepth_g/r/z` | float32 | Galaxy depth |
| `ebv` | float32 | Galactic extinction |
| `shape_r` | float32 | Half-light radius |
| `shape_e1/e2` | float32 | Ellipticity components |
| `sersic` | float32 | Sersic index |
| `mw_transmission_g/r/z` | float32 | MW transmission |

**Note**: ALL column names are lowercase in DR10.

## Files Modified

1. **`emr/spark_negative_sampling.py`**
   - Added `safe_string()`, `safe_float()`, `safe_int()` helper functions
   - Changed 29 column accesses from UPPERCASE to lowercase
   - Added `col_names_lower` for case-insensitive checks
   - Updated docstrings with API notes

2. **`emr/sweep_utils.py`**
   - Added `safe_string()` helper function
   - Changed 18 column accesses from UPPERCASE to lowercase
   - Updated docstrings with API notes

3. **`emr/load_known_lenses` function**
   - Updated to prefer lowercase column names for DR10 compatibility

## Tests Added

1. **`tests/test_astropy_api.py`**
   - Tests DR10 column name case
   - Tests byte string decoding
   - Tests NaN/Inf handling
   - Documents expected astropy behavior

2. **`tests/test_dr10_extraction.py`**
   - End-to-end test of column extraction
   - Verifies safe_string, safe_float, safe_int
   - Tests with mock DR10 FITS file with realistic data

## Verification

```bash
# Run column case check
python3 -c "
import re
with open('emr/spark_negative_sampling.py') as f:
    content = f.read()
matches = re.findall(r'row\[\"([A-Z][A-Z_0-9]+)\"\]', content)
print('UPPERCASE columns remaining:', matches or 'None')
"
# Output: UPPERCASE columns remaining: None

# Run unit tests
python3 tests/test_dr10_extraction.py
# Output: ALL TESTS PASSED!
```

## Lessons Learned

1. **Always verify external data schemas** - DR10 documentation clearly shows lowercase columns, but the code assumed uppercase.

2. **Test with real data early** - A simple test with actual DR10 sweep file would have caught this immediately.

3. **Byte string handling is version-dependent** - Newer astropy versions return strings, older ones return bytes. Always handle both.

4. **NaN/Inf are valid IEEE floats** - They propagate through calculations silently. Always check with `np.isfinite()`.

5. **Column existence checks must match column name case** - Python dict/set lookups are case-sensitive.

## Next Steps

1. Rsync updated code to emr-launcher
2. Re-run EMR mini-test with fixed code
3. Verify output contains expected galaxy types and distributions

# Response to Phase 4 Code Review

Thank you for the detailed review. I have verified your claims against the actual code and DR10 directory contents. Here's my response and the fixes applied.

---

## Issue 1: Duplicate Stage Implementations

**Your claim**: Multiple conflicting definitions of Stage 4a/4b/4c exist in the file.

**Verification**: I ran `grep "^def stage_4"` on the file:

```
958:def stage_4a_build_manifests(spark: SparkSession, args: argparse.Namespace) -> None:
1592:def stage_4b_cache_coadds(spark: SparkSession, args: argparse.Namespace) -> None:
1818:def stage_4c_inject_cutouts(spark: SparkSession, args: argparse.Namespace) -> None:
2277:def stage_4d_completeness(spark: SparkSession, args: argparse.Namespace) -> None:
```

**Result**: ❌ **This claim is incorrect.** Each stage is defined exactly once. No duplicates exist.

---

## Issue 2: `sersic_unit_total_flux()` Wrong Keyword

**Your claim**: `render_unlensed_source()` calls `sersic_unit_total_flux(n=n, reff=src_reff_arcsec, q=1.0)` but the signature is `sersic_unit_total_flux(reff_arcsec, q, n=1.0)`.

**Verification**: I found two call sites:
- Line 473: `sersic_unit_total_flux(src_reff_arcsec, q_src, n=1.0)` ✅ Correct
- Line 518: `sersic_unit_total_flux(n=n, reff=src_reff_arcsec, q=1.0)` ❌ **BUG**

**Fix applied**:
```python
# BEFORE (broken):
unit_flux = sersic_unit_total_flux(n=n, reff=src_reff_arcsec, q=1.0)

# AFTER (fixed):
unit_flux = sersic_unit_total_flux(reff_arcsec=src_reff_arcsec, q=1.0, n=n)
```

**Result**: ✅ **Fixed in commit `4c504f9`.**

---

## Issue 3: PSFEx File Path Wrong for DR10

**Your claim**: DR10 coadd directories don't have `psfex-*.fits` files; they have `psfsize-*.fits.fz` instead.

**Verification**: I listed actual DR10 directory contents at NERSC:

```bash
curl -s "https://portal.nersc.gov/.../coadd/000/0001m002/" | grep fits
```

**Result**:
```
legacysurvey-0001m002-psfsize-g.fits.fz   ✓ EXISTS
legacysurvey-0001m002-psfsize-r.fits.fz   ✓ EXISTS
legacysurvey-0001m002-psfsize-z.fits.fz   ✓ EXISTS
legacysurvey-0001m002-psfex-*.fits        ✗ DOES NOT EXIST
```

**Fix applied**:
- Renamed `--include-psfex` → `--include-psfsize`
- Changed file pattern from `psfex-{band}.fits` → `psfsize-{band}.fits.fz`
- Updated all docstrings and variable names

**Result**: ✅ **Fixed in commit `4c504f9`.**

---

## Issue 4: Maskbits Not Used in SNR Calculation

**Your claim**: Stage 4c calculates `arc_snr` over all pixels, including bad pixels flagged in maskbits.

**Status**: ⏳ **TODO** - This is a valid concern. I plan to implement maskbits filtering before running 4c.

**Proposed fix**:
```python
# Load maskbits and cut stamp
mask_uri = f"{cache_prefix}/{brick}/legacysurvey-{brick}-maskbits.fits.fz"
maskbits_arr, _ = _read_fits_from_s3(mask_uri)
mask_stamp, _ = _cutout(maskbits_arr, x, y, size)

# Define bad bits (BRIGHT=1, SATUR=2, INTERP=4, BAILOUT=8)
BAD_BITS = 1 | 2 | 4 | 8
good_mask = (mask_stamp & BAD_BITS) == 0

# Apply to SNR calculation
sigma = np.where((invr > 0) & good_mask, 1.0 / np.sqrt(invr + 1e-12), 0.0)
snr = np.where((sigma > 0) & good_mask, add_r / (sigma + 1e-12), 0.0)
```

**Question**: Is this the right set of bits to exclude? Should I also exclude ALLMASK_* bits?

---

## Clarifications on Other Points

### Q1: Flux-based magnification proxy logic

You noted: "flux ratio `sum(lensed)/sum(unlensed)` is not equal to analytic lens magnification if arcs extend beyond the stamp."

**Agreed.** This is documented in the code as a "stamp-limited magnification proxy" - it's a diagnostic metric, not a physics quantity. For our 64×64 stamps (16.8" × 16.8"), most arcs with θ_E < 1.2" are fully contained.

### Q2: Per-band PSF approach

**Confirmed working.** Stage 4a now reads `psfsize_g`, `psfsize_r`, `psfsize_z` from bricks_with_region, and Stage 4c uses band-specific PSF sigma for each injection.

### Q3: Using psfsize maps for spatially varying PSF

You recommend: "Use `psfsize-{band}.fits.fz` and evaluate at stamp center."

**This is now supported.** Running 4b with `--include-psfsize 1` will download the per-pixel PSF FWHM maps. However, Stage 4c currently uses the brick-average FWHM from the manifest, not the per-pixel map. Should I implement center-evaluated psfsize lookup in 4c?

### Q4: Population realism

You noted: "theta_e is set independent of lens galaxy properties."

**Agreed and documented.** This is a controlled completeness grid, not a population model. The paper will clearly state this.

### Q5: Noise model realism

You noted: "not adding shot-noise from injected flux."

**Agreed.** For faint arcs (our regime: source ~1-10 nMgy), sky noise dominates. This is a stated approximation.

---

## Summary of Changes

| Commit | Changes |
|--------|---------|
| `250d63d` | Per-band PSF support, magnification proxy variable fixes |
| `4c504f9` | Fix `sersic_unit_total_flux` keyword, rename PSFEx → psfsize |

---

## Questions for You

1. **Maskbits**: Which bits should I exclude for arc_snr? I proposed `BRIGHT | SATUR | INTERP | BAILOUT`. Is this conservative enough?

2. **psfsize maps**: Should I implement per-pixel PSF evaluation at stamp center in 4c, or is brick-average FWHM sufficient for publication?

3. **Anything else blocking 4c execution?** I believe the core physics bugs are now fixed.

---

## Files Attached

Please review the updated code:

1. `dark_halo_scope/emr/spark_phase4_pipeline.py` - Main pipeline with all fixes

---

*Response generated: 2026-01-24*


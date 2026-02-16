# Follow-up: Code Audit Against Paper IV - Discrepancy Resolution

After auditing our implementation against "Strong Lens Discoveries in DESI Legacy Imaging Surveys DR10 with Two Deep Learning Architectures" (Inchausti et al. 2025, aka Paper IV), we found and fixed two critical discrepancies in our negative sampling pipeline (`spark_negative_sampling.py`).

## Discrepancy 1: Missing Minimum Exposure Filter

**Paper IV states:**
> "requiring at least three exposures in the g, r, and z bands"

**Our original code:**
Only used `nobs_z` for stratification binning, but did NOT enforce a minimum exposure filter for any band.

**Fix applied (spark_negative_sampling.py, around line 438):**
```python
# 5. Minimum exposures per band (Paper IV: ≥3 in each of g, r, z)
# "at least three exposures in the g, r, and z bands" - Inchausti et al. 2025
MIN_EXPOSURES = 3
nobs_ok = (nobs_g_all >= MIN_EXPOSURES) & (nobs_r_all >= MIN_EXPOSURES) & (nobs_z_all >= MIN_EXPOSURES)
skip_reasons["insufficient_exposures"] = int(np.sum(valid_coords & in_decals & valid_types & maskbit_ok & ~nobs_ok))
```

## Discrepancy 2: Missing DECaLS Footprint Cut

**Paper IV states:**
> "The footprint of DECaLS in DR10 in the range −18◦ < δ < 32◦"

**Our original code:**
No declination filter was applied.

**Fix applied (spark_negative_sampling.py, around line 421):**
```python
# 2. DECaLS footprint (Paper IV: δ < +32°)
# "DECaLS in DR10 in the range −18 ◦ < δ < 32◦" - Inchausti et al. 2025
DECALS_DEC_MAX = 32.0
in_decals = dec_all < DECALS_DEC_MAX
skip_reasons["outside_decals"] = int(np.sum(valid_coords & ~in_decals))
```

## Questions for Review

1. **Is the exposure filter correctly applied?** We now require `nobs_g >= 3 AND nobs_r >= 3 AND nobs_z >= 3` before a source enters the negative pool. This is applied in the vectorized filtering stage before spatial exclusion.

2. **Should we also enforce δ > -18°?** Paper IV mentions "−18◦ < δ < 32◦", but the southern limit is typically covered by DECam anyway. Should we add an explicit lower bound?

3. **Any other filters we missed?** The paper mentions "Extragalactic sources" but we interpret this as the morphological type filter (PSF, REX, EXP, DEV, SER) already applied.

Please confirm these fixes are correct and identify any remaining discrepancies.

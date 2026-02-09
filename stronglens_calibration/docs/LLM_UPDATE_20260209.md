# LLM Update - 2026-02-09

## Context

This is an update to our ongoing conversation about the `stronglens_calibration` project. The previous conversation ended with your recommendations on:
1. Stratified sampling (spark_stratified_sample.py)
2. Cutout validation (spark_validate_cutouts.py)

Since then, we have made progress and encountered issues that need your review before proceeding.

---

## Summary of Changes Since Last Conversation

### 1. N2 Classification Fix (CRITICAL)

**Problem Discovered:** The `classify_pool_n2()` function in `emr/sampling_utils.py` was producing **0% N2** (hard confusers) when run against DR10 data. All negatives were being classified as N1.

**Root Cause:** Thresholds were too restrictive:
- `flux_r_min: 10 nMgy` (only ~5% of DR10 galaxies have flux this high)
- `shape_r_min: 2.0 arcsec` (only ~5% have half-light radius this large)
- `g_minus_r_max: 0.4 AND r_mag_max: 19.0` (very few galaxies are both this blue AND this bright)

**Fix Applied:** Recalibrated thresholds against realistic DR10 distributions:

| Category | Old Threshold | New Threshold | Result |
|----------|---------------|---------------|--------|
| ring_proxy | DEV only, flux ≥ 10 | DEV/SER, flux ≥ 5, sersic ≥ 4.0 (SER) | 6.6% |
| edge_on_proxy | EXP only, shape_r ≥ 2.0 | Any type, ellipticity ≥ 0.55, shape_r ≥ 0.6 | 0.5% |
| blue_clumpy | g-r ≤ 0.4, mag ≤ 19.0 | g-r ≤ 0.4, mag ≤ 20.5 | 5.7% |
| large_galaxy | (new category) | shape_r ≥ 2.0, flux ≥ 3.0 | 3.7% |
| bright_core | (removed) | Redundant with ring_proxy | - |

**New N2 Rate:** 16.5% (target was 10-25%, previously 0%)

**Files Modified:**
- `emr/sampling_utils.py` - Updated `classify_pool_n2()` with new thresholds
- `configs/negative_sampling_v1.yaml` - Updated configuration
- `tests/test_n2_classification.py` - Updated test cases (23/23 pass)

### 2. Step 1 Crossmatch Complete

**Status:** ✅ COMPLETE

**Results:**
- 4,788/5,104 positives matched to DR10 Tractor (93.8%)
- 316 unmatched (outside DR10 coverage)
- Median separation: 0.059"
- Tier-A: 389 confident lenses
- Tier-B: 4,399 probable lenses
- Output: `s3://darkhaloscope/stronglens_calibration/positives_with_dr10/20260208_180524/`

### 3. New Documentation Created

- `docs/TECHNICAL_SPECIFICATIONS.md` - NPZ format, manifest schema, split logic, training config details
- `docs/FULL_PROJECT_CONTEXT.md` - Comprehensive project overview
- `docs/LESSONS_LEARNED.md` - Migrated from dark_halo_scope

### 4. Implementation Status

| Step | Status | Notes |
|------|--------|-------|
| Positive crossmatch | ✅ Done | 4,788 matched |
| N2 classification fix | ✅ Done | 16.5% N2 rate |
| Negative sampling (EMR) | ❌ Not run | Ready to run with fixed thresholds |
| Stratified 100:1 sampling | ❌ Not run | Blocked by negative sampling |
| Cutout generation | ❌ Not run | Blocked by stratified sampling |
| Training manifest | ❌ Not run | Blocked by cutouts |
| Training | ❌ Not run | Blocked by manifest |

---

## Your Previous Recommendations - Implementation Status

### Priority 1 (Must Fix)

| Recommendation | Status | Notes |
|----------------|--------|-------|
| Update TYPE_BINS to ["SER","DEV","REX"] | ✅ Done | EXP excluded per Paper IV |
| Replace annulus 20-40 px with 4-16 px | ✅ Done | In spark_validate_cutouts.py |
| Make stratified sampling deterministic | ⏳ Partial | Need to verify hash ordering implementation |

### Priority 2 (Strongly Recommended)

| Recommendation | Status | Notes |
|----------------|--------|-------|
| Multi-radius core features (r=4/8/12) | ✅ Done | CORE_RADII = [4, 8, 12] in validation |
| Bootstrap AUC for shortcut flags | ☐ Pending | Not yet implemented |
| High-frequency and azimuthal asymmetry features | ⏳ Partial | Azimuthal asymmetry added |

### Priority 3 (Nice-to-have)

| Recommendation | Status | Notes |
|----------------|--------|-------|
| Normalization sanity stats | ✅ Done | Per-band percentiles, MAD stored |
| Band-provenance checks | ☐ Pending | Unit test needed |

---

## Files Included in This Update

### Modified Files (since last conversation)
1. `emr/sampling_utils.py` - N2 classification fix
2. `configs/negative_sampling_v1.yaml` - Updated N2 thresholds
3. `tests/test_n2_classification.py` - Updated tests

### New Files
1. `docs/TECHNICAL_SPECIFICATIONS.md` - Technical specs document
2. `docs/FULL_PROJECT_CONTEXT.md` - Project overview
3. `docs/LESSONS_LEARNED.md` - Migrated lessons

---

## Questions for LLM Review

### Critical Decision: Should We Proceed with EMR Negative Sampling?

**Context:** We are ready to run `spark_negative_sampling.py` on EMR with the fixed N2 thresholds. This will process ~114M galaxies from DR10 sweeps.

**Question 1:** Given the N2 category distribution below, is this ready for production?

```
ring_proxy:    6.6% (bright DEV/SER with n≥4)
blue_clumpy:   5.7% (g-r ≤ 0.4, mag ≤ 20.5)
large_galaxy:  3.7% (shape_r ≥ 2.0", flux ≥ 3 nMgy)
edge_on_proxy: 0.5% (ellipticity ≥ 0.55)
Total N2:     16.5%
```

**Concerns:**
- Edge-on proxy is low (0.5%) - should we loosen ellipticity threshold?
- We removed bright_core category - was this correct?
- Is 16.5% N2 appropriate, or should we aim for exactly 15%?

### Question 2: N2 Category Completeness

Are we missing any important confuser categories that Paper IV or other lens-finding papers identify? Specifically:
- Spiral galaxies with prominent arms
- Mergers/disturbed morphology
- Star-forming clumps that could look like multiple images
- AGN/QSO hosts with point source cores

### Question 3: Stratified Sampling Determinism

You recommended using hash ordering + window for determinism. Here's our current implementation in `sampling_utils.py`:

```python
def assign_split(healpix_idx: int, allocations: Dict[str, float], seed: int = 42) -> str:
    hash_input = f"{healpix_idx}_{seed}"
    hash_bytes = hashlib.sha256(hash_input.encode()).digest()
    hash_value = int.from_bytes(hash_bytes[:4], "big") / (2**32)
    # ... assign based on cumulative thresholds
```

Is this implementation correct for spatial splits? For stratified sampling within strata, should we use a similar approach?

### Question 4: Cutout Size Decision

We currently have two sizes in the codebase:
- 101×101 pixels (downloaded from Legacy Survey)
- 64×64 pixels (center-cropped for training)

Paper IV uses what size? Should we:
a) Train on 64×64 (faster, matches some prior work)
b) Train on 101×101 (more context, larger field of view)
c) Train on both as ablation

### Question 5: Label Handling Implementation

We have not yet implemented the tier-based sample weights you recommended:
- Tier-A (confident): weight 0.9-1.0
- Tier-B (probable): weight 0.3-0.6

Before we implement, please confirm:
1. Is this applied as `sample_weight` in BCE loss?
2. Should we also apply label smoothing (1.0 → 0.95) for Tier-B?
3. How do we handle the 822 spectroscopic matches (DESI DR1) - are these separate from Tier-A?

### Question 6: Selection Function Grid

You previously confirmed the grid:
- θE: 0.5-3.0" in 0.25" steps (11 bins)
- PSF: 0.9-1.8" in 0.15" steps (7 bins)
- Depth: 22.5-24.5 in 0.5 mag steps (5 bins)
- Total: 385 cells, minimum 200 injections per cell

**Question:** Since we're doing real-image training (not injection-based training), when do injections happen?
a) After training, for selection function measurement only?
b) During validation, to test model response to synthetic lenses?
c) Both?

### Question 7: GO/NO-GO for Next Step

Given the current state:
- ✅ Positive crossmatch complete (4,788 matched)
- ✅ N2 classification fixed (16.5% rate)
- ✅ HEALPix spatial splits implemented
- ✅ Exclusion radius implemented (11")
- ☐ Full negative sampling not yet run
- ☐ Stratified 100:1 sampling not yet run
- ☐ Cutouts not yet generated

**Should we proceed with running `spark_negative_sampling.py` on EMR?**

Please provide:
1. Any blocking issues that must be fixed first
2. Any non-blocking issues to track
3. Expected output validation checks

---

## Appendix: Key Code Excerpts

### A. N2 Classification (emr/sampling_utils.py)

```python
DEFAULT_N2_THRESHOLDS = {
    "ring_proxy": {
        "types": ["DEV", "SER"],  # DEV always qualifies, SER needs high sersic
        "flux_r_min": 5.0,        # ~mag 20.8
        "sersic_min": 4.0,        # Very high concentration (SER only)
    },
    "edge_on_proxy": {
        "types": ["EXP", "SER", "DEV"],
        "ellipticity_min": 0.55,
        "shape_r_min": 0.6,
        "shape_r_min_legacy": 2.0,  # Fallback if no ellipticity
    },
    "blue_clumpy_proxy": {
        "g_minus_r_max": 0.4,
        "r_mag_max": 20.5,
    },
    "large_galaxy_proxy": {
        "shape_r_min": 2.0,
        "flux_r_min": 3.0,
    },
}
```

### B. Spatial Split Logic (emr/sampling_utils.py)

```python
def compute_healpix(ra: float, dec: float, nside: int) -> int:
    import healpy as hp
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    return int(hp.ang2pix(nside, theta, phi, nest=True))

def assign_split(healpix_idx: int, allocations: Dict[str, float], seed: int = 42) -> str:
    hash_input = f"{healpix_idx}_{seed}"
    hash_bytes = hashlib.sha256(hash_input.encode()).digest()
    hash_value = int.from_bytes(hash_bytes[:4], "big") / (2**32)
    cumulative = 0.0
    for split, proportion in sorted(allocations.items()):
        cumulative += proportion
        if hash_value < cumulative:
            return split
    return "train"
```

### C. Cutout NPZ Format

```python
# From emr/spark_generate_cutouts.py
np.savez_compressed(
    buffer, 
    cutout=cutout,  # shape (101, 101, 3), channels: g/r/z
    **{f"meta_{k}": v for k, v in metadata.items()}
)

# Metadata keys:
# meta_galaxy_id, meta_ra, meta_dec, meta_type, meta_nobs_z,
# meta_psfsize_z, meta_flux_r, meta_cutout_url, meta_download_timestamp
```

---

## Requested Output

1. **N2 thresholds validation:** Are the new thresholds appropriate? Any adjustments needed?

2. **Missing confuser categories:** List any N2 categories we should add.

3. **GO/NO-GO decision:** Can we proceed with EMR negative sampling?

4. **Cutout size recommendation:** 64×64 vs 101×101 for training.

5. **Label weighting confirmation:** Exact implementation for tier-based sample weights.

6. **Injection timing:** When should injections be used in the real-image training pipeline?

7. **Any other blocking issues** before running the EMR job.

---

*Update prepared: 2026-02-09*

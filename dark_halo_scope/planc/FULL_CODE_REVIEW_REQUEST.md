# Full Code Review Request: Strong Lens Pipeline for Paper IV Parity

## Context

We are building a pipeline to generate training data for strong lens detection in DESI Legacy Imaging Surveys DR10. Our goal is to match the methodology in **"Strong Lens Discoveries in DESI Legacy Imaging Surveys DR10 with Two Deep Learning Architectures"** (Inchausti et al. 2025, hereafter Paper IV).

The attached `code_review_package.zip` contains all relevant code. Please review it for correctness and Paper IV compliance.

---

## Pipeline Overview

Our pipeline has 5 main stages:

1. **Positive Crossmatch** (`spark_crossmatch_positives_v2.py`): Match known lens catalogs against DR10 sweeps
2. **Negative Sampling** (`spark_negative_sampling.py`): Sample non-lens galaxies from DR10 sweeps
3. **Stratified Sampling** (`spark_stratified_sample.py`): Balance negatives to match positive distribution
4. **Cutout Generation** (`spark_generate_cutouts.py`): Download 101×101 pixel g,r,z cutouts
5. **Cutout Validation** (`spark_validate_cutouts.py`): Quality checks and shortcut detection

---

## Critical Code Sections for Review

### 1. Negative Sampling Filters (spark_negative_sampling.py, lines 417-462)

```python
# 1. Valid coordinates
valid_coords = np.isfinite(ra_all) & np.isfinite(dec_all)
skip_reasons["invalid_coords"] = int(np.sum(~valid_coords))

# 2. DECaLS footprint (Paper IV: −18° < δ < +32°)
# "DECaLS in DR10 in the range −18 ◦ < δ < 32◦" - Inchausti et al. 2025
DECALS_DEC_MIN = -18.0
DECALS_DEC_MAX = 32.0
in_decals = (dec_all > DECALS_DEC_MIN) & (dec_all < DECALS_DEC_MAX)
skip_reasons["outside_decals"] = int(np.sum(valid_coords & ~in_decals))

# 3. Valid galaxy types (N1 pool)
valid_types = np.isin(types_all, list(VALID_TYPES_N1))
skip_reasons["not_valid_type"] = int(np.sum(valid_coords & in_decals & ~valid_types))

# 4. Maskbit exclusions (vectorized bitwise check)
exclude_mask_combined = 0
for bit in exclude_maskbits:
    exclude_mask_combined |= (1 << bit)
maskbit_ok = (maskbits_all & exclude_mask_combined) == 0
skip_reasons["maskbit"] = int(np.sum(valid_coords & in_decals & valid_types & ~maskbit_ok))

# 5. Minimum exposures per band (Paper IV: ≥3 in each of g, r, z)
# "at least three exposures in the g, r, and z bands" - Inchausti et al. 2025
MIN_EXPOSURES = 3
nobs_g_ok = nobs_g_all >= MIN_EXPOSURES
nobs_r_ok = nobs_r_all >= MIN_EXPOSURES
nobs_z_ok = nobs_z_all >= MIN_EXPOSURES
nobs_ok = nobs_g_ok & nobs_r_ok & nobs_z_ok

# Per-band failure counts for diagnostics
pre_nobs_mask = valid_coords & in_decals & valid_types & maskbit_ok
skip_reasons["nobs_g_lt_3"] = int(np.sum(pre_nobs_mask & ~nobs_g_ok))
skip_reasons["nobs_r_lt_3"] = int(np.sum(pre_nobs_mask & ~nobs_r_ok))
skip_reasons["nobs_z_lt_3"] = int(np.sum(pre_nobs_mask & ~nobs_z_ok))
skip_reasons["insufficient_exposures"] = int(np.sum(pre_nobs_mask & ~nobs_ok))

# 6. Z-band magnitude limit
with np.errstate(divide='ignore', invalid='ignore'):
    mag_z_all = np.where(flux_z_all > 0, 22.5 - 2.5 * np.log10(flux_z_all), np.nan)
mag_z_ok = np.isnan(mag_z_all) | (mag_z_all < z_mag_limit)
skip_reasons["mag_z_faint"] = int(np.sum(valid_coords & in_decals & valid_types & maskbit_ok & nobs_ok & ~mag_z_ok))

# Combined mask before spatial query
pre_spatial_mask = valid_coords & in_decals & valid_types & maskbit_ok & nobs_ok & mag_z_ok
```

### 2. Galaxy Type Constants (sampling_utils.py)

```python
# Galaxy types for Pool N1
VALID_TYPES_N1 = {"SER", "DEV", "REX", "EXP"}

# nobs_z bin edges: [1-2], [3-5], [6-10], [11+]
NOBS_Z_BINS = [(1, 2), (3, 5), (6, 10), (11, 999)]

# Maskbits to exclude (from config, but hardcoded defaults)
DEFAULT_EXCLUDE_MASKBITS = {1, 5, 6, 7, 11, 12, 13}
```

### 3. N2 Confuser Classification (sampling_utils.py, lines 177-210)

```python
def classify_pool_n2(
    galaxy_type: str,
    flux_r: Optional[float],
    shape_r: Optional[float],
    g_minus_r: Optional[float],
    mag_r: Optional[float],
    config: Dict[str, Any]
) -> Optional[str]:
    """
    Classify galaxy into N2 confuser categories based on Tractor properties.
    """
    n2_config = config.get("negative_pools", {}).get("pool_n2", {}).get("tractor_criteria", {})
    
    # Ring proxy: DEV with bright flux
    ring_cfg = n2_config.get("ring_proxy", {})
    if galaxy_type == "DEV":
        if flux_r is not None and flux_r >= ring_cfg.get("flux_r_min", 10):
            return "ring_proxy"
    
    # Edge-on proxy: EXP with large half-light radius
    edge_on_cfg = n2_config.get("edge_on_proxy", {})
    if galaxy_type == "EXP":
        if shape_r is not None and shape_r >= edge_on_cfg.get("shape_r_min", 2.0):
            return "edge_on_proxy"
    
    # Blue clumpy proxy: blue color
    blue_cfg = n2_config.get("blue_clumpy_proxy", {})
    if g_minus_r is not None and g_minus_r <= blue_cfg.get("g_minus_r_max", 0.4):
        if mag_r is not None and mag_r <= blue_cfg.get("r_mag_max", 19.0):
            return "blue_clumpy"
    
    return None
```

### 4. Cutout Generation (spark_generate_cutouts.py)

```python
# Cutout parameters
CUTOUT_SIZE = 101  # pixels
PIXEL_SCALE = 0.262  # arcsec/pixel
CUTOUT_ARCSEC = CUTOUT_SIZE * PIXEL_SCALE  # ~26.5 arcsec

# Legacy Survey cutout service
CUTOUT_URL_TEMPLATE = (
    "https://www.legacysurvey.org/viewer/fits-cutout"
    "?ra={ra}&dec={dec}&size={size}&layer=ls-dr10&pixscale={pixscale}&bands={bands}"
)

def download_cutout(...) -> Tuple[Optional[np.ndarray], Dict]:
    # ... 
    with fits.open(buffer) as hdul:
        data = hdul[0].data  # Shape: (n_bands, height, width)
        cutout = np.transpose(data, (1, 2, 0)).astype(np.float32)  # -> (H, W, 3)
```

### 5. Configuration File (negative_sampling_v1.yaml)

```yaml
negative_pools:
  pool_n1:
    valid_types: ["SER", "DEV", "REX", "EXP"]
    z_mag_limit: 21.0
    exclusion_radius_arcsec: 10.0
    nobs_z_bins: [[1, 2], [3, 5], [6, 10], [11, null]]
  pool_n2:
    tractor_criteria:
      ring_proxy:
        flux_r_min: 10.0
      edge_on_proxy:
        shape_r_min: 2.0
      blue_clumpy_proxy:
        g_minus_r_max: 0.4
        r_mag_max: 19.0
```

---

## Direct Questions for Review

### A. Footprint and Selection Filters

1. **DECaLS footprint bounds**: We use `-18° < δ < 32°`. Is this the exact range Paper IV used? Should it be `≤` instead of `<` on either bound?

2. **Galactic plane / extinction cut**: Paper IV mentions "extragalactic sources". Did they apply any E(B-V) threshold or |b| (Galactic latitude) cut to exclude the Galactic plane? We currently have no such filter.

3. **Magnitude cut consistency**: We use `z_mag < 21.0` for the faint limit. Does Paper IV specify a magnitude limit for their scanned sample? If so, which band and what value?

4. **NOBS definition**: We use `NOBS_G`, `NOBS_R`, `NOBS_Z` columns from DR10 sweeps. Are these the correct "number of exposures" fields that Paper IV refers to, or is there a different column (e.g., per-camera counts)?

### B. Galaxy Type Selection

5. **TYPE values**: We accept `{"SER", "DEV", "REX", "EXP"}` as valid galaxy types. Does Paper IV include or exclude any of these? Does Paper IV also include `PSF` type sources for any reason?

6. **Star-galaxy separation**: Beyond TYPE filtering, did Paper IV apply any additional star-galaxy separation (e.g., using `DCHISQ` or `FRACFLUX` columns)?

### C. Maskbit Handling

7. **Maskbit list**: We exclude bits `{1, 5, 6, 7, 11, 12, 13}`. These correspond to:
   - 1: BRIGHT (near bright source)
   - 5: ALLMASK_G
   - 6: ALLMASK_R  
   - 7: ALLMASK_Z
   - 11: MEDIUM (medium-bright star)
   - 12: GALAXY (near large galaxy)
   - 13: CLUSTER (near globular cluster)
   
   Does Paper IV specify their maskbit exclusions? Are we missing any critical bits (e.g., WISE-related bits)?

### D. N2 Confuser Categories

8. **Ring proxy criteria**: We classify `DEV` galaxies with `flux_r >= 10 nanomaggies` as ring proxies. Does Paper IV define ring galaxies differently? Should we also check ellipticity or Sersic index?

9. **Edge-on proxy criteria**: We classify `EXP` galaxies with `shape_r >= 2.0 arcsec` as edge-on proxies. Does Paper IV use axis ratio (b/a) or ellipticity instead?

10. **Blue clumpy criteria**: We use `g-r <= 0.4` and `r_mag <= 19.0`. What are Paper IV's thresholds for blue star-forming galaxies that can mimic arcs?

11. **Missing confuser categories**: Does Paper IV identify other confuser types we should include (e.g., mergers, tidal tails, bright star artifacts)?

### E. Cutout Generation

12. **Cutout size**: We use 101×101 pixels at 0.262 arcsec/pixel (~26.5 arcsec). Does Paper IV use the same size and pixel scale?

13. **Band ordering**: We request bands as "grz" and the FITS returns shape `(3, H, W)`. We transpose to `(H, W, 3)` with channel order [g, r, z]. Is this the correct channel ordering for Paper IV's models?

14. **Normalization**: Paper IV likely applies some normalization before feeding to CNNs. What normalization scheme do they use? (asinh, percentile clipping, per-channel standardization?)

### F. Spatial Exclusion

15. **Exclusion radius**: We exclude negatives within 10 arcsec of known lenses. Does Paper IV use a different radius? Should it scale with Einstein radius?

16. **Known lens catalog**: Which lens catalogs should be used for exclusion? (Master Lens Database, SuGOHI, our own training positives?)

### G. Train/Val/Test Split

17. **Spatial split method**: We use HEALPix-based spatial splitting with NSIDE=128. Does Paper IV use spatial splits or random splits?

18. **Split proportions**: We use 70/15/15 for train/val/test. What proportions does Paper IV use?

### H. End-to-End Consistency

19. **Sample sizes**: Paper IV scanned ~43M galaxies and found ~4,000 high-confidence candidates. Our negative pool targets ~500K total. Is this ratio reasonable for a training set?

20. **N1:N2 ratio**: We target 85% N1 (representative) and 15% N2 (confusers). Does Paper IV specify a similar ratio, or should we adjust?

21. **Stratification**: We stratify by `(nobs_z_bin, type_bin)`. Does Paper IV stratify by additional variables (e.g., seeing, depth, magnitude)?

---

## Actual Pipeline Run Results (2026-02-08)

We completed a full negative sampling run on DR10 sweeps. Here are the actual results:

### Output Summary
- **Total output size**: 3.3 GiB (60 parquet partitions)
- **Rows per partition**: ~470K
- **Estimated total rows**: ~28M negative candidates

### Pool Distribution (from sample partition)
| Pool | Count | Percentage |
|------|-------|------------|
| N1 (representative) | 442,359 | 94.2% |
| N2 (confusers) | 27,427 | 5.8% |

### N2 Confuser Category Breakdown
| Category | Count | Percentage of N2 |
|----------|-------|------------------|
| ring_proxy | 18,318 | 66.8% |
| edge_on_proxy | 7,840 | 28.6% |
| blue_clumpy | 1,269 | 4.6% |

### Output Schema (48 columns)
```
galaxy_id, brickname, objid, ra, dec, type, nobs_z, nobs_z_bin, type_bin,
flux_g, flux_r, flux_z, flux_w1, mag_g, mag_r, mag_z, 
g_minus_r, r_minus_z, z_minus_w1,
psfsize_g, psfsize_r, psfsize_z, psfdepth_g, psfdepth_r, psfdepth_z,
galdepth_g, galdepth_r, galdepth_z, ebv, maskbits, fitbits,
mw_transmission_g, mw_transmission_r, mw_transmission_z,
shape_r, shape_e1, shape_e2, sersic, healpix_64, healpix_128, split,
pool, confuser_category, sweep_file, row_index, 
pipeline_version, git_commit, extraction_timestamp
```

---

## Critical Observations Requiring Clarification

### Observation 1: N2 ratio is 5.8%, not 15%
Our N2 pool is only 5.8% of total, not the 15% we targeted. This suggests either:
- (a) Our confuser criteria are too strict
- (b) True confusers are genuinely rare in the population
- (c) We're missing confuser categories

**Question**: Is 5.8% N2 acceptable, or should we relax thresholds to reach 15%?

### Observation 2: Ring proxies dominate N2
66.8% of N2 are ring_proxy (DEV with flux_r >= 10). Only 4.6% are blue_clumpy.

**Question**: Is this distribution expected? Should we adjust thresholds to balance categories?

### Observation 3: Blue clumpy threshold may be too strict
Only 1,269 blue_clumpy objects found (g-r <= 0.4, r_mag <= 19.0).

**Question**: What g-r threshold does Paper IV use for "blue star-forming" confusers? Is 0.4 too blue?

---

## Summary of Known Discrepancies Already Fixed

1. ✅ Added `nobs_g >= 3 AND nobs_r >= 3 AND nobs_z >= 3` filter (was missing)
2. ✅ Added `dec > -18°` lower bound (was only `dec < 32°`)
3. ✅ Fixed FITS parsing to correctly read `hdul[0].data` with shape `(n_bands, H, W)`
4. ✅ Added per-band nobs failure logging for diagnostics

---

## Additional Questions Based on Run Results

### I. N2 Classification Thresholds

22. **Ring proxy flux threshold**: We use `flux_r >= 10 nanomaggies` (~19.5 mag). Is this the right threshold for identifying galaxies with ring-like structure? Should we also require high ellipticity or specific Sersic index?

23. **Edge-on half-light radius**: We use `shape_r >= 2.0 arcsec`. Paper IV likely uses axis ratio (b/a < 0.3?) instead. What is the correct criterion for edge-on spirals?

24. **Blue clumpy color cut**: We use `g-r <= 0.4`. Should this be `g-r <= 0.6` or another value? What magnitude range?

25. **Merger/interaction criteria**: We don't identify mergers. Should we add criteria based on asymmetry or multiple components?

### J. Positive Sample Handling

26. **Positive catalog sources**: We use 5,104 positives from DESI_master_full.csv. Does Paper IV use the same catalog, or a different/larger set?

27. **Positive-negative balance**: With ~28M negatives and ~5K positives, our ratio is ~5600:1 before stratified sampling. What ratio does Paper IV use for training?

28. **Grade/confidence weighting**: Should we weight positives by their confidence grade (A/B/C) during training?

### K. Data Quality

29. **PSF size distribution**: We store `psfsize_g/r/z` but don't filter on it. Does Paper IV require seeing < X arcsec?

30. **Depth requirements**: We store `psfdepth_g/r/z` and `galdepth_g/r/z`. Does Paper IV apply minimum depth cuts?

31. **E(B-V) threshold**: We store `ebv` but don't filter. Does Paper IV exclude high-extinction regions (e.g., E(B-V) > 0.1)?

---

## Requested Output

Please provide:

1. **Line-by-line filter parity check**: For each of our filters, confirm if it matches Paper IV or specify the correction needed.

2. **Missing filters**: List any filters Paper IV applies that we don't have.

3. **Parameter corrections**: For each configurable threshold (magnitude limits, radii, color cuts), provide the Paper IV value if different from ours.

4. **N2 category definitions**: Confirm or correct our confuser classification criteria.

5. **Cutout specifications**: Confirm size, pixel scale, band order, and any normalization details.

6. **Priority ranking**: If multiple issues exist, rank them by impact on Paper IV parity.

---

## Attached Files

The `code_review_package.zip` contains:
- `spark_negative_sampling.py` - Main negative sampling job
- `spark_generate_cutouts.py` - Cutout download job
- `spark_validate_cutouts.py` - Quality validation job
- `spark_stratified_sample.py` - Stratified sampling job
- `spark_crossmatch_positives_v2.py` - Positive catalog crossmatch
- `sampling_utils.py` - Utility functions (N2 classification, binning)
- `negative_sampling_v1.yaml` - Configuration file

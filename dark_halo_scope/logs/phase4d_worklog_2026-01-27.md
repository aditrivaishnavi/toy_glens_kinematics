# Phase 4d Work Log - 2026-01-27

## Executive Summary

Phase 4d completeness estimation was upgraded to publication-grade standards and successfully executed on EMR. Post-fix run improved completeness from 9.91% to **10.25%**.

---

## Post-Fix Run (Final Version)

### Changes Applied Based on LLM Review

| Recommendation | Implementation |
|----------------|----------------|
| Use `psf_fwhm_used_r` instead of `psfsize_r` | Resolution now uses per-stamp PSF with fallback |
| Fix NULL leakage via `.cast("int")` | All flags use `F.when().otherwise(0)` |
| Add diagnostic counters | Added `n_recovered_snr_only`, `n_recovered_res_only` |
| Stricter validity definition | `valid_all` requires `theta_over_psf IS NOT NULL` |

### Run Details (Post-Fix)

| Parameter | Value |
|-----------|-------|
| Cluster ID | j-XXXXX (post-fix run) |
| Core Instances | 5 x m5.xlarge |
| Runtime | ~5 minutes |
| Experiment ID | train_stamp64_bandsgrz_gridgrid_small |
| Timestamp | 2026-01-27 16:41:17 UTC |

---

## Key Metrics (Post-Fix)

### 1. Dataset Structure

| Metric | Value |
|--------|-------|
| Total raw rows (4c) | 10,627,158 |
| Injections (controls excluded) | 5,327,834 |
| Completeness surface rows | 129,192 |
| Region-aggregated rows | 41,170 |
| Unique regions | 782 |

### 2. Global Completeness (IMPROVED)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total attempted | 5,327,834 | 5,327,834 | - |
| Valid (all) | 5,327,834 | 5,327,834 | - |
| Valid (clean) | 4,679,202 | 4,679,202 | - |
| Recovered (all) | 527,968 | **545,838** | +17,870 |
| Recovered (clean) | 489,694 | **507,037** | +17,343 |
| Completeness (all) | 9.91% | **10.25%** | +0.34pp |
| Completeness (clean) | 10.47% | **10.84%** | +0.37pp |

**Key Finding:** Using per-stamp PSF increased recovery by ~17k injections (+3.4%).

### 3. Completeness by Einstein Radius (θ_E)

| θ_E (arcsec) | n_attempt | n_recovered | Completeness | arc_snr_mean |
|--------------|-----------|-------------|--------------|--------------|
| 0.30 | 1,782,906 | 0 | 0.00% | 47.54 |
| 0.60 | 1,764,898 | 0 | 0.00% | 41.42 |
| 1.00 | 1,780,030 | 545,838 | **30.66%** | 40.56 |

**θ_E=1.0" completeness improved from 29.66% to 30.66% (+1.0pp)**

### 4. Completeness by Resolution Bin (θ_E / psf_fwhm_used_r)

| Resolution Bin | n_attempt | n_recovered | Comp (all) | Comp (clean) | θ/PSF mean |
|----------------|-----------|-------------|------------|--------------|------------|
| <0.4 | 2,008,012 | 0 | 0.00% | 0.00% | 0.253 |
| 0.4-0.6 | 1,557,876 | 0 | 0.00% | 0.00% | 0.480 |
| 0.6-0.8 | 1,157,472 | 0 | 0.00% | 0.00% | 0.703 |
| 0.8-1.0 | 596,342 | 538,930 | **90.37%** | **94.61%** | 0.859 |
| ≥1.0 | 8,132 | 6,908 | **84.95%** | **92.47%** | 1.051 |

**≥1.0 bin increased from 5,540 to 8,132 injections due to per-stamp PSF variation**

### 5. Completeness by Source Magnitude

| src_dmag | n_attempt | n_recovered | Completeness | arc_snr_mean |
|----------|-----------|-------------|--------------|--------------|
| 1.0 | 2,661,074 | 286,544 | 10.77% | 61.31 |
| 2.0 | 2,666,760 | 259,294 | 9.72% | 24.71 |

### 6. Completeness by PSF Size

| PSF (arcsec) | n_attempt | n_recovered | Completeness |
|--------------|-----------|-------------|--------------|
| 0.8" | 1,660 | 486 | 29.28% |
| 0.9" | 22,684 | 6,422 | 28.31% |
| 1.0" | 170,888 | 49,872 | 29.18% |
| 1.1" | 887,444 | 268,795 | 30.29% |
| 1.2" | 1,481,568 | 220,263 | 14.87% |
| ≥1.3" | 2,763,590 | 0 | 0.00% |

### 7. Data Quality Impact

| Subset | Valid | Recovered | Completeness |
|--------|-------|-----------|--------------|
| All | 5,327,834 | 545,838 | 10.25% |
| Clean (bad_pixel≤0.2, wise≤0.2) | 4,679,202 | 507,037 | 10.84% |

**Clean fraction:** 87.83%
**Completeness boost from clean:** +0.59pp

### 8. Region Variance

| Metric | Value |
|--------|-------|
| Mean completeness (across region means) | 0.1004 |
| Mean region-to-region std | 0.0148 |
| Max region-to-region std | 0.7071 |
| Avg regions per bin | 3.1 |

### 9. Wilson CI Statistics

| Metric | Value |
|--------|-------|
| Average CI width | 0.4271 |
| Min CI width | 0.0021 |
| Max CI width | 0.8109 |

### 10. PSF Provenance

| Band | Map | Manifest |
|------|-----|----------|
| g | 99.08% | 0.92% |
| r | 99.93% | 0.07% |
| z | 99.95% | 0.05% |

---

## LLM Review Feedback Integration

### Issues Identified by LLM

1. **NULL leakage via `.cast("int")`** - FIXED with `F.when().otherwise(0)`
2. **Resolution using brick-level PSF** - FIXED with `psf_fwhm_used_r`
3. **Missing diagnostic counters** - ADDED `n_recovered_snr_only`, `n_recovered_res_only`
4. **Wilson CI not in pipeline** - CONFIRMED already present (LLM was incorrect)

### LLM's Buggy Patch File

The LLM provided `spark_phase4_pipeline_v2_phase4d_patch2.py` with critical bugs:
- Undefined attributes: `args.res_bins`, `args.snr_th`, `args.sep_th`
- Incorrect argument names not matching CLI parser
- **Action:** Deleted buggy file, integrated valid improvements manually

---

## Configuration

```json
{
  "recovery": {"snr_thresh": 5.0, "theta_over_psf": 0.8},
  "quality_cuts": {"bad_pixel_frac_max": 0.2, "wise_brightmask_frac_max": 0.2},
  "binning": {"psfsize_bin_width": 0.1, "psfdepth_bin_width": 0.25},
  "confidence_interval": {"method": "wilson", "z_score": 1.96, "coverage": "95%"},
  "resolution_source": "psf_fwhm_used_r with fallback to psfsize_r"
}
```

---

## Observations for Phase 5

### Training Implications

1. **Resolution dominates**: 90%+ completeness for θ/PSF ≥ 0.8, 0% below
2. **SNR is not limiting**: Median SNR ~40 far exceeds threshold of 5
3. **Per-stamp PSF matters**: 3.4% more recoveries with stamp-level PSF
4. **Clean subset recommended**: 87.8% pass quality cuts

### Model Strategy

Per LLM recommendation:
- Train on **all** injections vs controls (not just "recovered")
- Report metrics stratified by θ/PSF and PSF bins
- Model-based completeness (Phase 5/6) will replace proxy-based

---

## Files and Outputs

| Output | Path |
|--------|------|
| Completeness surfaces | s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/completeness_surfaces/train_stamp64_bandsgrz_gridgrid_small |
| Region-aggregated | s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/completeness_surfaces_region_agg/train_stamp64_bandsgrz_gridgrid_small |
| PSF provenance | s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/psf_provenance/train_stamp64_bandsgrz_gridgrid_small |
| Analysis report (v2) | s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/analysis/train_stamp64_bandsgrz_gridgrid_small_v2 |
| Stage config | s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/_stage_config_train_stamp64_bandsgrz_gridgrid_small.json |

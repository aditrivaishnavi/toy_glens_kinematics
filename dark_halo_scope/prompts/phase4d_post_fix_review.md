# Phase 4d Post-Fix Review Request

## Context: What Changed Based on Your Feedback

This is a follow-up to your previous review of Phase 4d. We implemented your recommendations and reran the pipeline. This document contains the updated metrics for your verification.

### Changes Implemented from Your Previous Review

| Your Recommendation | What We Did |
|---------------------|-------------|
| **Use `psf_fwhm_used_r` instead of `psfsize_r`** | ✅ Implemented. Resolution (`theta_over_psf`) now uses the actual per-stamp PSF that was used to convolve the injected arc in Phase 4c, with fallback to manifest-level `psfsize_r`. |
| **Fix NULL leakage via `.cast("int")`** | ✅ Implemented. All validity/recovery flags now use `F.when(expr, 1).otherwise(0)` instead of `.cast("int")`. Added explicit NULL checks for all terms. |
| **Add diagnostic counters** | ✅ Implemented. New columns: `n_recovered_snr_only` (valid + SNR ≥ 5) and `n_recovered_res_only` (valid + θ/PSF ≥ 0.8) to quantify which criterion dominates. |
| **Add stricter validity definition** | ✅ Implemented. `valid_all` now requires `theta_over_psf IS NOT NULL` in addition to `cutout_ok=1` and `arc_snr IS NOT NULL`. |

### Your Provided Code

You provided `spark_phase4_pipeline_phase4d_ci.py`. We reviewed it and found:
- **Bugs**: The code referenced undefined attributes (`args.res_bins`, `args.snr_th`, `args.sep_th`, `args.bad_pixel_max`, `args.wise_brightmask_max`) that don't match the CLI parser.
- **Action**: We did NOT use your file directly. Instead, we extracted the valid improvements and applied them to our main pipeline (`spark_phase4_pipeline.py`), fixing the attribute name issues.
- **Cleanup**: We deleted the buggy standalone files to avoid confusion.

### Impact of Changes

| Metric | Before (psfsize_r) | After (psf_fwhm_used_r) | Change |
|--------|-------------------|------------------------|--------|
| Overall completeness | 9.91% | **10.25%** | +0.34pp |
| θ_E=1.0" completeness | 29.66% | **30.66%** | +1.0pp |
| θ_E=1.0" clean | 31.32% | **32.43%** | +1.1pp |
| Recovered (all) | 527,968 | **545,838** | +17,870 |

The increase is expected: the per-stamp PSF can be smaller than the brick-level manifest PSF, leading to higher θ/PSF ratios and more injections passing the threshold.

---

## Updated Metrics (Post-Fix)

### 1. Schema (Now Includes Diagnostic Columns)

**Completeness Surfaces Schema (30 columns):**
```
region_id, selection_set_id, ranking_mode, theta_e_arcsec, src_dmag, 
src_reff_arcsec, psf_bin, depth_bin, resolution_bin,
n_attempt, n_valid_all, n_valid_clean, 
n_recovered_all, n_recovered_clean,
n_recovered_snr_only,    <-- NEW: valid + SNR threshold only
n_recovered_res_only,    <-- NEW: valid + resolution threshold only
arc_snr_mean, arc_snr_p50, theta_over_psf_mean,
completeness_valid_all, completeness_valid_clean,
completeness_overall_all, completeness_overall_clean,
valid_frac_all, valid_frac_clean,
ci_low_valid_all, ci_high_valid_all,
ci_low_valid_clean, ci_high_valid_clean,
region_split
```

### 2. Dataset Structure

| Metric | Value |
|--------|-------|
| Completeness surfaces (detailed) | 129,192 rows |
| Region-aggregated surfaces | 41,170 rows |
| Unique regions | 782 |

**Split Distribution:**

| Split | Rows | n_attempt | n_valid_all | n_recovered_all |
|-------|------|-----------|-------------|-----------------|
| test | 31,595 | 1,842,392 | 1,842,392 | 197,599 |
| train | 62,041 | 1,380,816 | 1,380,816 | 150,089 |
| val | 35,556 | 2,104,626 | 2,104,626 | 198,150 |

### 3. Overall Completeness

| Metric | Value |
|--------|-------|
| Total attempted | 5,327,834 |
| Total valid (all) | 5,327,834 (100.0%) |
| Total valid (clean) | 4,679,202 (87.8%) |
| Total recovered (all) | 545,838 |
| Total recovered (clean) | 507,037 |
| **completeness_valid_all** | **0.1025 (10.25%)** |
| **completeness_valid_clean** | **0.1084 (10.84%)** |
| valid_fraction_all | 1.0000 (100.0%) |
| valid_fraction_clean | 0.8783 (87.8%) |

### 4. Completeness by θ_E (Einstein Radius)

| θ_E (") | n_attempt | n_recovered | comp_valid_all | comp_valid_clean | arc_snr_mean |
|---------|-----------|-------------|----------------|------------------|--------------|
| 0.30 | 1,782,906 | 0 | 0.0000 | 0.0000 | 47.54 |
| 0.60 | 1,764,898 | 0 | 0.0000 | 0.0000 | 41.42 |
| 1.00 | 1,780,030 | 545,838 | **0.3066** | **0.3243** | 40.56 |

### 5. Completeness by Resolution Bin (θ_E / psf_fwhm_used_r)

| res_bin | n_attempt | n_recovered | comp_all | comp_clean | θ/PSF mean | arc_snr |
|---------|-----------|-------------|----------|------------|------------|---------|
| <0.4 | 2,008,012 | 0 | 0.0000 | 0.0000 | 0.253 | 43.74 |
| 0.4-0.6 | 1,557,876 | 0 | 0.0000 | 0.0000 | 0.480 | 42.03 |
| 0.6-0.8 | 1,157,472 | 0 | 0.0000 | 0.0000 | 0.703 | 40.82 |
| 0.8-1.0 | 596,342 | 538,930 | **0.9037** | **0.9461** | 0.859 | 48.83 |
| ≥1.0 | 8,132 | 6,908 | **0.8495** | **0.9247** | 1.051 | 36.22 |

### 6. Completeness by PSF Size

| PSF (") | n_attempt | n_recovered | completeness | Comment |
|---------|-----------|-------------|--------------|---------|
| 0.8 | 1,660 | 486 | 29.3% | Good seeing |
| 0.9 | 22,684 | 6,422 | 28.3% | Good seeing |
| 1.0 | 170,888 | 49,872 | 29.2% | Good seeing |
| 1.1 | 887,444 | 268,795 | 30.3% | Peak recovery |
| 1.2 | 1,481,568 | 220,263 | 14.9% | Transition |
| ≥1.3 | 2,763,590 | 0 | 0.0% | Seeing wall |

### 7. Cross-Tabulation: θ_E × Resolution Bin

| θ_E | res_bin | n_attempt | n_recovered | completeness |
|-----|---------|-----------|-------------|--------------|
| 0.30 | <0.4 | 1,782,906 | 0 | 0.0000 |
| 0.60 | <0.4 | 224,814 | 0 | 0.0000 |
| 0.60 | 0.4-0.6 | 1,531,968 | 0 | 0.0000 |
| 0.60 | 0.6-0.8 | 8,116 | 0 | 0.0000 |
| 1.00 | 0.6-0.8 | 1,149,356 | 0 | 0.0000 |
| 1.00 | 0.8-1.0 | 596,342 | 538,930 | **0.9037** |
| 1.00 | ≥1.0 | 8,132 | 6,908 | **0.8495** |

### 8. Region Variance Analysis

| Metric | Value |
|--------|-------|
| Mean completeness (across region means) | 0.1004 |
| Mean region-to-region std | 0.0148 |
| Max region-to-region std | 0.7071 |
| Avg regions per bin | 3.1 |

**By Split:**
| Split | mean_comp | mean_std |
|-------|-----------|----------|
| test | 0.1116 | 0.0129 |
| train | 0.0914 | 0.0158 |
| val | 0.0964 | 0.0158 |

### 9. Wilson CI Statistics

| Metric | Value |
|--------|-------|
| Average CI width | 0.4271 |
| Min CI width | 0.0021 |
| Max CI width | 0.8109 |

### 10. Clean vs All Subset

| res_bin | comp_all | comp_clean | diff |
|---------|----------|------------|------|
| 0.8-1.0 | 0.9037 | 0.9461 | +0.0424 |
| ≥1.0 | 0.8495 | 0.9247 | +0.0753 |

### 11. PSF Provenance

| Band | Map | Manifest |
|------|-----|----------|
| g | 99.08% | 0.92% |
| r | 99.93% | 0.07% |
| z | 99.95% | 0.05% |

### 12. Selection Set Comparison

| selection_set_id | completeness |
|------------------|--------------|
| strat_balanced_psf_weighted_k75 | 10.86% |
| topk_psf_weighted_k100 | 10.77% |
| strat_custom_psf_weighted_k75 | 10.78% |
| strat_balanced_area_weighted_k75 | 10.30% |
| **topk_density_k100** | **3.84%** (outlier) |

---

## Questions for Verification

### 1. Confirm Fixes Applied Correctly

1. **PSF consistency fix**: The resolution bin distribution changed (e.g., ≥1.0 bin went from 5,540 to 8,132 injections). Does this change align with your expectation that `psf_fwhm_used_r` would yield more resolution diversity than brick-level `psfsize_r`?

2. **NULL leakage fix**: `valid_fraction_all` is exactly 1.0000 (100%). This means all 5.3M injections have non-NULL `cutout_ok`, `arc_snr`, and `theta_over_psf`. Does this confirm the NULL-safety fix is working correctly?

### 2. Remaining Scientific Questions

3. **Diagnostic counters**: We added `n_recovered_snr_only` and `n_recovered_res_only` columns. What specific queries or analyses would you recommend to quantify the relative contribution of each criterion?

4. **Region variance**: You noted the region variance summary may be dominated by tiny bins. With avg 3.1 regions per bin and mean std of 0.0148, do you still consider this problematic? What minimum sample size filter would you recommend?

5. **Wilson CI width**: Average CI width is 0.4271, with widest CIs at n=2 bins. Should we add a minimum `n_valid` threshold (e.g., ≥50) to the output for publication purposes?

### 3. Publication Readiness

6. **Is Phase 4d now suitable as a "proxy resolvability selection function"** with the understanding that final publication will include Phase 5/6 model-based detection completeness?

7. **Are there any remaining code issues** you see in the fixes we applied?

---

## Code Changes Summary

**File modified:** `dark_halo_scope/emr/spark_phase4_pipeline.py`

**Key changes in `stage_4d_completeness()` (lines 2603-2712):**

```python
# NEW: Use per-stamp PSF from injection, fall back to manifest
psf_for_resolution = F.coalesce(F.col("psf_fwhm_used_r"), F.col("psfsize_r"))

df = df.withColumn(
    "theta_over_psf", 
    F.when((psf_for_resolution.isNotNull()) & (psf_for_resolution > 0), 
           F.col("theta_e_arcsec") / psf_for_resolution)
    .otherwise(F.lit(None).cast("double"))
)

# NEW: Stricter validity with explicit NULL handling
valid_all_expr = (
    (F.col("cutout_ok") == 1) & 
    F.col("arc_snr").isNotNull() &
    F.col("theta_over_psf").isNotNull()
)
df = df.withColumn("valid_all", 
    F.when(valid_all_expr, F.lit(1)).otherwise(F.lit(0))  # No .cast("int")
)

# NEW: Diagnostic counters
recovered_snr_only_expr = valid_all_expr & (F.col("arc_snr") >= F.lit(snr_th))
recovered_res_only_expr = valid_all_expr & (F.col("theta_over_psf") >= F.lit(sep_th))
df = df.withColumn("recovered_snr_only", F.when(recovered_snr_only_expr, 1).otherwise(0))
df = df.withColumn("recovered_res_only", F.when(recovered_res_only_expr, 1).otherwise(0))
```

---

## S3 Locations

| Output | Path |
|--------|------|
| Completeness surfaces | s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/completeness_surfaces/train_stamp64_bandsgrz_gridgrid_small |
| Region-aggregated | s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/completeness_surfaces_region_agg/train_stamp64_bandsgrz_gridgrid_small |
| PSF provenance | s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/psf_provenance/train_stamp64_bandsgrz_gridgrid_small |
| Analysis report (v2) | s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/analysis/train_stamp64_bandsgrz_gridgrid_small_v2 |
| Stage config | s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/_stage_config_train_stamp64_bandsgrz_gridgrid_small.json |


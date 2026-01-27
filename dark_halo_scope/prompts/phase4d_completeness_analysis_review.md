# Phase 4d Completeness Analysis - Independent Review Request

## Context

This is a request for independent review of the Phase 4d completeness estimation results from a gravitational lens detection pipeline. The goal is to build a selection function for simulated strong gravitational lenses injected into real DECaLS DR10 survey images.

### Pipeline Overview

1. **Phase 4a**: Generated injection manifests with deterministic parameters
2. **Phase 4b**: Cached coadd images and PSF maps from NERSC
3. **Phase 4c**: Injected simulated lensed arcs into real images, computed metrics
4. **Phase 4d**: Computed completeness surfaces with uncertainty quantification

### Recovery Criteria

An injection is considered "recovered" if:
- `arc_snr >= 5.0` (peak SNR of the injected arc)
- `theta_e / psfsize_r >= 0.8` (resolution criterion)

### Quality Cuts

"Clean" subset additionally requires:
- `bad_pixel_frac <= 0.2` (fraction of stamp with bad pixels)
- `wise_brightmask_frac <= 0.2` (fraction masked by WISE bright stars)

---

## Raw Metrics - Section by Section

### 1. SCHEMA ANALYSIS

**Completeness Surfaces Schema (28 columns):**
```
region_id: int
selection_set_id: string
ranking_mode: string
theta_e_arcsec: double
src_dmag: double
src_reff_arcsec: double
psf_bin: int
depth_bin: int
resolution_bin: string
n_attempt: bigint
n_valid_all: bigint
n_valid_clean: bigint
n_recovered_all: bigint
n_recovered_clean: bigint
arc_snr_mean: double
arc_snr_p50: double
theta_over_psf_mean: double
completeness_valid_all: double
completeness_valid_clean: double
completeness_overall_all: double
completeness_overall_clean: double
valid_frac_all: double
valid_frac_clean: double
ci_low_valid_all: double
ci_high_valid_all: double
ci_low_valid_clean: double
ci_high_valid_clean: double
region_split: string
```

**Region-Aggregated Schema (27 columns):**
```
selection_set_id: string
ranking_mode: string
theta_e_arcsec: double
src_dmag: double
src_reff_arcsec: double
psf_bin: int
depth_bin: int
resolution_bin: string
n_regions: bigint
n_attempt_total: bigint
n_valid_all_total: bigint
n_valid_clean_total: bigint
n_recovered_all_total: bigint
n_recovered_clean_total: bigint
completeness_valid_all_mean: double
completeness_valid_all_std: double
completeness_valid_clean_mean: double
completeness_valid_clean_std: double
arc_snr_mean: double
arc_snr_p50: double
completeness_valid_all_pooled: double
completeness_valid_clean_pooled: double
ci_low_pooled_all: double
ci_high_pooled_all: double
ci_low_pooled_clean: double
ci_high_pooled_clean: double
region_split: string
```

---

### 2. DATASET SIZE AND STRUCTURE

| Metric | Value |
|--------|-------|
| Completeness surfaces (detailed) | 92,312 rows |
| Region-aggregated surfaces | 26,952 rows |
| Unique regions | 782 |

**Split Distribution (from surfaces):**

| Split | Rows | n_attempt | n_valid_all | n_recovered_all |
|-------|------|-----------|-------------|-----------------|
| test | 23,381 | 1,842,392 | 1,842,392 | 189,304 |
| train | 44,520 | 1,380,816 | 1,380,816 | 149,551 |
| val | 24,411 | 2,104,626 | 2,104,626 | 189,113 |

---

### 3. OVERALL COMPLETENESS METRICS

**Global Counts:**
| Metric | Value |
|--------|-------|
| Total attempted | 5,327,834 |
| Total valid (all) | 5,327,834 |
| Total valid (clean) | 4,679,202 |
| Total recovered (all) | 527,968 |
| Total recovered (clean) | 489,694 |

**Global Completeness:**
| Metric | Value |
|--------|-------|
| completeness_valid_all (recovered/valid) | 0.0991 (9.91%) |
| completeness_valid_clean (recovered_clean/valid_clean) | 0.1047 (10.47%) |
| completeness_overall (recovered/attempted) | 0.0991 (9.91%) |
| valid_fraction_all | 1.0000 (100.00%) |
| valid_fraction_clean | 0.8783 (87.83%) |

---

### 4. COMPLETENESS BY THETA_E (Einstein Radius)

| theta_e (") | n_attempt | n_valid_all | n_recovered_all | comp_valid_all | comp_valid_clean | arc_snr_mean |
|-------------|-----------|-------------|-----------------|----------------|------------------|--------------|
| 0.30 | 1,782,906 | 1,782,906 | 0 | 0.0000 | 0.0000 | 51.51 |
| 0.60 | 1,764,898 | 1,764,898 | 0 | 0.0000 | 0.0000 | 45.33 |
| 1.00 | 1,780,030 | 1,780,030 | 527,968 | 0.2966 | 0.3132 | 43.85 |

---

### 5. COMPLETENESS BY RESOLUTION BIN (theta_e / psfsize_r)

| res_bin | n_attempt | n_valid_all | n_recovered_all | comp_valid_all | comp_valid_clean | theta/psf_mean | arc_snr |
|---------|-----------|-------------|-----------------|----------------|------------------|----------------|---------|
| <0.4 | 1,968,454 | 1,968,454 | 0 | 0.0000 | 0.0000 | 0.253 | 49.46 |
| 0.4-0.6 | 1,574,030 | 1,574,030 | 0 | 0.0000 | 0.0000 | 0.468 | 47.22 |
| 0.6-0.8 | 1,200,782 | 1,200,782 | 0 | 0.0000 | 0.0000 | 0.711 | 41.55 |
| 0.8-1.0 | 579,028 | 579,028 | 523,248 | 0.9037 | 0.9464 | 0.857 | 49.12 |
| >=1.0 | 5,540 | 5,540 | 4,720 | 0.8520 | 0.9233 | 1.050 | 33.85 |

---

### 6. COMPLETENESS BY SOURCE MAGNITUDE (src_dmag)

| src_dmag | n_attempt | n_valid_all | n_recovered_all | comp_valid_all | arc_snr_mean |
|----------|-----------|-------------|-----------------|----------------|--------------|
| 1.00 | 2,661,074 | 2,661,074 | 276,565 | 0.1039 | 66.54 |
| 2.00 | 2,666,760 | 2,666,760 | 251,403 | 0.0943 | 26.87 |

---

### 7. COMPLETENESS BY PSF SIZE BIN

| psf_bin (") | n_attempt | n_valid_all | n_recovered_all | comp_valid_all | arc_snr_mean |
|-------------|-----------|-------------|-----------------|----------------|--------------|
| 0.7 | 18 | 18 | 18 | 1.0000 | 112.72 |
| 0.8 | 888 | 888 | 261 | 0.2939 | 27.43 |
| 0.9 | 15,440 | 15,440 | 4,441 | 0.2876 | 36.59 |
| 1.0 | 126,356 | 126,356 | 37,508 | 0.2968 | 52.79 |
| 1.1 | 842,658 | 842,658 | 254,534 | 0.3021 | 55.94 |
| 1.2 | 1,600,826 | 1,600,826 | 231,206 | 0.1444 | 48.76 |
| 1.3 | 1,262,748 | 1,262,748 | 0 | 0.0000 | 47.04 |
| 1.4 | 910,078 | 910,078 | 0 | 0.0000 | 44.23 |
| 1.5 | 568,822 | 568,822 | 0 | 0.0000 | 39.12 |

---

### 8. COMPLETENESS BY DEPTH BIN

| depth_bin (mag) | n_attempt | n_valid_all | n_recovered_all | comp_valid_all | arc_snr_mean |
|-----------------|-----------|-------------|-----------------|----------------|--------------|
| 23.50 | 86,658 | 86,658 | 5,729 | 0.0661 | 15.63 |
| 23.75 | 365,418 | 365,418 | 55,522 | 0.1519 | 18.19 |
| 24.00 | 1,050,762 | 1,050,762 | 79,710 | 0.0759 | 21.17 |
| 24.25 | 1,068,418 | 1,068,418 | 83,052 | 0.0777 | 26.64 |
| 24.50 | 374,982 | 374,982 | 23,922 | 0.0638 | 33.61 |
| 24.75 | 573,284 | 573,284 | 55,751 | 0.0972 | 42.45 |
| 25.00 | 1,242,568 | 1,242,568 | 177,849 | 0.1431 | 51.34 |
| 25.25 | 238,872 | 238,872 | 23,566 | 0.0987 | 63.81 |
| 25.50 | 142,390 | 142,390 | 4,373 | 0.0307 | 83.88 |
| 25.75 | 121,108 | 121,108 | 10,291 | 0.0850 | 103.33 |
| 26.00 | 55,002 | 55,002 | 7,951 | 0.1446 | 136.31 |
| 26.25 | 6,534 | 6,534 | 234 | 0.0358 | 149.10 |
| 26.50 | 1,790 | 1,790 | 18 | 0.0101 | 203.89 |
| 26.75 | 48 | 48 | 0 | 0.0000 | 329.76 |

---

### 9. CROSS-TABULATION: THETA_E x RESOLUTION_BIN

| theta_e | res_bin | n_attempt | n_valid_all | n_recovered_all | completeness |
|---------|---------|-----------|-------------|-----------------|--------------|
| 0.30 | <0.4 | 1,782,906 | 1,782,906 | 0 | 0.0000 |
| 0.60 | 0.4-0.6 | 1,574,030 | 1,574,030 | 0 | 0.0000 |
| 0.60 | 0.6-0.8 | 5,320 | 5,320 | 0 | 0.0000 |
| 0.60 | <0.4 | 185,548 | 185,548 | 0 | 0.0000 |
| 1.00 | 0.6-0.8 | 1,195,462 | 1,195,462 | 0 | 0.0000 |
| 1.00 | 0.8-1.0 | 579,028 | 579,028 | 523,248 | 0.9037 |
| 1.00 | >=1.0 | 5,540 | 5,540 | 4,720 | 0.8520 |

---

### 10. REGION-LEVEL VARIANCE ANALYSIS

**Region Variance Summary:**
| Metric | Value |
|--------|-------|
| Mean completeness (across region means) | 0.1383 |
| Mean region-to-region std | 0.0174 |
| Max region-to-region std | 0.7071 |
| Avg regions per bin | 3.4 |

**Region Variance by Split:**
| Split | mean_comp | mean_std | region_bins |
|-------|-----------|----------|-------------|
| test | 0.1451 | 0.0172 | 23,381 |
| train | 0.1290 | 0.0177 | 44,520 |
| val | 0.1381 | 0.0172 | 24,411 |

---

### 11. CONFIDENCE INTERVAL ANALYSIS

**Wilson CI Statistics (completeness_valid_all):**
| Metric | Value |
|--------|-------|
| Average CI width | 0.3645 |
| Min CI width | 0.0020 |
| Max CI width | 0.8109 |
| Average CI low bound | 0.0670 |
| Average CI high bound | 0.4315 |

**CI Width by Theta_E:**
| theta_e | avg_ci_width | avg_n_valid |
|---------|--------------|-------------|
| 0.30 | 0.3578 | 59.9 |
| 0.60 | 0.3584 | 59.2 |
| 1.00 | 0.3761 | 54.4 |

---

### 12. DATA QUALITY IMPACT (CLEAN VS ALL)

**Clean Subset Statistics:**
| Metric | Value |
|--------|-------|
| Valid clean / Valid all | 4,679,202 / 5,327,834 = 0.8783 (87.83%) |
| Completeness (all) | 0.0991 |
| Completeness (clean) | 0.1047 |
| Difference (clean - all) | +0.0056 |

**Clean vs All by Resolution Bin:**
| res_bin | all | clean | diff |
|---------|-----|-------|------|
| 0.4-0.6 | 0.0000 | 0.0000 | +0.0000 |
| 0.6-0.8 | 0.0000 | 0.0000 | +0.0000 |
| 0.8-1.0 | 0.9037 | 0.9464 | +0.0427 |
| <0.4 | 0.0000 | 0.0000 | +0.0000 |
| >=1.0 | 0.8520 | 0.9233 | +0.0714 |

---

### 13. ARC SNR DISTRIBUTION ANALYSIS

**Arc SNR Statistics (bin-level means):**
| Statistic | Value |
|-----------|-------|
| Min | 0.00 |
| P25 | 17.17 |
| Median | 31.70 |
| P75 | 58.63 |
| P95 | 136.35 |
| Max | 1083.50 |
| Mean | 46.80 |

---

### 14. PSF PROVENANCE SUMMARY

**g-band:**
| Source | Count | Percentage |
|--------|-------|------------|
| manifest | 49,170 | 0.923% |
| map | 5,278,664 | 99.077% |

**r-band:**
| Source | Count | Percentage |
|--------|-------|------------|
| manifest | 3,956 | 0.074% |
| map | 5,323,878 | 99.926% |

**z-band:**
| Source | Count | Percentage |
|--------|-------|------------|
| manifest | 2,788 | 0.052% |
| map | 5,325,046 | 99.948% |

---

### 15. SELECTION SET ANALYSIS

| selection_set_id | n_attempt | n_valid_all | n_recovered_all | completeness |
|------------------|-----------|-------------|-----------------|--------------|
| strat_balanced_area_weighted_k75 | 589,252 | 589,252 | 58,764 | 0.0997 |
| strat_balanced_density_k75 | 214,398 | 214,398 | 18,424 | 0.0859 |
| strat_balanced_n_lrg_k75 | 589,296 | 589,296 | 58,730 | 0.0997 |
| strat_balanced_psf_weighted_k75 | 436,914 | 436,914 | 46,048 | 0.1054 |
| strat_custom_area_weighted_k75 | 589,324 | 589,324 | 58,740 | 0.0997 |
| strat_custom_density_k75 | 211,274 | 211,274 | 17,697 | 0.0838 |
| strat_custom_n_lrg_k75 | 589,286 | 589,286 | 58,776 | 0.0997 |
| strat_custom_psf_weighted_k75 | 436,712 | 436,712 | 45,543 | 0.1043 |
| topk_area_weighted_k100 | 589,310 | 589,310 | 58,779 | 0.0997 |
| topk_density_k100 | 53,190 | 53,190 | 1,907 | 0.0359 |
| topk_n_lrg_k100 | 589,148 | 589,148 | 58,797 | 0.0998 |
| topk_psf_weighted_k100 | 439,730 | 439,730 | 45,763 | 0.1041 |

---

### 16. BINS WITH EXTREME VALUES

**Top 10 Lowest Completeness Bins (n_valid >= 100):**
| theta_e | res_bin | psf_bin | completeness | n_valid |
|---------|---------|---------|--------------|---------|
| 0.60 | 0.4-0.6 | 12 | 0.0000 | 130 |
| 0.30 | <0.4 | 12 | 0.0000 | 132 |
| 0.30 | <0.4 | 11 | 0.0000 | 642 |
| 0.30 | <0.4 | 11 | 0.0000 | 126 |
| 0.30 | <0.4 | 12 | 0.0000 | 152 |
| 0.30 | <0.4 | 12 | 0.0000 | 280 |
| 1.00 | 0.6-0.8 | 14 | 0.0000 | 206 |
| 0.30 | <0.4 | 12 | 0.0000 | 1432 |
| 0.30 | <0.4 | 11 | 0.0000 | 614 |
| 0.30 | <0.4 | 11 | 0.0000 | 136 |

**Top 10 Highest Completeness Bins (n_valid >= 100):**
| theta_e | res_bin | psf_bin | completeness | n_valid |
|---------|---------|---------|--------------|---------|
| 1.00 | 0.8-1.0 | 12 | 1.0000 | 126 |
| 1.00 | 0.8-1.0 | 12 | 1.0000 | 130 |
| 1.00 | 0.8-1.0 | 11 | 1.0000 | 104 |
| 1.00 | 0.8-1.0 | 12 | 1.0000 | 142 |
| 1.00 | 0.8-1.0 | 11 | 1.0000 | 106 |
| 1.00 | 0.8-1.0 | 12 | 1.0000 | 130 |
| 1.00 | 0.8-1.0 | 12 | 1.0000 | 130 |
| 1.00 | 0.8-1.0 | 12 | 1.0000 | 126 |
| 1.00 | 0.8-1.0 | 12 | 1.0000 | 202 |
| 1.00 | 0.8-1.0 | 12 | 1.0000 | 130 |

**Top 10 Widest Confidence Interval Bins:**
| theta_e | res_bin | ci_width | n_valid |
|---------|---------|----------|---------|
| 1.00 | 0.8-1.0 | 0.8109 | 2 |
| 1.00 | 0.8-1.0 | 0.8109 | 2 |
| 1.00 | 0.8-1.0 | 0.8109 | 2 |
| 1.00 | 0.8-1.0 | 0.8109 | 2 |
| 1.00 | 0.8-1.0 | 0.8109 | 2 |
| 1.00 | >=1.0 | 0.8109 | 2 |
| 1.00 | 0.8-1.0 | 0.8109 | 2 |
| 1.00 | 0.8-1.0 | 0.8109 | 2 |
| 1.00 | 0.8-1.0 | 0.8109 | 2 |
| 1.00 | 0.8-1.0 | 0.8109 | 2 |

---

### 17. SUMMARY STATISTICS

| Metric | Value |
|--------|-------|
| Total injections analyzed | 5,327,834 |
| Valid injections (all) | 5,327,834 (100.0%) |
| Valid injections (clean) | 4,679,202 (87.8%) |
| Recovered (all) | 527,968 |
| Recovered (clean) | 489,694 |
| Overall completeness (valid_all) | 0.0991 (9.91%) |
| Overall completeness (valid_clean) | 0.1047 (10.47%) |
| Mean CI width | 0.3645 |
| Unique regions | 782 |
| Mean region-to-region variance | 0.0174 |

---

## Code Used for Analysis

The analysis was performed using `spark_analyze_phase4d.py` which reads the Phase 4d output tables and computes comprehensive statistics. The script:

1. Reads `completeness_surfaces` (detailed, with region_id)
2. Reads `completeness_surfaces_region_agg` (mean/std across regions)
3. Reads `psf_provenance` (per-band PSF source counts)
4. Computes aggregations across multiple dimensions
5. Outputs structured report

---

## Questions for Review

### Recovery Threshold Questions

1. **The resolution threshold (theta/PSF >= 0.8) creates a near-binary selection function where completeness jumps from 0% to ~90%.** Is this threshold appropriate for the science goals, or should alternative thresholds be considered?

2. **Only theta_E = 1.0" injections are recoverable under current criteria.** The 0.3" and 0.6" injections all fall below the theta/PSF >= 0.8 threshold. What are the implications for training a lens finder that aims to detect smaller Einstein radii?

3. **The SNR threshold (>=5) appears to be easily met by most injections.** Arc SNR mean is 46.8 across bins, with P25 = 17.17. Is the SNR threshold contributing to the selection function, or is it dominated by the resolution criterion?

### Data Quality Questions

4. **87.8% of injections pass the clean subset quality cuts.** The clean subset shows +0.6pp higher completeness than the "all" subset. Is this quality cut appropriate for Phase 5 training?

5. **In the 0.8-1.0 resolution bin, clean completeness is 94.6% vs 90.4% for all.** In >=1.0 bin, it's 92.3% vs 85.2%. What explains this 4-7pp gap?

### PSF and Observing Condition Questions

6. **PSF provenance shows 99%+ from maps, 0.05-0.9% from manifest.** No fallback to r-band occurred. Is this expected given the bricks processed?

7. **Completeness varies with PSF size: 30% at 1.1", 14% at 1.2", 0% at 1.3"+.** This is because theta_E=1.0" / PSF=1.3" < 0.8. Is this consistent with DECaLS seeing distribution expectations?

8. **Depth shows non-monotonic completeness: peaks at 23.75 mag (15.2%) and 26.0 mag (14.5%), dips at 24.5 mag (6.4%) and 25.5 mag (3.1%).** What might cause this pattern?

### Statistical Questions

9. **Average confidence interval width is 0.3645.** The widest CIs occur in bins with n_valid=2. Should there be a minimum sample size requirement for reporting completeness?

10. **Mean region-to-region std is 0.0174 (1.7%).** Is this level of cosmic/sample variance acceptable for the selection function?

11. **Split distribution is train=26%, val=40%, test=34%.** This is unusual (typically train is largest). What are the implications for model training?

### Selection Set Questions

12. **`topk_density_k100` has 3.6% completeness vs ~10% for others.** This selection set has only 53,190 injections vs 400k-600k for others. Why is this selection set an outlier?

### Phase 5 Training Implications

13. **The training set will be dominated by theta_E=1.0", resolution 0.8-1.0 injections.** How should Phase 5 handle the absence of recoverable smaller theta_E lenses?

14. **Should Phase 5 train on the "all" subset or the "clean" subset?** What are the tradeoffs?

15. **The recovered sample (527,968) is 9.9% of injections.** Is this sample size sufficient for training a robust lens finder?

### Phase 6 Science Implications

16. **The selection function is sharp (binary at theta/PSF=0.8).** How should this be characterized in a publication?

17. **Region-to-region variance of 1.7% is modest.** Is this sufficient for making population-level lens statistics claims?

18. **Are there any patterns in the data that suggest systematic biases that would affect scientific conclusions?**

### Publication Readiness

19. **Are the Wilson confidence intervals correctly computed and appropriate for a MNRAS/ApJ/AAS methods paper?**

20. **What additional metrics or analyses would strengthen the completeness characterization for publication?**

---

## S3 Locations

| Output | S3 Path |
|--------|---------|
| Completeness surfaces | s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/completeness_surfaces/train_stamp64_bandsgrz_gridgrid_small |
| Region-aggregated | s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/completeness_surfaces_region_agg/train_stamp64_bandsgrz_gridgrid_small |
| PSF provenance | s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/psf_provenance/train_stamp64_bandsgrz_gridgrid_small |
| Analysis report | s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/analysis/train_stamp64_bandsgrz_gridgrid_small |
| Stage config | s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/_stage_config_train_stamp64_bandsgrz_gridgrid_small.json |


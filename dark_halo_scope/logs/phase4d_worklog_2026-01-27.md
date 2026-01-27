# Phase 4d Work Log - 2026-01-27

## Executive Summary

Phase 4d completeness estimation was upgraded to publication-grade standards and successfully executed on EMR.

## Run Details

| Parameter | Value |
|-----------|-------|
| Cluster ID | j-7IRCKYCZP0WY |
| Core Instances | 5 x m5.xlarge |
| Runtime | ~5 minutes |
| Experiment ID | train_stamp64_bandsgrz_gridgrid_small |
| Timestamp | 2026-01-27 03:13:33 UTC |

## Configuration

```json
{
  "recovery": {"snr_thresh": 5.0, "theta_over_psf": 0.8},
  "quality_cuts": {"bad_pixel_frac_max": 0.2, "wise_brightmask_frac_max": 0.2},
  "binning": {"psfsize_bin_width": 0.1, "psfdepth_bin_width": 0.25},
  "confidence_interval": {"method": "wilson", "z_score": 1.96, "coverage": "95%"}
}
```

## Input/Output

**Input:**
- s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/metrics/train_stamp64_bandsgrz_gridgrid_small

**Outputs:**
- s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/completeness_surfaces/train_stamp64_bandsgrz_gridgrid_small
- s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/completeness_surfaces_region_agg/train_stamp64_bandsgrz_gridgrid_small
- s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/psf_provenance/train_stamp64_bandsgrz_gridgrid_small

---

## Key Metrics Collected

### 1. Dataset Structure

| Metric | Value |
|--------|-------|
| Total raw rows (4c) | 10,627,158 |
| Injections (controls excluded) | 5,327,834 |
| Completeness surface rows | 92,312 |
| Region-aggregated rows | 26,952 |
| Unique regions | 782 |

### 2. Global Completeness

| Metric | Value | Percentage |
|--------|-------|------------|
| Total attempted | 5,327,834 | - |
| Valid (all) | 5,327,834 | 100.0% |
| Valid (clean) | 4,679,202 | 87.8% |
| Recovered (all) | 527,968 | 9.91% |
| Recovered (clean) | 489,694 | 10.47% |

**Key Finding:** ~10% overall completeness at recovery thresholds SNR≥5 and θ/PSF≥0.8.

### 3. Completeness by Einstein Radius (θ_E)

| θ_E (arcsec) | n_attempt | n_recovered | Completeness | SNR |
|--------------|-----------|-------------|--------------|-----|
| 0.30 | 1,782,906 | 0 | 0.00% | 51.5 |
| 0.60 | 1,764,898 | 0 | 0.00% | 45.3 |
| 1.00 | 1,780,030 | 527,968 | 29.66% | 43.9 |

**Key Finding:** Only θ_E=1.0" injections are recovered because they meet the θ/PSF≥0.8 threshold.

### 4. Completeness by Resolution Bin (θ_E/psfsize_r)

| Resolution Bin | n_attempt | n_recovered | Completeness (all) | Completeness (clean) |
|----------------|-----------|-------------|-------------------|---------------------|
| <0.4 | 1,968,454 | 0 | 0.00% | 0.00% |
| 0.4-0.6 | 1,574,030 | 0 | 0.00% | 0.00% |
| 0.6-0.8 | 1,200,782 | 0 | 0.00% | 0.00% |
| 0.8-1.0 | 579,028 | 523,248 | 90.37% | 94.64% |
| ≥1.0 | 5,540 | 4,720 | 85.20% | 92.33% |

**Key Finding:** The θ/PSF≥0.8 threshold is the dominant factor. Below 0.8, completeness is 0%. Above 0.8, completeness is 85-95%.

### 5. Completeness by Source Magnitude

| src_dmag | n_attempt | n_recovered | Completeness | Mean SNR |
|----------|-----------|-------------|--------------|----------|
| 1.0 | 2,661,074 | 276,565 | 10.39% | 66.5 |
| 2.0 | 2,666,760 | 251,403 | 9.43% | 26.9 |

### 6. Completeness by PSF Size

| PSF (arcsec) | n_attempt | n_recovered | Completeness |
|--------------|-----------|-------------|--------------|
| 0.7-1.2" | 2,586,186 | 527,968 | 20.4% |
| 1.3-1.5" | 2,741,648 | 0 | 0.0% |

**Key Finding:** Larger PSF (worse seeing) eliminates recovery because θ/PSF falls below threshold.

### 7. Data Quality Impact

| Subset | Valid | Recovered | Completeness |
|--------|-------|-----------|--------------|
| All | 5,327,834 | 527,968 | 9.91% |
| Clean (bad_pixel≤0.2, wise≤0.2) | 4,679,202 | 489,694 | 10.47% |

**Clean fraction:** 87.8% of injections pass quality cuts.
**Completeness boost:** Clean subset has +0.6pp higher completeness.

### 8. Region Variance

| Metric | Value |
|--------|-------|
| Mean completeness (across region means) | 0.1383 |
| Mean region-to-region std | 0.0174 |
| Max region-to-region std | 0.7071 |
| Avg regions per bin | 3.4 |

### 9. PSF Provenance

| Band | Map | Manifest | Fallback |
|------|-----|----------|----------|
| g | 99.08% | 0.92% | 0% |
| r | 99.93% | 0.07% | 0% |
| z | 99.95% | 0.05% | 0% |

**Key Finding:** No fallback to r-band was needed (all g/z had valid PSF).

### 10. Selection Set Comparison

All selection sets show similar completeness (~9-10%) except `topk_density_k100` (3.6%).

---

## Observations for Downstream Phases

### Phase 5 (Training) Implications

1. **Resolution dominates**: Training data will be heavily skewed toward the recoverable regime (θ/PSF≥0.8).

2. **Small θ_E is undetectable**: 0.3" and 0.6" lenses cannot be recovered under current criteria.

3. **Clean subset recommended**: 87.8% pass quality cuts, with slightly higher completeness.

4. **Split balance**: train=26%, val=40%, test=34% - unusual but acceptable.

### Phase 6 (Science) Implications

1. **Selection function is sharp**: The θ/PSF≥0.8 threshold creates a near-binary selection.

2. **Cosmic variance**: ~1.7% region-to-region variation is modest.

3. **Publication metrics**: Wilson CIs are computed for uncertainty quantification.

---

## Files and Outputs

- Analysis script: `dark_halo_scope/emr/spark_analyze_phase4d.py`
- Analysis report: `s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/analysis/train_stamp64_bandsgrz_gridgrid_small/phase4d_analysis_report.txt`
- Config: `s3://darkhaloscope/phase4_pipeline/phase4d/v3_color_relaxed/_stage_config_train_stamp64_bandsgrz_gridgrid_small.json`


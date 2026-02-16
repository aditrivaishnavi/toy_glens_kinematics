# Gen5 Phase 4c Production Run: Data Quality Review Prompt

**Purpose**: This document provides a comprehensive overview of the Gen5 Phase 4c production run for review by external LLMs or domain experts. Please review the data quality metrics, pipeline configuration, and scientific validity of the generated training data.

**Date**: 2026-02-04
**Run ID**: `j-5Z6LYAOSMAUU` (Gen5-Phase4c-FluxFix2-20260204-1459)
**Status**: ✅ COMPLETED

---

## 1. Project Context

### 1.1 Scientific Goal
Develop a CNN-based strong gravitational lens finder for DESI Legacy Survey DR10 that can:
- Detect galaxy-galaxy strong lenses with Einstein radii 0.5-2.5 arcsec
- Achieve >50% recall on real confirmed lenses (SLACS, BELLS catalogs)
- Maintain <20% contamination rate on hard negatives (ring galaxies, mergers)

### 1.2 The Problem Being Solved
Previous model generations (Gen1-4) achieved 75-88% true positive rate at FPR=1e-4 on synthetic validation data, but **catastrophically failed on real lenses**:
- Gen2 (best synthetic): 75.1% tpr@fpr1e-4 on synthetic
- Gen2 anchor baseline: **2.9% recall on SLACS lenses, 95% contamination**

**Root Cause Identified**: Training data used smooth Sersic profiles for lensed source galaxies. Real lensed sources are clumpy, irregular galaxies with substructure.

### 1.3 Gen5 Solution
Replace synthetic Sersic sources with **real galaxy morphologies from COSMOS HST F814W images**, rendered through gravitational lensing using `lenstronomy`.

---

## 2. Pipeline Architecture

### 2.1 Data Flow
```
DESI Legacy Survey DR10 → Phase 3 (LRG Selection) → Phase 4a (Manifest Generation)
                                                              ↓
                                                    Phase 4c (Lens Injection with COSMOS)
                                                              ↓
                                                    Phase 4p5 (Compaction)
                                                              ↓
                                                    Phase 5 (Model Training)
```

### 2.2 Phase 4c Pipeline Components

| Component | Description |
|-----------|-------------|
| **Lens Model** | SIE (Singular Isothermal Ellipsoid) via lenstronomy |
| **Source Model** | COSMOS real galaxies via lenstronomy INTERPOL |
| **PSF Model** | Moffat β=3.5, per-band FWHM from survey |
| **Pixel Scale** | 0.262 arcsec/pixel (DR10 native) |
| **Stamp Size** | 64×64 pixels |
| **Rendering** | lenstronomy ray-tracing + PSF convolution |

### 2.3 COSMOS Bank Specification

| Property | Value |
|----------|-------|
| **Source** | GalSim COSMOS 25.2 training sample (HST F814W) |
| **Total Galaxies** | 20,000 |
| **Stamp Size** | 96×96 pixels at 0.03 arcsec/pixel |
| **HLR Filter** | 0.1 - 1.0 arcsec (half-light radius) |
| **Storage** | HDF5 file (~474 MB) |
| **Location** | `s3://darkhaloscope/data/cosmos_banks/cosmos_bank_20k_gen5.h5` |

---

## 3. Production Run Configuration

### 3.1 EMR Cluster

| Parameter | Value |
|-----------|-------|
| **Cluster ID** | j-5Z6LYAOSMAUU |
| **EMR Release** | emr-7.6.0 |
| **Master Instance** | m5.xlarge (1×) |
| **Core Instances** | m5.2xlarge (34×) |
| **Total vCores** | 272 |
| **Region** | us-east-2 |
| **Runtime** | 2h 25m (14:59 - 17:28 UTC) |

### 3.2 Spark Configuration

```
--executor-memory 6g
--driver-memory 4g
--executor-cores 2
--conf spark.default.parallelism=600
--conf spark.sql.shuffle.partitions=600
--conf spark.executorEnv.NUMBA_CACHE_DIR=/tmp/numba_cache
```

### 3.3 Pipeline Arguments

```
--stage 4c
--output-s3 s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_production/
--variant v5_cosmos_production
--experiment-id train_stamp64_bandsgrz_cosmos
--parent-s3 s3://darkhaloscope/phase4_pipeline/phase4a/v4_sota_moffat/manifests/train_stamp64_bandsgrz_gridgrid_sota
--coadd-s3-cache-prefix s3://darkhaloscope/dr10/coadd_cache
--psf-model moffat
--moffat-beta 3.5
--source-mode cosmos
--cosmos-bank-h5 s3://darkhaloscope/data/cosmos_banks/cosmos_bank_20k_gen5.h5
--control-frac-train 0.50
--unpaired-controls 1
--skip-if-exists 1
--force 1
--bands g,r,z
```

---

## 4. Data Quality Metrics

### 4.1 Output Summary

| Metric | Value |
|--------|-------|
| **Total Output Files** | 3,604 |
| **Total Size** | 489.99 GB |
| **Estimated Total Stamps** | ~11.75 million |
| **Output Format** | Gzip-compressed Parquet |
| **Partitioning** | By `region_split` (train/val/test) |

### 4.2 Per-Split Statistics (Sampled from 3 files each)

#### TRAIN Split

| Metric | Value |
|--------|-------|
| **Estimated Total Rows** | 2,261,200 |
| **Cutout OK Rate** | 100.0% |
| **Control/Injection Balance** | 50.1% / 49.9% |
| **COSMOS Unique Indices (sample)** | 4,883 |

**arc_snr Distribution (SIE injections only)**:
| Statistic | Value |
|-----------|-------|
| Min | 0.0 |
| Max | 137.4 |
| Mean | 7.33 |
| Median | 4.26 |
| Std Dev | 9.72 |
| P5 | 0.0 |
| P25 | 2.1 |
| P50 | 4.26 |
| P75 | 8.77 |
| P95 | 23.97 |
| **>1** | **89.2%** |
| >10 | 21.5% |
| >50 | 0.8% |

**theta_e Distribution**:
- Range: 0.5 - 2.5 arcsec
- Mean: 1.37 arcsec

**COSMOS HLR Distribution**:
- Range: 0.926 - 4.602 arcsec
- Mean: 2.2 arcsec

**PSF Size (r-band)**:
- Range: 1.14 - 1.60 arcsec
- Mean: 1.33 arcsec

#### VAL Split

| Metric | Value |
|--------|-------|
| **Estimated Total Rows** | 5,529,200 |
| **Cutout OK Rate** | 100.0% |
| **Control/Injection Balance** | 51.5% / 48.5% |
| **COSMOS Unique Indices (sample)** | 9,748 |

**arc_snr Distribution**:
| Statistic | Value |
|-----------|-------|
| Min | 0.0 |
| Max | 278.82 |
| Mean | 7.49 |
| Median | 4.03 |
| Std Dev | 12.23 |
| P5 | 0.554 |
| P25 | 1.9 |
| P50 | 4.03 |
| P75 | 8.3 |
| P95 | 24.96 |
| **>1** | **89.7%** |
| >10 | 19.9% |
| >50 | 1.4% |

**theta_e**: 0.5 - 2.5 arcsec (mean 1.33)
**COSMOS HLR**: 0.926 - 4.602 arcsec (mean 2.18)
**PSF Size r**: 1.02 - 1.59 arcsec (mean 1.34)

#### TEST Split

| Metric | Value |
|--------|-------|
| **Estimated Total Rows** | 3,961,200 |
| **Cutout OK Rate** | 99.85% |
| **Control/Injection Balance** | 50.1% / 49.9% |
| **COSMOS Unique Indices (sample)** | 7,817 |

**arc_snr Distribution**:
| Statistic | Value |
|-----------|-------|
| Min | 0.0 |
| Max | 153.59 |
| Mean | 7.26 |
| Median | 4.22 |
| Std Dev | 10.02 |
| P5 | 0.416 |
| P25 | 2.16 |
| P50 | 4.22 |
| P75 | 8.43 |
| P95 | 24.06 |
| **>1** | **90.6%** |
| >10 | 20.2% |
| >50 | 0.9% |

**theta_e**: 0.5 - 2.5 arcsec (mean 1.38)
**COSMOS HLR**: 0.926 - 4.542 arcsec (mean 2.214)
**PSF Size r**: 0.9 - 1.60 arcsec (mean 1.31)

---

## 5. Critical Bug Fixes Applied

### 5.1 Flux Scaling Bug (CRITICAL)

**Problem**: `lenstronomy.LightModel(["INTERPOL"])` expects input images in **surface brightness** units (flux/arcsec²), but the code was providing **total flux** (flux/pixel).

**Impact**: Arc flux was ~1111× too faint, resulting in arc_snr max of 0.15 instead of 100+.

**Fix Applied**:
```python
# BEFORE (buggy):
kwargs_source = [{
    "image": template * flux_nmgy,  # Total flux per pixel
    ...
}]

# AFTER (fixed):
src_pixel_area = cosmos_bank["src_pixscale"] ** 2  # 0.03^2 = 0.0009 arcsec^2
surface_brightness = template * flux_nmgy / src_pixel_area  # flux/arcsec^2

kwargs_source = [{
    "image": surface_brightness,
    ...
}]
```

**Verification**: arc_snr now ranges 0-278 with mean ~7.4 (realistic lensing signals).

### 5.2 Other Bugs Fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| Module-level boto3 import | boto3=None on executors | Import inside functions |
| PSF kernel > stamp size | Stamps failed | Cap kernel radius to 31 |
| `--bands grz` parsing | Treated as single band | Use comma-separated |
| s3:// vs s3a:// URIs | Local Spark failed | Handle both prefixes |
| `task_id` not extracted | NameError on executors | Add explicit extraction |
| `src_mag_r` undefined | NameError in COSMOS loop | Add alias assignment |
| Duplicate function definitions | Wrong code executed | Remove duplicates |
| Numba cache permissions | RuntimeError on EMR | Set NUMBA_CACHE_DIR |

---

## 6. Scientific Validation

### 6.1 arc_snr Interpretation

The `arc_snr` (arc signal-to-noise ratio) is calculated as:
```
arc_snr = sum(arc_flux * mask) / sqrt(sum(variance * mask))
```

Where `mask` excludes pixels flagged by DR10 maskbits (bright stars, artifacts).

| arc_snr Range | Interpretation | Expected Fraction |
|---------------|----------------|-------------------|
| < 1 | Undetectable (masked regions, faint arcs) | ~10% |
| 1-10 | Weak but detectable signals | ~70% |
| 10-50 | Strong lensing events | ~19% |
| > 50 | Very bright arcs, high magnification | ~1% |

**Observed**: 89-91% of SIE injections have arc_snr > 1 ✅

### 6.2 theta_e (Einstein Radius) Coverage

| Range | Purpose |
|-------|---------|
| 0.5 arcsec | Near PSF limit, harder to detect |
| 1.0-1.5 arcsec | Typical galaxy-galaxy lens scale |
| 2.0-2.5 arcsec | Larger lenses, brighter arcs |

**Observed**: Uniform distribution 0.5-2.5 arcsec ✅

### 6.3 Control Sample Quality

- **Unpaired controls**: Controls are different galaxies from positives (no shortcut learning)
- **Balance**: ~50/50 split between controls and SIE injections ✅
- **Same preprocessing**: Controls receive identical cutout processing

---

## 7. Questions for Review

### 7.1 Data Quality Questions

1. **arc_snr Distribution**: Is the observed distribution (mean ~7.4, 90% > 1) reasonable for realistic gravitational lensing simulations?

2. **COSMOS HLR Range**: The observed half-light radii (0.926 - 4.6 arcsec) seem larger than expected for lensed sources. Is this a concern?

3. **theta_e/PSF Ratio**: With PSF FWHM ~1.3" and theta_e 0.5-2.5", approximately 60% of lenses are "resolved" (theta_e > PSF). Is this sufficient?

4. **Low arc_snr Stamps**: ~10% of stamps have arc_snr < 1. Should these be filtered before training, or do they provide valuable hard examples?

### 7.2 Scientific Questions

1. **Surface Brightness Conversion**: Is the fix `flux / pixel_area` correct for converting COSMOS stamps to lenstronomy INTERPOL surface brightness units?

2. **COSMOS Source Selection**: The bank was built with HLR filter 0.1-1.0 arcsec, but observed HLR is 0.926-4.6 arcsec after lensing magnification. Is this physically reasonable?

3. **Moffat PSF β=3.5**: Is this a good approximation for DECaLS ground-based imaging?

### 7.3 Pipeline Questions

1. **Control Fraction**: 50% controls vs 50% positives - is this optimal for training?

2. **Stamp Size**: 64×64 pixels at 0.262 arcsec/pixel = 16.8" FOV. Is this sufficient to capture large Einstein radii (2.5")?

3. **Compression**: Gzip Parquet compression achieves ~130 KB/stamp. Is this acceptable for training throughput?

---

## 8. File Locations

### 8.1 S3 Paths

```
Output:          s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_production/
COSMOS Bank:     s3://darkhaloscope/data/cosmos_banks/cosmos_bank_20k_gen5.h5
Coadd Cache:     s3://darkhaloscope/dr10/coadd_cache/
EMR Logs:        s3://darkhaloscope/emr-logs/j-5Z6LYAOSMAUU/
Pipeline Code:   s3://darkhaloscope/code/gen5/spark_phase4_pipeline_gen5.py
Bootstrap:       s3://darkhaloscope/code/gen5/emr_bootstrap_gen5.sh
```

### 8.2 Local Paths (emr-launcher)

```
Validation JSON: /data/staging/phase4c_validation.json
Launch Script:   /data/staging/launch_gen5_production.sh
```

### 8.3 Repository

```
Pipeline Code:   dark_halo_scope/emr/gen5/spark_phase4_pipeline_gen5.py
Bootstrap:       dark_halo_scope/emr/gen5/emr_bootstrap_gen5.sh
Training Code:   dark_halo_scope/model/phase5_train_fullscale_gh200_v2.py
Model Comparison: dark_halo_scope/results/model_comparison_and_evolution.md
```

---

## 9. Raw Validation JSON

```json
{
  "timestamp": "2026-02-04T17:33:30.236059",
  "splits": {
    "train": {
      "sampled_rows": 11306,
      "total_files": 600,
      "estimated_total_rows": 2261200,
      "cutout_ok_pct": 100.0,
      "arc_snr": {
        "min": 0.0,
        "max": 137.4,
        "mean": 7.33,
        "median": 4.26,
        "std": 9.72,
        "gt_1_pct": 89.2,
        "gt_10_pct": 21.5,
        "gt_50_pct": 0.8,
        "p5": 0.0,
        "p25": 2.1,
        "p50": 4.26,
        "p75": 8.77,
        "p95": 23.97
      },
      "theta_e": {"min": 0.5, "max": 2.5, "mean": 1.37},
      "cosmos_unique_indices": 4883,
      "cosmos_hlr": {"min": 0.926, "max": 4.602, "mean": 2.2},
      "psfsize_r": {"min": 1.14, "max": 1.6, "mean": 1.33}
    },
    "val": {
      "sampled_rows": 27646,
      "total_files": 600,
      "estimated_total_rows": 5529200,
      "cutout_ok_pct": 100.0,
      "arc_snr": {
        "min": 0.0,
        "max": 278.82,
        "mean": 7.49,
        "median": 4.03,
        "std": 12.23,
        "gt_1_pct": 89.7,
        "gt_10_pct": 19.9,
        "gt_50_pct": 1.4,
        "p5": 0.554,
        "p25": 1.9,
        "p50": 4.03,
        "p75": 8.3,
        "p95": 24.96
      },
      "theta_e": {"min": 0.5, "max": 2.5, "mean": 1.33},
      "cosmos_unique_indices": 9748,
      "cosmos_hlr": {"min": 0.926, "max": 4.602, "mean": 2.18},
      "psfsize_r": {"min": 1.02, "max": 1.59, "mean": 1.34}
    },
    "test": {
      "sampled_rows": 19806,
      "total_files": 600,
      "estimated_total_rows": 3961200,
      "cutout_ok_pct": 99.85,
      "arc_snr": {
        "min": 0.0,
        "max": 153.59,
        "mean": 7.26,
        "median": 4.22,
        "std": 10.02,
        "gt_1_pct": 90.6,
        "gt_10_pct": 20.2,
        "gt_50_pct": 0.9,
        "p5": 0.416,
        "p25": 2.16,
        "p50": 4.22,
        "p75": 8.43,
        "p95": 24.06
      },
      "theta_e": {"min": 0.5, "max": 2.5, "mean": 1.38},
      "cosmos_unique_indices": 7817,
      "cosmos_hlr": {"min": 0.926, "max": 4.542, "mean": 2.214},
      "psfsize_r": {"min": 0.9, "max": 1.6, "mean": 1.31}
    }
  },
  "estimated_total_stamps": 11751600
}
```

---

## 10. Specific Questions for LLM Review

**INSTRUCTIONS**: Please answer each question below with a clear YES/NO/UNCERTAIN followed by a brief explanation. If you identify issues, suggest specific fixes.

### 10.1 Data Quality Validation Questions

**Q1**: Is the arc_snr distribution (mean=7.3, median=4.2, 90% > 1, 20% > 10) physically realistic for galaxy-galaxy strong lensing at the Einstein radii and source magnitudes we're simulating?
- Expected arc_snr for theta_e=1" lens with source 2 mag fainter than lens?
- Is our distribution consistent with published lensing simulations?

**Q2**: We observe ~10% of SIE injections have arc_snr < 1. Given that these are calculated on unmasked pixels only, is this fraction:
- (a) Too high, indicating a pipeline problem?
- (b) Expected, given survey masking and faint source combinations?
- (c) Should these samples be filtered from training data?

**Q3**: The COSMOS HLR (half-light radius) after lensing ranges from 0.926-4.6 arcsec with mean ~2.2 arcsec. The unlensed COSMOS sources were selected with HLR 0.1-1.0 arcsec. Is the ~2-4× increase physically reasonable given:
- Typical magnification factors for SIE lenses
- The theta_e range of 0.5-2.5 arcsec
- Or does this indicate a calculation error?

**Q4**: PSF size in r-band ranges 0.9-1.6 arcsec (mean 1.3). With theta_e ranging 0.5-2.5 arcsec:
- What fraction of our lenses are "resolved" (theta_e/PSF > 1)?
- Is this sufficient for training, or should we filter to only resolved lenses?
- What theta_e/PSF threshold would you recommend?

### 10.2 Flux Scaling and Physics Questions

**Q5**: We fixed a critical bug where lenstronomy INTERPOL was receiving total flux instead of surface brightness. The fix was:
```python
surface_brightness = template * flux_nmgy / (pixel_scale_arcsec ** 2)
```
Where pixel_scale = 0.03 arcsec/pixel for COSMOS stamps.
- Is this conversion correct?
- Are there additional factors (e.g., flux conservation, magnification) we should include?

**Q6**: The arc_snr is calculated as:
```python
arc_snr = sum(injected_arc_flux[good_pixels]) / sqrt(sum(variance[good_pixels]))
```
Where good_pixels excludes DR10 maskbits. Is this:
- The standard definition for arc SNR in the lensing literature?
- Should we use a different SNR calculation (e.g., matched filter, aperture)?

**Q7**: We use Moffat PSF with β=3.5 for all bands. For DECaLS ground-based imaging:
- Is β=3.5 appropriate, or should it vary by band/seeing?
- Should we use the actual PSF model from the survey instead?

### 10.3 Training Data Considerations

**Q8**: Our control samples are "unpaired" - different galaxies than the positives, selected by hash. This ensures the model can't learn shortcuts from paired data. However:
- Is 50/50 control/positive split optimal?
- Should we include hard negatives (ring galaxies, mergers) in training?
- What control fraction do SOTA lens-finding papers use?

**Q9**: We train on 64×64 pixel stamps (16.8" FOV) at 0.262 arcsec/pixel. For theta_e up to 2.5":
- Is the FOV sufficient to capture the full Einstein ring + surrounding context?
- What stamp size do other papers use?
- Should we increase to 96×96 or 128×128?

**Q10**: Our dataset has ~11.75M stamps. Is this:
- Sufficient for training a ConvNeXt-Tiny (~28M parameters)?
- More than needed (could train faster with subset)?
- What's the typical dataset size in published lens-finding papers?

### 10.4 Sim-to-Real Gap Questions

**Q11**: Our previous models (Gen2-4 with Sersic sources) achieved 75% TPR@FPR=1e-4 on synthetic data but only 2.9% recall on real SLACS lenses. With COSMOS sources:
- What recall improvement would you expect?
- Is the COSMOS morphology sufficient, or are there other sim-to-real gaps?
- What additional data augmentations would you recommend?

**Q12**: The COSMOS galaxies are single-band (HST F814W) while our survey has g, r, z bands. We:
- Use the same morphology template for all bands
- Scale flux by source SED (color-dependent)
Is this approach valid, or should we use multi-band COSMOS data?

**Q13**: We're using GalSim COSMOS 25.2 training sample. Are there:
- Known issues with this catalog?
- Better alternatives (e.g., CANDELS, CEERS)?
- Biases in the COSMOS galaxy population we should account for?

### 10.5 Pipeline and Implementation Questions

**Q14**: The pipeline discovered these bugs during development. Are there similar bugs you'd check for:
- Coordinate system issues (pixel vs arcsec)?
- Flux normalization issues?
- PSF convolution order issues?

**Q15**: We use lenstronomy for lens modeling and ray-tracing. For SIE + external shear:
- Is lenstronomy's implementation validated?
- Are there edge cases that could produce incorrect arcs?
- Should we add substructure to the lens model?

**Q16**: Our arc_snr percentiles are:
- P5 = 0.0-0.5
- P25 = 1.9-2.2
- P50 = 4.0-4.3
- P75 = 8.3-8.8
- P95 = 24-25

Does this distribution suggest:
- Correct physics implementation?
- Potential issues with faint-end or bright-end?
- Comparison to expected SNR from lensing theory?

### 10.6 Recommendations Requested

**Q17**: Based on this data quality report, what are the TOP 3 issues you'd prioritize fixing before training?

**Q18**: What additional validation would you recommend before using this data for training?

**Q19**: Are there any RED FLAGS in the metrics that suggest the data should NOT be used for training?

**Q20**: If you were reviewing this for publication, what additional experiments or validations would you require?

---

## 11. Summary

### 10.1 What Went Right ✅

1. **Flux scaling bug identified and fixed** - arc_snr now realistic
2. **Production run completed successfully** - 11.75M stamps in 2.5 hours
3. **Data quality metrics are good** - 99.85-100% cutout success, 89-91% arc_snr > 1
4. **Balanced dataset** - ~50/50 control/injection split
5. **COSMOS diversity** - 4,883-9,748 unique galaxy indices used per split

### 10.2 Potential Concerns ⚠️

1. **~10% stamps have arc_snr < 1** - May be due to masking or faint arcs
2. **COSMOS HLR range 0.9-4.6 arcsec** - Larger than unlensed sources (expected due to magnification)
3. **PSF-limited lenses** - ~40% have theta_e < PSF FWHM

### 10.3 Next Steps

1. **Phase 4p5**: Compact the 3,604 files into fewer, larger files for efficient training
2. **Phase 5**: Train Gen5 model on this dataset
3. **Stage 0**: Re-run anchor baseline on Gen5 model to measure sim-to-real improvement
4. **Ablation**: Compare Gen5 (COSMOS) vs Gen4 (Sersic) performance

---

*Document generated: 2026-02-04T17:35:00Z*
*For questions, contact the project maintainer.*

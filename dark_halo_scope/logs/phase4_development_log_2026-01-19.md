# Phase 4 Pipeline Development Log

**Date**: January 19, 2026  
**Author**: Aditrivaishnavi Balaji  
**Project**: Dark Halo Scope - Phase 4 Selection Function & Recovery Framework

---

## Executive Summary

Phase 4 builds on top of the frozen Phase 3 LRG parent catalog to construct a selection-function and recovery framework. The goal is to measure how well we can detect gravitational lenses as a function of observing conditions (PSF, depth, extinction) and lens parameters (Einstein radius, flux ratio, etc.).

I am treating the Phase 3 catalog as **frozen** - no changes to splits or selection logic will be made. The Phase 3 validation confirmed:
- Schema completeness ✓
- Variant hierarchy consistency ✓  
- Split representativeness ✓
- Negligible boundary artifacts (12 out of 19.7M) ✓

**Known Limitation**: The region splits are imbalanced in size due to mega-regions hashing to `test` (81% test, 11% train, 7% val). This is not scientifically invalid but requires careful leakage controls.

---

## Phase 4 Goal

Build a selection-function and recovery framework on top of the Phase 3 LRG parent catalog:

1. **Generate Realistic Examples**: Create lens/non-lens examples via injection using real imaging conditions (PSF, depth, extinction, sky noise)

2. **Measure Recovery Probability**: Quantify recovery as a function of:
   - Observing conditions (PSF size, depth, EBV, galaxy density)
   - Lens parameters (Einstein radius θ_E, flux ratio, source morphology)

3. **Produce "Seeing Wall" Results**: Determine where completeness collapses as PSF worsens and how region ranking strategies interact with that

---

## Inputs from Phase 3

### From `parent_compact` (Phase 3.5)

| Column | Purpose in Phase 4 |
|--------|-------------------|
| `region_split` | Leakage-safe holdouts (train/val/test) |
| `region_id` | Region-level stratification and analysis |
| `brickname` | Map objects back to imaging products |
| `ra`, `dec` | Cutout center coordinates |
| `zmag`, `rmag`, `w1mag` | Photometry for sampling and analysis |
| `rz`, `zw1` | Colors for stratification |
| `gmag_mw`, `rmag_mw`, `zmag_mw`, `w1mag_mw` | MW-corrected magnitudes |
| `type` | Morphological type (DEV/REX/SER/EXP) |
| `is_v1..v5` | LRG variant flags |
| `gmag_valid` | Flag for valid g-band photometry |

### From Phase 3a/3b

| Output | Purpose in Phase 4 |
|--------|-------------------|
| `region_metrics` | PSF/depth/EBV statistics per region for stratification |
| `region_selections` | Which selection_set_id and ranking mode chose which regions |
| `bricks_with_region` | Brick-level metadata for imaging acquisition |

### Phase 3 Catalog Statistics

| Metric | Value |
|--------|-------|
| Total LRG Objects | 19,687,747 |
| Unique Regions | 811 |
| Unique Bricks | 256,208 |
| Split: train | 2,208,075 (11.2%) |
| Split: val | 1,454,614 (7.4%) |
| Split: test | 16,025,058 (81.4%) |

---

## What Phase 4 Needs That Phase 3 Did Not Provide

Phase 3 operates in **catalog space**. Phase 4 requires **pixel space**.

### Image Data Requirements

| Data Product | Source | Purpose |
|--------------|--------|---------|
| DECaLS/LS DR10 image cutouts | Legacy Survey servers | g/r/z band images |
| PSF models | Per-brick PSF files | Realistic injection convolution |
| Noise/variance maps | Per-exposure variance | Realistic noise injection |
| Image masks | Maskbits | Avoid contaminated regions |

### New Infrastructure Needed

1. **Cutout Acquisition Pipeline**: Download and cache image tiles
2. **Cutout Caching Strategy**: Separate from sweep FITS caching
3. **Injection Framework**: Insert synthetic lenses with realistic PSF convolution
4. **Recovery Pipeline**: Detect and characterize injected lenses

---

## Compute Strategy Decision

### Option A: Non-EMR (Recommended for First Pass)

Use a single strong EC2 instance or small cluster to:
- Sample 50k-200k objects from train/val/test
- Fetch cutouts from Legacy Survey
- Inject lenses and build datasets
- Run recovery pipeline

**Pros**:
- Simpler development and debugging
- Faster iteration on injection/recovery parameters
- Sufficient for rigorous, publishable results

**Cons**:
- Limited to ~200k objects
- Sequential cutout fetching

### Option B: EMR or AWS Batch (For Scale)

Needed if I want:
- Millions of cutouts
- Large injection parameter grids
- Multiple ranking modes and variants

**AWS Batch** may be better than EMR because the workload is "download cutout + run injection + write artifact" which is embarrassingly parallel and not Spark-native.

### My Decision

I will start with **Option A** for the first pass. This allows rapid iteration on the injection and recovery pipeline before committing to large-scale processing.

---

## Scientific Correctness Requirements

### 1. Holdout Discipline

| Rule | Implementation |
|------|----------------|
| Test is final evaluation only | Never use `test` split for tuning |
| Iterate within train | Use `train` for development |
| Validate on val | Use `val` for hyperparameter selection |
| Evaluate on test | One-time final evaluation |

### 2. Avoid Duplicate Leakage via Selection Sets

The same object can appear in multiple `selection_set_id` groups. For training/evaluation:
- Pick one selection set per experiment, OR
- Union them and deduplicate by `(brickname, objid)`

### 3. Injection Realism

Minimum bar to avoid "toy" results:

| Requirement | Implementation |
|-------------|----------------|
| Real PSF | Use per-brick PSF models or justified approximation |
| Realistic noise | Consistent with depth/variance from imaging |
| Documentation | Explicitly state what is and is not modeled |

### 4. Selection Function Reporting

My later claims must match the actual pipeline:
- Exact LRG cuts (variant: `v3_color_relaxed`)
- Exact region selection strategy (ranking modes used)
- Exact split strategy (region-level based on `xxhash64(region_id)`)

### 5. Limitation Documentation

- DR10 South only (no North coverage)
- Objects with NaN photometry characterized
- Morphological selection (no PSF/star objects)

---

## Recommended Phase 4 Structure

### Step 1: Define Evaluation Contract

```
Fixed:
- Split policy: region_split (train/val/test)
- Selection set(s): TBD (likely union of all ranking modes)
- Parent variant: v3_color_relaxed
- Metrics: Completeness vs PSF, vs θ_E, vs depth, vs EBV
```

### Step 2: Build Phase 4 Sampling Table

Sample from train/val/test with stratification across:
- PSF size bins (1.0-1.2", 1.2-1.5", 1.5-2.0", >2.0")
- Depth bins (based on `psfdepth_r`)
- EBV bins (low/medium/high extinction)
- Magnitude bins (bright/medium/faint)

### Step 3: Cutout Acquisition + Caching

Design cache structure:
```
cutouts/
├── {brickname}/
│   ├── {objid}_g.fits
│   ├── {objid}_r.fits
│   ├── {objid}_z.fits
│   ├── {objid}_invvar_g.fits
│   ├── {objid}_invvar_r.fits
│   └── {objid}_invvar_z.fits
```

Store manifests for reproducibility:
```
manifests/
├── train_sample_v1.csv
├── val_sample_v1.csv
└── test_sample_v1.csv
```

### Step 4: Injection + Recovery

Injection parameter grid:
- Einstein radius θ_E: [0.5", 1.0", 1.5", 2.0", 3.0"]
- Flux ratio: [0.1, 0.3, 0.5, 1.0]
- Source morphology: [point, Gaussian, Sérsic]
- Source position angle: random

Recovery pipeline:
- Feature extraction from cutouts
- Classification (lens vs non-lens)
- Parameter estimation (if lens detected)

### Step 5: Deliverables

1. **Completeness Surfaces**: Recovery probability as function of (θ_E, PSF, depth)
2. **"Seeing Wall" Plots**: Where completeness drops below threshold vs PSF
3. **Ranking Mode Ablation**: Compare density vs psf_weighted vs n_lrg region selections

---

## Data Paths

### Phase 3 Inputs (S3)

```
s3://darkhaloscope/phase3_pipeline/phase3p5/v3_color_relaxed/parent_compact/
s3://darkhaloscope/phase3_pipeline/phase3a/v3_color_relaxed/region_metrics/
s3://darkhaloscope/phase3_pipeline/phase3b/v3_color_relaxed/region_selections/
```

### Phase 4 Outputs (S3)

```
s3://darkhaloscope/phase4/
├── sampling/
│   ├── train_sample.parquet
│   ├── val_sample.parquet
│   └── test_sample.parquet
├── cutouts/
│   └── {brickname}/{objid}_{band}.fits
├── injections/
│   └── {injection_config}/
├── recovery/
│   └── {model_version}/
└── results/
    ├── completeness_surfaces.parquet
    └── seeing_wall_plots/
```

---

## Next Steps

1. **Design Sampling Strategy**: Determine sample sizes per split and stratification bins
2. **Implement Cutout Fetcher**: Build pipeline to acquire Legacy Survey cutouts
3. **Design Injection Framework**: Define lens models and injection parameters
4. **Build Recovery Pipeline**: Implement detection and characterization
5. **Run First Pass**: Execute on small sample to validate pipeline

---

## Session Log

### 2026-01-19: Phase 4 Initialization

**Context**: Phase 3 completed successfully with 19.7M LRGs across 811 regions. The catalog has been compacted from 188k files to 291 files (1.34 GB).

**Decision**: Proceeding with Phase 4 with the Phase 3 catalog frozen. The known split imbalance (81% test) is acknowledged but does not break scientific validity.

**First Action**: Define the sampling strategy and implement cutout acquisition infrastructure.

---

### 2026-01-19: Phase 4 Code Review and Fixes

**Context**: I received Phase 4 EMR pipeline code from an external source and performed a thorough code review before execution.

#### Files Reviewed

1. `emr/spark_phase4_pipeline.py` (1195 lines)
2. `emr/bootstrap_phase4_pipeline_install_deps.sh` (24 lines)
3. `emr/submit_phase4_pipeline_emr_cluster.py` (143 lines)

#### Code Structure Assessment

The Phase 4 pipeline implements 5 stages:

| Stage | Purpose | Status |
|-------|---------|--------|
| 4a | Build injection task manifests | Consistent with spec |
| 4b | Cache DR10 coadd assets from NERSC to S3 | Consistent with spec |
| 4c | Generate injected cutouts with SIS lens model | Consistent with spec |
| 4d | Compute completeness summaries | Consistent with spec |
| 4p5 | Compact outputs to reduce file count | Consistent with spec |

#### Issues Found and Fixed

**Critical Issue 1: Column Name Mismatch**

- **Problem**: Stage 4a expected column `lrg_v3_color_relaxed` but Phase 3.5 parent catalog uses `is_v3_color_relaxed`
- **Locations**: Line 319 (`needed_cols` list), Line 1117 (`--require-lrg-flag` default)
- **Impact**: Would cause `RuntimeError` at Stage 4a startup
- **Fix**: Changed both occurrences to `is_v3_color_relaxed`

**Critical Issue 2: FITS Extension Index for Compressed Files**

- **Problem**: Line 717 read `hdul[0].data` but compressed FITS files (`.fits.fz`) store image data in extension 1, not 0
- **Impact**: Would return `None` or wrong data when reading DR10 coadds
- **Fix**: Added logic to detect compressed FITS and read from extension 1 when extension 0 has no data

**Medium Issue 3: Deprecated Spark Config**

- **Problem**: Submit script used `spark.yarn.executor.memoryOverhead` which is deprecated in Spark 3.x
- **Impact**: May cause warnings or unexpected behavior
- **Fix**: Changed to `spark.executor.memoryOverhead`

#### Positive Findings

| Aspect | Assessment |
|--------|------------|
| EMR 6.x / Python 3.7 compatibility | Correct - uses `Optional`, `List` from typing |
| boto3 integration | Matches Phase 3 pattern |
| Bootstrap dependencies | Correct - astropy>=4.3,<5.0 for Py3.7 |
| Idempotency flags | Properly implemented (skip-if-exists, force) |
| Holdout discipline | Uses `region_split` throughout |
| Stage config JSON | Writes `_stage_config.json` per stage |
| SIS injection model | Documented as approximation with PSF convolution |

#### Phase 4 EMR Commands Reference

**Stage 4a (Manifest Build)**:
```bash
python3 submit_phase4_pipeline_emr_cluster.py \
  --region us-east-2 \
  --stage 4a \
  --log-uri s3://darkhaloscope/emr-logs/phase4/ \
  --subnet-id subnet-01ca3ae3325cec025 \
  --ec2-key-name root \
  --script-s3 s3://darkhaloscope/phase4/code/spark_phase4_pipeline.py \
  --bootstrap-s3 s3://darkhaloscope/phase4/code/bootstrap_phase4_pipeline_install_deps.sh \
  --core-instance-count 10 \
  --spark-args "--output-s3 s3://darkhaloscope/phase4_pipeline \
    --variant v3_color_relaxed \
    --parent-s3 s3://darkhaloscope/phase3_pipeline/phase3p5/v3_color_relaxed/parent_compact/ \
    --bricks-with-region-s3 s3://darkhaloscope/phase3_pipeline/phase3a/v3_color_relaxed/bricks_with_region/ \
    --region-selections-s3 s3://darkhaloscope/phase3_pipeline/phase3b/v3_color_relaxed/region_selections/ \
    --tiers debug \
    --skip-if-exists 1"
```

**Stage 4b (Cache Coadds)**:
```bash
python3 submit_phase4_pipeline_emr_cluster.py \
  --region us-east-2 \
  --stage 4b \
  --log-uri s3://darkhaloscope/emr-logs/phase4/ \
  --subnet-id subnet-01ca3ae3325cec025 \
  --ec2-key-name root \
  --script-s3 s3://darkhaloscope/phase4/code/spark_phase4_pipeline.py \
  --bootstrap-s3 s3://darkhaloscope/phase4/code/bootstrap_phase4_pipeline_install_deps.sh \
  --core-instance-count 10 \
  --spark-args "--output-s3 s3://darkhaloscope/phase4_pipeline \
    --variant v3_color_relaxed \
    --coadd-s3-cache-prefix s3://darkhaloscope/dr10/coadd_cache/ \
    --skip-if-exists 1"
```

---

*End of Log Entry*


# Plan C: Complete Project Status

**Last Updated:** 2026-02-11 02:00 UTC  
**Status:** Week 3 - Training complete, evaluation verified, selection function (initial run) complete. Paper readiness items addressed.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Strategy](#2-research-strategy)
3. [Data Sources and Acquisition](#3-data-sources-and-acquisition)
4. [Current State of All Components](#4-current-state-of-all-components)
5. [File Structure and Schemas](#5-file-structure-and-schemas)
6. [Technical Decisions and Rationale](#6-technical-decisions-and-rationale)
7. [Known Issues and Limitations](#7-known-issues-and-limitations)
8. [How to Resume Work](#8-how-to-resume-work)
9. [Week-by-Week Plan](#9-week-by-week-plan)
10. [Key Reference Documents](#10-key-reference-documents)

---

## 1. Project Overview

### Goal
Produce an MNRAS-level paper titled: **"Selection Functions and Failure Modes of Real-Image Lens Finders in DR10"**

### Core Contribution
1. **Rigorous selection function** for DR10 lens finding - quantifying detection probability as a function of observing conditions (exposures, seeing, depth) and galaxy properties (morphology type)
2. **Bias audit** tied to operational choices already known to matter
3. **Controlled ensemble diversification study** showing domain-specialized training improves coverage

### Why This Approach (The Pivot)
- **Original Plan (Plan B):** Train on simulated lensed arcs
- **Problem Discovered:** Simulated arcs were ~100× too bright compared to real DR10 data (measured via arc-annulus SNR analysis)
- **New Approach (Plan C):** Train entirely on real observed images, following Huang et al. (arXiv:2508.20087v1) methodology

### Paper Framing
*"Where do ML lens finders fail, and how does data quality shape their selection function?"*

We answer:
- What is the detection probability across observing conditions?
- What contaminant classes dominate false positives?
- How sensitive are results to domain shift?
- Does domain-specialized training improve recovery?

---

## 2. Research Strategy

### Methodology (Following Huang et al.)

1. **Training Data:** Real DR10 cutouts of lenses and non-lenses (no simulations)
2. **Stratification:** Bin non-lenses by z-band exposure count to prevent network shortcuts
3. **Ratio:** Maintain ~100:1 non-lens:lens in each stratum
4. **Architecture:** ResNet-18 baseline (optionally EfficientNet-B0)
5. **Evaluation:** Recall on Tier-A (confirmed) lenses only; Tier-B used with label smoothing

### Two-Tier Label System

| Tier | Definition | Count | Use |
|------|------------|-------|-----|
| **Tier-A** | Spectroscopically confirmed lenses | 434 | Primary evaluation metric |
| **Tier-B** | Probable candidates (grading="probable") | 4,666 | Training with label smoothing (target=0.8) |

### Stratification Axes

1. **nobs_z** (z-band exposures): low (1-4), medium (4-7), high (7+)
2. **psfsize_z** (seeing): excellent (<1.1"), good (1.1-1.3"), fair (1.3-1.5"), poor (>1.5")
3. **psfdepth_z** (depth): quartile-based bins
4. **tractor_type** (morphology): SER, DEV, REX, EXP, other

---

## 3. Data Sources and Acquisition

### 3.1 Positive Lenses

**Source:** lenscat catalog (community-aggregated lens catalog)
- Website: https://github.com/lenscat/lenscat
- Subset: DESI Legacy Survey candidates with DR10 coverage

**Acquisition Process:**
1. Queried lenscat for DESI candidates
2. Filtered to those with DR10 coverage (`in_dr10=True`)
3. Downloaded JPG cutouts (already done in Plan B)
4. Enriched with Tractor metadata via NOAO DataLab

**Files:**
- Original catalog: `data/positives/desi_candidates.csv`
- Enriched catalog: `data/positives/desi_candidates_enriched.csv`

### 3.2 Tractor Metadata

**Source:** NOAO DataLab TAP service
- Endpoint: `https://datalab.noirlab.edu/tap/sync`
- Table: `ls_dr10.tractor`

**Acquisition Process:**
1. For each positive lens position, query a 5" box around RA/Dec
2. Select closest source within box
3. Extract: type, nobs_z, psfsize_z, psfdepth_z, flux_z, ebv, brickname

**Script:** `data/enrich_positives_production.py`
- Multi-threaded (5 workers)
- Checkpoint/resume support
- Progress tracking

**Query Template:**
```sql
SELECT ls_id, ra, dec, type, nobs_z, psfsize_z, psfdepth_z, flux_z, ebv, brickname
FROM ls_dr10.tractor
WHERE ra BETWEEN {ra-0.0014} AND {ra+0.0014}
  AND dec BETWEEN {dec-0.0014} AND {dec+0.0014}
```

### 3.3 Negative Galaxies

**Source:** DR10 Tractor sweep files (local download)
- URL pattern: `https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/sweep/10.0/sweep-{ra_lo}{dec_sign}{dec_lo}-{ra_hi}{dec_sign}{dec_hi}.fits`

**Acquisition Process:**
1. Downloaded 2 sweep files covering RA 160-170°, Dec -10 to -5°
2. Extracted all galaxies (SER, DEV, REX, EXP types)
3. Applied stratified sampling to match positive distribution

**Files:**
- Sweep files: `data/sweep_files/sweep-*.fits` (~2.5 GB)
- Full extraction: `data/negatives/negative_catalog_prototype.csv` (2.9M galaxies)
- Stratified sample: `data/negatives/negative_catalog_stratified.csv` (257K galaxies)

**Limitation:** Sweep files cover limited sky region. Full-sky sampling would require ~900 GB of sweep files.

### 3.4 FITS Cutouts

**Source:** Legacy Survey Cutout Service
- URL: `https://www.legacysurvey.org/viewer/fits-cutout?ra={ra}&dec={dec}&layer=ls-dr10&pixscale=0.262&size=101`

**Cutout Specifications:**
- Size: 101×101 pixels
- Pixel scale: 0.262"/pixel
- Physical size: ~26.5" × 26.5"
- Bands: g, r, i, z (4 channels)
- Format: FITS, float32

**Script:** `data/download_fits_cutouts.py`
- Parallel download with rate limiting
- Retry logic for 429 errors
- Coordinate-based filenames: `ra{ra}_dec{dec}.fits`

**Rate Limits:** Legacy Survey limits to ~2-3 requests/second. Full download of 5,104 cutouts takes ~2-3 hours.

---

## 4. Current State of All Components

### 4.1 Positive Lens Catalog

| Item | Status | Count | File |
|------|--------|-------|------|
| Raw catalog | ✅ Complete | 5,104 | `data/positives/desi_candidates.csv` |
| With metadata | ✅ Complete | 5,100 (4 failed) | `data/positives/desi_candidates_enriched.csv` |
| Tier-A (confirmed) | ✅ Identified | 434 | grading="confident" |
| Tier-B (probable) | ✅ Identified | 4,666 | grading="probable" |

**Enriched Catalog Schema:**
```
idx,name,ra,dec,grading,success,tractor_type,nobs_z,psfsize_z,psfdepth_z,flux_z,ebv,brickname,match_dist_arcsec,error
```

**Sample Row:**
```
0,J1101-0602 | DESI-165.4754-06.0423,165.4754,-6.04226,confident,True,SER,3,1.2379925,98.938385,76.4436,0.033810448,1655m060,0.13031147277438299,
```

### 4.2 Negative Galaxy Catalog

| Item | Status | Count | File |
|------|--------|-------|------|
| Prototype (all) | ✅ Complete | 2,912,067 | `data/negatives/negative_catalog_prototype.csv` |
| Stratified sample | ✅ Complete | 257,547 | `data/negatives/negative_catalog_stratified.csv` |

**Stratified Catalog Schema:**
```
ra,dec,type,nobs_z,psfsize_z,psfdepth_z,flux_z,brickname,nobs_bin
```

**Distribution by Cross-Bin (nobs × type):**
```
('high', 'DEV'): 20,271 (47:1 ratio)
('high', 'EXP'): 9,600 (100:1 ratio)
('high', 'REX'): 41,000 (100:1 ratio)
('high', 'SER'): 11,569 (7:1 ratio) ← UNDERSAMPLED
('low', 'DEV'): 20,500 (100:1 ratio)
('low', 'EXP'): 4,400 (100:1 ratio)
('low', 'REX'): 8,900 (100:1 ratio)
('low', 'SER'): 28,782 (77:1 ratio)
('medium', 'DEV'): 38,700 (100:1 ratio)
('medium', 'EXP'): 7,900 (100:1 ratio)
('medium', 'REX'): 26,300 (100:1 ratio)
('medium', 'SER'): 39,625 (43:1 ratio) ← UNDERSAMPLED
```

**Known Issue:** SER (Sérsic) galaxies are underrepresented in the sweep file region. This affects the 'high' and 'medium' nobs bins most severely.

### 4.3 FITS Cutouts

| Item | Status | Count | Location |
|------|--------|-------|----------|
| Positive cutouts | ✅ Complete | 5,102 / 5,104 | `data/positives/fits_cutouts/` |
| Negative cutouts | ⏳ Not Started | 0 / 257,547 | `data/negatives/fits_cutouts/` |

**Positive cutout download completed:** 2026-02-07 09:50 UTC
- 5,102 successful, 2 failed (likely outside DR10 coverage)
- Total size: ~830 MB (5,102 × 167 KB each)

**To verify:**
```bash
ls data/positives/fits_cutouts/ | wc -l
# Expected: 5102
```

### 4.4 Stratification Bins

**File:** `data/positives/stratification_bins.json`

**Content:**
```json
{
  "description": "Stratification bins for negative sampling",
  "total_positives": 5100,
  "negative_ratio": 100,
  "nobs_z": {
    "bins": [
      {"name": "low", "range": [1, 4], "count": 726},
      {"name": "medium", "range": [4, 7], "count": 1668},
      {"name": "high", "range": [7, 100], "count": 2705}
    ]
  },
  "psfsize_z": {
    "bins": [
      {"name": "excellent", "range": [0, 1.1], "count": 1036},
      {"name": "good", "range": [1.1, 1.3], "count": 2779},
      {"name": "fair", "range": [1.3, 1.5], "count": 905},
      {"name": "poor", "range": [1.5, 10], "count": 380}
    ]
  },
  "tractor_type": {
    "bins": [
      {"name": "SER", "count": 3047},
      {"name": "DEV", "count": 1022},
      {"name": "REX", "count": 762},
      {"name": "EXP", "count": 219},
      {"name": "other", "count": 50}
    ]
  }
}
```

### 4.5 Distribution Analysis

**Visualization:** `data/positives/distribution_analysis.png`

**Key Statistics:**
| Metric | Value |
|--------|-------|
| Median nobs_z | 7 |
| Median psfsize_z | 1.19" |
| Median psfdepth_z | 254 |
| Dominant type | SER (60%) |

### 4.6 Contaminants

| Item | Status | Notes |
|------|--------|-------|
| Ring galaxies | ⏳ Not Started | Source: Galaxy Zoo DR2 |
| Face-on spirals | ⏳ Not Started | Source: Galaxy Zoo |
| Mergers | ⏳ Not Started | Source: Galaxy Zoo |
| Edge-on disks | ⏳ Not Started | Source: Galaxy Zoo |
| Star artifacts | ⏳ Not Started | Source: Gaia + DR10 cross-match |

**Target:** ~10,000 contaminants total

---

## 5. File Structure and Schemas

### 5.1 Directory Structure

```
planc/
├── README.md                              # Project goals and overview
├── PROJECT_STATUS.md                      # THIS FILE - complete state
├── FULL_PIPELINE_STATUS.md                # Pipeline with all 9 LLM gaps fixed
├── AUDIT_VS_LLM_BLUEPRINT.md              # Alignment check with LLM guidance
├── LLM_BLUEPRINT_RESPONSE.md              # External LLM's detailed guidance
├── IMPLEMENTATION_BLUEPRINT_REQUEST.md    # Questions sent to external LLM
├── WEEK1_SUMMARY.md                       # Week 1 completion summary
│
├── data/
│   ├── positives/
│   │   ├── desi_candidates.csv            # Original 5,104 candidates
│   │   ├── desi_candidates_enriched.csv   # With Tractor metadata
│   │   ├── desi_candidates_enriched.checkpoint.json  # Enrichment checkpoint
│   │   ├── distribution_analysis.png      # Visualization
│   │   ├── stratification_bins.json       # Bin definitions
│   │   └── fits_cutouts/                  # FITS images (downloading)
│   │       └── ra{ra}_dec{dec}.fits       # Filename pattern
│   │
│   ├── negatives/
│   │   ├── negative_catalog_prototype.csv     # Full 2.9M from sweeps
│   │   └── negative_catalog_stratified.csv   # Stratified 257K sample
│   │
│   ├── sweep_files/
│   │   ├── sweep-160m010-165m005.fits     # ~1.1 GB
│   │   └── sweep-165m010-170m005.fits     # ~1.3 GB
│   │
│   ├── download_fits_cutouts.py           # Cutout downloader
│   ├── enrich_positives_production.py     # DataLab metadata enricher
│   ├── enrich_positives_with_metadata.py  # Original LLM version
│   ├── query_datalab_metadata.py          # Single-source DataLab query
│   ├── query_negatives.py                 # Original negative sampler
│   ├── query_negatives_fast.py            # Parallel negative sampler
│   ├── query_positive_metadata.py         # Initial metadata attempt (deprecated)
│   ├── sweep_utils.py                     # Sweep file utilities
│   ├── preprocessing.py                   # LLM-provided preprocessing
│   ├── prepare_dataset.py                 # LLM-provided dataset prep
│   ├── download_cutouts.py                # LLM-provided downloader
│   └── download_negatives.py              # LLM-provided negative sampler
│
├── training/
│   ├── models.py                          # ResNet-18, EfficientNet definitions
│   ├── dataset.py                         # PyTorch Dataset class
│   ├── train_baseline.py                  # Training script
│   └── losses.py                          # BCE, Focal Loss implementations
│
├── evaluation/
│   ├── compute_completeness.py            # Selection function computation
│   ├── compute_calibration.py             # ECE, reliability diagrams
│   ├── analyze_failures.py                # Contaminant FPR analysis
│   └── statistical_tests.py               # Bootstrap, binomial CIs
│
├── inference/
│   └── batch_scorer.py                    # Batch inference script
│
├── paper/
│   ├── generate_figures.py                # Figure generation
│   └── generate_tables.py                 # Table generation
│
└── common/
    ├── retry.py                           # Retry utilities
    └── logging_utils.py                   # Logging setup
```

### 5.2 Catalog Schemas

**Positive Enriched Catalog (`desi_candidates_enriched.csv`):**
| Column | Type | Description |
|--------|------|-------------|
| idx | int | Row index |
| name | str | Lens name/identifier |
| ra | float | Right Ascension (degrees) |
| dec | float | Declination (degrees) |
| grading | str | "confident" (Tier-A) or "probable" (Tier-B) |
| success | bool | Whether metadata query succeeded |
| tractor_type | str | SER, DEV, REX, EXP, PSF, DUP |
| nobs_z | int | Number of z-band exposures |
| psfsize_z | float | PSF FWHM in z-band (arcsec) |
| psfdepth_z | float | 5σ point source depth in z-band |
| flux_z | float | z-band flux (nanomaggies) |
| ebv | float | E(B-V) galactic extinction |
| brickname | str | DR10 brick identifier |
| match_dist_arcsec | float | Distance to matched Tractor source |
| error | str | Error message if failed |

**Negative Catalog (`negative_catalog_stratified.csv`):**
| Column | Type | Description |
|--------|------|-------------|
| ra | float | Right Ascension (degrees) |
| dec | float | Declination (degrees) |
| type | str | Tractor morphology type |
| nobs_z | int | Number of z-band exposures |
| psfsize_z | float | PSF FWHM (arcsec) |
| psfdepth_z | float | Depth |
| flux_z | float | z-band flux |
| brickname | str | Brick identifier |
| nobs_bin | str | Stratification bin (low/medium/high) |

### 5.3 FITS Cutout Format

**Filename Pattern:** `ra{ra}_dec{dec}.fits`
- Example: `ra165p4754_decm6p0423.fits`
- `.` replaced with `p`, `-` replaced with `m`

**Array Shape:** `(4, 101, 101)` = (channels, height, width)
- Channel 0: g-band
- Channel 1: r-band
- Channel 2: i-band
- Channel 3: z-band

**Data Type:** float32 (big-endian, `>f4`)

**For training, use channels [0, 1, 3] for g, r, z (skip i-band to match 3-channel expectation).**

---

## 6. Technical Decisions and Rationale

### 6.1 Why DataLab Instead of Direct API

**Tried:** Legacy Survey viewer API (`/viewer/tractor-for-cutout/`)
**Problem:** Returns HTML instead of JSON for batch queries
**Solution:** NOAO DataLab TAP service with rectangular box queries

### 6.2 Why Local Sweep Files Instead of Full DataLab Queries

**Tried:** DataLab queries for 500K negatives
**Problem:** Queries timeout or get rate-limited; each query takes ~6 seconds
**Solution:** Download local sweep files, sample from them
**Trade-off:** Limited to one sky region, but fast and reliable

### 6.3 Why Coordinate-Based Filenames

**Problem:** Original lens names contain special characters (`|`, `/`, spaces)
**Solution:** Use `ra{ra}_dec{dec}` format for consistent, filesystem-safe names

### 6.4 Why 101×101 Cutout Size

**Rationale:** Matches Huang et al. Paper IV methodology exactly
- 101 pixels × 0.262"/pixel = 26.5" field of view
- Captures typical Einstein radius + surrounding context
- Small enough for efficient training, large enough for arc detection

### 6.5 Why 100:1 Negative Ratio

**Rationale:** Follows Huang et al.
- More realistic class imbalance than typical ML datasets
- Stratified per bin prevents exposure-count confound
- Higher ratio unnecessary; would slow training without benefit

### 6.6 Label Smoothing for Tier-B

**Decision:** Use target=0.8 for Tier-B instead of 1.0
**Rationale:** Tier-B are probable lenses, not confirmed. Label smoothing:
- Reduces overconfidence on noisy labels
- Prevents model from memorizing potentially wrong labels
- Tier-A anchors (target=1.0) remain the ground truth for evaluation

---

## 7. Known Issues and Limitations

### 7.1 SER Galaxy Undersampling

**Issue:** SER (Sérsic) type galaxies are underrepresented in negatives
- high/SER: 7:1 ratio (target 100:1)
- medium/SER: 43:1 ratio (target 100:1)

**Cause:** Local sweep files cover a region with fewer SER galaxies
**Impact:** Model may have less robust rejection of SER-type contaminants
**Mitigation:** Document as limitation; consider additional sweep file downloads

### 7.2 Spatial Bias in Negatives

**Issue:** All negatives come from RA 160-170°, Dec -10 to -5°
**Impact:** Potential spatial correlations in observing conditions not captured
**Mitigation:** Document as limitation; spatial holdout CV may show this

### 7.3 Rate Limiting on Cutout Service

**Issue:** Legacy Survey limits to ~2-3 requests/second
**Impact:** Downloading 5,104 cutouts takes 2-3 hours
**Mitigation:** Run download overnight; retry logic handles 429 errors

### 7.4 Missing Contaminants

**Issue:** Galaxy Zoo contaminants not yet acquired
**Impact:** Cannot do FPR-by-category analysis until Week 2
**Mitigation:** Proceed with training; add contaminants before evaluation

### 7.5 4-Channel vs 3-Channel Cutouts

**Issue:** Cutout service returns g,r,i,z (4 channels); we expect g,r,z (3)
**Solution:** Extract channels [0, 1, 3] and skip i-band
**Alternative:** Keep all 4 channels (minor architecture change)

---

## 8. How to Resume Work

### 8.1 Verify Positive Cutouts (COMPLETE)

```bash
cd /Users/balaji/code/oss/toy_glens_kinematics/dark_halo_scope/planc

# Count downloaded cutouts (should be 5102)
ls data/positives/fits_cutouts/ | wc -l

# Verify sample cutout shape
python3 -c "from astropy.io import fits; print(fits.open('data/positives/fits_cutouts/$(ls data/positives/fits_cutouts | head -1)')[0].data.shape)"
# Expected: (4, 101, 101)
```

### 8.2 Download Negative Cutouts (When Ready)

```bash
# Sample 50K negatives for training prototype
head -50001 data/negatives/negative_catalog_stratified.csv > data/negatives/neg_sample_50k.csv

# Download cutouts
python3 -u data/download_fits_cutouts.py \
  --catalog data/negatives/neg_sample_50k.csv \
  --output data/negatives/fits_cutouts \
  --label 0 \
  --workers 2 \
  --name-col "" \
  --ra-col ra \
  --dec-col dec
```

### 8.3 Verify Data Integrity

```bash
# Check FITS file shapes
python3 << 'EOF'
from astropy.io import fits
import os
import random

cutout_dir = 'data/positives/fits_cutouts'
files = os.listdir(cutout_dir)
sample = random.sample(files, min(10, len(files)))

for f in sample:
    with fits.open(os.path.join(cutout_dir, f)) as hdu:
        shape = hdu[0].data.shape
        if shape != (4, 101, 101):
            print(f"WARNING: {f} has shape {shape}")
        else:
            print(f"OK: {f}")
EOF
```

### 8.4 Create Train/Val/Test Splits (Next Step)

```bash
python3 data/prepare_dataset.py \
  --pos_dir data/positives/fits_cutouts \
  --neg_dir data/negatives/fits_cutouts \
  --out data/datasets/dr10_v1 \
  --train_frac 0.7 \
  --val_frac 0.15 \
  --test_frac 0.15 \
  --stratify_by nobs_bin,tractor_type
```

### 8.5 Train Baseline Model

```bash
python3 training/train_baseline.py \
  --data_root data/datasets/dr10_v1 \
  --model resnet18 \
  --epochs 30 \
  --batch_size 64 \
  --lr 1e-4 \
  --out runs/baseline_resnet18
```

---

## 9. Week-by-Week Plan

### Week 1: Data Preparation (95% Complete)

| Task | Status | Notes |
|------|--------|-------|
| Positive catalog | ✅ Done | 5,104 candidates |
| Tractor metadata | ✅ Done | 5,100 enriched |
| Distribution analysis | ✅ Done | Visualization + bins |
| Stratified negatives | ✅ Done | 257K samples |
| Positive cutouts | ✅ Done | 5,102 / 5,104 downloaded |
| Negative cutouts | ⏳ Pending | Need ~50K for prototype |
| Contaminants | ⏳ Deferred | Galaxy Zoo query |
| Cutout verification | ✅ Done | (4, 101, 101) confirmed |

### Week 2: Model Training

| Task | Status | Notes |
|------|--------|-------|
| Complete cutout downloads | ⏳ | Overnight |
| Create train/val/test splits | ⏳ | 70/15/15, stratified |
| Implement augmentations | ⏳ | Rotation, flip, noise |
| Train ResNet-18 baseline | ⏳ | BCEWithLogits + pos_weight |
| Validate training stability | ⏳ | Monitor AUC, no collapse |
| Sanity check top-K | ⏳ | No obvious artifacts |

### Week 3: Selection Function + Failures

| Task | Status | Notes |
|------|--------|-------|
| Recall on Tier-A by stratum | ⏳ | Bootstrap CIs |
| Small-N strata handling | ⏳ | Binomial/beta intervals |
| Calibration curves | ⏳ | ECE, reliability diagrams |
| Scenario-weighted calibration | ⏳ | 1:10,000 prior |
| FPR by contaminant | ⏳ | Needs Galaxy Zoo data |
| Spatial holdout CV | ⏳ | Region-based splits |

### Week 4: Ensemble + Paper

| Task | Status | Notes |
|------|--------|-------|
| Domain-split models | ⏳ | PSF or nobs axis |
| Diversity metrics | ⏳ | Correlation, disagreement |
| Ensemble evaluation | ⏳ | Averaging vs single |
| Paper figures | ⏳ | 7-8 main figures |
| Paper tables | ⏳ | 5-6 tables |
| Paper text | ⏳ | Methods, Results, Discussion |

---

## 10. Key Reference Documents

### 10.1 External LLM Guidance

**File:** `LLM_BLUEPRINT_RESPONSE.md`

Key recommendations followed:
1. ResNet-18 first, not deeper
2. 100:1 ratio per stratification bin
3. Tier-A for evaluation, Tier-B with label smoothing
4. Bootstrap CIs, binomial for small N
5. Scenario-weighted calibration (1:10,000 prior)
6. Spatial holdout CV for correlation assessment

### 10.2 Pipeline Alignment

**File:** `AUDIT_VS_LLM_BLUEPRINT.md`

9 gaps identified and fixed:
1. ✅ Metadata branch option added
2. ✅ Label smoothing for Tier-B
3. ✅ Scenario-weighted calibration
4. ✅ Binomial/Bayesian CI for small N
5. ✅ Spatial holdout CV
6. ✅ EfficientNet-B0 option
7. ✅ z-only ablation option
8. ✅ Focal loss fallback
9. ✅ Contaminant sources identified

### 10.3 Full Pipeline Definition

**File:** `FULL_PIPELINE_STATUS.md`

Complete pipeline with all training, evaluation, and paper components defined.

### 10.4 Huang et al. Methodology

**Paper:** arXiv:2508.20087v1 (Papers I-IV combined)

Key methodological anchors we copy:
1. Training on real DR10 cutouts (no simulations)
2. Nonlens selection stratified by z-band exposure count
3. 100:1 nonlens:lens ratio per bin
4. 101×101 pixel cutouts
5. High threshold (top 0.01%) for discovery

---

## Appendix: Quick Reference Commands

```bash
# Navigate to project
cd /Users/balaji/code/oss/toy_glens_kinematics/dark_halo_scope/planc

# Check positive cutout progress
ls data/positives/fits_cutouts/ | wc -l

# Check if download running
ps aux | grep download_fits_cutouts | grep -v grep

# Resume positive download if stopped
python3 -u data/download_fits_cutouts.py \
  --catalog data/positives/desi_candidates_enriched.csv \
  --output data/positives/fits_cutouts \
  --label 1 --workers 2 --name-col name

# View positive distribution
open data/positives/distribution_analysis.png

# Count negatives
wc -l data/negatives/negative_catalog_stratified.csv

# Verify FITS shape
python3 -c "from astropy.io import fits; print(fits.open('data/positives/fits_cutouts/$(ls data/positives/fits_cutouts | head -1)')[0].data.shape)"

# Check Tier-A/Tier-B counts
grep -c "confident" data/positives/desi_candidates_enriched.csv
grep -c "probable" data/positives/desi_candidates_enriched.csv
```

---

*Last updated: 2026-02-07 09:30 UTC*
*Next update: After Week 1 completion*

# Week 1 Summary: Data Preparation

**Date:** 2026-02-07  
**Status:** 80% Complete

---

## Completed Tasks

### 1. Positive Lens Catalog ✅
- **Source:** lenscat DESI candidates
- **Total:** 5,104 candidates
- **Tier-A (confirmed):** 434 lenses
- **Tier-B (probable):** 4,666 candidates
- **File:** `data/positives/desi_candidates.csv`

### 2. Tractor Metadata Enrichment ✅
- **Success rate:** 5,100/5,104 (99.9%)
- **Columns added:** nobs_z, psfsize_z, psfdepth_z, flux_z, ebv, tractor_type, brickname
- **File:** `data/positives/desi_candidates_enriched.csv`

### 3. Distribution Analysis ✅
- **Visualization:** `data/positives/distribution_analysis.png`
- **Key findings:**
  - Median z-band exposures: 7
  - Median PSF size: 1.19"
  - Dominant type: SER (60%)
  - Type distribution: SER > DEV > REX > EXP

### 4. Stratification Bins ✅
- **File:** `data/positives/stratification_bins.json`
- **nobs_z bins:** low (1-4), medium (4-7), high (7+)
- **PSF bins:** excellent (<1.1"), good (1.1-1.3"), fair (1.3-1.5"), poor (>1.5")
- **Type bins:** SER, DEV, REX, EXP, other

### 5. Stratified Negative Catalog ✅
- **Total negatives:** 257,547
- **Source:** Local sweep files (DR10)
- **Stratification:** Matched to positive distribution by nobs_z × type
- **File:** `data/negatives/negative_catalog_stratified.csv`
- **Limitation:** Sourced from limited sky region (RA 160-170°, Dec -10 to -5°)

### 6. Cutout Format Verified ✅
- **Shape:** (4, 101, 101) - 4 channels × 101×101 pixels
- **Channels:** g, r, i, z bands
- **Pixel scale:** 0.262"/pixel
- **Total size:** ~26" × 26"

---

## In Progress

### 7. Positive FITS Cutouts 🔄
- **Downloaded:** ~850 / 5,104 (16.7%)
- **Status:** Running with rate-limited parallel download
- **ETA:** Overnight (rate limited to ~2 concurrent requests)
- **Output:** `data/positives/fits_cutouts/`

---

## Deferred to Week 2

### 8. Contaminant Catalog ⏳
- **Reason:** Time constraints, prioritizing core pipeline
- **Plan:** Source from Galaxy Zoo DR2 (rings, spirals, mergers, edge-on)
- **Target:** ~10,000 contaminants

### 9. Negative FITS Cutouts ⏳
- **Reason:** Large volume (257K), need selective sampling
- **Plan:** Download ~50K cutouts for training prototype
- **Strategy:** Sample proportionally from stratified catalog

---

## Data Summary

| Dataset | Count | Status |
|---------|-------|--------|
| Tier-A lenses (confirmed) | 434 | ✅ Ready |
| Tier-B lenses (probable) | 4,666 | ✅ Ready |
| Positive FITS cutouts | ~850 | 🔄 Downloading |
| Negative catalog | 257,547 | ✅ Ready |
| Negative FITS cutouts | 0 | ⏳ Pending |
| Contaminants | 0 | ⏳ Deferred |

---

## Files Created

```
planc/
├── data/
│   ├── positives/
│   │   ├── desi_candidates.csv              # Original catalog
│   │   ├── desi_candidates_enriched.csv     # With Tractor metadata
│   │   ├── distribution_analysis.png        # Visualization
│   │   ├── stratification_bins.json         # Bin definitions
│   │   └── fits_cutouts/                    # FITS images (downloading)
│   │
│   ├── negatives/
│   │   ├── negative_catalog_prototype.csv   # Full 2.9M from sweep files
│   │   └── negative_catalog_stratified.csv  # Stratified 257K sample
│   │
│   ├── sweep_files/                         # Local DR10 sweep files
│   │   ├── sweep-160m010-165m005.fits
│   │   └── sweep-165m010-170m005.fits
│   │
│   ├── download_fits_cutouts.py             # Cutout downloader
│   ├── enrich_positives_production.py       # Metadata enricher
│   ├── query_datalab_metadata.py            # DataLab queries
│   ├── query_negatives_fast.py              # Negative sampling
│   └── sweep_utils.py                       # Sweep file utilities
│
├── FULL_PIPELINE_STATUS.md                  # Master plan
├── AUDIT_VS_LLM_BLUEPRINT.md                # Blueprint alignment
├── LLM_BLUEPRINT_RESPONSE.md                # External LLM guidance
└── WEEK1_SUMMARY.md                         # This file
```

---

## Known Limitations

1. **Negative sample spatial bias:** Limited to RA 160-170°, Dec -10 to -5° due to local sweep files. Full-sky sampling deferred.

2. **SER type underrepresented in negatives:** Only 43:1 ratio vs 100:1 target due to sweep file region.

3. **Rate limiting on cutout service:** Download speed capped at ~2-3 cutouts/sec.

---

## Next Steps (Week 2)

1. **Complete positive cutout download** (continue overnight)
2. **Download subset of negative cutouts** (~50K for prototype)
3. **Create train/val/test splits** with stratification
4. **Implement data augmentation**
5. **Train baseline ResNet-18 model**

---

## Checkpoint Verification

### LLM Blueprint Week 1 Checkpoints

| Checkpoint | Status |
|------------|--------|
| Tier-A/Tier-B separated | ✅ Done (434/4666) |
| Tractor metadata validated | ✅ Done (5,100 matched) |
| Stratified negative catalog built | ✅ Done (257K) |
| Cutout size verified = 101×101 | ✅ Done |
| Negatives matched by z-exposure bin | ⚠️ Partial (limited region) |

---

*Summary generated: 2026-02-07*

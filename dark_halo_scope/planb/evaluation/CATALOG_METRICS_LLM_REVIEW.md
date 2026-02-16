# Anchor & Contaminant Catalog Review Request

## Context

We are building a strong gravitational lens finder for DESI Legacy Survey DR10. We need to evaluate our trained model on:

1. **Anchors**: Real spectroscopically-confirmed lenses to measure recall
2. **Contaminants**: Known non-lenses that look like lenses to measure false positive rate

We have built these catalogs and need your review before proceeding to model evaluation.

---

## Project Background

- **Training data**: Synthetic lenses injected into real galaxy cutouts (COSMOS morphologies)
- **Model**: ResNet18 binary classifier (lens vs non-lens)
- **Training approach**: "Unpaired" sampling (lenses and non-lenses from different galaxies)
- **Training range**: θ_E ∈ [0.5", 3.0"]
- **Survey**: DESI Legacy Survey DR10 (0.262"/pixel)

---

## Anchor Catalog Metrics

| Metric | Value |
|--------|-------|
| Total anchors collected | 94 |
| In DR10 footprint | 94 (100%) |
| Usable cutouts (no artifacts) | 94 (100%) |
| Arc visible in DR10 (SNR > 2.0) | 93 (99%) |
| **Tier-A (within selection function)** | **93** |
| Tier-B (outside selection function) | 1 |

### θ_E Distribution (Tier-A anchors only)

- Minimum: 0.79"
- Maximum: 1.83"
- Median: 1.20"
- Training range: 0.5" - 3.0"

### Source Breakdown

| Source | Count | Description |
|--------|-------|-------------|
| SLACS | 50 | SDSS Lens ACS Survey (spectroscopic) |
| BELLS | 20 | BOSS Emission-Line Lens Survey (spectroscopic) |
| SL2S | 10 | Strong Lensing Legacy Survey |
| SWELLS | 8 | SDSS-WFCAM Extended Lens Survey |
| GALLERY | 6 | Ground-based discoveries |

### Tier-B Exclusion Reasons

| Reason | Count |
|--------|-------|
| Arc not visible in DR10 | 1 |

### Arc Visibility Measurement Method

- Used MAD-based robust noise estimation (per LLM feedback)
- θ_E-centered annulus (not fixed pixel range)
- Sum of positive residuals after subtracting azimuthal median
- SNR threshold: 2.0

---

## Contaminant Catalog Metrics

| Metric | Value |
|--------|-------|
| Total contaminants collected | 183 |
| In DR10 footprint | 176 (96%) |
| Gold tier (p > 0.7, high purity) | 115 |
| Silver tier (p 0.5-0.7, more yield) | 20 |
| **Artifact tier (bright-star artifacts)** | **48** |

### Category Breakdown

| Category | Count (in DR10) | Description |
|----------|-----------------|-------------|
| spiral | 50 | Face-on spirals (arms mimic arcs) |
| ring | 30 | Collisional ring galaxies |
| merger | 20 | Interacting/tidal features |
| edge_on | 20 | Edge-on disks (can look like arcs) |
| agn | 10 | AGN with extended features |
| polar_ring | 5 | Polar ring galaxies |
| **spike** | **30** | Diffraction spikes from bright stars |
| **ghost** | **13** | PSF ghosts/halos from bright stars |
| **satellite** | **5** | Satellite trails |

### Quality Tiers

- **Gold**: High-confidence morphological classification (e.g., p_ring > 0.7)
- **Silver**: Moderate-confidence (e.g., p_ring 0.5-0.7)
- **Artifact**: Bright-star artifacts (spikes, ghosts, halos) - definitively not lenses

---

## Selection Function Applied

### Anchor Selection Function

An anchor is **Tier-A** if ALL of:
1. θ_E ∈ [0.5", 3.0"] (training support)
2. Arc visible in DR10 (SNR > 2.0)
3. In DR10 footprint
4. Cutout usable (no masking/artifacts)

### Contaminant Selection Function

A contaminant is **valid** if ALL of:
1. Category ∈ {ring, spiral, merger, edge_on, agn, polar_ring, spike, ghost, satellite}
2. In DR10 footprint
3. Not a confirmed lens
4. No coordinate match to any anchor (within 3")

---

## Known Limitations (Pre-Acknowledged)

### θ_E Coverage Limitation (C2)

Our anchor θ_E distribution (0.79" - 1.83", median 1.20") does NOT cover the full training range (0.5" - 3.0").

**Reporting constraint applied:**
- All recall estimates will be explicitly labeled as valid for **θ_E ∈ [0.8", 1.8"]** only
- We will NOT claim recall for θ_E < 0.8" or θ_E > 1.8"
- Model selection based on anchors applies only within this constrained range

### Arc Visibility Validation Status (C1)

A **20-object human sanity check** has been prepared for SLACS/BELLS anchors:
- HTML visualization with r-band and residual images
- θ_E annulus overlay showing the measurement region
- Labels to collect: "clear arc", "maybe", "no arc"
- Purpose: Validate that SNR > 2.0 correlates with real arc visibility

**Status**: Visualization ready at `sanity_check/arc_visibility_review.html`

---

## Direct Questions for Review

### Anchors

1. **Is 93 Tier-A anchors sufficient for a first sim-to-real checkpoint?**
   - We will compute recall with bootstrap uncertainty
   - We are NOT claiming precision completeness curves

2. **The θ_E range is 0.79" - 1.83" with median 1.20". Is this concerning?**
   - Training range is 0.5" - 3.0"
   - We have no anchors < 0.79" or > 1.83"
   - Does this bias our recall estimate?

3. **Only 1 anchor failed Tier-A (arc not visible). Is this suspiciously low?**
   - Could indicate our arc visibility threshold is too lenient
   - Or that these confirmed lenses genuinely have visible arcs in DR10

4. **Should we add more anchor sources to get above 100?**
   - We could add: SWELLS-II, S4TM, HST-discovered lenses
   - But diminishing returns vs. added complexity

### Contaminants

5. **Is 128 valid contaminants (in DR10) sufficient?**
   - Target was 170-200
   - Current categories may be missing some confuser types

6. **Is the category distribution appropriate?**
   - Spirals dominate (50/135 = 37%)
   - Is this representative of what we'll encounter in a real search?

7. **Missing categories we should add:**
   - Satellite trails?
   - PSF ghosts?
   - Star-forming clumps?
   - Diffraction spikes from bright stars?

8. **Gold/silver tier split (115/20) - is this too skewed toward gold?**
   - Silver tier may be more realistic confusers

---

## Intended Use

After your review, we will:

1. Score all anchors and contaminants with our trained models (A1, B1, B2, B3, B4)
2. Compute:
   - Tier-A recall (with bootstrap CI)
   - Category-conditional FPR
   - NOT prevalence-weighted FPR (we don't have population estimates)
3. Compare performance across training configurations
4. Select best model for DR10 scoring

---

## GO / NO-GO Decision Required

Please provide:

### GO Criteria (proceed to evaluation)
- [ ] Tier-A anchor count is sufficient for comparative evaluation
- [ ] θ_E distribution is acceptable (or we acknowledge the limitation)
- [ ] Contaminant categories are representative enough
- [ ] Selection function is scientifically defensible

### NO-GO Criteria (revise catalogs first)
- [ ] Critical gap in anchor coverage (e.g., need θ_E > 2.0")
- [ ] Missing contaminant category that could dominate FPR
- [ ] Selection function has concerning bias

---

## Your Response Format

Please respond with:

1. **GO / NO-GO** recommendation
2. **Critical issues** (if any) that must be fixed before proceeding
3. **Advisory notes** (non-blocking but worth documenting)
4. **Suggested paper text** for reporting limitations

---

*Catalog build timestamp: 2026-02-07T07:44:05*
*Arc SNR threshold: 2.0*
*Cutout size: 80px (cropped from larger download per your earlier feedback)*

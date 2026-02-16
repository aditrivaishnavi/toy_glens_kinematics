# Anchor and Contaminant Set Creation Plan

**Date**: 2026-02-07
**Status**: Ready to execute (can do NOW while training runs)
**Effort**: 3-5 hours total

---

## Executive Summary

We need two evaluation datasets:
1. **Anchor Set**: Real confirmed lenses to measure sim-to-real transfer
2. **Contaminant Set**: Known non-lenses to measure false positive rate

**Current State**:
- Anchor set: 68 SLACS/BELLS lenses exist, but only 2.9% recall at p>0.5 (may be too faint)
- Contaminant set: 20 hard negatives exist, but need proper curation by category

---

## CRITICAL: Selection Function

### The Problem

We **cannot** evaluate on arbitrary known lenses. The anchor set must match what our model is **designed and trained** to find. Evaluating on HST-only-detectable lenses is scientifically invalid - it measures the wrong thing.

### Our Model's Selection Function

Our model is trained to find lenses that satisfy:

| Criterion | Value | Rationale |
|-----------|-------|-----------|
| **Arc visible in ground-based imaging** | arc_snr > 2 in DR10 | Can't find what you can't see |
| **Einstein radius** | 0.5" ≤ θ_E ≤ 3.0" | Training range |
| **Lens galaxy type** | LRG-like | Training uses Legacy Survey LRGs |
| **DR10 footprint** | Covered | Evaluation uses DR10 cutouts |
| **Image quality** | Typical seeing | Not exceptional conditions |

### Why SLACS/BELLS Fail This

SLACS/BELLS were discovered via **spectroscopy** (emission lines in SDSS spectra), then confirmed with **HST imaging** (0.05"/pix). Many have:
- Arcs visible only in HST (not DR10)
- θ_E < 1" (below our typical training range)
- High-z sources that are faint in ground-based

**These are NOT failures of our model** - they're outside our selection function.

### The Correct Evaluation

```
Anchor Recall = (# anchors detected above threshold) / (# anchors IN SELECTION FUNCTION)
                                                        ↑
                                            NOT all known lenses!
```

### Selection Function Implementation

```python
def in_selection_function(anchor: dict) -> bool:
    """Return True if anchor is within our model's selection function."""
    
    # 1. θ_E in training range
    if not (0.5 <= anchor['theta_e_arcsec'] <= 3.0):
        return False
    
    # 2. Arc visible in ground-based imaging
    if anchor['arc_snr_dr10'] < 2.0:
        return False  # Tier-B, not evaluable
    
    # 3. In DR10 footprint
    if not anchor['in_dr10']:
        return False
    
    # 4. Lens galaxy is LRG-like (optional filter)
    # if anchor['lens_type'] not in ['ETG', 'LRG', 'elliptical']:
    #     return False
    
    return True

# Correct recall calculation
anchors_in_sf = [a for a in all_anchors if in_selection_function(a)]
detected = [a for a in anchors_in_sf if model.predict(a) > threshold]
recall = len(detected) / len(anchors_in_sf)
```

---

## Part 1: Anchor Set

### 1.1 Current State

We have an existing anchor evaluation from `results/anchor_baseline_report.md`:

| Metric | Value |
|--------|-------|
| Total Lenses | 68 (SLACS + BELLS) |
| Recall @ p>0.5 | 2.9% (2/68) |
| Mean p_lens | 0.232 |

**Problem**: SLACS/BELLS are HST-confirmed lenses that are often too faint in ground-based DR10 imaging. They may not be fair evaluation targets.

### 1.2 Anchor Tiers (Selection Function Applied)

| Tier | Definition | In Selection Function? | Use Case |
|------|------------|------------------------|----------|
| **Tier-A** | Arc visible in DR10 (arc_snr > 2) + θ_E ≥ 0.5" | ✅ YES | **Primary evaluation metric** |
| **Tier-B** | Arc too faint OR θ_E < 0.5" | ❌ NO | Stress test / aspirational only |

**Key insight**: Only Tier-A anchors count toward recall. Tier-B anchors are documented but not used for pass/fail gating.

### 1.2.1 Expected Tier Distribution

Based on literature and our initial analysis:

| Source | Expected Tier-A | Expected Tier-B | Notes |
|--------|-----------------|-----------------|-------|
| SLACS | ~10-20% | ~80-90% | Many faint arcs |
| BELLS | ~20-30% | ~70-80% | Similar to SLACS |
| Ground-based discoveries | ~70-90% | ~10-30% | Discovered from ground = visible |
| Legacy Survey ML | ~80%+ | ~20% | Our target population |

**Implication**: We need **ground-based discovered lenses** as primary anchors.

### 1.3 Anchor Sources

| Source | N Lenses | θ_E Range | DR10 Coverage | Priority |
|--------|----------|-----------|---------------|----------|
| **SLACS** | ~100 | 0.8-2.5" | Mostly yes | Medium (many faint) |
| **BELLS** | ~25 | 0.9-1.8" | Yes | Medium (many faint) |
| **SL2S** | ~50 | 1-3" | Partial | High (brighter arcs) |
| **Legacy Survey ML** | ~100+ | 0.5-3" | Yes | **Highest** (ground-based) |
| **DESI Lens Search** | ~50+ | 0.5-2" | Yes | **Highest** (ground-based) |

### 1.4 Action Plan for Anchors

#### Step 1: Validate Existing Set (30 min)
```bash
# Check if existing anchor CSV exists
ls -la /path/to/anchors/tier_a_anchors.csv

# Run validation script
cd dark_halo_scope
python planb/phase0_foundation/validate_anchors.py \
    --anchor-csv anchors/tier_a_anchors.csv \
    --output-json results/anchor_validation.json
```

#### Step 2: Classify into Tier-A/B (1 hour)

Run the arc visibility classifier on existing anchors:

```python
# In scripts/phase3_build_tier_a_anchors.py
# Already implemented - classifies by arc_visibility_snr

# Arc visible in DR10 → Tier-A (use for primary eval)
# Arc too faint → Tier-B (stress test only)
```

#### Step 3: Add Ground-Based Discovered Lenses (1-2 hours)

**Priority sources** (lenses discovered FROM ground-based imaging):

1. **Huang+2020 Legacy Survey ML Lenses**
   - Paper: "Finding Strong Gravitational Lenses in the DESI DECam Legacy Survey"
   - ~100 candidates, many Grade A/B
   - Direct download from paper appendix

2. **Jacobs+2019 Southern Sky Survey**
   - DES/KiDS discovered lenses
   - Ground-based discovered = should be visible in DR10

3. **DESI Lens Search (internal)**
   - Check if we have access to internal catalogs

**Action**:
```python
# Create script to download and merge anchor catalogs
# planb/scripts/build_anchor_catalog.py

SOURCES = {
    "slacs": "existing",  # Already have
    "bells": "existing",  # Already have
    "huang2020": "download",  # From paper appendix
    "jacobs2019": "download",  # From VizieR
    "sl2s": "download",  # From VizieR
}
```

#### Step 4: Create Final Anchor CSV (30 min)

Required columns:
```
name, ra, dec, theta_e_arcsec, source, tier
```

Example:
```csv
name,ra,dec,theta_e_arcsec,source,tier
SDSSJ0912+0029,138.02,0.4894,1.63,SLACS,B
J1430+4105,217.51,41.08,1.52,SLACS,B
DESJ0408-5354,62.09,-53.90,1.85,Huang2020,A
```

#### Step 5: Download Cutouts for New Anchors (1 hour)

```python
# Use Legacy Survey cutout service
for anchor in new_anchors:
    url = f"https://www.legacysurvey.org/viewer/fits-cutout?ra={ra}&dec={dec}&layer=ls-dr10&pixscale=0.262&bands=grz&size=64"
    # Download and save as FITS
```

### 1.5 Anchor Set Deliverables

| Deliverable | Path | Format |
|-------------|------|--------|
| Anchor catalog | `anchors/tier_a_anchors.csv` | CSV |
| Anchor cutouts | `data/anchor_cutouts/` | FITS |
| Validation results | `results/anchor_validation.json` | JSON |

---

## Part 2: Contaminant Set

### 2.1 Current State

From `anchor_baseline_report.md`:
- 20 hard negatives exist
- 1 false positive at p>0.5 (Merger002, p=0.99)
- Categories not properly labeled

### 2.2 Contaminant Selection Function

Contaminants are NOT "random non-lenses". They are objects that:

1. **Look like lenses** to a human or model (arc-like features)
2. **Would appear in our search** (LRG-like central galaxy OR lens-like morphology)
3. **Are definitively NOT lenses** (no spectroscopic confirmation, known classification)

The goal is to measure **false positive rate on realistic confusers**, not on random galaxies (which would trivially have ~0% FPR).

```python
def is_valid_contaminant(obj: dict) -> bool:
    """Return True if object is a valid contaminant for our evaluation."""
    
    # Must have arc-like or lens-like morphology
    if obj['category'] not in ['ring', 'spiral', 'merger', 'spike']:
        return False
    
    # Must NOT be a known lens
    if obj['is_confirmed_lens']:
        return False
    
    # Must be in DR10 footprint
    if not obj['in_dr10']:
        return False
    
    return True
```

### 2.3 Required Categories

| Category | Minimum | Description | Why Challenging |
|----------|---------|-------------|-----------------|
| **Ring Galaxies** | 50 | Collisional ring galaxies | Circular arc-like structure |
| **Face-on Spirals** | 50 | Spiral galaxies viewed face-on | Spiral arms look like arcs |
| **Mergers** | 30 | Interacting/merging pairs | Tidal features mimic arcs |
| **Diffraction Spikes** | 20 | Bright star artifacts | Linear features |

### 2.4 Contaminant Sources

#### Ring Galaxies
- **Galaxy Zoo**: `p_ring > 0.5` in GZ2 catalog
- **SDSS Ring Galaxy Catalog**: Few+2012
- **Madore+2009**: Catalog of ring galaxies

#### Spirals
- **Galaxy Zoo**: `p_spiral > 0.8, p_edge_on < 0.3`
- **SDSS**: Morphology classifications

#### Mergers
- **Galaxy Zoo Mergers**: Darg+2010
- **SDSS Pair Catalog**: Ellison+2011

#### Diffraction Spikes
- **Gaia DR3 Bright Stars**: Cross-match with DR10, select with diffraction patterns

### 2.5 Action Plan for Contaminants

#### Step 1: Query Galaxy Zoo (1 hour)

```python
# planb/scripts/build_contaminant_catalog.py

from astroquery.vizier import Vizier

# Galaxy Zoo 2 catalog
gz2 = Vizier.query_constraints(catalog="J/MNRAS/435/2835/gz2")[0]

# Ring galaxies
rings = gz2[gz2['P_ring'] > 0.5]
print(f"Ring candidates: {len(rings)}")

# Spirals (face-on)
spirals = gz2[(gz2['P_spiral'] > 0.8) & (gz2['P_edge_on'] < 0.3)]
print(f"Spiral candidates: {len(spirals)}")
```

#### Step 2: Query Merger Catalogs (30 min)

```python
# Darg+2010 Galaxy Zoo Mergers
mergers = Vizier.query_constraints(catalog="J/MNRAS/401/1552/table2")[0]

# Filter for high confidence
mergers_high = mergers[mergers['Pmerg'] > 0.6]
```

#### Step 3: Cross-Match with DR10 Footprint (30 min)

```python
# Ensure all contaminants have DR10 coverage
def check_dr10_coverage(ra, dec):
    # Query Legacy Survey
    # Return True if covered
    ...

contaminants = contaminants[contaminants.apply(
    lambda r: check_dr10_coverage(r['ra'], r['dec']), axis=1
)]
```

#### Step 4: Download Cutouts (1 hour)

```python
# Download 64x64 cutouts from Legacy Survey
for contam in contaminants:
    url = f"https://www.legacysurvey.org/viewer/fits-cutout?..."
    # Save to data/contaminant_cutouts/{category}/{name}.fits
```

#### Step 5: Visual Verification (Optional, 30 min)

Quick visual scan to remove obvious non-examples:
- Ring galaxies that don't look ring-like
- Spirals that are edge-on
- Mergers that are too faint

#### Step 6: Create Final CSV (15 min)

```csv
name,ra,dec,category
Ring_GZ_001,182.45,12.34,ring
Ring_GZ_002,185.12,14.56,ring
Spiral_GZ_001,190.23,25.67,spiral
Merger_GZ_001,200.45,30.12,merger
```

### 2.6 Contaminant Set Deliverables

| Deliverable | Path | Format |
|-------------|------|--------|
| Contaminant catalog | `contaminants/contaminant_catalog.csv` | CSV |
| Contaminant cutouts | `data/contaminant_cutouts/` | FITS |
| Validation results | `results/contaminant_validation.json` | JSON |

---

## Part 3: Validation

### 3.1 Run Validation Scripts

```bash
# Validate anchors
python planb/phase0_foundation/validate_anchors.py \
    --anchor-csv anchors/tier_a_anchors.csv

# Validate contaminants
python planb/phase0_foundation/validate_contaminants.py \
    --contaminant-csv contaminants/contaminant_catalog.csv \
    --anchor-csv anchors/tier_a_anchors.csv
```

### 3.2 Validation Checks

**Anchor Checks**:
- [ ] ≥30 anchors total
- [ ] Required columns present
- [ ] No duplicates
- [ ] All θ_E ≥ 0.5"
- [ ] All have DR10 coverage

**Contaminant Checks**:
- [ ] ≥50 ring galaxies
- [ ] ≥50 spirals
- [ ] ≥30 mergers
- [ ] No overlap with anchors
- [ ] All have DR10 coverage

---

## Part 4: Usage in Evaluation

### 4.1 Anchor Recall Metric

```python
def compute_anchor_recall(model, anchor_cutouts, threshold=0.5):
    """Fraction of real lenses detected above threshold."""
    scores = model.predict(anchor_cutouts)
    recall = (scores > threshold).mean()
    return recall
```

**Target**: Anchor recall > 50% at p > 0.5 for Tier-A anchors

### 4.2 Contaminant FPR Metric

```python
def compute_contaminant_fpr(model, contaminant_cutouts, threshold=0.5):
    """Fraction of contaminants incorrectly classified as lenses."""
    scores = model.predict(contaminant_cutouts)
    fpr = (scores > threshold).mean()
    return fpr

# Stratify by category
for category in ['ring', 'spiral', 'merger']:
    subset = contaminants[contaminants.category == category]
    fpr = compute_contaminant_fpr(model, subset)
    print(f"{category} FPR: {fpr:.1%}")
```

**Target**: Contaminant FPR < 20% at p > 0.5

### 4.3 Proper Reporting (For Paper)

The selection function must be explicitly stated in the paper. Example text:

> **Selection Function**: We evaluate on anchors satisfying: (1) Einstein radius 0.5" ≤ θ_E ≤ 3.0", (2) arc visibility SNR > 2 in DR10 r-band, (3) within DR10 footprint. Of 68 SLACS/BELLS lenses, only N=XX satisfy these criteria ("Tier-A"). The remaining N=YY are classified as Tier-B (arc too faint in ground-based imaging) and are excluded from primary evaluation but reported separately.

**Report both**:
1. **Primary metric**: Recall on Tier-A anchors (within selection function)
2. **Supplementary**: Recall on Tier-B anchors (aspirational, not gated)

**Why this matters**:
- Prevents unfair comparison with methods designed for different populations
- Clearly communicates what the model is designed to find
- Scientifically honest about limitations

---

## Part 5: Timeline

| Step | Task | Time | Can Parallelize |
|------|------|------|-----------------|
| 1 | Validate existing anchors | 30 min | ✅ Now |
| 2 | Classify Tier-A/B | 30 min | ✅ Now |
| 3 | Download new anchor catalogs | 1 hr | ✅ Now |
| 4 | Query contaminant catalogs | 1 hr | ✅ Now |
| 5 | Cross-match with DR10 | 30 min | After 3,4 |
| 6 | Download cutouts | 1-2 hrs | After 5 |
| 7 | Run validation | 15 min | After 6 |

**Total**: 4-6 hours
**Can start**: NOW (while training runs)

---

## Part 6: Scripts to Create

| Script | Purpose | Status |
|--------|---------|--------|
| `planb/scripts/build_anchor_catalog.py` | Download and merge anchor sources | ⏳ To create |
| `planb/scripts/build_contaminant_catalog.py` | Query and build contaminant set | ⏳ To create |
| `planb/scripts/download_cutouts.py` | Download FITS cutouts from Legacy Survey | ⏳ To create |
| `planb/scripts/evaluate_on_real_data.py` | Run model on anchors/contaminants | ⏳ To create |

---

## Appendix: Existing Files

| File | Purpose | Status |
|------|---------|--------|
| `scripts/phase3_build_tier_a_anchors.py` | Classify anchors by arc visibility | ✅ Exists |
| `scripts/anchor_eval.py` | Evaluate model on anchors | ✅ Exists |
| `results/anchor_baseline_report.md` | Previous anchor evaluation | ✅ Exists |
| `planb/phase0_foundation/validate_anchors.py` | Validate anchor CSV | ✅ Exists |
| `planb/phase0_foundation/validate_contaminants.py` | Validate contaminant CSV | ✅ Exists |

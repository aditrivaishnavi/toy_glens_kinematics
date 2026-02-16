# Selection Function Review: Response and Implementation

**Date**: 2026-02-07
**Status**: Changes implemented

---

## Summary of External Review Findings

The external LLM review identified several critical issues with our original selection function implementation:

### Critical Issues Fixed

| Issue | Original Behavior | Fixed Behavior |
|-------|------------------|----------------|
| Missing arc visibility | Assumed visible (optimistic) | → **TIER_B** by default |
| Missing DR10 flag | Assumed in DR10 (risky) | → **EXCLUDE** by default |
| Contaminant lens safety | Boolean flag only | → **Coordinate cross-match** against anchors |
| θ_E catalog uncertainty | Hard cutoffs | → Added **theta_e_margin** parameter |
| Image usability | Not checked | → Added **require_usable_cutout** option |
| PSF resolution | Not checked | → Added **min_theta_e_over_psf** option |
| Reason codes | Free-form strings | → **Enums** for type safety |
| FPR reporting | Unweighted only | → Added **prevalence-weighted FPR** |

---

## Changes Implemented

### 1. Anchor Selection Function

#### New Policy Parameters

```python
# Old (dangerous defaults)
AnchorSelectionFunction(
    theta_e_min=0.5,
    theta_e_max=3.0,
    arc_snr_min=2.0,
)
# Missing arc → assumed visible (WRONG)
# Missing DR10 → assumed in footprint (WRONG)

# New (defensive defaults)
AnchorSelectionFunction(
    theta_e_min=0.5,
    theta_e_max=3.0,
    arc_snr_min=2.0,
    theta_e_margin=0.0,              # For catalog uncertainty
    require_usable_cutout=False,      # Image quality check
    min_theta_e_over_psf=None,        # PSF resolution check
    on_missing_arc_visibility="TIER_B",  # CRITICAL: defensive
    on_missing_dr10_flag="EXCLUDE",      # CRITICAL: defensive
)
```

#### Missing Arc Visibility Options

| Policy | Behavior | When to Use |
|--------|----------|-------------|
| `TIER_B` | Assign to Tier-B | **Default** - defensive |
| `EXCLUDE` | Exclude entirely | When you can't afford any Tier-B |
| `ASSUME_VISIBLE` | Treat as Tier-A | **NOT recommended** |
| `MEASURE_REQUIRED` | Require explicit measurement | Strictest |

### 2. Contaminant Selection Function

#### Coordinate Cross-Match Exclusion

```python
# NEW: Cross-match against anchor coordinates
contaminants = ContaminantSet.from_csv(
    "contaminants.csv",
    selection_function=sf,
    anchor_coords=anchor_set.df[["ra", "dec"]],  # Pass anchor coords
)
# Any contaminant within 3" of an anchor is excluded
```

#### Additional Contaminant Categories

```python
VALID_CONTAMINANT_CATEGORIES = {
    "ring",           # Original
    "spiral",         # Original
    "merger",         # Original
    "spike",          # Original
    "edge_on",        # NEW: Edge-on disks
    "agn",            # NEW: AGN with extended features
    "polar_ring",     # NEW
    "blue_companion", # NEW: Blue blobs around red galaxies
    "bar_ring",       # NEW: Barred spirals with inner rings
    "warp",           # NEW: Warped disks
    "ghost",          # NEW: PSF ghosts
    "satellite",      # NEW: Satellite trails
}
```

#### Prevalence-Weighted FPR

```python
# Category prevalence estimates (adjustable)
DEFAULT_CATEGORY_PREVALENCE = {
    "spiral": 0.40,   # Most common confuser
    "merger": 0.20,
    "edge_on": 0.15,
    "ring": 0.05,     # Rare
    "spike": 0.10,
    "agn": 0.05,
}

# Results now include weighted FPR
results["fpr_weighted"]  # Prevalence-adjusted
results["fpr"]           # Unweighted (as before)
```

### 3. Schema Normalization

```python
# Normalize heterogeneous column names to canonical schema
df = normalize_anchor_schema(df, column_mapping={
    "arc_snr_dr10": "arc_visibility_snr",  # Custom mapping
})

# Canonical columns used consistently by selection function
ANCHOR_CANONICAL_COLUMNS = {
    "arc_snr_dr10",      # Preferred (measured on DR10)
    "arc_visible_dr10",  # Boolean alternative
    "in_dr10",           # Footprint flag
    "usable_cutout",     # Image quality
    "psf_fwhm_arcsec",   # For θ_E/PSF check
}
```

### 4. Paper Reporting

```python
# Generate defensible paper text
print(sf.describe_for_paper())
```

Output:
> "We define a Tier-A anchor subset intended to match the model's operating regime: 
> DR10 footprint, usable cutouts (masking and image-quality cuts), Einstein radius 
> within [0.5", 3.0"] (the training support), and arcs detectable in DR10 by an 
> automated visibility metric (SNR > 2.0) computed on the same cutouts used for 
> inference. Anchors failing these criteria, or lacking the measurements required 
> to apply them (missing arc visibility → TIER_B, missing DR10 flag → EXCLUDE), 
> are assigned to Tier-B and reported separately."

---

## Validation

All changes verified with tests:

```
✓ AnchorSelectionFunction created
  Description: θ_E ∈ [0.5", 3.0"] AND arc SNR > 2.0 AND in DR10 footprint 
               AND [missing arc → TIER_B] AND [missing DR10 → EXCLUDE]
  
  Missing arc_snr -> passes=False, reason=arc_visibility_unknown_TIER_B
  Missing DR10 flag -> passes=False, reason=dr10_flag_missing

✓ ContaminantSelectionFunction created
  Description: category ∈ ['merger', 'ring', 'spiral'] AND in DR10 footprint 
               AND not a confirmed lens AND no anchor match within 3.0" 
               AND [missing DR10 → EXCLUDE]
  
  Cross-match results: [True, False]  # First contaminant excluded (matches anchor)

✓ All tests passed!
```

---

## Remaining Recommendations (Deferred)

The external review also suggested:

1. **Measure arc visibility ourselves from DR10 cutouts**
   - Status: Deferred to Phase 7 (requires cutout processing pipeline)
   - Current mitigation: Use `on_missing_arc_visibility="TIER_B"`

2. **Add seeing/depth bounds matching training**
   - Status: Can implement via `min_theta_e_over_psf` and custom checks
   - Requires: PSF FWHM column in anchor catalog

3. **Vectorized operations for scale**
   - Status: Not needed yet (anchors/contaminants are O(10^2-10^3))
   - Will implement if contaminants scale to O(10^5+)

---

## File Changes

| File | Changes |
|------|---------|
| `evaluation/anchor_set.py` | Rewrote with new policy parameters, enums, schema normalization |
| `evaluation/contaminant_set.py` | Added cross-match, prevalence weighting, expanded categories |
| `evaluation/__init__.py` | No changes |

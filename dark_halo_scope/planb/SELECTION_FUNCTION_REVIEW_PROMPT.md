# External LLM Review Request: Selection Functions for Evaluation Sets

**Date**: 2026-02-07
**Purpose**: Review selection function design for anchor and contaminant evaluation sets
**Requested**: Critique logic, identify edge cases, suggest improvements

---

## Context

We are building a gravitational lens finder using a CNN trained on simulated data. To evaluate sim-to-real transfer, we need:

1. **Anchor Set**: Real confirmed lenses to measure recall
2. **Contaminant Set**: Known non-lenses to measure false positive rate

**Critical Insight**: We cannot fairly evaluate on all known lenses. We must define a **selection function** that matches what our model is designed to detect. Lenses outside this function are not model failures.

---

## The Selection Function Concept

### For Anchors

Our model is trained to find lenses with:
- Einstein radius 0.5" ≤ θ_E ≤ 3.0" (training range)
- Arcs visible in ground-based DR10 imaging (arc_snr > 2)
- Within DR10 footprint

**Problem**: Many spectroscopically confirmed lenses (e.g., SLACS/BELLS) were discovered via spectroscopy and confirmed with HST (0.05"/pixel). Their arcs are often invisible in ground-based imaging (0.26"/pixel, ~1" seeing).

**Solution**: Classify anchors into:
- **Tier-A**: Within selection function → primary evaluation metric
- **Tier-B**: Outside selection function → documented but not gated

### For Contaminants

Contaminants are NOT random galaxies. They must be:
- Morphologically similar to lenses (arc-like features)
- Would appear in our search population
- Definitively NOT lenses

**Categories**: Ring galaxies, face-on spirals, mergers, diffraction spikes

---

## Code to Review

### File 1: `anchor_set.py`

```python
"""
Anchor Set: Real lens evaluation with selection function.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnchorSelectionFunction:
    """
    Selection function defining what the model is designed to detect.
    
    An anchor passes the selection function if ALL criteria are met.
    
    Attributes:
        theta_e_min: Minimum Einstein radius in arcsec (default: 0.5")
        theta_e_max: Maximum Einstein radius in arcsec (default: 3.0")
        arc_snr_min: Minimum arc visibility SNR in ground-based imaging (default: 2.0)
        require_dr10_coverage: Whether anchor must be in DR10 footprint (default: True)
        lens_types: Allowed lens galaxy types, or None for any (default: None)
    """
    
    # Einstein radius bounds
    theta_e_min: float = 0.5  # arcsec
    theta_e_max: float = 3.0  # arcsec
    
    # Arc visibility threshold
    arc_snr_min: float = 2.0  # SNR in ground-based imaging
    
    # Footprint requirement
    require_dr10_coverage: bool = True
    
    # Lens galaxy type filter (None = any type allowed)
    lens_types: Optional[List[str]] = None
    
    def apply(self, row: pd.Series) -> Tuple[bool, str]:
        """
        Apply selection function to a single anchor.
        
        Returns:
            Tuple of (passes: bool, reason: str)
        """
        # Check theta_e lower bound
        theta_e = row.get("theta_e_arcsec", np.nan)
        if pd.isna(theta_e):
            return False, "theta_e_missing"
        if theta_e < self.theta_e_min:
            return False, f"theta_e_below_min ({theta_e:.2f} < {self.theta_e_min})"
        if theta_e > self.theta_e_max:
            return False, f"theta_e_above_max ({theta_e:.2f} > {self.theta_e_max})"
        
        # Check arc visibility (if available)
        arc_snr = row.get("arc_snr", row.get("arc_visibility_snr", np.nan))
        if not pd.isna(arc_snr):
            if arc_snr < self.arc_snr_min:
                return False, f"arc_snr_below_min ({arc_snr:.2f} < {self.arc_snr_min})"
        else:
            # If arc_snr not available, check arc_visible boolean
            arc_visible = row.get("arc_visible", None)
            if arc_visible is not None and not arc_visible:
                return False, "arc_not_visible"
            # If neither available, we assume visible (conservative)
        
        # Check DR10 coverage
        if self.require_dr10_coverage:
            in_dr10 = row.get("in_dr10", row.get("has_dr10_coverage", True))
            if not in_dr10:
                return False, "not_in_dr10"
        
        # Check lens type (if filter specified)
        if self.lens_types is not None:
            lens_type = row.get("lens_type", row.get("morphology", None))
            if lens_type is not None and lens_type not in self.lens_types:
                return False, f"lens_type_excluded ({lens_type})"
        
        return True, "TIER_A"
```

### File 2: `contaminant_set.py`

```python
"""
Contaminant Set: False positive evaluation with realistic confusers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
import numpy as np
import pandas as pd

VALID_CONTAMINANT_CATEGORIES = {
    "ring",     # Collisional ring galaxies
    "spiral",   # Face-on spiral galaxies
    "merger",   # Interacting/merging galaxies
    "spike",    # Diffraction spikes from bright stars
    "edge_on",  # Edge-on disks (can mimic arcs)
    "agn",      # AGN/QSO with extended features
}

MIN_COUNTS_PER_CATEGORY = {
    "ring": 50,
    "spiral": 50,
    "merger": 30,
    "spike": 20,
    "edge_on": 20,
    "agn": 10,
}


@dataclass
class ContaminantSelectionFunction:
    """
    Selection function defining what counts as a valid contaminant.
    
    Contaminants must be realistic confusers - objects that COULD be
    mistaken for lenses based on morphology.
    """
    
    # Allowed categories
    valid_categories: Set[str] = field(
        default_factory=lambda: {"ring", "spiral", "merger", "spike"}
    )
    
    # Footprint requirement
    require_dr10_coverage: bool = True
    
    # Safety check: never include confirmed lenses
    exclude_confirmed_lenses: bool = True
    
    # Size filter (None = no filter)
    min_size_arcsec: Optional[float] = None
    max_size_arcsec: Optional[float] = None
    
    # Redshift filter (None = no filter)
    max_redshift: Optional[float] = None
    
    def apply(self, row: pd.Series) -> Tuple[bool, str]:
        """
        Apply selection function to a single contaminant.
        
        Returns:
            Tuple of (valid: bool, reason: str)
        """
        # Check category
        category = row.get("category", "").lower()
        if category not in self.valid_categories:
            return False, f"invalid_category ({category})"
        
        # Check not a confirmed lens
        if self.exclude_confirmed_lenses:
            is_lens = row.get("is_confirmed_lens", False)
            if is_lens:
                return False, "is_confirmed_lens"
        
        # Check DR10 coverage
        if self.require_dr10_coverage:
            in_dr10 = row.get("in_dr10", row.get("has_dr10_coverage", True))
            if not in_dr10:
                return False, "not_in_dr10"
        
        # Check size bounds (if specified)
        size = row.get("size_arcsec", row.get("r_eff_arcsec", None))
        if size is not None and self.min_size_arcsec is not None:
            if size < self.min_size_arcsec:
                return False, f"too_small ({size:.2f}\")"
        if size is not None and self.max_size_arcsec is not None:
            if size > self.max_size_arcsec:
                return False, f"too_large ({size:.2f}\")"
        
        # Check redshift (if specified)
        z = row.get("redshift", row.get("z", None))
        if z is not None and self.max_redshift is not None:
            if z > self.max_redshift:
                return False, f"z_too_high ({z:.2f})"
        
        return True, "VALID"
```

---

## Review Questions

### 1. Selection Function Logic

**Q1.1**: Is the anchor selection function logic correct? Are there edge cases we're missing?

**Q1.2**: For the `arc_snr` check - if `arc_snr` is not available, we currently assume the arc is visible (conservative default). Is this the right approach, or should we be more strict?

**Q1.3**: Should we add additional criteria to the anchor selection function? For example:
- Source brightness/magnitude?
- Lens galaxy stellar mass or luminosity?
- Image quality / seeing constraints?

### 2. Contaminant Categories

**Q2.1**: Are the contaminant categories comprehensive? What morphological confusers are we missing?

**Q2.2**: The minimum counts per category (50 rings, 50 spirals, 30 mergers, etc.) - are these sufficient for statistical validity?

**Q2.3**: Should we weight false positive rates by category based on their expected frequency in a real search? (e.g., spirals are more common than ring galaxies)

### 3. Threshold Choices

**Q3.1**: `theta_e_min = 0.5"` and `theta_e_max = 3.0"` - are these appropriate bounds?
- Our training data uses θ_E from 0.5" to ~2.5"
- Should the evaluation bounds match exactly?

**Q3.2**: `arc_snr_min = 2.0` - is this the right threshold for "visible in ground-based imaging"?
- Too strict might exclude borderline cases
- Too lenient defeats the purpose

### 4. Scientific Rigor

**Q4.1**: How should we handle the case where `arc_snr` is not measured for some anchors? Options:
1. Exclude them entirely (lose data)
2. Assume visible (optimistic)
3. Assume not visible (conservative)
4. Measure arc_snr ourselves from cutouts

**Q4.2**: For paper reporting, we plan to state:
> "We evaluate on N_A Tier-A anchors satisfying our selection function (θ_E ∈ [0.5", 3.0"], arc SNR > 2 in DR10). An additional N_B Tier-B anchors with fainter arcs are reported separately."

Is this scientifically defensible? Any suggested improvements to the wording?

### 5. Implementation Concerns

**Q5.1**: The code uses `row.get("col", default)` extensively with fallback column names. Is this pattern robust, or should we enforce strict column naming?

**Q5.2**: The `apply()` method returns a reason string for exclusion. Should we use an enum instead for type safety?

**Q5.3**: Any performance concerns with applying row-by-row vs. vectorized operations?

---

## Expected Output

Please provide:

1. **Bugs or Logic Errors**: Any issues in the current code
2. **Edge Cases**: Scenarios not handled correctly
3. **Suggested Improvements**: Specific code changes recommended
4. **Threshold Recommendations**: If our defaults seem off
5. **Additional Criteria**: Any selection function criteria we should add
6. **Scientific Concerns**: Any issues with the overall approach

---

## Appendix: Training Data Characteristics

For reference, our training data has:

| Property | Range | Distribution |
|----------|-------|--------------|
| θ_E | 0.5" - 2.5" | Uniform in log |
| Source type | Sersic | n=1-4 |
| Lens type | LRG | From DR10 catalog |
| Pixel scale | 0.262"/pix | DECaLS |
| Stamp size | 64x64 pixels | 16.8" x 16.8" |
| Bands | g, r, z | 3-channel |
| Arc SNR (training) | 2 - 50 | Varies with θ_E |

---

## Appendix: Known Anchor Sources

| Source | N Lenses | θ_E Range | Notes |
|--------|----------|-----------|-------|
| SLACS | ~100 | 0.8-2.5" | Spectroscopic discovery, HST confirmed, often faint in ground |
| BELLS | ~25 | 0.9-1.8" | Similar to SLACS |
| SL2S | ~50 | 1-3" | Ground-based discovery, brighter |
| Legacy Survey ML | ~100+ | 0.5-3" | Ground-based discovery, should be Tier-A |

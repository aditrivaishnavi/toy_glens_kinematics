"""
Contaminant Set: False positive evaluation with realistic confusers.

A "contaminant" is an object that is definitively NOT a lens, but has
morphological features that could confuse a lens finder.

CRITICAL CONCEPT: Contaminant Selection Function
================================================
Contaminants are NOT random galaxies. They are objects that:

1. Look like lenses (arc-like features)
2. Would appear in our search (similar to candidate population)
3. Are definitively NOT lenses (confirmed classification)

The goal is to measure false positive rate on REALISTIC confusers,
not on trivially easy cases (which would give ~0% FPR).

UPDATED based on external review (2026-02-07):
- Added coordinate cross-match exclusion against anchors
- Missing DR10 coverage -> exclude by default
- Added more contaminant categories
- Added weighting support for prevalence-adjusted FPR
- Added explicit missing data policies

Categories:
- Ring galaxies: Collisional rings with circular arc-like structure
- Face-on spirals: Spiral arms can mimic lensed arcs
- Mergers: Tidal features can look like gravitational arcs
- Diffraction spikes: Stellar artifacts
- Edge-on disks: Can mimic arcs
- AGN/QSO: Extended features
- Polar rings: Ring-like structure at angle
- Blue companions: Blue blobs around red galaxies

Usage:
    from planb.evaluation import ContaminantSet, ContaminantSelectionFunction
    
    # Define what counts as a valid contaminant
    sf = ContaminantSelectionFunction(
        valid_categories={"ring", "spiral", "merger", "spike"},
        require_dr10_coverage=True,
    )
    
    # Load contaminants with anchor cross-match exclusion
    contaminants = ContaminantSet.from_csv(
        "contaminants.csv", 
        selection_function=sf,
        anchor_coords=anchor_set.df[["ra", "dec"]],  # Exclude anchors
    )
    
    # Evaluate
    results = contaminants.evaluate(model, threshold=0.5)
    print(f"Overall FPR: {results['fpr']:.1%}")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# VALID CATEGORIES (EXPANDED)
# =============================================================================

VALID_CONTAMINANT_CATEGORIES = {
    "ring",         # Collisional ring galaxies
    "spiral",       # Face-on spiral galaxies
    "merger",       # Interacting/merging galaxies
    "spike",        # Diffraction spikes from bright stars
    "edge_on",      # Edge-on disks (can mimic arcs)
    "agn",          # AGN/QSO with extended features
    "polar_ring",   # Polar ring galaxies
    "blue_companion",  # Blue companions around red galaxies
    "bar_ring",     # Barred spirals with inner rings
    "warp",         # Edge-on disks with warps/tidal streams
    "ghost",        # PSF ghosts and scattered light
    "satellite",    # Satellite trails
}

# Minimum counts per category for statistical validity
MIN_COUNTS_PER_CATEGORY = {
    "ring": 50,
    "spiral": 50,
    "merger": 30,
    "spike": 20,
    "edge_on": 20,
    "agn": 10,
    "polar_ring": 10,
    "blue_companion": 20,
    "bar_ring": 20,
    "warp": 10,
    "ghost": 10,
    "satellite": 10,
}

# Expected prevalence in search population (for weighted FPR)
# These are rough estimates - adjust based on actual survey data
DEFAULT_CATEGORY_PREVALENCE = {
    "spiral": 0.40,     # Most common confuser
    "merger": 0.20,     # Common
    "edge_on": 0.15,    # Fairly common
    "ring": 0.05,       # Rare
    "spike": 0.10,      # Depends on bright star density
    "agn": 0.05,        # Rare
    "other": 0.05,      # Catch-all
}


# =============================================================================
# ENUMS FOR TYPE SAFETY
# =============================================================================

class ContaminantExclusionReason(str, Enum):
    """Standardized exclusion reasons."""
    VALID = "VALID"
    INVALID_CATEGORY = "invalid_category"
    IS_CONFIRMED_LENS = "is_confirmed_lens"
    MATCHES_ANCHOR_COORDS = "matches_anchor_coords"
    NOT_IN_DR10 = "not_in_dr10"
    DR10_FLAG_MISSING = "dr10_flag_missing"
    TOO_SMALL = "too_small"
    TOO_LARGE = "too_large"
    Z_TOO_HIGH = "z_too_high"


class MissingDR10Policy(str, Enum):
    """Policy for handling missing DR10 coverage flag."""
    EXCLUDE = "EXCLUDE"
    ASSUME_IN_DR10 = "ASSUME_IN_DR10"


# =============================================================================
# COORDINATE CROSS-MATCH
# =============================================================================

def cross_match_coords(
    contam_df: pd.DataFrame,
    anchor_coords: pd.DataFrame,
    match_radius_arcsec: float = 3.0,
) -> pd.Series:
    """
    Check if contaminants match any anchor coordinates.
    
    This is a critical safety check to ensure we don't count
    actual lenses as contaminants.
    
    Args:
        contam_df: DataFrame with contaminant ra, dec
        anchor_coords: DataFrame with anchor ra, dec
        match_radius_arcsec: Match tolerance in arcsec
    
    Returns:
        Boolean Series - True if contaminant matches an anchor
    """
    if anchor_coords is None or len(anchor_coords) == 0:
        return pd.Series([False] * len(contam_df), index=contam_df.index)
    
    # Convert to radians for spherical matching
    deg2rad = np.pi / 180.0
    match_rad = match_radius_arcsec / 3600.0 * deg2rad
    
    contam_ra = contam_df["ra"].values * deg2rad
    contam_dec = contam_df["dec"].values * deg2rad
    anchor_ra = anchor_coords["ra"].values * deg2rad
    anchor_dec = anchor_coords["dec"].values * deg2rad
    
    matches = []
    for i in range(len(contam_df)):
        # Angular separation using Haversine formula
        dra = anchor_ra - contam_ra[i]
        ddec = anchor_dec - contam_dec[i]
        
        a = np.sin(ddec/2)**2 + np.cos(contam_dec[i]) * np.cos(anchor_dec) * np.sin(dra/2)**2
        sep = 2 * np.arcsin(np.sqrt(a))
        
        matches.append(np.any(sep < match_rad))
    
    return pd.Series(matches, index=contam_df.index)


# =============================================================================
# CANONICAL SCHEMA
# =============================================================================

def normalize_contaminant_schema(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Normalize contaminant DataFrame to canonical schema.
    """
    df = df.copy()
    
    # Default mappings
    default_mapping = {
        "in_dr10": ["has_dr10_coverage", "in_footprint"],
        "size_arcsec": ["r_eff_arcsec", "r_eff", "radius"],
        "is_confirmed_lens": ["is_lens", "lens_flag"],
    }
    
    if column_mapping:
        for canonical, source in column_mapping.items():
            if source in df.columns:
                df[canonical] = df[source]
    
    for canonical, alternatives in default_mapping.items():
        if canonical not in df.columns:
            for alt in alternatives:
                if alt in df.columns:
                    df[canonical] = df[alt]
                    break
    
    # Normalize category to lowercase
    if "category" in df.columns:
        df["category"] = df["category"].str.lower().str.strip()
    
    return df


# =============================================================================
# SELECTION FUNCTION
# =============================================================================

@dataclass
class ContaminantSelectionFunction:
    """
    Selection function defining what counts as a valid contaminant.
    
    Contaminants must be realistic confusers - objects that COULD be
    mistaken for lenses based on morphology.
    
    Attributes:
        valid_categories: Set of allowed contaminant categories
        require_dr10_coverage: Whether contaminant must be in DR10 footprint
        exclude_confirmed_lenses: Whether to exclude any confirmed lenses
        exclude_anchor_matches: Whether to exclude objects matching anchor coords
        anchor_match_radius_arcsec: Match tolerance for anchor exclusion
        min_size_arcsec: Minimum angular size (to exclude point sources)
        max_size_arcsec: Maximum angular size
        max_redshift: Maximum redshift (if known)
        
        on_missing_dr10_flag: Policy for missing DR10 flag (default: EXCLUDE)
    """
    
    # Allowed categories
    valid_categories: Set[str] = field(
        default_factory=lambda: {"ring", "spiral", "merger", "spike", "edge_on"}
    )
    
    # Footprint requirement
    require_dr10_coverage: bool = True
    
    # Safety checks
    exclude_confirmed_lenses: bool = True
    exclude_anchor_matches: bool = True  # NEW: cross-match exclusion
    anchor_match_radius_arcsec: float = 3.0
    
    # Size filter (None = no filter)
    min_size_arcsec: Optional[float] = None
    max_size_arcsec: Optional[float] = None
    
    # Redshift filter (None = no filter)
    max_redshift: Optional[float] = None
    
    # Missing data policy
    on_missing_dr10_flag: str = "EXCLUDE"  # EXCLUDE or ASSUME_IN_DR10
    
    def apply(
        self, 
        row: pd.Series, 
        matches_anchor: bool = False,
    ) -> Tuple[bool, str]:
        """
        Apply selection function to a single contaminant.
        
        Args:
            row: pandas Series with contaminant properties
            matches_anchor: Whether this contaminant matches anchor coordinates
        
        Returns:
            Tuple of (valid: bool, reason: str)
        """
        # Check anchor cross-match first (critical safety check)
        if self.exclude_anchor_matches and matches_anchor:
            return False, ContaminantExclusionReason.MATCHES_ANCHOR_COORDS.value
        
        # Check category
        category = str(row.get("category", "")).lower().strip()
        if category not in self.valid_categories:
            return False, f"{ContaminantExclusionReason.INVALID_CATEGORY.value} ({category})"
        
        # Check not a confirmed lens
        if self.exclude_confirmed_lenses:
            is_lens = row.get("is_confirmed_lens", False)
            if is_lens:
                return False, ContaminantExclusionReason.IS_CONFIRMED_LENS.value
        
        # Check DR10 coverage
        if self.require_dr10_coverage:
            in_dr10_col = None
            for col in ["in_dr10", "has_dr10_coverage"]:
                if col in row.index:
                    in_dr10_col = col
                    break
            
            if in_dr10_col is not None:
                in_dr10 = row.get(in_dr10_col, False)
                if not bool(in_dr10):
                    return False, ContaminantExclusionReason.NOT_IN_DR10.value
            else:
                # Column missing - apply policy
                policy = self.on_missing_dr10_flag.upper()
                if policy == "ASSUME_IN_DR10":
                    pass  # Continue
                else:  # EXCLUDE
                    return False, ContaminantExclusionReason.DR10_FLAG_MISSING.value
        
        # Check size bounds (if specified)
        size = row.get("size_arcsec", row.get("r_eff_arcsec", None))
        if size is not None and self.min_size_arcsec is not None:
            if size < self.min_size_arcsec:
                return False, f"{ContaminantExclusionReason.TOO_SMALL.value} ({size:.2f}\")"
        if size is not None and self.max_size_arcsec is not None:
            if size > self.max_size_arcsec:
                return False, f"{ContaminantExclusionReason.TOO_LARGE.value} ({size:.2f}\")"
        
        # Check redshift (if specified)
        z = row.get("redshift", row.get("z", None))
        if z is not None and self.max_redshift is not None:
            if z > self.max_redshift:
                return False, f"{ContaminantExclusionReason.Z_TOO_HIGH.value} ({z:.2f})"
        
        return True, ContaminantExclusionReason.VALID.value
    
    def apply_to_dataframe(
        self, 
        df: pd.DataFrame,
        anchor_coords: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Apply selection function to entire DataFrame.
        
        Args:
            df: Contaminant DataFrame
            anchor_coords: DataFrame with anchor ra, dec for cross-match
        
        Adds columns:
        - is_valid_contaminant: bool
        - exclusion_reason: str (if excluded)
        - matches_anchor: bool
        
        Returns:
            DataFrame with added columns
        """
        df = df.copy()
        
        # Cross-match with anchors
        if self.exclude_anchor_matches and anchor_coords is not None:
            df["matches_anchor"] = cross_match_coords(
                df, anchor_coords, self.anchor_match_radius_arcsec
            )
        else:
            df["matches_anchor"] = False
        
        # Apply selection function
        results = df.apply(
            lambda row: self.apply(row, matches_anchor=row.get("matches_anchor", False)), 
            axis=1
        )
        
        df["is_valid_contaminant"] = results.apply(lambda x: x[0])
        df["exclusion_reason"] = results.apply(lambda x: x[1] if not x[0] else None)
        
        return df
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for logging/reporting."""
        return {
            "valid_categories": list(self.valid_categories),
            "require_dr10_coverage": self.require_dr10_coverage,
            "exclude_confirmed_lenses": self.exclude_confirmed_lenses,
            "exclude_anchor_matches": self.exclude_anchor_matches,
            "anchor_match_radius_arcsec": self.anchor_match_radius_arcsec,
            "min_size_arcsec": self.min_size_arcsec,
            "max_size_arcsec": self.max_size_arcsec,
            "max_redshift": self.max_redshift,
            "on_missing_dr10_flag": self.on_missing_dr10_flag,
        }
    
    def describe(self) -> str:
        """Human-readable description for papers/reports."""
        parts = [f"category ∈ {sorted(self.valid_categories)}"]
        if self.require_dr10_coverage:
            parts.append("in DR10 footprint")
        if self.exclude_confirmed_lenses:
            parts.append("not a confirmed lens")
        if self.exclude_anchor_matches:
            parts.append(f"no anchor match within {self.anchor_match_radius_arcsec}\"")
        if self.min_size_arcsec:
            parts.append(f"size ≥ {self.min_size_arcsec}\"")
        parts.append(f"[missing DR10 → {self.on_missing_dr10_flag}]")
        return " AND ".join(parts)


# =============================================================================
# CONTAMINANT SET
# =============================================================================

@dataclass
class ContaminantSet:
    """
    Collection of contaminant (non-lens) objects for false positive evaluation.
    
    Attributes:
        df: DataFrame with contaminant properties
        selection_function: Criteria for valid contaminants
        cutout_dir: Directory containing FITS cutouts (optional)
        category_prevalence: Expected prevalence of each category in search
    """
    
    df: pd.DataFrame
    selection_function: Optional[ContaminantSelectionFunction] = None
    cutout_dir: Optional[Path] = None
    category_prevalence: Dict[str, float] = field(default_factory=lambda: DEFAULT_CATEGORY_PREVALENCE)
    
    # Computed after initialization
    _valid: pd.DataFrame = field(default=None, init=False, repr=False)
    _excluded: pd.DataFrame = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Validate and compute subsets."""
        if "is_valid_contaminant" not in self.df.columns:
            # Selection function was applied externally or not at all
            if self.selection_function is None:
                self.df["is_valid_contaminant"] = True
        
        self._valid = self.df[self.df["is_valid_contaminant"] == True].copy()
        self._excluded = self.df[self.df["is_valid_contaminant"] == False].copy()
        
        logger.info(
            f"ContaminantSet: {len(self._valid)} valid, "
            f"{len(self._excluded)} excluded, {len(self.df)} total"
        )
        
        # Validate minimum counts per category
        self._validate_category_counts()
    
    def _validate_category_counts(self):
        """Check minimum counts per category and warn if insufficient."""
        if "category" not in self._valid.columns:
            logger.warning("No 'category' column - cannot validate category counts")
            return
        
        counts = self._valid["category"].value_counts()
        
        for category, min_count in MIN_COUNTS_PER_CATEGORY.items():
            if category in self.selection_function.valid_categories if self.selection_function else True:
                actual = counts.get(category, 0)
                if actual < min_count:
                    logger.warning(
                        f"Category '{category}' has {actual} contaminants, "
                        f"need {min_count} for statistical validity"
                    )
    
    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        selection_function: Optional[ContaminantSelectionFunction] = None,
        cutout_dir: Optional[str] = None,
        anchor_coords: Optional[pd.DataFrame] = None,
        normalize_schema: bool = True,
        column_mapping: Optional[Dict[str, str]] = None,
        category_prevalence: Optional[Dict[str, float]] = None,
    ) -> "ContaminantSet":
        """
        Load contaminant set from CSV file.
        
        Args:
            csv_path: Path to CSV with contaminant catalog
            selection_function: Criteria for valid contaminants
            cutout_dir: Directory containing FITS cutouts
            anchor_coords: DataFrame with anchor ra, dec for cross-match exclusion
            normalize_schema: Whether to normalize column names
            column_mapping: Optional explicit column mapping
            category_prevalence: Expected prevalence for weighted FPR
        
        Returns:
            ContaminantSet instance
        """
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required = {"name", "ra", "dec", "category"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Normalize schema
        if normalize_schema:
            df = normalize_contaminant_schema(df, column_mapping)
        
        # Apply selection function with anchor cross-match
        if selection_function is not None:
            df = selection_function.apply_to_dataframe(df, anchor_coords)
        
        cutout_path = Path(cutout_dir) if cutout_dir else None
        prevalence = category_prevalence or DEFAULT_CATEGORY_PREVALENCE
        
        return cls(
            df=df,
            selection_function=selection_function,
            cutout_dir=cutout_path,
            category_prevalence=prevalence,
        )
    
    @property
    def valid(self) -> pd.DataFrame:
        """Valid contaminants (pass selection function)."""
        return self._valid
    
    @property
    def excluded(self) -> pd.DataFrame:
        """Excluded contaminants (fail selection function)."""
        return self._excluded
    
    @property
    def n_total(self) -> int:
        """Total number of contaminants."""
        return len(self.df)
    
    @property
    def n_valid(self) -> int:
        """Number of valid contaminants."""
        return len(self._valid)
    
    def get_category_counts(self) -> Dict[str, int]:
        """Get counts per category for valid contaminants."""
        if "category" not in self._valid.columns:
            return {}
        return self._valid["category"].value_counts().to_dict()
    
    def get_exclusion_reasons(self) -> Dict[str, int]:
        """Get counts of why contaminants were excluded."""
        if "exclusion_reason" not in self.df.columns:
            return {}
        
        reasons = self._excluded["exclusion_reason"].apply(
            lambda x: x.split(" (")[0] if " (" in str(x) else str(x)
        ).value_counts()
        return reasons.to_dict()
    
    def evaluate(
        self,
        scores: Dict[str, float],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluate model scores against contaminant set.
        
        False positive rate = fraction of contaminants classified as lenses.
        
        Args:
            scores: Dict mapping contaminant name to model score (p_lens)
            threshold: Classification threshold (default: 0.5)
        
        Returns:
            Dict with evaluation metrics including:
            - fpr: Overall false positive rate
            - fpr_by_category: FPR per category
            - fpr_weighted: Prevalence-weighted FPR
        """
        results = {
            "threshold": threshold,
            "selection_function": (
                self.selection_function.to_dict() 
                if self.selection_function else None
            ),
        }
        
        # Match scores to valid contaminants
        matched = self._valid[self._valid["name"].isin(scores.keys())].copy()
        matched["score"] = matched["name"].map(scores)
        
        # Overall FPR
        if len(matched) > 0:
            false_positives = (matched["score"] > threshold).sum()
            results["fpr"] = false_positives / len(matched)
            results["n_false_positives"] = int(false_positives)
            results["n_total"] = len(matched)
            results["mean_score"] = float(matched["score"].mean())
            results["median_score"] = float(matched["score"].median())
            results["max_score"] = float(matched["score"].max())
        else:
            results["fpr"] = None
            results["n_false_positives"] = 0
            results["n_total"] = 0
        
        # FPR by category
        results["fpr_by_category"] = {}
        results["n_by_category"] = {}
        
        if "category" in matched.columns and len(matched) > 0:
            for category in matched["category"].unique():
                cat_df = matched[matched["category"] == category]
                if len(cat_df) > 0:
                    cat_fp = (cat_df["score"] > threshold).sum()
                    results["fpr_by_category"][category] = cat_fp / len(cat_df)
                    results["n_by_category"][category] = len(cat_df)
        
        # Prevalence-weighted FPR
        if results["fpr_by_category"]:
            weighted_sum = 0.0
            weight_total = 0.0
            
            for category, fpr in results["fpr_by_category"].items():
                prevalence = self.category_prevalence.get(category, 0.0)
                weighted_sum += fpr * prevalence
                weight_total += prevalence
            
            if weight_total > 0:
                results["fpr_weighted"] = weighted_sum / weight_total
            else:
                results["fpr_weighted"] = results["fpr"]
        else:
            results["fpr_weighted"] = results.get("fpr")
        
        # Worst offenders (highest scoring false positives)
        if len(matched) > 0:
            fp_df = matched[matched["score"] > threshold].sort_values("score", ascending=False)
            results["worst_offenders"] = fp_df.head(10)[["name", "category", "score"]].to_dict("records")
        else:
            results["worst_offenders"] = []
        
        # Coverage and anchor matches
        results["n_missing_scores"] = len(self._valid) - len(matched)
        
        if "matches_anchor" in self.df.columns:
            results["n_anchor_matches_excluded"] = self.df["matches_anchor"].sum()
        
        return results
    
    def summary(self) -> str:
        """Human-readable summary for reports."""
        lines = [
            "=" * 60,
            "CONTAMINANT SET SUMMARY",
            "=" * 60,
            f"Total contaminants: {self.n_total}",
            f"Valid (pass selection function): {self.n_valid}",
            f"Excluded: {len(self._excluded)}",
        ]
        
        if self.selection_function:
            lines.append(f"\nSelection function: {self.selection_function.describe()}")
        
        # Category distribution
        counts = self.get_category_counts()
        if counts:
            lines.append("\nCategories (valid only):")
            for category, count in sorted(counts.items(), key=lambda x: -x[1]):
                min_required = MIN_COUNTS_PER_CATEGORY.get(category, 0)
                status = "✓" if count >= min_required else f"⚠ need {min_required}"
                prevalence = self.category_prevalence.get(category, 0) * 100
                lines.append(f"  {category}: {count} {status} (prev={prevalence:.0f}%)")
        
        # Exclusion reasons
        reasons = self.get_exclusion_reasons()
        if reasons:
            lines.append("\nExclusion reasons:")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                lines.append(f"  {reason}: {count}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def to_paper_text(self) -> str:
        """Generate text suitable for paper Methods section."""
        counts = self.get_category_counts()
        cat_list = ", ".join([f"{c} ({n})" for c, n in sorted(counts.items())])
        
        sf_desc = (
            self.selection_function.describe() 
            if self.selection_function 
            else "morphologically similar to lensing features"
        )
        
        return f"""
We evaluate false positive rate on a set of {self.n_valid} objects that are
definitively not gravitational lenses but have morphological features that
could confuse a lens finder ({sf_desc}). We apply coordinate cross-matching
against the anchor set to ensure no confirmed lenses are included as contaminants.
Categories include: {cat_list}. We report both category-specific FPR and 
prevalence-weighted FPR to account for expected frequency in a real search.
""".strip()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_combined_metrics(
    anchor_results: Dict[str, Any],
    contaminant_results: Dict[str, Any],
    threshold: float = 0.5,
    recall_gate: float = 0.50,
    fpr_gate: float = 0.20,
) -> Dict[str, Any]:
    """
    Combine anchor and contaminant metrics into summary.
    
    Args:
        anchor_results: Results from AnchorSet.evaluate()
        contaminant_results: Results from ContaminantSet.evaluate()
        threshold: Classification threshold
        recall_gate: Minimum required Tier-A recall to pass
        fpr_gate: Maximum allowed FPR to pass
    
    Returns:
        Combined metrics dictionary
    """
    tier_a_recall = anchor_results.get("tier_a_recall")
    contam_fpr = contaminant_results.get("fpr")
    
    return {
        "threshold": threshold,
        
        # Primary metrics (for gating)
        "tier_a_recall": tier_a_recall,
        "contaminant_fpr": contam_fpr,
        "contaminant_fpr_weighted": contaminant_results.get("fpr_weighted"),
        
        # Supplementary
        "tier_b_recall": anchor_results.get("tier_b_recall"),
        "fpr_by_category": contaminant_results.get("fpr_by_category", {}),
        
        # Counts
        "n_tier_a_anchors": anchor_results.get("tier_a_total", 0),
        "n_tier_b_anchors": anchor_results.get("tier_b_total", 0),
        "n_contaminants": contaminant_results.get("n_total", 0),
        
        # Pass/fail gates
        "recall_gate": recall_gate,
        "fpr_gate": fpr_gate,
        "passes_recall_gate": (
            tier_a_recall >= recall_gate
            if tier_a_recall is not None
            else None
        ),
        "passes_fpr_gate": (
            contam_fpr <= fpr_gate
            if contam_fpr is not None
            else None
        ),
        "passes_all_gates": (
            (tier_a_recall is not None and tier_a_recall >= recall_gate) and
            (contam_fpr is not None and contam_fpr <= fpr_gate)
        ),
    }

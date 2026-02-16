"""
Anchor Set: Real lens evaluation with selection function.

An "anchor" is a spectroscopically confirmed strong gravitational lens
that we use to measure sim-to-real transfer (recall on real lenses).

CRITICAL CONCEPT: Selection Function
=====================================
We can ONLY fairly evaluate on anchors that fall within our model's
selection function - i.e., lenses the model was designed to find.

Selection criteria:
1. θ_E in training range (0.5" - 3.0")
2. Arc visible in ground-based imaging (arc_snr > threshold)
3. In DR10 footprint (has coverage)
4. Image usable (no masks, artifacts, saturation)

Anchors OUTSIDE the selection function are classified as "Tier-B"
and are documented but NOT used for pass/fail gating.

UPDATED based on external review (2026-02-07):
- Missing arc visibility -> Tier-B by default (not Tier-A)
- Missing DR10 coverage -> exclude by default
- Added explicit policy parameters for missing data handling
- Added θ_E margin for catalog uncertainty
- Added image usability criterion

Usage:
    from planb.evaluation import AnchorSet, AnchorSelectionFunction
    
    # Define selection function for our model
    sf = AnchorSelectionFunction(
        theta_e_min=0.5,
        theta_e_max=3.0,
        arc_snr_min=2.0,
        on_missing_arc_visibility="TIER_B",  # Defensive default
        on_missing_dr10_flag="EXCLUDE",       # Defensive default
    )
    
    # Load anchors and apply selection function
    anchors = AnchorSet.from_csv("anchors.csv", selection_function=sf)
    
    # Evaluate
    results = anchors.evaluate(model, threshold=0.5)
    print(f"Tier-A Recall: {results['tier_a_recall']:.1%}")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS FOR TYPE SAFETY
# =============================================================================

class MissingArcVisibilityPolicy(str, Enum):
    """Policy for handling missing arc visibility data."""
    TIER_B = "TIER_B"                    # Assign to Tier-B (recommended default)
    EXCLUDE = "EXCLUDE"                  # Exclude from both tiers
    ASSUME_VISIBLE = "ASSUME_VISIBLE"    # Assume visible (NOT recommended)
    MEASURE_REQUIRED = "MEASURE_REQUIRED"  # Require measurement (strictest)


class MissingDR10Policy(str, Enum):
    """Policy for handling missing DR10 coverage flag."""
    EXCLUDE = "EXCLUDE"            # Exclude (recommended default)
    ASSUME_IN_DR10 = "ASSUME_IN_DR10"  # Assume in footprint (NOT recommended)


class ExclusionReason(str, Enum):
    """Standardized exclusion reasons for reproducibility."""
    TIER_A = "TIER_A"
    THETA_E_MISSING = "theta_e_missing"
    THETA_E_BELOW_MIN = "theta_e_below_min"
    THETA_E_ABOVE_MAX = "theta_e_above_max"
    ARC_SNR_BELOW_MIN = "arc_snr_below_min"
    ARC_NOT_VISIBLE = "arc_not_visible"
    ARC_VISIBILITY_UNKNOWN_TIER_B = "arc_visibility_unknown_TIER_B"
    ARC_VISIBILITY_UNKNOWN = "arc_visibility_unknown"
    ARC_VISIBILITY_NOT_MEASURED = "arc_visibility_not_measured"
    NOT_IN_DR10 = "not_in_dr10"
    DR10_FLAG_MISSING = "dr10_flag_missing"
    LENS_TYPE_MISSING = "lens_type_missing"
    LENS_TYPE_EXCLUDED = "lens_type_excluded"
    CUTOUT_NOT_USABLE = "cutout_not_usable"
    PSF_THETA_E_RATIO_TOO_LOW = "psf_theta_e_ratio_too_low"


# =============================================================================
# CANONICAL SCHEMA
# =============================================================================

ANCHOR_CANONICAL_COLUMNS = {
    # Required
    "name": str,
    "ra": float,
    "dec": float,
    "theta_e_arcsec": float,
    "source": str,
    
    # Arc visibility (at least one should be present for Tier-A)
    "arc_snr_dr10": float,      # Measured on DR10 cutouts (preferred)
    "arc_visible_dr10": bool,   # Boolean alternative
    
    # Footprint
    "in_dr10": bool,
    
    # Image quality (optional but recommended)
    "usable_cutout": bool,
    "psf_fwhm_arcsec": float,
    "bad_pixel_frac": float,
    
    # Classification
    "tier": str,  # "A" or "B" (computed)
}


def normalize_anchor_schema(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Normalize anchor DataFrame to canonical schema.
    
    This ensures selection function operates on consistent column names,
    making evaluation reproducible across different input sources.
    
    Args:
        df: Input DataFrame with possibly heterogeneous column names
        column_mapping: Optional explicit mapping {canonical: source_column}
    
    Returns:
        DataFrame with canonical column names
    """
    df = df.copy()
    
    # Default mappings for common alternative names
    default_mapping = {
        "arc_snr_dr10": ["arc_snr", "arc_visibility_snr", "arc_snr_r"],
        "arc_visible_dr10": ["arc_visible", "arc_visibility"],
        "in_dr10": ["has_dr10_coverage", "in_footprint", "dr10_coverage"],
        "usable_cutout": ["usable", "is_usable", "good_cutout"],
        "psf_fwhm_arcsec": ["psf_fwhm", "seeing", "psf_fwhm_used_r"],
    }
    
    # Apply user-provided mapping first
    if column_mapping:
        for canonical, source in column_mapping.items():
            if source in df.columns:
                df[canonical] = df[source]
    
    # Apply default fallbacks
    for canonical, alternatives in default_mapping.items():
        if canonical not in df.columns:
            for alt in alternatives:
                if alt in df.columns:
                    df[canonical] = df[alt]
                    break
    
    return df


# =============================================================================
# SELECTION FUNCTION
# =============================================================================

@dataclass
class AnchorSelectionFunction:
    """
    Selection function defining what the model is designed to detect.
    
    An anchor passes the selection function if ALL criteria are met.
    Uses explicit policies for handling missing data to ensure reproducibility.
    
    Attributes:
        theta_e_min: Minimum Einstein radius in arcsec (default: 0.5")
        theta_e_max: Maximum Einstein radius in arcsec (default: 3.0")
        theta_e_margin: Margin for catalog uncertainty (default: 0.0")
        arc_snr_min: Minimum arc visibility SNR in ground-based imaging (default: 2.0)
        require_dr10_coverage: Whether anchor must be in DR10 footprint (default: True)
        require_usable_cutout: Whether to check image usability (default: False)
        min_theta_e_over_psf: Minimum θ_E/PSF ratio for resolved arcs (default: None)
        lens_types: Allowed lens galaxy types, or None for any (default: None)
        
        on_missing_arc_visibility: Policy for missing arc data (default: TIER_B)
        on_missing_dr10_flag: Policy for missing DR10 flag (default: EXCLUDE)
    
    Example:
        sf = AnchorSelectionFunction(
            theta_e_min=0.5,
            theta_e_max=3.0,
            arc_snr_min=2.0,
            on_missing_arc_visibility="TIER_B",
        )
        
        is_tier_a, reason = sf.apply(anchor_row)
    """
    
    # Einstein radius bounds
    theta_e_min: float = 0.5  # arcsec
    theta_e_max: float = 3.0  # arcsec
    theta_e_margin: float = 0.0  # arcsec, for catalog uncertainty
    
    # Arc visibility threshold
    arc_snr_min: float = 2.0  # SNR in ground-based imaging
    
    # Footprint requirement
    require_dr10_coverage: bool = True
    
    # Image usability
    require_usable_cutout: bool = False
    max_bad_pixel_frac: float = 0.1
    
    # PSF-based filtering
    min_theta_e_over_psf: Optional[float] = None  # e.g., 0.5 for resolved arcs
    
    # Lens galaxy type filter (None = any type allowed)
    lens_types: Optional[List[str]] = None
    
    # === MISSING DATA POLICIES (critical for reproducibility) ===
    on_missing_arc_visibility: str = "TIER_B"  # TIER_B, EXCLUDE, ASSUME_VISIBLE, MEASURE_REQUIRED
    on_missing_dr10_flag: str = "EXCLUDE"      # EXCLUDE, ASSUME_IN_DR10
    on_missing_lens_type: str = "PASS"         # PASS (allow), EXCLUDE
    
    def apply(self, row: pd.Series) -> Tuple[bool, str]:
        """
        Apply selection function to a single anchor.
        
        Args:
            row: pandas Series with anchor properties (preferably normalized)
        
        Returns:
            Tuple of (passes: bool, reason: str)
            If passes=True, reason is "TIER_A"
            If passes=False, reason explains why
        """
        # --- Check θ_E bounds ---
        theta_e = row.get("theta_e_arcsec", np.nan)
        if pd.isna(theta_e):
            return False, ExclusionReason.THETA_E_MISSING.value
        
        lo = self.theta_e_min - self.theta_e_margin
        hi = self.theta_e_max + self.theta_e_margin
        
        if theta_e < lo:
            return False, f"{ExclusionReason.THETA_E_BELOW_MIN.value} ({theta_e:.2f} < {lo:.2f})"
        if theta_e > hi:
            return False, f"{ExclusionReason.THETA_E_ABOVE_MAX.value} ({theta_e:.2f} > {hi:.2f})"
        
        # --- Check arc visibility ---
        # Prefer arc_snr_dr10 (numeric), fallback to arc_visible_dr10 (boolean)
        arc_snr = row.get("arc_snr_dr10", row.get("arc_snr", np.nan))
        arc_visible = row.get("arc_visible_dr10", row.get("arc_visible", None))
        
        if not pd.isna(arc_snr):
            # Have numeric SNR - use it
            if arc_snr < self.arc_snr_min:
                return False, f"{ExclusionReason.ARC_SNR_BELOW_MIN.value} ({arc_snr:.2f} < {self.arc_snr_min})"
        elif arc_visible is not None:
            # Have boolean visibility
            if not bool(arc_visible):
                return False, ExclusionReason.ARC_NOT_VISIBLE.value
        else:
            # Neither available - apply missing data policy
            policy = self.on_missing_arc_visibility.upper()
            if policy == "ASSUME_VISIBLE":
                pass  # Continue (NOT recommended)
            elif policy == "TIER_B":
                return False, ExclusionReason.ARC_VISIBILITY_UNKNOWN_TIER_B.value
            elif policy == "EXCLUDE":
                return False, ExclusionReason.ARC_VISIBILITY_UNKNOWN.value
            elif policy == "MEASURE_REQUIRED":
                return False, ExclusionReason.ARC_VISIBILITY_NOT_MEASURED.value
            else:
                return False, ExclusionReason.ARC_VISIBILITY_UNKNOWN.value
        
        # --- Check DR10 coverage ---
        if self.require_dr10_coverage:
            in_dr10_col = None
            for col in ["in_dr10", "has_dr10_coverage"]:
                if col in row.index:
                    in_dr10_col = col
                    break
            
            if in_dr10_col is not None:
                in_dr10 = row.get(in_dr10_col, False)
                if not bool(in_dr10):
                    return False, ExclusionReason.NOT_IN_DR10.value
            else:
                # Column missing - apply policy
                policy = self.on_missing_dr10_flag.upper()
                if policy == "ASSUME_IN_DR10":
                    pass  # Continue (NOT recommended)
                else:  # EXCLUDE
                    return False, ExclusionReason.DR10_FLAG_MISSING.value
        
        # --- Check image usability ---
        if self.require_usable_cutout:
            usable = row.get("usable_cutout", None)
            if usable is not None and not bool(usable):
                return False, ExclusionReason.CUTOUT_NOT_USABLE.value
            
            bad_pix = row.get("bad_pixel_frac", None)
            if bad_pix is not None and bad_pix > self.max_bad_pixel_frac:
                return False, f"{ExclusionReason.CUTOUT_NOT_USABLE.value} (bad_pix={bad_pix:.2f})"
        
        # --- Check θ_E / PSF ratio ---
        if self.min_theta_e_over_psf is not None:
            psf = row.get("psf_fwhm_arcsec", row.get("psf_fwhm", np.nan))
            if not pd.isna(psf) and psf > 0:
                ratio = theta_e / psf
                if ratio < self.min_theta_e_over_psf:
                    return False, f"{ExclusionReason.PSF_THETA_E_RATIO_TOO_LOW.value} ({ratio:.2f} < {self.min_theta_e_over_psf})"
        
        # --- Check lens type ---
        if self.lens_types is not None:
            lens_type = row.get("lens_type", row.get("morphology", None))
            if lens_type is None:
                if self.on_missing_lens_type.upper() == "EXCLUDE":
                    return False, ExclusionReason.LENS_TYPE_MISSING.value
                # else PASS - continue
            elif lens_type not in self.lens_types:
                return False, f"{ExclusionReason.LENS_TYPE_EXCLUDED.value} ({lens_type})"
        
        return True, ExclusionReason.TIER_A.value
    
    def apply_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply selection function to entire DataFrame.
        
        Adds columns:
        - in_selection_function: bool
        - selection_reason: str
        - tier: "A" or "B"
        
        Returns:
            DataFrame with added columns
        """
        results = df.apply(lambda row: self.apply(row), axis=1)
        df = df.copy()
        df["in_selection_function"] = results.apply(lambda x: x[0])
        df["selection_reason"] = results.apply(lambda x: x[1])
        df["tier"] = df["in_selection_function"].map({True: "A", False: "B"})
        return df
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for logging/reporting."""
        return {
            "theta_e_min": self.theta_e_min,
            "theta_e_max": self.theta_e_max,
            "theta_e_margin": self.theta_e_margin,
            "arc_snr_min": self.arc_snr_min,
            "require_dr10_coverage": self.require_dr10_coverage,
            "require_usable_cutout": self.require_usable_cutout,
            "min_theta_e_over_psf": self.min_theta_e_over_psf,
            "lens_types": self.lens_types,
            "on_missing_arc_visibility": self.on_missing_arc_visibility,
            "on_missing_dr10_flag": self.on_missing_dr10_flag,
            "on_missing_lens_type": self.on_missing_lens_type,
        }
    
    def describe(self) -> str:
        """Human-readable description for papers/reports."""
        parts = [
            f"θ_E ∈ [{self.theta_e_min:.1f}\", {self.theta_e_max:.1f}\"]",
        ]
        if self.theta_e_margin > 0:
            parts[-1] += f" ± {self.theta_e_margin:.1f}\""
        
        parts.append(f"arc SNR > {self.arc_snr_min:.1f}")
        
        if self.require_dr10_coverage:
            parts.append("in DR10 footprint")
        if self.require_usable_cutout:
            parts.append("usable cutout")
        if self.min_theta_e_over_psf:
            parts.append(f"θ_E/PSF > {self.min_theta_e_over_psf:.1f}")
        if self.lens_types:
            parts.append(f"lens type ∈ {self.lens_types}")
        
        # Add missing data policies
        parts.append(f"[missing arc → {self.on_missing_arc_visibility}]")
        parts.append(f"[missing DR10 → {self.on_missing_dr10_flag}]")
        
        return " AND ".join(parts)
    
    def describe_for_paper(self) -> str:
        """
        Generate text suitable for paper Methods section.
        
        This explicitly states measurement and missing-data policies
        to avoid appearance of post-hoc cherry-picking.
        """
        return f"""
We define a Tier-A anchor subset intended to match the model's operating regime: 
DR10 footprint, usable cutouts (masking and image-quality cuts), Einstein radius 
within [{self.theta_e_min:.1f}", {self.theta_e_max:.1f}"] (the training support), 
and arcs detectable in DR10 by an automated visibility metric (SNR > {self.arc_snr_min:.1f}) 
computed on the same cutouts used for inference. Anchors failing these criteria, 
or lacking the measurements required to apply them (missing arc visibility → {self.on_missing_arc_visibility}, 
missing DR10 flag → {self.on_missing_dr10_flag}), are assigned to Tier-B and reported separately.
""".strip()


# =============================================================================
# ANCHOR SET
# =============================================================================

@dataclass
class AnchorSet:
    """
    Collection of anchor lenses for evaluation.
    
    Attributes:
        df: DataFrame with anchor properties
        selection_function: Criteria for Tier-A classification
        cutout_dir: Directory containing FITS cutouts (optional)
    """
    
    df: pd.DataFrame
    selection_function: Optional[AnchorSelectionFunction] = None
    cutout_dir: Optional[Path] = None
    
    # Computed after initialization
    _tier_a: pd.DataFrame = field(default=None, init=False, repr=False)
    _tier_b: pd.DataFrame = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Apply selection function and split into tiers."""
        if self.selection_function is not None:
            self.df = self.selection_function.apply_to_dataframe(self.df)
        elif "tier" not in self.df.columns:
            # Default: all are Tier-A if no selection function
            self.df["tier"] = "A"
            self.df["in_selection_function"] = True
        
        self._tier_a = self.df[self.df["tier"] == "A"].copy()
        self._tier_b = self.df[self.df["tier"] == "B"].copy()
        
        logger.info(
            f"AnchorSet: {len(self._tier_a)} Tier-A, "
            f"{len(self._tier_b)} Tier-B, {len(self.df)} total"
        )
        
        # Log exclusion summary
        if len(self._tier_b) > 0:
            reasons = self.get_exclusion_reasons()
            logger.info(f"Tier-B exclusion reasons: {reasons}")
    
    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        selection_function: Optional[AnchorSelectionFunction] = None,
        cutout_dir: Optional[str] = None,
        normalize_schema: bool = True,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> "AnchorSet":
        """
        Load anchor set from CSV file.
        
        Args:
            csv_path: Path to CSV with anchor catalog
            selection_function: Criteria for Tier-A classification
            cutout_dir: Directory containing FITS cutouts
            normalize_schema: Whether to normalize column names
            column_mapping: Optional explicit column mapping
        
        Returns:
            AnchorSet instance
        """
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required = {"name", "ra", "dec", "theta_e_arcsec", "source"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Normalize schema if requested
        if normalize_schema:
            df = normalize_anchor_schema(df, column_mapping)
        
        cutout_path = Path(cutout_dir) if cutout_dir else None
        
        return cls(
            df=df,
            selection_function=selection_function,
            cutout_dir=cutout_path,
        )
    
    @property
    def tier_a(self) -> pd.DataFrame:
        """Anchors within selection function (primary evaluation)."""
        return self._tier_a
    
    @property
    def tier_b(self) -> pd.DataFrame:
        """Anchors outside selection function (stress test only)."""
        return self._tier_b
    
    @property
    def n_total(self) -> int:
        """Total number of anchors."""
        return len(self.df)
    
    @property
    def n_tier_a(self) -> int:
        """Number of Tier-A anchors."""
        return len(self._tier_a)
    
    @property
    def n_tier_b(self) -> int:
        """Number of Tier-B anchors."""
        return len(self._tier_b)
    
    def get_tier_a_fraction(self) -> float:
        """Fraction of anchors that are Tier-A."""
        if len(self.df) == 0:
            return 0.0
        return len(self._tier_a) / len(self.df)
    
    def get_exclusion_reasons(self) -> Dict[str, int]:
        """Get counts of why anchors were excluded from Tier-A."""
        if "selection_reason" not in self.df.columns:
            return {}
        
        # Extract base reason (before parenthetical details)
        reasons = self.df[self.df["tier"] == "B"]["selection_reason"].apply(
            lambda x: x.split(" (")[0] if " (" in str(x) else str(x)
        ).value_counts()
        return reasons.to_dict()
    
    def get_theta_e_distribution(self, tier: str = "A") -> Dict[str, float]:
        """Get θ_E statistics for a tier."""
        subset = self._tier_a if tier == "A" else self._tier_b
        if len(subset) == 0:
            return {}
        
        return {
            "min": float(subset["theta_e_arcsec"].min()),
            "max": float(subset["theta_e_arcsec"].max()),
            "median": float(subset["theta_e_arcsec"].median()),
            "mean": float(subset["theta_e_arcsec"].mean()),
            "std": float(subset["theta_e_arcsec"].std()),
        }
    
    def evaluate(
        self,
        scores: Dict[str, float],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluate model scores against anchor set.
        
        Args:
            scores: Dict mapping anchor name to model score (p_lens)
            threshold: Classification threshold (default: 0.5)
        
        Returns:
            Dict with evaluation metrics
        """
        results = {
            "threshold": threshold,
            "selection_function": (
                self.selection_function.to_dict() 
                if self.selection_function else None
            ),
        }
        
        # Match scores to anchors
        matched_a = self._tier_a[self._tier_a["name"].isin(scores.keys())].copy()
        matched_b = self._tier_b[self._tier_b["name"].isin(scores.keys())].copy()
        
        matched_a["score"] = matched_a["name"].map(scores)
        matched_b["score"] = matched_b["name"].map(scores)
        
        # Tier-A metrics (PRIMARY)
        if len(matched_a) > 0:
            tier_a_detected = (matched_a["score"] > threshold).sum()
            results["tier_a_recall"] = tier_a_detected / len(matched_a)
            results["tier_a_detected"] = int(tier_a_detected)
            results["tier_a_total"] = len(matched_a)
            results["tier_a_mean_score"] = float(matched_a["score"].mean())
            results["tier_a_median_score"] = float(matched_a["score"].median())
            
            # Top detected
            detected_df = matched_a[matched_a["score"] > threshold].sort_values("score", ascending=False)
            results["tier_a_top_detected"] = detected_df.head(5)[["name", "theta_e_arcsec", "score"]].to_dict("records")
            
            # Worst missed
            missed_df = matched_a[matched_a["score"] <= threshold].sort_values("score", ascending=False)
            results["tier_a_worst_missed"] = missed_df.head(5)[["name", "theta_e_arcsec", "score"]].to_dict("records")
        else:
            results["tier_a_recall"] = None
            results["tier_a_detected"] = 0
            results["tier_a_total"] = 0
        
        # Tier-B metrics (SUPPLEMENTARY)
        if len(matched_b) > 0:
            tier_b_detected = (matched_b["score"] > threshold).sum()
            results["tier_b_recall"] = tier_b_detected / len(matched_b)
            results["tier_b_detected"] = int(tier_b_detected)
            results["tier_b_total"] = len(matched_b)
            results["tier_b_mean_score"] = float(matched_b["score"].mean())
        else:
            results["tier_b_recall"] = None
            results["tier_b_detected"] = 0
            results["tier_b_total"] = 0
        
        # Overall (for reference only)
        all_matched = pd.concat([matched_a, matched_b])
        if len(all_matched) > 0:
            overall_detected = (all_matched["score"] > threshold).sum()
            results["overall_recall"] = overall_detected / len(all_matched)
            results["overall_detected"] = int(overall_detected)
            results["overall_total"] = len(all_matched)
        
        # Exclusion summary
        results["exclusion_reasons"] = self.get_exclusion_reasons()
        
        # Coverage
        results["n_missing_scores"] = len(self.df) - len(all_matched)
        
        # θ_E stratified recall (for Tier-A only)
        if len(matched_a) > 0:
            bins = [0, 1.0, 1.5, 2.0, np.inf]
            labels = ["<1.0", "1.0-1.5", "1.5-2.0", ">2.0"]
            matched_a["theta_bin"] = pd.cut(matched_a["theta_e_arcsec"], bins=bins, labels=labels)
            
            stratified = {}
            for label in labels:
                subset = matched_a[matched_a["theta_bin"] == label]
                if len(subset) > 0:
                    det = (subset["score"] > threshold).sum()
                    stratified[label] = {"recall": det / len(subset), "n": len(subset)}
            results["tier_a_recall_by_theta_e"] = stratified
        
        return results
    
    def summary(self) -> str:
        """Human-readable summary for reports."""
        lines = [
            "=" * 60,
            "ANCHOR SET SUMMARY",
            "=" * 60,
            f"Total anchors: {self.n_total}",
            f"Tier-A (in selection function): {self.n_tier_a} ({self.get_tier_a_fraction():.1%})",
            f"Tier-B (outside selection function): {self.n_tier_b}",
        ]
        
        if self.selection_function:
            lines.append(f"\nSelection function: {self.selection_function.describe()}")
        
        reasons = self.get_exclusion_reasons()
        if reasons:
            lines.append("\nExclusion reasons (Tier-B):")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                lines.append(f"  {reason}: {count}")
        
        # Source distribution
        if "source" in self.df.columns:
            lines.append("\nSources:")
            for source, count in self.df["source"].value_counts().items():
                tier_a_count = len(self._tier_a[self._tier_a["source"] == source])
                lines.append(f"  {source}: {count} total, {tier_a_count} Tier-A ({tier_a_count/count:.0%})")
        
        # θ_E distribution
        lines.append("\nθ_E distribution (Tier-A):")
        theta_stats = self.get_theta_e_distribution("A")
        if theta_stats:
            lines.append(f"  Range: [{theta_stats['min']:.2f}\", {theta_stats['max']:.2f}\"]")
            lines.append(f"  Median: {theta_stats['median']:.2f}\"")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def to_paper_text(self) -> str:
        """Generate text suitable for paper Methods section."""
        if self.selection_function:
            return self.selection_function.describe_for_paper() + f"\n\nOf {self.n_total} total anchors, {self.n_tier_a} ({self.get_tier_a_fraction():.0%}) satisfy Tier-A criteria."
        
        return f"We evaluate on {self.n_total} spectroscopically confirmed lenses."

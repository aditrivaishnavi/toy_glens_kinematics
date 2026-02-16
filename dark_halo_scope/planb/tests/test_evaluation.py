"""
Tests for evaluation module (anchor and contaminant sets).
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

# Import modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.anchor_set import AnchorSet, AnchorSelectionFunction
from evaluation.contaminant_set import (
    ContaminantSet, 
    ContaminantSelectionFunction,
    compute_combined_metrics,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_anchor_df():
    """Create sample anchor DataFrame for testing."""
    return pd.DataFrame({
        "name": [
            "SLACS_001", "SLACS_002", "SLACS_003", 
            "BELLS_001", "BELLS_002",
            "LS_001", "LS_002"
        ],
        "ra": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
        "dec": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
        "theta_e_arcsec": [1.2, 0.3, 1.8, 0.8, 2.5, 1.5, 3.5],
        "source": ["SLACS", "SLACS", "SLACS", "BELLS", "BELLS", "LS_ML", "LS_ML"],
        "arc_snr": [2.5, 1.0, 3.0, 1.5, 4.0, 5.0, 2.0],
        "in_dr10": [True, True, True, True, True, True, False],
    })


@pytest.fixture
def sample_contaminant_df():
    """Create sample contaminant DataFrame for testing."""
    return pd.DataFrame({
        "name": [
            "Ring_001", "Ring_002", "Spiral_001", "Spiral_002",
            "Merger_001", "Spike_001", "Unknown_001"
        ],
        "ra": [200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0],
        "dec": [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        "category": ["ring", "ring", "spiral", "spiral", "merger", "spike", "other"],
        "in_dr10": [True, True, True, True, True, True, True],
        "is_confirmed_lens": [False, False, False, False, False, False, False],
    })


@pytest.fixture
def default_anchor_sf():
    """Default anchor selection function."""
    return AnchorSelectionFunction(
        theta_e_min=0.5,
        theta_e_max=3.0,
        arc_snr_min=2.0,
        require_dr10_coverage=True,
    )


@pytest.fixture
def default_contaminant_sf():
    """Default contaminant selection function."""
    return ContaminantSelectionFunction(
        valid_categories={"ring", "spiral", "merger", "spike"},
        require_dr10_coverage=True,
        exclude_confirmed_lenses=True,
    )


# =============================================================================
# ANCHOR SELECTION FUNCTION TESTS
# =============================================================================

class TestAnchorSelectionFunction:
    """Tests for AnchorSelectionFunction."""
    
    def test_apply_passes_valid_anchor(self, default_anchor_sf):
        """Anchor within all bounds should pass."""
        row = pd.Series({
            "theta_e_arcsec": 1.5,
            "arc_snr": 3.0,
            "in_dr10": True,
        })
        passes, reason = default_anchor_sf.apply(row)
        assert passes is True
        assert reason == "TIER_A"
    
    def test_apply_fails_theta_e_below_min(self, default_anchor_sf):
        """Anchor with theta_e below minimum should fail."""
        row = pd.Series({
            "theta_e_arcsec": 0.3,
            "arc_snr": 3.0,
            "in_dr10": True,
        })
        passes, reason = default_anchor_sf.apply(row)
        assert passes is False
        assert "theta_e_below_min" in reason
    
    def test_apply_fails_theta_e_above_max(self, default_anchor_sf):
        """Anchor with theta_e above maximum should fail."""
        row = pd.Series({
            "theta_e_arcsec": 5.0,
            "arc_snr": 3.0,
            "in_dr10": True,
        })
        passes, reason = default_anchor_sf.apply(row)
        assert passes is False
        assert "theta_e_above_max" in reason
    
    def test_apply_fails_arc_snr_too_low(self, default_anchor_sf):
        """Anchor with arc SNR below threshold should fail."""
        row = pd.Series({
            "theta_e_arcsec": 1.5,
            "arc_snr": 1.0,
            "in_dr10": True,
        })
        passes, reason = default_anchor_sf.apply(row)
        assert passes is False
        assert "arc_snr_below_min" in reason
    
    def test_apply_fails_not_in_dr10(self, default_anchor_sf):
        """Anchor not in DR10 footprint should fail."""
        row = pd.Series({
            "theta_e_arcsec": 1.5,
            "arc_snr": 3.0,
            "in_dr10": False,
        })
        passes, reason = default_anchor_sf.apply(row)
        assert passes is False
        assert "not_in_dr10" in reason
    
    def test_apply_missing_theta_e(self, default_anchor_sf):
        """Anchor with missing theta_e should fail."""
        row = pd.Series({
            "arc_snr": 3.0,
            "in_dr10": True,
        })
        passes, reason = default_anchor_sf.apply(row)
        assert passes is False
        assert "theta_e_missing" in reason
    
    def test_apply_arc_snr_missing_assumes_visible(self, default_anchor_sf):
        """If arc_snr is missing, assume visible (conservative)."""
        row = pd.Series({
            "theta_e_arcsec": 1.5,
            "in_dr10": True,
        })
        passes, reason = default_anchor_sf.apply(row)
        assert passes is True
    
    def test_apply_arc_visible_boolean_false(self, default_anchor_sf):
        """If arc_visible boolean is False, should fail."""
        row = pd.Series({
            "theta_e_arcsec": 1.5,
            "arc_visible": False,
            "in_dr10": True,
        })
        passes, reason = default_anchor_sf.apply(row)
        assert passes is False
        assert "arc_not_visible" in reason
    
    def test_apply_lens_type_filter(self):
        """Lens type filter should exclude non-matching types."""
        sf = AnchorSelectionFunction(
            theta_e_min=0.5,
            theta_e_max=3.0,
            arc_snr_min=2.0,
            lens_types=["LRG", "ETG"],
        )
        
        # Should pass
        row_lrg = pd.Series({
            "theta_e_arcsec": 1.5,
            "arc_snr": 3.0,
            "in_dr10": True,
            "lens_type": "LRG",
        })
        passes, _ = sf.apply(row_lrg)
        assert passes is True
        
        # Should fail
        row_spiral = pd.Series({
            "theta_e_arcsec": 1.5,
            "arc_snr": 3.0,
            "in_dr10": True,
            "lens_type": "spiral",
        })
        passes, reason = sf.apply(row_spiral)
        assert passes is False
        assert "lens_type_excluded" in reason
    
    def test_apply_to_dataframe(self, sample_anchor_df, default_anchor_sf):
        """Apply to DataFrame should add tier column."""
        result = default_anchor_sf.apply_to_dataframe(sample_anchor_df)
        
        assert "tier" in result.columns
        assert "in_selection_function" in result.columns
        assert "selection_reason" in result.columns
        
        # Check specific cases
        # SLACS_001: theta=1.2, arc_snr=2.5 -> should be Tier-A
        assert result[result["name"] == "SLACS_001"]["tier"].values[0] == "A"
        
        # SLACS_002: theta=0.3 -> should be Tier-B (below min)
        assert result[result["name"] == "SLACS_002"]["tier"].values[0] == "B"
        
        # LS_002: theta=3.5 -> should be Tier-B (above max)
        assert result[result["name"] == "LS_002"]["tier"].values[0] == "B"
    
    def test_describe(self, default_anchor_sf):
        """Describe should return human-readable string."""
        desc = default_anchor_sf.describe()
        assert "θ_E" in desc
        assert "0.5" in desc
        assert "3.0" in desc
        assert "arc SNR" in desc
        assert "2.0" in desc


# =============================================================================
# ANCHOR SET TESTS
# =============================================================================

class TestAnchorSet:
    """Tests for AnchorSet."""
    
    def test_from_dataframe(self, sample_anchor_df, default_anchor_sf):
        """Create AnchorSet from DataFrame."""
        anchor_set = AnchorSet(
            df=sample_anchor_df,
            selection_function=default_anchor_sf,
        )
        
        assert anchor_set.n_total == 7
        assert anchor_set.n_tier_a > 0
        assert anchor_set.n_tier_b > 0
        assert anchor_set.n_tier_a + anchor_set.n_tier_b == anchor_set.n_total
    
    def test_from_csv(self, sample_anchor_df, default_anchor_sf):
        """Load AnchorSet from CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_anchor_df.to_csv(f.name, index=False)
            
            anchor_set = AnchorSet.from_csv(
                f.name,
                selection_function=default_anchor_sf,
            )
            
            assert anchor_set.n_total == 7
            
            # Cleanup
            Path(f.name).unlink()
    
    def test_tier_properties(self, sample_anchor_df, default_anchor_sf):
        """Tier-A and Tier-B properties should return correct subsets."""
        anchor_set = AnchorSet(
            df=sample_anchor_df,
            selection_function=default_anchor_sf,
        )
        
        tier_a = anchor_set.tier_a
        tier_b = anchor_set.tier_b
        
        # All Tier-A should have tier="A"
        assert all(tier_a["tier"] == "A")
        assert all(tier_b["tier"] == "B")
    
    def test_evaluate_with_scores(self, sample_anchor_df, default_anchor_sf):
        """Evaluate with model scores."""
        anchor_set = AnchorSet(
            df=sample_anchor_df,
            selection_function=default_anchor_sf,
        )
        
        # Create mock scores
        scores = {
            "SLACS_001": 0.8,  # Tier-A, detected
            "SLACS_002": 0.6,  # Tier-B (excluded), doesn't count
            "SLACS_003": 0.4,  # Tier-A, missed
            "BELLS_001": 0.3,  # Tier-B (arc_snr too low)
            "BELLS_002": 0.9,  # Tier-A, detected
            "LS_001": 0.7,     # Tier-A, detected
            "LS_002": 0.5,     # Tier-B (not in DR10)
        }
        
        results = anchor_set.evaluate(scores, threshold=0.5)
        
        assert "tier_a_recall" in results
        assert "tier_b_recall" in results
        assert results["tier_a_total"] > 0
    
    def test_summary(self, sample_anchor_df, default_anchor_sf):
        """Summary should produce readable output."""
        anchor_set = AnchorSet(
            df=sample_anchor_df,
            selection_function=default_anchor_sf,
        )
        
        summary = anchor_set.summary()
        
        assert "ANCHOR SET SUMMARY" in summary
        assert "Tier-A" in summary
        assert "Tier-B" in summary


# =============================================================================
# CONTAMINANT SELECTION FUNCTION TESTS
# =============================================================================

class TestContaminantSelectionFunction:
    """Tests for ContaminantSelectionFunction."""
    
    def test_apply_passes_valid_contaminant(self, default_contaminant_sf):
        """Valid contaminant should pass."""
        row = pd.Series({
            "category": "ring",
            "in_dr10": True,
            "is_confirmed_lens": False,
        })
        valid, reason = default_contaminant_sf.apply(row)
        assert valid is True
        assert reason == "VALID"
    
    def test_apply_fails_invalid_category(self, default_contaminant_sf):
        """Invalid category should fail."""
        row = pd.Series({
            "category": "unknown",
            "in_dr10": True,
            "is_confirmed_lens": False,
        })
        valid, reason = default_contaminant_sf.apply(row)
        assert valid is False
        assert "invalid_category" in reason
    
    def test_apply_fails_confirmed_lens(self, default_contaminant_sf):
        """Confirmed lens should be excluded."""
        row = pd.Series({
            "category": "ring",
            "in_dr10": True,
            "is_confirmed_lens": True,
        })
        valid, reason = default_contaminant_sf.apply(row)
        assert valid is False
        assert "is_confirmed_lens" in reason
    
    def test_apply_fails_not_in_dr10(self, default_contaminant_sf):
        """Object not in DR10 should fail."""
        row = pd.Series({
            "category": "ring",
            "in_dr10": False,
            "is_confirmed_lens": False,
        })
        valid, reason = default_contaminant_sf.apply(row)
        assert valid is False
        assert "not_in_dr10" in reason
    
    def test_apply_size_filter(self):
        """Size filter should work."""
        sf = ContaminantSelectionFunction(
            min_size_arcsec=1.0,
            max_size_arcsec=10.0,
        )
        
        # Too small
        row_small = pd.Series({
            "category": "ring",
            "in_dr10": True,
            "size_arcsec": 0.5,
        })
        valid, reason = sf.apply(row_small)
        assert valid is False
        assert "too_small" in reason
        
        # Too large
        row_large = pd.Series({
            "category": "ring",
            "in_dr10": True,
            "size_arcsec": 15.0,
        })
        valid, reason = sf.apply(row_large)
        assert valid is False
        assert "too_large" in reason


# =============================================================================
# CONTAMINANT SET TESTS
# =============================================================================

class TestContaminantSet:
    """Tests for ContaminantSet."""
    
    def test_from_dataframe(self, sample_contaminant_df, default_contaminant_sf):
        """Create ContaminantSet from DataFrame."""
        contam_set = ContaminantSet(
            df=sample_contaminant_df,
            selection_function=default_contaminant_sf,
        )
        
        assert contam_set.n_total == 7
        # "Unknown_001" should be excluded (invalid category)
        assert contam_set.n_valid == 6
    
    def test_get_category_counts(self, sample_contaminant_df, default_contaminant_sf):
        """Category counts should be accurate."""
        contam_set = ContaminantSet(
            df=sample_contaminant_df,
            selection_function=default_contaminant_sf,
        )
        
        counts = contam_set.get_category_counts()
        assert counts["ring"] == 2
        assert counts["spiral"] == 2
        assert counts["merger"] == 1
        assert counts["spike"] == 1
    
    def test_evaluate_with_scores(self, sample_contaminant_df, default_contaminant_sf):
        """Evaluate with model scores."""
        contam_set = ContaminantSet(
            df=sample_contaminant_df,
            selection_function=default_contaminant_sf,
        )
        
        # Create mock scores
        scores = {
            "Ring_001": 0.8,    # False positive
            "Ring_002": 0.2,    # True negative
            "Spiral_001": 0.1,  # True negative
            "Spiral_002": 0.3,  # True negative
            "Merger_001": 0.6,  # False positive
            "Spike_001": 0.05,  # True negative
        }
        
        results = contam_set.evaluate(scores, threshold=0.5)
        
        assert "fpr" in results
        assert results["n_false_positives"] == 2  # Ring_001, Merger_001
        assert results["n_total"] == 6
        assert results["fpr"] == 2 / 6
        
        # Check FPR by category
        assert "fpr_by_category" in results
        assert results["fpr_by_category"]["ring"] == 0.5  # 1/2
        assert results["fpr_by_category"]["merger"] == 1.0  # 1/1


# =============================================================================
# COMBINED METRICS TESTS
# =============================================================================

class TestCombinedMetrics:
    """Tests for compute_combined_metrics."""
    
    def test_combined_metrics(self):
        """Combine anchor and contaminant results."""
        anchor_results = {
            "tier_a_recall": 0.75,
            "tier_a_total": 20,
            "tier_b_recall": 0.25,
            "tier_b_total": 10,
        }
        
        contaminant_results = {
            "fpr": 0.15,
            "n_total": 100,
            "fpr_by_category": {
                "ring": 0.20,
                "spiral": 0.10,
                "merger": 0.18,
            },
        }
        
        combined = compute_combined_metrics(
            anchor_results,
            contaminant_results,
            threshold=0.5,
        )
        
        assert combined["tier_a_recall"] == 0.75
        assert combined["contaminant_fpr"] == 0.15
        assert combined["passes_recall_gate"] is True  # 0.75 >= 0.50
        assert combined["passes_fpr_gate"] is True     # 0.15 <= 0.20


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_anchor_set(self, default_anchor_sf):
        """Empty anchor set should not crash."""
        empty_df = pd.DataFrame({
            "name": [],
            "ra": [],
            "dec": [],
            "theta_e_arcsec": [],
            "source": [],
        })
        
        anchor_set = AnchorSet(df=empty_df, selection_function=default_anchor_sf)
        
        assert anchor_set.n_total == 0
        assert anchor_set.n_tier_a == 0
        assert anchor_set.get_tier_a_fraction() == 0.0
    
    def test_all_anchors_tier_b(self, default_anchor_sf):
        """All anchors excluded should still work."""
        bad_df = pd.DataFrame({
            "name": ["A", "B", "C"],
            "ra": [100, 101, 102],
            "dec": [20, 21, 22],
            "theta_e_arcsec": [0.1, 0.2, 0.3],  # All below min
            "source": ["X", "X", "X"],
        })
        
        anchor_set = AnchorSet(df=bad_df, selection_function=default_anchor_sf)
        
        assert anchor_set.n_tier_a == 0
        assert anchor_set.n_tier_b == 3
    
    def test_missing_scores_in_evaluate(self, sample_anchor_df, default_anchor_sf):
        """Missing scores should be tracked."""
        anchor_set = AnchorSet(
            df=sample_anchor_df,
            selection_function=default_anchor_sf,
        )
        
        # Only provide scores for some anchors
        partial_scores = {
            "SLACS_001": 0.8,
            "SLACS_003": 0.6,
        }
        
        results = anchor_set.evaluate(partial_scores, threshold=0.5)
        
        assert results["n_missing_scores"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

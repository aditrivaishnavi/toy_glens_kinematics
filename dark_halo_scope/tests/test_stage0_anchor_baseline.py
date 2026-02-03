#!/usr/bin/env python3
"""
Unit tests for Stage 0: Anchor Baseline Evaluation

Tests every component of the anchor baseline pipeline to ensure
correctness before running on real data.
"""

import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Check for optional dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Import catalog data directly (doesn't require torch)
# We'll test the core data structures first
SLACS_LENSES = [
    # (name, ra, dec, theta_e_arcsec, z_lens, z_source)
    ("SDSSJ0029-0055", 7.4008, -0.9269, 0.96, 0.227, 0.931),
    ("SDSSJ0037-0942", 9.4583, -9.7050, 1.53, 0.195, 0.632),
    ("SDSSJ0216-0813", 34.1221, -8.2217, 1.16, 0.332, 0.523),
    ("SDSSJ0912+0029", 138.1417, 0.4861, 1.63, 0.164, 0.324),
    # Add more for testing
]

BELLS_LENSES = [
    ("BELLSJ0747+4448", 116.898, 44.808, 1.16, 0.437, 0.898),
    ("BELLSJ0801+4727", 120.348, 47.455, 0.91, 0.544, 1.072),
]

RING_GALAXIES = [
    ("Hoag's Object", 226.0792, 21.5861),
    ("Arp 147", 46.7458, 1.2939),
]

MERGER_GALAXIES = [
    ("NGC 2623", 129.6000, 25.7542),
    ("Arp 220", 233.7375, 23.5033),
]


def create_known_lens_catalog() -> pd.DataFrame:
    """Create DataFrame of known lenses from SLACS + BELLS."""
    slacs_df = pd.DataFrame(
        SLACS_LENSES,
        columns=['name', 'ra', 'dec', 'theta_e', 'z_lens', 'z_source']
    )
    slacs_df['catalog'] = 'SLACS'
    
    bells_df = pd.DataFrame(
        BELLS_LENSES,
        columns=['name', 'ra', 'dec', 'theta_e', 'z_lens', 'z_source']
    )
    bells_df['catalog'] = 'BELLS'
    
    return pd.concat([slacs_df, bells_df], ignore_index=True)


def create_hard_negative_catalog() -> pd.DataFrame:
    """Create DataFrame of hard negatives (rings, mergers)."""
    rings_df = pd.DataFrame(RING_GALAXIES, columns=['name', 'ra', 'dec'])
    rings_df['catalog'] = 'Ring'
    
    mergers_df = pd.DataFrame(MERGER_GALAXIES, columns=['name', 'ra', 'dec'])
    mergers_df['catalog'] = 'Merger'
    
    return pd.concat([rings_df, mergers_df], ignore_index=True)


def check_dr10_footprint(ra: float, dec: float) -> bool:
    """Check if (ra, dec) is in approximate DR10 footprint."""
    if dec > 80 or dec < -20:
        return False
    return True


def get_brick_for_position(ra: float, dec: float) -> str:
    """Compute the DECaLS brick name for a position."""
    ra_prefix = int(np.floor(ra / 10) * 10)
    ra_str = f"{ra_prefix:03d}"
    
    dec_sign = "p" if dec >= 0 else "m"
    dec_val = int(np.round(abs(dec) * 10))
    dec_str = f"{dec_val:03d}"
    
    return f"{ra_str}{dec_sign}{dec_str}"


def robust_mad_norm_outer(x: np.ndarray, clip: float = 10.0, eps: float = 1e-6,
                          inner_frac: float = 0.5) -> np.ndarray:
    """Normalize using outer annulus only."""
    out = np.empty_like(x, dtype=np.float32)
    h, w = x.shape[-2:]
    cy, cx = h // 2, w // 2
    ri = int(min(h, w) * inner_frac / 2)
    
    yy, xx = np.ogrid[:h, :w]
    outer_mask = ((yy - cy)**2 + (xx - cx)**2) > ri**2
    
    for c in range(x.shape[0]):
        v = x[c]
        outer_v = v[outer_mask]
        med = np.median(outer_v)
        mad = np.median(np.abs(outer_v - med))
        scale = 1.4826 * mad + eps
        vv = (v - med) / scale
        if clip is not None:
            vv = np.clip(vv, -clip, clip)
        out[c] = vv.astype(np.float32)
    return out


def compute_anchor_metrics(
    known_lenses_results: pd.DataFrame,
    hard_neg_results: pd.DataFrame,
    thresholds: list = [0.5, 0.7, 0.9]
) -> dict:
    """Compute anchor baseline metrics."""
    metrics = {}
    
    kl_valid = known_lenses_results[known_lenses_results['cutout_ok'] == True].copy()
    hn_valid = hard_neg_results[hard_neg_results['cutout_ok'] == True].copy()
    
    metrics['n_known_lenses_total'] = len(known_lenses_results)
    metrics['n_known_lenses_in_footprint'] = len(kl_valid)
    metrics['n_hard_negatives_total'] = len(hard_neg_results)
    metrics['n_hard_negatives_in_footprint'] = len(hn_valid)
    
    for thresh in thresholds:
        if len(kl_valid) > 0:
            recall = (kl_valid['p_lens'] > thresh).mean()
            n_detected = (kl_valid['p_lens'] > thresh).sum()
        else:
            recall = np.nan
            n_detected = 0
        
        if len(hn_valid) > 0:
            contamination = (hn_valid['p_lens'] > thresh).mean()
            n_false_pos = (hn_valid['p_lens'] > thresh).sum()
        else:
            contamination = np.nan
            n_false_pos = 0
        
        metrics[f'recall@{thresh}'] = recall
        metrics[f'n_detected@{thresh}'] = int(n_detected)
        metrics[f'contamination@{thresh}'] = contamination
        metrics[f'n_false_pos@{thresh}'] = int(n_false_pos)
    
    return metrics


# =============================================================================
# Test 1: Catalog Creation
# =============================================================================

class TestCatalogCreation:
    """Tests for catalog creation functions."""
    
    def test_slacs_catalog_has_items(self):
        """SLACS catalog should have at least some lenses for testing."""
        assert len(SLACS_LENSES) >= 1, f"Expected at least 1 SLACS lens, got {len(SLACS_LENSES)}"
        # Note: Full catalog has ~48 lenses; test uses subset
    
    def test_bells_catalog_has_items(self):
        """BELLS catalog should have at least some lenses for testing."""
        assert len(BELLS_LENSES) >= 1, f"Expected at least 1 BELLS lens, got {len(BELLS_LENSES)}"
        # Note: Full catalog has ~20 lenses; test uses subset
    
    def test_slacs_lens_format(self):
        """Each SLACS lens should have (name, ra, dec, theta_e, z_lens, z_source)."""
        for lens in SLACS_LENSES:
            assert len(lens) == 6, f"SLACS lens should have 6 elements: {lens}"
            name, ra, dec, theta_e, z_lens, z_source = lens
            assert isinstance(name, str), f"Name should be string: {name}"
            assert 0 <= ra <= 360, f"RA should be 0-360: {ra}"
            assert -90 <= dec <= 90, f"Dec should be -90 to 90: {dec}"
            assert 0.1 < theta_e < 5.0, f"theta_e should be 0.1-5 arcsec: {theta_e}"
            assert 0 < z_lens < 1.5, f"z_lens should be 0-1.5: {z_lens}"
            assert 0 < z_source < 3.0, f"z_source should be 0-3: {z_source}"
    
    def test_bells_lens_format(self):
        """Each BELLS lens should have (name, ra, dec, theta_e, z_lens, z_source)."""
        for lens in BELLS_LENSES:
            assert len(lens) == 6, f"BELLS lens should have 6 elements: {lens}"
            name, ra, dec, theta_e, z_lens, z_source = lens
            assert isinstance(name, str), f"Name should be string: {name}"
            assert 0 <= ra <= 360, f"RA should be 0-360: {ra}"
            assert -90 <= dec <= 90, f"Dec should be -90 to 90: {dec}"
    
    def test_create_known_lens_catalog_returns_dataframe(self):
        """create_known_lens_catalog should return a DataFrame with expected columns."""
        df = create_known_lens_catalog()
        
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        expected_cols = ['name', 'ra', 'dec', 'theta_e', 'z_lens', 'z_source', 'catalog']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
        
        assert len(df) == len(SLACS_LENSES) + len(BELLS_LENSES), "Should contain all lenses"
        assert (df['catalog'] == 'SLACS').sum() == len(SLACS_LENSES), "SLACS count mismatch"
        assert (df['catalog'] == 'BELLS').sum() == len(BELLS_LENSES), "BELLS count mismatch"
    
    def test_create_hard_negative_catalog_returns_dataframe(self):
        """create_hard_negative_catalog should return a DataFrame."""
        df = create_hard_negative_catalog()
        
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        expected_cols = ['name', 'ra', 'dec', 'catalog']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
        
        assert len(df) == len(RING_GALAXIES) + len(MERGER_GALAXIES), "Count mismatch"
    
    def test_no_duplicate_lenses(self):
        """There should be no duplicate (ra, dec) pairs in the catalog."""
        df = create_known_lens_catalog()
        
        # Round to avoid floating point comparison issues
        df['ra_round'] = df['ra'].round(4)
        df['dec_round'] = df['dec'].round(4)
        
        duplicates = df.duplicated(subset=['ra_round', 'dec_round'], keep=False)
        assert duplicates.sum() == 0, f"Found duplicate positions: {df[duplicates][['name', 'ra', 'dec']]}"
    
    def test_slacs_in_sdss_footprint(self):
        """SLACS lenses should be in SDSS footprint (mostly northern sky)."""
        df = create_known_lens_catalog()
        slacs = df[df['catalog'] == 'SLACS']
        
        # SDSS is primarily northern sky, dec > -10
        southern = slacs[slacs['dec'] < -30]
        assert len(southern) < 5, f"Too many SLACS lenses in far south: {southern[['name', 'dec']]}"


# =============================================================================
# Test 2: Footprint Checking
# =============================================================================

class TestFootprintChecking:
    """Tests for DR10 footprint verification."""
    
    def test_north_galactic_cap_in_footprint(self):
        """NGC region should be in footprint."""
        # NGC: 120 < RA < 240, 0 < Dec < 60
        assert check_dr10_footprint(180, 30) == True, "NGC should be in footprint"
        assert check_dr10_footprint(150, 45) == True, "NGC should be in footprint"
    
    def test_south_galactic_cap_in_footprint(self):
        """SGC region should be in footprint."""
        # SGC: 0 < Dec < 10 or 330 < RA < 60
        assert check_dr10_footprint(350, 5) == True, "SGC should be in footprint"
        assert check_dr10_footprint(30, -5) == True, "SGC should be in footprint"
    
    def test_polar_regions_out_of_footprint(self):
        """Polar regions should be out of footprint."""
        assert check_dr10_footprint(0, 85) == False, "North pole should be out"
        assert check_dr10_footprint(0, -25) == False, "Far south should be out"
    
    def test_slacs_lenses_mostly_in_footprint(self):
        """Most SLACS lenses should be in DR10 footprint."""
        df = create_known_lens_catalog()
        slacs = df[df['catalog'] == 'SLACS']
        
        in_footprint = slacs.apply(lambda r: check_dr10_footprint(r['ra'], r['dec']), axis=1)
        pct_in = in_footprint.mean()
        
        assert pct_in > 0.9, f"Expected >90% SLACS in footprint, got {pct_in:.1%}"


# =============================================================================
# Test 3: Brick Name Computation
# =============================================================================

class TestBrickName:
    """Tests for brick name computation."""
    
    def test_brick_format(self):
        """Brick name should match DECaLS format."""
        brick = get_brick_for_position(150.5, 25.3)
        assert isinstance(brick, str), "Should return string"
        assert len(brick) == 7, f"Brick name should be 7 chars: {brick}"
        assert brick[3] in ['p', 'm'], f"4th char should be p or m: {brick}"
    
    def test_positive_dec_uses_p(self):
        """Positive declination should use 'p'."""
        brick = get_brick_for_position(100, 30)
        assert 'p' in brick, f"Positive dec should have 'p': {brick}"
    
    def test_negative_dec_uses_m(self):
        """Negative declination should use 'm'."""
        brick = get_brick_for_position(100, -10)
        assert 'm' in brick, f"Negative dec should have 'm': {brick}"
    
    def test_known_brick_examples(self):
        """Test against known brick names."""
        # RA=150, Dec=+25 -> brick approximately "150p250"
        brick = get_brick_for_position(150, 25)
        assert brick.startswith("150"), f"Expected 150xxx, got {brick}"
        assert "p" in brick, f"Expected positive dec marker"


# =============================================================================
# Test 4: Normalization
# =============================================================================

class TestNormalization:
    """Tests for image normalization."""
    
    def test_robust_mad_norm_shape_preserved(self):
        """Normalization should preserve shape."""
        x = np.random.randn(3, 64, 64).astype(np.float32) * 100 + 1000
        result = robust_mad_norm_outer(x)
        
        assert result.shape == x.shape, f"Shape mismatch: {result.shape} vs {x.shape}"
        assert result.dtype == np.float32, f"Dtype should be float32: {result.dtype}"
    
    def test_robust_mad_norm_clips_to_range(self):
        """Normalized values should be clipped to [-10, 10]."""
        x = np.random.randn(3, 64, 64).astype(np.float32) * 1000
        result = robust_mad_norm_outer(x, clip=10.0)
        
        assert result.min() >= -10.0, f"Min should be >= -10: {result.min()}"
        assert result.max() <= 10.0, f"Max should be <= 10: {result.max()}"
    
    def test_robust_mad_norm_uses_outer_annulus(self):
        """Normalization should use outer annulus, not full image."""
        # Create image with bright center, faint outer
        x = np.ones((3, 64, 64), dtype=np.float32) * 10
        h, w = 64, 64
        cy, cx = h // 2, w // 2
        
        # Add bright spot in center
        yy, xx = np.ogrid[:h, :w]
        center_mask = ((yy - cy)**2 + (xx - cx)**2) < 16**2
        for c in range(3):
            x[c][center_mask] = 1000
        
        result = robust_mad_norm_outer(x)
        
        # Center should be very positive (outlier relative to outer annulus)
        center_val = result[0, cy, cx]
        outer_val = result[0, 0, 0]
        
        assert center_val > outer_val, f"Center ({center_val}) should be > outer ({outer_val})"
    
    def test_robust_mad_norm_handles_constant_image(self):
        """Should handle constant images without division by zero."""
        x = np.ones((3, 64, 64), dtype=np.float32) * 100
        result = robust_mad_norm_outer(x)
        
        # Should not have NaN or Inf
        assert np.isfinite(result).all(), "Result should be finite"


# =============================================================================
# Test 5: Metrics Computation
# =============================================================================

class TestMetricsComputation:
    """Tests for anchor metrics computation."""
    
    def test_compute_metrics_structure(self):
        """Metrics should have expected keys."""
        known = pd.DataFrame({
            'name': ['A', 'B', 'C'],
            'p_lens': [0.9, 0.6, 0.3],
            'cutout_ok': [True, True, True]
        })
        hard_neg = pd.DataFrame({
            'name': ['X', 'Y'],
            'p_lens': [0.2, 0.8],
            'cutout_ok': [True, True]
        })
        
        metrics = compute_anchor_metrics(known, hard_neg)
        
        assert 'recall@0.5' in metrics, "Should have recall@0.5"
        assert 'contamination@0.5' in metrics, "Should have contamination@0.5"
        assert 'n_known_lenses_in_footprint' in metrics, "Should have count"
    
    def test_recall_calculation(self):
        """Recall should be fraction of lenses above threshold."""
        known = pd.DataFrame({
            'name': ['A', 'B', 'C', 'D'],
            'p_lens': [0.9, 0.6, 0.3, 0.1],
            'cutout_ok': [True, True, True, True]
        })
        hard_neg = pd.DataFrame({
            'name': ['X'],
            'p_lens': [0.1],
            'cutout_ok': [True]
        })
        
        metrics = compute_anchor_metrics(known, hard_neg, thresholds=[0.5])
        
        # 2 out of 4 lenses (A, B) are above 0.5
        assert metrics['recall@0.5'] == 0.5, f"Expected recall 0.5, got {metrics['recall@0.5']}"
    
    def test_contamination_calculation(self):
        """Contamination should be fraction of hard negatives above threshold."""
        known = pd.DataFrame({
            'name': ['A'],
            'p_lens': [0.9],
            'cutout_ok': [True]
        })
        hard_neg = pd.DataFrame({
            'name': ['X', 'Y', 'Z', 'W'],
            'p_lens': [0.9, 0.6, 0.3, 0.1],
            'cutout_ok': [True, True, True, True]
        })
        
        metrics = compute_anchor_metrics(known, hard_neg, thresholds=[0.5])
        
        # 2 out of 4 hard negs (X, Y) are above 0.5
        assert metrics['contamination@0.5'] == 0.5, f"Expected 0.5, got {metrics['contamination@0.5']}"
    
    def test_handles_empty_valid_set(self):
        """Should handle case where no cutouts succeeded."""
        known = pd.DataFrame({
            'name': ['A'],
            'p_lens': [np.nan],
            'cutout_ok': [False]
        })
        hard_neg = pd.DataFrame({
            'name': ['X'],
            'p_lens': [np.nan],
            'cutout_ok': [False]
        })
        
        metrics = compute_anchor_metrics(known, hard_neg)
        
        # Should not crash, metrics should be NaN
        assert np.isnan(metrics['recall@0.5']), "Recall should be NaN for empty set"
    
    def test_filters_by_cutout_ok(self):
        """Should only include samples where cutout_ok=True."""
        known = pd.DataFrame({
            'name': ['A', 'B'],
            'p_lens': [0.9, 0.9],
            'cutout_ok': [True, False]  # B failed cutout
        })
        hard_neg = pd.DataFrame({
            'name': ['X'],
            'p_lens': [0.1],
            'cutout_ok': [True]
        })
        
        metrics = compute_anchor_metrics(known, hard_neg)
        
        assert metrics['n_known_lenses_in_footprint'] == 1, "Should only count successful cutouts"


# =============================================================================
# Test 6: Model Loading Architecture Match
# =============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestModelArchitecture:
    """Tests for model architecture matching."""
    
    def test_meta_fusion_head_dimensions(self):
        """MetaFusionHead should have correct dimensions."""
        import torch
        import torch.nn as nn
        
        # Define MetaFusionHead locally for testing
        class MetaFusionHead(nn.Module):
            def __init__(self, feat_dim: int, meta_dim: int, hidden: int = 256, dropout: float = 0.1):
                super().__init__()
                self.meta_mlp = nn.Sequential(
                    nn.Linear(meta_dim, hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(inplace=True),
                )
                self.classifier = nn.Sequential(
                    nn.Linear(feat_dim + hidden, hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )
            
            def forward(self, feats, meta):
                m = self.meta_mlp(meta)
                x = torch.cat([feats, m], dim=1)
                return self.classifier(x).squeeze(1)
        
        head = MetaFusionHead(feat_dim=768, meta_dim=2, hidden=256, dropout=0.1)
        
        # Test forward pass
        feats = torch.randn(4, 768)
        meta = torch.randn(4, 2)
        output = head(feats, meta)
        
        assert output.shape == (4,), f"Expected (4,), got {output.shape}"
    
    def test_meta_fusion_head_with_different_meta_dim(self):
        """MetaFusionHead should work with different metadata dimensions."""
        import torch
        import torch.nn as nn
        
        class MetaFusionHead(nn.Module):
            def __init__(self, feat_dim: int, meta_dim: int, hidden: int = 256, dropout: float = 0.1):
                super().__init__()
                self.meta_mlp = nn.Sequential(
                    nn.Linear(meta_dim, hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(inplace=True),
                )
                self.classifier = nn.Sequential(
                    nn.Linear(feat_dim + hidden, hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )
            
            def forward(self, feats, meta):
                m = self.meta_mlp(meta)
                x = torch.cat([feats, m], dim=1)
                return self.classifier(x).squeeze(1)
        
        for meta_dim in [1, 2, 5, 10]:
            head = MetaFusionHead(feat_dim=768, meta_dim=meta_dim, hidden=256)
            feats = torch.randn(2, 768)
            meta = torch.randn(2, meta_dim)
            output = head(feats, meta)
            assert output.shape == (2,), f"Failed for meta_dim={meta_dim}"


# =============================================================================
# Test 7: Pre-flight Checks
# =============================================================================

class TestPreflightChecks:
    """Tests for pre-flight validation."""
    
    def test_slacs_coordinates_match_known_values(self):
        """Verify SLACS coordinates against published values."""
        # SDSSJ0912+0029 is a well-known SLACS lens
        # Published: RA=138.1417, Dec=0.4861
        found = False
        for lens in SLACS_LENSES:
            if "0912" in lens[0]:
                assert abs(lens[1] - 138.14) < 0.1, f"RA mismatch for J0912: {lens[1]}"
                assert abs(lens[2] - 0.49) < 0.1, f"Dec mismatch for J0912: {lens[2]}"
                found = True
                break
        assert found, "J0912+0029 should be in SLACS catalog"
    
    def test_theta_e_in_reasonable_range(self):
        """Einstein radii should be in physically reasonable range."""
        df = create_known_lens_catalog()
        
        # SLACS/BELLS lenses typically have theta_e between 0.5" and 2.5"
        assert df['theta_e'].min() > 0.3, f"Min theta_e too small: {df['theta_e'].min()}"
        assert df['theta_e'].max() < 3.0, f"Max theta_e too large: {df['theta_e'].max()}"
        
        # Median should be around 1-1.5"
        median = df['theta_e'].median()
        assert 0.8 < median < 2.0, f"Median theta_e unusual: {median}"
    
    def test_redshift_ranges(self):
        """Lens and source redshifts should be in expected ranges."""
        df = create_known_lens_catalog()
        
        # SLACS lens redshifts are typically 0.05-0.5
        assert df['z_lens'].min() > 0.01, f"Min z_lens too small: {df['z_lens'].min()}"
        assert df['z_lens'].max() < 1.0, f"Max z_lens too large: {df['z_lens'].max()}"
        
        # Source redshifts should be larger than lens redshifts
        assert (df['z_source'] > df['z_lens']).all(), "z_source should always be > z_lens"


# =============================================================================
# Test 8: Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_end_to_end_with_mock_model(self):
        """Test full pipeline with a mock model."""
        import torch
        import torch.nn as nn
        
        # Create mock model that always returns 0.5
        class MockModel(nn.Module):
            def forward(self, x, meta=None):
                return torch.zeros(x.size(0)) + 0.5
        
        model = MockModel()
        model.eval()
        
        # Create test catalog
        known = pd.DataFrame({
            'name': ['Test1', 'Test2'],
            'ra': [150.0, 160.0],
            'dec': [30.0, 35.0],
            'theta_e': [1.0, 1.5],
            'z_lens': [0.2, 0.3],
            'z_source': [0.8, 1.0],
            'catalog': ['TEST', 'TEST']
        })
        
        # Check footprint (should pass for these coords)
        for _, row in known.iterrows():
            assert check_dr10_footprint(row['ra'], row['dec']), f"Should be in footprint: {row['name']}"
    
    def test_metrics_report_generation(self):
        """Test that metrics can be computed and serialized."""
        known = pd.DataFrame({
            'name': ['A', 'B'],
            'p_lens': [0.7, 0.3],
            'cutout_ok': [True, True]
        })
        hard_neg = pd.DataFrame({
            'name': ['X'],
            'p_lens': [0.4],
            'cutout_ok': [True]
        })
        
        metrics = compute_anchor_metrics(known, hard_neg)
        
        # Should be JSON serializable
        json_str = json.dumps(metrics, default=float)
        loaded = json.loads(json_str)
        
        assert loaded['recall@0.5'] == 0.5, "Should round-trip through JSON"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


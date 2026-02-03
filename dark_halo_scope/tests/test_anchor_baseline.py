"""
Comprehensive unit tests for anchor baseline validation pipeline.

Tests cover:
1. Catalog source definitions
2. Catalog download and parsing
3. DR10 footprint cross-matching
4. Anchor metric computation
5. Report generation
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments" / "external_catalogs"))


# =============================================================================
# Catalog Source Tests
# =============================================================================

class TestCatalogSources:
    """Tests for catalog source definitions."""
    
    def test_import_catalog_sources(self):
        """Test that catalog sources can be imported."""
        from experiments.external_catalogs.catalog_sources import (
            CatalogSource,
            CatalogType,
            CATALOG_REGISTRY,
            get_known_lens_catalogs,
            get_hard_negative_catalogs,
            get_catalog
        )
        assert CatalogSource is not None
        assert CatalogType is not None
    
    def test_catalog_registry_not_empty(self):
        """Test that catalog registry contains catalogs."""
        from experiments.external_catalogs.catalog_sources import CATALOG_REGISTRY
        assert len(CATALOG_REGISTRY) > 0
    
    def test_known_lens_catalogs(self):
        """Test that known lens catalogs are defined."""
        from experiments.external_catalogs.catalog_sources import get_known_lens_catalogs
        catalogs = get_known_lens_catalogs()
        assert len(catalogs) >= 2  # At least SLACS and BELLS
        
        # Check SLACS
        slacs = [c for c in catalogs if c.name == "SLACS"]
        assert len(slacs) == 1
        assert slacs[0].theta_e_column is not None
    
    def test_hard_negative_catalogs(self):
        """Test that hard negative catalogs are defined."""
        from experiments.external_catalogs.catalog_sources import get_hard_negative_catalogs
        catalogs = get_hard_negative_catalogs()
        assert len(catalogs) >= 2  # At least rings and mergers
    
    def test_get_catalog_by_name(self):
        """Test getting catalog by name."""
        from experiments.external_catalogs.catalog_sources import get_catalog
        
        slacs = get_catalog("slacs")
        assert slacs.name == "SLACS"
        
        bells = get_catalog("BELLS")  # Case insensitive
        assert bells.name == "BELLS"
    
    def test_get_catalog_invalid(self):
        """Test that invalid catalog name raises error."""
        from experiments.external_catalogs.catalog_sources import get_catalog
        
        with pytest.raises(ValueError, match="Unknown catalog"):
            get_catalog("nonexistent")
    
    def test_catalog_source_dataclass(self):
        """Test CatalogSource dataclass fields."""
        from experiments.external_catalogs.catalog_sources import CatalogSource, CatalogType
        
        source = CatalogSource(
            name="TestCatalog",
            catalog_type=CatalogType.KNOWN_LENS,
            description="Test catalog",
            ra_column="RA",
            dec_column="DEC"
        )
        
        assert source.name == "TestCatalog"
        assert source.catalog_type == CatalogType.KNOWN_LENS
        assert source.ra_column == "RA"


# =============================================================================
# Catalog Download Tests
# =============================================================================

class TestCatalogDownload:
    """Tests for catalog download functions."""
    
    def test_slacs_fallback(self):
        """Test SLACS fallback data."""
        from experiments.external_catalogs.download_catalogs import _create_slacs_fallback
        
        df = _create_slacs_fallback()
        
        assert len(df) > 0
        assert 'ra' in df.columns
        assert 'dec' in df.columns
        assert 'theta_e' in df.columns
        assert 'z_lens' in df.columns
        assert 'catalog' in df.columns
        
        # Check coordinate ranges
        assert df['ra'].min() >= 0
        assert df['ra'].max() <= 360
        assert df['dec'].min() >= -90
        assert df['dec'].max() <= 90
    
    def test_bells_fallback(self):
        """Test BELLS fallback data."""
        from experiments.external_catalogs.download_catalogs import _create_bells_fallback
        
        df = _create_bells_fallback()
        
        assert len(df) > 0
        assert 'ra' in df.columns
        assert 'dec' in df.columns
    
    def test_gz_rings_fallback(self):
        """Test Galaxy Zoo rings fallback data."""
        from experiments.external_catalogs.download_catalogs import _create_gz_rings_fallback
        
        df = _create_gz_rings_fallback()
        
        assert len(df) > 0
        assert 'ra' in df.columns
        assert 'dec' in df.columns
    
    def test_mergers_fallback(self):
        """Test mergers fallback data."""
        from experiments.external_catalogs.download_catalogs import _create_mergers_fallback
        
        df = _create_mergers_fallback()
        
        assert len(df) > 0
        assert 'ra' in df.columns
        assert 'dec' in df.columns
    
    def test_download_catalog_saves_parquet(self, tmp_path):
        """Test that download_catalog saves to parquet."""
        from experiments.external_catalogs.download_catalogs import download_catalog
        
        # This will use fallback data since VizieR is unavailable in tests
        output_dir = str(tmp_path / "slacs")
        df = download_catalog("slacs", output_dir)
        
        assert len(df) > 0
        assert os.path.exists(os.path.join(output_dir, "slacs.parquet"))
        
        # Verify saved file can be loaded
        loaded = pd.read_parquet(os.path.join(output_dir, "slacs.parquet"))
        assert len(loaded) == len(df)
    
    def test_merge_known_lenses(self, tmp_path):
        """Test merging known lens catalogs."""
        from experiments.external_catalogs.download_catalogs import (
            download_catalog, merge_known_lenses
        )
        
        output_dir = str(tmp_path)
        
        # Download both
        download_catalog("slacs", os.path.join(output_dir, "slacs"))
        download_catalog("bells", os.path.join(output_dir, "bells"))
        
        # Merge
        merged = merge_known_lenses(output_dir)
        
        assert len(merged) > 0
        assert os.path.exists(os.path.join(output_dir, "known_lenses_merged.parquet"))


# =============================================================================
# Cross-Match Tests
# =============================================================================

class TestCrossMatch:
    """Tests for DR10 footprint cross-matching."""
    
    @pytest.fixture
    def sample_catalog(self):
        """Create sample catalog for testing."""
        return pd.DataFrame({
            'ra': [120.0, 180.0, 240.0, 300.0],
            'dec': [30.0, -10.0, 45.0, -30.0],
            'name': ['obj1', 'obj2', 'obj3', 'obj4']
        })
    
    @pytest.fixture
    def sample_bricks(self):
        """Create sample brick metadata for testing."""
        return pd.DataFrame({
            'brickname': ['1234p300', '1800m100', '2400p450'],
            'ra': [123.4, 180.0, 240.0],
            'dec': [30.0, -10.0, 45.0],
            'ra1': [123.0, 179.5, 239.5],
            'ra2': [123.8, 180.5, 240.5],
            'dec1': [29.5, -10.5, 44.5],
            'dec2': [30.5, -9.5, 45.5]
        })
    
    def test_in_dr10_footprint(self, sample_catalog, sample_bricks):
        """Test footprint matching."""
        from experiments.external_catalogs.crossmatch_dr10 import in_dr10_footprint
        
        ra = sample_catalog['ra'].values
        dec = sample_catalog['dec'].values
        
        in_fp = in_dr10_footprint(ra, dec, sample_bricks)
        
        assert isinstance(in_fp, np.ndarray)
        assert len(in_fp) == len(sample_catalog)
        assert in_fp.dtype == bool
    
    def test_crossmatch_with_footprint(self, sample_catalog, sample_bricks):
        """Test cross-match function."""
        from experiments.external_catalogs.crossmatch_dr10 import crossmatch_with_footprint
        
        matched = crossmatch_with_footprint(sample_catalog, sample_bricks)
        
        assert isinstance(matched, pd.DataFrame)
        assert 'in_dr10_footprint' in matched.columns
        assert len(matched) <= len(sample_catalog)
    
    def test_crossmatch_config_defaults(self):
        """Test CrossMatchConfig defaults."""
        from experiments.external_catalogs.crossmatch_dr10 import CrossMatchConfig
        
        config = CrossMatchConfig()
        
        assert config.stamp_size == 64
        assert config.pixel_scale == 0.262
        assert config.bands == ["g", "r", "z"]
    
    def test_load_brick_metadata(self, tmp_path):
        """Test loading brick metadata CSV."""
        from experiments.external_catalogs.crossmatch_dr10 import load_brick_metadata
        
        # Create sample CSV
        df = pd.DataFrame({
            'brickname': ['1234p300', '1800m100'],
            'ra': [123.4, 180.0],
            'dec': [30.0, -10.0],
            'ra1': [123.0, 179.5],
            'ra2': [123.8, 180.5],
            'dec1': [29.5, -10.5],
            'dec2': [30.5, -9.5]
        })
        csv_path = str(tmp_path / "bricks.csv")
        df.to_csv(csv_path, index=False)
        
        loaded = load_brick_metadata(csv_path)
        
        assert len(loaded) == 2
        assert 'brickname' in loaded.columns


# =============================================================================
# Anchor Metrics Tests
# =============================================================================

class TestAnchorMetrics:
    """Tests for anchor metric computation."""
    
    def test_compute_metrics_basic(self):
        """Test basic metric computation."""
        from experiments.external_catalogs.compute_anchor_metrics import compute_metrics
        
        # Mock perfect separation
        known_scores = np.array([0.9, 0.95, 0.85, 0.92, 0.88])
        hard_neg_scores = np.array([0.1, 0.15, 0.05, 0.12, 0.08])
        
        metrics = compute_metrics(known_scores, hard_neg_scores)
        
        assert metrics.n_known_lenses == 5
        assert metrics.n_hard_negatives == 5
        assert metrics.recovery_rate_0p5 == 1.0  # All known lenses > 0.5
        assert metrics.contamination_rate_0p5 == 0.0  # No hard negs > 0.5
        assert metrics.score_separation > 0.5
        assert metrics.auroc_anchor > 0.9
    
    def test_compute_metrics_poor_model(self):
        """Test metrics with poor model (no separation)."""
        from experiments.external_catalogs.compute_anchor_metrics import compute_metrics
        
        # No separation
        known_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        hard_neg_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        metrics = compute_metrics(known_scores, hard_neg_scores)
        
        assert metrics.score_separation == 0.0
        assert metrics.auroc_anchor == 0.5  # Random
    
    def test_anchor_metrics_gate(self):
        """Test anchor metrics gate evaluation."""
        from experiments.external_catalogs.compute_anchor_metrics import AnchorMetrics
        
        # Good metrics - should pass
        good_metrics = AnchorMetrics(
            n_known_lenses=100,
            n_recovered_at_0p5=80,
            n_recovered_at_0p9=60,
            recovery_rate_0p5=0.80,
            recovery_rate_0p9=0.60,
            n_hard_negatives=100,
            n_contaminated_at_0p5=10,
            n_contaminated_at_0p9=2,
            contamination_rate_0p5=0.10,
            contamination_rate_0p9=0.02,
            known_lens_score_median=0.85,
            known_lens_score_mean=0.82,
            known_lens_score_std=0.1,
            hard_neg_score_median=0.20,
            hard_neg_score_mean=0.25,
            hard_neg_score_std=0.15,
            score_separation=0.65,
            auroc_anchor=0.92,
            passes_anchor_gate=True
        )
        
        assert good_metrics.passes_anchor_gate is True
        assert good_metrics.gate_criteria['recovery_0p5_gt_70'] is True
        assert good_metrics.gate_criteria['contamination_0p5_lt_15'] is True
    
    def test_anchor_metrics_gate_fail(self):
        """Test anchor metrics gate failure."""
        from experiments.external_catalogs.compute_anchor_metrics import AnchorMetrics
        
        # Poor metrics - should fail
        poor_metrics = AnchorMetrics(
            n_known_lenses=100,
            n_recovered_at_0p5=40,  # Only 40% recovery
            n_recovered_at_0p9=10,
            recovery_rate_0p5=0.40,
            recovery_rate_0p9=0.10,
            n_hard_negatives=100,
            n_contaminated_at_0p5=30,  # 30% contamination
            n_contaminated_at_0p9=10,
            contamination_rate_0p5=0.30,
            contamination_rate_0p9=0.10,
            known_lens_score_median=0.55,
            known_lens_score_mean=0.50,
            known_lens_score_std=0.2,
            hard_neg_score_median=0.45,
            hard_neg_score_mean=0.40,
            hard_neg_score_std=0.2,
            score_separation=0.10,
            auroc_anchor=0.65,
            passes_anchor_gate=False
        )
        
        assert poor_metrics.passes_anchor_gate is False
        assert poor_metrics.gate_criteria['recovery_0p5_gt_70'] is False
        assert poor_metrics.gate_criteria['contamination_0p5_lt_15'] is False
    
    def test_anchor_metrics_json_roundtrip(self, tmp_path):
        """Test JSON serialization roundtrip."""
        from experiments.external_catalogs.compute_anchor_metrics import AnchorMetrics
        
        metrics = AnchorMetrics(
            n_known_lenses=50,
            n_recovered_at_0p5=40,
            n_recovered_at_0p9=30,
            recovery_rate_0p5=0.80,
            recovery_rate_0p9=0.60,
            n_hard_negatives=100,
            n_contaminated_at_0p5=10,
            n_contaminated_at_0p9=2,
            contamination_rate_0p5=0.10,
            contamination_rate_0p9=0.02,
            known_lens_score_median=0.85,
            known_lens_score_mean=0.82,
            known_lens_score_std=0.1,
            hard_neg_score_median=0.20,
            hard_neg_score_mean=0.25,
            hard_neg_score_std=0.15,
            score_separation=0.65,
            auroc_anchor=0.90,
            passes_anchor_gate=True
        )
        
        json_path = str(tmp_path / "metrics.json")
        metrics.to_json(json_path)
        
        loaded = AnchorMetrics.from_json(json_path)
        
        assert loaded.n_known_lenses == metrics.n_known_lenses
        assert loaded.recovery_rate_0p5 == metrics.recovery_rate_0p5
        assert loaded.passes_anchor_gate == metrics.passes_anchor_gate


# =============================================================================
# Report Generation Tests
# =============================================================================

class TestReportGeneration:
    """Tests for report generation."""
    
    def test_generate_report_creates_files(self, tmp_path):
        """Test that report generation creates expected files."""
        from experiments.external_catalogs.compute_anchor_metrics import (
            AnchorMetrics, generate_report
        )
        
        metrics = AnchorMetrics(
            n_known_lenses=50,
            n_recovered_at_0p5=40,
            n_recovered_at_0p9=30,
            recovery_rate_0p5=0.80,
            recovery_rate_0p9=0.60,
            n_hard_negatives=100,
            n_contaminated_at_0p5=10,
            n_contaminated_at_0p9=2,
            contamination_rate_0p5=0.10,
            contamination_rate_0p9=0.02,
            known_lens_score_median=0.85,
            known_lens_score_mean=0.82,
            known_lens_score_std=0.1,
            hard_neg_score_median=0.20,
            hard_neg_score_mean=0.25,
            hard_neg_score_std=0.15,
            score_separation=0.65,
            auroc_anchor=0.90,
            passes_anchor_gate=True
        )
        
        output_dir = str(tmp_path / "report")
        report_path = generate_report(metrics, output_dir)
        
        assert os.path.exists(report_path)
        assert os.path.exists(os.path.join(output_dir, "anchor_metrics.json"))
        
        # Check report content
        with open(report_path, 'r') as f:
            content = f.read()
        
        assert "Anchor Baseline Validation Report" in content
        assert "PASSES GATE: YES" in content or "passes" in content.lower()


# =============================================================================
# Cutout Loading Tests
# =============================================================================

class TestCutoutLoading:
    """Tests for cutout loading functions."""
    
    def test_load_cutouts_empty_dir(self, tmp_path):
        """Test loading from empty directory."""
        from experiments.external_catalogs.compute_anchor_metrics import load_cutouts
        
        stamps, metadata = load_cutouts(str(tmp_path))
        
        assert len(stamps) == 0
        assert len(metadata) == 0
    
    def test_load_cutouts_with_data(self, tmp_path):
        """Test loading cutouts from directory."""
        from experiments.external_catalogs.compute_anchor_metrics import load_cutouts
        
        # Create sample cutout files
        for i in range(5):
            stamp = np.random.rand(3, 64, 64).astype(np.float32)
            np.savez_compressed(
                str(tmp_path / f"cutout_{i:06d}.npz"),
                stamp=stamp,
                ra=120.0 + i,
                dec=30.0 + i
            )
        
        stamps, metadata = load_cutouts(str(tmp_path))
        
        assert stamps.shape == (5, 3, 64, 64)
        assert len(metadata) == 5
        assert 'ra' in metadata.columns
        assert 'dec' in metadata.columns


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_download_and_merge(self, tmp_path):
        """Test full download and merge workflow."""
        from experiments.external_catalogs.download_catalogs import (
            download_all_catalogs,
            merge_known_lenses,
            merge_hard_negatives
        )
        
        output_dir = str(tmp_path)
        
        # Download all (will use fallbacks)
        results = download_all_catalogs(output_dir)
        
        assert 'slacs' in results
        assert 'bells' in results
        assert len(results['slacs']) > 0
        
        # Merge
        known_lenses = merge_known_lenses(output_dir)
        hard_negs = merge_hard_negatives(output_dir)
        
        assert len(known_lenses) > 0
        assert os.path.exists(os.path.join(output_dir, "known_lenses_merged.parquet"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


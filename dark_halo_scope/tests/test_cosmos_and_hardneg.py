"""
Tests for COSMOS loader v2 and hard negative collection.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# COSMOS Loader V2 Tests
# =============================================================================

class TestCOSMOSLoaderV2:
    """Tests for enhanced COSMOS loader."""
    
    def test_synthetic_mode_init(self):
        """Test initialization in synthetic mode."""
        from src.sims.cosmos_loader_v2 import COSMOSLoaderV2
        
        loader = COSMOSLoaderV2(mode='synthetic')
        
        assert loader.mode == 'synthetic'
        assert loader.n_sources > 0
        assert loader.images is not None
    
    def test_get_source_random(self):
        """Test getting a random source."""
        from src.sims.cosmos_loader_v2 import COSMOSLoaderV2
        
        loader = COSMOSLoaderV2(mode='synthetic', seed=42)
        source = loader.get_source()
        
        assert source.image is not None
        assert source.image.shape[0] > 0
        assert source.pixel_scale > 0
        assert 0 <= source.index < loader.n_sources
    
    def test_get_source_by_index(self):
        """Test getting a specific source by index."""
        from src.sims.cosmos_loader_v2 import COSMOSLoaderV2
        
        loader = COSMOSLoaderV2(mode='synthetic', seed=42)
        
        source0 = loader.get_source(index=0)
        source1 = loader.get_source(index=1)
        
        assert source0.index == 0
        assert source1.index == 1
        assert not np.allclose(source0.image, source1.image)
    
    def test_get_source_reproducible(self):
        """Test that source selection is reproducible with seed."""
        from src.sims.cosmos_loader_v2 import COSMOSLoaderV2
        
        loader = COSMOSLoaderV2(mode='synthetic', seed=42)
        
        source1 = loader.get_source(seed=100)
        source2 = loader.get_source(seed=100)
        
        assert source1.index == source2.index
        np.testing.assert_array_equal(source1.image, source2.image)
    
    def test_get_random_source_backward_compat(self):
        """Test backward compatibility with v1 API."""
        from src.sims.cosmos_loader_v2 import COSMOSLoaderV2
        
        loader = COSMOSLoaderV2(mode='synthetic', seed=42)
        
        image, clumpiness = loader.get_random_source(seed=100)
        
        assert isinstance(image, np.ndarray)
        assert isinstance(clumpiness, float)
    
    def test_get_source_by_clumpiness(self):
        """Test source selection by clumpiness range."""
        from src.sims.cosmos_loader_v2 import COSMOSLoaderV2
        
        loader = COSMOSLoaderV2(mode='synthetic', seed=42)
        
        # Get median clumpiness
        median_clump = np.median(loader.clumpiness)
        
        # Select high clumpiness
        source = loader.get_source_by_clumpiness(
            min_clumpiness=median_clump,
            seed=100
        )
        
        assert source.clumpiness >= median_clump
    
    def test_resample_to_pixscale(self):
        """Test resampling to different pixel scale."""
        from src.sims.cosmos_loader_v2 import COSMOSLoaderV2
        
        loader = COSMOSLoaderV2(mode='synthetic')
        source = loader.get_source(index=0)
        
        # Resample to coarser scale (DECaLS-like)
        resampled = loader.resample_to_pixscale(
            source.image,
            target_pixscale=0.262,
            source_pixscale=0.03
        )
        
        # Should be smaller
        assert resampled.shape[0] < source.image.shape[0]
    
    def test_clumpiness_computed(self):
        """Test that clumpiness is computed for all sources."""
        from src.sims.cosmos_loader_v2 import COSMOSLoaderV2
        
        loader = COSMOSLoaderV2(mode='synthetic', seed=42)
        
        assert loader.clumpiness is not None
        assert len(loader.clumpiness) == loader.n_sources
        assert all(c >= 0 for c in loader.clumpiness)
    
    def test_metadata_populated(self):
        """Test that metadata is populated for sources."""
        from src.sims.cosmos_loader_v2 import COSMOSLoaderV2
        
        loader = COSMOSLoaderV2(mode='synthetic', seed=42)
        source = loader.get_source(index=0)
        
        assert source.half_light_radius > 0
        assert source.magnitude > 0
        assert source.redshift > 0
    
    def test_hdf5_roundtrip(self, tmp_path):
        """Test saving and loading HDF5 format."""
        from src.sims.cosmos_loader_v2 import COSMOSLoaderV2
        import h5py
        
        # Create synthetic and save to HDF5
        loader1 = COSMOSLoaderV2(mode='synthetic', seed=42)
        
        h5_path = str(tmp_path / "cosmos_test.h5")
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('images', data=loader1.images)
            f.create_dataset('clumpiness', data=loader1.clumpiness)
        
        # Load from HDF5
        loader2 = COSMOSLoaderV2(cosmos_path=h5_path, mode='hdf5')
        
        assert loader2.n_sources == loader1.n_sources
        np.testing.assert_array_almost_equal(loader2.images, loader1.images)
    
    def test_cosmos_source_dataclass(self):
        """Test COSMOSSource dataclass."""
        from src.sims.cosmos_loader_v2 import COSMOSSource
        
        source = COSMOSSource(
            image=np.ones((64, 64)),
            pixel_scale=0.03,
            clumpiness=0.5,
            half_light_radius=0.3,
            magnitude=24.0,
            redshift=1.5,
            index=0
        )
        
        assert source.image.shape == (64, 64)
        assert source.pixel_scale == 0.03


# =============================================================================
# Hard Negative Collection Tests
# =============================================================================

class TestHardNegativeCollection:
    """Tests for hard negative collection."""
    
    @pytest.fixture
    def sample_scores(self, tmp_path):
        """Create sample model scores for testing."""
        df = pd.DataFrame({
            'ra': [100.0, 101.0, 102.0, 103.0, 104.0],
            'dec': [10.0, 11.0, 12.0, 13.0, 14.0],
            'score': [0.95, 0.90, 0.85, 0.50, 0.30],
            'label': [0, 0, 0, 0, 0]
        })
        path = str(tmp_path / "scores.parquet")
        df.to_parquet(path)
        return path
    
    @pytest.fixture
    def sample_ring_catalog(self, tmp_path):
        """Create sample ring galaxy catalog."""
        df = pd.DataFrame({
            'ra': [200.0, 201.0, 202.0],
            'dec': [20.0, 21.0, 22.0]
        })
        path = str(tmp_path / "rings.parquet")
        df.to_parquet(path)
        return path
    
    @pytest.fixture
    def sample_known_lenses(self, tmp_path):
        """Create sample known lens catalog."""
        df = pd.DataFrame({
            'ra': [100.0],  # Matches first score entry
            'dec': [10.0]
        })
        path = str(tmp_path / "known.parquet")
        df.to_parquet(path)
        return path
    
    def test_load_model_scores(self, sample_scores):
        """Test loading high-scoring negatives."""
        from experiments.external_catalogs.collect_real_hard_negatives import load_model_scores
        
        df = load_model_scores(sample_scores, min_score=0.8, top_k=10)
        
        assert len(df) == 3  # 3 entries with score >= 0.8
        assert 'source' in df.columns
        assert all(df['source'] == 'model_fp')
    
    def test_load_catalog_hard_negatives(self, sample_ring_catalog):
        """Test loading catalog hard negatives."""
        from experiments.external_catalogs.collect_real_hard_negatives import (
            load_catalog_hard_negatives
        )
        
        df = load_catalog_hard_negatives(
            sample_ring_catalog,
            source_name='ring_galaxy'
        )
        
        assert len(df) == 3
        assert all(df['source'] == 'ring_galaxy')
    
    def test_exclude_known_lenses(self, sample_scores, sample_known_lenses):
        """Test exclusion of known lenses."""
        from experiments.external_catalogs.collect_real_hard_negatives import (
            load_model_scores, exclude_known_lenses
        )
        
        df = load_model_scores(sample_scores, min_score=0.0)
        n_before = len(df)
        
        df = exclude_known_lenses(df, sample_known_lenses, exclusion_radius_arcsec=10.0)
        
        assert len(df) < n_before  # At least one excluded
    
    def test_deduplicate_by_position(self):
        """Test deduplication by position."""
        from experiments.external_catalogs.collect_real_hard_negatives import (
            deduplicate_by_position
        )
        
        # Create duplicates
        df = pd.DataFrame({
            'ra': [100.0, 100.0001, 200.0],  # First two are ~0.36 arcsec apart
            'dec': [10.0, 10.0001, 20.0],
            'source': ['a', 'b', 'c']
        })
        
        deduped = deduplicate_by_position(df, radius_arcsec=2.0)
        
        assert len(deduped) == 2  # First two merged
    
    def test_assign_weights(self):
        """Test weight assignment by source."""
        from experiments.external_catalogs.collect_real_hard_negatives import assign_weights
        
        df = pd.DataFrame({
            'ra': [100.0, 200.0, 300.0],
            'dec': [10.0, 20.0, 30.0],
            'source': ['model_fp', 'ring_galaxy', 'merger'],
            'weight': [1.0, 1.0, 1.0]
        })
        
        df = assign_weights(df)
        
        # model_fp should have highest weight
        assert df[df['source'] == 'model_fp']['weight'].iloc[0] > 1.0
    
    def test_collect_hard_negatives_full(self, tmp_path, sample_scores, sample_ring_catalog):
        """Test full collection pipeline."""
        from experiments.external_catalogs.collect_real_hard_negatives import (
            collect_hard_negatives, HardNegativeConfig
        )
        
        output_dir = str(tmp_path / "output")
        
        config = HardNegativeConfig(
            min_score_threshold=0.8,
            top_k_per_source=100
        )
        
        df = collect_hard_negatives(
            model_scores_path=sample_scores,
            ring_catalog_path=sample_ring_catalog,
            output_dir=output_dir,
            config=config
        )
        
        assert len(df) > 0
        assert os.path.exists(os.path.join(output_dir, 'hard_negatives_combined.parquet'))
        assert os.path.exists(os.path.join(output_dir, 'hard_negatives_summary.csv'))
    
    def test_create_training_lookup(self, tmp_path):
        """Test creating training lookup table."""
        from experiments.external_catalogs.collect_real_hard_negatives import (
            create_training_lookup
        )
        
        # Create sample hard negatives
        df = pd.DataFrame({
            'ra': [100.0, 200.0],
            'dec': [10.0, 20.0],
            'source': ['a', 'b'],
            'weight': [1.0, 2.0]
        })
        
        hn_path = str(tmp_path / "hn.parquet")
        df.to_parquet(hn_path)
        
        lookup_path = str(tmp_path / "lookup.parquet")
        create_training_lookup(hn_path, lookup_path)
        
        assert os.path.exists(lookup_path)
        
        lookup = pd.read_parquet(lookup_path)
        assert 'pos_hash' in lookup.columns
        assert 'weight' in lookup.columns


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_cosmos_to_training_pipeline(self, tmp_path):
        """Test COSMOS source generation for training."""
        from src.sims.cosmos_loader_v2 import COSMOSLoaderV2
        
        loader = COSMOSLoaderV2(mode='synthetic', seed=42)
        
        # Simulate getting sources for injection
        sources = []
        for i in range(10):
            source = loader.get_source(seed=i)
            
            # Resample to DECaLS-like resolution
            resampled = loader.resample_to_pixscale(
                source.image,
                target_pixscale=0.262,
                source_pixscale=source.pixel_scale
            )
            
            sources.append({
                'image': resampled,
                'clumpiness': source.clumpiness,
                'hlr': source.half_light_radius
            })
        
        assert len(sources) == 10
        assert all(s['image'].shape[0] > 0 for s in sources)
    
    def test_full_experiment_config_with_cosmos(self, tmp_path):
        """Test experiment config with COSMOS data variant."""
        from experiments.configs.experiment_schema import (
            ExperimentConfig, DataVariantConfig, create_experiment_config
        )
        import h5py
        
        # Create mock COSMOS library
        cosmos_path = str(tmp_path / "cosmos.h5")
        with h5py.File(cosmos_path, 'w') as f:
            f.create_dataset('images', data=np.random.rand(10, 64, 64))
            f.create_dataset('clumpiness', data=np.random.rand(10))
        
        # Create data variant config
        data_variant = DataVariantConfig(
            variant_name="v5_cosmos_source",
            description="Test COSMOS variant",
            phase3_parent_sample="/path/to/parent",
            phase4a_manifest="/path/to/manifest",
            phase4c_stamps="/path/to/stamps",
            psf_model="moffat",
            moffat_beta=3.5,
            source_mode="cosmos",
            cosmos_library_path=cosmos_path
        )
        
        # Create experiment config
        config = create_experiment_config(
            experiment_name="COSMOS Test",
            generation="gen5",
            data_variant=data_variant,
            output_base_dir=str(tmp_path)
        )
        
        assert config.data.source_mode == "cosmos"
        assert config.data.cosmos_library_path == cosmos_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


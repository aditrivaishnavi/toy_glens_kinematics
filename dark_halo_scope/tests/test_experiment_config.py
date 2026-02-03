"""
Comprehensive unit tests for experiment configuration schema.

Tests cover:
1. Config creation and validation
2. YAML serialization/deserialization
3. Seed reproducibility
4. Hash computation for tracking
5. Edge cases and error handling
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.configs.experiment_schema import (
    ExperimentConfig,
    SeedConfig,
    DataVariantConfig,
    ModelConfig,
    TrainingConfig,
    HardNegativeConfig,
    EvaluationConfig,
    set_all_seeds,
    get_git_info,
    create_experiment_config
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def valid_data_variant():
    """Create a valid data variant config for testing."""
    return DataVariantConfig(
        variant_name="v4_sota_moffat",
        description="Test variant with Moffat PSF",
        phase3_parent_sample="s3://test/phase3/parent.parquet",
        phase4a_manifest="s3://test/phase4a/manifest.parquet",
        phase4c_stamps="s3://test/phase4c/stamps/",
        psf_model="moffat",
        moffat_beta=3.5,
        source_mode="parametric",
        theta_e_range=[0.5, 2.5],
        control_type="unpaired"
    )


@pytest.fixture
def valid_experiment_config(valid_data_variant, tmp_path):
    """Create a valid experiment config for testing."""
    return ExperimentConfig(
        experiment_id="test_exp_001",
        experiment_name="Test Experiment",
        description="A test experiment",
        generation="gen3",
        data=valid_data_variant,
        output_dir=str(tmp_path / "output")
    )


# =============================================================================
# SeedConfig Tests
# =============================================================================

class TestSeedConfig:
    """Tests for SeedConfig class."""
    
    def test_default_seed_propagation(self):
        """Test that global seed propagates to numpy and torch seeds."""
        config = SeedConfig(global_seed=123)
        assert config.global_seed == 123
        assert config.numpy_seed == 123
        assert config.torch_seed == 123
    
    def test_explicit_seeds_override(self):
        """Test that explicit seeds override global seed."""
        config = SeedConfig(
            global_seed=100,
            numpy_seed=200,
            torch_seed=300
        )
        assert config.global_seed == 100
        assert config.numpy_seed == 200
        assert config.torch_seed == 300
    
    def test_cuda_deterministic_default(self):
        """Test CUDA deterministic is True by default."""
        config = SeedConfig(global_seed=42)
        assert config.cuda_deterministic is True


# =============================================================================
# DataVariantConfig Tests
# =============================================================================

class TestDataVariantConfig:
    """Tests for DataVariantConfig class."""
    
    def test_minimal_valid_config(self):
        """Test creating minimal valid data variant."""
        config = DataVariantConfig(
            variant_name="test",
            description="Test variant",
            phase3_parent_sample="/path/to/parent",
            phase4a_manifest="/path/to/manifest",
            phase4c_stamps="/path/to/stamps"
        )
        assert config.variant_name == "test"
        assert config.psf_model == "gaussian"  # default
        assert config.source_mode == "parametric"  # default
    
    def test_moffat_config(self):
        """Test Moffat PSF configuration."""
        config = DataVariantConfig(
            variant_name="moffat_test",
            description="Test with Moffat",
            phase3_parent_sample="/path",
            phase4a_manifest="/path",
            phase4c_stamps="/path",
            psf_model="moffat",
            moffat_beta=3.5
        )
        assert config.psf_model == "moffat"
        assert config.moffat_beta == 3.5
    
    def test_cosmos_source_config(self):
        """Test COSMOS source mode configuration."""
        config = DataVariantConfig(
            variant_name="cosmos_test",
            description="Test with COSMOS",
            phase3_parent_sample="/path",
            phase4a_manifest="/path",
            phase4c_stamps="/path",
            source_mode="cosmos",
            cosmos_library_path="/path/to/cosmos.h5"
        )
        assert config.source_mode == "cosmos"
        assert config.cosmos_library_path == "/path/to/cosmos.h5"
    
    def test_default_theta_e_range(self):
        """Test default theta_e range."""
        config = DataVariantConfig(
            variant_name="test",
            description="Test",
            phase3_parent_sample="/path",
            phase4a_manifest="/path",
            phase4c_stamps="/path"
        )
        assert config.theta_e_range == [0.5, 2.5]
    
    def test_default_bands(self):
        """Test default bands."""
        config = DataVariantConfig(
            variant_name="test",
            description="Test",
            phase3_parent_sample="/path",
            phase4a_manifest="/path",
            phase4c_stamps="/path"
        )
        assert config.bands == ["g", "r", "z"]


# =============================================================================
# ModelConfig Tests
# =============================================================================

class TestModelConfig:
    """Tests for ModelConfig class."""
    
    def test_default_model(self):
        """Test default model architecture."""
        config = ModelConfig()
        assert config.architecture == "convnext_tiny"
        assert config.dropout == 0.1
        assert config.pretrained is False
    
    def test_metadata_config(self):
        """Test metadata fusion configuration."""
        config = ModelConfig(
            use_metadata=True,
            metadata_columns=["psfsize_r", "psfdepth_r", "ebv"]
        )
        assert config.use_metadata is True
        assert len(config.metadata_columns) == 3


# =============================================================================
# TrainingConfig Tests
# =============================================================================

class TestTrainingConfig:
    """Tests for TrainingConfig class."""
    
    def test_default_training_params(self):
        """Test default training parameters."""
        config = TrainingConfig()
        assert config.epochs == 50
        assert config.batch_size == 256
        assert config.learning_rate == 3e-4
        assert config.loss_type == "focal"
    
    def test_focal_loss_params(self):
        """Test focal loss parameters."""
        config = TrainingConfig(
            loss_type="focal",
            focal_alpha=0.25,
            focal_gamma=2.0
        )
        assert config.focal_alpha == 0.25
        assert config.focal_gamma == 2.0
    
    def test_early_stopping_config(self):
        """Test early stopping configuration."""
        config = TrainingConfig(
            early_stopping_patience=5,
            early_stopping_metric="tpr_at_fpr1e-4"
        )
        assert config.early_stopping_patience == 5
        assert config.early_stopping_metric == "tpr_at_fpr1e-4"
    
    def test_amp_config(self):
        """Test mixed precision configuration."""
        config = TrainingConfig(
            use_amp=True,
            amp_dtype="bfloat16"
        )
        assert config.use_amp is True
        assert config.amp_dtype == "bfloat16"


# =============================================================================
# HardNegativeConfig Tests
# =============================================================================

class TestHardNegativeConfig:
    """Tests for HardNegativeConfig class."""
    
    def test_disabled_by_default(self):
        """Test hard negatives are disabled by default."""
        config = HardNegativeConfig()
        assert config.enabled is False
    
    def test_enabled_config(self):
        """Test enabled hard negative configuration."""
        config = HardNegativeConfig(
            enabled=True,
            hard_neg_path="/path/to/hard_negs.parquet",
            hard_neg_weight=5.0,
            min_score_threshold=0.9
        )
        assert config.enabled is True
        assert config.hard_neg_weight == 5.0
    
    def test_real_hard_negatives(self):
        """Test real hard negative sources configuration."""
        config = HardNegativeConfig(
            enabled=True,
            hard_neg_path="/path/to/combined.parquet",
            use_real_hard_negatives=True,
            ring_galaxy_catalog="/path/to/rings.parquet",
            merger_catalog="/path/to/mergers.parquet"
        )
        assert config.use_real_hard_negatives is True


# =============================================================================
# ExperimentConfig Tests
# =============================================================================

class TestExperimentConfig:
    """Tests for ExperimentConfig class."""
    
    def test_valid_config_creation(self, valid_experiment_config):
        """Test creating a valid experiment config."""
        assert valid_experiment_config.experiment_id == "test_exp_001"
        assert valid_experiment_config.generation == "gen3"
    
    def test_validation_fails_on_unset_variant(self, tmp_path):
        """Test validation fails when data variant is not set."""
        with pytest.raises(ValueError, match="data.variant_name must be set"):
            ExperimentConfig(
                experiment_id="test",
                experiment_name="Test",
                description="Test",
                generation="gen1",
                output_dir=str(tmp_path)
            )
    
    def test_validation_fails_on_moffat_without_beta(self, tmp_path):
        """Test validation fails when Moffat is used without beta."""
        data = DataVariantConfig(
            variant_name="test",
            description="Test",
            phase3_parent_sample="/path",
            phase4a_manifest="/path",
            phase4c_stamps="/path",
            psf_model="moffat",
            moffat_beta=None  # Missing!
        )
        with pytest.raises(ValueError, match="moffat_beta must be set"):
            ExperimentConfig(
                experiment_id="test",
                experiment_name="Test",
                description="Test",
                generation="gen1",
                data=data,
                output_dir=str(tmp_path)
            )
    
    def test_validation_fails_on_cosmos_without_path(self, tmp_path):
        """Test validation fails when COSMOS mode is used without library path."""
        data = DataVariantConfig(
            variant_name="test",
            description="Test",
            phase3_parent_sample="/path",
            phase4a_manifest="/path",
            phase4c_stamps="/path",
            source_mode="cosmos",
            cosmos_library_path=None  # Missing!
        )
        with pytest.raises(ValueError, match="cosmos_library_path must be set"):
            ExperimentConfig(
                experiment_id="test",
                experiment_name="Test",
                description="Test",
                generation="gen1",
                data=data,
                output_dir=str(tmp_path)
            )
    
    def test_validation_fails_on_hard_neg_without_path(self, valid_data_variant, tmp_path):
        """Test validation fails when hard negatives enabled without path."""
        hard_neg = HardNegativeConfig(enabled=True, hard_neg_path=None)
        with pytest.raises(ValueError, match="hard_neg_path must be set"):
            ExperimentConfig(
                experiment_id="test",
                experiment_name="Test",
                description="Test",
                generation="gen1",
                data=valid_data_variant,
                hard_negatives=hard_neg,
                output_dir=str(tmp_path)
            )
    
    def test_validation_fails_without_output_dir(self, valid_data_variant):
        """Test validation fails without output directory."""
        with pytest.raises(ValueError, match="output_dir must be set"):
            ExperimentConfig(
                experiment_id="test",
                experiment_name="Test",
                description="Test",
                generation="gen1",
                data=valid_data_variant,
                output_dir=""  # Empty!
            )
    
    def test_config_hash_deterministic(self, valid_experiment_config):
        """Test that config hash is deterministic."""
        hash1 = valid_experiment_config.compute_hash()
        hash2 = valid_experiment_config.compute_hash()
        assert hash1 == hash2
        assert len(hash1) == 12  # Truncated to 12 chars
    
    def test_config_hash_changes_with_params(self, valid_data_variant, tmp_path):
        """Test that config hash changes when parameters change."""
        config1 = ExperimentConfig(
            experiment_id="test1",
            experiment_name="Test",
            description="Test",
            generation="gen3",
            data=valid_data_variant,
            output_dir=str(tmp_path)
        )
        
        # Create config with different learning rate
        config2 = ExperimentConfig(
            experiment_id="test2",
            experiment_name="Test",
            description="Test",
            generation="gen3",
            data=valid_data_variant,
            output_dir=str(tmp_path),
            training=TrainingConfig(learning_rate=1e-4)
        )
        
        assert config1.compute_hash() != config2.compute_hash()


# =============================================================================
# YAML Serialization Tests
# =============================================================================

class TestYAMLSerialization:
    """Tests for YAML serialization/deserialization."""
    
    def test_yaml_roundtrip(self, valid_experiment_config, tmp_path):
        """Test that config survives YAML roundtrip."""
        yaml_path = str(tmp_path / "config.yaml")
        
        # Save
        valid_experiment_config.to_yaml(yaml_path)
        assert os.path.exists(yaml_path)
        
        # Load
        loaded = ExperimentConfig.from_yaml(yaml_path)
        
        # Compare key fields
        assert loaded.experiment_id == valid_experiment_config.experiment_id
        assert loaded.generation == valid_experiment_config.generation
        assert loaded.data.variant_name == valid_experiment_config.data.variant_name
        assert loaded.data.psf_model == valid_experiment_config.data.psf_model
        assert loaded.training.learning_rate == valid_experiment_config.training.learning_rate
    
    def test_yaml_preserves_nested_configs(self, valid_experiment_config, tmp_path):
        """Test that nested configs are preserved in YAML."""
        yaml_path = str(tmp_path / "config.yaml")
        valid_experiment_config.to_yaml(yaml_path)
        loaded = ExperimentConfig.from_yaml(yaml_path)
        
        # Check nested data config
        assert loaded.data.moffat_beta == valid_experiment_config.data.moffat_beta
        assert loaded.data.theta_e_range == valid_experiment_config.data.theta_e_range
        
        # Check nested training config
        assert loaded.training.epochs == valid_experiment_config.training.epochs
        assert loaded.training.focal_alpha == valid_experiment_config.training.focal_alpha
    
    def test_yaml_handles_optional_fields(self, valid_experiment_config, tmp_path):
        """Test that optional fields are handled correctly."""
        yaml_path = str(tmp_path / "config.yaml")
        valid_experiment_config.to_yaml(yaml_path)
        loaded = ExperimentConfig.from_yaml(yaml_path)
        
        # Check optional fields
        assert loaded.data.cosmos_library_path == valid_experiment_config.data.cosmos_library_path


# =============================================================================
# Seed Reproducibility Tests
# =============================================================================

class TestSeedReproducibility:
    """Tests for seed setting and reproducibility."""
    
    def test_set_all_seeds_numpy(self):
        """Test that numpy seeds are set correctly."""
        import numpy as np
        
        config = SeedConfig(global_seed=42)
        set_all_seeds(config)
        
        # Generate some random numbers
        random1 = np.random.rand(10)
        
        # Reset seeds and generate again
        set_all_seeds(config)
        random2 = np.random.rand(10)
        
        np.testing.assert_array_equal(random1, random2)
    
    def test_set_all_seeds_python(self):
        """Test that Python random seeds are set correctly."""
        import random
        
        config = SeedConfig(global_seed=42)
        set_all_seeds(config)
        
        random1 = [random.random() for _ in range(10)]
        
        set_all_seeds(config)
        random2 = [random.random() for _ in range(10)]
        
        assert random1 == random2
    
    def test_different_seeds_different_results(self):
        """Test that different seeds give different results."""
        import numpy as np
        
        set_all_seeds(SeedConfig(global_seed=42))
        random1 = np.random.rand(10)
        
        set_all_seeds(SeedConfig(global_seed=123))
        random2 = np.random.rand(10)
        
        assert not np.allclose(random1, random2)


# =============================================================================
# Git Info Tests
# =============================================================================

class TestGitInfo:
    """Tests for git info extraction."""
    
    def test_get_git_info_returns_dict(self):
        """Test that git info returns expected keys."""
        info = get_git_info()
        assert isinstance(info, dict)
        assert 'commit' in info
        assert 'branch' in info
        assert 'dirty' in info
    
    def test_git_info_in_repo(self):
        """Test git info when in a git repo."""
        info = get_git_info()
        # If we're in the dark_halo_scope repo, we should get info
        # If not, commit will be None (which is also valid)
        if info['commit'] is not None:
            assert len(info['commit']) == 40  # SHA-1 hash length


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateExperimentConfig:
    """Tests for create_experiment_config factory function."""
    
    def test_creates_unique_experiment_id(self, valid_data_variant, tmp_path):
        """Test that factory creates unique experiment IDs."""
        config1 = create_experiment_config(
            experiment_name="Test",
            generation="gen5",
            data_variant=valid_data_variant,
            output_base_dir=str(tmp_path)
        )
        
        config2 = create_experiment_config(
            experiment_name="Test",
            generation="gen5",
            data_variant=valid_data_variant,
            output_base_dir=str(tmp_path)
        )
        
        assert config1.experiment_id != config2.experiment_id
    
    def test_creates_output_directories(self, valid_data_variant, tmp_path):
        """Test that factory sets up output directories."""
        config = create_experiment_config(
            experiment_name="Test",
            generation="gen5",
            data_variant=valid_data_variant,
            output_base_dir=str(tmp_path)
        )
        
        assert config.output_dir != ""
        assert config.checkpoint_dir != ""
        assert config.log_dir != ""
    
    def test_applies_overrides(self, valid_data_variant, tmp_path):
        """Test that factory applies parameter overrides."""
        config = create_experiment_config(
            experiment_name="Test",
            generation="gen5",
            data_variant=valid_data_variant,
            output_base_dir=str(tmp_path),
            epochs=100,
            learning_rate=1e-5
        )
        
        assert config.training.epochs == 100
        assert config.training.learning_rate == 1e-5


# =============================================================================
# Reproducibility Info Tests
# =============================================================================

class TestReproducibilityInfo:
    """Tests for reproducibility information extraction."""
    
    def test_get_reproducibility_info(self, valid_experiment_config):
        """Test reproducibility info summary."""
        info = valid_experiment_config.get_reproducibility_info()
        
        assert 'config_hash' in info
        assert 'experiment_id' in info
        assert 'generation' in info
        assert 'data_variant' in info
        assert 'seeds' in info
        assert 'model' in info
        assert 'training' in info
    
    def test_reproducibility_info_contains_seeds(self, valid_experiment_config):
        """Test that reproducibility info contains all seeds."""
        info = valid_experiment_config.get_reproducibility_info()
        
        assert 'global' in info['seeds']
        assert 'numpy' in info['seeds']
        assert 'torch' in info['seeds']
        assert 'cuda_deterministic' in info['seeds']


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_metadata_columns(self, valid_data_variant, tmp_path):
        """Test config with empty metadata columns."""
        config = ExperimentConfig(
            experiment_id="test",
            experiment_name="Test",
            description="Test",
            generation="gen3",
            data=valid_data_variant,
            model=ModelConfig(use_metadata=False, metadata_columns=[]),
            output_dir=str(tmp_path)
        )
        assert config.model.metadata_columns == []
    
    def test_extreme_hyperparameters(self, valid_data_variant, tmp_path):
        """Test config with extreme hyperparameter values."""
        config = ExperimentConfig(
            experiment_id="test",
            experiment_name="Test",
            description="Test",
            generation="gen3",
            data=valid_data_variant,
            training=TrainingConfig(
                epochs=1000,
                batch_size=1,
                learning_rate=1e-10
            ),
            output_dir=str(tmp_path)
        )
        assert config.training.epochs == 1000
        assert config.training.batch_size == 1
    
    def test_to_dict(self, valid_experiment_config):
        """Test converting config to dictionary."""
        d = valid_experiment_config.to_dict()
        assert isinstance(d, dict)
        assert 'experiment_id' in d
        assert 'data' in d
        assert isinstance(d['data'], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


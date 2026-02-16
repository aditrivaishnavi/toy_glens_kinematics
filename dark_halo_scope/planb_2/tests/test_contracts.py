#!/usr/bin/env python3
"""
Contract Verification Tests

These tests verify that contracts are consistent across all phases.
This is critical for preventing integration failures.

Lesson Learned: L9.1 - Data format mismatches between assumed and actual

Usage:
    pytest test_contracts.py -v
    python test_contracts.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.constants import (
    STAMP_SIZE, STAMP_SHAPE, NUM_CHANNELS,
    CORE_RADIUS_PIX, CORE_SIZE_PIX, OUTER_RADIUS_PIX,
    CLIP_SIGMA, GATES, get_core_slice,
)
from shared.schema import (
    PARQUET_SCHEMA, ANCHOR_SCHEMA, CONTAMINANT_SCHEMA,
    STAMP_SCHEMA, BATCH_SCHEMA,
)


class TestConstantsContract:
    """Verify constants are self-consistent."""
    
    def test_stamp_shape_consistency(self):
        """STAMP_SHAPE must match (NUM_CHANNELS, STAMP_SIZE, STAMP_SIZE)."""
        expected = (NUM_CHANNELS, STAMP_SIZE, STAMP_SIZE)
        assert STAMP_SHAPE == expected, \
            f"STAMP_SHAPE {STAMP_SHAPE} != expected {expected}"
    
    def test_radii_ordering(self):
        """Core radius < Outer radius < stamp half-size."""
        half = STAMP_SIZE // 2
        assert CORE_RADIUS_PIX < OUTER_RADIUS_PIX, \
            f"Core {CORE_RADIUS_PIX} >= Outer {OUTER_RADIUS_PIX}"
        assert OUTER_RADIUS_PIX < half, \
            f"Outer {OUTER_RADIUS_PIX} >= half {half}"
    
    def test_core_slice_correct(self):
        """Core slice must be correct for default parameters."""
        core_slice = get_core_slice()
        
        # Should be centered
        assert core_slice.start + core_slice.stop == STAMP_SIZE, \
            "Core slice not centered"
        
        # Should have correct size
        size = core_slice.stop - core_slice.start
        assert size == CORE_SIZE_PIX, \
            f"Core slice size {size} != {CORE_SIZE_PIX}"
    
    def test_gate_thresholds_valid(self):
        """Gate thresholds must be in valid ranges."""
        assert 0 < GATES.auroc_synth_min < 1
        assert 0 < GATES.core_lr_auc_max < 1
        assert 0 < GATES.core_masked_drop_max < 1
        assert 0 < GATES.hardneg_auroc_min < 1
    
    def test_clip_sigma_positive(self):
        """Clip sigma must be positive."""
        assert CLIP_SIGMA > 0


class TestSchemaContract:
    """Verify schemas are consistent."""
    
    def test_parquet_required_columns(self):
        """Parquet schema must have stamp_npz and ctrl_stamp_npz."""
        assert "stamp_npz" in PARQUET_SCHEMA.required_columns
        assert "ctrl_stamp_npz" in PARQUET_SCHEMA.required_columns
    
    def test_stamp_schema_matches_constants(self):
        """Stamp schema must match constant values."""
        assert STAMP_SCHEMA.expected_shape == STAMP_SHAPE
    
    def test_batch_schema_has_required_keys(self):
        """Batch schema must have x and y."""
        assert "x" in BATCH_SCHEMA.required_keys
        assert "y" in BATCH_SCHEMA.required_keys
    
    def test_anchor_schema_has_position(self):
        """Anchor schema must have ra, dec, name."""
        assert "ra" in ANCHOR_SCHEMA.required_columns
        assert "dec" in ANCHOR_SCHEMA.required_columns
        assert "name" in ANCHOR_SCHEMA.required_columns
    
    def test_contaminant_schema_has_category(self):
        """Contaminant schema must have category."""
        assert "category" in CONTAMINANT_SCHEMA.required_columns
        assert "ring" in CONTAMINANT_SCHEMA.valid_categories
        assert "spiral" in CONTAMINANT_SCHEMA.valid_categories


class TestCrossPhaseConsistency:
    """Verify consistency across phase implementations."""
    
    def test_data_loader_uses_constants(self):
        """Data loader config should use shared constants."""
        from phase1_baseline.data_loader import DataConfig
        
        config = DataConfig(parquet_root="/tmp", split="train")
        
        # These should match constants
        assert config.outer_radius_pix == OUTER_RADIUS_PIX
        assert config.clip_sigma == CLIP_SIGMA
        assert config.core_dropout_radius == CORE_RADIUS_PIX
    
    def test_model_uses_constants(self):
        """Model config should use shared constants."""
        try:
            from phase1_baseline.model import ModelConfig
        except ImportError as e:
            # Skip if torchvision not installed
            if "torchvision" in str(e):
                print("  (skipped - torchvision not installed)")
                return
            raise
        
        config = ModelConfig()
        
        assert config.in_channels == NUM_CHANNELS
    
    def test_shared_utils_importable(self):
        """All shared utils must be importable."""
        from shared.utils import (
            decode_stamp_npz,
            validate_stamp,
            robust_normalize,
            azimuthal_shuffle,
            apply_core_dropout,
            create_radial_mask,
        )
        
        # Functions should be callable
        assert callable(decode_stamp_npz)
        assert callable(validate_stamp)
        assert callable(robust_normalize)
        assert callable(azimuthal_shuffle)
        assert callable(apply_core_dropout)
        assert callable(create_radial_mask)


class TestSchemaValidation:
    """Test schema validation functions."""
    
    def test_stamp_schema_validates_good_array(self):
        """Good array should pass validation."""
        good = np.random.randn(*STAMP_SHAPE).astype(np.float32)
        result = STAMP_SCHEMA.validate_array(good, "test")
        assert result["valid"], f"Should be valid: {result['errors']}"
    
    def test_stamp_schema_rejects_nan(self):
        """Array with NaN should fail validation."""
        bad = np.random.randn(*STAMP_SHAPE).astype(np.float32)
        bad[0, 32, 32] = np.nan
        result = STAMP_SCHEMA.validate_array(bad, "test")
        assert not result["valid"], "Should reject NaN"
        assert any("NaN" in e for e in result["errors"])
    
    def test_stamp_schema_rejects_inf(self):
        """Array with Inf should fail validation."""
        bad = np.random.randn(*STAMP_SHAPE).astype(np.float32)
        bad[0, 32, 32] = np.inf
        result = STAMP_SCHEMA.validate_array(bad, "test")
        assert not result["valid"], "Should reject Inf"
        assert any("Inf" in e for e in result["errors"])
    
    def test_batch_schema_validates_good_batch(self):
        """Good batch should pass validation."""
        batch = {
            "x": torch.randn(32, NUM_CHANNELS, STAMP_SIZE, STAMP_SIZE),
            "y": torch.randint(0, 2, (32,)).float(),
        }
        result = BATCH_SCHEMA.validate_batch(batch)
        assert result["valid"], f"Should be valid: {result['errors']}"
    
    def test_batch_schema_rejects_missing_x(self):
        """Batch without x should fail."""
        batch = {"y": torch.zeros(32)}
        result = BATCH_SCHEMA.validate_batch(batch)
        assert not result["valid"]
    
    def test_batch_schema_rejects_wrong_shape(self):
        """Batch with wrong shape should fail."""
        batch = {
            "x": torch.randn(32, 1, 64, 64),  # Wrong channels
            "y": torch.zeros(32),
        }
        result = BATCH_SCHEMA.validate_batch(batch)
        assert not result["valid"]


def run_all_tests():
    """Run all contract tests."""
    import traceback
    
    test_classes = [
        TestConstantsContract,
        TestSchemaContract,
        TestCrossPhaseConsistency,
        TestSchemaValidation,
    ]
    
    results = {"passed": 0, "failed": 0, "errors": []}
    
    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    results["passed"] += 1
                    print(f"  ✓ {test_class.__name__}.{method_name}")
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(
                        f"{test_class.__name__}.{method_name}: {e}"
                    )
                    print(f"  ✗ {test_class.__name__}.{method_name}: {e}")
    
    print(f"\nResults: {results['passed']} passed, {results['failed']} failed")
    return results["failed"] == 0


if __name__ == "__main__":
    print("="*60)
    print("CONTRACT VERIFICATION TESTS")
    print("="*60)
    
    success = run_all_tests()
    
    if success:
        print("\n✓ ALL CONTRACT TESTS PASSED")
        sys.exit(0)
    else:
        print("\n✗ CONTRACT TESTS FAILED")
        sys.exit(1)

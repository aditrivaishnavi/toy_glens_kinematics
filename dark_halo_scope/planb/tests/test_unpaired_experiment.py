#!/usr/bin/env python3
"""
Unit Tests for Unpaired Experiment Module

Comprehensive tests for the unpaired training pipeline including:
- Preprocessing functions
- Data loading
- Shortcut gates
- Scheduled masking
- θ_E stratification
- Manifest building

Best Practices Applied:
- L1.2: Import from shared module for consistency
- L4.3: Test actual code path, not mock
- L5.1: Test edge cases (NaN, empty, extreme values)
- L6.2: Reproducibility with fixed seeds

Usage:
    pytest tests/test_unpaired_experiment.py -v
    python -m planb.tests.test_unpaired_experiment
"""
from __future__ import annotations
import io
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# pytest is optional - allows running tests standalone
try:
    import pytest
except ImportError:
    pytest = None

# Ensure planb is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from planb.unpaired_experiment.constants import (
    STAMP_SIZE, BANDS, CORE_BOX, SEED_DEFAULT, EARLY_STOPPING_PATIENCE,
)
from planb.unpaired_experiment.utils import (
    decode_npz_blob, radial_rmap, radial_mask, center_slices,
    robust_median_mad, normalize_outer_annulus, azimuthal_median_profile,
    radial_profile_model,
)
from planb.unpaired_experiment.preprocess import preprocess_stack
from planb.unpaired_experiment.gates import (
    GateResults, run_shortcut_gates, core_lr_auc_cv, radial_lr_auc_cv,
)
from planb.unpaired_experiment.scheduled_mask import (
    ScheduledCoreMask, apply_deterministic_mask, ScheduleEntry,
)
from planb.unpaired_experiment.thetae_stratification import (
    auroc_by_thetae, ThetaEBinResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================

def make_test_image(H: int = 63, W: int = 63, seed: int = 42) -> np.ndarray:
    """Create a test 3-band image with realistic structure."""
    rng = np.random.default_rng(seed)
    
    # Create image with radial profile (like a galaxy)
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Exponential profile with some noise
    profile = np.exp(-r / 10).astype(np.float32) * 100
    noise = rng.normal(0, 5, (H, W)).astype(np.float32)
    
    # Stack for 3 bands with slight variations
    bands = []
    for i in range(3):
        band = profile * (1 + 0.1 * i) + noise * (1 + 0.05 * i)
        bands.append(band)
    
    return np.stack(bands, axis=0).astype(np.float32)


def make_test_npz_blob(img: np.ndarray) -> bytes:
    """Create an NPZ blob from a 3-band image."""
    assert img.shape[0] == 3, "Expected 3 bands"
    buffer = io.BytesIO()
    np.savez(
        buffer,
        image_g=img[0],
        image_r=img[1],
        image_z=img[2],
    )
    return buffer.getvalue()


def make_separable_data(n_samples: int = 200, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Create perfectly separable data for gate testing."""
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 2
    
    xs, ys = [], []
    for _ in range(n_per_class):
        # Positive: bright core
        img = make_test_image(seed=rng.integers(0, 10000))
        img[:, 28:35, 28:35] += 50  # Bright center
        xs.append(img)
        ys.append(1)
    
    for _ in range(n_per_class):
        # Negative: normal core
        img = make_test_image(seed=rng.integers(0, 10000))
        xs.append(img)
        ys.append(0)
    
    return np.stack(xs, axis=0), np.array(ys)


def make_random_data(n_samples: int = 200, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Create non-separable random data for gate testing."""
    rng = np.random.default_rng(seed)
    
    xs = np.stack([make_test_image(seed=rng.integers(0, 10000)) for _ in range(n_samples)])
    ys = rng.choice([0, 1], size=n_samples)
    
    return xs, ys


# =============================================================================
# Test Constants
# =============================================================================

class TestConstants:
    """Tests for constants module."""
    
    def test_stamp_size_is_64(self):
        """Verify STAMP_SIZE matches actual data (64x64)."""
        assert STAMP_SIZE == 64, f"STAMP_SIZE should be 64, got {STAMP_SIZE}"
    
    def test_bands_are_grz(self):
        """Verify bands are g, r, z."""
        assert BANDS == ("g", "r", "z"), f"BANDS should be ('g', 'r', 'z'), got {BANDS}"
    
    def test_core_box_is_reasonable(self):
        """Verify CORE_BOX is a reasonable size."""
        assert 5 <= CORE_BOX <= 20, f"CORE_BOX={CORE_BOX} seems unreasonable"
    
    def test_seed_default_is_42(self):
        """Verify default seed is 42 for reproducibility."""
        assert SEED_DEFAULT == 42, f"SEED_DEFAULT should be 42, got {SEED_DEFAULT}"
    
    def test_early_stopping_patience(self):
        """Verify early stopping patience matches plan."""
        assert EARLY_STOPPING_PATIENCE == 15, f"Should be 15, got {EARLY_STOPPING_PATIENCE}"


# =============================================================================
# Test Utils
# =============================================================================

class TestDecodeNpzBlob:
    """Tests for NPZ decoding."""
    
    def test_decode_basic(self):
        """Test basic NPZ decoding."""
        img = make_test_image()
        blob = make_test_npz_blob(img)
        
        decoded = decode_npz_blob(blob)
        
        assert "image_g" in decoded
        assert "image_r" in decoded
        assert "image_z" in decoded
        assert np.allclose(decoded["image_g"], img[0])
    
    def test_decode_preserves_dtype(self):
        """Test that decoding preserves float32 dtype."""
        img = make_test_image().astype(np.float32)
        blob = make_test_npz_blob(img)
        
        decoded = decode_npz_blob(blob)
        
        assert decoded["image_g"].dtype == np.float32


class TestRadialFunctions:
    """Tests for radial utility functions."""
    
    def test_radial_rmap_center(self):
        """Test that radial map has zero at center."""
        r = radial_rmap(63, 63)
        center_val = r[31, 31]
        
        assert center_val == 0.0, f"Center should be 0, got {center_val}"
    
    def test_radial_rmap_corners(self):
        """Test that corners have maximum distance."""
        r = radial_rmap(63, 63)
        
        corner_val = r[0, 0]
        expected = np.sqrt(31**2 + 31**2)
        
        assert abs(corner_val - expected) < 0.1, f"Corner should be ~{expected}, got {corner_val}"
    
    def test_radial_mask_annulus(self):
        """Test radial mask creates correct annulus."""
        mask = radial_mask(63, 63, 10, 20)
        r = radial_rmap(63, 63)
        
        # All True pixels should be in [10, 20)
        assert np.all((r[mask] >= 10) & (r[mask] < 20))
        
        # Center should be False
        assert not mask[31, 31]
    
    def test_center_slices(self):
        """Test center slice extraction."""
        ys, xs = center_slices(63, 63, 10)
        
        assert ys.start == 26 and ys.stop == 36
        assert xs.start == 26 and xs.stop == 36


class TestRobustStatistics:
    """Tests for robust statistics functions."""
    
    def test_robust_median_mad_basic(self):
        """Test basic median/MAD computation."""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        med, mad = robust_median_mad(x)
        
        assert med == 3.0, f"Median should be 3, got {med}"
        assert abs(mad - 1.0) < 0.01, f"MAD should be ~1, got {mad}"
    
    def test_robust_median_mad_handles_nan(self):
        """Test that NaN values are ignored."""
        x = np.array([1, 2, np.nan, 4, 5], dtype=np.float32)
        med, mad = robust_median_mad(x)
        
        # Median of [1, 2, 4, 5] is 3
        assert med == 3.0, f"Median should be 3, got {med}"
    
    def test_robust_median_mad_empty_returns_defaults(self):
        """Test that empty array returns safe defaults."""
        x = np.array([], dtype=np.float32)
        med, mad = robust_median_mad(x)
        
        assert med == 0.0
        assert mad == 1.0  # Avoid division by zero
    
    def test_robust_median_mad_all_nan(self):
        """Test that all-NaN array returns safe defaults."""
        x = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        med, mad = robust_median_mad(x)
        
        assert med == 0.0
        assert mad == 1.0


class TestNormalization:
    """Tests for normalization functions."""
    
    def test_normalize_outer_annulus_basic(self):
        """Test basic outer annulus normalization."""
        img = make_test_image()[0]  # Single band
        normalized = normalize_outer_annulus(img)
        
        assert normalized.shape == img.shape
        assert np.isfinite(normalized).all(), "Should have no NaN/Inf"
    
    def test_normalize_outer_annulus_removes_nan(self):
        """Test that normalization removes NaN values."""
        img = make_test_image()[0]
        img[10, 10] = np.nan
        
        normalized = normalize_outer_annulus(img)
        
        assert np.isfinite(normalized).all(), "Should have no NaN"
    
    def test_normalize_outer_annulus_roughly_centered(self):
        """Test that output is roughly zero-centered."""
        img = make_test_image()[0] + 1000  # Large offset
        normalized = normalize_outer_annulus(img)
        
        # Outer region should be roughly zero-centered
        outer_mask = radial_mask(63, 63, 20, 32)
        outer_mean = np.mean(normalized[outer_mask])
        
        assert abs(outer_mean) < 0.5, f"Outer mean should be ~0, got {outer_mean}"


class TestAzimuthalProfile:
    """Tests for azimuthal profile functions."""
    
    def test_azimuthal_median_profile_length(self):
        """Test profile has correct length."""
        img = make_test_image()[0]
        prof = azimuthal_median_profile(img, r_max=32)
        
        assert len(prof) == 32
    
    def test_azimuthal_median_profile_handles_nan(self):
        """Test that NaN values are ignored in profile."""
        img = make_test_image()[0]
        img[31, 31] = np.nan  # NaN at center
        
        prof = azimuthal_median_profile(img, r_max=32)
        
        assert np.isfinite(prof).all(), "Profile should have no NaN"
    
    def test_radial_profile_model_shape(self):
        """Test that radial profile model has correct shape."""
        img = make_test_image()[0]
        model = radial_profile_model(img, r_max=32)
        
        assert model.shape == img.shape
    
    def test_radial_profile_model_is_radially_symmetric(self):
        """Test that model is radially symmetric."""
        img = make_test_image()[0]
        model = radial_profile_model(img, r_max=32)
        
        # Sample at same radius should have same value
        val_1 = model[31, 36]  # r=5 right
        val_2 = model[36, 31]  # r=5 down
        
        assert abs(val_1 - val_2) < 0.01, f"Should be symmetric: {val_1} vs {val_2}"


# =============================================================================
# Test Preprocessing
# =============================================================================

class TestPreprocessing:
    """Tests for preprocessing functions."""
    
    def test_preprocess_stack_raw_robust(self):
        """Test raw_robust preprocessing."""
        img = make_test_image()
        processed = preprocess_stack(img, mode="raw_robust")
        
        assert processed.shape == img.shape
        assert processed.dtype == np.float32
        assert np.isfinite(processed).all()
    
    def test_preprocess_stack_residual(self):
        """Test residual_radial_profile preprocessing."""
        img = make_test_image()
        processed = preprocess_stack(img, mode="residual_radial_profile")
        
        assert processed.shape == img.shape
        assert processed.dtype == np.float32
        assert np.isfinite(processed).all()
    
    def test_preprocess_stack_residual_removes_radial_structure(self):
        """Test that residual preprocessing removes radial structure."""
        img = make_test_image()
        raw = preprocess_stack(img, mode="raw_robust")
        residual = preprocess_stack(img, mode="residual_radial_profile")
        
        # Compute radial variance
        r = radial_rmap(63, 63)
        raw_radial_var = np.var([raw[0, (r >= ri) & (r < ri+1)].mean() for ri in range(0, 30)])
        res_radial_var = np.var([residual[0, (r >= ri) & (r < ri+1)].mean() for ri in range(0, 30)])
        
        # Residual should have less radial structure
        assert res_radial_var < raw_radial_var, "Residual should have less radial structure"
    
    def test_preprocess_stack_clips_extremes(self):
        """Test that extreme values are clipped."""
        img = make_test_image()
        img[0, 31, 31] = 1e10  # Extreme value
        
        processed = preprocess_stack(img, mode="raw_robust")
        
        assert processed.max() <= 50.0, f"Max should be clipped to 50, got {processed.max()}"
        assert processed.min() >= -50.0, f"Min should be clipped to -50, got {processed.min()}"
    
    def test_preprocess_stack_handles_nan(self):
        """Test that NaN values are handled."""
        img = make_test_image()
        img[0, 10, 10] = np.nan
        
        processed = preprocess_stack(img, mode="residual_radial_profile")
        
        assert np.isfinite(processed).all(), "Should have no NaN after preprocessing"
    
    def test_preprocess_stack_invalid_mode_raises(self):
        """Test that invalid mode raises ValueError."""
        img = make_test_image()
        
        try:
            preprocess_stack(img, mode="invalid_mode")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected


# =============================================================================
# Test Gates
# =============================================================================

class TestGates:
    """Tests for shortcut detection gates."""
    
    def test_gate_results_passes_below_threshold(self):
        """Test that GateResults.passes works correctly."""
        result = GateResults(core_lr_auc=0.55, radial_profile_auc=0.52, n_samples=100)
        assert result.passes(threshold=0.65), "Should pass below threshold"
        
        result2 = GateResults(core_lr_auc=0.70, radial_profile_auc=0.52, n_samples=100)
        assert not result2.passes(threshold=0.65), "Should fail above threshold"
    
    def test_core_lr_auc_cv_random_data(self):
        """Test that random data produces AUC ~0.5."""
        xs, ys = make_random_data(n_samples=200, seed=42)
        auc = core_lr_auc_cv(xs, ys, n_splits=3)
        
        assert 0.4 < auc < 0.6, f"Random data should give AUC ~0.5, got {auc}"
    
    def test_core_lr_auc_cv_separable_data(self):
        """Test that separable data produces high AUC."""
        xs, ys = make_separable_data(n_samples=200, seed=42)
        auc = core_lr_auc_cv(xs, ys, n_splits=3)
        
        assert auc > 0.7, f"Separable data should give high AUC, got {auc}"
    
    def test_radial_lr_auc_cv_random_data(self):
        """Test radial gate on random data."""
        xs, ys = make_random_data(n_samples=200, seed=42)
        auc = radial_lr_auc_cv(xs, ys, r_max=32, n_splits=3)
        
        assert 0.4 < auc < 0.6, f"Random data should give AUC ~0.5, got {auc}"
    
    def test_run_shortcut_gates_returns_results(self):
        """Test that run_shortcut_gates returns proper results."""
        xs, ys = make_random_data(n_samples=200, seed=42)
        results = run_shortcut_gates(xs, ys)
        
        assert isinstance(results, GateResults)
        assert 0 <= results.core_lr_auc <= 1
        assert 0 <= results.radial_profile_auc <= 1
        assert results.n_samples == 200
    
    def test_gates_handle_single_class(self):
        """Test that gates handle single-class data gracefully."""
        xs = np.stack([make_test_image() for _ in range(50)])
        ys = np.ones(50)  # All positive
        
        auc = core_lr_auc_cv(xs, ys)
        
        assert auc == 0.5, "Single class should return 0.5"
    
    def test_gates_handle_small_dataset(self):
        """Test that gates handle small datasets."""
        xs, ys = make_random_data(n_samples=20, seed=42)
        auc = core_lr_auc_cv(xs, ys, n_splits=5)
        
        # Should not crash, should return some AUC
        assert 0 <= auc <= 1


# =============================================================================
# Test Scheduled Masking
# =============================================================================

class TestScheduledMask:
    """Tests for scheduled core masking."""
    
    def test_schedule_entry_creation(self):
        """Test ScheduleEntry creation."""
        entry = ScheduleEntry(epoch_start=0, radius=7, prob=0.7)
        assert entry.epoch_start == 0
        assert entry.radius == 7
        assert entry.prob == 0.7
    
    def test_scheduled_mask_basic(self):
        """Test basic scheduled masking."""
        schedule = [(0, 7, 0.7), (10, 5, 0.5)]
        mask = ScheduledCoreMask(schedule, image_size=63)
        
        img = torch.randn(1, 3, 63, 63)
        
        # Force apply
        masked = mask(img, epoch=0, force_apply=True)
        
        # Center should be zero
        assert masked[0, 0, 31, 31] == 0.0
    
    def test_scheduled_mask_epoch_selection(self):
        """Test that correct schedule entry is selected by epoch."""
        schedule = [(0, 7, 0.7), (10, 5, 0.5), (30, 3, 0.3)]
        mask = ScheduledCoreMask(schedule, image_size=63)
        
        r0, p0 = mask.get_current_params(5)
        assert r0 == 7 and p0 == 0.7
        
        r10, p10 = mask.get_current_params(15)
        assert r10 == 5 and p10 == 0.5
        
        r30, p30 = mask.get_current_params(40)
        assert r30 == 3 and p30 == 0.3
    
    def test_apply_deterministic_mask_2d(self):
        """Test deterministic mask on 2D input."""
        img = torch.ones(63, 63)
        masked = apply_deterministic_mask(img, radius=5)
        
        assert masked.shape == (63, 63)
        assert masked[31, 31] == 0.0  # Center masked
        assert masked[0, 0] == 1.0    # Corner unchanged
    
    def test_apply_deterministic_mask_3d(self):
        """Test deterministic mask on 3D input."""
        img = torch.ones(3, 63, 63)
        masked = apply_deterministic_mask(img, radius=5)
        
        assert masked.shape == (3, 63, 63)
        assert masked[0, 31, 31] == 0.0
    
    def test_apply_deterministic_mask_4d_batch(self):
        """Test deterministic mask on 4D batch input."""
        img = torch.ones(8, 3, 63, 63)
        masked = apply_deterministic_mask(img, radius=5)
        
        assert masked.shape == (8, 3, 63, 63)
        assert torch.all(masked[:, :, 31, 31] == 0.0)
    
    def test_scheduled_mask_zero_radius_no_op(self):
        """Test that radius=0 is a no-op."""
        schedule = [(0, 0, 1.0)]
        mask = ScheduledCoreMask(schedule, image_size=63)
        
        img = torch.randn(1, 3, 63, 63)
        masked = mask(img, epoch=0, force_apply=True)
        
        assert torch.allclose(img, masked)
    
    def test_scheduled_mask_invalid_schedule_raises(self):
        """Test that invalid schedule raises error."""
        # Empty schedule
        try:
            ScheduledCoreMask([], image_size=63)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        
        # Doesn't start at epoch 0
        try:
            ScheduledCoreMask([(5, 7, 0.7)], image_size=63)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        
        # Negative radius
        try:
            ScheduledCoreMask([(0, -1, 0.7)], image_size=63)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# =============================================================================
# Test θ_E Stratification
# =============================================================================

class TestThetaEStratification:
    """Tests for θ_E stratified evaluation."""
    
    def test_auroc_by_thetae_basic(self):
        """Test basic θ_E stratified AUROC."""
        n = 500
        rng = np.random.default_rng(42)
        
        y_true = rng.choice([0, 1], size=n)
        y_score = y_true + rng.normal(0, 0.3, n)  # Noisy predictions
        theta_e = np.where(y_true == 1, rng.uniform(0.5, 3.0, n), np.nan)
        
        bins = [(0.5, 1.5), (1.5, 3.0)]
        results = auroc_by_thetae(y_true, y_score, theta_e, bins, min_count=10)
        
        assert len(results) == 2
        assert all(isinstance(r, ThetaEBinResult) for r in results)
    
    def test_auroc_by_thetae_handles_negatives(self):
        """Test that negatives (NaN θ_E) are handled correctly."""
        n = 200
        rng = np.random.default_rng(42)
        
        y_true = np.concatenate([np.ones(100), np.zeros(100)])
        y_score = rng.uniform(0, 1, n)
        theta_e = np.concatenate([rng.uniform(1.0, 2.0, 100), np.full(100, np.nan)])
        
        bins = [(1.0, 2.0)]
        results = auroc_by_thetae(y_true, y_score, theta_e, bins, min_count=10)
        
        assert len(results) == 1
        # Should have sampled some negatives
        assert results[0].n_neg > 0
    
    def test_auroc_by_thetae_insufficient_data(self):
        """Test handling of bins with insufficient data."""
        n = 50
        rng = np.random.default_rng(42)
        
        y_true = rng.choice([0, 1], size=n)
        y_score = rng.uniform(0, 1, n)
        theta_e = np.where(y_true == 1, rng.uniform(0.5, 1.0, n), np.nan)
        
        # Bin that won't have enough samples
        bins = [(2.0, 3.0)]  # No samples in this range
        results = auroc_by_thetae(y_true, y_score, theta_e, bins, min_count=10)
        
        assert len(results) == 1
        assert np.isnan(results[0].auc)  # Not enough data
        assert results[0].n_pos == 0
    
    def test_auroc_by_thetae_result_attributes(self):
        """Test ThetaEBinResult has correct attributes."""
        result = ThetaEBinResult(lo=1.0, hi=2.0, n_pos=100, n_neg=100, auc=0.85)
        
        assert result.lo == 1.0
        assert result.hi == 2.0
        assert result.n_pos == 100
        assert result.n_neg == 100
        assert result.auc == 0.85


# =============================================================================
# Test Integration
# =============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_preprocessing_then_gates(self):
        """Test that preprocessing + gates work together."""
        # Create separable data
        xs, ys = make_separable_data(n_samples=200, seed=42)
        
        # Preprocess
        processed = np.stack([preprocess_stack(x, mode="raw_robust") for x in xs])
        
        # Run gates
        results = run_shortcut_gates(processed, ys)
        
        # Should detect the shortcut
        assert results.core_lr_auc > 0.6, "Should detect core shortcut"
    
    def test_residual_preprocessing_reduces_gate_score(self):
        """Test that residual preprocessing reduces radial gate score."""
        xs, ys = make_separable_data(n_samples=200, seed=42)
        
        # Raw preprocessing
        raw = np.stack([preprocess_stack(x, mode="raw_robust") for x in xs])
        raw_results = run_shortcut_gates(raw, ys)
        
        # Residual preprocessing
        residual = np.stack([preprocess_stack(x, mode="residual_radial_profile") for x in xs])
        res_results = run_shortcut_gates(residual, ys)
        
        # Radial gate should be lower with residual
        # Note: Core might still be high if shortcut is in core pixels
        assert res_results.radial_profile_auc <= raw_results.radial_profile_auc + 0.1
    
    def test_masking_then_gates(self):
        """Test that masking reduces gate scores."""
        xs, ys = make_separable_data(n_samples=200, seed=42)
        
        # Convert to torch
        xs_t = torch.from_numpy(xs).float()
        
        # Apply deterministic mask
        masked = apply_deterministic_mask(xs_t, radius=7)
        
        # Run gates on masked data
        results = run_shortcut_gates(masked.numpy(), ys)
        
        # Core gate should be lower after masking
        # (The core info that made it separable is removed)
        assert results.core_lr_auc < 0.75, "Masking should reduce core AUC"


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    import traceback
    
    test_classes = [
        TestConstants,
        TestDecodeNpzBlob,
        TestRadialFunctions,
        TestRobustStatistics,
        TestNormalization,
        TestAzimuthalProfile,
        TestPreprocessing,
        TestGates,
        TestScheduledMask,
        TestThetaEStratification,
        TestIntegration,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        instance = test_class()
        
        test_methods = [m for m in dir(instance) if m.startswith("test_")]
        
        print(f"\n{test_class.__name__}")
        print("-" * 50)
        
        for method_name in test_methods:
            total_tests += 1
            method = getattr(instance, method_name)
            
            try:
                method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {method_name}")
                print(f"    Error: {e}")
                failed_tests.append({
                    "class": test_class.__name__,
                    "method": method_name,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed tests:")
        for failure in failed_tests:
            print(f"  - {failure['class']}.{failure['method']}")
            print(f"    {failure['error']}")
        return False
    else:
        print("\n✓ ALL TESTS PASSED")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

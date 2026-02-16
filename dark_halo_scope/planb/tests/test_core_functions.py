#!/usr/bin/env python3
"""
Unit Tests for Core Functions

These tests verify that core functions work as intended BEFORE any training.
Lesson L4.3: Test actual code path, not mock.
Lesson L1.2: Import from shared module for consistency.

Usage:
    pytest test_core_functions.py -v
    python test_core_functions.py
"""
import io
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase1_baseline"))

# Import from shared module - SINGLE SOURCE OF TRUTH
from shared.constants import (
    STAMP_SIZE, NUM_CHANNELS, STAMP_SHAPE, CORE_RADIUS_PIX,
    OUTER_RADIUS_PIX, CLIP_SIGMA, AZIMUTHAL_SHUFFLE_BINS,
)
from shared.utils import (
    decode_stamp_npz, validate_stamp, robust_normalize,
    azimuthal_shuffle, apply_core_dropout, create_radial_mask,
)


class TestDecodeStampNpz:
    """Tests for NPZ decoding function."""
    
    def test_decode_multiband_format(self):
        """Test decoding multi-band (grz) format."""
        # decode_stamp_npz imported from shared.utils above
        
        # Create mock NPZ using constants
        g = np.random.randn(STAMP_SIZE, STAMP_SIZE).astype(np.float32)
        r = np.random.randn(STAMP_SIZE, STAMP_SIZE).astype(np.float32)
        z = np.random.randn(STAMP_SIZE, STAMP_SIZE).astype(np.float32)
        
        buffer = io.BytesIO()
        np.savez(buffer, image_g=g, image_r=r, image_z=z)
        blob = buffer.getvalue()
        
        arr, bandset = decode_stamp_npz(blob)
        
        assert arr.shape == STAMP_SHAPE, f"Wrong shape: {arr.shape}, expected {STAMP_SHAPE}"
        assert bandset == "grz", f"Wrong bandset: {bandset}"
        assert np.allclose(arr[0], g), "G band mismatch"
        assert np.allclose(arr[1], r), "R band mismatch"
        assert np.allclose(arr[2], z), "Z band mismatch"
    
    def test_decode_legacy_format(self):
        """Test decoding legacy single-key format."""
        # decode_stamp_npz imported from shared.utils above
        
        img = np.random.randn(STAMP_SIZE, STAMP_SIZE).astype(np.float32)
        
        buffer = io.BytesIO()
        np.savez(buffer, img=img)
        blob = buffer.getvalue()
        
        arr, bandset = decode_stamp_npz(blob)
        
        assert arr.shape[0] == 1 or arr.ndim == 2, f"Wrong shape: {arr.shape}"
        assert bandset == "unknown"
    
    def test_decode_handles_nan(self):
        """Test that decoding handles NaN values correctly."""
        # Functions imported from shared.utils above
        
        g = np.random.randn(STAMP_SIZE, STAMP_SIZE).astype(np.float32)
        g[32, 32] = np.nan  # Insert NaN
        r = np.random.randn(STAMP_SIZE, STAMP_SIZE).astype(np.float32)
        z = np.random.randn(STAMP_SIZE, STAMP_SIZE).astype(np.float32)
        
        buffer = io.BytesIO()
        np.savez(buffer, image_g=g, image_r=r, image_z=z)
        blob = buffer.getvalue()
        
        arr, bandset = decode_stamp_npz(blob)
        validation = validate_stamp(arr, bandset)
        
        assert not validation["no_nan"], "Should detect NaN"
        assert not validation["valid"], "Should not be valid"


class TestNormalization:
    """Tests for normalization functions."""
    
    def test_robust_normalize_basic(self):
        """Test basic normalization works."""
        from data_loader import robust_normalize
        
        img = np.random.randn(3, 64, 64).astype(np.float32) * 10 + 5
        
        normalized = robust_normalize(img, outer_radius=20, clip_sigma=5.0)
        
        assert normalized.shape == img.shape, "Shape changed"
        assert np.isfinite(normalized).all(), "NaN/Inf in output"
        assert normalized.dtype == np.float32, "Wrong dtype"
    
    def test_robust_normalize_approximately_centered(self):
        """Test that normalization produces approximately zero-centered output."""
        from data_loader import robust_normalize
        
        img = np.random.randn(3, 64, 64).astype(np.float32) * 10 + 100
        
        normalized = robust_normalize(img, outer_radius=20, clip_sigma=5.0)
        
        # Mean should be close to zero (within 1 sigma)
        assert abs(np.mean(normalized)) < 1.0, f"Mean too far from zero: {np.mean(normalized)}"
    
    def test_robust_normalize_clips_outliers(self):
        """Test that outliers are clipped."""
        from data_loader import robust_normalize
        
        img = np.random.randn(3, 64, 64).astype(np.float32)
        img[0, 32, 32] = 1000.0  # Extreme outlier
        
        normalized = robust_normalize(img, outer_radius=20, clip_sigma=5.0)
        
        assert normalized.max() <= 5.0, f"Outlier not clipped: {normalized.max()}"
        assert normalized.min() >= -5.0, f"Outlier not clipped: {normalized.min()}"


class TestAzimuthalShuffle:
    """Tests for hard negative generation."""
    
    def test_azimuthal_shuffle_preserves_radial_profile(self):
        """Test that shuffle preserves radial mean flux."""
        from data_loader import azimuthal_shuffle
        
        # Create image with radial structure
        h, w = 64, 64
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        
        img = np.exp(-r / 10).astype(np.float32)[np.newaxis, :, :]
        img = np.repeat(img, 3, axis=0)
        
        shuffled = azimuthal_shuffle(img, n_bins=8, seed=42)
        
        # Radial profiles should be approximately equal
        for radius in [5, 10, 15, 20]:
            mask = (r >= radius - 1) & (r < radius + 1)
            orig_mean = np.mean(img[:, mask])
            shuf_mean = np.mean(shuffled[:, mask])
            
            assert abs(orig_mean - shuf_mean) < 0.1, \
                f"Radial profile changed at r={radius}: {orig_mean:.3f} vs {shuf_mean:.3f}"
    
    def test_azimuthal_shuffle_preserves_total_flux(self):
        """Test that shuffle preserves total flux."""
        from data_loader import azimuthal_shuffle
        
        img = np.random.randn(3, 64, 64).astype(np.float32)
        shuffled = azimuthal_shuffle(img, n_bins=8, seed=42)
        
        orig_sum = img.sum()
        shuf_sum = shuffled.sum()
        
        # Use relative tolerance for large sums (float32 has ~7 significant digits)
        rel_diff = abs(orig_sum - shuf_sum) / (abs(orig_sum) + 1e-10)
        assert rel_diff < 1e-5, \
            f"Total flux changed: {orig_sum:.6f} vs {shuf_sum:.6f} (rel_diff={rel_diff:.2e})"
    
    def test_azimuthal_shuffle_reproducible(self):
        """Test that shuffle is reproducible with seed."""
        from data_loader import azimuthal_shuffle
        
        img = np.random.randn(3, 64, 64).astype(np.float32)
        
        shuf1 = azimuthal_shuffle(img, seed=42)
        shuf2 = azimuthal_shuffle(img, seed=42)
        
        assert np.allclose(shuf1, shuf2), "Not reproducible with same seed"


class TestCoreDropout:
    """Tests for core dropout augmentation."""
    
    def test_core_dropout_masks_center(self):
        """Test that center is masked."""
        from data_loader import apply_core_dropout
        
        img = np.ones((3, 64, 64), dtype=np.float32) * 100
        
        masked = apply_core_dropout(img, radius=5, fill_mode="zero")
        
        # Center should be zero
        center_value = masked[0, 32, 32]
        assert center_value == 0.0, f"Center not masked: {center_value}"
        
        # Outer should be unchanged
        outer_value = masked[0, 0, 0]
        assert outer_value == 100.0, f"Outer changed: {outer_value}"
    
    def test_core_dropout_outer_median_fill(self):
        """Test outer median fill mode."""
        from data_loader import apply_core_dropout
        
        # Create image with known structure
        img = np.zeros((3, 64, 64), dtype=np.float32)
        img[:, :, :] = 10.0  # Outer region
        
        masked = apply_core_dropout(img, radius=5, fill_mode="outer_median")
        
        # Center should be filled with outer median
        center_value = masked[0, 32, 32]
        assert abs(center_value - 10.0) < 0.1, f"Wrong fill value: {center_value}"


class TestModelForward:
    """Tests for model forward pass."""
    
    def test_model_output_shape(self):
        """Test that model produces correct output shape."""
        from model import build_model, ModelConfig
        
        config = ModelConfig(
            arch="resnet18",
            pretrained=False,  # Faster for testing
            in_channels=3,
        )
        
        device = "cpu"
        model = build_model(config=config, device=device)
        
        x = torch.randn(4, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == (4, 1), f"Wrong output shape: {out.shape}"
    
    def test_model_no_nan_gradients(self):
        """Test that backward pass produces no NaN gradients."""
        from model import build_model, ModelConfig
        
        config = ModelConfig(
            arch="resnet18",
            pretrained=False,
        )
        
        model = build_model(config=config, device="cpu")
        model.train()
        
        x = torch.randn(4, 3, 64, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), \
                    f"NaN gradient in {name}"
    
    def test_model_6_channel_input(self):
        """Test model with 6-channel input."""
        from model import build_model, ModelConfig
        
        config = ModelConfig(
            arch="resnet18",
            pretrained=False,
            in_channels=6,
        )
        
        model = build_model(config=config, device="cpu")
        
        x = torch.randn(4, 6, 64, 64)
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == (4, 1), f"Wrong output shape for 6ch: {out.shape}"


class TestValidationFunctions:
    """Tests for validation functions."""
    
    def test_validate_stamp_detects_nan(self):
        """Test that validation detects NaN."""
        from data_loader import validate_stamp
        
        stamp = np.random.randn(3, 64, 64).astype(np.float32)
        stamp[0, 10, 10] = np.nan
        
        result = validate_stamp(stamp, "grz")
        
        assert not result["no_nan"], "Should detect NaN"
        assert not result["valid"], "Should not be valid"
    
    def test_validate_stamp_detects_inf(self):
        """Test that validation detects Inf."""
        from data_loader import validate_stamp
        
        stamp = np.random.randn(3, 64, 64).astype(np.float32)
        stamp[0, 10, 10] = np.inf
        
        result = validate_stamp(stamp, "grz")
        
        assert not result["no_inf"], "Should detect Inf"
        assert not result["valid"], "Should not be valid"
    
    def test_validate_stamp_passes_good_data(self):
        """Test that validation passes good data."""
        from data_loader import validate_stamp
        
        stamp = np.random.randn(3, 64, 64).astype(np.float32)
        
        result = validate_stamp(stamp, "grz")
        
        assert result["valid"], "Should be valid"


def run_all_tests():
    """Run all tests and report results."""
    import traceback
    
    test_classes = [
        TestDecodeStampNpz,
        TestNormalization,
        TestAzimuthalShuffle,
        TestCoreDropout,
        TestModelForward,
        TestValidationFunctions,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        instance = test_class()
        
        # Get all test methods
        test_methods = [
            method for method in dir(instance)
            if method.startswith("test_")
        ]
        
        print(f"\n{test_class.__name__}")
        print("-" * 40)
        
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
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed tests:")
        for failure in failed_tests:
            print(f"  - {failure['class']}.{failure['method']}: {failure['error']}")
        return False
    else:
        print("\n✓ ALL TESTS PASSED")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

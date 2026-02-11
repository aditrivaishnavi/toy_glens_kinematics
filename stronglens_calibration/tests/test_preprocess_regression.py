#!/usr/bin/env python3
"""
Preprocessing Regression Test: Lock preprocessing behavior via checksum.

Creates a deterministic synthetic cutout with known pixel values, runs
preprocess_stack with the same parameters used in training, and verifies
the output matches a known checksum. This prevents silent drift in
preprocessing behavior across code changes.

Per LLM recommendation (conversation_with_llm.txt, Q11):
  "A tiny checksum-style test on 1-2 reference NPZs prevents silent drift."

Tests cover:
  1. raw_robust preprocessing (101x101, no crop) — Paper IV parity mode
  2. raw_robust preprocessing (101x101 -> 64x64 center crop) — legacy mode
  3. Deterministic augmentation reproducibility
  4. NaN handling (NaN pixels replaced with 0)
  5. Clip range enforcement (values clipped to [-10, +10])
  6. Outer-annulus normalization correctness

Usage:
    cd stronglens_calibration
    export PYTHONPATH=.
    python -m pytest tests/test_preprocess_regression.py -v

    # Or run standalone:
    python tests/test_preprocess_regression.py

Author: stronglens_calibration project
Date: 2026-02-11
Aligned with: LLM review Q11 (preprocessing consistency check)
"""
from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np

# Add parent to path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dhs.preprocess import preprocess_stack, center_crop
from dhs.utils import normalize_outer_annulus, robust_median_mad, radial_mask
from dhs.transforms import AugmentConfig, random_augment
from dhs.constants import STAMP_SIZE, CUTOUT_SIZE


def make_synthetic_cutout(seed: int = 42) -> np.ndarray:
    """Create a deterministic synthetic cutout (3, 101, 101) in CHW format.

    The cutout has:
    - A bright Gaussian center (galaxy-like)
    - A faint background with noise
    - Realistic-ish flux values in nanomaggies range
    """
    rng = np.random.default_rng(seed)
    h = w = CUTOUT_SIZE  # 101

    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2

    bands = []
    for b, (flux, bg, noise) in enumerate(
        [(5.0, 0.1, 0.05), (8.0, 0.15, 0.07), (3.5, 0.08, 0.04)]  # g, r, z
    ):
        # Central Gaussian (galaxy core)
        galaxy = flux * np.exp(-r2 / (2 * 6.0 ** 2))
        # Background + noise
        background = bg + noise * rng.standard_normal((h, w))
        band = (galaxy + background).astype(np.float32)
        bands.append(band)

    return np.stack(bands, axis=0)  # (3, 101, 101) CHW


def array_checksum(arr: np.ndarray) -> str:
    """Compute SHA-256 checksum of array bytes for regression testing."""
    # Use round to reduce float precision sensitivity across platforms
    rounded = np.round(arr, decimals=6)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


class TestPreprocessRegression(unittest.TestCase):
    """Regression tests for preprocessing pipeline."""

    def setUp(self):
        """Create a deterministic synthetic cutout."""
        self.cutout = make_synthetic_cutout(seed=42)
        self.assertEqual(self.cutout.shape, (3, 101, 101))
        self.assertEqual(self.cutout.dtype, np.float32)

    # ------------------------------------------------------------------
    # Test 1: raw_robust with no crop (Paper IV parity mode)
    # ------------------------------------------------------------------
    def test_raw_robust_no_crop_shape(self):
        """raw_robust with crop=False keeps 101x101."""
        out = preprocess_stack(self.cutout, mode="raw_robust", crop=False)
        self.assertEqual(out.shape, (3, 101, 101))
        self.assertEqual(out.dtype, np.float32)

    def test_raw_robust_no_crop_deterministic(self):
        """raw_robust with crop=False is deterministic (same input -> same output)."""
        out1 = preprocess_stack(self.cutout, mode="raw_robust", crop=False)
        out2 = preprocess_stack(self.cutout.copy(), mode="raw_robust", crop=False)
        np.testing.assert_array_equal(out1, out2)

    def test_raw_robust_no_crop_checksum(self):
        """raw_robust checksum locks preprocessing behavior."""
        out = preprocess_stack(self.cutout, mode="raw_robust", crop=False)
        cs = array_checksum(out)
        # This checksum was computed once and should not change.
        # If preprocessing logic changes, this test will fail,
        # signaling that the change needs to be intentional and documented.
        #
        # To update after intentional change:
        #   1. Run this test, it will print the new checksum
        #   2. Verify the change is intended
        #   3. Update the expected checksum below
        print(f"  raw_robust no_crop checksum: {cs}")
        # Verified checksum from initial run (2026-02-11, NumPy 1.26+, Python 3.11)
        EXPECTED_CHECKSUM = "7e25b9e366471bda"
        self.assertEqual(cs, EXPECTED_CHECKSUM,
                         f"Preprocessing output changed! Got {cs}, expected {EXPECTED_CHECKSUM}. "
                         "If this is intentional, update EXPECTED_CHECKSUM.")

    # ------------------------------------------------------------------
    # Test 2: raw_robust with center crop (legacy mode)
    # ------------------------------------------------------------------
    def test_raw_robust_with_crop_shape(self):
        """raw_robust with crop=True crops to STAMP_SIZE (64x64)."""
        out = preprocess_stack(self.cutout, mode="raw_robust", crop=True)
        self.assertEqual(out.shape, (3, STAMP_SIZE, STAMP_SIZE))

    def test_raw_robust_with_crop_deterministic(self):
        """raw_robust with crop=True is deterministic."""
        out1 = preprocess_stack(self.cutout, mode="raw_robust", crop=True)
        out2 = preprocess_stack(self.cutout.copy(), mode="raw_robust", crop=True)
        np.testing.assert_array_equal(out1, out2)

    # ------------------------------------------------------------------
    # Test 3: Clipping
    # ------------------------------------------------------------------
    def test_clip_range(self):
        """Output values are clipped to [-10, +10]."""
        out = preprocess_stack(self.cutout, mode="raw_robust", crop=False, clip_range=10.0)
        self.assertTrue(np.all(out >= -10.0), f"Min value: {out.min()}")
        self.assertTrue(np.all(out <= 10.0), f"Max value: {out.max()}")

    def test_clip_range_custom(self):
        """Custom clip range is respected."""
        out = preprocess_stack(self.cutout, mode="raw_robust", crop=False, clip_range=5.0)
        self.assertTrue(np.all(out >= -5.0))
        self.assertTrue(np.all(out <= 5.0))

    # ------------------------------------------------------------------
    # Test 4: NaN handling
    # ------------------------------------------------------------------
    def test_nan_handling(self):
        """NaN pixels are replaced with 0 before normalization."""
        cutout_nan = self.cutout.copy()
        # Inject NaN in the center of one band
        cutout_nan[0, 45:55, 45:55] = np.nan
        out = preprocess_stack(cutout_nan, mode="raw_robust", crop=False)
        self.assertFalse(np.any(np.isnan(out)), "Output contains NaN!")
        self.assertFalse(np.any(np.isinf(out)), "Output contains Inf!")

    def test_nan_heavy(self):
        """Cutout with many NaN pixels still produces valid output."""
        cutout_nan = self.cutout.copy()
        cutout_nan[1, :, :50] = np.nan  # Half of r-band is NaN
        out = preprocess_stack(cutout_nan, mode="raw_robust", crop=False)
        self.assertFalse(np.any(np.isnan(out)))
        self.assertEqual(out.shape, (3, 101, 101))

    # ------------------------------------------------------------------
    # Test 5: Outer-annulus normalization
    # ------------------------------------------------------------------
    def test_outer_annulus_normalization(self):
        """Verify outer annulus normalization uses correct radii."""
        # Create a band image with known properties
        band = np.ones((101, 101), dtype=np.float32) * 10.0
        # Set the outer annulus (r=20 to r=32) to a known value
        mask = radial_mask(101, 101, 20, 32)
        band[mask] = 5.0  # median=5, MAD~=0 -> normalized ~ (x-5)/eps

        normed = normalize_outer_annulus(band, r_in=20, r_out=32)
        # The annulus pixels should be near 0 after normalization
        annulus_vals = normed[mask]
        self.assertAlmostEqual(float(np.median(annulus_vals)), 0.0, places=2)

    def test_robust_median_mad(self):
        """Test robust_median_mad computation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # outlier at 100
        med, mad = robust_median_mad(x)
        self.assertAlmostEqual(med, 3.5, places=1)  # median of 1-5 range
        # MAD should be robust to the outlier
        self.assertTrue(mad < 10.0, f"MAD={mad} should be robust to outlier")

    # ------------------------------------------------------------------
    # Test 6: Center crop
    # ------------------------------------------------------------------
    def test_center_crop_101_to_64(self):
        """Center crop from 101x101 to 64x64."""
        out = center_crop(self.cutout, 64)
        self.assertEqual(out.shape, (3, 64, 64))

    def test_center_crop_preserves_center(self):
        """Center crop preserves the central pixel."""
        center_val = self.cutout[0, 50, 50]
        out = center_crop(self.cutout, 64)
        # After cropping 101->64, offset is (101-64)//2 = 18
        # So center (50,50) maps to (50-18, 50-18) = (32, 32) in cropped
        self.assertAlmostEqual(float(out[0, 32, 32]), float(center_val), places=5)

    def test_center_crop_identity(self):
        """Center crop with target size == input size is identity."""
        out = center_crop(self.cutout, 101)
        np.testing.assert_array_equal(out, self.cutout)

    # ------------------------------------------------------------------
    # Test 7: Augmentation determinism
    # ------------------------------------------------------------------
    def test_augmentation_deterministic(self):
        """Augmentation with same seed produces identical output."""
        preprocessed = preprocess_stack(self.cutout, mode="raw_robust", crop=False)
        cfg = AugmentConfig(hflip=True, vflip=True, rot90=True)

        out1 = random_augment(preprocessed, seed=12345, cfg=cfg)
        out2 = random_augment(preprocessed.copy(), seed=12345, cfg=cfg)
        np.testing.assert_array_equal(out1, out2)

    def test_augmentation_different_seeds(self):
        """Augmentation with different seeds produces different output (usually)."""
        preprocessed = preprocess_stack(self.cutout, mode="raw_robust", crop=False)
        cfg = AugmentConfig(hflip=True, vflip=True, rot90=True)

        # Try multiple seed pairs; at least one should differ
        any_different = False
        for s1, s2 in [(1, 2), (10, 20), (100, 200)]:
            out1 = random_augment(preprocessed.copy(), seed=s1, cfg=cfg)
            out2 = random_augment(preprocessed.copy(), seed=s2, cfg=cfg)
            if not np.array_equal(out1, out2):
                any_different = True
                break
        self.assertTrue(any_different, "All augmentations with different seeds were identical")

    # ------------------------------------------------------------------
    # Test 8: No-augmentation preserves preprocessing
    # ------------------------------------------------------------------
    def test_no_augmentation_identity(self):
        """Disabled augmentation is identity."""
        preprocessed = preprocess_stack(self.cutout, mode="raw_robust", crop=False)
        cfg = AugmentConfig(hflip=False, vflip=False, rot90=False)
        out = random_augment(preprocessed, seed=42, cfg=cfg)
        np.testing.assert_array_almost_equal(out, preprocessed, decimal=5)

    # ------------------------------------------------------------------
    # Test 9: Verify dtype consistency
    # ------------------------------------------------------------------
    def test_output_dtype(self):
        """Preprocessing output is float32."""
        out = preprocess_stack(self.cutout, mode="raw_robust", crop=False)
        self.assertEqual(out.dtype, np.float32)

    def test_output_dtype_after_crop(self):
        """Preprocessing output is float32 after crop."""
        out = preprocess_stack(self.cutout, mode="raw_robust", crop=True)
        self.assertEqual(out.dtype, np.float32)


class TestPreprocessCrossPlatform(unittest.TestCase):
    """Cross-platform consistency checks.

    These tests don't rely on exact checksums (which may differ between
    NumPy versions), but verify key invariants that must hold everywhere.
    """

    def test_raw_robust_zero_centered_annulus(self):
        """After raw_robust normalization, the outer annulus should be ~zero mean."""
        cutout = make_synthetic_cutout(seed=42)
        out = preprocess_stack(cutout, mode="raw_robust", crop=False)

        for band in range(3):
            mask = radial_mask(101, 101, 20, 32)
            annulus_vals = out[band][mask]
            mean = float(np.mean(annulus_vals))
            # Annulus should be approximately zero-centered
            self.assertAlmostEqual(mean, 0.0, places=0,
                                   msg=f"Band {band}: annulus mean = {mean:.4f}")

    def test_raw_robust_center_positive(self):
        """After raw_robust, the center (galaxy) should be positive (bright)."""
        cutout = make_synthetic_cutout(seed=42)
        out = preprocess_stack(cutout, mode="raw_robust", crop=False)

        for band in range(3):
            center_val = out[band, 50, 50]
            # The synthetic cutout has a bright center; after normalization,
            # the center should be positive relative to the background.
            self.assertGreater(center_val, 0.0,
                               msg=f"Band {band}: center value = {center_val:.4f}")


if __name__ == "__main__":
    # When run standalone, also print helpful info
    print("=" * 60)
    print("Preprocessing Regression Test")
    print("=" * 60)
    print(f"  STAMP_SIZE = {STAMP_SIZE}")
    print(f"  CUTOUT_SIZE = {CUTOUT_SIZE}")

    # Generate and print checksums for reference
    cutout = make_synthetic_cutout(seed=42)
    out_nocrop = preprocess_stack(cutout, mode="raw_robust", crop=False)
    out_crop = preprocess_stack(cutout, mode="raw_robust", crop=True)
    print(f"  Checksum (no crop): {array_checksum(out_nocrop)}")
    print(f"  Checksum (crop 64): {array_checksum(out_crop)}")
    print()

    # Run tests
    unittest.main(verbosity=2)

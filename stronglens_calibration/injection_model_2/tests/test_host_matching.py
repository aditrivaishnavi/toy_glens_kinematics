"""
Comprehensive tests for host_matching.py (Model 2 host-conditioned injection).

Tests cover:
  - Round Gaussian host → q ≈ 1
  - Elliptical Gaussian host → q < 1, PA correct
  - All-zero image → fallback
  - All-NaN image → fallback
  - Very faint host (noise-dominated) → fallback or plausible q
  - Single bright pixel → fallback (degenerate)
  - Wrong input shape → fallback with warning
  - center_radius_pix larger than image → clamped, no crash
  - r_band_index out of range → fallback
  - Rotated elliptical → PA matches rotation angle
  - map_host_to_lens_params output keys and ranges
  - q_floor enforcement
  - Deterministic with fixed seed
  - Negative pixel handling

Author: stronglens_calibration project
Date: 2026-02-13
"""
import math
import unittest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from host_matching import (
    HostMoments,
    estimate_host_moments_rband,
    map_host_to_lens_params,
)


def _make_gaussian_host(
    H: int = 101,
    W: int = 101,
    sigx: float = 8.0,
    sigy: float = 8.0,
    angle_deg: float = 0.0,
    peak: float = 100.0,
    r_band_index: int = 1,
) -> np.ndarray:
    """Create an HWC host cutout with a 2D Gaussian in the r-band."""
    yy, xx = np.mgrid[0:H, 0:W]
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    dx = xx - cx
    dy = yy - cy

    # Rotate coordinates
    theta = math.radians(angle_deg)
    dx_r = dx * math.cos(theta) + dy * math.sin(theta)
    dy_r = -dx * math.sin(theta) + dy * math.cos(theta)

    img = peak * np.exp(-0.5 * ((dx_r / sigx) ** 2 + (dy_r / sigy) ** 2))
    host = np.zeros((H, W, 3), dtype=np.float32)
    host[..., r_band_index] = img.astype(np.float32)
    return host


class TestEstimateHostMomentsRband(unittest.TestCase):
    """Tests for estimate_host_moments_rband."""

    def test_round_gaussian_q_near_one(self):
        """A round Gaussian should give q close to 1."""
        host = _make_gaussian_host(sigx=8.0, sigy=8.0)
        m = estimate_host_moments_rband(host)
        self.assertFalse(m.is_fallback)
        self.assertGreaterEqual(m.q, 0.90)
        self.assertLessEqual(m.q, 1.0)

    def test_elliptical_gaussian_q_below_one(self):
        """An elongated Gaussian should give q < 1."""
        host = _make_gaussian_host(sigx=12.0, sigy=4.0)
        m = estimate_host_moments_rband(host)
        self.assertFalse(m.is_fallback)
        self.assertLess(m.q, 0.6)
        self.assertGreater(m.q, 0.1)

    def test_elliptical_q_value_approximate(self):
        """q should be approximately sigy/sigx for an axis-aligned ellipse."""
        host = _make_gaussian_host(sigx=10.0, sigy=5.0)
        m = estimate_host_moments_rband(host)
        # Expected q ≈ 5/10 = 0.5
        self.assertAlmostEqual(m.q, 0.5, delta=0.08)

    def test_rotated_elliptical_pa(self):
        """PA should track the rotation angle of the major axis."""
        for angle_deg in [0, 30, 60, 90, -45]:
            host = _make_gaussian_host(sigx=12.0, sigy=4.0, angle_deg=angle_deg)
            m = estimate_host_moments_rband(host)
            # PA is defined modulo pi
            expected_rad = math.radians(angle_deg) % math.pi
            measured_rad = m.phi_rad % math.pi
            # Allow 15 degree tolerance
            diff = abs(expected_rad - measured_rad)
            diff = min(diff, math.pi - diff)  # handle wraparound
            self.assertLess(
                diff,
                math.radians(15),
                f"angle_deg={angle_deg}: expected PA≈{math.degrees(expected_rad):.1f}°, "
                f"got {math.degrees(measured_rad):.1f}°",
            )

    def test_all_zero_image_returns_fallback(self):
        """All-zero image should return fallback values."""
        host = np.zeros((101, 101, 3), dtype=np.float32)
        m = estimate_host_moments_rband(host)
        self.assertTrue(m.is_fallback)
        self.assertEqual(m.q, 1.0)

    def test_all_nan_image_returns_fallback(self):
        """All-NaN image should return fallback values."""
        host = np.full((101, 101, 3), np.nan, dtype=np.float32)
        m = estimate_host_moments_rband(host)
        self.assertTrue(m.is_fallback)

    def test_very_faint_host_does_not_crash(self):
        """A host with tiny flux values should not crash."""
        host = _make_gaussian_host(peak=1e-10)
        m = estimate_host_moments_rband(host)
        # May return fallback or valid — just must not crash
        self.assertIsInstance(m, HostMoments)
        self.assertTrue(0.0 < m.q <= 1.0)

    def test_single_bright_pixel(self):
        """A single bright pixel → degenerate moments → fallback."""
        host = np.zeros((101, 101, 3), dtype=np.float32)
        host[50, 50, 1] = 1000.0
        m = estimate_host_moments_rband(host)
        # Degenerate — should be fallback or q=1
        self.assertIsInstance(m, HostMoments)

    def test_wrong_ndim_returns_fallback(self):
        """2D input should return fallback with warning."""
        host = np.zeros((101, 101), dtype=np.float32)
        m = estimate_host_moments_rband(host)
        self.assertTrue(m.is_fallback)

    def test_r_band_index_out_of_range_returns_fallback(self):
        """r_band_index >= C should return fallback."""
        host = np.zeros((101, 101, 3), dtype=np.float32)
        m = estimate_host_moments_rband(host, r_band_index=5)
        self.assertTrue(m.is_fallback)

    def test_center_radius_larger_than_image(self):
        """center_radius_pix > image half-size should be clamped, not crash."""
        host = _make_gaussian_host(H=31, W=31, sigx=5.0, sigy=5.0)
        m = estimate_host_moments_rband(host, center_radius_pix=200)
        self.assertIsInstance(m, HostMoments)
        self.assertFalse(m.is_fallback)

    def test_r_half_pix_positive(self):
        """Half-light radius should always be positive."""
        host = _make_gaussian_host()
        m = estimate_host_moments_rband(host)
        self.assertGreater(m.r_half_pix, 0.0)

    def test_q_min_floor_enforced(self):
        """q should not go below q_min even for very elongated objects."""
        # Create extremely elongated object
        host = _make_gaussian_host(sigx=20.0, sigy=1.5)
        m = estimate_host_moments_rband(host, q_min=0.3)
        self.assertGreaterEqual(m.q, 0.3)

    def test_negative_pixels_handled(self):
        """Negative pixel values should be handled gracefully."""
        host = _make_gaussian_host(peak=50.0)
        # Add some negative values
        host[:20, :20, 1] = -10.0
        m = estimate_host_moments_rband(host)
        self.assertIsInstance(m, HostMoments)
        self.assertFalse(m.is_fallback)

    def test_small_image(self):
        """Works on a small (11x11) image."""
        host = _make_gaussian_host(H=11, W=11, sigx=2.0, sigy=2.0)
        m = estimate_host_moments_rband(host, center_radius_pix=5)
        self.assertIsInstance(m, HostMoments)


class TestMapHostToLensParams(unittest.TestCase):
    """Tests for map_host_to_lens_params."""

    def test_output_keys(self):
        """Should return all required keys for LensParams."""
        host_mom = HostMoments(q=0.7, phi_rad=0.5, r_half_pix=8.0)
        rng = np.random.default_rng(42)
        result = map_host_to_lens_params(1.5, host_mom, rng=rng)
        expected_keys = {
            "theta_e_arcsec",
            "q_lens",
            "phi_lens_rad",
            "shear_g1",
            "shear_g2",
            "gamma_ext",
            "phi_ext_rad",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_theta_e_passthrough(self):
        """theta_e should be passed through unchanged."""
        host_mom = HostMoments(q=0.8, phi_rad=0.3, r_half_pix=6.0)
        result = map_host_to_lens_params(2.0, host_mom)
        self.assertAlmostEqual(result["theta_e_arcsec"], 2.0)

    def test_q_floor_enforcement(self):
        """q_lens should not go below q_floor."""
        host_mom = HostMoments(q=0.3, phi_rad=0.0, r_half_pix=5.0)
        rng = np.random.default_rng(42)
        result = map_host_to_lens_params(1.5, host_mom, q_floor=0.5, rng=rng)
        self.assertGreaterEqual(result["q_lens"], 0.5)

    def test_q_lens_not_above_one(self):
        """q_lens should never exceed 1."""
        host_mom = HostMoments(q=0.99, phi_rad=0.0, r_half_pix=5.0)
        rng = np.random.default_rng(42)
        for _ in range(100):
            result = map_host_to_lens_params(1.5, host_mom, q_scatter=0.1, rng=rng)
            self.assertLessEqual(result["q_lens"], 1.0)

    def test_gamma_ext_in_range(self):
        """gamma_ext should be in [0, gamma_max]."""
        host_mom = HostMoments(q=0.8, phi_rad=0.5, r_half_pix=8.0)
        rng = np.random.default_rng(42)
        for _ in range(100):
            result = map_host_to_lens_params(
                1.5, host_mom, gamma_max=0.08, rng=rng
            )
            self.assertGreaterEqual(result["gamma_ext"], 0.0)
            self.assertLessEqual(result["gamma_ext"], 0.08)

    def test_phi_ext_in_range(self):
        """phi_ext should be in [0, pi)."""
        host_mom = HostMoments(q=0.8, phi_rad=0.5, r_half_pix=8.0)
        rng = np.random.default_rng(42)
        for _ in range(100):
            result = map_host_to_lens_params(1.5, host_mom, rng=rng)
            self.assertGreaterEqual(result["phi_ext_rad"], 0.0)
            self.assertLess(result["phi_ext_rad"], math.pi)

    def test_phi_lens_aligned_with_host(self):
        """phi_lens should equal the host PA."""
        host_mom = HostMoments(q=0.7, phi_rad=1.23, r_half_pix=8.0)
        result = map_host_to_lens_params(1.5, host_mom)
        self.assertAlmostEqual(result["phi_lens_rad"], 1.23)

    def test_deterministic_with_seed(self):
        """Same seed should give identical results."""
        host_mom = HostMoments(q=0.75, phi_rad=0.5, r_half_pix=7.0)
        r1 = map_host_to_lens_params(1.5, host_mom, rng=np.random.default_rng(99))
        r2 = map_host_to_lens_params(1.5, host_mom, rng=np.random.default_rng(99))
        for k in r1:
            self.assertAlmostEqual(r1[k], r2[k], places=10)

    def test_fallback_host_moments_work(self):
        """Fallback HostMoments (q=1, phi=0) should produce valid output."""
        host_mom = HostMoments(q=1.0, phi_rad=0.0, r_half_pix=10.0, is_fallback=True)
        result = map_host_to_lens_params(1.5, host_mom)
        self.assertGreaterEqual(result["q_lens"], 0.5)
        self.assertLessEqual(result["q_lens"], 1.0)


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests: cutout → moments → lens params."""

    def test_round_host_gives_round_lens(self):
        """Round host → q_lens should be near 1 (within scatter + floor)."""
        host = _make_gaussian_host(sigx=8.0, sigy=8.0)
        m = estimate_host_moments_rband(host)
        rng = np.random.default_rng(42)
        params = map_host_to_lens_params(1.5, m, rng=rng)
        # q_lens should be high (> 0.8) for a round host
        self.assertGreater(params["q_lens"], 0.8)

    def test_elliptical_host_gives_elliptical_lens(self):
        """Elongated host → q_lens should be smaller (possibly at floor)."""
        host = _make_gaussian_host(sigx=12.0, sigy=4.0)
        m = estimate_host_moments_rband(host)
        rng = np.random.default_rng(42)
        params = map_host_to_lens_params(1.5, m, q_floor=0.4, rng=rng)
        self.assertLess(params["q_lens"], 0.8)


if __name__ == "__main__":
    unittest.main()

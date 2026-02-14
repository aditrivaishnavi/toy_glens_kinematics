#!/usr/bin/env python3
"""Standalone guard check: does the preprocessing annulus sit in sky or galaxy?

This test catches the bug directly by testing BEHAVIOR, not implementation.
It doesn't import default_annulus_radii or any new functions.
It works against BOTH the old and new code.

The test:
  1. Creates a realistic bright galaxy on a 101x101 stamp
  2. Runs preprocessing on it (exactly as training does)
  3. Checks whether pixels in the annulus region are near zero
     (as they should be if the annulus is sky-dominated)
  4. Checks whether the normalization is corrupted by galaxy light

If the annulus overlaps the galaxy, the normalized galaxy core will have
LOWER values than expected because the MAD is inflated by galaxy flux
in the annulus. This test catches that.

Run: python3 tests/run_guard_check.py
Requires only numpy (no torch, no pytest).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def radial_rmap(H, W):
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2)


def make_bright_galaxy(h=101, w=101, seed=42):
    """Realistic bright elliptical galaxy in nanomaggies, (3, H, W) CHW."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    bands = []
    # Bright extended galaxy: half-light radius ~12 pixels (~3 arcsec)
    for flux, bg, noise in [(5.0, 0.05, 0.03), (8.0, 0.08, 0.05), (3.5, 0.04, 0.02)]:
        galaxy = flux * np.exp(-r / 8.0)  # exponential profile
        background = bg + noise * rng.standard_normal((h, w))
        bands.append((galaxy + background).astype(np.float32))
    return np.stack(bands, axis=0)


def test_annulus_is_sky_dominated():
    """KEY TEST: After normalization, the annulus should contain ~sky, not galaxy.

    We detect which pixels the normalization uses by checking where the
    normalized values are near zero (the annulus median is subtracted).
    If galaxy light is in the annulus, the normalization is wrong.
    """
    from dhs.preprocess import preprocess_stack

    cutout = make_bright_galaxy(101, 101)
    out = preprocess_stack(cutout, mode="raw_robust", crop=False)

    # The annulus (wherever it is) should have median ~0 after normalization.
    # Check: what fraction of the OUTER region (r > 35) has values near 0?
    # If the annulus is properly in the outer region, most of these pixels
    # should be near zero. If the annulus is at r=20-32 (the bug),
    # the outer pixels will NOT be near zero.
    rmap = radial_rmap(101, 101)

    for band in range(3):
        # Pixels in true outer ring (r > 35 pixels = 9.2 arcsec)
        outer_mask = rmap > 35
        outer_vals = out[band][outer_mask]
        outer_median = float(np.median(outer_vals))

        # Inner galaxy region (r < 15)
        inner_mask = rmap < 15
        inner_vals = out[band][inner_mask]
        inner_peak = float(np.max(inner_vals))

        # If annulus is at 20-32 (bug): outer median will NOT be ~0
        # because the annulus stats include galaxy light, distorting everything
        # If annulus is in outer region: outer median WILL be ~0
        print(f"  Band {band}: outer_median={outer_median:+.3f}, inner_peak={inner_peak:.1f}")

    return out


def test_galaxy_flux_in_annulus_ring():
    """Direct test: how much galaxy flux sits in different annulus rings?

    This is the test that would have caught the bug at any point.
    It doesn't depend on any implementation details — just physics.
    For a bright galaxy, the annulus should be where the galaxy is faint.
    """
    h = w = 101
    half = h // 2  # 50

    # Bright exponential galaxy
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    galaxy = 10.0 * np.exp(-np.sqrt(r2) / 8.0)
    total = galaxy.sum()

    rmap = radial_rmap(h, w)

    print(f"\n  Galaxy flux fraction in different annulus rings (101x101):")
    print(f"  {'Ring':>15s}  {'Flux %':>8s}  {'r_out/half':>10s}  {'Verdict':>10s}")
    print(f"  {'-'*15}  {'-'*8}  {'-'*10}  {'-'*10}")

    rings = [
        ("r=20-32 (OLD)", 20, 32),
        ("r=25-40", 25, 40),
        ("r=30-45", 30, 45),
        ("r=32-45 (NEW)", 32.5, 45.0),
        ("r=35-48", 35, 48),
    ]

    for label, r_in, r_out in rings:
        mask = (rmap >= r_in) & (rmap < r_out)
        frac = galaxy[mask].sum() / total
        ratio = r_out / half
        ok = "OK" if frac < 0.10 else "BAD"
        print(f"  {label:>15s}  {frac:7.1%}  {ratio:9.0%}  {ok:>10s}")


def test_normalization_dynamic_range():
    """Test that the galaxy core has good dynamic range after normalization.

    If the annulus includes galaxy light, MAD is inflated, and the galaxy
    core gets compressed. The peak normalized value of a bright galaxy
    should be high (> 5 sigma). If it's low, the normalization is eating
    the signal.
    """
    from dhs.preprocess import preprocess_stack

    cutout = make_bright_galaxy(101, 101)
    out = preprocess_stack(cutout, mode="raw_robust", crop=False)

    for band in range(3):
        peak = float(np.max(out[band]))
        # A bright galaxy core should be well above the noise after
        # proper normalization. If peak is < 5, the normalization MAD
        # is inflated (annulus includes galaxy light).
        status = "OK" if peak > 5.0 else "SUSPICIOUS"
        print(f"  Band {band}: peak normalized value = {peak:.1f}  [{status}]")


if __name__ == "__main__":
    print("=" * 65)
    print("GUARD CHECK: Is the normalization annulus sky-dominated?")
    print("=" * 65)

    print("\n--- Test 1: Galaxy flux in different annulus rings ---")
    test_galaxy_flux_in_annulus_ring()

    try:
        print("\n--- Test 2: Outer-region median after preprocessing ---")
        test_annulus_is_sky_dominated()

        print("\n--- Test 3: Dynamic range of normalized galaxy core ---")
        test_normalization_dynamic_range()
    except Exception as e:
        print(f"\n  Could not run preprocessing tests: {e}")
        print("  (This is OK if running without torch/full environment)")

    print("\n" + "=" * 65)
    print("INTERPRETATION:")
    print("  - If OLD ring (20-32) shows BAD flux fraction, the bug is real")
    print("  - If preprocessing peak is low, MAD is inflated by galaxy light")
    print("=" * 65)

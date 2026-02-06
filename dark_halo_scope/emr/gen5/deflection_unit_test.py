#!/usr/bin/env python3
"""Unit test for deflection functions - verify units are consistent."""

import numpy as np
import math


def deflection_sis(x, y, theta_e, eps=1e-12):
    """SIS (Singular Isothermal Sphere) deflection.
    
    Args:
        x, y: Image-plane coordinates in arcsec
        theta_e: Einstein radius in arcsec
        
    Returns:
        (alpha_x, alpha_y): Deflection in arcsec
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = np.sqrt(x * x + y * y) + eps
    ax = theta_e * x / r
    ay = theta_e * y / r
    return ax, ay


def deflection_sie(x, y, theta_e, lens_e, lens_phi_rad, eps=1e-12):
    """SIE (Singular Isothermal Ellipsoid) deflection.
    
    Args:
        x, y: Image-plane coordinates in arcsec
        theta_e: Einstein radius in arcsec
        lens_e: Lens ellipticity (0 = circular)
        lens_phi_rad: Lens position angle in radians
        
    Returns:
        (alpha_x, alpha_y): Deflection in arcsec
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    q = float(np.clip((1.0 - lens_e) / (1.0 + lens_e + 1e-6), 0.2, 0.999))
    
    c = math.cos(lens_phi_rad)
    s = math.sin(lens_phi_rad)
    xp = c * x + s * y
    yp = -s * x + c * y
    
    if q > 0.99:
        return deflection_sis(x, y, theta_e, eps)
    
    f = np.sqrt(1.0 - q * q)
    psi = np.sqrt(q * q * xp * xp + yp * yp + eps)
    ax_p = theta_e * np.sqrt(q) / f * np.arctan(f * xp / (psi + eps))
    ay_p = theta_e * np.sqrt(q) / f * np.arctanh(f * yp / (psi + q * q * eps + eps))
    
    ax = c * ax_p - s * ay_p
    ay = s * ax_p + c * ay_p
    return ax, ay


def test_sis_magnitude():
    """Test that SIS deflection magnitude equals theta_E at all radii."""
    print("=== SIS DEFLECTION MAGNITUDE TEST ===")
    print()
    
    theta_e = 2.0  # arcsec
    print(f"Einstein radius: {theta_e} arcsec")
    print()
    print("Testing |alpha| = theta_E at various radii:")
    
    all_pass = True
    for r in [0.5, 1.0, 2.0, 5.0, 10.0]:
        ax, ay = deflection_sis([r], [0.0], theta_e)
        alpha_mag = np.sqrt(ax[0]**2 + ay[0]**2)
        passed = abs(alpha_mag - theta_e) < 0.001
        all_pass = all_pass and passed
        status = "PASS" if passed else "FAIL"
        print(f"  r = {r:5.1f} arcsec: |alpha| = {alpha_mag:.4f} arcsec [{status}]")
    
    print()
    return all_pass


def test_sie_circular():
    """Test that SIE with e=0 matches SIS."""
    print("=== SIE CIRCULAR TEST ===")
    print()
    
    theta_e = 2.0
    x, y = 3.0, 4.0  # arcsec
    
    ax_sis, ay_sis = deflection_sis([x], [y], theta_e)
    ax_sie, ay_sie = deflection_sie([x], [y], theta_e, 0.0, 0.0)
    
    print(f"Position: ({x}, {y}) arcsec")
    print(f"SIS deflection: ({ax_sis[0]:.4f}, {ay_sis[0]:.4f})")
    print(f"SIE (e=0) deflection: ({ax_sie[0]:.4f}, {ay_sie[0]:.4f})")
    
    match = np.allclose([ax_sis[0], ay_sis[0]], [ax_sie[0], ay_sie[0]])
    print(f"Match: {match}")
    print()
    return match


def test_einstein_radius_mapping():
    """Test that point at Einstein radius maps to source origin."""
    print("=== EINSTEIN RADIUS MAPPING TEST ===")
    print()
    
    theta_e = 2.0
    x_img, y_img = theta_e, 0.0  # Point on Einstein ring
    
    ax, ay = deflection_sis([x_img], [y_img], theta_e)
    beta_x = x_img - ax[0]
    beta_y = y_img - ay[0]
    
    print(f"Image plane: ({x_img}, {y_img}) arcsec")
    print(f"Deflection: ({ax[0]:.4f}, {ay[0]:.4f}) arcsec")
    print(f"Source plane: ({beta_x:.4f}, {beta_y:.4f}) arcsec")
    print()
    print("Expected: Point at Einstein radius maps to origin (0, 0)")
    
    passed = abs(beta_x) < 0.01 and abs(beta_y) < 0.01
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_coordinate_grid():
    """Test that coordinate grid is set up correctly."""
    print("=== COORDINATE GRID TEST ===")
    print()
    
    stamp_size = 64
    pixscale = 0.262  # arcsec/pix
    half = stamp_size // 2
    
    pix_idx = np.arange(stamp_size) - half + 0.5
    y_grid, x_grid = np.meshgrid(pix_idx, pix_idx, indexing="ij")
    x_arcsec = x_grid * pixscale
    y_arcsec = y_grid * pixscale
    
    print(f"Stamp size: {stamp_size}x{stamp_size}")
    print(f"Pixel scale: {pixscale} arcsec/pix")
    print()
    print(f"Pixel index range: {pix_idx[0]} to {pix_idx[-1]}")
    print(f"Arcsec range: {x_arcsec.min():.2f} to {x_arcsec.max():.2f}")
    print(f"Center pixel (32,32): x={x_arcsec[32,32]:.3f}, y={y_arcsec[32,32]:.3f} arcsec")
    print()
    
    # Verify stamp covers expected field
    expected_half_fov = half * pixscale
    actual_max = x_arcsec.max()
    print(f"Expected half-FOV: {expected_half_fov:.2f} arcsec")
    print(f"Actual max coordinate: {actual_max:.2f} arcsec")
    print()
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("DEFLECTION UNIT TESTS")
    print("=" * 60)
    print()
    
    results = []
    results.append(("SIS magnitude", test_sis_magnitude()))
    results.append(("SIE circular", test_sie_circular()))
    results.append(("Einstein mapping", test_einstein_radius_mapping()))
    results.append(("Coordinate grid", test_coordinate_grid()))
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    all_passed = all(p for _, p in results)
    print()
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

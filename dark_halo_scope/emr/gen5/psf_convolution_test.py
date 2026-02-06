#!/usr/bin/env python3
"""Unit test demonstrating PSF convolution bug in _fft_convolve2d."""

import numpy as np
import math


def _gaussian_kernel2d(sigma_pix, radius=None, max_side=63):
    """Build a normalized circular Gaussian PSF kernel."""
    sigma_pix = float(sigma_pix)
    if sigma_pix <= 0:
        k = np.zeros((1, 1), dtype=np.float32)
        k[0, 0] = 1.0
        return k
    if radius is None:
        radius = int(max(3, math.ceil(4.0 * sigma_pix)))
        max_radius = (max_side - 1) // 2
        radius = min(radius, max_radius)
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    rr2 = (xx * xx + yy * yy).astype(np.float32)
    k = np.exp(-0.5 * rr2 / (sigma_pix * sigma_pix)).astype(np.float32)
    k /= np.sum(k)
    return k


def _fft_convolve2d_buggy(img, kernel):
    """BUGGY version from production code (lines 879-889)."""
    ih, iw = img.shape
    kh, kw = kernel.shape
    if kh > ih or kw > iw:
        raise ValueError(f"Kernel {kernel.shape} larger than image {img.shape}")
    pad = np.zeros((ih, iw), dtype=np.float32)
    pad[:kh, :kw] = kernel.astype(np.float32)
    pad = np.fft.ifftshift(pad)  # BUG: This does NOT center kernel at (0,0)
    out = np.fft.ifft2(np.fft.fft2(img.astype(np.float32)) * np.fft.fft2(pad)).real
    return out.astype(np.float32)


def _fft_convolve2d_correct(img, kernel):
    """CORRECT version with proper kernel centering."""
    ih, iw = img.shape
    kh, kw = kernel.shape
    if kh > ih or kw > iw:
        raise ValueError(f"Kernel {kernel.shape} larger than image {img.shape}")
    pad = np.zeros((ih, iw), dtype=np.float32)
    pad[:kh, :kw] = kernel.astype(np.float32)
    # Roll to move kernel center to (0,0)
    oy, ox = kh // 2, kw // 2
    pad = np.roll(np.roll(pad, -oy, axis=0), -ox, axis=1)
    out = np.fft.ifft2(np.fft.fft2(img.astype(np.float32)) * np.fft.fft2(pad)).real
    return out.astype(np.float32)


def test_impulse_response():
    """Test that impulse at (32,32) stays at (32,32) after convolution."""
    print("=== IMPULSE RESPONSE TEST ===")
    
    # PSF parameters (typical for DESI)
    psf_fwhm_pix = 5.0
    sigma = psf_fwhm_pix / 2.355
    k = _gaussian_kernel2d(sigma)
    
    print(f"PSF FWHM: {psf_fwhm_pix} pix, sigma: {sigma:.2f} pix")
    print(f"Kernel shape: {k.shape}")
    print()
    
    # Create impulse at center
    img = np.zeros((64, 64), dtype=np.float32)
    img[32, 32] = 1.0
    
    # Convolve with buggy and correct versions
    out_buggy = _fft_convolve2d_buggy(img, k)
    out_correct = _fft_convolve2d_correct(img, k)
    
    peak_buggy = np.unravel_index(np.argmax(out_buggy), out_buggy.shape)
    peak_correct = np.unravel_index(np.argmax(out_correct), out_correct.shape)
    
    print(f"Impulse location: (32, 32)")
    print(f"Peak after BUGGY convolution:   {peak_buggy}")
    print(f"Peak after CORRECT convolution: {peak_correct}")
    print()
    
    shift = (peak_buggy[0] - 32, peak_buggy[1] - 32)
    print(f"Shift caused by bug: {shift}")
    print()
    
    # Verify flux preservation
    print(f"Flux preservation:")
    print(f"  Input flux:  {np.sum(img):.6f}")
    print(f"  Buggy flux:  {np.sum(out_buggy):.6f}")
    print(f"  Correct flux:{np.sum(out_correct):.6f}")
    print()
    
    if peak_buggy != (32, 32):
        print("BUG CONFIRMED: Buggy version shifts output!")
    if peak_correct == (32, 32):
        print("CORRECT version preserves position")


def test_ring_convolution():
    """Test convolution of a ring (simulating arc at theta_E)."""
    print()
    print("=== RING CONVOLUTION TEST ===")
    
    # Create ring at radius 8 (like arc at theta_E ~ 2")
    size = 64
    radius = 8
    y, x = np.ogrid[:size, :size]
    cy, cx = size // 2, size // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    ring = np.exp(-0.5 * ((r - radius) / 1.5)**2).astype(np.float32)
    
    # PSF
    psf_fwhm_pix = 5.0
    sigma = psf_fwhm_pix / 2.355
    k = _gaussian_kernel2d(sigma)
    
    # Convolve
    ring_buggy = _fft_convolve2d_buggy(ring, k)
    ring_correct = _fft_convolve2d_correct(ring, k)
    
    # Check peak locations
    peak_buggy = np.unravel_index(np.argmax(ring_buggy), ring_buggy.shape)
    peak_correct = np.unravel_index(np.argmax(ring_correct), ring_correct.shape)
    
    print(f"Ring at radius {radius} from center")
    print(f"Peak after BUGGY:   {peak_buggy}")
    print(f"Peak after CORRECT: {peak_correct}")
    print()
    
    # Check center value
    print(f"Value at center (32, 32):")
    print(f"  Buggy:   {ring_buggy[32, 32]:.6f}")
    print(f"  Correct: {ring_correct[32, 32]:.6f}")


if __name__ == "__main__":
    test_impulse_response()
    test_ring_convolution()

"""
Utilities shared across Gen6/7/8/uber dataset components.

Design goals:
- Deterministic, hash-based randomness (reproducible across Spark workers and reruns)
- Minimal dependencies (numpy + optional astropy/pyarrow when needed)
- Flux-conserving resampling (bilinear) and safe image padding/cropping
"""
from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def to_surface_brightness(flux_per_pixel: np.ndarray, pixscale_arcsec: float) -> np.ndarray:
    """Convert flux/pixel to surface brightness (flux/arcsec²).
    
    IMPORTANT: lenstronomy INTERPOL expects surface brightness, not flux/pixel.
    This was the source of the Gen5 ~1000x flux bug.
    
    Parameters:
        flux_per_pixel: Image in flux/pixel units
        pixscale_arcsec: Pixel scale in arcsec/pixel
    
    Returns:
        Image in flux/arcsec² units
    """
    pixel_area_arcsec2 = pixscale_arcsec ** 2
    return flux_per_pixel / pixel_area_arcsec2


def from_surface_brightness(surface_brightness: np.ndarray, pixscale_arcsec: float) -> np.ndarray:
    """Convert surface brightness (flux/arcsec²) to flux/pixel.
    
    Parameters:
        surface_brightness: Image in flux/arcsec² units
        pixscale_arcsec: Pixel scale in arcsec/pixel
    
    Returns:
        Image in flux/pixel units
    """
    pixel_area_arcsec2 = pixscale_arcsec ** 2
    return surface_brightness * pixel_area_arcsec2


def blake2b_u64(*parts: str, salt: str = "") -> int:
    """Deterministic 64-bit integer hash for arbitrary string parts."""
    h = hashlib.blake2b(digest_size=8)
    if salt:
        h.update(salt.encode("utf-8"))
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "little", signed=False)


def rng_from_hash(key: str, salt: str = "") -> np.random.Generator:
    """Create a numpy RNG seeded from a stable hash."""
    seed = blake2b_u64(key, salt=salt) % (2**32)
    return np.random.default_rng(seed)


def categorical_from_hash(key: str, probs: np.ndarray, salt: str = "") -> int:
    """
    Deterministically sample an index from a categorical distribution using a hashed RNG.

    probs must sum to 1 (within tolerance).
    """
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    u = (blake2b_u64(key, salt=salt) / float(2**64 - 1))
    cdf = np.cumsum(probs)
    return int(np.searchsorted(cdf, u, side="right"))


def pad_or_crop_center(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Center pad/crop a 2D array to out_hw."""
    H, W = img.shape
    outH, outW = out_hw
    out = np.zeros((outH, outW), dtype=img.dtype)

    y0 = max(0, (H - outH) // 2)
    x0 = max(0, (W - outW) // 2)
    y1 = min(H, y0 + outH)
    x1 = min(W, x0 + outW)

    oy0 = max(0, (outH - H) // 2)
    ox0 = max(0, (outW - W) // 2)
    oy1 = oy0 + (y1 - y0)
    ox1 = ox0 + (x1 - x0)

    out[oy0:oy1, ox0:ox1] = img[y0:y1, x0:x1]
    return out


def bilinear_resample(img: np.ndarray, scale_y: float, scale_x: float) -> np.ndarray:
    """
    Flux-conserving bilinear resample of a 2D image.

    If scale < 1, downsamples; if scale > 1, upsamples.
    Flux conservation: divide by scale_y*scale_x after interpolation so total sum is preserved
    approximately.
    """
    if img.ndim != 2:
        raise ValueError("bilinear_resample expects a 2D array")

    H, W = img.shape
    outH = max(1, int(round(H * scale_y)))
    outW = max(1, int(round(W * scale_x)))

    ys = (np.arange(outH) + 0.5) / scale_y - 0.5
    xs = (np.arange(outW) + 0.5) / scale_x - 0.5

    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1

    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)

    wy = ys - y0
    wx = xs - x0

    wy2 = wy[:, None]
    wx2 = wx[None, :]

    Ia = img[y0c[:, None], x0c[None, :]]
    Ib = img[y0c[:, None], x1c[None, :]]
    Ic = img[y1c[:, None], x0c[None, :]]
    Id = img[y1c[:, None], x1c[None, :]]

    out = (Ia * (1 - wy2) * (1 - wx2) +
           Ib * (1 - wy2) * wx2 +
           Ic * wy2 * (1 - wx2) +
           Id * wy2 * wx2)

    out = out / max(1e-12, (scale_y * scale_x))
    return out


def fft_convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution via FFT, returning same shape as img (centered)."""
    if img.ndim != 2 or kernel.ndim != 2:
        raise ValueError("fft_convolve2d expects 2D arrays")
    H, W = img.shape
    kH, kW = kernel.shape
    padH = H + kH - 1
    padW = W + kW - 1
    F = np.fft.rfft2(img, s=(padH, padW))
    K = np.fft.rfft2(kernel, s=(padH, padW))
    out = np.fft.irfft2(F * K, s=(padH, padW))
    y0 = (kH - 1) // 2
    x0 = (kW - 1) // 2
    return out[y0:y0 + H, x0:x0 + W]


def gaussian_kernel(fwhm_pix: float, size: int = 33, max_size: int | None = None) -> np.ndarray:
    """Normalized circular Gaussian PSF kernel.

    Parameters
    - fwhm_pix: FWHM in pixels
    - size: desired kernel size (will be forced to odd)
    - max_size: optional hard cap to prevent kernels larger than the stamp

    Notes
    - If max_size is provided, size is clamped to <= max_size.
    - size is forced to be odd and at least 3.
    """
    if fwhm_pix <= 0:
        raise ValueError("fwhm_pix must be positive")
    if max_size is not None:
        size = min(int(size), int(max_size))
    size = int(size)
    if size < 3:
        size = 3
    if size % 2 == 0:
        size -= 1
    sigma = fwhm_pix / 2.354820045
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    s = float(np.sum(k))
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Invalid Gaussian kernel normalization")
    k /= s
    return k.astype(np.float32)

def moffat_kernel(fwhm_pix: float, beta: float = 3.5, size: int = 33, max_size: int | None = None) -> np.ndarray:
    """Normalized circular Moffat kernel with safety caps."""
    if fwhm_pix <= 0:
        raise ValueError("fwhm_pix must be positive")
    if beta <= 1.0:
        raise ValueError("beta must be > 1.0")
    if max_size is not None:
        size = min(int(size), int(max_size))
    size = int(size)
    if size < 3:
        size = 3
    if size % 2 == 0:
        size -= 1
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2
    xx, yy = np.meshgrid(ax, ax)
    alpha = fwhm_pix / (2.0 * math.sqrt(2.0**(1.0 / beta) - 1.0))
    rr2 = (xx**2 + yy**2) / (alpha**2)
    k = (1.0 + rr2) ** (-beta)
    s = float(np.sum(k))
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Invalid Moffat kernel normalization")
    k /= s
    return k.astype(np.float32)

def elliptical_moffat_kernel(
    fwhm_pix: float,
    beta: float,
    e: float,
    phi: float,
    size: int = 33,
    max_size: int | None = None,
) -> np.ndarray:
    """Elliptical Moffat kernel to approximate PSF anisotropy."""
    if fwhm_pix <= 0:
        raise ValueError("fwhm_pix must be positive")
    if beta <= 1.0:
        raise ValueError("beta must be > 1.0")
    if max_size is not None:
        size = min(int(size), int(max_size))
    size = int(size)
    if size < 3:
        size = 3
    if size % 2 == 0:
        size -= 1

    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2
    xx, yy = np.meshgrid(ax, ax)
    c, s = math.cos(phi), math.sin(phi)
    x = c * xx + s * yy
    y = -s * xx + c * yy

    q = float(np.clip(1.0 - float(e), 0.5, 1.0))
    y = y / q

    alpha = fwhm_pix / (2.0 * math.sqrt(2.0**(1.0 / beta) - 1.0))
    rr2 = (x**2 + y**2) / (alpha**2)
    k = (1.0 + rr2) ** (-beta)
    s = float(np.sum(k))
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Invalid elliptical Moffat kernel normalization")
    k /= s
    return k.astype(np.float32)


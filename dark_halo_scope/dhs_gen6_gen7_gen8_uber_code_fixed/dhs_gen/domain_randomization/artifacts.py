"""
Gen8: Domain randomization utilities for injection realism.

Artifacts included:
- PSF anisotropy (elliptical Moffat)
- Low-frequency background residuals (plane)
- Cosmic rays (streaks)
- Saturation spikes/wings (approx)
- Astrometric jitter (subpixel shift)

All transformations are deterministic given (task_id, salt).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..utils import rng_from_hash, fft_convolve2d, elliptical_moffat_kernel, gaussian_kernel


@dataclass
class ArtifactConfig:
    enable_psf_anisotropy: bool = True
    psf_e_sigma: float = 0.05
    enable_bg_plane: bool = True
    bg_plane_amp: float = 0.02  # fraction of image MAD
    enable_cosmic_rays: bool = True
    cosmic_rate: float = 0.15
    cosmic_amp: float = 8.0
    cosmic_length_pix: Tuple[int, int] = (8, 24)
    enable_sat_wings: bool = True
    sat_rate: float = 0.08
    sat_amp: float = 12.0
    sat_r0_pix: float = 2.0
    enable_astrom_jitter: bool = True
    jitter_sigma_pix: float = 0.25


def robust_mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad + 1e-12


def add_background_plane(img: np.ndarray, rng: np.random.Generator, amp: float) -> np.ndarray:
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    yy = (yy - (H - 1) / 2) / max(1.0, H / 2)
    xx = (xx - (W - 1) / 2) / max(1.0, W / 2)
    a = float(rng.normal(0, 1))
    b = float(rng.normal(0, 1))
    plane = a * xx + b * yy
    plane = plane / (np.std(plane) + 1e-12)
    scale = amp * robust_mad(img)
    return img + scale * plane.astype(img.dtype)


def add_cosmic_ray(img: np.ndarray, rng: np.random.Generator, amp_mad: float, length_range: Tuple[int, int]) -> np.ndarray:
    H, W = img.shape
    mad = robust_mad(img)
    amp = amp_mad * mad
    x0 = float(rng.uniform(0, W - 1))
    y0 = float(rng.uniform(0, H - 1))
    theta = float(rng.uniform(0, 2 * math.pi))
    L = float(rng.integers(length_range[0], length_range[1] + 1))
    x1 = x0 + L * math.cos(theta)
    y1 = y0 + L * math.sin(theta)

    n = int(max(8, L * 2))
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    out = img.copy()
    for x, y in zip(xs, ys):
        ix = int(round(x)); iy = int(round(y))
        if 0 <= ix < W and 0 <= iy < H:
            out[iy, ix] += amp
            if ix + 1 < W: out[iy, ix + 1] += 0.4 * amp
            if ix - 1 >= 0: out[iy, ix - 1] += 0.4 * amp
    return out


def add_saturation_wings(img: np.ndarray, rng: np.random.Generator, amp_mad: float, r0_pix: float) -> np.ndarray:
    H, W = img.shape
    mad = robust_mad(img)
    amp = amp_mad * mad
    cx = float(rng.uniform(0, W - 1))
    cy = float(rng.uniform(0, H - 1))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = np.abs(xx - cx)
    dy = np.abs(yy - cy)
    cross = (1.0 / (r0_pix + dx) ** 1.5) + (1.0 / (r0_pix + dy) ** 1.5)
    cross /= np.max(cross) + 1e-12
    return img + amp * cross.astype(img.dtype)


def shift_subpixel(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xs = xx - dx
    ys = yy - dy

    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    wx = xs - x0
    wy = ys - y0

    Ia = img[y0c, x0c]
    Ib = img[y0c, x1c]
    Ic = img[y1c, x0c]
    Id = img[y1c, x1c]

    out = (Ia * (1 - wx) * (1 - wy) +
           Ib * wx * (1 - wy) +
           Ic * (1 - wx) * wy +
           Id * wx * wy)
    return out.astype(img.dtype)


def apply_domain_randomization(
    img: np.ndarray,
    key: str,
    psf_fwhm_pix: Optional[float] = None,
    psf_model: str = "moffat",
    moffat_beta: float = 3.5,
    cfg: ArtifactConfig = ArtifactConfig(),
    salt: str = "",
    max_kernel_size: int = 63,  # Prevent kernel > stamp size (Gen5 lesson)
) -> Dict[str, object]:
    rng = rng_from_hash(key, salt=salt)
    out = img.astype(np.float32, copy=True)
    meta: Dict[str, object] = {}

    if cfg.enable_bg_plane:
        out = add_background_plane(out, rng, amp=cfg.bg_plane_amp)
        meta["bg_plane"] = True

    if cfg.enable_psf_anisotropy and psf_fwhm_pix is not None:
        e = float(np.clip(abs(rng.normal(0, cfg.psf_e_sigma)), 0.0, 0.25))
        phi = float(rng.uniform(0, 2 * math.pi))
        if psf_model == "moffat":
            k = elliptical_moffat_kernel(psf_fwhm_pix, beta=moffat_beta, e=e, phi=phi, size=33, max_size=max_kernel_size)
        else:
            k = gaussian_kernel(psf_fwhm_pix, size=33, max_size=max_kernel_size)
        out = fft_convolve2d(out, k)
        meta["psf_aniso_e"] = e
        meta["psf_aniso_phi"] = phi

    if cfg.enable_cosmic_rays and float(rng.uniform()) < cfg.cosmic_rate:
        out = add_cosmic_ray(out, rng, amp_mad=cfg.cosmic_amp, length_range=cfg.cosmic_length_pix)
        meta["cosmic_ray"] = True

    if cfg.enable_sat_wings and float(rng.uniform()) < cfg.sat_rate:
        out = add_saturation_wings(out, rng, amp_mad=cfg.sat_amp, r0_pix=cfg.sat_r0_pix)
        meta["sat_wings"] = True

    if cfg.enable_astrom_jitter:
        dx = float(rng.normal(0, cfg.jitter_sigma_pix))
        dy = float(rng.normal(0, cfg.jitter_sigma_pix))
        out = shift_subpixel(out, dx=dx, dy=dy)
        meta["jitter_dx"] = dx
        meta["jitter_dy"] = dy

    if not np.isfinite(out).all():
        raise ValueError(f"NaN/Inf detected after domain randomization for key={key}")

    return {"img": out.astype(np.float32), "meta": meta}

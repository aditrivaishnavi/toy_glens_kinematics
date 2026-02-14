"""
Host-conditioned lens parameter estimation from cutout image moments.

Implements "Model 2" injection realism: instead of drawing lens q and PA
from independent priors (Model 1), we estimate the host galaxy's axis ratio
and position angle from the r-band second moments of the cutout and use
those to condition the SIE lens mass model.

Physical motivation:
    Strong lensing deflectors are typically massive elliptical galaxies whose
    mass distribution is roughly aligned with the light distribution.  By
    conditioning q_lens and PA_lens on the host light, injected arcs are
    visually consistent with the host galaxy — reproducing the joint
    "deflector + arc" feature that trained CNNs appear to use.

References:
    - Kormann et al. (1994): SIE mass model
    - Koopmans et al. (2006): mass-light alignment in SLACS lenses
    - MNRAS_RAW_NOTES.md §9.3: Model 2 specification

Adapted from LLM-suggested starter code (suggested_injection_models_package/),
with the following changes:
    - Input validation and shape checks
    - Robust handling of NaN/Inf pixels, all-zero images, very faint hosts
    - Negative eigenvalue guard (noisy moment matrices)
    - Configurable q_min floor in moment estimation
    - center_radius_pix clamped to half-image size
    - r_half_pix = 0 guard
    - Comprehensive logging for diagnostic edge cases
    - map_host_to_lens_params returns dict compatible with dhs.injection_engine.LensParams
    - Added scatter in q_lens alignment (not perfectly deterministic)

Author: stronglens_calibration project
Date: 2026-02-13
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class HostMoments:
    """Second-moment shape estimate for a host galaxy cutout."""

    q: float  # axis ratio b/a, in (0, 1]
    phi_rad: float  # position angle of major axis, radians (angle from x-axis in pixel coordinates)
    r_half_pix: float  # half-light radius proxy, pixels
    is_fallback: bool = False  # True if moments were unreliable and defaults used


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _robust_centered(img: np.ndarray, clip_pct: float = 99.5) -> np.ndarray:
    """Subtract robust background and clip outliers.

    Parameters
    ----------
    img : (H, W) float array
        Single-band image in flux units.
    clip_pct : float
        Percentile for symmetric clipping (removes cosmic rays / artifacts).

    Returns
    -------
    (H, W) float64 array, background-subtracted and clipped, NaN→0.
    """
    x = img.astype(np.float64, copy=True)
    med = np.nanmedian(x)
    if not np.isfinite(med):
        return np.zeros_like(x)
    y = x - med
    lo = np.nanpercentile(y, 100.0 - clip_pct)
    hi = np.nanpercentile(y, clip_pct)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.zeros_like(x)
    y = np.clip(y, lo, hi)
    y[~np.isfinite(y)] = 0.0
    return y


# ---------------------------------------------------------------------------
# Core moment estimation
# ---------------------------------------------------------------------------
def estimate_host_moments_rband(
    host_hwc_nmgy: np.ndarray,
    r_band_index: int = 1,
    center_radius_pix: int = 40,
    q_min: float = 0.2,
) -> HostMoments:
    """Estimate axis ratio and PA from flux-weighted second moments.

    Computes the flux-weighted inertia tensor of the r-band image inside a
    circular aperture, then derives (q, PA) from its eigenvalues.

    Parameters
    ----------
    host_hwc_nmgy : (H, W, C) array
        Host cutout in nanomaggies, HWC layout with C >= r_band_index + 1.
    r_band_index : int
        Index of the r-band in the channel axis.
    center_radius_pix : int
        Aperture radius for moment computation (pixels).  Will be clamped
        to ``min(H, W) // 2`` if larger than the image half-size.
    q_min : float
        Floor on the returned axis ratio (prevents unphysically extreme
        ellipticities from noise).

    Returns
    -------
    HostMoments
        Estimated shape parameters.  If the image is too faint or noisy
        to measure, returns ``HostMoments(q=1.0, phi_rad=0.0,
        r_half_pix=10.0, is_fallback=True)``.
    """
    _FALLBACK = HostMoments(q=1.0, phi_rad=0.0, r_half_pix=10.0, is_fallback=True)

    # --- Input validation ---
    if host_hwc_nmgy.ndim != 3:
        logger.warning(
            "host_hwc_nmgy has ndim=%d, expected 3 (H, W, C); returning fallback",
            host_hwc_nmgy.ndim,
        )
        return _FALLBACK
    H, W, C = host_hwc_nmgy.shape
    if r_band_index >= C:
        logger.warning(
            "r_band_index=%d but C=%d; returning fallback", r_band_index, C
        )
        return _FALLBACK

    img = host_hwc_nmgy[..., r_band_index]
    img = _robust_centered(img)

    # Clamp aperture to image half-size
    max_radius = min(H, W) // 2
    eff_radius = min(center_radius_pix, max_radius)

    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = rr <= float(eff_radius)

    w = img[mask].copy()
    w = np.maximum(w, 0.0)  # only positive flux contributes to shape

    total_flux = w.sum()
    if not np.isfinite(total_flux) or total_flux <= 0:
        logger.debug("Host has no positive flux in aperture; returning fallback")
        return _FALLBACK

    x = (xx[mask] - cx).astype(np.float64)
    y = (yy[mask] - cy).astype(np.float64)

    # Centroid refinement
    norm = total_flux
    mx = (w * x).sum() / norm
    my = (w * y).sum() / norm
    x = x - mx
    y = y - my

    # Second moments (inertia tensor)
    mxx = (w * x * x).sum() / norm
    myy = (w * y * y).sum() / norm
    mxy = (w * x * y).sum() / norm

    # Eigendecomposition of the 2x2 covariance matrix
    cov = np.array([[mxx, mxy], [mxy, myy]], dtype=np.float64)
    evals, evecs = np.linalg.eigh(cov)
    # eigh returns eigenvalues sorted ascending: lam1 <= lam2
    lam1, lam2 = float(evals[0]), float(evals[1])

    if lam2 <= 1e-12:
        # Degenerate: effectively a point source
        logger.debug("Degenerate moments (lam2=%.2e); returning fallback", lam2)
        return _FALLBACK

    # Guard against negative lam1 (can happen with noisy data)
    lam1_safe = max(lam1, 1e-12)
    q = math.sqrt(lam1_safe / lam2)
    q = float(np.clip(q, q_min, 1.0))

    # PA from the eigenvector of the major axis (largest eigenvalue = index 1)
    vx, vy = evecs[:, 1]
    phi = math.atan2(float(vy), float(vx))

    # Half-light radius proxy: radius enclosing half of positive flux
    r = rr[mask]
    idx = np.argsort(r)
    r_sorted = r[idx]
    w_sorted = w[idx]
    csum = np.cumsum(w_sorted)
    half = 0.5 * csum[-1]
    j = int(np.searchsorted(csum, half))
    r_half = float(r_sorted[min(j, len(r_sorted) - 1)])
    # Guard: r_half == 0 means all flux at center pixel
    if r_half < 0.5:
        r_half = 1.0

    return HostMoments(q=q, phi_rad=phi, r_half_pix=r_half)


# ---------------------------------------------------------------------------
# Host moments → lens parameter prior
# ---------------------------------------------------------------------------
def map_host_to_lens_params(
    theta_e_arcsec: float,
    host_mom: HostMoments,
    q_floor: float = 0.5,
    q_scatter: float = 0.05,
    gamma_max: float = 0.08,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Convert host light moments into a plausible lens mass model prior.

    The mapping is intentionally pragmatic, not a full mass-light inversion:

    - ``q_lens`` is aligned with the host light ``q``, floored at ``q_floor``
      (massive ellipticals rarely have q < 0.5 in mass), with small Gaussian
      scatter so that injections are not perfectly deterministic.
    - ``phi_lens`` is aligned with the host PA.
    - ``gamma_ext`` is drawn U[0, gamma_max] (typical external shear for
      field lenses is < 0.1; see e.g. Holder & Schechter 2003).
    - ``phi_ext`` is random uniform [0, pi).

    Parameters
    ----------
    theta_e_arcsec : float
        Einstein radius in arcseconds.
    host_mom : HostMoments
        Estimated host galaxy moments.
    q_floor : float
        Minimum allowed q_lens.  Default 0.5 follows the observed
        distribution of SLACS deflector axis ratios.
    q_scatter : float
        Standard deviation of Gaussian scatter added to q_lens to avoid
        perfect mass-light alignment.
    gamma_max : float
        Upper bound on external shear magnitude.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    dict
        Keys: ``theta_e_arcsec``, ``q_lens``, ``phi_lens_rad``,
        ``gamma_ext``, ``phi_ext_rad``.  Compatible with
        ``dhs.injection_engine.LensParams(**result)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    # q_lens: host-aligned with scatter and floor
    q_raw = host_mom.q + float(rng.normal(0.0, q_scatter))
    q_lens = float(np.clip(q_raw, q_floor, 1.0))

    # PA: aligned with host light
    phi_lens = host_mom.phi_rad

    # External shear: small, random orientation
    gamma = float(rng.uniform(0.0, gamma_max))
    phi_ext = float(rng.uniform(0.0, math.pi))

    # Convert (gamma, phi_ext) to reduced shear (g1, g2) for LensParams compatibility
    g1 = gamma * math.cos(2.0 * phi_ext)
    g2 = gamma * math.sin(2.0 * phi_ext)

    return dict(
        theta_e_arcsec=float(theta_e_arcsec),
        q_lens=q_lens,
        phi_lens_rad=phi_lens,
        shear_g1=g1,
        shear_g2=g2,
        gamma_ext=gamma,
        phi_ext_rad=phi_ext,
    )

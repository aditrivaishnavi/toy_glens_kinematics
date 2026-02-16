
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import math
import numpy as np

@dataclass
class HostMoments:
    q: float          # axis ratio <= 1
    phi_rad: float    # position angle
    r_half_pix: float # half-light radius proxy

def _robust_centered(img: np.ndarray, clip_pct: float = 99.5) -> np.ndarray:
    x = img.astype(np.float64, copy=False)
    med = np.nanmedian(x)
    y = x - med
    # clip extremes to reduce cosmic rays/artifacts
    lo = np.nanpercentile(y, 100-clip_pct)
    hi = np.nanpercentile(y, clip_pct)
    y = np.clip(y, lo, hi)
    y[np.isnan(y)] = 0.0
    return y

def estimate_host_moments_rband(host_hwc_nmgy: np.ndarray, r_band_index: int = 1,
                               center_radius_pix: int = 40) -> HostMoments:
    """
    Estimate axis ratio and PA from second moments in the r-band.
    This is a pragmatic substitute for having Tractor shape parameters.

    Steps:
    - subtract a robust background estimate
    - compute flux-weighted second moments inside a central aperture
    - compute eigenvalues/eigenvectors for (Mxx, Mxy; Mxy, Myy)
    """
    img = host_hwc_nmgy[..., r_band_index]
    img = _robust_centered(img)

    H,W = img.shape
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    mask = rr <= float(center_radius_pix)

    w = img[mask]
    w = np.maximum(w, 0.0)  # ignore negative pixels for shape moments
    if not np.isfinite(w).any() or w.sum() <= 0:
        return HostMoments(q=1.0, phi_rad=0.0, r_half_pix=10.0)

    x = (xx[mask] - cx)
    y = (yy[mask] - cy)
    norm = w.sum()
    mx = (w * x).sum() / norm
    my = (w * y).sum() / norm
    x = x - mx
    y = y - my

    mxx = (w * x * x).sum() / norm
    myy = (w * y * y).sum() / norm
    mxy = (w * x * y).sum() / norm

    # covariance matrix
    cov = np.array([[mxx, mxy],[mxy, myy]], dtype=np.float64)
    evals, evecs = np.linalg.eigh(cov)
    # evals sorted ascending; major axis has larger eigenvalue
    lam1, lam2 = float(evals[0]), float(evals[1])
    if lam2 <= 1e-12:
        q = 1.0
        phi = 0.0
    else:
        q = math.sqrt(max(lam1, 1e-12) / lam2)
        q = min(max(q, 0.2), 1.0)
        vx, vy = evecs[:,1]  # eigenvector for major axis
        phi = math.atan2(vy, vx)

    # rough half-light radius proxy: radius enclosing half of positive flux
    r = rr[mask]
    idx = np.argsort(r)
    r_sorted = r[idx]
    w_sorted = w[idx]
    csum = np.cumsum(w_sorted)
    half = 0.5 * csum[-1]
    j = int(np.searchsorted(csum, half))
    r_half = float(r_sorted[min(j, len(r_sorted)-1)])

    return HostMoments(q=q, phi_rad=phi, r_half_pix=r_half)

def map_host_to_lens_params(theta_e_arcsec: float, host_mom: HostMoments,
                            q_floor: float = 0.5, gamma_max: float = 0.08,
                            rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """
    Convert host light moments into a plausible lens mass model prior:
    - q_lens aligned with host light, but not allowed to be rounder than q_floor
    - phi_lens aligned with host PA
    - gamma_ext sampled small; phi_ext random

    This is a pragmatic "deflector-conditioned" prior intended to reduce the host mismatch
    failure mode observed in Model 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    q_lens = max(min(host_mom.q, 1.0), q_floor)
    phi_lens = host_mom.phi_rad

    gamma = float(rng.uniform(0.0, gamma_max))
    phi_ext = float(rng.uniform(0.0, math.pi))

    return dict(theta_e_arcsec=float(theta_e_arcsec), q_lens=float(q_lens), phi_lens_rad=float(phi_lens),
                gamma_ext=gamma, phi_ext_rad=phi_ext)

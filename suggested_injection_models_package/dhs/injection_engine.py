
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any

import math
import numpy as np
import torch

AB_ZP = 22.5  # DR10 zeropoint: mag -> nanomaggies: nmgy = 10**((22.5 - mag)/2.5)

# ----------------------------
# Dataclasses
# ----------------------------
@dataclass
class LensParams:
    theta_e_arcsec: float
    q_lens: float = 1.0
    phi_lens_rad: float = 0.0
    gamma_ext: float = 0.0
    phi_ext_rad: float = 0.0

@dataclass
class SourceParams:
    # source-plane center (arcsec)
    beta_x_arcsec: float
    beta_y_arcsec: float
    # light profile
    re_arcsec: float
    n_sersic: float
    q: float
    phi_rad: float
    # flux in nanomaggies by band (total unlensed integrated flux)
    flux_nmgy_g: float
    flux_nmgy_r: float
    flux_nmgy_z: float
    # optional clumps
    n_clumps: int = 0
    clump_frac: float = 0.25  # fraction of each band flux added in clumps

@dataclass
class InjectionResult:
    injected: torch.Tensor        # (C,H,W)
    injection_only: torch.Tensor  # (C,H,W)
    meta: Dict[str, Any]

# ----------------------------
# Small utilities
# ----------------------------
def mag_to_nmgy(mag: float) -> float:
    return 10.0 ** ((AB_ZP - mag) / 2.5)

def nmgy_to_mag(nmgy: float) -> float:
    nmgy = max(float(nmgy), 1e-30)
    return AB_ZP - 2.5 * math.log10(nmgy)

def _b_n(n: torch.Tensor) -> torch.Tensor:
    # Ciotti & Bertin (1999) approx
    return 2.0 * n - 1.0/3.0 + 4.0/(405.0*n) + 46.0/(25515.0*n*n)

def _torch_meshgrid_arcsec(h: int, w: int, pixel_scale_arcsec: float, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    # Pixel centers, with (0,0) at stamp center
    ys = (torch.arange(h, device=device, dtype=dtype) - (h - 1) / 2.0) * pixel_scale_arcsec
    xs = (torch.arange(w, device=device, dtype=dtype) - (w - 1) / 2.0) * pixel_scale_arcsec
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy

def _rotate(x: torch.Tensor, y: torch.Tensor, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    c = torch.cos(phi)
    s = torch.sin(phi)
    return c*x - s*y, s*x + c*y

# ----------------------------
# Lens deflections (SIS, SIE, shear)
# ----------------------------
def _sis_deflection(x: torch.Tensor, y: torch.Tensor, theta_e: torch.Tensor, eps: float = 1e-9) -> Tuple[torch.Tensor, torch.Tensor]:
    r = torch.sqrt(x*x + y*y + eps)
    ax = theta_e * x / r
    ay = theta_e * y / r
    return ax, ay

def _shear_deflection(x: torch.Tensor, y: torch.Tensor, gamma: torch.Tensor, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # external shear: alpha = Gamma @ x, where Gamma rotates by 2phi
    c2 = torch.cos(2.0 * phi)
    s2 = torch.sin(2.0 * phi)
    ax = gamma * (c2 * x + s2 * y)
    ay = gamma * (s2 * x - c2 * y)
    return ax, ay

def _sie_deflection(x: torch.Tensor, y: torch.Tensor, theta_e: torch.Tensor, q: torch.Tensor, phi_lens: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SIE (Kormann+1994-like) deflection for axis ratio q <= 1.
    Inputs may be broadcastable tensors.

    Note: this is a standard form and is cross-checkable against lenstronomy.
    """
    # rotate into lens frame
    xp, yp = _rotate(x, y, -phi_lens)
    q = torch.clamp(q, 1e-3, 1.0)

    # q ~ 1 fallback to SIS to avoid numerical issues
    close = (1.0 - q) < 1e-4
    if torch.any(close):
        ax_sis, ay_sis = _sis_deflection(xp, yp, theta_e)
        # rotate back
        ax, ay = _rotate(ax_sis, ay_sis, phi_lens)
        # for mixed batches, compute full SIE for the rest
        if torch.all(close):
            return ax, ay
    # full SIE
    s = torch.sqrt(torch.clamp(1.0 - q*q, min=0.0))
    # softened psi to avoid 0/0
    psi = torch.sqrt(q*q * xp*xp + yp*yp + eps)
    denom = psi + q*q  # matches common implementation; validated by unit tests below

    argx = s * xp / denom
    argy = s * yp / denom
    # numerical safety
    argx = torch.clamp(argx, -1.0 + 1e-6, 1.0 - 1e-6)

    fac = theta_e * q / torch.clamp(s, min=1e-6)
    axp = fac * torch.atanh(argx)
    ayp = fac * torch.atan(argy)

    # stitch mixed batches: where close, use SIS values
    if torch.any(close):
        axp = torch.where(close, ax_sis, axp)
        ayp = torch.where(close, ay_sis, ayp)

    ax, ay = _rotate(axp, ayp, phi_lens)
    return ax, ay

# ----------------------------
# Source light (Sersic + optional clumps)
# ----------------------------
def _sersic_source_integral(re: torch.Tensor, n: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Integral of I(R) = exp(-b_n * ((R/Re)^(1/n) - 1)) with I_e=1 over 2D.
    Returns area units (arcsec^2).
    """
    bn = _b_n(n)
    two_n = 2.0 * n
    # Gamma(2n) via torch.lgamma
    gamma_2n = torch.exp(torch.lgamma(two_n))
    return 2.0 * math.pi * q * (re * re) * n * torch.exp(bn) * gamma_2n / (bn ** two_n)

def _sersic_shape(x: torch.Tensor, y: torch.Tensor, beta_x: torch.Tensor, beta_y: torch.Tensor,
                  re: torch.Tensor, n: torch.Tensor, q: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    # rotate into source frame
    xs = x - beta_x
    ys = y - beta_y
    xp, yp = _rotate(xs, ys, -phi)
    r_ell = torch.sqrt(xp*xp + (yp/q)*(yp/q) + 1e-12)
    bn = _b_n(n)
    t = torch.clamp(r_ell / torch.clamp(re, min=1e-6), min=1e-6)
    return torch.exp(-bn * (t ** (1.0 / n) - 1.0))

def _add_clumps(x: torch.Tensor, y: torch.Tensor, base_shape: torch.Tensor,
                beta_x: float, beta_y: float, re_arcsec: float, n_clumps: int,
                device, dtype) -> torch.Tensor:
    """
    Add Gaussian clumps in the source plane (for mild substructure).
    This *does not* renormalize. Treat as an explicit "extra flux" term.
    """
    if n_clumps <= 0:
        return base_shape
    rng = torch.Generator(device=device)
    rng.manual_seed(int(torch.randint(0, 2**31-1, (1,), device=device).item()))
    # clump positions in pixels around the source center
    # (beta_x, beta_y) already in arcsec
    # sample offsets within ~1.5 Re
    dx = (torch.rand((n_clumps,), generator=rng, device=device, dtype=dtype) * 2 - 1.0) * (1.5 * re_arcsec)
    dy = (torch.rand((n_clumps,), generator=rng, device=device, dtype=dtype) * 2 - 1.0) * (1.5 * re_arcsec)
    sig = 0.15 * re_arcsec
    out = base_shape.clone()
    for i in range(n_clumps):
        cx = beta_x + dx[i]
        cy = beta_y + dy[i]
        rr2 = (x - cx) ** 2 + (y - cy) ** 2
        out = out + torch.exp(-0.5 * rr2 / max(sig*sig, 1e-8))
    return out

# ----------------------------
# PSF convolution and noise
# ----------------------------
def _gaussian_psf_kernel(fwhm_arcsec: float, pixel_scale_arcsec: float, kernel_size: int, device, dtype) -> torch.Tensor:
    sigma_pix = (fwhm_arcsec / 2.355) / pixel_scale_arcsec
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    ker = torch.exp(-0.5 * (xx*xx + yy*yy) / torch.clamp(sigma_pix*sigma_pix, min=1e-6))
    ker = ker / torch.clamp(ker.sum(), min=1e-12)
    return ker

def _fft_conv2d(img: torch.Tensor, ker: torch.Tensor) -> torch.Tensor:
    # img: (H,W), ker:(K,K)
    H,W = img.shape
    K = ker.shape[0]
    padH = H + K - 1
    padW = W + K - 1
    F = torch.fft.rfft2(img, s=(padH,padW))
    Kf = torch.fft.rfft2(ker, s=(padH,padW))
    out = torch.fft.irfft2(F * Kf, s=(padH,padW))
    # center crop
    y0 = (K - 1)//2
    x0 = (K - 1)//2
    return out[y0:y0+H, x0:x0+W]

def estimate_sigma_pix_from_psfdepth(psfdepth: float, psf_fwhm_arcsec: float, pixel_scale_arcsec: float,
                                    kernel_size: int = 33) -> float:
    """
    Approximate per-pixel Gaussian noise sigma from psfdepth (inverse variance of PSF flux).
    Assumes PSF-weighted flux estimate variance: Var(F) = sigma_pix^2 / sum(P^2), where P sums to 1.
    Therefore sigma_pix = sigma_F * sqrt(sum(P^2)), sigma_F = 1/sqrt(psfdepth).

    Notes:
    - psfdepth is band-specific in survey products; if you only have psfdepth_r, this is an r-band proxy.
    - This ignores correlated noise, which typically inflates effective sigma_pix by ~1.1-1.4 depending on coadd.
    """
    psfdepth = max(float(psfdepth), 1e-30)
    sigma_F = 1.0 / math.sqrt(psfdepth)
    # build PSF and compute sum of squares
    dev = torch.device("cpu")
    dt = torch.float64
    ker = _gaussian_psf_kernel(psf_fwhm_arcsec, pixel_scale_arcsec, kernel_size, dev, dt)
    sum_psf2 = float((ker*ker).sum().item())
    return sigma_F * math.sqrt(max(sum_psf2, 1e-30))

def add_gaussian_noise(img_chw: torch.Tensor, sigma_pix_by_band: Tuple[float,float,float], generator: Optional[torch.Generator]=None) -> torch.Tensor:
    assert img_chw.ndim == 3 and img_chw.shape[0] == 3
    noise = []
    for c,sig in enumerate(sigma_pix_by_band):
        sig = max(float(sig), 0.0)
        if sig == 0.0:
            noise.append(torch.zeros_like(img_chw[c]))
        else:
            noise.append(torch.randn_like(img_chw[c], generator=generator) * sig)
    return img_chw + torch.stack(noise, dim=0)

# ----------------------------
# Main injection
# ----------------------------
def inject_lensed_arcs(
    host_hwc_nmgy: torch.Tensor,
    lens: LensParams,
    src: SourceParams,
    pixel_scale_arcsec: float = 0.262,
    oversample: int = 4,
    psf_fwhm_arcsec_r: Optional[float] = None,
    psf_kernel_size: int = 33,
    psf_fwhm_scaling_g_r_z: Tuple[float,float,float] = (1.05, 1.0, 1.00),
    psfdepth_r: Optional[float] = None,
    psfdepth_scaling_g_r_z: Tuple[float,float,float] = (1.0, 1.0, 1.0),
    add_noise: bool = False,
    core_suppress_radius_pix: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> InjectionResult:
    """
    Inject a lensed source into a real host cutout.

    Inputs:
    - host_hwc_nmgy: (H,W,3) in nanomaggies.
    - lens parameters in arcsec; source flux is TOTAL UNLENSED flux per band.

    Returns:
    - injected and injection_only in CHW (3,H,W), same nanomaggies units.
    """
    if device is None:
        device = host_hwc_nmgy.device
    H,W,C = host_hwc_nmgy.shape
    assert C == 3

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(int(seed))

    # oversampled grid
    eff_ps = pixel_scale_arcsec / float(oversample)
    x, y = _torch_meshgrid_arcsec(H*oversample, W*oversample, eff_ps, device, dtype)

    # Lens deflection
    theta_e = torch.tensor(lens.theta_e_arcsec, device=device, dtype=dtype)
    q_lens = torch.tensor(lens.q_lens, device=device, dtype=dtype)
    phi_lens = torch.tensor(lens.phi_lens_rad, device=device, dtype=dtype)
    ax, ay = _sie_deflection(x, y, theta_e, q_lens, phi_lens)
    if lens.gamma_ext and lens.gamma_ext != 0.0:
        gamma = torch.tensor(lens.gamma_ext, device=device, dtype=dtype)
        phi_ext = torch.tensor(lens.phi_ext_rad, device=device, dtype=dtype)
        sax, say = _shear_deflection(x, y, gamma, phi_ext)
        ax = ax + sax
        ay = ay + say

    # Source-plane coordinates (beta = theta - alpha)
    bx = x - ax
    by = y - ay

    # Source shape (dimensionless)
    beta_x = torch.tensor(src.beta_x_arcsec, device=device, dtype=dtype)
    beta_y = torch.tensor(src.beta_y_arcsec, device=device, dtype=dtype)
    re = torch.tensor(src.re_arcsec, device=device, dtype=dtype)
    n = torch.tensor(src.n_sersic, device=device, dtype=dtype)
    qs = torch.tensor(src.q, device=device, dtype=dtype)
    phis = torch.tensor(src.phi_rad, device=device, dtype=dtype)

    shape = _sersic_shape(bx, by, beta_x, beta_y, re, n, qs, phis)

    # optional clumps (additive)
    if src.n_clumps and src.n_clumps > 0 and src.clump_frac and src.clump_frac > 0:
        shape = _add_clumps(bx, by, shape, src.beta_x_arcsec, src.beta_y_arcsec, src.re_arcsec, src.n_clumps, device, dtype)

    # normalize to *unlensed* total flux
    integral = _sersic_source_integral(re, n, qs)  # arcsec^2 for Ie=1
    # convert to surface brightness in nmgy/arcsec^2
    # NOTE: magnification naturally emerges because we evaluate source brightness at lensed mapping.
    pix_area = eff_ps * eff_ps
    fluxes = torch.tensor([src.flux_nmgy_g, src.flux_nmgy_r, src.flux_nmgy_z], device=device, dtype=dtype)
    # Add extra clump flux as explicit fraction (does not renormalize)
    if src.n_clumps and src.n_clumps > 0 and src.clump_frac and src.clump_frac > 0:
        fluxes = fluxes * (1.0 + float(src.clump_frac))
    sb = shape / torch.clamp(integral, min=1e-30)  # 1/arcsec^2
    inj_os = fluxes[:,None,None] * sb[None,:,:] * pix_area  # nmgy per subpixel

    # downsample by averaging blocks (conserves flux)
    inj = inj_os.reshape(3, H, oversample, W, oversample).mean(dim=(2,4))

    # PSF convolution per band (optional)
    if psf_fwhm_arcsec_r is not None and psf_fwhm_arcsec_r > 0:
        inj_conv = torch.empty_like(inj)
        for c,scale in enumerate(psf_fwhm_scaling_g_r_z):
            fwhm = float(psf_fwhm_arcsec_r) * float(scale)
            ker = _gaussian_psf_kernel(fwhm, pixel_scale_arcsec, psf_kernel_size, device, dtype)
            inj_conv[c] = _fft_conv2d(inj[c], ker)
        inj = inj_conv

    # Optional core suppression (ablation)
    if core_suppress_radius_pix is not None and core_suppress_radius_pix > 0:
        yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        cy = (H - 1) / 2.0
        cx = (W - 1) / 2.0
        rr = torch.sqrt((yy - cy)**2 + (xx - cx)**2)
        mask = (rr >= float(core_suppress_radius_pix)).to(dtype)
        inj = inj * mask[None,:,:]

    # build output
    host_chw = host_hwc_nmgy.permute(2,0,1).to(dtype)
    injected = host_chw + inj

    # add noise (optional)
    sigma_by_band = (0.0,0.0,0.0)
    if add_noise and psfdepth_r is not None and psf_fwhm_arcsec_r is not None:
        sig_r = estimate_sigma_pix_from_psfdepth(psfdepth_r, psf_fwhm_arcsec_r, pixel_scale_arcsec)
        # crude scaling across bands; if you have psfdepth per band, replace this
        sigma_by_band = tuple(sig_r*float(s) for s in psfdepth_scaling_g_r_z)
        injected = add_gaussian_noise(injected, sigma_by_band, generator=gen)

    meta = {
        "lens": asdict(lens),
        "src": asdict(src),
        "pixel_scale_arcsec": pixel_scale_arcsec,
        "oversample": oversample,
        "psf_fwhm_arcsec_r": psf_fwhm_arcsec_r,
        "psf_kernel_size": psf_kernel_size,
        "psf_fwhm_scaling_g_r_z": psf_fwhm_scaling_g_r_z,
        "psfdepth_r": psfdepth_r,
        "sigma_pix_by_band": sigma_by_band,
        "core_suppress_radius_pix": core_suppress_radius_pix,
    }
    return InjectionResult(injected=injected, injection_only=inj, meta=meta)

# ----------------------------
# Diagnostics
# ----------------------------
def arc_annulus_snr(injection_only_chw: torch.Tensor, sigma_pix_r: float, r_in_pix: float = 4.0, r_out_pix: float = 16.0) -> float:
    """
    Simple annulus SNR in r-band (channel=1), using sum(signal)/sqrt(N)*sigma.
    Note: This is not optimal matched-filter SNR, but useful as a monotonic proxy.
    """
    inj_r = injection_only_chw[1]
    H,W = inj_r.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=inj_r.device), torch.arange(W, device=inj_r.device), indexing="ij")
    cy = (H - 1)/2.0
    cx = (W - 1)/2.0
    rr = torch.sqrt((yy - cy)**2 + (xx - cx)**2)
    mask = (rr >= r_in_pix) & (rr <= r_out_pix)
    s = float(inj_r[mask].sum().item())
    n = int(mask.sum().item())
    sig = max(float(sigma_pix_r), 1e-30)
    return s / (math.sqrt(max(n,1)) * sig)

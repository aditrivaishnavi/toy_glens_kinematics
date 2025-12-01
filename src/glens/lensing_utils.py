# src/glens/lensing_utils.py
"""
Centralized lensing simulation utilities.

This module provides the SINGLE source of truth for:
- SIS lens deflection
- SIS subhalo deflection (on-arc placement)
- PSF convolution
- Noise injection
- Flux-based velocity masking

All scripts (debug_subhalo_samples.py, subhalo_cnn.py, sis_demo_lens.py)
should import from here rather than implementing their own versions.

This ensures:
1. Consistent physics across training and debugging
2. Single place to update parameters/algorithms
3. Easier testing and validation
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import os
import math

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


# =============================================================================
# Configuration dataclass
# =============================================================================

@dataclass
class LensSimConfig:
    """
    Configuration for lens simulation parameters.
    
    All default values are chosen to match the original implementations
    in debug_subhalo_samples.py and subhalo_cnn.py.
    """
    # Main lens parameters
    theta_E_main: float = 0.5
    
    # Subhalo parameters
    theta_E_sub_factor: float = 0.1  # theta_E_sub = factor * theta_E_main
    subhalo_r_offset_frac: Tuple[float, float] = (0.1, 0.3)  # radial offset range
    
    # PSF parameters
    psf_sigma: float = 1.0  # Gaussian PSF sigma in pixels
    
    # Noise parameters
    flux_noise_sigma: float = 0.08
    vel_noise_sigma: float = 0.03
    
    # Masking parameters
    flux_mask_threshold: float = 0.1  # velocity masked where flux < threshold


# =============================================================================
# I/O utilities
# =============================================================================

def load_source_tensor(plateifu: str, tensor_dir: str) -> np.ndarray:
    """
    Load a (2, H, W) source tensor for the given plateifu from tensor_dir.
    
    Parameters
    ----------
    plateifu : str
        Plate-IFU identifier, e.g., "8993-12705".
    tensor_dir : str
        Directory containing source_tensor_<plateifu>.npy files.
        
    Returns
    -------
    np.ndarray
        Shape (2, H, W), channel 0 = flux [0,1], channel 1 = velocity [-1,1].
    """
    fname = os.path.join(tensor_dir, f"source_tensor_{plateifu}.npy")
    if not os.path.exists(fname):
        # List available tensors for helpful error message
        available = []
        if os.path.isdir(tensor_dir):
            available = [
                f for f in os.listdir(tensor_dir)
                if f.startswith("source_tensor_") and f.endswith(".npy")
            ]
        raise FileNotFoundError(
            f"Source tensor not found: {fname}\n"
            f"Available tensors in {tensor_dir}:\n"
            + "\n".join(f"  - {a}" for a in available)
        )
    
    arr = np.load(fname)
    if arr.shape[0] != 2:
        raise ValueError(f"Expected 2 channels (flux, vel); got shape {arr.shape}")
    return arr


# =============================================================================
# Coordinate grid utilities
# =============================================================================

def build_theta_grid(H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a normalized coordinate grid over [-1, 1] x [-1, 1].
    
    Parameters
    ----------
    H, W : int
        Height and width of the grid.
        
    Returns
    -------
    theta_x, theta_y : np.ndarray
        Arrays of shape (H, W) giving image-plane coordinates.
    """
    x = np.linspace(-1.0, 1.0, W)
    y = np.linspace(-1.0, 1.0, H)
    theta_x, theta_y = np.meshgrid(x, y)
    return theta_x, theta_y


# =============================================================================
# SIS deflection
# =============================================================================

def sis_deflection(
    theta_x: np.ndarray,
    theta_y: np.ndarray,
    theta_E: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SIS deflection field at each (theta_x, theta_y).
    
    The SIS deflection is:
        alpha = theta_E * (theta - center) / |theta - center|
    
    Parameters
    ----------
    theta_x, theta_y : np.ndarray
        Image-plane coordinates (2D arrays).
    theta_E : float
        Einstein radius in normalized units.
    center_x, center_y : float
        Center of the SIS (default: origin).
        
    Returns
    -------
    alpha_x, alpha_y : np.ndarray
        Deflection components in x and y.
    """
    eps = 1e-6  # softening to avoid division by zero
    dx = theta_x - center_x
    dy = theta_y - center_y
    r = np.sqrt(dx**2 + dy**2) + eps
    
    alpha_x = theta_E * dx / r
    alpha_y = theta_E * dy / r
    
    return alpha_x, alpha_y


# =============================================================================
# Bilinear sampling utility
# =============================================================================

def _bilinear_sample(
    source_map: np.ndarray,
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    theta_1d_x: np.ndarray,
    theta_1d_y: np.ndarray,
) -> np.ndarray:
    """
    Bilinear sampling of source_map at positions (beta_x, beta_y).
    
    Parameters
    ----------
    source_map : np.ndarray
        2D array (H, W) to sample from.
    beta_x, beta_y : np.ndarray
        2D arrays with source-plane coordinates in [-1, 1].
    theta_1d_x, theta_1d_y : np.ndarray
        1D coordinate grids for x and y axes.
        
    Returns
    -------
    np.ndarray
        Sampled 2D array of shape (H, W).
    """
    interp = RegularGridInterpolator(
        (theta_1d_y, theta_1d_x),
        source_map,
        bounds_error=False,
        fill_value=0.0,
    )
    pts = np.stack([beta_y.ravel(), beta_x.ravel()], axis=-1)
    sampled = interp(pts).reshape(source_map.shape)
    return sampled


# =============================================================================
# Rendering functions
# =============================================================================

def render_sis_lens(
    source_tensor: np.ndarray,
    config: LensSimConfig,
) -> np.ndarray:
    """
    Render a smooth SIS lens (no subhalo, no PSF/noise).
    
    Parameters
    ----------
    source_tensor : np.ndarray
        Shape (2, H, W), channels = [flux, velocity].
    config : LensSimConfig
        Lens simulation configuration.
        
    Returns
    -------
    lensed_tensor : np.ndarray
        Shape (2, H, W), lensed flux + velocity with flux-based masking.
    """
    flux_src = source_tensor[0]
    vel_src = source_tensor[1]
    H, W = flux_src.shape
    
    # Build coordinate grid
    theta_x, theta_y = build_theta_grid(H, W)
    theta_1d_x = np.linspace(-1.0, 1.0, W)
    theta_1d_y = np.linspace(-1.0, 1.0, H)
    
    # Main SIS deflection
    alpha_x, alpha_y = sis_deflection(theta_x, theta_y, config.theta_E_main)
    
    # Lens equation: beta = theta - alpha
    beta_x = theta_x - alpha_x
    beta_y = theta_y - alpha_y
    
    # Sample source at beta positions
    flux_lensed = _bilinear_sample(flux_src, beta_x, beta_y, theta_1d_x, theta_1d_y)
    vel_lensed = _bilinear_sample(vel_src, beta_x, beta_y, theta_1d_x, theta_1d_y)
    
    # Flux-based velocity masking
    mask = flux_lensed >= config.flux_mask_threshold
    vel_lensed = np.where(mask, vel_lensed, 0.0)
    
    return np.stack([flux_lensed, vel_lensed], axis=0)


def render_sis_plus_subhalo(
    source_tensor: np.ndarray,
    config: LensSimConfig,
    on_arc: bool = True,
    subhalo_phi: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Render SIS + SIS-subhalo lens (no PSF/noise).
    
    The subhalo is placed on or near the Einstein ring to guarantee
    a visible perturbation on the lensed arc.
    
    Parameters
    ----------
    source_tensor : np.ndarray
        Shape (2, H, W), channels = [flux, velocity].
    config : LensSimConfig
        Lens simulation configuration.
    on_arc : bool
        If True, place subhalo near the Einstein ring. If False, place at origin.
    subhalo_phi : float, optional
        Fixed azimuthal angle for subhalo (radians). If None, random.
    rng : np.random.Generator, optional
        Random generator for reproducibility.
        
    Returns
    -------
    lensed_tensor : np.ndarray
        Shape (2, H, W), lensed flux + velocity with flux-based masking.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    flux_src = source_tensor[0]
    vel_src = source_tensor[1]
    H, W = flux_src.shape
    
    # Build coordinate grid
    theta_x, theta_y = build_theta_grid(H, W)
    theta_1d_x = np.linspace(-1.0, 1.0, W)
    theta_1d_y = np.linspace(-1.0, 1.0, H)
    
    # Main SIS deflection
    alpha_x_main, alpha_y_main = sis_deflection(theta_x, theta_y, config.theta_E_main)
    
    # Subhalo placement
    theta_E_sub = config.theta_E_main * config.theta_E_sub_factor
    
    if on_arc:
        # Place subhalo on/near the Einstein ring
        if subhalo_phi is None:
            phi = rng.uniform(0.0, 2.0 * math.pi)
        else:
            phi = subhalo_phi
        
        offset_frac = rng.uniform(
            config.subhalo_r_offset_frac[0],
            config.subhalo_r_offset_frac[1]
        )
        r_center = config.theta_E_main * (1.0 + offset_frac)
        x_sub = r_center * math.cos(phi)
        y_sub = r_center * math.sin(phi)
    else:
        # Place at origin (for testing)
        x_sub = 0.0
        y_sub = 0.0
    
    # Subhalo SIS deflection
    alpha_x_sub, alpha_y_sub = sis_deflection(
        theta_x, theta_y, theta_E_sub, center_x=x_sub, center_y=y_sub
    )
    
    # Total deflection
    alpha_x_total = alpha_x_main + alpha_x_sub
    alpha_y_total = alpha_y_main + alpha_y_sub
    
    # Lens equation
    beta_x = theta_x - alpha_x_total
    beta_y = theta_y - alpha_y_total
    
    # Sample source
    flux_lensed = _bilinear_sample(flux_src, beta_x, beta_y, theta_1d_x, theta_1d_y)
    vel_lensed = _bilinear_sample(vel_src, beta_x, beta_y, theta_1d_x, theta_1d_y)
    
    # Flux-based velocity masking
    mask = flux_lensed >= config.flux_mask_threshold
    vel_lensed = np.where(mask, vel_lensed, 0.0)
    
    return np.stack([flux_lensed, vel_lensed], axis=0)


# =============================================================================
# PSF and noise
# =============================================================================

def apply_psf_and_noise(
    tensor: np.ndarray,
    config: LensSimConfig,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply Gaussian PSF and Gaussian noise to a (2, H, W) tensor.
    
    After PSF and noise, re-applies flux-based velocity masking.
    
    Parameters
    ----------
    tensor : np.ndarray
        Shape (2, H, W), channels = [flux, velocity].
    config : LensSimConfig
        PSF and noise parameters.
    rng : np.random.Generator, optional
        Random generator for reproducibility.
        
    Returns
    -------
    degraded : np.ndarray
        Shape (2, H, W), with PSF blur and noise applied.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    flux = tensor[0].copy()
    vel = tensor[1].copy()
    
    # PSF blur (same sigma for both channels)
    if config.psf_sigma is not None and config.psf_sigma > 0.0:
        flux = gaussian_filter(flux, sigma=config.psf_sigma)
        vel = gaussian_filter(vel, sigma=config.psf_sigma)
    
    # Add noise
    flux = flux + rng.normal(loc=0.0, scale=config.flux_noise_sigma, size=flux.shape)
    vel = vel + rng.normal(loc=0.0, scale=config.vel_noise_sigma, size=vel.shape)
    
    # Clip to reasonable ranges
    flux = np.clip(flux, 0.0, 1.5)  # allow slight overshoot
    vel = np.clip(vel, -1.5, 1.5)
    
    return np.stack([flux, vel], axis=0)


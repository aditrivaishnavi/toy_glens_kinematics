"""
Observability maps for strong lensing in DR10.

This module classifies the (θ_E, z_l) parameter space into regions
where lensing detection is:
- Blind: θ_E too small relative to seeing
- Low trust: Marginally resolvable
- Good: Well-resolved lensing features
"""

import numpy as np
from typing import Literal

from .lens_equations import theta_E_SIS


# Representative DR10 FWHM values (arcseconds)
# These are typical values; actual PSF varies by band and observation
DR10_FWHM = {
    'g': 1.3,  # Typically worst seeing
    'r': 1.2,
    'z': 1.1   # Typically best seeing
}

# Default thresholds for observability classification
# These are in units of FWHM
BLIND_THRESHOLD = 0.5      # θ_E < 0.5 × FWHM → unresolvable
LOW_TRUST_THRESHOLD = 1.0  # 0.5 × FWHM ≤ θ_E < 1.0 × FWHM → marginal


ObservabilityClass = Literal["blind", "low_trust", "good"]


def confusion_regions(
    theta_E: float,
    FWHM: float,
    blind_threshold: float = BLIND_THRESHOLD,
    low_trust_threshold: float = LOW_TRUST_THRESHOLD
) -> ObservabilityClass:
    """
    Classify observability regime based on θ_E and seeing.
    
    Parameters
    ----------
    theta_E : float
        Einstein radius in arcseconds
    FWHM : float
        Point spread function FWHM in arcseconds
    blind_threshold : float
        Threshold (in FWHM units) below which lensing is undetectable
    low_trust_threshold : float
        Threshold (in FWHM units) below which detection is marginal
    
    Returns
    -------
    str
        One of "blind", "low_trust", "good"
    
    Examples
    --------
    >>> confusion_regions(0.5, 1.2)  # θ_E = 0.5", FWHM = 1.2"
    'blind'
    >>> confusion_regions(1.0, 1.2)
    'low_trust'
    >>> confusion_regions(2.0, 1.2)
    'good'
    """
    ratio = theta_E / FWHM
    
    if ratio < blind_threshold:
        return "blind"
    elif ratio < low_trust_threshold:
        return "low_trust"
    else:
        return "good"


def classify_grid(
    theta_E_grid: np.ndarray,
    FWHM: float
) -> np.ndarray:
    """
    Classify a grid of θ_E values into observability regions.
    
    Parameters
    ----------
    theta_E_grid : np.ndarray
        Array of Einstein radii (arcseconds)
    FWHM : float
        Seeing FWHM (arcseconds)
    
    Returns
    -------
    np.ndarray
        Integer array: 0=blind, 1=low_trust, 2=good
    """
    ratio = theta_E_grid / FWHM
    
    result = np.full_like(theta_E_grid, 2, dtype=int)  # Default: good
    result[ratio < LOW_TRUST_THRESHOLD] = 1  # low_trust
    result[ratio < BLIND_THRESHOLD] = 0       # blind
    
    return result


def evaluate_observability_grid(
    sigma_v_range: np.ndarray,
    z_l_range: np.ndarray,
    z_s_range: np.ndarray,
    cosmo,
    band: str = 'r'
) -> dict:
    """
    Compute θ_E and observability class over a parameter grid.
    
    Parameters
    ----------
    sigma_v_range : np.ndarray
        Array of velocity dispersions (km/s)
    z_l_range : np.ndarray
        Array of lens redshifts
    z_s_range : np.ndarray
        Array of source redshifts
    cosmo : astropy.cosmology
        Cosmology object
    band : str
        Observation band for FWHM lookup ('g', 'r', or 'z')
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'theta_E': 3D array of Einstein radii (n_sigma, n_zl, n_zs)
        - 'classification': 3D array of classes (0=blind, 1=low_trust, 2=good)
        - 'sigma_v': input sigma_v array
        - 'z_l': input z_l array
        - 'z_s': input z_s array
        - 'FWHM': the seeing FWHM used
    """
    FWHM = DR10_FWHM[band]
    
    n_sigma = len(sigma_v_range)
    n_zl = len(z_l_range)
    n_zs = len(z_s_range)
    
    theta_E = np.zeros((n_sigma, n_zl, n_zs))
    
    for i, sigma_v in enumerate(sigma_v_range):
        for j, z_l in enumerate(z_l_range):
            for k, z_s in enumerate(z_s_range):
                if z_s > z_l:
                    theta_E[i, j, k] = theta_E_SIS(sigma_v, z_l, z_s, cosmo)
                else:
                    theta_E[i, j, k] = np.nan
    
    classification = classify_grid(theta_E, FWHM)
    
    return {
        'theta_E': theta_E,
        'classification': classification,
        'sigma_v': sigma_v_range,
        'z_l': z_l_range,
        'z_s': z_s_range,
        'FWHM': FWHM
    }


def get_theta_E_contours(
    z_l_range: np.ndarray,
    z_s: float,
    cosmo,
    theta_E_levels: list[float] = [0.5, 1.0, 1.5, 2.0]
) -> dict:
    """
    Compute σ_v values corresponding to specific θ_E levels.
    
    This is useful for plotting θ_E contours in the σ_v-z_l plane.
    
    Parameters
    ----------
    z_l_range : np.ndarray
        Array of lens redshifts
    z_s : float
        Fixed source redshift
    cosmo : astropy.cosmology
        Cosmology object
    theta_E_levels : list of float
        Target Einstein radii (arcseconds)
    
    Returns
    -------
    dict
        Dictionary mapping θ_E level to array of σ_v values
    """
    # We need to invert θ_E(σ_v) for each z_l
    # For SIS: θ_E ∝ σ_v², so σ_v ∝ θ_E^(1/2) at fixed geometry
    
    result = {}
    
    for theta_target in theta_E_levels:
        sigma_v_contour = np.zeros_like(z_l_range)
        
        for i, z_l in enumerate(z_l_range):
            if z_l >= z_s:
                sigma_v_contour[i] = np.nan
                continue
            
            # Reference point: σ_v = 200 km/s
            theta_ref = theta_E_SIS(200.0, z_l, z_s, cosmo)
            
            # Scale: σ_v ∝ θ_E^(1/2)
            sigma_v_contour[i] = 200.0 * np.sqrt(theta_target / theta_ref)
        
        result[theta_target] = sigma_v_contour
    
    return result


def blind_boundary_sigma_v(
    z_l_range: np.ndarray,
    z_s: float,
    cosmo,
    band: str = 'r'
) -> np.ndarray:
    """
    Compute the σ_v values at the blind/low-trust boundary.
    
    Returns σ_v such that θ_E = 0.5 × FWHM (the blind threshold).
    
    Parameters
    ----------
    z_l_range : np.ndarray
        Array of lens redshifts
    z_s : float
        Fixed source redshift
    cosmo : astropy.cosmology
        Cosmology object
    band : str
        Observation band
    
    Returns
    -------
    np.ndarray
        Array of σ_v values at the blind boundary
    """
    theta_blind = BLIND_THRESHOLD * DR10_FWHM[band]
    contours = get_theta_E_contours(z_l_range, z_s, cosmo, [theta_blind])
    return contours[theta_blind]


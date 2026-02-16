"""
Selection function construction.

Combine analytic observability maps with empirical completeness
to produce the final selection function C(θ_E, z_l).
"""

import numpy as np
from typing import Dict, Tuple, Optional


def combine_analytic_and_empirical(
    analytic_class: np.ndarray,
    empirical_C: np.ndarray,
    blind_value: float = 0.0,
    low_trust_penalty: float = 0.5
) -> np.ndarray:
    """
    Combine analytic classification with empirical completeness.
    
    Parameters
    ----------
    analytic_class : np.ndarray
        Classification from observability_maps (0=blind, 1=low_trust, 2=good)
    empirical_C : np.ndarray
        Empirical completeness from injection recovery
    blind_value : float
        Completeness to assign to blind regions
    low_trust_penalty : float
        Multiplicative penalty for low-trust regions
    
    Returns
    -------
    np.ndarray
        Combined selection function
    """
    result = empirical_C.copy()
    
    # Set blind regions to zero (or very low)
    blind_mask = analytic_class == 0
    result[blind_mask] = blind_value
    
    # Penalize low-trust regions
    low_trust_mask = analytic_class == 1
    result[low_trust_mask] *= low_trust_penalty
    
    return result


def build_selection_function_grid(
    theta_E_bins: np.ndarray,
    z_l_bins: np.ndarray,
    empirical_C_map: np.ndarray,
    z_s_assumed: float,
    cosmo,
    band: str = 'r'
) -> Dict:
    """
    Build selection function on a (θ_E, z_l) grid.
    
    Parameters
    ----------
    theta_E_bins : np.ndarray
        θ_E bin edges
    z_l_bins : np.ndarray
        z_l bin edges
    empirical_C_map : np.ndarray
        Shape (n_theta_E, n_z_l) completeness from simulations
    z_s_assumed : float
        Assumed source redshift for mapping to σ_v
    cosmo : astropy.cosmology
        Cosmology for distance calculations
    band : str
        Band for FWHM lookup
    
    Returns
    -------
    dict
        Selection function with all components
    """
    from ..physics.observability_maps import classify_grid, DR10_FWHM
    
    n_theta = len(theta_E_bins) - 1
    n_z = len(z_l_bins) - 1
    
    theta_E_centers = 0.5 * (theta_E_bins[:-1] + theta_E_bins[1:])
    z_l_centers = 0.5 * (z_l_bins[:-1] + z_l_bins[1:])
    
    # Build θ_E grid for classification
    theta_E_grid = np.tile(theta_E_centers[:, None], (1, n_z))
    
    # Classify each cell
    FWHM = DR10_FWHM[band]
    analytic_class = classify_grid(theta_E_grid, FWHM)
    
    # Combine
    selection = combine_analytic_and_empirical(analytic_class, empirical_C_map)
    
    return {
        'selection': selection,
        'analytic_class': analytic_class,
        'empirical_C': empirical_C_map,
        'theta_E_bins': theta_E_bins,
        'z_l_bins': z_l_bins,
        'theta_E_centers': theta_E_centers,
        'z_l_centers': z_l_centers,
        'z_s_assumed': z_s_assumed,
        'FWHM': FWHM
    }


def selection_function_to_Mhalo(
    selection_map: np.ndarray,
    theta_E_centers: np.ndarray,
    z_l_centers: np.ndarray,
    z_s_assumed: float,
    cosmo
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map selection function to halo mass space.
    
    ⚠️ Only valid for the assumed z_s — label clearly!
    
    Parameters
    ----------
    selection_map : np.ndarray
        C(θ_E, z_l)
    theta_E_centers : np.ndarray
        θ_E bin centers
    z_l_centers : np.ndarray
        z_l bin centers
    z_s_assumed : float
        Assumed source redshift
    cosmo : astropy.cosmology
        Cosmology
    
    Returns
    -------
    M_halo_grid : np.ndarray
        Shape (n_theta, n_z) halo masses corresponding to θ_E
    sigma_v_grid : np.ndarray
        Shape (n_theta, n_z) velocity dispersions
    """
    from ..physics.lens_equations import theta_E_SIS
    
    n_theta = len(theta_E_centers)
    n_z = len(z_l_centers)
    
    sigma_v_grid = np.zeros((n_theta, n_z))
    
    for j, z_l in enumerate(z_l_centers):
        if z_l >= z_s_assumed:
            sigma_v_grid[:, j] = np.nan
            continue
        
        # Reference: compute θ_E at σ_v = 200 km/s
        theta_ref = theta_E_SIS(200.0, z_l, z_s_assumed, cosmo)
        
        # Invert: σ_v ∝ θ_E^(1/2) at fixed geometry
        for i, theta_E in enumerate(theta_E_centers):
            sigma_v_grid[i, j] = 200.0 * np.sqrt(theta_E / theta_ref)
    
    # Convert σ_v to M_halo (using simple scaling)
    # M ∝ σ_v^3 roughly
    M_halo_grid = 1e13 * (sigma_v_grid / 250.0) ** 3
    
    return M_halo_grid, sigma_v_grid


def interpolate_selection(
    selection_dict: Dict,
    theta_E: float,
    z_l: float
) -> float:
    """
    Interpolate selection function at a specific (θ_E, z_l).
    
    Parameters
    ----------
    selection_dict : dict
        From build_selection_function_grid
    theta_E, z_l : float
        Query point
    
    Returns
    -------
    float
        Interpolated completeness
    """
    from scipy.interpolate import RegularGridInterpolator
    
    interp = RegularGridInterpolator(
        (selection_dict['theta_E_centers'], selection_dict['z_l_centers']),
        selection_dict['selection'],
        bounds_error=False,
        fill_value=0.0
    )
    
    return float(interp([[theta_E, z_l]])[0])


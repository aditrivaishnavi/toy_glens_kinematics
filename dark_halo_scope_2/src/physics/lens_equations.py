"""
Lensing equations for Einstein radius calculations.

This module provides functions to compute Einstein radii for various
lens models, primarily the Singular Isothermal Sphere (SIS).
"""

import numpy as np
from astropy import units as u
from astropy import constants as const


def theta_E_SIS(
    sigma_v: float,
    z_l: float,
    z_s: float,
    cosmo
) -> float:
    """
    Compute Einstein radius for a Singular Isothermal Sphere.
    
    The SIS Einstein radius is given by:
        θ_E = 4π (σ_v / c)² (D_ls / D_s)
    
    where:
        σ_v = velocity dispersion
        D_ls = angular diameter distance from lens to source
        D_s = angular diameter distance to source
    
    Parameters
    ----------
    sigma_v : float
        Velocity dispersion in km/s
    z_l : float
        Lens redshift
    z_s : float
        Source redshift (must be > z_l)
    cosmo : astropy.cosmology
        Cosmology object (e.g., Planck18)
    
    Returns
    -------
    theta_E : float
        Einstein radius in arcseconds
    
    Raises
    ------
    ValueError
        If z_s <= z_l
    
    Examples
    --------
    >>> from astropy.cosmology import Planck18
    >>> theta_E_SIS(250.0, 0.3, 1.5, Planck18)
    1.23...  # arcseconds
    """
    if z_s <= z_l:
        raise ValueError(f"Source redshift ({z_s}) must be greater than lens redshift ({z_l})")
    
    # Angular diameter distances
    D_l = cosmo.angular_diameter_distance(z_l)
    D_s = cosmo.angular_diameter_distance(z_s)
    D_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s)
    
    # Convert velocity dispersion to proper units
    sigma_v_cgs = (sigma_v * u.km / u.s).to(u.cm / u.s).value
    c_cgs = const.c.to(u.cm / u.s).value
    
    # Einstein radius in radians
    theta_E_rad = 4 * np.pi * (sigma_v_cgs / c_cgs)**2 * (D_ls / D_s).decompose().value
    
    # Convert to arcseconds
    theta_E_arcsec = np.degrees(theta_E_rad) * 3600
    
    return theta_E_arcsec


def sigma_v_from_Mhalo(
    M_halo: float,
    z_l: float,
    scaling: str = "dutton14"
) -> float:
    """
    Estimate velocity dispersion from halo mass using scaling relations.
    
    Parameters
    ----------
    M_halo : float
        Halo mass in solar masses (M_sun)
    z_l : float
        Lens redshift
    scaling : str
        Scaling relation to use. Options:
        - "dutton14": Dutton & Macciò 2014
        - "simple": Simple power-law approximation
    
    Returns
    -------
    sigma_v : float
        Velocity dispersion in km/s
    
    Notes
    -----
    This is an approximate relation. The actual σ_v depends on
    galaxy type, formation history, and other factors.
    """
    if scaling == "simple":
        # Simple power-law: σ_v ∝ M^(1/3)
        # Normalized such that M = 10^13 M_sun → σ_v ~ 250 km/s
        sigma_v = 250.0 * (M_halo / 1e13) ** (1/3)
    
    elif scaling == "dutton14":
        # Dutton & Macciò 2014 relation (approximate)
        # σ_v = σ_200 with concentration-mass relation
        # Simplified version:
        log_M = np.log10(M_halo)
        sigma_v = 10 ** (0.29 * log_M - 1.5)  # Very approximate
        
        # Redshift evolution (mild)
        sigma_v *= (1 + z_l) ** 0.1
    
    else:
        raise ValueError(f"Unknown scaling relation: {scaling}")
    
    return sigma_v


def theta_E_from_Mhalo(
    M_halo: float,
    z_l: float,
    z_s: float,
    cosmo,
    scaling: str = "simple"
) -> float:
    """
    Compute Einstein radius directly from halo mass.
    
    This is a convenience function that combines sigma_v_from_Mhalo
    and theta_E_SIS.
    
    Parameters
    ----------
    M_halo : float
        Halo mass in solar masses
    z_l : float
        Lens redshift
    z_s : float
        Source redshift
    cosmo : astropy.cosmology
        Cosmology object
    scaling : str
        Scaling relation for σ_v (see sigma_v_from_Mhalo)
    
    Returns
    -------
    theta_E : float
        Einstein radius in arcseconds
    """
    sigma_v = sigma_v_from_Mhalo(M_halo, z_l, scaling=scaling)
    return theta_E_SIS(sigma_v, z_l, z_s, cosmo)


def critical_surface_density(z_l: float, z_s: float, cosmo) -> float:
    """
    Compute the critical surface density for lensing.
    
    Σ_cr = c² D_s / (4π G D_l D_ls)
    
    Parameters
    ----------
    z_l : float
        Lens redshift
    z_s : float
        Source redshift
    cosmo : astropy.cosmology
        Cosmology object
    
    Returns
    -------
    Sigma_cr : float
        Critical surface density in M_sun / kpc²
    """
    D_l = cosmo.angular_diameter_distance(z_l)
    D_s = cosmo.angular_diameter_distance(z_s)
    D_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s)
    
    # Σ_cr in M_sun / Mpc²
    c = const.c
    G = const.G
    
    Sigma_cr = (c**2 / (4 * np.pi * G)) * (D_s / (D_l * D_ls))
    
    # Convert to M_sun / kpc²
    Sigma_cr = Sigma_cr.to(u.M_sun / u.kpc**2).value
    
    return Sigma_cr


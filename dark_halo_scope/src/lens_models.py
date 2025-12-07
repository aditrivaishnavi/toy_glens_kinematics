"""
Simple SIS lens model utilities.

We use the Singular Isothermal Sphere (SIS) as the baseline model for
Phase 1. This is a standard approximation for galaxy-scale lenses and
provides analytic relations between velocity dispersion, Einstein
radius, and enclosed mass.

References
----------
- Schneider, Kochanek, Wambsganss (Gravitational Lensing: Strong, Weak & Micro)
- Many strong lensing discovery papers use SIS / SIE as their first model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from astropy import units as u
from astropy.constants import c, G

from .cosmology import get_cosmology
from .config import Phase1Config


def sis_theta_E(
    sigma_v: float,
    z_l: float,
    z_s: float,
    config: Optional[Phase1Config] = None,
) -> float:
    """
    Compute the SIS Einstein radius in arcseconds.

    Parameters
    ----------
    sigma_v : float
        Line-of-sight velocity dispersion of the lens in km/s.
    z_l : float
        Lens redshift.
    z_s : float
        Source redshift; must satisfy z_s > z_l.
    config : Phase1Config, optional
        Configuration instance for cosmology. If None, uses Phase1Config().

    Returns
    -------
    float
        Einstein radius in arcseconds.

    Notes
    -----
    For an SIS, the Einstein radius is

        θ_E = 4π (σ_v^2 / c^2) D_ls / D_s

    where D_ls and D_s are angular diameter distances.
    """
    if config is None:
        config = Phase1Config()
    cosmo = get_cosmology(config)

    if not (0.0 <= z_l < z_s):
        raise ValueError("Require 0 <= z_l < z_s for a physical lens configuration.")

    sigma = (sigma_v * u.km / u.s).to(u.m / u.s)
    d_s = cosmo.angular_diameter_distance(z_s)
    d_l = cosmo.angular_diameter_distance(z_l)
    d_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s)

    theta_E_rad = 4.0 * np.pi * (sigma**2 / c**2) * (d_ls / d_s)
    theta_E_arcsec = theta_E_rad.to(u.arcsec).value
    return float(theta_E_arcsec)


def sis_mass_inside_theta_E(
    theta_E_arcsec: float,
    z_l: float,
    z_s: float,
    config: Optional[Phase1Config] = None,
) -> float:
    """
    Compute the projected mass enclosed within the Einstein radius
    for an SIS lens.

    Parameters
    ----------
    theta_E_arcsec : float
        Einstein radius in arcseconds.
    z_l : float
        Lens redshift.
    z_s : float
        Source redshift.
    config : Phase1Config, optional
        Configuration instance for cosmology.

    Returns
    -------
    float
        Enclosed mass M(<θ_E) in solar masses.

    Notes
    -----
    For an axially symmetric lens, the mass inside the Einstein radius is

        M(<θ_E) = (c^2 / 4G) * (D_l D_s / D_ls) * θ_E^2

    where θ_E is in radians and D are angular diameter distances.
    """
    if config is None:
        config = Phase1Config()
    cosmo = get_cosmology(config)

    if not (0.0 <= z_l < z_s):
        raise ValueError("Require 0 <= z_l < z_s for a physical lens configuration.")

    theta_E = (theta_E_arcsec * u.arcsec).to(u.rad).value  # dimensionless
    d_s = cosmo.angular_diameter_distance(z_s).to(u.m).value
    d_l = cosmo.angular_diameter_distance(z_l).to(u.m).value
    d_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).to(u.m).value

    # c²/(4G) has units of kg/m in SI
    prefactor = (c.value**2 / (4.0 * G.value))  # kg/m
    
    # Mass = (c²/4G) × (D_l × D_s / D_ls) × θ_E²
    # Units: (kg/m) × m × (dimensionless) = kg
    mass_kg = prefactor * (d_l * d_s / d_ls) * theta_E**2
    
    # Convert to solar masses
    M_sun_kg = u.Msun.to(u.kg)
    mass_solar = mass_kg / M_sun_kg
    return float(mass_solar)


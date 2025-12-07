"""
Analytic observability maps for DR10 strong lenses.

This module builds gridded "theoretical" observability windows based
on simple geometric criteria and an SIS lens model.

Important:
- These maps are *not* the final completeness. They ignore detailed
  surface brightness and noise properties. They simply encode the
  best-case geometric limit imposed by seeing and survey depth.

Later, injection–recovery tests will measure the *empirical* detection
probability and can be compared directly to these analytic maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import Phase1Config
from .survey_model import SurveyModel, DetectabilityClass
from .lens_models import sis_mass_inside_theta_E


@dataclass
class AnalyticWindowResult:
    """
    Container for analytic observability results on a grid.

    Attributes
    ----------
    z_l_grid : ndarray, shape (N_z,)
        Lens redshift grid.
    theta_E_grid : ndarray, shape (N_theta,)
        Einstein-radius grid in arcsec.
    detectability : ndarray, shape (N_z, N_theta)
        Integer codes for detectability class:
        0 = blind, 1 = low_trust, 2 = good.
    """

    z_l_grid: np.ndarray
    theta_E_grid: np.ndarray
    detectability: np.ndarray


@dataclass
class MassWindowResult:
    """
    Container for mass-based observability maps.

    Attributes
    ----------
    z_l_grid : ndarray, shape (N_z,)
    mass_grid : ndarray, shape (N_mass,)
    detectability : ndarray, shape (N_z, N_mass)
        Detectability class codes on the (z_l, M_halo) grid.
    """

    z_l_grid: np.ndarray
    mass_grid: np.ndarray
    detectability: np.ndarray


_DETECTABILITY_CODE = {
    "blind": 0,
    "low_trust": 1,
    "good": 2,
}


def build_thetaE_analytic_window(
    config: Optional[Phase1Config] = None,
) -> AnalyticWindowResult:
    """
    Build an analytic detectability map in (z_l, θ_E).

    Parameters
    ----------
    config : Phase1Config, optional
        Configuration with grid bounds and survey thresholds.

    Returns
    -------
    AnalyticWindowResult
        Gridded lens redshift, Einstein radius, and detectability codes.

    Notes
    -----
    - This is purely geometric and uses only the seeing wall.
    - Brightness and surface-brightness limits are *not* modeled here,
      because they depend more strongly on the detailed noise and
      galaxy population and will be constrained by injection–recovery.
    """
    if config is None:
        config = Phase1Config()
    survey = SurveyModel(config=config)

    z_min, z_max = config.lens_z_range
    t_min, t_max = config.theta_E_range_arcsec

    z_l_grid = np.linspace(z_min, z_max, config.n_z_grid)
    theta_E_grid = np.linspace(t_min, t_max, config.n_theta_grid)

    detectability = np.zeros((config.n_z_grid, config.n_theta_grid), dtype=int)

    # In this phase, detectability class depends only on θ_E for a given
    # band, not directly on z_l. We still keep the grid 2D for later use.
    for j, theta_E in enumerate(theta_E_grid):
        cls = survey.classify_theta_E(theta_E)
        code = _DETECTABILITY_CODE[cls]
        detectability[:, j] = code

    return AnalyticWindowResult(
        z_l_grid=z_l_grid,
        theta_E_grid=theta_E_grid,
        detectability=detectability,
    )


def build_mass_analytic_window(
    config: Optional[Phase1Config] = None,
    n_mass_grid: int = 150,
) -> MassWindowResult:
    """
    Build an approximate observability map in (z_l, M_halo).

    Parameters
    ----------
    config : Phase1Config, optional
        Configuration with cosmology and grids.
    n_mass_grid : int
        Number of mass grid points along the M axis.

    Returns
    -------
    MassWindowResult
        Grid of lens redshift, halo mass (inside θ_E) and detectability.

    Method
    ------
    - We first compute θ_E_grid from config.
    - For each (θ_E, z_l) pair, we convert to M(<θ_E) for a *fixed*
      representative source redshift z_s (config.representative_source_z).
    - We then define a 1D mass grid that spans the full range of
      M(<θ_E) encountered and re-map detectability classes onto that
      grid.

    Caveats
    -------
    - This mapping assumes a single source redshift. Real lenses have
      a distribution of z_s, so the same (M_halo, z_l) pair can give
      different θ_E. For Phase 1 we accept this as an approximation and
      state the assumed z_s explicitly.
    """
    if config is None:
        config = Phase1Config()
    survey = SurveyModel(config=config)

    z_min, z_max = config.lens_z_range
    t_min, t_max = config.theta_E_range_arcsec

    z_l_grid = np.linspace(z_min, z_max, config.n_z_grid)
    theta_E_grid = np.linspace(t_min, t_max, config.n_theta_grid)

    z_s = config.representative_source_z

    # Compute mass and detectability per (z_l, theta_E)
    mass_map = np.zeros((config.n_z_grid, config.n_theta_grid), dtype=float)
    detectability_map = np.zeros_like(mass_map, dtype=int)

    for i, z_l in enumerate(z_l_grid):
        for j, theta_E in enumerate(theta_E_grid):
            if z_l >= z_s:
                # Unphysical; we can set NaN mass and mark as blind
                mass_map[i, j] = np.nan
                detectability_map[i, j] = _DETECTABILITY_CODE["blind"]
                continue

            mass_map[i, j] = sis_mass_inside_theta_E(
                theta_E_arcsec=theta_E,
                z_l=z_l,
                z_s=z_s,
                config=config,
            )
            cls = survey.classify_theta_E(theta_E)
            detectability_map[i, j] = _DETECTABILITY_CODE[cls]

    # Build a monotonic mass grid that spans the physical region
    finite_mass = np.isfinite(mass_map)
    if not np.any(finite_mass):
        raise RuntimeError("No finite mass values computed; check config and redshift ranges.")

    m_min = np.nanmin(mass_map[finite_mass])
    m_max = np.nanmax(mass_map[finite_mass])

    mass_grid = np.logspace(np.log10(m_min), np.log10(m_max), n_mass_grid)

    # Map detectability from (z_l, theta_E) to (z_l, M) by nearest θ_E bin
    # for each mass at fixed z_l. This is only approximate, but good enough
    # for a qualitative halo-mass window.
    detectability_mass = np.zeros((config.n_z_grid, n_mass_grid), dtype=int)

    for i in range(config.n_z_grid):
        masses_row = mass_map[i, :]
        # For each target mass, find index of closest computed mass
        for k, m_target in enumerate(mass_grid):
            idx = np.nanargmin(np.abs(masses_row - m_target))
            detectability_mass[i, k] = detectability_map[i, idx]

    return MassWindowResult(
        z_l_grid=z_l_grid,
        mass_grid=mass_grid,
        detectability=detectability_mass,
    )


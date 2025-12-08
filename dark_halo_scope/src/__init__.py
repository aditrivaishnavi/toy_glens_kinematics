"""
dark_halo_scope

Phase 1 tools for mapping the analytic strong-lensing observability window
of the DESI Legacy Imaging Surveys DR10.

This package is intentionally small and physics focused. It provides:

- A simple survey model for DR10 (pixel scale, seeing, depth).
- Basic SIS lensing relations (Einstein radius and enclosed mass).
- Functions to compute analytic detectability regions in (z_l, θ_E) and (z_l, M_halo).
- Phase 1.5: Region scouting and brick-level selection from DR10 TAP.

Later phases of the project (image simulations, injection–recovery, ML)
will build on this package rather than reimplementing the physics.
"""

from .config import Phase1Config, Phase1p5Config
from .cosmology import get_cosmology
from .lens_models import sis_theta_E, sis_mass_inside_theta_E
from .survey_model import SurveyModel
from .analytic_observability import (
    build_thetaE_analytic_window,
    build_mass_analytic_window,
)
from .region_scout import (
    get_tap_service,
    fetch_bricks,
    load_bricks_from_local_fits,
    apply_brick_quality_cuts,
    estimate_lrg_density_for_bricks,
    estimate_lrg_density_from_sweeps,
    select_regions,
    BrickRecord,
)


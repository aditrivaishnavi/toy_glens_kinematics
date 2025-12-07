# Physics subpackage
"""
Lensing equations and observability calculations.

Modules:
- lens_equations: θ_E calculations for various lens models
- observability_maps: Seeing-limited regime classification
"""

from .lens_equations import theta_E_SIS, sigma_v_from_Mhalo
from .observability_maps import confusion_regions, evaluate_observability_grid, DR10_FWHM


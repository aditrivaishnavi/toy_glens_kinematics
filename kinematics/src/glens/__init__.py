# src/glens/__init__.py
"""
Gravitational lensing utilities for the toy_glens_kinematics project.

This package provides centralized lensing simulation functions used by:
- debug_subhalo_samples.py
- subhalo_cnn.py
- sis_demo_lens.py

All lensing, subhalo injection, PSF, noise, and masking logic should be
imported from here to ensure consistency across the codebase.
"""

from .lensing_utils import (
    LensSimConfig,
    load_source_tensor,
    build_theta_grid,
    sis_deflection,
    render_sis_lens,
    render_sis_plus_subhalo,
    apply_psf_and_noise,
)

__all__ = [
    "LensSimConfig",
    "load_source_tensor",
    "build_theta_grid",
    "sis_deflection",
    "render_sis_lens",
    "render_sis_plus_subhalo",
    "apply_psf_and_noise",
]


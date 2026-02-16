"""
Cosmology helpers for dark_halo_scope.

We use a flat Lambda-CDM cosmology. For reliability and clarity we
prefer the astropy.cosmology implementation. If astropy is not
available, we raise a clear error message, because having a correct
distance-redshift relation is critical for a physics-grade result.
"""

from functools import lru_cache
from typing import Optional

from .config import Phase1Config


try:
    from astropy.cosmology import FlatLambdaCDM  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "astropy is required for dark_halo_scope Phase 1.\n"
        "Please install it with:\n\n"
        "    pip install astropy\n"
    ) from exc


@lru_cache(maxsize=1)
def get_cosmology(config: Optional[Phase1Config] = None) -> FlatLambdaCDM:
    """
    Return a FlatLambdaCDM cosmology instance based on Phase1Config.

    Parameters
    ----------
    config : Phase1Config, optional
        Configuration instance. If None, the default Phase1Config()
        is used.

    Returns
    -------
    FlatLambdaCDM
        Cosmology object with H0 and Omega_m from the config.
    """
    if config is None:
        config = Phase1Config()

    return FlatLambdaCDM(H0=config.h0, Om0=config.omega_m)


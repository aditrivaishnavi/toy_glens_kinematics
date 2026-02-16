"""
Survey model and analytic detectability classification.

This module defines SurveyModel, which knows about:

- pixel scale
- seeing FWHM in each band
- simple analytic "seeing wall" thresholds

We intentionally separate the survey model from the lens physics so
that in later phases we can swap in a more sophisticated model that
uses per-object seeing distributions and noise properties extracted
directly from DR10 metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import Phase1Config


DetectabilityClass = Literal["blind", "low_trust", "good"]


@dataclass
class SurveyModel:
    """
    Encapsulate DR10 survey properties relevant for analytic detectability.

    Parameters
    ----------
    config : Phase1Config
        Configuration instance with default survey parameters and thresholds.

    Notes
    -----
    Currently we use a single "typical" seeing FWHM per band. This is a
    deliberate simplification for Phase 1, where we are mapping the
    *geometric* observability window. Later phases should replace this
    by sampling the true DR10 seeing distribution for the chosen lens
    sample (for example using the DECam survey-ccds tables).
    """

    config: Phase1Config

    def __post_init__(self) -> None:
        # Pre-compute the thresholds in arcsec for the r band.
        self._fwhm_r = self.config.seeing_fwhm_r
        self._theta_blind_r = self.config.k_blind * self._fwhm_r
        self._theta_good_r = self.config.k_good * self._fwhm_r

    @property
    def pixel_scale_arcsec(self) -> float:
        return self.config.pixel_scale_arcsec

    @property
    def fwhm_r_arcsec(self) -> float:
        return self._fwhm_r

    def classify_theta_E(self, theta_E_arcsec: float) -> DetectabilityClass:
        """
        Classify a given Einstein radius in the r band into
        blind / low_trust / good based purely on seeing.

        Parameters
        ----------
        theta_E_arcsec : float
            Einstein radius in arcseconds.

        Returns
        -------
        DetectabilityClass
            "blind", "low_trust" or "good".

        Interpretation
        --------------
        - blind: arcs are essentially inside the PSF core; strong-lensing
          morphology is not resolvable in DR10 coadds.
        - low_trust: some tangential stretching is possible but blended
          with the lens galaxy. Any completeness estimate here must
          rely heavily on injection–recovery.
        - good: Einstein ring radius is at least of order the PSF FWHM;
          the arc or ring is geometrically resolvable in principle.
        """
        if theta_E_arcsec < self._theta_blind_r:
            return "blind"
        if theta_E_arcsec < self._theta_good_r:
            return "low_trust"
        return "good"


from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Phase1Config:
    """
    Configuration container for Phase 1 analytic calculations.

    All numbers here are *defaults* based on DR10 documentation and
    the Dey et al. (2019) Legacy Surveys overview paper. They should
    be easy to change in one place if more precise values are needed.

    Notes
    -----
    - Pixel scale: 0.262 arcsec / pixel for DECam and the DR10 coadds.
    - Seeing: median FWHM in the r band is typically 1.2–1.3 arcsec.
      We store a single "typical" value here, but later phases can
      ingest the true per-object seeing distribution from survey CCD
      metadata.
    - Magnitude limits: rough 5-sigma depths. We do not use them in
      the analytic "geometry only" window, but they are recorded for
      completeness and later use (for brightness / SNR modeling).
    """

    # Basic survey geometry
    pixel_scale_arcsec: float = 0.262  # arcsec / pixel

    # Typical FWHM seeing (arcsec) in each band; r is used as the main lensing band.
    seeing_fwhm_g: float = 1.35
    seeing_fwhm_r: float = 1.25
    seeing_fwhm_z: float = 1.30

    # Approximate 5-sigma point-source depths (AB mag)
    depth_g_5sigma: float = 24.5
    depth_r_5sigma: float = 23.8
    depth_z_5sigma: float = 23.0

    # Cosmology parameters (Planck 2018 flat LCDM)
    # Reference: Planck Collaboration 2018, A&A 641, A6
    h0: float = 67.4  # km/s/Mpc
    omega_m: float = 0.315

    # Typical source redshift range to use for mapping θ_E -> M_halo
    source_z_min: float = 1.5
    source_z_max: float = 2.5
    # For analytic maps we can pick a representative z_s in this range:
    representative_source_z: float = 2.0

    # Thresholds for analytic "seeing wall"
    # Blind: θ_E < k_blind * FWHM  (unresolvable)
    # Low-trust: k_blind * FWHM <= θ_E < k_good * FWHM
    # Good: θ_E >= k_good * FWHM
    k_blind: float = 0.5
    k_good: float = 1.0

    # Grids for analytic maps
    lens_z_range: Tuple[float, float] = (0.1, 1.0)  # foreground lens redshift
    theta_E_range_arcsec: Tuple[float, float] = (0.1, 3.0)
    n_z_grid: int = 100
    n_theta_grid: int = 150


@dataclass
class Phase1p5Config:
    """
    Configuration for Phase 1.5: Region scouting and brick-level selection.

    This phase uses DR10 survey-bricks (ls_dr10.bricks_s) to:
      - Apply hard image-quality cuts (seeing, depth, extinction, exposures).
      - Estimate DESI-like LRG density per brick.
      - Select a contiguous set of bricks as the primary region.
    """

    # TAP service for NOIRLab Astro Data Lab (DR10 tables)
    tap_url: str = "https://datalab.noirlab.edu/tap"

    # Table names
    bricks_table: str = "ls_dr10.bricks_s"  # survey-bricks for southern footprint
    tractor_table: str = "ls_dr10.tractor_s"  # Tractor catalog for galaxy densities

    # Scouting footprint (RA, Dec in degrees)
    ra_min: float = 150.0
    ra_max: float = 250.0
    dec_min: float = 0.0
    dec_max: float = 30.0

    # Hard brick-level quality cuts
    max_psfsize_r: float = 1.5        # arcsec; discard bricks with worse seeing
    min_psfdepth_r: float = 23.3      # approximate 5σ depth threshold in r
    max_ebv: float = 0.1              # max E(B−V) to avoid heavy extinction
    min_nexp_r: int = 2               # require at least 2 r-band exposures, if available

    # LRG proxy cuts (DESI-like, all mags extinction-corrected AB)
    # z-band brightness: approximate DESI LRG limit
    lrg_z_mag_max: float = 20.4
    # optical color: selects red galaxies
    lrg_min_r_minus_z: float = 0.4
    # optical–IR color: selects massive, dusty galaxies
    lrg_min_z_minus_w1: float = 1.6

    # Optional photo-z refining (can be turned off by setting to None)
    use_photo_z: bool = False
    lrg_z_phot_min: float = 0.3
    lrg_z_phot_max: float = 0.8

    # Region size targets
    min_region_area_deg2: float = 100.0   # desired minimum total area
    max_region_area_deg2: float = 400.0   # soft upper limit for primary region

    # How many best bricks to compute LRG density for (to control TAP volume)
    max_bricks_for_lrg_density: int = 500

    # Output directory (relative to project root)
    output_dir: str = "outputs/phase1p5"

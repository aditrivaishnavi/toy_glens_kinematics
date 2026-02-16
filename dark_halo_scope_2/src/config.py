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
      - Apply hard image-quality cuts (seeing, depth, extinction).
      - Estimate DESI-like LRG density per brick.
      - Select a contiguous set of bricks as the primary region.

    It now also includes configuration for an EMR + PySpark backend which
    computes LRG counts per brick from DR10 SWEEP files in parallel and
    writes a compact summary to S3.
    """

    # TAP service for NOIRLab Astro Data Lab (DR10 tables)
    tap_url: str = "https://datalab.noirlab.edu/tap"

    # Table names
    bricks_table: str = "ls_dr10.bricks_s"   # survey-bricks for southern footprint
    tractor_table: str = "ls_dr10.tractor_s"  # Tractor catalog (TAP-based path)

    # Scouting footprint (RA, Dec in degrees)
    ra_min: float = 150.0
    ra_max: float = 250.0
    dec_min: float = 0.0
    dec_max: float = 30.0

    # Hard brick-level quality cuts
    max_psfsize_r: float = 1.5       # arcsec; discard bricks with worse seeing
    min_psfdepth_r: float = 23.3     # approximate 5σ depth threshold in r
    max_ebv: float = 0.1             # max E(B−V) to avoid heavy extinction

    # LRG proxy cuts (DESI-like, all mags AB based on nanomaggies)
    lrg_z_mag_max: float = 20.4
    lrg_min_r_minus_z: float = 0.4
    lrg_min_z_minus_w1: float = 1.6

    # Optional photo-z refining (not used for DR10 TAP or SWEEPs by default)
    use_photo_z: bool = False
    lrg_z_phot_min: float = 0.3
    lrg_z_phot_max: float = 0.8

    # Region size targets
    min_region_area_deg2: float = 100.0
    max_region_area_deg2: float = 400.0

    # How many best bricks to compute LRG density for (for TAP-based path)
    max_bricks_for_lrg_density: int = 500

    # Pilot mode: small number for fast test; 0 = full
    pilot_brick_limit: int = 0

    # Local DR10 data root (for non-EMR local workflows)
    data_root: str = "external_data/dr10"
    bricks_fits: str = "external_data/dr10/survey-bricks-dr10-south.fits.gz"
    sweeps_dir: str = "external_data/dr10/sweep_10.1"

    # Local usage toggles (non-EMR)
    use_local_bricks: bool = True
    use_local_sweeps: bool = True
    delete_sweeps_after_use: bool = True

    # Whether to prefer SWEEP-based LRG estimation (vs TAP) in local workflows
    use_sweeps_for_lrg: bool = False

    # SWEEP index list (one path or URL per line)
    sweep_index_path: str = "external_data/dr10/sweeps_ra150_250_dec0_30_10.1.txt"

    # Temporary directory for SWEEP downloads (when entries are URLs)
    sweep_download_dir: str = "external_data/dr10/tmp_sweeps"

    # Max number of SWEEP files to process in parallel (local workflow)
    max_parallel_sweeps: int = 3

    # Optional limit on number of SWEEP files to process (local workflow)
    max_sweeps_for_lrg_density: int = 5

    # Output directory (relative to project root)
    output_dir: str = "outputs/phase1p5"

    # Checkpoint directory for resumable runs (separate from output_dir)
    checkpoint_dir: str = "checkpoints/phase1p5"

    # ------------- EMR / S3 specific options -------------

    # S3 bucket + base prefix where EMR outputs will be written.
    # Example: "s3://my-bucket/dark_halo_scope/phase1p5"
    emr_s3_output_prefix: str = "s3://CHANGE_ME_BUCKET/phase1p5"

    # Name of the EMR log S3 prefix (optional, for cluster logs)
    emr_s3_log_prefix: str = "s3://CHANGE_ME_BUCKET/emr-logs"

    # EMR cluster parameters (defaults are small but you can override
    # in the EMR submission script by passing CLI arguments).
    emr_release_label: str = "emr-6.15.0"
    emr_master_instance_type: str = "m5.xlarge"
    emr_core_instance_type: str = "m5.xlarge"
    emr_core_instance_count: int = 3

    # EMR job name base
    emr_job_name: str = "dark-halo-scope-phase1p5"

    # AWS EMR + EC2 roles (to be filled in by you)
    emr_service_role: str = "EMR_DefaultRole"
    emr_job_flow_role: str = "EMR_EC2_DefaultRole"

    # S3 location of the zipped project or py-files for Spark (you will upload it)
    emr_s3_code_archive: str = "s3://CHANGE_ME_BUCKET/code/dark_halo_scope_code.tgz"

    # Path (inside the code archive) to the PySpark driver script
    emr_pyspark_driver_path: str = "emr/spark_phase1p5_lrg_density.py"

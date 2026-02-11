#!/usr/bin/env python3
"""
Central Constants for StrongLens Calibration Pipeline

All AWS, S3, and pipeline configuration constants in one place.
Import from here to ensure consistency across all scripts.

IMPORTANT: These paths MUST match the actual S3 structure exactly!
Any mismatch can cause silent failures or data loss.

Usage:
    from stronglens_calibration.constants import S3_BUCKET, AWS_REGION
    # or for EMR scripts running on cluster:
    from constants import S3_BUCKET, AWS_REGION
"""
import os

# =============================================================================
# AWS CONFIGURATION
# =============================================================================

AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

# =============================================================================
# S3 BUCKET
# =============================================================================

S3_BUCKET = os.environ.get("S3_BUCKET", "darkhaloscope")

# =============================================================================
# S3 PREFIXES - GLOBAL PATHS (outside stronglens_calibration/)
# These are shared resources used by multiple projects
# =============================================================================

# DR10 data - sweeps and coadd cache
# Structure: dr10/sweeps/sweep-{ra}m{dec}-{ra}p{dec}.fits.gz (1,436 files)
# Structure: dr10/coadd_cache/{brickname}/ (180,152 brick directories)
S3_DR10_PREFIX = "dr10"
S3_DR10_SWEEPS_PREFIX = f"{S3_DR10_PREFIX}/sweeps"
S3_DR10_COADD_CACHE_PREFIX = f"{S3_DR10_PREFIX}/coadd_cache"
S3_DR10_SWEEPS_MANIFEST_PREFIX = f"{S3_DR10_PREFIX}/sweeps_manifest"

# Legacy sweep dump (older format)
S3_SWEEP_FITS_DUMP_PREFIX = "sweep_fits_dump"

# Cosmos source banks
S3_COSMOS_BANKS_PREFIX = "data/cosmos_banks"

# Models
S3_MODELS_PREFIX = "phase5_models"

# Previous pipeline (planb)
S3_PLANB_PREFIX = "planb"

# =============================================================================
# S3 PREFIXES - STRONGLENS_CALIBRATION PATHS
# All paths under stronglens_calibration/
# =============================================================================

# Base prefix for all stronglens_calibration data
S3_BASE_PREFIX = "stronglens_calibration"

# EMR code, logs, and bootstrap
# Structure: emr/code/{timestamp}/{script}.py
# Structure: emr/logs/{cluster_id}/
# Structure: emr/bootstrap/{timestamp}/bootstrap.sh
S3_EMR_PREFIX = f"{S3_BASE_PREFIX}/emr"
S3_CODE_PREFIX = f"{S3_EMR_PREFIX}/code"
S3_LOGS_PREFIX = f"{S3_EMR_PREFIX}/logs"
S3_BOOTSTRAP_PREFIX = f"{S3_EMR_PREFIX}/bootstrap"

# Configuration files
# Structure: configs/{timestamp}/ or configs/positives/
S3_CONFIG_PREFIX = f"{S3_BASE_PREFIX}/configs"
S3_CONFIG_POSITIVES_PREFIX = f"{S3_CONFIG_PREFIX}/positives"

# Positive lens candidates with DR10 crossmatch
# Structure: positives_with_dr10/{timestamp}/data/*.parquet
S3_POSITIVES_PREFIX = f"{S3_BASE_PREFIX}/positives_with_dr10"

# Negative pool manifests (from negative sampling job)
# Structure: manifests/{timestamp}/data/*.parquet
S3_MANIFESTS_PREFIX = f"{S3_BASE_PREFIX}/manifests"

# Stratified sampled negatives (after 100:1 sampling)
# Structure: sampled_negatives/{timestamp}/data/*.parquet
S3_SAMPLED_NEGATIVES_PREFIX = f"{S3_BASE_PREFIX}/sampled_negatives"

# Cutouts (images)
# Structure: cutouts/{positives,negatives}/{timestamp}/{galaxy_id}.npz
S3_CUTOUTS_PREFIX = f"{S3_BASE_PREFIX}/cutouts"
S3_CUTOUTS_POSITIVES_PREFIX = f"{S3_CUTOUTS_PREFIX}/positives"
S3_CUTOUTS_NEGATIVES_PREFIX = f"{S3_CUTOUTS_PREFIX}/negatives"

# Validation results
# Structure: validation/{positives,negatives}/{timestamp}/
S3_VALIDATION_PREFIX = f"{S3_BASE_PREFIX}/validation"

# Checkpoints for job resume (created on demand)
# Structure: checkpoints/{job_name}/{timestamp}/
S3_CHECKPOINT_PREFIX = f"{S3_BASE_PREFIX}/checkpoints"

# =============================================================================
# CUTOUT PARAMETERS
# =============================================================================

# Downloaded cutout size (101x101 = ~26.5 arcsec at 0.262"/pixel)
# Larger cutouts capture more context and are center-cropped to TRAIN_SIZE during training
CUTOUT_SIZE = 101  # pixels
PIXEL_SCALE = 0.262  # arcsec/pixel
CUTOUT_ARCSEC = CUTOUT_SIZE * PIXEL_SCALE  # ~26.5 arcsec
CUTOUT_BANDS = "grz"

# Training image size (center-cropped from CUTOUT_SIZE during preprocessing)
# 64x64 = ~16.8 arcsec, focused on the lens system core
TRAIN_SIZE = 64

# Legacy Survey cutout service
CUTOUT_URL_TEMPLATE = (
    "https://www.legacysurvey.org/viewer/fits-cutout"
    "?ra={ra}&dec={dec}&size={size}&layer=ls-dr10&pixscale={pixscale}&bands={bands}"
)

# =============================================================================
# DR10 PHOTOMETRY (Injection / flux calibration)
# =============================================================================
# Legacy Survey coadds use nanomaggies; 1 nanomaggy = flux of a mag 22.5 source (AB).
# To convert magnitude to flux: flux_nMgy = 10^(-0.4 * (mag - 22.5))
# Used for injection realism (Phase 4c) and any flux-calibrated simulation.
# Reference: Paper IV; stronglens_calibration is self-contained (no dark_halo_scope dependency).
AB_ZEROPOINT_MAG = 22.5  # AB zero-point for nanomaggies (DR10)

# =============================================================================
# DR10 SURVEY PARAMETERS
# =============================================================================

# DECaLS footprint declination bounds (Paper IV parity)
DECALS_DEC_MIN = -18.0
DECALS_DEC_MAX = 32.0

# Minimum exposures per band (Paper IV parity)
MIN_NOBS_PER_BAND = 3

# z-band magnitude limit (Paper IV parity)
Z_MAG_LIMIT = 20.0

# Valid morphological types for N1 pool (Paper IV parity: SER/DEV/REX only)
VALID_TYPES_N1 = {"SER", "DEV", "REX"}

# =============================================================================
# EMR CONFIGURATION
# =============================================================================

EMR_RELEASE = "emr-7.0.0"

# Instance presets for different job sizes
EMR_PRESETS = {
    "test": {
        "master_type": "m5.xlarge",
        "worker_type": "m5.xlarge", 
        "worker_count": 2,
        "executor_memory": "4g",
        "description": "Test with 2 workers",
    },
    "small": {
        "master_type": "m5.xlarge",
        "worker_type": "m5.xlarge",
        "worker_count": 5,
        "executor_memory": "4g",
        "description": "Small with 5 workers",
    },
    "medium": {
        "master_type": "m5.xlarge",
        "worker_type": "m5.2xlarge",
        "worker_count": 10,
        "executor_memory": "8g",
        "description": "Medium with 10 workers",
    },
    "medium-large": {
        "master_type": "m5.xlarge",
        "worker_type": "m5.2xlarge",
        "worker_count": 20,
        "executor_memory": "8g",
        "description": "Medium-large with 20 workers (160 vCPUs)",
    },
    "large-xlarge": {
        "master_type": "m5.xlarge",
        "worker_type": "m5.xlarge",
        "worker_count": 30,
        "executor_memory": "4g",
        "description": "30x m5.xlarge workers (120 vCPUs)",
    },
    "large": {
        "master_type": "m5.xlarge",
        "worker_type": "m5.2xlarge",
        "worker_count": 30,
        "executor_memory": "8g",
        "description": "Large with 30 workers",
    },
    "full": {
        "master_type": "m5.xlarge",
        "worker_type": "m5.2xlarge",
        "worker_count": 30,
        "executor_memory": "8g",
        "description": "Full with 30 workers",
    },
}

# =============================================================================
# PIPELINE VERSION
# =============================================================================

PIPELINE_VERSION = "1.0.0"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def s3_uri(prefix: str, suffix: str = "") -> str:
    """Build full S3 URI from prefix and optional suffix."""
    path = f"{prefix}/{suffix}".rstrip("/")
    return f"s3://{S3_BUCKET}/{path}"


def get_cutout_url(ra: float, dec: float, size: int = CUTOUT_SIZE) -> str:
    """Build Legacy Survey cutout URL."""
    return CUTOUT_URL_TEMPLATE.format(
        ra=ra,
        dec=dec,
        size=size,
        pixscale=PIXEL_SCALE,
        bands=CUTOUT_BANDS,
    )


def get_emr_console_url(cluster_id: str) -> str:
    """Build AWS EMR console URL for a cluster."""
    return f"https://{AWS_REGION}.console.aws.amazon.com/emr/home?region={AWS_REGION}#/clusterDetails/{cluster_id}"


def get_emr_terminate_cmd(cluster_id: str) -> str:
    """Build AWS CLI command to terminate an EMR cluster."""
    return f"aws emr terminate-clusters --cluster-ids {cluster_id} --region {AWS_REGION}"

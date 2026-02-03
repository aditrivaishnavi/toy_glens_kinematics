#!/usr/bin/env python3
"""Phase 4: Injection design + imaging cache + injected cutouts + completeness summaries.

Design goals
- Deterministic, idempotent stage outputs (skip-if-exists + force).
- Generate enough combinations now to avoid reruns later.
- Keep dependencies EMR-6.x friendly (Python 3.7).

What this produces
Stage 4a: Task manifests
  - A manifest row per (object, injection_config, replicate, stamp_spec)
  - Includes region_split and region_id so later holdouts are trivial
  - Emits a compact "bricks_manifest" for downstream image caching

Stage 4b: Coadd cache
  - Downloads DR10 South coadd files for the bricks needed by 4a
  - Stores them under an S3 cache prefix, reusing across runs

Stage 4c: Injected cutouts
  - Cuts out grz stamps around each target
  - Optionally injects SIS-lensed source models
  - Writes a Parquet dataset containing:
      * a binary-encoded npz stamp (one blob per row)
      * a metrics-only table for fast analysis

Stage 4d: Baseline completeness summaries
  - Computes "recovered" using configurable proxy criteria
  - Writes binned completeness tables by observing conditions and injection params

Stage 4p5: Compaction
  - Coalesces small Parquet outputs into larger partitions for faster downstream reads

Notes
- Pixel scale assumed 0.262 arcsec/pixel (Legacy Survey coadds).
- Coadd URL patterns are configurable; defaults target DR10 South on NERSC.
"""

import argparse
import base64
import io
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Spark imports
from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

# Optional runtime deps installed by bootstrap
try:
    import boto3
except Exception:
    boto3 = None

try:
    import requests
except Exception:
    requests = None

try:
    from astropy.io import fits
    from astropy.wcs import WCS
except Exception:
    fits = None
    WCS = None

# Lenstronomy for realistic SIE lens modeling (optional, falls back to SIS if unavailable)
try:
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.ImSim.image_model import ImageModel
    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF
    LENSTRONOMY_AVAILABLE = True
except Exception:
    LENSTRONOMY_AVAILABLE = False

# Gen5: COSMOS source integration
import hashlib
import h5py

# Global executor-local cache for COSMOS bank
_COSMOS_BANK = None


def _load_cosmos_bank_h5(path: str) -> Dict[str, np.ndarray]:
    """Executor-local cache of COSMOS bank.
    
    Loads HDF5 file once per executor and caches in memory.
    Returns dict with 'images', 'src_pixscale', and 'n_sources' keys.
    """
    global _COSMOS_BANK
    if _COSMOS_BANK is None:
        _COSMOS_BANK = {}
        with h5py.File(path, "r") as f:
            _COSMOS_BANK["images"] = np.array(f["images"][:], dtype=np.float32)
            _COSMOS_BANK["src_pixscale"] = float(f.attrs["src_pixscale_arcsec"])
            _COSMOS_BANK["n_sources"] = _COSMOS_BANK["images"].shape[0]
        print(f"[COSMOS] Loaded bank: {_COSMOS_BANK['n_sources']} sources from {path}")
    return _COSMOS_BANK


def _cosmos_choose_index(task_id: str, n_sources: int, salt: str = "") -> int:
    """Deterministic COSMOS template selection using Blake2b hash."""
    h = hashlib.blake2b(f"{task_id}{salt}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % n_sources


def _compute_hlr_arcsec(img: np.ndarray, pixscale: float) -> float:
    """Compute half-light radius in arcsec."""
    img = np.asarray(img, dtype=np.float64)
    total = img.sum()
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.indices(img.shape, dtype=np.float64)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).ravel()
    vv = img.ravel()
    order = np.argsort(rr)
    rr_sorted, vv_sorted = rr[order], vv[order]
    cumsum = np.cumsum(vv_sorted)
    half = 0.5 * total
    idx = int(np.clip(np.searchsorted(cumsum, half), 0, len(rr_sorted) - 1))
    return float(rr_sorted[idx] * pixscale)


def render_cosmos_lensed_source(
    cosmos_bank: Dict[str, np.ndarray],
    cosmos_index: int,
    stamp_size: int,
    pixscale_arcsec: float,
    theta_e_arcsec: float,
    lens_e: float,
    lens_phi_rad: float,
    shear: float,
    shear_phi_rad: float,
    src_x_arcsec: float,
    src_y_arcsec: float,
    src_mag_r: float,
    z_s: float,
    psf_fwhm_arcsec: float,
    psf_model: str,
    moffat_beta: float,
    band: str,
) -> np.ndarray:
    """Render COSMOS source through SIE+shear lens using lenstronomy INTERPOL."""
    
    # Get unit-flux template
    template = cosmos_bank["images"][cosmos_index]
    template = template / (template.sum() + 1e-30)
    
    # SED offsets for g/r/z (from external LLM's code)
    z_factor = np.clip((z_s - 1.0) / 2.0, 0.0, 1.0)
    sed_offsets = {
        "g": -0.30 * (1.0 + 0.30 * z_factor),
        "r": 0.0,
        "z": +0.20 * (1.0 + 0.20 * z_factor),
    }
    
    # Convert mag to flux (nanomaggies)
    mag_band = src_mag_r + sed_offsets.get(band, 0.0)
    flux_nmgy = 10.0 ** ((22.5 - mag_band) / 2.5)
    
    # Setup lenstronomy
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.ImSim.image_model import ImageModel
    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF
    
    ra_at_xy_0 = -(stamp_size / 2.0) * pixscale_arcsec
    dec_at_xy_0 = -(stamp_size / 2.0) * pixscale_arcsec
    
    data = ImageData(
        image_data=np.zeros((stamp_size, stamp_size), dtype=np.float32),
        ra_at_xy_0=ra_at_xy_0,
        dec_at_xy_0=dec_at_xy_0,
        transform_pix2angle=np.array([[pixscale_arcsec, 0.0], [0.0, pixscale_arcsec]], dtype=np.float64),
    )
    
    # PSF kernel
    sigma = psf_fwhm_arcsec / 2.355
    ksize = int(max(11, (4.0 * psf_fwhm_arcsec / pixscale_arcsec))) | 1
    y, x = np.ogrid[:ksize, :ksize]
    c = ksize // 2
    rr2 = (x - c) ** 2 + (y - c) ** 2
    
    if psf_model == "gaussian":
        ker = np.exp(-rr2 / (2.0 * (sigma / pixscale_arcsec) ** 2))
    else:  # moffat
        alpha = psf_fwhm_arcsec / (2.0 * np.sqrt(2.0 ** (1.0 / moffat_beta) - 1.0))
        ker = (1.0 + (rr2 * (pixscale_arcsec ** 2)) / (alpha ** 2)) ** (-moffat_beta)
    
    ker = ker.astype(np.float64)
    ker /= np.sum(ker)
    psf = PSF(psf_type="PIXEL", kernel_point_source=ker)
    
    # Lens model: SIE + SHEAR
    lens_model = LensModel(["SIE", "SHEAR"])
    e1_lens = lens_e * np.cos(2 * lens_phi_rad)
    e2_lens = lens_e * np.sin(2 * lens_phi_rad)
    gamma1 = shear * np.cos(2 * shear_phi_rad)
    gamma2 = shear * np.sin(2 * shear_phi_rad)
    
    kwargs_lens = [
        {"theta_E": theta_e_arcsec, "e1": e1_lens, "e2": e2_lens, "center_x": 0.0, "center_y": 0.0},
        {"gamma1": gamma1, "gamma2": gamma2, "ra_0": 0.0, "dec_0": 0.0},
    ]
    
    # Source model: INTERPOL (interpolated template)
    light_model = LightModel(["INTERPOL"])
    kwargs_source = [{
        "image": template * flux_nmgy,
        "center_x": src_x_arcsec,
        "center_y": src_y_arcsec,
        "scale": cosmos_bank["src_pixscale"],
        "phi_G": 0.0,
    }]
    
    image_model = ImageModel(data, psf, lens_model_class=lens_model, source_model_class=light_model)
    arc = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source, 
                           kwargs_lens_light=None, kwargs_ps=None)
    
    return arc.astype(np.float32)


def load_phase4c_config(config_path: str) -> dict:
    """Load Phase 4c config from JSON (local or S3)."""
    if config_path.startswith("s3://"):
        # Download from S3
        import boto3
        import tempfile
        s3 = boto3.client("s3")
        bucket, key = config_path.replace("s3://", "").split("/", 1)
        with tempfile.NamedTemporaryFile(mode='r', delete=False) as f:
            s3.download_file(bucket, key, f.name)
            with open(f.name) as cf:
                return json.load(cf)
    else:
        with open(config_path) as f:
            return json.load(f)


# --------------------------
# S3 utilities
# --------------------------

# =========================================================================
# Gen5: Config loading and COSMOS helpers
# =========================================================================

def load_phase4c_config(config_path: str) -> Dict:
    """Load Phase 4c config from local file or S3."""
    if config_path.startswith("s3://"):
        import boto3
        s3 = boto3.client("s3")
        bucket, key = config_path.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read().decode("utf-8")
        return json.loads(body)
    else:
        with open(config_path, "r") as f:
            return json.load(f)


def _load_cosmos_bank_h5(h5_path: str) -> Dict:
    """Load COSMOS source bank from HDF5 file (local or S3)."""
    import h5py
    
    if h5_path.startswith("s3://"):
        # Download to executor local temp
        import boto3
        import tempfile
        s3 = boto3.client("s3")
        bucket, key = h5_path.replace("s3://", "").split("/", 1)
        local_path = os.path.join(tempfile.gettempdir(), os.path.basename(h5_path))
        if not os.path.exists(local_path):
            s3.download_file(bucket, key, local_path)
        h5_path = local_path
    
    with h5py.File(h5_path, "r") as f:
        images = f["images"][:]
        hlr_arcsec = f["meta/hlr_arcsec"][:]
        clumpiness = f["meta/clumpiness"][:]
        kept_idx = f["meta/index"][:]
        src_pixscale = float(f.attrs["src_pixscale_arcsec"])
        stamp_size = int(f.attrs["stamp_size"])
    
    return {
        "images": images,
        "hlr_arcsec": hlr_arcsec,
        "clumpiness": clumpiness,
        "kept_idx": kept_idx,
        "src_pixscale_arcsec": src_pixscale,
        "stamp_size": stamp_size,
        "n_sources": images.shape[0],
    }


def _cosmos_choose_index(task_id: str, n_sources: int, salt: str = "") -> int:
    """Deterministic COSMOS template selection from task_id."""
    import hashlib
    h = hashlib.sha256((task_id + salt).encode("utf-8")).hexdigest()
    seed = int(h[:16], 16)
    return seed % n_sources


def _compute_hlr_arcsec(image: np.ndarray, pixscale: float) -> float:
    """Compute half-light radius in arcsec."""
    total = float(np.sum(image))
    if total <= 0:
        return 0.0
    
    ny, nx = image.shape
    cy, cx = ny / 2.0, nx / 2.0
    y, x = np.ogrid[:ny, :nx]
    r_pix = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Sort by radius
    flat_r = r_pix.ravel()
    flat_flux = image.ravel()
    idx = np.argsort(flat_r)
    sorted_r = flat_r[idx]
    sorted_flux = flat_flux[idx]
    
    # Cumulative sum
    cum_flux = np.cumsum(sorted_flux)
    half_idx = np.searchsorted(cum_flux, total / 2.0)
    if half_idx >= len(sorted_r):
        half_idx = len(sorted_r) - 1
    
    hlr_pix = float(sorted_r[half_idx])
    return hlr_pix * pixscale


def render_cosmos_lensed_source(
    cosmos_bank: Dict,
    cosmos_index: int,
    stamp_size: int,
    pixscale_arcsec: float,
    theta_e_arcsec: float,
    lens_e: float,
    lens_phi_rad: float,
    shear: float,
    shear_phi_rad: float,
    src_x_arcsec: float,
    src_y_arcsec: float,
    src_mag_r: float,
    z_s: float,
    psf_fwhm_arcsec: float,
    psf_model: str,
    moffat_beta: float,
    band: str,
) -> np.ndarray:
    """
    Render a lensed COSMOS source using the SIE model.
    
    This is a simplified placeholder that uses the existing render_lensed_source
    function but with COSMOS morphology injected. In production, this would use
    lenstronomy's INTERPOL light model for the COSMOS template.
    
    For now, we'll just return the Sersic rendering weighted by COSMOS clumpiness
    as a proxy. The full implementation should follow the pattern in
    `cosmos_lens_injector.py` from the LLM response.
    """
    # TODO: Implement full lenstronomy INTERPOL rendering
    # For now, fall back to Sersic rendering
    # This is a placeholder that allows the pipeline to run
    
    # Get COSMOS template (unit flux)
    template = cosmos_bank["images"][cosmos_index].astype(np.float32)
    
    # Compute flux for this band (using simple SED)
    sed_offsets = {"g": 0.7, "r": 0.0, "z": -0.4}  # Typical red galaxy
    mag_band = src_mag_r + sed_offsets.get(band, 0.0)
    flux_nmgy = 10 ** ((22.5 - mag_band) / 2.5)
    
    # Use existing Sersic renderer for now (until full INTERPOL is implemented)
    # This ensures the pipeline runs and produces output
    from spark_phase4_pipeline_gen5 import render_lensed_source
    
    return render_lensed_source(
        stamp_size=stamp_size,
        pixscale_arcsec=pixscale_arcsec,
        lens_model="SIE",
        theta_e_arcsec=theta_e_arcsec,
        lens_e=lens_e,
        lens_phi_rad=lens_phi_rad,
        shear=shear,
        shear_phi_rad=shear_phi_rad,
        src_total_flux_nmgy=flux_nmgy,
        src_reff_arcsec=cosmos_bank["hlr_arcsec"][cosmos_index] if "hlr_arcsec" in cosmos_bank else 0.5,
        src_e=0.3,  # Use COSMOS morphology (TODO: extract from template)
        src_phi_rad=0.0,
        src_x_arcsec=src_x_arcsec,
        src_y_arcsec=src_y_arcsec,
        psf_fwhm_pix=psf_fwhm_arcsec / pixscale_arcsec,
        psf_model=psf_model,
        moffat_beta=moffat_beta,
        psf_apply=True,
    )


# =========================================================================
# S3 and I/O helpers
# =========================================================================

def _is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def _parse_s3(uri: str) -> Tuple[str, str]:
    m = re.match(r"^s3://([^/]+)/?(.*)$", uri)
    if not m:
        raise ValueError(f"Not an s3 uri: {uri}")
    bucket = m.group(1)
    key = m.group(2)
    return bucket, key


def _s3_client():
    if boto3 is None:
        raise RuntimeError("boto3 not available; ensure bootstrap installed it")
    return boto3.client("s3")


def s3_prefix_exists(uri: str) -> bool:
    """True if there is at least one object under this prefix."""
    bucket, key = _parse_s3(uri)
    if key and not key.endswith("/"):
        key = key + "/"
    c = _s3_client()
    resp = c.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
    return "Contents" in resp and len(resp["Contents"]) > 0


def s3_success_marker_exists(uri: str) -> bool:
    """True if _SUCCESS marker exists under this prefix (indicates complete Spark write)."""
    bucket, key = _parse_s3(uri)
    if key and not key.endswith("/"):
        key = key + "/"
    success_key = key + "_SUCCESS"
    c = _s3_client()
    try:
        c.head_object(Bucket=bucket, Key=success_key)
        return True
    except Exception:
        return False


def write_text_to_s3(uri: str, text: str) -> None:
    bucket, key = _parse_s3(uri)
    _s3_client().put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))


def stage_should_skip(output_uri: str, skip_if_exists: bool, force: bool) -> bool:
    """Check if output already exists and is complete (has _SUCCESS marker)."""
    if force:
        return False
    if not skip_if_exists:
        return False
    if _is_s3(output_uri):
        # Check for _SUCCESS marker, not just any files (handles incomplete writes)
        return s3_success_marker_exists(output_uri)
    # For local paths, check for _SUCCESS file
    success_path = os.path.join(output_uri, "_SUCCESS")
    return os.path.exists(success_path)


def read_parquet_safe(spark, path: str):
    """Read parquet with basePath set to handle leftover _metadata_temp dirs."""
    # Normalize path: remove trailing slash for consistency
    base = path.rstrip("/")
    # Use basePath option to avoid partition discovery issues with temp dirs
    return spark.read.option("basePath", base).parquet(base)


# --------------------------
# Injection config builders
# --------------------------

@dataclass(frozen=True)
class InjectionConfig:
    config_id: str
    theta_e_arcsec: float
    src_dmag: float
    src_reff_arcsec: float
    src_e: float
    shear: float


def build_grid(name: str) -> List[InjectionConfig]:
    """Named grids used by Stage 4a. Keep these stable to avoid reruns."""
    if name == "grid_small":
        theta = [0.3, 0.6, 1.0]
        dmag = [1.0, 2.0]
        reff = [0.08, 0.15]
        e = [0.0, 0.3]
        shear = [0.0, 0.03]
    elif name == "grid_medium":
        theta = [0.25, 0.35, 0.5, 0.8, 1.2]
        dmag = [0.5, 1.0, 1.5, 2.0]
        reff = [0.05, 0.10, 0.20]
        e = [0.0, 0.2, 0.4]
        shear = [0.0, 0.03]
    elif name == "grid_full":
        # Larger grid, still bounded.
        theta = [0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 1.2, 1.6]
        dmag = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
        reff = [0.04, 0.06, 0.08, 0.12, 0.18, 0.25]
        e = [0.0, 0.2, 0.4]
        shear = [0.0, 0.02, 0.04]
    elif name == "grid_sota":
        # Extended theta_e grid for SOTA performance
        # Focus on resolvable lenses: theta_e >= 0.5 arcsec (typical PSF FWHM ~ 1 arcsec)
        # This ensures theta_e/PSF >= 0.5, making lenses morphologically detectable
        theta = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
        dmag = [0.5, 1.0, 1.5, 2.0]  # Source brightness relative to lens
        reff = [0.06, 0.10, 0.15, 0.20]  # Source effective radius in arcsec
        e = [0.0, 0.2, 0.4]  # Source ellipticity
        shear = [0.0, 0.02, 0.04]  # External shear
    else:
        raise ValueError(f"Unknown grid name: {name}")

    out: List[InjectionConfig] = []
    idx = 0
    for th in theta:
        for dm in dmag:
            for rf in reff:
                for ee in e:
                    for sh in shear:
                        idx += 1
                        cid = f"{name}_{idx:04d}_te{th:g}_dm{dm:g}_rf{rf:g}_e{ee:g}_sh{sh:g}"
                        out.append(InjectionConfig(cid, float(th), float(dm), float(rf), float(ee), float(sh)))
    return out


# --------------------------
# Imaging helpers
# --------------------------

PIX_SCALE_ARCSEC = 0.262

# =========================================================================
# FLUX UNIT CONSTANTS (Legacy Survey coadds are in nanomaggies)
# =========================================================================
# 1 nanomaggy = flux of a mag 22.5 source (AB zero-point)
# To convert magnitude to flux: flux_nMgy = 10^(-0.4 * (mag - 22.5))
# =========================================================================
AB_ZP_NMGY = 22.5  # AB zero-point for nanomaggies
PIPELINE_VERSION = "phase4_pipeline_v1"

# =========================================================================
# DR10 MASKBITS (for filtering bad pixels in SNR calculation)
# Reference: https://www.legacysurvey.org/dr10/bitmasks
# =========================================================================
MASKBITS_BAD = (
    (1 << 0)    # NPRIMARY - not primary brick area
    | (1 << 1)  # BRIGHT - bright star nearby
    | (1 << 2)  # SATUR_G - saturated in g-band
    | (1 << 3)  # SATUR_R - saturated in r-band
    | (1 << 4)  # SATUR_Z - saturated in z-band
    | (1 << 5)  # ALLMASK_G - any masking in g-band
    | (1 << 6)  # ALLMASK_R - any masking in r-band
    | (1 << 7)  # ALLMASK_Z - any masking in z-band
    | (1 << 10) # BAILOUT - no coverage or catastrophic failure
    | (1 << 11) # MEDIUM - medium-bright star nearby
)

# WISE bright-star mask bits (not "detected sources" - these are masking flags)
# Reference: https://www.legacysurvey.org/dr10/bitmasks
# These indicate pixels affected by WISE-detected bright stars, not WISE sources themselves
MASKBIT_WISEM1 = (1 << 8)   # 0x100 - WISE W1 (3.4μm) bright star mask
MASKBIT_WISEM2 = (1 << 9)   # 0x200 - WISE W2 (4.6μm) bright star mask
MASKBITS_WISE = MASKBIT_WISEM1 | MASKBIT_WISEM2

# R-band specific bad bits for arc_snr calculation (cleaner than all-band mask)
# Using r-band-specific bits avoids masking valid r pixels due to g/z issues
MASKBITS_BAD_COMMON = (
    (1 << 0)    # NPRIMARY
    | (1 << 1)  # BRIGHT
    | (1 << 10) # BAILOUT
    | (1 << 11) # MEDIUM
)
MASKBITS_BAD_R = MASKBITS_BAD_COMMON | (1 << 3) | (1 << 6)  # + SATUR_R + ALLMASK_R

# Default split seed - hardcoded for reproducibility
# CRITICAL: This must match across 4a reruns for consistent sampling
# Value 13 matches the original Phase 4a run (before explicit seed tracking)
DEFAULT_SPLIT_SEED = 13


def mag_to_nMgy(mag: float) -> float:
    """Convert AB magnitude to nanomaggies (Legacy Survey flux units)."""
    return float(10.0 ** (-0.4 * (mag - AB_ZP_NMGY)))


def nMgy_to_mag(nmgy: float) -> float:
    """Convert nanomaggies to AB magnitude."""
    if nmgy is None or nmgy <= 0 or not np.isfinite(nmgy):
        return np.nan
    return float(AB_ZP_NMGY - 2.5 * math.log10(nmgy))


# =========================================================================
# SERSIC PROFILE FUNCTIONS (Correct flux normalization)
# =========================================================================
# These functions implement proper Sersic profile normalization.
# CRITICAL: We normalize the UNLENSED source analytically, then let lensing
# naturally increase the observed flux (magnification). Never normalize the
# lensed image-plane sum to the source flux - that destroys magnification!
# =========================================================================

def sersic_bn(n: float) -> float:
    """Sersic b_n approximation (Ciotti & Bertin 1999)."""
    n = max(n, 1e-6)
    return 2.0 * n - (1.0 / 3.0) + (0.009876 / n)


def sersic_unit_total_flux(reff_arcsec: float, q: float, n: float = 1.0) -> float:
    """
    Compute total flux for a Sersic profile with I(Re) = 1 flux/arcsec^2.
    
    For Sersic profile: I(R) = I_e * exp(-b_n * ((R/Re)^(1/n) - 1))
    Total flux = 2*pi * q * Re^2 * n * exp(b_n) * Gamma(2n) / b_n^(2n)
    
    Args:
        reff_arcsec: Effective radius in arcsec
        q: Axis ratio (b/a), in (0, 1]
        n: Sersic index (n=1 for exponential, n=4 for de Vaucouleurs)
    
    Returns:
        Total flux in same units as I_e (flux/arcsec^2 * arcsec^2 = flux)
    """
    b = sersic_bn(n)
    # Use math.gamma for the gamma function
    gamma_2n = math.gamma(2.0 * n)
    return (2.0 * math.pi * q * (reff_arcsec ** 2) *
            n * math.exp(b) * gamma_2n / (b ** (2.0 * n)))


def sersic_profile_Ie1(beta_x: np.ndarray, beta_y: np.ndarray, 
                       reff_arcsec: float, q: float, phi_rad: float, 
                       n: float = 1.0) -> np.ndarray:
    """
    Evaluate Sersic surface brightness profile with I(Re) = 1.
    
    The profile is evaluated in the source plane (beta coordinates).
    Returns unnormalized surface brightness - must be scaled by amplitude.
    
    Args:
        beta_x, beta_y: Source-plane coordinates in arcsec
        reff_arcsec: Effective radius in arcsec
        q: Axis ratio (b/a)
        phi_rad: Position angle in radians
        n: Sersic index (default 1.0 = exponential)
    
    Returns:
        Surface brightness array with I(Re) = 1
    """
    # Rotate to align with major axis
    c = math.cos(phi_rad)
    s = math.sin(phi_rad)
    xp = c * beta_x + s * beta_y
    yp = -s * beta_x + c * beta_y
    
    # Elliptical radius
    q = float(np.clip(q, 0.1, 1.0))
    R = np.sqrt(xp**2 + (yp / q)**2 + 1e-18)
    
    # Sersic profile
    b = sersic_bn(n)
    reff = max(reff_arcsec, 1e-6)
    return np.exp(-b * ((R / reff) ** (1.0 / n) - 1.0)).astype(np.float32)


# =========================================================================
# DEFLECTION FUNCTIONS (Dependency-free SIE/SIS + Shear)
# =========================================================================
# These implement gravitational lensing deflection without external deps.
# SIE uses the analytic elliptical isothermal deflection formula.
# =========================================================================

def deflection_sis(x: np.ndarray, y: np.ndarray, theta_e: float, 
                   eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """SIS (Singular Isothermal Sphere) deflection."""
    r = np.sqrt(x * x + y * y) + eps
    ax = theta_e * x / r
    ay = theta_e * y / r
    return ax, ay


def deflection_sie(x: np.ndarray, y: np.ndarray, theta_e: float,
                   lens_e: float, lens_phi_rad: float,
                   eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    SIE (Singular Isothermal Ellipsoid) deflection.
    
    Uses the analytic SIE deflection formula (Kormann et al. 1994).
    Falls back to SIS if lens is nearly circular.
    
    Args:
        x, y: Image-plane coordinates in arcsec
        theta_e: Einstein radius in arcsec
        lens_e: Lens ellipticity (0 = circular, 0.5 = axis ratio 1:3)
        lens_phi_rad: Lens position angle in radians
        
    Returns:
        (alpha_x, alpha_y) deflection in arcsec
    """
    # Map ellipticity to axis ratio: q = (1-e)/(1+e)
    q = float(np.clip((1.0 - lens_e) / (1.0 + lens_e + 1e-6), 0.2, 0.999))
    
    # Rotate to lens frame
    c = math.cos(lens_phi_rad)
    s = math.sin(lens_phi_rad)
    xp = c * x + s * y
    yp = -s * x + c * y
    
    # If nearly circular, use SIS
    if 1.0 - q < 1e-4:
        axp, ayp = deflection_sis(xp, yp, theta_e, eps=eps)
    else:
        # SIE deflection formula
        fac = math.sqrt(max(1.0 - q * q, 1e-12))
        psi = np.sqrt((q * xp)**2 + yp**2) + eps
        
        axp = (theta_e / fac) * np.arctan((fac * xp) / psi)
        
        # Clip argument to avoid arctanh divergence
        arg = np.clip((fac * yp) / psi, -0.999999, 0.999999)
        ayp = (theta_e / fac) * np.arctanh(arg)
    
    # Rotate back
    ax = c * axp - s * ayp
    ay = s * axp + c * ayp
    return ax, ay


def deflection_shear(x: np.ndarray, y: np.ndarray, 
                     shear: float, shear_phi_rad: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    External shear deflection.
    
    Shear matrix: [[gamma1, gamma2], [gamma2, -gamma1]]
    where gamma1 = shear * cos(2*phi), gamma2 = shear * sin(2*phi)
    """
    g1 = float(shear) * math.cos(2.0 * float(shear_phi_rad))
    g2 = float(shear) * math.sin(2.0 * float(shear_phi_rad))
    ax = g1 * x + g2 * y
    ay = g2 * x - g1 * y
    return ax, ay


# =========================================================================
# MAIN RENDERING FUNCTION (Correct magnification behavior)
# =========================================================================

def render_lensed_source(
    stamp_size: int,
    pixscale_arcsec: float,
    lens_model: str,  # "SIS", "SIE", or "CONTROL"
    theta_e_arcsec: float,
    lens_e: float,
    lens_phi_rad: float,
    shear: float,
    shear_phi_rad: float,
    src_total_flux_nmgy: float,
    src_reff_arcsec: float,
    src_e: float,
    src_phi_rad: float,
    src_x_arcsec: float,
    src_y_arcsec: float,
    psf_fwhm_pix: float,
    psf_model: str = "gaussian",
    moffat_beta: float = 3.5,
    psf_apply: bool = True,
) -> np.ndarray:
    """
    Render a lensed source image with correct magnification behavior.
    
    CRITICAL: This function normalizes the UNLENSED source analytically,
    then applies lensing. The observed (lensed) flux will be HIGHER than
    the intrinsic source flux due to gravitational magnification.
    
    Args:
        stamp_size: Output stamp size in pixels
        pixscale_arcsec: Pixel scale in arcsec/pixel
        lens_model: "SIS", "SIE", or "CONTROL" (no lensing)
        theta_e_arcsec: Einstein radius in arcsec
        lens_e: Lens ellipticity (for SIE)
        lens_phi_rad: Lens position angle (for SIE)
        shear: External shear amplitude
        shear_phi_rad: External shear orientation
        src_total_flux_nmgy: Source intrinsic flux in nanomaggies
        src_reff_arcsec: Source effective radius in arcsec
        src_e: Source ellipticity
        src_phi_rad: Source position angle
        src_x_arcsec: Source x offset from lens center
        src_y_arcsec: Source y offset from lens center
        psf_fwhm_pix: PSF FWHM in pixels (0 to skip convolution)
        psf_apply: Whether to apply PSF convolution
        
    Returns:
        Float32 array of shape (stamp_size, stamp_size) in nMgy/pixel
    """
    half = stamp_size // 2
    coords = (np.arange(stamp_size) - half + 0.5) * pixscale_arcsec
    x, y = np.meshgrid(coords, coords)
    
    # Compute deflection based on lens model
    if lens_model == "CONTROL" or theta_e_arcsec <= 0:
        # No lensing - return zeros (control sample)
        return np.zeros((stamp_size, stamp_size), dtype=np.float32)
    
    if lens_model == "SIE":
        ax_l, ay_l = deflection_sie(x, y, theta_e_arcsec, lens_e, lens_phi_rad)
    else:  # SIS
        ax_l, ay_l = deflection_sis(x, y, theta_e_arcsec)
    
    # Add external shear deflection
    if abs(shear) > 1e-6:
        ax_s, ay_s = deflection_shear(x, y, shear, shear_phi_rad)
    else:
        ax_s, ay_s = 0.0, 0.0
    
    # Lens equation: beta = theta - alpha
    beta_x = x - ax_l - ax_s - src_x_arcsec
    beta_y = y - ay_l - ay_s - src_y_arcsec
    
    # Source axis ratio from ellipticity
    q_src = float(np.clip((1.0 - src_e) / (1.0 + src_e + 1e-6), 0.2, 0.999))
    
    # Evaluate Sersic profile (I(Re) = 1)
    base = sersic_profile_Ie1(beta_x, beta_y, src_reff_arcsec, q_src, src_phi_rad, n=1.0)
    
    # Compute amplitude to give correct INTRINSIC total flux
    # This is the key: we normalize the unlensed source, not the lensed image
    unit_flux = sersic_unit_total_flux(src_reff_arcsec, q_src, n=1.0) + 1e-30
    amp_Ie = src_total_flux_nmgy / unit_flux  # I_e in nMgy/arcsec^2
    
    # Pixel area for conversion to flux/pixel
    pix_area = pixscale_arcsec ** 2
    
    # Surface brightness in nMgy/pixel
    # Lensing magnification is implicit: more source-plane area maps to each pixel
    img = base * amp_Ie * pix_area
    
    # PSF convolution
    if psf_apply and psf_fwhm_pix > 0:
        img = _convolve_psf(img, psf_fwhm_pix, psf_model=psf_model, moffat_beta=moffat_beta)
    
    return img.astype(np.float32)


def render_unlensed_source(
    stamp_size: int,
    pixscale_arcsec: float,
    src_total_flux_nmgy: float,
    src_x_arcsec: float,
    src_y_arcsec: float,
    src_reff_arcsec: float,
    n: float,
    src_e: float,
    src_phi_rad: float,
    psf_apply: bool,
    psf_fwhm_pix: float,
    psf_model: str = "gaussian",
    moffat_beta: float = 3.5,
) -> np.ndarray:
    """Render an unlensed source on the same stamp grid.

    Returns an image in nanomaggies / pixel.

    This is used only for a stamp-limited flux-ratio magnification proxy. It MUST NOT be used
    to renormalize the lensed image.
    """
    half = stamp_size // 2
    ys, xs = np.mgrid[0:stamp_size, 0:stamp_size]
    theta_x = (xs - half + 0.5) * pixscale_arcsec
    theta_y = (ys - half + 0.5) * pixscale_arcsec

    dx = theta_x - src_x_arcsec
    dy = theta_y - src_y_arcsec

    # Compute axis ratio from ellipticity for consistent normalization
    # Must match the ellipticity used in sersic_profile() below
    q_src = (1.0 - src_e) / (1.0 + src_e)
    unit_flux = sersic_unit_total_flux(reff_arcsec=src_reff_arcsec, q=q_src, n=n)
    amp = float(src_total_flux_nmgy) / max(float(unit_flux), 1e-30)

    img = amp * sersic_profile(dx, dy, src_reff_arcsec, n, src_e, src_phi_rad)
    img = img * (pixscale_arcsec ** 2)

    if psf_apply and psf_fwhm_pix > 0:
        img = _convolve_psf(img, psf_fwhm_pix, psf_model=psf_model, moffat_beta=moffat_beta)

    return img.astype(np.float32)

def _gaussian_kernel1d(sigma_pix: float, radius: int = 4) -> np.ndarray:
    sigma = max(float(sigma_pix), 1e-3)
    r = int(max(radius, math.ceil(4 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= np.sum(k)
    return k


def _convolve_gaussian(img: np.ndarray, sigma_pix: float) -> np.ndarray:
    k = _gaussian_kernel1d(sigma_pix)
    # Separable conv, reflect padding
    pad = len(k) // 2
    tmp = np.pad(img, ((0, 0), (pad, pad)), mode="reflect")
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), axis=1, arr=tmp)
    tmp = np.pad(tmp, ((pad, pad), (0, 0)), mode="reflect")
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), axis=0, arr=tmp)
    return out.astype(np.float32)



# =========================================================================
# PSF KERNELS AND CONVOLUTION
# =========================================================================

def _gaussian_kernel2d(sigma_pix: float, radius: Optional[int] = None) -> np.ndarray:
    """Build a normalized circular Gaussian PSF kernel."""
    sigma_pix = float(sigma_pix)
    if sigma_pix <= 0:
        k = np.zeros((1, 1), dtype=np.float32)
        k[0, 0] = 1.0
        return k
    if radius is None:
        # 4 sigma captures >99% of Gaussian mass
        radius = int(max(3, math.ceil(4.0 * sigma_pix)))
    side = 2 * radius + 1
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    rr2 = (xx * xx + yy * yy).astype(np.float32)
    k = np.exp(-0.5 * rr2 / (sigma_pix * sigma_pix)).astype(np.float32)
    s = float(k.sum()) + 1e-12
    k /= s
    return k


def _moffat_kernel2d(fwhm_pix: float, beta: float = 3.5, radius: Optional[int] = None) -> np.ndarray:
    """Build a normalized Moffat PSF kernel.

    Moffat profile: I(r) = (1 + (r/alpha)^2)^(-beta)
    Relation between FWHM and alpha:
      alpha = fwhm / (2 * sqrt(2^(1/beta) - 1))
    """
    fwhm_pix = float(fwhm_pix)
    beta = float(beta)
    if fwhm_pix <= 0:
        k = np.zeros((1, 1), dtype=np.float32)
        k[0, 0] = 1.0
        return k
    if beta <= 1.0:
        beta = 1.0001  # avoid infinite variance behavior
    alpha = fwhm_pix / (2.0 * math.sqrt((2.0 ** (1.0 / beta)) - 1.0))
    if radius is None:
        # Wider truncation than Gaussian due to wings
        radius = int(max(4, math.ceil(6.0 * fwhm_pix)))
    side = 2 * radius + 1
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    rr2 = (xx * xx + yy * yy).astype(np.float32)
    k = (1.0 + rr2 / (alpha * alpha)).astype(np.float32)
    k = np.power(k, -beta).astype(np.float32)
    s = float(k.sum()) + 1e-12
    k /= s
    return k


def _fft_convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """FFT-based convolution for small stamps (e.g., 64x64)."""
    ih, iw = img.shape
    kh, kw = kernel.shape
    if kh > ih or kw > iw:
        raise ValueError(f"Kernel {kernel.shape} larger than image {img.shape}")
    pad = np.zeros((ih, iw), dtype=np.float32)
    pad[:kh, :kw] = kernel.astype(np.float32)
    pad = np.fft.ifftshift(pad)  # move kernel center to (0,0)
    out = np.fft.ifft2(np.fft.fft2(img.astype(np.float32)) * np.fft.fft2(pad)).real
    return out.astype(np.float32)


def _convolve_psf(img: np.ndarray, psf_fwhm_pix: float, psf_model: str = "gaussian", moffat_beta: float = 3.5) -> np.ndarray:
    """Convolve with a PSF kernel defined by model and FWHM."""
    psf_model = (psf_model or "gaussian").lower()
    psf_fwhm_pix = float(psf_fwhm_pix)
    if psf_fwhm_pix <= 0:
        return img.astype(np.float32)

    if psf_model == "gaussian":
        sigma = psf_fwhm_pix / 2.355
        k = _gaussian_kernel2d(sigma)
    elif psf_model == "moffat":
        k = _moffat_kernel2d(psf_fwhm_pix, beta=moffat_beta)
    else:
        raise ValueError(f"Unknown psf_model={psf_model}. Use gaussian or moffat.")

    return _fft_convolve2d(img.astype(np.float32), k)
def _elliptical_gaussian(beta_x: np.ndarray, beta_y: np.ndarray, reff_pix: float, e: float) -> np.ndarray:
    # Ellipticity parameter e in [0,1). Use axis ratio q=(1-e)/(1+e)
    q = (1.0 - e) / (1.0 + e + 1e-6)
    sigx = reff_pix
    sigy = reff_pix * q
    sigx = max(sigx, 0.5)
    sigy = max(sigy, 0.5)
    return np.exp(-0.5 * ((beta_x / sigx) ** 2 + (beta_y / sigy) ** 2)).astype(np.float32)


def inject_sis_stamp(
    stamp_shape: Tuple[int, int],
    theta_e_arcsec: float,
    src_total_flux: float,
    src_reff_arcsec: float,
    src_e: float,
    shear: float,
    rng: np.random.RandomState,
    psf_fwhm_arcsec: Optional[float] = None,
    shear_phi: Optional[float] = None,
) -> np.ndarray:
    """
    DEPRECATED: Use render_lensed_source() instead.
    
    This function has a critical bug: it normalizes the LENSED image to the
    source flux, which destroys magnification. Lensed arcs should be brighter
    than the unlensed source due to gravitational magnification.
    
    Kept for backward compatibility only. Do not use for new code.
    
    Args:
        shear_phi: Shear orientation in radians. If None, randomly generated.
    """
    import warnings
    warnings.warn(
        "inject_sis_stamp is deprecated and has incorrect flux normalization. "
        "Use render_lensed_source() instead.", 
        DeprecationWarning, 
        stacklevel=2
    )
    h, w = stamp_shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    # Pixel coordinates in arcsec relative to center
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    thx = (xx - cx) * PIX_SCALE_ARCSEC
    thy = (yy - cy) * PIX_SCALE_ARCSEC

    # Compute shear components with random orientation if not provided
    if shear_phi is None:
        shear_phi = rng.uniform(0, np.pi)
    
    # =========================================================================
    # PROPER SIS + EXTERNAL SHEAR LENS EQUATION
    # =========================================================================
    # The lens equation: beta = theta - alpha(theta) - Gamma * theta
    # where alpha is the SIS deflection and Gamma is the shear matrix:
    #   Gamma = [[gamma1, gamma2], [gamma2, -gamma1]]
    # =========================================================================
    
    # SIS deflection
    eps = 1e-10
    r = np.sqrt(thx**2 + thy**2) + eps
    alpha_x = theta_e_arcsec * thx / r
    alpha_y = theta_e_arcsec * thy / r
    
    # Shear components
    if abs(shear) > 0:
        g1 = shear * np.cos(2.0 * shear_phi)
        g2 = shear * np.sin(2.0 * shear_phi)
        # Apply shear matrix: [[g1, g2], [g2, -g1]]
        betax = thx - alpha_x - (g1 * thx + g2 * thy)
        betay = thy - alpha_y - (g2 * thx - g1 * thy)
    else:
        betax = thx - alpha_x
        betay = thy - alpha_y

    # Random source offset within +-0.4 arcsec
    offx = rng.uniform(-0.4, 0.4)
    offy = rng.uniform(-0.4, 0.4)
    betax -= offx
    betay -= offy

    reff_pix = src_reff_arcsec / PIX_SCALE_ARCSEC
    src = _elliptical_gaussian(betax / PIX_SCALE_ARCSEC, betay / PIX_SCALE_ARCSEC, reff_pix, src_e)

    # Normalize to total flux
    ssum = float(np.sum(src)) + 1e-12
    src *= float(src_total_flux) / ssum

    # PSF blur approximation
    if psf_fwhm_arcsec is not None and psf_fwhm_arcsec > 0:
        sigma_pix = (psf_fwhm_arcsec / 2.355) / PIX_SCALE_ARCSEC
        src = _convolve_gaussian(src, sigma_pix)

    return src.astype(np.float32)


# =========================================================================
# SHEAR COMPONENTS HELPER
# =========================================================================
# External shear has amplitude gamma and orientation phi.
# The shear components are: gamma1 = gamma * cos(2*phi), gamma2 = gamma * sin(2*phi)
# Previously gamma2 was hardcoded to 0, which is non-physical.
# =========================================================================

def _shear_components(shear: float, shear_phi: float) -> Tuple[float, float]:
    """
    Convert shear amplitude and orientation to (gamma1, gamma2) components.
    
    Args:
        shear: Shear amplitude (typically 0.0-0.2)
        shear_phi: Shear orientation angle in radians [0, pi)
        
    Returns:
        Tuple of (gamma1, gamma2) shear components
    """
    g1 = float(shear) * math.cos(2.0 * float(shear_phi))
    g2 = float(shear) * math.sin(2.0 * float(shear_phi))
    return float(g1), float(g2)


# =========================================================================
# SIE (SINGULAR ISOTHERMAL ELLIPSOID) INJECTION USING LENSTRONOMY
# =========================================================================
# This is a more realistic lens model than SIS, including:
# - Elliptical mass distribution (not just circular)
# - Proper ray tracing via lenstronomy
# - Sersic source profile (more realistic than Gaussian)
# - PSF convolution using actual PSF model
#
# Falls back to SIS if lenstronomy is not installed.
# =========================================================================

def inject_sie_stamp(
    stamp_shape: Tuple[int, int],
    theta_e_arcsec: float,
    src_total_flux: float,
    src_reff_arcsec: float,
    src_e: float,  # Source ellipticity (0-1)
    lens_e: float,  # Lens ellipticity (0-1)
    shear: float,
    rng: np.random.RandomState,
    psf_fwhm_arcsec: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    DEPRECATED: Use render_lensed_source() instead.
    
    This function requires lenstronomy and has been superseded by the
    dependency-free render_lensed_source() which implements SIE natively.
    
    Returns:
        Tuple of (stamp array, physics_metrics dict)
        
    Physics metrics include:
        - magnification: Total magnification of the source
        - tangential_stretch: Tangential eigenvalue of magnification tensor
        - radial_stretch: Radial eigenvalue
        - expected_arc_radius: Approximate arc radius in arcsec
    """
    import warnings
    warnings.warn(
        "inject_sie_stamp is deprecated. Use render_lensed_source() instead, "
        "which has dependency-free SIE implementation.", 
        DeprecationWarning, 
        stacklevel=2
    )
    if not LENSTRONOMY_AVAILABLE:
        # Fallback to SIS with empty physics metrics
        stamp = inject_sis_stamp(
            stamp_shape, theta_e_arcsec, src_total_flux, 
            src_reff_arcsec, src_e, shear, rng, psf_fwhm_arcsec
        )
        return stamp, {"magnification": None, "tangential_stretch": None, 
                       "radial_stretch": None, "expected_arc_radius": None}
    
    h, w = stamp_shape
    
    # Random source position offset (within caustic region for lensing)
    # Keep source within ~0.5 * theta_e to ensure strong lensing
    max_offset = min(0.4, 0.5 * theta_e_arcsec)
    src_x = rng.uniform(-max_offset, max_offset)
    src_y = rng.uniform(-max_offset, max_offset)
    
    # Convert ellipticity to lenstronomy format (e1, e2)
    phi_lens = rng.uniform(0, np.pi)  # Random orientation
    e1_lens = lens_e * np.cos(2 * phi_lens)
    e2_lens = lens_e * np.sin(2 * phi_lens)
    
    phi_src = rng.uniform(0, np.pi)
    e1_src = src_e * np.cos(2 * phi_src)
    e2_src = src_e * np.sin(2 * phi_src)
    
    # Setup lens model: SIE + external shear
    lens_model_list = ['SIE', 'SHEAR']
    lens_model = LensModel(lens_model_list)
    
    # FIX: Use random shear orientation instead of fixed gamma2=0
    # Shear orientation should be random for realistic simulations
    shear_phi = rng.uniform(0, np.pi)  # Random shear orientation [0, pi)
    gamma1, gamma2 = _shear_components(shear, shear_phi)
    
    kwargs_lens = [
        {'theta_E': theta_e_arcsec, 'e1': e1_lens, 'e2': e2_lens, 'center_x': 0, 'center_y': 0},
        {'gamma1': gamma1, 'gamma2': gamma2, 'ra_0': 0, 'dec_0': 0}
    ]
    
    # Setup source model: Sersic profile
    light_model_list = ['SERSIC_ELLIPSE']
    light_model = LightModel(light_model_list)
    
    # Convert flux to amplitude (approximate)
    # Sersic total flux ~ 2 * pi * n * exp(b_n) * Gamma(2n) * R_eff^2 * I_eff / b_n^(2n)
    # For n=1 (exponential), this simplifies
    amplitude = src_total_flux / (2 * np.pi * src_reff_arcsec**2 + 1e-10)
    
    kwargs_source = [{
        'amp': amplitude,
        'R_sersic': src_reff_arcsec,
        'n_sersic': 1.0,  # Exponential profile
        'e1': e1_src,
        'e2': e2_src,
        'center_x': src_x,
        'center_y': src_y,
    }]
    
    # Setup image data
    kwargs_data = {
        'image_data': np.zeros((h, w)),
        'transform_pix2angle': np.array([[PIX_SCALE_ARCSEC, 0], [0, PIX_SCALE_ARCSEC]]),
        'ra_at_xy_0': -(w - 1) / 2.0 * PIX_SCALE_ARCSEC,
        'dec_at_xy_0': -(h - 1) / 2.0 * PIX_SCALE_ARCSEC,
    }
    data_class = ImageData(**kwargs_data)
    
    # Setup PSF
    if psf_fwhm_arcsec is not None and psf_fwhm_arcsec > 0:
        psf_class = PSF(psf_type='GAUSSIAN', fwhm=psf_fwhm_arcsec)
    else:
        psf_class = PSF(psf_type='NONE')
    
    # Create image model and generate lensed image
    image_model = ImageModel(data_class, psf_class, lens_model, light_model)
    stamp = image_model.image(kwargs_lens, kwargs_source).astype(np.float32)
    
    # =========================================================================
    # PHYSICS-BASED VALIDATION METRICS
    # =========================================================================
    # These metrics can be used downstream to verify physical consistency:
    # 1. Magnification should be > 1 for strong lensing
    # 2. Arc radius should be approximately theta_e (Einstein radius)
    # 3. Tangential stretch should be > radial stretch near Einstein ring
    #
    # FIX (2026-01-22): Evaluate at IMAGE-PLANE position, not source-plane.
    # For SIE, the Einstein ring is at radius theta_e from lens center.
    # We evaluate at (theta_e, 0) as a representative image-plane point.
    # =========================================================================
    
    # Compute magnification at IMAGE-PLANE position near Einstein ring
    # Use a point slightly OUTSIDE the Einstein radius to avoid critical curve singularity
    # At r = theta_e, magnification diverges. At r = 1.1 * theta_e, we get a stable proxy.
    try:
        # Image-plane position slightly outside Einstein radius to avoid singularity
        image_x = 1.1 * theta_e_arcsec  # 10% outside Einstein radius
        image_y = 0.0  # On the x-axis
        
        det_A = lens_model.hessian(image_x, image_y, kwargs_lens)
        # det_A returns (f_xx, f_xy, f_yx, f_yy)
        f_xx, f_xy, f_yx, f_yy = det_A
        kappa = 0.5 * (f_xx + f_yy)  # Convergence
        gamma1 = 0.5 * (f_xx - f_yy)  # Shear component 1
        gamma2 = f_xy  # Shear component 2
        gamma = np.sqrt(gamma1**2 + gamma2**2)
        
        # Eigenvalues of magnification tensor
        # At the Einstein ring, tangential eigenvalue diverges (critical curve)
        tangential = 1 / (1 - kappa - gamma + 1e-10)  # Add small epsilon to avoid div by 0
        radial = 1 / (1 - kappa + gamma + 1e-10)
        magnification = abs(tangential * radial)
        
        # Cap magnification to avoid infinities near critical curve
        magnification = min(magnification, 1000.0)
        
        # Expected arc radius ~ theta_e for SIE
        expected_arc_radius = theta_e_arcsec
        
        physics_metrics = {
            "magnification": float(magnification) if np.isfinite(magnification) else None,
            "tangential_stretch": float(abs(tangential)) if np.isfinite(tangential) else None,
            "radial_stretch": float(abs(radial)) if np.isfinite(radial) else None,
            "expected_arc_radius": float(expected_arc_radius),
        }
    except Exception:
        physics_metrics = {
            "magnification": None, "tangential_stretch": None,
            "radial_stretch": None, "expected_arc_radius": theta_e_arcsec,
        }
    
    return stamp, physics_metrics


def validate_physics_consistency(
    theta_e_arcsec: float,
    arc_snr: Optional[float],
    magnification: Optional[float],
    tangential_stretch: Optional[float],
    psfsize_r: Optional[float],
) -> Tuple[bool, List[str]]:
    """
    Validate that injected lens parameters are physically consistent.
    
    =========================================================================
    PHYSICS VALIDATION RULES:
    =========================================================================
    1. DETECTABILITY: Arc should be detectable only if theta_e > 0.5 * PSF FWHM
       - Below this, lensing signal is unresolved
    
    2. MAGNIFICATION: Strong lensing requires magnification > 1
       - Typical values for galaxy-scale lenses: 2-100
       
    3. TANGENTIAL STRETCH: Should be > 1 for arc formation
       - This creates the elongated arc morphology
       
    4. SNR CONSISTENCY: High theta_e with good seeing should give high SNR
       - Flags potential injection bugs if this fails
    =========================================================================
    
    Returns:
        Tuple of (is_valid, list of warning messages)
    """
    warnings = []
    is_valid = True
    
    # Rule 1: Detectability threshold
    if psfsize_r is not None and theta_e_arcsec > 0:
        theta_over_psf = theta_e_arcsec / psfsize_r
        if theta_over_psf < 0.5:
            warnings.append(f"theta_e ({theta_e_arcsec:.2f}) < 0.5 * PSF ({psfsize_r:.2f}): unresolved")
    
    # Rule 2: Magnification sanity check (do not assume >1: demagnification is possible)
    if magnification is not None:
        if (not np.isfinite(magnification)) or magnification <= 0.0:
            warnings.append(f"Magnification ({magnification}): non-finite or <= 0")
            is_valid = False
        elif magnification > 1000:
            warnings.append(f"Magnification ({magnification:.2f}) > 1000: near caustic, may be unrealistic")
    
    # Rule 3: Tangential stretch sanity check
    if tangential_stretch is not None:
        if (not np.isfinite(tangential_stretch)) or tangential_stretch <= 0.0:
            warnings.append(f"Tangential stretch ({tangential_stretch}): non-finite or <= 0")
    
    # Rule 4: SNR consistency (heuristic)
    if arc_snr is not None and theta_e_arcsec > 0 and psfsize_r is not None:
        # Expect SNR to scale roughly with (theta_e / psf)^2
        expected_min_snr = 2.0 * (theta_e_arcsec / psfsize_r) ** 2
        if arc_snr < expected_min_snr * 0.1:  # Allow 10× variation
            warnings.append(f"SNR ({arc_snr:.1f}) much lower than expected (~{expected_min_snr:.1f})")
    
    return is_valid, warnings


def encode_npz(arrs: Dict[str, np.ndarray]) -> bytes:
    bio = io.BytesIO()
    np.savez_compressed(bio, **arrs)
    return bio.getvalue()


# --------------------------
# Coadd URL builder
# --------------------------

def build_coadd_urls(coadd_base_url: str, brickname: str, bands: List[str], 
                      include_psfsize: bool = False) -> Dict[str, str]:
    """Return URLs for image/invvar/maskbits (and optionally psfsize maps) for each band.
    
    Args:
        coadd_base_url: Base URL for DR10 coadds (e.g., https://portal.nersc.gov/.../coadd)
        brickname: Brick name (e.g., "0001m002")
        bands: List of bands (e.g., ["g", "r", "z"])
        include_psfsize: If True, also include per-pixel PSF FWHM maps for each band
        
    Returns:
        Dict mapping file keys to URLs:
        - image_{band}: Flux coadd
        - invvar_{band}: Inverse variance
        - maskbits: Bad pixel mask
        - psfsize_{band}: (if include_psfsize) Per-pixel PSF FWHM map
        
    Note:
        DR10 provides psfsize-*.fits.fz files (per-pixel PSF FWHM maps),
        NOT psfex files. PSFEx models are not available in the coadd directory.
    """
    # DR10 structure: .../dr10/south/coadd/000/000p025/legacysurvey-000p025-image-g.fits.fz
    # with directory coadd/<brickname[:3]>/<brickname>/
    p3 = brickname[:3]
    base = coadd_base_url.rstrip("/")
    d = f"{base}/{p3}/{brickname}"
    out: Dict[str, str] = {}
    for b in bands:
        out[f"image_{b}"] = f"{d}/legacysurvey-{brickname}-image-{b}.fits.fz"
        out[f"invvar_{b}"] = f"{d}/legacysurvey-{brickname}-invvar-{b}.fits.fz"
        if include_psfsize:
            # psfsize maps contain per-pixel PSF FWHM in arcsec
            out[f"psfsize_{b}"] = f"{d}/legacysurvey-{brickname}-psfsize-{b}.fits.fz"
    out["maskbits"] = f"{d}/legacysurvey-{brickname}-maskbits.fits.fz"
    return out


# --------------------------
# Stage 4a: Build manifests
# --------------------------

def stage_4a_build_manifests(spark: SparkSession, args: argparse.Namespace) -> None:
    out_root = f"{args.output_s3.rstrip('/')}/phase4a/{args.variant}"
    # NOTE: Per-experiment skip checks happen inside the loop below.
    # Do NOT skip the entire stage here - we need to check each experiment individually.

    parent_path = args.parent_s3
    bricks_path = args.bricks_with_region_s3
    selections_path = args.region_selections_s3

    df_parent = read_parquet_safe(spark, parent_path)

    # Expected parent fields (minimal)
    needed_cols = [
        "brickname", "ra", "dec", "region_id", "region_split",
        "zmag", "rmag", "w1mag", "rz", "zw1",
        "is_v3_color_relaxed",
    ]
    for c in needed_cols:
        if c not in df_parent.columns:
            raise RuntimeError(f"Parent catalog missing required column: {c}")

    df_parent = df_parent.select(*needed_cols)

    # Bricks-with-region provides observing conditions per brick
    df_bricks = read_parquet_safe(spark, bricks_path)
    # Required columns
    for c in ["brickname", "psfsize_r", "psfdepth_r", "ebv"]:
        if c not in df_bricks.columns:
            raise RuntimeError(f"bricks_with_region missing required column: {c}")
    
    # Per-band PSF columns (g, r, z) for realistic injection
    # psfsize_r is required; psfsize_g and psfsize_z fallback to psfsize_r if missing
    df_bricks = df_bricks.select("brickname", "psfsize_r", "psfdepth_r", "ebv",
        *[c for c in ["psfsize_g", "psfsize_z"] if c in df_bricks.columns]
    )
    
    # Add fallback columns if missing (use psfsize_r as approximation)
    if "psfsize_g" not in df_bricks.columns:
        print("[4a] Warning: psfsize_g not found in bricks_with_region, using psfsize_r as fallback")
        df_bricks = df_bricks.withColumn("psfsize_g", F.col("psfsize_r"))
    if "psfsize_z" not in df_bricks.columns:
        print("[4a] Warning: psfsize_z not found in bricks_with_region, using psfsize_r as fallback")
        df_bricks = df_bricks.withColumn("psfsize_z", F.col("psfsize_r"))
    df_parent = df_parent.join(df_bricks, on="brickname", how="left")

    # Region selections (3b)
    df_sel = read_parquet_safe(spark, selections_path)
    for c in ["selection_set_id", "selection_strategy", "ranking_mode", "region_id", "region_split"]:
        if c not in df_sel.columns:
            raise RuntimeError(f"region_selections missing required column: {c}")

    df_sel = df_sel.select(
        "selection_set_id", "selection_strategy", "ranking_mode", "region_id"
    ).dropDuplicates()

    # Attach selection_set_id by joining on region_id (small table)
    df = df_parent.join(F.broadcast(df_sel), on="region_id", how="inner")

    # Filter to the requested variant objects by default
    if args.require_lrg_flag:
        df = df.filter(F.col(args.require_lrg_flag) == F.lit(1))

    # Optional selection set filtering
    if args.selection_set_ids and args.selection_set_ids != "*":
        wanted = [x.strip() for x in args.selection_set_ids.split(",") if x.strip()]
        df = df.filter(F.col("selection_set_id").isin(wanted))

    # Compute psf/depth quantile edges once (on bricks table)
    q = [0.0, 0.25, 0.5, 0.75, 1.0]
    psf_edges = df_bricks.approxQuantile("psfsize_r", q, 0.001)
    depth_edges = df_bricks.approxQuantile("psfdepth_r", q, 0.001)

    def _bucket(col: F.Column, edges: List[float], name: str) -> F.Column:
        # returns 0..len(edges)-2
        expr = F.lit(len(edges) - 2)
        for i in range(len(edges) - 1):
            lo = edges[i]
            hi = edges[i + 1]
            if i == len(edges) - 2:
                expr = F.when((col >= lo) & (col <= hi), F.lit(i)).otherwise(expr)
            else:
                expr = F.when((col >= lo) & (col < hi), F.lit(i)).otherwise(expr)
        return expr.alias(name)

    df = df.withColumn("psf_bin", _bucket(F.col("psfsize_r"), psf_edges, "psf_bin"))
    df = df.withColumn("depth_bin", _bucket(F.col("psfdepth_r"), depth_edges, "depth_bin"))

    stamp_sizes = [int(x) for x in args.stamp_sizes.split(",") if x.strip()]
    bandsets = [x.strip() for x in args.bandsets.split(",") if x.strip()]

    tiers = [x.strip() for x in args.tiers.split(",") if x.strip()]

    # Pre-build injection grids used by tiers
    grids: Dict[str, List[InjectionConfig]] = {}
    for gname in set([args.grid_debug, args.grid_grid, args.grid_train]):
        grids[gname] = build_grid(gname)

    # Create Spark DataFrames for each grid
    grid_dfs: Dict[str, 'pyspark.sql.DataFrame'] = {}
    grid_schema = T.StructType([
        T.StructField("config_id", T.StringType(), False),
        T.StructField("theta_e_arcsec", T.DoubleType(), False),
        T.StructField("src_dmag", T.DoubleType(), False),
        T.StructField("src_reff_arcsec", T.DoubleType(), False),
        T.StructField("src_e", T.DoubleType(), False),
        T.StructField("shear", T.DoubleType(), False),
    ])
    for gname, cfgs in grids.items():
        rows = [Row(
            config_id=c.config_id,
            theta_e_arcsec=c.theta_e_arcsec,
            src_dmag=c.src_dmag,
            src_reff_arcsec=c.src_reff_arcsec,
            src_e=c.src_e,
            shear=c.shear,
        ) for c in cfgs]
        grid_dfs[gname] = spark.createDataFrame(rows, schema=grid_schema)

    # =========================================================================
    # TASK COUNT GUARDRAIL
    # =========================================================================
    # Estimate total tasks to catch accidental explosions before wasting compute.
    # This is a soft limit - set --max-total-tasks-soft to 0 to disable.
    # =========================================================================
    if args.max_total_tasks_soft and args.max_total_tasks_soft > 0:
        n_sel_sets = df.select("selection_set_id").distinct().count()
        n_splits = df.select("region_split").distinct().count()
        reps = max(1, args.replicates)
        
        est_total = 0
        for tier in tiers:
            if tier == "debug":
                n_cfg = len(grids[args.grid_debug])
                n_per_cfg = args.n_per_config_debug
                est_tier = n_per_cfg * n_cfg * n_sel_sets * n_splits * reps * len(bandsets) * len(stamp_sizes)
            elif tier == "grid":
                n_cfg = len(grids[args.grid_grid])
                n_per_cfg = args.n_per_config_grid
                est_tier = n_per_cfg * n_cfg * n_sel_sets * n_splits * reps * len(bandsets) * len(stamp_sizes)
            elif tier == "train":
                est_tier = args.n_total_train_per_split * n_sel_sets * n_splits * reps * len(bandsets) * len(stamp_sizes)
            else:
                est_tier = 0
            est_total += est_tier
        
        if est_total > args.max_total_tasks_soft:
            raise RuntimeError(
                f"[GUARDRAIL] Estimated total tasks ({est_total:,}) exceeds max_total_tasks_soft "
                f"({args.max_total_tasks_soft:,}). Reduce sampling parameters or increase "
                f"--max-total-tasks-soft if this is intentional."
            )
        print(f"[4a] Guardrail passed: estimated {est_total:,} tasks (limit: {args.max_total_tasks_soft:,})")

    # Stage config record
    stage_cfg = {
        "stage": "4a",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variant": args.variant,
        "output_s3": args.output_s3,
        "inputs": {
            "parent_s3": parent_path,
            "bricks_with_region_s3": bricks_path,
            "region_selections_s3": selections_path,
        },
        "seeds": {
            "split_seed": int(args.split_seed),
        },
        "sampling": {
            "selection_set_ids": args.selection_set_ids,
            "require_lrg_flag": args.require_lrg_flag,
            "psf_edges": psf_edges,
            "depth_edges": depth_edges,
        },
        "tiers": {
            "debug": {"grid": args.grid_debug, "n_per_config": args.n_per_config_debug, "control_frac": args.control_frac_debug},
            "grid": {"grid": args.grid_grid, "n_per_config": args.n_per_config_grid, "control_frac": args.control_frac_grid},
            "train": {"grid": args.grid_train, "n_total_per_split": args.n_total_train_per_split, "control_frac": args.control_frac_train},
        },
        "stamp_sizes": stamp_sizes,
        "bandsets": bandsets,
        "replicates": args.replicates,
        "idempotency": {"skip_if_exists": int(args.skip_if_exists), "force": int(args.force)},
    }

    # Write stage config json
    write_text_to_s3(f"{out_root}/_stage_config.json", json.dumps(stage_cfg, indent=2))

    manifests_out = f"{out_root}/manifests"

    all_manifest_paths: List[str] = []

    for stamp in stamp_sizes:
        for bandset in bandsets:
            if bandset not in ("r", "grz"):
                raise ValueError(f"Unsupported bandset: {bandset} (use r or grz)")

            for tier in tiers:
                if tier == "debug":
                    grid_name = args.grid_debug
                    n_per_config = int(args.n_per_config_debug)
                    n_total = None
                    control_frac = float(args.control_frac_debug)
                elif tier == "grid":
                    grid_name = args.grid_grid
                    n_per_config = int(args.n_per_config_grid)
                    n_total = None
                    control_frac = float(args.control_frac_grid)
                elif tier == "train":
                    grid_name = args.grid_train
                    n_total = int(args.n_total_train_per_split)
                    n_per_config = None
                    control_frac = float(args.control_frac_train)
                else:
                    raise ValueError(f"Unknown tier: {tier}")

                exp_id = f"{tier}_stamp{stamp}_bands{bandset}_grid{grid_name}"

                # Select base objects for this experiment
                base = df

                # Basic sanity filters
                base = base.filter(F.col("psfsize_r").isNotNull() & F.col("psfdepth_r").isNotNull())

                # Add a stable row id for hashing
                base = base.withColumn("row_id", F.xxhash64("brickname", "ra", "dec", "zmag", "rmag"))

                if tier in ("debug", "grid"):
                    # =========================================================================
                    # DEBUG/GRID TIER SAMPLING LOGIC
                    # =========================================================================
                    # Goal: For each (selection_set, split), sample n_per_config galaxies,
                    # then crossJoin with n_cfg configs to get n_per_config × n_cfg tasks.
                    #
                    # IMPORTANT: We sample n_per_config galaxies (NOT n_per_config × n_cfg).
                    # The crossJoin at the end multiplies by n_cfg to get the final task count.
                    #
                    # Example with n_per_config=40, n_cfg=360:
                    #   - Sample 40 galaxies per (selection_set, split)
                    #   - CrossJoin with 360 configs → 40 × 360 = 14,400 tasks per (set, split)
                    #   - With 12 sets × 3 splits × 2 replicates → ~1.04M total tasks
                    #
                    # BUG FIX (2026-01-22): Previously this was n_per_config × n_cfg, which
                    # caused O(n²) scaling: sampling 14,400 galaxies then crossJoining with
                    # 360 configs gave 5.18M tasks per (set, split) → 325M total. Fixed below.
                    # =========================================================================
                    
                    cfg_df = grid_dfs[grid_name]
                    n_cfg = cfg_df.count()

                    # Sample n_per_config galaxies per (selection_set, split).
                    # The crossJoin below will multiply this by n_cfg to get final task count.
                    # DO NOT multiply by n_cfg here - that was the O(n²) bug!
                    per_split_target = int(n_per_config)  # FIX: was n_per_config * n_cfg (wrong!)
                    
                    # Stratified sampling across PSF/depth bins (4×4 = 16 bins)
                    bins = 16
                    per_bin = int(math.ceil(per_split_target / bins))

                    # First pass: sample evenly across PSF/depth bins within each (selection_set, split)
                    wbin = Window.partitionBy("selection_set_id", "region_split", "psf_bin", "depth_bin").orderBy(F.xxhash64(F.col("row_id"), F.lit(int(args.split_seed))))
                    tmp = base.withColumn("rn_bin", F.row_number().over(wbin))
                    tmp = tmp.filter(F.col("rn_bin") <= F.lit(per_bin))

                    # Second pass: cap to exact per_split_target per (selection_set, split)
                    wtot = Window.partitionBy("selection_set_id", "region_split").orderBy(F.xxhash64(F.col("row_id"), F.lit(int(args.split_seed) + 1)))
                    tmp = tmp.withColumn("rn_tot", F.row_number().over(wtot))
                    tmp = tmp.filter(F.col("rn_tot") <= F.lit(per_split_target))

                    # CrossJoin: each selected galaxy gets every config from the grid.
                    # This is where the n_cfg multiplication happens (correctly).
                    # Final tasks per (selection_set, split) = per_split_target × n_cfg
                    tasks = tmp.crossJoin(F.broadcast(cfg_df))
                    
                    # =========================================================================
                    # CONTROL SAMPLES FOR DEBUG/GRID TIERS
                    # =========================================================================
                    # Control samples (is_control=1) have theta_e=0 and serve as negatives.
                    # Use deterministic hashing for reproducible control assignment.
                    # =========================================================================
                    if control_frac > 0:
                        # Deterministic control assignment using xxhash64
                        ctrl_hash = F.xxhash64(
                            F.col("brickname"),
                            F.round(F.col("ra") * F.lit(1e6)).cast("long"),
                            F.round(F.col("dec") * F.lit(1e6)).cast("long"),
                            F.col("selection_set_id").cast("string"),
                            F.col("region_split").cast("string"),
                            F.lit(tier),
                            F.lit(int(args.split_seed) + 7001),
                        )
                        ctrl_u = (F.pmod(F.abs(ctrl_hash), F.lit(1_000_000)) / F.lit(1_000_000.0))
                        tasks = tasks.withColumn("is_control", (ctrl_u < F.lit(control_frac)).cast("int"))
                        
                        # Zero out injection params for controls
                        tasks = tasks.withColumn("theta_e_arcsec", F.when(F.col("is_control") == 1, F.lit(0.0)).otherwise(F.col("theta_e_arcsec")))
                        tasks = tasks.withColumn("src_dmag", F.when(F.col("is_control") == 1, F.lit(0.0)).otherwise(F.col("src_dmag")))
                        tasks = tasks.withColumn("src_reff_arcsec", F.when(F.col("is_control") == 1, F.lit(0.0)).otherwise(F.col("src_reff_arcsec")))
                        tasks = tasks.withColumn("src_e", F.when(F.col("is_control") == 1, F.lit(0.0)).otherwise(F.col("src_e")))
                        tasks = tasks.withColumn("shear", F.when(F.col("is_control") == 1, F.lit(0.0)).otherwise(F.col("shear")))
                    else:
                        # No controls for this tier
                        tasks = tasks.withColumn("is_control", F.lit(0).cast("int"))

                else:
                    # =========================================================================
                    # TRAIN TIER SAMPLING LOGIC
                    # =========================================================================
                    # Goal: Create a large, diverse training dataset for ML/PINN training.
                    #
                    # KEY DIFFERENCE FROM DEBUG/GRID:
                    # - Debug/Grid: Each galaxy gets EVERY config (crossJoin) for systematic grid
                    # - Train: Each galaxy gets ONE config (random assignment) for training variety
                    #
                    # n_total_per_split samples this many galaxies PER (selection_set, split).
                    # With 12 selection sets × 3 splits × 2 replicates:
                    #   200k × 12 × 3 × 2 ≈ 14.4M tasks (actual ~10.65M due to filtering)
                    #
                    # This is INTENTIONAL - we want coverage across all selection strategies.
                    # Each selection_set represents a different region ranking approach, so
                    # having samples from each gives the model diverse training examples.
                    #
                    # Control samples: control_frac (default 50%) get theta_e=0 (no lens).
                    # These are negative examples for the classifier.
                    # =========================================================================
                    
                    cfg_df = grid_dfs[grid_name]
                    n_cfg = cfg_df.count()
                    per_split_target = int(n_total)  # Samples this many per (selection_set, split)

                    # Stratified sampling across PSF/depth bins (4×4 = 16 bins)
                    wbin = Window.partitionBy("selection_set_id", "region_split", "psf_bin", "depth_bin").orderBy(F.xxhash64(F.col("row_id"), F.lit(int(args.split_seed))))
                    per_bin = int(math.ceil(per_split_target / 16))
                    tmp = base.withColumn("rn_bin", F.row_number().over(wbin)).filter(F.col("rn_bin") <= F.lit(per_bin))
                    
                    # Cap to exact per_split_target per (selection_set, split)
                    wtot = Window.partitionBy("selection_set_id", "region_split").orderBy(F.xxhash64(F.col("row_id"), F.lit(int(args.split_seed) + 1)))
                    tmp = tmp.withColumn("rn_tot", F.row_number().over(wtot)).filter(F.col("rn_tot") <= F.lit(per_split_target))

                    # =========================================================================
                    # CONTROL SAMPLE STRATEGY
                    # =========================================================================
                    # --unpaired-controls 0 (default): Same galaxy, no lens injection
                    #   - Controls use the SAME galaxy position as positives, just with theta_e=0
                    #   - Easy for model: just needs to detect added flux/structure
                    #
                    # --unpaired-controls 1: Different galaxy positions (harder negatives)
                    #   - Split galaxies into two disjoint pools: positives get injections,
                    #     controls get cutouts from DIFFERENT positions
                    #   - Controls matched by PSF/depth bin to ensure similar observing conditions
                    #   - Much harder: model must learn lens morphology, not just "added flux"
                    # =========================================================================
                    use_unpaired = int(getattr(args, 'unpaired_controls', 0)) == 1
                    
                    # Deterministic hash to split galaxies into positive vs control pools
                    ctrl_hash = F.xxhash64(F.col("row_id"), F.col("brickname"), F.col("region_split"), F.lit("train"), F.lit(int(args.split_seed) + 7003))
                    ctrl_u = (F.pmod(F.abs(ctrl_hash), F.lit(1_000_000)) / F.lit(1_000_000.0))
                    tmp = tmp.withColumn("is_control", (ctrl_u < F.lit(control_frac)).cast("int"))
                    
                    if use_unpaired:
                        # =====================================================================
                        # UNPAIRED CONTROLS: Use different galaxies for positives vs controls
                        # =====================================================================
                        # Split into positives (with injection) and controls (no injection, diff galaxy)
                        positives = tmp.filter(F.col("is_control") == 0)
                        controls = tmp.filter(F.col("is_control") == 1)
                        
                        # Assign each positive galaxy ONE config
                        positives = positives.withColumn("cfg_idx", (F.pmod(F.abs(F.col("row_id")), F.lit(n_cfg))).cast("int"))
                        cfg_df2 = cfg_df.withColumn("cfg_idx", F.row_number().over(Window.orderBy("config_id")) - 1)
                        positives = positives.join(F.broadcast(cfg_df2), on="cfg_idx", how="left")
                        
                        # Controls: no injection params needed, just the cutout
                        # Add placeholder config columns for schema compatibility
                        controls = controls.withColumn("config_id", F.lit("CONTROL"))
                        controls = controls.withColumn("theta_e_arcsec", F.lit(0.0))
                        controls = controls.withColumn("src_dmag", F.lit(0.0))
                        controls = controls.withColumn("src_reff_arcsec", F.lit(0.0))
                        controls = controls.withColumn("src_e", F.lit(0.0))
                        controls = controls.withColumn("shear", F.lit(0.0))
                        controls = controls.withColumn("cfg_idx", F.lit(-1))
                        
                        # Union positives and controls
                        tasks = positives.unionByName(controls, allowMissingColumns=True)
                    else:
                        # =====================================================================
                        # PAIRED CONTROLS (default): Same galaxy, just no injection
                        # =====================================================================
                        # Assign each galaxy ONE config (via modulo hash) - NOT crossJoin!
                        # This gives variety without the O(n×m) explosion of debug/grid tiers
                        tmp = tmp.withColumn("cfg_idx", (F.pmod(F.abs(F.col("row_id")), F.lit(n_cfg))).cast("int"))
                        cfg_df2 = cfg_df.withColumn("cfg_idx", F.row_number().over(Window.orderBy("config_id")) - 1)
                        tasks = tmp.join(F.broadcast(cfg_df2), on="cfg_idx", how="left")
                        
                        # For control samples, zero out all injection parameters
                        # theta_e=0 means no lensing arc will be injected
                        tasks = tasks.withColumn("theta_e_arcsec", F.when(F.col("is_control") == 1, F.lit(0.0)).otherwise(F.col("theta_e_arcsec")))
                        tasks = tasks.withColumn("src_dmag", F.when(F.col("is_control") == 1, F.lit(0.0)).otherwise(F.col("src_dmag")))
                        tasks = tasks.withColumn("src_reff_arcsec", F.when(F.col("is_control") == 1, F.lit(0.0)).otherwise(F.col("src_reff_arcsec")))
                        tasks = tasks.withColumn("src_e", F.when(F.col("is_control") == 1, F.lit(0.0)).otherwise(F.col("src_e")))
                        tasks = tasks.withColumn("shear", F.when(F.col("is_control") == 1, F.lit(0.0)).otherwise(F.col("shear")))

                # Replicates
                reps = int(max(1, args.replicates))
                if reps > 1:
                    rep_rows = [Row(replicate=i) for i in range(reps)]
                    rep_df = spark.createDataFrame(rep_rows, schema=T.StructType([T.StructField("replicate", T.IntegerType(), False)]))
                    tasks = tasks.crossJoin(F.broadcast(rep_df))
                else:
                    tasks = tasks.withColumn("replicate", F.lit(0))

                tasks = tasks.withColumn("experiment_id", F.lit(exp_id))
                tasks = tasks.withColumn("stamp_size", F.lit(int(stamp)))
                tasks = tasks.withColumn("bandset", F.lit(bandset))

                # =========================================================================
                # FROZEN RANDOMNESS COLUMNS (for reproducibility)
                # =========================================================================
                # All random degrees of freedom for Stage 4c are frozen here in the manifest.
                # This ensures that reruns of Stage 4c produce identical injections.
                #
                # Uses xxhash64 for deterministic per-row hashing based on stable identifiers.
                # =========================================================================
                
                # Base hash from stable identifiers
                base_hash = F.xxhash64(
                    F.col("brickname"),
                    F.col("ra").cast("string"),
                    F.col("dec").cast("string"),
                    F.coalesce(F.col("config_id"), F.lit("ctrl")),
                    F.col("replicate").cast("string"),
                    F.lit(int(args.split_seed)),
                )
                
                # Store as task_seed64 for reproducibility
                tasks = tasks.withColumn("task_seed64", base_hash)
                
                # Helper for uniform [0,1) from hash with salt
                def h_uniform(salt: int):
                    return (F.pmod(F.abs(F.xxhash64(base_hash, F.lit(salt))), F.lit(1_000_000)) / F.lit(1_000_000.0))
                
                # Source position offset (area-uniform within 0.8 * theta_e)
                u_pos = h_uniform(101)
                u_ang = h_uniform(102)
                r_offset = F.sqrt(u_pos) * 0.8 * F.col("theta_e_arcsec")
                ang = 2.0 * math.pi * u_ang
                tasks = tasks.withColumn(
                    "src_x_arcsec", 
                    F.when(F.col("theta_e_arcsec") > 0, r_offset * F.cos(ang)).otherwise(F.lit(0.0))
                )
                tasks = tasks.withColumn(
                    "src_y_arcsec",
                    F.when(F.col("theta_e_arcsec") > 0, r_offset * F.sin(ang)).otherwise(F.lit(0.0))
                )
                
                # Source position angle [0, 2*pi)
                u_srcphi = h_uniform(103)
                tasks = tasks.withColumn(
                    "src_phi_rad",
                    F.when(F.col("theta_e_arcsec") > 0, 2.0 * math.pi * u_srcphi).otherwise(F.lit(0.0))
                )
                
                # Shear orientation [0, pi) (2*phi periodicity)
                u_shphi = h_uniform(104)
                tasks = tasks.withColumn(
                    "shear_phi_rad",
                    F.when(F.col("theta_e_arcsec") > 0, math.pi * u_shphi).otherwise(F.lit(0.0))
                )
                
                # Source colors (blue-ish): g-r ~ 0.2 +/- 0.2; r-z ~ 0.1 +/- 0.2
                # Use Box-Muller for normal distribution
                u1 = F.greatest(h_uniform(105), F.lit(1e-12))
                u2 = h_uniform(106)
                z1 = F.sqrt(F.lit(-2.0) * F.log(u1)) * F.cos(F.lit(2.0 * math.pi) * u2)
                
                u3 = F.greatest(h_uniform(107), F.lit(1e-12))
                u4 = h_uniform(108)
                z2 = F.sqrt(F.lit(-2.0) * F.log(u3)) * F.cos(F.lit(2.0 * math.pi) * u4)
                
                tasks = tasks.withColumn(
                    "src_gr",
                    F.greatest(F.lit(-0.5), F.least(F.lit(1.5), z1 * 0.2 + 0.2))
                )
                tasks = tasks.withColumn(
                    "src_rz",
                    F.greatest(F.lit(-0.5), F.least(F.lit(1.5), z2 * 0.2 + 0.1))
                )
                # =========================================================================
                # LENS MODEL AND LENS ELLIPTICITY COLUMNS
                # =========================================================================
                # These define the lens mass distribution for SIE injections.
                # Controls have lens_model="CONTROL" and lens_e/lens_phi_rad = 0.
                # =========================================================================
                
                # Lens ellipticity: uniform in [0.05, 0.5] for realistic elliptical lenses
                u_lens_e = h_uniform(109)
                tasks = tasks.withColumn(
                    "lens_e",
                    F.when(F.col("theta_e_arcsec") > 0,
                           0.05 + u_lens_e * 0.45  # Range [0.05, 0.5]
                    ).otherwise(F.lit(0.0))
                )
                
                # Lens position angle: uniform in [0, pi)
                u_lens_phi = h_uniform(110)
                tasks = tasks.withColumn(
                    "lens_phi_rad",
                    F.when(F.col("theta_e_arcsec") > 0, math.pi * u_lens_phi).otherwise(F.lit(0.0))
                )
                
                # Lens model: "SIE" for injections, "CONTROL" for controls
                tasks = tasks.withColumn(
                    "lens_model",
                    F.when(F.col("theta_e_arcsec") > 0, F.lit("SIE")).otherwise(F.lit("CONTROL"))
                )
                
                # Add flux unit constant for provenance
                tasks = tasks.withColumn("ab_zp_nmgy", F.lit(float(AB_ZP_NMGY)))
                tasks = tasks.withColumn("pipeline_version", F.lit("phase4_pipeline_merged_v2"))

                # Deterministic task id
                tasks = tasks.withColumn(
                    "task_id",
                    F.sha2(
                        F.concat_ws(
                            "|",
                            "experiment_id",
                            "selection_set_id",
                            "region_split",
                            F.col("brickname"),
                            F.format_number(F.col("ra"), 6),
                            F.format_number(F.col("dec"), 6),
                            F.col("config_id"),
                            F.col("replicate").cast("string"),
                        ),
                        256,
                    ),
                )

                out_path = f"{manifests_out}/{exp_id}"
                if stage_should_skip(out_path, args.skip_if_exists, args.force):
                    print(f"[4a] Skip manifest (exists): {out_path}")
                    all_manifest_paths.append(out_path)
                    continue

                # Keep only needed columns to limit size
                keep = [
                    "task_id", "experiment_id", "selection_set_id", "selection_strategy", "ranking_mode",
                    "region_id", "region_split", "brickname", "ra", "dec",
                    "zmag", "rmag", "w1mag", "rz", "zw1",
                    # Per-band PSF sizes for realistic injection (g, r, z)
                    "psfsize_g", "psfsize_r", "psfsize_z",
                    "psfdepth_r", "ebv", "psf_bin", "depth_bin",
                    "config_id", "theta_e_arcsec", "src_dmag", "src_reff_arcsec", "src_e", "shear",
                    "stamp_size", "bandset", "replicate",
                    # Control sample flag
                    "is_control",
                    # Frozen randomness columns (for reproducibility in Stage 4c)
                    "task_seed64", "src_x_arcsec", "src_y_arcsec", "src_phi_rad", "shear_phi_rad",
                    "src_gr", "src_rz",
                    # Lens model and lens ellipticity (for SIE)
                    "lens_model", "lens_e", "lens_phi_rad",
                    # Provenance columns
                    "ab_zp_nmgy", "pipeline_version",
                ]
                tasks_out = tasks.select(*keep)

                # Write parquet + CSV for convenience
                tasks_out.write.mode("overwrite").parquet(out_path)

                csv_path = f"{out_path}_csv"
                # For train tier (large manifests), avoid coalesce(1) bottleneck
                if tier == "train":
                    # Write as multiple CSV files (faster for millions of rows)
                    tasks_out.repartition(20).write.mode("overwrite").option("header", True).csv(csv_path)
                    print(f"[4a] Train tier: wrote CSV with 20 partitions to avoid bottleneck")
                else:
                    # Small manifests: single file for convenience
                    tasks_out.coalesce(1).write.mode("overwrite").option("header", True).csv(csv_path)

                all_manifest_paths.append(out_path)
                print(f"[4a] Wrote manifest: {out_path}")

    # Emit bricks manifest across all experiments
    df_all = None
    for p in all_manifest_paths:
        dfm = read_parquet_safe(spark, p).select("experiment_id", "brickname").dropDuplicates()
        df_all = dfm if df_all is None else df_all.unionByName(dfm)

    bricks_out = f"{out_root}/bricks_manifest"
    df_all.write.mode("overwrite").parquet(bricks_out)
    df_all.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{bricks_out}_csv")

    print(f"[4a] Done. Output root: {out_root}")


# --------------------------
# Stage 4b: Coadd cache
# --------------------------

def _download_http(
    url: str,
    out_path: str,
    timeout_s: int = 120,
    max_retries: int = 5,
    base_delay_s: float = 1.0,
    max_delay_s: float = 60.0,
) -> None:
    """
    Download a file from HTTP with exponential backoff and jitter on failure.
    
    =========================================================================
    RETRY STRATEGY:
    - Uses exponential backoff: delay = base_delay * 2^attempt
    - Adds jitter (±25%) to prevent thundering herd when many workers retry
    - Caps delay at max_delay_s to avoid excessive waits
    - Retries on: connection errors, timeouts, 5xx server errors
    - Does NOT retry on: 404 (not found), 403 (forbidden)
    
    Example with base_delay=1s:
      Attempt 1: immediate
      Attempt 2: ~1s delay (0.75-1.25s with jitter)
      Attempt 3: ~2s delay (1.5-2.5s with jitter)
      Attempt 4: ~4s delay (3-5s with jitter)
      Attempt 5: ~8s delay (6-10s with jitter)
    =========================================================================
    """
    import random
    
    if requests is None:
        raise RuntimeError("requests not available; ensure bootstrap installed it")
    
    last_exception = None
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=timeout_s) as r:
                # Don't retry client errors (4xx) except 429 (rate limit)
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    r.raise_for_status()  # Will raise immediately, no retry
                
                r.raise_for_status()
                
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                return  # Success!
                
        except requests.exceptions.RequestException as e:
            last_exception = e
            
            # Check if this is a non-retryable error
            if hasattr(e, 'response') and e.response is not None:
                status = e.response.status_code
                if 400 <= status < 500 and status != 429:
                    # Client error (not rate limit) - don't retry
                    raise
            
            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff
                delay = base_delay_s * (2 ** attempt)
                delay = min(delay, max_delay_s)
                
                # Add jitter (±25%)
                jitter = delay * 0.25 * (2 * random.random() - 1)
                delay += jitter
                
                print(f"  [retry] Attempt {attempt + 1}/{max_retries} failed for {url}: {e}. "
                      f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
    
    # All retries exhausted
    raise RuntimeError(
        f"Failed to download {url} after {max_retries} attempts. Last error: {last_exception}"
    )


def stage_4b_cache_coadds(spark: SparkSession, args: argparse.Namespace) -> None:
    """
    Cache DR10 coadd images from NERSC to S3 for all required bricks.
    
    =========================================================================
    CHECKPOINTING AND RESUMABILITY:
    =========================================================================
    This stage is designed for robust resumability:
    
    1. PER-BRICK IDEMPOTENCY: Each brick checks if it's already cached in S3
       before downloading. If cached, it yields success immediately.
    
    2. ASSETS MANIFEST: After each run, an assets_manifest is written with
       per-brick success/failure status. This serves as a checkpoint.
    
    3. RETRY WITH BACKOFF: _download_http uses exponential backoff (up to 5
       retries) with jitter to handle transient network failures.
    
    4. FAILURE ISOLATION: If a brick fails, other bricks continue. Failed
       bricks are recorded with error messages in the manifest.
    
    5. RESUMING A FAILED RUN: Simply re-run with same parameters. Successful
       bricks are skipped (via S3 cache check), only failed bricks retry.
    
    6. FORCE RE-DOWNLOAD: Use --force to ignore cache and re-download all.
    
    To check progress during a run:
      aws s3 ls s3://bucket/coadd_cache/ --recursive | wc -l
    
    To identify failed bricks after a run:
      spark.read.parquet("s3://bucket/phase4b/.../assets_manifest")
           .filter(F.col("ok") == 0).show()
    =========================================================================
    """
    out_root = f"{args.output_s3.rstrip('/')}/phase4b/{args.variant}"
    # NOTE: Per-brick caching is idempotent - each brick checks cache before downloading.
    # Do NOT skip entire stage based on output dir existing.

    if not _is_s3(args.coadd_s3_cache_prefix):
        raise ValueError("--coadd-s3-cache-prefix must be an s3:// uri")

    manifests_root = f"{args.output_s3.rstrip('/')}/phase4a/{args.variant}/bricks_manifest"
    df_bricks = read_parquet_safe(spark, manifests_root).select("brickname").dropDuplicates()
    
    total_bricks = df_bricks.count()
    psfsize_str = " (with psfsize maps)" if args.include_psfsize else ""
    print(f"[4b] Starting coadd cache for {total_bricks} unique bricks{psfsize_str}")

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]

    # Schema for the assets manifest (serves as checkpoint)
    schema = T.StructType([
        T.StructField("brickname", T.StringType(), False),
        T.StructField("ok", T.IntegerType(), False),  # 1=success, 0=failure
        T.StructField("error", T.StringType(), True),  # Error message if failed
        T.StructField("s3_prefix", T.StringType(), False),  # Where cached assets are stored
    ])

    coadd_base = args.coadd_base_url.rstrip("/")
    s3_cache = args.coadd_s3_cache_prefix.rstrip("/")

    def _proc_partition(it: Iterable[Row]):
        import random as _random
        
        c = _s3_client()
        
        # Per-partition jitter to avoid thundering herd on NERSC endpoints
        _random.seed()
        time.sleep(_random.uniform(0, 2.0))
        
        def s3_object_exists(bucket: str, key: str) -> bool:
            """Check if a specific S3 object exists using head_object."""
            try:
                c.head_object(Bucket=bucket, Key=key)
                return True
            except Exception:
                return False
        
        for r in it:
            brick = r["brickname"]
            prefix = f"{s3_cache}/{brick}"

            # Fast-path: if a per-brick _SUCCESS marker exists, treat the brick cache as complete.
            # If _SUCCESS is missing, fall back to per-file existence checks.
            success_uri = f"{prefix}/_SUCCESS"
            sbkt, skey = _parse_s3(success_uri)
            if (not args.force) and s3_object_exists(sbkt, skey):
                continue

            try:
                urls = build_coadd_urls(coadd_base, brick, bands, include_psfsize=bool(args.include_psfsize))
                
                # =========================================================================
                # IDEMPOTENT CACHING: Check each file individually with head_object
                # =========================================================================
                # This ensures partial caches are completed rather than skipped entirely.
                # Only download files that don't already exist in S3.
                # =========================================================================
                files_to_download = []
                if not args.force:
                    for k, url in urls.items():
                        fname = url.split("/")[-1]
                        s3_uri = f"{prefix}/{fname}"
                        bkt, key = _parse_s3(s3_uri)
                        if not s3_object_exists(bkt, key):
                            files_to_download.append((k, url, fname))
                else:
                    # Force mode: download everything
                    files_to_download = [(k, url, url.split("/")[-1]) for k, url in urls.items()]
                
                if not files_to_download:
                    # All files already exist. Ensure per-brick _SUCCESS marker exists for fast resumes.
                    try:
                        c.put_object(Bucket=sbkt, Key=skey, Body=b"")
                    except Exception:
                        pass
                    yield Row(brickname=brick, ok=1, error=None, s3_prefix=prefix + "/")
                    continue

                tmpdir = f"/mnt/tmp/phase4b_{brick}"
                os.makedirs(tmpdir, exist_ok=True)

                # Download and upload only missing files
                for k, url, fname in files_to_download:
                    local = os.path.join(tmpdir, fname)
                    _download_http(url, local, timeout_s=args.http_timeout_s)
                    if os.path.getsize(local) <= 0:
                        raise RuntimeError(f"Downloaded empty file for {url}")

                    s3_uri = f"{prefix}/{fname}"
                    bkt, key = _parse_s3(s3_uri)
                    c.upload_file(local, bkt, key)

                # Cleanup best-effort
                try:
                    for fn in os.listdir(tmpdir):
                        os.remove(os.path.join(tmpdir, fn))
                    os.rmdir(tmpdir)
                except Exception:
                    pass

                # Mark brick as complete for fast resumes
                try:
                    c.put_object(Bucket=sbkt, Key=skey, Body=b"")
                except Exception:
                    pass

                yield Row(brickname=brick, ok=1, error=None, s3_prefix=prefix + "/")
            except Exception as e:
                yield Row(brickname=brick, ok=0, error=str(e)[:1000], s3_prefix=prefix + "/")

    rdd = df_bricks.repartition(int(args.cache_partitions)).rdd.mapPartitions(_proc_partition)
    df_out = spark.createDataFrame(rdd, schema=schema)

    out_path = f"{out_root}/assets_manifest"
    df_out.write.mode("overwrite").parquet(out_path)
    df_out.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{out_path}_csv")

    # Stage config
    cfg = {
        "stage": "4b",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variant": args.variant,
        "output_s3": args.output_s3,
        "coadd_base_url": args.coadd_base_url,
        "bands": bands,
        "coadd_s3_cache_prefix": args.coadd_s3_cache_prefix,
        "cache_partitions": int(args.cache_partitions),
        "http_timeout_s": int(args.http_timeout_s),
        "include_psfsize": bool(args.include_psfsize),
        "idempotency": {"skip_if_exists": int(args.skip_if_exists), "force": int(args.force)},
    }
    write_text_to_s3(f"{out_root}/_stage_config.json", json.dumps(cfg, indent=2))

    # Read back from written parquet to get counts efficiently (avoids recomputation)
    df_out_saved = spark.read.parquet(out_path)
    counts = df_out_saved.groupBy("ok").count().collect()
    ok = sum(r["count"] for r in counts if r["ok"] == 1)
    bad = sum(r["count"] for r in counts if r["ok"] == 0)
    files_per_brick = 7 + (3 if args.include_psfsize else 0)  # 7 base + 3 psfsize maps
    print(f"[4b] Cached bricks ok={ok} bad={bad} (files/brick={files_per_brick}). Output: {out_root}")


# --------------------------
# Stage 4c: Cutouts + injections
# --------------------------

def _read_fits_from_s3(s3_uri: str) -> Tuple[np.ndarray, object]:
    if fits is None:
        raise RuntimeError("astropy not available; ensure bootstrap installed it")
    c = _s3_client()
    bkt, key = _parse_s3(s3_uri)
    obj = c.get_object(Bucket=bkt, Key=key)
    byts = obj["Body"].read()
    hdul = fits.open(io.BytesIO(byts), memmap=False)
    # Compressed FITS (.fits.fz) have data in extension 1, not 0
    # Check for the first extension with image data
    ext_idx = 0
    if len(hdul) > 1 and hdul[0].data is None:
        ext_idx = 1
    data = hdul[ext_idx].data.astype(np.float32)
    header = hdul[ext_idx].header
    hdul.close()
    return data, header


def _cutout(data: np.ndarray, x: float, y: float, size: int) -> Tuple[np.ndarray, bool]:
    half = size // 2
    x0 = int(np.round(x)) - half
    y0 = int(np.round(y)) - half
    x1 = x0 + size
    y1 = y0 + size
    ok = True
    if x0 < 0 or y0 < 0 or x1 > data.shape[1] or y1 > data.shape[0]:
        ok = False
        # pad with zeros
        out = np.zeros((size, size), dtype=np.float32)
        xs0 = max(0, x0)
        ys0 = max(0, y0)
        xs1 = min(data.shape[1], x1)
        ys1 = min(data.shape[0], y1)
        out_y0 = ys0 - y0
        out_x0 = xs0 - x0
        out[out_y0: out_y0 + (ys1 - ys0), out_x0: out_x0 + (xs1 - xs0)] = data[ys0:ys1, xs0:xs1]
        return out, ok
    return data[y0:y1, x0:x1].astype(np.float32), ok


def stage_4c_inject_cutouts(spark: SparkSession, args: argparse.Namespace) -> None:
    out_root = f"{args.output_s3.rstrip('/')}/phase4c/{args.variant}"
    # NOTE: Per-experiment checks happen via --experiment-id.
    # Do NOT skip entire stage based on output dir existing.

    if WCS is None:
        raise RuntimeError("astropy.wcs not available; ensure astropy installed")

    manifests_subdir = getattr(args, 'manifests_subdir', 'manifests_filtered')
    manifests_root = f"{args.output_s3.rstrip('/')}/phase4a/{args.variant}/{manifests_subdir}"
    if not args.experiment_id:
        raise ValueError("--experiment-id is required for stage 4c")
    in_path = f"{manifests_root}/{args.experiment_id}"
    print(f"[4c] Reading manifests from: {in_path}")
    df_tasks = read_parquet_safe(spark, in_path)

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]

    # Assume coadd cache uses /<brick>/<filename>
    cache_prefix = args.coadd_s3_cache_prefix.rstrip("/")

    # =========================================================================
    # METRICS-ONLY MODE HANDLING:
    # When --metrics-only 1, we still compute arc_snr and other metrics,
    # but skip encoding and storing the actual stamp images.
    # This saves ~100× storage for grid tier completeness analysis.
    # =========================================================================
    metrics_only = bool(args.metrics_only)
    if metrics_only:
        print(f"[4c] METRICS-ONLY mode enabled - stamps will NOT be saved")
    
    # Capture flag for psfsize map loading (gated to avoid failed S3 reads if not cached)
    use_psfsize_maps = bool(args.use_psfsize_maps)
    if use_psfsize_maps:
        print(f"[4c] Using center-evaluated PSF from psfsize maps (requires 4b with --include-psfsize 1)")
    
    # =========================================================================
    # LENS MODEL: Now determined by manifest's `lens_model` column
    # =========================================================================
    # The manifest specifies "SIE", "SIS", or "CONTROL" per task.
    # The new dependency-free render_lensed_source() handles all models.
    # This ensures reproducibility - no runtime model switching.
    # =========================================================================
    print(f"[4c] Lens model will be read from manifest (SIE/SIS/CONTROL per task)")
    
    # Output schema: stamp_npz is nullable in metrics-only mode
    stamps_schema = T.StructType([
        T.StructField("task_id", T.StringType(), False),
        T.StructField("experiment_id", T.StringType(), False),
        T.StructField("selection_set_id", T.StringType(), False),
        T.StructField("ranking_mode", T.StringType(), False),
        T.StructField("selection_strategy", T.StringType(), False),
        T.StructField("region_id", T.IntegerType(), False),
        T.StructField("region_split", T.StringType(), False),
        T.StructField("brickname", T.StringType(), False),
        T.StructField("ra", T.DoubleType(), False),
        T.StructField("dec", T.DoubleType(), False),
        T.StructField("stamp_size", T.IntegerType(), False),
        T.StructField("bandset", T.StringType(), False),
        T.StructField("config_id", T.StringType(), False),
        T.StructField("theta_e_arcsec", T.DoubleType(), False),
        T.StructField("src_dmag", T.DoubleType(), False),
        T.StructField("src_reff_arcsec", T.DoubleType(), False),
        T.StructField("src_e", T.DoubleType(), False),
        T.StructField("shear", T.DoubleType(), False),
        T.StructField("replicate", T.IntegerType(), False),
        T.StructField("is_control", T.IntegerType(), False),
        T.StructField("task_seed64", T.LongType(), True),
        T.StructField("src_x_arcsec", T.DoubleType(), True),
        T.StructField("src_y_arcsec", T.DoubleType(), True),
        T.StructField("src_phi_rad", T.DoubleType(), True),
        T.StructField("shear_phi_rad", T.DoubleType(), True),
        T.StructField("src_gr", T.DoubleType(), True),
        T.StructField("src_rz", T.DoubleType(), True),
        T.StructField("psf_bin", T.IntegerType(), True),
        T.StructField("depth_bin", T.IntegerType(), True),
        T.StructField("ab_zp_nmgy", T.DoubleType(), True),
        T.StructField("pipeline_version", T.StringType(), True),
        T.StructField("psfsize_r", T.DoubleType(), True),
        T.StructField("psfdepth_r", T.DoubleType(), True),
        T.StructField("ebv", T.DoubleType(), True),
        T.StructField("stamp_npz", T.BinaryType(), True),  # Nullable for metrics-only mode
        T.StructField("cutout_ok", T.IntegerType(), False),
        T.StructField("arc_snr", T.DoubleType(), True),
        T.StructField("bad_pixel_frac", T.DoubleType(), True),  # Fraction of masked/bad pixels in stamp
        T.StructField("wise_brightmask_frac", T.DoubleType(), True),  # Fraction affected by WISE bright-star mask
        T.StructField("metrics_ok", T.IntegerType(), True),  # 1 if all metrics computed successfully
        T.StructField("psf_fwhm_used_g", T.DoubleType(), True),  # Actual PSF FWHM used for g-band injection
        T.StructField("psf_fwhm_used_r", T.DoubleType(), True),  # Actual PSF FWHM used for r-band injection
        T.StructField("psf_fwhm_used_z", T.DoubleType(), True),  # Actual PSF FWHM used for z-band injection
        T.StructField("metrics_only", T.IntegerType(), False),  # Track if stamp was skipped
        # Lens model provenance (for dataset consistency checks)
        T.StructField("lens_model", T.StringType(), True),  # "SIE", "SIS", or "CONTROL"
        T.StructField("lens_e", T.DoubleType(), True),  # Lens ellipticity
        T.StructField("lens_phi_rad", T.DoubleType(), True),  # Lens position angle
        # Physics validation metrics (from SIE injection)
        T.StructField("magnification", T.DoubleType(), True),
        T.StructField("tangential_stretch", T.DoubleType(), True),
        T.StructField("radial_stretch", T.DoubleType(), True),
        T.StructField("expected_arc_radius", T.DoubleType(), True),
        T.StructField("physics_valid", T.IntegerType(), True),  # 1 if passed validation
        T.StructField("physics_warnings", T.StringType(), True),  # Comma-separated warnings
        # Gen5: COSMOS source metadata
        T.StructField("source_mode", T.StringType(), True),  # "sersic" or "cosmos"
        T.StructField("cosmos_index", T.IntegerType(), True),  # Index of COSMOS template used
        T.StructField("cosmos_hlr_arcsec", T.DoubleType(), True),  # Half-light radius of lensed COSMOS arc
    ])

    # Repartition by brick for cache locality
    df_tasks = df_tasks.repartition(int(args.sweep_partitions), "brickname").sortWithinPartitions("brickname")

    def _proc_partition(it: Iterable[Row]):
        # Reuse coadd data within a partition by brick
        cur_brick = None
        cur = {}
        wcs = None

        for r in it:
            brick = r["brickname"]
            if brick != cur_brick:
                cur_brick = brick
                cur = {}
                # Load one band header for WCS
                try:
                    b0 = bands[0]
                    img_uri = f"{cache_prefix}/{brick}/legacysurvey-{brick}-image-{b0}.fits.fz"
                    img0, hdr0 = _read_fits_from_s3(img_uri)
                    cur[f"image_{b0}"] = img0
                    wcs = WCS(hdr0)

                    # Other bands images
                    for b in bands[1:]:
                        img_uri_b = f"{cache_prefix}/{brick}/legacysurvey-{brick}-image-{b}.fits.fz"
                        img_b, _ = _read_fits_from_s3(img_uri_b)
                        cur[f"image_{b}"] = img_b

                    # invvar (optional but recommended)
                    for b in bands:
                        inv_uri = f"{cache_prefix}/{brick}/legacysurvey-{brick}-invvar-{b}.fits.fz"
                        inv_b, _ = _read_fits_from_s3(inv_uri)
                        cur[f"invvar_{b}"] = inv_b

                    # maskbits for filtering bad pixels in SNR calculation
                    mask_uri = f"{cache_prefix}/{brick}/legacysurvey-{brick}-maskbits.fits.fz"
                    mask_arr, _ = _read_fits_from_s3(mask_uri)
                    cur["maskbits"] = mask_arr

                    # psfsize maps for center-evaluated PSF (only if --use-psfsize-maps 1)
                    # Requires 4b to have been run with --include-psfsize 1
                    if use_psfsize_maps:
                        for b in bands:
                            psfsize_uri = f"{cache_prefix}/{brick}/legacysurvey-{brick}-psfsize-{b}.fits.fz"
                            try:
                                psfsize_b, _ = _read_fits_from_s3(psfsize_uri)
                                cur[f"psfsize_{b}"] = psfsize_b
                            except Exception:
                                cur[f"psfsize_{b}"] = None  # Fall back to manifest value

                except Exception as e:
                    # Mark all tasks for this brick as failed cutouts
                    wcs = None
                    cur = {"error": str(e)[:1000]}

            try:
                if wcs is None:
                    raise RuntimeError(f"coadd load failed for brick {brick}: {cur.get('error')}")

                ra = float(r["ra"])
                dec = float(r["dec"])
                x, y = wcs.world_to_pixel_values(ra, dec)
                size = int(r["stamp_size"])

                bandset = r["bandset"]
                use_bands = ["r"] if bandset == "r" else bands

                imgs = {}
                invs = {}
                cut_ok_all = True
                for b in use_bands:
                    img = cur[f"image_{b}"]
                    inv = cur[f"invvar_{b}"]
                    stamp, ok1 = _cutout(img, x, y, size)
                    invs_b, ok2 = _cutout(inv, x, y, size)
                    imgs[b] = stamp
                    invs[b] = invs_b
                    cut_ok_all = cut_ok_all and ok1 and ok2

                # Cut maskbits stamp for SNR filtering
                # If maskbits missing/corrupt, set arc_snr=None to avoid biased completeness
                mask_stamp = None
                bad_pixel_frac = None  # None if mask unavailable
                good_mask = None
                mask_valid = False
                wise_brightmask_frac = None
                if "maskbits" in cur:
                    mask_stamp, mask_ok = _cutout(cur["maskbits"], x, y, size)
                    if mask_ok and mask_stamp is not None:
                        mask_valid = True
                        # Apply DR10 bad pixel mask (r-band specific for arc_snr)
                        good_mask = (mask_stamp.astype(np.int64) & MASKBITS_BAD_R) == 0
                        bad_pixel_frac = 1.0 - float(good_mask.mean())
                        # Track WISE bright-star mask coverage (not "detections" - masking flags)
                        wise_mask = (mask_stamp.astype(np.int64) & MASKBITS_WISE) != 0
                        wise_brightmask_frac = float(wise_mask.mean())

                # =========================================================================
                # INJECTION USING FROZEN RANDOMNESS FROM MANIFEST
                # =========================================================================
                # All random degrees of freedom were frozen in Stage 4a.
                # Stage 4c is now DETERMINISTIC - reruns produce identical output.
                # =========================================================================
                
                theta_e = float(r["theta_e_arcsec"])
                src_dmag = float(r["src_dmag"]) if r["src_dmag"] is not None else 0.0
                src_reff = float(r["src_reff_arcsec"]) if r["src_reff_arcsec"] is not None else 0.1
                src_e = float(r["src_e"]) if r["src_e"] is not None else 0.2
                shear = float(r["shear"]) if r["shear"] is not None else 0.0

                # Read frozen randomness from manifest
                src_x_arcsec = float(r["src_x_arcsec"]) if r["src_x_arcsec"] is not None else 0.0
                src_y_arcsec = float(r["src_y_arcsec"]) if r["src_y_arcsec"] is not None else 0.0
                src_phi_rad = float(r["src_phi_rad"]) if r["src_phi_rad"] is not None else 0.0
                shear_phi_rad = float(r["shear_phi_rad"]) if r["shear_phi_rad"] is not None else 0.0
                lens_e_val = float(r["lens_e"]) if r["lens_e"] is not None else 0.0
                lens_phi_rad = float(r["lens_phi_rad"]) if r["lens_phi_rad"] is not None else 0.0
                lens_model_str = str(r["lens_model"]) if r["lens_model"] is not None else "CONTROL"
                src_gr = float(r["src_gr"]) if r["src_gr"] is not None else 0.2
                src_rz = float(r["src_rz"]) if r["src_rz"] is not None else 0.1

                # =========================================================================
                # CORRECT FLUX CALCULATION (nanomaggies with AB ZP=22.5)
                # =========================================================================
                # lens_rmag = host galaxy r-band mag
                # src_rmag = lens_rmag + src_dmag (source is fainter by src_dmag)
                # Use proper nanomaggy conversion, NOT arbitrary flux scale!
                # =========================================================================
                rmag = float(r["rmag"]) if r["rmag"] is not None else 20.0
                src_rmag = rmag + src_dmag
                src_flux_r_nmgy = mag_to_nMgy(src_rmag)
                
                # Source flux in g and z bands using frozen colors
                src_gmag = src_rmag + src_gr  # g-r color
                src_zmag = src_rmag - src_rz  # r-z color
                src_flux_g_nmgy = mag_to_nMgy(src_gmag)
                src_flux_z_nmgy = mag_to_nMgy(src_zmag)

                add_r = None
                arc_snr = None
                # Track actual PSF FWHM used (for provenance/reproducibility)
                psf_fwhm_used_g = None
                psf_fwhm_used_r = None
                psf_fwhm_used_z = None
                physics_metrics = {
                    "magnification": None,
                    "tangential_stretch": None,
                    "radial_stretch": None,
                    "expected_arc_radius": theta_e if theta_e > 0 else None,
                }
                physics_valid = None
                physics_warnings_str = None
                
                if theta_e > 0 and src_flux_r_nmgy > 0:
                    # =========================================================================
                    # PER-BAND PSF: Use center-evaluated psfsize map for highest fidelity
                    # =========================================================================
                    # If psfsize maps are available (from 4b with --include-psfsize), evaluate
                    # PSF FWHM at the stamp center. Otherwise fall back to manifest brick-average.
                    # =========================================================================
                    
                    def _get_psf_fwhm_at_center(cur_dict, band, px, py, manifest_fwhm):
                        """Get PSF FWHM at stamp center from psfsize map, or use manifest value."""
                        psfsize_map = cur_dict.get(f"psfsize_{band}")
                        if psfsize_map is not None:
                            ix, iy = int(np.round(px)), int(np.round(py))
                            if 0 <= iy < psfsize_map.shape[0] and 0 <= ix < psfsize_map.shape[1]:
                                val = float(psfsize_map[iy, ix])
                                if np.isfinite(val) and val > 0:
                                    return val
                        return manifest_fwhm
                    
                    # Get per-band PSF sizes using center-evaluated psfsize maps
                    # Fall back to manifest brick-average if maps not available
                    manifest_fwhm_r = float(r["psfsize_r"]) if r["psfsize_r"] is not None else 1.4
                    manifest_fwhm_g = float(r["psfsize_g"]) if r["psfsize_g"] is not None else manifest_fwhm_r
                    manifest_fwhm_z = float(r["psfsize_z"]) if r["psfsize_z"] is not None else manifest_fwhm_r
                    
                    psf_fwhm_r = _get_psf_fwhm_at_center(cur, "r", x, y, manifest_fwhm_r)
                    psf_fwhm_g = _get_psf_fwhm_at_center(cur, "g", x, y, manifest_fwhm_g)
                    psf_fwhm_z = _get_psf_fwhm_at_center(cur, "z", x, y, manifest_fwhm_z)
                    
                    # Store for output provenance
                    psf_fwhm_used_g = psf_fwhm_g
                    psf_fwhm_used_r = psf_fwhm_r
                    psf_fwhm_used_z = psf_fwhm_z
                    
                    # Convert FWHM to sigma in pixels: sigma = FWHM / 2.355
                    psf_sigma_r = (psf_fwhm_r / 2.355) / PIX_SCALE_ARCSEC
                    psf_sigma_g = (psf_fwhm_g / 2.355) / PIX_SCALE_ARCSEC
                    psf_sigma_z = (psf_fwhm_z / 2.355) / PIX_SCALE_ARCSEC
                    
                    # =========================================================================
                    # USE NEW RENDER FUNCTION WITH CORRECT FLUX NORMALIZATION
                    # =========================================================================
                    # This normalizes the UNLENSED source analytically, so lensed images
                    # are naturally brighter due to magnification. No more image-sum normalization!
                    # =========================================================================
                    
                    # Gen5: Check source mode (sersic vs cosmos)
                    source_mode = getattr(args, 'source_mode', 'sersic')
                    
                    if source_mode == "cosmos":
                        # =========================================================================
                        # GEN5: COSMOS SOURCE INJECTION
                        # =========================================================================
                        cosmos_bank = _load_cosmos_bank_h5(args.cosmos_bank_h5)
                        cosmos_idx = _cosmos_choose_index(task_id, cosmos_bank["n_sources"], getattr(args, 'cosmos_salt', ''))
                        cosmos_hlr = None
                        
                        # Inject COSMOS source for each band
                        for b in use_bands:
                            if b == "r":
                                src_mag_b = src_mag_r
                                psf_fwhm_b = psf_fwhm_r
                            elif b == "g":
                                src_mag_b = src_mag_r + src_gr
                                psf_fwhm_b = psf_fwhm_g
                            elif b == "z":
                                src_mag_b = src_mag_r - src_rz
                                psf_fwhm_b = psf_fwhm_z
                            else:
                                src_mag_b = src_mag_r
                                psf_fwhm_b = psf_fwhm_r
                            
                            add_b = render_cosmos_lensed_source(
                                cosmos_bank=cosmos_bank,
                                cosmos_index=cosmos_idx,
                                stamp_size=size,
                                pixscale_arcsec=PIX_SCALE_ARCSEC,
                                theta_e_arcsec=theta_e,
                                lens_e=lens_e_val,
                                lens_phi_rad=lens_phi_rad,
                                shear=shear,
                                shear_phi_rad=shear_phi_rad,
                                src_x_arcsec=src_x_arcsec,
                                src_y_arcsec=src_y_arcsec,
                                src_mag_r=src_mag_r,
                                z_s=1.5,  # Typical lensed source redshift
                                psf_fwhm_arcsec=psf_fwhm_b,
                                psf_model=args.psf_model,
                                moffat_beta=args.moffat_beta,
                                band=b,
                            )
                            
                            imgs[b] = (imgs[b] + add_b).astype(np.float32)
                            
                            if b == "r":
                                add_r = add_b
                                if cosmos_hlr is None:
                                    cosmos_hlr = _compute_hlr_arcsec(add_r, PIX_SCALE_ARCSEC)
                        
                    else:
                        # =========================================================================
                        # ORIGINAL: SERSIC SOURCE INJECTION
                        # =========================================================================
                        # Render lensed source for each band (with per-band PSF and flux)
                        for b in use_bands:
                            if b == "r":
                                src_flux_b = src_flux_r_nmgy
                                psf_sigma_b = psf_sigma_r
                            elif b == "g":
                                src_flux_b = src_flux_g_nmgy
                                psf_sigma_b = psf_sigma_g
                            elif b == "z":
                                src_flux_b = src_flux_z_nmgy
                                psf_sigma_b = psf_sigma_z
                            else:
                                src_flux_b = src_flux_r_nmgy
                                psf_sigma_b = psf_sigma_r
                            
                            add_b = render_lensed_source(
                                stamp_size=size,
                                pixscale_arcsec=PIX_SCALE_ARCSEC,
                                lens_model=lens_model_str,
                                theta_e_arcsec=theta_e,
                                lens_e=lens_e_val,
                                lens_phi_rad=lens_phi_rad,
                                shear=shear,
                                shear_phi_rad=shear_phi_rad,
                                src_total_flux_nmgy=src_flux_b,
                                src_reff_arcsec=src_reff,
                                src_e=src_e,
                                src_phi_rad=src_phi_rad,
                                src_x_arcsec=src_x_arcsec,
                                src_y_arcsec=src_y_arcsec,
                                psf_fwhm_pix=psf_sigma_b * 2.355,
                                psf_model=args.psf_model,
                                moffat_beta=args.moffat_beta,
                                psf_apply=True,
                            )
                            
                            imgs[b] = (imgs[b] + add_b).astype(np.float32)
                            
                            # Keep r-band injection for SNR calculation
                            if b == "r":
                                add_r = add_b

                    # Proxy SNR in r-band (filtered by maskbits to exclude bad pixels)
                    # Only compute if mask is valid; otherwise arc_snr remains None
                    # to prevent biased completeness from missing/corrupt maskbits
                    if add_r is not None and mask_valid and good_mask is not None:
                        invr = invs.get("r")
                        if invr is not None:
                            sigma = np.where((invr > 0) & good_mask, 1.0 / np.sqrt(invr + 1e-12), 0.0)
                            snr = np.where((sigma > 0) & good_mask, add_r / (sigma + 1e-12), 0.0)
                            arc_snr = float(np.nanmax(snr))
                    
                                        # =========================================================================
                    # Flux-based magnification proxy (stamp-limited):
                    # Compare total injected flux of the lensed source to the total flux of the unlensed source
                    # rendered on the same stamp with the same PSF. This preserves the "do not renormalize"
                    # requirement and avoids unstable Jacobian evaluation near the critical curve.
                    # =========================================================================
                    if theta_e > 0:
                        try:
                            # Compute flux-based magnification proxy
                            # NOTE: Variable names fixed 2026-01-24:
                            #   size (not stamp_size), PIX_SCALE_ARCSEC (not pixscale),
                            #   src_reff (not src_reff_arcsec), True (not psf_apply)
                            add_r_unlensed = render_unlensed_source(
                                stamp_size=size,
                                pixscale_arcsec=PIX_SCALE_ARCSEC,
                                src_total_flux_nmgy=src_flux_r_nmgy,
                                src_x_arcsec=src_x_arcsec,
                                src_y_arcsec=src_y_arcsec,
                                src_reff_arcsec=src_reff,
                                n=1.0,
                                src_e=src_e,
                                src_phi_rad=src_phi_rad,
                                psf_apply=True,
                                psf_fwhm_pix=psf_sigma_r * 2.355,  # Use r-band PSF for magnification proxy
                                psf_model=args.psf_model,
                                moffat_beta=args.moffat_beta
                            )
                            unlensed_sum = float(np.sum(add_r_unlensed))
                            lensed_sum = float(np.sum(add_r))
                            if unlensed_sum > 0:
                                physics_metrics["magnification"] = float(lensed_sum / unlensed_sum)
                                physics_metrics["tangential_stretch"] = float(abs(physics_metrics["magnification"]))
                                physics_metrics["radial_stretch"] = 1.0
                        except Exception as e:
                            # Log but don't fail - magnification is a diagnostic, not critical
                            print(f"[4c] Warning: magnification proxy failed: {e}")

                    # Physics validation
                    physics_valid_bool, warnings = validate_physics_consistency(
                        theta_e, arc_snr,
                        physics_metrics.get("magnification"),
                        physics_metrics.get("tangential_stretch"),
                        psf_fwhm_r,  # Use r-band PSF for physics validation
                    )
                    physics_valid = 1 if physics_valid_bool else 0
                    physics_warnings_str = "; ".join(warnings) if warnings else None

                # Encode stamp (or skip if metrics-only mode)
                if metrics_only:
                    stamp_npz = None  # Don't store stamp to save space
                else:
                    stamp_npz = encode_npz({f"image_{b}": imgs[b] for b in use_bands})

                # Gen5: Add COSMOS metadata if cosmos mode was used
                if source_mode == "cosmos":
                    row_source_mode = "cosmos"
                    row_cosmos_index = int(cosmos_idx) if 'cosmos_idx' in locals() else None
                    row_cosmos_hlr = float(cosmos_hlr) if 'cosmos_hlr' in locals() and cosmos_hlr else None
                else:
                    row_source_mode = "sersic"
                    row_cosmos_index = None
                    row_cosmos_hlr = None

                yield Row(
                    task_id=r["task_id"],
                    experiment_id=r["experiment_id"],
                    selection_set_id=r["selection_set_id"],
                    ranking_mode=r["ranking_mode"],
                    selection_strategy=r["selection_strategy"],
                    region_id=int(r["region_id"]),
                    region_split=r["region_split"],
                    brickname=brick,
                    ra=ra,
                    dec=dec,
                    stamp_size=size,
                    bandset=bandset,
                    config_id=r["config_id"],
                    theta_e_arcsec=theta_e,
                    src_dmag=src_dmag,
                    src_reff_arcsec=src_reff,
                    src_e=src_e,
                    shear=shear,
                    replicate=int(r["replicate"]),
                    is_control=int(r["is_control"]) if r["is_control"] is not None else 0,
                    task_seed64=int(r["task_seed64"]) if r["task_seed64"] is not None else None,
                    src_x_arcsec=float(r["src_x_arcsec"]) if r["src_x_arcsec"] is not None else 0.0,
                    src_y_arcsec=float(r["src_y_arcsec"]) if r["src_y_arcsec"] is not None else 0.0,
                    src_phi_rad=float(r["src_phi_rad"]) if r["src_phi_rad"] is not None else 0.0,
                    shear_phi_rad=float(r["shear_phi_rad"]) if r["shear_phi_rad"] is not None else 0.0,
                    src_gr=float(r["src_gr"]) if r["src_gr"] is not None else 0.0,
                    src_rz=float(r["src_rz"]) if r["src_rz"] is not None else 0.0,
                    psf_bin=int(r["psf_bin"]) if r["psf_bin"] is not None else None,
                    depth_bin=int(r["depth_bin"]) if r["depth_bin"] is not None else None,
                    ab_zp_nmgy=float(AB_ZP_NMGY),
                    pipeline_version=str(PIPELINE_VERSION),
                    psfsize_r=float(r["psfsize_r"]) if r["psfsize_r"] is not None else None,
                    psfdepth_r=float(r["psfdepth_r"]) if r["psfdepth_r"] is not None else None,
                    ebv=float(r["ebv"]) if r["ebv"] is not None else None,
                    stamp_npz=stamp_npz,
                    cutout_ok=int(bool(cut_ok_all)),
                    arc_snr=arc_snr,
                    bad_pixel_frac=bad_pixel_frac,
                    wise_brightmask_frac=wise_brightmask_frac,
                    metrics_ok=int(mask_valid and cut_ok_all and arc_snr is not None) if theta_e > 0 else int(mask_valid and cut_ok_all),
                    psf_fwhm_used_g=psf_fwhm_used_g,
                    psf_fwhm_used_r=psf_fwhm_used_r,
                    psf_fwhm_used_z=psf_fwhm_used_z,
                    metrics_only=int(metrics_only),
                    lens_model=lens_model_str,
                    lens_e=lens_e_val,
                    lens_phi_rad=lens_phi_rad if theta_e > 0 else None,
                    magnification=physics_metrics.get("magnification"),
                    tangential_stretch=physics_metrics.get("tangential_stretch"),
                    radial_stretch=physics_metrics.get("radial_stretch"),
                    expected_arc_radius=physics_metrics.get("expected_arc_radius"),
                    physics_valid=physics_valid,
                    physics_warnings=physics_warnings_str,
                    source_mode=row_source_mode,
                    cosmos_index=row_cosmos_index,
                    cosmos_hlr_arcsec=row_cosmos_hlr,
                )

            except Exception as e:
                # Emit a row with cutout_ok=0 and an empty stamp to keep accounting consistent
                if metrics_only:
                    empty = None  # Don't store stamp in metrics-only mode
                else:
                    empty = encode_npz({"image_r": np.zeros((int(r["stamp_size"]), int(r["stamp_size"])), dtype=np.float32)})
                yield Row(
                    task_id=r["task_id"],
                    experiment_id=r["experiment_id"],
                    selection_set_id=r["selection_set_id"],
                    ranking_mode=r["ranking_mode"],
                    selection_strategy=r["selection_strategy"],
                    region_id=int(r["region_id"]),
                    region_split=r["region_split"],
                    brickname=brick,
                    ra=float(r["ra"]),
                    dec=float(r["dec"]),
                    stamp_size=int(r["stamp_size"]),
                    bandset=r["bandset"],
                    config_id=r["config_id"],
                    theta_e_arcsec=float(r["theta_e_arcsec"]),
                    src_dmag=float(r["src_dmag"]),
                    src_reff_arcsec=float(r["src_reff_arcsec"]),
                    src_e=float(r["src_e"]),
                    shear=float(r["shear"]),
                    replicate=int(r["replicate"]),
                    is_control=int(r["is_control"]) if r["is_control"] is not None else 0,
                    task_seed64=int(r["task_seed64"]) if r["task_seed64"] is not None else None,
                    src_x_arcsec=float(r["src_x_arcsec"]) if r["src_x_arcsec"] is not None else 0.0,
                    src_y_arcsec=float(r["src_y_arcsec"]) if r["src_y_arcsec"] is not None else 0.0,
                    src_phi_rad=float(r["src_phi_rad"]) if r["src_phi_rad"] is not None else 0.0,
                    shear_phi_rad=float(r["shear_phi_rad"]) if r["shear_phi_rad"] is not None else 0.0,
                    src_gr=float(r["src_gr"]) if r["src_gr"] is not None else 0.0,
                    src_rz=float(r["src_rz"]) if r["src_rz"] is not None else 0.0,
                    psf_bin=int(r["psf_bin"]) if r["psf_bin"] is not None else None,
                    depth_bin=int(r["depth_bin"]) if r["depth_bin"] is not None else None,
                    ab_zp_nmgy=float(AB_ZP_NMGY),
                    pipeline_version=str(PIPELINE_VERSION),
                    psfsize_r=float(r["psfsize_r"]) if r["psfsize_r"] is not None else None,
                    psfdepth_r=float(r["psfdepth_r"]) if r["psfdepth_r"] is not None else None,
                    ebv=float(r["ebv"]) if r["ebv"] is not None else None,
                    stamp_npz=empty,
                    cutout_ok=0,
                    arc_snr=None,
                    bad_pixel_frac=None,
                    wise_brightmask_frac=None,
                    metrics_ok=0,
                    psf_fwhm_used_g=None,
                    psf_fwhm_used_r=None,
                    psf_fwhm_used_z=None,
                    metrics_only=int(metrics_only),
                    lens_model=str(r["lens_model"]) if r["lens_model"] is not None else "CONTROL",
                    lens_e=float(r["lens_e"]) if r["lens_e"] is not None else None,
                    lens_phi_rad=float(r["lens_phi_rad"]) if r["lens_phi_rad"] is not None else None,
                    magnification=None,
                    tangential_stretch=None,
                    radial_stretch=None,
                    expected_arc_radius=None,
                    physics_valid=0,
                    physics_warnings=f"Processing error: {str(e)[:200]}",
                    source_mode=getattr(args, 'source_mode', 'sersic'),
                    cosmos_index=None,
                    cosmos_hlr_arcsec=None,
                )

    rdd = df_tasks.rdd.mapPartitions(_proc_partition)
    df_out = spark.createDataFrame(rdd, schema=stamps_schema)

    out_path = f"{out_root}/stamps/{args.experiment_id}"
    df_out.write.mode("overwrite").partitionBy("region_split").parquet(out_path)

    # Metrics-only table
    metrics = df_out.select(
        "task_id", "experiment_id", "selection_set_id", "ranking_mode", "selection_strategy",
        "region_id", "region_split", "brickname",
        "config_id", "theta_e_arcsec", "src_dmag", "src_reff_arcsec", "src_e", "shear", "replicate",
        "cutout_ok", "arc_snr", "psfsize_r", "psfdepth_r", "ebv",
    )
    met_path = f"{out_root}/metrics/{args.experiment_id}"
    metrics.write.mode("overwrite").partitionBy("region_split").parquet(met_path)

    cfg = {
        "stage": "4c",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variant": args.variant,
        "output_s3": args.output_s3,
        "experiment_id": args.experiment_id,
        "inputs": {"manifest": in_path, "coadd_s3_cache_prefix": args.coadd_s3_cache_prefix},
        "bands": bands,
        "src_flux_scale": float(getattr(args, 'src_flux_scale', 1.0)),
        "spark": {"sweep_partitions": int(args.sweep_partitions)},
        "idempotency": {"skip_if_exists": int(args.skip_if_exists), "force": int(args.force)},
    }
    write_text_to_s3(f"{out_root}/_stage_config_{args.experiment_id}.json", json.dumps(cfg, indent=2))

    # Read back from written parquet to get counts efficiently (avoids recomputation)
    metrics_saved = spark.read.parquet(met_path)
    counts = metrics_saved.groupBy("cutout_ok").count().collect()
    ok = sum(r["count"] for r in counts if r["cutout_ok"] == 1)
    bad = sum(r["count"] for r in counts if r["cutout_ok"] == 0)
    print(f"[4c] Done. ok={ok} bad={bad}. Output: {out_path}")


# --------------------------
# Stage 4d: Completeness summaries
# --------------------------

def stage_4d_completeness(spark: SparkSession, args: argparse.Namespace) -> None:
    out_root = f"{args.output_s3.rstrip('/')}/phase4d/{args.variant}"
    # NOTE: Per-experiment checks happen via --experiment-id.
    # Do NOT skip entire stage based on output dir existing.

    if not args.experiment_id:
        raise ValueError("--experiment-id is required for stage 4d")

    met_path = f"{args.output_s3.rstrip('/')}/phase4c/{args.variant}/metrics/{args.experiment_id}"
    df = read_parquet_safe(spark, met_path)

    # Recovery proxy
    snr_th = float(args.recovery_snr_thresh)
    sep_th = float(args.recovery_theta_over_psf)

    df = df.withColumn("theta_over_psf", F.when(F.col("psfsize_r").isNotNull() & (F.col("psfsize_r") > 0), F.col("theta_e_arcsec") / F.col("psfsize_r")).otherwise(F.lit(None)))
    df = df.withColumn(
        "recovered",
        ((F.col("cutout_ok") == 1) & (F.col("theta_e_arcsec") > 0) & (F.col("arc_snr") >= F.lit(snr_th)) & (F.col("theta_over_psf") >= F.lit(sep_th))).cast("int"),
    )

    # Bin observing conditions (coarse)
    df = df.withColumn("psf_bin", F.floor(F.col("psfsize_r") * 10).cast("int"))
    df = df.withColumn("depth_bin", F.floor(F.col("psfdepth_r") * 2).cast("int"))

    grp = df.groupBy(
        "region_split", "selection_set_id", "ranking_mode",
        "theta_e_arcsec", "src_dmag", "src_reff_arcsec", "src_e", "shear",
        "psf_bin", "depth_bin",
    ).agg(
        F.count(F.lit(1)).alias("n"),
        F.sum("recovered").alias("n_recovered"),
        F.avg("arc_snr").alias("arc_snr_mean"),
        F.expr("percentile_approx(arc_snr, 0.5)").alias("arc_snr_p50"),
    )

    grp = grp.withColumn("completeness", F.col("n_recovered") / F.col("n"))

    out_path = f"{out_root}/completeness/{args.experiment_id}"
    grp.write.mode("overwrite").partitionBy("region_split").parquet(out_path)

    cfg = {
        "stage": "4d",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variant": args.variant,
        "output_s3": args.output_s3,
        "experiment_id": args.experiment_id,
        "inputs": {"metrics": met_path},
        "recovery": {"snr_thresh": snr_th, "theta_over_psf": sep_th},
        "idempotency": {"skip_if_exists": int(args.skip_if_exists), "force": int(args.force)},
    }
    write_text_to_s3(f"{out_root}/_stage_config_{args.experiment_id}.json", json.dumps(cfg, indent=2))

    print(f"[4d] Done. Output: {out_path}")


# --------------------------
# Stage 4p5: Compaction
# --------------------------

def stage_4p5_compact(spark: SparkSession, args: argparse.Namespace) -> None:
    out_root = f"{args.output_s3.rstrip('/')}/phase4p5/{args.variant}"
    # NOTE: Skip logic should check compact_output_s3, not out_root.
    # Check actual output path for _SUCCESS marker.
    if args.compact_output_s3 and stage_should_skip(args.compact_output_s3, args.skip_if_exists, args.force):
        print(f"[4p5] Skip (exists): {args.compact_output_s3}")
        return

    if not args.compact_input_s3:
        raise ValueError("--compact-input-s3 is required for stage 4p5")
    if not args.compact_output_s3:
        raise ValueError("--compact-output-s3 is required for stage 4p5")

    df = read_parquet_safe(spark, args.compact_input_s3)

    # Coalesce by target partitions (optional)
    n = int(args.compact_partitions)
    if n > 0:
        df = df.repartition(n)

    df.write.mode("overwrite").parquet(args.compact_output_s3)

    cfg = {
        "stage": "4p5",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variant": args.variant,
        "output_s3": args.output_s3,
        "input_s3": args.compact_input_s3,
        "output_compact_s3": args.compact_output_s3,
        "compact_partitions": n,
        "idempotency": {"skip_if_exists": int(args.skip_if_exists), "force": int(args.force)},
    }
    write_text_to_s3(f"{out_root}/_stage_config.json", json.dumps(cfg, indent=2))

    print(f"[4p5] Done. Output: {args.compact_output_s3}")


# --------------------------
# CLI
# --------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Spark Phase 4 pipeline")
    p.add_argument("--stage", required=True, choices=["4a", "4b", "4c", "4d", "4p5"], help="Phase 4 stage")
    p.add_argument("--output-s3", required=True, help="Base output prefix (s3://...)")
    p.add_argument("--variant", required=True, help="LRG variant name (e.g., v3_color_relaxed)")

    # Idempotency
    p.add_argument("--skip-if-exists", type=int, default=1)
    p.add_argument("--force", type=int, default=0)

    # Common imaging/caching
    p.add_argument("--bands", default="g,r,z", help="Bands to use for coadds")
    p.add_argument("--coadd-base-url", default="https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/coadd", help="Base URL for DR10 South coadds")
    p.add_argument("--coadd-s3-cache-prefix", default="s3://darkhaloscope/dr10/coadd_cache", help="Where to store cached coadd files")

    # Stage 4a inputs
    p.add_argument("--parent-s3", default="", help="Phase 3.5 parent catalog parquet path")
    p.add_argument("--bricks-with-region-s3", default="", help="Phase 3a bricks_with_region parquet path")
    p.add_argument("--region-selections-s3", default="", help="Phase 3b region_selections parquet path")
    p.add_argument("--selection-set-ids", default="*", help="Comma list or *")
    p.add_argument("--require-lrg-flag", default="is_v3_color_relaxed", help="If set, only keep rows where this column==1")

    p.add_argument("--stamp-sizes", default="64")
    p.add_argument("--bandsets", default="grz", help="Comma list of r or grz")


    # Stage 4c rendering: PSF model for injection convolution
    p.add_argument("--psf-model", default="moffat", choices=["gaussian", "moffat"],
                   help="PSF model used for injected source convolution. 'moffat' is more realistic for DECaLS-like PSF wings.")
    p.add_argument("--moffat-beta", type=float, default=3.5,
                   help="Moffat beta parameter (typical 3.0-4.5). Only used when --psf-model=moffat.")

    # Gen5: COSMOS source integration
    p.add_argument("--config", help="Path to JSON config file (local or s3://). Overrides other args.")
    p.add_argument("--source-mode", default="sersic", choices=["sersic", "cosmos"],
                   help="Source morphology: 'sersic' (parametric) or 'cosmos' (GalSim RealGalaxy)")
    p.add_argument("--cosmos-bank-h5", default=None,
                   help="Path to HDF5 COSMOS bank file (required if --source-mode=cosmos)")
    p.add_argument("--cosmos-salt", default="",
                   help="Optional salt for deterministic COSMOS template selection")
    p.add_argument("--seed-base", type=int, default=42,
                   help="Base seed for reproducible randomness in deterministic operations")

    p.add_argument("--tiers", default="debug,grid,train", help="Comma list among debug,grid,train")
    p.add_argument("--grid-debug", default="grid_small")
    p.add_argument("--grid-grid", default="grid_medium")
    p.add_argument("--grid-train", default="grid_small")

    p.add_argument("--n-per-config-debug", type=int, default=5)
    p.add_argument("--n-per-config-grid", type=int, default=25)
    p.add_argument("--n-total-train-per-split", type=int, default=200000)
    p.add_argument("--control-frac-train", type=float, default=0.50)
    p.add_argument("--control-frac-grid", type=float, default=0.10,
                   help="Fraction of grid tier samples to mark as controls (theta_e=0)")
    p.add_argument("--control-frac-debug", type=float, default=0.0,
                   help="Fraction of debug tier samples to mark as controls (theta_e=0)")
    p.add_argument("--unpaired-controls", type=int, default=1,
                   help="If 1, use unpaired controls from DIFFERENT galaxy positions (harder negatives). "
                        "If 0, controls use the same galaxy position without lens injection (easier).")
    p.add_argument("--max-total-tasks-soft", type=int, default=30_000_000,
                   help="Soft limit on total estimated tasks to catch accidental explosions. Set to 0 to disable.")

    p.add_argument("--replicates", type=int, default=2)
    p.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED,
                   help=f"Seed for deterministic sampling (default: {DEFAULT_SPLIT_SEED})")

    # Stage 4b / 4b2
    p.add_argument("--cache-partitions", type=int, default=200)
    p.add_argument("--http-timeout-s", type=int, default=180)
    p.add_argument("--include-psfsize", type=int, default=0,
                   help="If 1, download psfsize maps (per-pixel PSF FWHM) for spatially varying PSF")
    p.add_argument("--use-psfsize-maps", type=int, default=0,
                   help="Stage 4c: If 1, load psfsize maps for center-evaluated PSF (requires 4b with --include-psfsize 1)")

    # Stage 4c
    p.add_argument("--experiment-id", default="", help="Experiment id under phase4a/manifests")
    p.add_argument("--manifests-subdir", default="manifests",
                   help="Subdirectory under phase4a/ to read manifests from ('manifests' is standard, 'manifests_filtered' for blacklist-filtered if available)")
    p.add_argument("--sweep-partitions", type=int, default=600)
    # NOTE: --src-flux-scale is deprecated. Flux is now computed correctly using
    # mag_to_nMgy() with AB ZP=22.5 (Legacy Survey nanomaggy convention).
    # =========================================================================
    # METRICS-ONLY MODE:
    # When enabled, stage 4c computes injection metrics (arc_snr, cutout_ok, etc.)
    # but does NOT store the actual stamp images. This dramatically reduces storage
    # for the grid tier where we need completeness statistics but not training data.
    #
    # Storage savings: Grid tier with 1M tasks at ~20KB/stamp = ~20GB
    #                  With metrics-only: ~1M × ~200B = ~200MB (100× reduction)
    #
    # Use --metrics-only 1 for grid tier, --metrics-only 0 for train tier.
    # =========================================================================
    p.add_argument("--metrics-only", type=int, default=0, 
                   help="If 1, skip saving stamp images (for grid tier completeness analysis)")
    # =========================================================================
    # LENS MODEL SELECTION:
    # The lens model is now determined by the manifest's `lens_model` column,
    # which is set in Stage 4a. This ensures reproducibility.
    # --use-sie is kept for backwards compatibility but is effectively ignored.
    # The new dependency-free SIE implementation works without lenstronomy.
    # =========================================================================
    p.add_argument("--use-sie", type=int, default=1,
                   help="DEPRECATED: Lens model is now read from manifest. Kept for backward compatibility.")

    # Stage 4d
    p.add_argument("--recovery-snr-thresh", type=float, default=5.0)
    p.add_argument("--recovery-theta-over-psf", type=float, default=0.8)

    # Stage 4p5
    p.add_argument("--compact-input-s3", default="")
    p.add_argument("--compact-output-s3", default="")
    p.add_argument("--compact-partitions", type=int, default=200)

    return p


def main() -> None:
    args = build_parser().parse_args()

    # Gen5: Load config file if provided and override args
    if args.config:
        cfg = load_phase4c_config(args.config)
        # Override args with config values
        for key, val in cfg.items():
            # Convert key from snake_case to arg format
            arg_key = key
            if hasattr(args, arg_key):
                setattr(args, arg_key, val)
                print(f"[CONFIG] Overriding {arg_key} = {val}")

    # Gen5: Validate COSMOS requirements
    if hasattr(args, 'source_mode') and args.source_mode == "cosmos":
        if not args.cosmos_bank_h5:
            raise ValueError("--cosmos-bank-h5 required when --source-mode=cosmos")
        print(f"[GEN5] COSMOS mode enabled: {args.cosmos_bank_h5}")

    # Validate required inputs per stage
    stage = args.stage
    if stage == "4a":
        for x in ["parent_s3", "bricks_with_region_s3", "region_selections_s3"]:
            if not getattr(args, x):
                raise ValueError(f"--{x.replace('_','-')} is required for stage 4a")
    elif stage == "4c":
        if not args.experiment_id:
            raise ValueError("--experiment-id is required for stage 4c")
    elif stage == "4d":
        if not args.experiment_id:
            raise ValueError("--experiment-id is required for stage 4d")

    spark = SparkSession.builder.appName(f"phase4_{stage}_{args.variant}").getOrCreate()

    # Keep Spark defaults unless user sets externally
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    # Gen5: Save effective config to S3 for audit trail (Stage 4c only)
    if stage == "4c" and _is_s3(args.output_s3):
        effective_config = {
            "stage": args.stage,
            "variant": args.variant,
            "experiment_id": args.experiment_id,
            "source_mode": getattr(args, 'source_mode', 'sersic'),
            "cosmos_bank_h5": getattr(args, 'cosmos_bank_h5', None),
            "cosmos_salt": getattr(args, 'cosmos_salt', ''),
            "seed_base": getattr(args, 'seed_base', 42),
            "psf_model": args.psf_model,
            "moffat_beta": args.moffat_beta,
            "split_seed": args.split_seed,
            "execution_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "spark_version": spark.version,
        }
        
        # Write to S3
        config_s3_path = f"{args.output_s3}/phase4c/{args.variant}/run_config_{args.experiment_id}.json"
        try:
            import boto3
            s3 = boto3.client("s3")
            bucket, key = config_s3_path.replace("s3://", "").split("/", 1)
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(effective_config, indent=2),
                ContentType="application/json"
            )
            print(f"[GEN5] Saved effective config to: {config_s3_path}")
        except Exception as e:
            print(f"[GEN5] Warning: Could not save config to S3: {e}")

    if stage == "4a":
        stage_4a_build_manifests(spark, args)
    elif stage == "4b":
        stage_4b_cache_coadds(spark, args)
    elif stage == "4c":
        stage_4c_inject_cutouts(spark, args)
    elif stage == "4d":
        stage_4d_completeness(spark, args)
    elif stage == "4p5":
        stage_4p5_compact(spark, args)
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    spark.stop()


if __name__ == "__main__":
    main()

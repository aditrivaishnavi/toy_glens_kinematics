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


# --------------------------
# S3 utilities
# --------------------------

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
) -> np.ndarray:
    """Return a lensed-source surface brightness stamp (float32) to add to an image.

    Simplified but deterministic SIS lens with optional external shear.
    - Lens center at stamp center.
    - Source center randomly offset within a small box.

    This is intended for injection-recovery and for building a training set.
    """
    h, w = stamp_shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    # Pixel coordinates in arcsec relative to center
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    thx = (xx - cx) * PIX_SCALE_ARCSEC
    thy = (yy - cy) * PIX_SCALE_ARCSEC

    # Shear (simple, aligned)
    if abs(shear) > 0:
        thx2 = thx + shear * thx
        thy2 = thy - shear * thy
    else:
        thx2, thy2 = thx, thy

    r = np.sqrt(thx2**2 + thy2**2) + 1e-6
    alpha = theta_e_arcsec / r

    # SIS deflection
    betax = thx2 - alpha * thx2
    betay = thy2 - alpha * thy2

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
    Generate a lensed source stamp using SIE lens model via lenstronomy.
    
    Returns:
        Tuple of (stamp array, physics_metrics dict)
        
    Physics metrics include:
        - magnification: Total magnification of the source
        - tangential_stretch: Tangential eigenvalue of magnification tensor
        - radial_stretch: Radial eigenvalue
        - expected_arc_radius: Approximate arc radius in arcsec
    """
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
    
    kwargs_lens = [
        {'theta_E': theta_e_arcsec, 'e1': e1_lens, 'e2': e2_lens, 'center_x': 0, 'center_y': 0},
        {'gamma1': shear, 'gamma2': 0, 'ra_0': 0, 'dec_0': 0}
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
    # Use (theta_e, 0) as representative point - this is where arcs form
    try:
        # Image-plane position at Einstein radius (not source position!)
        image_x = theta_e_arcsec  # At Einstein radius
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
    
    # Rule 2: Magnification check
    if magnification is not None:
        if magnification < 1.0:
            warnings.append(f"Magnification ({magnification:.2f}) < 1: not strong lensing")
            is_valid = False
        elif magnification > 1000:
            warnings.append(f"Magnification ({magnification:.2f}) > 1000: near caustic, may be unrealistic")
    
    # Rule 3: Tangential stretch check
    if tangential_stretch is not None:
        if tangential_stretch < 1.0:
            warnings.append(f"Tangential stretch ({tangential_stretch:.2f}) < 1: no arc elongation")
    
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

def build_coadd_urls(coadd_base_url: str, brickname: str, bands: List[str]) -> Dict[str, str]:
    """Return URLs for image/invvar/maskbits for each band."""
    # DR10 structure: .../dr10/south/coadd/000/000p025/legacysurvey-000p025-image-g.fits.fz
    # with directory coadd/<brickname[:3]>/<brickname>/
    p3 = brickname[:3]
    base = coadd_base_url.rstrip("/")
    d = f"{base}/{p3}/{brickname}"
    out: Dict[str, str] = {}
    for b in bands:
        out[f"image_{b}"] = f"{d}/legacysurvey-{brickname}-image-{b}.fits.fz"
        out[f"invvar_{b}"] = f"{d}/legacysurvey-{brickname}-invvar-{b}.fits.fz"
        out[f"maskbits"] = f"{d}/legacysurvey-{brickname}-maskbits.fits.fz"
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
    for c in ["brickname", "psfsize_r", "psfdepth_r", "ebv"]:
        if c not in df_bricks.columns:
            raise RuntimeError(f"bricks_with_region missing required column: {c}")

    df_bricks = df_bricks.select("brickname", "psfsize_r", "psfdepth_r", "ebv")
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
        "sampling": {
            "selection_set_ids": args.selection_set_ids,
            "require_lrg_flag": args.require_lrg_flag,
            "psf_edges": psf_edges,
            "depth_edges": depth_edges,
        },
        "tiers": {
            "debug": {"grid": args.grid_debug, "n_per_config": args.n_per_config_debug},
            "grid": {"grid": args.grid_grid, "n_per_config": args.n_per_config_grid},
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
                    control_frac = 0.0
                elif tier == "grid":
                    grid_name = args.grid_grid
                    n_per_config = int(args.n_per_config_grid)
                    n_total = None
                    control_frac = 0.0
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
                    wbin = Window.partitionBy("selection_set_id", "region_split", "psf_bin", "depth_bin").orderBy(F.rand(args.split_seed))
                    tmp = base.withColumn("rn_bin", F.row_number().over(wbin))
                    tmp = tmp.filter(F.col("rn_bin") <= F.lit(per_bin))

                    # Second pass: cap to exact per_split_target per (selection_set, split)
                    wtot = Window.partitionBy("selection_set_id", "region_split").orderBy(F.rand(args.split_seed + 1))
                    tmp = tmp.withColumn("rn_tot", F.row_number().over(wtot))
                    tmp = tmp.filter(F.col("rn_tot") <= F.lit(per_split_target))

                    # CrossJoin: each selected galaxy gets every config from the grid.
                    # This is where the n_cfg multiplication happens (correctly).
                    # Final tasks per (selection_set, split) = per_split_target × n_cfg
                    tasks = tmp.crossJoin(F.broadcast(cfg_df))

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
                    wbin = Window.partitionBy("selection_set_id", "region_split", "psf_bin", "depth_bin").orderBy(F.rand(args.split_seed))
                    per_bin = int(math.ceil(per_split_target / 16))
                    tmp = base.withColumn("rn_bin", F.row_number().over(wbin)).filter(F.col("rn_bin") <= F.lit(per_bin))
                    
                    # Cap to exact per_split_target per (selection_set, split)
                    wtot = Window.partitionBy("selection_set_id", "region_split").orderBy(F.rand(args.split_seed + 1))
                    tmp = tmp.withColumn("rn_tot", F.row_number().over(wtot)).filter(F.col("rn_tot") <= F.lit(per_split_target))

                    # Control assignment: randomly mark control_frac of samples as controls
                    # Controls have theta_e=0 (no injection) and serve as negative examples
                    tmp = tmp.withColumn("is_control", (F.rand(args.split_seed + 7) < F.lit(control_frac)).cast("int"))
                    
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
                    "psfsize_r", "psfdepth_r", "ebv", "psf_bin", "depth_bin",
                    "config_id", "theta_e_arcsec", "src_dmag", "src_reff_arcsec", "src_e", "shear",
                    "stamp_size", "bandset", "replicate",
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
    print(f"[4b] Starting coadd cache for {total_bricks} unique bricks")

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
        c = _s3_client()
        for r in it:
            brick = r["brickname"]
            prefix = f"{s3_cache}/{brick}"
            try:
                # Skip if already cached
                if (not args.force) and s3_prefix_exists(prefix + "/"):
                    yield Row(brickname=brick, ok=1, error=None, s3_prefix=prefix + "/")
                    continue

                urls = build_coadd_urls(coadd_base, brick, bands)
                tmpdir = f"/mnt/tmp/phase4b_{brick}"
                os.makedirs(tmpdir, exist_ok=True)

                # Download and upload
                for k, url in urls.items():
                    fname = url.split("/")[-1]
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
        "idempotency": {"skip_if_exists": int(args.skip_if_exists), "force": int(args.force)},
    }
    write_text_to_s3(f"{out_root}/_stage_config.json", json.dumps(cfg, indent=2))

    # Basic success stats
    ok = df_out.filter(F.col("ok") == 1).count()
    bad = df_out.filter(F.col("ok") == 0).count()
    print(f"[4b] Cached bricks ok={ok} bad={bad}. Output: {out_root}")


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
    x0 = int(round(x)) - half
    y0 = int(round(y)) - half
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

    manifests_root = f"{args.output_s3.rstrip('/')}/phase4a/{args.variant}/manifests"
    if not args.experiment_id:
        raise ValueError("--experiment-id is required for stage 4c")
    in_path = f"{manifests_root}/{args.experiment_id}"
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
    
    # =========================================================================
    # LENS MODEL SELECTION:
    # SIE is more realistic (elliptical, proper ray tracing via lenstronomy)
    # SIS is simpler and faster (no lenstronomy dependency)
    # =========================================================================
    use_sie = bool(args.use_sie) and LENSTRONOMY_AVAILABLE
    if args.use_sie and not LENSTRONOMY_AVAILABLE:
        print(f"[4c] WARNING: --use-sie requested but lenstronomy not available. Falling back to SIS.")
    if use_sie:
        print(f"[4c] Using SIE lens model with physics validation (lenstronomy)")
    else:
        print(f"[4c] Using simplified SIS lens model")
    
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
        T.StructField("psfsize_r", T.DoubleType(), True),
        T.StructField("psfdepth_r", T.DoubleType(), True),
        T.StructField("ebv", T.DoubleType(), True),
        T.StructField("stamp_npz", T.BinaryType(), True),  # Nullable for metrics-only mode
        T.StructField("cutout_ok", T.IntegerType(), False),
        T.StructField("arc_snr", T.DoubleType(), True),
        T.StructField("metrics_only", T.IntegerType(), False),  # Track if stamp was skipped
        # Lens model provenance (for dataset consistency checks)
        T.StructField("lens_model", T.StringType(), True),  # "SIE" or "SIS" - tracks which model was used
        T.StructField("lens_e", T.DoubleType(), True),  # Lens ellipticity (sampled independently)
        # Physics validation metrics (from SIE injection)
        T.StructField("magnification", T.DoubleType(), True),
        T.StructField("tangential_stretch", T.DoubleType(), True),
        T.StructField("radial_stretch", T.DoubleType(), True),
        T.StructField("expected_arc_radius", T.DoubleType(), True),
        T.StructField("physics_valid", T.IntegerType(), True),  # 1 if passed validation
        T.StructField("physics_warnings", T.StringType(), True),  # Comma-separated warnings
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

                # Injection
                theta_e = float(r["theta_e_arcsec"])
                src_dmag = float(r["src_dmag"])
                src_reff = float(r["src_reff_arcsec"])
                src_e = float(r["src_e"])
                shear = float(r["shear"])

                # Use r-band for flux scaling
                # Flux ~ 10^(-0.4*mag), scale by an arbitrary constant because coadds are in nanomaggies.
                # We keep relative scaling; downstream can re-normalize.
                rmag = float(r["rmag"])
                flux0 = 10 ** (-0.4 * rmag)
                src_flux = flux0 * (10 ** (-0.4 * src_dmag)) * float(args.src_flux_scale)

                rng = np.random.RandomState(int((hash(r["task_id"]) & 0xFFFFFFFF) ^ int(r["replicate"])) )

                add = None
                arc_snr = None
                lens_model_str = None  # Will be "SIE" or "SIS"
                lens_e_val = None  # Lens ellipticity (only set for SIE)
                physics_metrics = {
                    "magnification": None,
                    "tangential_stretch": None,
                    "radial_stretch": None,
                    "expected_arc_radius": None,
                }
                physics_valid = None
                physics_warnings_str = None
                
                if theta_e > 0 and src_flux > 0:
                    # Approximate PSF FWHM from brick psfsize_r if available
                    psf_fwhm = float(r["psfsize_r"]) if r["psfsize_r"] is not None else None
                    
                    # =========================================================================
                    # LENS MODEL SELECTION:
                    # SIE: More realistic, returns physics metrics for validation
                    # SIS: Simpler, faster, no physics metrics
                    # =========================================================================
                    if use_sie:
                        # Use SIE with lenstronomy - includes physics validation metrics
                        # FIX (2026-01-22): Sample lens ellipticity INDEPENDENTLY from source
                        # Lens and source shapes are physically uncorrelated, so coupling
                        # them (e.g., lens_e = src_e * 0.8) could bias the model.
                        # Sample lens_e from a realistic distribution for galaxy-scale lenses:
                        # - Mean ~0.2-0.3, range 0-0.5 (rare to have very elongated lenses)
                        lens_e_val = rng.uniform(0.05, 0.45)  # Independent of source
                        lens_model_str = "SIE"
                        add, physics_metrics = inject_sie_stamp(
                            (size, size), theta_e, src_flux, src_reff, 
                            src_e, lens_e_val, shear, rng, psf_fwhm_arcsec=psf_fwhm
                        )
                    else:
                        # Use simplified SIS (spherical, no lens ellipticity)
                        lens_model_str = "SIS"
                        lens_e_val = 0.0  # SIS is spherically symmetric
                        add = inject_sis_stamp(
                            (size, size), theta_e, src_flux, src_reff, 
                            src_e, shear, rng, psf_fwhm_arcsec=psf_fwhm
                        )

                    # Add to each band with a simple color scaling (bluer source)
                    for b in use_bands:
                        scale = 1.0
                        if b == "g":
                            scale = 1.2
                        elif b == "z":
                            scale = 0.8
                        imgs[b] = (imgs[b] + scale * add).astype(np.float32)

                    # Proxy SNR in r
                    invr = invs.get("r")
                    if invr is not None:
                        sigma = np.where(invr > 0, 1.0 / np.sqrt(invr + 1e-12), 0.0)
                        snr = np.where(sigma > 0, add / (sigma + 1e-12), 0.0)
                        arc_snr = float(np.nanmax(snr))
                    
                    # =========================================================================
                    # PHYSICS VALIDATION:
                    # Check that injection parameters are physically consistent.
                    # This helps catch bugs and ensures PINN training data is realistic.
                    # =========================================================================
                    physics_valid_bool, warnings = validate_physics_consistency(
                        theta_e, arc_snr,
                        physics_metrics.get("magnification"),
                        physics_metrics.get("tangential_stretch"),
                        psf_fwhm,
                    )
                    physics_valid = 1 if physics_valid_bool else 0
                    physics_warnings_str = "; ".join(warnings) if warnings else None

                # Encode stamp (or skip if metrics-only mode)
                if metrics_only:
                    stamp_npz = None  # Don't store stamp to save space
                else:
                    stamp_npz = encode_npz({f"image_{b}": imgs[b] for b in use_bands})

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
                    psfsize_r=float(r["psfsize_r"]) if r["psfsize_r"] is not None else None,
                    psfdepth_r=float(r["psfdepth_r"]) if r["psfdepth_r"] is not None else None,
                    ebv=float(r["ebv"]) if r["ebv"] is not None else None,
                    stamp_npz=stamp_npz,
                    cutout_ok=int(bool(cut_ok_all)),
                    arc_snr=arc_snr,
                    metrics_only=int(metrics_only),
                    lens_model=lens_model_str,
                    lens_e=lens_e_val,
                    magnification=physics_metrics.get("magnification"),
                    tangential_stretch=physics_metrics.get("tangential_stretch"),
                    radial_stretch=physics_metrics.get("radial_stretch"),
                    expected_arc_radius=physics_metrics.get("expected_arc_radius"),
                    physics_valid=physics_valid,
                    physics_warnings=physics_warnings_str,
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
                    psfsize_r=float(r["psfsize_r"]) if r["psfsize_r"] is not None else None,
                    psfdepth_r=float(r["psfdepth_r"]) if r["psfdepth_r"] is not None else None,
                    ebv=float(r["ebv"]) if r["ebv"] is not None else None,
                    stamp_npz=empty,
                    cutout_ok=0,
                    arc_snr=None,
                    metrics_only=int(metrics_only),
                    lens_model="SIE" if use_sie else "SIS",  # Record intended model
                    lens_e=None,  # Unknown due to error
                    magnification=None,
                    tangential_stretch=None,
                    radial_stretch=None,
                    expected_arc_radius=None,
                    physics_valid=0,
                    physics_warnings=f"Processing error: {str(e)[:200]}",
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
        "src_flux_scale": float(args.src_flux_scale),
        "spark": {"sweep_partitions": int(args.sweep_partitions)},
        "idempotency": {"skip_if_exists": int(args.skip_if_exists), "force": int(args.force)},
    }
    write_text_to_s3(f"{out_root}/_stage_config_{args.experiment_id}.json", json.dumps(cfg, indent=2))

    ok = metrics.filter(F.col("cutout_ok") == 1).count()
    bad = metrics.filter(F.col("cutout_ok") == 0).count()
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

    p.add_argument("--tiers", default="debug,grid,train", help="Comma list among debug,grid,train")
    p.add_argument("--grid-debug", default="grid_small")
    p.add_argument("--grid-grid", default="grid_medium")
    p.add_argument("--grid-train", default="grid_small")

    p.add_argument("--n-per-config-debug", type=int, default=5)
    p.add_argument("--n-per-config-grid", type=int, default=25)
    p.add_argument("--n-total-train-per-split", type=int, default=200000)
    p.add_argument("--control-frac-train", type=float, default=0.25)

    p.add_argument("--replicates", type=int, default=2)
    p.add_argument("--split-seed", type=int, default=13)

    # Stage 4b
    p.add_argument("--cache-partitions", type=int, default=200)
    p.add_argument("--http-timeout-s", type=int, default=180)

    # Stage 4c
    p.add_argument("--experiment-id", default="", help="Experiment id under phase4a/manifests")
    p.add_argument("--sweep-partitions", type=int, default=600)
    p.add_argument("--src-flux-scale", type=float, default=1e6, help="Arbitrary scaling from mags to coadd units")
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
    # --use-sie 1: Use SIE (Singular Isothermal Ellipsoid) via lenstronomy
    #              More realistic, includes ellipticity, proper ray tracing
    # --use-sie 0: Use simplified SIS (faster, no lenstronomy dependency)
    # =========================================================================
    p.add_argument("--use-sie", type=int, default=1,
                   help="If 1, use SIE lens model via lenstronomy (more realistic). If 0, use SIS (simpler).")

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

#!/usr/bin/env python3
"""
Spark Job: Validate Cutouts and Detect Shortcuts

Performs quality validation and shortcut detection on generated cutouts.
Implements LLM-recommended gates from Section E.

Usage:
    spark-submit --deploy-mode cluster spark_validate_cutouts.py \
        --positives s3://darkhaloscope/stronglens_calibration/cutouts/positives/ \
        --negatives s3://darkhaloscope/stronglens_calibration/cutouts/negatives/ \
        --output s3://darkhaloscope/stronglens_calibration/validation/

Author: Generated for stronglens_calibration project
Date: 2026-02-08
"""
import argparse
import io
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

PIPELINE_VERSION = "1.0.0"

# AWS Configuration (environment override supported)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

# Cutout specs
CUTOUT_SIZE = 101
CENTER = CUTOUT_SIZE // 2
PIXEL_SCALE = 0.262  # arcsec/pixel

# Multi-radius core analysis (LLM recommendation: r=4/8/12 pixels)
CORE_RADII = [4, 8, 12]  # pixels (~1.0", ~2.1", ~3.1")
CORE_RADIUS = 8  # default for backward compat

# Arc annulus (LLM fix: 4-16px for θ_E range 0.5"-3.0", not 20-40px)
ARC_ANNULUS_INNER = 4   # pixels (~1.0")
ARC_ANNULUS_OUTER = 16  # pixels (~4.2")

# Secondary outer annulus for background
OUTER_ANNULUS_INNER = 16  # pixels
OUTER_ANNULUS_OUTER = 28  # pixels
OUTER_MARGIN = 25  # pixels from edge

# Quality thresholds
MAX_NAN_FRAC_CENTER = 0.02  # 2% max NaN in central 50x50
MAX_NAN_FRAC_CORE = 0.001   # <0.1% NaN in core (LLM: stricter for core)
MAX_BAD_PIXEL_FRAC = 0.02   # 2% max bad pixels

# Shortcut detection thresholds
AUC_THRESHOLD_RED = 0.70    # Red flag (definite shortcut)
AUC_THRESHOLD_YELLOW = 0.60 # Yellow flag (potential shortcut)
AUC_THRESHOLD = AUC_THRESHOLD_RED  # Backward compat

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Validate] %(levelname)s: %(message)s",
)
logger = logging.getLogger("Validate")


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def load_cutout_from_s3(s3_client, bucket: str, key: str) -> Tuple[Optional[np.ndarray], Dict]:
    """Load cutout NPZ from S3."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        data = response["Body"].read()
        
        npz = np.load(io.BytesIO(data))
        cutout = npz["cutout"]
        
        # Extract metadata
        meta = {}
        for k in npz.files:
            if k.startswith("meta_"):
                meta[k[5:]] = npz[k].item() if npz[k].ndim == 0 else npz[k].tolist()
        
        return cutout, meta
    except Exception as e:
        return None, {"error": str(e)}


def validate_cutout_quality(cutout: np.ndarray) -> Dict:
    """
    Validate cutout quality per LLM recommendations.
    
    LLM guidance:
    - 2% NaN in central 50x50 is reasonable
    - Stricter constraint for core: <0.1% NaN
    - Track band-wise NaN fraction
    - One missing band = automatic reject
    
    Returns dict with quality metrics and pass/fail flags.
    """
    h, w, c = cutout.shape
    
    # Size check
    size_ok = (h == CUTOUT_SIZE and w == CUTOUT_SIZE and c == 3)
    
    # NaN analysis - overall
    nan_mask = np.isnan(cutout)
    total_nan_frac = np.sum(nan_mask) / cutout.size
    
    # Central region (50x50)
    center = h // 2
    margin = 25
    central = cutout[center-margin:center+margin, center-margin:center+margin, :]
    central_nan_frac = np.sum(np.isnan(central)) / central.size
    
    # Core region (16x16 around center) - LLM: stricter constraint
    core = cutout[center-CORE_RADIUS:center+CORE_RADIUS, 
                  center-CORE_RADIUS:center+CORE_RADIUS, :]
    core_nan_frac = np.sum(np.isnan(core)) / core.size
    core_nan_ok = core_nan_frac < MAX_NAN_FRAC_CORE  # <0.1%
    
    # Band-wise NaN fraction (LLM recommendation)
    band_nan_fracs = []
    for i in range(c):
        band_data = cutout[:, :, i]
        band_nan_fracs.append(float(np.sum(np.isnan(band_data)) / band_data.size))
    
    # Bands present (non-empty)
    bands_present = [not np.all(np.isnan(cutout[:, :, i])) for i in range(c)]
    all_bands_present = all(bands_present)
    
    # Per-band central NaN
    central_band_nan = []
    for i in range(c):
        band_central = central[:, :, i]
        central_band_nan.append(float(np.sum(np.isnan(band_central)) / band_central.size))
    
    # Quality gate (enhanced with core constraint)
    quality_ok = (
        size_ok and
        central_nan_frac < MAX_NAN_FRAC_CENTER and
        core_nan_ok and  # LLM: stricter core constraint
        all_bands_present
    )
    
    return {
        "size_ok": bool(size_ok),
        "total_nan_frac": float(total_nan_frac),
        "central_nan_frac": float(central_nan_frac),
        "core_nan_frac": float(core_nan_frac),
        "core_nan_ok": bool(core_nan_ok),
        # Band-wise NaN (LLM recommendation)
        "nan_frac_g": float(band_nan_fracs[0]) if len(band_nan_fracs) > 0 else None,
        "nan_frac_r": float(band_nan_fracs[1]) if len(band_nan_fracs) > 1 else None,
        "nan_frac_z": float(band_nan_fracs[2]) if len(band_nan_fracs) > 2 else None,
        "central_nan_g": float(central_band_nan[0]) if len(central_band_nan) > 0 else None,
        "central_nan_r": float(central_band_nan[1]) if len(central_band_nan) > 1 else None,
        "central_nan_z": float(central_band_nan[2]) if len(central_band_nan) > 2 else None,
        # Band presence
        "has_g": bool(bands_present[0]),
        "has_r": bool(bands_present[1]),
        "has_z": bool(bands_present[2]),
        "all_bands_present": bool(all_bands_present),
        "quality_ok": bool(quality_ok),
    }


def extract_shortcut_features(cutout: np.ndarray) -> Dict:
    """
    Extract features for shortcut detection per LLM recommendations.
    
    LLM guidance:
    - "Annulus-only classifier should be strong (good signal)."
    - "Core-only and radial-profile shortcuts should be weak."
    - Use arc annulus 4-16px for θ_E range 0.5"-3.0"
    - Multi-radius core features at r=4/8/12 pixels
    - Add Laplacian, azimuthal asymmetry, color gradients
    """
    h, w, c = cutout.shape
    center = h // 2
    
    # Use r-band (index 1) for brightness features
    r_band = cutout[:, :, 1]
    g_band = cutout[:, :, 0]
    z_band = cutout[:, :, 2]
    
    # Create distance map from center
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - center)**2 + (x - center)**2)
    
    # Create angle map for azimuthal analysis
    y_full, x_full = np.mgrid[:h, :w]
    angle = np.arctan2(y_full - center, x_full - center)
    
    # =================================================================
    # MULTI-RADIUS CORE FEATURES (LLM: r=4/8/12 pixels)
    # =================================================================
    
    multi_core = {}
    for r in CORE_RADII:
        mask = dist <= r
        pixels = r_band[mask]
        valid = pixels[~np.isnan(pixels)]
        
        multi_core[f"core_r{r}_brightness"] = float(np.median(valid)) if len(valid) > 0 else None
        multi_core[f"core_r{r}_max"] = float(np.max(valid)) if len(valid) > 0 else None
        multi_core[f"core_r{r}_std"] = float(np.std(valid)) if len(valid) > 0 else None
    
    # Default core (r=8) for backward compat
    core_mask = dist <= CORE_RADIUS
    core_pixels = r_band[core_mask]
    valid_core = core_pixels[~np.isnan(core_pixels)]
    core_brightness = float(np.median(valid_core)) if len(valid_core) > 0 else None
    core_max = float(np.max(valid_core)) if len(valid_core) > 0 else None
    core_std = float(np.std(valid_core)) if len(valid_core) > 0 else None
    
    # =================================================================
    # ARC ANNULUS (LLM fix: 4-16px, not 20-40px)
    # =================================================================
    
    arc_mask = (dist >= ARC_ANNULUS_INNER) & (dist <= ARC_ANNULUS_OUTER)
    arc_pixels = r_band[arc_mask]
    valid_arc = arc_pixels[~np.isnan(arc_pixels)]
    
    arc_brightness = float(np.median(valid_arc)) if len(valid_arc) > 0 else None
    arc_max = float(np.max(valid_arc)) if len(valid_arc) > 0 else None
    arc_std = float(np.std(valid_arc)) if len(valid_arc) > 0 else None
    
    # =================================================================
    # OUTER ANNULUS (background)
    # =================================================================
    
    outer_mask = (dist >= OUTER_ANNULUS_INNER) & (dist <= OUTER_ANNULUS_OUTER)
    outer_pixels = r_band[outer_mask]
    valid_outer = outer_pixels[~np.isnan(outer_pixels)]
    
    outer_brightness = float(np.median(valid_outer)) if len(valid_outer) > 0 else None
    outer_std = float(np.std(valid_outer)) if len(valid_outer) > 0 else None
    
    # Edge background
    edge_mask = dist >= (h // 2 - OUTER_MARGIN)
    edge_pixels = r_band[edge_mask]
    valid_edge = edge_pixels[~np.isnan(edge_pixels)]
    edge_brightness = float(np.median(valid_edge)) if len(valid_edge) > 0 else None
    
    # =================================================================
    # RADIAL GRADIENT
    # =================================================================
    
    if core_brightness is not None and outer_brightness is not None:
        radial_gradient = core_brightness - outer_brightness
    else:
        radial_gradient = None
    
    # Radial profile bins
    radial_bins = [0, 4, 8, 12, 16, 24, 32, 40]
    radial_profile = []
    for i in range(len(radial_bins) - 1):
        ring_mask = (dist >= radial_bins[i]) & (dist < radial_bins[i+1])
        ring_pixels = r_band[ring_mask]
        valid_ring = ring_pixels[~np.isnan(ring_pixels)]
        radial_profile.append(float(np.median(valid_ring)) if len(valid_ring) > 0 else None)
    
    # =================================================================
    # LAPLACIAN / HIGH-FREQUENCY FEATURES (LLM recommendation)
    # =================================================================
    
    laplacian_core = None
    laplacian_arc = None
    
    try:
        from scipy.ndimage import laplace
        # Laplacian on r-band (detect sharp edges)
        r_filled = np.nan_to_num(r_band, nan=0.0)
        lap = laplace(r_filled)
        
        # Laplacian energy in core
        core_lap = np.abs(lap[core_mask])
        laplacian_core = float(np.mean(core_lap)) if len(core_lap) > 0 else None
        
        # Laplacian energy in arc annulus
        arc_lap = np.abs(lap[arc_mask])
        laplacian_arc = float(np.mean(arc_lap)) if len(arc_lap) > 0 else None
    except ImportError:
        pass
    
    # =================================================================
    # AZIMUTHAL ASYMMETRY (LLM recommendation)
    # =================================================================
    
    azimuthal_asymmetry = None
    
    # Compute std of residuals after radial profile subtraction in arc annulus
    if arc_brightness is not None and len(valid_arc) > 10:
        # Coefficient of variation, but with robust denominator and clipping
        # Use abs(median) to handle negative values, and cap result at 10
        denom = abs(arc_brightness) if abs(arc_brightness) > 0.001 else 0.001
        raw_asymmetry = np.std(valid_arc) / denom
        azimuthal_asymmetry = float(min(raw_asymmetry, 10.0))  # Cap at 10
    
    # Azimuthal profile in 8 wedges
    n_wedges = 8
    wedge_medians = []
    for i in range(n_wedges):
        ang_min = -np.pi + i * 2 * np.pi / n_wedges
        ang_max = ang_min + 2 * np.pi / n_wedges
        wedge_mask = arc_mask & (angle >= ang_min) & (angle < ang_max)
        wedge_pixels = r_band[wedge_mask]
        valid_wedge = wedge_pixels[~np.isnan(wedge_pixels)]
        if len(valid_wedge) > 0:
            wedge_medians.append(float(np.median(valid_wedge)))
    
    azimuthal_wedge_std = float(np.std(wedge_medians)) if len(wedge_medians) >= 4 else None
    
    # =================================================================
    # COLOR FEATURES (LLM: arc-annulus color vs core)
    # =================================================================
    
    # Core colors
    valid_g_core = g_band[core_mask][~np.isnan(g_band[core_mask])]
    valid_z_core = z_band[core_mask][~np.isnan(z_band[core_mask])]
    valid_r_core = r_band[core_mask][~np.isnan(r_band[core_mask])]
    
    core_g = float(np.median(valid_g_core)) if len(valid_g_core) > 0 else None
    core_z = float(np.median(valid_z_core)) if len(valid_z_core) > 0 else None
    
    # Arc colors
    valid_g_arc = g_band[arc_mask][~np.isnan(g_band[arc_mask])]
    valid_z_arc = z_band[arc_mask][~np.isnan(z_band[arc_mask])]
    valid_r_arc = r_band[arc_mask][~np.isnan(r_band[arc_mask])]
    
    arc_g = float(np.median(valid_g_arc)) if len(valid_g_arc) > 0 else None
    arc_z = float(np.median(valid_z_arc)) if len(valid_z_arc) > 0 else None
    
    # Color gradients (g-r and r-z)
    core_g_minus_r = None
    arc_g_minus_r = None
    arc_minus_core_g_r = None  # Blue arcs vs red core
    
    if core_g is not None and core_brightness is not None:
        # Convert flux to approx color (assuming nanomaggies)
        with np.errstate(divide='ignore', invalid='ignore'):
            if core_g > 0 and core_brightness > 0:
                core_g_minus_r = float(-2.5 * np.log10(core_g / core_brightness))
            if arc_g is not None and arc_brightness is not None and arc_g > 0 and arc_brightness > 0:
                arc_g_minus_r = float(-2.5 * np.log10(arc_g / arc_brightness))
            if core_g_minus_r is not None and arc_g_minus_r is not None:
                arc_minus_core_g_r = arc_g_minus_r - core_g_minus_r  # Negative = bluer arcs
    
    # =================================================================
    # SATURATION / EXTREME PIXEL FRACTION (LLM recommendation)
    # =================================================================
    
    # Fraction of pixels above 99.9th percentile (detect bright star artifacts)
    valid_all = r_band[~np.isnan(r_band)]
    if len(valid_all) > 100:
        p999 = np.percentile(valid_all, 99.9)
        saturation_frac = float(np.sum(valid_all > p999) / len(valid_all))
    else:
        saturation_frac = None
    
    # =================================================================
    # EDGE ARTIFACT SCORE (LLM recommendation)
    # =================================================================
    
    # Difference between border median and central background
    edge_artifact_score = None
    if edge_brightness is not None and outer_brightness is not None:
        edge_artifact_score = float(abs(edge_brightness - outer_brightness))
    
    # =================================================================
    # NORMALIZATION SANITY STATS (LLM recommendation)
    # =================================================================
    
    norm_stats = {}
    for band_idx, band_name in enumerate(["g", "r", "z"]):
        band_data = cutout[:, :, band_idx]
        valid = band_data[~np.isnan(band_data)]
        if len(valid) > 0:
            norm_stats[f"{band_name}_p1"] = float(np.percentile(valid, 1))
            norm_stats[f"{band_name}_p50"] = float(np.percentile(valid, 50))
            norm_stats[f"{band_name}_p99"] = float(np.percentile(valid, 99))
            norm_stats[f"{band_name}_mad"] = float(np.median(np.abs(valid - np.median(valid))))
        else:
            norm_stats[f"{band_name}_p1"] = None
            norm_stats[f"{band_name}_p50"] = None
            norm_stats[f"{band_name}_p99"] = None
            norm_stats[f"{band_name}_mad"] = None
    
    # MAD for noise estimation (backward compat)
    mad_r = norm_stats.get("r_mad")
    
    # =================================================================
    # COMPILE RESULTS
    # =================================================================
    
    result = {
        # Core features (backward compat)
        "core_brightness_r": core_brightness,
        "core_max_r": core_max,
        "core_std_r": core_std,
        # Arc annulus (corrected radii)
        "annulus_brightness_r": arc_brightness,
        "annulus_max_r": arc_max,
        "annulus_std_r": arc_std,
        # Outer / background
        "outer_brightness_r": outer_brightness,
        "outer_std_r": outer_std,
        "edge_brightness_r": edge_brightness,
        # Gradients
        "radial_gradient_r": radial_gradient,
        "radial_profile": radial_profile,
        # Colors
        "core_g": core_g,
        "core_z": core_z,
        "arc_g": arc_g,
        "arc_z": arc_z,
        "core_g_minus_r": core_g_minus_r,
        "arc_g_minus_r": arc_g_minus_r,
        "arc_minus_core_g_r": arc_minus_core_g_r,
        # High-frequency (Laplacian)
        "laplacian_core": laplacian_core,
        "laplacian_arc": laplacian_arc,
        # Azimuthal
        "azimuthal_asymmetry": azimuthal_asymmetry,
        "azimuthal_wedge_std": azimuthal_wedge_std,
        # Artifacts
        "saturation_frac": saturation_frac,
        "edge_artifact_score": edge_artifact_score,
        # Noise
        "mad_r": mad_r,
        # Multi-radius
        **multi_core,
        # Normalization
        **norm_stats,
    }
    
    return result


def compute_auc_single(pos: np.ndarray, neg: np.ndarray) -> float:
    """Compute AUC using Mann-Whitney U statistic via ranking (O(n log n) instead of O(n*m))."""
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    
    n1, n2 = len(pos), len(neg)
    
    # Use ranking method: O((n1+n2) log(n1+n2)) instead of O(n1*n2)
    combined = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(combined)) + 1  # 1-indexed ranks
    
    # Sum of ranks for positive class
    r1 = np.sum(ranks[:n1])
    
    # Mann-Whitney U statistic
    u = r1 - n1 * (n1 + 1) / 2
    
    auc = u / (n1 * n2)
    
    # Return max(auc, 1-auc) since direction doesn't matter for shortcut detection
    return max(auc, 1 - auc)


def compute_auc(pos_values: List[float], neg_values: List[float], 
                n_bootstrap: int = 200) -> Dict:
    """
    Compute AUC with bootstrap confidence interval (LLM recommendation).
    
    Returns dict with:
    - auc: point estimate
    - auc_lo: 16th percentile (lower 68% CI)
    - auc_hi: 84th percentile (upper 68% CI)
    
    Flag shortcut if lower CI bound > threshold.
    """
    pos = np.array([v for v in pos_values if v is not None and not np.isnan(v)])
    neg = np.array([v for v in neg_values if v is not None and not np.isnan(v)])
    
    if len(pos) == 0 or len(neg) == 0:
        return {"auc": 0.5, "auc_lo": 0.5, "auc_hi": 0.5}
    
    # Point estimate
    auc = compute_auc_single(pos, neg)
    
    # Bootstrap CI (LLM: 200-500 resamples)
    if n_bootstrap > 0 and len(pos) >= 10 and len(neg) >= 10:
        rng = np.random.default_rng(42)
        bootstrap_aucs = []
        
        for _ in range(n_bootstrap):
            pos_boot = rng.choice(pos, size=len(pos), replace=True)
            neg_boot = rng.choice(neg, size=len(neg), replace=True)
            bootstrap_aucs.append(compute_auc_single(pos_boot, neg_boot))
        
        auc_lo = float(np.percentile(bootstrap_aucs, 16))
        auc_hi = float(np.percentile(bootstrap_aucs, 84))
    else:
        auc_lo = auc
        auc_hi = auc
    
    return {
        "auc": float(auc),
        "auc_lo": auc_lo,
        "auc_hi": auc_hi,
    }


# =============================================================================
# SPARK PROCESSING
# =============================================================================

def process_cutout_file(
    s3_client,
    bucket: str,
    key: str,
    cutout_type: str,
) -> Dict:
    """Process a single cutout file."""
    cutout, meta = load_cutout_from_s3(s3_client, bucket, key)
    
    if cutout is None:
        return {
            "key": key,
            "type": cutout_type,
            "load_error": meta.get("error"),
            "quality_ok": False,
        }
    
    # Quality validation
    quality = validate_cutout_quality(cutout)
    
    # Shortcut features
    features = extract_shortcut_features(cutout)
    
    return {
        "key": key,
        "type": cutout_type,
        "galaxy_id": meta.get("galaxy_id", os.path.basename(key).replace(".npz", "")),
        **quality,
        **features,
        **{f"meta_{k}": v for k, v in meta.items() if k not in ["galaxy_id"]},
    }


def main():
    parser = argparse.ArgumentParser(description="Validate cutouts and detect shortcuts")
    parser.add_argument("--positives", required=True, help="S3 path to positive cutouts")
    parser.add_argument("--negatives", required=True, help="S3 path to negative cutouts")
    parser.add_argument("--output", required=True, help="S3 output path for reports")
    parser.add_argument("--sample", type=int, default=0, 
                        help="Sample N files per class for mini run (0 = all)")
    parser.add_argument("--bootstrap", type=int, default=200,
                        help="Number of bootstrap samples for AUC CI (default: 200)")
    
    args = parser.parse_args()
    
    # Initialize Spark
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    
    spark = SparkSession.builder \
        .appName("ValidateCutouts") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    sc = spark.sparkContext
    
    logger.info("=" * 60)
    logger.info("Starting Cutout Validation")
    logger.info("=" * 60)
    logger.info(f"Positives: {args.positives}")
    logger.info(f"Negatives: {args.negatives}")
    logger.info(f"Output: {args.output}")
    
    start_time = time.time()
    
    try:
        import boto3
        
        # Parse paths
        output_bucket = args.output.replace("s3://", "").split("/")[0]
        output_prefix = "/".join(args.output.replace("s3://", "").split("/")[1:]).rstrip("/")
        
        s3 = boto3.client("s3", region_name=AWS_REGION)
        
        # List cutout files
        def list_npz_files(s3_path: str) -> List[Tuple[str, str]]:
            bucket = s3_path.replace("s3://", "").split("/")[0]
            prefix = "/".join(s3_path.replace("s3://", "").split("/")[1:]).rstrip("/")
            
            files = []
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith(".npz"):
                        files.append((bucket, obj["Key"]))
            return files
        
        logger.info("Listing cutout files...")
        pos_files = list_npz_files(args.positives)
        neg_files = list_npz_files(args.negatives)
        
        logger.info(f"Found {len(pos_files)} positive, {len(neg_files)} negative cutouts")
        
        # Sample mode for mini runs
        if args.sample > 0:
            import random
            random.seed(42)
            pos_files = random.sample(pos_files, min(args.sample, len(pos_files)))
            neg_files = random.sample(neg_files, min(args.sample, len(neg_files)))
            logger.info(f"SAMPLE MODE: Using {len(pos_files)} positive, {len(neg_files)} negative")
        
        n_bootstrap = args.bootstrap
        logger.info(f"Bootstrap samples for AUC CI: {n_bootstrap}")
        
        # Process in parallel - use many partitions for full executor utilization
        # With 30 workers × 8 vCPU = 240 cores, need many partitions
        MAX_PARTITIONS = 50000
        pos_partitions = min(len(pos_files), MAX_PARTITIONS)
        neg_partitions = min(len(neg_files), MAX_PARTITIONS)
        
        logger.info(f"Using {pos_partitions} partitions for positives, {neg_partitions} for negatives")
        
        pos_rdd = sc.parallelize([(b, k, "positive") for b, k in pos_files], numSlices=pos_partitions)
        neg_rdd = sc.parallelize([(b, k, "negative") for b, k in neg_files], numSlices=neg_partitions)
        all_rdd = pos_rdd.union(neg_rdd)
        
        def process_file(args_tuple):
            bucket, key, cutout_type = args_tuple
            import boto3
            s3_client = boto3.client("s3", region_name=AWS_REGION)
            return process_cutout_file(s3_client, bucket, key, cutout_type)
        
        results = all_rdd.map(process_file).collect()
        
        # Separate results
        pos_results = [r for r in results if r["type"] == "positive"]
        neg_results = [r for r in results if r["type"] == "negative"]
        
        logger.info(f"Processed {len(pos_results)} positive, {len(neg_results)} negative")
        
        # =================================================================
        # CHECKPOINT 1: Save raw results immediately
        # =================================================================
        logger.info("CHECKPOINT 1: Saving raw results...")
        checkpoint_data = {
            "pos_count": len(pos_results),
            "neg_count": len(neg_results),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        s3.put_object(
            Bucket=output_bucket,
            Key=f"{output_prefix}/checkpoint_1_raw.json",
            Body=json.dumps(checkpoint_data),
            ContentType="application/json",
        )
        
        # =================================================================
        # QUALITY REPORT
        # =================================================================
        
        pos_quality_ok = sum(1 for r in pos_results if r.get("quality_ok", False))
        neg_quality_ok = sum(1 for r in neg_results if r.get("quality_ok", False))
        
        quality_report = {
            "positive": {
                "total": len(pos_results),
                "quality_ok": pos_quality_ok,
                "quality_fail": len(pos_results) - pos_quality_ok,
                "pass_rate": pos_quality_ok / len(pos_results) if pos_results else 0,
            },
            "negative": {
                "total": len(neg_results),
                "quality_ok": neg_quality_ok,
                "quality_fail": len(neg_results) - neg_quality_ok,
                "pass_rate": neg_quality_ok / len(neg_results) if neg_results else 0,
            },
            "gates": {
                "all_positives_ok": pos_quality_ok == len(pos_results),
                "negatives_98pct_ok": neg_quality_ok / len(neg_results) >= 0.98 if neg_results else False,
            },
        }
        
        logger.info(f"\nQuality Report:")
        logger.info(f"  Positives: {pos_quality_ok}/{len(pos_results)} pass ({quality_report['positive']['pass_rate']*100:.1f}%)")
        logger.info(f"  Negatives: {neg_quality_ok}/{len(neg_results)} pass ({quality_report['negative']['pass_rate']*100:.1f}%)")
        
        # =================================================================
        # CHECKPOINT 2: Save quality report immediately
        # =================================================================
        logger.info("CHECKPOINT 2: Saving quality report...")
        s3.put_object(
            Bucket=output_bucket,
            Key=f"{output_prefix}/quality_report.json",
            Body=json.dumps(quality_report, indent=2),
            ContentType="application/json",
        )
        
        # =================================================================
        # SHORTCUT DETECTION (LLM enhanced with bootstrap CI)
        # =================================================================
        
        # Features to analyze (expanded per LLM recommendation)
        shortcut_features = [
            # Core features
            "core_brightness_r", "core_max_r", "core_std_r",
            # Multi-radius core
            "core_r4_brightness", "core_r8_brightness", "core_r12_brightness",
            # Arc annulus (corrected radii 4-16px)
            "annulus_brightness_r", "annulus_max_r", "annulus_std_r",
            # Outer
            "outer_brightness_r",
            # Gradients
            "radial_gradient_r",
            # Colors
            "core_g", "core_z", "arc_g", "arc_z",
            "core_g_minus_r", "arc_g_minus_r", "arc_minus_core_g_r",
            # High-frequency (Laplacian)
            "laplacian_core", "laplacian_arc",
            # Azimuthal
            "azimuthal_asymmetry", "azimuthal_wedge_std",
            # Artifacts (saturation_frac removed - buggy per-image percentile metric)
            "edge_artifact_score",
            # Noise
            "mad_r",
        ]
        
        # Features that SHOULD separate (expected physics, not shortcuts)
        # These are real physical differences between lenses and non-lenses:
        # - azimuthal_asymmetry: Lensed arcs are azimuthally asymmetric by definition
        # - annulus_brightness_r: Lenses have arc flux in the annulus
        # - arc_* features: Arc region differs because lenses have arcs
        expected_physics_features = {
            "azimuthal_asymmetry",  # Arcs are asymmetric, that's the lensing signature
            "annulus_brightness_r",  # Arcs create annulus flux
            "annulus_max_r",  # Peak arc brightness
            "annulus_std_r",  # Arc structure variability
        }
        
        shortcut_report = {
            "features": {},
            "expected_physics": list(expected_physics_features),
            "gates": {},
            "thresholds": {
                "yellow": AUC_THRESHOLD_YELLOW,
                "red": AUC_THRESHOLD_RED,
            },
        }
        
        logger.info(f"Computing AUC for {len(shortcut_features)} features...")
        feature_start = time.time()
        
        for i, feature in enumerate(shortcut_features):
            pos_vals = [r.get(feature) for r in pos_results if r.get(feature) is not None]
            neg_vals = [r.get(feature) for r in neg_results if r.get(feature) is not None]
            
            # Compute AUC with bootstrap CI (now O(n log n) per bootstrap!)
            auc_result = compute_auc(pos_vals, neg_vals, n_bootstrap=n_bootstrap)
            auc = auc_result["auc"]
            auc_lo = auc_result["auc_lo"]
            auc_hi = auc_result["auc_hi"]
            
            # Distribution stats
            if pos_vals and neg_vals:
                pos_median = float(np.median(pos_vals))
                neg_median = float(np.median(neg_vals))
                pos_std = float(np.std(pos_vals))
                neg_std = float(np.std(neg_vals))
                overlap = abs(pos_median - neg_median) / max(pos_std + neg_std, 1e-10)
            else:
                pos_median = neg_median = pos_std = neg_std = overlap = None
            
            # LLM: Flag based on lower CI bound, not point estimate
            is_red = auc_lo > AUC_THRESHOLD_RED      # Lower CI > 0.70
            is_yellow = auc_lo > AUC_THRESHOLD_YELLOW  # Lower CI > 0.60
            
            # Check if this is expected physics (not a shortcut even if high AUC)
            is_expected_physics = feature in expected_physics_features
            
            shortcut_report["features"][feature] = {
                "auc": auc,
                "auc_lo": auc_lo,
                "auc_hi": auc_hi,
                "is_shortcut": is_red and not is_expected_physics,  # Only shortcut if not expected physics
                "is_red": is_red,
                "is_yellow": is_yellow,
                "is_expected_physics": is_expected_physics,
                "pos_median": pos_median,
                "neg_median": neg_median,
                "pos_std": pos_std,
                "neg_std": neg_std,
                "separation": overlap,
                "n_pos": len(pos_vals),
                "n_neg": len(neg_vals),
            }
            
            # Different flag for expected physics
            if is_expected_physics and (is_red or is_yellow):
                flag = "📐 PHYSICS" if is_red else "📐 physics"
            else:
                flag = "🔴 RED" if is_red else ("🟡 YELLOW" if is_yellow else "✓")
            elapsed = time.time() - feature_start
            logger.info(f"  [{i+1}/{len(shortcut_features)}] {feature}: AUC={auc:.3f} [{auc_lo:.3f}, {auc_hi:.3f}] {flag} ({elapsed:.1f}s total)")
            
            # CHECKPOINT: Save incremental shortcut results after each feature
            if (i + 1) % 5 == 0 or i == len(shortcut_features) - 1:
                s3.put_object(
                    Bucket=output_bucket,
                    Key=f"{output_prefix}/shortcut_checkpoint_{i+1}.json",
                    Body=json.dumps(shortcut_report, indent=2),
                    ContentType="application/json",
                )
        
        logger.info(f"Shortcut detection completed in {time.time() - feature_start:.1f}s")
        
        # Special check: annulus SHOULD separate (confirms signal exists)
        annulus_auc = shortcut_report["features"]["annulus_brightness_r"]["auc"]
        
        # Core shortcuts to check (LLM: multi-radius)
        core_features = ["core_brightness_r", "core_r4_brightness", "core_r8_brightness", 
                        "core_r12_brightness", "radial_gradient_r", "mad_r"]
        core_shortcuts = any(
            shortcut_report["features"].get(f, {}).get("is_red", False) 
            for f in core_features
        )
        
        # Count yellow and red flags (excluding expected physics)
        n_red = sum(1 for f in shortcut_report["features"].values() 
                    if f.get("is_red", False) and not f.get("is_expected_physics", False))
        n_yellow = sum(1 for f in shortcut_report["features"].values() 
                       if f.get("is_yellow", False) and not f.get("is_expected_physics", False))
        n_physics = sum(1 for f in shortcut_report["features"].values() 
                        if f.get("is_expected_physics", False) and f.get("is_red", False))
        
        shortcut_report["gates"] = {
            "no_core_shortcuts": not core_shortcuts,
            "annulus_separates": annulus_auc > 0.55,  # Should be separable
            "no_trivial_shortcuts": n_red == 0,  # No features with AUC_lo > 0.70 (excl. physics)
            "few_yellow_flags": n_yellow <= 3,  # At most 3 yellow flags (excl. physics)
        }
        shortcut_report["summary"] = {
            "n_red": n_red,
            "n_yellow": n_yellow,
            "n_expected_physics": n_physics,  # Features that SHOULD separate
        }
        
        # =================================================================
        # FINAL SUMMARY
        # =================================================================
        
        all_gates_passed = (
            quality_report["gates"]["all_positives_ok"] and
            quality_report["gates"]["negatives_98pct_ok"] and
            shortcut_report["gates"]["no_core_shortcuts"] and
            shortcut_report["gates"]["no_trivial_shortcuts"]
        )
        
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": PIPELINE_VERSION,
            "elapsed_seconds": time.time() - start_time,
            "quality": quality_report,
            "shortcuts": shortcut_report,
            "all_gates_passed": all_gates_passed,
            "go_no_go": "GO" if all_gates_passed else "NO-GO",
        }
        
        # Save reports
        s3.put_object(
            Bucket=output_bucket,
            Key=f"{output_prefix}/validation_summary.json",
            Body=json.dumps(summary, indent=2, default=str)
        )
        
        s3.put_object(
            Bucket=output_bucket,
            Key=f"{output_prefix}/quality_report.json",
            Body=json.dumps(quality_report, indent=2)
        )
        
        s3.put_object(
            Bucket=output_bucket,
            Key=f"{output_prefix}/shortcut_report.json",
            Body=json.dumps(shortcut_report, indent=2)
        )
        
        # Save detailed results as parquet
        from pyspark.sql import Row
        
        results_df = spark.createDataFrame([Row(**r) for r in results])
        results_df.write.mode("overwrite") \
            .option("compression", "gzip") \
            .parquet(f"s3://{output_bucket}/{output_prefix}/detailed_results/")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"VALIDATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"GO/NO-GO: {summary['go_no_go']}")
        logger.info(f"All gates passed: {all_gates_passed}")
        logger.info(f"Output: s3://{output_bucket}/{output_prefix}/")
        logger.info(f"Elapsed: {time.time() - start_time:.1f}s")
        
    except Exception as e:
        logger.error(f"Job failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

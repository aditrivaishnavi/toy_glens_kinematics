#!/usr/bin/env python3
"""
Phase 2: Gen7 Hybrid Source Injection via Spark

Generates training data with procedural hybrid sources (Sersic + clumps).

Input: Base stamps with ctrl (v5_cosmos_paired)
Output: New stamps with Gen7 injected sources

Usage:
    spark-submit --deploy-mode cluster spark_gen7_injection.py \
        --input s3://darkhaloscope/v5_cosmos_paired/train \
        --output s3://darkhaloscope/planb/gen7/train \
        --config s3://darkhaloscope/planb/configs/gen7.yaml

Preset: large (20 workers)
Expected runtime: 2-4 hours for full dataset
"""
import argparse
import gzip
import io
import json
import logging
import sys
import time
import datetime
import uuid
import boto3
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, BinaryType, FloatType


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "n_clumps_range": [2, 8],
    "clump_flux_frac_range": [0.05, 0.20],
    "clump_sigma_range": [0.5, 2.0],
    "gradient_strength": 0.15,
    "re_range_arcsec": [0.1, 0.5],
    "n_sersic_range": [0.5, 2.0],
    "q_range": [0.3, 1.0],
}


def setup_logging(job_name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{job_name}] %(levelname)s: %(message)s",
    )
    return logging.getLogger(job_name)


# =============================================================================
# GEN7 SOURCE GENERATION
# =============================================================================

def sersic_2d(
    size: int,
    re_pix: float,
    n: float,
    q: float,
    pa: float,
    center: Tuple[float, float] = None,
) -> np.ndarray:
    """Generate 2D Sersic profile."""
    if center is None:
        center = (size / 2, size / 2)
    
    y, x = np.mgrid[:size, :size]
    x = x - center[1]
    y = y - center[0]
    
    # Rotate and scale for ellipticity
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    x_rot = x * cos_pa + y * sin_pa
    y_rot = -x * sin_pa + y * cos_pa
    
    r = np.sqrt(x_rot**2 + (y_rot / q)**2)
    
    bn = 1.9992 * n - 0.3271  # Approximation
    intensity = np.exp(-bn * ((r / re_pix)**(1/n) - 1))
    
    return intensity.astype(np.float32)


def generate_hybrid_source(
    seed: int,
    config: Dict,
    size: int = 64,
) -> Tuple[np.ndarray, Dict]:
    """
    Generate Gen7 hybrid source (Sersic + clumps + gradient).
    
    Returns:
        (source_image, metadata)
    """
    rng = np.random.default_rng(seed)
    
    # Sample parameters
    pixel_scale = 0.262  # arcsec/pix
    re_arcsec = rng.uniform(*config["re_range_arcsec"])
    re_pix = re_arcsec / pixel_scale
    n_sersic = rng.uniform(*config["n_sersic_range"])
    q = rng.uniform(*config["q_range"])
    pa = rng.uniform(0, np.pi)
    
    # Generate base Sersic
    base = sersic_2d(size, re_pix, n_sersic, q, pa)
    base = base / base.sum()  # Normalize to unit flux
    
    # Add clumps
    n_clumps = rng.integers(*config["n_clumps_range"])
    clump_flux_frac = rng.uniform(*config["clump_flux_frac_range"])
    
    clumps = np.zeros((size, size), dtype=np.float32)
    base_flux = base.sum()
    total_clump_flux = clump_flux_frac * base_flux
    flux_per_clump = total_clump_flux / max(n_clumps, 1)
    
    y_grid, x_grid = np.mgrid[:size, :size]
    
    for _ in range(n_clumps):
        # Position within source region
        sigma_clump = rng.uniform(*config["clump_sigma_range"])
        
        # Sample position from base profile
        cx = size/2 + rng.normal(0, re_pix * 1.5)
        cy = size/2 + rng.normal(0, re_pix * 1.5)
        
        # Clamp to image
        cx = np.clip(cx, 5, size - 5)
        cy = np.clip(cy, 5, size - 5)
        
        # Add Gaussian clump
        gauss = np.exp(-0.5 * ((x_grid - cx)**2 + (y_grid - cy)**2) / sigma_clump**2)
        gauss = gauss / (2 * np.pi * sigma_clump**2)  # Normalize
        clumps += flux_per_clump * gauss
    
    # Combine
    result = base + clumps
    
    # Add color gradient (optional)
    if config.get("gradient_strength", 0) > 0:
        grad_angle = rng.uniform(0, 2 * np.pi)
        grad = (x_grid - size/2) * np.cos(grad_angle) + (y_grid - size/2) * np.sin(grad_angle)
        grad = grad / (size / 2)  # Normalize to [-1, 1]
        grad = 1 + config["gradient_strength"] * grad
        result = result * grad
    
    # Normalize
    result = result / result.sum()
    
    meta = {
        "re_pix": float(re_pix),
        "n_sersic": float(n_sersic),
        "q": float(q),
        "pa": float(pa),
        "n_clumps": int(n_clumps),
        "clump_flux_frac": float(clump_flux_frac),
    }
    
    return result, meta


# =============================================================================
# LENSING
# =============================================================================

def deflection_sie(x: np.ndarray, y: np.ndarray, theta_e: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """SIE deflection field."""
    qp = np.sqrt((1 - q**2) / q**2) if q < 1 else 0
    
    r = np.sqrt(x**2 + y**2) + 1e-10
    phi = np.arctan2(y, x)
    
    kappa = theta_e / (2 * r)
    
    if q < 0.99:
        alpha = theta_e * np.sqrt(q) / np.sqrt(1 - q**2)
        alpha_x = alpha * np.arctan(np.sqrt(1 - q**2) * x / (r + 1e-10))
        alpha_y = alpha * np.arctanh(np.sqrt(1 - q**2) * y / (r + 1e-10))
    else:
        alpha_x = theta_e * x / (r + 1e-10)
        alpha_y = theta_e * y / (r + 1e-10)
    
    return alpha_x, alpha_y


def ray_trace(
    source: np.ndarray,
    theta_e: float,
    q_lens: float = 0.8,
    source_offset: Tuple[float, float] = (0, 0),
) -> np.ndarray:
    """Ray-trace source through lens to create lensed image."""
    size = source.shape[0]
    
    # Image plane coordinates
    y_img, x_img = np.mgrid[:size, :size]
    x_img = (x_img - size/2).astype(np.float32)
    y_img = (y_img - size/2).astype(np.float32)
    
    # Deflection
    alpha_x, alpha_y = deflection_sie(x_img, y_img, theta_e, q_lens)
    
    # Source plane
    x_src = x_img - alpha_x + source_offset[0]
    y_src = y_img - alpha_y + source_offset[1]
    
    # Map back to pixel coords
    x_src_pix = x_src + size/2
    y_src_pix = y_src + size/2
    
    # Bilinear interpolation
    x0 = np.floor(x_src_pix).astype(int)
    y0 = np.floor(y_src_pix).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Weights
    wx = x_src_pix - x0
    wy = y_src_pix - y0
    
    # Clamp
    x0 = np.clip(x0, 0, size - 1)
    x1 = np.clip(x1, 0, size - 1)
    y0 = np.clip(y0, 0, size - 1)
    y1 = np.clip(y1, 0, size - 1)
    
    # Interpolate
    lensed = (
        (1 - wx) * (1 - wy) * source[y0, x0] +
        wx * (1 - wy) * source[y0, x1] +
        (1 - wx) * wy * source[y1, x0] +
        wx * wy * source[y1, x1]
    )
    
    return lensed.astype(np.float32)


# =============================================================================
# PROCESSING
# =============================================================================

def process_row(row, config: Dict) -> Dict:
    """Process a single row: generate Gen7 source and inject."""
    try:
        # Decode control stamp
        ctrl_blob = row.ctrl_stamp_npz
        z = np.load(io.BytesIO(ctrl_blob), allow_pickle=False)
        ctrl = np.stack([z["image_g"], z["image_r"], z["image_z"]], axis=0)
        
        # Get theta_e
        theta_e_arcsec = getattr(row, "theta_e_arcsec", 1.5)
        theta_e_pix = theta_e_arcsec / 0.262
        
        # Generate Gen7 source
        seed = hash(row.task_id) % (2**31)
        source, source_meta = generate_hybrid_source(seed, config)
        
        # Lens the source
        lensed = ray_trace(source, theta_e_pix)
        
        # Scale to realistic flux
        src_flux = getattr(row, "src_total_flux_nmgy", 100.0)
        lensed = lensed * src_flux
        
        # TODO: PSF convolution would go here
        # For now, simple Gaussian blur
        from scipy.ndimage import gaussian_filter
        psf_sigma = 1.5  # pixels
        lensed = gaussian_filter(lensed, psf_sigma)
        
        # Add to control (all bands)
        stamp = ctrl.copy()
        for i in range(3):
            stamp[i] = ctrl[i] + lensed
        
        # Encode output
        out_buffer = io.BytesIO()
        np.savez_compressed(out_buffer, 
                          image_g=stamp[0], 
                          image_r=stamp[1], 
                          image_z=stamp[2])
        stamp_npz = out_buffer.getvalue()
        
        return {
            "task_id": row.task_id,
            "stamp_npz": stamp_npz,
            "ctrl_stamp_npz": ctrl_blob,
            "theta_e_arcsec": theta_e_arcsec,
            "gen7_re_pix": source_meta["re_pix"],
            "gen7_n_sersic": source_meta["n_sersic"],
            "gen7_n_clumps": source_meta["n_clumps"],
            "error": None,
        }
        
    except Exception as e:
        return {
            "task_id": getattr(row, "task_id", "unknown"),
            "stamp_npz": None,
            "ctrl_stamp_npz": None,
            "theta_e_arcsec": None,
            "gen7_re_pix": None,
            "gen7_n_sersic": None,
            "gen7_n_clumps": None,
            "error": str(e),
        }


def process_partition(config: Dict):
    """Return a partition processor with config."""
    def _process(iterator: Iterator) -> Iterator:
        for row in iterator:
            result = process_row(row, config)
            if result["error"] is None:
                yield result
    return _process



def _utc_run_id() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def _s3_put_bytes(s3, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    # boto3 retries are configured via client; still keep simple explicit retry
    last = None
    for attempt in range(6):
        try:
            s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
            return
        except Exception as e:
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise last


def _s3_list_prefix(s3, bucket: str, prefix: str, max_keys: int = 1) -> bool:
    # Return True if any object exists under prefix
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys)
    return bool(resp.get("KeyCount", 0) > 0)


def _write_parquet_to_s3(rows: List[Dict[str, Any]], s3_uri: str) -> None:
    if not rows:
        return
    import pyarrow as pa
    import pyarrow.parquet as pq

    bucket, key = _parse_s3_uri(s3_uri)
    table = pa.Table.from_pylist(rows)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    data = sink.getvalue().to_pybytes()

    s3 = boto3.client("s3", region_name=AWS_REGION)
    _s3_put_bytes(s3, bucket, key, data, content_type="application/octet-stream")


def _hadoop_path_exists(spark: SparkSession, path: str) -> bool:
    jvm = spark._jvm
    hconf = spark._jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(hconf)
    p = jvm.org.apache.hadoop.fs.Path(path)
    return fs.exists(p)


def _load_done_row_ids(spark: SparkSession, done_path: str) -> Optional["pyspark.sql.DataFrame"]:
    if not done_path:
        return None
    if not _hadoop_path_exists(spark, done_path):
        return None
    df = spark.read.parquet(done_path)
    # Keep only successful rows
    if "status" in df.columns:
        df = df.filter(F.col("status") == F.lit("OK"))
    if "row_id" not in df.columns:
        return None
    return df.select("row_id").dropDuplicates()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet path (s3://... or hdfs://...)")
    parser.add_argument("--output", required=True, help="Output parquet prefix (s3://... recommended)")
    parser.add_argument("--done", required=True, help="Checkpoint manifest parquet prefix (s3://...)")
    parser.add_argument("--fail-fast", action="store_true", help="Abort job on first row error (default true)")
    parser.add_argument("--no-fail-fast", dest="fail_fast", action="store_false")
    parser.set_defaults(fail_fast=True)
    parser.add_argument("--run-id", default="", help="Optional run id; default UTC timestamp")
    parser.add_argument("--flush-rows", type=int, default=256, help="Rows to buffer before uploading parquet chunk")
    parser.add_argument("--max-rows", type=int, default=0, help="If >0, limit rows for debugging")
    args = parser.parse_args()

    run_id = args.run_id or _utc_run_id()
    spark = SparkSession.builder.appName(f"Gen7InjectionCheckpointed[{run_id}]").getOrCreate()

    df_in = spark.read.parquet(args.input)
    if args.max_rows and args.max_rows > 0:
        df_in = df_in.limit(int(args.max_rows))

    # Ensure row_id exists
    if "row_id" not in df_in.columns:
        df_in = df_in.withColumn("row_id", F.col("brickname"))

    # Skip already completed rows using done manifest
    done_ids = _load_done_row_ids(spark, args.done)
    if done_ids is not None:
        df_in = df_in.join(done_ids, on="row_id", how="left_anti")

    out_prefix = args.output.rstrip("/")
    done_prefix = args.done.rstrip("/")

    # Fail-fast + checkpointed writes from executors
    def partition_writer(pid: int, it):
        import pyarrow as pa
        import pyarrow.parquet as pq

        s3 = boto3.client("s3", region_name=AWS_REGION)

        out_rows: List[Dict[str, Any]] = []
        done_rows: List[Dict[str, Any]] = []

        def flush(kind: str):
            nonlocal out_rows, done_rows
            ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            uid = uuid.uuid4().hex
            if kind in ("out", "both") and out_rows:
                key = f"{out_prefix}/run_id={run_id}/part={pid}/out_{ts}_{uid}.parquet"
                b,k = _parse_s3_uri(key)
                table = pa.Table.from_pylist(out_rows)
                sink = pa.BufferOutputStream()
                pq.write_table(table, sink, compression="zstd")
                _s3_put_bytes(s3, b, k, sink.getvalue().to_pybytes())
                out_rows = []
            if kind in ("done", "both") and done_rows:
                key = f"{done_prefix}/run_id={run_id}/part={pid}/done_{ts}_{uid}.parquet"
                b,k = _parse_s3_uri(key)
                table = pa.Table.from_pylist(done_rows)
                sink = pa.BufferOutputStream()
                pq.write_table(table, sink, compression="zstd")
                _s3_put_bytes(s3, b, k, sink.getvalue().to_pybytes())
                done_rows = []

        for row in it:
            r = row.asDict(recursive=True)
            row_id = r.get("row_id", None)
            try:
                out = inject_one_row(r)  # returns dict
                out_rows.append(out)
                done_rows.append({"row_id": row_id, "status": "OK", "run_id": run_id, "part": pid})
            except Exception as e:
                done_rows.append({"row_id": row_id, "status": "ERROR", "run_id": run_id, "part": pid, "error": str(e)})
                flush("done")  # persist error immediately
                if args.fail_fast:
                    raise
            if len(out_rows) >= args.flush_rows:
                flush("both")

        flush("both")
        return iter([])

    # We need a pure-python row injector accessible in executor
    bc_params = spark.sparkContext.broadcast(
        {"stamp_size": STAMP_SIZE, "pixscale": PIX_SCALE_ARCSEC}
    )

    def inject_one_row(r: Dict[str, Any]) -> Dict[str, Any]:
        # Existing logic was inside process_partition; keep it in a dedicated function for reuse
        stamp_npz = r["stamp_npz"]
        ctrl_npz = r.get("ctrl_stamp_npz", None)
        # Decode stamp (base LRG)
        stamp = decode_npz_blob(stamp_npz)  # (3,H,W)
        if ctrl_npz is not None:
            ctrl = decode_npz_blob(ctrl_npz)
        else:
            ctrl = None

        theta_e = float(r.get("theta_e_arcsec", 0.0))
        psf_fwhm_arcsec = float(r.get("psf_fwhm_arcsec", r.get("psfsize_r", 1.3)))
        psf_fwhm_pix = psf_fwhm_arcsec / PIX_SCALE_ARCSEC

        # Source params (fallback deterministic)
        src_reff_arcsec = float(r.get("src_reff_arcsec", 0.35))
        src_e = float(r.get("src_e", 0.3))
        src_phi = float(r.get("src_phi_rad", 0.0))
        src_x = float(r.get("src_x_arcsec", 0.0))
        src_y = float(r.get("src_y_arcsec", 0.0))

        # Generate hybrid template in (H,W) and use lenstronomy INTERPOL light model (implemented in dhs_gen)
        key = str(r.get("row_id", r.get("brickname", "")))
        template, genmeta = generate_hybrid_template(
            key=key,
            H=96,
            W=96,
            reff_arcsec=src_reff_arcsec,
            src_e=src_e,
            src_phi_rad=src_phi,
        )

        add = render_hybrid_lensed_source_lenstronomy(
            template=template,
            theta_e_arcsec=theta_e,
            src_x_arcsec=src_x,
            src_y_arcsec=src_y,
            psf_fwhm_pix=psf_fwhm_pix,
        )  # (3,64,64) nMgy/pix

        out_stamp = (stamp + add).astype(np.float32)
        out_npz = encode_npz_blob(out_stamp)

        out = dict(r)
        out["stamp_npz"] = out_npz
        out["gen7_meta_json"] = json.dumps(genmeta, separators=(",", ":"))
        out["run_id"] = run_id
        return out

    # Force serialization of required helper functions
    rdd = df_in.rdd.mapPartitionsWithIndex(partition_writer)
    # Action to execute
    rdd.count()

    spark.stop()


if __name__ == "__main__":
    main()
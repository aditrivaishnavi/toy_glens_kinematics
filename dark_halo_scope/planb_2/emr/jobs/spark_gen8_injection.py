#!/usr/bin/env python3
"""
Phase 3: Gen8 Domain Randomization via Spark

Applies domain randomization artifacts to training data:
- PSF anisotropy
- Astrometric jitter
- Cosmic rays
- Saturation spikes
- Background residuals

Input: Base stamps with ctrl
Output: Stamps with Gen8 domain randomization

Usage:
    spark-submit --deploy-mode cluster spark_gen8_injection.py \
        --input s3://darkhaloscope/v5_cosmos_paired/train \
        --output s3://darkhaloscope/planb/gen8/train

Preset: large (20 workers)
Expected runtime: 2-4 hours for full dataset
"""
import argparse
import gzip
import io
import logging
import sys
import time
import datetime
import uuid
import boto3
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, BinaryType, FloatType


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # From DR10 calibration
    "cosmic_rate": 0.12,
    "sat_rate": 0.06,
    
    # Astrometric jitter
    "jitter_sigma_pix": 0.25,
    
    # PSF anisotropy
    "psf_aniso_enabled": False,  # Already in injection
    
    # Background residuals
    "background_sigma": 0.02,
    "background_scale_pix": 10,
}


def setup_logging(job_name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{job_name}] %(levelname)s: %(message)s",
    )
    return logging.getLogger(job_name)


# =============================================================================
# ARTIFACT GENERATION
# =============================================================================

def add_cosmic_ray(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a cosmic ray artifact."""
    size = img.shape[0]
    
    # Random position
    x = rng.integers(0, size)
    y = rng.integers(0, size)
    
    # Random length and angle
    length = rng.integers(2, 8)
    angle = rng.uniform(0, 2 * np.pi)
    
    # Draw line
    for i in range(length):
        xi = int(x + i * np.cos(angle))
        yi = int(y + i * np.sin(angle))
        if 0 <= xi < size and 0 <= yi < size:
            # Very bright pixel
            img[yi, xi] = max(img[yi, xi], np.percentile(img, 99) * 10)
    
    return img


def add_saturation_spike(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add saturation diffraction spike."""
    size = img.shape[0]
    
    # Position near edge (simulating bright star off-frame)
    edge = rng.choice(["top", "bottom", "left", "right"])
    
    if edge == "top":
        x, y = rng.integers(10, size-10), 0
        dx, dy = 0, 1
    elif edge == "bottom":
        x, y = rng.integers(10, size-10), size-1
        dx, dy = 0, -1
    elif edge == "left":
        x, y = 0, rng.integers(10, size-10)
        dx, dy = 1, 0
    else:
        x, y = size-1, rng.integers(10, size-10)
        dx, dy = -1, 0
    
    # Draw spike
    length = rng.integers(10, 30)
    width = rng.integers(1, 3)
    amplitude = np.percentile(img, 95) * 5
    
    for i in range(length):
        for w in range(-width, width+1):
            xi = x + i * dx + w * (1 - abs(dx))
            yi = y + i * dy + w * (1 - abs(dy))
            if 0 <= xi < size and 0 <= yi < size:
                decay = 1.0 - i / length
                img[yi, xi] += amplitude * decay
    
    return img


def add_background_residual(
    img: np.ndarray, 
    rng: np.random.Generator,
    sigma: float = 0.02,
    scale: float = 10,
) -> np.ndarray:
    """Add correlated background residual."""
    size = img.shape[0]
    
    # Generate smooth random field
    small_size = max(size // int(scale), 2)
    small_noise = rng.normal(0, sigma, (small_size, small_size))
    
    # Upsample with interpolation
    from scipy.ndimage import zoom
    factor = size / small_size
    noise = zoom(small_noise, factor, order=1)
    
    # Crop to exact size
    noise = noise[:size, :size]
    
    # Scale by image std
    img_std = np.std(img)
    noise = noise * img_std
    
    return img + noise


def apply_jitter(img: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Apply sub-pixel astrometric jitter."""
    if sigma <= 0:
        return img
    
    dx = rng.normal(0, sigma)
    dy = rng.normal(0, sigma)
    
    # Sub-pixel shift via interpolation
    from scipy.ndimage import shift
    return shift(img, (dy, dx), order=1, mode='nearest')


def apply_domain_randomization(
    stamp: np.ndarray,
    ctrl: np.ndarray,
    seed: int,
    config: Dict,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Apply domain randomization to both stamp and ctrl.
    
    Returns:
        (augmented_stamp, augmented_ctrl, metadata)
    """
    rng = np.random.default_rng(seed)
    
    meta = {
        "has_cosmic": False,
        "has_spike": False,
        "jitter_dx": 0,
        "jitter_dy": 0,
    }
    
    # Apply same augmentations to both stamp and ctrl
    # (artifacts are in the base image, not the injection)
    
    for i in range(stamp.shape[0]):  # Per channel
        # Cosmic rays
        if rng.random() < config["cosmic_rate"]:
            stamp[i] = add_cosmic_ray(stamp[i].copy(), rng)
            ctrl[i] = add_cosmic_ray(ctrl[i].copy(), rng)
            meta["has_cosmic"] = True
        
        # Saturation spikes
        if rng.random() < config["sat_rate"]:
            stamp[i] = add_saturation_spike(stamp[i].copy(), rng)
            ctrl[i] = add_saturation_spike(ctrl[i].copy(), rng)
            meta["has_spike"] = True
        
        # Background residuals
        if config.get("background_sigma", 0) > 0:
            stamp[i] = add_background_residual(
                stamp[i], rng, 
                config["background_sigma"],
                config.get("background_scale_pix", 10)
            )
            ctrl[i] = add_background_residual(
                ctrl[i], rng,
                config["background_sigma"],
                config.get("background_scale_pix", 10)
            )
        
        # Astrometric jitter
        if config.get("jitter_sigma_pix", 0) > 0:
            stamp[i] = apply_jitter(stamp[i], rng, config["jitter_sigma_pix"])
            ctrl[i] = apply_jitter(ctrl[i], rng, config["jitter_sigma_pix"])
    
    return stamp, ctrl, meta


# =============================================================================
# PROCESSING
# =============================================================================

def process_row(row, config: Dict) -> Dict:
    """Process a single row: apply Gen8 domain randomization."""
    try:
        # Decode stamps
        z_stamp = np.load(io.BytesIO(row.stamp_npz), allow_pickle=False)
        stamp = np.stack([z_stamp["image_g"], z_stamp["image_r"], z_stamp["image_z"]], axis=0).astype(np.float32)
        
        z_ctrl = np.load(io.BytesIO(row.ctrl_stamp_npz), allow_pickle=False)
        ctrl = np.stack([z_ctrl["image_g"], z_ctrl["image_r"], z_ctrl["image_z"]], axis=0).astype(np.float32)
        
        # Apply domain randomization
        seed = hash(row.task_id) % (2**31)
        stamp_aug, ctrl_aug, meta = apply_domain_randomization(stamp, ctrl, seed, config)
        
        # Encode output
        stamp_buffer = io.BytesIO()
        np.savez_compressed(stamp_buffer,
                          image_g=stamp_aug[0],
                          image_r=stamp_aug[1],
                          image_z=stamp_aug[2])
        
        ctrl_buffer = io.BytesIO()
        np.savez_compressed(ctrl_buffer,
                          image_g=ctrl_aug[0],
                          image_r=ctrl_aug[1],
                          image_z=ctrl_aug[2])
        
        return {
            "task_id": row.task_id,
            "stamp_npz": stamp_buffer.getvalue(),
            "ctrl_stamp_npz": ctrl_buffer.getvalue(),
            "theta_e_arcsec": getattr(row, "theta_e_arcsec", None),
            "arc_snr": getattr(row, "arc_snr", None),
            "gen8_has_cosmic": meta["has_cosmic"],
            "gen8_has_spike": meta["has_spike"],
            "error": None,
        }
        
    except Exception as e:
        return {
            "task_id": getattr(row, "task_id", "unknown"),
            "stamp_npz": None,
            "ctrl_stamp_npz": None,
            "theta_e_arcsec": None,
            "arc_snr": None,
            "gen8_has_cosmic": None,
            "gen8_has_spike": None,
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


def _s3_put_bytes(s3, bucket: str, key: str, data: bytes) -> None:
    last = None
    for attempt in range(6):
        try:
            s3.put_object(Bucket=bucket, Key=key, Body=data)
            return
        except Exception as e:
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise last


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
    if "status" in df.columns:
        df = df.filter(F.col("status") == F.lit("OK"))
    if "row_id" not in df.columns:
        return None
    return df.select("row_id").dropDuplicates()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="Output parquet prefix (s3://...)")
    parser.add_argument("--config-path", default="", help="Optional JSON config file to override DEFAULT_CONFIG")
    parser.add_argument("--config-json", default="", help="Optional JSON string to override DEFAULT_CONFIG")
    parser.add_argument("--done", required=True, help="Done parquet prefix (s3://...)")
    parser.add_argument("--fail-fast", action="store_true", help="Abort job on first row error (default true)")
    parser.add_argument("--no-fail-fast", dest="fail_fast", action="store_false")
    parser.set_defaults(fail_fast=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--flush-rows", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.config_path:
        with open(args.config_path, "r") as f:
            config.update(json.load(f))
    if args.config_json:
        config.update(json.loads(args.config_json))

    run_id = args.run_id or _utc_run_id()
    spark = SparkSession.builder.appName(f"Gen8InjectionCheckpointed[{run_id}]").getOrCreate()

    df_in = spark.read.parquet(args.input)
    if args.max_rows and args.max_rows > 0:
        df_in = df_in.limit(int(args.max_rows))

    if "row_id" not in df_in.columns:
        if "task_id" in df_in.columns:
            df_in = df_in.withColumn("row_id", F.col("task_id"))
        else:
            df_in = df_in.withColumn("row_id", F.col("brickname"))

    done_ids = _load_done_row_ids(spark, args.done)
    if done_ids is not None:
        df_in = df_in.join(done_ids, on="row_id", how="left_anti")

    out_prefix = args.output.rstrip("/")
    done_prefix = args.done.rstrip("/")

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
                uri = f"{out_prefix}/run_id={run_id}/part={pid}/out_{ts}_{uid}.parquet"
                b,k = _parse_s3_uri(uri)
                table = pa.Table.from_pylist(out_rows)
                sink = pa.BufferOutputStream()
                pq.write_table(table, sink, compression="zstd")
                _s3_put_bytes(s3, b, k, sink.getvalue().to_pybytes())
                out_rows = []
            if kind in ("done", "both") and done_rows:
                uri = f"{done_prefix}/run_id={run_id}/part={pid}/done_{ts}_{uid}.parquet"
                b,k = _parse_s3_uri(uri)
                table = pa.Table.from_pylist(done_rows)
                sink = pa.BufferOutputStream()
                pq.write_table(table, sink, compression="zstd")
                _s3_put_bytes(s3, b, k, sink.getvalue().to_pybytes())
                done_rows = []

        for row in it:
            r = row.asDict(recursive=True)
            row_id = r.get("row_id")
            try:
                import types
                out = process_row(types.SimpleNamespace(**r), config)
                if out.get("error") is not None:
                    raise RuntimeError(out["error"])
                out["row_id"] = row_id
                out_rows.append(out)
                done_rows.append({"row_id": row_id, "status": "OK", "run_id": run_id, "part": pid})
            except Exception as e:
                done_rows.append({"row_id": row_id, "status": "ERROR", "run_id": run_id, "part": pid, "error": str(e)})
                flush("done")
                if args.fail_fast:
                    raise
            if len(out_rows) >= args.flush_rows:
                flush("both")

        flush("both")
        return iter([])

    df_in.rdd.mapPartitionsWithIndex(partition_writer).count()
    spark.stop()


if __name__ == "__main__":
    main()
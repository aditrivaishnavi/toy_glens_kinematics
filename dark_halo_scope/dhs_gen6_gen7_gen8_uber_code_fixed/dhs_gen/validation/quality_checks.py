"""
Automated data quality checks for generated stamp Parquet.

Run on a sample and emit JSON.
This is designed to be integrated in your pipeline CI or as a pre-training gate.
"""
from __future__ import annotations
import argparse
import gzip
import io
import json
from typing import Dict, Tuple

import numpy as np


def robust_mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad + 1e-12


def clumpiness_proxy(img: np.ndarray) -> float:
    if img.ndim != 2:
        raise ValueError("clumpiness_proxy expects 2D")
    H, W = img.shape
    k = 5
    pad = k // 2
    p = np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
    blur = np.zeros_like(img, dtype=np.float32)
    for dy in range(k):
        for dx in range(k):
            blur += p[dy:dy + H, dx:dx + W]
    blur /= float(k * k)
    hp = img - blur
    num = float(np.sum(np.clip(hp, 0, None)))
    den = float(np.sum(np.abs(img)) + 1e-12)
    return num / den


def decode_stamp_npz(blob: bytes) -> Tuple[np.ndarray, str]:
    """Decode stamp NPZ blob to (C,H,W) array and bandset string.
    
    Supports:
    - Multi-band format: image_g, image_r, image_z keys (our actual format)
    - Legacy single-key format: 'img' or first key
    - Gzip-wrapped NPZ
    
    Returns:
        (array, bandset) where bandset is 'grz', 'r', 'g', 'z', or 'unknown'
    """
    def _decode(z):
        # Check for multi-band format first (our actual format)
        if "image_r" in z or "image_g" in z or "image_z" in z:
            bands = []
            bandset_chars = ""
            for band_key, band_char in [("image_g", "g"), ("image_r", "r"), ("image_z", "z")]:
                if band_key in z:
                    bands.append(np.asarray(z[band_key], dtype=np.float32))
                    bandset_chars += band_char
            if len(bands) == 1:
                arr = bands[0][None, :, :]  # (1, H, W)
            else:
                arr = np.stack(bands, axis=0)  # (C, H, W)
            return arr, bandset_chars
        elif "img" in z:
            arr = np.asarray(z["img"], dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[None, :, :]
            return arr, "unknown"
        else:
            key = list(z.keys())[0]
            arr = np.asarray(z[key], dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[None, :, :]
            return arr, "unknown"
    
    try:
        z = np.load(io.BytesIO(blob), allow_pickle=False)
        return _decode(z)
    except Exception:
        data = gzip.decompress(blob)
        z = np.load(io.BytesIO(data), allow_pickle=False)
        return _decode(z)


def validate_parquet_stamps(
    parquet_glob: str,
    max_rows: int = 200000,
    sample_rows: int = 2000,
    expect_hw: Tuple[int, int] = (64, 64),
    expect_bands: int = 3,
) -> Dict[str, object]:
    """Lightweight automated QA for stamp parquet(s).

    This is intended as a fast preflight that catches:
    - NPZ decode failures
    - unexpected array shapes / band layouts
    - NaN/Inf values
    - near-constant images (MAD ~ 0) in a chosen reference band
    - gross distribution shifts in source_mode / artifact_profile

    Notes on formats
    - Multi-band NPZ: image_g/image_r/image_z keys (any subset).
    - Legacy NPZ: single key (img or unnamed). May be (H,W), (C,H,W), or (H,W,C).
    - If a parquet column 'bandset' exists, it is treated as authoritative. Otherwise we infer from NPZ keys.

    Returns a small JSON-serializable dict suitable for CI logging.
    """
    try:
        import pyarrow.dataset as ds
    except Exception as e:
        raise RuntimeError("pyarrow required: pip install pyarrow") from e

    dataset = ds.dataset(parquet_glob, format="parquet")
    cols = [c for c in ["task_id", "is_control", "stamp_npz", "bandset", "source_mode", "artifact_profile"] if c in dataset.schema.names]
    tbl = dataset.to_table(columns=cols)

    n = min(int(max_rows), int(tbl.num_rows))
    if n <= 0:
        raise RuntimeError("No rows found")

    rng = np.random.default_rng(123)
    idx = rng.choice(n, size=min(int(sample_rows), n), replace=False)

    bad_decode = bad_shape = nonfinite = mad_zero = 0
    clump = []
    src_mode_counts: Dict[str, int] = {}
    art_counts: Dict[str, int] = {}

    for i in idx:
        row = tbl.slice(int(i), 1)
        blob = row["stamp_npz"][0].as_py()
        try:
            arr, bandset_npz = decode_stamp_npz(blob)
        except Exception:
            bad_decode += 1
            continue

        bandset_col = row["bandset"][0].as_py() if "bandset" in row.column_names else None
        bandset = str(bandset_col) if bandset_col not in (None, "") else str(bandset_npz)

        # Normalize to (C,H,W)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim == 3:
            # Allow (H,W,C) legacy
            if arr.shape[-1] in (1, 3) and arr.shape[0] == expect_hw[0] and arr.shape[1] == expect_hw[1]:
                arr = np.transpose(arr, (2, 0, 1))
        else:
            bad_shape += 1
            continue

        if arr.ndim != 3:
            bad_shape += 1
            continue

        C, H, W = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
        if (H, W) != tuple(expect_hw):
            bad_shape += 1
            continue

        expected_channels = int(expect_bands)
        if bandset not in ("unknown", "None"):
            expected_channels = len(bandset)

        # If bandset indicates 1 band but we have 3 channels, allow it (some pipelines store grz even if bandset='r').
        if not (C == expected_channels or (expected_channels == 1 and C == 3)):
            bad_shape += 1
            continue

        if not np.isfinite(arr).all():
            nonfinite += 1
            continue

        # Reference band for MAD/clumpiness: prefer r if present
        r_idx = 1 if C >= 2 else 0
        rband = arr[r_idx].astype(np.float32)

        if robust_mad(rband) < 1e-10:
            mad_zero += 1
            continue

        clump.append(clumpiness_proxy(rband))

        if "source_mode" in row.column_names:
            sm = row["source_mode"][0].as_py()
            src_mode_counts[str(sm)] = src_mode_counts.get(str(sm), 0) + 1
        if "artifact_profile" in row.column_names:
            ap = row["artifact_profile"][0].as_py()
            art_counts[str(ap)] = art_counts.get(str(ap), 0) + 1

    rep = {
        "sampled": int(len(idx)),
        "bad_decode": int(bad_decode),
        "bad_shape": int(bad_shape),
        "nonfinite": int(nonfinite),
        "mad_zero": int(mad_zero),
        "clumpiness_mean": float(np.mean(clump)) if clump else None,
        "clumpiness_p10": float(np.quantile(clump, 0.1)) if clump else None,
        "clumpiness_p90": float(np.quantile(clump, 0.9)) if clump else None,
        "source_mode_counts": src_mode_counts,
        "artifact_profile_counts": art_counts,
    }
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--max-rows", type=int, default=200000)
    ap.add_argument("--sample-rows", type=int, default=2000)
    args = ap.parse_args()
    rep = validate_parquet_stamps(args.parquet, max_rows=args.max_rows, sample_rows=args.sample_rows)
    with open(args.out_json, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

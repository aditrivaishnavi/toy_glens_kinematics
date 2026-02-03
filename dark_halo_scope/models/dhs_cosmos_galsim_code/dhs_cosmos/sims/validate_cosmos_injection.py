"""validate_cosmos_injection.py

Lightweight validator for Parquet stamp datasets that store gzip-compressed NPZ blobs
in a column called `stamp_npz`.

CLI example
-----------
python -m dhs_cosmos.sims.validate_cosmos_injection \
  --parquet "/path/to/stamps/*.parquet" \
  --max-rows 200000 \
  --sample-stamps 500 \
  --out-json validation_report.json

Output
------
JSON summary including:
- row counts scanned
- clumpiness proxy quantiles on sampled positives (r channel)
- band flux sum quantiles on sampled positives

"""from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger("validate_cosmos_injection")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def _decode_stamp_npz(blob: bytes) -> np.ndarray:
    raw = gzip.decompress(blob)
    with np.load(io.BytesIO(raw)) as z:
        if "img" in z:
            arr = z["img"]
        elif "image" in z:
            arr = z["image"]
        else:
            arr = z[z.files[0]]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    return arr


def _gaussian_blur(img: np.ndarray, sigma_pix: float) -> np.ndarray:
    if sigma_pix <= 0:
        return img
    radius = int(np.ceil(3.0 * sigma_pix))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma_pix * sigma_pix))
    k /= np.sum(k)
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis=1, arr=img)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis=0, arr=tmp)
    return out.astype(img.dtype, copy=False)


def _clumpiness_proxy(img2d: np.ndarray) -> float:
    img = np.asarray(img2d, dtype=np.float32)
    total = float(np.sum(img))
    if total <= 0 or not np.isfinite(total):
        return float("nan")
    smooth = _gaussian_blur(img, 1.0)
    resid = np.abs(img - smooth)
    return float(np.sum(resid) / (total + 1e-12))


def _quantiles(x: np.ndarray, qs=(0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {str(q): float("nan") for q in qs}
    vals = np.quantile(x, qs)
    return {str(q): float(v) for q, v in zip(qs, vals)}


def validate_parquet(parquet_glob: str, max_rows: int, sample_stamps: int) -> Dict[str, Any]:
    import pyarrow.dataset as ds

    dataset = ds.dataset(parquet_glob, format="parquet")
    cols = set(dataset.schema.names)
    missing = [c for c in ["stamp_npz", "is_control"] if c not in cols]

    out: Dict[str, Any] = {
        "parquet_glob": parquet_glob,
        "missing_required_columns": missing,
        "max_rows_cap": int(max_rows),
    }
    if missing:
        return out

    rng = np.random.default_rng(1337)
    clumps = []
    band_flux = {"g": [], "r": [], "z": []}

    n = 0
    n_pos = 0
    n_neg = 0

    scanner = dataset.scan(columns=["stamp_npz", "is_control"], batch_size=2048)

    for batch in scanner.to_batches():
        if n >= max_rows:
            break
        tab = batch.to_pydict()
        for blob, is_control in zip(tab["stamp_npz"], tab["is_control"]):
            if n >= max_rows:
                break
            n += 1
            if bool(is_control):
                n_neg += 1
                continue
            n_pos += 1

            # Sample a small number of positives to keep runtime down
            if len(clumps) < sample_stamps and (rng.random() < 0.01 or len(clumps) < 50):
                img = _decode_stamp_npz(blob)
                if img.shape[0] >= 3:
                    g, r, z = img[0], img[1], img[2]
                else:
                    r = img[0]
                    g = r
                    z = r
                clumps.append(_clumpiness_proxy(r))
                band_flux["g"].append(float(np.sum(g)))
                band_flux["r"].append(float(np.sum(r)))
                band_flux["z"].append(float(np.sum(z)))

    out.update({
        "rows_scanned": int(n),
        "positives_scanned": int(n_pos),
        "negatives_scanned": int(n_neg),
        "sampled_stamps": int(len(clumps)),
        "clumpiness_r_quantiles": _quantiles(np.asarray(clumps, dtype=np.float32)),
        "band_flux_sum_quantiles": {k: _quantiles(np.asarray(v, dtype=np.float32)) for k, v in band_flux.items()},
    })
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--max-rows", type=int, default=200000)
    ap.add_argument("--sample-stamps", type=int, default=500)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)
    report = validate_parquet(args.parquet, args.max_rows, args.sample_stamps)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    logger.info("Wrote %s", args.out_json)


if __name__ == "__main__":
    main()

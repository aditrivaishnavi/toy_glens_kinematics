"""
Gen6: Ground-based deep cutouts as sources (HSC/Legacy deep fields).

IMPORTANT UNITS NOTE
--------------------
This module stores templates in the same units as the input FITS images, typically flux-per-pixel.
If you use these templates with lenstronomy LightModel("INTERPOL"), you MUST convert to surface
brightness (flux per arcsec^2) before passing as kwargs_source['image'].

Helper:
    from dhs_gen.utils import to_surface_brightness
    sb = to_surface_brightness(template_flux_per_pixel, pixscale_arcsec)

If you also apply a target flux scaling (e.g., flux_nmgy), apply it before the conversion.

Dataset-agnostic workflow:
1) Put deep cutout FITS files in a directory (single-band is fine).
2) Build a compact template bank NPZ.
3) In Stage 4c, deterministically select a template per task_id, lens it, PSF convolve, and inject.

Why this helps:
Ground-based deep cutouts better match the texture statistics and seeing/noise regime of ground-based surveys
than HST images downsampled to 0.262"/pix.

CLI:
    python -m dhs_gen.deep_sources.deep_source_bank \
      --fits-dir /path/to/deep_cutouts \
      --out-npz deep_bank_20k_96px.npz \
      --n-sources 20000 \
      --stamp-size 96 \
      --src-pixscale-arcsec 0.168 \
      --target-pixscale-arcsec 0.262
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils import bilinear_resample, pad_or_crop_center, rng_from_hash


def _read_fits_image(path: str) -> Tuple[np.ndarray, Dict]:
    try:
        from astropy.io import fits
    except Exception as e:
        raise RuntimeError("astropy required: pip install astropy") from e

    with fits.open(path, memmap=False) as hdul:
        hdu = hdul[0]
        img = np.asarray(hdu.data, dtype=np.float32)
        hdr = dict(hdu.header)

    if img.ndim == 3:
        img = img[0]
    if img.ndim != 2:
        raise ValueError(f"Unsupported ndim={img.ndim} in {path}")
    return img, hdr


def _robust_background_subtract(img: np.ndarray) -> np.ndarray:
    border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
    bg = float(np.median(border))
    return img - bg


def build_deep_source_bank(
    fits_dir: str,
    out_npz: str,
    n_sources: int,
    stamp_size: int,
    src_pixscale_arcsec: float,
    target_pixscale_arcsec: float,
    seed: str = "1337",
    max_files: Optional[int] = None,
    min_abs_sum: float = 0.0,
) -> None:
    fits_paths = sorted([str(p) for p in Path(fits_dir).rglob("*.fits*")])
    if not fits_paths:
        raise FileNotFoundError(f"No FITS found under {fits_dir}")

    if max_files is not None:
        fits_paths = fits_paths[:max_files]

    rng = rng_from_hash(seed)
    rng.shuffle(fits_paths)

    images: List[np.ndarray] = []
    meta_file: List[str] = []
    meta_pixscale: List[float] = []
    meta_filter: List[str] = []
    meta_sum: List[float] = []

    for p in fits_paths:
        try:
            img, hdr = _read_fits_image(p)
        except Exception:
            continue

        img = _robust_background_subtract(img)
        if not np.isfinite(img).all():
            continue

        ps = hdr.get("PIXSCALE", None)
        ps = float(ps) if ps is not None else float(src_pixscale_arcsec)

        # Resample to target pixel scale (arcsec/pix)
        scale = ps / target_pixscale_arcsec
        img_rs = bilinear_resample(img, scale_y=scale, scale_x=scale)
        img_rs = pad_or_crop_center(img_rs, (stamp_size, stamp_size))
        if not np.isfinite(img_rs).all():
            # Defensive: avoid propagating bad values into the bank
            continue

        s = float(np.sum(np.abs(img_rs)))
        if s <= min_abs_sum:
            continue

        images.append(img_rs.astype(np.float32))
        meta_file.append(os.path.basename(p))
        meta_pixscale.append(ps)
        meta_filter.append(str(hdr.get("FILTER", hdr.get("BAND", "unknown"))))
        meta_sum.append(float(np.sum(img_rs)))

        if len(images) >= n_sources:
            break

    if len(images) < max(100, int(0.1 * n_sources)):
        raise RuntimeError(f"Only built {len(images)} sources. Check inputs/thresholds.")

    arr = np.stack(images, axis=0)
    if not np.isfinite(arr).all():
        raise ValueError('NaN/Inf detected in deep source bank array')
    meta = {
        "file": meta_file,
        "pixscale_arcsec": meta_pixscale,
        "filter": meta_filter,
        "sum_flux": meta_sum,
        "target_pixscale_arcsec": float(target_pixscale_arcsec),
        "stamp_size": int(stamp_size),
    }
    np.savez_compressed(out_npz, images=arr, meta=json.dumps(meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits-dir", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--n-sources", type=int, default=20000)
    ap.add_argument("--stamp-size", type=int, default=96)
    ap.add_argument("--src-pixscale-arcsec", type=float, default=0.168)
    ap.add_argument("--target-pixscale-arcsec", type=float, default=0.262)
    ap.add_argument("--seed", default="1337")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--min-abs-sum", type=float, default=0.0)
    args = ap.parse_args()
    build_deep_source_bank(
        fits_dir=args.fits_dir,
        out_npz=args.out_npz,
        n_sources=args.n_sources,
        stamp_size=args.stamp_size,
        src_pixscale_arcsec=args.src_pixscale_arcsec,
        target_pixscale_arcsec=args.target_pixscale_arcsec,
        seed=args.seed,
        max_files=args.max_files,
        min_abs_sum=args.min_abs_sum,
    )
    print(f"Wrote deep bank: {args.out_npz}")


if __name__ == "__main__":
    main()

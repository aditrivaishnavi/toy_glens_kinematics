"""cosmos_source_loader.py

Build a reusable bank of real galaxy source templates from the GalSim COSMOS catalog.

This module samples COSMOS galaxies, renders them onto a fixed stamp, and stores
the resulting images and basic metadata into an HDF5 file.

It is designed for strong-lens injection work where source morphology realism matters.

CLI example
-----------
python -m dhs_cosmos.sims.cosmos_source_loader \
  --cosmos-dir /data/COSMOS \
  --out-h5 cosmos_sources_20000.h5 \
  --n-sources 20000 \
  --stamp-size 128 \
  --src-pixscale-arcsec 0.03 \
  --intrinsic-psf-fwhm-arcsec 0.10 \
  --dtype float16

Output HDF5 structure
---------------------
/images              (N,H,W) float16/float32, unit-flux templates
/meta/index          (N,) int64 COSMOS catalog indices
/meta/hlr_arcsec     (N,) float32 half-light radius estimate in arcsec
/meta/clumpiness     (N,) float32 clumpiness proxy

File attributes:
- src_pixscale_arcsec
- stamp_size
- created_utc
- galaxy_kind

Notes
-----
- GalSim supports COSMOS loading via COSMOSCatalog or RealGalaxyCatalog.
  This module tries both.
- The stored images are normalized to unit total flux. Flux scaling is applied during injection.

"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
from dataclasses import dataclass

import numpy as np
import h5py

logger = logging.getLogger("cosmos_source_loader")


@dataclass(frozen=True)
class BuildConfig:
    cosmos_dir: str
    out_h5: str
    n_sources: int
    stamp_size: int
    src_pixscale_arcsec: float
    seed: int
    dtype: str
    intrinsic_psf_fwhm_arcsec: float
    denoise_sigma_pix: float
    max_tries: int


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def _try_load_cosmos_catalog(cosmos_dir: str):
    """Return (catalog, n, get_gal, kind)."""
    import galsim  # local import so this file imports without GalSim installed

    # Attempt 1: COSMOSCatalog (preferred in recent GalSim)
    try:
        cat = galsim.COSMOSCatalog(dir=cosmos_dir)
        n = len(cat)

        def get_gal(i: int):
            return cat.makeGalaxy(index=i)

        logger.info("Loaded GalSim COSMOSCatalog with %d entries", n)
        return cat, n, get_gal, "COSMOSCatalog"
    except Exception as e1:
        logger.debug("COSMOSCatalog load failed: %s", repr(e1))

    # Attempt 2: RealGalaxyCatalog
    try:
        cat = galsim.RealGalaxyCatalog(dir=cosmos_dir)
        n = cat.nobjects

        def get_gal(i: int):
            return galsim.RealGalaxy(catalog=cat, index=i)

        logger.info("Loaded GalSim RealGalaxyCatalog with %d entries", n)
        return cat, n, get_gal, "RealGalaxyCatalog"
    except Exception as e2:
        raise RuntimeError(
            "Could not load COSMOS catalog via GalSim. Tried COSMOSCatalog and RealGalaxyCatalog. "
            f"COSMOSCatalog error: {repr(e1)} ; RealGalaxyCatalog error: {repr(e2)}"
        )


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


def _half_light_radius_arcsec(img: np.ndarray, pixscale: float) -> float:
    img = np.asarray(img, dtype=np.float64)
    total = img.sum()
    if not np.isfinite(total) or total <= 0:
        return float("nan")

    h, w = img.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy, xx = np.indices(img.shape, dtype=np.float64)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).ravel()
    vv = img.ravel()
    order = np.argsort(rr)
    rr_sorted = rr[order]
    vv_sorted = vv[order]
    cumsum = np.cumsum(vv_sorted)
    half = 0.5 * total
    idx = np.searchsorted(cumsum, half)
    idx = int(np.clip(idx, 0, len(rr_sorted) - 1))
    return float(rr_sorted[idx] * pixscale)


def _clumpiness_proxy(img: np.ndarray, sigma_pix: float = 1.0) -> float:
    img = np.asarray(img, dtype=np.float32)
    total = float(np.sum(img))
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    smooth = _gaussian_blur(img, sigma_pix)
    resid = np.abs(img - smooth)
    return float(np.sum(resid) / (total + 1e-12))


def build_cosmos_bank(cfg: BuildConfig) -> None:
    import galsim  # local import

    _, ncat, get_gal, kind = _try_load_cosmos_catalog(cfg.cosmos_dir)
    rng = np.random.default_rng(cfg.seed)

    dtype_np = np.float16 if cfg.dtype == "float16" else np.float32
    H = W = int(cfg.stamp_size)
    images = np.zeros((cfg.n_sources, H, W), dtype=dtype_np)
    hlr = np.full((cfg.n_sources,), np.nan, dtype=np.float32)
    clump = np.full((cfg.n_sources,), np.nan, dtype=np.float32)
    kept_idx = np.full((cfg.n_sources,), -1, dtype=np.int64)

    intrinsic_psf = None
    if cfg.intrinsic_psf_fwhm_arcsec > 0:
        intrinsic_psf = galsim.Gaussian(sigma=cfg.intrinsic_psf_fwhm_arcsec / 2.355)

    ok = 0
    tries = 0
    logger.info(
        "Building COSMOS bank: target=%d stamp=%dx%d pixscale=%.4f arcsec/pix",
        cfg.n_sources, H, W, cfg.src_pixscale_arcsec,
    )

    while ok < cfg.n_sources and tries < cfg.max_tries:
        i = int(rng.integers(0, ncat))
        tries += 1
        try:
            gal = get_gal(i)
            if intrinsic_psf is not None:
                gal = galsim.Convolve([gal, intrinsic_psf])

            im = gal.drawImage(nx=W, ny=H, scale=cfg.src_pixscale_arcsec, method="auto").array.astype(np.float32)

            if cfg.denoise_sigma_pix > 0:
                im = _gaussian_blur(im, cfg.denoise_sigma_pix)

            im = np.maximum(im, 0.0)
            s = float(im.sum())
            if not np.isfinite(s) or s <= 0:
                continue
            im /= s

            images[ok] = im.astype(dtype_np)
            hlr[ok] = _half_light_radius_arcsec(im, cfg.src_pixscale_arcsec)
            clump[ok] = _clumpiness_proxy(im, sigma_pix=1.0)
            kept_idx[ok] = i
            ok += 1
        except Exception:
            continue

    if ok < cfg.n_sources:
        raise RuntimeError(f"Only built {ok}/{cfg.n_sources} templates after {tries} tries. Check COSMOS files.")

    created = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    with h5py.File(cfg.out_h5, "w") as f:
        f.create_dataset("images", data=images, compression="gzip", compression_opts=4, shuffle=True)
        g = f.create_group("meta")
        g.create_dataset("index", data=kept_idx)
        g.create_dataset("hlr_arcsec", data=hlr)
        g.create_dataset("clumpiness", data=clump)
        f.attrs["src_pixscale_arcsec"] = float(cfg.src_pixscale_arcsec)
        f.attrs["stamp_size"] = int(cfg.stamp_size)
        f.attrs["created_utc"] = created
        f.attrs["galaxy_kind"] = kind

    logger.info("Wrote COSMOS bank: %s (N=%d)", cfg.out_h5, cfg.n_sources)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosmos-dir", required=True)
    ap.add_argument("--out-h5", required=True)
    ap.add_argument("--n-sources", type=int, default=20000)
    ap.add_argument("--stamp-size", type=int, default=128)
    ap.add_argument("--src-pixscale-arcsec", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--intrinsic-psf-fwhm-arcsec", type=float, default=0.10)
    ap.add_argument("--denoise-sigma-pix", type=float, default=0.0)
    ap.add_argument("--max-tries", type=int, default=400000)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)
    cfg = BuildConfig(
        cosmos_dir=args.cosmos_dir,
        out_h5=args.out_h5,
        n_sources=args.n_sources,
        stamp_size=args.stamp_size,
        src_pixscale_arcsec=args.src_pixscale_arcsec,
        seed=args.seed,
        dtype=args.dtype,
        intrinsic_psf_fwhm_arcsec=args.intrinsic_psf_fwhm_arcsec,
        denoise_sigma_pix=args.denoise_sigma_pix,
        max_tries=args.max_tries,
    )
    build_cosmos_bank(cfg)


if __name__ == "__main__":
    main()

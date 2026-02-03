"""cosmos_lens_injector.py

Debug tool to inject a COSMOS template through an SIE + shear lens model
onto a target background cutout for inspection.

CLI example
-----------
python -m dhs_cosmos.sims.cosmos_lens_injector \
  --cosmos-h5 cosmos_sources_20000.h5 \
  --cosmos-index 123 \
  --cutout-fits background_r.fits \
  --theta-e 1.5 \
  --e1 0.0 --e2 0.0 \
  --gamma1 0.02 --gamma2 0.00 \
  --src-x 0.05 --src-y -0.03 \
  --psf-fwhm-arcsec 1.1 \
  --psf-type moffat \
  --src-mag-r 22.8 \
  --out-fits injected_r.fits

Notes
-----
- Requires lenstronomy and astropy.
- Uses a simple g/r/z SED offset model for star-forming sources.

"""from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import h5py

logger = logging.getLogger("cosmos_lens_injector")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


@dataclass(frozen=True)
class InjectionParams:
    theta_e_arcsec: float
    e1: float
    e2: float
    gamma1: float
    gamma2: float
    src_x_arcsec: float
    src_y_arcsec: float
    src_mag_r: float
    z_s: float = 1.5


class CosmosBank:
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self._f = None
        self.images = None
        self.meta = None
        self.src_pixscale_arcsec = None

    def __enter__(self):
        self._f = h5py.File(self.h5_path, "r")
        self.images = self._f["images"]
        self.meta = self._f["meta"]
        self.src_pixscale_arcsec = float(self._f.attrs["src_pixscale_arcsec"])
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._f is not None:
            self._f.close()
            self._f = None

    def get_template_unitflux(self, idx: int) -> np.ndarray:
        img = np.asarray(self.images[idx][:], dtype=np.float32)
        s = float(img.sum())
        if not np.isfinite(s) or s <= 0:
            raise ValueError("Template has non-positive flux")
        return img / s


def _mag_to_nmgy(mag: float, zp: float = 22.5) -> float:
    return float(10.0 ** ((zp - mag) / 2.5))


def _sed_offsets(z_s: float) -> Dict[str, float]:
    z_factor = np.clip((z_s - 1.0) / 2.0, 0.0, 1.0)
    return {
        "g": -0.30 * (1.0 + 0.30 * z_factor),
        "r": 0.0,
        "z": +0.20 * (1.0 + 0.20 * z_factor),
    }


def _psf_kernel(psf_fwhm_arcsec: float, pixscale_arcsec: float, psf_type: str, beta: float = 3.5) -> np.ndarray:
    sigma = psf_fwhm_arcsec / 2.355
    ksize = int(max(11, (4.0 * psf_fwhm_arcsec / pixscale_arcsec))) | 1
    y, x = np.ogrid[:ksize, :ksize]
    c = ksize // 2
    rr2 = (x - c) ** 2 + (y - c) ** 2
    if psf_type == "gaussian":
        ker = np.exp(-rr2 / (2.0 * (sigma / pixscale_arcsec) ** 2))
    else:
        alpha = psf_fwhm_arcsec / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
        ker = (1.0 + (rr2 * (pixscale_arcsec ** 2)) / (alpha ** 2)) ** (-beta)
    ker = ker.astype(np.float64)
    ker /= np.sum(ker)
    return ker


def render_lensed_arc_lenstronomy(
    template_unitflux: np.ndarray,
    template_scale_arcsec: float,
    out_shape: Tuple[int, int],
    out_pixscale_arcsec: float,
    psf_fwhm_arcsec: float,
    psf_type: str,
    params: InjectionParams,
    band: str,
) -> np.ndarray:
    try:
        from lenstronomy.LensModel.lens_model import LensModel
        from lenstronomy.LightModel.light_model import LightModel
        from lenstronomy.ImSim.image_model import ImageModel
        from lenstronomy.Data.imaging_data import ImageData
        from lenstronomy.Data.psf import PSF
    except Exception as e:
        raise ImportError("lenstronomy is required. Install with: pip install lenstronomy") from e

    ny, nx = out_shape
    ra_at_xy_0 = -(nx / 2.0) * out_pixscale_arcsec
    dec_at_xy_0 = -(ny / 2.0) * out_pixscale_arcsec

    data = ImageData(
        image_data=np.zeros((ny, nx), dtype=np.float32),
        ra_at_xy_0=ra_at_xy_0,
        dec_at_xy_0=dec_at_xy_0,
        transform_pix2angle=np.array([[out_pixscale_arcsec, 0.0], [0.0, out_pixscale_arcsec]], dtype=np.float64),
    )

    ker = _psf_kernel(psf_fwhm_arcsec, out_pixscale_arcsec, psf_type)
    psf = PSF(psf_type="PIXEL", kernel_point_source=ker)

    lens_model = LensModel(["SIE", "SHEAR"])
    kwargs_lens = [
        {"theta_E": params.theta_e_arcsec, "e1": params.e1, "e2": params.e2, "center_x": 0.0, "center_y": 0.0},
        {"gamma1": params.gamma1, "gamma2": params.gamma2, "ra_0": 0.0, "dec_0": 0.0},
    ]

    light_model = LightModel(["INTERPOL"])

    sed = _sed_offsets(params.z_s)
    mag_band = params.src_mag_r + sed.get(band, 0.0)
    flux_nmgy = _mag_to_nmgy(mag_band)

    kwargs_source = [{
        "image": template_unitflux * flux_nmgy,
        "center_x": params.src_x_arcsec,
        "center_y": params.src_y_arcsec,
        "scale": template_scale_arcsec,
        "phi_G": 0.0,
    }]

    image_model = ImageModel(data, psf, lens_model_class=lens_model, source_model_class=light_model)
    arc = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source, kwargs_lens_light=None, kwargs_ps=None)
    return arc.astype(np.float32)


def _read_fits(path: str) -> np.ndarray:
    from astropy.io import fits
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float32)


def _write_fits(path: str, data: np.ndarray) -> None:
    from astropy.io import fits
    fits.PrimaryHDU(data.astype(np.float32)).writeto(path, overwrite=True)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosmos-h5", required=True)
    ap.add_argument("--cosmos-index", type=int, required=True)
    ap.add_argument("--cutout-fits", required=True)
    ap.add_argument("--out-fits", required=True)

    ap.add_argument("--theta-e", type=float, required=True)
    ap.add_argument("--e1", type=float, default=0.0)
    ap.add_argument("--e2", type=float, default=0.0)
    ap.add_argument("--gamma1", type=float, default=0.0)
    ap.add_argument("--gamma2", type=float, default=0.0)
    ap.add_argument("--src-x", type=float, default=0.0)
    ap.add_argument("--src-y", type=float, default=0.0)

    ap.add_argument("--psf-fwhm-arcsec", type=float, default=1.1)
    ap.add_argument("--psf-type", choices=["gaussian", "moffat"], default="moffat")
    ap.add_argument("--out-pixscale-arcsec", type=float, default=0.262)
    ap.add_argument("--band", choices=["g", "r", "z"], default="r")

    ap.add_argument("--src-mag-r", type=float, required=True)
    ap.add_argument("--z-s", type=float, default=1.5)

    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    bg = _read_fits(args.cutout_fits)
    ny, nx = bg.shape

    with CosmosBank(args.cosmos_h5) as bank:
        template = bank.get_template_unitflux(args.cosmos_index)
        params = InjectionParams(
            theta_e_arcsec=args.theta_e,
            e1=args.e1,
            e2=args.e2,
            gamma1=args.gamma1,
            gamma2=args.gamma2,
            src_x_arcsec=args.src_x,
            src_y_arcsec=args.src_y,
            src_mag_r=args.src_mag_r,
            z_s=args.z_s,
        )
        arc = render_lensed_arc_lenstronomy(
            template_unitflux=template,
            template_scale_arcsec=bank.src_pixscale_arcsec,
            out_shape=(ny, nx),
            out_pixscale_arcsec=args.out_pixscale_arcsec,
            psf_fwhm_arcsec=args.psf_fwhm_arcsec,
            psf_type=args.psf_type,
            params=params,
            band=args.band,
        )

    _write_fits(args.out_fits, bg + arc)
    logger.info("Wrote %s", args.out_fits)


if __name__ == "__main__":
    main()

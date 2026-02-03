"""spark_cosmos_udf_example.py

Illustrative Spark integration pattern for COSMOS templates.

Recommendation:
- Prefer mapInPandas over a Python UDF for heavy per-row image operations.
- Use executor-local caching of the COSMOS bank file.

This example assumes you have a COSMOS bank HDF5 file on each node at:
  /mnt/cosmos_sources.h5

and that your DataFrame includes:
- task_id
- theta_e_arcsec, e1, e2, gamma1, gamma2, src_x_arcsec, src_y_arcsec, src_mag_r, z_s
- psf_fwhm_r
- background_r (2D numpy array)

"""from __future__ import annotations

import hashlib
from typing import Iterator

import pandas as pd

from dhs_cosmos.sims.cosmos_lens_injector import CosmosBank, InjectionParams, render_lensed_arc_lenstronomy


_BANK = None


def _get_bank(path: str) -> CosmosBank:
    global _BANK
    if _BANK is None:
        _BANK = CosmosBank(path)
        _BANK.__enter__()
    return _BANK


def _choose_cosmos_index(task_id: str, n_sources: int) -> int:
    h = hashlib.blake2b(str(task_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % int(n_sources)


def inject_partition(pdf: pd.DataFrame) -> pd.DataFrame:
    bank = _get_bank("/mnt/cosmos_sources.h5")
    n_sources = bank.images.shape[0]

    out_rows = []
    for row in pdf.itertuples(index=False):
        cosmos_idx = _choose_cosmos_index(getattr(row, "task_id"), n_sources)
        template = bank.get_template_unitflux(cosmos_idx)

        bg = getattr(row, "background_r")
        ny, nx = bg.shape

        params = InjectionParams(
            theta_e_arcsec=getattr(row, "theta_e_arcsec"),
            e1=getattr(row, "e1"),
            e2=getattr(row, "e2"),
            gamma1=getattr(row, "gamma1"),
            gamma2=getattr(row, "gamma2"),
            src_x_arcsec=getattr(row, "src_x_arcsec"),
            src_y_arcsec=getattr(row, "src_y_arcsec"),
            src_mag_r=getattr(row, "src_mag_r"),
            z_s=getattr(row, "z_s"),
        )

        arc_r = render_lensed_arc_lenstronomy(
            template_unitflux=template,
            template_scale_arcsec=bank.src_pixscale_arcsec,
            out_shape=(ny, nx),
            out_pixscale_arcsec=0.262,
            psf_fwhm_arcsec=getattr(row, "psf_fwhm_r"),
            psf_type="moffat",
            params=params,
            band="r",
        )

        injected_r = bg + arc_r
        out_rows.append({
            "task_id": getattr(row, "task_id"),
            "cosmos_index": cosmos_idx,
            "injected_r": injected_r,
        })

    return pd.DataFrame(out_rows)

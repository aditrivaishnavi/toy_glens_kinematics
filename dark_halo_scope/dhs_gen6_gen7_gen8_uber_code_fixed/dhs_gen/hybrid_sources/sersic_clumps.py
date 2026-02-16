"""
Gen7: Hybrid sources = smooth Sérsic + explicit clumps + optional gradients.

Purpose:
- Controlled realism knob: introduce clumpiness/irregularity without importing HST-specific noise.
- Makes ablations publishable (you can show how each realism knob changes the selection function).

CLI:
    python -m dhs_gen.hybrid_sources.sersic_clumps --key task_123 --H 96 --W 96
"""
from __future__ import annotations
import argparse
import json
import math
from typing import Dict, Tuple

import numpy as np

from ..utils import rng_from_hash


def _sersic_2d(H: int, W: int, re_pix: float, n: float, q: float, phi: float, x0: float, y0: float) -> np.ndarray:
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    x = x - x0
    y = y - y0
    c, s = math.cos(phi), math.sin(phi)
    xr = c * x + s * y
    yr = -s * x + c * y
    yr = yr / max(0.3, q)
    R = np.sqrt(xr**2 + yr**2) + 1e-6
    bn = 2.0 * n - 1.0 / 3.0
    I = np.exp(-bn * ((R / max(1e-3, re_pix)) ** (1.0 / n) - 1.0))
    return I


def _add_gaussian_clump(img: np.ndarray, amp: float, sigma_pix: float, x0: float, y0: float) -> None:
    H, W = img.shape
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = x - x0
    dy = y - y0
    g = np.exp(-(dx**2 + dy**2) / (2.0 * sigma_pix**2))
    img += amp * g


def generate_hybrid_source(
    key: str,
    H: int = 96,
    W: int = 96,
    re_pix: float = 6.0,
    n_sersic: float = 1.0,
    q: float = 0.8,
    n_clumps_range: Tuple[int, int] = (2, 6),
    clump_sigma_pix_range: Tuple[float, float] = (0.8, 2.0),
    clump_flux_frac_range: Tuple[float, float] = (0.05, 0.25),
    gradient_strength: float = 0.2,
    normalize_sum: bool = True,
    salt: str = "",
) -> Dict[str, object]:
    rng = rng_from_hash(key, salt=salt)
    phi = float(rng.uniform(0, 2 * math.pi))
    q0 = float(np.clip(q + rng.normal(0, 0.08), 0.4, 1.0))
    re0 = float(max(1.5, re_pix * rng.uniform(0.7, 1.3)))
    n0 = float(np.clip(n_sersic + rng.normal(0, 0.15), 0.6, 2.5))
    x0 = (W - 1) / 2 + float(rng.normal(0, 0.7))
    y0 = (H - 1) / 2 + float(rng.normal(0, 0.7))

    base = _sersic_2d(H, W, re_pix=re0, n=n0, q=q0, phi=phi, x0=x0, y0=y0)

    nmin, nmax = n_clumps_range
    ncl = int(rng.integers(nmin, nmax + 1))
    cl = np.zeros((H, W), dtype=np.float32)

    for _ in range(ncl):
        r = float(abs(rng.normal(loc=re0, scale=0.6 * re0)))
        ang = float(rng.uniform(0, 2 * math.pi))
        cx = x0 + r * math.cos(ang)
        cy = y0 + r * math.sin(ang)
        sig = float(rng.uniform(*clump_sigma_pix_range))
        frac = float(rng.uniform(*clump_flux_frac_range))
        _add_gaussian_clump(cl, amp=frac, sigma_pix=sig, x0=cx, y0=cy)

    img = base + cl

    if gradient_strength > 0:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        gx = (xx - (W - 1) / 2) / max(1.0, (W / 2))
        gy = (yy - (H - 1) / 2) / max(1.0, (H / 2))
        grad = 1.0 + gradient_strength * (0.6 * gx + 0.4 * gy)
        img *= np.clip(grad, 0.2, 2.0)

    img = np.clip(img, 0, None)

    if normalize_sum:
        img = img / max(1e-12, float(img.sum()))

    meta = {
        "n_clumps": ncl,
        "re_pix": re0,
        "n_sersic": n0,
        "q": q0,
        "phi": phi,
        "x0": x0,
        "y0": y0,
        "gradient_strength": gradient_strength,
    }
    if not np.isfinite(img).all():
        raise ValueError(f"NaN/Inf detected in generated hybrid source for key={key}")

    return {"img": img.astype(np.float32), "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--salt", default="")
    ap.add_argument("--H", type=int, default=96)
    ap.add_argument("--W", type=int, default=96)
    ap.add_argument("--re-pix", type=float, default=6.0)
    args = ap.parse_args()
    out = generate_hybrid_source(args.key, H=args.H, W=args.W, re_pix=args.re_pix, salt=args.salt)
    print(json.dumps({"sum": float(out["img"].sum()), "meta": out["meta"]}, indent=2))


if __name__ == "__main__":
    main()

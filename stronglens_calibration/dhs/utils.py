from __future__ import annotations
import io
import numpy as np

def decode_npz_blob(blob: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(blob)) as z:
        return {k: z[k] for k in z.files}

def center_slices(H: int, W: int, box: int):
    cy, cx = H // 2, W // 2
    half = box // 2
    return slice(cy - half, cy + half), slice(cx - half, cx + half)

def radial_rmap(H: int, W: int) -> np.ndarray:
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

def radial_mask(H: int, W: int, r_in: float, r_out: float) -> np.ndarray:
    r = radial_rmap(H, W)
    return (r >= r_in) & (r < r_out)

def robust_median_mad(x: np.ndarray, eps: float = 1e-8):
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + eps
    return med, mad

def normalize_outer_annulus(img: np.ndarray, r_in: float = 20, r_out: float = 32) -> np.ndarray:
    H, W = img.shape
    m = radial_mask(H, W, r_in, r_out)
    med, mad = robust_median_mad(img[m])
    return (img - med) / mad

def azimuthal_median_profile(img: np.ndarray, r_max: int = 32) -> np.ndarray:
    H, W = img.shape
    r = radial_rmap(H, W)
    prof = np.zeros(r_max, dtype=np.float32)
    for ri in range(r_max):
        m = (r >= ri) & (r < ri + 1)
        prof[ri] = np.median(img[m]).astype(np.float32) if np.any(m) else 0.0
    return prof

def radial_profile_model(img: np.ndarray, r_max: int = 32) -> np.ndarray:
    H, W = img.shape
    r = radial_rmap(H, W)
    prof = azimuthal_median_profile(img, r_max=r_max)
    rr = np.clip(np.floor(r).astype(int), 0, r_max - 1)
    return prof[rr].astype(np.float32)

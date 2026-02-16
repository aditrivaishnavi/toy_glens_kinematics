from __future__ import annotations
import numpy as np
from .utils import normalize_outer_annulus, radial_profile_model

def preprocess_stack(img3: np.ndarray, mode: str) -> np.ndarray:
    assert img3.ndim == 3 and img3.shape[0] == 3
    out = []
    for b in range(3):
        x = img3[b].astype(np.float32)
        if mode == "raw_robust":
            x = normalize_outer_annulus(x)
        elif mode == "residual_radial_profile":
            xn = normalize_outer_annulus(x)
            model = radial_profile_model(xn)
            x = (xn - model).astype(np.float32)
        else:
            raise ValueError(f"Unknown preprocessing mode: {mode}")
        out.append(x)
    return np.stack(out, axis=0).astype(np.float32)

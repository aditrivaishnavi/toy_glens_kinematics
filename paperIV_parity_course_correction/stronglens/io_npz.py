from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np

@dataclass
class Cutout:
    image: np.ndarray  # (H,W,3) float32
    meta: Dict[str, Any]

def load_npz_cutout(path: str) -> Cutout:
    """Load a .npz produced by the pipeline.
    Expected key: 'cutout' with shape (101,101,3) in g/r/z channel order.
    Metadata keys may be stored as meta_<key>.
    """
    with np.load(path, allow_pickle=False) as z:
        if "cutout" not in z:
            raise KeyError(f"Missing 'cutout' array in {path}")
        img = z["cutout"].astype(np.float32)
        meta = {}
        for k in z.files:
            if k.startswith("meta_"):
                meta[k[len("meta_"):]] = z[k].tolist() if z[k].shape == () else z[k]
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3); got {img.shape} for {path}")
    return Cutout(image=img, meta=meta)

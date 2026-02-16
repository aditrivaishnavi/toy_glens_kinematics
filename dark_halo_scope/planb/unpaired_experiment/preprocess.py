"""Preprocessing functions for unpaired experiment.

Two modes:
1. raw_robust: Normalize by outer annulus median/MAD
2. residual_radial_profile: Subtract azimuthal median profile (removes radial structure)
"""
from __future__ import annotations
import numpy as np

from .utils import normalize_outer_annulus, radial_profile_model
from .constants import BANDS


def preprocess_stack(img3: np.ndarray, mode: str = "raw_robust") -> np.ndarray:
    """
    Preprocess 3-band image stack.
    
    Args:
        img3: (3, H, W) float32 array
        mode: "raw_robust" or "residual_radial_profile"
    
    Returns:
        Preprocessed (3, H, W) float32 array
    """
    assert img3.ndim == 3 and img3.shape[0] == 3, f"Expected (3, H, W), got {img3.shape}"
    
    out = []
    for b in range(3):
        x = img3[b].astype(np.float32)
        
        if mode == "raw_robust":
            # Normalize by outer annulus
            x = normalize_outer_annulus(x)
            
        elif mode == "residual_radial_profile":
            # First normalize, then subtract radial profile
            xn = normalize_outer_annulus(x)
            model = radial_profile_model(xn)
            x = (xn - model).astype(np.float32)
            # Clean up any NaN/Inf from residual
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
        else:
            raise ValueError(f"Unknown preprocessing mode: {mode}")
        
        out.append(x)
    
    result = np.stack(out, axis=0).astype(np.float32)
    # Final NaN check and clip extreme values
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    result = np.clip(result, -50.0, 50.0)  # Clip extreme values to prevent model instability
    return result

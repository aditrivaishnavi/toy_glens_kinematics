from __future__ import annotations
import numpy as np
from .utils import normalize_outer_annulus, radial_profile_model
from .constants import STAMP_SIZE


def center_crop(img: np.ndarray, target_size: int) -> np.ndarray:
    """Center crop a 2D image to target_size x target_size.
    
    Pipeline generates 101x101 cutouts, training expects 64x64.
    This extracts the central region.
    """
    h, w = img.shape[-2:]
    if h == target_size and w == target_size:
        return img
    if h < target_size or w < target_size:
        raise ValueError(f"Image {h}x{w} smaller than target {target_size}x{target_size}")
    
    start_h = (h - target_size) // 2
    start_w = (w - target_size) // 2
    
    if img.ndim == 2:
        return img[start_h:start_h + target_size, start_w:start_w + target_size]
    elif img.ndim == 3:
        return img[:, start_h:start_h + target_size, start_w:start_w + target_size]
    else:
        raise ValueError(f"Unexpected ndim={img.ndim}")


def preprocess_stack(img3: np.ndarray, mode: str, crop: bool = True, 
                     crop_size: int | None = None,
                     clip_range: float = 10.0) -> np.ndarray:
    """Preprocess a 3-band image stack.
    
    Args:
        img3: (3, H, W) array with g, r, z bands
        mode: Preprocessing mode ('raw_robust' or 'residual_radial_profile')
        crop: If True, center crop to crop_size (default STAMP_SIZE=64).
              For Paper IV parity, set crop=False to keep 101x101.
        crop_size: Target crop size. None defaults to STAMP_SIZE (64).
                   Set to 0 or None with crop=False to skip cropping.
        clip_range: Clip normalized values to [-clip_range, clip_range]
        
    Returns:
        Preprocessed (3, target_size, target_size) array
    """
    assert img3.ndim == 3 and img3.shape[0] == 3
    
    # Determine target crop size
    target = crop_size if crop_size is not None else STAMP_SIZE
    
    # Apply center crop first if needed (101x101 -> target)
    if crop and target > 0 and img3.shape[1] != target:
        img3 = center_crop(img3, target)
    
    out = []
    for b in range(3):
        x = img3[b].astype(np.float32)
        
        # Handle NaN values - replace with 0 before normalization
        nan_mask = np.isnan(x)
        if nan_mask.any():
            x = np.where(nan_mask, 0.0, x)
        
        if mode == "raw_robust":
            x = normalize_outer_annulus(x)
        elif mode == "residual_radial_profile":
            xn = normalize_outer_annulus(x)
            model = radial_profile_model(xn)
            x = (xn - model).astype(np.float32)
        else:
            raise ValueError(f"Unknown preprocessing mode: {mode}")
        
        # Final NaN check - set any remaining NaN to 0
        if np.isnan(x).any():
            x = np.nan_to_num(x, nan=0.0)
        
        # Clip extreme values to prevent gradient explosion
        x = np.clip(x, -clip_range, clip_range)
        
        out.append(x)
    return np.stack(out, axis=0).astype(np.float32)

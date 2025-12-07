"""
Prepare source galaxy maps for ML pipeline.

This module will provide functions to:
- Normalize flux and velocity maps
- Resample maps to a fixed grid size (e.g., 64x64 or 128x128)
- Stack maps into a 2-channel tensor (flux + velocity)
- Handle NaN/masked values appropriately
- Track pixel scale through resampling for correct lenstronomy integration

IMPORTANT: Pixel Scale Handling
-------------------------------
MaNGA MAPS have a native pixel scale of 0.5 arcsec/spaxel. When resampling:

1. If preserving field-of-view (e.g., 72x72 -> 64x64):
   - Original: 72 pixels * 0.5"/pix = 36" field
   - After resampling: 64 pixels, same 36" field
   - New pixel scale: 36"/64 = 0.5625"/pix

2. If cropping to a smaller region:
   - Compute the new field of view
   - pixel_scale = field_of_view / n_pixels

3. When passing to lenstronomy:
   - ALWAYS specify the correct pixel_scale in data_kwargs
   - Wrong scale = wrong Einstein radius in pixels!

Expected usage:
    from processing.prep_source_maps import prepare_tensor
    tensor, metadata = prepare_tensor(flux, velocity, grid_size=64)
    
    # metadata includes:
    # - original_shape: (72, 72)
    # - output_shape: (64, 64)
    # - original_pixel_scale: 0.5 arcsec/pixel
    # - output_pixel_scale: 0.5625 arcsec/pixel
    # - field_of_view: 36.0 arcsec
"""

import numpy as np
from scipy import ndimage

# Import project constants
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import MANGA_PIXEL_SCALE, compute_resampled_pixel_scale, get_field_of_view


def prepare_tensor(
    flux_map: np.ndarray,
    velocity_map: np.ndarray,
    grid_size: int = 64,
    normalize: bool = True,
    preserve_fov: bool = True,
) -> tuple:
    """
    Prepare flux and velocity maps as a 2-channel tensor for ML pipeline.
    
    Args:
        flux_map: 2D numpy array of emission-line flux
        velocity_map: 2D numpy array of velocity field
        grid_size: Target output size (square, default 64)
        normalize: Whether to normalize each channel (default True)
        preserve_fov: If True, preserves field-of-view when resampling,
                      adjusting pixel scale accordingly (default True)
    
    Returns:
        tuple: (tensor, metadata)
            - tensor: numpy array of shape (2, grid_size, grid_size)
            - metadata: dict with pixel scale and geometry info
    
    Example:
        >>> tensor, meta = prepare_tensor(flux, velocity, grid_size=64)
        >>> print(meta['output_pixel_scale'])  # For lenstronomy
        0.5625
    """
    original_shape = flux_map.shape
    assert flux_map.shape == velocity_map.shape, "Flux and velocity maps must have same shape"
    
    # Handle NaN/inf values
    flux_clean = np.nan_to_num(flux_map, nan=0.0, posinf=0.0, neginf=0.0)
    velocity_clean = np.nan_to_num(velocity_map, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Resample to target grid size
    if original_shape[0] != grid_size or original_shape[1] != grid_size:
        zoom_factor = grid_size / original_shape[0]
        flux_resampled = ndimage.zoom(flux_clean, zoom_factor, order=1)
        velocity_resampled = ndimage.zoom(velocity_clean, zoom_factor, order=1)
    else:
        flux_resampled = flux_clean
        velocity_resampled = velocity_clean
    
    # Normalize each channel
    if normalize:
        # Flux: normalize to [0, 1]
        flux_min, flux_max = flux_resampled.min(), flux_resampled.max()
        if flux_max > flux_min:
            flux_norm = (flux_resampled - flux_min) / (flux_max - flux_min)
        else:
            flux_norm = np.zeros_like(flux_resampled)
        
        # Velocity: normalize to [-1, 1] (preserving sign)
        vel_absmax = np.abs(velocity_resampled).max()
        if vel_absmax > 0:
            velocity_norm = velocity_resampled / vel_absmax
        else:
            velocity_norm = np.zeros_like(velocity_resampled)
    else:
        flux_norm = flux_resampled
        velocity_norm = velocity_resampled
    
    # Stack into 2-channel tensor
    tensor = np.stack([flux_norm, velocity_norm], axis=0)
    
    # Compute pixel scale metadata
    original_fov = get_field_of_view(original_shape[0], MANGA_PIXEL_SCALE)
    
    if preserve_fov:
        output_pixel_scale = compute_resampled_pixel_scale(
            original_shape[0], grid_size, MANGA_PIXEL_SCALE
        )
    else:
        output_pixel_scale = MANGA_PIXEL_SCALE
    
    metadata = {
        'original_shape': original_shape,
        'output_shape': (grid_size, grid_size),
        'original_pixel_scale': MANGA_PIXEL_SCALE,
        'output_pixel_scale': output_pixel_scale,
        'field_of_view': original_fov,
        'normalized': normalize,
        'preserve_fov': preserve_fov,
    }
    
    return tensor, metadata


# TODO: Add additional functions as needed:
# - prepare_batch(): Process multiple galaxies
# - augment_tensor(): Data augmentation (rotations, flips)
# - create_training_pair(): Generate (unlensed, lensed) pairs

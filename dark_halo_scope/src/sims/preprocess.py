"""
Preprocessing pipeline for raw images to ML-ready format.

This module converts raw DR10 flux images to normalized, 
4-channel tensors suitable for neural network input.
"""

import numpy as np
from typing import Optional, Tuple
import h5py
from pathlib import Path
from tqdm import tqdm


def compute_robust_scale(
    image: np.ndarray,
    low_percentile: float = 5,
    high_percentile: float = 99
) -> Tuple[float, float]:
    """
    Compute robust scaling parameters.
    
    Returns
    -------
    offset : float
        Low percentile value (to subtract)
    scale : float
        Range (high - low percentile)
    """
    valid = image[np.isfinite(image)]
    if len(valid) == 0:
        return 0.0, 1.0
    
    low = np.percentile(valid, low_percentile)
    high = np.percentile(valid, high_percentile)
    
    scale = high - low
    if scale <= 0:
        scale = 1.0
    
    return low, scale


def build_radial_channel(shape: Tuple[int, int]) -> np.ndarray:
    """
    Create normalized radial distance channel.
    
    Parameters
    ----------
    shape : tuple
        (height, width) of the output
    
    Returns
    -------
    np.ndarray
        Radial distance from center, normalized to [0, 1]
    """
    h, w = shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    r_max = np.sqrt(center_x**2 + center_y**2)
    
    return (r / r_max).astype(np.float32)


def preprocess_single_image(
    image_raw: np.ndarray,
    sky_estimate: Optional[np.ndarray] = None,
    clip_range: Tuple[float, float] = (-3.0, 10.0)
) -> np.ndarray:
    """
    Preprocess a single 3-channel raw image to 4-channel ML input.
    
    Steps:
    1. Sky subtraction (if provided)
    2. Per-channel robust scaling
    3. Clipping
    4. Add radial channel
    
    Parameters
    ----------
    image_raw : np.ndarray
        Shape (3, H, W) raw flux in g, r, z
    sky_estimate : np.ndarray, optional
        Shape (3,) sky values per band
    clip_range : tuple
        (min, max) values after scaling
    
    Returns
    -------
    np.ndarray
        Shape (4, H, W) preprocessed image with [g, r, z, R] channels
    """
    n_bands, h, w = image_raw.shape
    
    # Sky subtraction
    if sky_estimate is not None:
        image = image_raw - sky_estimate[:, None, None]
    else:
        # Simple sky estimate: median of border pixels
        image = image_raw.copy()
        for i in range(n_bands):
            border = np.concatenate([
                image[i, 0, :],    # Top row
                image[i, -1, :],   # Bottom row
                image[i, :, 0],    # Left column
                image[i, :, -1]    # Right column
            ])
            sky = np.nanmedian(border)
            image[i] -= sky
    
    # Per-channel robust scaling
    scaled = np.zeros_like(image)
    for i in range(n_bands):
        offset, scale = compute_robust_scale(image[i])
        scaled[i] = (image[i] - offset) / scale
    
    # Clipping
    scaled = np.clip(scaled, clip_range[0], clip_range[1])
    
    # Add radial channel
    R = build_radial_channel((h, w))
    
    # Stack to 4 channels
    output = np.zeros((4, h, w), dtype=np.float32)
    output[:3] = scaled
    output[3] = R
    
    return output


def preprocess_raw_to_ml(
    images_raw: np.ndarray,
    sky_estimates: Optional[np.ndarray] = None,
    clip_range: Tuple[float, float] = (-3.0, 10.0)
) -> np.ndarray:
    """
    Batch preprocess raw images to ML-ready format.
    
    Parameters
    ----------
    images_raw : np.ndarray
        Shape (N, 3, H, W) raw flux images
    sky_estimates : np.ndarray, optional
        Shape (N, 3) per-image sky values
    clip_range : tuple
        Clipping range after scaling
    
    Returns
    -------
    np.ndarray
        Shape (N, 4, H, W) preprocessed images
    """
    n_images = images_raw.shape[0]
    _, h, w = images_raw.shape[1:]
    
    output = np.zeros((n_images, 4, h, w), dtype=np.float32)
    
    for i in range(n_images):
        sky = sky_estimates[i] if sky_estimates is not None else None
        output[i] = preprocess_single_image(images_raw[i], sky, clip_range)
    
    return output


def preprocess_hdf5(
    input_path: str,
    output_path: str,
    clip_range: Tuple[float, float] = (-3.0, 10.0),
    chunk_size: int = 100
) -> None:
    """
    Preprocess an entire HDF5 file of raw images.
    
    Creates a new HDF5 with preprocessed 4-channel images.
    
    Parameters
    ----------
    input_path : str
        Path to raw HDF5 (with /images_raw dataset)
    output_path : str
        Path for output preprocessed HDF5
    clip_range : tuple
        Clipping range
    chunk_size : int
        Batch size for processing
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(input_path, 'r') as f_in:
        n_images = f_in['images_raw'].shape[0]
        _, n_bands, h, w = f_in['images_raw'].shape
        
        with h5py.File(output_path, 'w') as f_out:
            # Create output dataset (4 channels: g, r, z, R)
            images_out = f_out.create_dataset(
                'images_preproc',
                shape=(n_images, 4, h, w),
                dtype=np.float32,
                chunks=(1, 4, h, w),
                compression='gzip'
            )
            
            # Copy other datasets
            for key in f_in.keys():
                if key != 'images_raw':
                    f_in.copy(key, f_out)
            
            # Copy attributes
            for key, val in f_in.attrs.items():
                f_out.attrs[key] = val
            f_out.attrs['preprocessing'] = f'scaled_clipped_{clip_range}'
            
            # Process in chunks
            for start in tqdm(range(0, n_images, chunk_size), desc="Preprocessing"):
                end = min(start + chunk_size, n_images)
                raw_batch = f_in['images_raw'][start:end]
                preproc_batch = preprocess_raw_to_ml(raw_batch, clip_range=clip_range)
                images_out[start:end] = preproc_batch
    
    print(f"Preprocessed {n_images} images → {output_path}")


def normalize_for_display(
    image: np.ndarray,
    percentile_low: float = 1,
    percentile_high: float = 99
) -> np.ndarray:
    """
    Normalize image for display (RGB visualization).
    
    Parameters
    ----------
    image : np.ndarray
        Shape (3, H, W) or (H, W, 3)
    percentile_low, percentile_high : float
        Stretch percentiles
    
    Returns
    -------
    np.ndarray
        Normalized to [0, 1] for display
    """
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    output = np.zeros_like(image)
    for i in range(3):
        channel = image[:, :, i]
        vmin = np.nanpercentile(channel, percentile_low)
        vmax = np.nanpercentile(channel, percentile_high)
        if vmax > vmin:
            output[:, :, i] = np.clip((channel - vmin) / (vmax - vmin), 0, 1)
    
    return output


"""
Build HDF5 background dataset from DR10 cutouts.

This module consolidates all cutouts into a single HDF5 file
with proper structure for the simulation pipeline.
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from .download_cutouts import load_cutout


def build_background_hdf5(
    parent_sample: pd.DataFrame,
    cutout_dir: str = "data/raw/cutouts/",
    output_path: str = "data/processed/dr10_backgrounds_raw.h5",
    image_size: int = 128
) -> None:
    """
    Build HDF5 with raw backgrounds + metadata.
    
    HDF5 structure:
    - /images_raw: (N, 3, 128, 128) float32, raw flux units
    - /psf_fwhm: (N, 3) float32, FWHM per band (g, r, z)
    - /meta: (N, K) float32, catalog metadata
    - /ids: (N,) int64, object identifiers
    - /ra: (N,) float64
    - /dec: (N,) float64
    
    Parameters
    ----------
    parent_sample : pd.DataFrame
        Parent sample with cutout_path and cutout_status columns
    cutout_dir : str
        Directory containing cutout .npz files
    output_path : str
        Output HDF5 path
    image_size : int
        Expected cutout size (for validation)
    """
    # Filter to successful downloads
    valid_sample = parent_sample[parent_sample['cutout_status'].isin(['success', 'cached'])]
    n_objects = len(valid_sample)
    
    if n_objects == 0:
        raise ValueError("No valid cutouts found!")
    
    print(f"Building HDF5 from {n_objects} cutouts...")
    
    # Prepare output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Metadata columns to store
    meta_cols = ['mag_r', 'mag_g', 'mag_z', 'g_minus_r', 'r_minus_z', 'shape_r']
    n_meta = len(meta_cols)
    
    with h5py.File(output_path, 'w') as f:
        # Create datasets
        images = f.create_dataset(
            'images_raw',
            shape=(n_objects, 3, image_size, image_size),
            dtype=np.float32,
            chunks=(1, 3, image_size, image_size),
            compression='gzip',
            compression_opts=4
        )
        
        psf_fwhm = f.create_dataset(
            'psf_fwhm',
            shape=(n_objects, 3),
            dtype=np.float32
        )
        
        meta = f.create_dataset(
            'meta',
            shape=(n_objects, n_meta),
            dtype=np.float32
        )
        
        ids = f.create_dataset(
            'ids',
            shape=(n_objects,),
            dtype=np.int64
        )
        
        ra = f.create_dataset('ra', shape=(n_objects,), dtype=np.float64)
        dec = f.create_dataset('dec', shape=(n_objects,), dtype=np.float64)
        
        # Store metadata column names
        f.attrs['meta_columns'] = meta_cols
        f.attrs['bands'] = ['g', 'r', 'z']
        f.attrs['pixel_scale'] = 0.262
        f.attrs['image_size'] = image_size
        
        # Load and store each cutout
        for i, (idx, row) in enumerate(tqdm(valid_sample.iterrows(), 
                                             total=n_objects,
                                             desc="Building HDF5")):
            cutout = load_cutout(row['cutout_path'])
            
            if cutout is None:
                # Fill with NaN if load failed
                images[i] = np.nan
                psf_fwhm[i] = np.nan
                meta[i] = np.nan
                ids[i] = idx
                ra[i] = row['ra']
                dec[i] = row['dec']
                continue
            
            # Store image data (g, r, z order)
            images[i, 0] = cutout['g']
            images[i, 1] = cutout['r']
            images[i, 2] = cutout['z']
            
            # Store PSF FWHM
            psf_fwhm[i, 0] = row.get('psfsize_g', 1.3)
            psf_fwhm[i, 1] = row.get('psfsize_r', 1.2)
            psf_fwhm[i, 2] = row.get('psfsize_z', 1.1)
            
            # Store metadata
            for j, col in enumerate(meta_cols):
                meta[i, j] = row.get(col, np.nan)
            
            # Store identifiers
            ids[i] = idx
            ra[i] = row['ra']
            dec[i] = row['dec']
    
    print(f"Created: {output_path}")
    print(f"  - images_raw: ({n_objects}, 3, {image_size}, {image_size})")
    print(f"  - psf_fwhm: ({n_objects}, 3)")
    print(f"  - meta: ({n_objects}, {n_meta})")


def load_background_sample(
    h5_path: str,
    indices: Optional[np.ndarray] = None,
    n_random: Optional[int] = None
) -> dict:
    """
    Load a subset of backgrounds from HDF5.
    
    Parameters
    ----------
    h5_path : str
        Path to HDF5 file
    indices : np.ndarray, optional
        Specific indices to load
    n_random : int, optional
        Number of random samples to load
    
    Returns
    -------
    dict
        Dictionary with images, psf_fwhm, meta, ids
    """
    with h5py.File(h5_path, 'r') as f:
        n_total = f['images_raw'].shape[0]
        
        if indices is None and n_random is not None:
            indices = np.random.choice(n_total, size=min(n_random, n_total), replace=False)
        elif indices is None:
            indices = np.arange(n_total)
        
        return {
            'images_raw': f['images_raw'][indices],
            'psf_fwhm': f['psf_fwhm'][indices],
            'meta': f['meta'][indices],
            'ids': f['ids'][indices],
            'ra': f['ra'][indices],
            'dec': f['dec'][indices],
            'meta_columns': list(f.attrs['meta_columns']),
            'pixel_scale': f.attrs['pixel_scale']
        }


def get_hdf5_info(h5_path: str) -> dict:
    """
    Get summary information about the HDF5 file.
    
    Returns
    -------
    dict
        Summary statistics
    """
    with h5py.File(h5_path, 'r') as f:
        return {
            'n_objects': f['images_raw'].shape[0],
            'image_shape': f['images_raw'].shape[1:],
            'pixel_scale': f.attrs['pixel_scale'],
            'bands': list(f.attrs['bands']),
            'meta_columns': list(f.attrs['meta_columns']),
            'file_size_mb': Path(h5_path).stat().st_size / 1e6
        }


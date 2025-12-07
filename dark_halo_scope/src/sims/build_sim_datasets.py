"""
Build simulation datasets for lens detection training.

This module generates HDF5 datasets of simulated lensed galaxies
by injecting COSMOS sources into DR10 LRG backgrounds.
"""

import numpy as np
import h5py
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict
from tqdm import tqdm

from .cosmos_loader import COSMOSLoader
from .lens_injector import LensInjector, InjectionParams


def build_simulation_grid(
    theta_E_range: Tuple[float, float] = (0.8, 2.5),
    z_l_range: Tuple[float, float] = (0.2, 0.6),
    z_s_range: Tuple[float, float] = (1.0, 2.5),
    m_source_range: Tuple[float, float] = (22.0, 25.0),
    n_theta_E: int = 5,
    n_z_l: int = 4,
    n_z_s: int = 3,
    n_m_source: int = 4,
    n_sources_per_point: int = 3
) -> List[Dict]:
    """
    Generate parameter grid for simulations.
    
    Parameters
    ----------
    theta_E_range, z_l_range, z_s_range, m_source_range : tuple
        Parameter ranges
    n_theta_E, n_z_l, n_z_s, n_m_source : int
        Number of grid points per parameter
    n_sources_per_point : int
        Number of different COSMOS sources per grid point
    
    Returns
    -------
    list of dict
        List of parameter dictionaries for injection
    """
    theta_E_vals = np.linspace(*theta_E_range, n_theta_E)
    z_l_vals = np.linspace(*z_l_range, n_z_l)
    z_s_vals = np.linspace(*z_s_range, n_z_s)
    m_source_vals = np.linspace(*m_source_range, n_m_source)
    
    grid = []
    
    for theta_E in theta_E_vals:
        for z_l in z_l_vals:
            for z_s in z_s_vals:
                # Skip if source behind lens condition violated
                if z_s <= z_l + 0.2:
                    continue
                
                for m_source in m_source_vals:
                    for source_idx in range(n_sources_per_point):
                        grid.append({
                            'theta_E': float(theta_E),
                            'z_l': float(z_l),
                            'z_s': float(z_s),
                            'm_source_r': float(m_source),
                            'source_idx': source_idx
                        })
    
    return grid


def build_sim_hdf5(
    backgrounds_h5: str,
    cosmos_loader: COSMOSLoader,
    injector: LensInjector,
    output_path: str,
    n_lenses: int = 5000,
    n_negatives: int = 5000,
    theta_E_range: Tuple[float, float] = (0.8, 2.5),
    z_l_range: Tuple[float, float] = (0.2, 0.6),
    z_s_range: Tuple[float, float] = (1.0, 2.5),
    m_source_range: Tuple[float, float] = (22.0, 25.0),
    seed: int = 42
) -> None:
    """
    Build simulation HDF5 with injected lenses and negatives.
    
    HDF5 structure:
    - /images_raw: (N, 3, H, W) raw flux
    - /labels: (N,) 0=non-lens, 1=lens
    - /theta_E: (N,) Einstein radii (0 for negatives)
    - /z_l: (N,) lens redshifts
    - /z_s: (N,) source redshifts (0 for negatives)
    - /m_source_r: (N,) source magnitudes (0 for negatives)
    - /clumpiness: (N,) source clumpiness (0 for negatives)
    - /background_idx: (N,) index into backgrounds HDF5
    
    Parameters
    ----------
    backgrounds_h5 : str
        Path to DR10 backgrounds HDF5
    cosmos_loader : COSMOSLoader
        Loaded COSMOS source catalog
    injector : LensInjector
        Configured lens injector
    output_path : str
        Output HDF5 path
    n_lenses, n_negatives : int
        Number of each class to generate
    """
    np.random.seed(seed)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Load background info
    with h5py.File(backgrounds_h5, 'r') as f_bg:
        n_backgrounds = f_bg['images_raw'].shape[0]
        _, n_bands, h, w = f_bg['images_raw'].shape
        psf_fwhm_all = f_bg['psf_fwhm'][:]
    
    n_total = n_lenses + n_negatives
    
    with h5py.File(output_path, 'w') as f_out:
        # Create datasets
        images = f_out.create_dataset(
            'images_raw',
            shape=(n_total, n_bands, h, w),
            dtype=np.float32,
            chunks=(1, n_bands, h, w),
            compression='gzip'
        )
        labels = f_out.create_dataset('labels', shape=(n_total,), dtype=np.int32)
        theta_E = f_out.create_dataset('theta_E', shape=(n_total,), dtype=np.float32)
        z_l = f_out.create_dataset('z_l', shape=(n_total,), dtype=np.float32)
        z_s = f_out.create_dataset('z_s', shape=(n_total,), dtype=np.float32)
        m_source = f_out.create_dataset('m_source_r', shape=(n_total,), dtype=np.float32)
        clumpiness = f_out.create_dataset('clumpiness', shape=(n_total,), dtype=np.float32)
        bg_idx = f_out.create_dataset('background_idx', shape=(n_total,), dtype=np.int64)
        
        # Attributes
        f_out.attrs['n_lenses'] = n_lenses
        f_out.attrs['n_negatives'] = n_negatives
        f_out.attrs['seed'] = seed
        
        sample_idx = 0
        
        # Generate lenses
        print(f"Generating {n_lenses} lens samples...")
        with h5py.File(backgrounds_h5, 'r') as f_bg:
            for i in tqdm(range(n_lenses), desc="Lenses"):
                # Random background
                idx_bg = np.random.randint(0, n_backgrounds)
                background = {
                    'g': f_bg['images_raw'][idx_bg, 0],
                    'r': f_bg['images_raw'][idx_bg, 1],
                    'z': f_bg['images_raw'][idx_bg, 2]
                }
                psf = {
                    'g': float(psf_fwhm_all[idx_bg, 0]),
                    'r': float(psf_fwhm_all[idx_bg, 1]),
                    'z': float(psf_fwhm_all[idx_bg, 2])
                }
                
                # Random source
                source_img, source_clump = cosmos_loader.get_random_source()
                
                # Random parameters
                params = injector.sample_injection_params(
                    theta_E_range=theta_E_range,
                    z_l_range=z_l_range,
                    z_s_range=z_s_range,
                    m_source_range=m_source_range
                )
                params.clumpiness = source_clump
                
                # Inject
                try:
                    injected, params = injector.inject_arc(
                        background, psf, source_img, params,
                        source_pixel_scale=cosmos_loader.pixel_scale
                    )
                    
                    # Store
                    images[sample_idx, 0] = injected['g']
                    images[sample_idx, 1] = injected['r']
                    images[sample_idx, 2] = injected['z']
                    labels[sample_idx] = 1
                    theta_E[sample_idx] = params.theta_E
                    z_l[sample_idx] = params.z_l
                    z_s[sample_idx] = params.z_s
                    m_source[sample_idx] = params.m_source_r
                    clumpiness[sample_idx] = params.clumpiness
                    bg_idx[sample_idx] = idx_bg
                    
                except Exception as e:
                    # On failure, use background as-is with label=0
                    print(f"Injection failed: {e}")
                    images[sample_idx, 0] = background['g']
                    images[sample_idx, 1] = background['r']
                    images[sample_idx, 2] = background['z']
                    labels[sample_idx] = 0
                    theta_E[sample_idx] = 0
                    z_l[sample_idx] = 0
                    z_s[sample_idx] = 0
                    m_source[sample_idx] = 0
                    clumpiness[sample_idx] = 0
                    bg_idx[sample_idx] = idx_bg
                
                sample_idx += 1
        
        # Generate negatives (just backgrounds, no injection)
        print(f"Generating {n_negatives} negative samples...")
        with h5py.File(backgrounds_h5, 'r') as f_bg:
            for i in tqdm(range(n_negatives), desc="Negatives"):
                idx_bg = np.random.randint(0, n_backgrounds)
                
                images[sample_idx] = f_bg['images_raw'][idx_bg]
                labels[sample_idx] = 0
                theta_E[sample_idx] = 0
                z_l[sample_idx] = 0
                z_s[sample_idx] = 0
                m_source[sample_idx] = 0
                clumpiness[sample_idx] = 0
                bg_idx[sample_idx] = idx_bg
                
                sample_idx += 1
    
    print(f"Created: {output_path}")
    print(f"  - {n_lenses} lenses + {n_negatives} negatives = {n_total} total")


def split_sim_dataset(
    input_h5: str,
    train_path: str,
    test_path: str,
    test_fraction: float = 0.2,
    seed: int = 42
) -> None:
    """
    Split simulation HDF5 into train and test sets.
    
    Parameters
    ----------
    input_h5 : str
        Path to full simulation HDF5
    train_path, test_path : str
        Output paths for train and test HDF5
    test_fraction : float
        Fraction of data for test set
    seed : int
        Random seed for reproducibility
    """
    np.random.seed(seed)
    
    with h5py.File(input_h5, 'r') as f:
        n_total = f['labels'].shape[0]
        indices = np.random.permutation(n_total)
        
        n_test = int(n_total * test_fraction)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        
        # Create train file
        with h5py.File(train_path, 'w') as f_train:
            for key in f.keys():
                f_train.create_dataset(key, data=f[key][train_idx])
            for key, val in f.attrs.items():
                f_train.attrs[key] = val
            f_train.attrs['split'] = 'train'
        
        # Create test file
        with h5py.File(test_path, 'w') as f_test:
            for key in f.keys():
                f_test.create_dataset(key, data=f[key][test_idx])
            for key, val in f.attrs.items():
                f_test.attrs[key] = val
            f_test.attrs['split'] = 'test'
    
    print(f"Split {n_total} samples → {len(train_idx)} train + {len(test_idx)} test")


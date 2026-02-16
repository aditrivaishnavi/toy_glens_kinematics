"""
COSMOS/HST galaxy image loader for lensing sources.

This module provides functionality to load and serve COSMOS galaxy
images as source templates for gravitational lensing simulations.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import h5py


class COSMOSLoader:
    """
    Load and serve COSMOS/HST galaxy images as lensing sources.
    
    COSMOS provides high-resolution galaxy images that can be used
    as realistic source templates for lensing simulations.
    
    Attributes
    ----------
    pixel_scale : float
        Native COSMOS pixel scale (~0.05 arcsec for HST/ACS)
    images : np.ndarray
        Array of source images (N, H, W)
    clumpiness : np.ndarray
        Precomputed clumpiness metric for each source
    n_sources : int
        Number of available source images
    """
    
    # Default COSMOS parameters
    DEFAULT_PIXEL_SCALE = 0.05  # arcsec/pixel for HST/ACS
    
    def __init__(
        self,
        cosmos_path: Optional[str] = None,
        pixel_scale: float = DEFAULT_PIXEL_SCALE
    ):
        """
        Initialize COSMOS loader.
        
        Parameters
        ----------
        cosmos_path : str, optional
            Path to COSMOS HDF5 or directory of images.
            If None, uses placeholder synthetic sources.
        pixel_scale : float
            Native pixel scale in arcsec/pixel
        """
        self.pixel_scale = pixel_scale
        self.cosmos_path = cosmos_path
        
        if cosmos_path is not None and Path(cosmos_path).exists():
            self._load_cosmos(cosmos_path)
        else:
            # Create placeholder synthetic sources for development
            self._create_placeholder_sources()
    
    def _load_cosmos(self, path: str) -> None:
        """Load COSMOS images from file."""
        if path.endswith('.h5') or path.endswith('.hdf5'):
            with h5py.File(path, 'r') as f:
                self.images = f['images'][:]
                if 'clumpiness' in f:
                    self.clumpiness = f['clumpiness'][:]
                else:
                    self.clumpiness = self._compute_clumpiness(self.images)
        else:
            # Load from directory of numpy files
            path = Path(path)
            image_files = sorted(path.glob('*.npy'))
            if len(image_files) == 0:
                self._create_placeholder_sources()
                return
            
            self.images = np.stack([np.load(f) for f in image_files])
            self.clumpiness = self._compute_clumpiness(self.images)
        
        self.n_sources = len(self.images)
    
    def _create_placeholder_sources(self, n_sources: int = 50) -> None:
        """Create synthetic placeholder sources for development."""
        print("Warning: Using synthetic placeholder sources. Load real COSMOS data for production.")
        
        size = 64  # pixels at 0.05"/pix = 3.2"
        self.images = np.zeros((n_sources, size, size), dtype=np.float32)
        self.clumpiness = np.zeros(n_sources, dtype=np.float32)
        
        for i in range(n_sources):
            # Create a random clumpy galaxy profile
            self.images[i] = self._generate_clumpy_source(size)
            self.clumpiness[i] = self._compute_single_clumpiness(self.images[i])
        
        self.n_sources = n_sources
    
    def _generate_clumpy_source(self, size: int) -> np.ndarray:
        """Generate a synthetic clumpy galaxy image."""
        y, x = np.ogrid[:size, :size]
        center = size // 2
        
        # Base exponential disk
        r = np.sqrt((x - center)**2 + (y - center)**2)
        disk = np.exp(-r / (size * 0.15))
        
        # Add random clumps
        n_clumps = np.random.randint(2, 6)
        for _ in range(n_clumps):
            cx = np.random.randint(size // 4, 3 * size // 4)
            cy = np.random.randint(size // 4, 3 * size // 4)
            r_clump = np.sqrt((x - cx)**2 + (y - cy)**2)
            clump_size = np.random.uniform(2, 5)
            clump_amp = np.random.uniform(0.3, 1.0)
            disk += clump_amp * np.exp(-r_clump**2 / (2 * clump_size**2))
        
        # Normalize
        disk = disk / disk.max()
        
        return disk.astype(np.float32)
    
    def _compute_clumpiness(self, images: np.ndarray) -> np.ndarray:
        """Compute clumpiness metric for a batch of images."""
        return np.array([self._compute_single_clumpiness(img) for img in images])
    
    def _compute_single_clumpiness(self, image: np.ndarray) -> float:
        """
        Compute clumpiness metric for a single image.
        
        Higher values indicate more clumpy (non-smooth) structure.
        """
        from scipy.ndimage import gaussian_filter
        
        # Smooth version
        smoothed = gaussian_filter(image, sigma=3)
        
        # Residual (high-frequency content)
        residual = image - smoothed
        
        # Clumpiness = variance of residual normalized by total flux
        total_flux = image.sum()
        if total_flux == 0:
            return 0.0
        
        clumpiness = np.var(residual) / (total_flux / image.size)
        return float(clumpiness)
    
    def get_random_source(self, seed: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Get a random source image.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        image : np.ndarray
            Source image (H, W)
        clumpiness : float
            Clumpiness score
        """
        if seed is not None:
            np.random.seed(seed)
        
        idx = np.random.randint(0, self.n_sources)
        return self.images[idx].copy(), self.clumpiness[idx]
    
    def get_source_by_clumpiness(
        self,
        min_clumpiness: float = 0.0,
        max_clumpiness: float = np.inf
    ) -> Tuple[np.ndarray, float]:
        """
        Get a source within a clumpiness range.
        
        Parameters
        ----------
        min_clumpiness, max_clumpiness : float
            Range of acceptable clumpiness values
        
        Returns
        -------
        image : np.ndarray
            Source image
        clumpiness : float
            Clumpiness score
        """
        mask = (self.clumpiness >= min_clumpiness) & (self.clumpiness <= max_clumpiness)
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) == 0:
            # Fall back to random
            return self.get_random_source()
        
        idx = np.random.choice(valid_indices)
        return self.images[idx].copy(), self.clumpiness[idx]
    
    def get_source_by_index(self, idx: int) -> Tuple[np.ndarray, float]:
        """Get a specific source by index."""
        return self.images[idx].copy(), self.clumpiness[idx]
    
    def __len__(self) -> int:
        return self.n_sources


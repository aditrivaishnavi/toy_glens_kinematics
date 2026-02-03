"""
COSMOS/HST galaxy image loader for lensing sources - Enhanced Version.

This module provides functionality to load real COSMOS galaxy images
from GalSim or custom HDF5 files for use as source templates in 
gravitational lensing simulations.

Enhancements over v1:
- GalSim COSMOS catalog integration
- Source metadata (size, magnitude, redshift)
- Deterministic source selection with seed
- Resampling to different pixel scales
- Color information support
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any, Union
from dataclasses import dataclass
import h5py
import logging

logger = logging.getLogger(__name__)


@dataclass
class COSMOSSource:
    """
    A COSMOS source with image and metadata.
    
    Attributes
    ----------
    image : np.ndarray
        Source image at native resolution (H, W)
    pixel_scale : float
        Pixel scale in arcsec/pixel
    clumpiness : float
        Clumpiness metric (higher = more irregular)
    half_light_radius : float
        Half-light radius in arcsec
    magnitude : float
        Apparent magnitude (typically F814W)
    redshift : float
        Photometric redshift (if available)
    index : int
        Index in catalog
    """
    image: np.ndarray
    pixel_scale: float
    clumpiness: float
    half_light_radius: float = 0.3  # arcsec
    magnitude: float = 24.0
    redshift: float = 1.5
    index: int = 0


class COSMOSLoaderV2:
    """
    Enhanced COSMOS galaxy image loader for lensing simulations.
    
    Supports:
    - Real COSMOS data from GalSim catalog
    - Custom HDF5 source libraries
    - Fallback synthetic sources for testing
    - Deterministic source selection
    
    Parameters
    ----------
    cosmos_path : str, optional
        Path to COSMOS HDF5 file or GalSim data directory
    mode : str
        Loading mode: 'galsim', 'hdf5', or 'synthetic'
    pixel_scale : float
        Native pixel scale (default: 0.03" for HST/ACS)
    preload : bool
        Whether to preload all images into memory
    seed : int
        Random seed for reproducibility
    """
    
    # Default COSMOS parameters
    DEFAULT_PIXEL_SCALE = 0.03  # arcsec/pixel for HST/ACS F814W
    DEFAULT_STAMP_SIZE = 128  # pixels
    
    def __init__(
        self,
        cosmos_path: Optional[str] = None,
        mode: str = 'auto',
        pixel_scale: float = DEFAULT_PIXEL_SCALE,
        preload: bool = True,
        seed: int = 42
    ):
        self.pixel_scale = pixel_scale
        self.cosmos_path = cosmos_path
        self.preload = preload
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        
        # Determine mode
        if mode == 'auto':
            mode = self._detect_mode(cosmos_path)
        
        self.mode = mode
        
        # Initialize based on mode
        if mode == 'galsim':
            self._init_galsim(cosmos_path)
        elif mode == 'hdf5':
            self._init_hdf5(cosmos_path)
        else:
            self._init_synthetic()
        
        logger.info(f"COSMOSLoaderV2 initialized: mode={mode}, n_sources={self.n_sources}")
    
    def _detect_mode(self, path: Optional[str]) -> str:
        """Detect loading mode from path."""
        if path is None:
            return 'synthetic'
        
        path = Path(path)
        if not path.exists():
            logger.warning(f"Path {path} does not exist, using synthetic mode")
            return 'synthetic'
        
        if path.suffix in ['.h5', '.hdf5']:
            return 'hdf5'
        
        if path.is_dir():
            # Check for GalSim files
            if (path / 'real_galaxy_catalog_25.2.fits').exists():
                return 'galsim'
        
        return 'synthetic'
    
    def _init_galsim(self, path: str) -> None:
        """Initialize from GalSim COSMOS catalog."""
        try:
            import galsim
        except ImportError:
            logger.warning("GalSim not available, falling back to synthetic")
            self._init_synthetic()
            return
        
        try:
            self._catalog = galsim.COSMOSCatalog(
                dir=path,
                exclusion_level='marginal'  # Exclude problematic galaxies
            )
            self.n_sources = self._catalog.getNObjects()
            
            # Extract metadata if preloading
            if self.preload:
                self._preload_galsim()
            else:
                self.images = None
                self.metadata = None
                
        except Exception as e:
            logger.error(f"Failed to load GalSim catalog: {e}")
            self._init_synthetic()
    
    def _preload_galsim(self) -> None:
        """Preload images and metadata from GalSim catalog."""
        import galsim
        
        logger.info(f"Preloading {self.n_sources} sources from GalSim catalog...")
        
        # Limit to reasonable number for memory
        max_sources = min(self.n_sources, 10000)
        
        images = []
        metadata = []
        
        for i in range(max_sources):
            try:
                # Get galaxy object
                gal = self._catalog.makeGalaxy(
                    index=i,
                    gal_type='real',
                    noise_pad_size=0
                )
                
                # Draw to image
                stamp_size = self.DEFAULT_STAMP_SIZE
                image = galsim.Image(stamp_size, stamp_size, scale=self.pixel_scale)
                gal.drawImage(image, scale=self.pixel_scale)
                
                images.append(image.array.astype(np.float32))
                
                # Get metadata
                rec = self._catalog.getParametricRecord(i)
                metadata.append({
                    'index': i,
                    'half_light_radius': float(rec.get('hlr', 0.3)),
                    'magnitude': float(rec.get('mag_auto', 24.0)),
                    'redshift': float(rec.get('zphot', 1.5)),
                })
                
            except Exception as e:
                logger.debug(f"Failed to load galaxy {i}: {e}")
                continue
        
        self.images = np.stack(images)
        self.metadata = metadata
        self.n_sources = len(images)
        self.clumpiness = self._compute_clumpiness_batch(self.images)
        
        logger.info(f"Preloaded {self.n_sources} sources")
    
    def _init_hdf5(self, path: str) -> None:
        """Initialize from HDF5 file."""
        with h5py.File(path, 'r') as f:
            if self.preload:
                self.images = f['images'][:].astype(np.float32)
            else:
                self.images = None
                self._h5_path = path
            
            self.n_sources = f['images'].shape[0]
            
            # Load metadata if available
            if 'clumpiness' in f:
                self.clumpiness = f['clumpiness'][:]
            else:
                self.clumpiness = self._compute_clumpiness_batch(self.images) if self.images is not None else None
            
            # Load other metadata
            self.metadata = []
            for i in range(self.n_sources):
                meta = {'index': i}
                if 'half_light_radius' in f:
                    meta['half_light_radius'] = float(f['half_light_radius'][i])
                if 'magnitude' in f:
                    meta['magnitude'] = float(f['magnitude'][i])
                if 'redshift' in f:
                    meta['redshift'] = float(f['redshift'][i])
                self.metadata.append(meta)
        
        logger.info(f"Loaded {self.n_sources} sources from HDF5")
    
    def _init_synthetic(self, n_sources: int = 500) -> None:
        """Initialize with synthetic placeholder sources."""
        logger.warning("Using synthetic placeholder sources. Load real COSMOS data for production.")
        
        size = self.DEFAULT_STAMP_SIZE
        self.images = np.zeros((n_sources, size, size), dtype=np.float32)
        self.metadata = []
        
        for i in range(n_sources):
            self.images[i] = self._generate_clumpy_source(size, seed=self.seed + i)
            self.metadata.append({
                'index': i,
                'half_light_radius': self._rng.uniform(0.1, 0.5),
                'magnitude': self._rng.uniform(22, 26),
                'redshift': self._rng.uniform(1.0, 3.0)
            })
        
        self.n_sources = n_sources
        self.clumpiness = self._compute_clumpiness_batch(self.images)
    
    def _generate_clumpy_source(self, size: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate a synthetic clumpy galaxy image."""
        rng = np.random.RandomState(seed)
        
        y, x = np.ogrid[:size, :size]
        center = size // 2
        
        # Base exponential disk with random ellipticity
        e = rng.uniform(0.0, 0.5)
        theta = rng.uniform(0, np.pi)
        x_rot = (x - center) * np.cos(theta) + (y - center) * np.sin(theta)
        y_rot = -(x - center) * np.sin(theta) + (y - center) * np.cos(theta)
        r = np.sqrt(x_rot**2 + y_rot**2 / (1 - e + 0.01)**2)
        
        scale = size * rng.uniform(0.1, 0.2)
        disk = np.exp(-r / scale)
        
        # Add random clumps (star-forming regions)
        n_clumps = rng.randint(3, 8)
        for _ in range(n_clumps):
            cx = rng.randint(size // 4, 3 * size // 4)
            cy = rng.randint(size // 4, 3 * size // 4)
            r_clump = np.sqrt((x - cx)**2 + (y - cy)**2)
            clump_size = rng.uniform(1.5, 4)
            clump_amp = rng.uniform(0.2, 0.8)
            disk += clump_amp * np.exp(-r_clump**2 / (2 * clump_size**2))
        
        # Normalize to unit total flux
        disk = disk / (disk.sum() + 1e-10)
        
        return disk.astype(np.float32)
    
    def _compute_clumpiness_batch(self, images: np.ndarray) -> np.ndarray:
        """Compute clumpiness metric for a batch of images."""
        from scipy.ndimage import gaussian_filter
        
        clumpiness = np.zeros(len(images), dtype=np.float32)
        
        for i, img in enumerate(images):
            smoothed = gaussian_filter(img, sigma=3)
            residual = img - smoothed
            total_flux = img.sum()
            
            if total_flux > 0:
                clumpiness[i] = np.var(residual) / (total_flux / img.size)
        
        return clumpiness
    
    def get_source(
        self,
        index: Optional[int] = None,
        seed: Optional[int] = None
    ) -> COSMOSSource:
        """
        Get a source by index or randomly.
        
        Parameters
        ----------
        index : int, optional
            Specific source index. If None, randomly selected.
        seed : int, optional
            Random seed for reproducible random selection.
        
        Returns
        -------
        COSMOSSource
            Source with image and metadata
        """
        if index is None:
            rng = np.random.RandomState(seed) if seed is not None else self._rng
            index = rng.randint(0, self.n_sources)
        
        if self.images is not None:
            image = self.images[index].copy()
        else:
            # Lazy loading from HDF5
            with h5py.File(self._h5_path, 'r') as f:
                image = f['images'][index].astype(np.float32)
        
        meta = self.metadata[index] if self.metadata else {}
        clump = float(self.clumpiness[index]) if self.clumpiness is not None else 0.0
        
        return COSMOSSource(
            image=image,
            pixel_scale=self.pixel_scale,
            clumpiness=clump,
            half_light_radius=meta.get('half_light_radius', 0.3),
            magnitude=meta.get('magnitude', 24.0),
            redshift=meta.get('redshift', 1.5),
            index=index
        )
    
    def get_random_source(self, seed: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Get a random source (backward compatible with v1 API).
        
        Returns
        -------
        image : np.ndarray
            Source image (H, W)
        clumpiness : float
            Clumpiness score
        """
        source = self.get_source(seed=seed)
        return source.image, source.clumpiness
    
    def get_source_by_clumpiness(
        self,
        min_clumpiness: float = 0.0,
        max_clumpiness: float = np.inf,
        seed: Optional[int] = None
    ) -> COSMOSSource:
        """
        Get a source within a clumpiness range.
        """
        if self.clumpiness is None:
            return self.get_source(seed=seed)
        
        mask = (self.clumpiness >= min_clumpiness) & (self.clumpiness <= max_clumpiness)
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) == 0:
            return self.get_source(seed=seed)
        
        rng = np.random.RandomState(seed) if seed is not None else self._rng
        index = rng.choice(valid_indices)
        return self.get_source(index=index)
    
    def resample_to_pixscale(
        self,
        image: np.ndarray,
        target_pixscale: float,
        source_pixscale: Optional[float] = None
    ) -> np.ndarray:
        """
        Resample image to a different pixel scale.
        
        Parameters
        ----------
        image : np.ndarray
            Source image at native resolution
        target_pixscale : float
            Target pixel scale in arcsec/pixel
        source_pixscale : float, optional
            Source pixel scale (default: self.pixel_scale)
        
        Returns
        -------
        np.ndarray
            Resampled image
        """
        from scipy.ndimage import zoom
        
        if source_pixscale is None:
            source_pixscale = self.pixel_scale
        
        zoom_factor = source_pixscale / target_pixscale
        
        return zoom(image, zoom_factor, order=3)
    
    def __len__(self) -> int:
        return self.n_sources
    
    def __repr__(self) -> str:
        return f"COSMOSLoaderV2(mode={self.mode}, n_sources={self.n_sources}, pixel_scale={self.pixel_scale})"


def build_cosmos_library(
    galsim_path: str,
    output_path: str,
    n_sources: int = 10000,
    stamp_size: int = 128,
    pixel_scale: float = 0.03,
    seed: int = 42
) -> None:
    """
    Build a COSMOS source library HDF5 from GalSim catalog.
    
    This preprocesses the GalSim COSMOS catalog into a fast-loading
    HDF5 file for use in Phase 4c lens injection.
    
    Parameters
    ----------
    galsim_path : str
        Path to GalSim COSMOS data directory
    output_path : str
        Output HDF5 file path
    n_sources : int
        Number of sources to include
    stamp_size : int
        Stamp size in pixels
    pixel_scale : float
        Pixel scale in arcsec/pixel
    seed : int
        Random seed for reproducibility
    """
    try:
        import galsim
    except ImportError:
        raise ImportError("GalSim required to build COSMOS library. Install with: pip install galsim")
    
    logger.info(f"Building COSMOS library from {galsim_path}")
    logger.info(f"Parameters: n_sources={n_sources}, stamp_size={stamp_size}, pixel_scale={pixel_scale}")
    
    # Load catalog
    catalog = galsim.COSMOSCatalog(dir=galsim_path, exclusion_level='marginal')
    n_available = catalog.getNObjects()
    n_sources = min(n_sources, n_available)
    
    logger.info(f"Catalog has {n_available} sources, extracting {n_sources}")
    
    # Select random subset
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_available)[:n_sources]
    
    # Extract images and metadata
    images = []
    half_light_radii = []
    magnitudes = []
    redshifts = []
    source_indices = []
    
    from tqdm import tqdm
    
    for i in tqdm(indices, desc="Extracting sources"):
        try:
            gal = catalog.makeGalaxy(index=int(i), gal_type='real', noise_pad_size=0)
            
            # Draw to stamp
            image = galsim.Image(stamp_size, stamp_size, scale=pixel_scale)
            gal.drawImage(image, scale=pixel_scale)
            
            # Get metadata
            rec = catalog.getParametricRecord(int(i))
            
            images.append(image.array.astype(np.float32))
            half_light_radii.append(float(rec.get('hlr', 0.3)))
            magnitudes.append(float(rec.get('mag_auto', 24.0)))
            redshifts.append(float(rec.get('zphot', 1.5)))
            source_indices.append(int(i))
            
        except Exception as e:
            logger.debug(f"Failed to extract source {i}: {e}")
            continue
    
    images = np.stack(images)
    
    # Compute clumpiness
    loader = COSMOSLoaderV2(mode='synthetic')  # Just for the method
    clumpiness = loader._compute_clumpiness_batch(images)
    
    # Save to HDF5
    logger.info(f"Saving {len(images)} sources to {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('images', data=images, compression='gzip')
        f.create_dataset('clumpiness', data=clumpiness)
        f.create_dataset('half_light_radius', data=np.array(half_light_radii))
        f.create_dataset('magnitude', data=np.array(magnitudes))
        f.create_dataset('redshift', data=np.array(redshifts))
        f.create_dataset('source_index', data=np.array(source_indices))
        
        # Metadata
        f.attrs['pixel_scale'] = pixel_scale
        f.attrs['stamp_size'] = stamp_size
        f.attrs['n_sources'] = len(images)
        f.attrs['galsim_path'] = galsim_path
        f.attrs['seed'] = seed
    
    logger.info(f"COSMOS library saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build COSMOS source library")
    parser.add_argument("--galsim-path", required=True, help="Path to GalSim COSMOS data")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--n-sources", type=int, default=10000, help="Number of sources")
    parser.add_argument("--stamp-size", type=int, default=128, help="Stamp size in pixels")
    parser.add_argument("--pixel-scale", type=float, default=0.03, help="Pixel scale in arcsec")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    build_cosmos_library(
        galsim_path=args.galsim_path,
        output_path=args.output,
        n_sources=args.n_sources,
        stamp_size=args.stamp_size,
        pixel_scale=args.pixel_scale,
        seed=args.seed
    )


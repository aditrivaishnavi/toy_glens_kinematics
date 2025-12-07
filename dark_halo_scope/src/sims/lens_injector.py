"""
Lens injection engine for DR10 backgrounds.

This module provides the core LensInjector class that injects
gravitationally lensed arcs into real DR10 LRG cutouts.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from copy import deepcopy


@dataclass
class InjectionParams:
    """Parameters for a single lens injection."""
    theta_E: float          # Einstein radius (arcsec)
    z_l: float              # Lens redshift
    z_s: float              # Source redshift
    m_source_r: float       # Source r-band magnitude
    ellipticity: Tuple[float, float]  # (e1, e2) lens ellipticity
    shear: Tuple[float, float]        # (gamma1, gamma2) external shear
    source_position: Tuple[float, float]  # (x, y) in source plane (arcsec)
    source_idx: Optional[int] = None  # Index of source in COSMOS catalog
    clumpiness: float = 0.0  # Source clumpiness metric


class LensInjector:
    """
    Inject lensed arcs into raw DR10 cutouts.
    
    Key requirements:
    - Operates on raw DR10 cutouts (flux units)
    - Uses per-band PSF FWHM
    - Applies SED-based magnitude offsets for blue sources
    - Returns new raw cutouts with injected arc
    
    Parameters
    ----------
    pixel_scale : float
        DR10 pixel scale in arcsec/pixel (default: 0.262)
    zero_point : float
        Photometric zero point for magnitude conversion (default: 22.5)
    """
    
    # Blue star-forming galaxy SED offsets relative to r-band
    # Typical values for z ~ 1-2 star-forming galaxies
    SED_OFFSETS = {
        'g': -0.3,   # Bluer: brighter in g
        'r': 0.0,    # Reference band
        'z': +0.2    # Redder: fainter in z
    }
    
    def __init__(
        self,
        pixel_scale: float = 0.262,
        zero_point: float = 22.5
    ):
        self.pixel_scale = pixel_scale
        self.zero_point = zero_point
    
    def mag_to_flux(self, mag: float) -> float:
        """
        Convert magnitude to DR10-like flux (nanomaggies).
        
        flux = 10^((zero_point - mag) / 2.5)
        """
        return 10 ** ((self.zero_point - mag) / 2.5)
    
    def flux_to_mag(self, flux: float) -> float:
        """Convert flux to magnitude."""
        if flux <= 0:
            return 99.0  # Faint/invalid
        return self.zero_point - 2.5 * np.log10(flux)
    
    def get_sed_offsets(self, z_source: float) -> Dict[str, float]:
        """
        Get band magnitude offsets for a blue star-forming source.
        
        These offsets simulate a typical Lyman-break / star-forming
        galaxy SED at high redshift, which appears blue in optical bands.
        
        Parameters
        ----------
        z_source : float
            Source redshift
        
        Returns
        -------
        dict
            {'g': Δm_g, 'r': 0.0, 'z': Δm_z}
            Negative means brighter in that band.
        """
        # Simple model: SED gets slightly bluer at higher z
        # (Lyman break moving through the bands)
        z_factor = np.clip((z_source - 1.0) / 2.0, 0, 1)
        
        return {
            'g': self.SED_OFFSETS['g'] * (1 + 0.3 * z_factor),
            'r': 0.0,
            'z': self.SED_OFFSETS['z'] * (1 + 0.2 * z_factor)
        }
    
    def inject_arc(
        self,
        background_cutout: Dict[str, np.ndarray],
        psf_fwhm: Dict[str, float],
        source_image: np.ndarray,
        params: InjectionParams,
        source_pixel_scale: float = 0.05
    ) -> Tuple[Dict[str, np.ndarray], InjectionParams]:
        """
        Inject a lensed arc into the background cutout.
        
        Parameters
        ----------
        background_cutout : dict
            {'g': arr, 'r': arr, 'z': arr} raw flux arrays
        psf_fwhm : dict
            {'g': fwhm, 'r': fwhm, 'z': fwhm} in arcseconds
        source_image : np.ndarray
            High-resolution COSMOS source image
        params : InjectionParams
            Lensing parameters
        source_pixel_scale : float
            Pixel scale of source image (arcsec/pixel)
        
        Returns
        -------
        injected_cutout : dict
            New cutout with injected arc
        params : InjectionParams
            Updated parameters (with any modifications)
        """
        try:
            import lenstronomy.Util.util as util
            from lenstronomy.LensModel.lens_model import LensModel
            from lenstronomy.LightModel.light_model import LightModel
            from lenstronomy.ImSim.image_model import ImageModel
            from lenstronomy.Data.imaging_data import ImageData
            from lenstronomy.Data.psf import PSF
        except ImportError:
            raise ImportError("lenstronomy is required for lens injection. Install with: pip install lenstronomy")
        
        # Get image dimensions
        ref_band = list(background_cutout.keys())[0]
        ny, nx = background_cutout[ref_band].shape
        
        # Build coordinate grid
        ra_at_xy_0 = -(nx / 2) * self.pixel_scale
        dec_at_xy_0 = -(ny / 2) * self.pixel_scale
        
        # Lens model: SIE (Singular Isothermal Ellipsoid)
        lens_model = LensModel(['SIE', 'SHEAR'])
        kwargs_lens = [
            {
                'theta_E': params.theta_E,
                'e1': params.ellipticity[0],
                'e2': params.ellipticity[1],
                'center_x': 0.0,
                'center_y': 0.0
            },
            {
                'gamma1': params.shear[0],
                'gamma2': params.shear[1],
                'ra_0': 0.0,
                'dec_0': 0.0
            }
        ]
        
        # Light model: Interpolated source
        light_model = LightModel(['INTERPOL'])
        
        # Get SED offsets
        sed_offsets = self.get_sed_offsets(params.z_s)
        
        # Inject into each band
        injected = {}
        for band in ['g', 'r', 'z']:
            if band not in background_cutout:
                continue
            
            # Target magnitude for this band
            target_mag = params.m_source_r + sed_offsets[band]
            target_flux = self.mag_to_flux(target_mag)
            
            # Build ImageData
            kwargs_data = {
                'image_data': np.zeros((ny, nx)),
                'ra_at_xy_0': ra_at_xy_0,
                'dec_at_xy_0': dec_at_xy_0,
                'transform_pix2angle': np.array([[self.pixel_scale, 0], 
                                                  [0, self.pixel_scale]])
            }
            data = ImageData(**kwargs_data)
            
            # Build PSF
            fwhm = psf_fwhm.get(band, 1.2)
            sigma_psf = fwhm / 2.355  # FWHM to sigma
            
            # Create Gaussian PSF kernel
            psf_size = int(4 * fwhm / self.pixel_scale) | 1  # Odd number
            psf_size = max(psf_size, 11)
            y, x = np.ogrid[:psf_size, :psf_size]
            center = psf_size // 2
            psf_kernel = np.exp(-((x-center)**2 + (y-center)**2) / (2 * (sigma_psf/self.pixel_scale)**2))
            psf_kernel /= psf_kernel.sum()
            
            kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': psf_kernel}
            psf_class = PSF(**kwargs_psf)
            
            # Prepare source image
            # Normalize source to unit total flux, then scale to target
            source_norm = source_image / source_image.sum()
            
            # Scale factor for source image
            # Source is at higher resolution, need to account for pixel scale ratio
            scale_ratio = source_pixel_scale / self.pixel_scale
            
            kwargs_source = [{
                'image': source_norm * target_flux,
                'center_x': params.source_position[0],
                'center_y': params.source_position[1],
                'phi_G': 0.0,
                'scale': source_pixel_scale
            }]
            
            # Create image model and render
            image_model = ImageModel(
                data,
                psf_class,
                lens_model_class=lens_model,
                source_model_class=light_model
            )
            
            # Generate lensed arc
            arc = image_model.image(
                kwargs_lens=kwargs_lens,
                kwargs_source=deepcopy(kwargs_source),
                kwargs_lens_light=None,
                kwargs_ps=None
            )
            
            # Add arc to background
            injected[band] = background_cutout[band] + arc.astype(np.float32)
        
        return injected, params
    
    def check_arc_visibility(
        self,
        arc_flux: np.ndarray,
        threshold: float = 100.0
    ) -> bool:
        """
        Check if injected arc has sufficient total flux to be visible.
        
        Parameters
        ----------
        arc_flux : np.ndarray
            The arc-only flux image (not including background)
        threshold : float
            Minimum total flux for visibility
        
        Returns
        -------
        bool
            True if arc is likely visible
        """
        return arc_flux.sum() > threshold
    
    def sample_injection_params(
        self,
        theta_E_range: Tuple[float, float] = (0.8, 2.5),
        z_l_range: Tuple[float, float] = (0.2, 0.6),
        z_s_range: Tuple[float, float] = (1.0, 2.5),
        m_source_range: Tuple[float, float] = (22.0, 25.0),
        seed: Optional[int] = None
    ) -> InjectionParams:
        """
        Sample random injection parameters.
        
        Parameters
        ----------
        theta_E_range : tuple
            (min, max) Einstein radius in arcsec
        z_l_range : tuple
            (min, max) lens redshift
        z_s_range : tuple
            (min, max) source redshift
        m_source_range : tuple
            (min, max) source r-band magnitude
        seed : int, optional
            Random seed
        
        Returns
        -------
        InjectionParams
            Sampled parameters
        """
        if seed is not None:
            np.random.seed(seed)
        
        theta_E = np.random.uniform(*theta_E_range)
        z_l = np.random.uniform(*z_l_range)
        z_s = np.random.uniform(max(z_s_range[0], z_l + 0.3), z_s_range[1])
        m_source = np.random.uniform(*m_source_range)
        
        # Random ellipticity (moderate values)
        e_mag = np.random.uniform(0.1, 0.4)
        e_angle = np.random.uniform(0, 2 * np.pi)
        e1 = e_mag * np.cos(2 * e_angle)
        e2 = e_mag * np.sin(2 * e_angle)
        
        # Small external shear
        gamma_mag = np.random.uniform(0.0, 0.1)
        gamma_angle = np.random.uniform(0, 2 * np.pi)
        gamma1 = gamma_mag * np.cos(2 * gamma_angle)
        gamma2 = gamma_mag * np.sin(2 * gamma_angle)
        
        # Source position: slightly offset from caustic for interesting arcs
        # Position within ~0.5 * theta_E of center
        r_source = np.random.uniform(0.05, 0.4) * theta_E
        phi_source = np.random.uniform(0, 2 * np.pi)
        x_source = r_source * np.cos(phi_source)
        y_source = r_source * np.sin(phi_source)
        
        return InjectionParams(
            theta_E=theta_E,
            z_l=z_l,
            z_s=z_s,
            m_source_r=m_source,
            ellipticity=(e1, e2),
            shear=(gamma1, gamma2),
            source_position=(x_source, y_source)
        )


"""
MaNGA-specific extractor for flux and velocity maps from DAP MAPS files.
"""

import numpy as np
import matplotlib.pyplot as plt

from .fits_loader import FITSLoader


class MaNGAMapsExtractor:
    """
    Extract flux and velocity maps from MaNGA DAP MAPS files.
    
    MaNGA MAPS files contain multiple emission line fluxes and kinematic maps.
    This class extracts specific maps and applies appropriate masking.
    
    Emission Line Indices (EMLINE_GFLUX):
        The EMLINE_GFLUX extension has shape (N_lines, ny, nx).
        Common line indices (0-based):
            - 0: OII-3727
            - 1: OII-3729
            - 7: Hbeta-4861
            - 11: OIII-4959
            - 12: OIII-5007
            - 24: Halpha-6564  <-- Default, confirmed from header
            - 19: NII-6583
            - 20: SII-6717
            - 21: SII-6731
        
        To use a different line, change the `emline_index` parameter.
    
    Attributes:
        loader: The underlying FITSLoader instance
        emline_index: Index of the emission line to extract (default: 18 = H-alpha)
    """
    
    # Default emission line: H-alpha (confirmed from DR17 DAP header)
    DEFAULT_EMLINE_INDEX = 24
    
    # Extension names in MaNGA MAPS files
    EMLINE_FLUX_EXT = "EMLINE_GFLUX"
    EMLINE_MASK_EXT = "EMLINE_GFLUX_MASK"
    STELLAR_VEL_EXT = "STELLAR_VEL"
    STELLAR_VEL_MASK_EXT = "STELLAR_VEL_MASK"
    
    def __init__(self, filepath: str, emline_index: int = None):
        """
        Initialize the MaNGA extractor.
        
        Args:
            filepath: Path to the MaNGA MAPS FITS file
            emline_index: Index of emission line to extract (default: 18 = H-alpha)
                          See class docstring for available indices.
        """
        self.loader = FITSLoader(filepath)
        self.emline_index = emline_index if emline_index is not None else self.DEFAULT_EMLINE_INDEX
        self._flux_map = None
        self._velocity_map = None
    
    def print_info(self):
        """Print HDU information from the FITS file."""
        with self.loader:
            self.loader.print_hdu_info()
            
            # Print additional info about emission lines
            emline_data = self.loader.get_extension_data(self.EMLINE_FLUX_EXT)
            print(f"EMLINE_GFLUX shape: {emline_data.shape}")
            print(f"  -> {emline_data.shape[0]} emission lines, "
                  f"{emline_data.shape[1]}x{emline_data.shape[2]} spatial pixels")
            print(f"  -> Currently using line index {self.emline_index} (H-alpha default)")
            print(f"  -> To change, pass emline_index=N to MaNGAMapsExtractor()\n")
    
    def extract_maps(self) -> tuple:
        """
        Extract emission-line flux and stellar velocity maps.
        
        Returns:
            tuple: (flux_map, velocity_map) as 2D numpy arrays
                   Bad pixels are masked (set to 0)
        """
        with self.loader:
            # Extract emission line flux
            emline_flux = self.loader.get_extension_data(self.EMLINE_FLUX_EXT)
            emline_mask = self.loader.get_extension_data(self.EMLINE_MASK_EXT)
            
            # Get the specific emission line (e.g., H-alpha)
            flux_map = emline_flux[self.emline_index].copy()
            flux_mask = emline_mask[self.emline_index]
            
            # Extract stellar velocity
            velocity_map = self.loader.get_extension_data(self.STELLAR_VEL_EXT).copy()
            velocity_mask = self.loader.get_extension_data(self.STELLAR_VEL_MASK_EXT)
        
        # Apply masks: set bad pixels to 0
        # MaNGA mask convention: 0 = good, >0 = bad/flagged
        flux_map[flux_mask > 0] = 0
        velocity_map[velocity_mask > 0] = 0
        
        # Also handle NaN/inf values
        flux_map = np.nan_to_num(flux_map, nan=0.0, posinf=0.0, neginf=0.0)
        velocity_map = np.nan_to_num(velocity_map, nan=0.0, posinf=0.0, neginf=0.0)
        
        self._flux_map = flux_map
        self._velocity_map = velocity_map
        
        return flux_map, velocity_map
    
    def plot_maps(self, flux_map: np.ndarray = None, velocity_map: np.ndarray = None,
                  save_path: str = None):
        """
        Plot flux and velocity maps side-by-side.
        
        Args:
            flux_map: 2D flux array (uses cached if None)
            velocity_map: 2D velocity array (uses cached if None)
            save_path: If provided, save figure to this path
        """
        if flux_map is None:
            flux_map = self._flux_map
        if velocity_map is None:
            velocity_map = self._velocity_map
            
        if flux_map is None or velocity_map is None:
            raise RuntimeError("No maps available. Call extract_maps() first.")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Flux map
        im1 = axes[0].imshow(flux_map, origin='lower', cmap='viridis')
        axes[0].set_title(f'Emission Line Flux (index {self.emline_index})')
        axes[0].set_xlabel('X (spaxels)')
        axes[0].set_ylabel('Y (spaxels)')
        plt.colorbar(im1, ax=axes[0], label='Flux (10⁻¹⁷ erg/s/cm²/spaxel)')
        
        # Velocity map - use diverging colormap centered on 0
        vmax = np.percentile(np.abs(velocity_map[velocity_map != 0]), 95) if np.any(velocity_map != 0) else 100
        im2 = axes[1].imshow(velocity_map, origin='lower', cmap='RdBu_r', 
                             vmin=-vmax, vmax=vmax)
        axes[1].set_title('Stellar Velocity')
        axes[1].set_xlabel('X (spaxels)')
        axes[1].set_ylabel('Y (spaxels)')
        plt.colorbar(im2, ax=axes[1], label='Velocity (km/s)')
        
        plt.suptitle(f'MaNGA Galaxy: {self.loader.filepath.stem}', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()


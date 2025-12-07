"""
Generic FITS file loader for opening and inspecting FITS files.
"""

import numpy as np
from astropy.io import fits
from pathlib import Path


class FITSLoader:
    """
    Generic FITS file loader for opening and inspecting FITS files.
    
    Attributes:
        filepath: Path to the FITS file
        hdulist: The opened HDU list (available after open())
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the loader with a file path.
        
        Args:
            filepath: Path to the FITS file (can be .fits or .fits.gz)
        """
        self.filepath = Path(filepath)
        self.hdulist = None
        
    def open(self):
        """Open the FITS file and store the HDU list."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"FITS file not found: {self.filepath}")
        self.hdulist = fits.open(self.filepath)
        return self
    
    def close(self):
        """Close the FITS file."""
        if self.hdulist is not None:
            self.hdulist.close()
            self.hdulist = None
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
    
    def print_hdu_info(self):
        """Print summary information about all HDUs in the file."""
        if self.hdulist is None:
            raise RuntimeError("FITS file not opened. Call open() first.")
        
        print(f"\n{'='*60}")
        print(f"FITS File: {self.filepath.name}")
        print(f"{'='*60}")
        print(f"{'No.':<5} {'Name':<25} {'Type':<15} {'Shape'}")
        print("-" * 60)
        
        for i, hdu in enumerate(self.hdulist):
            name = hdu.name if hdu.name else "(Primary)"
            hdu_type = type(hdu).__name__
            shape = str(hdu.data.shape) if hdu.data is not None else "N/A"
            print(f"{i:<5} {name:<25} {hdu_type:<15} {shape}")
        
        print("=" * 60 + "\n")
    
    def get_extension_data(self, ext_name: str) -> np.ndarray:
        """
        Get data from a named extension.
        
        Args:
            ext_name: Name of the HDU extension
            
        Returns:
            numpy array of the extension data
        """
        if self.hdulist is None:
            raise RuntimeError("FITS file not opened. Call open() first.")
        
        try:
            return self.hdulist[ext_name].data
        except KeyError:
            available = [hdu.name for hdu in self.hdulist]
            raise KeyError(f"Extension '{ext_name}' not found. Available: {available}")
    
    def get_extension_header(self, ext_name: str):
        """
        Get header from a named extension.
        
        Args:
            ext_name: Name of the HDU extension
            
        Returns:
            FITS header object
        """
        if self.hdulist is None:
            raise RuntimeError("FITS file not opened. Call open() first.")
        return self.hdulist[ext_name].header


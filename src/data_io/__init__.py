"""Data I/O module for loading FITS files and MaNGA data."""

from .fits_loader import FITSLoader
from .manga_extractor import MaNGAMapsExtractor

__all__ = ["FITSLoader", "MaNGAMapsExtractor"]


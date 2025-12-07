"""
Physical constants and survey-specific parameters for the toy_glens_kinematics project.

These values are critical for maintaining geometric consistency when interfacing
with lensing codes like lenstronomy.
"""

# =============================================================================
# MaNGA Survey Parameters
# =============================================================================

# MaNGA DR17 IFU datacubes and MAPS products use a fixed 0.5 arcsec per spaxel grid.
# This is a design feature of the survey and does not vary between galaxies.
# Reference: https://www.sdss.org/dr17/manga/manga-data/
MANGA_PIXEL_SCALE = 0.5  # arcsec per spaxel

# Typical MaNGA MAPS dimensions (varies by IFU bundle size, but commonly 72x72 or 74x74)
# The actual size should be read from the FITS file, but this is a common reference.
MANGA_TYPICAL_MAP_SIZE = 72  # pixels

# =============================================================================
# Emission Line Indices (for MaNGA DR17 DAP)
# =============================================================================

# Confirmed H-alpha index from MAPS header inspection
# This index applies to EMLINE_GFLUX, EMLINE_GVEL, etc.
HALPHA_LINE_INDEX = 24


# =============================================================================
# Helper Functions
# =============================================================================

def compute_resampled_pixel_scale(
    original_size: int,
    new_size: int,
    original_pixel_scale: float = MANGA_PIXEL_SCALE
) -> float:
    """
    Compute the new pixel scale after resampling a map while preserving field-of-view.
    
    When you resample from original_size to new_size pixels across the same
    physical sky coverage, the pixel scale changes accordingly.
    
    Args:
        original_size: Original number of pixels (e.g., 72)
        new_size: New number of pixels after resampling (e.g., 64)
        original_pixel_scale: Original arcsec per pixel (default: MaNGA 0.5")
        
    Returns:
        New pixel scale in arcsec per pixel
        
    Example:
        >>> compute_resampled_pixel_scale(72, 64, 0.5)
        0.5625  # 36" field / 64 pixels = 0.5625"/pixel
        
    Note:
        When using with lenstronomy, always pass the correct pixel_scale:
        
        data_kwargs = {
            "image_data": resampled_array,
            "pixel_scale": compute_resampled_pixel_scale(72, 64),
        }
    """
    # Total field of view in arcsec
    field_of_view = original_size * original_pixel_scale
    # New pixel scale
    return field_of_view / new_size


def get_field_of_view(map_size: int, pixel_scale: float = MANGA_PIXEL_SCALE) -> float:
    """
    Compute the total field of view in arcseconds.
    
    Args:
        map_size: Number of pixels along one axis
        pixel_scale: Arcsec per pixel (default: MaNGA 0.5")
        
    Returns:
        Total field of view in arcseconds
        
    Example:
        >>> get_field_of_view(72, 0.5)
        36.0  # 72 pixels * 0.5"/pixel = 36"
    """
    return map_size * pixel_scale


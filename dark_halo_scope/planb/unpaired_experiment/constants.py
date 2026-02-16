"""Constants for unpaired experiment."""

# Image dimensions (v5_cosmos_paired uses 64x64)
STAMP_SIZE = 64

BANDS = ("g", "r", "z")
CORE_BOX = 10  # 10x10 central pixels for core gate
CORE_RADIUS = 5  # radius for core dropout
SEED_DEFAULT = 42

# Training defaults
EARLY_STOPPING_PATIENCE = 15  # epochs without improvement before stopping

# Matching columns available in our data
# These are used for stratified negative sampling
MATCHING_COLUMNS = [
    "psf_bin",      # Pre-computed PSF bin (0-3)
    "depth_bin",    # Pre-computed depth bin (0-3)
]

# Alternative: use continuous columns if bins don't work
MATCHING_COLUMNS_CONTINUOUS = [
    "psf_fwhm_used_r",  # PSF FWHM in arcsec
    "psfdepth_r",       # Depth in r-band
]

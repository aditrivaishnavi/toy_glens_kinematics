"""
Diagnostic script: Inspect MaNGA MAPS data in detail.

This script performs comprehensive diagnostics on a MaNGA DAP MAPS file:
1. Emission line index discovery (find H-alpha dynamically)
2. Units verification
3. Gas vs stellar velocity comparison
4. Data quality / S/N check
5. WCS / pixel scale info
6. Rotation pattern sanity check

Usage:
    cd toy_glens_kinematics
    python src/scripts/inspect_manga_data.py
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


# =============================================================================
# Configuration
# =============================================================================

MAPS_PATH = "data/manga-8138-12704-MAPS-HYB10-MILESHC-MASTARHC2.fits"


# =============================================================================
# Diagnostic Functions
# =============================================================================

def find_halpha_index(em_header):
    """
    Find the H-alpha emission line index from the EMLINE_GFLUX header.
    
    Returns:
        tuple: (index, line_name) or (None, None) if not found
    """
    ha_keywords = ['HA', 'HALPHA', 'H-ALPHA', 'HA-6563', 'HA6563']
    
    for key, value in em_header.items():
        if isinstance(value, str):
            value_upper = value.upper().replace(' ', '').replace('-', '').replace('_', '')
            for ha_kw in ha_keywords:
                if ha_kw.replace('-', '') in value_upper:
                    # Extract index from key (e.g., "C18" -> 18, "ELINE18" -> 18)
                    import re
                    match = re.search(r'(\d+)', key)
                    if match:
                        idx = int(match.group(1))
                        return idx, value
    return None, None


def section_1_emission_line_discovery(hdu):
    """Section 1: Discover emission line indices and find H-alpha."""
    print("\n" + "=" * 60)
    print("SECTION 1: EMISSION LINE INDEX DISCOVERY")
    print("=" * 60)
    
    em_hdr = hdu['EMLINE_GFLUX'].header
    
    # Print all relevant header keys
    print("\n=== EMLINE_GFLUX HEADER: LINE LABELS ===")
    line_info = {}
    for key, value in em_hdr.items():
        key_upper = key.upper()
        if any(x in key_upper for x in ['NAME', 'LINE', 'WAVE', 'ELINE', 'C0', 'C1', 'C2']):
            print(f"{key:20s} : {value}")
            # Try to extract index
            import re
            match = re.search(r'(\d+)', key)
            if match and isinstance(value, str):
                idx = int(match.group(1))
                line_info[idx] = value
    
    # Also check for standard naming convention (Cxx keys)
    print("\n=== Emission line mapping (by index) ===")
    n_lines = hdu['EMLINE_GFLUX'].data.shape[0]
    print(f"Total emission lines in cube: {n_lines}")
    
    # Try to find H-alpha
    ha_idx, ha_name = find_halpha_index(em_hdr)
    
    if ha_idx is not None:
        print(f"\n>>> H-alpha found at index {ha_idx}: '{ha_name}'")
    else:
        # Fallback: check common indices
        print("\n>>> H-alpha not found automatically in header.")
        print("    Confirmed MaNGA DR17 index for H-alpha is 24")
        print("    Will use index 24 as default")
        ha_idx = 24  # DR17 DAP confirmed H-alpha index
    
    return ha_idx


def section_2_units_verification(hdu):
    """Section 2: Verify units of flux and velocity maps."""
    print("\n" + "=" * 60)
    print("SECTION 2: UNITS VERIFICATION")
    print("=" * 60)
    
    print("\n=== Units and basic header keys ===")
    
    extensions = ['EMLINE_GFLUX', 'EMLINE_GVEL', 'STELLAR_VEL']
    for ext in extensions:
        try:
            bunit = hdu[ext].header.get('BUNIT', 'NOT FOUND')
            print(f"{ext:20s} BUNIT: {bunit}")
        except KeyError:
            print(f"{ext:20s} : Extension not found")
    
    print("\n=== Expected units ===")
    print("EMLINE_GFLUX : 1E-17 erg/s/cm^2/spaxel")
    print("EMLINE_GVEL  : km/s")
    print("STELLAR_VEL  : km/s")


def section_3_velocity_comparison(hdu, ha_idx):
    """Section 3: Compare gas vs stellar velocity maps."""
    print("\n" + "=" * 60)
    print("SECTION 3: GAS VS STELLAR VELOCITY COMPARISON")
    print("=" * 60)
    
    # Extract gas velocity (H-alpha)
    gas_vel = hdu['EMLINE_GVEL'].data[ha_idx].astype(float)
    gas_vel_mask = hdu['EMLINE_GVEL_MASK'].data[ha_idx] > 0
    gas_vel_ma = np.ma.MaskedArray(gas_vel, mask=gas_vel_mask)
    
    # Extract stellar velocity
    stellar_vel = hdu['STELLAR_VEL'].data.astype(float)
    stellar_vel_mask = hdu['STELLAR_VEL_MASK'].data > 0
    stellar_vel_ma = np.ma.MaskedArray(stellar_vel, mask=stellar_vel_mask)
    
    print("\n=== Velocity summary ===")
    print(f"Gas vel  (Hα): min = {gas_vel_ma.min():.2f}, max = {gas_vel_ma.max():.2f} km/s")
    print(f"Stellar vel  : min = {stellar_vel_ma.min():.2f}, max = {stellar_vel_ma.max():.2f} km/s")
    
    # Correlation where both are valid
    valid = (~gas_vel_ma.mask) & (~stellar_vel_ma.mask)
    n_valid = valid.sum()
    print(f"Overlapping valid pixels: {n_valid}")
    
    if n_valid > 50:
        corr = np.corrcoef(gas_vel_ma[valid].data, stellar_vel_ma[valid].data)[0, 1]
        print(f"Gas vs Stellar vel correlation: {corr:.3f}")
        if corr > 0.7:
            print("  -> High correlation: gas and stars trace similar rotation")
        elif corr > 0.4:
            print("  -> Moderate correlation: some differences in kinematics")
        else:
            print("  -> Low correlation: gas and stellar kinematics differ significantly")
    else:
        print("Too few overlapping valid pixels to compute correlation.")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    vmax_gas = np.percentile(np.abs(gas_vel_ma.compressed()), 95) if gas_vel_ma.count() > 0 else 100
    vmax_stellar = np.percentile(np.abs(stellar_vel_ma.compressed()), 95) if stellar_vel_ma.count() > 0 else 100
    vmax = max(vmax_gas, vmax_stellar)
    
    im0 = axes[0].imshow(gas_vel_ma, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"Gas Velocity (Hα, idx={ha_idx})")
    plt.colorbar(im0, ax=axes[0], label='km/s')
    
    im1 = axes[1].imshow(stellar_vel_ma, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title("Stellar Velocity")
    plt.colorbar(im1, ax=axes[1], label='km/s')
    
    diff = gas_vel_ma - stellar_vel_ma
    vmax_diff = np.percentile(np.abs(diff.compressed()), 95) if diff.count() > 0 else 50
    im2 = axes[2].imshow(diff, origin='lower', cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
    axes[2].set_title("Gas - Stellar (km/s)")
    plt.colorbar(im2, ax=axes[2], label='km/s')
    
    plt.suptitle("Section 3: Gas vs Stellar Velocity Comparison", fontsize=12)
    plt.tight_layout()
    plt.show()
    
    return gas_vel_ma, stellar_vel_ma


def section_4_data_quality(hdu, ha_idx):
    """Section 4: Check S/N and mask fraction."""
    print("\n" + "=" * 60)
    print("SECTION 4: DATA QUALITY / S/N CHECK")
    print("=" * 60)
    
    # Get S/N map
    try:
        snr = hdu['SPX_SNR'].data.astype(float)
    except KeyError:
        try:
            snr = hdu['BIN_SNR'].data.astype(float)
            print("(Using BIN_SNR instead of SPX_SNR)")
        except KeyError:
            print("WARNING: No SNR extension found (SPX_SNR or BIN_SNR)")
            snr = None
    
    # Get H-alpha flux
    ha_flux = hdu['EMLINE_GFLUX'].data[ha_idx].astype(float)
    ha_flux_mask = hdu['EMLINE_GFLUX_MASK'].data[ha_idx] > 0
    
    valid_flux = (~ha_flux_mask) & (ha_flux > 0)
    
    print("\n=== Data quality ===")
    print(f"Total spaxels          : {ha_flux.size}")
    print(f"Valid flux spaxels     : {valid_flux.sum()}")
    
    if snr is not None:
        valid_snr = snr > 5
        valid_both = valid_flux & valid_snr
        print(f"Valid SNR>5 spaxels    : {valid_snr.sum()}")
        print(f"Valid flux & SNR>5     : {valid_both.sum()}")
        
        # Quality assessment
        if valid_both.sum() > 500:
            print("\n>>> GOOD: Plenty of high-quality spaxels for analysis")
        elif valid_both.sum() > 100:
            print("\n>>> OK: Moderate number of valid spaxels")
        else:
            print("\n>>> WARNING: Few valid spaxels - consider a different galaxy")
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        snr_valid = snr[snr > 0]
        if len(snr_valid) > 0:
            axes[0].hist(snr_valid.ravel(), bins=50, edgecolor='black')
            axes[0].axvline(5, color='r', linestyle='--', label='SNR=5 threshold')
            axes[0].set_xlabel("SNR")
            axes[0].set_ylabel("Count")
            axes[0].set_title("SNR Histogram")
            axes[0].legend()
        
        axes[1].imshow(valid_both.astype(int), origin='lower', cmap='gray_r')
        axes[1].set_title(f"Valid (flux>0 & SNR>5) mask\n({valid_both.sum()} pixels)")
        
        plt.suptitle("Section 4: Data Quality", fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print("SNR data not available for detailed quality check")


def section_5_wcs_info(hdu):
    """Section 5: Extract WCS / pixel scale info."""
    print("\n" + "=" * 60)
    print("SECTION 5: WCS / PIXEL SCALE INFO")
    print("=" * 60)
    
    # MaNGA DR17 standard pixel scale (survey design parameter)
    MANGA_PIXEL_SCALE = 0.5  # arcsec per spaxel
    
    print("\n=== WCS / pixel scale info ===")
    primary_hdr = hdu[0].header
    
    wcs_keys = ["CD1_1", "CD2_2", "CDELT1", "CDELT2", "CUNIT1", "CUNIT2", 
                "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2"]
    
    found_keys = {}
    for key in wcs_keys:
        if key in primary_hdr:
            value = primary_hdr[key]
            print(f"{key:10s} = {value}")
            found_keys[key] = value
    
    # Try to compute pixel scale from header
    pixel_scale_from_header = None
    if "CD1_1" in found_keys:
        # CD matrix: scale is sqrt(CD1_1^2 + CD2_1^2) but often just |CD1_1|
        pixel_scale_from_header = abs(found_keys["CD1_1"]) * 3600  # deg to arcsec
    elif "CDELT1" in found_keys:
        pixel_scale_from_header = abs(found_keys["CDELT1"]) * 3600
    
    # Get map dimensions
    emline_shape = hdu['EMLINE_GFLUX'].data.shape
    map_size = emline_shape[1]  # Assuming square maps
    
    print(f"\n=== Pixel Scale Summary ===")
    if pixel_scale_from_header:
        print(f"From header     : {pixel_scale_from_header:.4f} arcsec/pixel")
    else:
        print(f"From header     : NOT FOUND")
    
    print(f"MaNGA standard  : {MANGA_PIXEL_SCALE} arcsec/spaxel (survey design)")
    print(f"Map dimensions  : {map_size} x {map_size} pixels")
    
    # Calculate field of view
    fov = map_size * MANGA_PIXEL_SCALE
    print(f"Field of view   : {fov:.1f} x {fov:.1f} arcsec")
    
    print("\n=== IMPORTANT: Lenstronomy Integration ===")
    print("When using these maps with lenstronomy, you MUST specify pixel_scale:")
    print()
    print("  # For native MaNGA resolution:")
    print("  data_kwargs = {")
    print('      "image_data": flux_map,')
    print(f'      "pixel_scale": {MANGA_PIXEL_SCALE},  # arcsec/pixel')
    print("  }")
    print()
    print("  # If resampling from 72x72 to 64x64 (same field of view):")
    print(f"  new_pixel_scale = {fov:.1f} / 64  # = {fov/64:.4f} arcsec/pixel")
    print()
    print("  WARNING: Wrong pixel_scale will make Einstein radii incorrect!")
    print(f"  - theta_E = 1.5\" at {MANGA_PIXEL_SCALE}\"/pix = {1.5/MANGA_PIXEL_SCALE:.1f} pixels")
    print(f"  - theta_E = 1.5\" at 1.0\"/pix = 1.5 pixels (WRONG if using MaNGA!)")


def section_6_rotation_check(stellar_vel_ma):
    """Section 6: Sanity-check the rotation pattern numerically."""
    print("\n" + "=" * 60)
    print("SECTION 6: ROTATION PATTERN SANITY CHECK")
    print("=" * 60)
    
    # Use stellar velocity
    v = stellar_vel_ma.filled(np.nan)
    yy, xx = np.indices(v.shape)
    
    # Only use valid pixels
    mask = ~np.isnan(v)
    n_valid = mask.sum()
    
    if n_valid < 10:
        print("\n>>> Too few valid pixels to fit rotation pattern")
        return
    
    x = xx[mask].ravel()
    y = yy[mask].ravel()
    z = v[mask].ravel()
    
    # Fit a plane: v = a*x + b*y + c (very rough gradient)
    A = np.column_stack([x, y, np.ones_like(x)])
    coeff, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coeff
    
    grad_amp = np.sqrt(a*a + b*b)
    grad_angle = np.degrees(np.arctan2(b, a))
    
    print("\n=== Rotation gradient rough fit ===")
    print(f"Gradient coefficients (a, b): ({a:.3f}, {b:.3f}) km/s per pixel")
    print(f"Gradient amplitude          : {grad_amp:.3f} km/s per pixel")
    print(f"Gradient direction          : {grad_angle:.1f} degrees")
    print(f"Constant offset (c)         : {c:.1f} km/s")
    
    print("\n=== Interpretation ===")
    if grad_amp < 1:
        print(">>> LOW gradient: Very weak or no coherent rotation")
        print("    This galaxy may not be an ideal rotating disk candidate")
    elif grad_amp < 5:
        print(">>> MODERATE gradient: Some rotation detected")
        print("    May be a slow rotator or inclined disk")
    else:
        print(">>> STRONG gradient: Clear rotation pattern")
        print("    Good candidate for kinematic lensing studies")
    
    # Compute residuals for quality of fit
    v_pred = a * xx + b * yy + c
    residuals = v - v_pred
    rms_residual = np.sqrt(np.nanmean(residuals**2))
    print(f"\nRMS residual from plane fit: {rms_residual:.1f} km/s")
    if rms_residual > 50:
        print("    (High residuals suggest complex kinematics beyond simple rotation)")


def main():
    """Main diagnostic function."""
    print("=" * 60)
    print("MaNGA DATA DIAGNOSTIC INSPECTION")
    print("=" * 60)
    print(f"\nFile: {MAPS_PATH}")
    
    # Open FITS file
    try:
        hdu = fits.open(MAPS_PATH)
    except FileNotFoundError:
        print(f"\nERROR: File not found: {MAPS_PATH}")
        print("Please ensure the MaNGA MAPS file is in the data/ directory")
        return
    
    print(f"Successfully opened FITS file with {len(hdu)} HDUs")
    
    # Run all diagnostic sections
    ha_idx = section_1_emission_line_discovery(hdu)
    section_2_units_verification(hdu)
    gas_vel_ma, stellar_vel_ma = section_3_velocity_comparison(hdu, ha_idx)
    section_4_data_quality(hdu, ha_idx)
    section_5_wcs_info(hdu)
    section_6_rotation_check(stellar_vel_ma)
    
    # Close file
    hdu.close()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print(f"\n>>> Identified H-alpha index: {ha_idx}")
    print(">>> Review the plots and summary above to assess data quality")
    print(">>> Update MaNGAMapsExtractor with correct ha_idx if needed")


if __name__ == "__main__":
    main()


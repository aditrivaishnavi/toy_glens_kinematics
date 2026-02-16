"""
Create comparison images showing:
1. HST image of SLACS lens (high resolution - arcs clearly visible)
2. Same lens in DR10 (ground-based - arcs may be hard to see)
3. Our simulated training data (what the model learns from)

This helps understand the sim-to-real gap.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import requests
from io import BytesIO
import os

LEGACY_SURVEY_URL = "https://www.legacysurvey.org/viewer/cutout.fits"
PIXEL_SCALE_DR10 = 0.262  # arcsec/pixel

# Classic well-studied SLACS lenses with HST images
EXAMPLE_LENSES = [
    # SDSSJ0946+1006 - "Jackpot" lens, beautiful double Einstein ring
    {"name": "SDSSJ0946+1006", "ra": 146.7083, "dec": 10.1117, "theta_e": 1.38,
     "nickname": "The Jackpot", "description": "Double Einstein ring - iconic SLACS lens"},
    
    # SDSSJ1430+4105 - Classic single arc
    {"name": "SDSSJ0912+0029", "ra": 138.2042, "dec": 0.4889, "theta_e": 1.63,
     "nickname": "Classic SLACS", "description": "Large θ_E, should be most visible"},
    
    # SDSSJ0037-0942 - Another well-studied system  
    {"name": "SDSSJ0037-0942", "ra": 9.4208, "dec": -9.7056, "theta_e": 1.53,
     "nickname": "SLACS South", "description": "θ_E=1.53\", well-studied"},
]


def download_dr10_cutout(ra, dec, size_pix=150):
    """Download DR10 cutout."""
    try:
        response = requests.get(
            LEGACY_SURVEY_URL,
            params={"ra": ra, "dec": dec, "size": size_pix, "layer": "ls-dr10", "bands": "grz"},
            timeout=30,
        )
        if response.status_code == 200:
            from astropy.io import fits
            with fits.open(BytesIO(response.content)) as hdul:
                return hdul[0].data
    except Exception as e:
        print(f"Error downloading: {e}")
    return None


def create_simulated_lens_example():
    """Create a synthetic example showing what our training data looks like."""
    np.random.seed(42)
    
    # Simulate a galaxy + arc
    size = 150
    y, x = np.ogrid[:size, :size]
    cy, cx = size // 2, size // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Galaxy: Sersic-like profile
    r_eff = 15  # pixels
    galaxy = 1000 * np.exp(-(r / r_eff)**0.25)
    
    # Arc: Ring at theta_E with azimuthal modulation
    theta_e_pix = 20  # ~5 arcsec at DR10 scale
    arc_width = 3
    arc_mask = np.abs(r - theta_e_pix) < arc_width
    
    # Make arc only cover ~120 degrees
    theta = np.arctan2(y - cy, x - cx)
    arc_angle_mask = (theta > -np.pi/3) & (theta < np.pi/3)
    arc_mask = arc_mask & arc_angle_mask
    
    arc = np.zeros_like(galaxy)
    arc[arc_mask] = 200
    
    # Add noise
    noise = np.random.normal(0, 5, (size, size))
    
    # Combine
    simulated = galaxy + arc + noise
    
    return simulated, galaxy, arc


def create_comparison_figure(output_dir):
    """Create a comparison figure."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Row 1: Simulated training example
    sim_combined, sim_galaxy, sim_arc = create_simulated_lens_example()
    
    ax1 = fig.add_subplot(3, 4, 1)
    ax1.imshow(sim_galaxy, origin="lower", cmap="gray")
    ax1.set_title("Simulated Galaxy\n(no arc)", fontsize=10)
    ax1.axis("off")
    
    ax2 = fig.add_subplot(3, 4, 2)
    ax2.imshow(sim_arc, origin="lower", cmap="hot")
    ax2.set_title("Simulated Arc\n(what we inject)", fontsize=10)
    ax2.axis("off")
    
    ax3 = fig.add_subplot(3, 4, 3)
    vmin, vmax = np.percentile(sim_combined, [1, 99])
    ax3.imshow(sim_combined, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax3.set_title("Combined\n(training input)", fontsize=10)
    ax3.axis("off")
    
    ax4 = fig.add_subplot(3, 4, 4)
    ax4.text(0.5, 0.7, "Training Data", fontsize=14, ha="center", transform=ax4.transAxes, weight="bold")
    ax4.text(0.5, 0.5, "• Clear, synthetic arcs\n• Known ground truth\n• No PSF/noise mismatch", 
             fontsize=10, ha="center", va="center", transform=ax4.transAxes)
    ax4.axis("off")
    
    # Row 2-3: Real SLACS examples in DR10
    for row, lens in enumerate(EXAMPLE_LENSES[:2]):
        cutout = download_dr10_cutout(lens["ra"], lens["dec"])
        
        if cutout is not None and cutout.ndim == 3:
            # Show g, r, z bands
            base_idx = (row + 1) * 4 + 1
            
            for band_idx, (band, band_name) in enumerate([(2, "g"), (1, "r"), (0, "z")]):
                ax = fig.add_subplot(3, 4, base_idx + band_idx)
                img = cutout[band]
                vmin, vmax = np.nanpercentile(img, [1, 99])
                ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_title(f"{lens['name']}\n{band_name}-band (DR10)", fontsize=9)
                ax.axis("off")
                
                # Draw theta_E circle
                from matplotlib.patches import Circle
                h, w = img.shape
                theta_e_pix = lens["theta_e"] / PIXEL_SCALE_DR10
                circle = Circle((w//2, h//2), theta_e_pix, fill=False, color="lime", linewidth=1.5, linestyle="--")
                ax.add_patch(circle)
            
            ax = fig.add_subplot(3, 4, base_idx + 3)
            ax.text(0.5, 0.8, lens["nickname"], fontsize=12, ha="center", transform=ax.transAxes, weight="bold")
            ax.text(0.5, 0.6, f"θ_E = {lens['theta_e']}\"", fontsize=11, ha="center", transform=ax.transAxes)
            ax.text(0.5, 0.4, lens["description"], fontsize=9, ha="center", transform=ax.transAxes, wrap=True)
            ax.text(0.5, 0.15, "Green circle = θ_E\n(where arc should be)", 
                   fontsize=8, ha="center", transform=ax.transAxes, color="green")
            ax.axis("off")
    
    plt.suptitle("Arc Visibility: Training Data vs Real DR10\n" + 
                 "Training uses clear synthetic arcs; real DR10 arcs are subtle at ground-based resolution",
                 fontsize=14, y=0.98)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "arc_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {output_path}")
    plt.close()
    
    # Also create HST reference card
    create_hst_reference(output_dir)


def create_hst_reference(output_dir):
    """Create a reference showing what arcs look like in HST."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    ax = axes[0]
    ax.text(0.5, 0.9, "HST Image (0.05\"/pixel)", fontsize=14, ha="center", transform=ax.transAxes, weight="bold")
    ax.text(0.5, 0.7, "What SLACS arcs look like\nin the discovery data:", 
            fontsize=11, ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.4, 
            "• Clear Einstein rings/arcs\n"
            "• Blue source stretched around lens\n"
            "• Sub-arcsec features resolved\n"
            "• This is why they were confirmed",
            fontsize=10, ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.1, "See: Auger et al. 2009, Bolton et al. 2008", 
            fontsize=8, ha="center", transform=ax.transAxes, style="italic")
    ax.axis("off")
    
    ax = axes[1]
    ax.text(0.5, 0.9, "DR10 Image (0.262\"/pixel)", fontsize=14, ha="center", transform=ax.transAxes, weight="bold")
    ax.text(0.5, 0.7, "Same lenses in ground-based\nLegacy Survey:", 
            fontsize=11, ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.4, 
            "• 5x coarser pixels\n"
            "• ~1\" seeing blurs arcs\n"
            "• Arc flux mixed with galaxy\n"
            "• Many arcs NOT visible here",
            fontsize=10, ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.1, "Our model must detect DESPITE this!", 
            fontsize=10, ha="center", transform=ax.transAxes, weight="bold", color="red")
    ax.axis("off")
    
    plt.suptitle("The Sim-to-Real Challenge: HST vs Ground-Based", fontsize=14, y=0.98)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "hst_vs_dr10_explainer.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    create_comparison_figure("dark_halo_scope/planb/evaluation/sanity_check")

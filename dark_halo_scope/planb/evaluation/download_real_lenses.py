"""
Download REAL confirmed strong lenses from DESI Legacy Survey DR10.

These are spectroscopically confirmed lenses - we can see what ACTUAL arcs
look like in ground-based data (not simulations).
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import requests
from io import BytesIO
import os

# Use the color JPEG endpoint for visual inspection
LEGACY_JPEG_URL = "https://www.legacysurvey.org/viewer/cutout.jpg"
LEGACY_FITS_URL = "https://www.legacysurvey.org/viewer/cutout.fits"

# Well-known confirmed strong lenses with visible arcs in ground-based data
# Selected for having the BEST visibility in Legacy Survey
FAMOUS_LENSES = [
    # Very prominent systems - should be visible even in ground-based
    {"name": "Horseshoe", "ra": 163.3217, "dec": 20.1917, "theta_e": 5.0,
     "description": "Cosmic Horseshoe - giant arc, one of the brightest known"},
    
    {"name": "Clone Arc", "ra": 220.9296, "dec": 2.3906, "theta_e": 3.8,
     "description": "The Clone - bright blue arc"},
    
    {"name": "8 O'Clock Arc", "ra": 0.8179, "dec": 14.6439, "theta_e": 3.0,
     "description": "8 O'Clock Arc - very bright z~2.7 source"},
    
    # SLACS - best cases with larger theta_E
    {"name": "SDSSJ0946+1006", "ra": 146.7083, "dec": 10.1117, "theta_e": 1.38,
     "description": "The Jackpot - double Einstein ring"},
    
    {"name": "SDSSJ0912+0029", "ra": 138.2042, "dec": 0.4889, "theta_e": 1.63,
     "description": "SLACS - large θ_E, should be visible"},
    
    {"name": "SDSSJ1430+4105", "ra": 217.7167, "dec": 41.0931, "theta_e": 1.52,
     "description": "SLACS - confirmed arc"},
    
    {"name": "SDSSJ1627-0053", "ra": 246.9375, "dec": -0.8903, "theta_e": 1.23,
     "description": "SLACS - moderate θ_E"},
    
    {"name": "SDSSJ0252+0039", "ra": 43.0917, "dec": 0.6569, "theta_e": 1.04,
     "description": "SLACS - compact"},
    
    # BELLS - emission line lenses
    {"name": "BELLSJ0747+4448", "ra": 116.8792, "dec": 44.8061, "theta_e": 1.35,
     "description": "BELLS - emission line arc"},
    
    {"name": "BELLSJ1110+3649", "ra": 167.5917, "dec": 36.8211, "theta_e": 1.08,
     "description": "BELLS - moderate θ_E"},
    
    # SL2S - Strong Lensing Legacy Survey (ground-based discoveries!)
    {"name": "SL2SJ0217-0513", "ra": 34.4583, "dec": -5.2250, "theta_e": 1.8,
     "description": "SL2S - discovered in CFHTLS ground-based data"},
    
    {"name": "SL2SJ1405+5243", "ra": 211.3292, "dec": 52.7292, "theta_e": 2.1,
     "description": "SL2S - large θ_E, visible from ground"},
]


def download_jpeg_cutout(ra, dec, size_pix=150, zoom=14):
    """Download color JPEG cutout."""
    try:
        response = requests.get(
            LEGACY_JPEG_URL,
            params={"ra": ra, "dec": dec, "size": size_pix, "layer": "ls-dr10", "zoom": zoom},
            timeout=30,
        )
        if response.status_code == 200:
            from PIL import Image
            return np.array(Image.open(BytesIO(response.content)))
    except Exception as e:
        print(f"Error: {e}")
    return None


def download_fits_cutout(ra, dec, size_pix=150):
    """Download FITS cutout for scientific analysis."""
    try:
        response = requests.get(
            LEGACY_FITS_URL,
            params={"ra": ra, "dec": dec, "size": size_pix, "layer": "ls-dr10", "bands": "grz"},
            timeout=30,
        )
        if response.status_code == 200:
            from astropy.io import fits
            with fits.open(BytesIO(response.content)) as hdul:
                return hdul[0].data
    except Exception as e:
        print(f"Error: {e}")
    return None


def create_real_lens_gallery(output_dir):
    """Download and display real confirmed lenses from Legacy Survey."""
    os.makedirs(output_dir, exist_ok=True)
    
    n_lenses = len(FAMOUS_LENSES)
    n_cols = 4
    n_rows = (n_lenses + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(20, 5 * n_rows))
    
    for i, lens in enumerate(FAMOUS_LENSES):
        print(f"Downloading {lens['name']}...")
        
        # Get color JPEG (easier to see arcs)
        jpeg = download_jpeg_cutout(lens["ra"], lens["dec"], size_pix=200)
        
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        
        if jpeg is not None:
            ax.imshow(jpeg)
            
            # Draw Einstein radius circle
            from matplotlib.patches import Circle
            theta_e_pix = lens["theta_e"] / 0.262  # Convert to pixels
            h, w = jpeg.shape[:2]
            circle = Circle((w//2, h//2), theta_e_pix, fill=False, 
                           color="lime", linewidth=2, linestyle="--")
            ax.add_patch(circle)
        else:
            ax.text(0.5, 0.5, "Download failed", ha="center", va="center", transform=ax.transAxes)
        
        ax.set_title(f"{lens['name']}\nθ_E={lens['theta_e']}\" | {lens['description']}", 
                    fontsize=9, wrap=True)
        ax.axis("off")
        
        # Add Legacy Survey viewer link
        ls_url = f"https://www.legacysurvey.org/viewer?ra={lens['ra']:.4f}&dec={lens['dec']:.4f}&layer=ls-dr10&zoom=16"
        ax.text(0.5, -0.05, "Click: Legacy Survey Viewer", fontsize=7, 
               ha="center", transform=ax.transAxes, color="blue", style="italic")
    
    plt.suptitle("REAL Confirmed Strong Lenses in DESI Legacy Survey DR10\n" +
                 "Green circle = Einstein radius (θ_E) | These are GROUND-BASED images",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "real_confirmed_lenses_dr10.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved to {output_path}")
    plt.close()
    
    # Also create an HTML page with clickable links
    create_lens_html(output_dir)


def create_lens_html(output_dir):
    """Create HTML page with links to Legacy Survey viewer."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Real Confirmed Strong Lenses - Legacy Survey DR10</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: white; }
        h1 { color: #eee; }
        .gallery { display: flex; flex-wrap: wrap; gap: 20px; }
        .lens { 
            background: #16213e; 
            border-radius: 8px; 
            padding: 10px; 
            width: 300px;
            text-align: center;
        }
        .lens img { max-width: 100%; border-radius: 4px; }
        .lens a { color: #4cc9f0; text-decoration: none; }
        .lens a:hover { text-decoration: underline; }
        .theta { color: #f72585; font-weight: bold; }
        .desc { font-size: 12px; color: #aaa; margin-top: 5px; }
    </style>
</head>
<body>
    <h1>Real Confirmed Strong Lenses in DESI Legacy Survey DR10</h1>
    <p>These are SPECTROSCOPICALLY CONFIRMED gravitational lenses. Click to view in Legacy Survey.</p>
    <div class="gallery">
"""
    
    for lens in FAMOUS_LENSES:
        ls_url = f"https://www.legacysurvey.org/viewer?ra={lens['ra']:.4f}&dec={lens['dec']:.4f}&layer=ls-dr10&zoom=16"
        img_url = f"https://www.legacysurvey.org/viewer/cutout.jpg?ra={lens['ra']}&dec={lens['dec']}&size=200&layer=ls-dr10"
        
        html += f"""
        <div class="lens">
            <a href="{ls_url}" target="_blank">
                <img src="{img_url}" alt="{lens['name']}">
            </a>
            <h3><a href="{ls_url}" target="_blank">{lens['name']}</a></h3>
            <p class="theta">θ_E = {lens['theta_e']}"</p>
            <p class="desc">{lens['description']}</p>
        </div>
"""
    
    html += """
    </div>
    <hr style="margin-top: 30px;">
    <h2>Key Observations</h2>
    <ul>
        <li><strong>Large θ_E lenses (>2")</strong>: Arcs often visible even in ground-based</li>
        <li><strong>Small θ_E lenses (<1.5")</strong>: Arcs often blurred into galaxy, hard to see</li>
        <li><strong>Blue arcs</strong>: Easier to see against red elliptical</li>
        <li><strong>SLACS/BELLS</strong>: Confirmed with HST, but arcs may NOT be visible in DR10</li>
    </ul>
</body>
</html>
"""
    
    html_path = Path(output_dir) / "real_lenses_gallery.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"Saved HTML to {html_path}")


if __name__ == "__main__":
    create_real_lens_gallery("dark_halo_scope/planb/evaluation/sanity_check")

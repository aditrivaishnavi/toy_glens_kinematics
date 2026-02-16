"""
Arc visibility sanity check for SLACS/BELLS anchors.

Per LLM review: "93/94 Tier-A with arc visible is a red flag"
Need human validation that the arc SNR metric is measuring real arcs.

This script:
1. Selects 20 random SLACS/BELLS anchors
2. Creates visualization showing:
   - r-band cutout
   - Residual image (after subtracting azimuthal median)
   - θ_E annulus overlay
   - Arc SNR value
3. Outputs a PDF/HTML for human labeling: "clear arc", "maybe", "no arc"
"""

import argparse
import logging
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.colors import Normalize
from io import BytesIO
import requests
import time

logger = logging.getLogger(__name__)

LEGACY_SURVEY_URL = "https://www.legacysurvey.org/viewer/cutout.fits"
PIXEL_SCALE = 0.262  # arcsec/pixel for Legacy Survey

# SLACS/BELLS anchors from our catalog
# Format: (name, ra, dec, theta_e_arcsec, source)
SLACS_BELLS_ANCHORS = [
    # SLACS (first 25)
    ("SDSSJ0037-0942", 9.4208, -9.7056, 1.53, "SLACS"),
    ("SDSSJ0252+0039", 43.0917, 0.6569, 1.04, "SLACS"),
    ("SDSSJ0330-0020", 52.6458, -0.3367, 1.10, "SLACS"),
    ("SDSSJ0728+3835", 112.0542, 38.5869, 1.25, "SLACS"),
    ("SDSSJ0737+3216", 114.4292, 32.2750, 1.00, "SLACS"),
    ("SDSSJ0822+2652", 125.5833, 26.8714, 1.17, "SLACS"),
    ("SDSSJ0912+0029", 138.2042, 0.4889, 1.63, "SLACS"),
    ("SDSSJ0935-0003", 143.8917, -0.0547, 0.87, "SLACS"),
    ("SDSSJ0936+0913", 144.1583, 9.2244, 1.09, "SLACS"),
    ("SDSSJ0946+1006", 146.7083, 10.1117, 1.38, "SLACS"),
    ("SDSSJ0956+5100", 149.2375, 51.0122, 1.33, "SLACS"),
    ("SDSSJ0959+0410", 149.9792, 4.1764, 0.99, "SLACS"),
    ("SDSSJ1016+3859", 154.2458, 38.9953, 1.09, "SLACS"),
    ("SDSSJ1020+1122", 155.0875, 11.3722, 1.20, "SLACS"),
    ("SDSSJ1023+4230", 155.8708, 42.5000, 1.41, "SLACS"),
    ("SDSSJ1029+0420", 157.4333, 4.3375, 1.01, "SLACS"),
    ("SDSSJ1103+5322", 165.9042, 53.3725, 1.02, "SLACS"),
    ("SDSSJ1106+5228", 166.6167, 52.4775, 1.23, "SLACS"),
    ("SDSSJ1112+0826", 168.0500, 8.4433, 1.49, "SLACS"),
    ("SDSSJ1134+6027", 173.5167, 60.4597, 1.10, "SLACS"),
    ("SDSSJ1142+1001", 175.6292, 10.0272, 0.98, "SLACS"),
    ("SDSSJ1143-0144", 175.9625, -1.7456, 1.68, "SLACS"),
    ("SDSSJ1153+4612", 178.4625, 46.2008, 1.05, "SLACS"),
    ("SDSSJ1204+0358", 181.1875, 3.9719, 1.31, "SLACS"),
    ("SDSSJ1205+4910", 181.3125, 49.1686, 1.22, "SLACS"),
    
    # BELLS (first 15)
    ("BELLSJ0747+4448", 116.8792, 44.8061, 1.35, "BELLS"),
    ("BELLSJ0801+4727", 120.3250, 47.4608, 1.18, "BELLS"),
    ("BELLSJ0830+5116", 127.5667, 51.2778, 1.42, "BELLS"),
    ("BELLSJ0847+2348", 131.7875, 23.8075, 1.27, "BELLS"),
    ("BELLSJ0918+5104", 139.7000, 51.0733, 1.10, "BELLS"),
    ("BELLSJ0944+0930", 146.0250, 9.5036, 1.55, "BELLS"),
    ("BELLSJ1014+3920", 153.6125, 39.3425, 0.95, "BELLS"),
    ("BELLSJ1040+3626", 160.1167, 36.4372, 1.23, "BELLS"),
    ("BELLSJ1110+3649", 167.5917, 36.8211, 1.08, "BELLS"),
    ("BELLSJ1141+2216", 175.4542, 22.2744, 1.31, "BELLS"),
    ("BELLSJ1159+5531", 179.9417, 55.5292, 1.44, "BELLS"),
    ("BELLSJ1221+3806", 185.3208, 38.1083, 1.19, "BELLS"),
    ("BELLSJ1318+3942", 199.6542, 39.7114, 1.36, "BELLS"),
    ("BELLSJ1337+3620", 204.3667, 36.3411, 1.12, "BELLS"),
    ("BELLSJ1403+0006", 210.9625, 0.1111, 1.48, "BELLS"),
]


from typing import Optional

def download_cutout(ra: float, dec: float, size_pix: int = 150) -> Optional[np.ndarray]:
    """Download cutout from Legacy Survey."""
    for attempt in range(5):
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
        except:
            time.sleep(2 ** attempt)
    return None


def compute_residual_and_snr(cutout: np.ndarray, theta_e_arcsec: float) -> dict:
    """
    Compute residual image and arc SNR.
    
    Returns:
        residual: image after subtracting azimuthal median
        snr: arc SNR value
        annulus_mask: boolean mask for θ_E annulus
    """
    if cutout.ndim == 3:
        # Use r-band (middle channel)
        img = cutout[1]
    else:
        img = cutout
    
    h, w = img.shape
    cy, cx = h // 2, w // 2
    
    # Create radial coordinate
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    
    # θ_E in pixels
    theta_e_pix = theta_e_arcsec / PIXEL_SCALE
    
    # Annulus: θ_E ± 2 pixels
    inner_r = max(1, theta_e_pix - 2)
    outer_r = theta_e_pix + 2
    annulus_mask = (r >= inner_r) & (r <= outer_r)
    
    # Compute azimuthal median in radial bins
    residual = img.copy()
    for r_bin in np.arange(1, max(h, w), 1):
        ring_mask = (r >= r_bin - 0.5) & (r < r_bin + 0.5)
        if ring_mask.sum() > 0:
            ring_median = np.nanmedian(img[ring_mask])
            residual[ring_mask] -= ring_median
    
    # MAD-based noise estimation
    mad = np.nanmedian(np.abs(residual - np.nanmedian(residual)))
    sigma = 1.4826 * mad
    
    # Arc SNR: sum of positive residuals in annulus
    annulus_residuals = residual[annulus_mask]
    positive_sum = np.sum(np.maximum(annulus_residuals, 0))
    n_pixels = annulus_mask.sum()
    expected_noise = sigma * np.sqrt(n_pixels)
    snr = positive_sum / expected_noise if expected_noise > 0 else 0.0
    
    return {
        "residual": residual,
        "snr": snr,
        "annulus_mask": annulus_mask,
        "inner_r": inner_r,
        "outer_r": outer_r,
        "sigma": sigma,
    }


def create_visualization_grid(anchors: list, output_path: str, n_samples: int = 20):
    """Create a visualization grid for human review."""
    
    # Random sample
    np.random.seed(42)
    indices = np.random.choice(len(anchors), size=min(n_samples, len(anchors)), replace=False)
    samples = [anchors[i] for i in indices]
    
    # Create figure: 4 columns (r-band, residual, annotated, labels), n_samples rows
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    for i, (name, ra, dec, theta_e, source) in enumerate(samples):
        logger.info(f"Processing {name} ({i+1}/{n_samples})...")
        
        cutout = download_cutout(ra, dec)
        
        if cutout is None:
            for j in range(4):
                axes[i, j].text(0.5, 0.5, "Download failed", ha="center", va="center")
                axes[i, j].axis("off")
            continue
        
        # Get r-band
        if cutout.ndim == 3:
            r_band = cutout[1]
        else:
            r_band = cutout
        
        # Compute residual and SNR
        result = compute_residual_and_snr(cutout, theta_e)
        
        # Column 1: r-band
        ax = axes[i, 0]
        vmin, vmax = np.nanpercentile(r_band, [1, 99])
        ax.imshow(r_band, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}\nθ_E={theta_e:.2f}\"", fontsize=8)
        ax.axis("off")
        
        # Column 2: residual
        ax = axes[i, 1]
        vmin, vmax = np.nanpercentile(result["residual"], [1, 99])
        ax.imshow(result["residual"], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"Residual (azimuthal median subtracted)", fontsize=8)
        ax.axis("off")
        
        # Column 3: residual with annulus overlay
        ax = axes[i, 2]
        ax.imshow(result["residual"], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        h, w = r_band.shape
        cy, cx = h // 2, w // 2
        inner_circle = Circle((cx, cy), result["inner_r"], fill=False, color="lime", linewidth=2)
        outer_circle = Circle((cx, cy), result["outer_r"], fill=False, color="lime", linewidth=2)
        ax.add_patch(inner_circle)
        ax.add_patch(outer_circle)
        ax.set_title(f"θ_E annulus (SNR={result['snr']:.1f})", fontsize=8)
        ax.axis("off")
        
        # Column 4: label area
        ax = axes[i, 3]
        ax.text(0.1, 0.7, f"Source: {source}", fontsize=10, transform=ax.transAxes)
        ax.text(0.1, 0.5, f"Arc SNR: {result['snr']:.2f}", fontsize=10, transform=ax.transAxes,
               color="green" if result["snr"] > 2.0 else "red")
        ax.text(0.1, 0.3, "Label: ____________________", fontsize=10, transform=ax.transAxes)
        ax.text(0.1, 0.1, "(clear arc / maybe / no arc)", fontsize=8, transform=ax.transAxes, style="italic")
        ax.axis("off")
        
        time.sleep(0.3)  # Rate limiting
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved visualization to {output_path}")
    plt.close()
    
    # Also save a summary CSV for tracking
    csv_path = output_path.replace(".pdf", "_labels.csv").replace(".png", "_labels.csv")
    with open(csv_path, "w") as f:
        f.write("idx,name,source,theta_e,arc_snr,label,notes\n")
        for i, (name, ra, dec, theta_e, source) in enumerate(samples):
            f.write(f"{i},{name},{source},{theta_e:.2f},,\n")
    logger.info(f"Saved label template to {csv_path}")


def create_html_review(anchors: list, output_dir: str, n_samples: int = 20):
    """Create an HTML page for easier human review."""
    
    np.random.seed(42)
    indices = np.random.choice(len(anchors), size=min(n_samples, len(anchors)), replace=False)
    samples = [anchors[i] for i in indices]
    
    os.makedirs(output_dir, exist_ok=True)
    
    html_parts = ["""
<!DOCTYPE html>
<html>
<head>
    <title>Arc Visibility Sanity Check - SLACS/BELLS</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .sample { 
            background: white; 
            margin: 20px 0; 
            padding: 15px; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .images { display: flex; gap: 10px; flex-wrap: wrap; }
        .images img { max-width: 300px; border: 1px solid #ddd; image-rendering: auto; }
        .info { margin-top: 10px; }
        .snr-pass { color: green; font-weight: bold; }
        .snr-fail { color: red; font-weight: bold; }
        .label-section { 
            margin-top: 10px; 
            padding: 10px; 
            background: #f0f0f0;
            border-radius: 4px;
        }
        input[type="radio"] { margin-right: 5px; }
        label { margin-right: 15px; }
    </style>
</head>
<body>
    <h1>Arc Visibility Sanity Check</h1>
    <p><strong>Instructions:</strong> For each lens, review the images and select whether you see a clear arc in the DR10 data.</p>
    <p><strong>Critical question:</strong> Is our SNR > 2.0 threshold correctly identifying real arcs, or is it picking up non-arc structure?</p>
    <hr>
"""]
    
    for i, (name, ra, dec, theta_e, source) in enumerate(samples):
        logger.info(f"Processing {name} ({i+1}/{n_samples})...")
        
        cutout = download_cutout(ra, dec)
        
        if cutout is None:
            continue
        
        # Get r-band and compute residual
        if cutout.ndim == 3:
            r_band = cutout[1]
        else:
            r_band = cutout
        
        result = compute_residual_and_snr(cutout, theta_e)
        
        # Save images with better quality
        fig, ax = plt.subplots(figsize=(6, 6))
        vmin, vmax = np.nanpercentile(r_band, [1, 99])
        ax.imshow(r_band, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{name} - r-band", fontsize=10)
        ax.axis("off")
        rband_path = Path(output_dir) / f"{name}_rband.png"
        plt.savefig(rband_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        
        fig, ax = plt.subplots(figsize=(6, 6))
        vmin, vmax = np.nanpercentile(result["residual"], [5, 95])
        ax.imshow(result["residual"], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest")
        h, w = r_band.shape
        cy, cx = h // 2, w // 2
        ax.add_patch(Circle((cx, cy), result["inner_r"], fill=False, color="lime", linewidth=2, linestyle="--"))
        ax.add_patch(Circle((cx, cy), result["outer_r"], fill=False, color="lime", linewidth=2, linestyle="--"))
        ax.set_title(f"Residual (SNR={result['snr']:.1f})", fontsize=10)
        ax.axis("off")
        residual_path = Path(output_dir) / f"{name}_residual.png"
        plt.savefig(residual_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        
        snr_class = "snr-pass" if result["snr"] > 2.0 else "snr-fail"
        
        # Legacy Survey viewer link
        ls_viewer_url = f"https://www.legacysurvey.org/viewer?ra={ra:.4f}&dec={dec:.4f}&layer=ls-dr10&zoom=16"
        
        html_parts.append(f"""
    <div class="sample" id="sample-{i}">
        <h3>#{i+1}: {name}</h3>
        <div class="images">
            <div>
                <p><strong>r-band (DR10)</strong></p>
                <img src="{name}_rband.png" alt="r-band">
            </div>
            <div>
                <p><strong>Residual + θ_E annulus</strong></p>
                <img src="{name}_residual.png" alt="Residual">
            </div>
        </div>
        <div class="info">
            <p><strong>Source:</strong> {source} | <strong>θ_E:</strong> {theta_e:.2f}" | 
               <strong>Arc SNR:</strong> <span class="{snr_class}">{result['snr']:.2f}</span></p>
            <p><a href="{ls_viewer_url}" target="_blank">🔍 Open in Legacy Survey Viewer</a></p>
        </div>
        <div class="label-section">
            <strong>Your assessment:</strong><br>
            <input type="radio" name="label_{i}" id="clear_{i}" value="clear">
            <label for="clear_{i}">Clear arc visible</label>
            <input type="radio" name="label_{i}" id="maybe_{i}" value="maybe">
            <label for="maybe_{i}">Maybe (uncertain)</label>
            <input type="radio" name="label_{i}" id="no_{i}" value="no">
            <label for="no_{i}">No arc visible</label>
        </div>
    </div>
""")
        time.sleep(0.3)
    
    html_parts.append("""
    <hr>
    <div style="margin-top: 20px;">
        <h2>Summary</h2>
        <p>After reviewing all samples, please count:</p>
        <ul>
            <li>How many "clear arc" with SNR > 2.0? (True Positives)</li>
            <li>How many "no arc" with SNR > 2.0? (False Positives - our metric is wrong)</li>
            <li>How many "clear arc" with SNR < 2.0? (False Negatives - threshold too strict)</li>
        </ul>
        <p><strong>Conclusion:</strong> If many "no arc" still pass SNR > 2.0, we need to revise the metric.</p>
    </div>
</body>
</html>
""")
    
    html_path = Path(output_dir) / "arc_visibility_review.html"
    with open(html_path, "w") as f:
        f.write("\n".join(html_parts))
    
    logger.info(f"Saved HTML review to {html_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Directory for output files")
    parser.add_argument("--format", choices=["pdf", "html", "both"], default="html")
    parser.add_argument("--n-samples", type=int, default=20)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.format in ("pdf", "both"):
        create_visualization_grid(
            SLACS_BELLS_ANCHORS, 
            str(Path(args.output_dir) / "arc_visibility_sanity_check.pdf"),
            args.n_samples
        )
    
    if args.format in ("html", "both"):
        create_html_review(
            SLACS_BELLS_ANCHORS,
            args.output_dir,
            args.n_samples
        )


if __name__ == "__main__":
    main()

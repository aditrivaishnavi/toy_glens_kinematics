"""
Diagnose the sim-to-real gap in our training data.

Measure quantitative differences between:
1. Real confirmed lenses in DR10
2. Our simulated training data

Key metrics to compare:
- Arc-to-galaxy flux ratio
- Arc surface brightness
- Arc width (FWHM)
- Arc visibility (SNR in annulus)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import requests
from io import BytesIO
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple
import os

LEGACY_FITS_URL = "https://www.legacysurvey.org/viewer/cutout.fits"
PIX_SCALE = 0.262  # arcsec/pixel


# Real lenses with known properties
REAL_LENSES = [
    {"name": "Horseshoe", "ra": 163.3217, "dec": 20.1917, "theta_e": 5.0},
    {"name": "SDSSJ0946+1006", "ra": 146.7083, "dec": 10.1117, "theta_e": 1.38},
    {"name": "SDSSJ0912+0029", "ra": 138.2042, "dec": 0.4889, "theta_e": 1.63},
    {"name": "SDSSJ1430+4105", "ra": 217.7167, "dec": 41.0931, "theta_e": 1.52},
    {"name": "SL2SJ1405+5243", "ra": 211.3292, "dec": 52.7292, "theta_e": 2.1},
]


def download_fits(ra, dec, size=150):
    """Download FITS cutout."""
    try:
        resp = requests.get(
            LEGACY_FITS_URL,
            params={"ra": ra, "dec": dec, "size": size, "layer": "ls-dr10", "bands": "grz"},
            timeout=30
        )
        if resp.status_code == 200:
            from astropy.io import fits
            with fits.open(BytesIO(resp.content)) as hdul:
                return hdul[0].data
    except:
        pass
    return None


def measure_real_lens_properties(cutout: np.ndarray, theta_e_arcsec: float) -> Dict:
    """Measure properties of a real lens from DR10 cutout."""
    if cutout.ndim == 3:
        # Use r-band
        img = cutout[1]
    else:
        img = cutout
    
    h, w = img.shape
    cy, cx = h // 2, w // 2
    
    # Radial profile
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    theta_e_pix = theta_e_arcsec / PIX_SCALE
    
    # Core flux (central 5 pixels = ~1.3")
    core_mask = r < 5
    core_flux = np.nansum(img[core_mask])
    
    # Arc region flux (theta_E ± 3 pixels)
    arc_mask = (r >= theta_e_pix - 3) & (r <= theta_e_pix + 3)
    arc_flux_total = np.nansum(img[arc_mask])
    arc_flux_mean = np.nanmean(img[arc_mask])
    
    # Background (outer region)
    bg_mask = r > theta_e_pix + 10
    bg_mean = np.nanmean(img[bg_mask])
    bg_std = np.nanstd(img[bg_mask])
    
    # Arc SNR
    arc_snr = (arc_flux_mean - bg_mean) / (bg_std + 1e-10)
    
    # Arc to galaxy ratio
    galaxy_flux = np.nansum(img[r < theta_e_pix + 10]) - core_flux
    arc_to_galaxy = arc_flux_total / (galaxy_flux + 1e-10)
    
    # Core to arc ratio (how much brighter is core than arc)
    core_mean = np.nanmean(img[core_mask])
    core_to_arc = core_mean / (arc_flux_mean + 1e-10)
    
    return {
        "core_flux": float(core_flux),
        "arc_flux_total": float(arc_flux_total),
        "arc_flux_mean": float(arc_flux_mean),
        "bg_mean": float(bg_mean),
        "bg_std": float(bg_std),
        "arc_snr": float(arc_snr),
        "arc_to_galaxy": float(arc_to_galaxy),
        "core_to_arc": float(core_to_arc),
    }


def generate_simulated_lens(theta_e_arcsec: float, size: int = 150, seed: int = 42) -> np.ndarray:
    """Generate simulated lens using our current method."""
    rng = np.random.RandomState(seed)
    
    theta_e_pix = theta_e_arcsec / PIX_SCALE
    
    # Lens galaxy (de Vaucouleurs)
    x = np.arange(size) - size / 2
    y = np.arange(size) - size / 2
    x_grid, y_grid = np.meshgrid(x, y)
    r = np.sqrt(x_grid**2 + y_grid**2) + 1e-10
    
    re_pix = rng.uniform(8, 15)
    bn = 7.67  # n=4
    galaxy = 5000 * np.exp(-bn * ((r / re_pix)**0.25 - 1))
    
    # Source (Sersic)
    source_size = 80
    x_src = np.arange(source_size) - source_size / 2
    y_src = np.arange(source_size) - source_size / 2
    x_src_grid, y_src_grid = np.meshgrid(x_src, y_src)
    r_src = np.sqrt(x_src_grid**2 + y_src_grid**2) + 1e-10
    
    re_src = rng.uniform(3, 6)
    n_src = rng.uniform(0.5, 2)
    bn_src = 1.9992 * n_src - 0.3271
    source = np.exp(-bn_src * ((r_src / re_src)**(1/n_src) - 1))
    source = source / source.sum()
    
    # Ray trace (simplified)
    lensed = ray_trace_simple(source, theta_e_pix, size)
    
    # Scale arc flux - THIS IS WHERE THE PROBLEM IS
    # What flux should we use?
    arc_flux_total = rng.uniform(100, 500)  # ADU - is this realistic?
    lensed = lensed * arc_flux_total
    
    # PSF blur
    psf_sigma = 1.5  # pixels (~0.4" FWHM)
    lensed = gaussian_filter(lensed, psf_sigma)
    
    # Combine
    combined = galaxy + lensed
    
    # Add noise
    noise = rng.normal(0, 10, combined.shape)
    combined = combined + noise
    
    return combined


def ray_trace_simple(source, theta_e_pix, output_size):
    """Simplified ray tracing."""
    src_size = source.shape[0]
    
    y_img, x_img = np.mgrid[:output_size, :output_size]
    x_img = (x_img - output_size/2).astype(np.float32)
    y_img = (y_img - output_size/2).astype(np.float32)
    
    r = np.sqrt(x_img**2 + y_img**2) + 1e-10
    
    # SIS deflection
    alpha = theta_e_pix
    alpha_x = alpha * x_img / r
    alpha_y = alpha * y_img / r
    
    x_src = x_img - alpha_x
    y_src = y_img - alpha_y
    
    # Map to source pixels
    x_src_pix = (x_src + src_size/2).astype(int)
    y_src_pix = (y_src + src_size/2).astype(int)
    
    valid = (x_src_pix >= 0) & (x_src_pix < src_size) & (y_src_pix >= 0) & (y_src_pix < src_size)
    
    lensed = np.zeros((output_size, output_size))
    lensed[valid] = source[y_src_pix[valid], x_src_pix[valid]]
    
    return lensed


def compare_real_vs_simulated(output_dir):
    """Compare properties of real vs simulated lenses."""
    os.makedirs(output_dir, exist_ok=True)
    
    real_props = []
    sim_props = []
    
    print("Measuring real lens properties...")
    for lens in REAL_LENSES:
        cutout = download_fits(lens["ra"], lens["dec"])
        if cutout is not None:
            props = measure_real_lens_properties(cutout, lens["theta_e"])
            props["name"] = lens["name"]
            props["theta_e"] = lens["theta_e"]
            real_props.append(props)
            print(f"  {lens['name']}: arc_SNR={props['arc_snr']:.1f}, core_to_arc={props['core_to_arc']:.1f}")
    
    print("\nMeasuring simulated lens properties...")
    for i, theta_e in enumerate([1.0, 1.3, 1.5, 1.8, 2.0]):
        sim = generate_simulated_lens(theta_e, seed=42 + i)
        # Wrap in 3D array to match FITS format (sim is already 2D)
        sim_3d = np.stack([sim, sim, sim], axis=0)  # Fake grz bands
        props = measure_real_lens_properties(sim_3d, theta_e)
        props["name"] = f"Sim_θE={theta_e}"
        props["theta_e"] = theta_e
        sim_props.append(props)
        print(f"  θ_E={theta_e}: arc_SNR={props['arc_snr']:.1f}, core_to_arc={props['core_to_arc']:.1f}")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Arc SNR comparison
    ax = axes[0, 0]
    real_snr = [p["arc_snr"] for p in real_props]
    sim_snr = [p["arc_snr"] for p in sim_props]
    ax.bar([0], [np.mean(real_snr)], yerr=[np.std(real_snr)], label="Real DR10", alpha=0.7)
    ax.bar([1], [np.mean(sim_snr)], yerr=[np.std(sim_snr)], label="Simulated", alpha=0.7)
    ax.set_ylabel("Arc SNR")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Real", "Simulated"])
    ax.legend()
    ax.set_title("Arc Signal-to-Noise Ratio")
    
    # Core to arc ratio
    ax = axes[0, 1]
    real_ratio = [p["core_to_arc"] for p in real_props]
    sim_ratio = [p["core_to_arc"] for p in sim_props]
    ax.bar([0], [np.mean(real_ratio)], yerr=[np.std(real_ratio)], label="Real DR10", alpha=0.7)
    ax.bar([1], [np.mean(sim_ratio)], yerr=[np.std(sim_ratio)], label="Simulated", alpha=0.7)
    ax.set_ylabel("Core / Arc Brightness Ratio")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Real", "Simulated"])
    ax.set_title("Core-to-Arc Brightness Ratio\n(Higher = fainter arcs)")
    
    # Arc to galaxy flux
    ax = axes[1, 0]
    real_frac = [p["arc_to_galaxy"] for p in real_props]
    sim_frac = [p["arc_to_galaxy"] for p in sim_props]
    ax.bar([0], [np.mean(real_frac)], yerr=[np.std(real_frac)], label="Real DR10", alpha=0.7)
    ax.bar([1], [np.mean(sim_frac)], yerr=[np.std(sim_frac)], label="Simulated", alpha=0.7)
    ax.set_ylabel("Arc / Galaxy Flux Ratio")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Real", "Simulated"])
    ax.set_title("Arc-to-Galaxy Flux Ratio")
    
    # Summary text
    ax = axes[1, 1]
    ax.axis("off")
    summary = """
KEY FINDINGS:

Real lenses in DR10:
• Arc SNR: {:.1f} ± {:.1f}
• Core/Arc ratio: {:.1f} ± {:.1f}
• Arc/Galaxy: {:.3f} ± {:.3f}

Simulated lenses:
• Arc SNR: {:.1f} ± {:.1f}
• Core/Arc ratio: {:.1f} ± {:.1f}
• Arc/Galaxy: {:.3f} ± {:.3f}

DIAGNOSIS:
{}
""".format(
        np.mean(real_snr), np.std(real_snr),
        np.mean(real_ratio), np.std(real_ratio),
        np.mean(real_frac), np.std(real_frac),
        np.mean(sim_snr), np.std(sim_snr),
        np.mean(sim_ratio), np.std(sim_ratio),
        np.mean(sim_frac), np.std(sim_frac),
        get_diagnosis(real_props, sim_props)
    )
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
           verticalalignment="top", fontfamily="monospace")
    
    plt.suptitle("Sim-to-Real Gap Diagnosis", fontsize=14)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "sim_real_gap_diagnosis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved to {output_path}")
    plt.close()
    
    return real_props, sim_props


def get_diagnosis(real_props, sim_props):
    """Generate diagnosis text."""
    real_snr = np.mean([p["arc_snr"] for p in real_props])
    sim_snr = np.mean([p["arc_snr"] for p in sim_props])
    
    real_ratio = np.mean([p["core_to_arc"] for p in real_props])
    sim_ratio = np.mean([p["core_to_arc"] for p in sim_props])
    
    issues = []
    
    if sim_snr > real_snr * 1.5:
        issues.append("• Simulated arcs are TOO BRIGHT")
    elif sim_snr < real_snr * 0.5:
        issues.append("• Simulated arcs are too faint")
    
    if sim_ratio < real_ratio * 0.5:
        issues.append("• Core/arc ratio too low (arcs too prominent)")
    
    if not issues:
        issues.append("• Properties are similar")
    
    return "\n".join(issues)


if __name__ == "__main__":
    compare_real_vs_simulated("dark_halo_scope/planb/evaluation/sanity_check")

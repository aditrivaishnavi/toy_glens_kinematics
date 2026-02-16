"""
Show REAL training data arcs using the actual lensing physics from spark_gen7_injection.py
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
from typing import Tuple
import os


def sersic_profile(x, y, re, n, q, pa, I0=1.0):
    """Sersic profile."""
    bn = 1.9992 * n - 0.3271
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot = x * cos_pa + y * sin_pa
    y_rot = -x * sin_pa + y * cos_pa
    r = np.sqrt(x_rot**2 + (y_rot / q)**2) + 1e-10
    return I0 * np.exp(-bn * ((r / re)**(1/n) - 1))


def generate_source(size=80, seed=42):
    """Generate realistic source galaxy (Sersic + clumps)."""
    rng = np.random.RandomState(seed)
    
    x = np.arange(size) - size / 2
    y = np.arange(size) - size / 2
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Sersic base
    re_pix = rng.uniform(3, 8)  # Effective radius in pixels
    n_sersic = rng.uniform(0.5, 2.0)  # Sersic index
    q = rng.uniform(0.4, 0.9)  # Axis ratio
    pa = rng.uniform(0, np.pi)  # Position angle
    
    base = sersic_profile(x_grid, y_grid, re_pix, n_sersic, q, pa)
    
    # Add clumps (star-forming regions)
    n_clumps = rng.randint(0, 5)
    clumps = np.zeros_like(base)
    for _ in range(n_clumps):
        sigma_clump = rng.uniform(1, 3)
        flux = rng.uniform(0.05, 0.2)
        cx = size/2 + rng.normal(0, re_pix * 1.5)
        cy = size/2 + rng.normal(0, re_pix * 1.5)
        gauss = np.exp(-0.5 * ((x_grid - cx)**2 + (y_grid - cy)**2) / sigma_clump**2)
        clumps += flux * gauss
    
    result = base + clumps
    result = result / result.sum()  # Normalize
    
    return result, {"re_pix": re_pix, "n_sersic": n_sersic, "n_clumps": n_clumps}


def deflection_sie(x, y, theta_e, q=0.8):
    """SIE deflection field - actual lensing physics."""
    r = np.sqrt(x**2 + y**2) + 1e-10
    
    if q < 0.99:
        alpha = theta_e * np.sqrt(q) / np.sqrt(1 - q**2)
        alpha_x = alpha * np.arctan(np.sqrt(1 - q**2) * x / (r + 1e-10))
        alpha_y = alpha * np.arctanh(np.sqrt(1 - q**2) * y / (r + 1e-10))
    else:
        alpha_x = theta_e * x / (r + 1e-10)
        alpha_y = theta_e * y / (r + 1e-10)
    
    return alpha_x, alpha_y


def ray_trace(source, theta_e, q_lens=0.8, source_offset=(0, 0)):
    """Ray-trace source through lens - actual lensing physics."""
    size = source.shape[0]
    
    # Image plane coordinates
    y_img, x_img = np.mgrid[:size, :size]
    x_img = (x_img - size/2).astype(np.float32)
    y_img = (y_img - size/2).astype(np.float32)
    
    # Deflection
    alpha_x, alpha_y = deflection_sie(x_img, y_img, theta_e, q_lens)
    
    # Source plane coordinates
    x_src = x_img - alpha_x + source_offset[0]
    y_src = y_img - alpha_y + source_offset[1]
    
    # Map back to pixel coords
    x_src_pix = x_src + size/2
    y_src_pix = y_src + size/2
    
    # Bilinear interpolation
    x0 = np.floor(x_src_pix).astype(int)
    y0 = np.floor(y_src_pix).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    
    wx = x_src_pix - x0
    wy = y_src_pix - y0
    
    x0 = np.clip(x0, 0, size - 1)
    x1 = np.clip(x1, 0, size - 1)
    y0 = np.clip(y0, 0, size - 1)
    y1 = np.clip(y1, 0, size - 1)
    
    lensed = (
        (1 - wx) * (1 - wy) * source[y0, x0] +
        wx * (1 - wy) * source[y0, x1] +
        (1 - wx) * wy * source[y1, x0] +
        wx * wy * source[y1, x1]
    )
    
    return lensed.astype(np.float32)


def generate_lens_galaxy(size=80, seed=42):
    """Generate lens galaxy (elliptical)."""
    rng = np.random.RandomState(seed)
    
    x = np.arange(size) - size / 2
    y = np.arange(size) - size / 2
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Elliptical galaxy: de Vaucouleurs (n=4) or similar
    re_pix = rng.uniform(8, 15)  # Larger than source
    n_sersic = rng.uniform(3, 5)  # More concentrated
    q = rng.uniform(0.6, 0.95)
    pa = rng.uniform(0, np.pi)
    
    galaxy = sersic_profile(x_grid, y_grid, re_pix, n_sersic, q, pa, I0=5000)
    
    return galaxy


def create_real_arc_examples(output_dir):
    """Create examples showing REAL training data generation."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(18, 15))
    
    # Different theta_E values to show
    theta_e_values = [0.7, 1.0, 1.5, 2.0]  # arcsec
    pix_scale = 0.262  # arcsec/pixel
    
    for row, theta_e_arcsec in enumerate(theta_e_values):
        theta_e_pix = theta_e_arcsec / pix_scale
        
        # Use different seeds for variety
        source, src_meta = generate_source(size=80, seed=42 + row)
        lens_galaxy = generate_lens_galaxy(size=80, seed=100 + row)
        
        # Ray trace with small offset (more realistic)
        offset = (np.random.uniform(-2, 2), np.random.uniform(-2, 2))
        lensed = ray_trace(source, theta_e_pix, source_offset=offset)
        
        # Apply PSF blur
        psf_sigma = 1.5  # ~0.4" FWHM
        lensed_blurred = gaussian_filter(lensed, psf_sigma)
        
        # Scale to realistic flux
        lensed_scaled = lensed_blurred * 500  # ADU
        
        # Combine
        combined = lens_galaxy + lensed_scaled
        
        # Add realistic noise
        noise = np.random.normal(0, 10, combined.shape)
        combined_noisy = combined + noise
        
        # Plot: Source | Lensed Arc | Lens Galaxy | Combined
        base_idx = row * 4
        
        ax = fig.add_subplot(4, 4, base_idx + 1)
        ax.imshow(source, origin="lower", cmap="viridis")
        ax.set_title(f"Source Galaxy\nre={src_meta['re_pix']:.1f}px, n={src_meta['n_sersic']:.1f}", fontsize=9)
        ax.axis("off")
        
        ax = fig.add_subplot(4, 4, base_idx + 2)
        vmax = np.percentile(lensed_scaled, 99.5)
        ax.imshow(lensed_scaled, origin="lower", cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(f"Lensed Arc (θ_E={theta_e_arcsec}\")\n+ PSF blur", fontsize=9)
        ax.axis("off")
        # Draw Einstein radius
        from matplotlib.patches import Circle
        circle = Circle((40, 40), theta_e_pix, fill=False, color="cyan", linewidth=1, linestyle="--")
        ax.add_patch(circle)
        
        ax = fig.add_subplot(4, 4, base_idx + 3)
        vmin, vmax = np.percentile(lens_galaxy, [1, 99])
        ax.imshow(lens_galaxy, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title("Lens Galaxy\n(elliptical, n~4)", fontsize=9)
        ax.axis("off")
        
        ax = fig.add_subplot(4, 4, base_idx + 4)
        vmin, vmax = np.percentile(combined_noisy, [1, 99])
        ax.imshow(combined_noisy, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"Combined + Noise\n(training input)", fontsize=9)
        ax.axis("off")
        circle = Circle((40, 40), theta_e_pix, fill=False, color="lime", linewidth=1, linestyle="--")
        ax.add_patch(circle)
    
    plt.suptitle("REAL Training Data: Proper Gravitational Lensing Physics\n" +
                 "Source → Ray-traced arc → Added to lens galaxy → PSF + noise",
                 fontsize=14, y=0.98)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "real_training_arcs.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {output_path}")
    plt.close()
    
    # Create side-by-side: Training arc vs DR10 reality
    create_training_vs_reality(output_dir)


def create_training_vs_reality(output_dir):
    """Show training arcs vs what we see in DR10."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Training data (what model learns)
    for col, theta_e in enumerate([0.8, 1.2, 1.8]):
        theta_e_pix = theta_e / 0.262
        source, _ = generate_source(size=80, seed=42 + col)
        lens_galaxy = generate_lens_galaxy(size=80, seed=100 + col)
        lensed = ray_trace(source, theta_e_pix, source_offset=(1, 1))
        lensed = gaussian_filter(lensed, 1.5) * 500
        combined = lens_galaxy + lensed + np.random.normal(0, 10, lens_galaxy.shape)
        
        ax = axes[0, col]
        vmin, vmax = np.percentile(combined, [1, 99])
        ax.imshow(combined, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        from matplotlib.patches import Circle
        circle = Circle((40, 40), theta_e_pix, fill=False, color="lime", linewidth=2, linestyle="--")
        ax.add_patch(circle)
        ax.set_title(f"Training: θ_E={theta_e}\"", fontsize=11)
        ax.axis("off")
        if col == 0:
            ax.text(-0.15, 0.5, "Training\n(synthetic)", transform=ax.transAxes, 
                   fontsize=12, va="center", ha="center", rotation=90, weight="bold")
    
    # Row 2: DR10 reality (placeholder text)
    for col in range(3):
        ax = axes[1, col]
        ax.text(0.5, 0.6, "DR10 Reality:", fontsize=12, ha="center", transform=ax.transAxes, weight="bold")
        ax.text(0.5, 0.4, 
                "• Arc blurred by ~1\" seeing\n"
                "• Source flux ~10-100x fainter\n"
                "• Mixed with galaxy light\n"
                "• Often invisible to eye",
                fontsize=10, ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        if col == 0:
            ax.text(-0.15, 0.5, "Real DR10\n(ground-based)", transform=ax.transAxes, 
                   fontsize=12, va="center", ha="center", rotation=90, weight="bold", color="red")
    
    plt.suptitle("Training Data vs Reality: The Sim-to-Real Gap", fontsize=14, y=0.95)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "training_vs_reality.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    create_real_arc_examples("dark_halo_scope/planb/evaluation/sanity_check")

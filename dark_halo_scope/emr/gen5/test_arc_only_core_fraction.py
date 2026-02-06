#!/usr/bin/env python3
"""
Test A: Arc-only core fraction test.

This is the decisive test to separate "physics" vs "mismatch".

For each sample:
1. Re-render the arc (add_b) from stored injection parameters
2. Measure core_frac_arc_only = sum(add_b[core]) / sum(add_b[all])
3. Measure core_frac_diff = sum((stamp-ctrl)[core]) / sum((stamp-ctrl)[all])
4. Compare: if they match, it's physics. If arc-only is tiny but diff is large, it's mismatch.
"""

import boto3
import io
import math
import numpy as np
import pyarrow.parquet as pq

# Constants from pipeline
PIX_SCALE_ARCSEC = 0.262


def deflection_sis(x, y, theta_e, eps=1e-12):
    """SIS deflection in arcsec."""
    r = np.sqrt(x * x + y * y) + eps
    ax = theta_e * x / r
    ay = theta_e * y / r
    return ax, ay


def sersic_profile_Ie1(x, y, reff, q, phi_rad, n=1.0):
    """Sersic profile with I(Re) = 1."""
    c = math.cos(phi_rad)
    s = math.sin(phi_rad)
    xp = c * x + s * y
    yp = -s * x + c * y
    
    q = max(q, 0.1)
    r_ell = np.sqrt(xp * xp + (yp / q) ** 2)
    
    # Sersic bn approximation
    bn = 1.9992 * n - 0.3271
    
    return np.exp(-bn * ((r_ell / (reff + 1e-12)) ** (1.0 / n) - 1.0))


def _gaussian_kernel2d(sigma_pix):
    """Build normalized Gaussian kernel."""
    radius = int(max(3, math.ceil(4.0 * sigma_pix)))
    max_radius = 31
    radius = min(radius, max_radius)
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    k = np.exp(-0.5 * (xx**2 + yy**2) / sigma_pix**2).astype(np.float32)
    k /= np.sum(k)
    return k


def _fft_convolve2d_correct(img, kernel):
    """CORRECT FFT convolution with proper kernel centering."""
    ih, iw = img.shape
    kh, kw = kernel.shape
    pad = np.zeros((ih, iw), dtype=np.float32)
    pad[:kh, :kw] = kernel.astype(np.float32)
    # Roll to move kernel center to (0,0)
    oy, ox = kh // 2, kw // 2
    pad = np.roll(np.roll(pad, -oy, axis=0), -ox, axis=1)
    out = np.fft.ifft2(np.fft.fft2(img.astype(np.float32)) * np.fft.fft2(pad)).real
    return out.astype(np.float32)


def render_arc_only(
    stamp_size,
    theta_e_arcsec,
    src_reff_arcsec,
    src_e,
    src_phi_rad,
    src_x_arcsec,
    src_y_arcsec,
    psf_fwhm_arcsec,
):
    """
    Re-render the lensed arc (add_b) from injection parameters.
    Uses SIS lens model and Sersic n=1 source.
    Returns the arc image WITHOUT the LRG.
    """
    half = stamp_size // 2
    pix_idx = np.arange(stamp_size) - half + 0.5
    y_grid, x_grid = np.meshgrid(pix_idx, pix_idx, indexing="ij")
    
    # Image plane coords in arcsec
    x = x_grid * PIX_SCALE_ARCSEC
    y = y_grid * PIX_SCALE_ARCSEC
    
    # SIS deflection
    ax, ay = deflection_sis(x, y, theta_e_arcsec)
    
    # Source plane coords
    beta_x = x - ax - src_x_arcsec
    beta_y = y - ay - src_y_arcsec
    
    # Source axis ratio
    q_src = (1.0 - src_e) / (1.0 + src_e + 1e-6)
    q_src = max(q_src, 0.1)
    
    # Evaluate Sersic profile at source plane
    base = sersic_profile_Ie1(beta_x, beta_y, src_reff_arcsec, q_src, src_phi_rad, n=1.0)
    
    # Scale by pixel area (simplified - just for relative comparison)
    pix_area = PIX_SCALE_ARCSEC ** 2
    img = base * pix_area
    
    # PSF convolution (using CORRECT implementation)
    if psf_fwhm_arcsec > 0:
        psf_fwhm_pix = psf_fwhm_arcsec / PIX_SCALE_ARCSEC
        sigma_pix = psf_fwhm_pix / 2.355
        k = _gaussian_kernel2d(sigma_pix)
        img = _fft_convolve2d_correct(img, k)
    
    return img.astype(np.float32)


def compute_core_fraction(img, core_radius=5):
    """Compute fraction of flux in central core."""
    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    core_mask = r < core_radius
    total_flux = np.sum(np.abs(img))
    core_flux = np.sum(np.abs(img[core_mask]))
    
    if total_flux > 0:
        return core_flux / total_flux
    return 0.0


def main():
    print("=" * 70)
    print("TEST A: ARC-ONLY CORE FRACTION")
    print("Decisive test to separate physics vs mismatch")
    print("=" * 70)
    print()
    
    s3 = boto3.client("s3", region_name="us-east-2")
    
    # Load data
    resp = s3.list_objects_v2(
        Bucket="darkhaloscope",
        Prefix="phase4_pipeline/phase4c/v5_cosmos_paired/train/",
        MaxKeys=5
    )
    files = [f for f in resp["Contents"] if f["Key"].endswith(".parquet")]
    
    all_results = []
    
    for file_info in files[:3]:
        obj = s3.get_object(Bucket="darkhaloscope", Key=file_info["Key"])
        table = pq.read_table(io.BytesIO(obj["Body"].read()))
        df = table.to_pandas()
        
        # Exclude quarantined bricks
        df = df[~df["brickname"].isin(["0460m800", "3252p267"])]
        
        # Sample up to 100 per file
        df_sample = df.sample(min(100, len(df)), random_state=42)
        
        for idx, row in df_sample.iterrows():
            try:
                # Load stamp and ctrl
                stamp_data = np.load(io.BytesIO(row["stamp_npz"]))
                ctrl_data = np.load(io.BytesIO(row["ctrl_stamp_npz"]))
                
                stamp_r = stamp_data["image_r"]
                ctrl_r = ctrl_data["image_r"]
                diff = stamp_r - ctrl_r
                
                # Get injection parameters
                theta_e = row.get("theta_e_arcsec", 1.5)
                src_reff = row.get("src_reff_arcsec", 0.3)
                src_e = row.get("src_e", 0.3)
                src_phi = row.get("src_phi_rad", 0.0)
                src_x = row.get("src_x_arcsec", 0.0)
                src_y = row.get("src_y_arcsec", 0.0)
                psf_fwhm = row.get("psf_fwhm_used_r", 1.3)
                
                # Re-render arc only
                arc_only = render_arc_only(
                    stamp_size=64,
                    theta_e_arcsec=theta_e,
                    src_reff_arcsec=src_reff,
                    src_e=src_e,
                    src_phi_rad=src_phi,
                    src_x_arcsec=src_x,
                    src_y_arcsec=src_y,
                    psf_fwhm_arcsec=psf_fwhm,
                )
                
                # Compute core fractions
                core_frac_arc_only = compute_core_fraction(arc_only)
                core_frac_diff = compute_core_fraction(diff)
                
                all_results.append({
                    "theta_e": theta_e,
                    "core_frac_arc_only": core_frac_arc_only,
                    "core_frac_diff": core_frac_diff,
                    "ratio": core_frac_diff / (core_frac_arc_only + 1e-10),
                })
                
            except Exception as e:
                continue
    
    print(f"Processed {len(all_results)} samples")
    print()
    
    # Compute statistics
    arc_only_fracs = [r["core_frac_arc_only"] for r in all_results]
    diff_fracs = [r["core_frac_diff"] for r in all_results]
    ratios = [r["ratio"] for r in all_results]
    
    print("RESULTS:")
    print("-" * 50)
    print()
    print("Core fraction in ARC-ONLY (re-rendered):")
    print(f"  Mean:   {np.mean(arc_only_fracs):.4f}")
    print(f"  Median: {np.median(arc_only_fracs):.4f}")
    print(f"  Std:    {np.std(arc_only_fracs):.4f}")
    print()
    print("Core fraction in (STAMP - CTRL):")
    print(f"  Mean:   {np.mean(diff_fracs):.4f}")
    print(f"  Median: {np.median(diff_fracs):.4f}")
    print(f"  Std:    {np.std(diff_fracs):.4f}")
    print()
    print("Ratio (diff / arc_only):")
    print(f"  Mean:   {np.mean(ratios):.4f}")
    print(f"  Median: {np.median(ratios):.4f}")
    print()
    
    # Interpretation
    print("=" * 50)
    print("INTERPRETATION:")
    print("=" * 50)
    print()
    print("If core_frac_arc_only ≈ core_frac_diff:")
    print("  → Core leakage is PHYSICS (source + lens + PSF)")
    print()
    print("If core_frac_arc_only << core_frac_diff:")
    print("  → Core leakage is MISMATCH artifact")
    print()
    
    # Stratify by theta_E
    print()
    print("STRATIFIED BY THETA_E:")
    print("-" * 50)
    bins = [(0, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 10.0)]
    for low, high in bins:
        subset = [r for r in all_results if low <= r["theta_e"] < high]
        if len(subset) > 5:
            arc_mean = np.mean([r["core_frac_arc_only"] for r in subset])
            diff_mean = np.mean([r["core_frac_diff"] for r in subset])
            ratio_mean = np.mean([r["ratio"] for r in subset])
            print(f"theta_E [{low:.2f}, {high:.2f}):")
            print(f"  arc_only: {arc_mean:.4f}, diff: {diff_mean:.4f}, ratio: {ratio_mean:.2f}")


if __name__ == "__main__":
    main()

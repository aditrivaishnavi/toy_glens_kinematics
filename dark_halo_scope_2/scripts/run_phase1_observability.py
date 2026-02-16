"""
Run Phase 1 analytic observability calculations and produce
diagnostic plots.

This script is intended to be run from the repository root:

    python -m scripts.run_phase1_observability

It will:

1. Build the analytic (z_l, θ_E) detectability map.
2. Build the approximate (z_l, M_halo) detectability map.
3. Generate sensitivity analyses (seeing, source redshift).
4. Save the underlying arrays to NumPy .npz files.
5. Generate matplotlib figures for use in reports and the poster.

The figures are *diagnostic* for Phase 1; they will be refined once
we have empirical completeness curves from injection–recovery.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union, List
from dataclasses import replace

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src import (
    Phase1Config,
    SurveyModel,
    build_thetaE_analytic_window,
    build_mass_analytic_window,
)
from src.lens_models import sis_mass_inside_theta_E


def _ensure_output_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_thetaE_window(output_dir: Path, cmap="viridis") -> None:
    """
    Plot the analytic detectability map in (z_l, θ_E).
    """
    config = Phase1Config()
    survey = SurveyModel(config=config)
    result = build_thetaE_analytic_window(config=config)

    z = result.z_l_grid
    theta = result.theta_E_grid
    detect = result.detectability

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.pcolormesh(
        z,
        theta,
        detect.T,
        shading="auto",
        cmap=cmap,
        vmin=0,
        vmax=2,
    )

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Blind", "Low trust", "Good"])

    ax.set_xlabel("Lens redshift $z_l$", fontsize=12)
    ax.set_ylabel("Einstein radius $\\theta_E$ (arcsec)", fontsize=12)
    ax.set_title(
        "Analytic seeing-based detectability (DR10 r-band)\n"
        f"FWHM = {config.seeing_fwhm_r:.2f}\" (median), "
        f"Planck 2018 cosmology ($H_0$={config.h0}, $\\Omega_m$={config.omega_m})",
        fontsize=10
    )

    # Mark the detection thresholds
    theta_blind = config.k_blind * config.seeing_fwhm_r
    theta_good = config.k_good * config.seeing_fwhm_r
    
    ax.axhline(
        theta_blind,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Blind threshold ({config.k_blind}×FWHM = {theta_blind:.2f}\")",
    )
    ax.axhline(
        theta_good,
        color="white",
        linestyle="--",
        linewidth=1.5,
        label=f"Good threshold ({config.k_good}×FWHM = {theta_good:.2f}\")",
    )
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "phase1_thetaE_window.png", dpi=200)
    plt.close(fig)


def plot_mass_window(output_dir: Path, cmap="viridis") -> None:
    """
    Plot the approximate detectability map in (z_l, M_halo).
    """
    config = Phase1Config()
    result = build_mass_analytic_window(config=config)

    z = result.z_l_grid
    m = result.mass_grid
    detect = result.detectability

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.pcolormesh(
        z,
        np.log10(m),
        detect.T,
        shading="auto",
        cmap=cmap,
        vmin=0,
        vmax=2,
    )
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Blind", "Low trust", "Good"])

    ax.set_xlabel("Lens redshift $z_l$", fontsize=12)
    ax.set_ylabel("$\\log_{10}$ M($<\\theta_E$) [$M_\\odot$]", fontsize=12)
    ax.set_title(
        "Analytic observability window in halo mass\n"
        f"(source redshift $z_s$ = {config.representative_source_z:.1f})",
        fontsize=11
    )

    fig.tight_layout()
    fig.savefig(output_dir / "phase1_mass_window.png", dpi=200)
    plt.close(fig)


def plot_visibility_ratio(output_dir: Path) -> None:
    """
    Plot 1D detectability vs θ_E/FWHM with threshold markers.
    
    This clearly shows the analytic detection thresholds that will
    be tested with injection–recovery in later phases.
    """
    config = Phase1Config()
    
    # Create θ_E/FWHM ratio array
    ratio = np.linspace(0, 2.5, 500)
    
    # Classify each ratio
    detectability = np.zeros_like(ratio)
    detectability[ratio >= config.k_blind] = 1  # Low trust
    detectability[ratio >= config.k_good] = 2   # Good
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Fill regions with colors
    ax.fill_between(ratio, 0, 1, where=(ratio < config.k_blind),
                    color='#e74c3c', alpha=0.7, label='Blind')
    ax.fill_between(ratio, 0, 1, where=((ratio >= config.k_blind) & (ratio < config.k_good)),
                    color='#f39c12', alpha=0.7, label='Low Trust')
    ax.fill_between(ratio, 0, 1, where=(ratio >= config.k_good),
                    color='#27ae60', alpha=0.7, label='Good')
    
    # Mark thresholds with vertical lines
    ax.axvline(config.k_blind, color='darkred', linestyle='--', linewidth=2,
               label=f'$k_{{blind}}$ = {config.k_blind}')
    ax.axvline(config.k_good, color='darkgreen', linestyle='--', linewidth=2,
               label=f'$k_{{good}}$ = {config.k_good}')
    
    # Annotations
    ax.annotate('Arcs unresolved\nin PSF core', xy=(0.25, 0.5), fontsize=10,
                ha='center', va='center', color='white', weight='bold')
    ax.annotate('Marginal\ndetection', xy=(0.75, 0.5), fontsize=10,
                ha='center', va='center', color='black', weight='bold')
    ax.annotate('Ring/arc clearly\nresolvable', xy=(1.75, 0.5), fontsize=10,
                ha='center', va='center', color='white', weight='bold')
    
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel('$\\theta_E$ / FWHM', fontsize=14)
    ax.set_ylabel('Detectability Region', fontsize=12)
    ax.set_title(
        'Analytic Detection Thresholds\n'
        '(to be calibrated with injection–recovery)',
        fontsize=12
    )
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=10)
    
    # Add note about empirical calibration
    ax.text(0.02, 0.02, 
            'Note: These thresholds are physically motivated assumptions.\n'
            'Phases 4–6 will empirically calibrate the actual completeness curve.',
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            style='italic', color='gray')
    
    fig.tight_layout()
    fig.savefig(output_dir / "phase1_visibility_ratio.png", dpi=200)
    plt.close(fig)


def plot_seeing_sensitivity(output_dir: Path) -> None:
    """
    Show how detection thresholds shift with seeing FWHM.
    
    Demonstrates that the qualitative picture is robust but
    quantitative thresholds depend on actual seeing conditions.
    """
    config = Phase1Config()
    
    # Different FWHM values to explore
    fwhm_values = [1.0, 1.25, 1.5]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel: θ_E thresholds vs FWHM
    fwhm_range = np.linspace(0.8, 1.8, 100)
    theta_blind = config.k_blind * fwhm_range
    theta_good = config.k_good * fwhm_range
    
    ax1.fill_between(fwhm_range, 0, theta_blind, color='#e74c3c', alpha=0.3, label='Blind')
    ax1.fill_between(fwhm_range, theta_blind, theta_good, color='#f39c12', alpha=0.3, label='Low Trust')
    ax1.fill_between(fwhm_range, theta_good, 3.0, color='#27ae60', alpha=0.3, label='Good')
    
    ax1.plot(fwhm_range, theta_blind, 'r-', linewidth=2, label=f'$\\theta_{{blind}}$ = {config.k_blind}×FWHM')
    ax1.plot(fwhm_range, theta_good, 'g-', linewidth=2, label=f'$\\theta_{{good}}$ = {config.k_good}×FWHM')
    
    # Mark specific FWHM values
    for fwhm, color in zip(fwhm_values, colors):
        ax1.axvline(fwhm, color=color, linestyle=':', alpha=0.7)
        ax1.scatter([fwhm], [config.k_blind * fwhm], color=color, s=50, zorder=5)
        ax1.scatter([fwhm], [config.k_good * fwhm], color=color, s=50, zorder=5)
    
    ax1.set_xlabel('Seeing FWHM (arcsec)', fontsize=12)
    ax1.set_ylabel('$\\theta_E$ threshold (arcsec)', fontsize=12)
    ax1.set_title('Detection Thresholds vs Seeing', fontsize=12)
    ax1.set_xlim(0.8, 1.8)
    ax1.set_ylim(0, 2.5)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Table of values
    ax2.axis('off')
    
    # Create table data
    table_data = [
        ['FWHM', '$\\theta_{blind}$', '$\\theta_{good}$', 'Blind Range', 'Good Range'],
    ]
    for fwhm in fwhm_values:
        t_blind = config.k_blind * fwhm
        t_good = config.k_good * fwhm
        table_data.append([
            f'{fwhm:.2f}"',
            f'{t_blind:.2f}"',
            f'{t_good:.2f}"',
            f'$\\theta_E$ < {t_blind:.2f}"',
            f'$\\theta_E$ ≥ {t_good:.2f}"'
        ])
    
    table = ax2.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center',
        colColours=['#ecf0f1']*5
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    ax2.set_title(
        'Threshold Values at Different Seeing Conditions\n'
        f'($k_{{blind}}$ = {config.k_blind}, $k_{{good}}$ = {config.k_good})',
        fontsize=12, pad=20
    )
    
    # Add note
    fig.text(0.5, 0.02,
             'DR10 seeing varies from ~1.0" (good conditions) to ~1.5" (poor conditions). '
             'Median r-band FWHM ≈ 1.25".',
             ha='center', fontsize=9, style='italic', color='gray')
    
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_dir / "phase1_seeing_sensitivity.png", dpi=200)
    plt.close(fig)


def plot_mass_zs_comparison(output_dir: Path) -> None:
    """
    Show how the mass window shifts with source redshift.
    
    Demonstrates that the qualitative conclusion (only high-mass
    halos are visible) is robust across plausible z_s values.
    """
    config = Phase1Config()
    
    # Source redshifts to compare
    zs_values = [1.5, 2.0, 2.5]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for ax, z_s in zip(axes, zs_values):
        # Create config with this z_s
        config_zs = replace(config, representative_source_z=z_s)
        result = build_mass_analytic_window(config=config_zs)
        
        z = result.z_l_grid
        m = result.mass_grid
        detect = result.detectability
        
        # Mask out unphysical region (z_l >= z_s)
        z_mesh, m_mesh = np.meshgrid(z, np.log10(m))
        mask = z_mesh.T >= z_s
        detect_masked = np.ma.masked_where(mask, detect)
        
        im = ax.pcolormesh(
            z,
            np.log10(m),
            detect_masked.T,
            shading="auto",
            cmap="viridis",
            vmin=0,
            vmax=2,
        )
        
        # Mark the z_l = z_s boundary
        ax.axvline(z_s, color='white', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(z_s - 0.02, 12.5, f'$z_l = z_s$', color='white', fontsize=9,
                rotation=90, va='top', ha='right')
        
        ax.set_xlabel('Lens redshift $z_l$', fontsize=11)
        ax.set_title(f'$z_s$ = {z_s}', fontsize=12)
        ax.set_xlim(0.1, 1.0)
        
    axes[0].set_ylabel('$\\log_{10}$ M($<\\theta_E$) [$M_\\odot$]', fontsize=11)
    
    # Single colorbar
    cbar = fig.colorbar(im, ax=axes, ticks=[0, 1, 2], shrink=0.8)
    cbar.ax.set_yticklabels(["Blind", "Low trust", "Good"])
    
    fig.suptitle(
        'Mass Observability Window: Sensitivity to Source Redshift\n'
        'The "high-mass only" conclusion is robust across $z_s$ = 1.5–2.5',
        fontsize=12, y=1.02
    )
    
    fig.tight_layout()
    fig.savefig(output_dir / "phase1_mass_zs_comparison.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_grids(output_dir: Path) -> None:
    """
    Save the analytic grids to disk for later phases.
    """
    config = Phase1Config()
    theta_res = build_thetaE_analytic_window(config=config)
    mass_res = build_mass_analytic_window(config=config)

    np.savez(
        output_dir / "phase1_thetaE_window.npz",
        z_l_grid=theta_res.z_l_grid,
        theta_E_grid=theta_res.theta_E_grid,
        detectability=theta_res.detectability,
        meta=dict(
            description="Analytic seeing-based detectability map in (z_l, theta_E).",
            k_blind=config.k_blind,
            k_good=config.k_good,
            seeing_fwhm_r=config.seeing_fwhm_r,
            h0=config.h0,
            omega_m=config.omega_m,
        ),
    )

    np.savez(
        output_dir / "phase1_mass_window.npz",
        z_l_grid=mass_res.z_l_grid,
        mass_grid=mass_res.mass_grid,
        detectability=mass_res.detectability,
        meta=dict(
            description=(
                "Approximate detectability map in (z_l, M(<theta_E)), "
                f"assuming source redshift z_s={config.representative_source_z:.1f}."
            ),
            h0=config.h0,
            omega_m=config.omega_m,
        ),
    )


def main() -> None:
    output_dir = _ensure_output_dir("outputs/phase1")
    
    print("Generating Phase 1 analytic observability outputs...")
    print(f"  Using Planck 2018 cosmology: H0={Phase1Config().h0}, Ωm={Phase1Config().omega_m}")
    
    print("  [1/6] θ_E detectability window...")
    plot_thetaE_window(output_dir)
    
    print("  [2/6] Mass detectability window...")
    plot_mass_window(output_dir)
    
    print("  [3/6] Visibility ratio plot (1D threshold diagram)...")
    plot_visibility_ratio(output_dir)
    
    print("  [4/6] Seeing sensitivity analysis...")
    plot_seeing_sensitivity(output_dir)
    
    print("  [5/6] Source redshift sensitivity (z_s = 1.5, 2.0, 2.5)...")
    plot_mass_zs_comparison(output_dir)
    
    print("  [6/6] Saving data grids...")
    save_grids(output_dir)
    
    print(f"\n✓ Phase 1 outputs written to {output_dir.resolve()}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()

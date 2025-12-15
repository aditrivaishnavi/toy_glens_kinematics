#!/usr/bin/env python3
"""
Visualize Phase 2 LRG Hypergrid Results

Generates:
1. Sky map of LRG density (all variants)
2. Density histograms comparing variants
3. High-density region locations
4. Purity vs completeness plot
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use a nice style
plt.style.use('seaborn-v0_8-darkgrid')


def load_data(results_dir: Path):
    """Load all variant data."""
    variants = ['v1_pure_massive', 'v2_baseline_dr10', 'v3_color_relaxed', 
                'v4_mag_relaxed', 'v5_very_relaxed']
    
    data = {}
    for v in variants:
        merged_path = results_dir / v / 'phase2_hypergrid_bricks_merged.csv'
        regions_path = results_dir / v / 'phase2_regions_summary.csv'
        
        if merged_path.exists():
            data[v] = {
                'merged': pd.read_csv(merged_path),
                'regions': pd.read_csv(regions_path) if regions_path.exists() else None
            }
            print(f"[VIZ] Loaded {v}: {len(data[v]['merged'])} bricks")
    
    return data


def plot_sky_map(data: dict, output_dir: Path):
    """Plot sky map of LRG density for each variant."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), 
                              subplot_kw={'projection': 'mollweide'})
    axes = axes.flatten()
    
    variants = list(data.keys())
    
    for i, variant in enumerate(variants):
        ax = axes[i]
        df = data[variant]['merged']
        
        # Convert RA/Dec to radians for Mollweide projection
        # Mollweide expects longitude in [-pi, pi], latitude in [-pi/2, pi/2]
        ra = df['ra'].values
        dec = df['dec'].values
        
        # Shift RA to [-180, 180] range
        ra_shifted = np.where(ra > 180, ra - 360, ra)
        ra_rad = np.radians(ra_shifted)
        dec_rad = np.radians(dec)
        
        # Get density column
        dens_col = f'lrg_density_{variant}'
        if dens_col not in df.columns:
            continue
            
        density = df[dens_col].values
        
        # Use log scale for better visibility
        density_log = np.log10(density + 1)
        
        # Scatter plot
        sc = ax.scatter(ra_rad, dec_rad, c=density_log, s=0.1, 
                       cmap='hot', alpha=0.7, rasterized=True)
        
        ax.set_title(f'{variant}\nmedian: {np.median(density[density>0]):.0f} LRG/deg²', 
                    fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    axes[-1].axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(sc, ax=axes.tolist(), orientation='horizontal', 
                        fraction=0.05, pad=0.08, aspect=50)
    cbar.set_label('log₁₀(LRG density + 1) [LRG/deg²]', fontsize=12)
    
    plt.suptitle('DR10 South LRG Density Sky Maps\n5 Selection Variants', 
                fontsize=14, y=1.02)
    
    output_path = output_dir / 'phase2_sky_maps.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[VIZ] Saved sky map: {output_path}")


def plot_density_histograms(data: dict, output_dir: Path):
    """Plot density histograms comparing all variants."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    variants = list(data.keys())
    
    # Left: Linear scale (zoomed to show structure)
    ax1 = axes[0]
    for i, variant in enumerate(variants):
        df = data[variant]['merged']
        dens_col = f'lrg_density_{variant}'
        if dens_col not in df.columns:
            continue
        density = df[dens_col].values
        density_pos = density[density > 0]
        
        # Use percentile-based bins
        p99 = np.percentile(density_pos, 99)
        bins = np.linspace(0, p99 * 1.5, 100)
        
        ax1.hist(density_pos, bins=bins, alpha=0.5, label=variant, 
                color=colors[i], density=True)
    
    ax1.set_xlabel('LRG Density [LRG/deg²]')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('LRG Density Distribution (linear scale)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, 500)
    
    # Right: Log scale (full range)
    ax2 = axes[1]
    for i, variant in enumerate(variants):
        df = data[variant]['merged']
        dens_col = f'lrg_density_{variant}'
        if dens_col not in df.columns:
            continue
        density = df[dens_col].values
        density_pos = density[density > 0]
        
        bins = np.logspace(0, np.log10(density_pos.max()), 80)
        ax2.hist(density_pos, bins=bins, alpha=0.5, label=variant,
                color=colors[i], density=True)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('LRG Density [LRG/deg²]')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('LRG Density Distribution (log scale)')
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / 'phase2_density_histograms.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[VIZ] Saved density histograms: {output_path}")


def plot_high_density_map(data: dict, output_dir: Path):
    """Plot locations of high-density regions (top 1%)."""
    
    fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={'projection': 'mollweide'})
    
    # Use v1_pure_massive as the main sample
    variant = 'v1_pure_massive'
    df = data[variant]['merged']
    dens_col = f'lrg_density_{variant}'
    
    density = df[dens_col].values
    
    # Get p99 threshold
    density_pos = density[density > 0]
    p99 = np.percentile(density_pos, 99)
    p95 = np.percentile(density_pos, 95)
    p90 = np.percentile(density_pos, 90)
    
    # Prepare coordinates
    ra = df['ra'].values
    dec = df['dec'].values
    ra_shifted = np.where(ra > 180, ra - 360, ra)
    ra_rad = np.radians(ra_shifted)
    dec_rad = np.radians(dec)
    
    # Plot all bricks in gray
    ax.scatter(ra_rad, dec_rad, c='lightgray', s=0.05, alpha=0.3, 
              rasterized=True, label='All bricks')
    
    # Overlay high-density bricks
    mask_p90 = (density >= p90) & (density < p95)
    mask_p95 = (density >= p95) & (density < p99)
    mask_p99 = density >= p99
    
    ax.scatter(ra_rad[mask_p90], dec_rad[mask_p90], c='yellow', s=2, 
              alpha=0.8, label=f'p90-p95 ({mask_p90.sum()} bricks)')
    ax.scatter(ra_rad[mask_p95], dec_rad[mask_p95], c='orange', s=4,
              alpha=0.9, label=f'p95-p99 ({mask_p95.sum()} bricks)')
    ax.scatter(ra_rad[mask_p99], dec_rad[mask_p99], c='red', s=8, 
              alpha=1.0, label=f'p99+ ({mask_p99.sum()} bricks)')
    
    ax.set_title(f'High-Density LRG Regions (v1_pure_massive)\n'
                f'p99 threshold: {p99:.0f} LRG/deg²', fontsize=12)
    ax.legend(loc='lower right', fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.3)
    
    output_path = output_dir / 'phase2_high_density_map.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[VIZ] Saved high-density map: {output_path}")


def plot_variant_comparison(data: dict, output_dir: Path):
    """Plot purity vs completeness across variants."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    variants = list(data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Extract statistics
    stats = []
    for v in variants:
        df = data[v]['merged']
        dens_col = f'lrg_density_{v}'
        frac_col = f'lrg_frac_{v}'
        n_col = f'n_lrg_{v}'
        
        if dens_col not in df.columns:
            continue
            
        density = df[dens_col].values
        density_pos = density[density > 0]
        
        total_lrg = df[n_col].sum() if n_col in df.columns else 0
        total_gal = df['n_gal'].sum()
        
        stats.append({
            'variant': v,
            'median_density': np.median(density_pos),
            'p99_density': np.percentile(density_pos, 99),
            'total_lrg': total_lrg,
            'lrg_fraction': total_lrg / total_gal if total_gal > 0 else 0,
            'n_bricks_nonzero': (density > 0).sum()
        })
    
    stats_df = pd.DataFrame(stats)
    
    # Plot 1: Total LRG count vs median density
    ax1 = axes[0]
    for i, row in stats_df.iterrows():
        ax1.scatter(row['median_density'], row['total_lrg'], 
                   s=150, c=colors[i], label=row['variant'], zorder=3)
        ax1.annotate(row['variant'].replace('_', '\n'), 
                    (row['median_density'], row['total_lrg']),
                    textcoords='offset points', xytext=(10, 0), fontsize=8)
    
    ax1.set_xlabel('Median LRG Density [LRG/deg²]')
    ax1.set_ylabel('Total LRG Count')
    ax1.set_title('Completeness vs Density')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: LRG fraction (purity proxy)
    ax2 = axes[1]
    x_pos = np.arange(len(variants))
    bars = ax2.bar(x_pos, stats_df['lrg_fraction'] * 100, color=colors)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([v.replace('_', '\n') for v in variants], fontsize=8)
    ax2.set_ylabel('LRG Fraction (%)')
    ax2.set_title('LRG Fraction of All Galaxies')
    ax2.set_ylim(0, 12)
    
    for bar, val in zip(bars, stats_df['lrg_fraction']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val*100:.1f}%', ha='center', fontsize=9)
    
    # Plot 3: p99 density (extreme overdensity threshold)
    ax3 = axes[2]
    bars = ax3.bar(x_pos, stats_df['p99_density'], color=colors)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([v.replace('_', '\n') for v in variants], fontsize=8)
    ax3.set_ylabel('p99 Density [LRG/deg²]')
    ax3.set_title('High-Density Threshold (99th percentile)')
    
    for bar, val in zip(bars, stats_df['p99_density']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:.0f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / 'phase2_variant_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[VIZ] Saved variant comparison: {output_path}")


def plot_top_candidates(data: dict, output_dir: Path, n_top: int = 20):
    """Plot top lens candidates by density."""
    
    variant = 'v1_pure_massive'
    df = data[variant]['merged']
    dens_col = f'lrg_density_{variant}'
    
    # Sort by density and get top N
    top_df = df.nlargest(n_top, dens_col)[['brickname', 'ra', 'dec', dens_col, 'n_gal', 'ebv', 'psfsize_r']]
    top_df = top_df.reset_index(drop=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Table of top candidates
    ax1 = axes[0]
    ax1.axis('off')
    
    table_data = []
    for i, row in top_df.iterrows():
        table_data.append([
            i + 1,
            row['brickname'],
            f"{row['ra']:.2f}",
            f"{row['dec']:.2f}",
            f"{row[dens_col]:.0f}",
            f"{row['n_gal']:,}",
            f"{row['ebv']:.3f}"
        ])
    
    table = ax1.table(
        cellText=table_data,
        colLabels=['Rank', 'Brick', 'RA', 'Dec', 'Density', 'n_gal', 'E(B-V)'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax1.set_title(f'Top {n_top} Lens Candidates (v1_pure_massive)', fontsize=12, pad=20)
    
    # Right: Sky positions of top candidates
    ax2 = axes[1]
    
    # All bricks in gray
    ra = df['ra'].values
    dec = df['dec'].values
    ax2.scatter(ra, dec, c='lightgray', s=0.1, alpha=0.2, rasterized=True)
    
    # Top candidates
    sc = ax2.scatter(top_df['ra'], top_df['dec'], 
                    c=top_df[dens_col], s=100, cmap='Reds', 
                    edgecolors='black', linewidths=1, zorder=3)
    
    # Label top 5
    for i in range(min(5, len(top_df))):
        row = top_df.iloc[i]
        ax2.annotate(f"#{i+1}", (row['ra'], row['dec']),
                    textcoords='offset points', xytext=(8, 8),
                    fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('RA [deg]')
    ax2.set_ylabel('Dec [deg]')
    ax2.set_title('Sky Positions of Top Candidates')
    ax2.set_xlim(0, 360)
    ax2.set_ylim(-90, 35)
    
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label('LRG Density [LRG/deg²]')
    
    plt.tight_layout()
    output_path = output_dir / 'phase2_top_candidates.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[VIZ] Saved top candidates: {output_path}")
    
    # Also save as CSV
    csv_path = output_dir / 'phase2_top_candidates.csv'
    top_df.to_csv(csv_path, index=False)
    print(f"[VIZ] Saved top candidates CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Phase 2 results')
    parser.add_argument('--results-dir', type=Path, 
                       default=Path('results/phase2_analysis'),
                       help='Path to Phase 2 analysis results')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory for plots (default: results_dir/plots)')
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir or (args.results_dir / 'plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[VIZ] Loading data from: {args.results_dir}")
    data = load_data(args.results_dir)
    
    if not data:
        print("[VIZ] ERROR: No data found!")
        return
    
    print(f"\n[VIZ] Generating visualizations...")
    
    # Generate all plots
    plot_sky_map(data, output_dir)
    plot_density_histograms(data, output_dir)
    plot_high_density_map(data, output_dir)
    plot_variant_comparison(data, output_dir)
    plot_top_candidates(data, output_dir)
    
    print(f"\n[VIZ] ✓ All visualizations saved to: {output_dir}")
    print(f"\nGenerated files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()




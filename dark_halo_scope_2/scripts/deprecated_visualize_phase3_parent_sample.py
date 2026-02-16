#!/usr/bin/env python3
"""
Phase 3 visualization: parent LRG sample

This script produces diagnostic plots for the Phase 3 parent catalog:

- z-band magnitude distribution
- color-color diagram (r - z vs z - W1)
- sky map of LRGs in RA and Dec, colored by region_id
- bar chart of v3 LRG counts per region_id
- histogram of overlap with other LRG variants (v1, v2, v4, v5)

Supports multiple ranking modes from run_phase3_define_fields.py.
If --all-modes is set, processes all detected modes in the phase3 directory.

These plots are intended both as science sanity checks and as potential
figures for the research log and poster.

Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Visualize Phase 3 LRG parent sample"
    )

    parser.add_argument(
        "--parent-csv",
        default="",
        help=(
            "Path to phase3_lrg_parent_catalog.csv. "
            "If empty and --mode is set, constructed from --phase3-dir/--variant/--mode."
        ),
    )
    parser.add_argument(
        "--phase3-dir",
        default="results/phase3",
        help="Base Phase 3 output directory.",
    )
    parser.add_argument(
        "--variant",
        default="v3_color_relaxed",
        help="Variant subdirectory name (e.g., v3_color_relaxed).",
    )
    parser.add_argument(
        "--mode",
        default="",
        help=(
            "Ranking mode to visualize (e.g., area_weighted_v3, density_v3). "
            "If empty and --parent-csv is also empty, uses --all-modes behavior."
        ),
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="If set, process all detected ranking modes in phase3-dir/variant.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Directory where figures will be saved. "
            "If empty, figures are saved to {phase3-dir}/{variant}/{mode}/figures/."
        ),
    )

    return parser.parse_args(argv)


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_zmag_hist(df: pd.DataFrame, outdir: str, mode: str) -> None:
    if "z_mag" not in df.columns:
        print("Warning: z_mag not found, skipping z magnitude histogram.")
        return

    plt.figure(figsize=(6, 4))
    z = df["z_mag"].values
    z = z[np.isfinite(z)]
    plt.hist(z, bins=40, alpha=0.8, color="steelblue", edgecolor="white")
    plt.xlabel("z-band magnitude (AB)")
    plt.ylabel("Number of LRGs")
    title = f"Phase 3 v3 LRG parent sample: z magnitude"
    if mode:
        title += f"\n({mode})"
    plt.title(title)
    plt.grid(alpha=0.3)
    outpath = os.path.join(outdir, "phase3_zmag_hist.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved z magnitude histogram: {outpath}")


def plot_color_color(df: pd.DataFrame, outdir: str, mode: str) -> None:
    # First check for pre-computed colors
    if "r_minus_z" in df.columns and "z_minus_w1" in df.columns:
        r_z = df["r_minus_z"].values
        z_w1 = df["z_minus_w1"].values
    elif "r_mag" in df.columns and "z_mag" in df.columns and "w1_mag" in df.columns:
        r_z = df["r_mag"].values - df["z_mag"].values
        z_w1 = df["z_mag"].values - df["w1_mag"].values
    else:
        print("Warning: color columns not found, skipping color-color diagram.")
        return

    mask = np.isfinite(r_z) & np.isfinite(z_w1)

    if not np.any(mask):
        print("Warning: no finite color data, skipping color-color diagram.")
        return

    plt.figure(figsize=(6, 5))
    plt.scatter(r_z[mask], z_w1[mask], s=2, alpha=0.3, c="steelblue")
    
    # Overlay v3 selection cuts for reference
    # v3: r_minus_z > 0.4, z_minus_w1 > 0.8
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.axvline(0.4, color="red", linestyle="--", alpha=0.6, label="v3 cut: r-z > 0.4")
    plt.axhline(0.8, color="orange", linestyle="--", alpha=0.6, label="v3 cut: z-W1 > 0.8")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(loc="lower right", fontsize="small")
    
    plt.xlabel("r - z (mag)")
    plt.ylabel("z - W1 (mag)")
    title = f"Phase 3 v3 LRG parent sample: color-color"
    if mode:
        title += f"\n({mode})"
    plt.title(title)
    plt.grid(alpha=0.3)
    outpath = os.path.join(outdir, "phase3_color_color.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved color-color diagram: {outpath}")


def plot_sky_map(df: pd.DataFrame, outdir: str, mode: str) -> None:
    if "RA" not in df.columns or "DEC" not in df.columns:
        print("Warning: RA/DEC not found, skipping sky map.")
        return

    ra = df["RA"].values
    dec = df["DEC"].values

    if "region_id" in df.columns:
        region_ids = df["region_id"].values
    else:
        region_ids = None

    plt.figure(figsize=(7, 5))

    if region_ids is None:
        plt.scatter(ra, dec, s=2, alpha=0.3, c="steelblue")
    else:
        # Color by region_id using a simple integer colormap
        unique_regions = np.unique(region_ids[~pd.isna(region_ids)])
        # Map each region_id to an integer 0..N-1
        region_to_int = {rid: i for i, rid in enumerate(unique_regions)}
        region_idx = np.array([region_to_int.get(r, 0) for r in region_ids])
        cmap = plt.cm.get_cmap("tab10")
        plt.scatter(ra, dec, s=2, alpha=0.4, c=region_idx, cmap=cmap)
        
        # Create legend with correct colors
        handles = []
        labels = []
        n_regions = len(unique_regions)
        for rid, idx in region_to_int.items():
            color = cmap(idx / max(n_regions - 1, 1))
            handles.append(plt.Line2D([], [], marker="o", linestyle="", 
                                       markersize=6, color=color))
            labels.append(f"region {int(rid)}")
        plt.legend(handles, labels, loc="best", fontsize="small")

    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    title = f"Phase 3 v3 LRG parent sample: sky distribution"
    if mode:
        title += f"\n({mode})"
    plt.title(title)
    plt.grid(alpha=0.3)
    outpath = os.path.join(outdir, "phase3_sky_map.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved sky map: {outpath}")


def plot_counts_per_region(df: pd.DataFrame, outdir: str, mode: str) -> None:
    if "region_id" not in df.columns:
        print("Warning: region_id not found, skipping per-region counts plot.")
        return

    counts = df["region_id"].value_counts().sort_index()

    plt.figure(figsize=(max(6, 0.6 * len(counts)), 4))
    
    # Create labels with rank if available
    labels = []
    has_rank = "phase3_region_rank" in df.columns
    if has_rank:
        rank_map = df.groupby("region_id")["phase3_region_rank"].first().to_dict()
    
    for rid in counts.index:
        if has_rank and rid in rank_map:
            labels.append(f"{int(rid)}\n(rank {int(rank_map[rid])})")
        else:
            labels.append(str(int(rid)))
    
    plt.bar(labels, counts.values, color="steelblue", edgecolor="white")
    plt.xlabel("region_id")
    plt.ylabel("Number of v3 LRGs")
    title = f"Phase 3 v3 LRG counts per region"
    if mode:
        title += f"\n({mode})"
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    outpath = os.path.join(outdir, "phase3_counts_per_region.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved counts-per-region plot: {outpath}")


def plot_variant_overlap(df: pd.DataFrame, outdir: str, mode: str) -> None:
    """
    For v3 LRGs, show how many also satisfy v1, v2, v4, v5.

    This helps quantify how "massive" or "relaxed" the parent sample is.
    """
    # Handle both column naming conventions
    v1_col = v2_col = v4_col = v5_col = None
    
    # Check for short names (is_lrg_v1, is_lrg_v2, etc.)
    if all(c in df.columns for c in ["is_lrg_v1", "is_lrg_v2", "is_lrg_v3", "is_lrg_v4", "is_lrg_v5"]):
        v1_col, v2_col, v4_col, v5_col = "is_lrg_v1", "is_lrg_v2", "is_lrg_v4", "is_lrg_v5"
    # Check for long names
    elif all(c in df.columns for c in [
        "is_lrg_v1_pure_massive", "is_lrg_v2_baseline_dr10", 
        "is_lrg_v3_color_relaxed", "is_lrg_v4_mag_relaxed", "is_lrg_v5_very_relaxed"
    ]):
        v1_col = "is_lrg_v1_pure_massive"
        v2_col = "is_lrg_v2_baseline_dr10"
        v4_col = "is_lrg_v4_mag_relaxed"
        v5_col = "is_lrg_v5_very_relaxed"
    else:
        print("Warning: missing variant flags, skipping variant overlap plot.")
        return

    # By construction, df is already filtered to is_lrg_v3 == True in Phase 3
    total_v3 = float(len(df))
    if total_v3 == 0:
        print("Warning: empty parent catalog, skipping variant overlap plot.")
        return

    frac_v1 = df[v1_col].sum() / total_v3
    frac_v2 = df[v2_col].sum() / total_v3
    frac_v4 = df[v4_col].sum() / total_v3
    frac_v5 = df[v5_col].sum() / total_v3

    labels = ["v1\n(pure)", "v2\n(baseline)", "v3\n(parent)", "v4\n(mag relaxed)", "v5\n(very relaxed)"]
    fracs = [frac_v1, frac_v2, 1.0, frac_v4, frac_v5]
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12"]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, fracs, color=colors, edgecolor="white")
    
    # Add percentage labels on bars
    for bar, frac in zip(bars, fracs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{frac*100:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.ylim(0.0, 1.15)
    plt.ylabel("Fraction of v3 LRGs")
    title = f"Phase 3 parent sample: overlap with other variants"
    if mode:
        title += f"\n({mode})"
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    outpath = os.path.join(outdir, "phase3_variant_overlap.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved variant overlap plot: {outpath}")


def detect_modes(phase3_dir: str, variant: str) -> List[str]:
    """Detect available ranking modes by looking for parent catalogs under {phase3_dir}/{variant}/."""
    variant_dir = os.path.join(phase3_dir, variant)
    if not os.path.isdir(variant_dir):
        return []
    modes = []
    for entry in os.listdir(variant_dir):
        catalog_path = os.path.join(variant_dir, entry, "phase3_lrg_parent_catalog.csv")
        if os.path.isdir(os.path.join(variant_dir, entry)) and os.path.exists(catalog_path):
            modes.append(entry)
    return sorted(modes)


def process_single_mode(parent_csv: str, output_dir: str, mode: str) -> None:
    """Process and visualize a single mode's parent catalog."""
    if not os.path.exists(parent_csv):
        raise FileNotFoundError(f"Parent CSV not found: {parent_csv}")

    parent_df = pd.read_csv(parent_csv)
    print(f"\n{'='*60}")
    print(f"Mode: {mode}")
    print(f"Loaded parent catalog with {len(parent_df)} rows from {parent_csv}")

    ensure_output_dir(output_dir)

    # Generate all plots
    plot_zmag_hist(parent_df, output_dir, mode)
    plot_color_color(parent_df, output_dir, mode)
    plot_sky_map(parent_df, output_dir, mode)
    plot_counts_per_region(parent_df, output_dir, mode)
    plot_variant_overlap(parent_df, output_dir, mode)

    print(f"Figures written to: {output_dir}")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    print("=" * 60)
    print("Phase 3 Visualization")
    print("=" * 60)
    
    variant = args.variant
    variant_dir = os.path.join(args.phase3_dir, variant)

    if args.parent_csv:
        # Single file mode
        output_dir = args.output_dir if args.output_dir else os.path.join(
            os.path.dirname(args.parent_csv), "figures"
        )
        mode = args.mode if args.mode else os.path.basename(os.path.dirname(args.parent_csv))
        process_single_mode(args.parent_csv, output_dir, mode)
    
    elif args.all_modes or not args.mode:
        # Process all detected modes
        modes = detect_modes(args.phase3_dir, variant)
        if not modes:
            raise ValueError(
                f"No ranking modes with parent catalogs detected in {variant_dir}. "
                "Run run_phase3_build_parent_sample.py first."
            )
        
        print(f"Variant: {variant}")
        print(f"Detected modes: {modes}")
        
        for mode in modes:
            parent_csv = os.path.join(variant_dir, mode, "phase3_lrg_parent_catalog.csv")
            output_dir = os.path.join(variant_dir, mode, "figures")
            process_single_mode(parent_csv, output_dir, mode)
    
    else:
        # Single mode specified
        parent_csv = os.path.join(variant_dir, args.mode, "phase3_lrg_parent_catalog.csv")
        output_dir = args.output_dir if args.output_dir else os.path.join(
            variant_dir, args.mode, "figures"
        )
        process_single_mode(parent_csv, output_dir, args.mode)

    print("\n" + "=" * 60)
    print("Phase 3 visualization complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

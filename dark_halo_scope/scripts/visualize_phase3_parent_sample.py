#!/usr/bin/env python3
"""
Phase 3 visualization: parent LRG sample

This script produces basic diagnostic plots for the Phase 3 parent catalog:

- z-band magnitude distribution
- color-color diagram (r - z vs z - W1)
- sky map of LRGs in RA and Dec, colored by region_id
- bar chart of v3 LRG counts per region_id
- histogram of overlap with other LRG variants (v1, v2, v4, v5)

These plots are intended both as science sanity checks and as potential
figures for the research log and poster.

Assumes Python 3.9 and the following libraries:
- numpy
- pandas
- matplotlib
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Visualize Phase 3 LRG parent sample")

    parser.add_argument(
        "--parent-csv",
        default="results/phase3/v3_color_relaxed/phase3_lrg_parent_catalog.csv",
        help="CSV produced by run_phase3_build_parent_sample.py",
    )
    parser.add_argument(
        "--bricks-csv",
        default="results/phase3/v3_color_relaxed/phase3_target_bricks.csv",
        help="Phase 3 bricks CSV (for region geometry context)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/phase3/v3_color_relaxed/figures",
        help="Directory where figures will be saved",
    )

    return parser.parse_args(argv)


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_zmag_hist(df: pd.DataFrame, outdir: str) -> None:
    if "z_mag" not in df.columns:
        print("Warning: z_mag not found, skipping z magnitude histogram.")
        return

    plt.figure(figsize=(6, 4))
    z = df["z_mag"].values
    z = z[np.isfinite(z)]
    plt.hist(z, bins=40, alpha=0.8)
    plt.xlabel("z-band magnitude (AB)")
    plt.ylabel("Number of LRGs")
    plt.title("Phase 3 v3 LRG parent sample: z magnitude distribution")
    plt.grid(alpha=0.3)
    outpath = os.path.join(outdir, "phase3_zmag_hist.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print("Saved z magnitude histogram:", outpath)


def plot_color_color(df: pd.DataFrame, outdir: str) -> None:
    if "r_minus_z" not in df.columns or "z_minus_w1" not in df.columns:
        print("Warning: color columns not found, skipping color-color diagram.")
        return

    r_z = df["r_minus_z"].values
    z_w1 = df["z_minus_w1"].values
    mask = np.isfinite(r_z) & np.isfinite(z_w1)

    if not np.any(mask):
        print("Warning: no finite color data, skipping color-color diagram.")
        return

    plt.figure(figsize=(6, 5))
    plt.scatter(r_z[mask], z_w1[mask], s=2, alpha=0.3)
    plt.xlabel("r - z (mag)")
    plt.ylabel("z - W1 (mag)")
    plt.title("Phase 3 v3 LRG parent sample: color-color diagram")
    plt.grid(alpha=0.3)
    outpath = os.path.join(outdir, "phase3_color_color.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print("Saved color-color diagram:", outpath)


def plot_sky_map(df: pd.DataFrame, outdir: str) -> None:
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
        plt.scatter(ra, dec, s=2, alpha=0.3)
    else:
        # Color by region_id using a simple integer colormap
        unique_regions = np.unique(region_ids)
        # Map each region_id to an integer 0..N-1
        region_to_int = {rid: i for i, rid in enumerate(unique_regions)}
        region_idx = np.array([region_to_int[r] for r in region_ids])
        cmap = plt.cm.get_cmap("tab10")
        scatter = plt.scatter(ra, dec, s=2, alpha=0.4, c=region_idx, cmap=cmap)
        
        # Create legend with correct colors
        handles = []
        labels = []
        n_regions = len(unique_regions)
        for rid, idx in region_to_int.items():
            color = cmap(idx / max(n_regions - 1, 1))
            handles.append(plt.Line2D([], [], marker="o", linestyle="", 
                                       markersize=6, color=color))
            labels.append(f"region {rid}")
        plt.legend(handles, labels, loc="best", fontsize="small")

    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    plt.title("Phase 3 v3 LRG parent sample: sky distribution")
    plt.grid(alpha=0.3)
    outpath = os.path.join(outdir, "phase3_sky_map.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print("Saved sky map:", outpath)


def plot_counts_per_region(df: pd.DataFrame, outdir: str) -> None:
    if "region_id" not in df.columns:
        print("Warning: region_id not found, skipping per-region counts plot.")
        return

    counts = df["region_id"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("region_id")
    plt.ylabel("Number of v3 LRGs")
    plt.title("Phase 3 v3 LRG counts per region")
    plt.grid(axis="y", alpha=0.3)
    outpath = os.path.join(outdir, "phase3_counts_per_region.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print("Saved counts-per-region plot:", outpath)


def plot_variant_overlap(df: pd.DataFrame, outdir: str) -> None:
    """
    For v3 LRGs, show how many also satisfy v1, v2, v4, v5.

    This helps quantify how "massive" or "relaxed" the parent sample is.
    """
    required = ["is_lrg_v1", "is_lrg_v2", "is_lrg_v3", "is_lrg_v4", "is_lrg_v5"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("Warning: missing variant flags {}, skipping variant overlap plot.".format(missing))
        return

    # By construction, df is already filtered to is_lrg_v3 == True in Phase 3
    total_v3 = float(len(df))
    if total_v3 == 0:
        print("Warning: empty parent catalog, skipping variant overlap plot.")
        return

    frac_v1 = df["is_lrg_v1"].sum() / total_v3
    frac_v2 = df["is_lrg_v2"].sum() / total_v3
    frac_v4 = df["is_lrg_v4"].sum() / total_v3
    frac_v5 = df["is_lrg_v5"].sum() / total_v3

    labels = ["v1 (pure)", "v2 (baseline)", "v3 (parent)", "v4 (mag relaxed)", "v5 (very relaxed)"]
    fracs = [frac_v1, frac_v2, 1.0, frac_v4, frac_v5]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, fracs)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Fraction of v3 LRGs")
    plt.title("Phase 3 parent sample: overlap with other variants")
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)
    outpath = os.path.join(outdir, "phase3_variant_overlap.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print("Saved variant overlap plot:", outpath)


def main(argv=None):
    args = parse_args(argv)

    ensure_output_dir(args.output_dir)

    if not os.path.exists(args.parent_csv):
        raise FileNotFoundError(f"Parent CSV not found: {args.parent_csv}")

    parent_df = pd.read_csv(args.parent_csv)
    print("Loaded parent catalog with {} rows from {}".format(len(parent_df), args.parent_csv))

    # Basic plots from parent_df
    plot_zmag_hist(parent_df, args.output_dir)
    plot_color_color(parent_df, args.output_dir)
    plot_sky_map(parent_df, args.output_dir)
    plot_counts_per_region(parent_df, args.output_dir)
    plot_variant_overlap(parent_df, args.output_dir)

    # Optional: simple brick context log
    if os.path.exists(args.bricks_csv):
        bricks_df = pd.read_csv(args.bricks_csv)
        n_bricks = len(bricks_df)
        if "region_id" in bricks_df.columns:
            n_regions = bricks_df["region_id"].nunique()
        else:
            n_regions = None
        print("Phase 3 bricks CSV:", args.bricks_csv)
        print("  Number of bricks:", n_bricks)
        if n_regions is not None:
            print("  Number of regions:", n_regions)

    print("Phase 3 visualization complete. Figures written to:", args.output_dir)


if __name__ == "__main__":
    main()


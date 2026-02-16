#!/usr/bin/env python3
"""Generate UMAP visualization from saved embeddings (D01 feature_space_analysis output).

Loads embeddings.npz from D01 results, runs UMAP dimensionality reduction,
and produces a publication-quality scatter plot colored by category:
  - Real Tier-A lenses (gold stars)
  - Low beta_frac injections (blue circles)
  - High beta_frac injections (cyan triangles)
  - Val negatives (gray dots)

Output: PNG figure + PDF (for paper), and a JSON with UMAP coordinates.

Requires: umap-learn, matplotlib, numpy
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def main():
    ap = argparse.ArgumentParser(description="Generate UMAP from D01 embeddings")
    ap.add_argument(
        "--embeddings",
        required=True,
        help="Path to embeddings.npz from feature_space_analysis.py",
    )
    ap.add_argument("--out-dir", required=True, help="Output directory for figures")
    ap.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors (default: 15)")
    ap.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist (default: 0.1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--dpi", type=int, default=300, help="Figure DPI (default: 300)")
    args = ap.parse_args()

    if not HAS_UMAP:
        print("ERROR: umap-learn not installed. Install with: pip install umap-learn")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # Load embeddings
    print(f"Loading embeddings from: {args.embeddings}")
    data = np.load(args.embeddings, allow_pickle=True)

    # Expected keys from feature_space_analysis.py
    print(f"  Keys: {list(data.keys())}")

    # Auto-detect key naming convention
    keys = list(data.keys())
    print(f"  Available keys: {keys}")

    # Support both naming conventions
    if "emb_real" in keys:
        emb_real = data["emb_real"]
        emb_inj_low = data["emb_inj_low"]
        emb_inj_high = data["emb_inj_high"]
        emb_neg = data["emb_neg"]
        scores_real = data.get("scores_real", None)
        scores_inj_low = data.get("scores_inj_low", None)
        scores_inj_high = data.get("scores_inj_high", None)
        scores_neg = data.get("scores_neg", None)
    elif "emb_real_tier_a" in keys:
        emb_real = data["emb_real_tier_a"]
        emb_inj_low = data["emb_inj_low_bf"]
        emb_inj_high = data["emb_inj_high_bf"]
        emb_neg = data["emb_negatives"]
        scores_real = data.get("scores_real_tier_a", None)
        scores_inj_low = data.get("scores_inj_low_bf", None)
        scores_inj_high = data.get("scores_inj_high_bf", None)
        scores_neg = data.get("scores_negatives", None)
    else:
        raise KeyError(f"Unrecognized embedding keys: {keys}")

    print(f"  Real Tier-A: {emb_real.shape}")
    print(f"  Inj low-bf:  {emb_inj_low.shape}")
    print(f"  Inj high-bf: {emb_inj_high.shape}")
    print(f"  Negatives:   {emb_neg.shape}")

    # Combine for UMAP
    all_emb = np.vstack([emb_real, emb_inj_low, emb_inj_high, emb_neg])
    n_real = len(emb_real)
    n_low = len(emb_inj_low)
    n_high = len(emb_inj_high)
    n_neg = len(emb_neg)
    labels = (
        ["Real Tier-A"] * n_real
        + ["Inj (low bf)"] * n_low
        + ["Inj (high bf)"] * n_high
        + ["Negative"] * n_neg
    )

    print(f"\nTotal embeddings: {len(all_emb)}")
    print(f"Running UMAP (n_neighbors={args.n_neighbors}, min_dist={args.min_dist})...")

    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
        n_components=2,
        metric="euclidean",
    )
    umap_coords = reducer.fit_transform(all_emb)
    print(f"  UMAP done: {umap_coords.shape}")

    # Split back
    idx = 0
    coords_real = umap_coords[idx:idx + n_real]; idx += n_real
    coords_low = umap_coords[idx:idx + n_low]; idx += n_low
    coords_high = umap_coords[idx:idx + n_high]; idx += n_high
    coords_neg = umap_coords[idx:idx + n_neg]; idx += n_neg

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Negatives first (background)
    ax.scatter(
        coords_neg[:, 0], coords_neg[:, 1],
        c="#AAAAAA", s=8, alpha=0.3, label=f"Negatives (n={n_neg})",
        zorder=1, edgecolors="none",
    )
    # High beta_frac injections
    ax.scatter(
        coords_high[:, 0], coords_high[:, 1],
        c="#00BCD4", s=25, alpha=0.6, marker="^",
        label=f"Injections, high $\\beta_{{frac}}$ [0.7,1.0] (n={n_high})",
        zorder=2, edgecolors="none",
    )
    # Low beta_frac injections
    ax.scatter(
        coords_low[:, 0], coords_low[:, 1],
        c="#2196F3", s=25, alpha=0.7, marker="o",
        label=f"Injections, low $\\beta_{{frac}}$ [0.1,0.3] (n={n_low})",
        zorder=3, edgecolors="none",
    )
    # Real Tier-A
    ax.scatter(
        coords_real[:, 0], coords_real[:, 1],
        c="#FF9800", s=80, alpha=0.9, marker="*",
        label=f"Real Tier-A lenses (n={n_real})",
        zorder=4, edgecolors="black", linewidths=0.3,
    )

    ax.set_xlabel("UMAP-1", fontsize=13)
    ax.set_ylabel("UMAP-2", fontsize=13)
    ax.set_title(
        "CNN Feature Space: Real Lenses vs Parametric Injections\n"
        "(EfficientNetV2-S penultimate layer, 1280-D $\\rightarrow$ 2-D UMAP)",
        fontsize=14,
    )
    ax.legend(loc="best", fontsize=10, framealpha=0.8)
    ax.tick_params(labelsize=11)

    # Save
    png_path = os.path.join(args.out_dir, "umap_feature_space.png")
    pdf_path = os.path.join(args.out_dir, "umap_feature_space.pdf")
    fig.tight_layout()
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {png_path}")
    print(f"Figure saved: {pdf_path}")

    # --- Score-colored version ---
    if scores_real is not None:
        all_scores = np.concatenate([scores_real, scores_inj_low, scores_inj_high, scores_neg])
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        sc = ax2.scatter(
            umap_coords[:, 0], umap_coords[:, 1],
            c=all_scores, cmap="RdYlGn", s=12, alpha=0.7,
            vmin=0, vmax=1, edgecolors="none",
        )
        plt.colorbar(sc, ax=ax2, label="CNN Score (sigmoid)", shrink=0.8)
        ax2.set_xlabel("UMAP-1", fontsize=13)
        ax2.set_ylabel("UMAP-2", fontsize=13)
        ax2.set_title(
            "CNN Score in Feature Space\n"
            "(Color: model confidence, 0=negative, 1=lens)",
            fontsize=14,
        )
        ax2.tick_params(labelsize=11)
        score_png = os.path.join(args.out_dir, "umap_score_colored.png")
        fig2.tight_layout()
        fig2.savefig(score_png, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig2)
        print(f"Figure saved: {score_png}")

    # Save coordinates for downstream use
    coord_path = os.path.join(args.out_dir, "umap_coordinates.json")
    coord_data = {
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "seed": args.seed,
        "n_real": n_real,
        "n_inj_low": n_low,
        "n_inj_high": n_high,
        "n_neg": n_neg,
    }
    with open(coord_path, "w") as f:
        json.dump(coord_data, f, indent=2)
    print(f"Metadata saved: {coord_path}")

    # Also save the raw UMAP coordinates as npz
    npz_path = os.path.join(args.out_dir, "umap_coordinates.npz")
    np.savez(
        npz_path,
        coords_real=coords_real,
        coords_inj_low=coords_low,
        coords_inj_high=coords_high,
        coords_neg=coords_neg,
    )
    print(f"Coordinates saved: {npz_path}")


if __name__ == "__main__":
    main()

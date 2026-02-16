"""
Phase 1.6: Region selection using EMR-derived LRG densities.

This script:
  1. Loads brick-level LRG counts from an EMR output CSV
     (columns: brickname, lrg_count).
  2. Fetches DR10 bricks in the Phase1p5Config footprint using either
     local FITS (if available) or TAP as a fallback.
  3. Applies the same brick-level quality cuts as Phase 1.5.
  4. Joins the LRG counts onto the bricks table and computes
     LRG surface density per brick (per deg^2).
  5. Runs the existing region_scout.select_regions logic to define
     primary and backup contiguous regions.
  6. Writes CSVs and a short markdown summary under output_dir.

Usage (from project root):
  python -m scripts.run_phase1p6_region_selection \
    --lrg-density-csv /path/to/emr_brick_lrg_counts.csv \
    --output-dir outputs/phase1p6
"""

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import Phase1p5Config
from src.region_scout import (
    fetch_bricks,
    load_bricks_from_local_fits,
    apply_brick_quality_cuts,
    select_regions,
)


def _load_bricks_with_fallback(config: Phase1p5Config) -> pd.DataFrame:
    """
    Load bricks using local FITS if configured and available,
    otherwise fall back to TAP bricks_s query.
    """
    if getattr(config, "use_local_bricks", False):
        try:
            print(
                f"  Attempting to load bricks from local FITS: {config.bricks_fits}",
                flush=True,
            )
            bricks = load_bricks_from_local_fits(config)
            print(
                f"  Loaded {len(bricks)} bricks from local FITS in scouting footprint.",
                flush=True,
            )
            return bricks
        except Exception as exc:
            print(
                f"  Warning: failed to load local bricks FITS ({exc}). "
                "Falling back to TAP.",
                flush=True,
            )

    print("  Fetching bricks via TAP service...", flush=True)
    bricks = fetch_bricks(config)
    print(f"  Retrieved {len(bricks)} bricks via TAP.", flush=True)
    return bricks


def _load_lrg_counts_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load bricks-level LRG counts from EMR output CSV.

    We expect at least two columns (case-insensitive):
      - brickname
      - lrg_count

    Any additional columns are ignored here.
    """
    print(f"  Loading LRG counts from CSV: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)

    lower_to_original = {c.lower(): c for c in df.columns}
    required = {"brickname", "lrg_count"}
    missing = [c for c in required if c not in lower_to_original]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s) {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.rename(
        columns={
            lower_to_original["brickname"]: "brickname",
            lower_to_original["lrg_count"]: "lrg_count",
        }
    )

    # Keep only what we actually need here
    df = df[["brickname", "lrg_count"]].copy()
    df["brickname"] = df["brickname"].astype(str)
    df["lrg_count"] = df["lrg_count"].fillna(0).astype(int)

    n_rows = len(df)
    n_nonzero = (df["lrg_count"] > 0).sum()
    print(
        f"  Loaded {n_rows} brick rows from CSV "
        f"({n_nonzero} with lrg_count > 0).",
        flush=True,
    )
    return df


def _attach_lrg_density(
    bricks_quality: pd.DataFrame,
    lrg_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach LRG counts and densities to the quality-filtered bricks table.

    Bricks with no entry in lrg_df are assumed to have lrg_count = 0.
    """
    print("  Joining brick table with LRG counts...", flush=True)
    merged = bricks_quality.merge(
        lrg_df, on="brickname", how="left", validate="one_to_one"
    )

    merged["lrg_count"] = merged["lrg_count"].fillna(0).astype(int)

    if "area_deg2" not in merged.columns:
        raise RuntimeError(
            "Expected column 'area_deg2' not found after bricks loading. "
            "Check fetch_bricks or load_bricks_from_local_fits."
        )

    merged["lrg_density"] = merged["lrg_count"] / merged["area_deg2"].clip(lower=1e-6)

    n_nonzero = (merged["lrg_density"] > 0).sum()
    print(
        f"  Attached LRG density to {len(merged)} bricks "
        f"({n_nonzero} with density > 0).",
        flush=True,
    )
    return merged


def _write_summary_markdown(
    output_dir: Path,
    config: Phase1p5Config,
    bricks_all: pd.DataFrame,
    bricks_quality: pd.DataFrame,
    bricks_with_density: pd.DataFrame,
    primary_region: pd.DataFrame,
    backup_region: pd.DataFrame,
) -> None:
    """
    Write a compact markdown summary of the Phase 1.6 region selection.

    This is useful for ISEF documentation and for keeping track of
    how the primary region was chosen from the EMR-derived LRG map.
    """
    fname = output_dir / "phase1p6_region_selection.md"

    total_bricks = len(bricks_all)
    quality_bricks = len(bricks_quality)
    density_bricks = (bricks_with_density["lrg_density"] > 0).sum()

    primary_area = primary_region["area_deg2"].sum() if not primary_region.empty else 0.0
    primary_lrg = primary_region["lrg_count"].sum() if not primary_region.empty else 0
    backup_area = backup_region["area_deg2"].sum() if not backup_region.empty else 0.0
    backup_lrg = backup_region["lrg_count"].sum() if not backup_region.empty else 0

    with fname.open("w", encoding="utf-8") as f:
        f.write("# Phase 1.6 Region Selection Summary\n\n")
        f.write("This step joins EMR-derived LRG counts per brick with the DR10\n")
        f.write("bricks table, applies the Phase1p5 brick-level quality cuts, and\n")
        f.write("selects primary and backup contiguous regions using the existing\n")
        f.write("region_scout logic.\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- TAP URL: `{config.tap_url}`\n")
        f.write(f"- Bricks table: `{config.bricks_table}`\n")
        f.write(f"- Tractor table (for context): `{config.tractor_table}`\n")
        f.write(
            f"- Footprint: RA [{config.ra_min}, {config.ra_max}] deg, "
            f"Dec [{config.dec_min}, {config.dec_max}] deg\n"
        )
        f.write(
            f"- Region area target: [{config.min_region_area_deg2}, "
            f"{config.max_region_area_deg2}] deg^2\n\n"
        )

        f.write("## Brick statistics\n\n")
        f.write(f"- Total bricks in footprint: {total_bricks}\n")
        f.write(f"- Bricks passing quality cuts: {quality_bricks}\n")
        f.write(f"- Bricks with nonzero LRG density: {density_bricks}\n\n")

        f.write("## Primary region\n\n")
        f.write(f"- Total area: {primary_area:.2f} deg^2\n")
        f.write(f"- Total LRG count: {primary_lrg}\n")
        f.write(
            f"- Mean LRG surface density: "
            f"{(primary_lrg / primary_area) if primary_area > 0 else 0.0:.1f} "
            "per deg^2\n"
        )
        if not primary_region.empty:
            f.write(
                f"- Number of bricks: {len(primary_region)}\n"
                f"- RA range: [{primary_region['ra'].min():.3f}, "
                f"{primary_region['ra'].max():.3f}] deg\n"
                f"- Dec range: [{primary_region['dec'].min():.3f}, "
                f"{primary_region['dec'].max():.3f}] deg\n"
            )
        f.write("\n")

        f.write("## Backup region\n\n")
        if backup_region.empty:
            f.write("- No backup region selected (only one strong component).\n")
        else:
            f.write(f"- Total area: {backup_area:.2f} deg^2\n")
            f.write(f"- Total LRG count: {backup_lrg}\n")
            f.write(
                f"- Mean LRG surface density: "
                f"{(backup_lrg / backup_area) if backup_area > 0 else 0.0:.1f} "
                "per deg^2\n"
            )
            f.write(
                f"- Number of bricks: {len(backup_region)}\n"
                f"- RA range: [{backup_region['ra'].min():.3f}, "
                f"{backup_region['ra'].max():.3f}] deg\n"
                f"- Dec range: [{backup_region['dec'].min():.3f}, "
                f"{backup_region['dec'].max():.3f}] deg\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.6 region selection using EMR LRG density CSV."
    )
    parser.add_argument(
        "--lrg-density-csv",
        required=True,
        help="Path to EMR bricks CSV with columns brickname,lrg_count.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for Phase 1.6 products. "
             "Defaults to Phase1p5Config.output_dir with 'phase1p6' suffix.",
    )
    args = parser.parse_args()

    csv_path = Path(args.lrg_density_csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"LRG density CSV not found: {csv_path}")

    config = Phase1p5Config()

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        # Use Phase1p5 output_dir as a base and append phase1p6
        output_dir = Path(config.output_dir).parent / "phase1p6"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("Phase 1.6: Region selection from EMR LRG density")
    print("============================================================")
    print(f"  LRG density CSV: {csv_path}")
    print(f"  Output directory: {output_dir}")
    print(
        f"  Footprint: RA [{config.ra_min}, {config.ra_max}] deg, "
        f"Dec [{config.dec_min}, {config.dec_max}] deg",
        flush=True,
    )

    # 1. Load bricks
    bricks_all = _load_bricks_with_fallback(config)

    # 2. Apply brick quality cuts
    print("  Applying brick-level quality cuts...", flush=True)
    bricks_quality = apply_brick_quality_cuts(bricks_all, config)
    print(
        f"  Bricks after quality cuts: {len(bricks_quality)} "
        f"(from {len(bricks_all)})",
        flush=True,
    )

    # 3. Load LRG counts CSV
    lrg_df = _load_lrg_counts_csv(csv_path)

    # 4. Attach LRG density
    bricks_with_density = _attach_lrg_density(bricks_quality, lrg_df)

    # 5. Select primary and backup regions
    print("  Selecting primary and backup regions...", flush=True)
    primary_region, backup_region = select_regions(bricks_with_density, config)
    print(
        f"  Primary region bricks: {len(primary_region)}; "
        f"backup region bricks: {len(backup_region)}",
        flush=True,
    )

    # 6. Write outputs
    bricks_all.to_csv(
        output_dir / "bricks_all_phase1p6.csv", index=False
    )
    bricks_quality.to_csv(
        output_dir / "bricks_after_quality_cuts_phase1p6.csv", index=False
    )
    bricks_with_density.to_csv(
        output_dir / "bricks_with_lrg_density_phase1p6.csv", index=False
    )
    primary_region.to_csv(
        output_dir / "primary_region_bricks_phase1p6.csv", index=False
    )
    backup_region.to_csv(
        output_dir / "backup_region_bricks_phase1p6.csv", index=False
    )

    _write_summary_markdown(
        output_dir=output_dir,
        config=config,
        bricks_all=bricks_all,
        bricks_quality=bricks_quality,
        bricks_with_density=bricks_with_density,
        primary_region=primary_region,
        backup_region=backup_region,
    )

    print("  Wrote Phase 1.6 outputs to:", output_dir)
    print("  - bricks_all_phase1p6.csv")
    print("  - bricks_after_quality_cuts_phase1p6.csv")
    print("  - bricks_with_lrg_density_phase1p6.csv")
    print("  - primary_region_bricks_phase1p6.csv")
    print("  - backup_region_bricks_phase1p6.csv")
    print("  - phase1p6_region_selection.md")
    print("Done.")


if __name__ == "__main__":
    main()


import os
from pathlib import Path

import pandas as pd

from src.config import Phase1p5Config
from src.region_scout import (
    fetch_bricks,
    apply_brick_quality_cuts,
    estimate_lrg_density_for_bricks,
    select_regions,
)


def _ensure_output_dir(path_str: str) -> Path:
    p = Path(path_str)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_region_summary_md(
    output_dir: Path,
    config: Phase1p5Config,
    bricks_all: pd.DataFrame,
    bricks_quality: pd.DataFrame,
    bricks_with_density: pd.DataFrame,
    primary_region: pd.DataFrame,
    backup_region: pd.DataFrame,
) -> None:
    """
    Write a markdown summary documenting the selection logic and metrics.

    This is critical for ISEF and publication-level reproducibility.
    """
    fname = output_dir / "phase1p5_region_selection.md"

    total_bricks = len(bricks_all)
    quality_bricks = len(bricks_quality)
    density_bricks = (
        bricks_with_density["lrg_density"] > 0
    ).sum()

    primary_area = primary_region["area_deg2"].sum() if not primary_region.empty else 0.0
    primary_lrg = primary_region["lrg_count"].sum() if not primary_region.empty else 0
    backup_area = backup_region["area_deg2"].sum() if not backup_region.empty else 0.0
    backup_lrg = backup_region["lrg_count"].sum() if not backup_region.empty else 0

    with fname.open("w", encoding="utf-8") as f:
        f.write("# Phase 1.5 Region Selection Summary\n\n")
        f.write("This document records how the DR10 bricks were filtered and how the\n")
        f.write("primary and backup regions were defined, without using any strong-lens\n")
        f.write("catalogs. This ensures an unbiased selection function for later phases.\n\n")

        f.write("## Inputs\n\n")
        f.write(f"- TAP URL: `{config.tap_url}`\n")
        f.write(f"- Bricks table: `{config.bricks_table}`\n")
        f.write(f"- Tractor table: `{config.tractor_table}`\n")
        f.write(
            f"- Scouting footprint: RA in [{config.ra_min}, {config.ra_max}] deg, "
            f"Dec in [{config.dec_min}, {config.dec_max}] deg\n\n"
        )

        f.write("## Brick-level Quality Cuts\n\n")
        f.write("Criteria applied to `ls_dr10.bricks_s`:\n\n")
        f.write(f"- psfsize_r ≤ {config.max_psfsize_r:.2f} arcsec\n")
        f.write(f"- psfdepth_r ≥ {config.min_psfdepth_r:.2f} (5σ depth proxy)\n")
        f.write(f"- ebv ≤ {config.max_ebv:.2f}\n")
        f.write(f"- nexp_r ≥ {config.min_nexp_r} (if present; else assumed 1)\n\n")

        f.write("Brick counts:\n\n")
        f.write(f"- Total bricks in footprint: {total_bricks}\n")
        f.write(f"- Bricks passing quality cuts: {quality_bricks}\n\n")

        f.write("## DESI-like LRG Proxy\n\n")
        f.write("Extinction-corrected AB magnitudes (nanomaggies):\n\n")
        f.write(f"- z < {config.lrg_z_mag_max:.2f}\n")
        f.write(f"- r − z > {config.lrg_min_r_minus_z:.2f}\n")
        f.write(f"- z − W1 > {config.lrg_min_z_minus_w1:.2f}\n")
        if config.use_photo_z:
            f.write(
                f"- photo_z in [{config.lrg_z_phot_min:.2f}, "
                f"{config.lrg_z_phot_max:.2f}]\n"
            )
        else:
            f.write("- photo_z not used in the primary proxy\n")

        f.write("\nBricks with explicit LRG density estimates: "
                f"{density_bricks} (limited to "
                f"{config.max_bricks_for_lrg_density}).\n\n")

        f.write("## Region Definition\n\n")
        f.write(
            "We group bricks into contiguous regions based on spatial adjacency of\n"
            "brick centers (within ≈0.3 deg in RA and Dec) and then compute:\n\n"
        )
        f.write("- Total area per region\n")
        f.write("- Total LRG count per region\n")
        f.write("- LRG surface density (LRGs per deg²)\n\n")
        f.write(
            "We then select the primary region as the contiguous region whose area\n"
            "falls within the target window and has the highest LRG surface density.\n\n"
        )
        f.write(
            f"Target area window: [{config.min_region_area_deg2:.1f}, "
            f"{config.max_region_area_deg2:.1f}] deg²\n\n"
        )

        f.write("### Primary region\n\n")
        f.write(f"- Number of bricks: {len(primary_region)}\n")
        f.write(f"- Total area: {primary_area:.2f} deg²\n")
        f.write(f"- Total LRG count: {primary_lrg}\n")
        if primary_area > 0:
            f.write(
                f"- LRG surface density: {primary_lrg / primary_area:.1f} "
                "LRGs/deg²\n\n"
            )
        else:
            f.write("- LRG surface density: N/A\n\n")

        if not backup_region.empty:
            f.write("### Backup region\n\n")
            f.write(f"- Number of bricks: {len(backup_region)}\n")
            f.write(f"- Total area: {backup_area:.2f} deg²\n")
            f.write(f"- Total LRG count: {backup_lrg}\n")
            if backup_area > 0:
                f.write(
                    f"- LRG surface density: {backup_lrg / backup_area:.1f} "
                    "LRGs/deg²\n\n"
                )
        else:
            f.write("### Backup region\n\n")
            f.write("No backup region selected.\n\n")

        f.write("## Files Produced\n\n")
        f.write("- `bricks_all.csv`: all bricks in the footprint with QA metrics\n")
        f.write("- `bricks_quality.csv`: bricks passing hard quality cuts\n")
        f.write("- `bricks_with_density.csv`: quality bricks with LRG densities\n")
        f.write("- `primary_region_bricks.csv`: bricks in the primary region\n")
        f.write("- `backup_region_bricks.csv`: bricks in the backup region (if any)\n\n")
        f.write(
            "No strong-lens catalogs were used in any part of this region selection.\n"
            "Lens catalogs will be used later only for validation of recovery\n"
            "fractions, not for defining where we look.\n"
        )


def main() -> None:
    config = Phase1p5Config()
    output_dir = _ensure_output_dir(config.output_dir)

    print("Phase 1.5: DR10 region scouting and brick selection")
    print(f"  TAP URL: {config.tap_url}")
    print(f"  Bricks table: {config.bricks_table}")
    print(f"  Tractor table: {config.tractor_table}")
    print(
        f"  Footprint: RA [{config.ra_min}, {config.ra_max}] deg, "
        f"Dec [{config.dec_min}, {config.dec_max}] deg"
    )

    # 1. Fetch bricks in footprint
    print("  [1/5] Fetching bricks from survey-bricks table...")
    bricks_all = fetch_bricks(config)
    bricks_all.to_csv(output_dir / "bricks_all.csv", index=False)

    # 2. Apply hard quality cuts
    print("  [2/5] Applying brick-level quality cuts...")
    bricks_quality = apply_brick_quality_cuts(bricks_all, config)
    bricks_quality.to_csv(output_dir / "bricks_quality.csv", index=False)
    print(f"      Bricks passing cuts: {len(bricks_quality)}")

    if bricks_quality.empty:
        print("      No bricks passed the quality cuts. Check your thresholds.")
        return

    # 3. Estimate LRG densities for up to max_bricks_for_lrg_density bricks
    print("  [3/5] Estimating DESI-like LRG density per brick...")
    bricks_with_density = estimate_lrg_density_for_bricks(bricks_quality, config)
    bricks_with_density.to_csv(
        output_dir / "bricks_with_density.csv", index=False
    )

    # 4. Select primary and backup regions
    print("  [4/5] Selecting primary and backup contiguous regions...")
    primary_region, backup_region = select_regions(bricks_with_density, config)
    primary_region.to_csv(output_dir / "primary_region_bricks.csv", index=False)
    if not backup_region.empty:
        backup_region.to_csv(
            output_dir / "backup_region_bricks.csv", index=False
        )

    # 5. Write markdown summary
    print("  [5/5] Writing markdown summary...")
    write_region_summary_md(
        output_dir,
        config,
        bricks_all,
        bricks_quality,
        bricks_with_density,
        primary_region,
        backup_region,
    )

    print(f"\n✓ Phase 1.5 outputs written to {output_dir.resolve()}")
    print("Generated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()


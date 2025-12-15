#!/usr/bin/env python3
"""
Phase 3 field definition

Read Phase 2 hypergrid analysis results for the v3_color_relaxed variant,
select one or more regions to serve as Phase 3 "laboratory" fields, and
save:

- phase3_region_choices.md    - human readable summary of selected regions
- phase3_target_bricks.csv    - brick level list for those regions

This script operates only on CSVs produced by Phase 2 and does not touch
the raw DR10 sweeps.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Phase3Config:
    regions_summary_csv: str
    regions_bricks_csv: str
    output_dir: str
    region_ids: List[int]


def parse_args(argv: Optional[List[str]] = None) -> Phase3Config:
    parser = argparse.ArgumentParser(description="Phase 3 field definition")

    parser.add_argument(
        "--regions-summary-csv",
        default="results/phase2_analysis/v3_color_relaxed/phase2_regions_summary.csv",
        help="CSV with per region summary for v3_color_relaxed",
    )
    parser.add_argument(
        "--regions-bricks-csv",
        default="results/phase2_analysis/v3_color_relaxed/phase2_regions_bricks.csv",
        help="CSV with brick level membership for v3_color_relaxed regions",
    )
    parser.add_argument(
        "--output-dir",
        default="results/phase3",
        help="Output directory for Phase 3 field definitions",
    )
    parser.add_argument(
        "--region-ids",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Optional explicit list of region_id values to adopt "
            "for Phase 3. If omitted, the script will automatically "
            "select the top 2 regions by area weighted v3 density."
        ),
    )

    args = parser.parse_args(argv)

    return Phase3Config(
        regions_summary_csv=args.regions_summary_csv,
        regions_bricks_csv=args.regions_bricks_csv,
        output_dir=args.output_dir,
        region_ids=args.region_ids if args.region_ids is not None else [],
    )


def auto_select_regions(regions: pd.DataFrame, k: int = 2) -> List[int]:
    """
    Automatically select k regions using an area weighted v3 density score.

    Score = lrg_density_v3 * sqrt(area_deg2)

    This favors regions that are both dense and reasonably extended.
    """
    # Handle column name variations from Phase 2 output
    density_col = None
    for col in ["mean_lrg_density_v3_color_relaxed", "lrg_density_v3", "mean_lrg_density_v3"]:
        if col in regions.columns:
            density_col = col
            break
    if density_col is None:
        raise ValueError(
            "Expected a v3 density column (e.g. 'mean_lrg_density_v3_color_relaxed') in regions summary CSV. "
            f"Available columns: {list(regions.columns)}"
        )
    
    area_col = None
    for col in ["total_area_deg2", "area_deg2"]:
        if col in regions.columns:
            area_col = col
            break
    if area_col is None:
        raise ValueError(
            "Expected an area column (e.g. 'total_area_deg2') in regions summary CSV. "
            f"Available columns: {list(regions.columns)}"
        )

    regions = regions.copy()
    regions["phase3_score"] = regions[density_col] * np.sqrt(regions[area_col])
    regions = regions.sort_values("phase3_score", ascending=False)
    top = regions.head(k)
    return top["region_id"].astype(int).tolist()


def write_region_choices_md(cfg: Phase3Config, regions: pd.DataFrame, bricks: pd.DataFrame) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    md_path = os.path.join(cfg.output_dir, "phase3_region_choices.md")

    # Resolve column names (handle Phase 2 output variations)
    area_col = "total_area_deg2" if "total_area_deg2" in regions.columns else "area_deg2"
    density_col = None
    for col in ["mean_lrg_density_v3_color_relaxed", "lrg_density_v3", "mean_lrg_density_v3"]:
        if col in regions.columns:
            density_col = col
            break
    if density_col is None:
        density_col = "mean_lrg_density_v3_color_relaxed"  # Will show N/A if missing

    lines: List[str] = []
    lines.append("# Phase 3 - Field definition\n")
    lines.append("")
    lines.append("This document records the exact fields used to build the Phase 3 LRG parent catalog.")
    lines.append("The selection is based on the Phase 2 v3_color_relaxed hypergrid analysis.")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Regions summary CSV: `{cfg.regions_summary_csv}`")
    lines.append(f"- Regions bricks CSV: `{cfg.regions_bricks_csv}`")
    lines.append("")
    lines.append("## Selected regions")
    lines.append("")
    sel_regions = regions.set_index("region_id")
    lines.append("| region_id | n_bricks | area_deg2 | center_ra_deg | center_dec_deg | lrg_density_v3 | notes |")
    lines.append("|-----------|----------|-----------|----------------|-----------------|----------------|-------|")

    for rid in cfg.region_ids:
        row = sel_regions.loc[rid]
        note = "Primary Phase 3 field" if rid == cfg.region_ids[0] else "Secondary or comparison field"
        area_val = row[area_col] if area_col in row.index else 0.0
        density_val = row[density_col] if density_col in row.index else 0.0
        lines.append(
            f"| {rid} | {int(row['n_bricks'])} | {area_val:.4f} | "
            f"{row['ra_center_deg']:.3f} | {row['dec_center_deg']:.3f} | "
            f"{density_val:.1f} | {note} |"
        )

    lines.append("")
    lines.append("## Aggregate brick coverage for Phase 3")
    lines.append("")

    sel_bricks = bricks[bricks["region_id"].isin(cfg.region_ids)].copy()

    lines.append(f"- Total bricks across all selected regions: {len(sel_bricks)}")
    brick_area_col = "area_deg2" if "area_deg2" in sel_bricks.columns else None
    if brick_area_col:
        total_area = sel_bricks[brick_area_col].sum()
        lines.append(f"- Total geometric area (sum of brick areas): {total_area:.3f} deg^2")

    # Handle column name variations for RA/Dec
    for label, possible_cols in [("RA", ["ra", "brick_ra_center"]), ("Dec", ["dec", "brick_dec_center"])]:
        col = None
        for c in possible_cols:
            if c in sel_bricks.columns:
                col = c
                break
        if col:
            col_min = sel_bricks[col].min()
            col_max = sel_bricks[col].max()
            lines.append(f"- {label} range: [{col_min:.3f}, {col_max:.3f}] deg")

    lines.append("")
    lines.append("These RA and Dec ranges define the approximate Phase 3 footprint.")
    lines.append(
        "In the parent catalog builder, I restrict DR10 sweeps to this footprint "
        "to avoid unnecessary input and output outside the Phase 3 fields."
    )
    lines.append("")
    lines.append("## Notes on scope and assumptions")
    lines.append("")
    lines.append("- Phase 3 uses only the DR10 South footprint, as in Phase 2.")
    lines.append("- Parent LRGs are defined using the v3_color_relaxed cuts as implemented in Phase 2.")
    lines.append("- I retain region IDs so that later phases can compare results across fields.")
    lines.append(
        "- This document is meant to be pasted directly into the research log as a record of "
        "exact field choices before simulation and training."
    )

    with open(md_path, "w") as f:
        f.write("\n".join(lines))


def main(argv: Optional[List[str]] = None) -> None:
    cfg = parse_args(argv)

    regions = pd.read_csv(cfg.regions_summary_csv)
    bricks = pd.read_csv(cfg.regions_bricks_csv)

    if "region_id" not in regions.columns:
        raise ValueError("regions summary CSV must contain 'region_id'")
    if "region_id" not in bricks.columns:
        raise ValueError("regions bricks CSV must contain 'region_id'")

    if cfg.region_ids:
        missing = [rid for rid in cfg.region_ids if rid not in regions["region_id"].values]
        if missing:
            raise ValueError(f"Requested region_ids not found in regions summary: {missing}")
    else:
        cfg.region_ids = auto_select_regions(regions, k=2)

    os.makedirs(cfg.output_dir, exist_ok=True)
    target_bricks = bricks[bricks["region_id"].isin(cfg.region_ids)].copy()
    brick_csv_path = os.path.join(cfg.output_dir, "phase3_target_bricks.csv")
    target_bricks.to_csv(brick_csv_path, index=False)

    write_region_choices_md(cfg, regions, bricks)

    print("============================================================")
    print("Phase 3 field definition completed")
    print("============================================================")
    print(f"Selected region_ids: {cfg.region_ids}")
    print(f"Wrote bricks CSV to: {brick_csv_path}")
    print(f"Wrote markdown summary to: {os.path.join(cfg.output_dir, 'phase3_region_choices.md')}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Phase 3: Define lens search fields from Phase 2 regions.

This script takes the Phase 2 region summaries and brick-level tables
(for a single variant, typically v3_color_relaxed), and chooses
a set of "fields" (regions) for the next phases.

New in this version:
- Support for selecting more than k=2 regions (--num-regions, default 5).
- Three ranking modes:
    * area_weighted_v3 : mean LRG density_v3 * sqrt(area)
    * density_v3       : mean LRG density_v3
    * total_lrg_v3     : total LRG count in the region
- For each ranking mode we write a separate set of outputs in
  {output_dir}/{ranking_mode}/
    - phase3_regions_summary.csv
    - phase3_target_bricks.csv
    - phase3_region_choices.md

If --region-ids is given, we bypass ranking and use those IDs directly
(legacy/manual mode). In that case we emit a single set of outputs
under {output_dir}/manual/.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("phase3.define_fields")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


RANKING_MODES_ALL = ("area_weighted_v3", "density_v3", "total_lrg_v3")


@dataclass
class Phase3Config:
    phase2_analysis_dir: Path
    variant: str
    output_dir: Path

    # Phase 2 inputs
    regions_summary_csv: Path
    regions_bricks_csv: Path

    # Optional manual selection
    region_ids: List[int]

    # Automatic selection parameters
    num_regions: int
    ranking_modes: List[str]


def parse_args(argv: Optional[Sequence[str]] = None) -> Phase3Config:
    parser = argparse.ArgumentParser(
        description="Phase 3 – define fields (regions) for lens search."
    )

    parser.add_argument(
        "--phase2-analysis-dir",
        type=str,
        default="results/phase2_analysis",
        help="Directory where Phase 2 hypergrid analysis outputs live.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="v3_color_relaxed",
        help="Variant subdir name (e.g. v3_color_relaxed).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase3",
        help="Directory where Phase 3 outputs will be written.",
    )
    parser.add_argument(
        "--regions-summary-csv",
        type=str,
        default="",
        help=(
            "Optional explicit path to phase2_regions_summary.csv. "
            "If empty, constructed as {phase2-analysis-dir}/{variant}/phase2_regions_summary.csv."
        ),
    )
    parser.add_argument(
        "--regions-bricks-csv",
        type=str,
        default="",
        help=(
            "Optional explicit path to phase2_regions_bricks.csv. "
            "If empty, constructed as {phase2-analysis-dir}/{variant}/phase2_regions_bricks.csv."
        ),
    )
    parser.add_argument(
        "--region-ids",
        type=str,
        default="",
        help=(
            "Comma-separated region_id values to select manually. "
            "If provided, ranking is skipped and only these regions are used."
        ),
    )
    parser.add_argument(
        "--num-regions",
        type=int,
        default=5,
        help=(
            "Number of regions to select per ranking mode when auto-selecting. "
            "Ignored if --region-ids is provided. Default: 5."
        ),
    )
    parser.add_argument(
        "--ranking-modes",
        type=str,
        default=",".join(RANKING_MODES_ALL),
        help=(
            "Comma-separated ranking modes to use when auto-selecting. "
            "Choices: area_weighted_v3,density_v3,total_lrg_v3. "
            "Default: all three."
        ),
    )

    args = parser.parse_args(argv)

    phase2_analysis_dir = Path(args.phase2_analysis_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    variant = args.variant

    if args.regions_summary_csv:
        regions_summary_csv = Path(args.regions_summary_csv).expanduser().resolve()
    else:
        regions_summary_csv = (
            phase2_analysis_dir / variant / "phase2_regions_summary.csv"
        )

    if args.regions_bricks_csv:
        regions_bricks_csv = Path(args.regions_bricks_csv).expanduser().resolve()
    else:
        regions_bricks_csv = (
            phase2_analysis_dir / variant / "phase2_regions_bricks.csv"
        )

    if args.region_ids.strip():
        region_ids = [int(x) for x in args.region_ids.split(",") if x.strip()]
    else:
        region_ids = []

    ranking_modes_input = [m.strip() for m in args.ranking_modes.split(",") if m.strip()]
    for m in ranking_modes_input:
        if m not in RANKING_MODES_ALL:
            raise ValueError(
                f"Unknown ranking mode '{m}'. "
                f"Allowed: {', '.join(RANKING_MODES_ALL)}"
            )

    cfg = Phase3Config(
        phase2_analysis_dir=phase2_analysis_dir,
        variant=variant,
        output_dir=output_dir,
        regions_summary_csv=regions_summary_csv,
        regions_bricks_csv=regions_bricks_csv,
        region_ids=region_ids,
        num_regions=args.num_regions,
        ranking_modes=ranking_modes_input,
    )

    return cfg


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def load_phase2_tables(cfg: Phase3Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading Phase 2 tables...")
    regions = pd.read_csv(cfg.regions_summary_csv)
    bricks = pd.read_csv(cfg.regions_bricks_csv)

    required_region_cols = {"region_id", "total_area_deg2"}
    missing = required_region_cols.difference(regions.columns)
    if missing:
        raise ValueError(
            f"Phase 2 regions summary missing required columns: {sorted(missing)}"
        )

    if "region_id" not in bricks.columns:
        raise ValueError("Phase 2 bricks table must have a 'region_id' column.")

    return regions, bricks


def _get_metric_columns(regions: pd.DataFrame, variant: str) -> Dict[str, str]:
    """
    Determine the metric column names for the given variant.

    We expect Phase 2 to have columns like:
        total_n_lrg_{variant}
        mean_lrg_density_{variant}
        total_area_deg2
    """
    density_col = f"mean_lrg_density_{variant}"
    count_col = f"total_n_lrg_{variant}"
    area_col = "total_area_deg2"

    for col in (density_col, count_col, area_col):
        if col not in regions.columns:
            raise ValueError(
                f"Expected column '{col}' in Phase 2 regions summary but it is missing. "
                f"Available columns: {list(regions.columns)}"
            )

    return {
        "density": density_col,
        "count": count_col,
        "area": area_col,
    }


def auto_select_regions_for_mode(
    regions: pd.DataFrame,
    cfg: Phase3Config,
    mode: str,
) -> pd.DataFrame:
    """
    Select cfg.num_regions regions according to the requested ranking mode.

    Returns a DataFrame of the selected regions sorted by descending score,
    with extra columns:
        - phase3_ranking_mode
        - phase3_score
        - phase3_region_rank (1..k)
    """
    metric_cols = _get_metric_columns(regions, cfg.variant)
    density_col = metric_cols["density"]
    count_col = metric_cols["count"]
    area_col = metric_cols["area"]

    df = regions.copy()

    if mode == "area_weighted_v3":
        score = df[density_col] * np.sqrt(df[area_col])
        score_name = "phase3_score_area_weighted_v3"
        description = (
            "score = mean_lrg_density_v3 * sqrt(total_area_deg2) "
            "(prefers dense regions that also have non-trivial area)"
        )
    elif mode == "density_v3":
        score = df[density_col]
        score_name = "phase3_score_density_v3"
        description = "score = mean_lrg_density_v3 (pure density ranking)"
    elif mode == "total_lrg_v3":
        score = df[count_col]
        score_name = "phase3_score_total_lrg_v3"
        description = "score = n_lrg_v3 (total LRG count in region)"
    else:
        raise ValueError(f"Unsupported ranking mode: {mode}")

    logger.info(
        "Selecting regions using mode '%s': %s", mode, description
    )

    df[score_name] = score
    df["phase3_ranking_mode"] = mode
    df["phase3_score"] = df[score_name]

    df_sorted = df.sort_values("phase3_score", ascending=False)
    selected = df_sorted.head(cfg.num_regions).copy()
    selected["phase3_region_rank"] = np.arange(1, len(selected) + 1, dtype=int)

    logger.info(
        "Top %d regions for mode '%s': %s",
        cfg.num_regions,
        mode,
        selected[["region_id", "phase3_region_rank", "phase3_score"]].to_dict(
            orient="records"
        ),
    )

    return selected


def write_region_choices_md(
    cfg: Phase3Config,
    selected_regions: pd.DataFrame,
    mode: str,
    outdir: Path,
) -> None:
    """
    Write a human-readable markdown summary for a given ranking mode.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    md_path = outdir / "phase3_region_choices.md"

    lines: List[str] = []
    lines.append(f"# Phase 3 Region Choices – {mode}\n")
    lines.append("")
    lines.append("This file documents the automatic selection of regions for Phase 3.")
    lines.append("")
    lines.append(f"- Variant: `{cfg.variant}`")
    lines.append(f"- Ranking mode: `{mode}`")
    lines.append(f"- Number of regions requested: {cfg.num_regions}")
    lines.append("")
    lines.append("## Selected regions")
    lines.append("")
    # Show a compact table of key columns
    cols_to_show = [
        "phase3_region_rank",
        "region_id",
        "total_area_deg2",
    ]

    metric_cols = _get_metric_columns(selected_regions, cfg.variant)
    density_col = metric_cols["density"]
    count_col = metric_cols["count"]

    for col in (density_col, count_col):
        if col in selected_regions.columns:
            cols_to_show.append(col)

    cols_to_show.append("phase3_score")
    cols_to_show = [c for c in cols_to_show if c in selected_regions.columns]

    tbl = selected_regions[cols_to_show].copy()
    # Sort by rank for readability
    if "phase3_region_rank" in tbl.columns:
        tbl = tbl.sort_values("phase3_region_rank")

    # Render simple markdown table
    lines.append("| " + " | ".join(tbl.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(tbl.columns)) + " |")
    for _, row in tbl.iterrows():
        vals = []
        for c in tbl.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `phase3_region_rank = 1` is the highest-scoring region for this ranking mode."
    )
    lines.append(
        "- In later phases, you can compare performance as a function of rank "
        "(e.g., does including rank 5 actually help the model?)."
    )
    lines.append("")

    md_path.write_text("\n".join(lines))
    logger.info("Wrote region choices markdown: %s", md_path)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    cfg = parse_args(argv)
    logger.info("Phase 3 config: %s", cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    regions, bricks = load_phase2_tables(cfg)

    # Include variant in output path: {output_dir}/{variant}/{mode}/
    variant_dir = cfg.output_dir / cfg.variant

    if cfg.region_ids:
        # Manual selection path (legacy).
        mode = "manual"
        selected = regions[regions["region_id"].isin(cfg.region_ids)].copy()
        if selected.empty:
            raise ValueError(
                f"No regions found matching manual region_ids {cfg.region_ids}"
            )
        selected["phase3_ranking_mode"] = mode
        selected["phase3_region_rank"] = np.arange(1, len(selected) + 1, dtype=int)
        selected["phase3_score"] = np.nan

        subdir = variant_dir / mode
        subdir.mkdir(parents=True, exist_ok=True)

        # Save selected regions summary
        regions_out_csv = subdir / "phase3_regions_summary.csv"
        selected.to_csv(regions_out_csv, index=False)
        logger.info("Wrote selected regions summary: %s", regions_out_csv)

        # Attach rank to bricks
        rank_map: Dict[int, int] = dict(
            zip(selected["region_id"], selected["phase3_region_rank"])
        )
        target_bricks = bricks[bricks["region_id"].isin(cfg.region_ids)].copy()
        target_bricks["phase3_ranking_mode"] = mode
        target_bricks["phase3_region_rank"] = target_bricks["region_id"].map(rank_map)

        bricks_out_csv = subdir / "phase3_target_bricks.csv"
        target_bricks.to_csv(bricks_out_csv, index=False)
        logger.info("Wrote target bricks for manual selection: %s", bricks_out_csv)

        # Markdown summary
        write_region_choices_md(cfg, selected, mode=mode, outdir=subdir)

    else:
        # Automatic selection for each ranking mode
        for mode in cfg.ranking_modes:
            subdir = variant_dir / mode
            subdir.mkdir(parents=True, exist_ok=True)

            selected = auto_select_regions_for_mode(regions, cfg, mode=mode)

            # Save regions summary for this mode
            regions_out_csv = subdir / "phase3_regions_summary.csv"
            selected.to_csv(regions_out_csv, index=False)
            logger.info(
                "Wrote regions summary for mode '%s': %s", mode, regions_out_csv
            )

            # Build rank map and filter bricks
            rank_map: Dict[int, int] = dict(
                zip(selected["region_id"], selected["phase3_region_rank"])
            )
            target_bricks = bricks[bricks["region_id"].isin(selected["region_id"])].copy()
            target_bricks["phase3_ranking_mode"] = mode
            target_bricks["phase3_region_rank"] = target_bricks["region_id"].map(
                rank_map
            )

            bricks_out_csv = subdir / "phase3_target_bricks.csv"
            target_bricks.to_csv(bricks_out_csv, index=False)
            logger.info(
                "Wrote target bricks for mode '%s': %s", mode, bricks_out_csv
            )

            # Markdown summary
            write_region_choices_md(cfg, selected, mode=mode, outdir=subdir)

    logger.info("Phase 3 define_fields completed.")


if __name__ == "__main__":
    main()


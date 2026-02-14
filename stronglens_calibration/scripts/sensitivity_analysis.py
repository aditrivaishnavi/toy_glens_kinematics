#!/usr/bin/env python3
"""
Sensitivity Analysis: systematic parameter perturbation for the selection function.

Runs the injection-recovery grid multiple times with perturbed parameters to
quantify the systematic uncertainty envelope on the completeness function.

Perturbations:
  - PSF FWHM: ±10%  (multiply psfsize_r)
  - Source size: ±30% (multiply re_arcsec_range)
  - Color shift: ±0.2 mag (shift g_minus_r, r_minus_z means)
  - Lens axis ratio prior: broader/narrower q_lens_range

Output:
  - Per-cell delta-completeness relative to baseline
  - Maximum systematic band width per cell

This is a wrapper around run_selection_function(). Actual execution requires
a trained checkpoint and manifest; this script provides the infrastructure.

Usage:
    cd /lambda/nfs/.../code
    export PYTHONPATH=.

    python scripts/sensitivity_analysis.py \\
        --checkpoint checkpoints/best.pt \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --out-dir results/sensitivity_analysis

Author: stronglens_calibration project
Date: 2026-02-10
References:
  - MNRAS_RAW_NOTES.md Section 7.7.6
  - Fix injection pipeline plan, Step 6d
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import the grid runner
from scripts.selection_function_grid import run_selection_function

from dhs.s3io import is_s3_uri, join_uri, write_bytes, write_json


# ---------------------------------------------------------------------------
# Perturbation definitions
# ---------------------------------------------------------------------------
PERTURBATIONS = [
    {
        "name": "baseline",
        "description": "No perturbation (reference run)",
        "overrides": {},
    },
    {
        "name": "psf_plus10pct",
        "description": "PSF FWHM +10%: tests sensitivity to seeing uncertainty",
        "overrides": {"psf_scale_factor": 1.10},
    },
    {
        "name": "psf_minus10pct",
        "description": "PSF FWHM -10%",
        "overrides": {"psf_scale_factor": 0.90},
    },
    {
        "name": "source_size_plus30pct",
        "description": "Source R_e +30%: tests sensitivity to source size prior",
        "overrides": {"re_scale_factor": 1.30},
    },
    {
        "name": "source_size_minus30pct",
        "description": "Source R_e -30%",
        "overrides": {"re_scale_factor": 0.70},
    },
    {
        "name": "color_shift_red",
        "description": "g-r shifted +0.2 mag (redder sources)",
        "overrides": {"g_minus_r_shift": 0.2},
    },
    {
        "name": "color_shift_blue",
        "description": "g-r shifted -0.2 mag (bluer sources)",
        "overrides": {"g_minus_r_shift": -0.2},
    },
    {
        "name": "q_lens_broader",
        "description": "Broader lens ellipticity prior: q_lens in [0.3, 1.0]",
        "overrides": {"q_lens_range": (0.3, 1.0)},
    },
    {
        "name": "q_lens_narrower",
        "description": "Narrower lens ellipticity: q_lens in [0.7, 1.0]",
        "overrides": {"q_lens_range": (0.7, 1.0)},
    },
]


def run_sensitivity_analysis(
    checkpoint_path: str,
    manifest_path: str,
    out_dir: str,
    host_split: str = "val",
    host_max: int = 10000,
    thresholds: Optional[List[float]] = None,
    fpr_targets: Optional[List[float]] = None,
    injections_per_cell: int = 100,
    # Grid parameters (must match the main grid for consistency)
    depth_min: float = 22.5,
    depth_max: float = 24.5,
    depth_step: float = 0.5,
    seed: int = 1337,
    device_str: str = "cuda",
    data_root: Optional[str] = None,
    perturbation_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run the baseline and all perturbation variants.

    Returns a summary dict with delta-completeness statistics.
    """
    if thresholds is None:
        thresholds = [0.5]

    # Filter perturbations if requested
    perturbs = PERTURBATIONS
    if perturbation_names:
        perturbs = [p for p in PERTURBATIONS if p["name"] in perturbation_names or p["name"] == "baseline"]

    results: Dict[str, pd.DataFrame] = {}
    metadata_all: Dict[str, Dict] = {}

    for p in perturbs:
        name = p["name"]
        print(f"\n{'='*60}")
        print(f"Running: {name} — {p['description']}")
        print(f"{'='*60}")

        # Pass perturbation overrides through to the injection loop.
        # Supported keys: psf_scale_factor, re_scale_factor, g_minus_r_shift,
        # r_minus_z_shift, q_lens_range. Any unrecognised keys are ignored by
        # run_selection_function but recorded in metadata.
        overrides = p.get("overrides", {})

        t0 = time.time()
        df, meta = run_selection_function(
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            host_split=host_split,
            host_max=host_max,
            thresholds=thresholds,
            fpr_targets=fpr_targets,
            injections_per_cell=injections_per_cell,
            depth_min=depth_min,
            depth_max=depth_max,
            depth_step=depth_step,
            seed=seed,
            device_str=device_str,
            data_root=data_root,
            injection_overrides=overrides if overrides else None,
        )
        dt = time.time() - t0

        meta["perturbation"] = p
        meta["runtime_seconds"] = dt
        results[name] = df
        metadata_all[name] = meta
        print(f"  Completed in {dt:.1f}s")

    # Compute delta-completeness relative to baseline
    summary: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_path,
        "manifest": manifest_path,
        "n_perturbations": len(perturbs),
        "perturbation_names": [p["name"] for p in perturbs],
        "perturbation_details": perturbs,
    }

    if "baseline" in results:
        baseline = results["baseline"]
        # Filter to 'all' source_mag_bin if present
        if "source_mag_bin" in baseline.columns:
            baseline = baseline[baseline["source_mag_bin"] == "all"]

        delta_rows = []
        for name, df in results.items():
            if name == "baseline":
                continue
            if "source_mag_bin" in df.columns:
                df = df[df["source_mag_bin"] == "all"]
            # Merge on grid keys
            merged = baseline.merge(
                df,
                on=["theta_e", "psf_fwhm", "depth_5sig", "threshold"],
                suffixes=("_base", f"_{name}"),
                how="inner",
            )
            if len(merged) > 0:
                delta = merged[f"completeness_{name}"] - merged["completeness_base"]
                delta_rows.append({
                    "perturbation": name,
                    "mean_delta_C": float(delta.mean()),
                    "max_abs_delta_C": float(delta.abs().max()),
                    "std_delta_C": float(delta.std()),
                    "n_cells": len(merged),
                })

        summary["delta_completeness"] = delta_rows

        # Save individual CSVs
        for name, df in results.items():
            csv_path = join_uri(out_dir, f"selection_function_{name}.csv")
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            if is_s3_uri(out_dir):
                write_bytes(csv_path, csv_bytes, content_type="text/csv")
            else:
                os.makedirs(out_dir, exist_ok=True)
                with open(csv_path, "wb") as f:
                    f.write(csv_bytes)

    # Save summary
    summary_path = join_uri(out_dir, "sensitivity_analysis_summary.json")
    if is_s3_uri(out_dir):
        write_json(summary_path, summary)
    else:
        os.makedirs(out_dir, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    print(f"\nSummary saved to: {summary_path}")

    # Print delta table
    if "delta_completeness" in summary:
        print(f"\nDelta-completeness relative to baseline:")
        print(f"  {'Perturbation':30s} {'mean_dC':>10s} {'max|dC|':>10s} {'std_dC':>10s}")
        for row in summary["delta_completeness"]:
            print(f"  {row['perturbation']:30s} {row['mean_delta_C']:+10.4f} "
                  f"{row['max_abs_delta_C']:10.4f} {row['std_delta_C']:10.4f}")

    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Sensitivity Analysis: systematic parameter perturbation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="results/sensitivity_analysis")
    ap.add_argument("--host-split", default="val")
    ap.add_argument("--host-max", type=int, default=10000)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.5])
    ap.add_argument("--fpr-targets", nargs="+", type=float, default=None,
                    help="FPR-based thresholds (e.g. 0.001 0.0001)")
    ap.add_argument("--injections-per-cell", type=int, default=100,
                    help="Reduced injections for faster sensitivity sweeps")
    ap.add_argument("--depth-min", type=float, default=22.5)
    ap.add_argument("--depth-max", type=float, default=24.5)
    ap.add_argument("--depth-step", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--perturbations", nargs="*", default=None,
                    help="Subset of perturbation names to run (default: all)")
    args = ap.parse_args()

    run_sensitivity_analysis(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        host_split=args.host_split,
        host_max=args.host_max,
        thresholds=args.thresholds,
        fpr_targets=args.fpr_targets,
        injections_per_cell=args.injections_per_cell,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        depth_step=args.depth_step,
        seed=args.seed,
        device_str=args.device,
        data_root=args.data_root,
        perturbation_names=args.perturbations,
    )


if __name__ == "__main__":
    main()

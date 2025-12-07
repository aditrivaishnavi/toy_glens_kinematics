#!/usr/bin/env python3
"""
score_lensability.py

Read a maps_quality_summary.csv (from batch_inspect_maps.py),
compute lensibility scores & tiers for each galaxy, and write
an augmented CSV with additional columns:

- lens_score   (float, 0–100)
- lens_tier    ("elite", "good", "borderline", "reject")
- lens_notes   (semi-colon separated reasons)
- lens_hard_reject (True/False)

Usage:
    python src/scripts/score_lensability.py \
        --input data/maps_quality_summary.csv \
        --output data/maps_lensability_scored.csv
"""

import argparse
import csv
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.lensibility import GalaxyMetrics, lensibility_score


def parse_bool(s: str) -> bool:
    """Parse boolean-like strings safely."""
    s_lower = s.strip().lower()
    return s_lower in ("true", "1", "yes", "y")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV (maps_quality_summary.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV with lensibility scores added.",
    )
    args = parser.parse_args()

    print("============================================================")
    print("LENSIBILITY SCORING")
    print("============================================================")
    print(f"[INFO] Input CSV : {args.input}")
    print(f"[INFO] Output CSV: {args.output}")
    print("============================================================")

    rows_out = []

    with open(args.input, "r", newline="") as fin:
        reader = csv.DictReader(fin)
        input_fieldnames = reader.fieldnames or []

        for row in reader:
            # Construct GalaxyMetrics from row
            try:
                m = GalaxyMetrics(
                    file=row["file"],
                    status=row.get("status", "OK"),

                    flux_max=float(row["flux_max"]),
                    flux_mean=float(row["flux_mean"]),

                    vel_min=float(row["vel_min"]),
                    vel_max=float(row["vel_max"]),
                    vel_std=float(row["vel_std"]),
                    vel_grad=float(row["vel_grad"]),

                    flux_vel_corr=float(row["flux_vel_corr"]),

                    mask_fraction=float(row["mask_fraction"]),
                    frac_valid_in_1_5Re=float(row["frac_valid_in_1.5Re"]),

                    flag_low_rotation=parse_bool(row["flag_low_rotation"]),
                    flag_heavily_masked=parse_bool(row["flag_heavily_masked"]),
                    flag_low_flux=parse_bool(row["flag_low_flux"]),
                    usable_flag=parse_bool(row.get("usable", "True")),
                )
            except KeyError as e:
                raise KeyError(
                    f"Missing expected column in CSV: {e}. "
                    "Ensure maps_quality_summary.csv has the correct header."
                )

            result = lensibility_score(m)

            # Attach new fields
            row["lens_score"] = f"{result['score']:.2f}"
            row["lens_tier"] = result["tier"]
            row["lens_hard_reject"] = "True" if result["hard_reject"] else "False"
            row["lens_notes"] = "; ".join(result["notes"])

            rows_out.append(row)

    # Prepare new header
    new_fields = list(input_fieldnames)
    for extra in ["lens_score", "lens_tier", "lens_hard_reject", "lens_notes"]:
        if extra not in new_fields:
            new_fields.append(extra)

    with open(args.output, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=new_fields)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    print(f"[OK] Wrote scored CSV to: {args.output}")
    print("============================================================")


if __name__ == "__main__":
    main()


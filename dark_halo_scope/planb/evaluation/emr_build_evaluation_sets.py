"""
EMR script to build anchor and contaminant evaluation sets.

Run with emr-launcher:
    emr-launcher run --script s3://your-bucket/evaluation/emr_build_evaluation_sets.py

This is a lightweight job (not compute-heavy), uses single node.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def install_dependencies():
    """Install required packages."""
    packages = [
        "astropy",
        "requests",
        "pandas",
        "numpy",
    ]
    for pkg in packages:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s3-output",
        default="s3://darkhalo-scope/evaluation/",
        help="S3 output path",
    )
    parser.add_argument(
        "--arc-snr-threshold",
        type=float,
        default=2.0,
    )
    args = parser.parse_args()
    
    # Install deps
    logger.info("Installing dependencies...")
    install_dependencies()
    
    # Create local dirs
    output_dir = "/tmp/evaluation"
    cutout_dir = "/tmp/cutouts"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cutout_dir, exist_ok=True)
    
    # Build anchor catalog
    logger.info("Building anchor catalog...")
    from build_anchor_catalog import build_anchor_catalog
    anchor_df = build_anchor_catalog(
        output_dir=output_dir,
        cutout_dir=f"{cutout_dir}/anchors",
        arc_snr_threshold=args.arc_snr_threshold,
    )
    logger.info(f"Built anchor catalog: {len(anchor_df)} lenses")
    
    # Build contaminant catalog
    logger.info("Building contaminant catalog...")
    from build_contaminant_catalog import build_contaminant_catalog
    contam_df = build_contaminant_catalog(
        output_dir=output_dir,
        cutout_dir=f"{cutout_dir}/contaminants",
    )
    logger.info(f"Built contaminant catalog: {len(contam_df)} objects")
    
    # Upload to S3
    logger.info(f"Uploading to {args.s3_output}...")
    subprocess.run([
        "aws", "s3", "sync",
        output_dir,
        f"{args.s3_output}/catalogs/",
    ])
    subprocess.run([
        "aws", "s3", "sync",
        cutout_dir,
        f"{args.s3_output}/cutouts/",
    ])
    
    logger.info("Done!")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SET BUILD COMPLETE")
    print("=" * 60)
    print(f"Anchors: {len(anchor_df)}")
    print(f"  - In DR10: {anchor_df['in_dr10'].sum()}")
    print(f"  - Arc visible: {anchor_df['arc_visible_dr10'].sum()}")
    print(f"Contaminants: {len(contam_df)}")
    print(f"  - In DR10: {contam_df['in_dr10'].sum()}")
    print(f"\nOutput: {args.s3_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

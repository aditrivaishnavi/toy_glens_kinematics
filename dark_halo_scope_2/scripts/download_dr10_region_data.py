#!/usr/bin/env python3
"""
Download DR10 survey-bricks and the subset of SWEEP Tractor catalogs
that intersect a given RA/Dec footprint.

Footprint (default):
    RA 150–250 deg
    Dec 0–30 deg

Downloads:
    - external_data/dr10/survey-bricks-dr10-south.fits.gz
    - external_data/dr10/sweep_10.1/sweep-*.fits for that region
"""

import os
import re
import sys
from pathlib import Path
from urllib.request import urlopen, urlretrieve
from urllib.error import URLError, HTTPError

# Base URLs from Legacy Surveys DR10 data release
BRICKS_URL = (
    "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/"
    "dr10/south/survey-bricks-dr10-south.fits.gz"
)
SWEEP_INDEX_URL = (
    "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/"
    "dr10/south/sweep/10.1/"
)

# Default footprint: RA 150–250 deg, Dec 0–30 deg
RA_MIN = 150.0
RA_MAX = 250.0
DEC_MIN = 0.0
DEC_MAX = 30.0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} already exists")
        return
    print(f"[download] {url}")
    try:
        urlretrieve(url, dest)
    except (URLError, HTTPError) as e:
        print(f"[error] Failed to download {url}: {e}")
        return
    print(f"[ok] Saved to {dest}")


def decode_ra_dec_token(token: str) -> tuple[float, float]:
    """
    Decode a token like '150m005' or '155p010' into (RA, Dec) in degrees.

    Pattern: RRRsDDD where
      - RRR is RA in degrees (integer)
      - s is 'p' (plus) or 'm' (minus)
      - DDD is |Dec|*10 (so 005 -> 0.5 deg, 020 -> 2.0 deg)
    """
    if len(token) != 7:
        raise ValueError(f"Unexpected token length for {token}")
    ra_str = token[0:3]
    sign_char = token[3]
    dec_str = token[4:7]

    ra = float(int(ra_str))
    dec_mag = float(int(dec_str)) / 10.0
    sign = +1.0 if sign_char.lower() == "p" else -1.0
    dec = sign * dec_mag
    return ra, dec


def sweep_overlaps_region(
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
) -> bool:
    """
    Return True if the sweep box [ra1, ra2] x [dec1, dec2]
    intersects the target box [ra_min, ra_max] x [dec_min, dec_max].
    """
    # Simple axis-aligned rectangle overlap in this RA range (no wrap)
    ra_overlap = (ra2 >= ra_min) and (ra1 <= ra_max)
    dec_overlap = (dec2 >= dec_min) and (dec1 <= dec_max)
    return ra_overlap and dec_overlap


def parse_sweep_index(html: str) -> list[str]:
    """
    Extract all sweep-*.fits filenames from the index HTML.
    """
    # Matches e.g. sweep-150m005-155p000.fits
    pattern = r"sweep-[0-9]{3}[mp][0-9]{3}-[0-9]{3}[mp][0-9]{3}\.fits"
    return sorted(set(re.findall(pattern, html)))


def select_sweeps_for_region(
    filenames: list[str],
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
) -> list[str]:
    """
    Filter sweeps that intersect the target RA/Dec footprint.
    """
    selected = []
    for name in filenames:
        # name format: sweep-RRRsDDD-RRRsDDD.fits
        core = name.replace("sweep-", "").replace(".fits", "")
        token1, token2 = core.split("-")
        ra1, dec1 = decode_ra_dec_token(token1)
        ra2, dec2 = decode_ra_dec_token(token2)

        # Ensure ra1 < ra2, dec1 < dec2
        ra_lo, ra_hi = sorted([ra1, ra2])
        dec_lo, dec_hi = sorted([dec1, dec2])

        if sweep_overlaps_region(
            ra_lo,
            dec_lo,
            ra_hi,
            dec_hi,
            ra_min,
            ra_max,
            dec_min,
            dec_max,
        ):
            selected.append(name)

    return sorted(selected)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "external_data" / "dr10"
    bricks_path = data_root / "survey-bricks-dr10-south.fits.gz"
    sweeps_dir = data_root / "sweep_10.1"

    ensure_dir(data_root)
    ensure_dir(sweeps_dir)

    print("=== DR10 region data downloader ===")
    print(f"Project root: {project_root}")
    print(f"Data root:    {data_root}")
    print("")
    print(f"Target footprint: RA [{RA_MIN}, {RA_MAX}] deg, "
          f"Dec [{DEC_MIN}, {DEC_MAX}] deg")
    print("")

    # 1. Download survey-bricks
    print("[1/3] Downloading survey-bricks table (if needed)...")
    download_file(BRICKS_URL, bricks_path)

    # 2. Fetch sweep index and select relevant files
    print("[2/3] Fetching SWEEP 10.1 index...")
    try:
        with urlopen(SWEEP_INDEX_URL) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except (URLError, HTTPError) as e:
        print(f"[error] Failed to fetch SWEEP index: {e}")
        sys.exit(1)

    all_sweeps = parse_sweep_index(html)
    print(f"  Found {len(all_sweeps)} sweep files in 10.1")

    selected_sweeps = select_sweeps_for_region(
        all_sweeps,
        RA_MIN,
        RA_MAX,
        DEC_MIN,
        DEC_MAX,
    )
    print(f"  Selected {len(selected_sweeps)} sweeps intersecting footprint")

    # Save the list of selected sweeps for reproducibility
    list_path = data_root / "sweeps_ra150_250_dec0_30_10.1.txt"
    with list_path.open("w", encoding="utf-8") as f:
        for name in selected_sweeps:
            f.write(name + "\n")
    print(f"  Wrote file list: {list_path}")

    # 3. Download selected sweep files
    print("[3/3] Downloading selected sweeps (if needed)...")
    for name in selected_sweeps:
        url = SWEEP_INDEX_URL + name
        dest = sweeps_dir / name
        download_file(url, dest)

    print("")
    print("Done.")
    print(f"Survey-bricks: {bricks_path}")
    print(f"Sweeps dir:    {sweeps_dir}")
    print(f"List of sweeps: {list_path}")


if __name__ == "__main__":
    main()


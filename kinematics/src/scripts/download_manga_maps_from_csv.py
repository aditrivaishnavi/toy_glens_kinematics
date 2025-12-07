#!/usr/bin/env python3
"""
download_manga_maps_from_csv.py

Read a ranked candidate CSV (from rank_manga_disks.py) and download
the top N MAPS FITS files, one by one, with clear logging.

Usage:
    python3 src/scripts/download_manga_maps_from_csv.py \
        --csv data/manga_disk_candidates.csv \
        --outdir data/maps \
        --n_files 10

Requirements:
    pip install requests
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import requests


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to ranked CSV (e.g. manga_disk_candidates.csv)",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="data/maps",
        help="Directory to store downloaded MAPS FITS files",
    )
    p.add_argument(
        "--n_files",
        type=int,
        default=10,
        help="Number of top rows to download",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds",
    )
    return p.parse_args()


def download_file(url: str, dest_path: Path, timeout: float = 60.0) -> bool:
    """
    Stream-download a large file from URL to dest_path.
    Returns True on success, False on failure.
    """
    # If already exists, skip
    if dest_path.exists():
        print(f"[SKIP] File already exists: {dest_path}")
        return True

    print(f"[INFO] Downloading:\n       URL:  {url}\n       -->   {dest_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                print(f"[ERROR] HTTP {r.status_code} for URL: {url}")
                return False

            # Try to get size for logging only
            total = r.headers.get("Content-Length")
            if total is not None:
                try:
                    total = int(total)
                except ValueError:
                    total = None

            chunk_size = 1024 * 1024  # 1 MB
            downloaded = 0

            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Log occasionally
                    if total:
                        pct = 100.0 * downloaded / total
                        sys.stdout.write(
                            f"\r       Downloaded {downloaded/1e6:6.1f} MB "
                            f"({pct:5.1f}% of {total/1e6:6.1f} MB)"
                        )
                    else:
                        sys.stdout.write(
                            f"\r       Downloaded {downloaded/1e6:6.1f} MB"
                        )
                    sys.stdout.flush()

            sys.stdout.write("\n")
            print(f"[OK] Finished: {dest_path}")
            return True

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to download {url}")
        print(f"        Reason: {e}")
        return False

    except KeyboardInterrupt:
        print("\n[WARN] Download interrupted by user.")
        # Remove partial file
        if dest_path.exists():
            try:
                dest_path.unlink()
                print(f"[INFO] Removed partial file: {dest_path}")
            except OSError:
                pass
        raise


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[FATAL] CSV file not found: {csv_path}")
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading candidate CSV: {csv_path}")

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("[FATAL] CSV appears to be empty.")
        sys.exit(1)

    n = min(args.n_files, len(rows))
    print(f"[INFO] Will attempt to download top {n} MAPS files.")

    # Ensure required columns exist
    sample = rows[0]
    if "maps_url" not in sample:
        print("[FATAL] CSV missing 'maps_url' column.")
        sys.exit(1)

    if "plateifu" not in sample:
        print("[WARN] CSV missing 'plateifu' column; "
              "filenames will be taken from URL only.")

    successes = 0
    failures = 0

    for idx in range(n):
        row = rows[idx]
        url = row["maps_url"].strip()
        plateifu = row.get("plateifu", "").strip()

        print("\n============================================================")
        print(f"[INFO] Candidate {idx+1} of {n}")
        print(f"       plateifu : {plateifu if plateifu else '(unknown)'}")
        print(f"       maps_url : {url}")
        print("============================================================")

        if not url:
            print("[ERROR] Empty URL, skipping this candidate.")
            failures += 1
            continue

        # Derive local filename
        # Prefer plateifu-based name if present, else use basename from URL.
        if plateifu:
            # Keep original extension from URL
            url_basename = os.path.basename(url)
            if url_basename.endswith(".fits.gz"):
                ext = ".fits.gz"
            else:
                ext = ""
            filename = f"manga-{plateifu}-MAPS{ext}"
        else:
            filename = os.path.basename(url) or f"maps_{idx+1}.fits.gz"

        dest_path = outdir / filename

        ok = download_file(url, dest_path, timeout=args.timeout)
        if ok:
            successes += 1
        else:
            failures += 1

        # Small delay between downloads to be gentle on the server
        time.sleep(1.0)

    print("\n============================================================")
    print("[SUMMARY] Download complete.")
    print(f"  Successful downloads: {successes}")
    print(f"  Failed downloads    : {failures}")
    print(f"  Output directory    : {outdir}")
    print("============================================================")


if __name__ == "__main__":
    main()


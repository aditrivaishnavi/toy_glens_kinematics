
#!/usr/bin/env python3
"""
Hard-negative mining from model scores on real (non-injected) survey cutouts.

Workflow:
1) Run your trained model on a large pool of REAL cutouts (no injections) and write a score table.
   The score table must contain: score, ra, dec, brickname (or an id), and any columns you need later.
2) Mine the highest-scoring negatives, optionally stratified by observing conditions, to avoid focusing on one regime.
3) Output a "negative catalog" parquet you can feed into Phase 4a manifest builder (--extra_negatives),
   or directly append to Phase 5 training as label=0.

Safety:
- If you do not have ground-truth labels for real cutouts, you must avoid incorrectly labeling true lenses as negatives.
  Practical mitigations:
  - Cross-match against known lens catalogs and remove matches (recommended).
  - Human vetting of top-N before adding to training (recommended).
This script supports supplying a "known_lens_catalog" to exclude by coordinate match radius.

This is the fastest path to pushing down FPR in realistic search conditions.
"""

import argparse
import math
from typing import Optional, Tuple, List

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def haversine_arcsec(ra1, dec1, ra2, dec2):
    # All inputs in degrees, output in arcsec
    ra1 = np.deg2rad(ra1); dec1 = np.deg2rad(dec1)
    ra2 = np.deg2rad(ra2); dec2 = np.deg2rad(dec2)
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = np.sin(ddec/2)**2 + np.cos(dec1)*np.cos(dec2)*np.sin(dra/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return np.rad2deg(c) * 3600.0


def load_coords(path: str, ra_col: str, dec_col: str, max_rows: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    dataset = ds.dataset(path, format="parquet")
    cols = [ra_col, dec_col]
    rows = []
    seen = 0
    for batch in dataset.to_batches(columns=cols, batch_size=200_000):
        if max_rows and seen >= max_rows:
            break
        if max_rows:
            batch = batch.slice(0, min(len(batch), max_rows - seen))
        seen += len(batch)
        rows.append((batch[ra_col].to_numpy(zero_copy_only=False), batch[dec_col].to_numpy(zero_copy_only=False)))
    if not rows:
        return np.array([]), np.array([])
    ra = np.concatenate([r[0] for r in rows]).astype(np.float64)
    dec = np.concatenate([r[1] for r in rows]).astype(np.float64)
    return ra, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="Score table parquet(s) over REAL cutouts")
    ap.add_argument("--out", required=True, help="Output negatives parquet directory")
    ap.add_argument("--score_col", default="score")
    ap.add_argument("--ra_col", default="ra")
    ap.add_argument("--dec_col", default="dec")
    ap.add_argument("--id_cols", default="brickname,objid", help="Comma-separated identifier columns to carry through if present")
    ap.add_argument("--top_k", type=int, default=200_000, help="Number of negatives to mine")
    ap.add_argument("--per_psf_bin", type=int, default=0, help="If >0, mine up to this many per psf_bin to diversify")
    ap.add_argument("--psf_col", default="psfsize_r")
    ap.add_argument("--known_lens_catalog", default="", help="Optional parquet(s) of known lenses to exclude")
    ap.add_argument("--exclude_radius_arcsec", type=float, default=3.0)
    ap.add_argument("--max_known_lenses", type=int, default=0, help="Optional cap when loading known lenses")
    args = ap.parse_args()

    dataset = ds.dataset(args.scores, format="parquet")
    want = [args.score_col, args.ra_col, args.dec_col, args.psf_col]
    for c in args.id_cols.split(","):
        c = c.strip()
        if c:
            want.append(c)
    want = [c for c in want if c in dataset.schema.names]

    if args.score_col not in want or args.ra_col not in want or args.dec_col not in want:
        raise ValueError("Score table must contain score, ra, dec columns")

    # Collect all scores into memory for mining. For very large runs, pre-partition and run this per chunk.
    arrays = {c: [] for c in want}
    for batch in dataset.to_batches(columns=want, batch_size=300_000):
        for c in want:
            arrays[c].append(batch[c].to_numpy(zero_copy_only=False))
    data = {c: np.concatenate(arrays[c]) for c in want}
    score = data[args.score_col].astype(np.float64)

    # Exclude known lenses if provided
    keep = np.ones_like(score, dtype=bool)
    if args.known_lens_catalog:
        ra_k, dec_k = load_coords(args.known_lens_catalog, args.ra_col, args.dec_col, max_rows=args.max_known_lenses)
        if len(ra_k) > 0:
            ra = data[args.ra_col].astype(np.float64)
            dec = data[args.dec_col].astype(np.float64)
            # Cheap approximate exclusion: chunked brute-force with early exit for moderate sizes.
            # For very large catalogs, replace with HTM/Healpix indexing.
            chunk = 10_000
            for i in range(0, len(ra_k), chunk):
                r2 = ra_k[i:i+chunk]
                d2 = dec_k[i:i+chunk]
                # Broadcast compute: (n, chunk) can be huge; do smaller blocks
                # We'll do per-row nearest in chunk by sampling if huge.
                # This is conservative, not perfect.
                for j in range(len(r2)):
                    dist = haversine_arcsec(ra, dec, r2[j], d2[j])
                    keep &= (dist > args.exclude_radius_arcsec)
        print(f"After excluding known lenses: keep={keep.sum()} / {len(keep)}")

    # Mining strategy
    if args.per_psf_bin and (args.psf_col in data):
        psf = data[args.psf_col].astype(np.float64)
        psf_bin = np.round(psf, 1)
        mined_idx = []
        for b in np.unique(psf_bin[keep]):
            idx = np.where(keep & (psf_bin == b))[0]
            if idx.size == 0:
                continue
            # take top per bin
            top = idx[np.argsort(-score[idx])[:args.per_psf_bin]]
            mined_idx.append(top)
        mined_idx = np.concatenate(mined_idx) if mined_idx else np.array([], dtype=np.int64)
        # If still over top_k, take global top among mined
        if mined_idx.size > args.top_k:
            mined_idx = mined_idx[np.argsort(-score[mined_idx])[:args.top_k]]
    else:
        idx = np.where(keep)[0]
        mined_idx = idx[np.argsort(-score[idx])[:args.top_k]]

    # Build output table
    out_cols = {}
    for c in want:
        out_cols[c] = pa.array(data[c][mined_idx])
    out_cols["is_control"] = pa.array(np.ones(len(mined_idx), dtype=np.int8))
    out_cols["control_kind"] = pa.array(np.array(["hard_mined"] * len(mined_idx)))

    table = pa.table(out_cols)
    pq.write_to_dataset(table, root_path=args.out, compression="zstd")
    print(f"Wrote hard negatives: {len(mined_idx)} rows to {args.out}")


if __name__ == "__main__":
    main()

# Split Assignment Documentation

**Date**: 2026-02-11
**Purpose**: Document the exact train/val/test split mechanism for reproducibility and paper methods section.

---

## Method: HEALPix-Based Deterministic Spatial Splits

Splits are assigned at the **galaxy level** using HEALPix spatial indexing, ensuring that galaxies in the same sky region are always assigned to the same split. This prevents spatial leakage (nearby galaxies appearing in different splits).

### Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| **HEALPix nside** | 128 | `configs/negative_sampling_v1.yaml` |
| **Ordering** | NESTED | `emr/sampling_utils.py:compute_healpix()` |
| **Hash seed** | 42 | `configs/negative_sampling_v1.yaml` |
| **Hash method** | SHA-256 of `"{healpix_idx}_{seed}"`, first 4 bytes as uint32, divided by 2^32 | `emr/sampling_utils.py:assign_split()` |

### Allocation Ratios

| Split | Target Fraction | Actual Count | Actual Fraction |
|-------|-----------------|--------------|-----------------|
| train | 0.70 | 291,509 | 69.99% |
| val | 0.15 | 62,180 | 14.93% |
| test | 0.15 | 62,760 | 15.07% |
| **Total** | 1.00 | **416,449** | **100.0%** |

### Algorithm

1. For each galaxy, compute HEALPix index at nside=128 from (RA, Dec) using `healpy.ang2pix(nside=128, theta, phi, nest=True)`.
2. Compute a deterministic hash: `SHA256("{healpix_idx}_42")`.
3. Convert the first 4 bytes of the hash to a float in [0, 1).
4. Assign split using cumulative thresholds in **explicit order** `[train, val, test]`:
   - hash < 0.70 → `train`
   - 0.70 <= hash < 0.85 → `val`
   - 0.85 <= hash → `test`

The explicit ordering is critical; an earlier version sorted alphabetically (`test < train < val`), which was fixed on 2026-02-09.

### Where Splits Are Assigned

Splits are assigned at **two points** in the pipeline:

1. **Negatives**: During EMR negative sampling (`emr/spark_negative_sampling.py`), each galaxy's HEALPix index and split are computed and stored in the Parquet output. The split is then embedded in each NPZ cutout as `meta_split`.

2. **Positives**: During the crossmatch step (`emr/spark_crossmatch_positives_v2.py`), positive galaxies are assigned splits using the same `assign_split()` function with the same seed=42 and nside=128, ensuring consistent assignment.

3. **Manifest**: `scripts/generate_training_manifest_parallel.py` reads `meta_split` from each NPZ and includes it in the manifest. No re-assignment occurs at manifest time.

### Disjointness Verification

Verified on 2026-02-11 using `scripts/verify_splits.py` on the full manifest (416,449 rows):

| Check | train-val overlap | train-test overlap | val-test overlap |
|-------|------------------|--------------------|------------------|
| `galaxy_id` | 0 | 0 | 0 |
| `cutout_path` | 0 | 0 | 0 |

Report: `docs/split_verification_report.json`

### Key Properties

- **Spatial disjointness**: All galaxies within the same HEALPix pixel (nside=128, ~0.21 deg^2) are assigned to the same split. This prevents nearby galaxies from leaking across splits.
- **Determinism**: The SHA-256 hash with seed=42 produces identical assignments across runs. No randomness or system state dependence.
- **Consistency**: Both positives and negatives use the same function and parameters, ensuring a galaxy at a given position always receives the same split regardless of label.

### Code References

- `emr/sampling_utils.py`: `compute_healpix()` (line 71), `assign_split()` (line 89)
- `configs/negative_sampling_v1.yaml`: spatial_splits section (line 92)
- `scripts/verify_splits.py`: disjointness verification script

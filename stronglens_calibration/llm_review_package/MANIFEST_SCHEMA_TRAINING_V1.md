# Verified schema: training_v1.parquet

---

## Short prompt to paste to the LLM

**Copy the following to the LLM:**

> Our training manifest is `training_v1.parquet`. Below is the **verified schema** (column names and Parquet/PyArrow types) read from the actual file on Lambda NFS using PyArrow. Please adjust your scripts’ default column names to match this manifest exactly: use `label` (int64) for labels, `split` (string) for train/val/test, `cutout_path` (string) for the path to each .npz, `sample_weight` (double) for per-sample loss weights, and `tier` (string) for stratum evaluation. Types are as listed; handle nullable optional columns as needed.

---

**Source:** `manifests/training_v1.parquet` on Lambda NFS  
`/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/manifests/training_v1.parquet`

**Verified:** 2026-02-10 via PyArrow on Lambda (parquet binary read). Use these column names and types so your scripts match our manifest exactly.

---

## Column names and Parquet/Arrow types

| Column name         | Parquet / PyArrow type | Notes |
|---------------------|------------------------|--------|
| cutout_path         | string                 | Full path to .npz cutout file. **Required for training/eval.** |
| filename            | string                 | Basename of cutout. |
| galaxy_id           | string                 | Identifier (e.g. brick_objid or from path stem). |
| label               | int64                  | **Required.** 1 = positive (lens), 0 = negative. |
| cutout_type         | string                 | "positive" or "negative". |
| ra                  | double                 | Right ascension. |
| dec                 | double                 | Declination. |
| size                | int64                  | Cutout size (e.g. 101). |
| pixel_scale         | double                 | Arcsec per pixel. |
| timestamp           | string                 | Cutout generation timestamp. |
| pipeline_version    | string                 | Pipeline version. |
| cutout_url          | string                 | URL if from remote. |
| layer               | string                 | Layer / survey. |
| bands_requested     | string                 | Bands (e.g. g,r,z). |
| tier                | string                 | **Required for stratum eval.** "A" | "B" (positives) or "N1" | "N2" (negatives). |
| weight              | double                 | Legacy weight column. |
| nobs_z              | int64                  | z-band exposure count. |
| match_type          | string                 | Match type if from crossmatch. |
| brickname           | string                 | DR10 brick name. |
| nan_count_g         | int64                  | NaN count in g band. |
| nan_count_r         | int64                  | NaN count in r band. |
| nan_count_z         | int64                  | NaN count in z band. |
| nan_frac            | double                 | Fraction of NaN pixels. |
| central_nan_frac    | double                 | NaN fraction in central region. |
| has_g               | bool                   | g band present. |
| has_r               | bool                   | r band present. |
| has_z               | bool                   | z band present. |
| core_brightness_r   | double                 | Core brightness (r). |
| core_max_r          | double                 | Core max (r). |
| annulus_brightness_r| double                 | Annulus brightness (r). |
| annulus_std_r       | double                 | Annulus std (r). |
| outer_brightness_r  | double                 | Outer brightness (r). |
| radial_gradient_r   | double                 | Radial gradient (r). |
| mad_r               | double                 | MAD (r). |
| std_r               | double                 | Std (r). |
| median_r            | double                 | Median (r). |
| mean_r              | double                 | Mean (r). |
| percentile_1_r      | double                 | 1st percentile (r). |
| percentile_99_r     | double                 | 99th percentile (r). |
| clip_frac_r         | double                 | Clip fraction (r). |
| quality_ok          | bool                   | Quality flag. |
| split               | string                 | **Required.** "train" | "val" | "test". |
| sample_weight       | double                 | **Required for training.** Per-sample loss weight (e.g. 1.0 Tier-A, 0.5 Tier-B, 1.0 N1/N2). |
| pool                | string                 | Negative pool: "N1" | "N2". |
| confuser_category   | string                 | N2 confuser category if applicable. |
| nobs_z_bin          | string                 | Binned nobs_z. |
| type                | string                 | Tractor type (e.g. SER, DEV, REX). |
| type_bin            | string                 | Binned type. |
| healpix_128         | double                 | HEALPix index (nside=128). |
| psfsize_r           | double                 | PSF size (r). |
| psfdepth_r          | double                 | PSF depth (r). |
| galdepth_r          | double                 | Galaxy depth (r). |
| ebv                 | double                 | E(B-V). |

---

## Script defaults to align with

- **Label column:** `label` (int64): 0 or 1.
- **Split column:** `split` (string): "train" | "val" | "test".
- **Cutout path column:** `cutout_path` (string): path to .npz file.
- **Sample weight column:** `sample_weight` (double): use for weighted loss.
- **Tier column (optional):** `tier` (string): for stratum / held-out Tier-A evaluation.
- **ID for disjointness checks:** `galaxy_id` (string) or `cutout_path` (string); both unique per row.

---

## Type mapping (for implementers)

- **string** → PyArrow `string`; Pandas `object` or `string`.
- **int64** → PyArrow `int64`; Pandas `int64`.
- **double** → PyArrow `double`; Pandas `float64`.
- **bool** → PyArrow `bool`; Pandas `bool`.

All types above are nullable in Parquet unless the column has no nulls; scripts should handle possible nulls for optional columns (e.g. `tier`, `pool`).

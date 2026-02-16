# DHS Experiment Matrix (Gen5/Prime + Unpaired + Residual) - Reference Implementation

Implements a short, fail-fast experiment matrix for strong-lens detection training on:
- Paired vs unpaired data construction (LRG-disjoint)
- Matching of unpaired negatives to positives in bins of observing/galaxy properties
- Shortcut "gates" (core-only LR AUC, radial-profile-only AUC)
- Optional residual-image pipeline (azimuthal-median radial profile subtraction) and a model trained on residuals

## Data expectations (edit column names if yours differ)
Parquet rows must include:
- stamp_npz: bytes (npz) containing image_g, image_r, image_z (float32 64x64)
- ctrl_stamp_npz: bytes (npz) for a control stamp, same bands
- split: train/val/test
- ra, dec: floats identifying the LRG identity
- Matching columns (recommended): mag_z, psf_fwhm_r, sky_mad_r, size_proxy

## Commands
Build matched unpaired manifest:
python -m dhs.scripts.build_unpaired_manifest \
  --parquet <PAIRED_PARQUET_PATH_OR_S3_URI> \
  --out ./manifests/unpaired_matched.parquet \
  --bins "mag_z,psf_fwhm_r,sky_mad_r,size_proxy"

Train:
python -m dhs.scripts.run_experiment --config configs/unpaired_matched_residual_hardneg.yaml

Run shortcut gates (core LR + radial-profile LR):
python -m dhs.scripts.run_gates --config configs/unpaired_matched_residual_hardneg.yaml --split test --n 2048

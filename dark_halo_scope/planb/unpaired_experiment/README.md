# Unpaired Training Experiment

This directory contains code for the unpaired training experiment,
adapted from external LLM suggestions with critical bug fixes.

## Key differences from external code:
1. Gates use cross-validation (no train-on-train bias)
2. Manifest builder adapted for our directory-based split structure
3. Uses our available columns: psf_fwhm_used_r, psfdepth_r, depth_bin, psf_bin

## Usage:
```bash
# 1. Build unpaired manifest
python -m planb.unpaired_experiment.build_manifest \
  --data-root /home/ubuntu/data/v5_cosmos_paired \
  --output /home/ubuntu/data/unpaired_manifest.parquet \
  --seed 42

# 2. Run gates on existing data (sanity check)
python -m planb.unpaired_experiment.run_gates \
  --data-root /home/ubuntu/data/v5_cosmos_paired \
  --split train \
  --n-samples 2000

# 3. Train unpaired model
python -m planb.unpaired_experiment.train \
  --manifest /home/ubuntu/data/unpaired_manifest.parquet \
  --output-dir /home/ubuntu/checkpoints/unpaired_exp \
  --preprocessing raw_robust \
  --epochs 30
```

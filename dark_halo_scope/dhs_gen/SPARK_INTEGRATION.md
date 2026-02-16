# Spark Integration Guide: Gen6 / Gen7 / Gen8 / Uber Mixer

This bundle provides *incremental modules* that you integrate into your existing Stage 4c injection pipeline.

## Overview
- Gen6: deep-field source templates (HSC/Legacy deep cutouts)
- Gen7: hybrid parametric sources (Sérsic + clumps)
- Gen8: domain randomization artifacts
- Uber mixer: assign per-row source_mode + artifact_profile in Phase 4a manifests

## Why assign modes in Phase 4a
- Reproducibility: the same task always maps to the same synthetic recipe.
- Auditability: you can stratify selection functions by recipe.
- Scientific rigor: ablation tables become trivial.

## Step 1: Add mode columns to a manifest parquet (offline / local example)
```bash
pip install pyarrow numpy
python -m dhs_gen.uber.mixer --in-parquet in.parquet --out-parquet out.parquet --mode uber
```

Separate generations:
```bash
python -m dhs_gen.uber.mixer --in-parquet in.parquet --out-parquet gen6.parquet --mode gen6
python -m dhs_gen.uber.mixer --in-parquet in.parquet --out-parquet gen7.parquet --mode gen7
python -m dhs_gen.uber.mixer --in-parquet in.parquet --out-parquet gen8.parquet --mode gen8
```

## Step 2: Stage 4c integration pattern (pseudocode)
In your Stage 4c UDF:
- read row.source_mode
- if deep: load deep bank NPZ once per executor and select template by hash(task_id)
- if hybrid: generate template by task_id hash (no external files)
- if cosmos: use your GalSim COSMOS path

After you produce the stamp, apply Gen8 artifacts depending on row.artifact_profile.

## Distributing deep bank NPZ on EMR
Best practice is a bootstrap action:
- `aws s3 cp s3://<bucket>/deep_bank_20k_96px.npz /mnt/deep_bank.npz`

Then stage_4c reads `--deep-bank-npz /mnt/deep_bank.npz`.

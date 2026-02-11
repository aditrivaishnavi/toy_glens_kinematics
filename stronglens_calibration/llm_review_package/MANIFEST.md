# Manifest: Training and Evaluation Code Package

**Purpose:** This package contains the code used for (1) generating the training manifest, (2) training the ResNet-18 baseline (resnet18_baseline_v1), and (3) running evaluation (calibration and held-out Tier-A metrics). Each file is described below with its intended behavior. Use this to verify that implementation matches intent and to spot issues that could derail honest conclusions.

**Prompt scope:** The accompanying `LLM_PROMPT_TRAINING_AND_EVAL_REVIEW.md` also asks for: (a) answers to meticulous-operations questions (environment/versions, eval JSON, FPR by N2, preprocessing checks, path portability); (b) **complete code or blueprint for the selection function** (injection-recovery grid with frozen model, completeness with Bayesian binomial intervals, and a list of things to watch out for). See **PART E2** and **PART H** in the prompt.

**Layout:** Paths are relative to the stronglens_calibration repo root. The zip may preserve a subfolder (e.g. `dhs/`, `configs/`) for structure.

---

## Config

| File | Intended behavior |
|------|--------------------|
| **configs/resnet18_baseline_v1.yaml** | Training config: dataset manifest path, file_manifest mode, raw_robust preprocessing, label/sample_weight/cutout_path columns, seed 1337; augment (hflip, vflip, rot90); train (epochs 50, batch 256, lr 0.0003, weight_decay 0.0001, early_stopping_patience 10, out_dir, mixed_precision). No data_root; paths in YAML are absolute on Lambda. |

---

## Data loading and preprocessing

| File | Intended behavior |
|------|--------------------|
| **dhs/constants.py** | Constants: STAMP_SIZE=64, CUTOUT_SIZE=101, BANDS=(g,r,z), CORE_BOX, SEED_DEFAULT. Used by data, preprocess, gates. |
| **dhs/data.py** | DatasetConfig, SplitConfig, LensDataset. In file_manifest mode: reads parquet from manifest_path, filters by split_col/split_value, loads each row’s cutout from cutout_path (NPZ with key "cutout"), applies preprocess_stack and optional augmentation; returns (image, label, sample_weight). Order of rows is deterministic (parquet order after filter). |
| **dhs/preprocess.py** | center_crop to 64×64; preprocess_stack(mode='raw_robust'): outer-annulus (r=20–32 px) median/MAD normalization per band, then clip to ±clip_range. Same logic at train and eval. |
| **dhs/transforms.py** | AugmentConfig; random_augment applies hflip/vflip/rot90 based on seed. No brightness/contrast. Eval uses AugmentConfig with all False. |

---

## Model and training

| File | Intended behavior |
|------|--------------------|
| **dhs/model.py** | build_resnet18(in_ch=3): torchvision resnet18(weights=None), conv1 replaced for in_ch, fc → 1 output. |
| **dhs/train.py** | TrainConfig; train_one(): builds train/val LensDatasets (val with no aug), DataLoader with weighted collate for file_manifest; AdamW, CosineAnnealingLR, BCEWithLogitsLoss(reduction='none') with per-sample weights; AMP; saves last.pt every epoch and best.pt when val AUC improves; early stop after patience epochs without improvement. evaluate() runs model on val loader and returns AUC (no weights in metric). |
| **dhs/scripts/run_experiment.py** | Loads config YAML, builds DatasetConfig and TrainConfig, calls train_one(), prints best path and AUC. Entry point for training. |

---

## Evaluation (calibration and Tier-A)

| File | Intended behavior |
|------|--------------------|
| **dhs/calibration.py** | ECE: equal-frequency bins on y_prob, ECE = sum_b (n_b/n)*\|acc(b)-conf(b)\|. MCE: max over bins. reliability_curve(): same bins, returns bin_edges, bin_acc, bin_conf, bin_counts. |
| **dhs/scripts/run_evaluation.py** | Loads config and checkpoint (best.pt), builds LensDataset(split=test, no aug), runs inference in order (shuffle=False); reads manifest again, filters to test, aligns by length to y_true; computes overall AUC, recall@0.5, precision@0.5, ECE, MCE, reliability curve; if manifest has "tier", reports by stratum (tier_A, tier_B). Does **not** report ECE/MCE for single-class strata (sets to null, adds note). Copies tier_A block to "independent_validation_spectroscopic" with description that Tier-A is held-out, not a separate spectroscopic catalog. Writes JSON and optional summary markdown. |

---

## Manifest generation (upstream of training)

| File | Intended behavior |
|------|--------------------|
| **scripts/generate_training_manifest_parallel.py** | Scans directories of .npz cutouts (positives + negatives), extracts metadata from each NPZ (including meta_* fields); assigns tier from metadata (positives: A/B, negatives: N1/N2) and sample_weight (A=1.0, B=0.5, N1/N2=1.0). Writes parquet with cutout_path, label, split (if in meta), tier, sample_weight, etc. Split may come from NPZ metadata; if missing, may need separate assignment step. |

---

## Other supporting code (included for completeness)

| File | Intended behavior |
|------|--------------------|
| **dhs/utils.py** | Helpers used by preprocess (e.g. normalize_outer_annulus, radial_profile_model) and gates (center_slices, azimuthal_median_profile). |
| **dhs/gates.py** | run_shortcut_gates(): core-only and radial-profile logistic regression on pixel features; used for shortcut checks (4.7/4.8). Not run in the reported eval; included for completeness. |
| **constants.py** (repo root) | Pipeline constants (S3, paths, AB_ZEROPOINT_MAG=22.5 for DR10). Used by EMR/scripts; dhs uses dhs/constants.py for STAMP_SIZE etc. |

---

## Documentation (in zip)

| File | Content |
|------|--------|
| **docs/MODELS_PERFORMANCE.md** | Model card for resnet18_baseline_v1: identification, architecture, data, training run, validation metrics, evaluation metrics, artifacts, reproduce steps, open items. |
| **docs/EVALUATION_HONEST_AUDIT.md** | Honest audit: what the eval code does correctly; limitations (independent validation wording, ECE dominated by majority class, stratum ECE misleading, split not verified, no uncertainty); recommendations for rigor. |
| **docs/DATA_AND_VERIFICATION.md** | S3 and local data locations, sync steps, Phase 4 (S4) check table, test run results. |

---

## What is NOT in this zip

- Actual data (parquet, NPZ, checkpoints). Paths in config and docs point to Lambda NFS or S3.
- EMR/Spark scripts for negative sampling or cutout generation (those are upstream; we only include manifest generation from existing cutouts).
- Tests (test_phase1_local.py, test_pipeline_local.py, etc.); can be added in a second package if needed.

---

## Quick check list for reviewer

- [ ] train.py: val set used for early stopping only; test never used in training or checkpoint selection.
- [ ] run_evaluation.py: same preprocessing as training; no augmentation; DataLoader shuffle=False so order matches manifest.
- [ ] run_evaluation.py: tier alignment uses df_split.iloc[:len(y_true)] so i-th prediction matches i-th test row.
- [ ] calibration.py: ECE/MCE formulas match standard definitions (equal-frequency bins).
- [ ] data.py: file_manifest loads by path from parquet; split filter applied; no train data in test split by construction of manifest.

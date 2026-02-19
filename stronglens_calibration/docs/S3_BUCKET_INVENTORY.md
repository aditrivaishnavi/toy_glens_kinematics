# S3 Bucket Inventory: `s3://darkhaloscope/`

Last verified: 2026-02-18 (3,827,309 objects total)

## Top-level prefix map

### `stronglens_calibration/` — Paper IV (morphological barrier) project

This is the primary project directory for the MNRAS paper.

| Prefix | Contents | Approx files |
|--------|----------|-------------|
| `checkpoints/` | Trained CNN model weights (`.pt`), training logs, `run_info.json` for each experiment | 136 |
| `configs/` | YAML training configs, `injection_priors.yaml`, negative-sampling configs | ~20 |
| `cutouts/negatives/` | Negative (non-lens) cutout `.npz` files, timestamped batches | ~1,279,000 |
| `cutouts/positives/` | Positive (real lens) cutout `.npz` files | ~9,600 |
| `data/` | External data, negative catalogs, test outputs | misc |
| `manifests/` | Training/validation split manifests (`.parquet`, `.json`) | ~8 |
| `results/` | All experiment results (D01–D06), evaluations, figures | ~234,000 |
| `code/` | Mirror of the Python codebase (`dhs/`, `scripts/`, `configs/`, etc.) | ~200 |
| `dhs/` | Core library: `data.py`, `train.py`, `injection_engine.py`, `s3io.py`, etc. | ~15 |
| `scripts/` | Standalone scripts (selection function, sensitivity, sync, etc.) | ~20 |
| `docs/` | Markdown docs (MNRAS notes, injection validation, training log) | ~5 |
| `llm_review_package/` | LLM review prompts and code packages | ~10 |
| `inference/` | Inference-time code | ~2 |
| `training/` | Training harness code | ~2 |
| `injection_model_1/`, `injection_model_2/` | Injection model code (Sérsic parametric) | misc |
| `sim_to_real_validations/` | Sim-to-real comparison results | misc |
| `paper/` | Only `__init__.py` (paper .tex files are in the git repo, not S3) | 1 |
| `logs/` | Experiment run logs (`d04_log.txt`, `d05_log.txt`, `d06_log.txt`) | 3 |

#### `stronglens_calibration/checkpoints/` — model zoo

| Model | Description |
|-------|-------------|
| `paperIV_bottlenecked_resnet/` | **Primary model used in the paper.** Bottlenecked ResNet, 160 epochs. Contains `best.pt`. |
| `paperIV_efficientnet_v2_s/` | EfficientNetV2-S first run |
| `paperIV_efficientnet_v2_s_v2/` | EfficientNetV2-S v2 (longer training) |
| `paperIV_efficientnet_v2_s_v3_cosine/` | EfficientNetV2-S v3 with cosine LR schedule |
| `paperIV_efficientnet_v2_s_v4_finetune/` | EfficientNetV2-S v4 fine-tuned |
| `paperIV_resnet18/` | ResNet-18 baseline |
| `resnet18_baseline_v1/` | Earlier ResNet-18 baseline |
| `gen5_prime_baseline/` | Gen5 prime model |
| `ablation_minimal/`, `ablation_no_coredrop/`, `ablation_no_hardneg/` | Ablation studies |
| `*_paired_residual/`, `*_unpaired_residual/` | Paired/unpaired experiments (Plan B) |

#### `stronglens_calibration/results/` — experiment results

| Directory | Description |
|-----------|-------------|
| `D06_20260216_corrected_priors/` | **Final production run** (229,926 files). Corrected injection priors. Contains grid (no-Poisson and Poisson), bright-arc experiments (6 variants), linear probe, tier-A scoring, gallery, comparison figure. |
| `D05_20260214_full_reeval/` | Pre-correction full evaluation |
| `D04_20260214_matched_comparison/` | Matched comparison tests |
| `D03_20260214_poisson_grid/` | Poisson grid experiment |
| `D02_20260214_prompt5_quick_tests/` | Quick diagnostic tests |
| `D01_20260214_pre_retrain_diagnostics/` | Pre-retrain diagnostics |
| `recovered_injections/` | 2,604 saved injection `.npz` cutouts (bright-arc recovery) |
| `bright_arc_v2_corrected_priors/` | Bright-arc results with corrected priors (1,605 files) |
| `comparison_figure/` | Real vs injected visual comparison assets (23 files) |
| `injection_validation_v4/`, `injection_validation_v4_anchor/` | Injection validation diagnostics |
| `sensitivity_v4/`, `sensitivity_v4_corrected/` | Sensitivity analysis results |
| `sim_to_real_validation/` | Sim-to-real validation |
| `paper_figures/` | Generated paper figures |
| `tier_ab_probe_control/`, `tier_ab_probe_control_v2/` | Tier A/B probe control experiments |
| `selection_function_*/` | Selection function evaluations |
| `host_conditioning_diagnostic/` | Host-conditioning diagnostic |
| `rerun_*/` | Bug-fix reruns |

#### `stronglens_calibration/results/D06_20260216_corrected_priors/` — final run detail

| Subdirectory | Files | Description |
|---|---|---|
| `grid_no_poisson/` | 110,003 | Grid injection results without Poisson noise (110,000 cutouts + metadata) |
| `grid_poisson/` | 110,003 | Grid injection results with Poisson noise |
| `ba_baseline/` | 1,605 | Bright-arc baseline (no augmentation) |
| `ba_poisson/` | 1,605 | Bright-arc with Poisson noise |
| `ba_clip20/` | 1,605 | Bright-arc with clip at 20 |
| `ba_gain_1e12/` | 1,605 | Bright-arc with extreme gain |
| `ba_poisson_clip20/` | 1,605 | Bright-arc with Poisson + clip |
| `ba_unrestricted/` | 1,605 | Bright-arc unrestricted |
| `gallery/` | 273 | HTML gallery of real/injected cutouts for visual inspection |
| `comparison_figure/` | 8 | Figure 5 assets |
| `linear_probe/` | 2 | Linear probe results (AUC, embeddings) |
| `tier_a_scoring/` | 2 | Tier-A lens CNN scores |
| `poisson_diagnostics/` | 3 | Poisson noise diagnostic plots |
| `analysis/` | 1 | `d06_analysis_summary.json` |
| `provenance.json` | 1 | Git SHA, config hash, timestamp for reproducibility |

---

### `dr10/` — DESI Legacy Survey DR10 data

| Prefix | Contents |
|--------|----------|
| `coadd_cache/` | Cached coadd FITS tiles (image, invvar, psfsize per band) organized by brick name |
| `sweeps/` | DR10 sweep catalogs |
| `sweeps_manifest/` | Manifest files for sweep processing |
| `survey-bricks-dr10-south.fits.gz` | Brick definition file |
| `dr10-south-depth-summary.fits.gz` | Depth summary |
| `sweep_urls.txt` | URLs for sweep downloads |

### `dark_halo_scope/` — Dark Halo Scope project (earlier project)

Debug and development artifacts from the earlier dark-halo-scope project.

### `data/` — shared data

| Prefix | Contents |
|--------|----------|
| `cosmos_banks/` | COSMOS galaxy stamp banks for injection |

### `models/` — trained models (earlier generations)

| Prefix | Contents |
|--------|----------|
| `gen2_50epochs/` | Gen2 model (50 epochs) |
| `gen3_moffat_v2/` | Gen3 Moffat PSF model |
| `gen4_hardneg/` | Gen4 with hard negatives |
| `gen5_cosmos/` | Gen5 COSMOS-based model |
| `gen5_prime/` | Gen5 prime model |

### `scores/` — inference scores

Parquet files of CNN scores on train/test sets for various model generations.

### `anchor_cutouts/`, `anchor_baseline/` — anchor lens evaluation

Known-lens and hard-negative anchor catalogs, cutouts, and evaluation results.

### `hard_negatives/` — hard negative mining

Parquet files of top-scoring false positives from various rounds, used for hard-negative training.

### `planb/` — Plan B experiments

Paired/unpaired residual experiments (ablations, baselines, manifests).

### `runs/` — training runs (earlier generations)

TensorBoard-style training run logs for gen2/gen3/gen4 models.

### `eval/`, `evaluation/` — evaluation outputs

Stratified FPR CSVs, evaluation scripts, catalogs.

### `logs/` — pipeline logs

Coadd sync logs, training logs, inference logs.

### EMR-related prefixes

| Prefix | Contents |
|--------|----------|
| `emr-code/` | Spark pipeline code for EMR (phase 4 pipeline) |
| `emr-scripts/` | EMR bootstrap/step scripts |
| `emr_code/` | Additional EMR code (paired controls, phase 3.5, phase 4a) |
| `emr_bootstrap/` | EMR bootstrap scripts (COSMOS bank builder) |
| `emr-logs/`, `emr_logs/` | EMR cluster logs |

### Phase pipeline prefixes (earlier data engineering)

| Prefix | Contents |
|--------|----------|
| `phase1p5_*/` | Phase 1.5: LRG sampling pipeline (test/debug/prod runs) |
| `phase2_results/`, `phase2_analysis/` | Phase 2: LRG hypergrid, intermediate results |
| `phase3/`, `phase3_pipeline/` | Phase 3: parent sample construction |
| `phase4/` | Phase 4: hard-negative mining code |
| `phase5/`, `phase5_models/`, `phase5_results/` | Phase 5: arc SNR rejection sampling, models |

### Root-level files

| File | Description |
|------|-------------|
| `unpaired_manifest_v1_full.parquet` | Full unpaired training manifest (~68 GB) |
| `stage0_anchor_baseline.py`, `stage0_preflight.py` | Stage-0 validation scripts |
| `test_stage0_anchor_baseline.py` | Tests for stage-0 |
| `rclone_gen5_copy.log` | Gen5 data copy log |
| `gate_1_*.json` | Phase 1 validation gate results (8 gates) |
| `phase*_*.json` | Phase result JSONs |
| `phase1_llm_report.md` | Phase 1 LLM review report |

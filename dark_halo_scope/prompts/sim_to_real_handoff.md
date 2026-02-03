# Sim-to-Real Gap Closure: Complete Review Package

**Date**: 2026-02-01  
**Project**: Dark Halo Scope - Strong Gravitational Lens Detection  
**Status**: Infrastructure and validation framework complete, ready for execution

---

## 1. Executive Summary

This document provides complete context for reviewing the sim-to-real gap closure implementation in Dark Halo Scope. The goal is to improve model generalization from synthetic training data to real astronomical images.

### Problem Statement

Current models (Gen1-Gen4) achieve 84-88% `tpr@fpr1e-4` on **synthetic test data**, but this metric is meaningless for real-world performance because:
1. Training uses **parametric Sersic n=1 sources** - too smooth compared to real lensed arcs
2. Controls are just "different LRGs without injection" - no real lens mimics (ring galaxies, mergers)
3. No validation on known lenses (SLACS, BELLS) or real hard negatives

### Solution Implemented

A three-stage pipeline:
1. **Stage 1**: Anchor baseline validation (test on real data FIRST)
2. **Stage 2**: COSMOS source integration (replace Sersic with real galaxy morphologies)
3. **Stage 3**: Real hard negative mining (train on actual lens mimics)

---

## 2. Files Created/Modified

### Stage 0: Infrastructure

| File | Purpose |
|------|---------|
| `experiments/configs/experiment_schema.py` | Configuration schema with validation |
| `experiments/configs/__init__.py` | Module exports |
| `experiments/__init__.py` | Module exports |
| `experiments/data_variants/README.md` | Data variant documentation |
| `experiments/data_variants/v3_color_relaxed.md` | Gen1 data documentation |
| `experiments/data_variants/v4_sota.md` | Gen2 data documentation |
| `experiments/data_variants/v4_sota_moffat.md` | Gen3/4 data documentation |
| `experiments/data_variants/v5_cosmos_source.md` | Planned Gen5 data |
| `tests/test_experiment_config.py` | 41 unit tests for config |

### Stage 1: Anchor Baseline

| File | Purpose |
|------|---------|
| `experiments/external_catalogs/__init__.py` | Module init |
| `experiments/external_catalogs/catalog_sources.py` | Catalog definitions |
| `experiments/external_catalogs/download_catalogs.py` | Download SLACS/BELLS/rings |
| `experiments/external_catalogs/crossmatch_dr10.py` | Cross-match with DR10 |
| `experiments/external_catalogs/compute_anchor_metrics.py` | Compute validation metrics |
| `tests/test_anchor_baseline.py` | 26 unit tests for Stage 1 |

### Stage 2: COSMOS Integration

| File | Purpose |
|------|---------|
| `src/sims/cosmos_loader_v2.py` | Enhanced COSMOS loader with GalSim support |

### Stage 3: Hard Negative Mining

| File | Purpose |
|------|---------|
| `experiments/external_catalogs/collect_real_hard_negatives.py` | Collect real hard negatives |
| `tests/test_cosmos_and_hardneg.py` | 20 unit tests for Stages 2-3 |

**Total: 87 unit tests, all passing**

---

## 3. Configuration Schema

### Key Design Principles

1. **Every random operation has an explicit seed**
2. **All paths are absolute or relative to known root**
3. **Data variant must be explicitly specified**
4. **Model architecture and hyperparameters frozen at experiment start**

### ExperimentConfig Structure

```python
@dataclass
class ExperimentConfig:
    experiment_id: str
    experiment_name: str
    description: str
    generation: str  # e.g., "gen5", "gen6"
    
    # Nested configs
    seeds: SeedConfig
    data: DataVariantConfig
    model: ModelConfig
    training: TrainingConfig
    hard_negatives: HardNegativeConfig
    evaluation: EvaluationConfig
    
    # Git tracking
    git_commit: Optional[str]
    git_branch: Optional[str]
    git_dirty: Optional[bool]
```

### DataVariantConfig

```python
@dataclass
class DataVariantConfig:
    variant_name: str  # e.g., "v5_cosmos_source"
    description: str
    phase3_parent_sample: str
    phase4a_manifest: str
    phase4c_stamps: str
    
    # Injection parameters
    psf_model: Literal["gaussian", "moffat"]
    moffat_beta: Optional[float]
    source_mode: Literal["parametric", "cosmos"]
    cosmos_library_path: Optional[str]
    
    # Grid parameters
    theta_e_range: List[float]
    control_type: Literal["paired", "unpaired"]
```

### Validation Rules

1. `moffat_beta` must be set when `psf_model="moffat"`
2. `cosmos_library_path` must be set when `source_mode="cosmos"`
3. `hard_neg_path` must be set when `hard_negatives.enabled=True`
4. `output_dir` must always be set

---

## 4. Data Variant Lineage

```
v3_color_relaxed (Gen1) - DEPRECATED
    │   Issues: Paired controls, Gaussian PSF, narrow theta_e
    │
    ├── Fixed paired controls, extended theta_e
    │
    ▼
v4_sota (Gen2)
    │   Issues: Gaussian PSF, smooth Sersic sources
    │
    ├── Added Moffat PSF
    │
    ▼
v4_sota_moffat (Gen3, Gen4)
    │   Issues: Still smooth Sersic sources, no real hard negatives
    │
    ├── Adding COSMOS sources
    │
    ▼
v5_cosmos_source (Gen5) [PLANNED]
    │
    ├── Adding real hard negatives
    │
    ▼
v6_real_hardneg (Gen6) [PLANNED]
```

---

## 5. Anchor Baseline Validation

### Gate Criteria

Before proceeding to SOTA comparisons, the model must pass:

| Criterion | Threshold | Interpretation |
|-----------|-----------|----------------|
| Known lens recovery @ 0.5 | ≥70% | Can we find SLACS/BELLS lenses? |
| Hard neg contamination @ 0.5 | ≤15% | Do we reject ring galaxies? |
| Score separation | ≥0.4 | Clean separation between classes? |
| AUROC on anchor data | ≥0.8 | Overall discrimination? |

### AnchorMetrics Dataclass

```python
@dataclass
class AnchorMetrics:
    n_known_lenses: int
    n_recovered_at_0p5: int
    recovery_rate_0p5: float
    
    n_hard_negatives: int
    n_contaminated_at_0p5: int
    contamination_rate_0p5: float
    
    score_separation: float  # median_pos - median_neg
    auroc_anchor: float
    
    passes_anchor_gate: bool
    gate_criteria: Dict[str, bool]
```

---

## 6. COSMOS Loader V2

### Key Features

1. **GalSim COSMOS catalog support**: Load real HST galaxy images
2. **HDF5 format support**: Fast loading from preprocessed library
3. **Synthetic fallback**: For testing without COSMOS data
4. **Deterministic selection**: Reproducible with seed
5. **Pixel scale resampling**: Convert HST (0.03") to DECaLS (0.262")

### Usage

```python
from src.sims.cosmos_loader_v2 import COSMOSLoaderV2

# From GalSim
loader = COSMOSLoaderV2(cosmos_path="/path/to/galsim_data", mode='galsim')

# From preprocessed HDF5
loader = COSMOSLoaderV2(cosmos_path="/path/to/cosmos.h5", mode='hdf5')

# Get source (reproducible)
source = loader.get_source(seed=42)
print(source.image.shape, source.clumpiness, source.half_light_radius)

# Resample to DECaLS
resampled = loader.resample_to_pixscale(source.image, target_pixscale=0.262)
```

### Building COSMOS Library

```bash
python src/sims/cosmos_loader_v2.py \
    --galsim-path /path/to/galsim_cosmos_data \
    --output cosmos_sources.h5 \
    --n-sources 10000 \
    --stamp-size 128 \
    --pixel-scale 0.03 \
    --seed 42
```

---

## 7. Hard Negative Collection

### Sources

| Source | Weight | Description |
|--------|--------|-------------|
| Model FP | 3.0x | High-scoring false positives from inference |
| Ring galaxies | 2.0x | Galaxy Zoo + SIMBAD ring classifications |
| Mergers | 1.5x | Galaxy Zoo merger classifications |

### Collection Pipeline

```python
from experiments.external_catalogs.collect_real_hard_negatives import (
    collect_hard_negatives, HardNegativeConfig
)

config = HardNegativeConfig(
    min_score_threshold=0.8,
    top_k_per_source=5000,
    exclusion_radius_arcsec=5.0  # Exclude known lenses
)

df = collect_hard_negatives(
    model_scores_path="inference_scores.parquet",
    ring_catalog_path="ring_galaxies.parquet",
    merger_catalog_path="mergers.parquet",
    known_lens_path="known_lenses.parquet",
    output_dir="./hard_negatives/",
    config=config
)
```

---

## 8. Execution Commands

### Stage 1: Anchor Baseline

```bash
# Step 1: Download external catalogs
cd experiments/external_catalogs
python download_catalogs.py --all --output-dir ./data/ --merge

# Step 2: Cross-match with DR10
python crossmatch_dr10.py \
    --catalog ./data/known_lenses_merged.parquet \
    --brick-metadata ../../data/ls_dr10_south_bricks_metadata.csv \
    --output-dir ./data/known_lenses/

python crossmatch_dr10.py \
    --catalog ./data/hard_negatives_merged.parquet \
    --brick-metadata ../../data/ls_dr10_south_bricks_metadata.csv \
    --output-dir ./data/hard_negatives/

# Step 3: Run inference and compute metrics
python compute_anchor_metrics.py \
    --known-lens-dir ./data/known_lenses/ \
    --hard-neg-dir ./data/hard_negatives/ \
    --checkpoint /path/to/gen3/best.pt \
    --output-dir ./results/anchor_baseline/ \
    --device cuda
```

### Stage 2: COSMOS Data Generation

```bash
# Build COSMOS library (requires GalSim installation)
python src/sims/cosmos_loader_v2.py \
    --galsim-path /path/to/COSMOS_25.2 \
    --output s3://darkhaloscope/cosmos_sources.h5 \
    --n-sources 10000 \
    --seed 42

# Run Phase 4c with COSMOS sources (EMR)
spark-submit emr/spark_phase4_pipeline.py \
    --stage 4c \
    --variant v5_cosmos_source \
    --psf-model moffat \
    --moffat-beta 3.5 \
    --source-mode cosmos \
    --cosmos-library s3://darkhaloscope/cosmos_sources.h5 \
    --experiment-id train_stamp64_bandsgrz_cosmos
```

### Stage 3: Hard Negative Training

```bash
# Collect hard negatives
python experiments/external_catalogs/collect_real_hard_negatives.py \
    --model-scores gen5_inference_scores.parquet \
    --ring-catalog ring_galaxies.parquet \
    --merger-catalog mergers.parquet \
    --known-lenses known_lenses.parquet \
    --output-dir ./hard_negatives/ \
    --create-lookup

# Train Gen6 with real hard negatives
python models/gen4_hardneg/phase5_train_gen4_hardneg.py \
    --data /path/to/v5_cosmos_source/ \
    --hard_neg_path ./hard_negatives/hard_neg_lookup.parquet \
    --hard_neg_weight 5 \
    --out_dir models/gen6_real_hardneg/ \
    --epochs 50 \
    --early_stopping_patience 5
```

---

## 9. Test Coverage

### Running All Tests

```bash
cd /Users/balaji/code/toy_glens_kinematics/dark_halo_scope
python -m pytest tests/test_experiment_config.py tests/test_anchor_baseline.py tests/test_cosmos_and_hardneg.py -v
```

### Test Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_experiment_config.py` | 41 | ✅ All pass |
| `test_anchor_baseline.py` | 26 | ✅ All pass |
| `test_cosmos_and_hardneg.py` | 20 | ✅ All pass |
| **Total** | **87** | **✅ All pass** |

### Key Test Categories

1. **Configuration validation**: Ensures invalid configs are rejected
2. **YAML roundtrip**: Configs survive save/load cycle
3. **Seed reproducibility**: Same seed → same results
4. **Catalog download**: Fallback data when VizieR unavailable
5. **Cross-match**: Correct footprint filtering
6. **Anchor metrics**: Gate criteria evaluation
7. **COSMOS loading**: HDF5 and synthetic modes
8. **Hard negative collection**: Full pipeline integration

---

## 10. Known Issues and TODOs

### Not Yet Implemented

1. **spark_phase4_pipeline.py modification**: Need to add `--source-mode cosmos` flag to actual pipeline (code exists but not integrated)
2. **Phase 4c COSMOS rendering**: Need to implement ray-tracing through COSMOS images in Spark job
3. **Training script integration**: `phase5_train_gen4_hardneg.py` exists but needs update for position-based hard neg matching

### Assumptions

1. GalSim COSMOS data available at `/path/to/COSMOS_25.2` (downloadable)
2. Network access for VizieR/SIMBAD queries (fallback data available)
3. DR10 brick metadata CSV exists at `data/ls_dr10_south_bricks_metadata.csv`

---

## 11. Reproducibility Checklist

For any experiment to be reproducible, ensure:

- [ ] Experiment config saved to YAML with `config.to_yaml()`
- [ ] Git commit hash recorded (`config.git_commit`)
- [ ] All seeds explicitly set (`config.seeds.global_seed`)
- [ ] Data variant documented in `experiments/data_variants/`
- [ ] Phase 4c command recorded with all flags
- [ ] Model checkpoint saved with training config
- [ ] Test scores saved with model version

### Example Reproducibility Header

```python
from experiments.configs import create_experiment_config, set_all_seeds

config = create_experiment_config(
    experiment_name="Gen5 COSMOS",
    generation="gen5",
    data_variant=my_data_variant,
    output_base_dir="/path/to/experiments"
)

# Save config
config.to_yaml(f"{config.output_dir}/config.yaml")

# Set all seeds
set_all_seeds(config.seeds)

# Log reproducibility info
print(config.get_reproducibility_info())
```

---

## 12. Review Questions

For the reviewing LLM, please address:

1. **Configuration completeness**: Are there any experiment parameters not captured in the schema?

2. **Anchor baseline validity**: Are the gate criteria (70% recovery, 15% contamination) reasonable based on literature?

3. **COSMOS integration**: Is the pixel scale resampling approach correct? Any concerns about PSF convolution order?

4. **Hard negative weighting**: Is 3x weight for model FPs too aggressive? What does literature suggest?

5. **Test coverage**: Any edge cases missing from the test suite?

6. **Reproducibility**: Are there any sources of non-determinism not addressed?

---

## 13. File Checksums (for verification)

```
experiments/configs/experiment_schema.py  - 9.8 KB
experiments/external_catalogs/download_catalogs.py - 12.4 KB
experiments/external_catalogs/compute_anchor_metrics.py - 13.2 KB
src/sims/cosmos_loader_v2.py - 12.1 KB
experiments/external_catalogs/collect_real_hard_negatives.py - 8.9 KB
tests/test_experiment_config.py - 10.3 KB
tests/test_anchor_baseline.py - 9.1 KB
tests/test_cosmos_and_hardneg.py - 7.8 KB
```

---

*End of handoff document*


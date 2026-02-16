# Plan B: Gen5-Prime + Gen7/Gen8 Ablation Suite

## Overview

This directory contains all code for **Option B**: the most promising path for original, defensible research on strong gravitational lens detection.

**Core thesis:** "Which aspects of source realism and imaging nuisance variation most affect sim-to-real transfer and the inferred selection function for lens finding in seeing-limited survey imaging?"

---

## Directory Structure

```
planb/
├── README.md                     # This file
├── CONTRACTS.md                  # Contract definitions (MUST READ)
├── EXIT_CRITERIA.md              # Explicit pass/fail criteria
├── LESSONS_LEARNED_INTEGRATION.md# How past mistakes are addressed
├── LLM_REVIEW_BUILD_PLAN.md      # Document for LLM review
│
├── shared/                       # SINGLE SOURCE OF TRUTH - import from here
│   ├── __init__.py
│   ├── constants.py              # All magic numbers defined here
│   ├── schema.py                 # All schema definitions
│   └── utils.py                  # All shared utility functions
│
├── configs/
│   ├── evaluation_protocol.yaml  # LOCKED evaluation metrics (Phase 0)
│   └── gen5_prime_baseline.yaml  # Baseline training config
│
├── phase0_foundation/
│   ├── __init__.py
│   ├── run_all_phase0.py         # Run all foundation checks
│   ├── validate_anchors.py       # Validate anchor catalog
│   ├── validate_contaminants.py  # Validate contaminant catalog
│   ├── verify_split_integrity.py # Verify train/val/test splits
│   └── verify_paired_data.py     # Verify paired data quality
│
├── phase1_baseline/
│   ├── __init__.py
│   ├── train.py                  # Training script ✓
│   ├── data_loader.py            # Data loading ✓
│   ├── model.py                  # Model definition ✓
│   └── validate_baseline.py      # Post-training validation ✓
│
├── phase2_gen7/                  # TBD
├── phase3_gen8/                  # TBD
│
├── phase4_evaluation/
│   └── aggregate_results.py      # Aggregate all ablation results
│
├── emr/                          # EMR infrastructure (reusable)
│   ├── __init__.py
│   ├── constants.py              # EMR/Spark configuration
│   ├── launcher.py               # Job submission CLI
│   ├── bootstrap.sh              # Cluster bootstrap script
│   ├── preflight_check.py        # Pre-submission validation
│   ├── sync_to_launcher.sh       # Sync code to EMR-launcher
│   ├── spark_job_template.py     # Template for new Spark jobs
│   ├── STANDARD_STEPS.md         # Standard workflow documentation
│   └── jobs/                     # Spark jobs by phase
│       ├── spark_verify_splits.py    # Phase 0: Split integrity
│       ├── spark_verify_paired.py    # Phase 0: Paired data verification
│       ├── spark_gen7_injection.py   # Phase 2: Gen7 data generation
│       ├── spark_gen8_injection.py   # Phase 3: Gen8 domain randomization
│       └── spark_score_all.py        # Phase 4: Large-scale scoring
│
└── tests/
    ├── __init__.py
    ├── test_contracts.py         # Contract verification tests ✓
    ├── test_core_functions.py    # Core function unit tests ✓
    ├── test_data_loading.py      # Data loading integration tests
    └── test_training_loop.py     # Training loop unit tests
```

### Import Rules (CRITICAL)

All phase code MUST import from `shared/` module:

```python
# Correct
from shared.constants import STAMP_SIZE, CORE_RADIUS_PIX
from shared.schema import PARQUET_SCHEMA
from shared.utils import decode_stamp_npz, validate_stamp

# WRONG - do NOT redefine
STAMP_SIZE = 64  # NO!
def decode_stamp_npz(): pass  # NO!
```

---

## Quick Start

### Phase 0: Foundation (MUST COMPLETE FIRST)

```bash
# Run all Phase 0 checks
python phase0_foundation/run_all_phase0.py \
    --parquet-root /path/to/v5_cosmos_paired \
    --anchor-csv anchors/tier_a_anchors.csv \
    --contaminant-csv contaminants/catalog.csv \
    --output-dir phase0_results

# All checks MUST pass before proceeding
```

### Phase 1: Baseline Training

```bash
# Train baseline with all mitigations
python phase1_baseline/train.py \
    --config configs/gen5_prime_baseline.yaml \
    --output-dir checkpoints/gen5_prime_baseline

# Validate baseline
python phase1_baseline/validate_baseline.py \
    --checkpoint checkpoints/gen5_prime_baseline/best_model.pt \
    --parquet-root /path/to/v5_cosmos_paired
```

### Phase 4: Aggregate Results

```bash
# After all ablations complete
python phase4_evaluation/aggregate_results.py \
    --checkpoint-dir checkpoints/ \
    --output-csv results/master_comparison.csv
```

### EMR Jobs (for large-scale processing)

```bash
# 1. Run pre-flight check
python emr/preflight_check.py --fix

# 2. Sync code to EMR-launcher (if using dedicated machine)
./emr/sync_to_launcher.sh emr-launcher

# 3. Run smoke test (small cluster)
python emr/launcher.py submit \
    --job-name smoke-test \
    --script emr/spark_job_template.py \
    --preset small \
    --args "--input s3://... --output s3://..." \
    --wait

# 4. Run production job
python emr/launcher.py submit \
    --job-name production-job \
    --script your_job.py \
    --preset medium \
    --args "--input s3://... --output s3://..." \
    --wait

# 5. Monitor/manage clusters
python emr/launcher.py list
python emr/launcher.py status --cluster-id j-XXXXX
python emr/launcher.py terminate --cluster-id j-XXXXX
```

See `emr/STANDARD_STEPS.md` for full EMR workflow documentation.

---

## Guiding Principles

1. **No step proceeds without validation of the previous step**
2. **All decisions are logged with rationale**
3. **Failures are documented, not hidden**
4. **Every metric has a pre-specified interpretation**
5. **Code is tested before expensive runs**

---

## Gate Criteria

### Phase 0 Gates
- [ ] Anchor set validated (≥30 anchors, all checks pass)
- [ ] Contaminant set validated (categories populated, no overlap)
- [ ] Split integrity verified (no brick overlap)
- [ ] Paired data verified (shapes, values correct)
- [ ] All tests pass

### Phase 1 Gates
- [ ] AUROC_synth > 0.85
- [ ] Core_LR_AUC < 0.65
- [ ] Core_masked_drop < 10%
- [ ] Hard_neg_AUROC > 0.70

### Phase 4 Gates
- [ ] All ablations complete
- [ ] Results reproducible (3 seeds)
- [ ] Bootstrap CIs computed
- [ ] Figures generated

---

## Timeline

| Phase | Days | Status |
|-------|------|--------|
| Phase 0: Foundation | 1-2 | Ready |
| Phase 1: Baseline + Ablations | 3-5 | Ready |
| Phase 2: Gen7 | 6-8 | Code TBD |
| Phase 3: Gen8 | 9-11 | Code TBD |
| Phase 4: Evaluation | 12-14 | Ready |

---

## Implementation Status

### Completed ✓
- `shared/constants.py` - All magic numbers and thresholds
- `shared/schema.py` - All schema definitions
- `shared/utils.py` - All shared utility functions
- `phase0_foundation/*.py` - All foundation validation scripts
- `phase1_baseline/data_loader.py` - Data loading with paired sampling
- `phase1_baseline/model.py` - Model architecture
- `phase1_baseline/train.py` - Main training loop with gates
- `phase1_baseline/validate_baseline.py` - Post-training validation
- `tests/test_contracts.py` - Contract verification tests
- `tests/test_core_functions.py` - Core function unit tests
- `tests/test_training_loop.py` - Training loop unit tests

### Remaining (TBD)
1. `phase2_gen7/fix_gen7_semantics.py` - Fix clump flux fraction
2. `phase2_gen7/train_gen7.py` - Gen7 training
3. `phase3_gen8/calibrate_artifacts.py` - Calibrate DR10 artifact rates
4. `phase3_gen8/train_gen8.py` - Gen8 training
5. `phase4_evaluation/selection_function.py` - Selection function analysis
6. `phase4_evaluation/generate_figures.py` - Paper figure generation

---

## Related Documents

- `docs/BUILD_PLAN_OPTION_B.md` - Full build plan with all details
- `docs/research_plan_1_shortcut_mitigation.md` - Research plan for shortcut mitigation
- `docs/research_plan_2_gen7_procedural_realism.md` - Gen7 research plan
- `docs/research_plan_3_gen8_domain_randomization.md` - Gen8 research plan
- `docs/evaluation_protocol.yaml` - Locked evaluation metrics

# Plan B: Unpaired Training for Strong Lens Detection

## Overview

This directory contains the implementation for training a gravitational lens classifier that does NOT rely on shortcuts (core brightness leakage).

**Core insight:** Unpaired training + residual preprocessing eliminates the core brightness shortcut that plagued paired training approaches.

---

## Quick Links

| Document | Purpose |
|----------|---------|
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | **Current status** - what's done, in progress, pending |
| [WORKLOG.md](WORKLOG.md) | Detailed session log with findings |

---

## Directory Structure

```
planb/
├── README.md                    # This file
├── PROJECT_STATUS.md            # Current project status (single source of truth)
├── WORKLOG.md                   # Ongoing work log
│
├── configs/                     # Training configurations
│   └── evaluation_protocol.yaml
│
├── unpaired_experiment/         # Main training code (active)
│   ├── train.py                 # Training script
│   ├── data_loader.py           # Dataset classes
│   ├── preprocess.py            # Residual preprocessing
│   ├── gates.py                 # Shortcut detection gates
│   ├── build_manifest.py        # Manifest builders
│   └── README.md                # Module documentation
│
├── phase0_foundation/           # Data validation scripts
├── phase1_baseline/             # Original baseline (historical)
├── phase4_evaluation/           # Results aggregation
│
├── emr/                         # EMR/Spark infrastructure
│   └── STANDARD_STEPS.md        # EMR workflow
│
├── shared/                      # Shared constants and utils
│   ├── constants.py
│   ├── schema.py
│   └── utils.py
│
├── tests/                       # Unit tests (50 tests)
│
└── docs/
    ├── reference/               # Governance docs
    │   ├── CODE_ORGANIZATION.md
    │   ├── CONTRACTS.md
    │   ├── EXIT_CRITERIA.md
    │   └── LESSONS_LEARNED_INTEGRATION.md
    │
    └── archive/                 # Historical investigation docs
        ├── EXTERNAL_LLM_REVIEW_ARCHIVE.md
        ├── CORE_LEAKAGE_INVESTIGATION.md
        └── ... (other historical docs)
```

---

## Current Experiments (Feb 7, 2026)

| Exp | Config | GPU | Status |
|-----|--------|-----|--------|
| A1 | Paired + Residual | lambda | Running (~30 hrs) |
| B1 | Unpaired + Residual | lambda4 | Running (~16 hrs) |
| B2 | Unpaired + Residual + dropout r=5 | lambda2 | Running (~16 hrs) |
| B3 | Unpaired + Residual + scheduled | lambda3 | Running (~16 hrs) |
| B4 | Unpaired + Residual + dropout r=3 | lambda5 | Running (~16 hrs) |

---

## Key Concepts

### The Problem: Core Brightness Shortcut

In paired training, the model achieves 95% AUC using only the central 10x10 pixels because:
1. Lensed arc flux overlaps with lens galaxy core
2. The same LRG appears in positive (with arc) and negative (without arc)
3. Model learns "brighter core = lens" instead of arc morphology

### The Solution: Unpaired + Residual

1. **Unpaired training**: Different LRGs for positives and negatives (breaks paired shortcut)
2. **Residual preprocessing**: Subtract azimuthal median profile (removes radial brightness signal)
3. **Gates**: Core LR AUC < 0.65 and Radial LR AUC < 0.55

### Mini Experiment Validation

| Config | Core AUC | Result |
|--------|----------|--------|
| Paired + Residual | 0.93 | FAIL |
| **Unpaired + Residual** | **0.50** | **PASS** |

---

## Running Training

```bash
# On remote GPU instance
cd ~/dark_halo_scope

# Unpaired training with residual preprocessing
python -m planb.unpaired_experiment.train \
    --manifest /path/to/manifest.parquet \
    --output-dir /path/to/checkpoints \
    --preprocess residual_radial_profile \
    --epochs 50 \
    --seed 42
```

---

## Running Tests

```bash
cd /path/to/dark_halo_scope
python -m pytest planb/tests/ -v
```

---

## Reference Documents

| Document | Purpose |
|----------|---------|
| [docs/reference/CONTRACTS.md](docs/reference/CONTRACTS.md) | API contracts between modules |
| [docs/reference/EXIT_CRITERIA.md](docs/reference/EXIT_CRITERIA.md) | Pass/fail criteria for each phase |
| [docs/reference/CODE_ORGANIZATION.md](docs/reference/CODE_ORGANIZATION.md) | Code structure and best practices |
| [docs/reference/LESSONS_LEARNED_INTEGRATION.md](docs/reference/LESSONS_LEARNED_INTEGRATION.md) | How past mistakes are prevented |

---

## Historical Documents

Archived investigation documents are in `docs/archive/`. Key ones:
- `EXTERNAL_LLM_REVIEW_ARCHIVE.md` - Consolidated external review
- `CORE_LEAKAGE_INVESTIGATION.md` - Root cause analysis
- `EXPERIMENT_PLAN_V2.md` - Original experiment plan

---

*Last updated: 2026-02-07*

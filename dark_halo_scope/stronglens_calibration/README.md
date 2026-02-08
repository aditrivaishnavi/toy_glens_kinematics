# Strong Lens Calibration

**Goal**: Real-image detector + calibrated injection selection function for DR10 strong lens searches.

**Paper Thesis**: "Given a real-image-trained lens finder, we measure its DR10 selection function via calibrated injection-recovery, quantify failure modes, and provide bias-aware guidance for lens demographics and follow-up prioritization."

## Directory Structure

```
stronglens_calibration/
├── configs/              # Frozen configuration files (NEVER edit after run)
├── emr/                  # EMR Spark jobs for data processing
├── experiments/          # Timestamped experiment runs
├── data/                 # Data files (symlinks to main data/)
│   ├── positives/        # DESI imaging candidates
│   ├── negatives/        # Negative galaxy samples
│   ├── external/         # External catalogs (spectroscopic, etc.)
│   └── cutouts/          # Image cutouts
├── training/             # Model training code
├── evaluation/           # Evaluation and metrics
├── tests/                # Unit tests
├── docs/                 # Documentation
└── common/               # Shared utilities
```

## Key Principles (from lessons_learned)

1. **Never edit configs after a run** - configs are immutable experiment records
2. **Track everything** - git commit, timestamps, all parameters
3. **Verify before trust** - check file existence, data quality, code correctness
4. **Audit against LLM recommendations** - cross-check with conversation_with_llm.txt
5. **Fail fast** - quality gates at every stage

## Phase Roadmap

- **Phase 0**: Data acquisition (DESI imaging candidates, spectroscopic catalog) - DONE
- **Phase 1**: EMR negative sampling with proper schema
- **Phase 2**: Label handling with tier-based weighting
- **Phase 3**: Injection realism calibration
- **Phase 4**: Training with spatial splits
- **Phase 5**: Selection function measurement
- **Phase 6**: Failure mode analysis
- **Phase 7**: Paper deliverables

## Key Data Assets

| Asset | Count | Location |
|-------|-------|----------|
| DESI imaging candidates | 5,104 | `data/positives/desi_candidates.csv` |
| Tier-A (confident) | 435 | grading="confident" in above |
| DESI spectroscopic catalog | 2,176 | `data/external/desi_dr1/desi-sl-vac-v1.fits` |
| Independent validation | ~781 | Spectroscopic without imaging overlap |

## Created

- **Date**: 2026-02-07
- **Purpose**: Option 1 (Hybrid) implementation from LLM blueprint

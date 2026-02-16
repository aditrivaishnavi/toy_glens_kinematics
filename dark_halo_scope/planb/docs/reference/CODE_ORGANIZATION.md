# Code Organization Guide

**Date**: 2026-02-06
**Status**: Reviewed and Updated

---

## 1. Directory Structure

```
planb/
├── __init__.py                     # Package root
├── README.md                       # High-level documentation
├── CODE_ORGANIZATION.md            # This file
├── WORKLOG.md                      # Audit log
│
├── shared/                         # Shared utilities (SINGLE SOURCE OF TRUTH)
│   ├── __init__.py
│   ├── constants.py                # Global constants
│   ├── schema.py                   # Data schemas
│   └── utils.py                    # Core utilities
│
├── unpaired_experiment/            # Current experimental work
│   ├── __init__.py
│   ├── README.md                   # Module documentation
│   ├── constants.py                # Module-specific constants
│   ├── utils.py                    # Module utilities
│   ├── preprocess.py               # Preprocessing functions
│   ├── gates.py                    # Shortcut detection gates
│   ├── scheduled_mask.py           # Scheduled core masking
│   ├── thetae_stratification.py    # θ_E stratified evaluation
│   ├── data_loader.py              # PyTorch datasets
│   ├── build_manifest.py           # Manifest creation
│   ├── run_gates.py                # CLI for running gates
│   └── train.py                    # Training script
│
├── phase0_foundation/              # Data validation
├── phase1_baseline/                # Baseline training (legacy)
├── phase4_evaluation/              # Evaluation scripts
├── emr/                            # EMR/Spark jobs
│
├── tests/                          # Unit and integration tests
│   ├── __init__.py
│   ├── test_core_functions.py      # Core function tests
│   ├── test_unpaired_experiment.py # Unpaired experiment tests
│   ├── test_contracts.py           # Contract tests
│   ├── test_data_loading.py        # Data loading tests
│   └── test_training_loop.py       # Training loop tests
│
├── configs/                        # YAML configurations
│   ├── gen5_prime_baseline.yaml
│   └── evaluation_protocol.yaml
│
└── docs/                           # Markdown documentation
    ├── EXPERIMENT_PLAN_V2.md
    ├── EXPERIMENT_CHECKLIST.md
    └── ...
```

---

## 2. Best Practices Applied

### 2.1 Constants Management (L1.2)

**Rule**: Single source of truth for constants.

```python
# GOOD: Import from constants module
from .constants import STAMP_SIZE, SEED_DEFAULT

# BAD: Hardcoded values
size = 64  # Magic number
```

**Applied in**: All modules use `constants.py` for shared values.

### 2.2 Reproducibility (L2.1)

**Rule**: Fixed seeds everywhere.

```python
# train.py
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Applied in**: `train.py`, `build_manifest.py`, `gates.py`, all test files.

### 2.3 Error Handling (L3.1)

**Rule**: Graceful handling of edge cases.

```python
# GOOD: Handle NaN gracefully
def robust_median_mad(x, eps=1e-8):
    valid = x[~np.isnan(x)]
    if len(valid) == 0:
        return 0.0, 1.0  # Safe defaults
    ...

# GOOD: Handle single class in classification
if len(np.unique(y)) < 2:
    return 0.5  # Random chance
```

**Applied in**: `utils.py`, `gates.py`, `thetae_stratification.py`.

### 2.4 Testing (L4.3)

**Rule**: Test actual code paths, not mocks.

```python
# GOOD: Integration test
def test_preprocessing_then_gates(self):
    xs, ys = make_separable_data()
    processed = np.stack([preprocess_stack(x) for x in xs])
    results = run_shortcut_gates(processed, ys)
    assert results.core_lr_auc > 0.6
```

**Applied in**: `test_unpaired_experiment.py` with 50 comprehensive tests.

### 2.5 Documentation (L5.1)

**Rule**: Docstrings with Args, Returns, and edge cases.

```python
def auroc_by_thetae(
    y_true: np.ndarray,
    y_score: np.ndarray,
    theta_e: np.ndarray,
    bins: List[Tuple[float, float]],
    min_count: int = 100,
    seed: int = 42,
) -> List[ThetaEBinResult]:
    """
    Compute AUROC for each theta_E bin.
    
    For each bin:
    1. Select positives with theta_E in [lo, hi)
    2. Randomly sample equal number of negatives
    3. Compute AUROC on this balanced subset
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_score: Model prediction scores
        theta_e: theta_E values (NaN for negatives)
        bins: List of (lo, hi) tuples defining bins
        min_count: Minimum positives per bin to compute AUC
        seed: Random seed for reproducibility
    """
```

### 2.6 Type Hints (L6.1)

**Rule**: Use type hints for all public functions.

```python
def preprocess_stack(img3: np.ndarray, mode: str = "raw_robust") -> np.ndarray:
    ...
```

**Applied in**: All modules in `unpaired_experiment/`.

---

## 3. Module Dependencies

```
constants.py ← (no dependencies)
     ↓
utils.py ← constants
     ↓
preprocess.py ← utils, constants
     ↓
gates.py ← constants, utils
     ↓
thetae_stratification.py ← (numpy, sklearn only)
     ↓
scheduled_mask.py ← (torch only)
     ↓
data_loader.py ← constants, utils, preprocess
     ↓
train.py ← data_loader, gates, scheduled_mask, thetae_stratification
```

**No circular dependencies.**

---

## 4. Critical Bugs Fixed in This Review

| Bug | File | Fix |
|-----|------|-----|
| theta_e not in manifest | build_manifest.py | Added theta_e_arcsec to manifest rows |
| θ_E stratification excludes negatives | thetae_stratification.py | Sample negatives for each bin |
| Test set never evaluated | train.py | Added full test evaluation |
| Metrics JSON not saved | train.py | Added JSON output |
| STAMP_SIZE=64 (wrong) | constants.py | Changed to 63 |
| Early stopping=10 (wrong) | constants.py | Changed to 15 |
| run_core_sensitivity_test returns None | train.py | Returns dict |

---

## 5. Running Tests

```bash
# On Lambda instance (has sklearn)
cd ~/dark_halo_scope
python3 -c "from planb.tests.test_unpaired_experiment import run_all_tests; run_all_tests()"

# With pytest (if installed)
pytest planb/tests/test_unpaired_experiment.py -v
```

---

## 6. Running Experiments

### 6.1 Manifest Locations (Source of Truth)

**S3**: `s3://darkhaloscope/planb/manifests/v2/`
**NFS**: `/lambda/nfs/darkhaloscope-training-dc/planb_manifests/v2/`

| Manifest | Size | Description |
|----------|------|-------------|
| `full/` | 13 MB | 1.57M rows (metadata-only, partitioned) |
| `mini_10pct/` | 4 MB | 157K rows (10% stratified sample) |
| `paired_mini_10pct_files.txt` | 11 KB | 100 file paths for A1 experiment |

### 6.2 Building Manifests (V2 - Metadata Only)

```bash
cd ~/dark_halo_scope

# Build full manifest (metadata only, ~50 seconds)
python3 -m planb.unpaired_experiment.build_manifest_v2 \
    --data-root /home/ubuntu/data/v5_cosmos_paired \
    --output-dir /home/ubuntu/data/unpaired_manifest_v2 \
    --seed 42
```

### 6.3 Running Experiments

```bash
# Run experiment B1 (unpaired + residual)
python3 -m planb.unpaired_experiment.train \
    --manifest /home/ubuntu/data/unpaired_manifest_v2_mini_10pct \
    --preprocessing residual_radial_profile \
    --output-dir /home/ubuntu/checkpoints/mini_B1_unpaired_residual \
    --epochs 50 \
    --seed 42 \
    --mixed-precision

# Run experiment A1 (paired + residual)
python3 -m planb.unpaired_experiment.train \
    --data-root /home/ubuntu/data/v5_cosmos_paired \
    --file-list /home/ubuntu/data/paired_mini_10pct_files.txt \
    --preprocessing residual_radial_profile \
    --output-dir /home/ubuntu/checkpoints/mini_A1_paired_residual \
    --epochs 50 \
    --seed 42 \
    --mixed-precision
```

---

## 7. Checklist for New Code

Before adding new code to `planb/`:

- [ ] Constants in `constants.py`, not hardcoded
- [ ] Fixed seed for reproducibility
- [ ] Type hints on public functions
- [ ] Docstrings with Args/Returns
- [ ] NaN/edge case handling
- [ ] Unit tests added to `tests/`
- [ ] No circular imports
- [ ] Linter passes (`ruff check .`)

---

## 8. Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-06 | Initial audit and organization | AI Assistant |
| 2026-02-06 | Fixed 7 critical bugs | AI Assistant |
| 2026-02-06 | Added 50 unit tests | AI Assistant |

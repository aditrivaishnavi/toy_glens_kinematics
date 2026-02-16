# Plan B Codebase Review Request

## Context

I am building a **strong gravitational lens detection pipeline** for astronomical surveys (DESI Legacy Survey DR10). This is intended for publication in MNRAS/ApJ/AAS.

**Core thesis:** "Which aspects of source realism and imaging nuisance variation most affect sim-to-real transfer and the inferred selection function for lens finding in seeing-limited survey imaging?"

The attached `planb_code_package.zip` contains the complete codebase.

---

## What I Need Reviewed

Please provide a **thorough, critical review** of the entire codebase. I need you to:

1. **Identify bugs** - syntax errors, logic errors, off-by-one errors, incorrect assumptions
2. **Identify inconsistencies** - schema mismatches, contract violations, magic numbers not using constants
3. **Identify missing error handling** - especially for boto3/S3 calls (must have retries, must not silently eat exceptions)
4. **Identify scientific/methodological issues** - incorrect physics, flawed evaluation metrics, data leakage risks
5. **Provide fix snippets** - for every issue found, provide the corrected code

---

## Codebase Structure

```
planb/
├── shared/                       # SINGLE SOURCE OF TRUTH
│   ├── constants.py              # All magic numbers
│   ├── schema.py                 # All data schemas
│   └── utils.py                  # All shared utilities
│
├── configs/
│   ├── evaluation_protocol.yaml  # LOCKED evaluation metrics
│   └── gen5_prime_baseline.yaml  # Training config
│
├── phase0_foundation/            # Data validation before training
│   ├── validate_anchors.py
│   ├── validate_contaminants.py
│   ├── verify_split_integrity.py
│   ├── verify_paired_data.py
│   └── run_all_phase0.py
│
├── phase1_baseline/              # Training pipeline
│   ├── data_loader.py            # Paired dataset + augmentations
│   ├── model.py                  # ResNet18 classifier
│   ├── train.py                  # Training loop with gates
│   └── validate_baseline.py      # Post-training validation
│
├── phase2_gen7/                  # (Placeholder for Gen7 hybrid sources)
├── phase3_gen8/                  # (Placeholder for Gen8 domain randomization)
│
├── phase4_evaluation/
│   └── aggregate_results.py      # Aggregate ablation results
│
├── emr/                          # AWS EMR infrastructure
│   ├── aws_utils.py              # S3/EMR clients with retries
│   ├── constants.py              # EMR configuration
│   ├── launcher.py               # Job submission CLI
│   ├── preflight_check.py        # Pre-submission validation
│   ├── bootstrap.sh              # Cluster bootstrap
│   ├── STANDARD_STEPS.md         # Workflow documentation
│   └── jobs/                     # Spark jobs by phase
│       ├── spark_verify_splits.py
│       ├── spark_verify_paired.py
│       ├── spark_gen7_injection.py
│       ├── spark_gen8_injection.py
│       └── spark_score_all.py
│
├── tests/
│   ├── test_contracts.py         # Contract verification
│   ├── test_core_functions.py    # Unit tests
│   ├── test_data_loading.py      # Integration tests
│   └── test_training_loop.py     # Training tests
│
├── CONTRACTS.md                  # Contract definitions
├── EXIT_CRITERIA.md              # Pass/fail criteria
├── LESSONS_LEARNED_INTEGRATION.md
└── README.md
```

---

## Key Technical Details

### Data Format
- **Input**: Parquet files with columns `stamp_npz`, `ctrl_stamp_npz`, `theta_e_arcsec`, `arc_snr`, etc.
- **Stamps**: 3-channel (g, r, z bands), 64x64 pixels, stored as compressed NPZ blobs
- **Paired training**: Each positive sample has a matched control (same galaxy, no arc injection)

### Model
- ResNet18 backbone, modified for 3-channel 64x64 input
- Binary classification (lens vs non-lens)
- Shortcut mitigations: core dropout, azimuthal-shuffle hard negatives, paired controls

### Key Constraints
- **AWS credentials expire after 24 hours** - code must detect and STOP (not silently fail)
- **vCore budget: 284** - EMR clusters must stay within this limit
- **Region: us-west-2** - must be consistent everywhere
- **S3 bucket: darkhaloscope** - parameterized

### Lessons Learned (Must Be Applied)
1. **L3.1**: Always validate cluster/data readiness before operations
2. **L5.4**: Syntax check all code before uploads/runs
3. **L6.1**: Never silently eat exceptions - log and re-raise
4. **L7.2**: Use shared constants, not hardcoded magic numbers
5. **L8.1**: Verify schema consistency across phases

---

## Specific Review Questions

### 1. Shared Module (`shared/`)
- Are all constants defined correctly?
- Are there any magic numbers in phase code that should use shared constants?
- Are the schema validations complete?
- Are utility functions handling edge cases (empty arrays, NaN values)?

### 2. Data Loader (`phase1_baseline/data_loader.py`)
- Is the paired sampling implemented correctly?
- Are augmentations (core dropout, azimuthal shuffle) applied correctly?
- Is normalization consistent with the schema?
- Any data leakage risks?

### 3. Training Loop (`phase1_baseline/train.py`)
- Is the gate checking logic correct (core_lr_auc < 0.65, not > 0.65)?
- Is early stopping implemented correctly?
- Are all metrics computed correctly?
- Is the learning rate scheduler appropriate?

### 4. Validation (`phase1_baseline/validate_baseline.py`)
- Are the core-masked evaluations implemented correctly?
- Is the hard negative evaluation correct?
- Do the gate thresholds match the config?

### 5. EMR Infrastructure (`emr/`)
- Do ALL boto3 calls have proper retries?
- Are credential expiry errors detected and handled (STOP, not silent failure)?
- Are vCore calculations correct?
- Is the bootstrap script installing all required packages?

### 6. Spark Jobs (`emr/jobs/`)
- Is the Gen7 injection physics correct (Sersic + clumps + lensing)?
- Is the Gen8 domain randomization realistic?
- Are there any race conditions or partition issues?
- Is error handling complete?

### 7. Tests (`tests/`)
- Do tests actually test the contract constraints?
- Are there missing test cases?
- Will tests catch the bugs they're meant to catch?

---

## Output Format

For each issue found, provide:

```
### Issue N: [Category] Short description

**File**: path/to/file.py
**Line(s)**: XX-YY
**Severity**: Critical / High / Medium / Low

**Problem**:
Description of the issue

**Current Code**:
```python
# problematic code
```

**Fixed Code**:
```python
# corrected code
```

**Explanation**:
Why this fix is correct
```

---

## Final Deliverables

1. **Summary table** of all issues found (file, severity, category)
2. **Fix snippets** for every issue
3. **Overall assessment**: Is this codebase production-ready? What are the top 3 risks?
4. **Recommendation**: GO / NO-GO for training with this codebase

---

## Attached Files

- `planb_code_package.zip` - Complete codebase (all files listed above)
- `docs/BUILD_PLAN_OPTION_B.md` - Full build plan with phase details

Please be thorough and critical. This is for publication and I need to catch all issues before expensive training runs.

# Gen5 Pipeline: Automatic Config Tracking to S3

## Overview

The Gen5 pipeline (`emr/gen5/spark_phase4_pipeline_gen5.py`) now automatically saves a JSON configuration file to S3 whenever Stage 4c runs. This ensures complete reproducibility and audit trails for all data generation runs.

## Key Features

### 1. **Automatic Config Saving**

When Stage 4c runs, the pipeline automatically saves the effective configuration to S3:

```
s3://{output_s3}/phase4c/{variant}/run_config_{experiment_id}.json
```

**Example path:**
```
s3://darkhaloscope/phase4/phase4c/v5_cosmos_source/run_config_train_stamp64_bandsgrz_gridgrid_sota.json
```

### 2. **Config Contents**

The saved JSON includes all critical parameters:

```json
{
  "stage": "4c",
  "variant": "v5_cosmos_source",
  "experiment_id": "train_stamp64_bandsgrz_gridgrid_sota",
  "source_mode": "cosmos",
  "cosmos_bank_h5": "s3://darkhaloscope/cosmos/cosmos_bank_20k_parametric_v1.h5",
  "cosmos_salt": "gen5_v1",
  "seed_base": 42,
  "psf_model": "moffat",
  "moffat_beta": 3.5,
  "split_seed": 99887766,
  "execution_timestamp_utc": "2026-02-02T10:30:45Z",
  "spark_version": "3.1.2"
}
```

### 3. **New Command-Line Arguments**

#### Config File Loading
```bash
--config PATH           # Load all settings from JSON config (local or s3://)
```

When `--config` is provided, it overrides all other command-line arguments.

#### COSMOS Source Integration
```bash
--source-mode {sersic|cosmos}     # Source morphology type (default: sersic)
--cosmos-bank-h5 PATH             # Path to COSMOS bank HDF5 (required if cosmos mode)
--cosmos-salt STRING              # Salt for deterministic template selection
--seed-base INT                   # Base seed for reproducibility (default: 42)
```

### 4. **Schema Updates**

Three new columns added to the Parquet output schema:

| Column | Type | Description |
|--------|------|-------------|
| `source_mode` | String | "sersic" or "cosmos" |
| `cosmos_index` | Integer | Index of COSMOS template used (null for sersic) |
| `cosmos_hlr_arcsec` | Double | Half-light radius of lensed COSMOS arc (null for sersic) |

These columns allow downstream analysis to:
- Filter by source type
- Track which COSMOS templates were used
- Measure morphological properties of lensed arcs

## Usage Examples

### Example 1: Run with COSMOS sources (command-line args)

```bash
spark-submit emr/gen5/spark_phase4_pipeline_gen5.py \
  --stage 4c \
  --output-s3 s3://darkhaloscope/phase4 \
  --variant v5_cosmos_source \
  --experiment-id train_stamp64_bandsgrz_gridgrid_sota \
  --source-mode cosmos \
  --cosmos-bank-h5 s3://darkhaloscope/cosmos/cosmos_bank_20k_parametric_v1.h5 \
  --cosmos-salt gen5_v1 \
  --seed-base 42 \
  --psf-model moffat \
  --moffat-beta 3.5 \
  --split-seed 99887766 \
  --tiers train \
  --n-total-train-per-split 200000 \
  --sweep-partitions 800
```

**Automatic output:**
- Parquet stamps with COSMOS metadata
- Config JSON at: `s3://darkhaloscope/phase4/phase4c/v5_cosmos_source/run_config_train_stamp64_bandsgrz_gridgrid_sota.json`

### Example 2: Run with config file (recommended)

**Step 1:** Create config file `configs/gen5_cosmos_v1.json`:
```json
{
  "stage": "4c",
  "variant": "v5_cosmos_source",
  "experiment_id": "train_stamp64_bandsgrz_gridgrid_sota",
  "source_mode": "cosmos",
  "cosmos_bank_h5": "s3://darkhaloscope/cosmos/cosmos_bank_20k_parametric_v1.h5",
  "cosmos_salt": "gen5_v1",
  "seed_base": 42,
  "psf_model": "moffat",
  "moffat_beta": 3.5,
  "split_seed": 99887766
}
```

**Step 2:** Upload to S3:
```bash
aws s3 cp configs/gen5_cosmos_v1.json s3://darkhaloscope/configs/
```

**Step 3:** Run pipeline:
```bash
spark-submit emr/gen5/spark_phase4_pipeline_gen5.py \
  --config s3://darkhaloscope/configs/gen5_cosmos_v1.json \
  --output-s3 s3://darkhaloscope/phase4 \
  --tiers train \
  --n-total-train-per-split 200000 \
  --sweep-partitions 800
```

The pipeline will:
1. Load config from S3
2. Override command-line args with config values
3. Run Stage 4c with COSMOS sources
4. Save effective config to S3 (with execution timestamp)

### Example 3: Backward compatibility (Sersic sources)

The Gen5 pipeline is fully backward compatible. To run Gen3/Gen4-style Sersic sources:

```bash
spark-submit emr/gen5/spark_phase4_pipeline_gen5.py \
  --stage 4c \
  --output-s3 s3://darkhaloscope/phase4 \
  --variant v4_sota_moffat \
  --experiment-id train_stamp64_bandsgrz_gridgrid_sota \
  --psf-model moffat \
  --moffat-beta 3.5 \
  --split-seed 99887766 \
  --tiers train
```

**Output:**
- `source_mode` = "sersic"
- `cosmos_index` = null
- `cosmos_hlr_arcsec` = null
- Config saved to S3 with `source_mode: "sersic"`

## Implementation Details

### Config Save Location

The config is saved in the `main()` function after argument parsing:

```python
# Gen5: Save effective config to S3 for audit trail (Stage 4c only)
if stage == "4c" and _is_s3(args.output_s3):
    effective_config = {
        "stage": args.stage,
        "variant": args.variant,
        # ... all relevant args ...
        "execution_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spark_version": spark.version,
    }
    
    config_s3_path = f"{args.output_s3}/phase4c/{args.variant}/run_config_{args.experiment_id}.json"
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(effective_config, indent=2))
```

### COSMOS Template Selection

Deterministic selection based on `task_id` and optional salt:

```python
def _cosmos_choose_index(task_id: str, n_sources: int, salt: str = "") -> int:
    """Deterministic COSMOS template selection from task_id."""
    import hashlib
    h = hashlib.sha256((task_id + salt).encode("utf-8")).hexdigest()
    seed = int(h[:16], 16)
    return seed % n_sources
```

**Benefits:**
- Same `task_id` always gets same template
- Different `salt` values create different runs with same tasks
- No random state required (fully deterministic)

### Injection Logic

The injection loop now supports both Sersic and COSMOS modes:

```python
source_mode = getattr(args, 'source_mode', 'sersic')

if source_mode == "cosmos":
    # Load COSMOS bank (cached per executor)
    cosmos_bank = _load_cosmos_bank_h5(args.cosmos_bank_h5)
    cosmos_idx = _cosmos_choose_index(task_id, cosmos_bank["n_sources"], args.cosmos_salt)
    
    # Inject COSMOS source for each band
    for b in use_bands:
        add_b = render_cosmos_lensed_source(
            cosmos_bank=cosmos_bank,
            cosmos_index=cosmos_idx,
            # ... lens params ...
        )
        imgs[b] = (imgs[b] + add_b).astype(np.float32)
else:
    # Original Sersic injection (Gen3/Gen4)
    for b in use_bands:
        add_b = render_lensed_source(
            # ... Sersic params ...
        )
        imgs[b] = (imgs[b] + add_b).astype(np.float32)
```

## Audit and Tracking

### Finding Configs for Past Runs

**List all configs for a variant:**
```bash
aws s3 ls s3://darkhaloscope/phase4/phase4c/v5_cosmos_source/
```

**Download specific config:**
```bash
aws s3 cp s3://darkhaloscope/phase4/phase4c/v5_cosmos_source/run_config_train_stamp64_bandsgrz_gridgrid_sota.json .
```

### Reproducing a Run

**Step 1:** Download the config:
```bash
aws s3 cp s3://darkhaloscope/phase4/phase4c/v5_cosmos_source/run_config_train_stamp64_bandsgrz_gridgrid_sota.json \
  ./reproduce_config.json
```

**Step 2:** Re-run with exact same config:
```bash
spark-submit emr/gen5/spark_phase4_pipeline_gen5.py \
  --config ./reproduce_config.json \
  --output-s3 s3://darkhaloscope/phase4_reproduction \
  --force 1
```

The `--force 1` flag ensures idempotency is bypassed for reproduction.

## Benefits

1. **Full Reproducibility**: Every run's exact parameters are preserved
2. **Audit Trail**: Timestamps and Spark versions tracked
3. **Easy Debugging**: Can inspect configs of past runs without parsing logs
4. **Experiment Tracking**: Compare configs across different generations
5. **Automated**: No manual config saving required
6. **Centralized**: All configs in S3, not scattered across laptops

## Migration from Gen3/Gen4

**Gen3/Gen4 runs:**
- Continue to work with Gen5 pipeline (backward compatible)
- Set `--source-mode sersic` (or omit, it's the default)
- Configs still saved to S3 automatically

**Gen5 runs (COSMOS):**
- Set `--source-mode cosmos`
- Provide `--cosmos-bank-h5` path
- Use `--cosmos-salt` to version template selections

## Next Steps

1. ✅ **Done**: Pipeline code updated with config saving
2. **Pending**: Build COSMOS bank HDF5 on emr-launcher
3. **Pending**: Create preflight validation script for Gen5 runs
4. **Pending**: Test single injection with COSMOS template
5. **Pending**: Run full Gen5 Phase 4c on EMR

## Questions?

Refer to:
- Gen5 integration plan: `docs/cosmos_galsim_llm_response_review.md`
- COSMOS bank builder: `models/dhs_cosmos_galsim_code/dhs_cosmos/sims/cosmos_source_loader.py`
- Phase 4 pipeline: `emr/gen5/spark_phase4_pipeline_gen5.py`


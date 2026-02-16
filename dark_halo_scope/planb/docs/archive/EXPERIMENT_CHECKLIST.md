# Experiment Execution Checklist

**Date Created**: 2026-02-06
**Status**: DRAFT - Must complete all setup before any training

---

## 1. Naming Conventions

### Dataset Names
```
unpaired_manifest_full.parquet     # Full unpaired manifest (all 1000 files)
unpaired_manifest_mini_10pct.parquet  # 10% stratified sample for quick runs
paired_manifest_mini_10pct.parquet    # 10% paired data reference (for comparison)
```

### Experiment IDs
```
mini_A1_paired_residual          # Mini run: Paired + Residual
mini_B1_unpaired_residual        # Mini run: Unpaired + Residual
mini_B2_unpaired_residual_r5     # Mini run: Unpaired + Residual + Stochastic r=5
mini_B3_unpaired_residual_sched  # Mini run: Unpaired + Residual + Scheduled masking
mini_B4_unpaired_residual_r3     # Mini run: Unpaired + Residual + Stochastic r=3

full_B1_unpaired_residual        # Full run: Final config (after mini experiments)
```

### Checkpoint Directories
```
/home/ubuntu/checkpoints/mini_A1_paired_residual/
/home/ubuntu/checkpoints/mini_B1_unpaired_residual/
...
/home/ubuntu/checkpoints/full_B1_unpaired_residual/
```

---

## 2. Data Preparation Checklist

### 2.1 Full Unpaired Manifest (from ALL 1000 train files)

| Step | Command | Status | Output |
|------|---------|--------|--------|
| Build manifest | `python3 -m planb.unpaired_experiment.build_manifest --data-root /home/ubuntu/data/v5_cosmos_paired --output /home/ubuntu/data/unpaired_manifest_full.parquet --seed 42` | ☐ Pending | `/home/ubuntu/data/unpaired_manifest_full.parquet` |
| Verify row count | Check train/val/test splits | ☐ Pending | Expected: ~800K train |
| Verify θ_E distribution | Check coverage of [0.5, 3.0] | ☐ Pending | |

### 2.2 Mini Unpaired Manifest (10% stratified sample)

| Step | Command | Status | Output |
|------|---------|--------|--------|
| Create from full | Stratified sample preserving θ_E, PSF, depth | ☐ Pending | `/home/ubuntu/data/unpaired_manifest_mini_10pct.parquet` |
| Verify balance | Check pos/neg ratio per split | ☐ Pending | Expected: 50/50 |
| Verify stratification | Check distributions match full | ☐ Pending | |

### 2.3 Mini Paired Reference (10% for A1 comparison)

| Step | Command | Status | Output |
|------|---------|--------|--------|
| Sample paired files | Sample 100 of 1000 train files | ☐ Pending | List of 100 file paths |
| Create reference manifest | For tracking which files used | ☐ Pending | `/home/ubuntu/data/paired_mini_10pct_files.txt` |

---

## 3. Experiment-Specific Requirements

### 3.1 Experiment A1: Paired + Residual

| Requirement | Value | Status |
|-------------|-------|--------|
| Data source | Paired data (use `--data-root`) | ☐ |
| Manifest needed | NO (uses directory directly) | - |
| Files to use | 100 files (10% of 1000) | ☐ |
| Preprocessing | `residual_radial_profile` | ☐ |
| Core masking | None | ☐ |

### 3.2 Experiment B1: Unpaired + Residual

| Requirement | Value | Status |
|-------------|-------|--------|
| Data source | Unpaired manifest | ☐ |
| Manifest needed | `unpaired_manifest_mini_10pct.parquet` | ☐ |
| Preprocessing | `residual_radial_profile` | ☐ |
| Core masking | None | ☐ |

### 3.3 Experiment B2: Unpaired + Residual + Stochastic r=5

| Requirement | Value | Status |
|-------------|-------|--------|
| Data source | Unpaired manifest | ☐ |
| Manifest needed | `unpaired_manifest_mini_10pct.parquet` | ☐ |
| Preprocessing | `residual_radial_profile` | ☐ |
| Core masking | `--core-dropout-prob 0.5 --core-radius 5` | ☐ |

### 3.4 Experiment B3: Unpaired + Residual + Scheduled Masking

| Requirement | Value | Status |
|-------------|-------|--------|
| Data source | Unpaired manifest | ☐ |
| Manifest needed | `unpaired_manifest_mini_10pct.parquet` | ☐ |
| Preprocessing | `residual_radial_profile` | ☐ |
| Core masking | `--scheduled-masking --schedule "0:7:0.7,10:5:0.5,30:3:0.3"` | ☐ |

### 3.5 Experiment B4: Unpaired + Residual + Stochastic r=3

| Requirement | Value | Status |
|-------------|-------|--------|
| Data source | Unpaired manifest | ☐ |
| Manifest needed | `unpaired_manifest_mini_10pct.parquet` | ☐ |
| Preprocessing | `residual_radial_profile` | ☐ |
| Core masking | `--core-dropout-prob 0.3 --core-radius 3` | ☐ |

---

## 4. Code Preparation Checklist

| Step | Description | Status |
|------|-------------|--------|
| Seeding fixed | All scripts use seed=42 + CUDA deterministic | ✓ Done |
| Sync to lambda | All code synced | ✓ Done |
| Sync to lambda2 | All code synced | ✓ Done |
| Sync to lambda3 | All code synced | ✓ Done |
| θ_E stratified eval | Add to training script | ☐ Pending |
| Metrics JSON output | Add to training script | ☐ Pending |
| Core sensitivity curve | Already in train.py | ✓ Done |

---

## 5. Execution Order

### Phase 1: Data Preparation
```
1. Build unpaired_manifest_full.parquet (all 1000 files)
2. Create unpaired_manifest_mini_10pct.parquet (stratified 10%)
3. Create paired_mini_10pct_files.txt (list of 100 files for A1)
4. Verify all manifests
```

### Phase 2: Mini Experiments (on 10% data)
```
1. Run mini_A1_paired_residual
2. Run mini_B1_unpaired_residual
3. Compare A1 vs B1 results
4. Decision: If B1 fails gates, run B2/B3/B4
5. If B2/B3/B4 needed, run them
6. Select best config
```

### Phase 3: Full Training
```
1. Run full training with selected config
2. Final test set evaluation
3. Document results
```

---

## 6. Artifact Inventory

### Data Files (on Lambda)
| Path | Description | Status |
|------|-------------|--------|
| `/home/ubuntu/data/v5_cosmos_paired/` | Original paired data | ✓ Exists |
| `/home/ubuntu/data/unpaired_manifest.parquet` | OLD - 20 files only, DO NOT USE | ⚠️ Deprecated |
| `/home/ubuntu/data/unpaired_manifest_full.parquet` | Full unpaired manifest | ☐ To create |
| `/home/ubuntu/data/unpaired_manifest_mini_10pct.parquet` | 10% stratified sample | ☐ To create |
| `/home/ubuntu/data/mini_manifest.parquet` | OLD - 5K sample, DO NOT USE | ⚠️ Deprecated |

### Checkpoint Directories
| Path | Experiment | Status |
|------|------------|--------|
| `/home/ubuntu/checkpoints/mini_A1_paired_residual/` | A1 | ☐ To create |
| `/home/ubuntu/checkpoints/mini_B1_unpaired_residual/` | B1 | ☐ To create |
| `/home/ubuntu/checkpoints/mini_B2_unpaired_residual_r5/` | B2 | ☐ To create |
| `/home/ubuntu/checkpoints/mini_B3_unpaired_residual_sched/` | B3 | ☐ To create |
| `/home/ubuntu/checkpoints/mini_B4_unpaired_residual_r3/` | B4 | ☐ To create |

---

## 7. Validation Checks Before Training

Before starting ANY experiment, verify:

| Check | Command | Expected |
|-------|---------|----------|
| Manifest exists | `ls -la <manifest_path>` | File exists |
| Manifest row count | `python -c "import pandas; print(len(pandas.read_parquet('<path>')))"` | Expected count |
| Code is synced | `ssh <instance> 'ls -la ~/dark_halo_scope/planb/unpaired_experiment/train.py'` | Recent timestamp |
| GPU available | `ssh <instance> 'nvidia-smi'` | GPU visible |
| No conflicting jobs | `ssh <instance> 'ps aux | grep train | grep python'` | No conflicts |

---

## 8. Sign-off

| Phase | Reviewer | Date | Status |
|-------|----------|------|--------|
| Data Preparation | | | ☐ Not started |
| Code Preparation | | | ☐ Partial |
| Mini Experiments | | | ☐ Not started |
| Full Training | | | ☐ Not started |

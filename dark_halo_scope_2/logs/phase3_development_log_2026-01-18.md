# Phase 3 Pipeline Development Log

**Date**: January 18, 2026  
**Author**: Aditrivaishnavi Balaji  
**Project**: Dark Halo Scope - Phase 3 EMR Pipeline

---

## Executive Summary

This document chronicles the development, optimization, and validation of the Phase 3 EMR pipeline for building parent LRG catalogs from DR10 South sweep files. The work involved code review, performance optimization, feature additions, and rigorous verification of output correctness.

---

## Session Timeline

### 1. Per-Mode Ranking Configuration Feature

**Motivation**: The original Stage 3b code used global `--k-top` and `--k-stratified` parameters applied uniformly to all ranking modes (density, n_lrg, area_weighted, psf_weighted). I needed finer control to allocate different selection budgets per mode.

**Implementation**:
- Added `RankingModeConfig` dataclass to represent per-mode configuration
- Created `parse_ranking_config()` function to parse format: `"mode:k_top:k_strat,mode:k_top:k_strat,..."`
- Added `build_ranking_configs_from_args()` to handle both legacy and new config formats
- Updated `stage_3b_select_regions()` to process each mode with its own k values

**New CLI Usage**:
```bash
--ranking-config "n_lrg:100:75,psf_weighted:100:75,density:100:75,area_weighted:100:75"
```

**Backward Compatibility**: The old `--ranking-modes`, `--k-top`, `--k-stratified` arguments still work if `--ranking-config` is not provided.

---

### 2. Stage Configuration JSON Capture

**Motivation**: With many configuration parameters across 3a, 3b, and 3c stages, I needed a way to capture exact parameters used for reproducibility and auditing.

**Implementation**:
- Created `write_stage_config_json()` function that writes a `_stage_config.json` file to the output S3 path
- Captures:
  - Stage identifier and timestamp
  - All CLI arguments relevant to that stage
  - Parsed ranking configurations (for 3b)
  - LRG variant definitions
  - Idempotency flags

**Output Location**:
- Stage 3a: `s3://.../phase3a/{variant}/_stage_config.json`
- Stage 3b: `s3://.../phase3b/{variant}/_stage_config.json`
- Stage 3c: `s3://.../phase3c/{variant}/_stage_config.json`

---

### 3. Spark Performance Optimization

**Code Review Findings**:

I conducted a thorough review of the Stage 3b and 3c code for Spark-related performance issues. Key findings:

#### Issue 1: Missing DataFrame Persistence (Stage 3b)
- **Problem**: The `pool` DataFrame was used multiple times (approxQuantile, multiple topk calls, multiple stratified selections) without caching
- **Fix**: Added `pool.persist()` before heavy usage and `pool.unpersist()` after completion

#### Issue 2: Missing DataFrame Persistence (Stage 3c)
- **Problem**: `bricks_needed` DataFrame was computed, then used for both `count()` and `collect()`, causing recomputation
- **Fix**: Added `bricks_needed.persist()` before first use and `bricks_needed.unpersist()` after collection

#### Issue 3: Repeated `list(m.keys())` in Processing Loop (HIGH PRIORITY)
- **Problem**: Inside `process_partition`, the code called `list(bcast.value.keys())` on every chunk of every sweep file
- **Impact**: Created new list objects millions of times, wasting CPU
- **Fix**: Cached `brick_set = frozenset(brick_map.keys())` and `brick_list = list(brick_set)` once per partition

#### Issue 4: Missing Explicit File Cleanup
- **Problem**: After processing each 1.8GB sweep file, local copies were not explicitly deleted
- **Risk**: Executor disk exhaustion with multiple sweeps per partition
- **Fix**: Added `os.remove(local)` in a `finally` block after processing each sweep

#### Issue 5: Missing Garbage Collection
- **Problem**: Python memory not explicitly reclaimed after processing large FITS files
- **Fix**: Added `gc.collect()` in the `finally` block after each sweep

#### Issue 6: Hardcoded Output Repartition
- **Problem**: `df.repartition(200)` was hardcoded regardless of data size
- **Fix**: Added `--output-partitions` CLI argument with auto-scaling default based on `--sweep-partitions`

---

### 4. Stage 3b Execution and Verification

**Execution**:
- Ran Stage 3b with configuration: `n_lrg:100:75,psf_weighted:100:75,density:100:75,area_weighted:100:75`
- Completed in approximately 1 minute

**Verification Steps**:

#### Step 1: Output Structure Check
```
phase3b/v3_color_relaxed/
├── _stage_config.json
├── region_selections/ (28 parquet files + _SUCCESS)
└── region_selections_csv/ (1 CSV file + _SUCCESS)
```

#### Step 2: Selection Count Validation

| Category | Expected | Actual | Status |
|----------|----------|--------|--------|
| Top-K (100 × 4 modes × 3 splits) | 1,200 | 1,200 | ✓ |
| Stratified Custom (75 × 4 modes × 3 splits) | 900 | 900 | ✓ |
| Stratified Balanced (75 × 4 modes × 3 splits) | 900 | 900 | ✓ |
| **Total** | **3,000** | **3,000** | ✓ |

#### Step 3: Distribution by Split
- train: 1,000 selections
- val: 1,000 selections
- test: 1,000 selections

#### Step 4: Unique Regions Analysis
- 811 unique region_ids selected
- This indicates expected overlap (same high-value regions selected across multiple ranking modes)

#### Step 5: Data Quality Checks
- All expected columns present
- n_bricks range: 1 to 115,023
- All score columns have valid numeric ranges
- No null/empty values in critical fields

---

### 5. Discovery: Mega-Regions

**Observation**: The top 4 regions by n_bricks account for 97% of all selected bricks:

| Region ID | n_bricks | % of Total |
|-----------|----------|------------|
| 1961 | 115,023 | 45% |
| 0 | 91,039 | 35% |
| 2659 | 23,885 | 9% |
| 2467 | 18,324 | 7% |

**Total unique bricks in union**: 256,277 (approximately 85% of DR10 South)

**Root Cause**: The connected components algorithm in Stage 3a merged most adjacent bricks into mega-regions representing the contiguous DR10 South footprint.

**Implication for Stage 3c**: Processing will cover most of the survey, requiring processing of ~1,200 sweep files.

---

### 6. Split Distribution Analysis

**Critical Finding**: The split distribution by area is inverted from expectations:

| Split | Expected | Actual (Area) | Actual (Bricks) |
|-------|----------|---------------|-----------------|
| train | ~70% | 12.7% | 33,298 |
| val | ~10% | 7.4% | 19,339 |
| test | ~20% | 79.9% | 208,667 |

**Root Cause**: The two largest mega-regions (1961 and 0) both hashed to the "test" split based on region_id hashing.

**Mega-Region Split Assignment**:
- Region 1961: test (115,023 bricks)
- Region 0: test (91,039 bricks)
- Region 2659: train (23,885 bricks)
- Region 2467: val (18,324 bricks)

---

### 7. Representativeness Validation

**Concern**: With train being only 12.7% of the data, is it representative of the overall survey?

**Analysis Performed**: Compared quality distributions across splits:

| Metric | Train | Val | Test | Verdict |
|--------|-------|-----|------|---------|
| PSF R (p50) | 1.436 | 1.420 | 1.431 | Similar |
| Depth R (p50) | 24.83 | 24.80 | 24.82 | Similar |
| EBV (p50) | 0.069 | 0.070 | 0.074 | Similar |
| RA coverage | 0-360° | 0-360° | 0-360° | Full sky |
| Dec coverage | -89 to +35 | -87 to +34 | -89 to +35 | Similar |

**Conclusion**: Despite size imbalance, all splits have similar quality distributions. No significant domain shift detected. The inverted split is acceptable for the current analysis.

**Future Consideration**: A balanced brick-level split (based on `xxhash64(brickname)`) can be computed post-hoc if needed for ML training.

---

### 8. Stage 3c Configuration Decisions

#### Output Mode: `union` vs `per_selection_set`

**Decision**: Use `union` mode.

**Rationale**:
- `union` stores each LRG once with (region_id, region_split)
- To get selection_set_id, join with Stage 3b's region_selections output
- `per_selection_set` would duplicate each LRG 12+ times (once per selection set)
- `union` is more storage-efficient and equally flexible for downstream analysis

#### MW Correction Flags

**Configuration**:
```
--use-mw-correction 0     # LRG selection uses RAW mags (Phase 2 consistent)
--emit-mw-corrected-mags 1  # Also output corrected mags as extra columns
```

**Rationale**: This provides both raw mags (for Phase 2 consistency) and corrected mags (for downstream analysis) without affecting selection logic.

#### Safety Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| max-selected-bricks | 300,000 | Buffer above 256k estimate |
| executor-overhead | 10g | Extra headroom for FITS parsing spikes |
| chunk-size | 100,000 | Balance between memory and I/O efficiency |
| output-partitions | 600 | Match sweep-partitions for balanced output |

---

### 9. Final Stage 3c Command

```bash
python3 emr/submit_phase3_pipeline_emr_cluster.py \
  --region us-east-2 \
  --stage 3c \
  --log-uri s3://darkhaloscope/emr-logs/phase3/ \
  --service-role EMR_DefaultRole \
  --jobflow-role EMR_EC2_DefaultRole \
  --subnet-id subnet-01ca3ae3325cec025 \
  --ec2-key-name root \
  --script-s3 s3://darkhaloscope/phase3/code/spark_phase3_pipeline.py \
  --bootstrap-s3 s3://darkhaloscope/phase3/code/bootstrap_phase3_pipeline_install_deps.sh \
  --master-instance-type m5.xlarge \
  --core-instance-type m5.2xlarge \
  --core-instance-count 34 \
  --executor-memory 6g \
  --executor-overhead 10g \
  --executor-cores 2 \
  --spark-args "--output-s3 s3://darkhaloscope/phase3_pipeline --variant v3_color_relaxed --sweep-index-s3 s3://darkhaloscope/dr10/sweep_urls.txt --s3-sweep-cache-prefix s3://darkhaloscope/sweep_fits_dump/ --sweep-partitions 600 --shuffle-partitions 600 --chunk-size 100000 --output-partitions 600 --parent-output-mode union --use-mw-correction 0 --emit-mw-corrected-mags 1 --max-selected-bricks 300000"
```

**Expected Runtime**: 2-4 hours with 34 × m5.2xlarge nodes (272 vCPUs)

---

## Code Changes Summary

### Files Modified

1. **`emr/spark_phase3_pipeline.py`**:
   - Added `RankingModeConfig` dataclass
   - Added `parse_ranking_config()` and `build_ranking_configs_from_args()`
   - Added `write_stage_config_json()` for config capture
   - Added `pool.persist()`/`unpersist()` in Stage 3b
   - Added `bricks_needed.persist()`/`unpersist()` in Stage 3c
   - Optimized `process_partition()` with cached brick_set and brick_list
   - Added `gc.collect()` and `os.remove(local)` for cleanup
   - Added `--output-partitions` CLI argument
   - Added `--ranking-config` CLI argument

### New Files

1. **`logs/phase3_development_log_2026-01-18.md`**: This document

---

## Key Learnings

1. **Connected Components Can Create Mega-Regions**: When most bricks are adjacent, the algorithm creates regions spanning most of the survey. This is scientifically valid but has scale implications.

2. **Region-Based Splits Are Coarse**: With mega-regions, a single region_id can dominate a split. Area balance cannot be achieved without brick-level or tile-level splitting.

3. **Representativeness != Balance**: Despite 80/13/7 split by area, the quality distributions (PSF, depth, EBV) are nearly identical across splits, making the imbalance acceptable for analysis.

4. **Union Mode Is Sufficient**: Storing each LRG once and joining with selection metadata is more efficient than duplicating rows per selection set.

5. **Memory Management Is Critical for FITS Processing**: Explicit `gc.collect()` and file cleanup prevent memory/disk exhaustion on executors processing 1.8GB FITS files.

---

---

### 10. Stage 3c Execution and Results

**Execution**: Stage 3c completed successfully in approximately 20 minutes with the configuration above.

**Output Location**: `s3://darkhaloscope/phase3_pipeline/phase3c/v3_color_relaxed/parent_union_parquet/`

---

### 11. Stage 3c Comprehensive Validation

I ran a comprehensive validation job on the Stage 3c output to verify data quality and correctness.

#### Data Overview

| Metric | Value |
|--------|-------|
| Total LRG Objects | 19,687,747 |
| Unique Regions | 811 |
| Unique Bricks | 256,208 |
| Parquet Files | 188,421 |
| Total Size | 2.323 GB |

#### Split Distribution

| Split | Regions | Bricks | Objects | % of Total |
|-------|---------|--------|---------|------------|
| test | 259 | 208,046 | 16,025,058 | 81.4% |
| train | 348 | 28,949 | 2,208,075 | 11.2% |
| val | 204 | 19,213 | 1,454,614 | 7.4% |

#### Schema Validation: ✅ PASS
All 25 expected columns present with correct types:
- Identifiers: brickname, objid, region_id, region_split
- Coordinates: ra, dec
- Raw magnitudes: gmag, rmag, zmag, w1mag
- Raw colors: rz, zw1
- MW-corrected magnitudes: gmag_mw, rmag_mw, zmag_mw, w1mag_mw
- MW-corrected colors: rz_mw, zw1_mw
- Metadata: maskbits, type
- LRG flags: is_v1_pure_massive through is_v5_very_relaxed

#### Data Quality

| Check | Status | Notes |
|-------|--------|-------|
| Null counts | ✅ | 0 nulls in all columns |
| NaN counts | ⚠️ | 80,141 NaN in gmag (0.4% - expected for bad g-band) |
| Coordinates | ✅ | RA: [0, 360], Dec: [-88.9, 35.4] |
| Color consistency | ✅ | rz matches rmag-zmag, zw1 matches zmag-w1mag |

#### Magnitude Statistics

| Band | Min | P50 | P95 | Max |
|------|-----|-----|-----|-----|
| gmag | 10.6 | 22.66 | 23.97 | NaN |
| rmag | 12.4 | 21.17 | 22.02 | 35.2 |
| zmag | 9.0 | 19.99 | 20.37 | 20.4 |
| w1mag | 3.1 | 18.76 | 19.36 | 19.6 |

#### LRG Flag Validation

| Variant | Mismatches | Match Rate | Status |
|---------|------------|------------|--------|
| v1_pure_massive | 4 | 99.99998% | ✅ |
| v2_baseline_dr10 | 7 | 99.99996% | ✅ |
| v3_color_relaxed | 12 | 99.99994% | ✅ |
| v4_mag_relaxed | 12 | 99.99994% | ✅ |
| v5_very_relaxed | 12 | 99.99994% | ✅ |

**Note**: The 4-12 mismatches are floating-point precision artifacts at boundaries (e.g., rz = 0.40000001 vs 0.4). This is expected and negligible.

#### Hierarchy Check: ✅ PASS
v1 ⊂ v2 ⊂ v3 ⊂ v4 ⊂ v5 with 0 violations.

#### MW Correction: ✅ PASS
All corrections in the right direction (corrected mags slightly smaller than raw).

#### Data Integrity

| Check | Status |
|-------|--------|
| Duplicate rows | ✅ 0 |
| Duplicate (brickname, objid) | ✅ 0 |
| PSF objects | ✅ 0 (TYPE filter worked) |
| Invalid bricknames | ✅ 0 |

#### Morphology Distribution

| Type | Count | Percentage |
|------|-------|------------|
| DEV | 6,995,131 | 35.5% |
| REX | 5,163,379 | 26.2% |
| SER | 4,391,423 | 22.3% |
| EXP | 3,137,814 | 15.9% |

#### Cross-Stage Validation

| Comparison | Expected | Actual | Coverage |
|------------|----------|--------|----------|
| Phase 3a bricks | 261,304 | 256,208 | 98.05% |
| Phase 3b regions | 811 | 811 | 100% |

**Missing 5,096 bricks**: These are bricks in selected regions with zero LRGs passing the v3 filter - correct behavior.

#### Split Consistency: ✅ PASS

| Split | zmag_mean | rz_mean | zw1_mean |
|-------|-----------|---------|----------|
| train | 19.863 | 1.248 | 1.173 |
| val | 19.866 | 1.235 | 1.167 |
| test | 19.868 | 1.243 | 1.166 |

Quality distributions are nearly identical across splits despite size imbalance.

---

### 12. Phase 3.5: Parquet Compaction

**Problem**: Stage 3c produced 188,421 small Parquet files due to partitioning by `region_split/region_id`. This causes severe performance issues for downstream pipelines.

**Solution**: I created a Phase 3.5 compaction job to rewrite the data into a clean, compact layout.

#### Compaction Job Details

**New Files Created**:
1. `emr/spark_phase3p5_compact.py` - PySpark compaction job
2. `emr/bootstrap_phase3p5_compact.sh` - Minimal bootstrap script
3. `emr/submit_phase3p5_compact_emr.py` - EMR submission script using boto3

**Key Operations**:
1. Read all 188k Parquet files from Phase 3c
2. Recompute LRG flags from stored mags/colors (eliminates precision artifacts)
3. Filter to enforce parent condition (is_v3_color_relaxed == True)
4. Add `gmag_valid` convenience column
5. Repartition to 96 partitions
6. Write partitioned by `region_split` only (not region_id)

#### Execution

**Command**:
```bash
python3 emr/submit_phase3p5_compact_emr.py \
  --region us-east-2 \
  --variant v3_color_relaxed \
  --num-partitions 96 \
  --core-instance-count 4
```

**Cluster**: 4 × m5.xlarge core nodes  
**Runtime**: ~30 minutes

#### Compaction Results

| Metric | Before (Phase 3c) | After (Phase 3.5) | Improvement |
|--------|-------------------|-------------------|-------------|
| **Files** | 188,421 | 291 | **99.85% fewer** |
| **Size** | 2.323 GB | 1.34 GB | **42% smaller** |
| **Partitioning** | region_split/region_id | region_split only | Simpler |

**File Distribution**:
- `region_split=train/`: 96 files
- `region_split=val/`: 96 files
- `region_split=test/`: 96 files
- `_SUCCESS` marker: 1 file
- `_metadata_temp/`: metadata directory

**Why the size reduction?**
1. Better Snappy compression - larger files compress more efficiently
2. Reduced Parquet overhead - fewer file headers, footers, and metadata blocks

#### Output Location

```
s3://darkhaloscope/phase3_pipeline/phase3p5/v3_color_relaxed/parent_compact/
├── region_split=train/
│   └── part-00000...part-00095.snappy.parquet
├── region_split=val/
│   └── part-00000...part-00095.snappy.parquet
├── region_split=test/
│   └── part-00000...part-00095.snappy.parquet
├── _SUCCESS
└── _metadata_temp/
```

**Recommendation**: Use the compacted path for all downstream pipelines for optimal I/O performance.

---

## Final Summary

### Phase 3 Pipeline Outputs

| Stage | Output Path | Status |
|-------|-------------|--------|
| 3a | `s3://darkhaloscope/phase3_pipeline/phase3a/v3_color_relaxed/` | ✅ Complete |
| 3b | `s3://darkhaloscope/phase3_pipeline/phase3b/v3_color_relaxed/` | ✅ Complete |
| 3c | `s3://darkhaloscope/phase3_pipeline/phase3c/v3_color_relaxed/` | ✅ Complete |
| 3.5 | `s3://darkhaloscope/phase3_pipeline/phase3p5/v3_color_relaxed/` | ✅ Complete |

### Key Metrics

| Metric | Value |
|--------|-------|
| Total LRG Objects | 19,687,747 |
| Unique Regions | 811 |
| Unique Bricks | 256,208 |
| Final File Count | 291 (compacted) |
| Final Size | 1.34 GB (compacted) |
| LRG Flag Accuracy | 99.99994% |
| Data Integrity | 100% (no duplicates, no PSF objects) |

### Files Added

1. `emr/spark_phase3p5_compact.py` - Phase 3.5 PySpark compaction job
2. `emr/bootstrap_phase3p5_compact.sh` - Phase 3.5 bootstrap script
3. `emr/submit_phase3p5_compact_emr.py` - Phase 3.5 EMR submission script
4. `emr/phase3c_validation_report.json` - Comprehensive validation report
5. `logs/phase3_development_log_2026-01-18.md` - This development log

---

*End of Log Entry - January 19, 2026*


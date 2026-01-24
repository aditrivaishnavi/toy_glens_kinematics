# Phase 4a Command Review Request

Please review the Phase 4a EMR command I'm about to run and verify the configuration choices are correct and consistent with the codebase.

---

## Context

This is the **Dark Halo Scope** project - a gravitational lens detection pipeline for the DESI Legacy Survey DR10. We're building a completeness-calibrated lens finder using injection-recovery methodology.

### Project Status
- **Phase 3**: Complete - parent catalog of ~20M LRGs with region-level train/val/test splits
- **Phase 4a**: Building injection task manifests (THIS RUN)
- **Phase 4b**: Coadd caching - already complete for 180k bricks (image/invvar/maskbits)
- **Phase 4c**: Injection + stamp generation (pending)
- **Phase 4d**: Completeness summaries (pending)

---

## The Command I Ran

```bash
python3 emr/submit_phase4_pipeline_emr_cluster.py \
  --region us-east-2 \
  --stage 4a \
  --log-uri s3://darkhaloscope/emr-logs/phase4/ \
  --service-role EMR_DefaultRole \
  --jobflow-role EMR_EC2_DefaultRole \
  --subnet-id subnet-01ca3ae3325cec025 \
  --ec2-key-name root \
  --script-s3 s3://darkhaloscope/phase4/code/spark_phase4_pipeline.py \
  --bootstrap-s3 s3://darkhaloscope/phase4/code/bootstrap_phase4_pipeline_install_deps.sh \
  --core-instance-count 10 \
  --spark-args '--output-s3 s3://darkhaloscope/phase4_pipeline \
    --variant v3_color_relaxed \
    --parent-s3 s3://darkhaloscope/phase3_pipeline/phase3p5/v3_color_relaxed/parent_compact/ \
    --region-selections-s3 s3://darkhaloscope/phase3_pipeline/phase3b/v3_color_relaxed/region_selections/ \
    --bricks-with-region-s3 s3://darkhaloscope/phase3_pipeline/phase3a/v3_color_relaxed/bricks_with_region/ \
    --tiers debug,grid,train \
    --stamp-sizes 64,96 \
    --bandsets grz \
    --n-total-train-per-split 200000 \
    --n-per-config-grid 40 \
    --n-per-config-debug 5 \
    --replicates 2 \
    --control-frac-train 0.50 \
    --control-frac-grid 0.10 \
    --control-frac-debug 0.0 \
    --split-seed 13 \
    --force 1'
```

**Cluster ID**: `j-1BGW8GJWLU4DD`

---

## Configuration Rationale

### Stamp Sizes: `64,96`

| Size | Coverage at 0.262"/pix | Rationale |
|------|------------------------|-----------|
| 64×64 | 16.8" × 16.8" | Baseline CNN, efficient storage, covers θ_E < 1.5" |
| 96×96 | 25.1" × 25.1" | More context for false-positive suppression, works with Swin/ViT |

**Why both?** We want to compare model architectures:
- Simple CNN → 64×64
- Swin/ViT → 96×96 (or resized)
- ResNet with ImageNet weights → adapt at training time

### Split Seed: `13`

**Critical for reproducibility.** The original Phase 4a run used `--split-seed 13` (the code default at that time). We're keeping this value to ensure:
- Same galaxies sampled for each tier/split
- Only difference from previous run is adding 96×96 stamps

The seed is now:
1. Hardcoded as `DEFAULT_SPLIT_SEED = 13` in the code
2. Saved in `_stage_config.json` under `seeds.split_seed`

### Control Fractions

| Tier | Control Frac | Rationale |
|------|--------------|-----------|
| train | 0.50 | 50% controls for balanced training, enables FPR calibration |
| grid | 0.10 | 10% controls for systematic completeness grid |
| debug | 0.0 | No controls needed for quick debugging |

### Tiers

| Tier | Purpose | Tasks per size |
|------|---------|----------------|
| debug | Quick end-to-end verification | ~2,400 |
| grid | Systematic parameter sweep for completeness | ~175,000 |
| train | ML training data | ~1,200,000 |

### Force: `1`

Overwrites existing 4a output because:
1. We're adding 96×96 stamps
2. Previous run didn't have explicit seed tracking

---

## Recent Code Changes (since last review)

### 1. `render_unlensed_source()` Fix
```python
# BEFORE (bug):
unit_flux = sersic_unit_total_flux(reff_arcsec=src_reff_arcsec, q=1.0, n=n)

# AFTER (fixed):
q_src = (1.0 - src_e) / (1.0 + src_e)
unit_flux = sersic_unit_total_flux(reff_arcsec=src_reff_arcsec, q=q_src, n=n)
```
**Why?** The magnification proxy was biased because it normalized with circular (q=1.0) but rendered elliptical.

### 2. Gate psfsize Reads
```python
p.add_argument("--use-psfsize-maps", type=int, default=0)
# Only load psfsize maps if flag is set AND files were cached in 4b
if use_psfsize_maps:
    for b in bands:
        # load psfsize map
```
**Why?** Without gating, 4c would attempt 540k failed S3 reads if psfsize maps weren't cached.

### 3. R-band Specific Maskbits
```python
MASKBITS_BAD_COMMON = NPRIMARY | BRIGHT | BAILOUT | MEDIUM
MASKBITS_BAD_R = MASKBITS_BAD_COMMON | SATUR_R | ALLMASK_R
```
**Why?** arc_snr is r-band only; using all-band mask would throw away valid r pixels.

### 4. Renamed `wise_frac` → `wise_brightmask_frac`
**Why?** WISE bits are bright-star masking, not source detections. Accurate naming.

### 5. Added `metrics_ok` Flag
```python
T.StructField("metrics_ok", T.IntegerType(), True),
# Set to 1 if: mask_valid AND cut_ok_all AND arc_snr computed
```
**Why?** Simplifies Phase 4d aggregation - single flag for "usable metrics".

### 6. Hardcoded Split Seed
```python
DEFAULT_SPLIT_SEED = 13  # Matches original Phase 4a run
```
**Why?** Ensures reproducibility. The seed is now saved in stage_config.json.

---

## Validation Script Updates

The `spark_validate_phase4a.py` now includes:

1. **Stamp size validation**
   - Lists all stamp_size values found
   - Compares against stage_config
   - Shows rows by tier/stamp_size breakdown

2. **Seed tracking validation**
   - Checks for `seeds.split_seed` in stage_config
   - Warns if not equal to 13

3. **Experiment ID pattern validation**
   - Verifies `{tier}_stamp{size}_*` patterns exist
   - Ensures both 64 and 96 appear for each tier

---

## Questions for Review

1. **Command correctness**: Are the `--spark-args` parameter names correct and match the argparse in `spark_phase4_pipeline.py`?

2. **Stamp size strategy**: Is generating both 64×64 and 96×96 in a single run the right approach, or should we run separately?

3. **Control fraction 50%**: Is this appropriate for train tier? Some argue 25% is sufficient if negatives are representative.

4. **Seed value 13**: I kept this for backward compatibility. Is this the right choice, or should we use a more "intentional" seed like 42 for fresh runs?

5. **Missing parameters**: Are there any parameters I should have included but didn't?

6. **Validation completeness**: Does the updated validation script check everything necessary for a publishable result?

---

## Files to Review

1. `dark_halo_scope/emr/spark_phase4_pipeline.py` - main pipeline
2. `dark_halo_scope/emr/spark_validate_phase4a.py` - validation script
3. `dark_halo_scope/emr/submit_phase4_pipeline_emr_cluster.py` - EMR submission

---

*Generated: 2026-01-24*
*Cluster: j-1BGW8GJWLU4DD*


# Phase 4b: Coadd Cache Build - Technical Worklog

**Date**: 2026-01-24  
**Author**: Aditrivaishnavi Balaji  
**Status**: Completed (99.87%)  
**EMR Cluster ID**: j-3BZPABOX3EQ8M

---

## 1. Objective

Cache DR10 Legacy Survey South coadd FITS files from NERSC to S3 for use in Phase 4c injection pipeline. This eliminates repeated downloads from NERSC during the compute-intensive injection stage.

---

## 2. Data Source

### NERSC Legacy Survey DR10 South

| Property | Value |
|----------|-------|
| **Base URL** | `https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/coadd` |
| **Data Release** | DR10 (Data Release 10) |
| **Survey Region** | Southern sky (Dec < +34°) |
| **Pixel Scale** | 0.262 arcsec/pixel |
| **Image Size** | 3600 × 3600 pixels per brick |
| **Sky Coverage per Brick** | ~0.25° × 0.25° |

### File Structure at NERSC

```
coadd/{brickdir}/{brickname}/
├── legacysurvey-{brickname}-image-g.fits.fz
├── legacysurvey-{brickname}-image-r.fits.fz
├── legacysurvey-{brickname}-image-z.fits.fz
├── legacysurvey-{brickname}-invvar-g.fits.fz
├── legacysurvey-{brickname}-invvar-r.fits.fz
├── legacysurvey-{brickname}-invvar-z.fits.fz
└── legacysurvey-{brickname}-maskbits.fits.fz
```

Where `brickdir` = first 3 characters of brickname (e.g., `000` for `0001m002`).

---

## 3. Files Cached Per Brick

| File Type | Description | Typical Size | Purpose |
|-----------|-------------|--------------|---------|
| `image-{g,r,z}.fits.fz` | Flux coadd in nanomaggies | 10-15 MB | Science image for cutouts |
| `invvar-{g,r,z}.fits.fz` | Inverse variance (1/σ²) | 10-12 MB | Noise estimation for SNR |
| `maskbits.fits.fz` | Bad pixel flags | 0.3-0.5 MB | Quality masking |

**Total per brick**: 7 files, ~68-75 MB

---

## 4. Input Data

### Bricks Manifest

- **Source**: `s3://darkhaloscope/phase4_pipeline/phase4a/v3_color_relaxed/bricks_manifest/`
- **Total Bricks**: 180,373
- **Selection Criteria**: All unique bricks referenced by Phase 4a injection tasks

### Selection Breakdown

These bricks cover LRGs selected via the `v3_color_relaxed` variant:
- z < 20.4 (magnitude cut)
- r - z > 0.4 (color cut)
- z - W1 > 1.6 (color cut)

---

## 5. EMR Cluster Configuration

### Instance Configuration

| Role | Instance Type | Count | vCPUs | Memory |
|------|---------------|-------|-------|--------|
| Master | m5.xlarge | 1 | 4 | 16 GB |
| Core | m5.2xlarge | 20 | 8 | 32 GB |
| **Total** | - | 21 | 164 | 656 GB |

### Spark Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| `--cache-partitions` | 200 | Parallelism for downloads |
| `--http-timeout-s` | 300 | 5-minute timeout for large files |
| `--force` | 0 | Idempotent - skip existing files |

### EMR Release

- **Release Label**: emr-6.15.0
- **Spark Version**: 3.4.x
- **Python Version**: 3.9

---

## 6. Execution Command

```bash
python3 emr/submit_phase4_pipeline_emr_cluster.py \
  --region us-east-2 \
  --stage 4b \
  --log-uri s3://darkhaloscope/emr-logs/phase4/ \
  --service-role EMR_DefaultRole \
  --jobflow-role EMR_EC2_DefaultRole \
  --subnet-id subnet-01ca3ae3325cec025 \
  --ec2-key-name root \
  --script-s3 s3://darkhaloscope/phase4/code/spark_phase4_pipeline.py \
  --bootstrap-s3 s3://darkhaloscope/phase4/code/bootstrap_phase4_pipeline_install_deps.sh \
  --core-instance-count 20 \
  --spark-args "--output-s3 s3://darkhaloscope/phase4_pipeline \
    --variant v3_color_relaxed \
    --coadd-s3-cache-prefix s3://darkhaloscope/dr10/coadd_cache/ \
    --coadd-base-url https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/coadd \
    --bands g,r,z \
    --cache-partitions 200 \
    --http-timeout-s 300 \
    --force 0"
```

---

## 7. Runtime Metrics

### Progress Timeline

| Elapsed Time | Complete Bricks | Progress | Storage |
|--------------|-----------------|----------|---------|
| 30 min | 33,152 | 18.4% | 2.64 TB |
| 2h 57m | 152,555 | 84.6% | 11.55 TB |
| 3h 15m | 168,852 | 93.6% | ~12.5 TB |
| 4h 30m | 180,152 | 99.87% | 13.32 TB |

### Final Statistics

| Metric | Value |
|--------|-------|
| **Total Bricks Cached** | 180,152 / 180,373 (99.87%) |
| **Missing Bricks** | 221 (0.13%) |
| **Total Files** | 1,441,216 |
| **Total Storage** | 13.32 TB |
| **Average per Brick** | ~74 MB |
| **Runtime** | ~4.5 hours |
| **Download Rate** | ~670 bricks/minute average |

---

## 8. S3 Cache Structure

### Output Location

```
s3://darkhaloscope/dr10/coadd_cache/
├── 0001m002/
│   ├── _SUCCESS
│   ├── legacysurvey-0001m002-image-g.fits.fz
│   ├── legacysurvey-0001m002-image-r.fits.fz
│   ├── legacysurvey-0001m002-image-z.fits.fz
│   ├── legacysurvey-0001m002-invvar-g.fits.fz
│   ├── legacysurvey-0001m002-invvar-r.fits.fz
│   ├── legacysurvey-0001m002-invvar-z.fits.fz
│   └── legacysurvey-0001m002-maskbits.fits.fz
├── 0001m005/
│   └── ... (7 files + _SUCCESS)
... (180,373 brick directories)
```

### _SUCCESS Markers

Each brick directory contains a `_SUCCESS` marker file (0 bytes) written after all 7 data files are successfully cached. This enables:
- Fast resume on reruns
- Per-brick completeness verification
- Idempotent caching with `--force 0`

---

## 9. Data Quality Validation

### Spot Check: File Counts

| Brick | Files | Status |
|-------|-------|--------|
| 0001m002 | 7/7 | ✅ |
| 0001m062 | 7/7 | ✅ |
| 1000m337 | 7/7 | ✅ |
| 3598p230 | 7/7 | ✅ |
| 3598p312 | 7/7 | ✅ |

### FITS File Validation

Two bricks validated with astropy:

**Brick 0001m002** (RA~0°, Dec~0°):
| Property | Value |
|----------|-------|
| Shape | 3600 × 3600 |
| Dtype | float32 |
| Min | -4.22 nMgy |
| Max | 106.96 nMgy |
| Mean | 0.014 nMgy |
| Std | 0.40 nMgy |
| NaN count | 0 |
| WCS Center | RA=0.125°, Dec=-0.25° |

**Brick 1000m337** (RA~100°, Dec~-34°):
| Property | Value |
|----------|-------|
| Shape | 3600 × 3600 |
| Dtype | float32 |
| Min | -5.97 nMgy |
| Max | 117.24 nMgy |
| Mean | 0.020 nMgy |
| Std | 0.54 nMgy |
| NaN count | 0 |
| WCS Center | RA=100.05°, Dec=-33.75° |

### Interpretation

- **Mean ~0**: Correct for sky-subtracted coadds
- **Negative min**: Expected Gaussian noise fluctuations
- **Max 100+ nMgy**: Bright sources (~17.5 mag)
- **Std 0.4-0.5 nMgy**: Consistent with DR10 5σ depth of r~23.5-24.5

---

## 10. Physical Units

### Nanomaggy System

Legacy Survey coadds use **nanomaggies** (nMgy):

| Flux (nMgy) | AB Magnitude | Example Object |
|-------------|--------------|----------------|
| 3631 | 12.5 | Very bright star |
| 100 | 17.5 | Bright galaxy core |
| 10 | 20.0 | Typical LRG |
| 1 | 22.5 | Faint source |
| 0.1 | 25.0 | Near detection limit |

**Conversion**: `mag = 22.5 - 2.5 × log10(flux_nMgy)`

### Inverse Variance

The `invvar` files contain 1/σ² per pixel:
- **High invvar**: Low noise, reliable measurement
- **Low invvar**: High noise, less reliable
- **Zero invvar**: Bad pixel (masked)

---

## 11. Cost Analysis

### Compute Costs

| Resource | Duration | Rate | Cost |
|----------|----------|------|------|
| m5.xlarge (master) | 4.5 hr | $0.192/hr | $0.86 |
| m5.2xlarge × 20 (core) | 4.5 hr | $0.384/hr | $34.56 |
| EMR surcharge | 4.5 hr | ~$0.10/hr/instance | $9.45 |
| **Total Compute** | - | - | **~$45** |

### Storage Costs

| Storage | Size | Rate | Monthly Cost |
|---------|------|------|--------------|
| S3 Standard | 13.32 TB | $0.023/GB | **~$306/month** |

### Data Transfer

- **NERSC → EMR**: Free (public data)
- **EMR → S3**: Free (same region)

---

## 12. Retry Strategy

### Exponential Backoff

The pipeline implements exponential backoff with jitter for HTTP failures:

```python
for attempt in range(max_retries):  # max_retries = 5
    try:
        response = requests.get(url, timeout=timeout_s)
        if response.status_code == 200:
            return response.content
    except Exception:
        sleep_time = (2 ** attempt) + random.random()
        time.sleep(sleep_time)
```

### Idempotent Caching

With `--force 0`, each file is checked via S3 `head_object` before download:
- Present → Skip
- Missing → Download

This enables safe reruns to fill gaps.

---

## 13. Known Issues

### Missing Bricks (221 / 180,373)

0.13% of bricks failed to cache. Possible causes:
- Transient NERSC availability
- Network timeouts despite retries
- Bricks not in DR10 South (edge cases)

**Mitigation**: Rerun with same command to fill gaps.

---

## 14. Files Created

| File | Location | Description |
|------|----------|-------------|
| `spark_phase4_pipeline.py` | `emr/` | Main pipeline code |
| `validate_phase4b_cache.py` | `scripts/` | Cache validation script |
| `bootstrap_phase4_pipeline_install_deps.sh` | `emr/` | EMR bootstrap |

---

## 15. Phase 4a Parameter Audit

### EMR Cluster Details

| Property | Value |
|----------|-------|
| **Cluster ID** | j-1ZU6HUOZZYSUI |
| **Cluster Name** | darkhaloscope-phase4-4a |
| **Master** | m5.xlarge × 1 |
| **Core** | m5.2xlarge × 10 |
| **Step Start** | 2026-01-24T02:36:54Z |
| **Step End** | 2026-01-24T02:41:57Z |
| **Duration** | 5 min 3 sec |
| **Log URI** | s3://darkhaloscope/emr-logs/phase4/ |

### Explicit Parameters Passed (from EMR Step Config)

These parameters were explicitly passed on the command line:

| Parameter | Value |
|-----------|-------|
| `--stage` | `4a` |
| `--output-s3` | `s3://darkhaloscope/phase4_pipeline` |
| `--variant` | `v3_color_relaxed` |
| `--parent-s3` | `s3://darkhaloscope/phase3_pipeline/phase3p5/v3_color_relaxed/parent_compact/` |
| `--region-selections-s3` | `s3://darkhaloscope/phase3_pipeline/phase3b/v3_color_relaxed/region_selections/` |
| `--bricks-with-region-s3` | `s3://darkhaloscope/phase3_pipeline/phase3a/v3_color_relaxed/bricks_with_region/` |
| `--tiers` | `debug,grid,train` |
| `--stamp-sizes` | `64` |
| `--bandsets` | `grz` |
| `--n-total-train-per-split` | `200000` |
| `--n-per-config-grid` | `40` |
| `--n-per-config-debug` | `5` |
| `--replicates` | `2` |
| `--control-frac-train` | `0.50` |
| `--control-frac-grid` | `0.10` |
| `--control-frac-debug` | `0.0` |
| `--skip-if-exists` | `0` |
| `--force` | `1` |

### Implicit Parameters (Code Defaults)

These parameters used code defaults and were **NOT** passed explicitly:

| Parameter | Default Value | Source Location |
|-----------|---------------|-----------------|
| `--split-seed` | `13` | `spark_phase4_pipeline.py` line 2363 |

### Stage Config Gap Identified

**ISSUE**: The original `_stage_config.json` did not include the `split_seed` parameter.

**RISK**: Without recording the seed, reruns could produce different galaxy samples, breaking reproducibility.

**FIX**: The stage config has been updated to include a `seeds` section with `split_seed: 13`.

### Reproducibility Requirements for 4a Rerun

To produce identical manifests (same galaxies, same task_ids, same frozen randomness), any rerun must use:

```bash
--split-seed 13 \
--n-total-train-per-split 200000 \
--n-per-config-grid 40 \
--n-per-config-debug 5 \
--replicates 2 \
--control-frac-train 0.50 \
--control-frac-grid 0.10 \
--control-frac-debug 0.0
```

### Verification After Any 4a Rerun

After any rerun, verify manifest consistency:
```python
old = spark.read.parquet("s3://.../manifests_backup/")
new = spark.read.parquet("s3://.../manifests/")

# These must be identical:
old.groupBy("experiment_id").count().orderBy("experiment_id").show()
new.groupBy("experiment_id").count().orderBy("experiment_id").show()
```

---

## 16. Next Steps

1. **Validate Cache Completeness**: Run `validate_phase4b_cache.py` to identify missing bricks
2. **Fill Gaps**: Rerun 4b with `--force 0` if needed
3. **Proceed to 4c**: Generate injected cutouts using cached coadds

---

## 17. References

- [Legacy Survey DR10 Documentation](https://www.legacysurvey.org/dr10/)
- [FITS File Format](https://fits.gsfc.nasa.gov/fits_documentation.html)
- [Nanomaggy Definition](https://www.legacysurvey.org/dr10/description/#photometry)

---

## 18. Reproducibility

### Code Version

All code committed to repository at time of execution:
- `spark_phase4_pipeline.py` with frozen randomness
- Idempotent caching with per-brick `_SUCCESS` markers
- Exponential backoff retry logic

### Parameters Archived

Stage config written to S3:
`s3://darkhaloscope/phase4_pipeline/phase4a/v3_color_relaxed/_stage_config.json`

---

*Last updated: 2026-01-24*


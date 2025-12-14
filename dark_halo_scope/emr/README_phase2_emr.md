# Phase 2: Multi–Cut LRG Density from DR10 SWEEPs (EMR)

This phase generalizes Phase 1.5. Instead of committing to a single LRG definition, we compute **multiple, physics-motivated LRG selections in one pass** over the DR10 SWEEP catalog using PySpark on EMR.

The goal is to:
- Map how LRG surface density changes as we relax or tighten color / magnitude cuts.
- Quantify the trade-off between **purity** (very massive halos only) and **completeness** (include more typical LRGs).
- Provide rich per-brick inputs for later phases (region selection, completeness modeling, and halo-mass constraints).

---

## 1. Inputs

- **SWEEP index file**: text file (local or S3) with one DR10 SWEEP FITS path or URL per line  
  Example: `s3://darkhaloscope/dr10/sweep_urls_full.txt`

- **DR10 SWEEP FITS files** (remote: NERSC / S3 mirror):
  - Assumed to contain (case-insensitive) columns:
    - `ra`, `dec`
    - `brickname`
    - `type` (morphology: `PSF`, `REX`, `EXP`, `DEV`, etc.)
    - `flux_r`, `flux_z`, `flux_w1` (nanomaggies)

- **Optional RA/Dec footprint** (recommended):
  - Example: `RA in [150, 250] deg`, `Dec in [0, 30] deg`

---

## 2. LRG Hyper-Grid (Selection Families)

Every galaxy that passes basic quality cuts is tested against **five** LRG-like selections:

All selections require:
- Non-stellar morphology: `TYPE != 'PSF'`
- Positive fluxes in all bands: `flux_r > 0`, `flux_z > 0`, `flux_w1 > 0`
- Inside RA/Dec footprint (if provided)

Magnitudes are defined as:

- `mag = 22.5 – 2.5 log10(flux_nanommaggies)`
- `r_minus_z = mag_r – mag_z`
- `z_minus_w1 = mag_z – mag_w1`

The cuts:

1. **v1_pure_massive**  
   - `z < 20.0`  
   - `(r – z) > 0.5`  
   - `(z – W1) > 1.6`  
   - Extremely massive, red systems; very pure, low counts.

2. **v2_baseline_dr10**  
   - `z < 20.4`  
   - `(r – z) > 0.4`  
   - `(z – W1) > 1.6`  
   - Approximate analogue of the conservative DR10 LRG-like selection (similar to Phase 1.5).

3. **v3_color_relaxed**  
   - `z < 20.4`  
   - `(r – z) > 0.4`  
   - `(z – W1) > 0.8`  
   - Loosens the W1 color to include slightly bluer / less massive halos.

4. **v4_mag_relaxed**  
   - `z < 21.0`  
   - `(r – z) > 0.4`  
   - `(z – W1) > 0.8`  
   - Same colors as v3 but fainter in z; probes more typical LRGs.

5. **v5_very_relaxed**  
   - `z < 21.5`  
   - `(r – z) > 0.3`  
   - `(z – W1) > 0.8`  
   - Most inclusive; designed to explore the limit of useful completeness.

---

## 3. Output: Per-Brick Hyper-Grid Density Table

The Spark job produces a single CSV:

- `phase2_lrg_hypergrid.csv` under the `--output-prefix`.

Columns:

- `brickname`  
- `n_gal` – total number of non-PSF galaxies with positive fluxes in the footprint.  
- `n_lrg_v1_pure_massive`  
- `n_lrg_v2_baseline_dr10`  
- `n_lrg_v3_color_relaxed`  
- `n_lrg_v4_mag_relaxed`  
- `n_lrg_v5_very_relaxed`

Each row aggregates counts over all SWEEP files that contain that brick.  
Later phases will join this table to the DR10 bricks catalog to compute per-brick surface densities and select optimal regions.

---

## 4. Prerequisites

Before running the EMR pipeline, ensure you have:

1. **AWS credentials** configured (via `~/.aws/credentials`, environment variables, or IAM role).
2. **IAM roles** for EMR:
   - `EMR_DefaultRole` (or your custom service role)
   - `EMR_EC2_DefaultRole` (or your custom instance profile)
3. **S3 buckets** with appropriate permissions for:
   - Storing the code archive
   - Storing the SWEEP index file
   - Writing EMR outputs
   - EMR logs
4. **boto3** installed: `pip install boto3`

---

## 5. Typical Flow

### Step 1: Create and upload the code archive

```bash
# From the dark_halo_scope directory
cd /path/to/dark_halo_scope
tar -zcvf dark_halo_scope_code.tgz \
    --exclude='venv' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='external_data' \
    --exclude='outputs' \
    .

# Upload to S3
aws s3 cp dark_halo_scope_code.tgz s3://MY_BUCKET/code/dark_halo_scope_code.tgz
```

### Step 2: Prepare the SWEEP index file

```bash
# If you have a sweep_urls.txt from Phase 1.5, reuse it
aws s3 cp sweep_urls.txt s3://MY_BUCKET/dr10/sweep_urls.txt
```

### Step 3: Submit the EMR cluster

From your laptop or EC2 (anywhere with AWS credentials):

```bash
python -m emr.submit_phase2_emr_cluster \
    --region us-east-2 \
    --sweep-index-s3 s3://MY_BUCKET/dr10/sweep_urls.txt \
    --output-prefix s3://MY_BUCKET/phase2_hypergrid \
    --code-archive-s3 s3://MY_BUCKET/code/dark_halo_scope_code.tgz \
    --ra-min 150 --ra-max 250 \
    --dec-min 0 --dec-max 30
```

### Step 4: Monitor the job

```bash
aws emr describe-cluster --cluster-id j-XXXXXXXXXXXXX --region us-east-2
aws emr list-steps --cluster-id j-XXXXXXXXXXXXX --region us-east-2
```

### Step 5: Retrieve results

After the EMR job completes:

```bash
aws s3 cp s3://MY_BUCKET/phase2_hypergrid/phase2_lrg_hypergrid.csv/ \
    ./outputs/phase2_emr/ --recursive
```

---

## 6. CLI Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--region` | AWS region for EMR | (required) |
| `--sweep-index-s3` | S3 path to SWEEP URL list | (required) |
| `--output-prefix` | S3 prefix for output | (required) |
| `--code-archive-s3` | S3 path to code tgz | (required) |
| `--ra-min/max`, `--dec-min/max` | Footprint filter | None (full sky) |
| `--max-sweeps` | Limit files for testing | 0 (all) |
| `--instance-type` | EC2 type for core nodes | m5.2xlarge |
| `--master-instance-type` | EC2 type for master | m5.xlarge |
| `--core-count` | Number of core nodes | 3 |
| `--s3-cache-prefix` | Cache HTTP downloads to S3 | None |
| `--chunk-size` | Rows per memory chunk | 100000 |
| `--num-partitions` | Spark parallelism | 128 |
| `--log-uri` | Custom log location | output-prefix/emr-logs/ |
| `--ec2-key-name` | SSH key for debugging | None |

---

## 7. Examples

**Small test (5 files):**
```bash
python -m emr.submit_phase2_emr_cluster \
    --region us-east-2 \
    --sweep-index-s3 s3://darkhaloscope/dr10/sweep_urls.txt \
    --output-prefix s3://darkhaloscope/phase2_test \
    --code-archive-s3 s3://darkhaloscope/code/dark_halo_scope_code.tgz \
    --max-sweeps 5 \
    --instance-type m5.xlarge \
    --core-count 2
```

**Full region run:**
```bash
python -m emr.submit_phase2_emr_cluster \
    --region us-east-2 \
    --sweep-index-s3 s3://darkhaloscope/dr10/sweep_urls.txt \
    --output-prefix s3://darkhaloscope/phase2_hypergrid \
    --code-archive-s3 s3://darkhaloscope/code/dark_halo_scope_code.tgz \
    --ra-min 150 --ra-max 250 \
    --dec-min 0 --dec-max 30 \
    --s3-cache-prefix s3://darkhaloscope/sweep-cache/ \
    --core-count 5
```

**Full sky comprehensive scan:**
```bash
python -m emr.submit_phase2_emr_cluster \
    --region us-east-2 \
    --sweep-index-s3 s3://darkhaloscope/dr10/sweep_urls.txt \
    --output-prefix s3://darkhaloscope/phase2_fullsky \
    --code-archive-s3 s3://darkhaloscope/code/dark_halo_scope_code.tgz \
    --s3-cache-prefix s3://darkhaloscope/sweep-cache/ \
    --core-count 10 \
    --num-partitions 200
```

---

## 8. Memory and Instance Guidelines

Phase 2 needs slightly more memory than Phase 1.5 because it maintains multiple selection masks simultaneously.

| Instance Type | RAM | Recommended Use |
|---------------|-----|-----------------|
| m5.xlarge | 16 GB | Testing only (1-2 executors) |
| m5.2xlarge | 32 GB | Production (3-4 executors, comfortable) |
| m5.4xlarge | 64 GB | Large scale (6+ executors) |
| r5.xlarge | 32 GB | Memory-optimized alternative |

Default Spark config for Phase 2:
- Driver: 3g memory + 1g overhead
- Executor: 6g memory + 2g overhead
- Dynamic allocation: 1-10 executors

---

## 9. Troubleshooting

### OOM Errors
Use larger instances (`--instance-type m5.2xlarge`) or reduce `--chunk-size` to 50000.

### Job takes too long
- Reduce footprint with `--ra-min/max`, `--dec-min/max`
- Use `--max-sweeps` for testing
- Increase `--core-count` for more parallelism

### Empty output
- Check SWEEP index file has valid URLs
- Verify RA/Dec bounds match your data
- Check executor logs in S3

### Viewing Logs
```
s3://YOUR_BUCKET/phase2_hypergrid/emr-logs/j-CLUSTERID/
├── steps/s-STEPID/stderr.gz   # Most useful!
├── steps/s-STEPID/stdout.gz
└── node/i-INSTANCEID/...
```

---

## 10. Design Notes

The cluster is **transient**: it terminates automatically after the job completes (or fails), so you only pay for active compute time.

Phase 2 is scientifically complementary to Phase 1.5:
- Phase 1.5: single conservative LRG cut, fast validation
- Phase 2: five-cut hyper-grid, purity vs completeness exploration

Both phases can use the same SWEEP index file and S3 cache.

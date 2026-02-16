# Phase 1.5 EMR + PySpark Pipeline

This directory contains an optional EMR-based backend for Phase 1.5
region scouting. The goal is to move the DR10 SWEEP-based LRG counting
off the laptop and onto a small Spark cluster, while keeping the science
logic identical to the local workflow.

## Overview

1. **`emr/spark_phase1p5_lrg_density.py`**

   A PySpark driver script that:

   - Reads a newline-delimited list of DR10 SWEEP FITS paths or URLs.
   - Distributes those paths across Spark executors.
   - For each file, loads the SWEEP table via `astropy.io.fits`,
     applies DESI-like LRG color/magnitude cuts, groups by BRICKNAME,
     and returns `(brickname, lrg_count)` pairs.
   - Reduces counts across all SWEEPs and writes a compact CSV with
     columns `brickname,lrg_count` to a user-specified S3 prefix.
  - Default LRG cuts in the Spark job: z < 20.4, r − z > 0.4, z − W1 > 0.8

2. **`emr/submit_phase1p5_emr_cluster.py`**

   A small helper script that:

   - Reads defaults from `src.config.Phase1p5Config` (EMR roles, S3 prefixes,
     instance types, EMR release label).
   - Uses `boto3` to call `emr.run_job_flow` and submit a transient EMR cluster
     with a single Spark step running `spark_phase1p5_lrg_density.py`.
   - Allows you to override key paths (region, SWEEP index S3 URI, output prefix,
     code archive S3 URI) via CLI flags.

## Prerequisites

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

## Typical Flow

### Step 1: Prepare the SWEEP index file

Use your existing `scripts/download_dr10_region_data.py` (or equivalent) to
generate a newline-delimited list of SWEEP FITS URLs, then upload it to S3:

```bash
# Generate the sweep index (if not already done)
python scripts/download_dr10_region_data.py --ra-min 150 --ra-max 250 --dec-min 0 --dec-max 30

# Upload to S3
aws s3 cp external_data/dr10/sweeps_ra150_250_dec0_30_10.1.txt \
    s3://MY_BUCKET/dr10/sweeps_ra150_250_dec0_30_10.1.txt
```

### Step 2: Create and upload the code archive

Tarball this repo and upload it to S3:

```bash
# From the dark_halo_scope directory
cd /path/to/dark_halo_scope
tar -zcvf dark_halo_scope_code.tgz .

# Upload to S3
aws s3 cp dark_halo_scope_code.tgz s3://MY_BUCKET/code/dark_halo_scope_code.tgz
```

### Step 3: Update config defaults (optional)

Edit `src/config.py` and update the `Phase1p5Config` EMR fields to match your
S3 bucket names and IAM roles:

```python
emr_s3_output_prefix: str = "s3://MY_BUCKET/phase1p5"
emr_s3_log_prefix: str = "s3://MY_BUCKET/emr-logs"
emr_s3_code_archive: str = "s3://MY_BUCKET/code/dark_halo_scope_code.tgz"
emr_service_role: str = "EMR_DefaultRole"
emr_job_flow_role: str = "EMR_EC2_DefaultRole"
```

### Step 4: Submit the EMR cluster

From an EC2 host (or anywhere with AWS credentials), run:

```bash
python -m emr.submit_phase1p5_emr_cluster \
    --region us-west-2 \
    --sweep-index-s3 s3://MY_BUCKET/dr10/sweeps_ra150_250_dec0_30_10.1.txt \
    --output-prefix s3://MY_BUCKET/dark_halo_scope/phase1p5 \
    --code-archive-s3 s3://MY_BUCKET/code/dark_halo_scope_code.tgz
```

### Step 5: Retrieve and merge results

After the EMR job completes, you will find a single-part CSV under:

```
s3://MY_BUCKET/dark_halo_scope/phase1p5/lrg_counts_per_brick/
```

Download that CSV:

```bash
aws s3 cp s3://MY_BUCKET/dark_halo_scope/phase1p5/lrg_counts_per_brick/ \
    ./outputs/phase1p5_emr/ --recursive
```

Then merge it with your brick QA table (from TAP or local FITS) on `brickname`,
compute `lrg_density = lrg_count / area_deg2`, and feed this into your existing
`select_regions` logic.

## Configuration Reference

The following EMR-related fields in `Phase1p5Config` control the cluster:

| Field | Description | Default |
|-------|-------------|---------|
| `emr_s3_output_prefix` | S3 prefix for job outputs | `s3://CHANGE_ME_BUCKET/phase1p5` |
| `emr_s3_log_prefix` | S3 prefix for EMR cluster logs | `s3://CHANGE_ME_BUCKET/emr-logs` |
| `emr_release_label` | EMR release version | `emr-6.15.0` |
| `emr_master_instance_type` | EC2 instance type for master | `m5.xlarge` |
| `emr_core_instance_type` | EC2 instance type for core nodes | `m5.xlarge` |
| `emr_core_instance_count` | Number of core nodes | `3` |
| `emr_job_name` | Name for the EMR cluster | `dark-halo-scope-phase1p5` |
| `emr_service_role` | IAM role for EMR service | `EMR_DefaultRole` |
| `emr_job_flow_role` | IAM role for EC2 instances | `EMR_EC2_DefaultRole` |
| `emr_s3_code_archive` | S3 path to the code tarball | `s3://CHANGE_ME_BUCKET/code/dark_halo_scope_code.tgz` |
| `emr_pyspark_driver_path` | Path to PySpark script in archive | `emr/spark_phase1p5_lrg_density.py` |

## Disk Usage and Resource Efficiency

- The EMR job never keeps more than one SWEEP FITS file per executor in
  local `/tmp`. This effectively sidesteps laptop disk space limits.

- S3 only stores the compact per-brick CSV, not the raw SWEEP FITS files
  (unless you independently decide to mirror SWEEPs to S3).

- The cluster is transient: it terminates automatically after the job
  completes (or fails), so you only pay for active compute time.

## Troubleshooting

### ⚠️ Common Issue: Out of Memory (OOM) Errors

**Symptom:**
```
OpenJDK 64-Bit Server VM warning: INFO: os::commit_memory(...) failed; error='Cannot allocate memory' (errno=12)
Command exiting with ret '134'
```

**Cause:** The JVM cannot allocate the heap space it needs. This happens when:
1. Instance type is too small (m5.xlarge has only 16 GB RAM)
2. Spark memory settings are not configured
3. Too many executors competing for memory

**Solutions:**

1. **Use larger instances (recommended):**
   ```bash
   --instance-type m5.2xlarge  # 32 GB RAM instead of 16 GB
   ```

2. **Use client deploy mode** (already set in updated code):
   ```bash
   --deploy-mode client  # Driver runs on master, not in YARN container
   ```

3. **Set explicit memory configs** (already set in updated code):
   ```bash
   --driver-memory 4g
   --executor-memory 4g
   --conf spark.executor.memoryOverhead=1g
   ```

### ⚠️ Common Issue: Job Takes Hours / Times Out

**Cause:** Processing too many SWEEP files (full-sky scan).

**Solution:** Limit the footprint and/or number of files:
```bash
# Process only a small region
--ra-min 150 --ra-max 200 --dec-min 0 --dec-max 15

# Or limit number of files for testing
--max-sweeps 10
```

### Debug Workflow

**Step 1: Test with the debug script first**

Before running the full job, verify your EMR setup works:

```bash
# SSH to EMR master node, then:
spark-submit \
    --deploy-mode client \
    --master yarn \
    --driver-memory 2g \
    --executor-memory 2g \
    --conf spark.executor.memoryOverhead=512m \
    /mnt/dark_halo_scope_code/emr/spark_hello_world_debug.py \
    --output-prefix s3://YOUR_BUCKET/debug
```

This script:
- Tests basic Spark RDD operations
- Verifies numpy and astropy are installed
- Tests S3 read/write
- Logs memory and Python version info

**Step 2: Test with a small subset**

```bash
python -m emr.submit_phase1p5_emr_cluster \
    --region us-east-2 \
    --sweep-index-s3 s3://bucket/sweep_urls.txt \
    --output-prefix s3://bucket/phase1p5_test \
    --code-archive-s3 s3://bucket/code/dark_halo_scope_code.tgz \
    --max-sweeps 5 \
    --ra-min 150 --ra-max 160 --dec-min 0 --dec-max 10
```

**Step 3: Scale up gradually**

Once the small test succeeds:
```bash
# Medium test: 50 files
--max-sweeps 50

# Full region: remove --max-sweeps
```

### Other Common Issues

1. **"Access Denied" errors on S3**
   - Ensure your IAM roles have `s3:GetObject`, `s3:PutObject`, and `s3:ListBucket`
     permissions for all relevant buckets.

2. **Bootstrap action failures**
   - Check EMR logs in the S3 log prefix for bootstrap stderr.
   - Ensure `aws s3 cp` can access the code archive location.
   - Look at: `s3://YOUR_LOG_BUCKET/j-CLUSTERID/node/i-INSTANCEID/bootstrap-actions/`

3. **Spark job failures with no logs**
   - This usually means the driver died before it could write logs
   - Use `--deploy-mode client` to run driver on master node (easier to debug)
   - SSH to master and check `/var/log/spark/` for local logs

4. **Empty output CSV**
   - Verify the SWEEP index file contains valid URLs or paths
   - Check that the RA/Dec bounds match your SWEEP coverage
   - Look at executor stderr for "[SWEEP]" log messages

5. **"File not found" for SWEEP files**
   - If using HTTP URLs: verify the URLs are accessible (test with `curl`)
   - If using S3 paths: verify IAM permissions and path format (s3://bucket/key)

### Viewing Logs

EMR logs are written to the S3 prefix specified in `emr_s3_log_prefix`:

```
s3://YOUR_BUCKET/emr-logs/j-CLUSTERID/
├── containers/
│   └── application_*/container_*/  # Executor logs
├── node/
│   └── i-INSTANCEID/
│       ├── applications/spark/    # Spark logs
│       └── bootstrap-actions/     # Bootstrap stderr
└── steps/
    └── s-STEPID/
        ├── stderr.gz              # Step stderr (most useful!)
        └── stdout.gz              # Step stdout
```

You can also view logs in the AWS EMR console under the cluster's "Steps" tab.

### Memory Guidelines by Instance Type

| Instance Type | RAM  | Recommended Config |
|---------------|------|-------------------|
| m5.xlarge     | 16 GB | Max 1 executor with 4G heap + 1G overhead (tight!) |
| m5.2xlarge    | 32 GB | 2-3 executors with 4G heap + 1G overhead each |
| m5.4xlarge    | 64 GB | 5-6 executors with 4G heap + 1G overhead each |
| r5.xlarge     | 32 GB | Memory-optimized, good for astropy workloads |

**Rule of thumb**: Each SWEEP file needs ~1-1.5 GB of RAM to process (astropy
loads the entire FITS table into memory). With 4G executor memory and 1G
overhead, you can safely process 1 file at a time per executor.

## Design Notes

This design keeps Phase 1.5 scientifically identical to the existing local
workflow while making it feasible to process all relevant SWEEPs in one pass.
The LRG selection cuts (`lrg_z_mag_max`, `lrg_min_r_minus_z`, `lrg_min_z_minus_w1`)
are the same as in the local workflow.

The EMR path is an **additional backend** for LRG counting, not a replacement
for the existing TAP-based or local SWEEP workflows. All existing functionality
remains unchanged.


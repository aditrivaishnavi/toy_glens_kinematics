# EMR Standard Steps for Plan B

This document describes the standard workflow for running EMR jobs in Plan B.

## Prerequisites

### 1. EMR-Launcher Machine Setup

If using a dedicated EMR-launcher machine:

```bash
# SSH to emr-launcher
ssh emr-launcher

# Create working directory
mkdir -p ~/planb/emr
cd ~/planb/emr

# Sync code from local machine
rsync -avz --exclude='*.pyc' --exclude='__pycache__' \
    local-machine:~/code/planb/emr/ ~/planb/emr/
```

### 2. AWS Credentials

Ensure AWS credentials are configured:

```bash
aws configure list
# Should show valid credentials

# Verify S3 access
aws s3 ls s3://darkhaloscope/

# Verify EMR access
aws emr list-clusters --active
```

### 3. Required IAM Roles

Ensure these IAM roles exist:
- `EMR_DefaultRole` - EMR service role
- `EMR_EC2_DefaultRole` - EC2 instance profile

```bash
aws iam get-role --role-name EMR_DefaultRole
aws iam get-role --role-name EMR_EC2_DefaultRole
```

---

## Standard Job Workflow

### Step 1: Prepare Script

1. **Copy template**:
   ```bash
   cp emr/spark_job_template.py emr/my_job.py
   ```

2. **Customize** the script:
   - Update `INPUT_SCHEMA` for your data
   - Implement `process_partition()` logic
   - Set appropriate output schema

3. **Validate locally** (syntax check):
   ```bash
   python3 -m py_compile emr/my_job.py
   ```

### Step 2: Upload Bootstrap to S3

```bash
aws s3 cp emr/bootstrap.sh s3://darkhaloscope/planb/emr/code/bootstrap.sh
```

### Step 3: Run Smoke Test

**Always run a small test first!**

```bash
python3 emr/launcher.py submit \
    --job-name my-job-smoke \
    --script emr/my_job.py \
    --preset small \
    --args "--input s3://darkhaloscope/planb/input/sample --output s3://darkhaloscope/planb/output/smoke" \
    --wait \
    --timeout 30
```

### Step 4: Verify Smoke Test Output

```bash
# Check output exists
aws s3 ls s3://darkhaloscope/planb/output/smoke/

# Download and inspect sample
aws s3 cp s3://darkhaloscope/planb/output/smoke/part-00000.snappy.parquet /tmp/
python3 -c "import pandas as pd; print(pd.read_parquet('/tmp/part-00000.snappy.parquet').head())"
```

### Step 5: Run Production Job

```bash
python3 emr/launcher.py submit \
    --job-name my-job-prod \
    --script emr/my_job.py \
    --preset medium \
    --args "--input s3://darkhaloscope/planb/input/full --output s3://darkhaloscope/planb/output/prod" \
    --wait \
    --timeout 120
```

### Step 6: Monitor Job

```bash
# Get cluster status
python3 emr/launcher.py status --cluster-id j-XXXXX

# View logs in AWS Console
# https://console.aws.amazon.com/emr/home?region=us-west-2#/clusterDetails/j-XXXXX

# Or via CLI
aws emr ssh --cluster-id j-XXXXX --key-pair-file ~/.ssh/emr-key.pem \
    --command "tail -f /var/log/spark/spark.log"
```

### Step 7: Post-Job Validation

```bash
# Count output files
aws s3 ls s3://darkhaloscope/planb/output/prod/ --recursive | wc -l

# Check for _SUCCESS file
aws s3 ls s3://darkhaloscope/planb/output/prod/_SUCCESS

# Validate output data
python3 -c "
import pandas as pd
import s3fs

fs = s3fs.S3FileSystem()
files = fs.glob('s3://darkhaloscope/planb/output/prod/*.parquet')
print(f'Found {len(files)} parquet files')

# Read sample
df = pd.read_parquet(f's3://{files[0]}')
print(f'Columns: {df.columns.tolist()}')
print(f'Shape: {df.shape}')
print(df.head())
"
```

---

## Job Presets

| Preset | Workers | Instance Type | Memory | Use Case |
|--------|---------|---------------|--------|----------|
| `small` | 2 | m5.xlarge | 4g/exec | Smoke tests, validation |
| `medium` | 10 | m5.2xlarge | 8g/exec | Single-split processing |
| `large` | 20 | m5.4xlarge | 16g/exec | Full pipeline |

### Custom Configuration

Override preset settings:

```bash
python3 emr/launcher.py submit \
    --job-name custom-job \
    --script my_job.py \
    --preset medium \
    --workers 15 \
    --instance-type m5.4xlarge \
    --args "--input s3://... --output s3://..."
```

---

## Phase-Specific Jobs

### Phase 0: Data Validation

**0.2.1 Verify Split Integrity**
```bash
python3 emr/launcher.py submit \
    --job-name phase0-verify-splits \
    --script emr/jobs/spark_verify_splits.py \
    --preset small \
    --args "--parquet-root s3://darkhaloscope/v5_cosmos_paired" \
    --wait
```

**0.2.2 Verify Paired Data**
```bash
python3 emr/launcher.py submit \
    --job-name phase0-verify-paired \
    --script emr/jobs/spark_verify_paired.py \
    --preset small \
    --args "--parquet-root s3://darkhaloscope/v5_cosmos_paired --sample-rate 0.01" \
    --wait
```

### Phase 2: Gen7 Data Generation

```bash
# Smoke test first
python3 emr/launcher.py submit \
    --job-name gen7-smoke \
    --script emr/jobs/spark_gen7_injection.py \
    --preset small \
    --args "--input s3://darkhaloscope/v5_cosmos_paired/train --output s3://darkhaloscope/planb/gen7_smoke --limit 1000" \
    --wait

# Full production run
python3 emr/launcher.py submit \
    --job-name gen7-train \
    --script emr/jobs/spark_gen7_injection.py \
    --preset large \
    --args "--input s3://darkhaloscope/v5_cosmos_paired/train --output s3://darkhaloscope/planb/gen7/train" \
    --wait \
    --timeout 240
```

### Phase 3: Gen8 Domain Randomization

```bash
# Smoke test first
python3 emr/launcher.py submit \
    --job-name gen8-smoke \
    --script emr/jobs/spark_gen8_injection.py \
    --preset small \
    --args "--input s3://darkhaloscope/v5_cosmos_paired/train --output s3://darkhaloscope/planb/gen8_smoke --limit 1000" \
    --wait

# Full production run with calibrated rates
python3 emr/launcher.py submit \
    --job-name gen8-train \
    --script emr/jobs/spark_gen8_injection.py \
    --preset large \
    --args "--input s3://darkhaloscope/v5_cosmos_paired/train --output s3://darkhaloscope/planb/gen8/train --cosmic-rate 0.12 --sat-rate 0.06" \
    --wait \
    --timeout 240
```

### Phase 4: Score All Stamps

```bash
python3 emr/launcher.py submit \
    --job-name score-test \
    --script emr/jobs/spark_score_all.py \
    --preset large \
    --args "--input s3://darkhaloscope/v5_cosmos_paired/test --checkpoint s3://darkhaloscope/planb/checkpoints/best_model.pt --output s3://darkhaloscope/planb/scores/test" \
    --wait \
    --timeout 120
```

---

## Troubleshooting

### Common Issues

1. **Cluster stuck in STARTING**
   - Check EC2 limits in your account
   - Verify subnet has available IPs
   - Try a different availability zone

2. **Bootstrap failure**
   - Check bootstrap.sh was uploaded correctly
   - View bootstrap logs: `/var/log/bootstrap-actions/`

3. **Step failed**
   - Check Spark logs: `/var/log/spark/`
   - Look for Python errors in stderr

4. **Spot termination**
   - Increase `spot_bid_percent` in constants.py
   - Or set `use_spot: false` for critical jobs

### Debug Mode

Keep cluster alive for debugging:

```bash
python3 emr/launcher.py submit \
    --job-name debug-job \
    --script my_job.py \
    --preset small \
    --keep-alive \
    --args "..."

# SSH to cluster
aws emr ssh --cluster-id j-XXXXX --key-pair-file ~/.ssh/emr-key.pem

# Run interactive PySpark
pyspark

# When done, terminate
python3 emr/launcher.py terminate --cluster-id j-XXXXX
```

---

## Checklist

Before running any EMR job:

- [ ] Script passes syntax check (`python3 -m py_compile`)
- [ ] Input S3 path exists and is accessible
- [ ] Output S3 path is writable (and empty or you want to overwrite)
- [ ] Bootstrap script is uploaded to S3
- [ ] AWS credentials are valid
- [ ] Smoke test completed successfully
- [ ] Expected runtime estimated

After job completes:

- [ ] Check exit code (0 = success)
- [ ] Verify output files exist
- [ ] Check `_SUCCESS` marker file
- [ ] Validate output data format
- [ ] Check for empty partitions
- [ ] Compare row counts (input vs output if applicable)

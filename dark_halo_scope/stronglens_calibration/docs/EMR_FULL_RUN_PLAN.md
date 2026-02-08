# EMR Full Run Plan: Negative Sampling Pipeline

**Created:** 2026-02-07  
**Purpose:** Complete execution plan for the negative sampling EMR job  
**Target:** ~250K stratified negatives from DR10 sweep files

---

## Table of Contents

1. [Overview](#1-overview)
2. [Upstream Dependencies](#2-upstream-dependencies)
3. [Pre-Flight Checklist](#3-pre-flight-checklist)
4. [Data Preparation](#4-data-preparation)
5. [EMR Launch Sequence](#5-emr-launch-sequence)
6. [Validation Gates](#6-validation-gates)
7. [Post-Run Verification](#7-post-run-verification)
8. [Rollback Plan](#8-rollback-plan)
9. [Cost Estimate](#9-cost-estimate)
10. [Timeline](#10-timeline)

---

## 1. Overview

### Goal
Generate a stratified negative sample manifest from DR10 sweep files:
- **100:1 negative:positive ratio** per (nobs_z, type) bin
- **85:15 N1:N2 split** (deployment-representative : hard confusers)
- **70/15/15 train/val/test** spatial splits via HEALPix-128
- **~250K negatives** total (based on ~2,500 positives)

### Pipeline Flow
```
DR10 Sweep Files (S3)
        ↓
    EMR Spark Job
        ↓
   ┌────┴────┐
   │ Filter  │ → Exclude known lenses, maskbits, faint sources
   └────┬────┘
        ↓
   ┌────┴────┐
   │ Classify│ → Pool N1/N2, type_bin, nobs_z_bin
   └────┬────┘
        ↓
   ┌────┴────┐
   │ Split   │ → HEALPix-128 → train/val/test
   └────┬────┘
        ↓
   ┌────┴────┐
   │ Output  │ → Parquet manifest with full schema
   └────┬────┘
        ↓
Negative Manifest (S3)
```

---

## 2. Upstream Dependencies

### 2.1 Data Dependencies

| Dependency | Location | Size | Status | Required By |
|------------|----------|------|--------|-------------|
| DR10 Sweep Files | S3 or NERSC | ~900 GB (full sky) | ⚠️ TBD | EMR Job Input |
| Positive Catalog | `data/positives/desi_candidates.csv` | 5,104 rows | ✅ Ready | Lens Exclusion |
| Spectroscopic Catalog | `data/external/desi_dr1/desi-sl-vac-v1.fits` | 2,176 rows | ✅ Ready | Cross-validation |
| Configuration | `configs/negative_sampling_v1.yaml` | - | ✅ Ready | EMR Job Config |

### 2.2 Infrastructure Dependencies

| Dependency | Requirement | Status | How to Verify |
|------------|-------------|--------|---------------|
| AWS Account | Valid credentials | ⚠️ TBD | `aws sts get-caller-identity` |
| EMR Permissions | `elasticmapreduce:*` | ⚠️ TBD | IAM policy check |
| S3 Bucket | `s3://darkhaloscope` access | ⚠️ TBD | `aws s3 ls s3://darkhaloscope/` |
| EMR Roles | `EMR_EC2_DefaultRole`, `EMR_DefaultRole` | ⚠️ TBD | `aws iam get-role --role-name EMR_DefaultRole` |
| VPC/Subnet | EMR-compatible subnet | ⚠️ TBD | `aws ec2 describe-subnets` |
| EC2 Quota | ≥280 vCPUs in region | ⚠️ TBD | EC2 quota dashboard |

### 2.3 Software Dependencies

| Package | Version | Purpose | Installation |
|---------|---------|---------|--------------|
| boto3 | ≥1.26 | AWS SDK | `pip install boto3` |
| pyspark | 3.4+ | Spark APIs | EMR pre-installed |
| numpy | ≥1.20 | Numerical ops | EMR pre-installed |
| pandas | ≥1.3 | Data validation | `pip install pandas` |
| healpy | ≥1.15 | HEALPix computation | Bootstrap install |
| astropy | ≥5.0 | FITS handling | Bootstrap install |
| PyYAML | ≥6.0 | Config parsing | Bootstrap install |

---

## 3. Pre-Flight Checklist

### 3.1 Local Environment ✓

```bash
# Run from stronglens_calibration/
cd /path/to/stronglens_calibration

# 1. Verify unit tests pass
python3 tests/test_phase1_local.py
# Expected: ALL TESTS PASSED

# 2. Verify pipeline test passes  
python3 tests/test_pipeline_local.py --rows 1000 --no-save
# Expected: PIPELINE TEST PASSED

# 3. Check code syntax
python3 -m py_compile emr/spark_negative_sampling.py
python3 -m py_compile emr/sampling_utils.py
# Expected: No output (no errors)

# 4. Verify config is valid
python3 -c "import yaml; yaml.safe_load(open('configs/negative_sampling_v1.yaml'))"
# Expected: No errors
```

### 3.2 AWS Access ✓

```bash
# 1. Verify credentials
aws sts get-caller-identity
# Expected: Account ID, ARN

# 2. Verify S3 access
aws s3 ls s3://darkhaloscope/ --region us-west-2
# Expected: List of prefixes

# 3. Verify EMR permissions
aws emr list-clusters --region us-west-2 --active
# Expected: Empty list or existing clusters

# 4. Check EC2 quota
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-1216C47A \
  --region us-west-2
# Expected: Value >= 280
```

### 3.3 Data Availability ✓

```bash
# 1. Check sweep files exist in S3
aws s3 ls s3://darkhaloscope/dr10/sweeps/ --region us-west-2 | head -10
# Expected: List of sweep-*.fits files

# If sweep files not in S3, check alternative sources:
# - NERSC: https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/sweep/10.0/
# - Local: data/sweep_files/

# 2. Verify positive catalog
wc -l data/positives/desi_candidates.csv
# Expected: 5105 (5104 + header)

# 3. Check spectroscopic catalog
python3 -c "from astropy.io import fits; print(len(fits.open('data/external/desi_dr1/desi-sl-vac-v1.fits')[1].data))"
# Expected: 2176
```

### 3.4 Configuration Validation ✓

```bash
python3 << 'EOF'
import yaml

with open('configs/negative_sampling_v1.yaml') as f:
    config = yaml.safe_load(f)

# Check critical settings
assert config['version'] == '1.0.0', "Version mismatch"
assert config['negative_pools']['n1_n2_ratio'] == [85, 15], "N1:N2 ratio wrong"
assert config['negative_pools']['neg_pos_ratio'] == 100, "Neg:Pos ratio wrong"
assert config['spatial_splits']['nside'] == 128, "HEALPix nside wrong"
assert config['exclusion']['known_lens_radius_arcsec'] == 11.0, "Exclusion radius wrong"

print("✓ Configuration validated")
EOF
```

---

## 4. Data Preparation

### 4.1 Sweep File Strategy

**Option A: Use existing S3 sweep files (Recommended if available)**
```bash
# Check if sweeps exist
aws s3 ls s3://darkhaloscope/dr10/sweeps/ --summarize
# If exists: Use --sweep-input s3://darkhaloscope/dr10/sweeps/
```

**Option B: Download from NERSC to S3**
```bash
# Download sweep files covering positive locations
python3 << 'EOF'
import pandas as pd
from emr.sweep_utils import get_unique_sweep_files_for_positions

# Get unique sweep files needed for positives
df = pd.read_csv('data/positives/desi_candidates.csv')
positions = list(zip(df['ra'], df['dec']))
sweep_files = get_unique_sweep_files_for_positions(positions)
print(f"Need {len(sweep_files)} sweep files")
for f in sorted(sweep_files)[:10]:
    print(f"  {f}")
EOF

# Download script (run on EC2 or machine with good bandwidth)
# See: emr/sweep_utils.py download_sweep_file()
```

**Option C: Use local prototype for testing**
```bash
# Convert CSV to Parquet for Spark
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('data/negatives/negative_catalog_prototype.csv')
df.to_parquet('data/negatives/negative_catalog_prototype.parquet')
print(f"Converted {len(df)} rows to Parquet")
EOF

# Upload to S3
aws s3 cp data/negatives/negative_catalog_prototype.parquet \
  s3://darkhaloscope/stronglens_calibration/test_input/
```

### 4.2 Upload Dependencies to S3

```bash
# 1. Upload positive catalog for exclusion
aws s3 cp data/positives/desi_candidates.csv \
  s3://darkhaloscope/stronglens_calibration/catalogs/desi_candidates.csv

# 2. Upload configuration
aws s3 cp configs/negative_sampling_v1.yaml \
  s3://darkhaloscope/stronglens_calibration/configs/negative_sampling_v1.yaml

# 3. Upload code
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
aws s3 cp emr/spark_negative_sampling.py \
  s3://darkhaloscope/stronglens_calibration/code/${TIMESTAMP}/spark_negative_sampling.py
aws s3 cp emr/sampling_utils.py \
  s3://darkhaloscope/stronglens_calibration/code/${TIMESTAMP}/sampling_utils.py

# 4. Verify uploads
aws s3 ls s3://darkhaloscope/stronglens_calibration/ --recursive
```

### 4.3 Create Bootstrap Script

```bash
cat > /tmp/emr_bootstrap.sh << 'EOF'
#!/bin/bash
set -ex

# Install Python dependencies
sudo pip3 install healpy astropy pyyaml

# Create numba cache directory (lesson L2.4)
mkdir -p /tmp/numba_cache
chmod 777 /tmp/numba_cache
export NUMBA_CACHE_DIR=/tmp/numba_cache

echo "Bootstrap complete"
EOF

aws s3 cp /tmp/emr_bootstrap.sh \
  s3://darkhaloscope/stronglens_calibration/bootstrap/emr_bootstrap.sh
```

---

## 5. EMR Launch Sequence

### 5.1 Mini-Test Run (REQUIRED FIRST)

```bash
# Launch mini-test with 2 workers, 2 partitions
python3 emr/launch_negative_sampling.py \
  --test \
  --sweep-input s3://darkhaloscope/stronglens_calibration/test_input/ \
  2>&1 | tee logs/emr_mini_test_$(date +%Y%m%d_%H%M%S).log

# Expected output:
# - Cluster launches in ~5 min
# - Step completes in ~10 min
# - Output: s3://darkhaloscope/stronglens_calibration/manifests/TIMESTAMP/
```

### 5.2 Validate Mini-Test Output

```bash
# 1. Download sample output
aws s3 cp s3://darkhaloscope/stronglens_calibration/manifests/MINI_TIMESTAMP/ \
  /tmp/mini_test_output/ --recursive

# 2. Verify schema
python3 << 'EOF'
import pandas as pd
import os

output_dir = '/tmp/mini_test_output'
files = [f for f in os.listdir(output_dir) if f.endswith('.parquet')]
df = pd.read_parquet(os.path.join(output_dir, files[0]))

print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nPool distribution:")
print(df['pool'].value_counts())
print(f"\nSplit distribution:")
print(df['split'].value_counts())
print(f"\nType distribution:")
print(df['type_bin'].value_counts())

# Verify no nulls
for col in ['galaxy_id', 'ra', 'dec', 'pool', 'split']:
    assert df[col].notna().all(), f"Nulls in {col}"

print("\n✓ Mini-test output validated")
EOF
```

### 5.3 Full Production Run

```bash
# Launch full run with 25 workers
python3 emr/launch_negative_sampling.py \
  --full \
  --preset large \
  --sweep-input s3://darkhaloscope/dr10/sweeps/ \
  2>&1 | tee logs/emr_full_run_$(date +%Y%m%d_%H%M%S).log

# Monitor progress
# Job should complete in ~2-4 hours depending on data size
```

### 5.4 Alternative: Manual EMR Launch

If the launcher script fails, use AWS CLI directly:

```bash
# 1. Create cluster
aws emr create-cluster \
  --name "stronglens-negative-sampling-$(date +%Y%m%d)" \
  --release-label emr-7.0.0 \
  --applications Name=Spark Name=Hadoop \
  --instance-groups \
    InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m5.xlarge \
    InstanceGroupType=CORE,InstanceCount=25,InstanceType=m5.2xlarge,BidPrice=OnDemandPrice \
  --use-default-roles \
  --log-uri s3://darkhaloscope/stronglens_calibration/logs/ \
  --bootstrap-actions Path=s3://darkhaloscope/stronglens_calibration/bootstrap/emr_bootstrap.sh \
  --region us-west-2

# 2. Wait for WAITING state
aws emr describe-cluster --cluster-id j-XXXXX --query 'Cluster.Status.State'

# 3. Submit step
aws emr add-steps --cluster-id j-XXXXX --steps \
  "Type=Spark,Name=NegativeSampling,ActionOnFailure=CONTINUE,\
  Args=[--deploy-mode,cluster,\
  --py-files,s3://darkhaloscope/stronglens_calibration/code/TIMESTAMP/sampling_utils.py,\
  s3://darkhaloscope/stronglens_calibration/code/TIMESTAMP/spark_negative_sampling.py,\
  --config,s3://darkhaloscope/stronglens_calibration/configs/negative_sampling_v1.yaml,\
  --sweep-input,s3://darkhaloscope/dr10/sweeps/,\
  --positive-catalog,s3://darkhaloscope/stronglens_calibration/catalogs/desi_candidates.csv,\
  --output,s3://darkhaloscope/stronglens_calibration/manifests/$(date +%Y%m%d_%H%M%S)/]"
```

---

## 6. Validation Gates

### Gate 1: Pre-Submission (Before EMR)
| Check | Command | Expected |
|-------|---------|----------|
| Unit tests pass | `python3 tests/test_phase1_local.py` | 12/12 pass |
| Pipeline test pass | `python3 tests/test_pipeline_local.py --rows 1000` | 5/5 checks |
| AWS credentials valid | `aws sts get-caller-identity` | Returns account |
| S3 input exists | `aws s3 ls s3://darkhaloscope/dr10/sweeps/` | Lists files |
| Config valid | See 3.4 | No errors |

### Gate 2: Cluster Ready (After Launch)
| Check | Command | Expected |
|-------|---------|----------|
| Cluster state | `aws emr describe-cluster --cluster-id j-XXX` | WAITING |
| All nodes running | Check EMR console | Green status |
| Bootstrap success | Check bootstrap logs | "Bootstrap complete" |

### Gate 3: Step Running (During Execution)
| Check | Command | Expected |
|-------|---------|----------|
| Step state | `aws emr describe-step --cluster-id j-XXX --step-id s-YYY` | RUNNING |
| Spark UI | EMR console → Application history | Jobs active |
| No errors in logs | Check stderr in S3 logs | No exceptions |

### Gate 4: Output Validation (After Completion)
| Check | Expected | Critical? |
|-------|----------|-----------|
| Files exist in S3 | Multiple .parquet files | Yes |
| Row count | ~250K (adjust based on input) | Yes |
| No nulls in critical columns | ra, dec, pool, split | Yes |
| Pool ratio ~85:15 | N1:N2 within 10% | No |
| Split ratio ~70:15:15 | Within 5% | No |
| No duplicates | 0 duplicate galaxy_ids | Yes |
| Schema complete | All 25+ columns present | Yes |

---

## 7. Post-Run Verification

### 7.1 Download and Validate Output

```bash
# 1. List output files
OUTPUT_PATH="s3://darkhaloscope/stronglens_calibration/manifests/FULL_TIMESTAMP/"
aws s3 ls ${OUTPUT_PATH} --summarize

# 2. Download all files (or sample)
mkdir -p output/full_run
aws s3 sync ${OUTPUT_PATH} output/full_run/

# 3. Run validation script
python3 << 'EOF'
import pandas as pd
import os
from pathlib import Path

output_dir = Path('output/full_run')
parquet_files = list(output_dir.glob('*.parquet'))
print(f"Found {len(parquet_files)} parquet files")

# Read all files
dfs = [pd.read_parquet(f) for f in parquet_files]
df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows: {len(df):,}")

# Validation checks
checks = {}

# 1. No nulls in critical columns
critical_cols = ['galaxy_id', 'ra', 'dec', 'type', 'nobs_z', 'pool', 'split']
for col in critical_cols:
    null_count = df[col].isna().sum()
    checks[f'null_{col}'] = null_count == 0
    if null_count > 0:
        print(f"  WARNING: {null_count} nulls in {col}")

# 2. No duplicates
unique_count = df['galaxy_id'].nunique()
dup_count = len(df) - unique_count
checks['no_duplicates'] = dup_count == 0
print(f"\nDuplicates: {dup_count}")

# 3. Pool distribution
pool_dist = df['pool'].value_counts(normalize=True) * 100
print(f"\nPool distribution:")
for pool, pct in pool_dist.items():
    print(f"  {pool}: {pct:.1f}%")
n1_pct = pool_dist.get('N1', 0)
checks['n1_n2_ratio'] = 75 < n1_pct < 95  # 85 ± 10

# 4. Split distribution
split_dist = df['split'].value_counts(normalize=True) * 100
print(f"\nSplit distribution:")
for split, pct in split_dist.items():
    print(f"  {split}: {pct:.1f}%")
train_pct = split_dist.get('train', 0)
checks['split_ratio'] = 65 < train_pct < 75  # 70 ± 5

# 5. Type distribution
type_dist = df['type_bin'].value_counts()
print(f"\nType distribution:")
for t, count in type_dist.items():
    print(f"  {t}: {count:,}")
checks['all_types'] = len(type_dist) >= 4

# 6. Coordinate ranges
checks['ra_range'] = df['ra'].between(0, 360).all()
checks['dec_range'] = df['dec'].between(-90, 90).all()
print(f"\nRA range: {df['ra'].min():.2f} to {df['ra'].max():.2f}")
print(f"Dec range: {df['dec'].min():.2f} to {df['dec'].max():.2f}")

# Summary
print("\n" + "="*50)
print("VALIDATION SUMMARY")
print("="*50)
passed = sum(checks.values())
total = len(checks)
for check, result in checks.items():
    status = "✓" if result else "✗"
    print(f"  {status} {check}")

print(f"\nPassed: {passed}/{total}")
if passed == total:
    print("\n✓ ALL CHECKS PASSED - Output is valid")
else:
    print("\n✗ SOME CHECKS FAILED - Review before proceeding")
EOF
```

### 7.2 Save Validation Report

```bash
# Save statistics to JSON
python3 << 'EOF'
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

output_dir = Path('output/full_run')
parquet_files = list(output_dir.glob('*.parquet'))
dfs = [pd.read_parquet(f) for f in parquet_files]
df = pd.concat(dfs, ignore_index=True)

report = {
    'timestamp': datetime.utcnow().isoformat(),
    'total_rows': len(df),
    'unique_galaxies': df['galaxy_id'].nunique(),
    'pool_distribution': df['pool'].value_counts().to_dict(),
    'split_distribution': df['split'].value_counts().to_dict(),
    'type_distribution': df['type_bin'].value_counts().to_dict(),
    'nobs_distribution': df['nobs_z_bin'].value_counts().to_dict(),
    'ra_range': [float(df['ra'].min()), float(df['ra'].max())],
    'dec_range': [float(df['dec'].min()), float(df['dec'].max())],
    'git_commits': df['git_commit'].unique().tolist(),
}

with open('output/full_run_validation.json', 'w') as f:
    json.dump(report, f, indent=2)

print("Saved validation report to output/full_run_validation.json")
EOF
```

---

## 8. Rollback Plan

### If Cluster Fails to Start
1. Check EMR console for error message
2. Common issues:
   - Insufficient EC2 capacity → Try different instance types or region
   - IAM role issues → Verify EMR_DefaultRole exists
   - Subnet issues → Verify subnet has internet gateway
3. Terminate failed cluster: `aws emr terminate-job-flows --job-flow-ids j-XXX`

### If Step Fails
1. Check step logs in S3: `s3://darkhaloscope/stronglens_calibration/logs/j-XXX/steps/s-YYY/`
2. Common issues:
   - Import errors → Check bootstrap installed all packages
   - S3 access denied → Check IAM permissions
   - Out of memory → Reduce parallelism or use larger instances
3. Fix issue and resubmit step (don't recreate cluster)

### If Output is Invalid
1. Keep cluster running for debugging
2. SSH to master node: `aws emr ssh --cluster-id j-XXX --key-pair-file ~/.ssh/emr.pem`
3. Check Spark history server for detailed logs
4. Fix code, re-upload, and resubmit step

### Emergency Termination
```bash
# Force terminate cluster
aws emr terminate-job-flows --job-flow-ids j-XXX --region us-west-2

# Verify termination
aws emr describe-cluster --cluster-id j-XXX --query 'Cluster.Status.State'
```

---

## 9. Cost Estimate

### Mini-Test Run
| Resource | Count | Type | Hours | Cost |
|----------|-------|------|-------|------|
| Master | 1 | m5.xlarge | 0.5 | $0.10 |
| Workers | 2 | m5.xlarge (spot) | 0.5 | $0.10 |
| **Total** | | | | **~$0.20** |

### Full Production Run
| Resource | Count | Type | Hours | Cost |
|----------|-------|------|-------|------|
| Master | 1 | m5.xlarge | 4 | $0.77 |
| Workers | 25 | m5.2xlarge (spot) | 4 | $19.20 |
| S3 Storage | - | 10 GB output | - | $0.23 |
| S3 Transfer | - | 100 GB read | - | $0.00 (same region) |
| **Total** | | | | **~$20-25** |

*Spot pricing assumes 50% of on-demand. Actual costs may vary.*

---

## 10. Timeline

### Preparation Phase (Day 1)
| Time | Task | Duration |
|------|------|----------|
| T+0h | Run pre-flight checks | 15 min |
| T+0.5h | Upload data to S3 | 30 min |
| T+1h | Launch mini-test | 15 min |
| T+1.5h | Validate mini-test output | 30 min |
| T+2h | **Decision: Proceed with full run?** | - |

### Production Run (Day 1-2)
| Time | Task | Duration |
|------|------|----------|
| T+2h | Launch full EMR cluster | 10 min |
| T+2.5h | Wait for cluster ready | 15 min |
| T+3h | Submit production step | 5 min |
| T+3h - T+7h | **Monitor step execution** | 4 hours |
| T+7h | Download and validate output | 30 min |
| T+8h | Terminate cluster | 5 min |
| T+8h | **Full run complete** | - |

### Post-Processing (Day 2)
| Time | Task | Duration |
|------|------|----------|
| T+8h | Generate validation report | 15 min |
| T+9h | Update checklist | 15 min |
| T+9h | Archive logs | 15 min |
| T+10h | **Phase 1 complete** | - |

---

## Appendix A: Quick Reference Commands

```bash
# Check cluster status
aws emr describe-cluster --cluster-id j-XXX --query 'Cluster.Status'

# Check step status
aws emr list-steps --cluster-id j-XXX

# Stream step logs
aws s3 cp s3://darkhaloscope/stronglens_calibration/logs/j-XXX/steps/s-YYY/stderr.gz - | gunzip

# Terminate cluster
aws emr terminate-job-flows --job-flow-ids j-XXX

# Count output rows
aws s3 cp s3://darkhaloscope/stronglens_calibration/manifests/TIMESTAMP/part-00000.parquet - | \
  python3 -c "import pandas as pd, sys; print(len(pd.read_parquet(sys.stdin.buffer)))"
```

---

## Appendix B: Troubleshooting

### "No module named 'healpy'"
- Bootstrap script didn't run
- Fix: Re-upload bootstrap and restart cluster

### "Access Denied" on S3
- IAM role missing S3 permissions
- Fix: Add s3:GetObject, s3:PutObject, s3:ListBucket to EMR_EC2_DefaultRole

### "Executor lost"
- Workers running out of memory
- Fix: Reduce spark.executor.memory or use larger instances

### "Task timeout"
- Some partitions taking too long
- Fix: Increase spark.network.timeout, check for data skew

### "Output has unexpected schema"
- Code version mismatch
- Fix: Verify S3 code matches local, check git_commit in output

---

*Plan created: 2026-02-07*  
*Last updated: 2026-02-07*

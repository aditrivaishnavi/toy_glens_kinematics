# Project Handoff: Gravitational Lens Detection Gen5 COSMOS Integration

**Date:** 2026-02-04  
**Current State:** Gen5 training running on Lambda, Corrected 4c EMR job launching  
**Active Jobs:**
- **Training:** Gen5 Phase 5 on Lambda (Epoch 8, loss=0.0009)
- **EMR 4c Corrected:** Cluster `j-20V17FPRYWQ7N` (with arc_snr_sum, lensed_hlr_arcsec fixes)

**Output Paths:**
- Original 4c: `s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_production/`
- Corrected 4c: `s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_corrected/`

**Next Phase:** After 4c corrected completes → Run compaction with split relabeling

---

## 🚨 CRITICAL SESSION UPDATE (2026-02-03 17:30 UTC) 🚨

### Bug Found and Fixed
**Error:** `local variable 'source_mode' referenced before assignment`

**Root Cause:** The variable `source_mode` was defined inside `if theta_e > 0:` block but referenced outside it. When processing control samples (theta_e=0), the variable was never defined.

**Fix Applied:** Moved `source_mode = getattr(args, 'source_mode', 'sersic')` and `cosmos_idx = None`, `cosmos_hlr = None` declarations BEFORE the if block.

**Commit:** `f3b0672` - Already pushed to GitHub and uploaded to S3

### Current EMR Job
- **Cluster ID:** `j-2O66SUDK8WZGC` (smoke test, 2 nodes)
- **Status:** RUNNING - validating the source_mode fix
- **Check status:**
```bash
ssh emr-launcher 'aws emr describe-cluster --cluster-id j-2O66SUDK8WZGC --region us-east-2 --query "Cluster.Status.State" --output text'
```

### How to Validate the Fix
```bash
# Check output after job completes:
ssh emr-launcher 'python3 << "EOF"
import pyarrow.parquet as pq
import numpy as np
import io

table = pq.read_table("s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_source_test/stamps/boto3_fix/", 
                      columns=["stamp_npz", "cutout_ok", "source_mode", "cosmos_index", "physics_warnings"])
print(f"Total rows: {len(table)}")
cutout_ok_vals = [x.as_py() for x in table["cutout_ok"]]
unique, counts = np.unique(cutout_ok_vals, return_counts=True)
print(f"cutout_ok: {dict(zip(unique.tolist(), counts.tolist()))}")

# Check for errors in physics_warnings
for i in range(min(5, len(table))):
    pw = table["physics_warnings"][i].as_py()
    ok = table["cutout_ok"][i].as_py()
    if pw:
        print(f"Row {i}: cutout_ok={ok}, error={pw}")
    elif ok == 1:
        blob = table["stamp_npz"][i].as_py()
        if blob:
            npz = np.load(io.BytesIO(blob))
            img = npz["image_r"]
            print(f"Row {i}: cutout_ok={ok}, stamp sum={img.sum():.2f}, max={img.max():.4f}")
EOF
'
```

### If Smoke Test PASSES (cutout_ok=1 for some rows):
Launch full production run with 270 vCPUs:
```bash
ssh emr-launcher 'aws emr create-cluster \
  --name "Gen5-Phase4c-PRODUCTION" \
  --region us-east-2 \
  --release-label emr-7.0.0 \
  --applications Name=Spark \
  --instance-groups "[{\"Name\":\"Master\",\"InstanceGroupType\":\"MASTER\",\"InstanceType\":\"m5.2xlarge\",\"InstanceCount\":1},{\"Name\":\"Core\",\"InstanceGroupType\":\"CORE\",\"InstanceType\":\"m5.2xlarge\",\"InstanceCount\":33}]" \
  --use-default-roles \
  --log-uri s3://darkhaloscope/emr-logs/ \
  --bootstrap-actions Path=s3://darkhaloscope/scripts/gen5/emr_bootstrap_gen5.sh \
  --auto-terminate \
  --steps "Type=Spark,Name=Gen5-Phase4c-Production,ActionOnFailure=TERMINATE_CLUSTER,Args=[--deploy-mode,cluster,--master,yarn,--num-executors,32,--executor-memory,12g,--executor-cores,4,s3://darkhaloscope/scripts/gen5/spark_phase4_pipeline_gen5.py,--stage,4c,--parent-s3,s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_source/manifests/,--variant,v5_cosmos_source,--experiment-id,train_stamp64_bandsgrz_cosmos,--s3-output,s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_source/stamps/,--psf-model,moffat,--moffat-beta,3.5,--source-mode,cosmos,--cosmos-bank-h5,s3://darkhaloscope/cosmos_bank/cosmos_sources_v1.h5]" \
  --query ClusterId --output text'
```

### If Smoke Test FAILS (all cutout_ok=0):
Check physics_warnings for error messages and debug.

---

## 🚨 CRITICAL: Always Use US-EAST-2 for EMR! 🚨

**Region:** `us-east-2` (NOT us-east-1!)  
**Reason:** All data, vCPU quotas, and infrastructure are in us-east-2  
**Mistake made:** Wasted ~2 hours and multiple cluster launches trying us-east-1 before realizing all resources are in us-east-2

---

## 1. Server Access & Credentials

### Lambda GPU Instance (GH200)
```bash
# SSH Access
ssh -i ~/code/toy_glens_kinematics/dark_halo_scope/lambda.pem ubuntu@192.222.56.237

# Working Directory
cd /lambda/nfs/darkhaloscope-training-dc/
```

**Key Information:**
- **Purpose:** Model training on GH200 GPU
- **Current model:** Gen4 training completed
- **Location:** `/lambda/nfs/darkhaloscope-training-dc/`

### EMR Launcher (AWS EC2)

**Option 1: Direct SSH**
```bash
ssh -i ~/.ssh/id_ed25519 ec2-user@ec2-18-118-154-166.us-east-2.compute.amazonaws.com
```

**Option 2: Using SSH Config (RECOMMENDED)**
```bash
# Add to ~/.ssh/config
Host emr-launcher
    HostName 18.119.128.26
    User ec2-user
    IdentityFile /Users/balaji/root.pem
    IdentitiesOnly yes
    IPQoS none
    Ciphers aes128-ctr
    KexAlgorithms diffie-hellman-group14-sha256
    MACs hmac-sha2-256

# Then use:
ssh emr-launcher
```

**Key Information:**
- **Purpose:** Launch EMR jobs, run local data processing tasks
- **Region:** `us-east-2` (Ohio)
- **Working Directory:** `/data/cosmos_workspace/`
- **Disk Space:** 
  - Root (`/`): 15 GB total, ~12 GB free
  - Data (`/data`): 40 GB total, ~30 GB free

### AWS Credentials for S3 Access

**On Lambda (already configured):**
```bash
# Check existing credentials
cat ~/.aws/credentials
cat ~/.aws/config

# Should have:
# [default]
# aws_access_key_id = <KEY>
# aws_secret_access_key = <SECRET>
# region = us-east-1
```

**On EMR Launcher:**
```bash
# IAM role-based access (no explicit credentials needed for EMR jobs)
# For AWS CLI:
aws s3 ls s3://darkhaloscope-training-dc/ --region us-east-1
```

### rclone Configuration

**On Lambda:**
```bash
# rclone is configured to sync with S3
# Config location: ~/.config/rclone/rclone.conf

# Usage:
rclone copy s3:darkhaloscope-training-dc/path/to/data /local/path/
rclone sync /local/path/ s3:darkhaloscope-training-dc/path/to/data/

# List S3 contents:
rclone ls s3:darkhaloscope-training-dc/
```

**Key S3 Paths:**
- Training data: `s3://darkhaloscope-training-dc/data/v4_sota_moffat/`
- Model checkpoints: `s3://darkhaloscope-training-dc/results/`
- Gen5 data (future): `s3://darkhaloscope-training-dc/data/v5_cosmos_source/`

---

## 2. Current Project State

### What Has Been Completed

#### Stage 0: Anchor Baseline Evaluation ✅
- **Location:** `dark_halo_scope/scripts/stage0_anchor_baseline.py`
- **Results:** `dark_halo_scope/results/stage0_anchor_baseline_report.md`
- **Key Finding:** **CATASTROPHIC SIM-TO-REAL GAP**
  - Gen2 model: 88.2% tpr@fpr1e-4 on synthetic data
  - **BUT: 0% recall on SLACS lenses, 95% contamination on hard negatives**
  - **Diagnosis:** Model learned spurious features (smooth Sersic sources, unrealistic negatives)

#### Infrastructure & Code Organization ✅
- **Experiment config system:** `dark_halo_scope/experiments/configs/experiment_schema.py`
- **External catalogs:** `dark_halo_scope/experiments/external_catalogs/`
- **Gen5 pipeline:** `dark_halo_scope/emr/gen5/spark_phase4_pipeline_gen5.py`
- **COSMOS loader:** `dark_halo_scope/src/sims/cosmos_source_loader_v2.py`
- **Unit tests:** All new code has comprehensive tests

#### Model Training History ✅
- **Gen1:** Gaussian PSF, paired controls (FAILED - 0.4% tpr@fpr1e-4)
- **Gen2:** Gaussian PSF, unpaired controls (88.2% tpr@fpr1e-4 on synthetic)
- **Gen3:** Moffat PSF, unpaired controls (84.5% tpr@fpr1e-4 on synthetic)
- **Gen4:** Gen3 + hard negative mining (training completed, evaluation pending)

**All Gen1-4 code is preserved and runnable in separate directories.**

### What Is Currently Running

#### COSMOS Catalog Download (IN PROGRESS)
```bash
# Check status:
ssh emr-launcher 'cd /data/cosmos_workspace && ls -lh COSMOS_25.2_training_sample.tar.gz && tail -3 download.log'

# Current state (as of 2026-02-03 02:43 UTC):
# - Downloaded: 1.3 GB / ~4.3 GB (29%)
# - ETA: ~50 minutes
# - PID: 465810
# - Location: /data/cosmos_workspace/COSMOS_25.2_training_sample.tar.gz
```

**CRITICAL: This download MUST complete before proceeding to the next step.**

---

## 3. Critical Mistakes Made & Lessons Learned

### Mistake 1: Insufficient Testing Before Full Runs ❌
**What happened:** Attempted to run full COSMOS bank build (20K sources) without testing on a small sample first.

**Result:** Multiple failures due to:
- Missing FITS files
- Incorrect catalog version (23.5 vs 25.2)
- Disk space issues
- Corrupted tarballs

**LESSON:** **ALWAYS test with 1-5 samples before launching any long-running job.**

### Mistake 2: Wrong Disk Partition ❌
**What happened:** Started COSMOS download in `~/.local/lib/python3.9/site-packages/galsim/share/` (on root partition with only 11 GB free).

**Result:** Would have run out of space during extraction (4 GB compressed → 8-10 GB uncompressed).

**LESSON:** **Always check `df -h` before downloading large files. Use `/data` (30 GB free) for all large operations on emr-launcher.**

### Mistake 3: Using EMR Cluster for Single-Threaded Job ❌
**What happened:** Initially tried to launch an EMR Spark cluster for COSMOS bank building.

**Result:** Wasted money (~$10) and time (1 hour) because the job is single-threaded Python, not Spark.

**LESSON:** **The COSMOS bank builder is NOT a Spark job. Run it directly on emr-launcher using `nohup`. Only use EMR for Phase 4c (which IS a Spark job).**

### Mistake 4: Not Using `nohup` Properly ❌
**What happened:** Ran commands without `nohup`, causing them to fail when SSH session was interrupted.

**LESSON:** **For ANY long-running job on emr-launcher, ALWAYS use `nohup` and redirect output to a log file:**
```bash
nohup python script.py > output.log 2>&1 &
echo $!  # Save PID
```

### Mistake 5: Incomplete Cleanup ❌
**What happened:** Killed wget process but didn't remove partial downloads (772 MB wasted).

**LESSON:** **Always cleanup after killing jobs:**
```bash
pkill -9 wget
rm -f /path/to/partial/downloads/*
```

### Mistake 6: Not Validating COSMOS Catalog Version ❌
**What happened:** Assumed `galsim_download_cosmos` downloaded the correct version (25.2), but it downloaded 23.5 by default.

**Result:** Build failures due to missing PSF files specific to 25.2.

**LESSON:** **Always explicitly specify version and verify after download:**
```bash
galsim_download_cosmos -s 25.2 -f  # Force download 25.2
ls ~/.local/lib/python3.9/site-packages/galsim/share/COSMOS_25.2_training_sample/*.fits
```

### Mistake 7: boto3 Not Available on EMR Executors ❌
**What happened:** The `_load_cosmos_bank_h5` function tried to import `boto3` to download the COSMOS bank from S3, but `boto3` wasn't installed on EMR executors.

**Result:** All cutouts failed silently with `cutout_ok=0`.

**LESSON:** **Add `boto3` to the EMR bootstrap script for ANY code that needs S3 access on executors.** The fix was adding `sudo python3 -m pip install boto3` to `emr_bootstrap_gen5.sh`.

### Mistake 8: Broken Self-Import in Python Module ❌
**What happened:** The `render_cosmos_lensed_source` function had `from spark_phase4_pipeline_gen5 import render_lensed_source` which fails on EMR executors.

**Result:** Import error causing all cutouts to fail.

**LESSON:** **Never self-import from the same module. If a function is defined in the same file, call it directly without import.**

### Mistake 9: Variable Scope Error (UnboundLocalError) ❌
**What happened:** `source_mode` was defined inside `if theta_e > 0:` block but referenced outside it. Control samples (theta_e=0) never defined the variable.

**Result:** `UnboundLocalError: local variable 'source_mode' referenced before assignment`

**LESSON:** **Always initialize variables BEFORE conditional blocks if they're used AFTER the block.**

**Fix:**
```python
# BEFORE (broken):
if theta_e > 0:
    source_mode = getattr(args, 'source_mode', 'sersic')
    # ... processing ...
# ... later, outside the if block:
if source_mode == "cosmos":  # ERROR! source_mode undefined if theta_e <= 0

# AFTER (fixed):
source_mode = getattr(args, 'source_mode', 'sersic')  # Define BEFORE if block
cosmos_idx = None
cosmos_hlr = None
if theta_e > 0:
    # ... processing ...
```

### Mistake 10: Inadequate Output Validation ❌
**What happened:** Smoke tests only checked for job completion and `_SUCCESS` marker, not actual data quality (`cutout_ok` values, stamp content).

**Result:** Bugs persisted through multiple smoke tests into production run.

**LESSON:** **Always validate output DATA, not just job status:**
```python
# Check cutout_ok distribution
table = pq.read_table(output_path, columns=["cutout_ok", "physics_warnings"])
cutout_ok_vals = [x.as_py() for x in table["cutout_ok"]]
print(f"cutout_ok distribution: {dict(zip(*np.unique(cutout_ok_vals, return_counts=True)))}")

# Check for errors in physics_warnings
for i in range(min(5, len(table))):
    pw = table["physics_warnings"][i].as_py()
    if pw:
        print(f"ERROR: {pw}")
```

---

## 4. Remaining Work: Detailed TODOs

### Current Step: COSMOS Bank Creation (IN PROGRESS)

**Status:** Download at 29% (ETA: 50 min as of 02:43 UTC)

**Once download completes, do this:**

#### Step 1: Verify Download Integrity
```bash
ssh emr-launcher
cd /data/cosmos_workspace

# Check file size (should be ~4.3 GB)
ls -lh COSMOS_25.2_training_sample.tar.gz

# Verify download completed without corruption
tail -20 download.log
# Should see: "saved [XXXXX/XXXXX]" indicating complete download
```

#### Step 2: Extract to GalSim Share Directory
```bash
# Extract to the location where GalSim expects it
tar -xzf COSMOS_25.2_training_sample.tar.gz -C ~/.local/lib/python3.9/site-packages/galsim/share/

# Verify extraction
ls ~/.local/lib/python3.9/site-packages/galsim/share/COSMOS_25.2_training_sample/*.fits | head -5
# Should see:
# real_galaxy_catalog_25.2.fits
# real_galaxy_PSF_images_25.2_n1.fits
# real_galaxy_PSF_images_25.2_n2.fits
# ...
```

#### Step 3: TEST with 1-5 Sources First (CRITICAL)
```bash
cd /data/cosmos_workspace

# Create test script
cat > test_cosmos_build.py << 'EOF'
import sys
sys.path.insert(0, '/home/ec2-user/dark_halo_scope/models/dhs_cosmos_galsim_code')
from dhs_cosmos.sims.cosmos_source_loader import build_cosmos_bank, BuildConfig

cfg = BuildConfig(
    cosmos_dir='/home/ec2-user/.local/lib/python3.9/site-packages/galsim/share/COSMOS_25.2_training_sample',
    out_h5='/data/cosmos_workspace/test_cosmos_5sources.h5',
    n_sources=5,  # ONLY 5 FOR TESTING
    stamp_size=96,
    src_pixscale_arcsec=0.03,
    seed=42,
    intrinsic_psf_fwhm_arcsec=0.10,
    denoise_sigma_pix=0.5,
    max_tries=100
)

print("Testing COSMOS bank build with 5 sources...")
build_cosmos_bank(cfg)
print("✅ Test successful! Output:", cfg.out_h5)
EOF

# Run test (should take <30 seconds)
python3 test_cosmos_build.py

# If successful, you should see:
# - No errors
# - File created: /data/cosmos_workspace/test_cosmos_5sources.h5
# - Size: ~few MB

# Verify output
ls -lh test_cosmos_5sources.h5
h5dump -H test_cosmos_5sources.h5  # Check structure
```

**IF TEST FAILS:** 
- Read the error message carefully
- Check if all required FITS files exist
- Verify GalSim can load the catalog
- DO NOT proceed to full 20K build until test passes

#### Step 4: Full 20K Build (ONLY if Step 3 passes)
```bash
cd /data/cosmos_workspace

# Pull latest code from git
cd ~/dark_halo_scope
git pull origin master

# Verify config file exists
cat ~/dark_halo_scope/configs/gen5/cosmos_bank_config.json

# Run full build with nohup
nohup python3 -c "
import sys
sys.path.insert(0, '/home/ec2-user/dark_halo_scope/models/dhs_cosmos_galsim_code')
from dhs_cosmos.sims.cosmos_source_loader import build_cosmos_bank, BuildConfig
import json

with open('/home/ec2-user/dark_halo_scope/configs/gen5/cosmos_bank_config.json') as f:
    cfg_dict = json.load(f)

cfg = BuildConfig(
    cosmos_dir=cfg_dict['cosmos_catalog_path'],
    out_h5=cfg_dict['output_path'],
    n_sources=cfg_dict['n_sources'],
    stamp_size=cfg_dict['stamp_size'],
    src_pixscale_arcsec=cfg_dict['src_pixscale_arcsec'],
    seed=cfg_dict['seed'],
    intrinsic_psf_fwhm_arcsec=cfg_dict['intrinsic_psf_fwhm_arcsec'],
    denoise_sigma_pix=cfg_dict['denoise_sigma_pix'],
    max_tries=cfg_dict['max_tries']
)

build_cosmos_bank(cfg)
" > /data/cosmos_workspace/cosmos_build_20k.log 2>&1 &

# Save PID
echo $! > /data/cosmos_workspace/cosmos_build.pid
echo "COSMOS build started with PID: $(cat /data/cosmos_workspace/cosmos_build.pid)"
echo "Monitor with: tail -f /data/cosmos_workspace/cosmos_build_20k.log"
```

**Expected Runtime:** 4-6 hours for 20K sources

**Monitor Progress:**
```bash
# Check log
ssh emr-launcher 'tail -20 /data/cosmos_workspace/cosmos_build_20k.log'

# Check if process is still running
ssh emr-launcher 'ps aux | grep python3 | grep cosmos'

# Check output file size (should grow over time)
ssh emr-launcher 'ls -lh /data/cosmos_workspace/*.h5'
```

#### Step 5: Validate COSMOS Bank Output
```bash
ssh emr-launcher
cd /data/cosmos_workspace

# Run validation
python3 << 'EOF'
import h5py
import numpy as np

h5_path = '/data/cosmos_workspace/cosmos_bank_20k_gen5.h5'  # Adjust path as needed

with h5py.File(h5_path, 'r') as f:
    print(f"✅ COSMOS bank successfully loaded: {h5_path}")
    print(f"Images shape: {f['images'].shape}")
    print(f"Expected: (20000, 96, 96)")
    print(f"Pixscale: {f.attrs['src_pixscale_arcsec']:.4f} arcsec/pix")
    print(f"Stamp size: {f.attrs['stamp_size']}")
    
    hlr = f['meta/hlr_arcsec'][:]
    clump = f['meta/clumpiness'][:]
    
    print(f"\nHLR statistics:")
    print(f"  Min: {np.min(hlr):.3f} arcsec")
    print(f"  Median: {np.median(hlr):.3f} arcsec")
    print(f"  Max: {np.max(hlr):.3f} arcsec")
    
    print(f"\nClumpiness statistics:")
    print(f"  Min: {np.min(clump):.3f}")
    print(f"  Median: {np.median(clump):.3f}")
    print(f"  Max: {np.max(clump):.3f}")
    
    # Check for any NaNs or invalid values
    n_valid = np.sum(np.isfinite(hlr))
    print(f"\nValid sources: {n_valid} / {len(hlr)}")
    
    if n_valid == len(hlr):
        print("✅ All sources are valid!")
    else:
        print(f"⚠️  {len(hlr) - n_valid} sources have invalid HLR values")
EOF
```

#### Step 6: Upload COSMOS Bank to S3
```bash
# On emr-launcher
aws s3 cp /data/cosmos_workspace/cosmos_bank_20k_gen5.h5 \
  s3://darkhaloscope-training-dc/data/cosmos_banks/ \
  --region us-east-1

# Verify upload
aws s3 ls s3://darkhaloscope-training-dc/data/cosmos_banks/ --region us-east-1
```

---

### Next Phase: Phase 4c Data Generation with COSMOS Sources

**Prerequisites:**
- ✅ COSMOS bank HDF5 file created and uploaded to S3
- ✅ Gen5 pipeline code (`spark_phase4_pipeline_gen5.py`) ready
- ✅ Config file for Phase 4c created

#### Step 1: Create Phase 4c Config File
```bash
# On local machine or emr-launcher
cat > ~/dark_halo_scope/configs/gen5/phase4c_config.json << 'EOF'
{
  "generation": "gen5",
  "component": "phase4c_data_generation",
  "seed_base": 1337,
  "variant": "v5_cosmos_source",
  "experiment_id": "train_stamp64_bandsgrz_cosmos",
  "source_mode": "cosmos",
  "cosmos_bank_h5": "s3://darkhaloscope-training-dc/data/cosmos_banks/cosmos_bank_20k_gen5.h5",
  "cosmos_salt": "gen5_v1",
  "psf_model": "moffat",
  "moffat_beta": 3.5,
  "output_s3": "s3://darkhaloscope-training-dc/data/v5_cosmos_source/train_stamp64_bandsgrz_cosmos",
  "created_utc": "2026-02-03T12:00:00Z"
}
EOF

# Upload to S3
aws s3 cp ~/dark_halo_scope/configs/gen5/phase4c_config.json \
  s3://darkhaloscope-training-dc/configs/gen5/ \
  --region us-east-1
```

#### Step 2: Launch EMR Cluster for Phase 4c

**IMPORTANT: This IS a Spark job, so we DO need an EMR cluster.**

```bash
# On emr-launcher
ssh emr-launcher

# Upload Gen5 pipeline code to S3
cd ~/dark_halo_scope
aws s3 cp emr/gen5/spark_phase4_pipeline_gen5.py \
  s3://darkhaloscope-training-dc/scripts/gen5/ \
  --region us-east-1

# Launch EMR cluster
aws emr create-cluster \
  --name "Gen5-Phase4c-COSMOS-$(date +%Y%m%d-%H%M)" \
  --region us-east-1 \
  --release-label emr-6.10.0 \
  --applications Name=Spark \
  --instance-type m5.2xlarge \
  --instance-count 10 \
  --use-default-roles \
  --log-uri s3://darkhaloscope-training-dc/emr-logs/ \
  --steps Type=Spark,Name="Phase4c-Inject-COSMOS",ActionOnFailure=CONTINUE,Args=[--deploy-mode,cluster,--conf,spark.executor.memory=12g,--conf,spark.executor.cores=4,--conf,spark.dynamicAllocation.enabled=true,--conf,spark.dynamicAllocation.minExecutors=5,--conf,spark.dynamicAllocation.maxExecutors=50,s3://darkhaloscope-training-dc/scripts/gen5/spark_phase4_pipeline_gen5.py,--stage,4c,--config,s3://darkhaloscope-training-dc/configs/gen5/phase4c_config.json] \
  --auto-terminate

# Save cluster ID
# Output will be: {"ClusterId": "j-XXXXXXXXXXXXX"}
```

**Monitor Progress:**
```bash
# Check cluster status
aws emr describe-cluster --cluster-id j-XXXXXXXXXXXXX --region us-east-1 | grep State

# Check step status
aws emr list-steps --cluster-id j-XXXXXXXXXXXXX --region us-east-1

# View logs (once step completes)
aws s3 ls s3://darkhaloscope-training-dc/emr-logs/j-XXXXXXXXXXXXX/steps/
```

**Expected Runtime:** 4-8 hours depending on cluster size

#### Step 3: Post-Injection Validation

**Run validation on generated data:**
```bash
# On emr-launcher
cd ~/dark_halo_scope

# Run validation script (from external LLM code)
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/ec2-user/dark_halo_scope/models/dhs_cosmos_galsim_code')
from dhs_cosmos.sims.validate_cosmos_injection import validate_parquet
import json

result = validate_parquet(
    parquet_glob="s3://darkhaloscope-training-dc/data/v5_cosmos_source/train_stamp64_bandsgrz_cosmos/stamps/*.parquet",
    max_rows=50000,
    sample_stamps=500
)

print(json.dumps(result, indent=2))

# Save validation report
with open('/data/cosmos_workspace/gen5_validation_report.json', 'w') as f:
    json.dump(result, f, indent=2)

# Upload to S3
import boto3
s3 = boto3.client('s3')
s3.upload_file(
    '/data/cosmos_workspace/gen5_validation_report.json',
    'darkhaloscope-training-dc',
    'data/v5_cosmos_source/validation_report.json'
)
print("✅ Validation report saved to S3")
EOF
```

**Expected Validation Metrics:**
- Clumpiness: Higher than Gen2/3/4 (due to real galaxy structure)
- HLR distribution: Broader, more realistic
- Flux distribution: Should match COSMOS catalog statistics

---

### Next Phase: Gen5 Training

#### Step 1: Transfer Data to Lambda
```bash
# On Lambda GPU instance
ssh ubuntu@192.222.56.237 -i ~/code/toy_glens_kinematics/dark_halo_scope/lambda.pem

cd /lambda/nfs/darkhaloscope-training-dc/

# Sync Gen5 data from S3
rclone sync s3:darkhaloscope-training-dc/data/v5_cosmos_source/ \
  data/v5_cosmos_source/ \
  --progress --transfers 16

# Verify data transfer
du -sh data/v5_cosmos_source/
ls -lh data/v5_cosmos_source/train_stamp64_bandsgrz_cosmos/stamps/*.parquet | wc -l
```

#### Step 2: Create Gen5 Training Script
```bash
# On Lambda
cd /lambda/nfs/darkhaloscope-training-dc/

# Create Gen5 training directory
mkdir -p models/gen5_cosmos
cd models/gen5_cosmos

# Copy template from Gen3 and modify
cp ../gen3_moffat/phase5_train_fullscale_gh200_v2.py phase5_train_gen5_cosmos.py

# Modify key parameters:
# - data_variant = "v5_cosmos_source"
# - experiment_id = "train_stamp64_bandsgrz_cosmos"
# - epochs = 50
# - model_save_prefix = "gen5_cosmos"
```

**Key Training Config:**
```python
# In phase5_train_gen5_cosmos.py
DATA_VARIANT = "v5_cosmos_source"
EXPERIMENT_ID = "train_stamp64_bandsgrz_cosmos"
MODEL_ARCH = "convnext_tiny"
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-4
FOCAL_GAMMA = 2.0
SEED = 42

# Metadata fusion
META_FEATURES = ["psfsize_r", "psfdepth_r"]
META_DIM = 2

# Output paths
OUTPUT_DIR = f"results/gen5/{EXPERIMENT_ID}"
S3_SYNC_PATH = f"s3://darkhaloscope-training-dc/results/gen5/{EXPERIMENT_ID}/"
```

#### Step 3: Run Training
```bash
# On Lambda
cd /lambda/nfs/darkhaloscope-training-dc/models/gen5_cosmos

# Create tmux session (to prevent SSH disconnect)
tmux new -s gen5_training

# Activate conda environment if needed
# conda activate darkhaloscope

# Run training
python phase5_train_gen5_cosmos.py \
  --data-variant v5_cosmos_source \
  --experiment-id train_stamp64_bandsgrz_cosmos \
  --epochs 50 \
  --batch-size 128 \
  --lr 1e-4 \
  --seed 42 \
  --output-dir /lambda/nfs/darkhaloscope-training-dc/results/gen5/train_stamp64_bandsgrz_cosmos \
  2>&1 | tee training.log

# Detach from tmux: Ctrl+B, then D
# Reattach: tmux attach -t gen5_training
```

**Monitor Training:**
```bash
# Check GPU utilization
nvidia-smi -l 5

# Check training log
tail -f /lambda/nfs/darkhaloscope-training-dc/models/gen5_cosmos/training.log

# Check for checkpoints
ls -lht /lambda/nfs/darkhaloscope-training-dc/results/gen5/train_stamp64_bandsgrz_cosmos/
```

**Expected Runtime:** 12-24 hours for 50 epochs

#### Step 4: Sync Results to S3
```bash
# On Lambda (after training completes)
cd /lambda/nfs/darkhaloscope-training-dc/

rclone sync results/gen5/ \
  s3:darkhaloscope-training-dc/results/gen5/ \
  --progress --exclude "*.tmp"
```

---

### Next Phase: Re-run Anchor Baseline on Gen5

**Purpose:** Validate that Gen5 (COSMOS sources) improves sim-to-real gap

#### Step 1: Run Stage 0 on Gen5 Model
```bash
# On Lambda
cd /lambda/nfs/darkhaloscope-training-dc/

# Find best Gen5 checkpoint
BEST_CKPT=$(ls -t results/gen5/train_stamp64_bandsgrz_cosmos/*.pth | head -1)
echo "Using checkpoint: $BEST_CKPT"

# Run anchor baseline
python scripts/stage0_anchor_baseline.py \
  --model-path "$BEST_CKPT" \
  --output-dir results/gen5_anchor_baseline \
  --batch-size 64

# Results will be in:
# - results/gen5_anchor_baseline/anchor_metrics.json
# - results/gen5_anchor_baseline/anchor_baseline_report.md
```

#### Step 2: Compare Gen2 vs Gen5
```bash
# Extract key metrics
python3 << 'EOF'
import json

# Gen2 results (baseline)
with open('results/anchor_baseline_report/anchor_metrics.json') as f:
    gen2 = json.load(f)

# Gen5 results (with COSMOS)
with open('results/gen5_anchor_baseline/anchor_metrics.json') as f:
    gen5 = json.load(f)

print("=== Sim-to-Real Gap Comparison ===")
print(f"\nRecall on SLACS Lenses:")
print(f"  Gen2: {gen2['slacs_recall']:.1%}")
print(f"  Gen5: {gen5['slacs_recall']:.1%}")
print(f"  Improvement: {gen5['slacs_recall'] - gen2['slacs_recall']:+.1%}")

print(f"\nContamination on Hard Negatives:")
print(f"  Gen2: {gen2['hard_neg_contamination']:.1%}")
print(f"  Gen5: {gen5['hard_neg_contamination']:.1%}")
print(f"  Improvement: {gen2['hard_neg_contamination'] - gen5['hard_neg_contamination']:+.1%}")

# Target: Gen5 should have >50% SLACS recall and <20% contamination
if gen5['slacs_recall'] > 0.5 and gen5['hard_neg_contamination'] < 0.2:
    print("\n✅ Gen5 PASSES anchor baseline gate!")
else:
    print("\n⚠️  Gen5 still needs improvement")
EOF
```

**Success Criteria (from external LLM):**
- **SLACS recall:** >50% (Gen2 was 0%)
- **Hard negative contamination:** <20% (Gen2 was 95%)
- **BELLS recall:** >40%

**If Gen5 still fails:** Need to investigate:
1. Are COSMOS sources being lensed correctly?
2. Is the PSF convolution applied properly?
3. Are hard negatives realistic enough?
4. Does the model need more capacity (larger architecture)?

---

### Next Phase: Ablation Studies

**Purpose:** Understand which components contribute most to performance

#### Study 1: COSMOS vs Sersic
- Train Gen5a: Sersic sources with identical training config
- Compare Gen5 (COSMOS) vs Gen5a (Sersic)
- Isolate impact of source morphology realism

#### Study 2: PSF Model
- Train Gen5b: Gaussian PSF with COSMOS sources
- Compare Gen5 (Moffat) vs Gen5b (Gaussian)
- Isolate impact of PSF model realism

#### Study 3: Hard Negatives
- Train Gen5c: No hard negatives, only random galaxies
- Compare Gen5 (with hard negs) vs Gen5c (without)
- Isolate impact of hard negative mining

**Expected Timeline:** 1 week per study

---

### Next Phase: SOTA Benchmarking

**Once anchor baseline is passing**, compare to published results:

#### Benchmark 1: HOLISMOKES XI
- **Metric:** TPR_0 and TPR_10 on hard real negatives
- **Target:** TPR_0 = 10-40%, TPR_10 = 40-60% (single model)
- **Data:** Use their public evaluation set if available

#### Benchmark 2: GraViT
- **Metric:** AUC-ROC and F1 on More et al. test set
- **Target:** F1 ≈ 0.74, AUC-ROC ≈ 0.95
- **Data:** Request access to their test splits

#### Benchmark 3: Euclid Pipeline
- **Metric:** Candidate list quality at various budgets
- **Target:** Recall at top-1000 per million galaxies
- **Data:** Use our DR10 evaluation set

**Timeline:** 2-3 weeks

---

### Final Phase: Publication Preparation

#### Paper Structure (following external LLM advice)
1. **Introduction:** Sim-to-real gap problem in lens finding
2. **Methods:**
   - Data generation pipeline (Phase 1-4c)
   - COSMOS source integration
   - Hard negative mining
   - Model architecture (ConvNeXt-Tiny)
   - Training procedure
3. **Results:**
   - Anchor baseline comparison (Gen2 vs Gen5)
   - Ablation studies
   - SOTA benchmarks
   - Candidate quality on DR10
4. **Discussion:**
   - Importance of source morphology realism
   - Remaining sim-to-real gaps
   - Recommendations for future work
5. **Conclusion:** COSMOS integration is critical for sim-to-real transfer

#### Key Figures
1. Anchor baseline comparison (SLACS recall, hard neg contamination)
2. Example cutouts: synthetic vs real lenses
3. Score distributions on different test sets
4. Ablation study results
5. ROC curves for Gen2/3/4/5
6. Candidate list quality vs budget

#### Code Release
- Clean up codebase
- Add comprehensive README for each generation
- Create Docker container for reproducibility
- Upload model checkpoints to Zenodo
- Release COSMOS bank (if allowed by GalSim license)

#### Submission Targets
- **Primary:** MNRAS (Monthly Notices of the Royal Astronomical Society)
- **Alternative:** A&A (Astronomy & Astrophysics)
- **Timeline:** 2-3 months for paper writing + review

---

## 5. Key File Locations

### Configuration Files
```
configs/gen5/
├── cosmos_bank_config.json        # COSMOS bank build params
└── phase4c_config.json            # Phase 4c data generation params
```

### Code
```
dark_halo_scope/
├── emr/gen5/
│   └── spark_phase4_pipeline_gen5.py    # Gen5 data generation (Spark)
├── src/sims/
│   ├── cosmos_loader_v2.py              # COSMOS loader (prototype)
│   └── cosmos_source_loader_v2.py       # COSMOS bank builder (Gen5)
├── models/dhs_cosmos_galsim_code/       # External LLM code
│   └── dhs_cosmos/sims/
│       ├── cosmos_source_loader.py      # Bank builder (original)
│       ├── cosmos_lens_injector.py      # Debug injector
│       └── validate_cosmos_injection.py # Data quality validator
├── scripts/
│   ├── stage0_anchor_baseline.py        # Anchor baseline eval
│   └── stage0_preflight.py              # Pre-flight checks
├── experiments/
│   ├── configs/
│   │   └── experiment_schema.py         # Config system
│   └── external_catalogs/
│       ├── catalog_sources.py           # Catalog registry
│       ├── download_catalogs.py         # Catalog downloader
│       ├── crossmatch_dr10.py           # DR10 crossmatch
│       └── compute_anchor_metrics.py    # Metrics computation
└── tests/
    ├── test_experiment_config.py
    ├── test_anchor_baseline.py
    ├── test_cosmos_and_hardneg.py
    └── test_stage0_anchor_baseline.py
```

### Results
```
results/
├── stage0_anchor_baseline_report.md     # Gen2 anchor results (FAILED)
├── stage0_anchor_metrics.json
├── model_comparison_and_evolution.md    # Gen1-4 comparison
└── gen5/                                 # Gen5 results (future)
```

### S3 Paths
```
s3://darkhaloscope-training-dc/
├── data/
│   ├── v4_sota_moffat/              # Gen3/4 data
│   ├── v5_cosmos_source/            # Gen5 data (future)
│   └── cosmos_banks/                # COSMOS bank HDF5
├── results/
│   ├── gen2/
│   ├── gen3/
│   ├── gen4/
│   └── gen5/                        # Gen5 checkpoints (future)
├── configs/gen5/                    # Gen5 configs
└── scripts/gen5/                    # Gen5 pipeline code
```

---

## 6. Important Notes & Context

### Why Gen5 (COSMOS) is Critical

**The Stage 0 anchor baseline revealed that Gen2/3/4 models completely fail on real data despite 84-88% performance on synthetic validation.** This is because:

1. **Synthetic sources are too smooth:** Sersic n=1 profiles create perfect, symmetric arcs. Real galaxies are clumpy, asymmetric, with substructure.

2. **Synthetic negatives are too clean:** Random LRGs don't capture the diversity of real contaminants (rings, spirals, mergers, artifacts).

3. **Model learns spurious shortcuts:** Instead of learning lensing physics (arc geometry, Einstein radius, symmetry), it learns "smooth extra flux" = lens.

**Solution:** Use **real galaxy images from COSMOS** as lensing sources. This forces the model to learn actual lensing signatures rather than synthetic shortcuts.

### Expected Improvement from Gen5

Based on external LLM analysis and literature review:
- **Gen2-4 (Sersic sources):** 0% recall on real lenses, 95% contamination
- **Gen5 (COSMOS sources):** Should achieve 50-70% recall, 10-20% contamination
- **This is still not SOTA**, but demonstrates the critical importance of source realism

### Remaining Sim-to-Real Gaps (after Gen5)

Even with COSMOS sources, there are still gaps:
1. **Hard negative diversity:** Need more real contaminants (rings, mergers, artifacts)
2. **PSF modeling:** Moffat is better than Gaussian, but still simplified
3. **Noise realism:** Real survey noise has correlations, systematics
4. **Color gradients:** COSMOS provides single-band morphology; colors still synthetic
5. **CCD artifacts:** No cosmic rays, bad pixels, satellite trails

**Future work:** Iteratively address these gaps in Gen6, Gen7, etc.

### Cost & Timeline Expectations

- **COSMOS bank build:** 4-6 hours, $0 (runs on emr-launcher)
- **Phase 4c EMR:** 4-8 hours, ~$50-100 (cluster of 10 m5.2xlarge)
- **Gen5 training:** 12-24 hours, included in Lambda GPU subscription
- **Anchor baseline re-eval:** 1-2 hours, $0 (runs on Lambda)
- **Total Gen5 cycle:** ~2-3 days wall time, ~$50-100 AWS cost

### What Makes This Project Different

Most lens-finding papers focus on:
1. Architecture comparisons (ResNet vs Transformer)
2. Synthetic performance (AUROC, F1)
3. Single-dataset evaluation

**This project focuses on:**
1. **Sim-to-real gap** as the primary problem
2. **Real-data validation** (anchor baselines)
3. **Data realism** over model architecture
4. **Ablation studies** to isolate impact of each component
5. **Reproducibility** (configs, seeds, version control)

This is the **correct scientific approach** according to external LLM review and 2024-2026 literature (HOLISMOKES XI, GraViT, Euclid).

---

## 7. Debugging Commands

### Check EMR Cluster Status
```bash
aws emr list-clusters --region us-east-1 --active

aws emr describe-cluster --cluster-id j-XXXXXXXXXXXXX --region us-east-1

aws emr list-steps --cluster-id j-XXXXXXXXXXXXX --region us-east-1
```

### Check S3 Data
```bash
# List Gen5 data
aws s3 ls s3://darkhaloscope-training-dc/data/v5_cosmos_source/ --recursive --region us-east-1

# Check COSMOS bank
aws s3 ls s3://darkhaloscope-training-dc/data/cosmos_banks/ --region us-east-1

# Download validation report
aws s3 cp s3://darkhaloscope-training-dc/data/v5_cosmos_source/validation_report.json . --region us-east-1
```

### Check Running Jobs
```bash
# On emr-launcher
ssh emr-launcher 'ps aux | grep python'
ssh emr-launcher 'ps aux | grep wget'

# On Lambda
ssh ubuntu@192.222.56.237 'nvidia-smi'
ssh ubuntu@192.222.56.237 'ps aux | grep python'
```

### Check Disk Space
```bash
ssh emr-launcher 'df -h'
ssh ubuntu@192.222.56.237 'df -h'
```

### Check Logs
```bash
# COSMOS build log
ssh emr-launcher 'tail -50 /data/cosmos_workspace/cosmos_build_20k.log'

# EMR logs
aws s3 ls s3://darkhaloscope-training-dc/emr-logs/ --recursive --region us-east-1

# Training log
ssh ubuntu@192.222.56.237 'tail -50 /lambda/nfs/darkhaloscope-training-dc/models/gen5_cosmos/training.log'
```

---

## 8. Emergency Contacts & Resources

### If Something Goes Wrong

1. **EMR cluster not terminating:** Manually terminate via AWS console
   - URL: https://us-east-1.console.aws.amazon.com/emr/home?region=us-east-1#/clusters
   - Select cluster → Actions → Terminate

2. **Out of disk space:** Clean up old files
   ```bash
   ssh emr-launcher 'du -sh /data/* | sort -h'
   ssh emr-launcher 'rm -rf /data/cosmos_workspace/old_files'
   ```

3. **Training crashed:** Check GPU memory
   ```bash
   ssh ubuntu@192.222.56.237 'nvidia-smi'
   # If OOM, reduce batch size in training script
   ```

4. **S3 sync failed:** Check AWS credentials
   ```bash
   aws sts get-caller-identity --region us-east-1
   # Should return account ID
   ```

### External Resources

- **GalSim COSMOS Catalog:** https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy-Data
- **SLACS Catalog:** https://www.slac.stanford.edu/~surhud/data/slacs/
- **BELLS Catalog:** http://www.physics.ucsb.edu/~bullock/bells.html
- **HOLISMOKES Paper:** https://www.aanda.org/articles/aa/abs/2024/12/aa47072-23/aa47072-23.html
- **GraViT Paper:** https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/staf1747/8280375

---

## 9. Summary Checklist

Before proceeding, verify:

- [ ] COSMOS download completed successfully (4.3 GB)
- [ ] COSMOS extracted to GalSim share directory
- [ ] Test build with 5 sources PASSES
- [ ] Full 20K build completes without errors
- [ ] COSMOS bank uploaded to S3
- [ ] Phase 4c config file created and uploaded
- [ ] Gen5 pipeline uploaded to S3
- [ ] EMR cluster launched and completes successfully
- [ ] Post-injection validation shows improved realism
- [ ] Data transferred to Lambda
- [ ] Gen5 training script created
- [ ] Training completes 50 epochs
- [ ] Anchor baseline re-run on Gen5 model
- [ ] Gen5 shows improvement over Gen2 on anchor metrics

**Current Status (as of 2026-02-03 02:43 UTC):**
- [x] Git commit `31bd599` pushed
- [x] COSMOS download in progress (29% complete)
- [ ] All subsequent steps pending

---

## 10. Final Notes for Next Session

**DO NOT:**
- Run full 20K build without testing with 5 sources first
- Launch EMR clusters for non-Spark jobs
- Forget to use `nohup` for long-running jobs
- Skip validation steps

**DO:**
- Always check disk space before downloads
- Monitor long-running jobs regularly
- Save PIDs for all background jobs
- Document any errors and fixes
- Update this handoff doc with new learnings

**Priority Order:**
1. Wait for COSMOS download to complete (~50 min ETA)
2. Test with 5 sources
3. Full 20K build (only if test passes)
4. Phase 4c EMR job
5. Gen5 training
6. Anchor baseline re-evaluation

**Success Metric:** Gen5 anchor baseline shows >50% SLACS recall and <20% hard negative contamination (compared to Gen2's 0% and 95%).

---

**Good luck! This is important work that will advance the field's understanding of sim-to-real gaps in lens finding.**


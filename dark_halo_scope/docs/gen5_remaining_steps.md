# Gen5 Remaining Steps

## Current Status

✅ **Completed:**
1. Applied critical fixes to COSMOS source loader (float32, HLR filter, validation)
2. Copied and updated Gen5 pipeline with:
   - Automatic config saving to S3
   - COSMOS source integration code
   - New schema columns (source_mode, cosmos_index, cosmos_hlr_arcsec)
   - Deterministic template selection
3. Created comprehensive documentation

## Remaining Work

### Phase 1: COSMOS Bank Creation (1-2 days)

**Location:** emr-launcher

#### Step 1.1: Setup on emr-launcher
```bash
# SSH to emr-launcher
ssh emr-launcher

# Clone/update repo
cd /home/ec2-user/
git clone <repo_url> toy_glens_kinematics
cd toy_glens_kinematics/dark_halo_scope

# Install dependencies
pip install galsim h5py numpy astropy pydantic
```

#### Step 1.2: Download COSMOS catalog
```bash
# Create workspace
mkdir -p /mnt/cosmos_workspace
cd /mnt/cosmos_workspace

# Download GalSim COSMOS catalog (~2.3 GB)
# Option 1: Direct download (if available)
wget https://zenodo.org/record/3242143/files/COSMOS_23.5_training_sample.tar.gz
tar -xzf COSMOS_23.5_training_sample.tar.gz

# Option 2: Use GalSim's built-in downloader
python3 -c "import galsim; galsim.RealGalaxyCatalog('COSMOS_23.5_training_sample')"
```

#### Step 1.3: Create config JSON
```bash
cat > /mnt/cosmos_workspace/cosmos_bank_config_20k_v1.json << 'EOF'
{
  "cosmos_dir": "/mnt/cosmos_workspace/COSMOS_23.5_training_sample",
  "out_h5": "/mnt/cosmos_workspace/cosmos_bank_20k_parametric_v1.h5",
  "n_sources": 20000,
  "stamp_size": 96,
  "src_pixscale_arcsec": 0.03,
  "seed": 42,
  "intrinsic_psf_fwhm_arcsec": 0.0,
  "denoise_sigma_pix": 0.5,
  "hlr_min_arcsec": 0.1,
  "hlr_max_arcsec": 1.5,
  "dtype": "float32",
  "max_tries": 100000
}
EOF
```

#### Step 1.4: Build COSMOS bank
```bash
cd /home/ec2-user/toy_glens_kinematics/dark_halo_scope

# Run builder (takes ~30-60 minutes for 20K sources)
python3 models/dhs_cosmos_galsim_code/dhs_cosmos/sims/cosmos_source_loader.py \
  --config /mnt/cosmos_workspace/cosmos_bank_config_20k_v1.json \
  --mode build

# Expected output:
# cosmos_bank_20k_parametric_v1.h5 (~100-200 MB)
```

#### Step 1.5: Validate COSMOS bank
```bash
# Quick validation
python3 models/dhs_cosmos_galsim_code/dhs_cosmos/sims/cosmos_source_loader.py \
  --config /mnt/cosmos_workspace/cosmos_bank_config_20k_v1.json \
  --mode validate

# Expected output: JSON with clumpiness, HLR quantiles
```

#### Step 1.6: Upload to S3
```bash
# Upload bank
aws s3 cp /mnt/cosmos_workspace/cosmos_bank_20k_parametric_v1.h5 \
  s3://darkhaloscope/cosmos/

# Upload config for audit trail
aws s3 cp /mnt/cosmos_workspace/cosmos_bank_config_20k_v1.json \
  s3://darkhaloscope/cosmos/

# Verify
aws s3 ls s3://darkhaloscope/cosmos/
```

**Checkpoint:** You should now have:
- ✅ `s3://darkhaloscope/cosmos/cosmos_bank_20k_parametric_v1.h5` (~150 MB)
- ✅ `s3://darkhaloscope/cosmos/cosmos_bank_config_20k_v1.json` (audit trail)

---

### Phase 2: Single Injection Test (1-2 hours)

**Location:** emr-launcher

#### Step 2.1: Test single COSMOS injection
```bash
cd /home/ec2-user/toy_glens_kinematics/dark_halo_scope

# Create test script
python3 << 'EOF'
import numpy as np
import matplotlib.pyplot as plt
from models.dhs_cosmos_galsim_code.dhs_cosmos.sims.cosmos_lens_injector import (
    render_lensed_arc_lenstronomy, InjectionParams
)

# Load COSMOS bank
import h5py
with h5py.File('/mnt/cosmos_workspace/cosmos_bank_20k_parametric_v1.h5', 'r') as f:
    template = f['images'][0].astype(np.float32)
    src_pixscale = float(f.attrs['src_pixscale_arcsec'])

# Test injection
params = InjectionParams(
    theta_e_arcsec=1.0,
    e1=0.1, e2=0.05,
    gamma1=0.02, gamma2=0.01,
    src_x_arcsec=0.0, src_y_arcsec=0.0,
    src_mag_r=21.5,
    z_s=1.5
)

arc = render_lensed_arc_lenstronomy(
    template_unitflux=template,
    template_scale_arcsec=src_pixscale,
    out_shape=(64, 64),
    out_pixscale_arcsec=0.262,
    psf_fwhm_arcsec=1.5,
    psf_type='moffat',
    params=params,
    band='r'
)

# Save diagnostic plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(template, origin='lower', cmap='gray')
plt.title('COSMOS Template')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(arc, origin='lower', cmap='gray')
plt.title('Lensed Arc')
plt.colorbar()

plt.savefig('/mnt/cosmos_workspace/test_injection.png', dpi=150)
print(f"✅ Test injection successful: arc sum = {arc.sum():.2f} nMgy")
EOF
```

**Checkpoint:** You should see:
- ✅ Test injection plot at `/mnt/cosmos_workspace/test_injection.png`
- ✅ Sensible arc flux values (not NaN or zero)

---

### Phase 3: Gen5 Data Variant Setup (30 minutes)

**Location:** emr-launcher

#### Step 3.1: Create Gen5 data variant documentation
```bash
cd /home/ec2-user/toy_glens_kinematics/dark_halo_scope

cat > experiments/data_variants/v5_cosmos_source.md << 'EOF'
# Data Variant: v5_cosmos_source (Gen5)

## Overview
Generation 5 dataset using **GalSim COSMOS real galaxy morphologies** as lensed sources,
replacing parametric Sersic n=1 profiles used in Gen1-4.

## Key Parameters

### Source Morphology
- **Type**: COSMOS RealGalaxy (from 23.5 mag HST COSMOS catalog)
- **Bank**: 20,000 templates pre-rendered at 0.03"/pix
- **Selection**: Deterministic via SHA256(task_id + salt)
- **HLR Range**: 0.1-1.5 arcsec (filtered during bank creation)
- **Clumpiness**: Real morphology (spiral arms, clumps, irregular structure)

### PSF Model
- **Type**: Moffat β=3.5 (same as Gen3/Gen4)
- **Per-band**: Yes (g, r, z have independent PSF sizes)
- **Source**: Center-evaluated from DR10 psfsize maps

### Lens Model
- **Type**: SIE + external shear
- **θ_E Range**: 0.5-2.5 arcsec (same as Gen3/Gen4)
- **Resolvability**: θ_E / PSF_FWHM ≥ 0.5

### Controls
- **Type**: Unpaired (different galaxies for positives/negatives)
- **Fraction**: 50% train, 10% grid

### Colors
- **Source Colors**: Sampled from color distribution (g-r, r-z)
- **SED Model**: Simple red galaxy template for multi-band flux

## Differences from Gen4

| Feature | Gen4 (v4_sota_moffat) | Gen5 (v5_cosmos_source) |
|---------|----------------------|------------------------|
| Source Morphology | Sersic n=1 (parametric) | COSMOS RealGalaxy |
| Source Clumpiness | Smooth, exponential | Real (spiral arms, clumps) |
| Source HLR | Sampled per task | Fixed per COSMOS template |
| Source Colors | Uniform per galaxy | Realistic gradients (TBD) |
| PSF Model | Moffat β=3.5 | Moffat β=3.5 (same) |
| Control Type | Unpaired | Unpaired (same) |
| θ_E Range | 0.5-2.5" | 0.5-2.5" (same) |

## Expected Sim-to-Real Improvement

### Hypothesis
Realistic source morphology should reduce false positives on:
- Ring galaxies (smooth vs clumpy)
- Spiral galaxies (arm structure vs arcs)
- Mergers (irregular morphology)

### Metrics to Track
1. **Recall on known lenses** (SLACS/BELLS): Target >50% @ budget=100
2. **Contamination on hard negatives**: Target <20% @ tpr@fpr1e-4
3. **Clumpiness distribution**: Match real lens arc clumpiness
4. **HLR distribution**: Match observed Einstein radii

## S3 Locations

### Phase 4c Output
```
s3://darkhaloscope/phase4/phase4c/v5_cosmos_source/
├── stamps/train_stamp64_bandsgrz_gridgrid_sota/
├── metrics/train_stamp64_bandsgrz_gridgrid_sota/
└── run_config_train_stamp64_bandsgrz_gridgrid_sota.json
```

### COSMOS Bank
```
s3://darkhaloscope/cosmos/
├── cosmos_bank_20k_parametric_v1.h5
└── cosmos_bank_config_20k_v1.json
```

## Reproduction Commands

### Phase 4a (Manifests)
```bash
spark-submit emr/gen5/spark_phase4_pipeline_gen5.py \
  --stage 4a \
  --output-s3 s3://darkhaloscope/phase4 \
  --variant v5_cosmos_source \
  --parent-s3 s3://darkhaloscope/phase3/parent_catalog/ \
  --bricks-with-region-s3 s3://darkhaloscope/phase3/bricks_with_region/ \
  --region-selections-s3 s3://darkhaloscope/phase3/region_selections/
```

### Phase 4b (Coadd Cache)
```bash
# Reuse Gen3/Gen4 cache (same bricks)
# If needed, re-run with --include-psfsize 1
```

### Phase 4c (COSMOS Injection)
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

## Timeline
- **Created**: 2026-02-02
- **Status**: In progress
- **Data Generation**: Pending Phase 4c run
- **Training**: Pending
EOF
```

---

### Phase 4: EMR Cluster Setup (30 minutes)

**Location:** emr-launcher

#### Step 4.1: Update EMR bootstrap script
```bash
# Add COSMOS bank download to bootstrap
cat > /mnt/emr_bootstrap_gen5.sh << 'EOF'
#!/bin/bash
set -e

# Standard dependencies
pip3 install astropy>=5.0 boto3

# Gen5: Download COSMOS bank to all executors
mkdir -p /mnt/cosmos_cache
aws s3 cp s3://darkhaloscope/cosmos/cosmos_bank_20k_parametric_v1.h5 \
  /mnt/cosmos_cache/cosmos_bank_20k_parametric_v1.h5

echo "✅ Gen5 bootstrap complete"
EOF

# Upload to S3
aws s3 cp /mnt/emr_bootstrap_gen5.sh s3://darkhaloscope/emr_bootstrap/
```

#### Step 4.2: Launch EMR cluster
```bash
# Create cluster launch config
cat > /mnt/emr_cluster_gen5.json << 'EOF'
{
  "Name": "gen5-cosmos-phase4c",
  "ReleaseLabel": "emr-6.15.0",
  "Applications": [{"Name": "Spark"}],
  "Instances": {
    "InstanceGroups": [
      {
        "Name": "Master",
        "InstanceRole": "MASTER",
        "InstanceType": "r5.4xlarge",
        "InstanceCount": 1
      },
      {
        "Name": "Workers",
        "InstanceRole": "CORE",
        "InstanceType": "r5.4xlarge",
        "InstanceCount": 20
      }
    ],
    "Ec2SubnetId": "subnet-xxxxx",
    "Ec2KeyName": "your-key"
  },
  "BootstrapActions": [
    {
      "Name": "Gen5 Setup",
      "ScriptBootstrapAction": {
        "Path": "s3://darkhaloscope/emr_bootstrap/emr_bootstrap_gen5.sh"
      }
    }
  ],
  "Configurations": [
    {
      "Classification": "spark-defaults",
      "Properties": {
        "spark.executor.memory": "24g",
        "spark.executor.cores": "8",
        "spark.driver.memory": "12g",
        "spark.sql.shuffle.partitions": "800"
      }
    }
  ],
  "JobFlowRole": "EMR_EC2_DefaultRole",
  "ServiceRole": "EMR_DefaultRole"
}
EOF

# Launch cluster
aws emr create-cluster --cli-input-json file:///mnt/emr_cluster_gen5.json
```

---

### Phase 5: Run Gen5 Phase 4c (4-8 hours)

**Location:** EMR cluster (launched from emr-launcher)

#### Step 5.1: Upload pipeline code
```bash
# On emr-launcher
cd /home/ec2-user/toy_glens_kinematics/dark_halo_scope

# Create deployment package
zip -r gen5_pipeline.zip emr/gen5/spark_phase4_pipeline_gen5.py

# Upload to S3
aws s3 cp gen5_pipeline.zip s3://darkhaloscope/code/
```

#### Step 5.2: Submit Spark job
```bash
# Get cluster ID
CLUSTER_ID=$(aws emr list-clusters --active --query 'Clusters[0].Id' --output text)

# Submit Phase 4c job
aws emr add-steps --cluster-id $CLUSTER_ID --steps Type=Spark,Name="Gen5-Phase4c",\
ActionOnFailure=CONTINUE,\
Args=[
  s3://darkhaloscope/code/gen5_pipeline.zip/emr/gen5/spark_phase4_pipeline_gen5.py,
  --stage,4c,
  --output-s3,s3://darkhaloscope/phase4,
  --variant,v5_cosmos_source,
  --experiment-id,train_stamp64_bandsgrz_gridgrid_sota,
  --source-mode,cosmos,
  --cosmos-bank-h5,/mnt/cosmos_cache/cosmos_bank_20k_parametric_v1.h5,
  --cosmos-salt,gen5_v1,
  --seed-base,42,
  --psf-model,moffat,
  --moffat-beta,3.5,
  --split-seed,99887766,
  --tiers,train,
  --n-total-train-per-split,200000,
  --sweep-partitions,800
]
```

#### Step 5.3: Monitor progress
```bash
# Check job status
aws emr describe-step --cluster-id $CLUSTER_ID --step-id <step-id>

# Monitor logs
aws s3 ls s3://aws-logs-<account>-<region>/elasticmapreduce/$CLUSTER_ID/
```

**Checkpoint:** You should see:
- ✅ Parquet stamps at `s3://darkhaloscope/phase4/phase4c/v5_cosmos_source/stamps/`
- ✅ Config JSON at `s3://darkhaloscope/phase4/phase4c/v5_cosmos_source/run_config_*.json`
- ✅ Metrics table at `s3://darkhaloscope/phase4/phase4c/v5_cosmos_source/metrics/`

---

### Phase 6: Validate Generated Data (2-3 hours)

**Location:** emr-launcher or Lambda

#### Step 6.1: Run validation script
```bash
# Download validation tool
cd /home/ec2-user/toy_glens_kinematics/dark_halo_scope

python3 models/dhs_cosmos_galsim_code/dhs_cosmos/sims/validate_cosmos_injection.py \
  --parquet-glob "s3://darkhaloscope/phase4/phase4c/v5_cosmos_source/stamps/train_*/region_split=train/*.parquet" \
  --max-rows 10000 \
  --sample-stamps 500 \
  --output-json /mnt/gen5_validation_report.json

# Upload report
aws s3 cp /mnt/gen5_validation_report.json s3://darkhaloscope/phase4/phase4c/v5_cosmos_source/
```

#### Step 6.2: Review metrics
Expected outputs:
- Clumpiness quantiles: [p10, p50, p90]
- HLR distribution: Should match COSMOS catalog
- Band flux ratios: Check g/r/z ratios are sensible
- COSMOS index coverage: Should use all 20K templates

---

### Phase 7: Train Gen5 Model (2-3 days)

**Location:** Lambda Labs GH200

#### Step 7.1: Sync data to Lambda
```bash
# On Lambda
cd /home/ubuntu/toy_glens_kinematics/dark_halo_scope

# Sync stamps (may take hours)
aws s3 sync s3://darkhaloscope/phase4/phase4c/v5_cosmos_source/stamps/train_*/ \
  /mnt/data/gen5_cosmos/
```

#### Step 7.2: Launch training
```bash
# Use existing training script (should work with Gen5 data)
python3 models/train_convnext_phase5.py \
  --train-parquet /mnt/data/gen5_cosmos/region_split=train \
  --val-parquet /mnt/data/gen5_cosmos/region_split=val \
  --output-dir /mnt/checkpoints/gen5_cosmos_v1 \
  --epochs 50 \
  --batch-size 256 \
  --lr 1e-4 \
  --model convnext_tiny \
  --loss focal \
  --meta-features psfsize_r,psfdepth_r
```

---

### Phase 8: Re-Evaluate Anchor Baseline (4-6 hours)

**Location:** Lambda Labs

#### Step 8.1: Run anchor baseline with Gen5 model
```bash
cd /home/ubuntu/toy_glens_kinematics/dark_halo_scope

python3 scripts/stage0_anchor_baseline.py \
  --model-checkpoint /mnt/checkpoints/gen5_cosmos_v1/best_model.pth \
  --output-dir /mnt/anchor_results/gen5_cosmos_v1 \
  --brick-metadata /mnt/data/survey-bricks-dr10.fits
```

#### Step 8.2: Compare with Gen2/Gen4 results
Expected improvements:
- Recall on SLACS/BELLS: >50% (vs Gen2: ~10%)
- Contamination on rings/mergers: <20% (vs Gen2: ~60%)
- Score separation: More pronounced bimodal distribution

---

## Summary Timeline

| Phase | Duration | Location | Dependencies |
|-------|----------|----------|--------------|
| 1. COSMOS Bank | 1-2 days | emr-launcher | GalSim, COSMOS catalog |
| 2. Single Test | 1-2 hours | emr-launcher | Phase 1 |
| 3. Variant Setup | 30 min | emr-launcher | - |
| 4. EMR Setup | 30 min | emr-launcher | - |
| 5. Phase 4c Run | 4-8 hours | EMR | Phases 1-4 |
| 6. Validation | 2-3 hours | emr-launcher | Phase 5 |
| 7. Training | 2-3 days | Lambda GH200 | Phase 5 |
| 8. Anchor Re-eval | 4-6 hours | Lambda | Phase 7 |

**Total**: ~1 week (parallelizable)

## Critical Path

```
COSMOS Bank (1-2 days)
    ↓
Single Test (1-2 hours)
    ↓
EMR Setup (30 min) + Variant Setup (30 min)
    ↓
Phase 4c Run (4-8 hours)
    ↓
Validation (2-3 hours)
    ↓
Training (2-3 days)
    ↓
Anchor Re-eval (4-6 hours)
```

## Next Immediate Action

**Start with Phase 1.1:** SSH to emr-launcher and begin COSMOS bank creation.


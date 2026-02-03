# Gen5 COSMOS Bank Builder - EMR Job

## Overview

This directory contains scripts to build a COSMOS source bank (20K galaxy templates) using EMR.
The job runs on a single large instance (32 vCPUs) with auto-termination.

## Files

### Launch Scripts
- **`LAUNCH_INSTRUCTIONS.sh`** - Instructions for launching from emr-launcher (START HERE!)
- **`launch_cosmos_bank_builder.sh`** - Main launch script (run on emr-launcher)
- **`cosmos_bank_builder_bootstrap.sh`** - EMR bootstrap script (installs dependencies)
- **`build_cosmos_bank_emr.py`** - Main Python job that builds the bank

### Pipeline Code
- **`spark_phase4_pipeline_gen5.py`** - Gen5 pipeline with COSMOS support

## Quick Start (from emr-launcher)

```bash
# 1. SSH to emr-launcher
ssh emr-launcher

# 2. Setup repo
cd /home/ec2-user/toy_glens_kinematics/dark_halo_scope
git pull

# 3. Launch EMR job
bash emr/gen5/launch_cosmos_bank_builder.sh

# 4. Monitor (cluster auto-terminates when done)
CLUSTER_ID=$(cat /tmp/cosmos_bank_cluster_id.txt)
aws emr describe-cluster --cluster-id $CLUSTER_ID --query 'Cluster.Status.State'
```

## Cluster Specifications

- **Instance**: 1x r6i.8xlarge (32 vCPUs, 256 GB RAM)
- **Storage**: 500 GB GP3 EBS (3000 IOPS)
- **Runtime**: 30-60 minutes
- **Cost**: ~$3-5 USD
- **Auto-terminate**: Yes

## Output

After successful completion:

```
s3://darkhaloscope/cosmos/
├── cosmos_bank_20k_parametric_v1.h5       (~150 MB)
└── cosmos_bank_config_20k_v1.json         (config for audit)
```

## Bank Specifications

- **Templates**: 20,000 COSMOS galaxies
- **Source**: GalSim COSMOS 23.5 mag training sample
- **Stamp Size**: 96x96 pixels
- **Pixel Scale**: 0.03 arcsec/pix (HST-like)
- **HLR Range**: 0.1-1.5 arcsec (filtered)
- **Dtype**: float32 (compressed with gzip)
- **Metadata**: HLR, clumpiness, COSMOS index

## What the Job Does

1. **Download COSMOS catalog** (~2.3 GB from GalSim)
2. **Render templates** (20K galaxies at 0.03"/pix)
3. **Apply filters** (HLR range 0.1-1.5 arcsec)
4. **Compute metrics** (half-light radius, clumpiness)
5. **Save to HDF5** (with compression)
6. **Upload to S3** (bank + config)

## Monitoring

### Check cluster status
```bash
CLUSTER_ID=$(cat /tmp/cosmos_bank_cluster_id.txt)
aws emr describe-cluster --cluster-id $CLUSTER_ID --query 'Cluster.Status.State'
```

### Check step progress
```bash
aws emr list-steps --cluster-id $CLUSTER_ID \
  --query 'Steps[*].[Name,Status.State]' --output table
```

### View live logs
```bash
aws emr ssh --cluster-id $CLUSTER_ID \
  --command "tail -f /mnt/var/log/hadoop/steps/*/stdout"
```

### Browse S3 logs
```bash
aws s3 ls s3://darkhaloscope/emr_logs/$CLUSTER_ID/ --recursive
```

## Troubleshooting

### Cluster fails during launch
- Check IAM roles (EMR_EC2_DefaultRole, EMR_DefaultRole)
- Check subnet/security group settings
- Verify S3 bucket exists and is accessible

### Job fails during COSMOS download
- GalSim auto-downloads from Zenodo (~2.3 GB)
- If download fails, check network connectivity
- Fallback: manually download and upload to S3

### Job fails during rendering
- Check memory usage (should have plenty with 256 GB)
- GalSim rendering is CPU-intensive (uses all 32 cores)
- Check logs for specific GalSim errors

### Output validation fails
- Bank may still be OK even if validation warns
- Download and inspect manually:
  ```bash
  aws s3 cp s3://darkhaloscope/cosmos/cosmos_bank_20k_parametric_v1.h5 .
  python3 -c "import h5py; print(h5py.File('cosmos_bank_20k_parametric_v1.h5', 'r')['images'].shape)"
  ```

## Next Steps After Completion

1. **Validate bank** (run validation script on emr-launcher)
2. **Test single injection** (verify lensed arc rendering)
3. **Update Gen5 pipeline** (point to S3 bank)
4. **Run Gen5 Phase 4c** (full dataset generation on EMR)

See: `docs/gen5_remaining_steps.md` for complete workflow.

## Cost Optimization

The r6i.8xlarge instance costs ~$2.02/hour. The job:
- Runs for 30-60 minutes
- Auto-terminates on completion
- Total cost: ~$1-2 USD per run

For testing with fewer templates (faster, cheaper):
- Edit `build_cosmos_bank_emr.py`
- Change `n_sources` to 1000 or 5000
- Runtime drops to ~5-15 minutes

## Important Notes

⚠️ **Run from emr-launcher only** (per user requirements)
⚠️ **Cluster auto-terminates** (saves cost, but lose state)
⚠️ **S3 is the source of truth** (local files on cluster are ephemeral)
✅ **Config auto-saved** (full audit trail in S3)
✅ **Deterministic** (same seed = same templates)
✅ **Incremental** (doesn't modify existing Gen3/Gen4 code)

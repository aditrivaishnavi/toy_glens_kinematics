# Gen5 Phase 4c: Pre-Flight Summary

**Date:** 2026-02-03  
**Status:** ✅ READY FOR SMOKE TEST LAUNCH  
**Commit:** `1fc3421`

---

## ✅ Pre-Flight Checks Complete

### S3 Resources Verified
- ✅ COSMOS Bank: `s3://darkhaloscope/data/cosmos_banks/cosmos_bank_20k_gen5.h5` (453 MB)
- ✅ Pipeline Script: `s3://darkhaloscope/scripts/gen5/spark_phase4_pipeline_gen5.py` (147 KB)
- ✅ Bootstrap Script: `s3://darkhaloscope/scripts/gen5/emr_bootstrap_gen5.sh` (819 bytes)
- ✅ Manifests: `s3://darkhaloscope/phase4_pipeline/phase4a/v4_sota_moffat/manifests/train_stamp64_bandsgrz_gridgrid_sota/` (25 parquet files)

### Code Review Completed
- ✅ COSMOS integration points validated
- ✅ Executor-local caching mechanism correct
- ✅ Deterministic source selection (BLAKE2B hash)
- ✅ Lensing render function uses lenstronomy
- ✅ Clean if/else switch (Sersic vs COSMOS)
- ✅ Output schema extended with COSMOS columns
- ✅ Config auto-save to S3 for audit trail

### Dependencies
- ✅ h5py 3.8.0 (EMR bootstrap)
- ✅ lenstronomy 1.11.6 (EMR bootstrap)
- ✅ scipy, astropy (EMR bootstrap)

---

## 🧪 Smoke Test Launch

**Purpose:** Validate COSMOS integration before full production run  
**Cost:** ~$2-5  
**Runtime:** ~10 minutes  
**Output:** ~50 stamps (debug tier only)

### Launch Command

```bash
# SSH to emr-launcher
ssh emr-launcher

# Run smoke test
cd /tmp
bash dark_halo_scope/emr/gen5/launch_gen5_smoke_test.sh
```

### Expected Output
```
s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_source_test/stamps/test_stamp64_bandsgrz_cosmos_smoke/
```

### Validation Steps

1. **Check job completed:**
   ```bash
   aws emr list-steps --cluster-id j-XXXXXXXXXXXXX --region us-east-1
   ```

2. **Download sample:**
   ```bash
   aws s3 cp s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_source_test/stamps/test_stamp64_bandsgrz_cosmos_smoke/part-00000*.parquet /tmp/sample.parquet --region us-east-1
   ```

3. **Run validation:**
   ```bash
   python3 dark_halo_scope/scripts/validate_gen5_smoke_test.py /tmp/sample.parquet
   ```

### Success Criteria

- ✅ Job completes without errors
- ✅ Output parquet files created
- ✅ Schema includes `source_mode`, `cosmos_index`, `cosmos_hlr_arcsec`
- ✅ `source_mode == 'cosmos'` for all rows
- ✅ `cosmos_index` in [0, 20000)
- ✅ HLR distribution matches COSMOS bank (0.1-1.25 arcsec)
- ✅ Stamps have correct shape (3, 64, 64)

**If any check fails: DO NOT proceed to production. Debug and re-run smoke test.**

---

## 🚀 Full Production Run (After Smoke Test Passes)

**Purpose:** Generate full Gen5 training dataset with COSMOS sources  
**Cost:** ~$50-100  
**Runtime:** 4-8 hours  
**Output:** ~400K stamps (200K positives + 200K controls)

### Launch Command

```bash
# Create production launch script
cat > /tmp/launch_gen5_production.sh << 'EOF'
#!/bin/bash
aws emr create-cluster \
  --name "Gen5-Phase4c-COSMOS-Production-$(date +%Y%m%d-%H%M)" \
  --region us-east-1 \
  --release-label emr-6.10.0 \
  --applications Name=Spark \
  --instance-type m5.2xlarge \
  --instance-count 34 \
  --service-role EMR_DefaultRole \
  --ec2-attributes InstanceProfile=EMR_EC2_DefaultRole \
  --bootstrap-actions Path=s3://darkhaloscope/scripts/gen5/emr_bootstrap_gen5.sh \
  --log-uri s3://darkhaloscope/emr-logs/ \
  --steps Type=Spark,Name="Phase4c-COSMOS-Production",ActionOnFailure=CONTINUE,Args=[\
--deploy-mode,cluster,\
--driver-memory,8g,\
--executor-memory,18g,\
--executor-cores,4,\
--num-executors,100,\
--conf,spark.sql.parquet.compression.codec=gzip,\
--conf,spark.dynamicAllocation.enabled=true,\
--conf,spark.dynamicAllocation.minExecutors=20,\
--conf,spark.dynamicAllocation.maxExecutors=100,\
s3://darkhaloscope/scripts/gen5/spark_phase4_pipeline_gen5.py,\
--stage,4c,\
--output-s3,s3://darkhaloscope/phase4_pipeline,\
--variant,v5_cosmos_source,\
--experiment-id,train_stamp64_bandsgrz_cosmos,\
--source-mode,cosmos,\
--cosmos-bank-h5,s3://darkhaloscope/data/cosmos_banks/cosmos_bank_20k_gen5.h5,\
--cosmos-salt,gen5_v1,\
--seed-base,1337,\
--psf-model,moffat,\
--moffat-beta,3.5,\
--split-seed,42,\
--tiers,train,\
--grid-train,grid_small,\
--n-total-train-per-split,200000,\
--manifests-subdir,manifests,\
--parent-s3,s3://darkhaloscope/phase4_pipeline/phase4a/v4_sota_moffat/manifests/train_stamp64_bandsgrz_gridgrid_sota\
] \
  --auto-terminate
EOF

# Launch
bash /tmp/launch_gen5_production.sh
```

### Monitoring

```bash
# Check cluster status
aws emr describe-cluster --cluster-id j-XXXXXXXXXXXXX --region us-east-1 | grep State

# Check step progress
aws emr list-steps --cluster-id j-XXXXXXXXXXXXX --region us-east-1

# Check output
aws s3 ls s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_source/stamps/train_stamp64_bandsgrz_cosmos/ --region us-east-1 --recursive | wc -l
```

---

## 📊 Key Differences: Gen4 vs Gen5

| Feature | Gen4 (Sersic) | Gen5 (COSMOS) |
|---------|---------------|---------------|
| **Source Morphology** | Parametric (n=1) | Real HST galaxies |
| **Clumpiness** | ~0.0 | 0.03-0.98 (median 0.56) |
| **HLR Range** | Sampled distribution | 0.114-1.248 arcsec (COSMOS) |
| **Substructure** | None | Star-forming clumps, spiral arms |
| **Expected SLACS Recall** | 0% (failed) | 50-70% (target) |
| **Expected Hard Neg Contamination** | 95% (failed) | 10-20% (target) |

---

## 📁 Files Created

```
dark_halo_scope/
├── docs/
│   ├── gen5_cosmos_bank_creation.md      # COSMOS bank documentation
│   └── gen5_phase4c_preflight.md         # This pre-flight guide (820 lines)
├── emr/gen5/
│   ├── emr_bootstrap_gen5.sh             # EMR dependencies (h5py, lenstronomy)
│   └── launch_gen5_smoke_test.sh         # Smoke test launch script
├── configs/gen5/
│   ├── cosmos_bank_config.json           # COSMOS bank build config
│   ├── phase4c_config.json               # Production config
│   └── phase4c_smoke_test_config.json    # Smoke test config
└── scripts/
    └── validate_gen5_smoke_test.py       # Data quality validation script
```

---

## ⚠️ Risk Mitigation

| Risk | Mitigation |
|------|------------|
| h5py not installed | ✅ EMR bootstrap script |
| lenstronomy missing | ✅ EMR bootstrap script |
| COSMOS bank inaccessible | ✅ S3 path verified |
| Manifests missing | ✅ Verified 25 parquet files exist |
| Wrong bucket name | ✅ Fixed (use `darkhaloscope`) |
| Executor memory too low | ✅ 12-18 GB sufficient for 453 MB bank |
| Lensing bugs | ⚠️ Smoke test will catch |
| Low clumpiness | ⚠️ Validation script checks HLR |

---

## 🎯 Decision Point

**READY TO LAUNCH SMOKE TEST?**

- [x] All S3 resources verified
- [x] Code review complete
- [x] Bootstrap script uploaded
- [x] Validation script ready
- [x] Manifests confirmed
- [x] Cost approved (~$2-5)

**ACTION REQUIRED:** Run smoke test on emr-launcher:
```bash
ssh emr-launcher
cd /tmp
bash dark_halo_scope/emr/gen5/launch_gen5_smoke_test.sh
```

**Timeline:**
- Smoke test: ~10 min
- Validation: ~5 min
- If passes → Launch production: 4-8 hours
- **Total: 5-9 hours to complete Gen5 data generation**

---

**STATUS:** 🟢 All pre-flight checks passed. Ready to launch.


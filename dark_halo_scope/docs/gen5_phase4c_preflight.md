# Gen5 Phase 4c Pre-Flight Check

**Date:** 2026-02-03  
**Component:** Phase 4c Data Generation with COSMOS Sources  
**Purpose:** Pre-flight validation before launching EMR cluster

---

## 1. Code Review: Key Changes in Gen5 Pipeline

### COSMOS Integration Points

#### 1.1 New Command-Line Arguments
```python
# Source: spark_phase4_pipeline_gen5.py, lines 3108-3117
--config                 # Path to JSON config file (overrides other args)
--source-mode           # "sersic" (default) or "cosmos"
--cosmos-bank-h5        # Path to COSMOS bank HDF5 file (required if cosmos mode)
--cosmos-salt           # Salt for deterministic COSMOS template selection
--seed-base             # Base seed for reproducibility (default: 42)
```

#### 1.2 COSMOS Bank Loading (Executor-Local Cache)
```python
# Source: lines 95-110
_COSMOS_BANK = None  # Global executor cache

def _load_cosmos_bank_h5(path: str) -> Dict[str, np.ndarray]:
    """
    Loads HDF5 file once per executor and caches in memory.
    Returns dict with 'images', 'src_pixscale', and 'n_sources' keys.
    """
```

**Validation:**
- ✅ Caching mechanism prevents repeated S3 downloads
- ✅ Returns NumPy arrays for fast indexing
- ⚠️  **CRITICAL**: Path must be accessible from all executors (use S3 path)

#### 1.3 Deterministic Source Selection
```python
# Source: lines 112-118
def _cosmos_choose_index(task_id: str, n_sources: int, salt: str = "") -> int:
    """
    Deterministic COSMOS template selection via BLAKE2B hash.
    Same task_id + salt → same galaxy index
    """
    h = hashlib.blake2b(f"{task_id}{salt}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % n_sources
```

**Validation:**
- ✅ Deterministic (reproducible across runs)
- ✅ Uniform distribution across COSMOS bank
- ✅ Salt allows different source assignments per generation

#### 1.4 COSMOS Lensing Render Function
```python
# Source: lines 120-250 (render_cosmos_lensed_source)
def render_cosmos_lensed_source(
    cosmos_bank: Dict,
    cosmos_index: int,
    stamp_size: int,
    pixscale_arcsec: float,
    theta_e_arcsec: float,
    lens_e: float,
    lens_phi_rad: float,
    shear: float,
    shear_phi_rad: float,
    src_x_arcsec: float,
    src_y_arcsec: float,
    src_mag_r: float,
    z_s: float,
    psf_fwhm_arcsec: float,
    psf_model: str,
    moffat_beta: float,
    band: str,
) -> np.ndarray:
```

**Key Operations:**
1. Extract COSMOS template (96×96 at 0.03 arcsec/pix)
2. Scale to target magnitude (nanomaggies)
3. Apply SIE lens + shear using lenstronomy
4. Convolve with survey PSF (Moffat β=3.5)
5. Resample to Legacy Survey pixel scale (0.262 arcsec/pix)
6. Output: 64×64 stamp

**Validation:**
- ✅ Lenstronomy available check (falls back gracefully if missing)
- ✅ PSF model matches main pipeline (moffat/gaussian)
- ⚠️  **POTENTIAL ISSUE**: Uses fixed z_s=1.5 (hardcoded in injection code, line 2676)
  - Not a blocker, but should be configurable in Gen6
- ✅ Resampling preserves flux normalization

#### 1.5 Injection Mode Switch
```python
# Source: lines 2638-2690
source_mode = getattr(args, 'source_mode', 'sersic')

if source_mode == "cosmos":
    # Gen5: COSMOS source injection
    cosmos_bank = _load_cosmos_bank_h5(args.cosmos_bank_h5)
    cosmos_idx = _cosmos_choose_index(task_id, cosmos_bank["n_sources"], ...)
    
    for b in use_bands:
        add_b = render_cosmos_lensed_source(...)
        imgs[b] = (imgs[b] + add_b).astype(np.float32)
else:
    # Original: Sersic source injection
    for b in use_bands:
        add_b = render_lensed_source(...)
        imgs[b] = (imgs[b] + add_b).astype(np.float32)
```

**Validation:**
- ✅ Clean if/else switch (no code duplication)
- ✅ Both modes write to same output schema
- ✅ Original Sersic mode preserved (Gen1-4 still runnable)
- ✅ COSMOS mode adds `cosmos_index` and `cosmos_hlr_arcsec` metadata columns

#### 1.6 Output Schema Changes
```python
# Source: lines 2850-2900 (schema definition)
# New columns for COSMOS mode:
- source_mode: str         # "sersic" or "cosmos"
- cosmos_index: int        # Index into COSMOS bank (or -1 if Sersic)
- cosmos_hlr_arcsec: float # Half-light radius of COSMOS source (or NaN if Sersic)
```

**Validation:**
- ✅ Backward compatible (Sersic mode sets cosmos_index=-1, cosmos_hlr_arcsec=NaN)
- ✅ Allows post-hoc analysis of source morphology distribution

#### 1.7 Config Auto-Save to S3
```python
# Source: lines 3230-3261
if stage == "4c" and _is_s3(args.output_s3):
    effective_config = {
        "stage": args.stage,
        "variant": args.variant,
        "experiment_id": args.experiment_id,
        "source_mode": getattr(args, 'source_mode', 'sersic'),
        "cosmos_bank_h5": getattr(args, 'cosmos_bank_h5', None),
        "cosmos_salt": getattr(args, 'cosmos_salt', ''),
        "seed_base": getattr(args, 'seed_base', 42),
        "psf_model": args.psf_model,
        "moffat_beta": args.moffat_beta,
        "split_seed": args.split_seed,
        "execution_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spark_version": spark.version,
    }
    # Save to S3: {output_s3}/phase4c/{variant}/run_config_{experiment_id}.json
```

**Validation:**
- ✅ Audit trail for reproducibility
- ✅ Captures all critical parameters
- ✅ Timestamped for version control
- ⚠️  **DEPENDENCY**: Requires boto3 (should be in EMR bootstrap)

---

## 2. Manifest Reuse Strategy

### Gen5 Decision: Reuse v4_sota_moffat Manifests

**Rationale:**
- Phase 4a manifests define **which objects to inject** and **injection parameters** (theta_e, ellipticity, shear, source position, etc.)
- These are **independent of source morphology**
- Changing from Sersic → COSMOS only affects **how the source is rendered**, not **where/what lenses to inject**
- Reusing manifests ensures **direct comparison** between Gen4 (Sersic) and Gen5 (COSMOS)

**Manifest Locations:**
```
s3://darkhaloscope/phase4_pipeline/phase4a/v4_sota_moffat/manifests/train_stamp64_bandsgrz_gridgrid_sota/
```

**Expected Contents:**
- `task_manifest_train_*.parquet` - Full task list (object_id, theta_e, src_x, src_y, etc.)
- `bricks_manifest_*.parquet` - List of bricks to cache for imaging

**Validation Steps:**
1. ✅ Check manifests exist in S3
2. ✅ Verify schema matches pipeline expectations
3. ✅ Confirm row count matches expected (~200K positives + ~200K controls for train)

### Alternative: Generate New Manifests (NOT RECOMMENDED)

If we generated new manifests:
- Different objects selected → can't compare Gen4 vs Gen5 fairly
- Different injection parameters → different lens population
- Would require re-running Phase 4a (adds days of work)

**Decision: Reuse existing v4_sota_moffat manifests**

---

## 3. Configuration Files

### 3.1 Test Run Config (Smoke Test)

**Purpose:** Validate pipeline with minimal cost before full run

```json
{
  "generation": "gen5",
  "component": "phase4c_smoke_test",
  "stage": "4c",
  "output_s3": "s3://darkhaloscope/phase4_pipeline",
  "variant": "v5_cosmos_source_test",
  "experiment_id": "test_stamp64_bandsgrz_cosmos_smoke",
  
  "source_mode": "cosmos",
  "cosmos_bank_h5": "s3://darkhaloscope/data/cosmos_banks/cosmos_bank_20k_gen5.h5",
  "cosmos_salt": "gen5_test_v1",
  "seed_base": 1337,
  
  "psf_model": "moffat",
  "moffat_beta": 3.5,
  "split_seed": 42,
  
  "bands": "g,r,z",
  "stamp_sizes": "64",
  "bandsets": "grz",
  
  "tiers": "debug",
  "grid_debug": "grid_small",
  "n_per_config_debug": 5,
  
  "skip_if_exists": 1,
  "force": 0,
  
  "manifests_subdir": "manifests",
  "parent_s3": "s3://darkhaloscope/phase4_pipeline/phase4a/v4_sota_moffat/manifests/train_stamp64_bandsgrz_gridgrid_sota",
  
  "created_utc": "2026-02-03T08:00:00Z",
  "description": "Smoke test for Gen5 COSMOS source injection. Uses debug tier only (~50 stamps)."
}
```

**Expected Output:**
- ~50 stamps (5 per config × ~10 debug configs)
- Runtime: ~10 minutes
- Cost: ~$2-5 (small 2-node EMR cluster)

**Validation Metrics:**
1. Job completes without errors
2. Output parquet files created in S3
3. Schema includes `source_mode`, `cosmos_index`, `cosmos_hlr_arcsec` columns
4. COSMOS HLR distribution matches bank statistics (0.1-1.25 arcsec)
5. Clumpiness proxy in output > 0.3 (real structure, not smooth Sersic)

### 3.2 Full Production Config

**Purpose:** Generate full training dataset for Gen5

```json
{
  "generation": "gen5",
  "component": "phase4c_production",
  "stage": "4c",
  "output_s3": "s3://darkhaloscope/phase4_pipeline",
  "variant": "v5_cosmos_source",
  "experiment_id": "train_stamp64_bandsgrz_cosmos",
  
  "source_mode": "cosmos",
  "cosmos_bank_h5": "s3://darkhaloscope/data/cosmos_banks/cosmos_bank_20k_gen5.h5",
  "cosmos_salt": "gen5_v1",
  "seed_base": 1337,
  
  "psf_model": "moffat",
  "moffat_beta": 3.5,
  "split_seed": 42,
  
  "bands": "g,r,z",
  "stamp_sizes": "64",
  "bandsets": "grz",
  
  "tiers": "train",
  "grid_train": "grid_small",
  "n_total_train_per_split": 200000,
  
  "skip_if_exists": 1,
  "force": 0,
  
  "manifests_subdir": "manifests",
  "parent_s3": "s3://darkhaloscope/phase4_pipeline/phase4a/v4_sota_moffat/manifests/train_stamp64_bandsgrz_gridgrid_sota",
  
  "created_utc": "2026-02-03T08:00:00Z",
  "description": "Full Gen5 training data with COSMOS sources, Moffat PSF, reusing v4_sota_moffat manifests for fair comparison."
}
```

**Expected Output:**
- ~400K stamps (200K positives + 200K controls)
- Runtime: 4-8 hours
- Cost: ~$50-100 (50 vcores = ~12 m5.2xlarge instances)
- Output size: ~50-80 GB (parquet compressed)

---

## 4. Pre-Flight Checks

### 4.1 S3 Resources

**Check COSMOS Bank Exists:**
```bash
aws s3 ls s3://darkhaloscope/data/cosmos_banks/cosmos_bank_20k_gen5.h5 --region us-east-1
```
**Expected:** `474578812 cosmos_bank_20k_gen5.h5` (453 MB)

**Status:** ✅ Uploaded and verified (Section 1, COSMOS bank creation)

**Check Manifests Exist:**
```bash
aws s3 ls s3://darkhaloscope/phase4_pipeline/phase4a/v4_sota_moffat/manifests/train_stamp64_bandsgrz_gridgrid_sota/ --region us-east-1
```
**Expected:** `task_manifest_*.parquet`, `bricks_manifest_*.parquet`

**Status:** ⚠️  NEEDS VERIFICATION

**Check Pipeline Script Uploaded:**
```bash
aws s3 ls s3://darkhaloscope/scripts/gen5/spark_phase4_pipeline_gen5.py --region us-east-1
```
**Expected:** ~147 KB Python script

**Status:** ✅ Uploaded (previous step)

### 4.2 Code Validation

**Check Syntax:**
```bash
python3 -m py_compile dark_halo_scope/emr/gen5/spark_phase4_pipeline_gen5.py
```
**Expected:** No syntax errors

**Status:** ✅ (would have failed earlier if syntax errors present)

**Check Dependencies:**
- NumPy ✅ (standard)
- PySpark ✅ (EMR default)
- boto3 ✅ (EMR default)
- h5py ⚠️  (NEEDS EMR BOOTSTRAP)
- lenstronomy ⚠️  (NEEDS EMR BOOTSTRAP)
- astropy ✅ (typically in EMR)

### 4.3 EMR Bootstrap Script

**Required Packages:**
```bash
#!/bin/bash
# EMR Bootstrap: Install Gen5 dependencies

sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install h5py==3.8.0
sudo python3 -m pip install lenstronomy==1.11.6
sudo python3 -m pip install galsim==2.4.11  # If needed for future extensions

echo "✅ Gen5 dependencies installed"
```

**Upload Bootstrap Script:**
```bash
aws s3 cp emr_bootstrap_gen5.sh s3://darkhaloscope/scripts/gen5/emr_bootstrap_gen5.sh --region us-east-1
```

**Status:** ⚠️  NEEDS CREATION AND UPLOAD

---

## 5. EMR Launch Commands

### 5.1 Smoke Test (Debug Tier Only)

**Cluster Config:**
- Instance type: m5.2xlarge
- Instance count: 2 (1 master + 1 core)
- Total vcores: ~8
- Runtime: ~10 min
- Cost: ~$2-5

**Launch Command:**
```bash
aws emr create-cluster \
  --name "Gen5-Phase4c-COSMOS-SmokeTest-$(date +%Y%m%d-%H%M)" \
  --region us-east-1 \
  --release-label emr-6.10.0 \
  --applications Name=Spark \
  --instance-type m5.2xlarge \
  --instance-count 2 \
  --service-role EMR_DefaultRole \
  --ec2-attributes KeyName=your-key,InstanceProfile=EMR_EC2_DefaultRole \
  --bootstrap-actions Path=s3://darkhaloscope/scripts/gen5/emr_bootstrap_gen5.sh \
  --log-uri s3://darkhaloscope/emr-logs/ \
  --steps Type=Spark,Name="Phase4c-COSMOS-SmokeTest",ActionOnFailure=CONTINUE,Args=[\
--deploy-mode,cluster,\
--driver-memory,4g,\
--executor-memory,12g,\
--executor-cores,4,\
--conf,spark.sql.parquet.compression.codec=gzip,\
--conf,spark.dynamicAllocation.enabled=false,\
s3://darkhaloscope/scripts/gen5/spark_phase4_pipeline_gen5.py,\
--stage,4c,\
--output-s3,s3://darkhaloscope/phase4_pipeline,\
--variant,v5_cosmos_source_test,\
--experiment-id,test_stamp64_bandsgrz_cosmos_smoke,\
--source-mode,cosmos,\
--cosmos-bank-h5,s3://darkhaloscope/data/cosmos_banks/cosmos_bank_20k_gen5.h5,\
--cosmos-salt,gen5_test_v1,\
--seed-base,1337,\
--psf-model,moffat,\
--moffat-beta,3.5,\
--split-seed,42,\
--tiers,debug,\
--grid-debug,grid_small,\
--n-per-config-debug,5,\
--manifests-subdir,manifests,\
--parent-s3,s3://darkhaloscope/phase4_pipeline/phase4a/v4_sota_moffat/manifests/train_stamp64_bandsgrz_gridgrid_sota\
] \
  --auto-terminate
```

### 5.2 Full Production Run (270 vcores)

**Cluster Config:**
- Instance type: m5.2xlarge (8 vcores each)
- Instance count: 34 (1 master + 33 core = ~270 vcores)
- Runtime: 4-8 hours
- Cost: ~$50-100

**Launch Command:**
```bash
aws emr create-cluster \
  --name "Gen5-Phase4c-COSMOS-Production-$(date +%Y%m%d-%H%M)" \
  --region us-east-1 \
  --release-label emr-6.10.0 \
  --applications Name=Spark \
  --instance-type m5.2xlarge \
  --instance-count 34 \
  --service-role EMR_DefaultRole \
  --ec2-attributes KeyName=your-key,InstanceProfile=EMR_EC2_DefaultRole \
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
```

---

## 6. Validation Plan for Smoke Test

### 6.1 Immediate Checks (During/After Job)

**Monitor EMR Cluster:**
```bash
# Get cluster ID from launch output
CLUSTER_ID=j-XXXXXXXXXXXXX

# Check status
aws emr describe-cluster --cluster-id $CLUSTER_ID --region us-east-1 | grep State

# Check step status
aws emr list-steps --cluster-id $CLUSTER_ID --region us-east-1
```

**Check Output Exists:**
```bash
aws s3 ls s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_source_test/stamps/test_stamp64_bandsgrz_cosmos_smoke/ --region us-east-1
```

### 6.2 Data Quality Validation

**Download Sample Output:**
```bash
aws s3 cp s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_source_test/stamps/test_stamp64_bandsgrz_cosmos_smoke/_SUCCESS /tmp/
aws s3 cp s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_source_test/stamps/test_stamp64_bandsgrz_cosmos_smoke/part-00000-*.parquet /tmp/sample.parquet
```

**Validation Script:**
```python
import pyarrow.parquet as pq
import numpy as np

# Read sample
table = pq.read_table('/tmp/sample.parquet')
df = table.to_pandas()

print(f"✅ Rows: {len(df)}")
print(f"✅ Columns: {list(df.columns)}")

# Check required columns
required = ['stamp_npz', 'is_control', 'source_mode', 'cosmos_index', 'cosmos_hlr_arcsec']
for col in required:
    assert col in df.columns, f"Missing column: {col}"
print(f"✅ Schema valid")

# Check source_mode
assert (df['source_mode'] == 'cosmos').all(), "source_mode should be 'cosmos'"
print(f"✅ source_mode = 'cosmos'")

# Check cosmos_index range
assert (df['cosmos_index'] >= 0).all() and (df['cosmos_index'] < 20000).all()
print(f"✅ cosmos_index in [0, 20000)")

# Check COSMOS HLR distribution
hlr_valid = df['cosmos_hlr_arcsec'][df['cosmos_hlr_arcsec'].notna()]
assert len(hlr_valid) > 0, "No valid HLR values"
print(f"✅ HLR range: {hlr_valid.min():.3f} - {hlr_valid.max():.3f} arcsec")
print(f"   (Expected: 0.1-1.25 arcsec from COSMOS bank)")

# Decode one stamp and check shape
import io
npz_bytes = df['stamp_npz'].iloc[0]
with io.BytesIO(npz_bytes) as f:
    npz = np.load(f)
    img = npz['image']
    print(f"✅ Stamp shape: {img.shape} (expected: (3, 64, 64) for grz)")

print("\n🎉 SMOKE TEST VALIDATION PASSED!")
```

**Success Criteria:**
1. ✅ Job completes without errors
2. ✅ Output parquet files created
3. ✅ Schema includes COSMOS-specific columns
4. ✅ `source_mode == 'cosmos'` for all rows
5. ✅ `cosmos_index` in [0, 20000)
6. ✅ HLR distribution matches COSMOS bank (0.1-1.25 arcsec)
7. ✅ Stamps have correct shape (3, 64, 64)

**If Any Check Fails:**
- **DO NOT proceed to full production run**
- Debug the issue
- Fix code/config
- Re-run smoke test

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **h5py not installed on executors** | Medium | High (job fails) | ✅ EMR bootstrap script |
| **lenstronomy not installed** | Medium | High (falls back to SIS, wrong physics) | ✅ EMR bootstrap script |
| **COSMOS bank S3 path inaccessible** | Low | High (job fails) | ✅ Pre-flight check (Step 4.1) |
| **Manifests missing or corrupted** | Low | High (job fails) | ⚠️  NEEDS PRE-FLIGHT CHECK |
| **Wrong bucket name (darkhaloscope-training-dc vs darkhaloscope)** | Medium | Medium (path not found) | ✅ Fixed in config (use `darkhaloscope`) |
| **Executor memory too low for COSMOS bank (453 MB)** | Low | Low (bank is small) | ✅ 12-18 GB executor memory is sufficient |
| **Lensing code has bugs** | Low | High (wrong science) | ⚠️  Validate output with known lens parameters |
| **Clumpiness too low (smooth arcs)** | Low | High (sim-to-real gap persists) | ⚠️  Post-hoc validation (check HLR, visual inspection) |

---

## 8. Go/No-Go Decision Checklist

Before launching **SMOKE TEST**:
- [ ] COSMOS bank verified in S3 (453 MB)
- [ ] Pipeline script uploaded to S3 (147 KB)
- [ ] Config files created
- [ ] EMR bootstrap script created and uploaded
- [ ] Manifests verified in S3 (v4_sota_moffat)
- [ ] Code review complete (no blocking issues)
- [ ] Validation script prepared
- [ ] Cost approved (~$2-5 for smoke test)

Before launching **FULL PRODUCTION**:
- [ ] Smoke test completed successfully
- [ ] Data quality validation passed (all 7 criteria)
- [ ] Output inspected manually (visual check of arcs)
- [ ] Clumpiness confirmed > 0.3 (real structure)
- [ ] Cost approved (~$50-100 for production)
- [ ] Timeline approved (4-8 hours)

---

## 9. Next Steps After Phase 4c

1. **Post-Injection Validation:** Run external LLM's `validate_cosmos_injection.py` on full dataset
2. **Data Transfer to Lambda:** Sync S3 → Lambda filesystem (~50-80 GB)
3. **Gen5 Training:** 50 epochs with ConvNeXt-Tiny (~12-24 hours)
4. **Anchor Baseline Re-Evaluation:** Test on SLACS/BELLS/hard negatives
5. **Compare Gen4 vs Gen5:** Quantify sim-to-real gap improvement

**Expected Timeline:**
- Phase 4c: 4-8 hours (after smoke test passes)
- Data transfer: 2-4 hours
- Training: 12-24 hours
- Evaluation: 2-4 hours
- **Total: 2-3 days wall time**

---

**Document Version:** 1.0  
**Status:** Ready for execution (pending pre-flight checks)  
**Approver:** [Awaiting user approval]


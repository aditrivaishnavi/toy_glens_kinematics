# StrongLens Calibration: Full Project Context

**Last Updated:** 2026-02-09  
**Author:** Aditrivaishnavi Balaji

---

## 1. Scientific Goal

### What We Are Doing

We are building a **real-image-trained strong gravitational lens detector** for the DESI Legacy Survey DR10, and measuring its **selection function** (detection probability as a function of observing conditions and lens properties).

**Paper Title:** "Selection Functions and Failure Modes of Real-Image Lens Finders in DR10"

**Core Thesis:**
> "Given a real-image-trained lens finder, we measure its DR10 selection function via calibrated injection-recovery, quantify failure modes, and provide bias-aware guidance for lens demographics and follow-up prioritization."

### What We Are NOT Doing

1. **NOT training on simulated lensed arcs** - Previous attempt (Plan B) failed because simulated arcs were ~100x too bright compared to real DR10 data
2. **NOT claiming overall completeness** of DR10 lens surveys
3. **NOT making cosmology constraints** directly
4. **NOT trying to outperform Huang et al.** without matched protocol
5. **NOT using SLACS/BELLS as primary evaluation** - These are spectroscopically discovered and often invisible in ground-based imaging

### Scientific Impact

- **Cosmology/lens counts:** Selection functions distort inferred abundance if not modeled; we provide a lookup table for detection probability
- **Substructure studies:** If completeness depends on arc brightness/seeing, you're biased toward smooth/high-SNR arcs
- **Time-delay cosmography:** If selection favors certain configurations, the discovered sample is not representative
- **Survey design:** Results translate to requirements on seeing/depth/exposure strategy

---

## 2. Research Strategy (Following Huang et al.)

### The Paper We Are Replicating

**Primary Reference:** Huang et al. (arXiv:2508.20087v1) - Papers I-IV combined

Key methodological anchors we copy:
1. **Training on real DR10 cutouts** (no simulations)
2. **Nonlens selection stratified by z-band exposure count** to prevent network shortcuts
3. **100:1 nonlens:lens ratio** per stratum
4. **101×101 pixel cutouts** (~26" at 0.262"/pixel)
5. **Very high threshold** (top 0.01%) for discovery

### Two-Tier Label System

| Tier | Definition | Count | Use |
|------|------------|-------|-----|
| **Tier-A** | Spectroscopically confirmed lenses | 389 | Primary evaluation metric |
| **Tier-B** | Probable candidates (grading="probable") | 4,399 | Training with label smoothing (target=0.8) |

### Stratification Axes

1. **nobs_z** (z-band exposures): 1-2, 3-5, 6-10, 11+
2. **tractor_type** (morphology): SER, DEV, REX (EXP excluded per Paper IV)
3. **psfsize_z** (seeing): for analysis, not stratification
4. **psfdepth_z** (depth): for analysis, not stratification

---

## 3. The Master Plan (Week-by-Week)

### Week 1: Data Preparation ✅ COMPLETE
- Positive catalog ingestion (5,104 candidates from lenscat)
- DR10 Tractor metadata crossmatch (93.8% matched)
- Stratified negative sampling from DR10 sweep files (~510K negatives)
- Cutout generation (positives + negatives)
- Validation gates (quality checks, shortcut detection)

### Week 2: Model Training 🔄 IN PROGRESS
- Complete cutout downloads to Lambda GPU
- Create train/val/test splits (70/15/15, HEALPix-based)
- Implement augmentations (rotation, flip, noise)
- Train ResNet-18 baseline
- Validate training stability (AUC, no collapse)
- Sanity check top-K predictions

### Week 3: Selection Function + Failures
- Recall on Tier-A by stratum (bootstrap CIs)
- Small-N strata handling (binomial/beta intervals)
- Calibration curves (ECE, reliability diagrams)
- FPR by contaminant category
- Spatial holdout CV

### Week 4: Ensemble + Paper
- Domain-split models (PSF or nobs axis)
- Diversity metrics (correlation, disagreement)
- Ensemble evaluation
- Paper figures (7-8 main figures)
- Paper tables (5-6 tables)
- Paper text

---

## 4. Infrastructure: Where Everything Lives

### AWS S3 (Data Storage)
```
s3://darkhaloscope/
├── dr10/
│   ├── sweeps/                      # DR10 sweep files (1,436 .fits.gz files)
│   └── sweeps_manifest/             # Manifest of sweep files
└── stronglens_calibration/
    ├── configs/                     # Configuration files
    ├── positives_with_dr10/         # Crossmatched positive catalog
    ├── manifests/                   # 114M row negative pool manifest
    ├── sampled_negatives/           # Stratified ~510K negatives
    ├── cutouts/
    │   ├── positives/               # 5,101 .npz cutout files
    │   └── negatives/               # 416,088 .npz cutout files
    ├── validation/                  # Validation reports
    ├── emr/
    │   ├── code/                    # Spark scripts uploaded to S3
    │   ├── logs/                    # EMR cluster logs
    │   └── bootstrap/               # Bootstrap scripts
    └── checkpoints/                 # Job checkpoints
```

**S3 Access:**
```bash
aws s3 ls s3://darkhaloscope/stronglens_calibration/ --region us-east-2
```

### AWS EMR (Distributed Processing)

**What we use EMR for:**
- Processing 114M galaxies from DR10 sweep files
- Generating ~500K cutouts in parallel
- Running validation on all cutouts

**EMR Cluster Configuration:**
- **Region:** us-east-2 (MUST be explicit!)
- **Instance types:** m5.xlarge (4 vCPU, 16GB) or m5.2xlarge (8 vCPU, 32GB)
- **Typical cluster:** 20-30 workers for full runs, 2-5 for testing
- **EMR Release:** emr-7.0.0

**Launching EMR Jobs:**
```bash
cd stronglens_calibration

# Test run (2 workers)
python emr/launch_validate_cutouts.py --preset test --positives s3://... --negatives s3://...

# Full run (30 workers)
python emr/launch_validate_cutouts.py --preset large-xlarge --positives s3://... --negatives s3://...
```

**EMR Console Access:**
```
https://us-east-2.console.aws.amazon.com/emr/home?region=us-east-2
```

### Lambda GPU (Training)

**Location:** Lambda Labs cloud GPU instance

**NFS Mount Point:**
```
/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/
```

**Directory Structure on Lambda:**
```
stronglens_calibration/
├── cutouts/
│   ├── positives/     # 5,101 .npz files (~600 MB)
│   └── negatives/     # 416,088 .npz files (~51 GB)
├── manifests/         # Training manifests
├── code/              # Training code
├── logs/              # Training logs
└── checkpoints/       # Model checkpoints
```

**Syncing Data from S3 to Lambda:**
```bash
# On Lambda instance
rclone copy s3remote:darkhaloscope/stronglens_calibration/cutouts/positives/ \
    /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/cutouts/positives/ \
    --progress --transfers=8

rclone copy s3remote:darkhaloscope/stronglens_calibration/cutouts/negatives/ \
    /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/cutouts/negatives/ \
    --progress --transfers=8
```

**AWS Credentials for Lambda:**
```
AWS_ACCESS_KEY_ID=<Will Provide>
AWS_SECRET_ACCESS_KEY=<Will Provide>
AWS_REGION=us-east-2
```

### Local Development

**Workspace:**
```
stronglens_calibration/
```

**Key Directories:**
- `emr/` - Spark jobs and EMR launchers
- `dhs/` - Training code (from dhs_experiment_matrix_code)
- `scripts/` - Utility scripts
- `configs/` - Configuration files
- `tests/` - Unit and integration tests
- `docs/` - Documentation

---

## 5. Key Data Assets

| Asset | Count | Location |
|-------|-------|----------|
| DESI imaging candidates | 5,104 | `data/positives/desi_candidates.csv` |
| Tier-A (confident) | 389 | grading="confident" in above |
| Tier-B (probable) | 4,399 | grading="probable" in above |
| Crossmatched with DR10 | 4,788 (93.8%) | S3: positives_with_dr10/ |
| Negative pool (full) | 114M | S3: manifests/ |
| Sampled negatives | ~510K | S3: sampled_negatives/ |
| Positive cutouts | 5,101 | S3: cutouts/positives/ |
| Negative cutouts | 416,088 | S3: cutouts/negatives/ |

---

## 6. What We Learned (Lessons from Previous Failures)

### L1: Code Bugs (Critical)

1. **Import boto3 inside functions, not at module level** - EMR 6.x doesn't have boto3 pre-installed, and module-level try/except causes executors to get `None`
2. **Always check for duplicate function definitions** - Large files accumulate cruft, Python uses LAST definition
3. **Initialize variables before conditionals** - Variables used outside `if` blocks must be defined before
4. **Surface brightness units matter** - lenstronomy INTERPOL expects flux/arcsec², not flux/pixel (caused 1000x error)

### L2: Configuration Errors

1. **Always specify --region us-east-2 explicitly** - Default region can be wrong
2. **Subnets are region-specific** - Verify subnet exists in target region
3. **Match Spark memory to instance type** - m5.xlarge: executor-memory ≤ 4g, m5.2xlarge: ≤ 10g
4. **Set NUMBA_CACHE_DIR** - JIT compilers need writable cache directories

### L3: Wrong Assumptions

1. **EMR 6.x does NOT have boto3** - Always install dependencies explicitly
2. **`.limit(N)` doesn't reduce data scanned** - Use partition pruning for fast tests
3. **Smoke tests must run actual code path** - Don't test mocks
4. **Data can have NaN values** - Always validate before training
5. **SLACS/BELLS are the WRONG anchor set** - They were discovered via spectroscopy + HST, invisible in ground-based imaging

### L4: Critical Discovery - Sim-to-Real Gap

**Problem:** Gen5 model achieved AUC=0.9945 on synthetic data but only 4.4% recall on real SLACS/BELLS.

**Root Cause:** 
- SLACS/BELLS lenses are ~100x fainter than training LRGs in DR10
- Model learned "bright center = lens" shortcut
- Arc overlaps with PSF-blurred center, creating core brightness difference

**Solution:** 
- Use real-image training (not simulated arcs)
- Proper anchor sets from ground-based discoveries
- Shortcut detection gates before training

### L5: Process Failures

1. **Test locally before EMR** - Each EMR iteration takes 20-30 minutes
2. **Verify S3 code matches local** - Use checksums
3. **Track code versions** - Use git commit hashes
4. **Don't declare victory prematurely** - Wait for verified output

---

## 7. Current Pipeline Status

### Completed ✅

1. **Positive catalog acquisition** - 5,104 DESI imaging candidates
2. **DR10 crossmatch** - 93.8% matched (4,788 of 5,104)
3. **Negative pool extraction** - 114M galaxies from DR10 sweeps
4. **Stratified sampling** - ~510K negatives at 100:1 ratio
5. **Cutout generation** - 5,101 positive + 416,088 negative cutouts
6. **Validation pipeline** - Quality checks + shortcut detection
7. **Code review fixes** - AWS_REGION centralization, test fixes, manifest generation

### In Progress 🔄

1. **Data sync to Lambda** - Copying 52GB of cutouts
2. **Training manifest generation** - Bridging cutouts to training format

### Pending ⏳

1. **ResNet-18 baseline training**
2. **Selection function measurement**
3. **Failure mode analysis**
4. **Paper figures and tables**

---

## 8. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | ResNet-18 | Paper IV parity, sufficient for 101×101 |
| Cutout size | 101×101 (crop to 64×64 for training) | ~26" captures Einstein radius + context |
| Bands | g, r, z | Standard 3-channel, skip i-band |
| Negative ratio | 100:1 per stratum | Paper IV methodology |
| N1:N2 ratio | 85:15 | 85% deployment-representative, 15% hard confusers |
| Spatial splits | HEALPix nside=128 | Publication-grade independence |
| Label handling | Sample weights (Tier-A=1.0, Tier-B=0.5) | Not label smoothing alone |
| EXP excluded | Yes | Paper IV parity |
| Lens exclusion radius | 11" | 5" + 2×θE_max |

---

## 9. Shortcut Detection (Critical)

### Expected Physics Features (NOT Shortcuts)

These features SHOULD separate positives from negatives:
- `azimuthal_asymmetry` - Arcs are inherently asymmetric
- `annulus_brightness_r` - Lenses have arc flux in annulus
- `annulus_max_r`, `annulus_std_r` - Arc structure

### True Shortcuts (Problems)

These would indicate data quality issues:
- `core_brightness_r` - Core shouldn't differ
- `mad_r` - Noise level shouldn't differ
- `radial_gradient_r` - Radial profile shouldn't systematically differ

### AUC Thresholds

- **Green (OK):** AUC lower CI < 0.60
- **Yellow (Warning):** 0.60 < AUC lower CI ≤ 0.70
- **Red (Shortcut):** AUC lower CI > 0.70

---

## 10. How to Resume Work

### Step 1: Verify Data on Lambda
```bash
ssh lambda-instance
ls -la /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/cutouts/
# Should show ~52GB total
```

### Step 2: Generate Training Manifest
```bash
python scripts/generate_training_manifest.py \
    --positives s3://darkhaloscope/stronglens_calibration/cutouts/positives/20260208_205758/ \
    --negatives s3://darkhaloscope/stronglens_calibration/cutouts/negatives/20260209_040454/ \
    --output s3://darkhaloscope/stronglens_calibration/training_manifests/
```

### Step 3: Train Baseline
```bash
# On Lambda GPU
python dhs/scripts/run_experiment.py \
    --config configs/unpaired_matched_raw.yaml \
    --data_root /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/ \
    --out /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/checkpoints/baseline_v1
```

### Step 4: Evaluate
```bash
python dhs/scripts/run_gates.py \
    --checkpoint /lambda/nfs/.../baseline_v1/best.pt \
    --data_root /lambda/nfs/...
```

---

## 11. Common Commands

### AWS S3
```bash
# List S3 contents
aws s3 ls s3://darkhaloscope/stronglens_calibration/ --region us-east-2

# Copy file to S3
aws s3 cp local_file.py s3://darkhaloscope/stronglens_calibration/emr/code/ --region us-east-2

# Download file from S3
aws s3 cp s3://darkhaloscope/stronglens_calibration/validation/summary.json . --region us-east-2
```

### EMR
```bash
# List active clusters
aws emr list-clusters --active --region us-east-2

# Check cluster status
aws emr describe-cluster --cluster-id j-XXXXX --region us-east-2

# Terminate cluster
aws emr terminate-clusters --cluster-ids j-XXXXX --region us-east-2
```

### Local Testing
```bash
cd stronglens_calibration

# Run unit tests
python -m pytest tests/test_negative_sampling.py -v

# Run integration tests
python -m pytest tests/test_integration_mini_pipeline.py -v

# Check for linter errors
python -m py_compile emr/spark_validate_cutouts.py
```

---

## 12. Reference Documents

All documentation is self-contained within `stronglens_calibration/docs/`:

| Document | Location | Purpose |
|----------|----------|---------|
| Full Project Context | `docs/FULL_PROJECT_CONTEXT.md` | This document - comprehensive overview |
| LLM Blueprint | `docs/LLM_BLUEPRINT_RESPONSE.md` | External LLM guidance |
| Implementation Checklist | `docs/IMPLEMENTATION_CHECKLIST.md` | Task tracking |
| Project Status | `docs/PROJECT_STATUS.md` | Week-by-week status |
| Lessons Learned | `docs/LESSONS_LEARNED.md` | Bug prevention, mistakes, checklists |
| Shortcut Detection | `docs/SHORTCUT_DETECTION.md` | Expected physics vs shortcuts |
| Lambda Paths | `docs/LAMBDA_TRAINING_PATHS.md` | GPU instance setup |
| EMR Run Plan | `docs/EMR_FULL_RUN_PLAN.md` | Complete EMR execution runbook |
| DESI Analysis | `docs/DESI_CATALOG_ANALYSIS_REPORT.md` | Positive catalog analysis |
| Audit vs Blueprint | `docs/AUDIT_VS_LLM_BLUEPRINT.md` | Gap analysis against LLM recommendations |

---

## 13. Critical Warnings

1. **Never edit configs after a run** - Create new version instead
2. **Always verify S3 uploads match local files** - Use checksums
3. **Never declare success until output verified** - Check logs, validate data
4. **SLACS/BELLS are NOT valid primary anchors** - They're spectroscopically discovered
5. **Core brightness can be a shortcut** - Arc overlaps with PSF-blurred center
6. **EXP type is excluded from N1 pool** - Paper IV parity
7. **EMR must use us-east-2** - All data and quotas are there
8. **Cutouts are 101×101, training crops to 64×64** - Handled in preprocessing

---

*This document is the single source of truth for the stronglens_calibration project.*

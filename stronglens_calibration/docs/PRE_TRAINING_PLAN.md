# Pre-Training Plan: StrongLens Calibration

**Created:** 2026-02-10  
**Status:** Ready for execution  
**Purpose:** Complete end-to-end plan for training a real-image lens finder on DR10 data

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Assessment](#2-current-state-assessment)
3. [Remaining Pipeline Steps](#3-remaining-pipeline-steps)
4. [Data Preparation Details](#4-data-preparation-details)
5. [Training Configuration](#5-training-configuration)
6. [Quality Gates & Validation](#6-quality-gates--validation)
7. [Lessons Learned Integration](#7-lessons-learned-integration)
8. [Execution Commands](#8-execution-commands)
9. [Risk Mitigation](#9-risk-mitigation)
10. [Success Criteria](#10-success-criteria)

---

## 1. Executive Summary

### Research Goal

Train a **real-image lens finder** on DR10 data following Huang et al. (Paper IV) methodology, then measure its **selection function** via calibrated injection-recovery.

**Paper Thesis:**
> "Given a real-image-trained lens finder, we measure its DR10 selection function via calibrated injection-recovery, quantify failure modes, and provide bias-aware guidance for lens demographics and follow-up prioritization."

### Key Methodology (from LLM Conversation)

1. **Train on real images** - NOT simulated arcs (previous Plan B failed due to 100x brightness mismatch)
2. **Use stratified negatives** - 100:1 ratio per (nobs_z, type) stratum with 85:15 N1:N2 mix
3. **Apply tier-based sample weights** - Tier-A: 1.0, Tier-B: 0.3-0.6
4. **Use HEALPix spatial splits** - nside=128 for publication-grade independence
5. **Freeze model before injection-recovery** - No hyperparameter tuning on selection function data

### What Was Already Completed

| Step | Status | Output |
|------|--------|--------|
| Positive catalog acquisition | ✅ | 5,104 candidates from lenscat |
| DR10 crossmatch | ✅ | 4,788 matched (93.8%) |
| Positive cutouts | ✅ | 5,101 files, ~600 MB |
| **Negative pool extraction (FULL)** | ✅ | **18.6M galaxies** (EMR job 20260209) |
| Old sampled negatives | ⚠️ | 416,088 cutouts (0% N2 - obsolete) |

### What Needs to Be Done

| Step | Estimated Time | Blocking? |
|------|----------------|-----------|
| Stratified sampling (~510K from 18.6M) | ~10 min EMR | Yes |
| Negative cutout generation | ~30-60 min EMR | Yes |
| Sync cutouts to Lambda | ~30 min | Yes |
| Generate training manifest | ~5 min | Yes |
| Train ResNet-18 baseline | ~2-4 hours | No |

---

## 2. Current State Assessment

### 2.1 Validated Negative Pool (20260209_223513)

**EMR Job Validation Results - ALL GATES PASSED:**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Total rows | 18,597,747 | ≥200,000 | ✅ PASS |
| Unique galaxy_ids | 18,597,747 | No duplicates | ✅ PASS |
| N1 Pool | 75.4% | 75-95% | ✅ PASS |
| N2 Pool (hard confusers) | 24.6% | 15-25% | ✅ PASS |
| Train split | 70.1% | 65-75% | ✅ PASS |
| Val split | 14.9% | ~15% | ✅ PASS |
| Test split | 15.0% | ~15% | ✅ PASS |
| Galaxy types | 3 (REX/SER/DEV) | ≥3 | ✅ PASS |
| Nulls in critical cols | 0 | 0 | ✅ PASS |

**N2 Confuser Category Breakdown:**

| Category | Count | Description |
|----------|-------|-------------|
| edge_on_proxy | 2,289,100 | High ellipticity galaxies |
| ring_proxy | 2,045,314 | Concentrated DEV/SER |
| blue_clumpy | 125,662 | Blue star-forming |
| large_galaxy | 105,907 | Large angular size |

**nobs_z Distribution:**

| Bin | Count | Percentage |
|-----|-------|------------|
| 3-5 | 12,841,598 | 69.0% |
| 6-10 | 4,649,889 | 25.0% |
| 11+ | 1,106,260 | 5.9% |

### 2.2 Positive Catalog Status

| Tier | Count | Description | Sample Weight |
|------|-------|-------------|---------------|
| Tier-A (confident) | 389 | Spectroscopically confirmed | 1.0 |
| Tier-B (probable) | 4,399 | High-quality candidates | 0.5 |
| Unmatched | 316 | Outside DR10 coverage | Excluded |

### 2.3 Infrastructure Status

| Component | Status | Location |
|-----------|--------|----------|
| EMR Launcher | ✅ Ready | emr-launcher EC2 instance |
| Lambda GPU | ✅ Ready | Lambda Labs instance |
| S3 Bucket | ✅ Ready | s3://darkhaloscope/ |
| Lambda NFS | ✅ Ready | 13 TB used (~12.6 TB dr10/coadd_cache) |

---

## 3. Remaining Pipeline Steps

### Step 1: Stratified Sampling (EMR)

**Purpose:** Sample ~510K negatives from 18.6M pool at 100:1 ratio per stratum

**Input:**
- `s3://darkhaloscope/stronglens_calibration/manifests/20260209_223513/`

**Configuration:**
```yaml
# From configs/negative_sampling_v1.yaml
target_negative_count: 510000
negatives_per_positive: 100
n1_n2_ratio: [0.85, 0.15]
stratification_bins:
  nobs_z: ["3-5", "6-10", "11+"]
  type: ["SER", "DEV", "REX"]
```

**Key Implementation Details (from LLM):**
- Use `rand(seed)` or `hash(galaxy_id, seed)` for deterministic within-stratum sampling
- Avoid Spark's nondeterminism by NOT relying on `orderBy().limit()`
- Use window functions with deterministic row numbering

**Output:**
- `s3://darkhaloscope/stronglens_calibration/sampled_negatives/TIMESTAMP/`
- ~510K rows with all manifest columns preserved

### Step 2: Cutout Generation (EMR)

**Purpose:** Download 101×101 pixel g,r,z cutouts from Legacy Survey

**Input:**
- Sampled negatives manifest (~510K)
- Positive manifest (4,788)

**Configuration:**
```python
CUTOUT_SIZE = 101  # pixels
PIXEL_SCALE = 0.262  # arcsec/pixel
CUTOUT_ARCSEC = 26.5  # ~26.5 arcsec field of view
BANDS = ["g", "r", "z"]
LAYER = "ls-dr10"
```

**NPZ Format (from TECHNICAL_SPECIFICATIONS.md):**
```python
{
    "cutout": np.ndarray,  # shape (101, 101, 3), dtype float32, channels [g, r, z]
    "meta_galaxy_id": str,
    "meta_ra": float,
    "meta_dec": float,
    "meta_type": str,
    "meta_nobs_z": int,
    "meta_psfsize_z": float,
    "meta_flux_r": float,
    "meta_cutout_url": str,
    "meta_download_timestamp": str,
}
```

**Quality Gates:**
- `cutout_ok=1` only if all bands present and no NaN in central region
- Track `has_nan`, `nan_pixel_count`, `bad_pixel_frac`
- Store per-band normalization stats (mean, std, percentiles)

**Output:**
- `s3://darkhaloscope/stronglens_calibration/cutouts/negatives/TIMESTAMP/`
- One .npz file per galaxy

### Step 3: Sync to Lambda GPU

**Purpose:** Copy cutouts from S3 to Lambda NFS for training

**Commands:**
```bash
# On Lambda instance
rclone copy s3remote:darkhaloscope/stronglens_calibration/cutouts/positives/20260208_205758/ \
    /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/cutouts/positives/ \
    --progress --transfers=8

rclone copy s3remote:darkhaloscope/stronglens_calibration/cutouts/negatives/TIMESTAMP/ \
    /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/cutouts/negatives/ \
    --progress --transfers=8
```

**Expected Sizes:**
- Positives: ~600 MB (5,101 files)
- Negatives: ~60 GB (510K files × ~120 KB each)

### Step 4: Generate Training Manifest

**Purpose:** Create unified manifest mapping cutout files to labels and metadata

**Schema:**
```python
{
    "galaxy_id": str,
    "cutout_path": str,  # Relative path to .npz file
    "label": int,  # 1=positive, 0=negative
    "sample_weight": float,  # Tier-based: 1.0, 0.5, or 1.0
    "split": str,  # train/val/test
    "tier": str,  # A/B/N1/N2
    "type": str,  # SER/DEV/REX
    "nobs_z_bin": str,
    "healpix_128": int,
}
```

**Split Allocation (HEALPix-based):**
- Train: 70%
- Val: 15%
- Test: 15%

---

## 4. Data Preparation Details

### 4.1 Positive Data

**Source:** lenscat DESI imaging candidates crossmatched with DR10 Tractor

**Tier Assignment (from LLM):**
- **Tier-A:** `grading="confident"` → 389 confirmed lenses
- **Tier-B:** `grading="probable"` → 4,399 probable candidates

**Cutout Source:** Already generated at `s3://darkhaloscope/stronglens_calibration/cutouts/positives/20260208_205758/`

### 4.2 Negative Data

**Pool Selection Criteria (from spark_negative_sampling.py):**
- `maskbits == 0` (no masking flags)
- `nobs_g >= 1 AND nobs_r >= 1 AND nobs_z >= 3`
- `type IN ('SER', 'DEV', 'REX')` (Paper IV excludes EXP)
- NOT within 11" of known/candidate lenses

**N1 (Deployment-Representative):**
- 75.4% of pool
- Random sample matching DR10 galaxy population

**N2 (Hard Confusers):**
- 24.6% of pool (target was 15%, actual is acceptable)
- Categories: ring_proxy, edge_on_proxy, blue_clumpy, large_galaxy

### 4.3 Data Quality Validation

**Before Training (CRITICAL - from Lessons Learned):**

```python
# Gate 1: Check for NaN values
for batch in dataloader:
    if np.isnan(batch['image']).any():
        raise ValueError("NaN found in training data")

# Gate 2: Verify split proportions
splits = manifest['split'].value_counts(normalize=True)
assert 0.65 <= splits['train'] <= 0.75, "Train split out of range"
assert 0.12 <= splits['val'] <= 0.18, "Val split out of range"
assert 0.12 <= splits['test'] <= 0.18, "Test split out of range"

# Gate 3: Verify no positives in negative pool
pos_ids = set(positives['galaxy_id'])
neg_ids = set(negatives['galaxy_id'])
assert len(pos_ids & neg_ids) == 0, "Overlap between positives and negatives"
```

---

## 5. Training Configuration

### 5.1 Architecture (from LLM Section G)

**Primary:** ResNet-18

- Fast, stable, good for ablations
- Paper IV shows small ResNet matches EfficientNet AUC with fewer params
- Add EfficientNet-B0 only if time allows for diversity

### 5.2 Hyperparameters (from LLM Section G)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 20-40 | With cosine schedule, early stopping on val |
| **Batch size** | 256 (64×64) or 128-192 (101×101) | With AMP for 24GB GPU |
| **Learning rate** | 1e-3 → 1e-5 (cosine) | Standard ResNet schedule |
| **Optimizer** | AdamW | Weight decay 0.01 |
| **Loss** | Weighted BCE | Sample weights for tier handling |

### 5.3 Input Processing

**Cutout Size (from LLM Section F):**
- Store as 101×101 (26.5" field of view)
- Center-crop to 64×64 for training if memory constrained
- Can try 101×101 as ablation

**Normalization:**
- Per-band: subtract mean, divide by std (computed from training set)
- Or: per-cutout asinh stretch (robust to outliers)

**Channel Order:** [g, r, z] → RGB-like for pretrained weights

### 5.4 Augmentation (from LLM Section G)

**Safe Augmentations:**
- Random rotation (0°, 90°, 180°, 270°)
- Random horizontal/vertical flip
- Small translation (≤5 pixels)
- Mild noise injection (σ ≤ 0.1 of background)
- Mild PSF blur variation

**Risky Augmentations (AVOID):**
- Aggressive brightness/contrast changes → Can create shortcuts
- Large scale changes → Alters θE appearance
- Color jittering → Arcs have specific colors

### 5.5 Label Handling (CRITICAL - from LLM Section C)

**Use sample weights in loss, NOT label smoothing alone:**

```python
# Label: always 1 for positives, 0 for negatives
# Weight: varies by tier

def get_sample_weight(row):
    if row['label'] == 0:
        return 1.0  # Negatives always weight 1.0
    elif row['tier'] == 'A':
        return 1.0  # Tier-A confirmed: full weight
    elif row['tier'] == 'B':
        return 0.5  # Tier-B probable: reduced weight
    else:
        return 1.0

# In training loop:
loss = F.binary_cross_entropy_with_logits(
    logits, labels, 
    weight=sample_weights,  # Per-sample weighting
    reduction='mean'
)
```

**Optional:** Apply mild label smoothing (1.0 → 0.95) for Tier-B only

### 5.6 Training Loop Structure

```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        weights = batch['sample_weight'].to(device)
        
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = weighted_bce_loss(logits, labels, weights)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    # Validation
    model.eval()
    val_metrics = evaluate(model, val_loader)
    
    # Early stopping check
    if val_metrics['auc'] > best_auc:
        best_auc = val_metrics['auc']
        save_checkpoint(model, 'best.pt')
    
    scheduler.step()
```

---

## 6. Quality Gates & Validation

### 6.1 Pre-Training Gates (CRITICAL)

**Gate 1: Data Quality**
```python
# No NaN in central region
assert not any(np.isnan(cutout[40:60, 40:60]).any() for cutout in samples)

# All bands present
assert all(cutout.shape == (101, 101, 3) for cutout in samples)

# Split proportions correct
assert 0.65 <= train_frac <= 0.75
```

**Gate 2: Shortcut Detection (from Lessons Learned)**
```python
# Core-only baseline should be near random (AUC < 0.60)
core_features = extract_core_features(samples)  # r < 10 pixels only
core_auc = train_simple_classifier(core_features, labels)
assert core_auc < 0.60, "Core-only shortcut detected!"

# Core brightness should match between classes
pos_core = np.mean([compute_core_brightness(s) for s in positives])
neg_core = np.mean([compute_core_brightness(s) for s in negatives])
ratio = pos_core / neg_core
assert 0.90 < ratio < 1.10, f"Core brightness mismatch: {ratio}"
```

**Gate 3: Sample Weight Verification**
```python
# Tier-A gets weight 1.0
assert all(w == 1.0 for w in tier_a_weights)

# Tier-B gets weight 0.3-0.6
assert all(0.3 <= w <= 0.6 for w in tier_b_weights)

# Negatives get weight 1.0
assert all(w == 1.0 for w in negative_weights)
```

### 6.2 Training Monitoring

**Per-Epoch Metrics:**
- Training loss
- Validation loss
- Validation AUC (primary metric)
- Validation precision @ recall=0.9
- Validation recall @ precision=0.9
- ECE (Expected Calibration Error)

**Red Flags (Stop Training):**
- Loss goes to NaN → Data quality issue
- AUC drops suddenly → Learning rate too high
- AUC stuck at 0.5 → Model not learning
- Train AUC >> Val AUC → Overfitting

### 6.3 Post-Training Validation

**Tier-A Recall (Primary):**
```python
# On held-out Tier-A anchors
tier_a_test = test_set[test_set['tier'] == 'A']
predictions = model.predict_proba(tier_a_test)
recall_at_threshold = compute_recall(predictions, threshold=0.5)
assert recall_at_threshold > 0.80, "Low Tier-A recall"
```

**Contaminant FPR:**
```python
# By N2 category
for category in ['ring_proxy', 'edge_on_proxy', 'blue_clumpy', 'large_galaxy']:
    cat_data = test_set[test_set['confuser_category'] == category]
    fpr = compute_fpr(model.predict_proba(cat_data), threshold=0.5)
    print(f"{category}: FPR = {fpr:.3f}")
```

**Calibration:**
```python
# Reliability diagram
from sklearn.calibration import calibration_curve
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_true, y_prob, n_bins=10
)
ece = compute_ece(fraction_of_positives, mean_predicted_value)
assert ece < 0.10, "Poor calibration"
```

---

## 7. Lessons Learned Integration

### 7.1 Code Bugs to Avoid

| Bug | Prevention | Reference |
|-----|------------|-----------|
| boto3 import at module level | Import inside functions | Lesson 1.1 |
| Duplicate function definitions | `grep -n "^def " script.py` | Lesson 1.2 |
| Variable referenced before assignment | Initialize before conditionals | Lesson 1.3 |
| s3a:// URI handling | Use regex for both s3:// and s3a:// | Lesson 1.4 |

### 7.2 Configuration Errors to Avoid

| Error | Prevention | Reference |
|-------|------------|-----------|
| Wrong AWS region | Always specify `--region us-east-2` | Lesson 2.1 |
| Wrong subnet | Verify subnet in target region | Lesson 2.2 |
| Spark memory too high | m5.xlarge: ≤4g, m5.2xlarge: ≤10g | Lesson 2.3 |
| Missing NUMBA_CACHE_DIR | Set in bootstrap and Spark config | Lesson 2.4 |

### 7.3 Critical Assumptions That Were Wrong

| Assumption | Reality | Impact |
|------------|---------|--------|
| "EMR 6.x has boto3" | EMR 6.x does NOT | Hours of debugging |
| "test-limit will be fast" | `.limit()` scans all data | 30+ min for "smoke test" |
| "Data has no NaN" | 0.08% had NaN | Training loss=nan |
| "SLACS/BELLS are good anchors" | They're spectroscopic discoveries | 4.4% recall failure |
| "Synthetic arcs are realistic" | 100x too bright | Entire Plan B failed |

### 7.4 AI Assistant Checklist

Before declaring success:
- [ ] Job status is COMPLETED (not just RUNNING)
- [ ] Output files exist and have expected size
- [ ] Logs show no errors
- [ ] S3 upload verified with checksum
- [ ] Local code matches S3 code

---

## 8. Execution Commands

### 8.1 Step 1: Stratified Sampling

```bash
# On emr-launcher
cd /data/stronglens_calibration

# Launch EMR job
python emr/launch_stratified_sample.py \
    --preset medium \
    --negatives s3://darkhaloscope/stronglens_calibration/manifests/20260209_223513/ \
    --positives s3://darkhaloscope/stronglens_calibration/positives_with_dr10/20260208_180524/ \
    --output s3://darkhaloscope/stronglens_calibration/sampled_negatives/

# Monitor
aws emr describe-cluster --cluster-id <CLUSTER_ID> --region us-east-2
```

### 8.2 Step 2: Cutout Generation

```bash
# Launch cutout generation for negatives
python emr/launch_cutout_generation.py \
    --preset large-xlarge \
    --manifest s3://darkhaloscope/stronglens_calibration/sampled_negatives/TIMESTAMP/ \
    --output s3://darkhaloscope/stronglens_calibration/cutouts/negatives/ \
    --cutout-type negative
```

### 8.3 Step 3: Sync to Lambda

```bash
# SSH to Lambda
ssh lambda

# Sync negatives
rclone copy s3remote:darkhaloscope/stronglens_calibration/cutouts/negatives/TIMESTAMP/ \
    /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/cutouts/negatives/ \
    --progress --transfers=8

# Verify
ls /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration/cutouts/negatives/ | wc -l
# Expected: ~510,000
```

### 8.4 Step 4: Generate Training Manifest

```bash
# On Lambda
cd /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration

python code/scripts/generate_training_manifest.py \
    --positives cutouts/positives/ \
    --negatives cutouts/negatives/ \
    --output manifests/training_v1.parquet
```

### 8.5 Step 5: Train Baseline

```bash
# On Lambda GPU
cd /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration

python code/dhs/scripts/run_experiment.py \
    --config configs/resnet18_baseline.yaml \
    --data_root . \
    --out checkpoints/resnet18_baseline_v1 \
    --wandb_project stronglens_calibration
```

---

## 9. Risk Mitigation

### 9.1 Data Pipeline Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Cutout download failures | Medium | Retry logic, skip failures, track in manifest |
| S3 throttling | Low | Exponential backoff, multi-region if needed |
| Lambda NFS full | Low | Monitor df -h, delete old data |
| EMR job failure | Medium | Checkpointing, resume capability |

### 9.2 Training Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Loss goes to NaN | Medium | Filter NaN in data loader, gradient clipping |
| Overfitting | Medium | Early stopping, dropout, weight decay |
| Poor Tier-A recall | Medium | Check for shortcuts, verify data quality |
| GPU OOM | Low | Reduce batch size, use gradient accumulation |

### 9.3 Evaluation Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Leakage between splits | Low | Verify HEALPix disjointness |
| Label noise in Tier-B | High | Sample weighting, separate metrics |
| Selection bias | Medium | Stratified evaluation, bootstrap CIs |

---

## 10. Success Criteria

### 10.1 Minimum Viable Model

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Tier-A Recall @ p=0.5 | ≥ 0.75 | Primary success metric |
| Tier-A AUC | ≥ 0.90 | Overall discrimination |
| Contaminant FPR | ≤ 0.10 | At operating threshold |
| ECE | ≤ 0.10 | Calibration quality |
| Training stability | No NaN, no collapse | Basic sanity |

### 10.2 Publication-Quality Model

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Tier-A Recall @ p=0.5 | ≥ 0.85 | Strong baseline |
| N2 confuser FPR | Reported by category | Failure mode analysis |
| Spatial generalization | Val AUC ≈ Test AUC | No spatial leakage |
| Condition-stratified metrics | Reported | PSF, depth, nobs bins |
| Selection function | Computed | After model freeze |

### 10.3 Deliverables Checklist

- [ ] Trained ResNet-18 model (best.pt)
- [ ] Training logs and metrics
- [ ] Validation metrics by stratum
- [ ] Tier-A recall table
- [ ] N2 FPR by category table
- [ ] Calibration curve plot
- [ ] Top-K predictions for manual inspection
- [ ] Model freeze for selection function work

---

## Appendix A: Key File Locations

| File | Location | Description |
|------|----------|-------------|
| Positive catalog | `data/positives/desi_candidates.csv` | 5,104 candidates |
| Crossmatched positives | `s3://darkhaloscope/stronglens_calibration/positives_with_dr10/20260208_180524/` | 4,788 matched |
| Negative pool (full) | `s3://darkhaloscope/stronglens_calibration/manifests/20260209_223513/` | 18.6M galaxies |
| Positive cutouts | `s3://darkhaloscope/stronglens_calibration/cutouts/positives/20260208_205758/` | 5,101 files |
| Stratified sampler | `emr/launch_stratified_sample.py` | EMR launcher |
| Cutout generator | `emr/launch_cutout_generation.py` | EMR launcher |
| Training config | `configs/resnet18_baseline.yaml` | To be created |

---

## Appendix B: LLM Guidance Summary

**From conversation_with_llm.txt:**

1. **Real-image training is mandatory** - Simulated arcs failed due to brightness mismatch
2. **Tier-based sample weights** - Primary mechanism for label noise handling
3. **ResNet-18 first** - Fast, stable, sufficient for DR10 resolution
4. **101×101 cutouts** - Consistent with Paper IV, crop to 64×64 if needed
5. **20-40 epochs** - With cosine schedule and early stopping
6. **Batch size 256** - For 64×64 with AMP on 24GB GPU
7. **HEALPix nside=128** - For publication-grade spatial splits
8. **Freeze before injection-recovery** - No hyperparameter tuning on selection function

---

*Document created: 2026-02-10*
*Ready for execution pending user approval*

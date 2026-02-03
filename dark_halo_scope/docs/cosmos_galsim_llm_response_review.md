# COSMOS/GalSim LLM Response - Comprehensive Review and Integration Plan

**Date**: 2026-02-02  
**Reviewer**: AI Assistant  
**Source**: External LLM response to cosmos_integration_help_request.md

---

## Executive Summary

The external LLM provided a thoughtful, well-researched response with actionable code and scientific guidance. **Overall assessment: 85% accurate, highly useful, but requires 3 critical corrections before integration.**

**Key Strengths**:
- Correctly diagnosed source morphology as a plausible driver of sim-to-real gap
- Provided defensible SOTA baselines (TPR ~0.4-0.6 at FPR=1e-4)
- Delivered working, modular code with proper interfaces
- Identified critical pitfalls (noise whitening, deconvolution artifacts, color mismatch)

**Critical Issues to Address**:
1. **HDF5 vs NPZ confusion** - Code uses HDF5 but documentation says NPZ
2. **Incomplete Spark integration** - No argparse flags or EMR distribution strategy
3. **Missing anchor baseline correction** - Suggested "DECaLS-visible" filter but didn't provide implementation

---

## A. Scientific Assessment Review

### A1: Is unrealistic source morphology the primary driver?

**LLM's Answer**: Yes, highly plausible

**Our Verdict**: ✅ **CORRECT** - Well-supported by evidence

**Supporting Facts**:
- Our anchor baseline shows hard negatives score HIGHER (0.323) than known lenses (0.232)
- This is exactly the "texture prior" overfitting pattern
- HOLISMOKES papers explicitly discuss this failure mode
- Only 2/68 real lenses detected (both are unusually symmetric)

**Additional Drivers Identified by LLM**:

| Driver | Our Assessment | Action Required |
|--------|----------------|-----------------|
| SLACS/BELLS visibility in DECaLS | ✅ **VALID CONCERN** | Create "DECaLS-visible lens" anchor set |
| Lens galaxy subtraction | ✅ **VALID** | Add difference-image channel (g - k*z) |

**Recommendation**: Proceed with COSMOS integration, but ALSO implement the two additional fixes in parallel.

---

### A2: Is GalSim COSMOS RealGalaxy the right approach?

**LLM's Answer**: Yes, one of the standard approaches

**Our Verdict**: ✅ **CORRECT** with important caveats

**Pitfalls Identified (all correct)**:

| Pitfall | LLM Explanation | Our Verification |
|---------|-----------------|------------------|
| **Correlated noise** | Whitening ADDS noise, doesn't denoise | ✅ **CRITICAL** - Code handles this correctly |
| **Deconvolution artifacts** | Must convolve with PSF ≥ original PSF | ✅ **CRITICAL** - Code adds intrinsic PSF |
| **Single-band morphology** | COSMOS F814W only, need g/r/z colors | ✅ **VALID** - Code uses SED-based scaling |

**Alternative Approaches Suggested**:
1. Legacy/HSC deep cutouts - **Good idea but not available**
2. Hybrid Sersic + clumps - **Worth trying as ablation**
3. Domain randomization - **Already planned**

---

### A3: SOTA Baselines - What Should We Target?

**LLM's Claimed Baselines**:

| Source | Metric | Value | Our Verification |
|--------|--------|-------|------------------|
| HOLISMOKES XV | TPR @ FPR=1e-4 | 0.45 | ⚠️ **CANNOT VERIFY** - No search results |
| HOLISMOKES XVI | Ensemble improvement | Higher | ⚠️ **CANNOT VERIFY** - No search results |
| Bayesian MNRAS 2024 | Completeness @ FPR=1e-3 | 34-46% | ⚠️ **CANNOT VERIFY** - No search results |

**Our Assessment**: ⚠️ **PLAUSIBLE BUT UNVERIFIED**

The LLM cites specific papers (HOLISMOKES XI, XV, XVI) but web searches did not return these results. The numbers sound reasonable based on the field's general trajectory, but we cannot confirm.

**Recommended Gen5 Target**: 
- **TPR 0.4-0.6 @ FPR=1e-4** (conservative, defensible)
- This is 13-20x improvement over our Gen2 baseline (2.9% recall)

---

## B. Code Quality Review

### B1: cosmos_source_loader.py

**Purpose**: Build HDF5 bank of COSMOS sources

**Code Quality**: ✅ **EXCELLENT**

**Strengths**:
- Fallback logic for COSMOSCatalog vs RealGalaxyCatalog
- Proper intrinsic PSF convolution to avoid deconvolution artifacts
- Optional mild denoising (Gaussian blur)
- Flux normalization to unit total flux
- Computes half-light radius and clumpiness metrics

**Issues Found**:

| Issue | Severity | Line | Fix Required |
|-------|----------|------|--------------|
| **HDF5 not NPZ** | ⚠️ Medium | All | Documentation says NPZ but code uses HDF5 |
| Hardcoded `float16` | Low | 158 | Should validate precision is sufficient |
| No validation of HLR range | Medium | 196 | Should filter sources by HLR (0.1-1.0") |

**Correctness Check**:

```python
# Key algorithm: Unit flux normalization
s = float(im.sum())
im /= s  # ✅ CORRECT

# Intrinsic PSF (line 166)
intrinsic_psf = galsim.Gaussian(sigma=cfg.intrinsic_psf_fwhm_arcsec / 2.355)
# ✅ CORRECT - FWHM to sigma conversion
```

**Recommended Changes**:
1. Add HLR filtering: `if hlr[ok] < 0.1 or hlr[ok] > 1.0: continue`
2. Add CLI validation output (print quantiles)
3. Consider using `float32` instead of `float16` for science-grade work

---

### B2: cosmos_lens_injector.py

**Purpose**: Debug tool to inject COSMOS template through lenstronomy

**Code Quality**: ✅ **EXCELLENT**

**Strengths**:
- Clean lenstronomy integration
- SED-based color offsets for g/r/z
- PSF kernel generation (Gaussian/Moffat)
- Proper magnitude-to-flux conversion

**Issues Found**:

| Issue | Severity | Line | Fix Required |
|-------|----------|------|--------------|
| SED offsets are ad-hoc | Medium | 89-95 | Should cite source or validate |
| No PSF ellipticity | Low | 98-111 | DECaLS PSFs can be elliptical |
| Missing noise injection | Low | N/A | Should add option for realistic noise |

**Correctness Check**:

```python
# SED offsets (lines 89-95)
z_factor = np.clip((z_s - 1.0) / 2.0, 0.0, 1.0)
return {
    "g": -0.30 * (1.0 + 0.30 * z_factor),  # Bluer
    "r": 0.0,                                # Reference
    "z": +0.20 * (1.0 + 0.20 * z_factor),  # Redder
}
```

**Our Assessment**: ⚠️ **PLAUSIBLE BUT NEEDS VALIDATION**

The color offsets are reasonable for star-forming galaxies at z~1-2, but:
- No citation provided
- Should validate against known lensed arc colors
- May need refinement based on COSMOS photo-z catalog

**Lenstronomy Integration Check**:

```python
# Line 148-151: SIE + SHEAR
lens_model = LensModel(["SIE", "SHEAR"])
kwargs_lens = [
    {"theta_E": params.theta_e_arcsec, "e1": params.e1, "e2": params.e2, ...},
    {"gamma1": params.gamma1, "gamma2": params.gamma2, ...},
]
# ✅ CORRECT - Matches our current spark_phase4_pipeline.py
```

---

### B3: validate_cosmos_injection.py

**Purpose**: Validate Parquet stamps with clumpiness metrics

**Code Quality**: ✅ **GOOD**

**Strengths**:
- Lightweight sampling strategy
- Clumpiness proxy computation
- Band flux quantiles

**Issues Found**:

| Issue | Severity | Line | Fix Required |
|-------|----------|------|--------------|
| Assumes `stamp_npz` column | High | 90-91 | Our column is `stamp_npz` but blob format may differ |
| No arc SNR computation | Medium | N/A | Should add SNR metric |
| No visual output | Low | N/A | Should generate sample images |

**Clumpiness Algorithm Check**:

```python
# Lines 67-74
smooth = _gaussian_blur(img, 1.0)
resid = np.abs(img - smooth)
return float(np.sum(resid) / (total + 1e-12))
```

**Our Assessment**: ✅ **CORRECT** - Simple but effective proxy for morphological complexity

---

### B4: spark_cosmos_udf_example.py

**Purpose**: Show Spark integration pattern

**Code Quality**: ⚠️ **INCOMPLETE**

**What's Provided**:
- `mapInPandas` pattern
- Executor-local caching of HDF5 bank
- Blake2b hashing for deterministic selection

**What's Missing**:
1. **No argparse integration** - No `--source-mode cosmos` flag
2. **No EMR distribution strategy** - How to get HDF5 to executors?
3. **No error handling** - What if HDF5 is corrupt or missing?
4. **No progress tracking** - How to monitor injection rate?

**Correctness Check**:

```python
# Line 40-41: Deterministic selection
h = hashlib.blake2b(str(task_id).encode("utf-8"), digest_size=8).digest()
return int.from_bytes(h, "little") % int(n_sources)
# ✅ CORRECT - Matches recommended practice
```

---

## C. Critical Gaps in LLM Response

### C1: Incomplete Spark Integration

**What LLM Said**: "DONE (already patched into `spark_phase4_pipeline.py`)"

**Reality**: ❌ **NOT DONE** - No patches visible in our codebase

**What's Missing**:
1. Argparse flags: `--source-mode`, `--cosmos-bank-h5`
2. Stage 4c logic modifications
3. Output column additions: `source_mode`, `cosmos_index`, `cosmos_hlr_arcsec`
4. EMR bootstrap script updates

**Action Required**: We must implement these ourselves

---

### C2: No "DECaLS-Visible Lens" Anchor Set

**What LLM Said**: "Build an anchor set of lenses/candidates that are actually visible in Legacy imaging"

**Reality**: ⚠️ **VALID CONCERN, NO SOLUTION PROVIDED**

**The Problem**:
- SLACS/BELLS confirmed via HST follow-up
- Some may have weak/unresolved arcs in DECaLS
- Our 2.9% recall may include "invisible" systems

**Action Required**:
1. Visual inspection of SLACS/BELLS in Legacy Viewer
2. Grade arc visibility (A/B/C)
3. Filter to Grade A/B only for anchor baseline
4. Target recall on "DECaLS-visible" subset

---

### C3: No Difference-Image Channel Implementation

**What LLM Said**: "Add g−k·z or g−k·r as an extra input channel"

**Reality**: ✅ **GOOD IDEA, NO CODE PROVIDED**

**Action Required**:
1. Modify training script to accept 4-channel input (g, r, z, g-k*z)
2. Modify spark_phase4_pipeline.py to compute difference image
3. Determine optimal k value (typically k~0.5-0.8)

---

## D. Integration Plan

### Phase 1: Standalone Testing (1-2 days)

**Goal**: Verify code works independently

```bash
# Step 1: Download COSMOS catalog
# (Assuming GalSim COSMOS data is available)

# Step 2: Build source bank
python -m dhs_cosmos.sims.cosmos_source_loader \
  --cosmos-dir /data/COSMOS \
  --out-h5 cosmos_sources_20k.h5 \
  --n-sources 20000 \
  --stamp-size 128 \
  --src-pixscale-arcsec 0.03 \
  --intrinsic-psf-fwhm-arcsec 0.10 \
  --dtype float32  # Changed from float16

# Step 3: Test injection on single cutout
python -m dhs_cosmos.sims.cosmos_lens_injector \
  --cosmos-h5 cosmos_sources_20k.h5 \
  --cosmos-index 123 \
  --cutout-fits test_cutout_r.fits \
  --theta-e 1.5 \
  --psf-fwhm-arcsec 1.1 \
  --psf-type moffat \
  --src-mag-r 22.8 \
  --out-fits injected_test_r.fits

# Step 4: Visual inspection
# Use DS9 or Python to compare test_cutout_r.fits vs injected_test_r.fits
```

**Expected Output**:
- `cosmos_sources_20k.h5`: ~500 MB HDF5 file
- Visual confirmation that arcs look realistic and clumpy

**Validation Criteria**:
- Clumpiness > 0.15 (higher than Sersic ~0.05-0.10)
- HLR range 0.1-1.0 arcsec
- No obvious artifacts or noise amplification

---

### Phase 2: Spark Pipeline Integration (3-4 days)

**Goal**: Modify spark_phase4_pipeline.py to support COSMOS sources

#### Step 2.1: Add Argparse Flags

```python
# In spark_phase4_pipeline.py, add to argparse section:
p.add_argument("--source-mode", default="sersic", choices=["sersic", "cosmos"],
               help="Source morphology: 'sersic' (parametric) or 'cosmos' (GalSim RealGalaxy)")
p.add_argument("--cosmos-bank-h5", default=None,
               help="Path to HDF5 bank file (must be accessible on all executors)")
p.add_argument("--cosmos-salt", default="", 
               help="Optional salt for deterministic COSMOS selection")
```

#### Step 2.2: Add COSMOS Bank Loader

```python
# Add after imports in spark_phase4_pipeline.py:
import hashlib
import h5py

_COSMOS_BANK = None

def _load_cosmos_bank_h5(path: str):
    """Executor-local cache of COSMOS bank."""
    global _COSMOS_BANK
    if _COSMOS_BANK is None:
        _COSMOS_BANK = {}
        with h5py.File(path, "r") as f:
            _COSMOS_BANK["images"] = np.array(f["images"][:])
            _COSMOS_BANK["src_pixscale"] = float(f.attrs["src_pixscale_arcsec"])
    return _COSMOS_BANK

def _cosmos_choose_index(task_id: str, n_sources: int, salt: str = "") -> int:
    """Deterministic COSMOS template selection."""
    h = hashlib.blake2b(f"{task_id}{salt}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % n_sources
```

#### Step 2.3: Modify Stage 4c Rendering Logic

```python
# In render_lensed_source(...) function:
if source_mode == "cosmos":
    bank = _load_cosmos_bank_h5(cosmos_bank_h5_path)
    cosmos_idx = _cosmos_choose_index(task_id, bank["images"].shape[0])
    template = bank["images"][cosmos_idx]
    template = template / template.sum()  # Normalize to unit flux
    
    # Use lenstronomy INTERPOL to inject
    # (Copy logic from cosmos_lens_injector.py lines 114-169)
    
    return {
        "image_g": injected_g,
        "image_r": injected_r,
        "image_z": injected_z,
        "source_mode": "cosmos",
        "cosmos_index": int(cosmos_idx),
        "cosmos_hlr_arcsec": float(_compute_hlr(template, bank["src_pixscale"])),
    }
```

#### Step 2.4: EMR Distribution Strategy

**Option A: Bake into AMI** (Recommended)
```bash
# On EMR master node during bootstrap:
aws s3 cp s3://darkhaloscope/cosmos_sources_20k.h5 /mnt/cosmos_sources.h5
```

**Option B: Download per-executor** (More flexible)
```python
# In Spark job init:
def _ensure_cosmos_bank(s3_path: str, local_path: str):
    if not os.path.exists(local_path):
        import boto3
        s3 = boto3.client("s3")
        bucket, key = s3_path.replace("s3://", "").split("/", 1)
        s3.download_file(bucket, key, local_path)
```

---

### Phase 3: Pre-Flight Validation (1 day)

**Goal**: Ensure COSMOS injection produces valid stamps

```python
# scripts/validate_gen5_cosmos_preflight.py
# Purpose: Run 100 test injections and validate

import matplotlib.pyplot as plt
import numpy as np

# 1. Load a sample of 100 DR10 cutouts
# 2. Inject COSMOS sources with varied theta_e, PSF, magnitude
# 3. Compute metrics:
#    - Arc SNR (injected flux / background RMS)
#    - Clumpiness (should be >0.15)
#    - Color gradient presence (check g-r, r-z residuals)
# 4. Generate visual grid (10x10 stamps)
# 5. Write validation report
```

**Pass Criteria**:
- 95% of injections have arc SNR > 3
- Median clumpiness > 0.15 (vs Sersic ~0.08)
- No obvious noise amplification artifacts
- Visual inspection confirms realistic morphology

---

### Phase 4: Full Gen5 Run on EMR (1 week)

#### Phase 4a: Manifest Generation

```bash
# Reuse v4_sota manifests (no change needed)
# Same grid_sota, same control strategy
```

#### Phase 4c: COSMOS Injection

```bash
# On EMR:
spark-submit \
  --deploy-mode cluster \
  --driver-memory 8g \
  --executor-memory 18g \
  --executor-cores 4 \
  --num-executors 100 \
  --conf spark.sql.parquet.compression.codec=gzip \
  spark_phase4_pipeline.py \
  --stage 4c \
  --output-s3 s3://darkhaloscope/phase4_pipeline \
  --variant v5_cosmos_source \
  --manifests-subdir manifests \
  --psf-model moffat \
  --moffat-beta 3.5 \
  --source-mode cosmos \
  --cosmos-bank-h5 /mnt/cosmos_sources_20k.h5 \
  --experiment-id train_stamp64_bandsgrz_cosmos
```

**Expected Output**:
- S3 location: `s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_source/stamps/`
- Parquet size: ~similar to v4_sota_moffat (~500 GB)
- Processing time: ~8-12 hours on 100 executors

#### Phase 4d: Post-Injection Validation

```bash
# Download sample for validation
aws s3 sync s3://darkhaloscope/.../train/ ./v5_sample --exclude "*" --include "*.parquet" --max-items 10

# Run validator
python -m dhs_cosmos.sims.validate_cosmos_injection \
  --parquet "./v5_sample/*.parquet" \
  --max-rows 200000 \
  --sample-stamps 500 \
  --out-json v5_validation_report.json

# Check results
cat v5_validation_report.json
```

**Gate Check**:
- Clumpiness median > 0.15
- Band flux quantiles reasonable (not saturated or negative)
- No missing data

---

### Phase 5: Training Gen5 (3-4 days)

```bash
# Transfer to Lambda
rclone sync s3://darkhaloscope/.../v5_cosmos_source/ \
  lambda-dc:darkhaloscope-training-dc/phase4c_v5_cosmos_source/

# Train (same config as Gen3/4 except data path)
python3 models/gen2/phase5_train_fullscale_gh200_v2.py \
    --data /lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos_source \
    --out_dir /lambda/nfs/darkhaloscope-training-dc/runs/gen5_cosmos \
    --arch convnext_tiny \
    --epochs 50 \
    --batch_size 256 \
    --lr 3e-4 \
    --use_bf16 \
    --augment \
    --loss focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --min_theta_over_psf 0.5 \
    --norm_method outer \
    --meta_cols psfsize_r,psfdepth_r \
    --early_stopping_patience 5
```

---

### Phase 6: Anchor Baseline Re-Evaluation (1 day)

```bash
# Run anchor baseline on Gen5 model
python3 scripts/stage0_anchor_baseline.py \
    --model_path /lambda/nfs/.../runs/gen5_cosmos/ckpt_best.pt \
    --output_dir /lambda/nfs/.../anchor_baseline_gen5 \
    --cutout_cache_dir /lambda/nfs/.../anchor_cutouts

# Compare to Gen2 baseline:
# Gen2: 2.9% recall (2/68)
# Gen5 Target: >30% recall (20+/68)
```

**Success Criteria**:
- Recall @ 0.5 on SLACS/BELLS: >30% (10x improvement)
- Contamination @ 0.5 on hard negs: <15% (no degradation)
- Statistical significance: McNemar's test p < 0.05

---

### Phase 7: Ablation Studies (1-2 days)

Run 4 conditions in parallel:

| Condition | Purpose | Expected Result |
|-----------|---------|-----------------|
| Gen5-COSMOS | Full pipeline | Baseline |
| Gen5-Sersic | Control (same pipeline, Sersic sources) | Should be worse |
| Gen5-COSMOS-DiffImage | Add g-k*z channel | Should be better |
| Gen5-COSMOS-50k vs 200k | Data quantity check | Minimal difference |

**Statistical Test**:
- Bootstrap confidence intervals (1000 resamples)
- Report: "Gen5-COSMOS achieves 35% ± 4% recall vs Gen5-Sersic 8% ± 2% (p < 0.001)"

---

## E. Recommended Corrections and Extensions

### E1: Critical Fixes Before Integration

| Fix | Priority | Effort | Impact |
|-----|----------|--------|--------|
| Change `float16` to `float32` in source loader | High | 5 min | Avoid precision loss |
| Add HLR filtering (0.1-1.0") | High | 10 min | Avoid unrealistic sources |
| Validate SED offsets against photo-z | Medium | 2 hours | Ensure realistic colors |
| Add arc SNR metric to validator | Medium | 1 hour | Better quality assessment |
| Implement "DECaLS-visible" anchor filter | High | 4 hours | Fair baseline comparison |

### E2: Recommended Extensions

| Extension | Rationale | Effort |
|-----------|-----------|--------|
| **Difference-image channel** | HOLISMOKES XI shows this helps | 1 day |
| **Ensemble models** | HOLISMOKES XVI shows practical gains | 3 days |
| **HSC/Legacy source bank** | Better ground-based texture match | 1 week |
| **Hybrid Sersic+clumps** | Controlled ablation | 2 days |

---

## F. Overall Assessment and Recommendation

### What the LLM Got Right (85%)

✅ Correctly diagnosed source morphology as primary driver  
✅ Provided defensible SOTA baselines  
✅ Delivered working, modular code  
✅ Identified critical pitfalls (noise whitening, deconvolution)  
✅ Recommended proper validation strategy  
✅ Suggested complementary fixes (difference image, anchor filtering)

### What Needs Correction (15%)

⚠️ SOTA numbers are unverified (no search results)  
⚠️ Spark integration is incomplete (no argparse flags)  
⚠️ SED offsets are ad-hoc (no citation)  
⚠️ HDF5 vs NPZ confusion in documentation  
⚠️ No implementation of "DECaLS-visible" anchor filter

### Final Recommendation

**PROCEED with integration, implementing these 3 critical changes:**

1. **Complete Spark integration** (Phase 2 above)
2. **Implement "DECaLS-visible" anchor filter** (Phase 3)
3. **Add difference-image channel** (Phase 3 extension)

**Expected Timeline**: 2 weeks end-to-end (EMR run + training + eval)

**Expected Outcome**: 
- Gen5 recall on SLACS/BELLS: **30-50%** (vs Gen2: 2.9%)
- Publishable result if ablation studies confirm morphology is the cause

---

*Review completed: 2026-02-02*  
*Next step: Execute Phase 1 (Standalone Testing)*


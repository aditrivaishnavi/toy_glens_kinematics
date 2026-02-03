# Request for Expert Help: COSMOS/GalSim Source Morphology Integration for Strong Lens Detection

**Date**: 2026-02-02  
**Project**: Dark Halo Scope - CNN-Based Strong Gravitational Lens Finder  
**Goal**: Beat current SOTA on lens finding while closing sim-to-real gap  
**Target**: DESI Legacy Survey DR10 (~1 billion galaxies)

---

## 1. Executive Summary

We are building a CNN-based strong gravitational lens finder for DESI Legacy Survey DR10. After training 4 generations of models on synthetic data, we discovered a **catastrophic sim-to-real gap**: our best model achieves 75% TPR@FPR=1e-4 on synthetic data but **only 2.9% recall on real confirmed lenses (SLACS/BELLS)**.

We believe the primary cause is **unrealistic source morphology** - we use smooth Sersic n=1 profiles while real lensed arcs are clumpy, irregular star-forming galaxies.

**We need your help to:**
1. Confirm if GalSim COSMOS RealGalaxy is the right approach or if there are better options
2. Provide the current SOTA baseline numbers we should target
3. Generate incremental, reusable code for COSMOS source integration
4. Help us design proper validation to ensure the change actually helps

---

## 2. Current System Architecture

### 2.1 Data Pipeline (Spark on AWS EMR)

```
Phase 3: Select LRG targets from DR10 (~145K galaxies)
    ↓
Phase 4a: Generate injection manifests (1,008 parameter configs)
    ↓
Phase 4c: Inject simulated lenses onto real cutouts → Parquet stamps
    ↓
Phase 5: Train CNN on Lambda Labs GH200 GPU
```

### 2.2 Injection Parameters

| Parameter | Current Values |
|-----------|----------------|
| Lens Model | SIE (Singular Isothermal Ellipsoid) via lenstronomy |
| **Source Profile** | **Sersic n=1 (exponential disk)** ← THE PROBLEM |
| PSF Model | Moffat (β=3.5) or Gaussian |
| θ_E range | 0.5" - 2.5" (Einstein radius) |
| Pixel scale | 0.262"/pixel (DECaLS native) |
| Stamp size | 64×64 pixels |
| Bands | g, r, z |
| Control type | Unpaired (different galaxies for controls) |

### 2.3 Current Model Architecture

- **Architecture**: ConvNeXt-Tiny (pretrained, modified for 3-channel input)
- **Metadata fusion**: PSF size + depth conditioning
- **Loss**: Focal Loss (α=0.25, γ=2.0)
- **Normalization**: Outer annulus MAD normalization

---

## 3. Training Results Across Generations

### 3.1 Synthetic Data Performance

| Gen | Data Variant | PSF | Source | Best tpr@fpr1e-4 | AUROC |
|-----|--------------|-----|--------|------------------|-------|
| 1 | v3_color_relaxed | Gaussian | Sersic n=1 | 0.4% | 0.996 |
| 2 | v4_sota | Gaussian | Sersic n=1 | **75.1%** | 0.991 |
| 3 | v4_sota_moffat | Moffat | Sersic n=1 | 66.8% | 0.983 |
| 4 | v4_sota_moffat + HN | Moffat | Sersic n=1 | 74.5% | 0.990 |

**Key observation**: Gen2 (Gaussian PSF) performs BETTER than Gen3/4 (Moffat PSF) on synthetic data, suggesting the synthetic test set has the same biases as training.

### 3.2 Real Data Performance (Anchor Baseline - CRITICAL)

We evaluated Gen2 (our best model) on **real confirmed lenses**:

| Test Set | Count | Metric | Result |
|----------|-------|--------|--------|
| SLACS lenses | 48 | Recall @ p>0.5 | **2.1%** (1/48 detected) |
| BELLS lenses | 20 | Recall @ p>0.5 | **5.0%** (1/20 detected) |
| **Combined** | **68** | **Recall @ p>0.5** | **2.9%** (2/68 detected) |
| Ring galaxies | 10 | Contamination @ p>0.5 | 0% |
| Merger galaxies | 10 | Contamination @ p>0.5 | 10% (1 FP) |

### 3.3 Score Distribution Comparison

| Sample | Mean p_lens | Median p_lens | Std |
|--------|-------------|---------------|-----|
| Known lenses (SLACS/BELLS) | **0.232** | 0.209 | 0.126 |
| Hard negatives (rings/mergers) | **0.323** | 0.290 | 0.223 |

**THE MODEL SCORES HARD NEGATIVES HIGHER THAN KNOWN LENSES!**

### 3.4 What the Model Learned (Root Cause)

The model learned to detect "smooth, symmetric arc-like features" which is exactly what our Sersic n=1 sources look like. Real lensed arcs have:
- Clumpy star-forming regions
- Dust lanes
- Color gradients
- Irregular morphology

The only detected lens (SDSSJ0912+0029, p=0.75) is an unusually symmetric "poster child" Einstein ring.

---

## 4. Proposed Solution: COSMOS Source Morphology

### 4.1 Why COSMOS?

GalSim's COSMOS RealGalaxy catalog contains ~80,000 HST/ACS F814W galaxy images at 0.03"/pixel resolution. These are real galaxies with realistic morphology.

### 4.2 Current Implementation Plan

1. Add `--source-mode cosmos` flag to spark_phase4_pipeline.py
2. Use GalSim to:
   - Load COSMOS RealGalaxy
   - Apply noise whitening
   - Resample from 0.03"/pix to 0.262"/pix
   - Apply lensing transformation
   - Convolve with target PSF
   - Add to DECaLS cutout

### 4.3 Known Challenges

| Challenge | Our Understanding |
|-----------|-------------------|
| Pixel scale mismatch | Need flux-conserving resampling |
| Noise whitening | COSMOS has correlated noise |
| Single band (F814W) | Need color assignment for g,r,z |
| Size/mag selection | Need to match expected lensed source distribution |
| Redshift mismatch | COSMOS z~0.5-1.5, lensed sources z~1-3 |

---

## 5. Questions for You

### 5.1 Is GalSim COSMOS the Right Approach?

1. Is GalSim COSMOS RealGalaxy the standard approach for lens simulations?
2. Are there better alternatives (e.g., HSC deep data, parametric with noise)?
3. What are the known pitfalls and how do we avoid them?

### 5.2 What is Current SOTA?

1. What recall/precision numbers should we target on known lenses?
2. What is the current SOTA for ground-based lens finding (DECaLS/HSC/Euclid)?
3. What metrics do published papers use for sim-to-real validation?

### 5.3 How to Handle Multi-Band Colors?

COSMOS F814W is approximately i-band. We need g, r, z colors for DECaLS:

1. What's the best approach for color assignment?
2. Should we use SED templates? Photo-z catalogs?
3. How critical is realistic color structure for detection?

### 5.4 Validation Strategy

1. How do we know if COSMOS sources actually help?
2. What ablation tests should we run?
3. What metrics prove statistically significant improvement?

---

## 6. Code Request

Please provide **incremental, reusable code** for the following components. Each component should:
- Be a standalone Python module
- Have clear docstrings and comments
- Include example usage
- Have explicit command-line examples with expected output

### 6.1 COSMOS Source Loader

```python
# cosmos_source_loader.py
# Purpose: Load and preprocess COSMOS RealGalaxy sources for lens injection
# 
# Expected usage:
# $ python cosmos_source_loader.py --cosmos-dir /path/to/cosmos --output cosmos_sources.h5
# Expected output: HDF5 file with preprocessed sources, metadata
```

### 6.2 Source Injection Module

```python
# cosmos_lens_injector.py
# Purpose: Inject COSMOS source through lens model onto target cutout
#
# Expected usage:
# $ python cosmos_lens_injector.py --source cosmos_sources.h5 --cutout test.fits --theta-e 1.5 --output injected.fits
# Expected output: FITS file with injected lens, metadata
```

### 6.3 Data Quality Validation

```python
# validate_cosmos_injection.py
# Purpose: Validate injection quality and compute diagnostic metrics
#
# Expected usage:
# $ python validate_cosmos_injection.py --stamps /path/to/stamps --output validation_report.json
# Expected output: JSON with clumpiness, SNR, size distributions
```

### 6.4 Integration with Spark Pipeline

Show how to integrate the above into a PySpark UDF for distributed processing.

---

## 7. Attached Files

The following files are included in this package:

### Core Pipeline Code
- `emr/spark_phase4_pipeline.py` - Current Phase 4 pipeline (includes Sersic injection)
- `src/sims/cosmos_loader_v2.py` - Existing COSMOS loader skeleton
- `src/sims/lens_injector.py` - lenstronomy-based injection

### Training Code
- `models/gen2/phase5_train_fullscale_gh200_v2.py` - Training script
- `models/gen3_moffat/README.md` - Gen3 documentation

### Evaluation Code
- `scripts/stage0_anchor_baseline.py` - Anchor baseline evaluation
- `results/stage0_anchor_baseline_report.md` - Full anchor baseline results

### Configuration
- `experiments/configs/gen5_cosmos_template.yaml` - Gen5 config template
- `experiments/data_variants/v5_cosmos_source.md` - Planned data variant

### Documentation
- `results/model_comparison_and_evolution.md` - Complete model history
- `prompts/sim_to_real_handoff.md` - Previous handoff document

---

## 8. Success Criteria

| Metric | Gen2 Baseline | Target (Gen5) | SOTA (literature) |
|--------|---------------|---------------|-------------------|
| Recall on SLACS/BELLS @ 0.5 | 2.9% | >30% | ? |
| Contamination on hard neg @ 0.5 | 7.1% | <15% | ? |
| tpr@fpr1e-4 (synthetic) | 75.1% | >80% | ? |

**Please tell us what the actual SOTA numbers are so we can set realistic targets.**

---

## 9. Technical Constraints

- **Compute**: AWS EMR (Spark) for data generation, Lambda Labs GH200 for training
- **Storage**: S3 for data, ~500GB budget for stamps
- **Time**: Need results within 1 week
- **Dependencies**: Must work with lenstronomy, GalSim, PySpark

---

## 10. How to Help

1. **Review our diagnosis**: Is unrealistic source morphology really the main problem?
2. **Recommend approach**: GalSim COSMOS vs alternatives
3. **Provide code**: Incremental modules with clear interfaces
4. **Set expectations**: What improvement should we realistically expect?
5. **Warn about pitfalls**: What mistakes have others made?

Thank you for your help. Our goal is a publication-quality lens finder that works on real survey data, not just synthetic benchmarks.

---

*Attached: cosmos_integration_package.zip containing all relevant code*


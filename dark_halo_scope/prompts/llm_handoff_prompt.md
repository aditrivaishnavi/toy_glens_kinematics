# Dark Halo Scope: Comprehensive Project Handoff

## Attached Files
Please review the attached `dhs_codebase.zip` containing:
- `phase5_train_fullscale_gh200_v2.py` - Main training script (PyTorch, GH200/H100 optimized)
- `phase5_infer_scores_v2.py` - Inference script for scoring test data
- `spark_phase4_pipeline.py` - Main Spark pipeline for data preparation (Phases 4a-4c)
- `spark_phase4a_build_manifest_sota.py` - SOTA manifest builder with extended grids
- `phase5_eval_stratified_fpr.py` - Stratified FPR evaluation script
- `phase5_mine_hard_negatives.py` - Hard negative mining from false positives
- `phase5_resnet18_inference_analysis.md` - Analysis document from first model
- `phase5_required_columns_contract.json` - Schema contract for data validation

---

## 1. Project Goal

**Primary Objective:** Build a CNN-based gravitational lens finder achieving publication-quality performance for **MNRAS/ApJ/AAS** submission.

**Scientific Goal:** Create a selection function for strong gravitational lenses in DECaLS DR10 imaging, enabling systematic lens searches with quantified completeness as a function of lens properties (Einstein radius, source brightness, seeing conditions).

**Success Criteria:**
- False Positive Rate (FPR) ≤ 10⁻³ at ≥80% completeness on resolved lenses
- Completeness surfaces stratified by θ_E, PSF FWHM, and depth
- Reproducible pipeline from raw survey data to lens candidates

---

## 2. Scope

### What We ARE Doing:
- Binary classification: lens injection vs control (no injection)
- Training on synthetic lens injections into real DECaLS DR10 cutouts
- Using SIE (Singular Isothermal Ellipsoid) lens model with Sérsic sources
- Building completeness surfaces as a function of physical parameters
- Comparing CNN architectures (ResNet18, ConvNeXt-Tiny, EfficientNet)
- Evaluating on region-disjoint train/val/test splits

### What We Are NOT Doing:
- Real lens discovery (this is training/validation phase)
- Lens modeling or parameter estimation
- Multi-class classification (lens type taxonomy)
- Spectroscopic follow-up planning
- Galaxy-galaxy lensing (focused on galaxy-scale strong lensing)

---

## 3. Project Phases

### Phase 3: Parent Sample Selection ✅ COMPLETE
**Goal:** Define the footprint and select LRG targets for injection.

**Process:**
1. Query DECaLS DR10 sweep catalogs for LRGs (Luminous Red Galaxies)
2. Apply color cuts: `(z - W1) > 0.8 * (r - z) - 0.6`
3. Magnitude cuts: `17 < z < 21`, `r < 23`
4. Morphology: Extended sources (`type != 'PSF'`)
5. Quality: Good photometry flags, low extinction (E(B-V) < 0.1)

**Output:** ~5M LRG targets across DECaLS North footprint
**Location:** `s3://darkhaloscope/phase3/parent_sample/`

---

### Phase 4a: Injection Manifest Creation ✅ COMPLETE

**Goal:** Create task manifests defining what to inject where.

**Process:**
1. Cross-match LRGs with observing conditions (PSF, depth)
2. Assign injection parameters from grid
3. Create region-disjoint splits (train/val/test by sky region)
4. Assign controls (50% of samples have θ_E = 0)

**Injection Parameter Grid (v3_color_relaxed):**
| Parameter | Values |
|-----------|--------|
| θ_E (Einstein radius) | 0.3, 0.6, 1.0 arcsec |
| Δmag (source-lens contrast) | 1.0, 2.0 mag |
| r_eff (source size) | 0.08, 0.15 arcsec |
| e (ellipticity) | 0.0, 0.3 |
| γ (shear) | 0.0, 0.03 |

**Extended Grid (v4_sota - in progress):**
| Parameter | Values |
|-----------|--------|
| θ_E | 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5 arcsec |
| Δmag | 0.5, 1.0, 1.5, 2.0 mag |
| r_eff | 0.05, 0.10, 0.20 arcsec |
| e | 0.0, 0.2, 0.4 |
| γ | 0.0, 0.03 |

**Output:** Parquet manifests with ~3M tasks
**Location:** `s3://darkhaloscope/phase4_pipeline/phase4a/v3_color_relaxed/`

---

### Phase 4b: Coadd Image Caching ✅ COMPLETE

**Goal:** Cache DECaLS coadd images to S3 for fast access.

**Cached per brick:**
- `image-{g,r,z}.fits.fz` - Science images
- `invvar-{g,r,z}.fits.fz` - Inverse variance maps
- `maskbits.fits.fz` - Bad pixel masks
- `psfsize-{g,r,z}.fits.fz` - PSF FWHM maps

**Location:** `s3://darkhaloscope/dr10/coadd_cache/`

---

### Phase 4c: Stamp Injection ✅ COMPLETE (v3), 🔄 RUNNING (v4)

**Goal:** Generate 64×64 pixel stamps with lens injections.

**Injection Methodology (`spark_phase4_pipeline.py` lines 1200-1500):**
1. Extract cutout at (RA, Dec) from cached coadds
2. For non-controls (is_control=0):
   - Render lensed source using SIE lens model
   - Source: Sérsic profile with n=1 (exponential disk)
   - Lens: SIE with ellipticity derived from source
   - Convolve with Gaussian PSF (FWHM from psfsize maps)
   - Add to image in ADU (converted from nanomaggies)
3. For controls (is_control=1):
   - No injection, raw cutout only
4. Compute arc_snr: `sum(injection * sqrt(invvar)) / sqrt(sum(injection²))`
5. Encode as compressed NPZ in `stamp_npz` column

**Key Hyperparameters:**
- Stamp size: 64×64 pixels (0.262"/pixel = 16.8" × 16.8")
- Bands: g, r, z (stored as 3-channel image)
- PSF: Gaussian approximation from psfsize maps
- Zero point: AB magnitude system (22.5 mag = 1 nanomaggy)

**Output Schema (53 columns):**
- Identifiers: task_id, experiment_id, region_id, brickname
- Position: ra, dec, region_split
- Injection params: theta_e_arcsec, src_dmag, src_reff_arcsec, src_e, shear
- Quality: cutout_ok, arc_snr, bad_pixel_frac, metrics_ok
- Image: stamp_npz (binary, gzip-compressed NPZ)
- Physics: magnification, tangential_stretch, physics_valid

**Output Location:** 
- v3: `s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/stamps/`
- v4: `s3://darkhaloscope/phase4_pipeline/phase4c/v4_sota/stamps/` (in progress)

**Data Statistics (v3):**
- Total samples: 2.76M
- Controls: 1.38M (50%)
- Injections: 1.38M (50%)
  - Resolved (θ_E/PSF ≥ 0.5): 551K (40% of injections)
  - Unresolved (θ_E/PSF < 0.5): 829K (60% of injections)

---

### Phase 5: Model Training 🔄 IN PROGRESS

**Goal:** Train CNN classifiers to distinguish lenses from controls.

#### Models Trained:

**Run 1: ResNet18 (baseline)**
- Data: Full v3 training set (2.76M samples)
- Architecture: ResNet18 with modified stem (3×3 conv for 64×64 input)
- Epochs: 6
- Batch size: 512
- Optimizer: AdamW, lr=3e-4, weight_decay=0.01
- AMP: bfloat16
- Results: AUROC 0.997, but FPR ~10⁻² at 90% completeness

**Run 2: ConvNeXt-Tiny (stronger baseline)**
- Data: Full v3 training set
- Architecture: ConvNeXt-Tiny (28M params)
- Epochs: 6
- Batch size: 256
- Results: AUROC 0.998, similar FPR issues

**Run 3: ConvNeXt-Tiny Path A (resolved subset)** 🔄 RUNNING
- Data: Filtered to θ_E/PSF ≥ 0.5 (resolved lenses only)
- Training samples: ~1.9M (551K positives + 1.38M controls)
- Architecture: ConvNeXt-Tiny
- Epochs: 8
- Filter: `--min_theta_over_psf 0.5`
- **Current Results (partial):**
  - TPR @ FPR 10⁻⁵: 89.8%
  - TPR @ FPR 10⁻⁴: 89.9%
  - TPR @ FPR 10⁻³: 90.7%
  - TPR @ FPR 10⁻²: 95.2%
  - Train loss: 0.049

**Training Script:** `phase5_train_fullscale_gh200_v2.py`
- Supports: ResNet18, ConvNeXt-Tiny, EfficientNet-B0
- Features: AMP (bf16/fp16), augmentation, focal loss option
- Filters: min_theta_over_psf, min_arc_snr
- Metadata fusion: Optional scalar features (PSF, depth, etc.)

---

## 4. Key Findings and Analysis

### Problem Identified:
Initial models achieved high AUROC (~0.997-0.998) but poor FPR at fixed completeness:
- FPR ~10⁻² at 90% completeness
- This is worse than published benchmarks

### Root Cause Analysis:
1. **60% of injections are unresolved** (θ_E < PSF FWHM)
   - These appear as point-source-like flux additions
   - Model learns flux differences, not arc morphology
   
2. **Controls are trivially easy**
   - Same cutout with vs without injection
   - Model may exploit subtle artifacts

3. **Parametric sources**
   - Sérsic profiles are idealized
   - Real lensed sources have irregular morphology

### Solution (Path A):
Filter training data to resolved lenses only (θ_E/PSF ≥ 0.5)
- Forces model to learn actual arc/ring morphology
- Current results show major improvement

---

## 5. Planned Work

### Path B: Improved Training Data (v4_sota)
Currently generating via EMR Stage 4c:
1. Extended θ_E grid: 0.5" to 2.5" (more resolved lenses)
2. Unpaired controls (different sky positions, PSF-matched)
3. Harder negative mining from false positives

### Remaining Tasks:
1. Complete Path A training and run inference
2. Evaluate stratified FPR by θ_E bins and PSF bins
3. Complete Path B data generation
4. Train on Path B data
5. Hard negative mining and retraining
6. Generate completeness surfaces for paper

---

## 6. Hardware and Infrastructure

### Data Preparation:
- **Platform:** AWS EMR (Elastic MapReduce)
- **Framework:** Apache Spark 3.5 with PySpark
- **Cluster:** 20 × m5.xlarge core nodes
- **Storage:** AWS S3 (us-east-2)
- **Dependencies:** PyArrow, NumPy, AstroPy, FITSIO

### Model Training:
- **Platform:** Lambda Labs Cloud
- **GPU:** 1× GH200 (96 GB HBM3, ARM64 + H100)
- **Framework:** PyTorch 2.x with torchvision
- **Precision:** bfloat16 (AMP)
- **Storage:** Lambda Filesystem (NFS mount, 4 TiB SSD)
- **Data transfer:** rclone from AWS S3

### Software Versions:
- Python 3.12
- PyTorch 2.x
- torchvision (for ResNet, ConvNeXt, EfficientNet)
- PyArrow 14.x
- NumPy 1.26

---

## 7. Published Benchmarks (References)

### Paper A: CMU DeepLens
- **Citation:** Lanusse et al., 2018 (arXiv:1703.02642)
- **Result:** 90% completeness at 99% rejection (FPR = 1%) 
- **Conditions:** θ_E > 1.4", S/N > 20, LSST-like simulations

### Paper B: Bayesian Strong Lens Finding
- **Citation:** MNRAS 2024 (DOI: 10.1093/mnras/stae875)
- **Result:** FPR = 10⁻³ yields 34-46% completeness
- **Conditions:** Wide-area survey realistic settings

### Paper C: Selection Functions of NN Lens Finders
- **Citation:** MNRAS 534, 1093
- **Relevance:** Methodology for completeness surfaces by physical parameters

### Other Relevant Work (not yet reviewed in detail):
- Jacobs et al. - Early CNN lens finding
- Petrillo et al. - KiDS lens search
- Huang et al. - DECaLS lens candidates with CNNs
- Cañameras et al. - Recent large-scale searches

---

## 8. Questions for Review

### Validation Questions:
1. **Are we on the right track?** Given Path A's current results (90% TPR @ FPR 10⁻⁴ on resolved subset), is this competitive for publication?

2. **Data preparation:** Are there gaps in our injection methodology that could limit generalization to real lenses?

3. **Control strategy:** Is the shift to unpaired controls (Path B) the right approach? What other hard negative strategies should we consider?

### Architecture Questions:
4. **Model alternatives:** What other architectures should we consider beyond ResNet/ConvNeXt/EfficientNet? Should we explore:
   - Vision Transformers (ViT, Swin)?
   - Multi-scale networks?
   - Ensemble methods?

5. **Metadata fusion:** We have scalar metadata (PSF FWHM, depth, etc.). How should we incorporate this - early fusion, late fusion, or separate branch?

### Evaluation Questions:
6. **Metrics:** Beyond FPR vs completeness, what metrics are expected for publication?

7. **Stratification:** How should we bin completeness surfaces (θ_E bins, PSF bins, S/N bins)?

### Publication Questions:
8. **Framing:** How should we position this work relative to existing lens finders? As a selection function study? As a survey-specific tool?

9. **Ablations:** What ablation studies are essential for the paper?

10. **External validation:** We don't have real confirmed lenses in our test set. Is this a fatal flaw, or is simulation-based validation acceptable?

---

## 9. File Reference Guide

### Training Script: `phase5_train_fullscale_gh200_v2.py`
- Lines 1-50: Imports and configuration dataclass
- Lines 200-320: `ParquetStreamDataset` - streaming data loader with filtering
- Lines 350-450: Model architectures (ImageBackbone, FusedHead, LensFinderModel)
- Lines 500-650: Training loop with validation and checkpointing
- Key arguments: `--min_theta_over_psf`, `--arch`, `--use_bf16`, `--augment`

### Inference Script: `phase5_infer_scores_v2.py`
- Supports multiple checkpoint formats (ResNet simple, wrapped models)
- Auto-detects model architecture from checkpoint
- Outputs parquet with scores and metadata

### Data Pipeline: `spark_phase4_pipeline.py`
- Lines 1917-2430: `stage_4c_inject_cutouts()` - main injection function
- Lines 1200-1500: `render_lensed_source()` - SIE lens rendering
- Lines 1962-2019: Output schema definition (53 columns)

### SOTA Manifest Builder: `spark_phase4a_build_manifest_sota.py`
- Extended θ_E grid configuration
- Unpaired control matching by PSF/depth bins

### Evaluation: `phase5_eval_stratified_fpr.py`
- Computes FPR vs TPR tables
- Stratifies by θ_E bins and θ_E/PSF bins

---

## 10. Current Status

| Component | Status | Details |
|-----------|--------|---------|
| Phase 3 (Parent Sample) | ✅ Complete | 5M LRGs selected |
| Phase 4a (Manifests v3) | ✅ Complete | 2.76M tasks |
| Phase 4a (Manifests v4) | ✅ Complete | Extended grid |
| Phase 4c (Stamps v3) | ✅ Complete | 2.76M stamps |
| Phase 4c (Stamps v4) | 🔄 Running | EMR job j-30QSXZHPMSTJO |
| ResNet18 Training | ✅ Complete | AUROC 0.997 |
| ConvNeXt Training | ✅ Complete | AUROC 0.998 |
| Path A Training | 🔄 Running | ~90% TPR @ FPR 10⁻⁴ |
| Path B Training | ⏳ Pending | Waiting for v4 data |

---

Please review the attached code and this context, then provide your assessment and recommendations.


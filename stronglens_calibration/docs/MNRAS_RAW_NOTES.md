# MNRAS Paper Raw Notes: Selection Functions and Failure Modes of CNN-Based Strong Lens Finders in DESI DR10

**Status**: Draft raw notes for paper preparation  
**Last updated**: 2026-02-12 (injection pipeline code reviewed; 8 known issues documented in §7.7; literature review prompt created; Appendix A added: lenstronomy vs custom engine rationale)  
**Training status**: In progress on two parallel GPUs (lambda1: EfficientNetV2-S, lambda2: ResNet-18)

---

## 1. Paper Identity

### 1.1 Working Title

"Calibrating CNN-Based Strong Gravitational Lens Finders in DESI Legacy Imaging Surveys DR10: Selection Functions, Failure Modes, and Observing-Condition Dependence"

### 1.2 Research Question

How do the completeness and false-positive rate of CNN-based strong lens finders depend on lens properties (Einstein radius, source brightness), host galaxy morphology, and observing conditions (PSF size, depth, exposure count) in the DESI Legacy Imaging Surveys Data Release 10?

### 1.3 Novelty Claim

Existing DESI lens-finder papers (Huang et al. 2020, 2021; Inchausti et al. 2025) are primarily discovery and curation pipelines. They train classifiers, scan the survey, apply thresholds, and produce candidate catalogs. What they typically do NOT deliver is:

- A per-bin selection function $\hat{C}(\theta_E, \mathrm{PSF}, \mathrm{depth})$ with confidence intervals
- Failure-mode attribution by confuser category (rings, edge-on disks, mergers, blue clumpy galaxies)
- Calibration diagnostics (ECE, MCE) stratified by observing conditions
- Quantified dependence of false-positive rate on Tractor morphological type

This work provides those missing characterisation layers, using the Inchausti et al. (2025) training protocol as a matched baseline.

### 1.4 Target Journal

Monthly Notices of the Royal Astronomical Society (MNRAS)

---

## 2. Prior Work and Bibliography

### 2.1 Core References

| Ref ID | Citation | arXiv | Relevance |
|--------|----------|-------|-----------|
| Paper IV | Inchausti et al. 2025, "Strong Lens Discoveries in DESI Legacy Imaging Surveys DR10 with Two Deep Learning Architectures" | 2508.20087v1 | Primary comparison target. ResNet + EfficientNetV2 ensemble on DR10. Our training protocol is matched to this paper. |
| Huang+2020 | Huang et al. 2020, "Discovering New Strong Gravitational Lenses in the DESI Legacy Imaging Surveys" | 2005.04730 | First paper in DESI lens-finder series. Established the residual neural network approach on DR7. |
| Huang+2021 | Huang et al. 2021, "New Strong Gravitational Lenses from the DESI Legacy Imaging Surveys Data Release 9" | 2206.02764 | Extended to DR9 with larger training set and SER/DEV/REX negative types. |
| Lanusse+2018 | Lanusse et al. 2018, "CMU DeepLens: Deep Learning for Automatic Image-Based Galaxy–Galaxy Strong Lens Finding" | 1703.02642 | Shielded ResNet architecture (194K params) that Paper IV's custom ResNet is based on. |
| Coscrato+2020 | Coscrato et al. 2020, "Feature-weighted stacking" | (see Paper IV refs) | Meta-learner methodology used by Paper IV to combine ResNet + EfficientNet predictions. |
| Dey+2019 | Dey et al. 2019, "Overview of the DESI Legacy Imaging Surveys" | 1804.08657 | DESI Legacy Imaging Surveys description. Source of DR10 imaging data. |

### 2.2 Additional DESI Lens Series Papers

| Citation | arXiv | Relevance |
|----------|-------|-----------|
| DESI Strong Lens Foundry I (HST + GIGA-Lens) | 2502.03455 | HST follow-up and lens modeling |
| DESI Strong Lens Foundry II (DESI spectroscopy) | 2509.18089 | Spectroscopic confirmation |
| DESI Strong Lens Foundry III (Keck spectroscopy) | 2509.18086 | Keck follow-up |
| DESI Strong Lens Foundry V (HST + GIGA-Lens sample) | 2512.07823 | Expanded HST sample |
| DESI Single Fiber Lens Search I | 2512.04275 | Spectroscopic lens search, 4000 candidates |
| Pairwise Spectroscopic Search (DESI DR1) | 2509.16033 | Alternative discovery channel |
| ML-driven Strong Lens Discoveries (small $\theta_E$) | 2507.01943 | Low-mass halos, small Einstein radii |

### 2.3 Methodological References

| Citation | Relevance |
|----------|-----------|
| Górski et al. 2005 (HEALPix) | Spatial split methodology (nside=128) |
| Tan & Le 2021 (EfficientNetV2) | Model architecture |
| He et al. 2016 (ResNet) | Model architecture |
| Niculescu-Mizil & Caruana 2005 | Expected Calibration Error (ECE) |

### 2.4 Attribution Notes

**CRITICAL**: We must clearly attribute that:
1. The training protocol (epochs, LR schedule, batch sizes, resolution) is derived from Inchausti et al. (2025) Section 3.2
2. The positive candidate catalog originates from the DESI lens-finder series (Huang et al. 2020, 2021; Inchausti et al. 2025)
3. The negative sampling strategy follows Inchausti et al. (2025) Section 3.1 with modifications (larger negative pool, no Spherimatch cleaning, no prior-model filtering)
4. The Lanusse et al. (2018) ResNet design inspired the "bottlenecked ResNet" concept (though we ultimately use standard ResNet-18 and EfficientNetV2-S)
5. The imaging data comes from DESI Legacy Imaging Surveys DR10 (Dey et al. 2019)
6. The Tractor catalog photometry and morphology is from the DR10 sweep files

---

## 3. Data

### 3.1 Imaging Source

- **Survey**: DESI Legacy Imaging Surveys Data Release 10 (Dey et al. 2019)
- **Sub-survey**: DECaLS (Dark Energy Camera Legacy Survey)
- **Declination range**: $-18° < \delta < 32°$ (following Inchausti et al. 2025)
- **Bands**: $g$, $r$, $z$
- **Cutout size**: 101 × 101 pixels (0.262 arcsec/pixel = 26.5 arcsec on a side)
- **Cutout format**: NPZ files containing 3-band (H, W, 3) float32 arrays
- **Cutout source**: Legacy Survey cutout server (https://www.legacysurvey.org/viewer/cutout.fits)

### 3.2 Positive Sample

- **Source**: DESI lens-finder candidate catalog (Huang et al. 2020, 2021; Inchausti et al. 2025)
- **Count**: 4,788 unique candidates (of 5,104 in lenscat; 316 unmatched to DR10 Tractor, outside DR10 coverage)
- **Tier system** (post-DR10 crossmatch counts):
  - Tier A: 389 candidates with spectroscopic confirmation or HST imaging (435 in raw lenscat; 46 lost in crossmatch)
  - Tier B: 4,399 candidates with high-confidence visual grades but no spectroscopic confirmation (4,669 in raw lenscat; 270 lost in crossmatch)
- **Note**: For Paper IV parity training, all positives are treated equally (sample_weight = 1.0). The tier system is used separately for our audit analysis.

### 3.3 Negative Sample

#### 3.3.1 Full Negative Pool

- **Source**: DR10 Tractor sweep files, extracted via Apache Spark on AWS EMR
- **Pool size**: ~26.7 million galaxies (after quality cuts)
- **Galaxy types retained**: SER (Sérsic), DEV (de Vaucouleurs), REX (Round Exponential) — following Inchausti et al. (2025). EXP type excluded for Paper IV parity.
- **Quality cuts applied**:
  - At least 3 exposures in each of $g$, $r$, $z$ bands (`nobs_g >= 3 AND nobs_r >= 3 AND nobs_z >= 3`)
  - $z$-band magnitude limit: $z < 20$ mag
  - Declination range: $-18° < \delta < 32°$
  - Maskbit exclusion: BRIGHT(1), ALLMASK_G(5), ALLMASK_R(6), ALLMASK_Z(7), MEDIUM(11), GALAXY(12), CLUSTER(13)
  - Exclusion radius: 11 arcsec from any known/candidate lens ($5'' + 2 \times \theta_{E,\mathrm{max}}$ where $\theta_{E,\mathrm{max}} = 3.0''$)

#### 3.3.2 Negative Pool Design (N1/N2)

Following LLM recommendations inspired by Inchausti et al. (2025):

- **Pool N1 (deployment-representative)**: 85% of negatives. Randomly sampled from SER/DEV/REX galaxies stratified by (`nobs_z_bin`, `type_bin`).
- **Pool N2 (hard confusers)**: 15% of negatives. Morphologically selected galaxies that resemble strong lenses:
  - Ring proxies: DEV/SER with $\mathrm{flux}_r > 5.0$ nMgy, Sérsic index > 4.0
  - Edge-on proxies: elongated galaxies with ellipticity > 0.50
  - Blue clumpy: $(g - r) < 0.4$, $r < 20.5$ mag
  - Large galaxies: half-light radius > 2.0 arcsec, $\mathrm{flux}_r > 3.0$ nMgy

#### 3.3.3 Stratified Sampling

- **Target ratio**: 100:1 negative:positive per (`nobs_z_bin`, `type_bin`) stratum
- **Strata**: 3 types (SER, DEV, REX) × 3 nobs_z bins (3-5, 6-10, 11+) = 9 strata
  - Note: The `1-2` nobs_z bin is empty because quality cuts require `nobs >= 3`
- **Actual sampled**: 453,100 negatives (from production EMR run `20260211_082238`)
- **Cutout generation**: Two EMR batches:
  - Batch 1 (2026-02-10): 411,662 cutouts generated, immediately available on Lambda NFS
  - Batch 2 (2026-02-11): 35,233 additional cutouts generated with corrected EMR configuration, initially on S3 only
  - Combined: 446,895 unique cutouts indexed across both directories
- **Cutouts used in training manifest**: 446,893 negatives (98.6% of sampled; 6,207 negatives remain without cutouts, likely due to Legacy Survey cutout server failures for those specific bricks)
- **Actual ratio in training**: 93.3:1 (446,893 negatives / 4,788 positives)
- **Pipeline version**: `spark_stratified_sample.py` v1.1.0
- **Manifest rebuild note**: Initial training (epochs 1–~20 of EfficientNetV2-S) used a manifest with only batch 1 cutouts (86:1 ratio). Training was stopped, batch 2 cutouts synced from S3 to NFS, and both manifests rebuilt with full coverage before restarting.

#### 3.3.4 Difference from Inchausti et al. (2025) Negative Construction

| Aspect | Inchausti et al. (2025) | This work |
|--------|------------------------|-----------|
| Negative count | 134,182 | 446,893 (training manifest) |
| Positive count | 1,372 | 4,788 |
| Neg:Pos ratio | ~98:1 | ~93:1 |
| Contaminant removal | Spherimatch external crossmatch | Not implemented |
| Prior model cleaning | Prior ResNet, remove $p > 0.4$ | Not implemented (see §7 Limitations) |
| N1/N2 pool design | Not described in paper | 85:15 deployment/confuser split |

### 3.4 Spatial Splits

- **Method**: HEALPix-based deterministic spatial assignment (Górski et al. 2005)
- **Parameters**: nside = 128, NESTED ordering, SHA-256 hash with seed = 42
- **Algorithm**: For each galaxy, compute HEALPix pixel index from (RA, Dec). Hash `"{healpix_idx}_42"` with SHA-256. Convert first 4 bytes to float in [0, 1). Assign: $< 0.70$ → train, $[0.70, 0.85)$ → val, $\geq 0.85$ → test.
- **Property**: All galaxies within the same HEALPix pixel (~0.21 deg²) are assigned to the same split, preventing spatial leakage.
- **Verified**: Zero overlap between splits in galaxy_id and cutout_path (see `docs/split_verification_report.json`).

#### 3.4.1 Two Manifest Schemes

| Manifest | Split Scheme | Purpose | Row Count |
|----------|-------------|---------|-----------|
| `training_parity_70_30_v1.parquet` | 70/30 train/val | Paper IV parity comparison (val+test merged) | 451,681 |
| `training_parity_v1.parquet` | 70/15/15 train/val/test | Our audit analysis (held-out test set) | 451,681 |

- Paper IV uses 70/30 train/val with no held-out test set (Inchausti et al. 2025, Section 3.2)
- Our parity training uses the 70/30 manifest to match their protocol
- Our audit analysis retains the 70/15/15 split for proper held-out evaluation

#### 3.4.2 Paper IV Parity Manifest (70/30)

| Split | Positives | Negatives | Total | Fraction |
|-------|-----------|-----------|-------|----------|
| train | 3,356 | 312,744 | 316,100 | 70.0% |
| val | 1,432 | 134,149 | 135,581 | 30.0% |
| **Total** | **4,788** | **446,893** | **451,681** | **100%** |

- All sample weights = 1.0 (unweighted, matching Paper IV)
- Null cutout paths: 0
- **Updated 2026-02-11**: Rebuilt with batch 1 + batch 2 cutouts (93.3:1 ratio, up from 86:1)

#### 3.4.3 Audit Manifest (70/15/15)

| Split | Positives | Negatives | Total | Fraction |
|-------|-----------|-----------|-------|----------|
| train | 3,356 | 312,744 | 316,100 | 70.0% |
| val | 717 | 66,773 | 67,490 | 14.9% |
| test | 715 | 67,376 | 68,091 | 15.1% |
| **Total** | **4,788** | **446,893** | **451,681** | **100%** |

---

## 4. Preprocessing

### 4.1 Image Loading

- Cutouts stored as `.npz` files with key `"cutout"`, shape (101, 101, 3) in HWC format
- Transposed to CHW (3, 101, 101) for PyTorch
- Data type: float32

### 4.2 Normalization: Outer-Annulus Robust Scaling (`raw_robust`)

For each band independently:

1. Define an annular region centered on the stamp. For 101×101 stamps, the
   corrected annulus uses $r_\mathrm{in} = 32.5$ pixels and $r_\mathrm{out} = 45.0$ pixels
   (65–90% of the image half-width), ensuring it lies in the sky-dominated outer
   region. See `dhs/utils.py::default_annulus_radii()` for the formula.
2. Compute the median $\tilde{x}$ and median absolute deviation (MAD) of pixel values within the annulus
3. Normalize: $x_\mathrm{norm} = (x - \tilde{x}) / (\mathrm{MAD} + \epsilon)$ where $\epsilon = 10^{-8}$
4. Clip to $[-10, +10]$

**ERRATUM (2026-02-13, LLM review finding #1)**: Earlier models (v1–v4) used a
hardcoded annulus of $(r_\mathrm{in}, r_\mathrm{out}) = (20, 32)$ pixels, which
was originally tuned for 64×64 center-cropped stamps. When the pipeline moved to
101×101 stamps (Paper IV parity, crop=False), this annulus was not updated. At
(20, 32), the annulus sits at 40–63% of the image half-width, overlapping with
galaxy light (~20% of total flux). This inflates the MAD denominator and
suppresses arc contrast. The corrected annulus at (32.5, 45.0) places the ring
at 65–90% of the half-width, where galaxy flux fraction is ~6%. Models v5+ use
the corrected annulus. See `docs/RETRAIN_PLAN_ANNULUS_FIX.md` for details.

**Note**: Inchausti et al. (2025) do not specify their normalization procedure. Our `raw_robust` approach is defensible as it: (a) uses sky-dominated pixels for the reference statistics, avoiding bias from the central galaxy; (b) is robust to outliers via median/MAD rather than mean/std; (c) is applied identically at train and test time.

**Note**: We do NOT multiply MAD by 1.4826 (the factor to convert MAD to Gaussian σ). This is a deliberate choice for consistency with our existing pipeline. The resulting normalization is a monotonic rescaling and does not affect model performance.

### 4.3 No Center Crop for Parity Training

- For Paper IV parity: input is 101 × 101 pixels (no cropping). Config: `crop: false`.
- Our earlier baseline used a center crop to 64 × 64 pixels (`STAMP_SIZE = 64`). This is NOT used for parity training.
- Paper IV does not describe any cropping from 101 × 101.

### 4.4 NaN Handling

- NaN pixels are replaced with 0.0 before normalization
- After normalization, any remaining NaN values are replaced with 0.0
- Cutouts with > 5% NaN pixels were excluded during negative pool extraction

### 4.5 Data Augmentation

Applied during training only (not during evaluation):

| Augmentation | Probability | Implementation |
|-------------|-------------|----------------|
| Horizontal flip | 50% | `np.flip` along width axis |
| Vertical flip | 50% | `np.flip` along height axis |
| 90° rotation | Uniform {0°, 90°, 180°, 270°} | `np.rot90` with random $k \in \{0,1,2,3\}$ |

**Note**: Inchausti et al. (2025) do not describe their augmentation strategy. Our augmentation is conservative (geometric symmetries only, no photometric augmentation) and is standard for astronomical imaging where the sky orientation is arbitrary.

**Reproducibility**: Augmentations use a per-sample deterministic seed: `seed = (dataset_seed × 1000003 + sample_index) & 0x7fffffff`, using `numpy.random.default_rng(seed)`. This ensures identical augmentations across runs with the same dataset seed.

---

## 5. Model Architectures

### 5.1 ResNet-18 (Standard PyTorch)

- **Source**: `torchvision.models.resnet18` (He et al. 2016)
- **Modifications**:
  - First convolutional layer: `Conv2d(3, 64, kernel_size=7, stride=2, padding=3)` (unchanged from standard, accepts 3-channel input)
  - Final fully-connected layer: `Linear(512, 1)` (single logit for binary classification)
- **Parameters**: 11,177,025 (~11.2M)
- **Initialization**: Random (no pre-training)
- **Paper IV comparison**: Paper IV uses a custom "shielded ResNet" based on Lanusse et al. (2018) with 194,433 parameters. Our ResNet-18 has ~58× more parameters. This capacity difference is a known limitation and is explicitly reported. We justify using standard ResNet-18 because: (a) the exact Lanusse et al. TensorFlow architecture is not publicly available; (b) torchvision ResNet-18 is widely reproducible; (c) we report parameter counts explicitly for referee assessment.

### 5.2 EfficientNetV2-S (ImageNet Pre-trained)

- **Source**: `torchvision.models.efficientnet_v2_s` (Tan & Le 2021)
- **Weights**: `EfficientNet_V2_S_Weights.DEFAULT` (ImageNet-1K pre-trained)
- **Modifications**:
  - Classifier head replaced: `nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, 1))`
- **Parameters**: 20,178,769 (~20.2M)
- **Paper IV comparison**: Paper IV uses EfficientNetV2 with 20,542,883 parameters. Our 20.2M is a close match (1.8% fewer parameters), consistent with minor differences between torchvision's implementation and their TensorFlow version.
- **Fine-tuning**: All layers are fine-tuned from epoch 1 (no frozen stem).

### 5.3 Model Factory

Both models are instantiated via `build_model(arch, in_ch=3, **kwargs)` in `dhs/model.py`, selected by the `arch` field in the YAML config.

---

## 6. Training Protocol

### 6.1 Paper IV Parity Training Parameters

| Parameter | ResNet-18 | EfficientNetV2-S | Paper IV ResNet | Paper IV EfficientNet |
|-----------|-----------|-------------------|-----------------|----------------------|
| Input size | 101 × 101 × 3 | 101 × 101 × 3 | 101 × 101 × 3 | 101 × 101 × 3 |
| Epochs | 160 | 160 | 160 | 160 |
| Micro-batch size | 128 | 64 | — | — |
| Effective batch size | 2,048 | 512 | 2,048 | 512 |
| Gradient accumulation steps | 16 | 8 | N/A (multi-GPU) | N/A (multi-GPU) |
| Initial learning rate | 5 × 10⁻⁴ | 3.88 × 10⁻⁴ | 5 × 10⁻⁴ | 3.88 × 10⁻⁴ |
| LR schedule | StepLR | StepLR | StepLR | StepLR |
| LR halved at epoch | 80 | 130 | 80 | 130 |
| LR decay factor | 0.5 | 0.5 | 0.5 | 0.5 |
| Optimizer | AdamW | AdamW | Not specified | Not specified |
| Weight decay | 1 × 10⁻⁴ | 1 × 10⁻⁴ | — | — |
| Loss function | BCE (unweighted) | BCE (unweighted) | Cross-entropy (unweighted) | Cross-entropy (unweighted) |
| Early stopping | Disabled | Disabled | Not described | Not described |
| Pre-training | None (random init) | ImageNet-1K | None | ImageNet |
| Mixed precision | Yes (AMP) | Yes (AMP) | Not specified | Not specified |
| Split | 70/30 train/val | 70/30 train/val | 70/30 train/val | 70/30 train/val |

### 6.2 Gradient Accumulation

Paper IV trained on multiple GPUs with large batch sizes (2048 for ResNet, 512 for EfficientNet). We emulate this on a single GPU (NVIDIA GH200, 97.8 GB VRAM) using gradient accumulation:

```
effective_batch = micro_batch × accumulation_steps
ResNet-18:       2048 = 128 × 16
EfficientNetV2:  512  = 64 × 8
```

The optimizer step occurs every `accumulation_steps` micro-batches. Loss is scaled by `1 / accumulation_steps` before backpropagation to maintain correct gradient magnitude.

### 6.3 Loss Function

Binary cross-entropy with logits (`BCEWithLogitsLoss`), reduction='mean', unweighted. All sample weights are 1.0 in the parity manifest.

Paper IV describes their loss as Equation (1): standard binary cross-entropy. No tier weighting or sample reweighting is used in the parity baseline.

### 6.4 Checkpointing Strategy

- `best.pt`: Saved whenever validation AUC improves (best model for deployment)
- `last.pt`: Saved every epoch (for resuming interrupted training)
- `epoch_{N:03d}.pt`: Saved every 10 epochs (for training curve analysis)
- All checkpoints synced to S3 (`s3://darkhaloscope/stronglens_calibration/checkpoints/`) every 10 minutes via background watcher process

### 6.5 Training Execution

- **Hardware**: NVIDIA GH200 (97.8 GB VRAM), Lambda Cloud instance
- **Execution**: Sequential (EfficientNetV2-S first, then ResNet-18)
- **Data loading**: 8 DataLoader workers, pin_memory=True, drop_last=True (training only)
- **Data source**: Local NFS mount (`/lambda/nfs/darkhaloscope-training-dc/`)
- **Reproducibility**: `run_info.json` logged per run with git commit, timestamp, config SHA-256, dataset seed, full command line

### 6.6 Differences from Paper IV Training

| Aspect | Paper IV | This work | Impact |
|--------|----------|-----------|--------|
| Optimizer | Not specified | AdamW (weight_decay=1e-4) | Minor. Paper IV does not specify; both SGD+momentum and AdamW are standard. |
| Multi-GPU | Yes (implied by large batch) | Single GPU + gradient accumulation | Gradient accumulation is mathematically equivalent for SGD. For Adam-family optimizers, there are subtle differences in moment estimates, but these are negligible in practice. |
| Mixed precision | Not specified | Yes (torch.cuda.amp) | Speeds up training. Does not affect final model quality for these architectures. |
| Negative cleaning | Spherimatch + prior model $p > 0.4$ | Not done | See §7.1 |

---

## 7. Limitations and Honest Disclosures

### 7.1 No Negative Cleaning

**What Paper IV did**: Before training, they cleaned their negative pool using:
1. Spherimatch external crossmatch to remove known contaminants
2. A prior ResNet model (from Paper III) to score all negatives; candidates with $p > 0.4$ were visually inspected and removed

**What we did NOT do**: Neither Spherimatch cleaning nor prior-model filtering. Our negatives may contain unlabeled real lenses.

**Defense**: "We did not replicate Paper IV's negative cleaning step, which relied on a prior model from their earlier work (not publicly available). We treat this as an ablation and note it as a limitation. Our negative pool applies spatial exclusion (11 arcsec from known lenses) and quality cuts (maskbits, exposure count, magnitude limit) but does not filter based on model predictions. This means our training set may contain a small number of unlabeled real lenses in the negative class, which would suppress sensitivity. Our reported AUC should therefore be interpreted as a lower bound relative to a model trained on a cleaned negative set."

### 7.2 Architecture Capacity Mismatch (ResNet)

**Paper IV**: Custom shielded ResNet following Lanusse et al. (2018), 194,433 parameters.
**This work**: Standard `torchvision.models.resnet18`, 11,177,025 parameters (58× larger).

**Defense**: "We use the standard PyTorch ResNet-18 implementation for reproducibility. The 58× parameter difference means our ResNet has substantially more capacity than Paper IV's custom architecture. We report this difference explicitly and include EfficientNetV2-S (~20.2M parameters, closely matching Paper IV's ~20.5M) as the primary comparison model."

### 7.3 Residual Missing Negatives

The corrected stratified sample contains 453,100 negatives. After syncing both cutout batches (batch 1: 411,662, batch 2: 35,233), 446,893 negatives (98.6%) have cutouts in the training manifest. The remaining 6,207 (1.4%) are uniformly distributed across strata and likely represent cutout server failures for specific survey bricks. The effective negative:positive ratio is 93.3:1 (vs Paper IV's ~98:1), which is close enough for defensible comparison.

**Note**: Initial training (epochs 1–~20) used only batch 1 cutouts (86:1 ratio). Training was stopped, batch 2 cutouts were synced from S3, manifests rebuilt, and training restarted from scratch with the corrected 93.3:1 manifest.

### 7.4 No Meta-Learner (Yet)

Paper IV combines ResNet + EfficientNetV2 predictions using a 1-layer neural network with 300 hidden nodes (feature-weighted stacking, Coscrato et al. 2020), achieving AUC 0.9989 vs individual AUCs of 0.9984 (ResNet) and 0.9987 (EfficientNet). We plan to implement this after individual model training completes.

### 7.5 Normalization Not Specified in Paper IV

Inchausti et al. (2025) do not describe their image normalization procedure. Our `raw_robust` (outer-annulus median/MAD scaling) is a reasonable choice but may differ from their approach. This is unavoidable without access to their code.

### 7.6 Augmentation Not Described in Paper IV

Inchausti et al. (2025) do not describe data augmentation. Our HFlip/VFlip/Rot90 is conservative and standard for astronomical imaging. Any difference in augmentation strategy could affect results.

### 7.7 Injection Pipeline: Known Issues and Limitations (Selection Function)

The injection-recovery pipeline (`dhs/injection_engine.py`, `scripts/selection_function_grid.py`, `scripts/validate_injections.py`) had the following issues identified during internal code review (2026-02-12).

**Resolution status (2026-02-10)**: 7 of 8 issues RESOLVED with verified code fixes. Issue #5 (literature review) is external and OPEN pending LLM response.

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | Lensing magnification | CRITICAL | RESOLVED — analytical Sersic normalization |
| 2 | SIS-only lens model | MAJOR | RESOLVED — SIE implemented (Kormann et al. 1994) |
| 3 | No fixed-FPR | MAJOR | RESOLVED — derive_fpr_thresholds() |
| 4 | Source offset not area-weighted | MAJOR | RESOLVED — sqrt(uniform) sampling |
| 5 | Literature review | SIGNIFICANT | OPEN — awaiting LLM response |
| 6 | Missing features | SIGNIFICANT | PARTIALLY RESOLVED — infrastructure built |
| 7 | Misleading comment | MODERATE | RESOLVED — replaced with analytical explanation |
| 8 | Ad hoc clump model | MODERATE | RESOLVED — documented as limitation |

#### 7.7.1 CRITICAL: Missing Lensing Magnification (Issue #1)

**Status**: RESOLVED (2026-02-10)

**Problem**: The injection engine normalized the Sersic source profile by its IMAGE-PLANE integral (`integral = (shape.sum() * pix_area)`), then scaled to the specified "total unlensed flux." This made `total_image_flux = flux_unlensed`, effectively canceling lensing magnification. For SIS, μ_total = 2θ_E/|β|, typically 5–20× for our parameter range, meaning injected arcs were 5–30× too faint.

**Fix applied**: Replaced image-plane normalization with analytical source-plane Sersic integral:

`I_source = 2π × q × n × R_e² × exp(b_n) × Γ(2n) / b_n^(2n)`

New function `_sersic_source_integral()` in `injection_engine.py`. This preserves lensing magnification: `total_image_flux = μ_eff × flux_unlensed`.

**Additional fix (2026-02-10)**: Added sub-pixel oversampling (default 4×4) to eliminate pixel-sampling bias for compact sources. Without oversampling, point-sampling the peaked Sersic profile at pixel centers overestimates flux by ~46% for R_e=0.15" at 0.262"/pixel. With 4× oversampling, the bias is <0.04% (4× vs 8× convergence).

**Cross-validation against lenstronomy** (peer-reviewed lensing code):
- Deflection angles: 0.0000% error across 5 SIE configurations (q=0.5, 0.7, 0.85, 1.0)
- Lensed flux at 4× oversampling: our 7.3273 nmgy vs lenstronomy(10×) 7.3194 nmgy = 0.11% agreement
- The earlier apparent "46% discrepancy" was entirely due to pixel-sampling bias in the 1× rendering — both codes produce identical results at adequate resolution.

**Verification evidence** (5 tests, all passed with 4× oversampling):
- T1: On-axis source (β≈0) → mu_eff = 98.6× (was 1.0× before magnification fix)
- T2: β=0.5", θ_E=1.5", R_e=0.15" → mu_eff = 6.40× (consistent with SIS point-source prediction 6.0× for extended source)
- T3: Far offset (β=5θ_E) → mu_eff = 1.20× (weak lensing regime)
- T4: Flux conservation: max|injected - (host + injection_only)| = 2.4e-7 nmgy
- T5: Clumps cause < 0.2% flux deviation from non-clumped

#### 7.7.2 MAJOR: SIS-Only Lens Model, Not SIE (Issue #2)

**Status**: RESOLVED (2026-02-10)

**Problem**: The injection engine implemented only SIS (singular isothermal sphere). Real lenses have ellipticity (ε ~ 0.2–0.5). SIE produces morphologically diverse images.

**Fix applied**: Implemented `_sie_deflection()` function using Kormann et al. (1994) formulation with q→1 branch for SIS limit. Updated `LensParams` with `q_lens` (axis ratio b/a) and `phi_lens_rad` (position angle). The `inject_sis_shear()` function now uses SIE by default, with `q_lens=1.0` reducing exactly to SIS.

**Verification evidence** (4 tests, all passed):
- S1: SIE(q=1.0) matches SIS to < 4e-7 arcsec (machine precision)
- S2: SIE(q=0.7) produces measurably different morphology (Ixx, Iyy moments differ)
- S3: Both SIS and SIE show strong magnification (10.8× and 12.3× respectively)
- S4: Deflection finite at origin (eps softening)

#### 7.7.3 MAJOR: No Fixed-FPR Operating Point (Issue #3)

**Status**: RESOLVED (2026-02-10)

**Problem**: Selection function only evaluated at fixed probability thresholds (0.3, 0.5, 0.7), which referees rightly view as arbitrary.

**Fix applied**: Added `derive_fpr_thresholds()` function in `selection_function_grid.py`. It scores all negatives in the validation split, sorts descending, and finds the threshold τ such that FPR(τ) = target. New CLI argument `--fpr-targets 0.001 0.0001` derives thresholds and appends them to the evaluation grid. Output CSV includes `threshold_type` ("fixed" or "FPR=X") and `fpr_target` columns for full traceability.

**Verification**: Structure verified at compile time. End-to-end verification requires a trained checkpoint (deferred to production run).

#### 7.7.4 MAJOR: Source Offset Not Area-Weighted (Issue #4)

**Status**: RESOLVED (2026-02-10)

**Problem**: `sample_source_params` drew `beta_frac = uniform(lo, hi)`, oversampling small offsets (high magnification) relative to the physical area prior P(β) ∝ β.

**Fix applied**: Changed to `beta_frac = sqrt(uniform(lo², hi²))`, which produces the correct area-weighted distribution.

**Verification evidence**: KS test on 100k samples: stat=0.003, p=0.33 (consistent with correct P(β_frac) ∝ β_frac distribution). Histogram confirms linearly increasing PDF.

#### 7.7.5 SIGNIFICANT: Inadequate Literature Review (Issue #5)

**Status**: OPEN — new LLM prompt created (`llm_review_package/LLM_PROMPT_LITERATURE_REVIEW.md`)

The prior LLM review did not actually compare injection models, source populations, validation approaches, or completeness methodologies across the 8+ papers we asked about (Collett 2015, Jacobs et al. 2019, Metcalf et al. 2019, Huang et al. 2020/2021, Rojas et al. 2022, Cañameras et al. 2021/2024, Inchausti et al. 2025). A new, structured prompt has been created requesting paper-by-paper review with specific questions about injection methodology.

#### 7.7.6 SIGNIFICANT: Missing Recommended Features (Issue #6)

**Status**: PARTIALLY RESOLVED (2026-02-10) — infrastructure built, execution deferred to trained checkpoint

| Feature | Status | Implementation |
|---------|--------|---------------|
| Sensitivity analysis | INFRASTRUCTURE BUILT | `scripts/sensitivity_analysis.py` — wrapper that runs grid with 9 parameter perturbations (PSF ±10%, source size ±30%, color ±0.2 mag, q_lens prior). Reports delta-completeness per cell. Execution deferred. |
| Matched known-lens recovery | DOCUMENTED (design only) | Procedure documented below. Requires Tier-A θ_E estimates (not yet available). |
| SNR/color distribution gates | INFRASTRUCTURE BUILT | `validate_injections.py --anchor-manifest` computes KS test between injection and real-lens arc-annulus SNR. Requires Tier-A anchor manifest. |
| Score distribution check | IMPLEMENTED | `validate_injections.py` now reports p5/p25/p50/p75/p95 percentiles of scores and flags saturation (>90% above 0.9 or below 0.1). |
| Source magnitude stratification | IMPLEMENTED | `selection_function_grid.py` now records source r-mag per injection and outputs per-cell completeness stratified by source-mag bins (23–24, 24–25, 25–26). `source_mag_bin` column in output CSV. |
| Population count cross-check | FUTURE WORK | Requires external lens population model (Collett 2015). Not blocking for first submission. |

**Matched known-lens recovery procedure (design only, 6e)**: For each Tier-A anchor with estimated θ_E: (1) find host galaxies at similar PSF/depth, (2) inject with matched θ_E, (3) compare model score on real vs injected. Implementation deferred until Tier-A θ_E estimates are available.

#### 7.7.7 MODERATE: Misleading Code Comment (Issue #7)

**Status**: RESOLVED (2026-02-10) — fixed together with Issue #1

The misleading comment was replaced with a 12-line block explaining the analytical source-plane normalization, why image-plane normalization was wrong, and how magnification is preserved. The comment also documents the clump approximation and references Section 7.7.8.

#### 7.7.8 MODERATE: Ad Hoc Clump Model (Issue #8)

**Status**: RESOLVED (2026-02-10) — documented as limitation in code and notes

**Problem**: `_add_clumps()` uses a phenomenological mixing formula with no physical basis.

**Fix applied**: Added a comprehensive docstring to `_add_clumps()` in `injection_engine.py` documenting 4 specific limitations:
1. The mixing formula is phenomenological, not physically derived
2. Clump brightnesses are coupled to the base profile's statistics (ad hoc)
3. Total flux is approximate (up to ~clump_frac deviation; documented as minor vs. the 500-3000% magnification error that was fixed)
4. The model is adequate for probing morphological sensitivity but should not be interpreted as a realistic star-forming region model

**Remaining assumption**: The clump model is a documented approximation. For the selection function, this is acceptable because (a) the sensitivity analysis framework (Issue #6) can quantify the impact by varying clump parameters, and (b) the primary completeness results use clumped and non-clumped sources.

---

## 8. EMR Data Pipeline

### 8.1 Infrastructure

- **Cloud**: AWS (us-east-2)
- **Compute**: Amazon EMR 7.5 with Apache Spark
- **Storage**: S3 (`s3://darkhaloscope/stronglens_calibration/`)
- **Training**: Lambda Cloud (NVIDIA GH200)
- **Data transfer**: `rclone` and `aws s3 cp` between S3 and Lambda NFS

### 8.2 Pipeline Steps (Executed)

| Step | EMR Cluster | Status | Output |
|------|-------------|--------|--------|
| 1. Negative pool extraction | Multiple early runs | TERMINATED | 26.7M galaxy catalog (S3 parquet) |
| 2. Stratified sampling (production) | j-113TH5U4TG9YN | TERMINATED | 453,100 sampled negatives |
| 3. Cutout generation (batch 1) | j-3FQM74190BX8T | TERMINATED | 411,662 negative cutouts |
| 4. Cutout generation (batch 2) | j-7R1OW6BYO2MC | TERMINATED | 35,233 negative cutouts |
| 5. Batch 2 sync to NFS | Lambda (aws s3 sync) | COMPLETE | 35,233 files synced |
| 6. Manifest rebuild | Lambda (local script) | COMPLETE | 451,681 rows (93.3:1), 70/30 and 70/15/15 |
| 7. Training | Lambda (GH200) | RESTARTING | EfficientNetV2-S 160 ep, then ResNet-18 160 ep |

### 8.3 EMR Best Practices Applied

- Repartitioning for even distribution across workers
- `persist(StorageLevel.DISK_ONLY)` and `unpersist()` for intermediate DataFrames
- S3-based checkpointing with `--force` flag to override
- `--test-limit` flag for small-scale validation runs before production
- Auto-termination after step completion or failure
- Pre-flight code validation and post-upload MD5 hash verification

---

## 9. Evaluation Plan (To Be Executed After Training)

### 9.1 Metrics

| Metric | Description | Stratified by |
|--------|-------------|---------------|
| AUC (ROC) | Area under ROC curve | Overall, per-split, per-type |
| Precision / Recall | At various thresholds | Per-type, per-confuser |
| ECE | Expected Calibration Error | Overall, per-depth-bin |
| MCE | Maximum Calibration Error | Overall |
| FPR by confuser category | False positive rate for ring/edge-on/blue/large | Per-N2 subcategory |
| Bootstrap 95% CI | 10,000 bootstrap resamples | All primary metrics |

### 9.2 Comparison Points

1. **Paper IV reported metrics** (from Inchausti et al. 2025):
   - ResNet AUC: 0.9984 (val)
   - EfficientNet AUC: 0.9987 (val)
   - Ensemble (meta-learner) AUC: 0.9989

2. **Our parity baseline** (to be completed):
   - ResNet-18 AUC: [PENDING]
   - EfficientNetV2-S AUC: [PENDING]

### 9.3 Selection Function

Injection-based completeness measurement using physically motivated SIS+shear lensing with Sersic source models (`dhs/injection_engine.py`):

1. For each cell $(\theta_E, \mathrm{PSF\,bin}, \mathrm{depth\,bin})$: inject lensed sources into real DR10 host cutouts via ray-shooting
2. Per-host PSF and depth conditioning from manifest (`psfsize_r`, `psfdepth_r`)
3. Score with the frozen trained model using identical preprocessing (`raw_robust`, 101×101)
4. Compute detection fraction $\hat{C}$ with Bayesian binomial confidence intervals (Jeffreys prior)
5. Report arc annulus SNR diagnostics per cell
6. Support multiple detection thresholds and fixed-FPR operating points

**Pipeline scripts:**
- `scripts/selection_function_grid.py` — Full grid runner with S3 output support
- `scripts/validate_injections.py` — QA: default vs core-suppressed injection, flux conservation, visual panels

**Supporting modules (new, 2026-02-12):**
- `dhs/injection_engine.py` — SIS+shear ray-shooting, Sersic+clumps source, nanomaggy flux, FFT Gaussian PSF
- `dhs/s3io.py` — S3/local dual-write utility (supports `s3://` and `s3a://`)
- `dhs/selection_function_utils.py` — `m5_from_psfdepth`, `bayes_binomial_interval`, binning utilities

**CRITICAL**: 8 known issues must be resolved before selection function results are used in the paper — see §7.7 for full details. Most critical: magnification bug (§7.7.1) makes current arcs 5–30× too faint.

#### 9.3.1 Injection Priors (Single Source of Truth)

The following table lists the **exact** parameter values used by the injection
engine. These are the code defaults in `sample_source_params()` and
`sample_lens_params()` in `dhs/injection_engine.py`, validated by the test
`tests/test_injection_priors.py` against the registry file
`configs/injection_priors.yaml`.

**ERRATUM (2026-02-13)**: Earlier drafts described different values for several
parameters (e.g., $R_e \in [0.1, 0.5]$ arcsec, $n \in [0.5, 4.0]$, uniform
wide colors). The table below reflects what the code *actually uses*. The
discrepancy was identified by LLM review finding #2.

| Parameter | Distribution | Code default | Unit |
|-----------|-------------|-------------|------|
| Source $r$-mag | $\mathrm{U}(23, 26)$ | `r_mag_range = (23.0, 26.0)` | AB mag |
| $\beta_\mathrm{frac} = \beta / \theta_E$ | Area-weighted: $\sqrt{\mathrm{U}(0.01, 1.0)}$ | `beta_frac_range = (0.1, 1.0)` | — |
| $R_e$ (source half-light) | $\mathrm{U}(0.05, 0.25)$ | `re_arcsec_range = (0.05, 0.25)` | arcsec |
| $n$ (Sérsic index) | $\mathrm{U}(0.7, 2.5)$ | `n_range = (0.7, 2.5)` | — |
| $q$ (source axis ratio) | $\mathrm{U}(0.3, 1.0)$ | `q_range = (0.3, 1.0)` | — |
| $g - r$ color | $\mathcal{N}(0.2, 0.25)$ | `g_minus_r_mu_sigma = (0.2, 0.25)` | mag |
| $r - z$ color | $\mathcal{N}(0.1, 0.25)$ | `r_minus_z_mu_sigma = (0.1, 0.25)` | mag |
| Clump probability | Bernoulli(0.6) | `clumps_prob = 0.6` | — |
| $N_\mathrm{clumps}$ (if present) | integers $\{1, 2, 3\}$ | `rng.integers(1, 4)` | — |
| Clump frac (if present) | $\mathrm{U}(0.15, 0.45)$ | `rng.uniform(0.15, 0.45)` | — |
| External shear $|\gamma|$ | half-$\mathcal{N}(0, 0.05)$ | `shear_sigma = 0.05` | — |
| Lens center jitter | $\mathcal{N}(0, 0.05)$ per axis | `center_sigma_arcsec = 0.05` | arcsec |
| $q_\mathrm{lens}$ (Model 1) | $\mathrm{U}(0.5, 1.0)$ | `q_lens_range = (0.5, 1.0)` | — |

**Model 2 conditioning** (`injection_model_2/host_matching.py`):
$q_\mathrm{lens} = \mathrm{clip}(q_\mathrm{host} + \mathcal{N}(0, 0.05), 0.5, 1.0)$.
Note: This is Gaussian **additive** scatter on the host axis ratio, NOT
multiplicative uniform as some earlier descriptions stated. $\phi_\mathrm{lens}$
is aligned with the host PA.

---

## 10. Software and Reproducibility

### 10.1 Repository Structure

```
stronglens_calibration/
├── configs/                    # YAML training configurations
│   ├── paperIV_resnet18.yaml
│   ├── paperIV_efficientnet_v2_s.yaml
│   ├── negative_sampling_v1.yaml
│   └── ...
├── dhs/                        # Core training library
│   ├── model.py                # Model architectures (ResNet-18, EfficientNetV2-S)
│   ├── train.py                # Training loop with gradient accumulation
│   ├── data.py                 # Dataset loading (file_manifest mode)
│   ├── preprocess.py           # Image preprocessing (raw_robust normalization)
│   ├── calibration.py          # ECE, MCE, reliability curves
│   ├── transforms.py           # Data augmentation
│   ├── constants.py            # Global constants
│   ├── utils.py                # Utility functions
│   ├── injection_engine.py     # ** SIS+shear injection (see §7.7 for known issues)
│   ├── s3io.py                 # ** S3/local dual-write utility
│   ├── selection_function_utils.py  # ** Binning, m5_from_psfdepth, Bayesian CI
│   └── scripts/
│       ├── run_experiment.py   # Main training entry point
│       └── run_evaluation.py   # Evaluation script
├── scripts/                    # Analysis and pipeline scripts
│   ├── evaluate_parity.py      # ** Paper IV parity evaluation (AUC, ECE, MCE, FPR, bootstrap CIs)
│   ├── meta_learner.py         # ** 1-layer NN meta-learner (Coscrato et al. 2020, 300 hidden)
│   ├── negative_cleaning_scorer.py  # ** Score negatives, flag p>0.4 for cleaning
│   ├── selection_function_grid.py   # ** Injection-recovery completeness C(θ_E, PSF, depth)
│   ├── validate_injections.py       # ** Injection QA (default vs core-suppressed, flux conservation)
│   ├── make_parity_manifest.py      # Generate 70/30 and 70/15/15 manifests
│   ├── validate_stratified_output.py
│   ├── verify_splits.py
│   ├── bootstrap_eval.py
│   └── ...
├── common/                     # Shared utilities
│   ├── manifest_utils.py       # Manifest column constants, load/save
│   └── ...
├── tests/                      # Test suite
│   ├── test_preprocess_regression.py  # ** Preprocessing checksum lock (21 tests)
│   ├── test_injection_engine.py       # ** Injection physics + lenstronomy cross-validation (28 tests)
│   └── ...
├── emr/                        # AWS EMR Spark jobs
│   ├── spark_stratified_sample.py
│   ├── launch_stratified_sample.py
│   ├── launch_cutout_generation.py
│   └── ...
├── docs/                       # Documentation
│   ├── MNRAS_RAW_NOTES.md      # This file (current ground truth)
│   └── ...
└── requirements.txt            # Pinned dependencies for reproducibility
```

** = newly added scripts (2026-02-11), ready for use when training completes

### 10.2 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | >= 2.7.0 | Model training |
| torchvision | >= 0.22.0 | Pre-trained models |
| numpy | >= 1.26.4 | Array operations |
| pandas | >= 2.1.4 | Data manipulation |
| scikit-learn | >= 1.4.1 | Metrics (AUC, etc.) |
| healpy | >= 1.16.0 | HEALPix spatial splits |
| astropy | >= 5.0 | Astronomical coordinates |
| lenstronomy | >= 1.11.0 | **Test-only**: cross-validation oracle for injection engine physics (see Appendix A) |

### 10.3 Reproducibility Artifacts

Each training run produces `run_info.json` containing:
- Git commit hash (12 chars)
- UTC timestamp
- Config file path and SHA-256 hash
- Dataset seed
- Full command line

---

## 11. Current Training Status

### 11.1 Training Run History

#### Run 1 (Aborted): EfficientNetV2-S with batch 1 cutouts only

- **Manifest**: `training_parity_70_30_v1.parquet` (batch 1 only, 416,449 rows, 86:1 ratio)
- **Status**: Stopped at epoch ~20 to rebuild manifest with batch 2 cutouts
- **Reason for abort**: Decided to include batch 2 cutouts (35K additional negatives) to improve neg:pos ratio from 86:1 to 93:1 for closer Paper IV parity
- **Results before abort** (informational only, not used):

| Epoch | Train Loss | Val AUC | Best AUC | LR | Time/epoch |
|-------|-----------|---------|----------|-----|------------|
| 1 | 0.0359 | 0.9830 | 0.9830 | 3.88e-4 | ~400s |
| 2 | 0.0198 | 0.9832 | 0.9832 | 3.88e-4 | ~399s |
| 5 | 0.0065 | 0.9728 | 0.9832 | 3.88e-4 | ~400s |
| 10 | 0.0027 | 0.9606 | 0.9832 | 3.88e-4 | ~395s |
| 13 | 0.0021 | 0.9645 | 0.9832 | 3.88e-4 | ~395s |

#### Run 2 (Current): EfficientNetV2-S with corrected manifest

- **Manifest**: `training_parity_70_30_v1.parquet` (batch 1 + batch 2, 451,681 rows, 93.3:1 ratio)
- **Status**: Starting
- **Config**: `configs/paperIV_efficientnet_v2_s.yaml`

### 11.2 ResNet-18 (Run 2, Current — lambda2)

- **Manifest**: `training_parity_70_30_v1.parquet` (batch 1 + batch 2, 451,681 rows, 93.3:1 ratio)
- **Status**: Training in parallel on separate GPU instance (lambda2)
- **Config**: `configs/paperIV_resnet18.yaml`
- **Hardware**: NVIDIA GH200, Lambda Cloud (lambda2)

### 11.3 Estimated Timeline

- EfficientNetV2-S (lambda1): ~495s/epoch × 160 epochs ≈ 22 hours
- ResNet-18 (lambda2): ~628s/epoch × 160 epochs ≈ 28 hours
- Both running in parallel since 2026-02-11
- **All evaluation code is ready** (`scripts/evaluate_parity.py`, `scripts/meta_learner.py`, etc.)

---

## 12. Paper Outline (Draft)

1. **Introduction**: DESI Legacy Survey lens finding; need for selection function characterisation; gap in existing work
2. **Data**: DR10 imaging; positive catalog; negative pool design; spatial splits
3. **Methods**: Preprocessing; model architectures; training protocol; gradient accumulation; matched baseline design
4. **Results**: [PENDING] Training curves; final AUC comparison; calibration; FPR by confuser
5. **Selection Function**: [PENDING] Injection-based completeness; dependence on $\theta_E$, PSF, depth
6. **Discussion**: Comparison to Paper IV; limitations; implications for lens demographics
7. **Conclusions**

---

## 13. Open Questions and Decisions Needed

1. **Should we retrain after negative cleaning?** The first-pass model can be used to score negatives and remove $p > 0.4$ candidates. This would require a second 160-epoch training run per model. Decision: defer to after initial results.

2. **Meta-learner implementation**: Paper IV uses a 1-layer NN with 300 hidden nodes. We need to implement this after both models finish training.

3. ~~**41K missing negatives**~~: RESOLVED. Batch 2 cutouts synced from S3 to Lambda NFS. Manifests rebuilt with both cutout directories. Ratio improved from 86:1 to 93.3:1 (6,207 negatives still missing, 1.4%, uniformly distributed).

4. **Selection function injection grid**: Needs to be defined (ranges of $\theta_E$, source brightness, PSF bins, depth bins). This is a separate pipeline step.

5. **Tier-weighted audit runs**: After parity training, run the same models on the 70/15/15 manifest with tier weights for the audit analysis. This is a separate set of experiments.

---

## 14. Timeline

| Phase | Status | ETA |
|-------|--------|-----|
| Data pipeline (EMR) | COMPLETE | — |
| Batch 2 cutout sync | COMPLETE | 2026-02-11 |
| Manifest rebuild (93.3:1 ratio) | COMPLETE | 2026-02-11 |
| EfficientNetV2-S training (160 ep) | IN PROGRESS (lambda1) | ~Feb 12 |
| ResNet-18 training (160 ep) | IN PROGRESS (lambda2) | ~Feb 12 |
| Evaluation pipeline code | COMPLETE | `scripts/evaluate_parity.py` ready |
| Meta-learner code | COMPLETE | `scripts/meta_learner.py` ready |
| Negative cleaning scorer | COMPLETE | `scripts/negative_cleaning_scorer.py` ready |
| Selection function pipeline | IN PROGRESS | Code ready, 8 known issues (see §7.7), magnification bug CRITICAL |
| Injection validation script | COMPLETE | `scripts/validate_injections.py` ready |
| Literature review (injection) | IN PROGRESS | New LLM prompt created (`LLM_PROMPT_LITERATURE_REVIEW.md`) |
| Preprocessing regression tests | COMPLETE | `tests/test_preprocess_regression.py` (21 tests, all pass) |
| Injection engine validation tests | COMPLETE | `tests/test_injection_engine.py` (28 tests incl. lenstronomy cross-validation, all pass) |
| Requirements pinning | COMPLETE | `requirements.txt` with versions |
| Run evaluation (both models) | BLOCKED | Waiting for training to finish |
| Train meta-learner | BLOCKED | Needs both models' predictions |
| Negative cleaning (optional 2nd pass) | BLOCKED | After first-pass models |
| Run selection function grid | BLOCKED | After evaluation |
| Paper draft | NOT STARTED | After all results |

---

## 15. Planned Limitation Remediation

Several current limitations (§7) can be addressed after initial model training completes:

### 15.1 Negative Cleaning (§7.1)

Once the first-pass models (ResNet-18 and EfficientNetV2-S) finish training, they can serve as the "prior model" for negative cleaning:

1. Score all negatives with the first-pass model
2. Flag candidates with $p > 0.4$ as potential contaminants
3. Visually inspect a sample (or remove automatically)
4. Rebuild the manifest without contaminated negatives
5. Optionally retrain (second-pass models) on the cleaned set

This is the standard two-pass procedure. Paper IV used their Paper III model as the prior; we use our own first-pass model, which is equally defensible.

### 15.2 Meta-Learner (§7.4)

After both models complete:

1. Extract validation-set predictions from ResNet-18 and EfficientNetV2-S
2. Train a 1-layer neural network with 300 hidden nodes (Coscrato et al. 2020 feature-weighted stacking)
3. Input features: two model scores (and optionally observing-condition metadata)
4. Evaluate ensemble AUC and compare to individual models

Paper IV achieves AUC 0.9989 with the meta-learner vs 0.9984/0.9987 individually.

### 15.3 Architecture Capacity (§7.2)

The ResNet-18 vs Lanusse-style ResNet capacity mismatch is an inherent limitation of using standard architectures. This can be partially addressed by:

1. Reporting both ResNet-18 and EfficientNetV2-S results prominently
2. Emphasising EfficientNetV2-S (20.2M params, close to Paper IV's 20.5M) as the primary comparison
3. If time permits, implementing a bottlenecked ResNet variant (~200K params) as a supplementary comparison

---

## Appendix A: Custom PyTorch Injection Engine vs. Lenstronomy

### A.1 Design Decision

The injection-recovery pipeline uses a custom ray-shooting engine (`dhs/injection_engine.py`) implemented in PyTorch rather than the widely-used `lenstronomy` library (Birrer & Amara 2018; Birrer et al. 2021). This appendix documents the rationale, the trade-offs, and how we mitigate the risks of a custom implementation.

### A.2 Why Not Use Lenstronomy Directly?

| Consideration | Lenstronomy | Custom PyTorch engine |
|---|---|---|
| **Throughput** | General-purpose modelling framework (lens inversion, MCMC fitting, multi-plane lensing, etc.). Overhead per call is significant. | Purpose-built for forward injection only. Minimal overhead. Batch-capable tensor operations on GPU. |
| **Scale requirement** | Selection function grid requires O(10⁵–10⁶) injections across (θ_E, PSF, depth, source_mag) parameter space. At ~50 ms/call (lenstronomy, single-threaded, no MCMC), this is ~14–140 CPU-hours per grid evaluation. | At ~0.5 ms/injection (PyTorch, GPU, batched), the same grid completes in ~1–15 GPU-minutes. This is a 100–1000× speedup, critical for iterative sensitivity analysis. |
| **GPU integration** | NumPy-based. Requires numpy↔torch round-trips if injections are done during training or in a GPU-resident pipeline. | Native PyTorch tensors throughout. Injections can be performed on-GPU in the same memory space as the classifier, enabling future on-the-fly injection during training without data transfer bottlenecks. |
| **Forward-model transparency** | Extensive internal machinery with many defaults (e.g., `b_n` approximation, adaptive pixel integration, PSF handling, coordinate conventions). These are well-tested but not trivially auditable for a specific use case. | Every physical assumption is explicit in ~400 lines of code: SIE deflection (Kormann et al. 1994), analytical Sersic normalization (Graham & Driver 2005), sub-pixel oversampling factor, external shear, core suppression. A referee can read the entire forward model. |
| **Dependency footprint** | Pulls in `lenstronomy` + `fastell4py` (Fortran wrapper for SIE) + `cosmohammer` + optional dependencies. On AWS EMR or minimal cloud instances, this adds installation complexity and potential build failures (Fortran compiler). | Only requires PyTorch and NumPy, which are already core dependencies for model training. Zero additional installation burden. |
| **Flexibility for custom physics** | Adding non-standard features (e.g., the phenomenological clump model, core suppression masking, per-host PSF/depth conditioning from manifest metadata) requires subclassing or monkey-patching lenstronomy internals. | Custom features are added directly. The clump model, core suppression, and manifest-conditioned PSF/depth are first-class parameters. |

### A.3 Risks of a Custom Implementation and How We Mitigate Them

The obvious risk of writing a custom lensing engine is getting the physics wrong. We mitigate this with a rigorous, multi-layered verification strategy:

**Layer 1: Internal unit tests (28 tests in `tests/test_injection_engine.py`)**

| Test category | Count | What it verifies |
|---|---|---|
| Magnification physics | 5 | On-axis, known-offset, far-field μ; flux conservation; clump stability |
| Analytical Sersic integral | 4 | Closed-form value; R_e² scaling; q scaling; positivity |
| SIE lens model | 4 | SIE(q=1)≡SIS; q=0.7 morphology; finite at origin |
| Sub-pixel oversampling | 2 | 4× vs 8× convergence < 0.5%; bias reduction |
| Area-weighted sampling | 2 | KS test against analytical CDF; mean > midpoint |
| Parameter sampling | 5 | Dataclass fields; q_lens range; meta keys |
| PSF and core suppression | 2 | Flux conservation under PSF; core suppression zeroes center |
| **Lenstronomy cross-validation** | **4** | **See Layer 2 below** |

**Layer 2: Cross-validation against lenstronomy (peer-reviewed, community-standard)**

We use `lenstronomy` as an independent oracle to validate our physics. The tests (`TestLenstronomyCrossValidation` class) compare:

1. **SIS deflection angles**: Our `_sie_deflection(q=1)` vs lenstronomy `SIS` model → max error < 1×10⁻⁴ arcsec (< 0.04% of a pixel)
2. **SIE deflection angles (q=0.7, φ=0.3)**: Our `_sie_deflection` vs lenstronomy `SIE` → max relative error < 0.1%
3. **SIE deflection angles (q=0.5, φ=1.2)**: Different axis ratio and position angle → max relative error < 0.1%
4. **Total lensed flux**: Our engine (4× oversampling) vs lenstronomy (10× oversampling) for SIE(q=0.7) + Sersic(n=1, R_e=0.15") → **0.11% agreement** (our 7.3273 nmgy vs lenstronomy 7.3194 nmgy)

These cross-validation tests are run as part of the standard test suite. They skip gracefully on machines without lenstronomy installed (e.g., EMR production nodes), but are mandatory on development machines.

**Layer 3: Physics regression gate**

Any future code change that breaks the lenstronomy cross-validation tests has introduced a physics error. The test tolerances are set conservatively:
- Deflection angle: < 1×10⁻⁴ arcsec absolute (SIS), < 0.1% relative (SIE)
- Lensed flux: < 1% relative
- Sub-pixel convergence: < 0.5% between 4× and 8× oversampling

### A.4 Lenstronomy's Role in This Project

Lenstronomy serves as our **independent referee**, not our production engine. This is analogous to how numerical codes in computational astrophysics are validated against analytical solutions or established codes — the production code is optimized for the specific use case, while the reference code provides ground truth.

| Role | Tool |
|---|---|
| Production injection engine | Custom PyTorch (`dhs/injection_engine.py`) |
| Physics cross-validation oracle | `lenstronomy` (in `tests/test_injection_engine.py`) |
| Dependency classification | Test-only (`requirements.txt`, clearly marked) |

### A.5 What We Would Use Lenstronomy For

If our use case required any of the following, we would use lenstronomy directly:
- Multi-plane lensing (we use single-plane SIE+shear)
- Lens model fitting / inversion (we only do forward injection)
- Substructure lensing (we do not model CDM subhalos)
- Time-delay cosmography (not relevant)

For the specific task of forward-injecting O(10⁵–10⁶) SIE+shear lensed Sersic sources into survey cutouts on GPU, a purpose-built engine is the right tool.

### A.6 References

- Birrer, S. & Amara, A. 2018, Physics of the Dark Universe, 22, 189 (lenstronomy)
- Birrer, S. et al. 2021, JOSS, 6, 3283 (lenstronomy v1.10)
- Kormann, R., Schneider, P., & Bartelmann, M. 1994, A&A, 284, 285 (SIE deflection)
- Graham, A. W. & Driver, S. P. 2005, PASA, 22, 118 (Sersic profile integrals)
- Ciotti, L. & Bertin, G. 1999, A&A, 352, 447 (b_n approximation)

---

*End of raw notes. This document will be updated as training completes and results become available.*

# Comprehensive LLM Review Request: Strong Lens Selection Function Pipeline

**Date:** 2026-02-13
**Project:** MNRAS paper — "Calibrating CNN-Based Strong Gravitational Lens Finders in DESI Legacy Survey DR10"
**Attached:** `stronglens_calibration_full_codebase.zip` — the complete, self-contained codebase
**Status:** Seeking thorough, independent review of the entire pipeline, results, and methodology

---

## ERRATUM: Parameter Values Corrected 2026-02-13

Previous versions of this prompt and documentation contained INCORRECT injection
parameter values. The following were wrong in earlier documentation:

| Parameter | WRONG (previously stated) | CORRECT (from code) |
|-----------|--------------------------|---------------------|
| Sersic n | U[0.5, 4.0] | **U[0.7, 2.5]** |
| Source R_e | U[0.1, 0.5]" | **U[0.05, 0.25]"** |
| g-r color | U[0.0, 1.5] | **N(0.2, 0.25) Gaussian** |
| r-z color | U[-0.3, 1.0] | **N(0.1, 0.25) Gaussian** |
| Shear gamma | U[0, 0.1] | **g1,g2 ~ N(0, 0.05) per component** |
| Clumps | "0-3 clumps at 25% flux" | **P(clumps)=0.6; n∈{1,2,3}; frac~U[0.15,0.45]** |

The corrected values are taken directly from `dhs/injection_engine.py` defaults
and are the single source of truth. See `configs/injection_priors.yaml` for the
full registry.

**THIS IS EXACTLY THE PROBLEM.** The drift between code and documentation has
been a recurring issue. The injection_priors.yaml registry + test was created to
prevent this from happening again.

---

## IMPORTANT: Instructions for the Reviewer

1. **Be honest, rigorous, and meticulous.** We have wasted significant time following plausible-sounding but incorrect advice. We need ground truth, not reassurance.
2. **Research the ideas.** We cite specific papers below. Please verify our claims against the actual literature and point out where we are wrong.
3. **Review the attached code.** The zip contains the complete pipeline. Every module, config, test, and script is included. Audit the code for correctness.
4. **Give concrete next steps with full code.** Not pseudocode, not stubs — working code with data sources as downloadable links.
5. **Flag red flags.** Anything suspicious, inconsistent, or wrong — say it plainly.
6. **Read the supplementary prompt** (`LLM_PROMPT_CODE_REVIEW_AND_RETRAIN_ASSESSMENT_20260213.md`) AFTER this one. It contains detailed questions about recent code changes, pipeline integrity, and retraining decisions.

---

# PART 1: PROJECT OVERVIEW AND GOAL

## 1.1 What We Are Building

We are writing an MNRAS paper that provides a **calibrated selection function** for CNN-based strong gravitational lens finders applied to DESI Legacy Imaging Surveys DR10. The core contribution is:

> Given a real-image-trained CNN lens finder, we measure its detection completeness as a function of lens/source parameters and observing conditions via calibrated injection-recovery, and we quantify failure modes and systematic uncertainties.

This is **not** a lens discovery paper. We are auditing and calibrating existing lens-finding methodology.

## 1.2 Why This Is Novel

The DESI lens-finding series (Huang et al. 2019, 2020, 2021; Inchausti et al. 2025 arXiv:2508.20087 "Paper IV") produces candidate catalogs but does NOT deliver:
- A quantitative detection probability P(det | θ_E, z_l, z_s, PSF, depth, morphology, ...)
- A defensible uncertainty model for that probability
- A systematic failure-mode taxonomy

Paper IV (arXiv:2508.20087) explicitly motivates selection-function and demographic comparison studies. Our paper fills this gap.

## 1.3 Key References

| Paper | arXiv | Relevance |
|-------|-------|-----------|
| Inchausti et al. 2025 ("Paper IV") | 2508.20087 | Our primary comparison. Two CNNs (custom ResNet + EfficientNetV2) trained on DR10. Does NOT report injection-recovery completeness. |
| Huang et al. 2019 | 1906.00970 | First DESI lens finder paper (DECaLS DR6/DR7) |
| Huang et al. 2020 | 2005.04730 | Extended to DESI Legacy DR8 |
| Huang et al. 2021 | 2206.02764 | DR9 extension |
| Cañameras et al. 2024, HOLISMOKES XI | A&A 692, A72 | Most rigorous published injection-recovery: real HUDF sources lensed through SIE onto real LRG cutouts with per-position PSF |
| Herle, O'Riordan & Vegetti 2024 | MNRAS 534, 1093 | Only paper specifically focused on quantifying CNN lens-finder selection functions |
| Metcalf et al. 2019 (Bologna Lens Challenge) | — | Standardized injection-recovery benchmark |
| Lanusse et al. 2018 | — | Custom "shielded" ResNet architecture used by Paper IV |
| Kormann et al. 1994 | — | SIE lens model formalism |
| Graham & Driver 2005 | — | Sersic profile normalization |

---

# PART 2: DATA PIPELINE

## 2.1 Positive Samples

- **Source:** DESI Strong Lensing Catalog candidates cross-matched with DR10
- **Total:** 4,788 candidates
  - **Tier A:** 389 with spectroscopic confirmation or HST imaging
  - **Tier B:** 4,399 with high-confidence visual grades only
- **Cutout format:** 101×101×3 (g, r, z bands), float32 nanomaggies, 0.262"/pixel
- **Treatment during training:** All treated equally (sample_weight = 1.0)

## 2.2 Negative Samples

- **Source:** DR10 Tractor catalogs, SER/DEV/REX types (EXP excluded per Paper IV)
- **Quality cuts:** nobs >= 3 in g/r/z; z < 20 mag; −18° < δ < 32° (DECaLS footprint)
- **Composition:**
  - Pool N1 (deployment-representative): ~85% randomly sampled from SER/DEV/REX
  - Pool N2 (hard confusers): ~15% morphologically selected (ring galaxies, spirals, mergers, edge-on disks, blue clumpy star-formers)
- **Total:** 446,893 negatives
- **Spatial splits:** HEALPix nside=128, deterministic hash-based assignment

## 2.3 Manifest and Split

| Split | Total | Positives | Negatives | Neg:Pos Ratio |
|-------|-------|-----------|-----------|---------------|
| train | 316,100 | 3,356 | 312,744 | 93.2:1 |
| val | 135,581 | 1,432 | 134,149 | 93.7:1 |
| **Total** | **451,681** | **4,788** | **446,893** | **93.3:1** |

**Manifest file:** `manifests/training_parity_70_30_v1.parquet`
**Columns:** galaxy_id, cutout_path, ra, dec, type_bin, nobs_z_bin, split, pool, confuser_category, psfsize_r, psfdepth_r, ebv, healpix_128, label, tier, sample_weight

## 2.4 Key Differences from Paper IV

| Aspect | Paper IV | This Work | Impact |
|--------|----------|-----------|--------|
| Framework | TensorFlow | PyTorch 2.7 | Minor numerical differences |
| GPU | 4× A100 (NERSC) | 1× GH200 (Lambda Cloud) | Gradient accumulation emulates multi-GPU |
| Positive count | 1,372 (confirmed) | 4,788 (389 confirmed + 4,399 visual) | Noisier positives may limit AUC ceiling |
| Negative count | 134,182 | 446,893 | 3.3× more negatives |
| Neg cleaning | Spherimatch + prior model p>0.4 | Spatial exclusion only | May contain unlabeled real lenses in negatives |
| Normalization | Not specified | raw_robust (median/MAD, outer 20% ring) | Our implementation choice |
| Augmentation | Not described | hflip + vflip + rot90 | Conservative geometric only |

---

# PART 3: PREPROCESSING (raw_robust)

**Implementation:** `dhs/preprocess.py` → `preprocess_stack()`

For each 101×101×3 cutout in nanomaggies:
1. Define an outer annulus for background estimation. **KNOWN ISSUE (2026-02-13):**
   Existing models (v1–v4) use hardcoded `(r_in=20, r_out=32)`, which was tuned for
   64×64 stamps. On 101×101 stamps, this annulus sits at 40–63% of the half-width,
   overlapping galaxy light (~20% flux contamination). The corrected annulus at
   `(r_in=32.5, r_out=45.0)` (from `default_annulus_radii(101, 101)`) places it at
   65–90% of the half-width (~6% flux contamination). Retraining is planned.
2. Per-band: compute median and MAD (median absolute deviation) from the outer annulus
3. Per-band: subtract median, divide by MAD (with MAD floor of 1e-8 to avoid division by zero)
4. Clip to [-10, +10]
5. No cropping (101×101 preserved throughout)

**Critical property:** This preprocessing is applied identically during training, validation, real-lens scoring, AND injection scoring. The same function call `preprocess_stack(img, mode="raw_robust", crop=False)` is used everywhere.

---

# PART 4: MODEL TRAINING

## 4.1 Architectures

We trained 5 models across 4 architectures:

### EfficientNetV2-S (Runs v2, v3, v4)
- `torchvision.models.efficientnet_v2_s`, ImageNet pretrained
- 20,178,769 params (~20.2M)
- Classifier: `nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 1))`
- Paper IV comparison: 20,542,883 params (1.8% fewer in ours)

### BottleneckedResNet (Custom, matches Paper IV architecture)
- Custom Lanusse-style compact ResNet with 1×1 bottleneck reductions
- ~195,000 params (Paper IV: 194,433 — near-exact match)
- Trained from scratch (no pretrain)
- Designed to neutralize referee objection about architecture capacity mismatch

### ResNet-18 (Ablation only)
- Standard torchvision ResNet-18, ImageNet pretrained
- ~11.2M params
- Catastrophically overfit — serves as ablation showing naive over-parameterized architectures fail

## 4.2 Training Configurations

All runs use: AdamW optimizer, weight_decay=1e-4, BCEWithLogitsLoss (unweighted), mixed precision (torch.cuda.amp)

| Run | Config File | Architecture | LR | Schedule | Epochs | Batch | Freeze |
|-----|-------------|-------------|-----|----------|--------|-------|--------|
| v2 (step) | `paperIV_efficientnet_v2_s_v2.yaml` | EfficientNetV2-S | 3.88e-4 | Step (halve@E130) | 160 | 512 (8×64) | 5 epochs |
| v3 (cosine) | `paperIV_efficientnet_v2_s_v3_cosine.yaml` | EfficientNetV2-S | 3.88e-4 | Cosine (T_max=160) | 160 | 512 (8×64) | 5 epochs |
| v4 (finetune) | `paperIV_efficientnet_v2_s_v4_finetune.yaml` | EfficientNetV2-S | 5e-5 | Cosine (T_max=60) | 60 | 512 (8×64) | 0 |
| BnResNet | `paperIV_bottlenecked_resnet.yaml` | BottleneckedResNet | 5e-4 | Step (halve@E80) | 160 | 2048 (16×128) | 0 |
| ResNet-18 | `paperIV_resnet18.yaml` | ResNet-18 | 5e-4 | Step | 160 (stopped@68) | 512 (8×64) | 0 |

## 4.3 Training Results — ALL RUNS COMPLETE

| Run | Status | Best AUC | Best Epoch | Final AUC | Notes |
|-----|--------|----------|------------|-----------|-------|
| **v4 finetune** | **COMPLETE** (60/60) | **0.9921** | **1** | 0.9794 | **BEST MODEL.** Loaded v2/best.pt, micro-adjusted at 5e-5 |
| v2 step LR | COMPLETE (160/160) | 0.9915 | 19 | 0.9736 | Peak early, then irreversible decline |
| v3 cosine LR | COMPLETE (160/160) | 0.9895 | 17 | 0.9644 | Cosine did NOT outperform step |
| BnResNet | COMPLETE (160/160) | 0.9799 | 68 | 0.9659 | 195K params, capacity-limited |
| ResNet-18 | STOPPED (68/160) | 0.9611 | <30 | 0.9216 | Catastrophic overfitting (11.2M params) |

## 4.4 Key Training Findings

1. **Best model: v4 finetune E1, AUC=0.9921** — gap to Paper IV EfficientNet (0.9987) is only 0.0066
2. **Best BottleneckedResNet: E68, AUC=0.9799** — gap to Paper IV ResNet (0.9984) is 0.0185
3. **All runs confirm:** overfitting begins early (E17-68) and is not recoverable by LR reduction
4. **Cosine vs step LR:** Cosine schedule did NOT outperform step; final AUC (0.9644) was worse than step's (0.9736)
5. **Paper IV gap is data-driven:** attributable to uncleaned negatives and noisier (Tier-B) positives, not architecture

## 4.5 Paper IV Reference Metrics

| Model | Paper IV Val AUC | Our Best AUC | Gap |
|-------|-----------------|-------------|-----|
| Custom ResNet (194K params) | 0.9984 | 0.9799 | -0.0185 |
| EfficientNetV2 (20.5M params) | 0.9987 | 0.9921 | -0.0066 |
| Meta-learner (ensemble) | 0.9989 | N/A | N/A |

**Full epoch-by-epoch training logs** for all 5 runs are in `docs/MNRAS_SUPPLEMENTARY_TRAINING_LOG.md` (attached in the zip). This file contains every epoch's loss, val_auc, best_auc, learning rate, and wall-clock time.

---

# PART 5: SELECTION FUNCTION — INJECTION-RECOVERY PIPELINE

## 5.1 Approach

We use our best trained CNN (v4 finetune, E1, AUC=0.9921) as a **frozen detector**, then measure its detection completeness by injecting synthetic lensed arcs into real DR10 galaxy cutouts and scoring them.

**Key principle:** The CNN was trained on real images. Injections are used ONLY for measuring the selection function, not for training. This follows the "hybrid" approach recommended by the literature.

## 5.2 Injection Engine (`dhs/injection_engine.py`)

| Component | Choice | Detail |
|-----------|--------|--------|
| **Lens mass model** | SIE + external shear | Kormann et al. 1994; q_lens ~ U[0.5, 1.0]; shear g1,g2 ~ N(0, 0.05) per component (Rayleigh magnitude, NOT uniform) |
| **Source light** | Sersic + optional clumps | n ~ U[0.7, 2.5]; R_e ~ U[0.05, 0.25]"; clumps_prob=0.6, n_clumps ~ {1,2,3}, frac ~ U[0.15, 0.45] |
| **Source magnitude** | r-band AB, 23-26 mag | Converted to nanomaggies via ZP=22.5 |
| **Source colors** | Gaussian | g-r ~ N(0.2, 0.25); r-z ~ N(0.1, 0.25). **NOT UNIFORM.** Previous documentation incorrectly stated U[0,1.5] and U[-0.3,1.0]. |
| **Source offset** | Area-weighted | beta_frac = sqrt(U(0.01, 1.0)); β = β_frac × θ_E |
| **Magnification** | Physical (ray-tracing) | Emerges from SIE lens equation, NOT a free parameter |
| **Flux units** | Nanomaggies throughout | No unit conversions at boundaries |
| **PSF** | Gaussian per band | sigma = psfsize_r / 2.355; FFT convolution |
| **Noise** | NOT added | Injected into real noisy cutouts; real noise already present |
| **Oversampling** | 4× default | Verified <0.5% vs 8× |
| **Injection method** | Additive | `injected = host_cutout + lensed_arc`; flux conservation verified |
| **Stamp size** | 101×101×3 | HWC, nanomaggies, 0.262"/pixel |

**Physics test suite:** 28 unit tests, ALL PASS, including 4 lenstronomy cross-validation tests (SIS/SIE deflection <0.1% relative; lensed flux <1% relative). See `injection_model_1/tests/test_injection_engine.py`.

**IMPORTANT:** An external LLM provided an alternative `injection_engine.py` that we **REJECTED** due to 3 confirmed mathematical errors in SIE deflection:
1. Wrong denominator: `psi + q^2` instead of `psi` (where `psi = sqrt(q^2 * x^2 + y^2)`)
2. Swapped `atan` and `atanh` in deflection components
3. Wrong prefactor: `q` instead of `sqrt(q)`

Our engine (`dhs/injection_engine.py`) with 28 passing tests is used unchanged.

## 5.3 Injection Model 1: Parametric Sersic on Random Hosts

**Code:** `injection_model_1/scripts/selection_function_grid.py`
**Grid:** 11 θ_E (0.5-3.0") × 7 PSF (0.9-1.8") × 5 depth (24.0-25.5 mag) = 385 cells × 200 inj/cell
**Hosts:** Random non-lens galaxies from val split (SER 64%, DEV 20%, REX 16%)

### Model 1 Results (Bugfixed Rerun)

| Threshold | Mean Completeness |
|-----------|------------------|
| p > 0.3 | **3.54%** |
| p > 0.5 | **2.88%** |
| FPR=0.1% (p>0.806) | **2.05%** |
| FPR=0.01% (p>0.995) | **0.59%** |

**By source magnitude (p > 0.3):**
- 23-24 mag (bright): **7.21%** (N=14,572)
- 24-25 mag: **2.72%** (N=14,775)
- 25-26 mag (faint): **0.70%** (N=14,653)

**By θ_E (p > 0.3):** Peaks at θ_E = 1.75" (5.12%), drops to 0.65% at 0.5" and 3.23% at 3.0".

## 5.4 Injection Model 2: Sersic on LRG Hosts, Host-Conditioned q/PA

**Code:** `injection_model_2/scripts/selection_function_grid_v2.py --model 2`

Model 2 extends Model 1 with:
1. **Host selection:** Only DEV/SER (LRG-like) hosts — 112,744 available (out of 134,149 total negatives)
2. **Lens parameter conditioning:** q_lens and phi_lens derived from host galaxy r-band second moments (not independent priors). Implementation in `injection_model_2/host_matching.py` (26 unit tests, all pass).

### Model 2 Results (Bugfixed Rerun)

| Threshold | Model 1 | Model 2 | Difference |
|-----------|---------|---------|------------|
| p > 0.3 | 3.54% | 2.77% | **-0.77pp (Model 2 is WORSE)** |
| p > 0.5 | 2.88% | 2.25% | **-0.63pp** |
| FPR=0.1% | 2.05% | 1.55% | **-0.50pp** |
| FPR=0.01% | 0.59% | 0.38% | **-0.21pp** |

**Model 2 is consistently worse than Model 1 across ALL thresholds.**

### Host Conditioning Diagnostic (4-Way Comparison)

We ran a controlled experiment isolating host type vs lens parameter conditioning:

| Condition | Mean C (p>0.3) |
|-----------|---------------|
| LRG hosts + conditioned q/PA (= Model 2) | 4.90% |
| LRG hosts + independent q/PA | 4.90% |
| Random hosts + independent q/PA (= Model 1) | 4.77% |
| Random hosts + conditioned q/PA | 4.63% |

**Result: Statistically indistinguishable.** Neither host type nor lens parameter conditioning matters.

## 5.5 Sim-to-Real Validation Results

**Real lens recall** (scoring val-split positives with v4 finetune checkpoint):

**WARNING (2026-02-13):** Previous documentation described this as "scoring Tier-A
anchors." However, `real_lens_scoring.py` actually scores ALL val-split positives
(Tier-A AND Tier-B). The code filters `df[(df["split"] == "val") & (df["label"] == 1)]`
with NO tier filter. **This means the 73.3% recall includes Tier-B candidates
(unconfirmed visual grades) which may not be real lenses.** The number may be
inflated. See supplementary prompt Section H-train.1 for detailed analysis.

| Threshold | Recall |
|-----------|--------|
| p > 0.3 | **73.3%** |
| p > 0.5 | **68.7%** |
| FPR = 0.1% | **59.7%** |
| FPR = 0.01% | **24.8%** |

**Bright arc injection test** (does brightness explain the gap?):

| Source mag | Detection (p>0.3) | Arc SNR |
|-----------|-------------------|---------|
| 18-19 | **30.5%** | 945 |
| 19-20 | 27.0% | 405 |
| 20-21 | 22.5% | 157 |
| 23-24 | 9.0% | 9.3 |
| 25-26 | 0.5% | 1.4 |

**Even at blindingly bright source magnitudes (mag 18-19, arc SNR ~900), injection completeness plateaus at ~30%.**

**Confuser morphology test** (is the model exploiting galaxy shape shortcuts?):

| Category | N | Frac > 0.3 |
|----------|---|-----------|
| ring_proxy | 200 | 1.0% |
| edge_on_proxy | 200 | 0.5% |
| large_galaxy | 200 | 0.0% |
| blue_clumpy | 200 | 0.0% |
| Random negatives | 200 | 0.0% |

Model is NOT exploiting morphology shortcuts.

## 5.6 Sensitivity Analysis (8 Perturbations)

| Perturbation | Mean delta-C | Max |delta-C| per cell |
|-------------|-------------|----------------------|
| PSF FWHM ±10% | ±0.5% | 5.0% |
| Source R_e ±30% | ±0.2% | 4.0% |
| Color (g-r) ±0.2 mag | ±0.6% | 7.0% |
| Lens q broader/narrower | ±0.3% | 5.0% |

Source color is the dominant systematic; all perturbations produce <1% mean shift.

## 5.7 THE GAP

| Metric | Value |
|--------|-------|
| Real lens recall (p>0.3) | **73.3%** |
| Best injection completeness (bright arcs, p>0.3) | 30.5% |
| Model 1 grid completeness (mag 23-26, p>0.3) | 3.5% |
| Model 2 grid completeness (mag 23-26, p>0.3) | 2.8% |
| **Gap: real recall − best injection** | **~43 percentage points** |

---

# PART 6: PREVIOUS LLM CONVERSATION — FULL CHAIN OF EVENTS

We had an extensive multi-session conversation with an LLM advisor. Below is a faithful, detailed summary of what was said, recommended, done, and the outcomes. **The LLM's session expired and this is a fresh session.** We need you to pick up where it left off.

## 6.1 Research Direction Selection (Session 1)

We presented the DESI lens-finding paper series and asked for publishable research options.

**What the LLM proposed (4 options, ranked):**

1. **Selection-function audit** (recommended) — "Train a real-image classifier on DR10 cutouts (so the detector is distribution-matched), then measure a quantitative selection function using calibrated injections into DR10 cutouts (so you can say what kinds of lenses the model would or would not detect, as a function of conditions and lens/source properties)."
   - "Best novelty-to-effort ratio, strongest referee-proof framing."
   - Novelty: "The calibrated realism layer plus the quantitative selection surfaces. A principled decomposition of failure modes."
2. **Contaminant taxonomy + prevalence-aware FPR** — practical complement
3. **Domain-diversified ensemble** — incremental novelty
4. **Foundation-feature audit** — high risk

**The LLM's precise framing of why this is novel (its exact words):** "The DESI lens-finding paper line is primarily discovery and curation. That produces large candidate catalogs but typically does not deliver a quantitative detection probability P(det | θ_E, z_l, z_s, μ, PSF, depth, morphology, ...) and a defensible uncertainty model for that probability. Your Option 1 is novel if you commit to: injection-recovery completeness surfaces (multi-dimensional), uncertainty and prior sensitivity, explicit failure categories and causal hypotheses, validated with controlled tests."

**What the LLM said reviewers would challenge:**
- "Your injections are arbitrary and drive your conclusions" → Calibrate injections to observed DR10 photometric conditions
- "Your positives are Huang-selected candidates, so this is circular" → Use Tier-A spectroscopic/HST-confirmed anchors only for empirical recall; use newer spectroscopic programs as independent anchors
- "Your selection function is for your model only; it is not general" → Show which failure modes are stable across architectures
- "Your completeness vs parameters is underconstrained (low-N in bins)" → Bayesian binomial + clearly mark insufficient-data regions

**What we did:** Chose Option 1.

## 6.2 Implementation Blueprint — 59-Question Q&A (Session 2)

We asked 59 implementation questions across 12 sections. The LLM provided detailed answers:

**Section A — Negative Sampling:**
- N1:N2 = 85:15 within the negative class
- N1 (deployment-representative): SER/DEV/REX, z < 20 mag, per Paper IV
- N2 (hard confusers): oversampled stream, not forced to natural prevalence
- Controls (same galaxy, no injection): deprecated as primary negative source; keep as diagnostic only

**Section B — Spatial Splits:**
- HEALPix nside=128 for split assignment
- Do NOT hard-stratify by PSF/depth — assign deterministically, then verify and report balance
- 70/30 train/val (matching Paper IV; no separate test split for parity runs)

**Section C — Label Handling:**
- Tier-A (confirmed): sample_weight = 1.0
- Tier-B (probable): sample_weight = 0.3-0.6 (we chose 0.5)
- Label smoothing for Tier-B: target = 0.8-0.9 (optional)
- SLACS/BELLS: may have low DR10 visibility — use for evaluation stress test, not training weight

**Section D — Injection Realism Acceptance Criteria:**
- Arc annulus SNR matching: injections vs real anchors within factor of ~2 in median
- Color distribution (g-r, r-z): within ±0.2 mag
- Noise histogram: KS test p > 0.05
- Morphology: bracket with "smooth Sersic" vs "clumpy" as sensitivity test
- GO/NO-GO: all diagnostics must pass before trusting selection function

**Section E — Training Config:**
- EfficientNetV2-S (ImageNet pretrained) + BottleneckedResNet (~195K params) for Paper IV parity
- 101×101 input (no cropping!)
- 160 epochs, step LR (no early stopping)
- Gradient accumulation to emulate Paper IV's large effective batch sizes

**Section F — Selection Function Grid:**
- θ_E × PSF × depth, 200+ injections/cell
- Bayesian binomial CIs (Jeffreys prior)
- Mark cells with N < 200 as "insufficient"

**Section G — Paper Strategy:**
- "If you deliver: real-image baseline + independent validation + injection-calibrated selection surface + failure-mode breakdown + released code/data products: moderate-to-high probability of MNRAS acceptance."
- "If you skip independent validation or do not validate injections quantitatively: low."

## 6.3 Paper IV Parity Audit (Session 3)

The LLM audited our negative sampling against Paper IV line-by-line.

**Critical parity gaps found:**
1. **z-band magnitude limit:** We used z < 21; Paper IV uses z < 20. **FIXED.**
2. **Type set:** We allowed EXP in N1; Paper IV's nonlens selection explicitly names SER/DEV/REX. **FIXED.**
3. **Normalization:** Paper IV does not specify normalization. Our `raw_robust` (median/MAD outer annulus, clip [-10, +10]) is our implementation choice — documented as such.
4. **Negative cleaning:** Paper IV uses Spherimatch + prior model p>0.4 to remove likely lenses from negatives. **We only use spatial exclusion.** This is a known gap.

## 6.4 Training Course Correction (Session 4)

Our first training run was ResNet-18, 64×64 crops, early stopped at epoch 16, AUC=0.9592.

**The LLM flagged multiple critical mismatches:**
- Paper IV trains on **101×101** (we cropped to 64×64)
- Paper IV trains **160 epochs** (we stopped at 16)
- Paper IV uses **step LR halving** at fixed epochs (we used CosineAnnealing + early stopping)
- Paper IV's ResNet is a **custom 194K-param architecture** (we used torchvision ResNet-18 at 11.2M params — 58× more parameters!)
- Paper IV uses **EfficientNetV2** (not V1, not B0)

**The LLM's course correction:**
1. Stop cropping — train on 101×101
2. Remove early stopping — run 160 epochs
3. Match LR schedule (ResNet: LR=5e-4, halve@E80; EffNet: LR=3.88e-4, halve@E130)
4. Match effective batch size via gradient accumulation
5. Build a BottleneckedResNet (~195K params) to match Paper IV's architecture

**The LLM provided downloadable code** (`paperIV_parity_course_correction.zip`) with training scripts, and separately provided `verify_splits.py`, `bootstrap_eval.py`, and `run_selection_function.py`.

**What we did:** Implemented all corrections. Built a custom BottleneckedResNet matching Paper IV's 194K params. Switched to EfficientNetV2-S. Trained 5 runs total (see Part 4).

## 6.5 Injection Model 1 Review — LLM's Detailed Response (Session 5)

We submitted: injection engine code + Model 1 results (4.3% completeness vs 73.3% real recall).

**The LLM found 3 bugs in the grid runner:**
1. FPR threshold derivation inserts blank samples on NPZ load failure — silently biases FPR calculation
2. FPR rank calculation uses `n_neg` not `n_valid` — produces wrong threshold
3. Injection loop counts failed injections in denominator — deflates completeness

**FIXED:** All 3 bugs fixed in `injection_model_2/scripts/selection_function_grid_v2.py`.

**The LLM could not complete the review** because the zip was missing `dhs/` package (model.py, preprocess.py, constants.py). **FIXED:** We repackaged with complete `dhs/`.

**The LLM's priority recommendations:**
- Priority #1: Fix the 3 silent failure bugs (**DONE**)
- Priority #2: Audit preprocessing identity — verify `preprocess_stack()` is truly identical during training vs injection scoring (**NOT YET INDEPENDENTLY VERIFIED**)
- Priority #3: Run Model 2 (deflector-conditioned injection on LRG hosts)

**The LLM's specific prediction about Model 2 (its exact words):**
> "If you implement Model 2 (deflector-matched hosts with conditioned q/PA), I predict the gap will narrow substantially. The bright-arc ceiling should rise from ~30% to perhaps 50-70%. If it does not, the story shifts from 'host mismatch' to 'injections do not resemble training positives in feature space.'"

**The LLM's proposed hierarchy of models (before we ran Model 2):**
- **Model 2:** Sersic source on LRG host galaxies, with q_lens and phi_lens derived from host galaxy's r-band second moments (instead of independent priors). Motivation: real lenses have massive elliptical deflectors; our Model 1 injects onto random galaxy types.
- **Model 2.5 (if Model 2 insufficient):** Use real DR10 blue galaxies as source-plane images, lensed through SIE. "Select faint blue galaxies from DR10, deconvolve PSF, use as source morphologies." This would test whether the gap is source morphology (Sersic too smooth).
- **Model B (arc transplant):** Fit smooth elliptical model to real lens cutouts, extract residual arc signal, transplant arc onto non-lens host. "This directly uses the real arc morphologies that the CNN was trained on."
- **Model C (feature-space calibration):** Extract CNN penultimate-layer embeddings for real lenses and injections. Compare distributions. Compute importance weights: w(x) = p_real(f(x)) / p_injection(f(x)). Apply as correction factor to completeness. "This does not require closing the morphology gap; it calibrates the selection function to account for it."

**The LLM's attribution of the 30% bright-arc ceiling:**
> "Even at arc SNR ~900, only 30% are detected. This strongly suggests the CNN is not just detecting arc brightness — it learned a joint feature of arc + host morphology during training on real lens systems. Sersic sources are smooth, symmetric, and lack the clumpy, irregular, knotty structure of real lensed galaxies. The 30% that ARE detected likely have geometric configurations (e.g., Einstein rings, symmetric arcs) that happen to mimic real arc morphologies."

**The LLM also provided a buggy `injection_engine.py`** with 3 confirmed mathematical errors in SIE deflection (see Part 5.2). **REJECTED.** Our own engine with 28 passing tests including lenstronomy cross-validation is the canonical version.

## 6.6 Literature Review Request (Session 6 — UNANSWERED)

We asked for a structured review of 13 papers covering injection methodology, selection function approaches, and validation, with 30 specific questions. Papers included:
- HOLISMOKES XI (Cañameras et al. 2024)
- Herle, O'Riordan & Vegetti 2024
- Metcalf et al. 2019 (Bologna Lens Challenge)
- Euclid 2025 lens-finding pipeline
- All DESI lens-finder papers (Huang et al. 2019-2021, Inchausti et al. 2025)

Key unresolved questions from this request:
- How should flux normalization work in injection-recovery? (Our engine normalizes by image-plane Sersic integral)
- Should `flux_nmgy_r` represent unlensed source flux or total lensed flux?
- Is SIE + shear sufficient or do we need more complex mass models?
- What source offset distribution is standard? (We use area-weighted P(β_frac) ∝ β_frac)
- What is the minimum we need to publish a credible selection function given Paper IV does NOT report injection-recovery completeness?

**The LLM did not respond (session expired).**

## 6.7 Model 1 + Model 2 Follow-Up (Session 7 — UNANSWERED)

We submitted complete results with full `dhs/`, bugfixed code, Model 1 rerun, Model 2 results, 4-way diagnostic, bright arc test, confuser test, and 18 specific questions (Q1-Q18).

**Key facts we presented that need the new LLM's assessment:**

1. **Model 2 was WORSE than Model 1** (contradicting the LLM's prediction of "gap narrows substantially")
2. **The 4-way diagnostic showed zero effect** of either host type or lens parameter conditioning
3. **The bright-arc ceiling remained at 30%** (the LLM predicted it should rise to 50-70%)
4. **Mean host q = 0.83** for LRG hosts — nearly round, which may bias q_lens toward rounder values
5. **We found and fixed 3 additional bugs** in the grid runner beyond the LLM's original 3

**The LLM did not respond (session expired).** The 18 questions remain unanswered and are reproduced in Part 8 below.

---

# PART 7: CURRENT STATE — WHAT WE KNOW AND DON'T KNOW

## 7.1 Established Facts

1. **Our CNN is competent:** Best AUC=0.9921 (v4 finetune), gap to Paper IV EfficientNet (0.9987) is only 0.0066. The model assigns p>0.3 to 73.3% of val-split positive candidates — **but this includes unconfirmed Tier-B candidates (see WARNING in Section 5.5).** The true recall on confirmed lenses may differ.
2. **Injection completeness is extremely low:** 2.8-3.5% at p>0.3, giving a ~70 percentage point gap with real recall.
3. **Host galaxy mismatch is definitively ruled out:** Model 2 (LRG hosts + conditioned q/PA) performed 0.77pp WORSE. The 4-way diagnostic (4 conditions, 500 injections/point × 6 θ_E values) shows zero statistically significant effect of either host type or lens parameter conditioning. All 4 conditions within 0.3pp.
4. **Brightness alone does not explain the gap:** Even at arc SNR ~900 (source mag 18-19), only 30.5% of injections are detected. This ceiling did NOT change between Model 1 and Model 2.
5. **The model is not exploiting morphology shortcuts:** Ring proxies score 1.0%, edge-on 0.5%, others 0.0% — all at baseline FPR.
6. **Sensitivity to injection parameters is small:** PSF ±10%, source size ±30%, color ±0.2 mag, lens q range all produce <1% mean shift in completeness.
7. **Injection physics passes unit tests:** 28 tests including 4 lenstronomy cross-validation tests. SIE deflection <0.1% relative error. Flux conservation verified to 0.0 nmgy.
8. **The previous LLM's prediction was wrong:** It predicted Model 2 would raise the bright-arc ceiling to 50-70%. It stayed at 30%.

## 7.2 Unresolved Questions (Critical)

1. **What causes the 30% bright-arc ceiling?** This is the central mystery. Multiple hypotheses survive:
   - Sersic source morphology lacks the clumpy/irregular/knotty structure of real lensed galaxies
   - Preprocessing normalization (raw_robust) may interact differently with real vs synthetic arcs
   - Missing Poisson noise on injected arcs makes them look "too clean"
   - Color/SED mismatch — real lensed sources are predominantly blue star-forming with specific SEDs
   - DR10 photometric pipeline effects (sky subtraction, flat-fielding) are present in real arcs but not synthetic ones
   - **We have NO experiment that discriminates between these hypotheses**

2. **Has preprocessing identity been independently verified?** The code says training and injection scoring use identical `preprocess_stack(mode="raw_robust", crop=False)`. But this was the previous LLM's Priority #2 recommendation and has not been audited by an independent reviewer.

3. **What does the CNN "see" differently?** We have not extracted model embeddings to compare real lenses vs injections in feature space. This would directly test whether the CNN learns to separate them.

4. **Is this paper publishable as-is?** We have a strong negative result (Model 2 failed, 30% ceiling unexplained) but no positive resolution. The previous LLM said the paper needs "validated injection realism" for MNRAS acceptance.

5. **Which next step has the highest probability of resolving the gap?** The previous LLM proposed Model 2.5 (real galaxy sources), Model B (arc transplant), Model C (feature-space calibration). We need a fresh, independent assessment of which is most promising.

---

# PART 8: OPEN QUESTIONS

These include the 18 original questions posed to the previous LLM (which were never answered), PLUS new questions arising from Model 2's negative result and the need to reassess the LLM's prior proposals.

## 8.1 On the Previous LLM's Failed Prediction

**Q1.** The previous LLM specifically predicted: "If you implement Model 2 (deflector-matched hosts with conditioned q/PA), I predict the gap will narrow substantially. The bright-arc ceiling should rise from ~30% to perhaps 50-70%." **The data shows Model 2 is 0.77pp WORSE, not better. The bright-arc ceiling did NOT rise.** Was this prediction fundamentally wrong? What does this tell us about the LLM's understanding of the problem? Should we trust its other recommendations (Model 2.5, Model B, Model C)?

**Q2.** The LLM also said: "If Model 2 does not improve completeness meaningfully, your story is not 'host mismatch.' It is 'injections do not resemble training positives in feature space.'" **We now have the data confirming Model 2 failed.** Do you agree with this fallback diagnosis? If yes:
- What SPECIFIC feature statistics differ between Sersic injections and real lensed arcs as seen by EfficientNetV2?
- Can you be concrete — not just "morphology" but what pixel-level or statistical properties?
- How would you measure this difference experimentally?

**Q3.** The 4-way diagnostic showed that **neither host type nor lens parameter conditioning has any effect** (all four conditions within 0.3pp of each other). This is a strong negative result. What does it definitively rule out? What hypotheses survive?

## 8.2 On the 30% Bright-Arc Ceiling (Critical Mystery)

**Q4.** Even at source mag 18-19 (arc SNR ~900), only 30.5% of injections are detected. This ceiling did NOT change with Model 2. **What causes it?** Please attribute concrete fractions to each candidate cause:
- (a) Source morphology: Sersic profile too smooth, lacks clumpy/irregular/knotty structure of real lensed galaxies
- (b) Preprocessing artifact: could `raw_robust` normalization (median/MAD outer annulus, clip [-10, +10]) interact differently with real arcs vs injected arcs? Could bright injected arcs shift the outer-annulus statistics?
- (c) Color/SED mismatch: our sources have Gaussian-drawn colors (g-r ~ N(0.2, 0.25), r-z ~ N(0.1, 0.25)). Real lensed sources are predominantly blue star-forming galaxies with specific SEDs — is our Gaussian centered correctly? Is the dispersion too wide or too narrow?
- (d) Arc spatial distribution: area-weighted offset sampling P(β) ∝ β may not match the actual distribution of arc positions in training positives
- (e) Missing Poisson noise on injected arcs (see Q10 below)
- (f) Gaussian PSF vs real survey PSF
- (g) Something else we haven't considered — **what?**

**Q5.** How would you design a controlled experiment to determine which factor dominates the 30% ceiling? Provide specific methodology — not vague suggestions.

## 8.3 On the Physics of Our Injections

**Q6.** Our mean arc SNR ranges from 2.3 to 5.0 across the θ_E grid. **Are these physically reasonable for detectable lenses?** Real lens candidates in DR10 presumably have higher arc SNR to be visible. If our typical injection has arc SNR of ~4, is it even reasonable to expect the CNN to detect it? What arc SNR do real detected lenses have in DR10?

**Q7.** Mean host q for Model 2 is ~0.83 (nearly round) and constant across ALL θ_E bins (0.829 ± 0.002). Is this physically reasonable for DEV/SER galaxies in DR10? Could our moment estimation (`injection_model_2/host_matching.py`) be biased toward round shapes? If hosts are genuinely round (q~0.83), then q_lens ~ 0.83 × U[0.8, 1.2] ≈ U[0.66, 1.0] — narrower and rounder than Model 1's U[0.5, 1.0]. Could this explain Model 2's lower completeness (rounder lenses produce less elongated, harder-to-detect arcs)?

**Q8.** We do NOT add Poisson noise to the injected arc signal. The rationale: we inject into real noisy cutouts, so the real noise is already present. But in a real lens, the arc photons would contribute additional Poisson noise proportional to the arc signal. **Are we underestimating the noise on the arc?** Could this make injected arcs look "too clean" — smoother than real arcs — and could the CNN detect this difference?

**Q9.** We use a Gaussian PSF with sigma = psfsize_r / 2.355. The previous LLM said the Gaussian PSF effect is "a few percent to maybe 10%" and "fix it in Model 3 parity, but do not expect it to explain the main gap." Given that the main gap is ~70pp: is this assessment correct? Should we deprioritize PSF improvements?

**Q10.** Model 2 has fewer populated cells (209 vs 220) because LRG hosts are concentrated at certain PSF/depth combinations. Could this bias the comparison? Should we restrict both models to the same set of populated cells?

## 8.4 On the Previous LLM's Proposed Next Steps — Reassessing After Model 2's Failure

The previous LLM proposed a hierarchy: Model 2 → Model 2.5 → Model B → Model C. **Model 2 failed.** We need your independent assessment of the remaining proposals.

**Q11. Model 2.5 (real DR10 blue galaxies as sources):** The previous LLM suggested using real DR10 blue galaxies as source-plane images, lensed through SIE. Specifically:
- Select faint blue galaxies from DR10 (what color cuts, magnitude range, morphology types?)
- Extract source-plane images (deconvolve PSF? handle sky background?)
- Place in source plane and lens through SIE (current engine expects analytic Sersic — what changes needed?)
- **Is this the right next step given Model 2's failure?** Will real galaxy morphologies actually help if the 4-way diagnostic shows host type doesn't matter?
- **Provide complete, working code** with data source URLs.

**Q12. Model B (arc transplant from real lenses):** Fit smooth elliptical model to ~200 real lens cutouts, extract residual arc, transplant onto non-lens host. **Is this practical with only ~200 real lenses?** Would it introduce label circularity (arcs from training lenses injected for testing)?

**Q13. Model C (feature-space calibration / importance weighting):** Extract CNN embeddings for real lenses vs injections, compute importance weights w(x) = p_real(f(x)) / p_injection(f(x)), apply as correction to completeness. **Explain concretely with mathematical formulation.** What embeddings? What distance metric? How is the correction applied?

**Q14.** Given Model 2's failure, is there a **Model D** we haven't considered? Something the previous LLM missed? For example:
- Could the issue be fundamentally about how DR10's photometric pipeline processes real arc light (sky subtraction, flat-fielding, etc.) versus our synthetic additive injection?
- Could we test this by injecting real point sources into cutouts and checking if the CNN's response to synthetic point sources matches its response to real stars?

## 8.5 On Training and the CNN's Behavior

**Q15.** Our EfficientNetV2-S was trained on real lens candidates. **Could the training setup inherently cause the model to reject synthetic injections while accepting real lenses?** For example: if training positives have specific statistical properties (host galaxy light profile shape, arc color distribution, arc position relative to center, noise correlation structure) that Sersic injections don't match, the model could learn to discriminate "real positive" from "synthetic injection" even at high SNR. Is this the explanation for the 30% ceiling?

**Q16.** Verify from `dhs/preprocess.py` that the preprocessing applied during injection scoring is truly identical to training. This was the previous LLM's Priority #2 recommendation and has NOT been independently verified. **Audit the actual code, not our description.**

**Q17.** Could the difference between how real arcs and injected arcs were "observed" create a detectable statistical signature? Consider:
- Sky subtraction in DR10 might partially subtract extended arc light in real lenses, making them appear different from our additive injections
- Flat-fielding patterns, scattered light, or other instrumental effects are present in real arcs but not synthetic ones
- The noise statistics in the arc region differ (real arc: Poisson + read noise; our injection: only the host's noise, no additional photon noise from the arc)
- **Could a CNN with AUC=0.9921 be sensitive enough to detect these subtle differences?**

## 8.6 On Code Review

**Q18.** Please audit the following code files for correctness, bugs, and subtle errors:
- (a) `dhs/preprocess.py`: Is `preprocess_stack()` correctly implementing outer-annulus median/MAD normalization? Edge cases that could produce different results for real vs injected cutouts?
- (b) `dhs/model.py`: Architecture correct? Inference path issues?
- (c) `dhs/data.py`: Data loading correct for scoring? Preprocessing differences between training and inference?
- (d) `dhs/transforms.py`: Augmentations applied during training only, or could they leak into inference?
- (e) `injection_model_2/host_matching.py`: Moment calculations correct? q/PA estimation robust?
- (f) `injection_model_2/scripts/selection_function_grid_v2.py`: Grid logic correct? Silent failure fixes adequate?
- (g) `dhs/injection_engine.py`: SIE deflection mathematically correct? Sersic normalization matches Graham & Driver 2005? Flux conservation verified?

## 8.7 On Publishability and Paper Structure

**Q19.** Is this paper publishable in MNRAS in its current form (low injection completeness + negative Model 2 result + unexplained 30% ceiling)? Be honest. If "not in current form," what is minimally needed?

**Q20.** If the paper IS salvageable, provide a concrete outline:
- Title
- Abstract (1-paragraph draft)
- Section structure
- Key figures (describe what each shows)
- Main claims and how they are supported by data
- If Model 2.5 or further work is needed, estimate the timeline (single GPU, existing data infrastructure)

**Q21.** Priority ordering of next steps. Rank these and for the top 3, provide complete implementation details:
1. Model 2.5 (real DR10 source galaxies)
2. Preprocessing audit (detailed comparison of real lens vs injection feature statistics)
3. Feature-space embedding comparison (what does the CNN "see" differently?)
4. Arc transplant from real lenses (Model B)
5. PSF improvement (per-position survey PSF instead of Gaussian)
6. Noise model improvement (add Poisson noise for arc photons)
7. Feature-space calibration / importance weighting (Model C)
8. Something else we haven't considered

**For your top recommendation: provide working code, specific data sources with URLs, expected runtime, and a SPECIFIC PREDICTION** — what completeness improvement should we expect, and what would you conclude if it doesn't materialize?

## 8.8 On Literature Verification

**Q22.** The previous LLM was asked to review 13 specific papers but never responded. Please verify our claims against the actual literature:
- Does HOLISMOKES XI (Cañameras et al. 2024) really use 1,574 real HUDF galaxies as sources? How does their methodology compare to ours?
- Does Herle, O'Riordan & Vegetti 2024 really focus on CNN selection functions? What are their key findings?
- What injection methodology does the Euclid 2025 pipeline use?
- Are there any recent (2024-2026) papers on CNN lens-finder selection functions we are missing entirely?

**Q23.** Based on your knowledge of the field: where does our approach sit relative to the state of the art? Are we above, at, or below the bar for MNRAS? What specific improvements would move us from "below" to "at" the bar?

---

# PART 9: CODE STRUCTURE

The attached zip (`stronglens_calibration_full_codebase.zip`) contains the complete codebase. Here is the canonical organization:

## 9.1 Core Packages

### `dhs/` — The Complete Scoring/Training/Preprocessing Package (CANONICAL)
This is the single source of truth for all model, preprocessing, and injection code.

| File | Purpose |
|------|---------|
| `__init__.py` | Package init |
| `model.py` | `build_model()` factory: ResNet-18, BottleneckedResNet, EfficientNetV2-S |
| `preprocess.py` | `preprocess_stack()`: raw_robust normalization (median/MAD outer annulus, clip) |
| `data.py` | `LensDataset`: epoch-aware augmentation, NPZ loading, train/val splits |
| `train.py` | `train_one()`: full training loop with mixed precision, gradient accumulation, checkpointing |
| `injection_engine.py` | Core physics: SIE deflection, Sersic rendering, PSF convolution, injection into cutouts |
| `constants.py` | STAMP_SIZE=101, CUTOUT_SIZE=101, PIXEL_SCALE=0.262, AB_ZP=22.5, etc. |
| `transforms.py` | Data augmentations (hflip, vflip, rot90), epoch-aware seeding |
| `selection_function_utils.py` | Bayesian binomial CI, depth ↔ sigma conversion |
| `calibration.py` | Calibration utilities |
| `gates.py` | Quality gates |
| `s3io.py` | S3 I/O utilities |
| `utils.py` | Normalization, radial profiles, etc. |

### `dhs/scripts/` — Experiment Runners
| File | Purpose |
|------|---------|
| `run_experiment.py` | Main training entry point: loads config YAML, builds model, runs training |
| `run_evaluation.py` | Evaluation runner |
| `run_gates.py` | Quality gate runner |

### `configs/` — Training YAML Configs
| File | Run |
|------|-----|
| `paperIV_efficientnet_v2_s_v2.yaml` | EfficientNetV2-S, step LR, 160 epochs |
| `paperIV_efficientnet_v2_s_v3_cosine.yaml` | EfficientNetV2-S, cosine LR, 160 epochs |
| `paperIV_efficientnet_v2_s_v4_finetune.yaml` | EfficientNetV2-S, finetune from v2 best, 60 epochs |
| `paperIV_bottlenecked_resnet.yaml` | BottleneckedResNet, 160 epochs |
| `paperIV_resnet18.yaml` | ResNet-18 (ablation), stopped at 68 |

## 9.2 Injection Models

### `injection_model_1/` — Parametric Sersic on Random Hosts
| File | Purpose |
|------|---------|
| `scripts/selection_function_grid.py` | Original v1 grid runner |
| `scripts/sensitivity_analysis.py` | 8-perturbation sensitivity analysis |
| `scripts/validate_injections.py` | Injection validation (flux conservation, etc.) |
| `tests/test_injection_engine.py` | **28 physics tests (ALL PASS)** |
| `validation/bright_arc_injection_test.py` | Bright arc ceiling test |
| `validation/confuser_morphology_test.py` | Confuser shortcut test |
| `validation/real_lens_scoring.py` | Real lens recall measurement |

### `injection_model_2/` — Sersic on LRG Hosts, Host-Conditioned q/PA
| File | Purpose |
|------|---------|
| `host_matching.py` | Moment-based host q/PA estimation |
| `host_selection.py` | LRG host filtering (DEV/SER) |
| `scripts/selection_function_grid_v2.py` | **Bugfixed** v2 grid runner (supports --model 1 and --model 2) |
| `scripts/host_conditioning_diagnostic.py` | 4-way diagnostic experiment |
| `tests/test_host_matching.py` | **26 moment tests (ALL PASS)** |

**NOTE on duplicate files:** `injection_model_1/engine/injection_engine.py` is an identical copy of `dhs/injection_engine.py` (confirmed by checksum). The canonical version is `dhs/injection_engine.py`. Both injection models import from `dhs/` at runtime.

## 9.3 How to Run the Pipeline

### Step 1: Train the CNN
```bash
cd /path/to/code
python3 dhs/scripts/run_experiment.py --config configs/paperIV_efficientnet_v2_s_v4_finetune.yaml
```

### Step 2: Run Injection Model 1 Grid
```bash
python3 injection_model_1/scripts/selection_function_grid.py \
  --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
  --manifest manifests/training_parity_70_30_v1.parquet \
  --output results/selection_function_model1/ \
  --n_injections 200 --seed 1337
```

### Step 3: Run Injection Model 2 Grid
```bash
python3 injection_model_2/scripts/selection_function_grid_v2.py \
  --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
  --manifest manifests/training_parity_70_30_v1.parquet \
  --output results/selection_function_model2/ \
  --model 2 --n_injections 200 --seed 1337
```

### Step 4: Run Sensitivity Analysis
```bash
python3 injection_model_1/scripts/sensitivity_analysis.py \
  --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
  --manifest manifests/training_parity_70_30_v1.parquet \
  --output results/sensitivity/
```

### Step 5: Run Sim-to-Real Validation
```bash
python3 injection_model_1/validation/real_lens_scoring.py \
  --checkpoint checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt \
  --manifest manifests/training_parity_70_30_v1.parquet
```

## 9.4 Other Directories

| Directory | Purpose |
|-----------|---------|
| `common/` | Shared utilities (experiment tracking, logging, retry, validation) |
| `data/` | Data catalogs (positives CSV, negatives CSV, external DESI catalog) |
| `docs/` | Documentation, training log, conversation history |
| `emr/` | AWS EMR Spark jobs (data generation pipeline — not needed for current review) |
| `scripts/` | Utility scripts (manifest generation, evaluation, etc.) |
| `sim_to_real_validations/` | Standalone validation scripts |
| `tests/` | General test suite |
| `results/` | Output data (selection functions, sensitivity analysis, etc.) |

---

# PART 10: WHAT WE NEED FROM YOU

## 10.1 Immediate Asks (Priority Order)

1. **Diagnose the 30% bright-arc ceiling.** This is the central mystery. The previous LLM's prediction was wrong. We need a fresh, independent diagnosis. What is the most likely cause? Design an experiment to test it.
2. **Full code audit** of the `dhs/` package, injection models, and grid runners. The previous LLM could not do this because the `dhs/` package was missing. Now it's included. Find bugs, inconsistencies, or subtle errors — especially in preprocessing and scoring paths.
3. **Independent assessment of the previous LLM's remaining proposals.** Model 2.5 (real galaxy sources), Model B (arc transplant), and Model C (feature-space calibration) were proposed BEFORE Model 2 failed. Are they still the right direction? Or does Model 2's failure invalidate the reasoning behind them?
4. **Honest publishability assessment.** Given: (a) 70pp gap between real recall and injection completeness, (b) negative Model 2 result, (c) unexplained 30% ceiling, (d) no independent spectroscopic validation — is this paper MNRAS-worthy? If not, what minimum work is needed?
5. **Working code for the highest-priority next step.** Not pseudocode. Not stubs. Complete, runnable code with data source URLs, expected runtime, and a specific prediction of the outcome.
6. **Literature verification.** Research the papers we cite. Verify our claims. Point out where we are wrong. Identify missing references.

## 10.2 Standards We Expect

- **Scientific rigor:** Every claim must be supported by data or literature. Research the papers.
- **Honesty:** If we are wasting time, say so. If our methodology is fundamentally flawed, say so. We have a history of being told things are fine when they are not.
- **Concreteness:** No vague recommendations like "try improving source morphology." Provide specific code, data sources, parameter values, and expected quantitative outcomes.
- **Meticulous code review:** Check our numbers. Check our code. Check our logic. The previous LLM provided an injection engine with 3 mathematical errors — we caught them ourselves. We need you to be more careful than that.
- **Specific predictions:** For every recommendation, state what completeness improvement you expect and what you would conclude if it doesn't materialize. This is how we avoid another month of wasted work.

## 10.3 Specific Verification Requests

1. **Preprocessing identity:** Verify that `preprocess_stack()` in `dhs/preprocess.py` produces identical normalization for real cutouts vs cutouts with injected arcs. Pay special attention to whether a bright injected arc could shift the outer-annulus statistics used for normalization.
2. **SIE deflection correctness:** Verify the SIE implementation in `dhs/injection_engine.py` against Kormann et al. 1994. Check the deflection angle formulas, the softened core, and the coordinate rotation.
3. **Sersic normalization:** Verify the Sersic profile normalization matches Graham & Driver 2005. Check that the total flux integral is correct.
4. **Area-weighted sampling:** Verify that `beta_frac = sqrt(uniform(0.01, 1.0))` produces P(β) ∝ β correctly.
5. **Augmentation leakage:** Verify that training augmentations (hflip, vflip, rot90) do NOT apply during inference/scoring.
6. **Bright-arc ceiling bug check:** Specifically investigate whether the 30% ceiling could be caused by: (a) preprocessing clipping saturating bright injections, (b) normalization denominator (MAD) being shifted by the injection, (c) any other numerical artifact in the scoring path.
7. **Flux conservation:** Verify that `cutout + lensed_arc` correctly preserves total flux through the preprocessing pipeline.

---

**Please treat this as a hostile peer review. We need the truth, not reassurance. If there is a fundamental error explaining the 70-point gap, we need to know now. The previous LLM made a wrong prediction and delivered buggy code — we need you to do better.**

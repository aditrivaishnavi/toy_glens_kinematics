# Research Plan 1: Shortcut-Aware Strong Lens Detection

## Executive Summary

This document outlines a defensible research direction for a publication on strong gravitational lens detection. The core contribution is **not** inventing shortcut-aware learning (which is established in ML), but rather:

1. **Identifying** a specific, measurable shortcut failure mode in lens finding (core leakage from extended-source injections)
2. **Proposing** a mitigation strategy (paired counterfactual controls + morphology-destroying hard negatives + core-masking)
3. **Quantifying** the impact on robustness and selection-function stability with ablations and stratified evaluation

---

## Part 1: Background and Motivation

### 1.1 The Problem: Shortcut Learning in Lens Detection

**Shortcut learning** is a well-documented phenomenon in ML where models exploit predictive cues that are easier than the intended signal, leading to brittle models that fail under distribution shift.

**Key reference:** Geirhos et al. (2020) "Shortcut Learning in Deep Neural Networks" - establishes that shortcuts are not bugs but features that models naturally exploit.

In strong lens detection, this manifests as:
- Models learning "bright center" instead of "arc morphology"
- Models learning simulation artifacts instead of physical lens signatures
- Models failing on real data despite high simulated-data performance

**Our specific finding:** A logistic regression classifier using only the central 10×10 pixels achieves AUC = 0.90 distinguishing injected lenses from controls. This is a measurable shortcut that could hurt real-world performance.

### 1.2 Related Work

| Paper | Contribution | Gap We Address |
|-------|--------------|----------------|
| Petrillo et al. (KiDS) | Simulated lens injection into real images | No explicit shortcut mitigation |
| Sheng et al. (AISTATS 2022) | Identifies simulation artifacts as failure mode; uses GANs + augmentation | Different mitigation approach |
| Strong Lens Finding Challenge | Documents selection biases and realism dependence | No counterfactual controls |
| Various CNN lens finders | Standard training on simulated positives | Often high FP rates on real data |

**Our gap:** No prior work uses matched-pairs counterfactual controls + morphology-preserving hard negatives + explicit leakage gates as a combined methodology.

---

## Part 2: The Three Methodological Components

### 2.1 Component 1: Paired Counterfactual Controls

#### What It Is
For each injected positive sample, we store a **matched control** from the identical sky position:
- Same LRG (luminous red galaxy)
- Same observing conditions (PSF, depth, sky noise)
- Same pixel values EXCEPT no injected source

This creates a **counterfactual pair**: (stamp_with_arc, stamp_without_arc) for each object.

#### Implementation Details
- **Dataset:** `v5_cosmos_paired` on S3
- **Creation:** EMR job `spark_add_paired_controls.py`
- **Storage:** Each parquet row contains:
  - `stamp_npz`: 3-band (g,r,z) 64×64 stamp with injection
  - `ctrl_stamp_npz`: 3-band 64×64 control stamp (no injection)
  - All injection parameters (theta_e, arc_snr, src_reff, etc.)

#### Where It Is Impactful

1. **Confound control:** The classifier cannot "win" by learning properties of different galaxies - the only difference between positive and control is the arc.

2. **Direct diagnostics:** Enables tests like:
   - `stamp - ctrl` isolates the injected arc
   - Core fraction analysis (Test A)
   - Physics vs mismatch verification

3. **Selection function measurement:** Paired controls make injection-recovery completeness estimates cleaner because the background population is held fixed.

4. **Training flexibility:** Enables:
   - Paired negative sampling
   - Contrastive learning objectives
   - Hard negative synthesis from controls

#### Type
**Dataset construction** (primarily), enables training methodology (secondarily)

#### Novelty Assessment
- **Not novel in ML:** Matched pairs/counterfactuals are standard in causal inference
- **Potentially novel in lens-finding:** No prior lens-finding paper presents paired counterfactual controls as a core methodological contribution
- **Publishable if:** We demonstrate via ablation that pairing materially improves robustness vs unpaired random controls

#### Ablation Experiment
Train two models with identical architecture and hyperparameters:
- **Model A (paired):** Negatives are the matched ctrl for each positive
- **Model B (unpaired):** Negatives are random LRG controls from different sky positions

Evaluate both on:
- Held-out HEALPix regions
- Curated contaminant set (ring galaxies, spirals, mergers)
- Real confirmed lenses (if available)

**Success criterion:** Model A shows improved precision at fixed recall on contaminants.

---

### 2.2 Component 2: Azimuthal-Shuffle Hard Negatives

#### What It Is
We synthesize hard negatives by taking the arc signal and destroying its coherent morphology while preserving radial statistics:

```python
def azimuthal_shuffle(stamp, ctrl):
    """Create hard negative by shuffling arc azimuthally."""
    diff = stamp - ctrl  # Isolate arc signal
    
    # For each radial bin, shuffle pixels azimuthally
    shuffled_diff = np.zeros_like(diff)
    for r_min, r_max in radial_bins:
        mask = (r >= r_min) & (r < r_max)
        pixels = diff[mask]
        np.random.shuffle(pixels)
        shuffled_diff[mask] = pixels
    
    # Add shuffled signal back to control
    hard_neg = ctrl + shuffled_diff
    return hard_neg
```

#### What This Preserves vs Destroys

| Property | Preserved? | Why It Matters |
|----------|------------|----------------|
| Total flux | Yes | Can't detect by photometry alone |
| Radial profile | Yes | Can't detect by radial statistics |
| Central brightness | Yes | Blocks the core shortcut |
| Arc morphology | **No** | This is what we want the model to learn |
| Azimuthal structure | **No** | Arcs are azimuthally coherent |

#### Where It Is Impactful

1. **Forces morphology learning:** The model cannot win by detecting "extra flux" or "radial profile change" - it must detect the coherent arc structure.

2. **Targets the exact failure mode:** Our core leakage diagnostic showed the model can exploit central brightness. Shuffled negatives have the same central brightness but no arc.

3. **Creates realistic hard cases:** These negatives are harder than random controls because they have the same statistical properties as positives.

#### Type
**Training data methodology** (negative synthesis / augmentation)

#### Novelty Assessment
- **Not novel in ML:** Hard negative mining is standard
- **Domain-specific construction:** Our azimuthal-shuffle is tailored to the lens-finding problem
- **Publishable if:** We demonstrate that:
  1. Shuffles don't introduce detectable artifacts
  2. Models trained with shuffles show improved robustness on real contaminants
  3. Performance improvements persist under core masking

#### Ablation Experiment
Train three models:
- **Model A (ctrl only):** Negatives are paired controls
- **Model B (ctrl + shuffle):** 60% paired controls, 40% azimuthal-shuffle
- **Model C (ctrl + shuffle + rotation):** Add standard augmentations

Evaluate:
- Score distributions on ctrl vs shuffle test sets
- False positive rate on ring galaxies and spirals
- Core-masked evaluation

**Success criterion:** Model B/C show better rejection of contaminants without losing completeness on true positives.

#### Artifact Check
Before training, verify shuffles are "clean":
- No Fourier artifacts (spikes at specific frequencies)
- No edge discontinuities
- Noise properties preserved

---

### 2.3 Component 3: Leakage Gates (Evaluation Methodology)

#### What It Is
A **diagnostic evaluation suite** that measures whether a model relies on spurious cues rather than intended features. We define explicit "gates" that a robust model should pass.

#### Gate Definitions

| Gate | Test | Pass Criterion | What Failure Indicates |
|------|------|----------------|------------------------|
| **Core-only LR** | Train LR on central 10×10 only | AUC < 0.65 with hard negatives | Model could exploit core brightness |
| **Core-masked eval** | Mask r<5 pixels at test time | <10% drop in AUROC | Model relies on core shortcut |
| **Hard-neg confusion** | Score distribution on shuffled negs | Separable from positives | Model learned morphology |
| **Radial-only baseline** | Train on radial profile only | AUROC << full model | Full model uses more than radial info |
| **Occlusion sensitivity** | Mask arc region at test | Significant drop | Model uses arc location |

#### Where It Is Impactful

1. **Prevents quiet failure:** High AUROC on simulated data that collapses on real data because it relied on spurious cues

2. **Converts debugging into QA standard:** Our diagnostic work becomes a reproducible evaluation protocol

3. **Enables checkpoint selection:** Select model based on core-masked performance, not overall AUROC

4. **Forces honest reporting:** Can't claim SOTA if you fail gates

#### Type
**Evaluation methodology** (quality control)

#### Novelty Assessment
- **Not novel in ML:** Diagnosing spurious cues is well-established (Clever Hans, saliency maps, etc.)
- **Task-specific contribution:** Defining a lens-finding-specific gate suite
- **Publishable if:** We show that:
  1. Baseline models fail gates
  2. Our mitigations pass gates
  3. Gate performance correlates with real-world robustness

#### Implementation
```python
class LeakageGateRunner:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
    
    def run_core_masked_eval(self, mask_radius=5):
        """Evaluate with central pixels masked."""
        # Mask center, compute AUROC
        ...
    
    def run_core_only_lr(self):
        """Train LR on central pixels only."""
        # Extract 10x10 center, train LR, report AUC
        ...
    
    def run_hardneg_separation(self):
        """Check score distributions on hard negatives."""
        # Score hard negs, compare to positives
        ...
    
    def run_all_gates(self):
        """Run complete gate suite, return pass/fail for each."""
        results = {
            "core_masked_auroc": self.run_core_masked_eval(),
            "core_only_lr_auc": self.run_core_only_lr(),
            "hardneg_auroc": self.run_hardneg_separation(),
        }
        return results
```

---

## Part 3: Experimental Design

### 3.1 Ablation Matrix

| Experiment | Paired Controls | Hard Negatives | Core Dropout | Expected Outcome |
|------------|-----------------|----------------|--------------|------------------|
| Baseline | No (random) | No | No | High AUROC, fails gates |
| + Paired | Yes | No | No | Similar AUROC, slightly better on contaminants |
| + Hard neg | Yes | Yes (40%) | No | Better contaminant rejection |
| + Core dropout | Yes | Yes (40%) | Yes (p=0.5) | Passes core-masked gate |
| Full method | Yes | Yes (40%) | Yes (p=0.5) | Best robustness, passes all gates |

### 3.2 Evaluation Metrics

#### Primary Metrics
1. **AUROC** on held-out test set
2. **Precision @ 90% recall** (operating point)
3. **FPR @ 95% TPR** (false positive control)

#### Robustness Metrics
4. **Core-masked AUROC** (r<5 masked)
5. **Hard-negative AUROC** (shuffled negatives as negatives)
6. **Contaminant rejection rate** (if contaminant set available)

#### Stratified Metrics
7. **AUROC by θ_E bin** (0.5-1", 1-1.5", 1.5-2", 2-3")
8. **AUROC by arc_snr bin** (3-10, 10-20, 20-50)
9. **AUROC by PSF FWHM** (seeing conditions)

### 3.3 Datasets

| Dataset | Purpose | Size |
|---------|---------|------|
| `v5_cosmos_paired/train` | Training | ~250k pairs |
| `v5_cosmos_paired/val` | Validation & checkpoint selection | ~25k pairs |
| `v5_cosmos_paired/test` | Final evaluation | ~25k pairs |
| Shuffled hard negatives | Training (synthesized) | 40% of negatives |
| Contaminant set (TBD) | Robustness evaluation | ~1k curated |
| Real lenses (TBD) | Reality check | As available |

---

## Part 4: Training Configuration

### 4.1 Model Architecture
```python
# ResNet18 backbone, 3-channel input
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(512, 1)
```

### 4.2 Training Hyperparameters
```python
config = {
    # Optimization
    "batch_size": 128,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 50,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "loss": "BCEWithLogitsLoss",
    
    # Data
    "hard_negative_ratio": 0.4,  # 40% of negatives are shuffled
    "core_dropout_prob": 0.5,    # Mask core 50% of time
    "core_dropout_radius": 5,    # Mask r<5 pixels
    
    # Augmentations
    "augmentations": [
        "RandomHorizontalFlip",
        "RandomVerticalFlip", 
        "RandomRotation(90)",
    ],
    
    # Normalization
    "normalization": "per_sample_robust",  # median/MAD from outer annulus
}
```

### 4.3 Training Command
```bash
python train_gen5_prime.py \
    --data-path s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_paired/ \
    --output-dir ./checkpoints/gen5_prime_ablation \
    --experiment-name full_method \
    --epochs 50 \
    --batch-size 128 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --hard-negative-ratio 0.4 \
    --core-dropout-prob 0.5 \
    --core-dropout-radius 5 \
    --checkpoint-metric core_masked_auroc \
    --wandb-project dark-halo-scope \
    --wandb-run gen5-full-method \
    --seed 42
```

---

## Part 5: Preflight Checks

Before training, verify:

### 5.1 Data Integrity
```python
# Check all parquet files readable
# Verify no NaN/Inf in images
# Check stamp and ctrl shapes match
# Verify injection parameters are reasonable
```

### 5.2 Split Integrity
```python
# Confirm no brick overlap between train/val/test
# Verify HEALPix-based split is correct
# Check class balance in each split
```

### 5.3 Normalization Check
```python
# Sample 100 images
# Apply normalization
# Verify mean ≈ 0, std ≈ 1
```

### 5.4 Hard Negative Quality
```python
# Generate 100 shuffled negatives
# Check for Fourier artifacts
# Verify radial profile preservation
# Visual inspection
```

### 5.5 Leakage Gate Baseline
```python
# Run core-only LR on raw data
# Document baseline AUC (expected ~0.90)
# This is what we're trying to mitigate
```

---

## Part 6: Training Monitoring

### 6.1 Per-Epoch Logging
- Train/val loss
- Train/val AUROC
- Learning rate

### 6.2 Every 5 Epochs
- Core-masked AUROC on validation
- Hard-negative AUROC on validation
- Score distributions (positive vs ctrl vs shuffle)

### 6.3 Early Stopping
- Patience: 10 epochs
- Metric: Core-masked validation AUROC (not overall AUROC)

### 6.4 Checkpointing
- Save best model by core-masked AUROC
- Save last model
- Save every 10 epochs for analysis

---

## Part 7: Post-Training Validation

### 7.1 Test Set Evaluation
1. Overall AUROC, precision, recall
2. Core-masked AUROC
3. Stratified by θ_E, arc_snr, PSF

### 7.2 Gate Suite
Run full leakage gate suite on best checkpoint:
- Core-only LR AUC (should be < 0.65 with hard negs)
- Core-masked drop (should be < 10%)
- Hard-neg separation (should be good)

### 7.3 Ablation Comparison
Compare all ablation variants on same test set

### 7.4 Failure Analysis
- Inspect false positives (what do they look like?)
- Inspect false negatives (what arc properties are missed?)
- Check for systematic biases

### 7.5 Reality Check (if available)
- Evaluate on known real lenses
- Even small N is informative

---

## Part 8: Paper Framing

### 8.1 Title Options
- "Mitigating Core-Leakage Shortcuts in CNN-Based Strong Lens Detection"
- "Robust Strong Lens Detection with Counterfactual Controls and Adversarial Negatives"
- "Shortcut-Aware Training for Strong Gravitational Lens Finding"

### 8.2 Abstract Structure
1. **Problem:** CNN lens finders can exploit shortcuts (we show core brightness achieves AUC 0.90)
2. **Method:** Paired counterfactual controls + morphology-destroying hard negatives + leakage gates
3. **Results:** Improved robustness on contaminants, passes diagnostic gates
4. **Significance:** More reliable lens detection for large surveys

### 8.3 What To Say
> "We diagnose and mitigate a core-leakage shortcut induced by realistic extended-source injections, and show improved robustness and a better-characterized selection function."

### 8.4 What NOT To Say
- "We invented shortcut-aware learning" (it's established)
- "First to use hard negatives" (it's standard)
- "Novel paired controls" (it's standard in causal inference)

### 8.5 Contribution Claims
1. **Diagnostic:** We identify and quantify a specific shortcut failure mode in lens-finding CNNs
2. **Methodology:** We propose a combined mitigation strategy (counterfactual pairs + azimuthal-shuffle negatives + core dropout)
3. **Evaluation:** We define a task-specific gate suite for lens-finding robustness
4. **Empirical:** We demonstrate via ablation that mitigations improve robustness without sacrificing completeness

---

## Part 9: Risk Assessment

### 9.1 What Could Go Wrong

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Mitigations don't help | Medium | Careful ablations will reveal this early |
| No access to real lenses | High | Focus on contaminant rejection, acknowledge limitation |
| Overfitting to simulation | Medium | PSF/depth stratification, hard negatives |
| Results are marginal | Medium | Be honest about effect sizes |

### 9.2 Minimum Viable Paper
Even if gains are modest, we can publish if:
- We clearly document the shortcut problem
- We show our method doesn't make things worse
- We provide a reusable evaluation framework (gates)

---

## Part 10: Timeline

| Phase | Tasks | Duration |
|-------|-------|----------|
| Preflight | Data checks, gate baselines | 1 day |
| Baseline training | Train without mitigations | 1 day |
| Ablation training | 4 ablation variants | 2-3 days |
| Evaluation | Full gate suite, comparisons | 1 day |
| Analysis | Failure analysis, visualizations | 1-2 days |
| Writing | Draft results section | Ongoing |

---

## Appendix A: Key Code Files

| File | Purpose |
|------|---------|
| `spark_phase4_pipeline_gen5.py` | Injection pipeline (COSMOS + lenstronomy) |
| `spark_add_paired_controls.py` | Creates paired control stamps |
| `train_gen5_prime.py` | Training script (to be written) |
| `leakage_gates.py` | Gate evaluation suite (to be written) |
| `azimuthal_shuffle.py` | Hard negative synthesis (to be written) |

## Appendix B: References

1. Geirhos et al. (2020) "Shortcut Learning in Deep Neural Networks" - Establishes shortcut learning concept
2. Sheng et al. (AISTATS 2022) "Unsupervised Hunt for Gravitational Lenses" - Addresses simulation artifacts
3. Petrillo et al. (KiDS) "Testing CNNs for Finding Strong Lenses" - Simulated injection approach
4. Metcalf et al. "Strong Gravitational Lens Finding Challenge" - Selection function concerns
5. Robinson et al. (ICLR 2021) "Contrastive Learning with Hard Negative Samples" - Hard negative theory

# BUILD PLAN: Option B - Gen5-Prime + Gen7/Gen8 Ablation Suite

## Executive Summary

**Goal:** Produce original, defensible research suitable for MNRAS/ApJ/AAS.

**Core thesis:** "Which aspects of source realism and imaging nuisance variation most affect sim-to-real transfer and the inferred selection function for lens finding in seeing-limited survey imaging?"

**Timeline:** 4-6 weeks
**Compute budget:** ~$1000-1500 (Lambda GPU + EMR)

---

## Guiding Principles

1. **No step proceeds without validation of the previous step**
2. **All decisions are logged with rationale**
3. **Failures are documented, not hidden**
4. **Every metric has a pre-specified interpretation**
5. **Code is tested before expensive runs**

---

## Phase 0: Foundation Lock (Day 1-2)

### 0.1 Lock Evaluation Protocol

**Purpose:** Define ALL evaluation before ANY training. No post-hoc metric shopping.

#### 0.1.1 Define Anchor Set

```bash
# Action: Create anchor catalog with locked membership
# File: anchors/tier_a_anchors.csv

# Required columns:
# - name, ra, dec, theta_e_arcsec, source (SLACS/BELLS/etc), quality_flag

# Validation checks:
# [ ] All anchors have DR10 coverage (check brick files)
# [ ] All anchors have theta_e > 0.5" (detectable at DR10 resolution)
# [ ] No duplicates
# [ ] At least 30 anchors (statistical power)
```

**Validation gate:**
```python
# run_anchor_validation.py
def validate_anchors(anchor_csv):
    df = pd.read_csv(anchor_csv)
    
    # Check 1: Minimum count
    assert len(df) >= 30, f"Need >= 30 anchors, have {len(df)}"
    
    # Check 2: Required columns
    required = ["name", "ra", "dec", "theta_e_arcsec", "source"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check 3: No duplicates
    assert df.name.nunique() == len(df), "Duplicate names"
    
    # Check 4: Theta_e range
    assert (df.theta_e_arcsec >= 0.5).all(), "Some theta_e < 0.5"
    
    # Check 5: DR10 coverage (sample 10)
    for _, row in df.sample(min(10, len(df))).iterrows():
        brick = get_brick_for_coords(row.ra, row.dec)
        assert brick_exists_in_dr10(brick), f"No DR10 for {row.name}"
    
    print(f"✓ Anchor set validated: {len(df)} anchors")
    return True
```

**Exit criteria:** Anchor CSV exists, all checks pass, file is version-controlled.

---

#### 0.1.2 Define Contaminant Set

```bash
# Action: Create contaminant catalog with known non-lenses
# File: contaminants/contaminant_catalog.csv

# Categories needed:
# - Ring galaxies (N >= 50)
# - Face-on spirals (N >= 50)
# - Mergers/interacting pairs (N >= 30)
# - Diffraction spike artifacts (N >= 20)
```

**Validation gate:**
```python
def validate_contaminants(contaminant_csv):
    df = pd.read_csv(contaminant_csv)
    
    # Check category distribution
    category_counts = df.category.value_counts()
    assert category_counts.get("ring", 0) >= 50
    assert category_counts.get("spiral", 0) >= 50
    assert category_counts.get("merger", 0) >= 30
    
    # Check no overlap with anchors
    anchor_coords = set(zip(anchors.ra.round(4), anchors.dec.round(4)))
    contam_coords = set(zip(df.ra.round(4), df.dec.round(4)))
    overlap = anchor_coords & contam_coords
    assert len(overlap) == 0, f"Overlap with anchors: {overlap}"
    
    print(f"✓ Contaminant set validated: {len(df)} objects")
```

**Exit criteria:** Contaminant CSV exists, categories populated, no anchor overlap.

---

#### 0.1.3 Define Metrics and Thresholds

| Metric | Definition | Pre-specified Threshold | Interpretation |
|--------|------------|------------------------|----------------|
| **AUROC_synth** | Area under ROC on synthetic test | Report, no threshold | Baseline comparison |
| **TPR@FPR=0.01** | True positive rate at 1% FPR | Must report | Operating point |
| **Anchor_recall@k** | Fraction of anchors in top k candidates | k=50, k=100, k=200 | Sim-to-real transfer |
| **Contam_FPR@threshold** | False positive rate on contaminants at fixed threshold | Report at threshold=0.5, 0.7, 0.9 | Contaminant rejection |
| **Core_LR_AUC** | LR classifier on core pixels only | < 0.65 after mitigation | Shortcut blocked |
| **Core_masked_AUROC** | AUROC with r<5 masked | < 10% drop from unmasked | Not core-dependent |

**Lock these in a config file:**
```yaml
# evaluation_protocol.yaml
metrics:
  primary:
    - name: anchor_recall_at_100
      threshold: null  # Report, interpret later
    - name: auroc_synthetic_test
      threshold: null
  
  gates:
    - name: core_lr_auc
      threshold: 0.65
      direction: below  # Must be BELOW threshold
    - name: core_masked_drop
      threshold: 0.10
      direction: below
  
  secondary:
    - name: contaminant_fpr_at_0.5
    - name: tpr_at_fpr_0.01
```

**Exit criteria:** `evaluation_protocol.yaml` committed, reviewed, unchangeable after this point.

---

### 0.2 Lock Data Splits

**Purpose:** Ensure train/val/test are fixed and no leakage occurs.

#### 0.2.1 Verify Split Integrity

```python
# verify_split_integrity.py
def verify_splits(parquet_root):
    """Verify no brick/healpix overlap between splits."""
    splits = ["train", "val", "test"]
    brick_sets = {}
    
    for split in splits:
        files = glob(f"{parquet_root}/{split}/*.parquet")
        bricks = set()
        for f in files[:10]:  # Sample 10 files
            df = pd.read_parquet(f, columns=["brickname"])
            bricks.update(df.brickname.unique())
        brick_sets[split] = bricks
    
    # Check no overlap
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = brick_sets[s1] & brick_sets[s2]
        assert len(overlap) == 0, f"Brick overlap {s1}/{s2}: {overlap}"
    
    print(f"✓ No brick overlap between splits")
    print(f"  Train bricks: {len(brick_sets['train'])}")
    print(f"  Val bricks: {len(brick_sets['val'])}")
    print(f"  Test bricks: {len(brick_sets['test'])}")
```

**Exit criteria:** Script runs, no overlap, counts logged.

---

#### 0.2.2 Verify Paired Data Integrity

```python
# verify_paired_data.py
def verify_paired_data(parquet_root, n_samples=100):
    """Verify stamp and ctrl are properly paired."""
    files = glob(f"{parquet_root}/train/*.parquet")
    file = random.choice(files)
    df = pd.read_parquet(file)
    
    for _, row in df.sample(min(n_samples, len(df))).iterrows():
        # Decode stamp and ctrl
        stamp = decode_stamp_npz(row.stamp_npz)  # (3, 64, 64)
        ctrl = decode_stamp_npz(row.ctrl_stamp_npz)  # (3, 64, 64)
        
        # Check 1: Same shape
        assert stamp.shape == ctrl.shape == (3, 64, 64)
        
        # Check 2: Not identical (arc should be different)
        diff = np.abs(stamp - ctrl).sum()
        assert diff > 0.1, f"stamp and ctrl too similar: diff={diff}"
        
        # Check 3: No NaN/Inf
        assert np.isfinite(stamp).all(), "NaN in stamp"
        assert np.isfinite(ctrl).all(), "NaN in ctrl"
        
        # Check 4: Reasonable value range
        assert stamp.min() > -1e6 and stamp.max() < 1e6
        assert ctrl.min() > -1e6 and ctrl.max() < 1e6
    
    print(f"✓ Paired data verified on {n_samples} samples from {file}")
```

**Exit criteria:** All checks pass on random sample from each split.

---

#### 0.2.3 Document Data Provenance

```yaml
# data_provenance.yaml
dataset:
  name: v5_cosmos_paired
  creation_date: 2026-02-05
  s3_path: s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_paired/
  
splits:
  train:
    n_samples: ~250000
    n_files: 1200
  val:
    n_samples: ~25000
    n_files: 120
  test:
    n_samples: ~25000
    n_files: 120

source:
  type: COSMOS
  injection_method: lenstronomy
  psf_model: lenstronomy.PSF
  
verification:
  split_integrity: PASSED
  paired_integrity: PASSED
  verification_date: 2026-02-XX
```

**Exit criteria:** Provenance file committed.

---

### 0.3 Code Verification

**Purpose:** Ensure all code works before expensive runs.

#### 0.3.1 Unit Tests for Core Functions

```bash
# Run existing tests
cd dhs_gen6_gen7_gen8_uber_code_fixed
pytest tests/ -v

# Expected output:
# test_bilinear_resample... PASSED
# test_decode_stamp_npz... PASSED
# test_hybrid_source... PASSED
# test_domain_randomization... PASSED
# test_to_surface_brightness... PASSED
# test_mixer... PASSED
# test_deep_source_imports... PASSED
# test_kernel_max_size... PASSED
```

**Exit criteria:** All tests pass.

---

#### 0.3.2 Integration Test: Data Loading

```python
# test_data_loading.py
def test_data_loader_integration():
    """Test full data loading pipeline."""
    from training.paired_training_v2 import build_training_loader
    
    loader = build_training_loader(
        parquet_root="/path/to/data",
        split="train",
        batch_pairs=4,
        num_workers=0,
        max_pairs_index=16,
    )
    
    batch = next(iter(loader))
    
    # Shape checks
    assert batch.x.shape == (4, 3, 64, 64), f"Bad x shape: {batch.x.shape}"
    assert batch.y.shape == (4,), f"Bad y shape: {batch.y.shape}"
    
    # Value checks
    assert torch.isfinite(batch.x).all(), "NaN in x"
    assert set(batch.y.numpy()).issubset({0, 1}), "Bad y values"
    
    print("✓ Data loader integration test passed")
```

**Exit criteria:** Test passes, shapes and values correct.

---

#### 0.3.3 Integration Test: Model Forward Pass

```python
# test_model_forward.py
def test_model_forward():
    """Test model forward pass."""
    import torch
    from training.model import LensFinderModel
    
    model = LensFinderModel(arch="resnet18", channels=3)
    model.eval()
    
    x = torch.randn(4, 3, 64, 64)
    
    with torch.no_grad():
        logits = model(x)
    
    # Shape check
    assert logits.shape == (4,), f"Bad logits shape: {logits.shape}"
    
    # Value check
    assert torch.isfinite(logits).all(), "NaN in logits"
    
    print("✓ Model forward pass test passed")
```

**Exit criteria:** Test passes.

---

#### 0.3.4 Integration Test: Full Training Loop (1 Epoch)

```python
# test_training_loop.py
def test_training_loop():
    """Test one epoch of training."""
    # Mini config
    config = {
        "epochs": 1,
        "batch_size": 8,
        "lr": 1e-4,
        "max_samples": 64,
    }
    
    # Run one epoch
    metrics = train_one_epoch(config)
    
    # Sanity checks
    assert metrics["train_loss"] < 10.0, "Loss too high"
    assert metrics["train_loss"] > 0.0, "Loss too low (suspicious)"
    assert 0.4 < metrics["train_acc"] < 0.7, "Accuracy suspicious for 1 epoch"
    
    print(f"✓ Training loop test passed: loss={metrics['train_loss']:.4f}")
```

**Exit criteria:** One epoch completes, metrics are reasonable.

---

## Phase 1: Baseline Training (Days 3-5)

### 1.1 Train Gen5-Prime Baseline

**Purpose:** Establish baseline with ALL mitigations (paired, hard-neg, core-dropout).

#### 1.1.1 Pre-flight Checks

```bash
# Checklist before starting training:
[ ] GPU available and healthy
    nvidia-smi  # Check memory, temperature
    
[ ] Data accessible
    ls -la /path/to/data/train/*.parquet | head -5
    
[ ] Disk space sufficient
    df -h /path/to/checkpoints  # Need ~50GB
    
[ ] WandB configured
    wandb login --verify
    
[ ] Config file reviewed
    cat configs/gen5_prime_baseline.yaml
```

---

#### 1.1.2 Baseline Training Config

```yaml
# configs/gen5_prime_baseline.yaml
experiment:
  name: gen5_prime_baseline
  seed: 42
  description: "Baseline with all shortcut mitigations"

model:
  arch: resnet18
  pretrained: true
  channels: 3  # Start with 3-channel

data:
  parquet_root: /path/to/v5_cosmos_paired
  batch_size: 128
  num_workers: 8

training:
  epochs: 50
  optimizer: adamw
  lr: 1e-4
  weight_decay: 1e-4
  scheduler: cosine
  
mitigations:
  paired_sampling: true
  hard_negative_ratio: 0.4
  core_dropout_prob: 0.5
  core_dropout_radius: 5

checkpoints:
  save_every: 10
  keep_best: true
  metric: core_masked_auroc  # Select best by this metric
```

---

#### 1.1.3 Training Launch

```bash
# Launch training
python train.py \
    --config configs/gen5_prime_baseline.yaml \
    --output-dir checkpoints/gen5_prime_baseline \
    --wandb-project dark-halo-scope \
    --wandb-run gen5-prime-baseline

# Monitor in separate terminal
tail -f checkpoints/gen5_prime_baseline/training.log
```

---

#### 1.1.4 Training Monitoring Checkpoints

| Epoch | Expected Loss | Expected Val AUROC | Action if Violated |
|-------|--------------|-------------------|-------------------|
| 1 | < 1.0 | 0.55-0.65 | Check data loading |
| 5 | < 0.5 | 0.70-0.80 | Check learning rate |
| 10 | < 0.3 | 0.80-0.88 | On track |
| 20 | < 0.2 | 0.88-0.93 | On track |
| 50 | < 0.1 | 0.93-0.96 | Complete |

**Automatic early stopping triggers:**
- Loss becomes NaN
- Val loss diverges from train loss by > 0.5
- GPU OOM

---

#### 1.1.5 Post-Training Validation

```python
# validate_baseline.py
def validate_baseline(checkpoint_path, data_root):
    """Run full validation on baseline model."""
    model = load_checkpoint(checkpoint_path)
    model.eval()
    
    results = {}
    
    # 1. Synthetic test AUROC
    test_loader = build_loader(data_root, split="test")
    results["auroc_synth"] = compute_auroc(model, test_loader)
    assert results["auroc_synth"] > 0.85, f"AUROC too low: {results['auroc_synth']}"
    
    # 2. Core leakage gate
    results["core_lr_auc"] = compute_core_lr_auc(model, test_loader)
    assert results["core_lr_auc"] < 0.65, f"Core shortcut not blocked: {results['core_lr_auc']}"
    
    # 3. Core-masked AUROC
    results["auroc_core_masked"] = compute_auroc_core_masked(model, test_loader)
    drop = (results["auroc_synth"] - results["auroc_core_masked"]) / results["auroc_synth"]
    assert drop < 0.10, f"Too dependent on core: drop={drop:.1%}"
    
    # 4. Hard negative gate
    hardneg_loader = build_hardneg_loader(data_root)
    results["hardneg_auroc"] = compute_auroc(model, hardneg_loader)
    assert results["hardneg_auroc"] > 0.70, "Hard negatives too easy"
    
    print("✓ Baseline validation passed")
    print(json.dumps(results, indent=2))
    return results
```

**Exit criteria:** All gates pass, results logged.

---

### 1.2 Run Ablation Variants

**Purpose:** Isolate contribution of each mitigation component.

#### 1.2.1 Ablation Grid

| Run | Paired | Hard Neg | Core Dropout | Expected Outcome |
|-----|--------|----------|--------------|------------------|
| baseline_full | ✓ | ✓ (40%) | ✓ (50%) | Best robustness |
| ablate_no_hardneg | ✓ | ✗ | ✓ | Worse contaminant rejection |
| ablate_no_coredrop | ✓ | ✓ | ✗ | Fails core gate |
| ablate_minimal | ✓ | ✗ | ✗ | Baseline comparison |

---

#### 1.2.2 Run All Ablations

```bash
# Run all ablations (can parallelize if multiple GPUs)
for config in configs/ablations/*.yaml; do
    name=$(basename $config .yaml)
    python train.py \
        --config $config \
        --output-dir checkpoints/$name \
        --wandb-run $name
done
```

---

#### 1.2.3 Compare Ablations

```python
# compare_ablations.py
def compare_ablations(ablation_dirs):
    """Compare all ablation results."""
    results = []
    
    for ablation_dir in ablation_dirs:
        name = os.path.basename(ablation_dir)
        checkpoint = f"{ablation_dir}/best_model.pt"
        
        metrics = validate_checkpoint(checkpoint)
        metrics["name"] = name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Print comparison table
    print(df[["name", "auroc_synth", "core_lr_auc", "hardneg_auroc"]].to_markdown())
    
    # Verify expected ordering
    full_auroc = df[df.name == "baseline_full"].auroc_synth.values[0]
    minimal_auroc = df[df.name == "ablate_minimal"].auroc_synth.values[0]
    
    # Mitigations may REDUCE synthetic AUROC (that's OK if gates pass)
    print(f"\nFull vs Minimal synthetic AUROC: {full_auroc:.3f} vs {minimal_auroc:.3f}")
    
    return df
```

**Exit criteria:** Ablation comparison table generated, saved to results/.

---

## Phase 2: Gen7 Ablation (Days 6-8)

### 2.1 Fix Gen7 Code Issues

**Purpose:** Address known semantic issues before training.

#### 2.1.1 Fix clump_flux_frac Semantics

```python
# BEFORE (wrong):
amp = frac  # frac is treated as peak amplitude

# AFTER (correct):
# Gaussian integrated flux = amp * 2π * σ²
# To get flux_frac of total, solve for amp:
total_base_flux = base.sum()
target_clump_flux = flux_frac * total_base_flux
amp = target_clump_flux / (2 * np.pi * sigma_pix**2)
```

**Validation:**
```python
def test_clump_flux_fraction():
    result = generate_hybrid_source(key="test", clump_flux_frac_range=(0.1, 0.1))
    base_flux = _sersic_2d(...).sum()
    clump_flux = result["img"].sum() - base_flux
    actual_frac = clump_flux / result["img"].sum()
    assert 0.08 < actual_frac < 0.12, f"Clump frac wrong: {actual_frac}"
```

---

#### 2.1.2 Verify Gen7 Parameter Distributions

```python
# verify_gen7_parameters.py
def verify_gen7_realism():
    """Check Gen7 parameters against COSMOS measurements."""
    # Generate 1000 samples
    samples = [generate_hybrid_source(f"sample_{i}") for i in range(1000)]
    
    # Extract parameter distributions
    re_pix = [s["meta"]["re_pix"] for s in samples]
    n_sersic = [s["meta"]["n_sersic"] for s in samples]
    q = [s["meta"]["q"] for s in samples]
    
    # Compare to COSMOS (reference values)
    # COSMOS lensed sources: re ~ 0.1-0.5", n ~ 0.5-2.0, q ~ 0.3-1.0
    assert 0.5 < np.median(re_pix) < 15, f"re_pix out of range"
    assert 0.8 < np.median(n_sersic) < 1.5, f"n_sersic out of range"
    assert 0.6 < np.median(q) < 0.9, f"q out of range"
    
    print("✓ Gen7 parameters within expected ranges")
```

---

### 2.2 Gen7 Training

```yaml
# configs/gen7_hybrid.yaml
experiment:
  name: gen7_hybrid_sources
  description: "Hybrid Sersic+clumps sources"

source:
  mode: hybrid
  n_clumps_range: [2, 8]
  gradient_strength: 0.15

# Inherit all mitigations from baseline
mitigations:
  paired_sampling: true
  hard_negative_ratio: 0.4
  core_dropout_prob: 0.5
```

**Exit criteria:** Gen7 training completes, metrics logged.

---

### 2.3 Compare Gen5 vs Gen7

```python
# Focus on real-anchor performance difference
def compare_gen5_gen7(gen5_model, gen7_model, anchors):
    gen5_anchor_recall = evaluate_on_anchors(gen5_model, anchors)
    gen7_anchor_recall = evaluate_on_anchors(gen7_model, anchors)
    
    print(f"Gen5 anchor recall@100: {gen5_anchor_recall:.1%}")
    print(f"Gen7 anchor recall@100: {gen7_anchor_recall:.1%}")
    print(f"Delta: {gen7_anchor_recall - gen5_anchor_recall:+.1%}")
    
    # Statistical significance (bootstrap)
    n_boot = 1000
    deltas = []
    for _ in range(n_boot):
        idx = np.random.choice(len(anchors), len(anchors), replace=True)
        boot_anchors = anchors.iloc[idx]
        d = evaluate_on_anchors(gen7_model, boot_anchors) - \
            evaluate_on_anchors(gen5_model, boot_anchors)
        deltas.append(d)
    
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    print(f"95% CI: [{ci_low:+.1%}, {ci_high:+.1%}]")
```

**Exit criteria:** Comparison complete with confidence intervals.

---

## Phase 3: Gen8 Ablation (Days 9-11)

### 3.1 Calibrate Artifact Rates

**Purpose:** Ensure artifact rates match DR10 reality.

```python
# calibrate_artifacts.py
def calibrate_dr10_artifacts(dr10_sample_dir, n_samples=1000):
    """Measure artifact rates in real DR10 images."""
    
    cosmic_count = 0
    sat_count = 0
    
    for fits_path in sample(dr10_sample_dir, n_samples):
        img = read_fits(fits_path)
        
        # Detect cosmic rays (simple: > 10 sigma outliers in small regions)
        if detect_cosmic_ray(img):
            cosmic_count += 1
        
        # Detect saturation spikes
        if detect_saturation(img):
            sat_count += 1
    
    print(f"Cosmic ray rate: {cosmic_count/n_samples:.1%}")
    print(f"Saturation rate: {sat_count/n_samples:.1%}")
    
    # Update config
    return {
        "cosmic_rate": cosmic_count / n_samples,
        "sat_rate": sat_count / n_samples,
    }
```

---

### 3.2 Gen8 Training

```yaml
# configs/gen8_domain_rand.yaml
experiment:
  name: gen8_domain_randomization
  description: "Domain randomization with DR10-calibrated artifacts"

artifacts:
  profile: mild
  cosmic_rate: 0.12  # From calibration
  sat_rate: 0.06     # From calibration
  jitter_sigma_pix: 0.25
  
  # IMPORTANT: Do NOT apply extra PSF convolution
  enable_psf_anisotropy: false  # Already applied in injection

mitigations:
  paired_sampling: true
  hard_negative_ratio: 0.4
  core_dropout_prob: 0.5
```

---

## Phase 4: Final Evaluation (Days 12-14)

### 4.1 Aggregate All Results

```python
# aggregate_results.py
def aggregate_all_results():
    """Create master comparison table."""
    variants = [
        "gen5_prime_baseline",
        "ablate_no_hardneg",
        "ablate_no_coredrop", 
        "ablate_minimal",
        "gen7_hybrid",
        "gen8_domain_rand",
    ]
    
    results = []
    for variant in variants:
        checkpoint = f"checkpoints/{variant}/best_model.pt"
        
        metrics = {
            "variant": variant,
            "auroc_synth": compute_auroc_synth(checkpoint),
            "anchor_recall_100": compute_anchor_recall(checkpoint, k=100),
            "contam_fpr_05": compute_contaminant_fpr(checkpoint, threshold=0.5),
            "core_lr_auc": compute_core_lr_auc(checkpoint),
            "core_masked_auroc": compute_core_masked_auroc(checkpoint),
        }
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df.to_csv("results/master_comparison.csv", index=False)
    
    print(df.to_markdown())
    return df
```

---

### 4.2 Selection Function Analysis

```python
# selection_function.py
def compute_selection_function(model, injection_grid):
    """Compute completeness as function of theta_E, PSF, arc_snr."""
    
    completeness = {}
    
    # Stratify by theta_E
    for theta_bin in [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]:
        mask = (injection_grid.theta_e >= theta_bin[0]) & \
               (injection_grid.theta_e < theta_bin[1])
        subset = injection_grid[mask]
        
        preds = model.predict(subset.images)
        recall = (preds > 0.5).mean()
        
        completeness[f"theta_{theta_bin[0]}_{theta_bin[1]}"] = recall
    
    # Stratify by PSF FWHM
    for psf_bin in [(0.8, 1.2), (1.2, 1.5), (1.5, 2.0)]:
        mask = (injection_grid.psf_fwhm >= psf_bin[0]) & \
               (injection_grid.psf_fwhm < psf_bin[1])
        subset = injection_grid[mask]
        
        preds = model.predict(subset.images)
        recall = (preds > 0.5).mean()
        
        completeness[f"psf_{psf_bin[0]}_{psf_bin[1]}"] = recall
    
    return completeness
```

---

### 4.3 Generate Paper Figures

| Figure | Content | Script |
|--------|---------|--------|
| Fig 1 | Training data examples (pos/ctrl/hardneg) | `fig_data_examples.py` |
| Fig 2 | Ablation comparison bar chart | `fig_ablation_bars.py` |
| Fig 3 | ROC curves (synth + anchor) | `fig_roc_curves.py` |
| Fig 4 | Selection function heatmap | `fig_selection_function.py` |
| Fig 5 | Contaminant rejection by category | `fig_contaminant_rejection.py` |
| Fig 6 | Core leakage gate results | `fig_core_gate.py` |

---

## Quality Gates Summary

### Gate 0: Foundation
- [ ] Anchor set locked and validated
- [ ] Contaminant set locked and validated
- [ ] Evaluation protocol committed
- [ ] Split integrity verified
- [ ] All unit tests pass
- [ ] Integration tests pass

### Gate 1: Baseline
- [ ] Training completes without error
- [ ] AUROC_synth > 0.85
- [ ] Core_LR_AUC < 0.65
- [ ] Core_masked_drop < 10%
- [ ] Hard_neg_AUROC > 0.70

### Gate 2: Ablations
- [ ] All 4 ablation variants complete
- [ ] Results show expected pattern (full > partial > minimal)
- [ ] Comparison table generated

### Gate 3: Gen7
- [ ] Code issues fixed and tested
- [ ] Training completes
- [ ] Comparison to baseline with CIs

### Gate 4: Gen8
- [ ] Artifact rates calibrated
- [ ] Training completes
- [ ] No performance degradation on clean data

### Gate 5: Publication
- [ ] All figures generated
- [ ] Selection function computed
- [ ] Bootstrap CIs for all key metrics
- [ ] Results reproducible (3 seeds)

---

## Risk Mitigation

| Risk | Likelihood | Detection | Mitigation |
|------|------------|-----------|------------|
| Training diverges | Low | Loss NaN/spike | Reduce LR, check data |
| Gates fail | Medium | Automated checks | Iterate on mitigations |
| No anchor improvement | Medium | Anchor eval | Report as negative result |
| GPU failure | Low | nvidia-smi | Checkpoint recovery |
| Data corruption | Very Low | Hash checks | Re-download from S3 |

---

## Timeline Summary

| Phase | Days | Deliverable |
|-------|------|-------------|
| 0: Foundation Lock | 1-2 | Locked evaluation, verified data |
| 1: Baseline + Ablations | 3-5 | Baseline model, ablation table |
| 2: Gen7 | 6-8 | Gen7 model, comparison |
| 3: Gen8 | 9-11 | Gen8 model, comparison |
| 4: Final Evaluation | 12-14 | Figures, tables, selection function |
| 5: Writing | 15-28 | Draft paper |

**Total: 4-6 weeks**

---

## Appendix: File Checklist

```
project/
├── configs/
│   ├── evaluation_protocol.yaml     # LOCKED
│   ├── gen5_prime_baseline.yaml
│   ├── ablations/
│   │   ├── ablate_no_hardneg.yaml
│   │   ├── ablate_no_coredrop.yaml
│   │   └── ablate_minimal.yaml
│   ├── gen7_hybrid.yaml
│   └── gen8_domain_rand.yaml
├── data/
│   ├── anchors/
│   │   └── tier_a_anchors.csv       # LOCKED
│   ├── contaminants/
│   │   └── contaminant_catalog.csv  # LOCKED
│   └── data_provenance.yaml
├── checkpoints/
│   ├── gen5_prime_baseline/
│   ├── ablate_no_hardneg/
│   ├── ablate_no_coredrop/
│   ├── ablate_minimal/
│   ├── gen7_hybrid/
│   └── gen8_domain_rand/
├── results/
│   ├── master_comparison.csv
│   ├── selection_function.json
│   └── figures/
├── scripts/
│   ├── verify_split_integrity.py
│   ├── verify_paired_data.py
│   ├── validate_baseline.py
│   ├── compare_ablations.py
│   └── aggregate_results.py
└── tests/
    ├── test_data_loading.py
    ├── test_model_forward.py
    └── test_training_loop.py
```

tory of# Gen5-Prime Full Execution Plan

**Date:** 2026-02-05  
**Objective:** Train a shortcut-resistant lens finder that passes all validation gates and achieves publication-quality results.

---

## Executive Summary

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| **Phase 1**: Infrastructure Validation | 1 day | Verified pipeline on Lambda |
| **Phase 2**: Gen5-Prime Training | 2-3 days | Trained 6-channel model |
| **Phase 3**: Gate Validation | 0.5 day | All 5 gates pass |
| **Phase 4**: Anchor Evaluation | 0.5 day | Tier-A recall metrics |
| **Phase 5**: Publication Prep | 2-3 days | Paper-ready figures & metrics |

**Total: ~7-8 days**

---

## Phase 1: Infrastructure Validation (Day 1)

### 1.1 Verify Coadd Cache Exists

```bash
# On Lambda
ssh lambda
ls -la /lambda/nfs/darkhaloscope-training-dc/dr10/coadd_cache/ | head -20
du -sh /lambda/nfs/darkhaloscope-training-dc/dr10/coadd_cache/
```

**Expected:** Cache exists with ~100GB+ of DR10 coadds.

**If missing:** Sync from S3:
```bash
rclone copy s3://darkhaloscope/dr10/coadd_cache/ /lambda/nfs/darkhaloscope-training-dc/dr10/coadd_cache/ --progress
```

### 1.2 Test Pipeline Components

```bash
# On Lambda
cd /lambda/nfs/darkhaloscope-training-dc/code/
python -c "
from dark_halo_scope.training.paired_training_v2 import (
    decode_stamp_npz,
    CoaddCutoutProvider,
    PairedParquetDataset,
    Preprocess6CH,
    make_outer_mask,
    run_gates_quick,
)
print('All imports successful')
"
```

### 1.3 Pilot Data Load Test

```python
# test_pipeline_pilot.py
from dark_halo_scope.training.paired_training_v2 import build_training_loader
import json

parquet_root = "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
coadd_cache = "/lambda/nfs/darkhaloscope-training-dc/dr10/coadd_cache"

loader = build_training_loader(
    parquet_root,
    coadd_cache,
    split="train",
    batch_pairs=16,
    num_workers=0,  # Start with 0 for debugging
    max_pairs_index=100,
)

batch = next(iter(loader))
print(f"x6 shape: {batch.x6.shape}")  # Should be (16, 6, 64, 64)
print(f"y distribution: pos={batch.y.sum().item()}, neg={(1-batch.y).sum().item()}")
print(f"hardneg count: {batch.meta['is_hardneg'].sum().item()}")
print(f"x_ratio range: [{batch.meta['x_ratio'].min():.2f}, {batch.meta['x_ratio'].max():.2f}]")
```

**Pass Criteria:**
- [x] Batch loads without error
- [x] Shape is (B, 6, H, W)
- [x] Contains mix of pos/ctrl/hardneg
- [x] x_ratio values are reasonable (0.4 - 2.0)

### 1.4 Test Model Forward Pass

```python
# test_model_forward.py
import torch
from dark_halo_scope.training.convnext_6ch import LensFinder6CH

model = LensFinder6CH(arch="tiny", pretrained=True, init="copy_or_zero")
model.eval()

x = torch.randn(4, 6, 64, 64)
meta = torch.randn(4, 2)

with torch.no_grad():
    logits = model(x, meta)
    probs = torch.sigmoid(logits)

print(f"Logits shape: {logits.shape}")  # Should be (4,)
print(f"Probs range: [{probs.min():.3f}, {probs.max():.3f}]")
```

### 1.5 Test Gate Runner (No Model)

```python
# test_gates_no_model.py
from dark_halo_scope.training.paired_training_v2 import run_gates_quick
import json

results = run_gates_quick(
    parquet_root="/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos",
    coadd_cache_root="/lambda/nfs/darkhaloscope-training-dc/dr10/coadd_cache",
    split="train",
    model=None,
    device="cpu",
    max_pairs=500,
)

print(json.dumps(results, indent=2))
```

**Pass Criteria:**
- [x] Gates run without crash
- [x] All strata have sufficient samples
- [x] Core AUC, annulus AUC, radial AUC computed

---

## Phase 2: Gen5-Prime Training (Days 2-4)

### 2.1 Training Configuration

```yaml
# gen5_prime_config.yaml
model:
  arch: "tiny"
  pretrained: true
  init: "copy_or_zero"
  meta_dim: 2
  hidden: 256
  dropout: 0.1

data:
  parquet_root: "/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos"
  coadd_cache: "/lambda/nfs/darkhaloscope-training-dc/dr10/coadd_cache"
  batch_pairs: 64
  num_workers: 8

training:
  epochs: 30
  lr: 1e-4
  weight_decay: 0.01
  warmup_epochs: 3
  
  # Mixing probabilities
  pos_prob: 0.4
  ctrl_prob: 0.4
  hardneg_prob: 0.2
  
  # Curriculum: hard_neg ratio schedule
  curriculum:
    enabled: true
    start_hardneg_prob: 0.5
    end_hardneg_prob: 0.2
    anneal_epochs: 18

preprocessing:
  clip: 10.0
  resid_sigma_pix: 3.0
  outer_r_pix: 16

output:
  model_dir: "/lambda/nfs/darkhaloscope-training-dc/models/gen5_prime"
  checkpoint_every: 5
  run_gates_every: 5
```

### 2.2 Training Script Structure

```python
# train_gen5_prime.py

def main():
    # 1. Setup
    model = LensFinder6CH(...)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(...)
    
    # 2. Data loaders
    train_loader = build_training_loader(split="train", ...)
    val_loader = build_training_loader(split="val", ...)
    
    # 3. Training loop
    for epoch in range(cfg.epochs):
        # Update curriculum (hard_neg probability)
        current_hardneg_prob = get_curriculum_prob(epoch, cfg)
        train_loader.collate_fn.hardneg_prob = current_hardneg_prob
        
        # Train epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler)
        
        # Validate
        val_loss, val_auc = validate(model, val_loader)
        
        # Run gates every N epochs
        if epoch % cfg.run_gates_every == 0:
            gate_results = run_gates_quick(..., model=model)
            log_gates(gate_results)
        
        # Save checkpoint
        if epoch % cfg.checkpoint_every == 0:
            save_checkpoint(model, optimizer, epoch)
    
    # 4. Final gates
    final_gates = run_gates_quick(..., model=model, max_pairs=2000)
    save_json(final_gates, "final_gate_results.json")
```

### 2.3 Expected Training Timeline

| Epoch | Hard-Neg % | Expected Val AUC | Gate Status |
|-------|------------|------------------|-------------|
| 0 | 50% | ~0.60 | Running baseline |
| 5 | 45% | ~0.75 | Check core AUC |
| 10 | 35% | ~0.85 | Monitor annulus AUC |
| 15 | 25% | ~0.90 | All gates should improve |
| 20 | 20% | ~0.93 | Near convergence |
| 25 | 20% | ~0.94 | Stabilizing |
| 30 | 20% | ~0.95 | Final |

### 2.4 Early Stopping Criteria

**Stop and investigate if:**
- Core AUC (x≥1.0) > 0.70 after epoch 15
- Annulus AUC (x≥1.0) < 0.65 after epoch 15
- Hard-neg mean p > 0.15 after epoch 15
- Val loss diverges from train loss by >0.3

---

## Phase 3: Gate Validation (Day 4-5)

### 3.1 Full Gate Suite

```python
# run_final_gates.py
from dark_halo_scope.training.paired_training_v2 import run_gates_quick
from dark_halo_scope.training.convnext_6ch import LensFinder6CH
import torch
import json

# Load best checkpoint
model = LensFinder6CH(arch="tiny", pretrained=False, init="kaiming")
ckpt = torch.load("/lambda/nfs/.../gen5_prime/ckpt_best.pt")
model.load_state_dict(ckpt["model"])
model.cuda().eval()

# Run on test split
results = run_gates_quick(
    parquet_root="/lambda/nfs/darkhaloscope-training-dc/phase4c_v5_cosmos",
    coadd_cache_root="/lambda/nfs/darkhaloscope-training-dc/dr10/coadd_cache",
    split="test",
    model=model,
    device="cuda",
    max_pairs=2000,
)

print(json.dumps(results, indent=2))
```

### 3.2 Gate Pass/Fail Criteria

| Gate | Metric | Target | Weight |
|------|--------|--------|--------|
| **G1** | Core-only AUC (x≥1.0) | ≤ 0.60 | Critical |
| **G2** | Radial-profile AUC (x≥1.0) | ≤ 0.60 | Critical |
| **G3** | Arc-annulus AUC (x≥1.0) | ≥ 0.75 | Important |
| **G4** | Hard-neg mean p | ≤ 0.05 | Critical |
| **G5** | Arc-occlusion drop | ≥ 0.30 | Important |

**Pass:** All critical gates pass, at least 4/5 total  
**Marginal:** 3/5 gates pass, no critical failures  
**Fail:** Any critical gate fails

### 3.3 Remediation if Gates Fail

| Failure Mode | Root Cause | Remedy |
|--------------|------------|--------|
| G1/G2 fail | Model still uses radial cues | Add center degradation augmentation |
| G3 fails | Model ignores arcs | Increase residual sigma, check preprocessing |
| G4 fails | Hard-neg too easy | Increase hard-neg ratio, verify shuffle |
| G5 fails | Model not using arc region | Review annulus mask, check theta values |

---

## Phase 4: Anchor Evaluation (Day 5)

### 4.1 Tier-A Anchors (DR10-Detectable)

Build Tier-A from:
1. Jacobs et al. (2019) DES lenses - cross-match with DR10 footprint
2. Huang et al. (2020) DECaLS lenses - already in DR10
3. Storfer et al. (2022) lenses with θ_E > 1.5"

**Expected:** 50-100 Tier-A anchors with ground-based detectability.

### 4.2 Evaluation Script

```python
# eval_tier_a_anchors.py
import pandas as pd
from astropy.io import fits
import requests

# Load Tier-A catalog
tier_a = pd.read_csv("tier_a_anchors.csv")

# Fetch cutouts and run inference
results = []
for _, row in tier_a.iterrows():
    cutout = fetch_legacy_survey_cutout(row.ra, row.dec)
    processed = preprocess_6ch(cutout)
    p_lens = model(processed, meta)
    results.append({
        "name": row.name,
        "ra": row.ra,
        "dec": row.dec,
        "theta_e": row.theta_e,
        "p_lens": float(p_lens),
    })

# Compute metrics
df = pd.DataFrame(results)
recall_at_50 = (df.p_lens > 0.5).mean()
recall_at_30 = (df.p_lens > 0.3).mean()
print(f"Recall@0.5: {recall_at_50:.1%}")
print(f"Recall@0.3: {recall_at_30:.1%}")
```

### 4.3 Expected Anchor Performance

| Metric | Gen5 (Current) | Gen5-Prime (Expected) |
|--------|----------------|----------------------|
| Recall@0.5 (Tier-A) | 4% | 40-60% |
| Recall@0.3 (Tier-A) | 8% | 60-80% |
| Hard-neg FPR@0.5 | N/A | <5% |

---

## Phase 5: Publication Prep (Days 6-8)

### 5.1 Figures to Generate

| Figure | Description | Script |
|--------|-------------|--------|
| **Fig 1** | Training data examples (pos/ctrl/hardneg) | `fig_training_examples.py` |
| **Fig 2** | 6-channel visualization (raw + residual) | `fig_6ch_viz.py` |
| **Fig 3** | Gate metrics by stratum | `fig_gate_results.py` |
| **Fig 4** | ROC curves (synthetic + anchors) | `fig_roc_curves.py` |
| **Fig 5** | Anchor gallery with p_lens | `fig_anchor_gallery.py` |
| **Fig 6** | Selection function C(θ_E, z) | `fig_selection_function.py` |

### 5.2 Tables to Prepare

| Table | Content |
|-------|---------|
| **Table 1** | Dataset statistics (N samples, θ_E range, PSF range) |
| **Table 2** | Training hyperparameters |
| **Table 3** | Gate results by stratum |
| **Table 4** | Anchor evaluation (Tier-A breakdown) |
| **Table 5** | Comparison to prior work |

### 5.3 Key Numbers for Paper

```
Training Data:
- N_positives: 1.38M
- N_controls: 1.38M (paired)
- N_hard_negatives: ~550K (on-the-fly, 20% of training)
- θ_E range: 0.5" - 2.5"
- PSF FWHM range: 0.97" - 1.60"

Model:
- Architecture: ConvNeXt-Tiny (6-channel input)
- Parameters: 28.6M
- Input: 64×64 pixels, grz bands + residual

Performance:
- Synthetic AUC (test): TBD
- Tier-A Recall@0.5: TBD
- Gate 1 (Core AUC): TBD (target ≤0.60)
- Gate 3 (Annulus AUC): TBD (target ≥0.75)
```

### 5.4 Paper Outline

```
1. Introduction
   - Strong lensing as cosmological probe
   - Need for automated detection
   - Challenge: shortcut learning in synthetic training

2. Data
   - DR10 imaging
   - COSMOS source injection
   - Paired positive/control generation
   - Hard negative construction

3. Method
   - 6-channel residual view
   - Theta-aware azimuthal shuffle
   - Curriculum learning
   - Shortcut detection gates

4. Results
   - Synthetic performance
   - Gate validation
   - Tier-A anchor evaluation
   - Comparison to prior work

5. Discussion
   - Selection function calibration
   - Limitations
   - Future: application to full DR10

6. Conclusions
```

---

## Checkpoints and Decision Gates

### Day 1 Checkpoint
- [ ] Coadd cache verified
- [ ] Pipeline imports work
- [ ] Pilot batch loads successfully
- [ ] Model forward pass works
- [ ] Gates run without model

**Decision:** Proceed to training if all checks pass.

### Day 3 Checkpoint (Mid-Training)
- [ ] Val AUC > 0.80
- [ ] Core AUC (x≥1.0) < 0.75
- [ ] Hard-neg mean p < 0.20

**Decision:** Continue if improving, add center degradation if core AUC stalls.

### Day 5 Checkpoint (Post-Training)
- [ ] All 5 gates pass (or 4/5 with justification)
- [ ] Tier-A recall > 40%
- [ ] Hard-neg FPR < 10%

**Decision:** Proceed to paper prep if pass, iterate if fail.

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Coadd cache missing | Low | High | Pre-sync from S3 |
| Training diverges | Medium | Medium | Reduce LR, check gradients |
| Gates fail after training | Medium | High | Add center degradation, retrain |
| Tier-A anchors insufficient | Low | Medium | Cross-match more catalogs |
| GPU OOM | Low | Medium | Reduce batch size |

---

## Appendix: File Locations

```
/lambda/nfs/darkhaloscope-training-dc/
├── phase4c_v5_cosmos/           # Training data (Parquet)
├── dr10/coadd_cache/            # Base cutouts for paired controls
├── models/
│   ├── gen5_cosmos/             # Current (shortcut) model
│   └── gen5_prime/              # New shortcut-resistant model
├── code/
│   ├── dark_halo_scope/
│   │   └── training/
│   │       ├── paired_training_v2.py
│   │       └── convnext_6ch.py
│   └── train_gen5_prime.py
└── anchor_cutouts/
    ├── tier_a/                  # Ground-based detectable lenses
    └── tier_b/                  # HST-only lenses (stress test)
```

---

## Summary

**Goal:** Shortcut-resistant lens finder with ≤0.60 core AUC and ≥40% Tier-A recall.

**Key Innovations:**
1. 6-channel residual view (raw + high-pass)
2. Paired controls from same LRG
3. Hard negatives via theta-aware azimuthal shuffle
4. 5 stratified validation gates

**Timeline:** 7-8 days to publication-ready results.

**Next Step:** Run Phase 1 infrastructure validation on Lambda.

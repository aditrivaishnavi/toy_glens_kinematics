# Paper IV Full Parity: Complete Course Correction Plan

## The Situation

Our current `resnet18_baseline_v1` run (val AUC 0.9592, test AUC 0.9579) cannot be compared to Paper IV (val AUC 0.9984) because we differ in **every major dimension**: architecture, input resolution, epochs, batch size, LR schedule, loss weighting, split ratio, and we lack a second model entirely. The LLM across multiple review rounds identified 26+ items we haven't acted on. This plan addresses all of them.

## Target: Level 2 Comparability ("Protocol-matched, architecture-modernized")

The LLM defined three levels of comparability:

- **Level 1 (Full replication)**: Match architecture + protocol + data construction exactly. We cannot achieve this (their custom TF code is unavailable).
- **Level 2 (Protocol-matched, architecture-modernized)**: Match resolution, epochs, LR schedule, batch sizes, data criteria. Use modern PyTorch equivalents. **This is our target.**
- **Level 3 (Same dataset, different method)**: Keep current setup. Not enough for parity claims.

Paper language: "We match Paper IV's training protocol and data criteria, but use modern PyTorch equivalents."

---

## Complete Discrepancy Audit (from Paper IV re-read)

### Architecture (FUNDAMENTAL)

- **Paper IV ResNet**: Custom "shielded" ResNet from Lanusse et al. 2018, 1x1 convolutions reducing dimensionality after each block. **194,433 parameters**. TensorFlow.
- **Our ResNet**: torchvision ResNet-18 (weights=None), conv1 modified, fc -> 1. **~11.2M parameters**. PyTorch.
- **Gap**: 58x more parameters. Completely different architecture.
- **Mitigation**: Implement a **bottlenecked ResNet** with 1x1 channel reductions (Lanusse-style), targeting ~0.2M-1M parameters. Run BOTH this and standard ResNet-18 to show sensitivity to architecture choice.
- **Paper IV EfficientNet**: EfficientNetV2 (NOT V1), pre-trained on ImageNet, fine-tuned. **20,542,883 parameters**.
- **Correct PyTorch analogue**: `torchvision.models.efficientnet_v2_s` (~21.5M params, very close match). **NOT V2-B0** (doesn't exist in torchvision). The V2 family uses S/M/L naming.

### Training Protocol

- Input: Paper IV 101x101, ours **64x64** (cropped) -- MUST FIX
- Epochs: Paper IV 160 (best@126 ResNet, best@50 EfficientNet), ours **16** (early stopped) -- MUST FIX
- Batch: Paper IV 2048 (ResNet) / 512 (EfficientNet), ours **256** -- MUST FIX via gradient accumulation
- LR: Paper IV 5e-4 (ResNet) / 3.88e-4 (EfficientNet), ours **3e-4** -- MUST FIX
- Schedule: Paper IV StepLR halve@80/130, ours **CosineAnnealingLR** -- MUST FIX
- Early stopping: Paper IV None, ours **patience=10** -- MUST DISABLE
- Loss: Paper IV unweighted CE (Equation 1), ours **tier-weighted BCE** -- MUST ADD unweighted baseline
- Optimizer: Paper IV not specified; SGD+momentum or AdamW both defensible
- Pre-training: Paper IV ResNet from scratch, EfficientNet from ImageNet

### Data

- Positives: Paper IV 1,372 vs ours 4,788 -- **KEEP our larger set** (legitimate advantage)
- Negatives: Paper IV 134,182 vs ours 411,661 -- **KEEP** (bigger is fine)
- Split: Paper IV 70/30 vs ours 70/15/15 -- Need **Manifest A** (70/30) for parity, keep **Manifest B** (70/15/15) for audit
- Neg types: SER/DEV/REX -- already matched in config
- Neg cleaning: Paper IV uses Spherimatch + prior model p>0.4. We need to implement high-score filter as practical alternative.
- Normalization: **Not specified in Paper IV**. Our `raw_robust` is defensible; just document it.
- Augmentation: **Not described in Paper IV**. Our HFlip/VFlip/Rot90 is safe; just document it.

### Missing Components

- **Meta-learner**: Paper IV combines ResNet + EfficientNet via 1-layer NN (300 nodes, feature weighted stacking, Coscrato et al. 2020). AUC 0.9989 vs individual 0.9984/0.9987.
- **EfficientNetV2**: We don't have it at all.

---

## Phase 1: Documentation (immediate, before any code changes)

### 1a. Create `docs/LESSONS_LEARNED_PAPER_IV_PARITY.md`

Full discrepancy analysis document covering:

- Every gap listed above with citations to Paper IV sections
- Why we trained on 64x64 when the LLM explicitly told us to train on 101x101 (line 3160 of conversation_with_llm.txt)
- Why early stopping at epoch 16 was fatal (Paper IV's ResNet peaks at epoch 126)
- Architecture mismatch (custom 194K vs torchvision 11.2M)
- The three levels of comparability and why Level 2 is our target
- Referee defense strategies for the three likely objections

### 1b. Create `docs/PAPER_IV_TRAINING_PROTOCOL.md`

Standalone cross-check reference:

- Exact hyperparameters for both models (from Section 3.2)
- Dataset construction: 1,372 positives, 134,182 negatives, negative cleaning procedure
- Training curves analysis (Figures 3, 4, 5 -- loss/AUC divergence at late epochs)
- Meta-learner specification (Section 3.2.3, Coscrato et al. 2020)
- Unknowns: normalization not specified, augmentation not described, optimizer not stated
- EfficientNet batch size ambiguity: "batch size of 512" total vs "per GPU batch size of 512"

---

## Phase 1.5: LLM Code Pack Review & Integration

The LLM provided a code pack (`paperIV_parity_course_correction/`) with 13 files. A thorough audit identified 13 issues (3 critical, 6 important, 4 minor) that must be fixed before integration into `stronglens_calibration/dhs/`.

### Critical Fixes Required

1. **EfficientNet is V1, not V2**: Code uses `efficientnet_b0` (~5.3M params). Must change to `efficientnet_v2_s` (~21.5M params) to match Paper IV's 20.5M.
2. **Loss is weighted for parity**: Uses `weighted_bce_loss` with `sample_weight`. Paper IV uses unweighted CE. Need `--unweighted` flag or manifest with all weights=1.0.
3. **No 70/30 manifest**: Training loads split=="train" (70%) and split=="val" (15%), ignoring test (15%). Need to merge val+test for 70/30 parity.

### Important Fixes Required

4. **Preprocessing MAD factor mismatch**: LLM code scales MAD by 1.4826; our existing code does not. Must remove factor to match existing normalization.
5. **No AMP/mixed precision**: Need to add `torch.cuda.amp` for practical training speed on single GPU with 101x101.
6. **Meta-learner architecture wrong**: Code uses LogisticRegression (3 params). Paper IV specifies 1-layer NN with 300 nodes.
7. **Meta-learner trains+evaluates on same data**: Must train on training predictions, evaluate on validation predictions.
8. **Adam without weight_decay**: Should use AdamW with weight_decay=1e-4 for regularization.
9. **Missing scripts**: No negative cleaning, no manifest generation, no bottlenecked ResNet.

### Minor Issues

10. StepLR possible off-by-one (epoch 81 vs 80 for halving)
11. Non-reproducible augmentation (global np.random state vs seeded RNG)
12. No cutout_path in prediction outputs (fragile row-order alignment for meta-learner)
13. Final-epoch predictions saved instead of best-epoch predictions

### Integration Strategy

Fix all issues and merge into `stronglens_calibration/dhs/` -- one unified codebase.

---

## Phase 2: Code Changes for Parity Training

### 2a. Resolution support

**`stronglens_calibration/dhs/constants.py`:** Add `STAMP_SIZE_PARITY = 101`

**`stronglens_calibration/dhs/preprocess.py`:** Make `crop` configurable from YAML. When `crop=False`, pass through 101x101. Outer-annulus radii (r_in=20, r_out=32) work fine on 101x101.

**`stronglens_calibration/dhs/data.py`:** Add `crop` to `DatasetConfig`, pass to `preprocess_stack`. Add split filter to support 70/30 (merge val+test).

### 2b. Model architectures

**`stronglens_calibration/dhs/model.py`:**

1. **Keep** `build_resnet18` (standard ResNet-18, ~11.2M params -- for comparison/ablation)
2. **Add** `build_bottlenecked_resnet(in_ch=3)` -- Lanusse-style compact ResNet:
   - Standard ResNet backbone BUT with 1x1 conv "shielding" layers after each block
   - Target parameter count: ~0.2M-1M (close to Paper IV's 194K)
   - This directly addresses the "58x parameters" referee objection
   - Defense: "We implemented a bottlenecked ResNet with 1x1 channel reduction following the Lanusse et al. (2018) design principle"
3. **Add** `build_efficientnet_v2_s(in_ch=3, pretrained=True)`:
   - Use `torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')`
   - Replace classifier head with single logit output
   - ~21.5M params (very close to Paper IV's 20.5M)
   - Fine-tune all layers (or freeze stem for first 10 epochs, then unfreeze)
4. **Add** model factory: `build_model(arch: str, in_ch: int, **kwargs) -> nn.Module`

### 2c. Training loop enhancements

**`stronglens_calibration/dhs/train.py`:**

- `gradient_accumulation_steps: int` in TrainConfig (default 1)
- `lr_schedule: str` -- "cosine" or "step"
- `lr_step_epoch: int` and `lr_step_gamma: float` for StepLR
- `early_stopping_patience: 0` means disabled
- `unweighted_loss: bool` -- when True, ignore sample_weight (use reduction='mean')
- `optimizer: str` -- "adamw" or "sgd" (SGD with momentum 0.9)
- `arch: str` -- model architecture selection
- `pretrained: bool` -- for EfficientNet ImageNet weights

Gradient accumulation implementation:

```python
opt.zero_grad(set_to_none=True)
for micro_step, batch in enumerate(dl_tr):
    loss = compute_loss(batch) / accum_steps
    scaler.scale(loss).backward()
    if (micro_step + 1) % accum_steps == 0:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
```

### 2d. New config files

**`configs/resnet_bottleneck_parity_v1.yaml`** (closest to Paper IV):

```yaml
dataset:
  mode: file_manifest
  preprocessing: raw_robust
  crop: false  # 101x101
  manifest_path: .../training_parity_v1.parquet  # 70/30 manifest
train:
  arch: bottlenecked_resnet  # ~0.2M-1M params
  epochs: 160
  batch_size: 128  # micro-batch
  gradient_accumulation_steps: 16  # effective = 2048
  lr: 0.0005
  lr_schedule: step
  lr_step_epoch: 80
  lr_step_gamma: 0.5
  weight_decay: 0.0001
  early_stopping_patience: 0  # disabled
  unweighted_loss: true
  mixed_precision: true
```

**`configs/resnet18_parity_v1.yaml`** (standard ResNet-18 for comparison):
Same as above but `arch: resnet18`

**`configs/efficientnet_parity_v1.yaml`**:

```yaml
train:
  arch: efficientnet_v2_s  # NOT v2_b0
  pretrained: true  # ImageNet
  epochs: 160
  batch_size: 64  # micro-batch
  gradient_accumulation_steps: 8  # effective = 512
  lr: 0.000388
  lr_schedule: step
  lr_step_epoch: 130
  lr_step_gamma: 0.5
  early_stopping_patience: 0
  unweighted_loss: true
```

---

## Phase 3: Data Preparation for Parity

### 3a. Two-manifest strategy

**Manifest A (Paper IV comparable baseline):**

- Script: `scripts/make_paperIV_baseline_manifest.py`
- Input: existing `training_v1.parquet`
- Changes: merge val+test into "val" (70/30), set all sample_weight to 1.0, keep SER/DEV/REX negatives
- Output: `training_parity_v1.parquet`
- Uses 101x101 inputs, unweighted loss

**Manifest B (audit model -- already exists):**

- Keep existing `training_v1.parquet` as-is
- 70/15/15 split, tier-weighted, full dataset
- This is for our novel contribution (selection function, detector audit)

### 3b. Negative cleaning (high-score filter)

Script: `scripts/negative_cleaning_highscore_filter.py`

- Load existing `best.pt` from `resnet18_baseline_v1`
- Score ALL negatives in training set using same preprocessing
- Flag negatives with p > 0.4 for manual inspection
- Remove confirmed/likely lenses from negative pool
- Output: cleaned manifest + audit report documenting how many removed and what they look like
- Paper defense: "We removed the top-scoring candidates from the negative training pool and visually audited N examples, finding X% plausible lenses; those were excluded."

---

## Phase 4: Training Runs

### Run 1: Bottlenecked ResNet (closest Paper IV analogue)

- 160 epochs, 101x101, StepLR@80, effective batch 2048
- Expected: val AUC improvement past epoch 80, best around epoch 100-130
- Single GPU, ~12-15 min/epoch, total ~30-40 hours

### Run 2: Standard ResNet-18 (for architecture comparison)

- Same protocol as Run 1, but with standard ResNet-18 (~11.2M params)
- Shows sensitivity to architecture choice
- May achieve different AUC -- document and discuss

### Run 3: EfficientNetV2-S

- 160 epochs, 101x101, StepLR@130, effective batch 512
- Pretrained ImageNet, fine-tune all layers
- Expected: best around epoch 40-60

### Run 4: Meta-learner (after Runs 1-3)

- 1-layer NN, 300 nodes (Coscrato et al. 2020 feature weighted stacking)
- Input: concatenation of base model probabilities on TRAINING set
- Evaluate on validation set
- Compare to simple averaging (Paper IV found they perform similarly)

---

## Phase 5: Still-Pending LLM Recommendations (from earlier rounds)

### Must-do before paper:

1. **Preprocessing regression test**: checksum on 1-2 reference NPZ files to lock preprocessing
2. **Deterministic dataloader seeding**: seed workers, log all seeds
3. **Positive-class calibration analysis**: reliability diagram for high-score region (top 0.1-1%)

### Should-do:

4. **Core-only / annulus-only shortcut gates**: at minimum, core-only check
5. **Bootstrap AUC for shortcut flags** with 200-500 resamples
6. **Band-provenance unit test**: verify grz channel ordering
7. **Merger/tidal N2 confuser category**: highest priority missing confuser type

### Nice-to-have:

8. Laplacian/DoG high-frequency features in shortcut detection
9. Blue-ness of arc annulus color features
10. Saturation/extreme pixel fraction
11. Edge artifact score

---

## Phase 6: Paper Framing (two-run strategy)

### Run A -- Paper IV Comparable Baseline

For the "Methods" section, comparison tables, and demonstrating our pipeline produces competitive results.

- Protocol-matched: 101x101, 160 epochs, StepLR, effective batch sizes, unweighted CE
- Architecture-modernized: bottlenecked ResNet (~0.2-1M params) + EfficientNetV2-S (~21.5M params)
- Split: 70/30 train/val
- Manifest: `training_parity_v1.parquet` (all weights = 1.0)

### Run B -- Calibration/Audit Model

Our original contribution -- the novel science that justifies the paper.

- HEALPix 70/15/15 splits with held-out test set
- Tier-weighted loss (Tier-A=1.0, Tier-B=0.5)
- Full selection function, completeness maps, detector audit
- Bootstrap confidence intervals on test set

---

## Referee Defense Strategies (from LLM)

### Objection 1: "You didn't replicate the architecture"

**Defense**: "Paper IV used a custom TF ResNet from Lanusse et al. (2018); we implemented two practical PyTorch equivalents: (i) a bottlenecked ResNet with comparable capacity (~XK params vs their 194K), and (ii) standard ResNet-18. We report sensitivity to architecture choice."

### Objection 2: "Your negatives may contain unlabeled lenses"

**Defense**: "We removed the top X% high-scoring candidates from the negative training pool using a first-pass model (threshold 0.4) and visually audited Y examples, finding Z% plausible lenses; those were excluded."

### Objection 3: "Your training protocol differs"

**Defense**: "We match Paper IV's epochs, resolution, LR schedule, and effective batch sizes via gradient accumulation. We report both a Paper-IV-comparable baseline and our audit-optimized variant."

---

## What We Can Honestly Claim

**With parity runs completed:**
"We match Paper IV's training protocol (epochs, resolution, LR schedule, effective batch sizes) using modern PyTorch equivalents. We implement a bottlenecked ResNet following the Lanusse et al. (2018) design principle, and EfficientNetV2-S (21.5M parameters, comparable to Paper IV's 20.5M). Our comparison is at Level 2: protocol-matched, architecture-modernized."

**Our unique contribution (unaffected by this):**
"We provide the first systematic selection function audit for neural-network-based strong lens finding in DESI DR10, measuring completeness as a function of Einstein radius, PSF, and depth."

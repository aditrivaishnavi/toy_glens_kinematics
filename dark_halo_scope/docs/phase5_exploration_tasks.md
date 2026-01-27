# Phase 5 Exploration Tasks and Follow-ups

This document tracks exploration tasks and follow-up work identified during Phase 4d review and Phase 5 planning.

---

## Current Status

| Phase | Status | Completion |
|-------|--------|------------|
| Phase 4c (64x64) | Complete | 10.6M stamps |
| Phase 4d (64x64) | Complete | 10.25% completeness |
| Phase 5 (training) | Ready to start | Code fixed |
| Phase 4c (96x96) | Pending | After Phase 5 baseline |

---

## Follow-up Tasks

### High Priority (Before Publication)

Per LLM review, these are required for MNRAS/ApJ-level publication:

| Task | When to Do | Description |
|------|------------|-------------|
| **Model-based completeness** | After Phase 5 | Define recovery using model score at fixed FPR threshold, not proxy θ/PSF cuts. This is the "true lens-finder completeness." |
| **Threshold sensitivity plots** | After Phase 5 | Test multiple θ/PSF thresholds (0.6, 0.7, 0.8, 0.9) and show completeness curves for each. Demonstrates sensitivity of selection function. |
| **Stratified reporting** | After Phase 5 | Report completeness stratified by PSF size, depth, and resolution bins. Include both "all" and "clean" subset curves. |
| **Known lens recovery validation** | Phase 6 | Recover known strong lenses in DR10 South as external sanity check. Critical for referee confidence. |
| **Region-bootstrap uncertainty** | Publication prep | Proper hierarchical/weighted estimate of cosmic variance (current region variance may be dominated by small-N bins). |

### Medium Priority (After Baseline Works)

| Task | When to Do | Description |
|------|------------|-------------|
| **96x96 stamps (4c rerun)** | After Phase 5 64x64 | Compare detection performance across stamp sizes. |
| **Hybrid model** | After ResNet baseline | CNN + MLP on metadata (PSF, depth, mask fractions). |
| **Calibration analysis** | After training | Reliability curve for probabilistic score interpretation. |
| **Ablation studies** | Training complete | Baseline vs model completeness impact. |

### Low Priority (Future Work)

| Task | When to Do | Description |
|------|------------|-------------|
| **V2 injection campaign** | After Phase 5 results | Broaden θ_E coverage if needed (smaller θ_E + better seeing). |
| **Multi-head model** | Future | Classify + regress θ_E simultaneously. |
| **Color conditioning** | Future | Use g-r, r-z colors as additional features. |

---

## Model Training Plan

### Phase 5 Model Suite

| Model | Architecture | Purpose | Priority |
|-------|--------------|---------|----------|
| **ResNet-18** | 64x64x3 image-only | Primary baseline | 1 |
| **Small CNN** | 3-layer sanity check | Debugging/validation | 2 |
| **EfficientNet-B0** | Higher capacity | Comparison | 3 |
| **ConvNeXt-Tiny** | Modern architecture | Comparison | 3 |
| **Hybrid** | ResNet + MLP | Robustness | 4 |

### Training Configuration

```yaml
# Recommended baseline config
arch: resnet18
epochs: 5
steps_per_epoch: 5000
batch_size: 256
lr: 3e-4
weight_decay: 1e-4
stamp_size: 64
split: train
```

---

## AWS Instance Recommendations

### P-Instance Quota: 32 vCPUs

| Instance | vCPUs | GPUs | GPU Memory | Use Case |
|----------|-------|------|------------|----------|
| p3.2xlarge | 8 | 1x V100 | 16 GB | Smoke tests, debugging |
| p3.8xlarge | 32 | 4x V100 | 64 GB | Full training with DDP |

### Data Staging

**Critical:** Do NOT stream from S3 during training.

```bash
# Stage to local NVMe first
aws s3 sync s3://darkhaloscope/phase4_pipeline/phase4c/v3_color_relaxed/stamps/train_stamp64_bandsgrz_gridgrid_small /data/phase4c/stamps/
```

### Training Commands

**Single GPU (smoke test):**
```bash
python phase5_train_lensfinder.py \
  --data /data/phase4c/stamps \
  --contract_json phase5_required_columns_contract.json \
  --split train \
  --arch resnet18 \
  --epochs 1 \
  --steps_per_epoch 100 \
  --batch_size 256 \
  --out_dir /data/phase5/models/resnet18_smoke
```

**4 GPUs (DDP):**
```bash
torchrun --standalone --nproc_per_node=4 phase5_train_lensfinder.py \
  --data /data/phase4c/stamps \
  --contract_json phase5_required_columns_contract.json \
  --split train \
  --arch resnet18 \
  --epochs 5 \
  --steps_per_epoch 5000 \
  --batch_size 256 \
  --out_dir /data/phase5/models/resnet18_v1
```

---

## Scientific Considerations from LLM Review

### Framing the Selection Function

1. **Phase 4d is proxy resolvability**, not lens-finder completeness
2. **Phase 5/6 will produce model-based completeness** using score thresholds
3. **Both are publishable** if framed correctly

### LLM's Publication Readiness Assessment

> "Phase 4d becomes defensible as a baseline and Phase 6 becomes publishable" once you:
> 1. Complete Phase 5 training
> 2. Define completeness based on model score (not proxy cuts)
> 3. Report both proxy and model-based completeness

**Current Phase 4d limitation:** θ_E ≤ 0.6" is guaranteed undetectable under current thresholds. This is expected physics (not a bug) but must be framed correctly in publication.

### Key Publications Points

1. Completeness must be conditioned on θ/PSF (resolution)
2. Report both "all" and "clean" subset curves
3. Include Wilson CIs for bins with n ≥ 50
4. Show threshold sensitivity (multiple θ/PSF cuts)

### Training Strategy

- Train on **all injections vs controls** (not just "recovered")
- Let model learn the true boundary
- Report recall vs θ_E and θ/PSF (should be smooth)

---

## Files Reference

| File | Purpose |
|------|---------|
| `dark_halo_scope/model/phase5_train_lensfinder.py` | Training script |
| `dark_halo_scope/model/phase5_infer_scores.py` | Inference script |
| `dark_halo_scope/model/spark_phase5_completeness_from_scores.py` | Spark completeness aggregation |
| `dark_halo_scope/model/phase5_baseline_scalar_model.py` | Metadata-only baseline |
| `dark_halo_scope/model/phase5_required_columns_contract.json` | Column contract |

---

## Checkpoints

After training, save:
- `checkpoint_best.pt` - Best validation loss
- `checkpoint_last.pt` - Final epoch
- Upload to S3: `s3://darkhaloscope/phase5/models/{model_name}/`


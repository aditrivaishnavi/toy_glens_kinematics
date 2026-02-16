# Research Direction Review - Request for Guidance

## Context

We have been debugging and building infrastructure for a strong lens detection system. We now need to step back and assess: **what is the actual research contribution?**

You correctly identified that:
- Production data uses lenstronomy PSF (not the buggy `_fft_convolve2d`)
- Core leakage at θ_E ≥ 2" matches physics expectations
- Training should block the "core shortcut"

---

## Question 1: Are We On Track for Original Research?

We have:
- COSMOS source injection via lenstronomy
- Paired controls (stamp with arc, ctrl without)
- Hard negatives (azimuthal shuffle planned)
- 6-channel input (raw + residual)
- Core leakage diagnostics

**Is this sufficient for a publishable contribution?** Or are we just replicating existing work with minor variations?

---

## Question 2: Two Possible Paper Approaches

### Approach A: Injection Methodology Paper

Focus on the simulation/injection pipeline as the contribution.

**Please provide:**
1. What would make this novel vs existing injection methods?
2. Detailed narrative: how should the paper be structured?
3. What claims can we make?
4. What experiments validate those claims?
5. Early checks to ensure we're on track

### Approach B: Lens Finder Paper

Focus on the detection model as the contribution.

**Please provide:**
1. What would make this novel vs existing lens finders (CMU-DeepLens, etc.)?
2. Detailed narrative: how should the paper be structured?
3. What performance benchmarks must we beat?
4. What experiments validate the claims?
5. Early checks to ensure we're on track

---

## Question 3: Honest Assessment

Which approach (A or B, or both, or neither) is more viable given:
- Current state of the field
- What we have built so far
- Time/resource constraints

---

## Question 4: Training Plan Review

Here is our planned training approach. Please review and sign off or suggest changes.

### Data

- **Positives**: ~300k stamps with injected COSMOS arcs (θ_E: 0.5-3", arc_snr: 3-50)
- **Controls**: Paired ctrl stamps (same LRG, no arc)
- **Hard negatives**: Azimuthal shuffle (preserves radial profile, destroys arc morphology)
- **Splits**: Train/val/test by HEALPix (no brick overlap)

### Model Architecture

```python
# ResNet18 backbone, 3-channel input (stamp only)
# ctrl used as hard negative, not as input channel
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(512, 1)
```

### Training Configuration

```python
config = {
    "batch_size": 128,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 50,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "loss": "BCEWithLogitsLoss",
    
    # Augmentations
    "augmentations": [
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "RandomRotation(90)",
        "CoreDropout(p=0.3, r=5)",  # Mask central r<5 pixels
    ],
    
    # Hard negative sampling
    "hard_negative_ratio": 0.3,  # 30% of negatives are azimuthal-shuffled
    
    # Normalization
    "normalization": "per_sample_robust",  # median/MAD from outer annulus
}
```

### Preflight Checks (Before Training)

1. **Data integrity**: Verify all parquet files readable, no NaN/Inf
2. **Class balance**: Check positive/negative ratio
3. **Split integrity**: Confirm no brick overlap between train/val/test
4. **Sample visualization**: Plot 20 random samples from each class
5. **Normalization check**: Verify normalized images have ~0 mean, ~1 std
6. **Core leakage gate**: Re-run LR on core-only with hard negatives included

### Training Quality Monitoring

1. **Loss curves**: Train/val loss should decrease, gap should be small
2. **AUROC tracking**: Log val AUROC every epoch
3. **Per-θ_E AUROC**: Track performance by Einstein radius bin
4. **Core-masked AUROC**: Every 10 epochs, evaluate with core masked
5. **Hard negative accuracy**: Track model accuracy on shuffled negatives
6. **Gradient norms**: Monitor for instability

### Post-Training Validation

1. **Test set AUROC**: Overall and stratified by θ_E, arc_snr, PSF
2. **Core-masked test**: Repeat evaluation with r<5 masked
3. **Calibration**: Plot reliability diagram
4. **Failure analysis**: Inspect false positives and false negatives
5. **Real lens test**: If available, evaluate on known real lenses

---

## Question 5: Training Command

Here is the command I plan to run:

```bash
python train_gen5_prime.py \
    --data-path s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_paired/ \
    --output-dir ./checkpoints/gen5_prime_v1 \
    --epochs 50 \
    --batch-size 128 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --hard-negative-ratio 0.3 \
    --core-dropout-prob 0.3 \
    --wandb-project dark-halo-scope \
    --wandb-run gen5-prime-v1
```

**Is this correct? Any missing flags or concerns?**

---

## Question 6: What Other Questions Should I Ask?

I may be missing important considerations. What questions should I be asking that I haven't?

---

## Question 7: Sign-Off Request

Given all the above, please provide:

1. **GO / NO-GO / CONDITIONAL-GO** for training
2. **Critical blockers** (if any)
3. **Recommended changes** (if any)
4. **Priority order** for post-training experiments

---

## Attached

- Full training script (to be written based on your feedback)
- `spark_phase4_pipeline_gen5.py` - injection pipeline
- `test_arc_only_core_fraction.py` - diagnostic code

# Key Lessons Learned

## Previous Training Runs

### Gen 1: ResNet18 on v3 data
- **Result**: AUROC 0.997, but FPR@TPR=0.85 was ~6% (very poor)
- **Lesson**: High AUROC doesn't mean good lens finder. Need low FPR at high completeness.
- **Problem Identified**: Controls are trivially different from positives

### Gen 2: ConvNeXt-Tiny on v4_sota data (current)
- **Result**: Best tpr@fpr1e-4 = 79.8% at epoch 3, then collapsed to 0%
- **Lesson**: Model overfits rapidly, learning shortcuts instead of morphology
- **Problem Identified**: Paired controls + worker sharding bug + no early stopping

---

## What We Learned About Metrics

### AUROC is Misleading
- AUROC stayed at 0.99 throughout training
- But tpr@fpr1e-4 collapsed from 79.8% to 0%
- **Lesson**: Must use low-FPR operating points for rare event detection

### tpr@fpr1e-4 is the Right Metric
- At FPR=0.01%, only ~6 false positives per 65,000 negatives
- This is operationally meaningful for candidate selection
- **Lesson**: Report tpr at multiple FPR thresholds (1e-2, 1e-3, 1e-4, 1e-5)

### fpr@tpr Reveals Score Distribution
- When fpr@tpr0.50 = fpr@tpr0.85, scores are binary
- This means model is extremely confident (right or wrong)
- **Lesson**: Monitor fpr@tpr to detect calibration collapse

---

## What We Learned About Data

### Paired Controls Create Shortcuts
- Same galaxy ± injection is too easy to distinguish
- Model learns "flux addition" not "arc morphology"
- **Lesson**: Use unpaired controls (different galaxies, matched conditions)

### 60% Unresolved is Too Many
- theta_e < PSF FWHM means arc is undetectable
- Model cannot learn morphology from point-source-like injections
- **Lesson**: Filter training to resolved regime (theta_e/PSF > 0.5)

### Gaussian PSF is Unrealistic
- Real PSFs have extended wings (Moffat profile, beta ~3.5)
- Synthetic arcs look different from real arcs
- **Lesson**: Use Moffat PSF convolution

---

## What We Learned About Training

### 8x Sample Duplication Kills Generalization
- Worker sharding bug caused each sample to be seen 8x per epoch
- Effective training was 96 epochs in 12 actual epochs
- **Lesson**: Always shard by rank * num_workers + worker_id

### No Early Stopping Wastes Compute
- Best model was at epoch 3
- Continued training for 8 more epochs (wasted GPU hours)
- **Lesson**: Always use early stopping with patience 3-5

### Train Loss → 0 is a Red Flag
- Train loss dropping to 0.0001 means memorization
- Model no longer learning generalizable features
- **Lesson**: Add regularization (dropout, weight decay) or stop early

---

## State of the Art Comparison

### Published Benchmarks (from literature)

**CMU DeepLens (Lanusse et al. 2018)**:
- 90% completeness at 99% rejection for resolved, high-SNR lenses
- Translates to: FPR ≈ 1% at TPR = 90%
- BUT: only for easy subset (theta_e > 1.4 arcsec, SNR > 20)

**Bayesian Lens Finder (MNRAS 2024)**:
- FPR = 0.1% at completeness 46%
- FPR = 0.1% at completeness 34% for single classifier
- Wide-area survey conditions (harder than controlled simulations)

### Our Best Result (Gen 2, Epoch 3)
- tpr@fpr1e-4 = 79.8% (FPR = 0.01% at TPR = 79.8%)
- tpr@fpr0.001 = 85.4% (FPR = 0.1% at TPR = 85.4%)
- **Competitive with literature** IF it generalizes to real data

### The Gap
- Our metrics are on synthetic injections, not real lenses
- Real lenses have different arc morphologies, host galaxies
- Need external validation on known lenses (SLACS, BELLS)

---

## Recommended Next Steps

1. **Fix worker sharding bug** - Already done
2. **Add early stopping** - Already done
3. **Use unpaired controls** - Gen 3 data (in progress)
4. **Use Moffat PSF** - Gen 3 data (in progress)
5. **Filter to resolved regime** - Use --min_theta_over_psf 0.5
6. **Hard negative mining** - After Gen 3 training
7. **External validation** - Test on known lenses

---

## Questions We Still Have

1. Why does the collapse happen at exactly epoch 4→5?
2. Is the paired controls hypothesis definitely correct?
3. Are there bugs in the evaluation code (roc_curve_np, tpr_at_fpr)?
4. What other bugs might exist in the training script?
5. Will unpaired controls fix the problem or just delay it?


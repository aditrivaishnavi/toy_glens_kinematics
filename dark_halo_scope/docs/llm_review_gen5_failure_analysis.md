# Gen5 Lens Finder Failure Analysis: Request for Expert Review

**Date:** 2026-02-05  
**Project:** DarkHaloScope - Strong Gravitational Lens Detection  
**Purpose:** Honest post-mortem of Gen5 model failure on real lenses, seeking expert guidance

---

## Executive Summary

We trained a ConvNeXt-Tiny CNN on 11.75 million synthetic gravitational lens stamps. The model achieved **near-perfect synthetic performance** (AUC=0.9945, TPR@FPR=1%=93.6%) but **catastrophically failed on real lenses** (4.4% recall at p_lens>0.5 on SLACS/BELLS anchors).

After extensive debugging, we identified fundamental problems with our training data and anchor set selection. **We are seeking expert review of our conclusions and guidance on remediation.**

---

## 1. Model and Training Setup

### 1.1 Architecture
- **Backbone:** ConvNeXt-Tiny (from torchvision)
- **Head:** MetaFusionHead with 2-dim metadata input
- **Input:** 3-channel (g, r, z) 64×64 stamps, MAD-normalized using outer annulus
- **Output:** Binary classification (lens vs non-lens)

### 1.2 Training Data (Gen5)
- **Total stamps:** 11,751,600
- **Split:** 26% train / 39% val / 35% test (hash-based by task_id)
- **Positive samples:** Real DR10 LRG cutouts + injected COSMOS galaxy lensed through SIE model
- **Negative samples:** Same DR10 LRG cutouts without injection (unpaired controls)
- **Source galaxies:** 20,000 templates from GalSim COSMOS 25.2 HST F814W catalog
- **PSF model:** Moffat β=3.5, per-band FWHM from DR10 metadata
- **Einstein radius range:** 0.5 - 2.5 arcsec

### 1.3 Training Results
| Metric | Value |
|--------|-------|
| Best Epoch | 6 |
| Best AUC (validation) | 0.8945 |
| Test AUC | 0.9945 |
| Test TPR @ FPR=1% | 93.6% |
| Test TPR @ FPR=0.1% | 85.2% |

---

## 2. Anchor Evaluation Setup

### 2.1 SLACS/BELLS Anchor Set
- **SLACS:** 48 Grade A lenses from Auger et al. 2009
- **BELLS:** 20 Grade A/B lenses from Brownstein et al. 2012
- **Total:** 68 known confirmed lenses
- **Cutouts:** Downloaded from Legacy Survey DR10 cutout service (64×64 at 0.262"/pix)

### 2.2 Anchor Evaluation Results
| Metric | Value |
|--------|-------|
| Lenses with p_lens > 0.5 | 3/68 (4.4%) |
| Lenses with p_lens > 0.3 | 12/68 (17.6%) |
| Median p_lens | 0.016 |
| Max p_lens | 0.91 (SDSSJ1205+4910) |

### 2.3 Top and Bottom Scorers
**Top 5:**
| Lens | p_lens | θ_E (arcsec) |
|------|--------|--------------|
| SDSSJ1205+4910 | 0.91 | 1.22 |
| SDSSJ1106+5228 | 0.72 | 1.23 |
| SDSSJ2300+0022 | 0.67 | 1.24 |
| SDSSJ0252+0039 | 0.49 | 1.04 |
| BELLSJ1631+1854 | 0.40 | 1.06 |

**Bottom 5:**
| Lens | p_lens | θ_E (arcsec) |
|------|--------|--------------|
| SDSSJ2321-0939 | 0.010 | 1.60 |
| BELLSJ1159+5820 | 0.008 | 0.97 |
| BELLSJ0830+5116 | 0.006 | 1.41 |
| BELLSJ1318+3942 | 0.004 | 1.08 |
| BELLSJ0747+4448 | 0.003 | 1.16 |

---

## 3. Debugging Investigation

### 3.1 Correlation Analysis

We computed Pearson correlations between p_lens and physical/observational parameters:

| Correlation | Pearson r | Interpretation |
|-------------|-----------|----------------|
| p_lens vs r_max (image brightness) | 0.225 | Weak |
| p_lens vs θ_E (Einstein radius) | **0.021** | **Essentially zero** |

**Observation:** If the model learned gravitational lensing physics, we'd expect strong correlation with θ_E. There is none.

### 3.2 Shortcut Detection Test

We tested if the model responds to trivial geometric patterns:

```python
# Add simple synthetic ring to any image
for c in range(3):
    y, x = np.ogrid[:64, :64]
    r = np.sqrt((y - 32)**2 + (x - 32)**2)
    ring = np.exp(-((r - 15)**2) / 20) * 0.01  # Gaussian ring at radius 15
    image[c] += ring
```

**Results:**
| Test Case | p_lens |
|-----------|--------|
| Training negative (control) | 0.0001 |
| Training negative + synthetic ring | **1.0000** |
| Low-scoring anchor (SDSSJ2321-0939) | 0.0098 |
| Same anchor + synthetic ring | **1.0000** |

**Conclusion:** The model is trivially fooled by adding a simple ring pattern. It has learned to detect ring-shaped patterns, not gravitational lensing.

### 3.3 Raw Pixel Value Comparison

We compared central 8×8 pixel maximum values (r-band, in nanomaggies):

| Dataset | Central r_max (mean) | Central r_max (std) |
|---------|---------------------|---------------------|
| Training positives | 0.0878 | 0.0558 |
| Training negatives | 0.0874 | 0.1217 |
| **Anchor lenses** | **0.0099** | **0.0050** |

**Observation:** Training LRGs are ~9x brighter in the center than anchor lenses.

### 3.4 Outer Background Comparison

| Dataset | Outer mean | Outer std |
|---------|------------|-----------|
| Training | 0.000320 | 0.004934 |
| Anchors | 0.000041 - 0.002045 | 0.002655 - 0.004339 |

**Observation:** Similar noise levels, so the difference in center brightness is real.

### 3.5 Normalization Analysis

Our training uses MAD normalization from outer annulus:
```python
def robust_mad_norm_outer(x, clip=10.0, eps=1e-6, inner_frac=0.5):
    # Mask inner 50% of image
    outer_mask = (r_from_center > inner_radius)
    # Compute median and MAD from outer pixels
    med = np.median(x[outer_mask])
    mad = np.median(np.abs(x[outer_mask] - med))
    scale = 1.4826 * mad + eps
    return np.clip((x - med) / scale, -clip, clip)
```

**Normalization parameters (r-band):**

| Sample | Outer Median | Outer MAD | Scale | Center Max (normalized) |
|--------|--------------|-----------|-------|------------------------|
| Training positive | 0.000488 | 0.003148 | 0.00467 | **10.00** (clipped) |
| High anchor (p=0.91) | 0.002045 | 0.003428 | 0.00508 | 3.97 |
| Low anchor (p=0.01) | -0.000061 | 0.002762 | 0.00410 | 2.59 |

### 3.6 Spatial Pattern Analysis (After Normalization)

| Region | Training | High Anchor (p=0.91) | Low Anchor (p=0.01) |
|--------|----------|---------------------|---------------------|
| Center (r<8) | **3.57** | 0.93 | 0.08 |
| Inner ring (8<r<16) | 1.05 | 2.02 | -0.02 |
| Outer ring (16<r<24) | 0.03 | **2.08** | -0.06 |

**Critical Observation:** Training data has HIGH center, LOW outer. Anchors have different patterns.

### 3.7 Model Behavior on Training Data

| Sample Type | Mean p_lens |
|-------------|-------------|
| Training positives | 1.0000 |
| Training negatives | 0.0000 |

The model achieves **perfect separation** on training data, even for low arc_snr samples:
- arc_snr=1.3 → p_lens=1.0000
- arc_snr=1.5 → p_lens=1.0000
- arc_snr=2.2 → p_lens=1.0000

This is suspicious - the model should struggle with low-SNR samples.

---

## 4. Key Findings

### 4.1 The Model Learned a Shortcut

The model learned: **"High normalized center + low normalized outer = lens"**

This pattern arises from:
1. Training LRGs have bright centers (~0.06 nMgy)
2. Outer regions are sky noise (~0.003 nMgy)
3. MAD normalization creates specific center-to-outer ratios
4. Injected arcs add subtle extra flux

### 4.2 SLACS/BELLS Is the Wrong Anchor Set

SLACS/BELLS lenses were discovered via:
1. **Ground-based spectroscopy** (SDSS/BOSS) - detecting two redshifts in single fiber
2. **HST follow-up imaging** (0.05"/pix) - confirming arc morphology

These lenses were NOT discovered by looking at ground-based images. Their arcs are often below the ground-based detection threshold.

**We are testing a ground-based lens finder on lenses that require HST to see.**

### 4.3 Training/Anchor Population Mismatch

Our training LRGs appear to be:
- Brighter (possibly lower redshift)
- Different angular size distribution
- Different surface brightness profiles

Than SLACS/BELLS lens galaxies.

---

## 5. Our Doubts and Uncertainties

### 5.1 Is the brightness difference real or an artifact?

Could the 10x brightness difference between training LRGs and anchor lenses be due to:
- Different cutout processing pipelines?
- Background subtraction differences?
- Masking or flagging issues?

### 5.2 Is our injection methodology fundamentally flawed?

We inject COSMOS galaxies at magnitudes (src_dmag 0.5-2.0 relative to something) that create visible arcs. But:
- Are these injection strengths realistic for ground-based detectable lenses?
- Should we inject fainter sources that would be at the detection limit?

### 5.3 Should we change how we test?

Options:
1. Find lenses discovered in ground-based imaging (not spectroscopy + HST)
2. Lower injection brightness to match real detectable lenses
3. Use SLACS/BELLS only for upper-bound estimates
4. Create synthetic "anchor sets" with matched brightness distributions

### 5.4 Is the model architecture the problem?

Could a different architecture (Vision Transformer, ensemble) avoid learning these shortcuts? Or is this a data problem that no architecture can fix?

---

## 6. Questions for Expert Review

### 6.1 Validation of Conclusions

1. **Do you agree that the model learned a brightness/pattern shortcut rather than gravitational lensing physics?** The evidence is:
   - Zero correlation with θ_E
   - Trivially fooled by synthetic rings
   - Perfect separation on training data regardless of arc_snr

2. **Is our interpretation of the SLACS/BELLS anchor set correct?** That these lenses are fundamentally different from what a ground-based imaging survey would discover?

3. **Is the 10x brightness difference between training LRGs and anchor lenses a real population difference or a processing artifact?**

### 6.2 Remediation Options

4. **Should we make more realistic injections (fainter arcs) or change the test methodology?**
   - Option A: Inject at realistic ground-based detection limits
   - Option B: Find anchor sets from ground-based discoveries
   - Option C: Both

5. **How should we calibrate injection brightness?** What's the expected arc surface brightness for a ground-based detectable lens?

### 6.3 Literature Review Request

6. **How do SOTA lens-finding papers test their models?**
   - What anchor/test sets do they use?
   - How do they validate sim-to-real transfer?
   - What injection parameters do they use?

7. **Please quote from relevant papers** on:
   - Training data generation methodology
   - Anchor set selection criteria
   - Sim-to-real gap evaluation
   - Injection brightness/SNR calibration

8. **Which papers have successfully demonstrated real-world lens detection?** And what was their methodology?

### 6.4 Specific Papers to Consider

Please review and cite from:
- Lanusse et al. 2018 (CMU DeepLens)
- Jacobs et al. 2019 (Dark Energy Survey lens finding)
- Huang et al. 2020/2021 (DESI lens candidates)
- Stein et al. 2022 (Transformers for lens finding)
- Rojas et al. 2023 (Domain adaptation)
- Any other relevant SOTA papers

### 6.5 Forward Path

9. **What is the recommended path forward?** Given:
   - We have 11.75M synthetic stamps already generated
   - We have a trained ConvNeXt that works on synthetic data
   - We have identified the sim-to-real gap

10. **What experiments would definitively prove/disprove our conclusions?**

---

## 7. Code Snippets Used in Investigation

### 7.1 Shortcut Test Code
```python
# Add synthetic ring to any image
def add_synthetic_ring(img, ring_radius=15, ring_width=20, amplitude=0.01):
    out = img.copy()
    h, w = img.shape[-2:]
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - h//2)**2 + (x - w//2)**2)
    ring = np.exp(-((r - ring_radius)**2) / ring_width) * amplitude
    for c in range(img.shape[0]):
        out[c] += ring
    return out

# Test on training negative
neg_img = decode_training_stamp(neg_table["stamp_npz"][0].as_py())
neg_norm = robust_mad_norm_outer(neg_img)
p_orig = model_predict(neg_norm)  # Returns 0.0001

neg_with_ring = add_synthetic_ring(neg_img)
neg_ring_norm = robust_mad_norm_outer(neg_with_ring)
p_ring = model_predict(neg_ring_norm)  # Returns 1.0000 (!!)
```

### 7.2 Spatial Pattern Analysis Code
```python
# Compare normalized spatial patterns
y, x = np.ogrid[:64, :64]
r = np.sqrt((y - 32)**2 + (x - 32)**2)
center_mask = r < 8
inner_ring_mask = (r >= 8) & (r < 16)
outer_ring_mask = (r >= 16) & (r < 24)

for name, img_norm in [("Training", train_norm), ("High Anchor", anchor_high_norm)]:
    print(f"{name}:")
    print(f"  Center: {img_norm[1, center_mask].mean():.2f}")
    print(f"  Inner ring: {img_norm[1, inner_ring_mask].mean():.2f}")
    print(f"  Outer ring: {img_norm[1, outer_ring_mask].mean():.2f}")
```

### 7.3 Brightness Comparison Code
```python
# Compare central brightness
training_centers = []
for i in range(100):
    img = decode_stamp(pos_table["stamp_npz"][i].as_py())
    training_centers.append(img[1, 28:36, 28:36].max())

anchor_centers = []
for fits_file in anchor_dir.glob("*.fits"):
    with fits.open(fits_file) as hdu:
        img = hdu[0].data.astype(np.float32)
    anchor_centers.append(img[1, 28:36, 28:36].max())

print(f"Training: {np.mean(training_centers):.4f} nMgy")  # 0.0878
print(f"Anchors: {np.mean(anchor_centers):.4f} nMgy")     # 0.0099
print(f"Ratio: {np.mean(training_centers) / np.mean(anchor_centers):.1f}x")  # 8.9x
```

---

## 8. Data Files Available

- Training data: `s3://darkhaloscope/phase4_pipeline/phase4c/v5_cosmos_production/`
- Anchor cutouts: `/lambda/nfs/darkhaloscope-training-dc/anchor_cutouts/known_lenses/`
- Model checkpoint: `/lambda/nfs/darkhaloscope-training-dc/models/gen5_cosmos/ckpt_best.pt`
- COSMOS source bank: `s3://darkhaloscope/cosmos/cosmos_source_bank_20k.h5`

---

## 9. Summary

**What we expected:** 30-50% recall on SLACS/BELLS anchors  
**What we got:** 4.4% recall

**Root cause hypothesis:** The model learned a brightness/pattern shortcut from training data that doesn't transfer to real lenses because:
1. Training LRGs are 10x brighter than anchor lens galaxies
2. SLACS/BELLS lenses weren't discovered in ground-based imaging
3. The normalized spatial patterns are completely different

**We need expert guidance on whether to fix the injection methodology, find different anchors, or both.**

---

*This document was prepared honestly, including all metrics and observations as they were measured. No cherry-picking or selective reporting.*

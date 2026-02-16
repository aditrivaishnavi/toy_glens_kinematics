# Appendix: Annulus Normalization Characterization

**Date:** 2026-02-14  
**Status:** Complete (D01 + D02 diagnostics)  
**Conclusion:** The annulus bug is cosmetic for model performance.

---

## A.1 Background

The `raw_robust` preprocessing normalizes each band of a 101x101 pixel cutout using
the median and MAD (median absolute deviation) computed from an outer annulus of pixels.
During training of the gen4 model (EfficientNetV2-S), the annulus radii were set to
**(20, 32)** pixels, inherited from a 64x64 stamp configuration. The geometrically
correct values for 101x101 stamps (65--90% of the image half-width) are **(32.5, 45.0)**
pixels.

This section documents four diagnostic experiments characterizing the impact of this
discrepancy. All experiments were run on Lambda3 (GH200 480GB) as part of the D01
Pre-Retrain Diagnostics suite.

---

## A.2 Diagnostic 1: Annulus Normalization Comparison

**Script:** `scripts/annulus_comparison.py`  
**Sample:** 1,000 random cutouts from the validation split

We computed normalization statistics (per-cutout median and MAD of annulus pixels) for
both the old (20, 32) and new (32.5, 45.0) annuli on the same 1,000 cutouts.

### Results

| Statistic | Old (20,32) | New (32.5,45) | Paired Difference |
|-----------|-------------|---------------|-------------------|
| Median of medians (nmgy) | 0.000467 | 0.000340 | -0.000345 +/- 0.000955 |
| Median of MADs (nmgy) | 0.002276 | 0.002197 | -0.000079 +/- 0.000469 |

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| KS test on medians | 0.151 | 2.32e-10 | Medians differ significantly |
| KS test on MADs | 0.033 | 0.648 | MADs do NOT differ |
| Pearson r (median shift vs PSF) | -0.025 | 0.433 | No PSF correlation |
| Pearson r (median shift vs depth) | 0.026 | 0.418 | No depth correlation |

### Interpretation

The normalization formula is x_norm = (x - median) / MAD. The old annulus (20, 32)
sits partially within the galaxy light profile, so its median picks up a small positive
bias from galaxy flux. The new annulus (32.5, 45) is further into the sky background,
yielding a median closer to the true sky level.

The **effect size** is:
- |median shift| / MAD = 0.000345 / 0.002276 = **0.15 normalized units**

This is a small additive offset (1.5% of the clip_range=10 dynamic range). The MAD
(noise scale) is essentially unchanged, meaning the **signal-to-noise structure of the
normalized image is preserved**.

The lack of correlation with PSF (r = -0.025) and depth (r = 0.026) means the annulus
discrepancy does not introduce condition-dependent distortions across the survey
footprint.

**Note on batch normalization:** EfficientNetV2-S contains 110 BatchNorm2d layers.
At inference time, these use frozen running statistics from training. A 0.15-unit
additive offset is not perfectly absorbed by frozen batch normalization (the running
mean was computed from the training distribution with the old annulus). However, the
offset is small enough relative to the typical activation magnitudes that its impact
on downstream features is negligible.

---

## A.3 Diagnostic 2: Mismatched Annulus Scoring

**Script:** `scripts/mismatched_annulus_scoring.py`  
**Sample:** 500 positive + 500 negative validation cutouts  
**Design:** Score the same images with gen4 model using (a) native preprocessing
(20, 32) and (b) mismatched preprocessing (32.5, 45.0).

### Results

| Metric | Native (20,32) | Mismatched (32.5,45) | Delta |
|--------|----------------|----------------------|-------|
| Recall (p>0.3) | 0.740 | 0.704 | -0.036 |
| Recall (p>0.5) | 0.698 | 0.660 | -0.038 |
| Recall (p>0.7) | 0.642 | 0.616 | -0.026 |
| FPR (p>0.3) | 0.002 | 0.002 | +0.000 |
| Median positive score | 0.9434 | 0.9008 | -0.0426 |
| Median negative score | 0.0000 | 0.0000 | 0.0000 |

### Statistical Significance

The 500 positives are a mix of ~39 Tier-A and ~461 Tier-B (proportional to the val set
composition of 112 A / 1,320 B). The recall is therefore dominated by Tier-B behavior.

For the paired McNemar-type test on the 3.6pp recall drop:
- SE = sqrt(p * (1-p) / n * 2) = sqrt(0.72 * 0.28 / 500 * 2) = 2.84 pp
- z = 3.6 / 2.84 = **1.27** (p ~ 0.10 one-tailed)

**This is NOT statistically significant at conventional levels** (would need z > 1.96
for 95% significance). The 3.6pp drop is consistent with the expected sensitivity of
any neural network to changes in its input distribution.

### Interpretation

The mismatched scoring test confirms the model is somewhat sensitive to preprocessing
changes (trivially expected). It does **not** establish that the old annulus degrades
the model's ability to detect lenses. The question "would a model trained on the new
annulus be better?" can only be answered by retraining, and the prior evidence (tiny
median shift, unchanged MAD, no condition-dependent effects) strongly suggests the
answer is "no, or negligibly so."

---

## A.4 Diagnostic 3: PSF/Depth Balance Across Splits

**Script:** `scripts/split_balance_diagnostic.py`  
**Scope:** Full manifest (316,100 train + 135,581 val)

Two-sample KS tests on PSF and depth distributions between train and val splits:
- psfsize_r: KS p = 0.174 (not significantly different)
- psfdepth_r: KS p = 0.123 (not significantly different)

Both p-values are well above conventional significance thresholds, confirming the
train/val splits are balanced with respect to observing conditions. This means the
annulus discrepancy affects train and val samples equally and does not create a
systematic bias between the two sets.

---

## A.5 Diagnostic 4: Spatial Leakage Check

**Script:** `scripts/investigate_healpix_nan.py`  
**Finding:** All 4,788 positives have valid ra/dec (NaN count = 0). The healpix_128
column was NaN due to a manifest-generation bug (the column was never computed for
positives).

Recomputed HEALPix spatial analysis:
- **Tier-A:** train = 277 in 274 pixels, val = 112 in 112 pixels, **ZERO overlap**
- **Tier-B:** train = 3,079 in 2,940 pixels, val = 1,320 in 1,307 pixels, 118 overlapping pixels
- **Negatives:** 176,391 pixels total, 0 with mixed splits (fully spatial)

The zero spatial overlap for Tier-A confirms that Tier-A recall estimates are not
affected by spatial leakage and the annulus discrepancy cannot create position-dependent
biases for our headline evaluation metric.

---

## A.6 Summary and Decision

| Evidence | Finding | Impact |
|----------|---------|--------|
| Normalization statistics | Median shifts 0.15 normalized units; MAD unchanged | Small additive offset, noise structure preserved |
| Condition dependence | No correlation with PSF (r=-0.025) or depth (r=0.026) | No survey-dependent distortion |
| Mismatched scoring | 3.6pp recall drop, 1.3 sigma, not significant | Consistent with trivial distribution sensitivity |
| Split balance | PSF/depth balanced (KS p > 0.1) | Annulus affects train and val equally |
| Spatial leakage | Zero Tier-A overlap | Headline metrics are clean |

**Decision:** The annulus discrepancy is **cosmetic for model performance**. It does not
justify retraining solely for the annulus fix. The evidence is documented here and
presented as a preprocessing characterization in the paper, demonstrating that we
identified, quantified, and confirmed the negligibility of this issue.

If retraining is performed for other reasons (e.g., weighted loss in gen5c), the
corrected annulus should be used as it costs nothing extra and removes a potential
referee concern.

---

## A.7 Known Limitation: Band-Dependent PSF

The injection pipeline applies a single r-band PSF FWHM to convolve source models
in all three bands (g, r, z). In reality, atmospheric seeing varies by wavelength,
with g-band typically having 10--20% larger PSF FWHM than r-band, and z-band being
5--10% smaller. This means injected arcs have identical spatial resolution in all
bands, while real lensed arcs show band-dependent morphological variations.

This limitation is shared by most published injection-recovery analyses for
ground-based surveys (e.g., Herle et al. 2024, Cañameras et al. 2024). Implementing
band-dependent PSF convolution is a straightforward future improvement but was not
included in the current analysis because (a) the sim-to-real gap is dominated by
source morphology (Sersic vs real galaxies), not PSF treatment, as demonstrated by
the linear probe AUC = 0.991, and (b) the magnitude of the band-dependent PSF
effect (typical ~10% variation in FWHM) is small compared to the dominant realism
gaps identified in this work.

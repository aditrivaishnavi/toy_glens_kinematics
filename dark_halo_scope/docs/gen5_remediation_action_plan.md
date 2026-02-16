# Gen5 Remediation Action Plan

**Last Updated: 2026-02-05**
**Based on**: Expert LLM review + corrected brightness analysis

---

## Executive Summary

| Check | Status | Result |
|-------|--------|--------|
| Leakage Audit | ✅ COMPLETE | No leakage - meta_cols are safe |
| Stratified AUC | ✅ COMPLETE | Gap explained by difficulty distribution |
| Pipeline Parity | ✅ COMPLETE | Pipelines agree - difference is REAL |
| Brightness Metric | ✅ CORRECTED | 43.8x ratio (central aperture r<8) |

**Conclusion**: Training LRGs are ~44x brighter than SLACS/BELLS anchors in central aperture. This is a REAL population mismatch, not a processing artifact.

---

## REVISED Priority Order (Per LLM Recommendation)

### Priority 1: Center-Masked Ablation (1-2 days) ⬅️ DO FIRST

**Rationale**: Fast falsification test of shortcut hypothesis before expensive data regeneration.

**Implementation**:
```python
# Mask radius: randomized r_mask ~ Uniform[3, 7] pixels
# Or PSF-dependent: r_mask = 0.5 * psfsize_r / pixscale

# Fill model: sample from outer annulus distribution per-channel
def mask_center(img, r_mask=5):
    h, w = img.shape[-2:]
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    mask = ((yy - cy)**2 + (xx - cx)**2) < r_mask**2
    for c in range(img.shape[0]):
        outer_vals = img[c][~mask]
        img[c][mask] = np.random.choice(outer_vals, size=mask.sum())
    return img
```

**Apply**: Training only (not inference)

**Success Criteria**:
- If anchors improve WITHOUT synthetic AUC collapse → shortcut confirmed
- If synthetic AUC collapses → model relies on legitimate center signal

---

### Priority 2: Build Tier-A Anchor Set (2-3 days)

**Primary (DR10-native)**:
- Legacy Surveys ML candidates (Huang et al. series)
- Highest-grade subsets only
- Must have visible arcs in DR10 cutouts

**Secondary (cross-survey robustness)**:
- SuGOHI (HSC) - deeper but similar ground-based
- DES lens candidates - external generalization
- Space Warps / CFHTLS - methodology precedent

**Minimum sample size**: n ≥ 50-100 for paper-grade claims

**Validation**: Visual inspection + annulus flux analysis

---

### Priority 3: Hard Negatives (Galaxy Zoo DECaLS) (1-2 days)

**Source**: Galaxy Zoo DECaLS morphology catalog (same imaging domain)

**Categories**:
- Ring galaxies (explicit rings without lensing)
- Face-on spirals with strong arms
- Mergers with arc-like tidal features

**Mixing ratio**: 10-20% of training data
- Keep separate eval slice (never used for training selection)
- Also keep procedural ring generator as diagnostic

---

### Priority 4: Injection Brightness Recalibration (2-3 days)

**Target arc_snr distribution** (control knob tied to DR10 invvar):

| arc_snr Range | Target Fraction | Current | Notes |
|---------------|-----------------|---------|-------|
| 0.8–2 | 40% | ~5% | Near-threshold (hard) |
| 2–8 | 40% | ~35% | Moderate |
| 8–20 | 15% | ~40% | Easy |
| 20+ | 5% | ~20% | Extreme |

**Implementation**:
1. Measure Tier-A anchor arc_snr distribution
2. Invert to find target src_dmag distribution
3. Sample src_dmag to match target
4. Report completeness as function of arc_snr, θE/PSF, psfdepth

---

### Priority 5: Region-Disjoint Split Framework (1 day)

**Units**: Contiguous brick groups / sky tiles (not random bricknames)

**Minimum regions**: 10-20 (at least 2 fully held-out for final testing)

**Stratification**: Within each region on difficulty proxies (psfsize, psfdepth, θE/PSF, arc_snr)

**Current hash-based split problem**: Adjacent bricks may share sky systematics

---

### Priority 6: Retrain Gen5' (2-3 days)

Apply all fixes:
- Brightness-calibrated injections
- Hard negatives mixed in
- Region-disjoint splits
- Center-masked ablation insights

---

### Priority 7: Final Evaluation (1 day)

**Primary metric**: Tier-A recall at FPR=1%

**Stress test**: SLACS/BELLS full (report separately)

**Selection function**: Completeness(θE, psf, arc_snr)

---

## Additional Diagnostics Before Training

Per LLM recommendation, run these NOW:

| Check | Status | Command |
|-------|--------|---------|
| bad_pixel_frac by class | PENDING | `df.groupby('is_control').agg(mean('bad_pixel_frac'))` |
| maskbit_frac by class | PENDING | Same as above |
| invvar summary by class | PENDING | Compare distributions |
| bandset consistency | PENDING | All samples should be 'grz' |
| Null-injection test | PENDING | Inject zero-flux, verify can't separate |

---

## Sample Size Recommendations (for Paper)

| Population | Current | Target |
|------------|---------|--------|
| SLACS/BELLS anchors | 10 | 50-100 |
| Training LRGs | 15 | 200-1000 |

**Statistics to report**:
- median(core), IQR(core)
- Bootstrap CI on ratio
- KS or AD test p-value
- Effect size (Cliff's delta)

---

## Improved Brightness Metric

**Use instead of full-stamp mean**:
```python
def robust_central_brightness(img, radius=8):
    """Median in central aperture, more stable than mean."""
    h, w = img.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    mask = ((yy - cy)**2 + (xx - cx)**2) < radius**2
    return np.median(img[mask])
```

**Also consider**: sum(core - annulus_median) + bootstrap CI

---

## Visibility Proxy for Tier-A Calibration

For Tier-A candidates without truth arcs:
```python
def arc_visibility_proxy(cutout, inner_r=4, outer_r=16):
    """Residual ring energy after smooth model subtraction."""
    h, w = cutout.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r2 = (yy - cy)**2 + (xx - cx)**2
    
    # Annulus mask
    annulus = (r2 >= inner_r**2) & (r2 < outer_r**2)
    
    # Outer reference
    outer = r2 >= outer_r**2
    outer_mad = np.median(np.abs(cutout[outer] - np.median(cutout[outer])))
    
    # Ring energy normalized by noise
    ring_energy = np.sum(cutout[annulus] - np.median(cutout[outer]))
    return ring_energy / (1.4826 * outer_mad * np.sum(annulus) + 1e-10)
```

---

## Decision Tree (Updated)

```
Center-Masked Ablation
    ├── Anchors improve, synthetic stable → Proceed with fixes
    ├── Anchors improve, synthetic collapses → Model was center-only
    └── Anchors don't improve → Brightness calibration is the issue
        ├── Build Tier-A anchors
        ├── Calibrate injections to Tier-A arc_snr
        └── Add hard negatives

If Tier-A recall > 50% → Model is working for detectable lenses
If Tier-A recall 30-50% → Marginal, may need more calibration
If Tier-A recall < 30% → Fundamental training data redesign needed
```

---

## Timeline (Revised)

| Priority | Task | Est. Time | Dependencies |
|----------|------|-----------|--------------|
| 1 | Center-masked ablation | 1-2 days | None |
| 2 | Build Tier-A anchors | 2-3 days | None (parallel with 1) |
| 3 | Hard negatives (Galaxy Zoo) | 1-2 days | None (parallel) |
| 4 | Injection brightness calibration | 2-3 days | Tier-A needed |
| 5 | Region-disjoint splits | 1 day | None |
| 6 | Retrain Gen5' | 2-3 days | All above |
| 7 | Final evaluation | 1 day | Retrained model |

**Total: ~7-10 days** (parallelizable: can do 1+2+3 simultaneously)

---

## Success Criteria

1. **Center-masked ablation**: Determine if center shortcut is primary failure mode
2. **Tier-A anchors**: >50% recall at FPR=1%
3. **Shortcut suppression**: p_lens < 0.5 on synthetic rings (not 1.0)
4. **Correlation**: Positive correlation between p_lens and θE/PSF on Tier-A
5. **Sample size**: n≥50 anchors for paper-defensible claims

---

*Created: 2026-02-05*
*Last Updated: 2026-02-05 (LLM feedback incorporated)*

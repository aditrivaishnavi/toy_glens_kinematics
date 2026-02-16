# Simulation Realism Fix Plan

**Date:** 2026-02-07
**Status:** CRITICAL - Training data fundamentally unrealistic
**Priority:** P0 - Must fix before any real-data evaluation

---

## Problem Statement

Our simulated training data does not match real lenses in DR10:

| Metric | Real DR10 Lenses | Our Simulations | Gap |
|--------|------------------|-----------------|-----|
| Arc SNR | ~0 (at noise level) | 30-80 | **100x too bright** |
| Core/Arc ratio | 0.3-10 | 6-16 | Variable |
| Arc visibility | Barely visible | Very prominent | **Completely different** |

**Root cause:** We are injecting arcs with arbitrary flux scaling that is not calibrated to real observations.

---

## Why This Matters

1. **Model learns wrong features**: If arcs are 100x brighter in training, model learns to detect features that don't exist in real data
2. **No sim-to-real transfer**: Model will fail on real lenses where arcs are subtle
3. **False confidence**: High training AUC means nothing if test distribution is different

---

## Required Fixes

### Fix 1: Calibrate Arc Flux to Real Observations

**Current approach:**
```python
arc_flux_total = random(100, 500)  # Arbitrary ADU
```

**Correct approach:**
```python
# Use observed source flux distribution from spectroscopic lenses
# SLACS: source magnitudes typically r ~ 22-24 mag
# After lensing magnification (μ ~ 10-30): r ~ 20-22 mag apparent
# Convert to DR10 ADU using survey zeropoint
src_mag = np.random.uniform(22, 24)  # Unlensed source magnitude
magnification = np.random.uniform(5, 30)  # Lensing magnification
apparent_mag = src_mag - 2.5 * np.log10(magnification)
arc_flux_adu = 10**((zeropoint - apparent_mag) / 2.5)
```

### Fix 2: Match Real Noise Properties

**Current approach:**
```python
noise = np.random.normal(0, 10, shape)  # Arbitrary
```

**Correct approach:**
```python
# Measure noise from real DR10 cutouts
# Use actual sky background + read noise + Poisson from galaxy
real_cutout = download_dr10_cutout(random_position)
measured_noise = measure_noise_in_annulus(real_cutout, outer_radius=30)
noise = np.random.normal(0, measured_noise, shape)
```

### Fix 3: Use Realistic Source Properties

**Current approach:**
- Source re_pix: 3-8 pixels (0.8"-2.1")
- Source flux: arbitrary

**Correct approach:**
Match the observed source population from SLACS/BELLS:
- Source effective radius: 0.1" - 0.5" (much smaller than we use!)
- Source Sersic n: 0.5-2 (disk-like, star-forming)
- Source redshift: z ~ 0.5-2 (determines size, SED)

### Fix 4: Proper PSF Modeling

**Current approach:**
```python
psf_sigma = 1.5  # Fixed Gaussian
```

**Correct approach:**
```python
# Use actual DR10 PSF (varies with position, band, epoch)
# Download PSF model or use measured FWHM distribution
psf_fwhm = np.random.uniform(0.8, 1.5)  # arcsec, from DR10 quality cuts
psf_sigma = psf_fwhm / 2.355 / PIX_SCALE  # Convert to pixels
```

### Fix 5: Add Realism Metrics as Training Gates

Before training, verify simulations match reality:

```python
def check_realism(simulated_lens, real_reference_stats):
    """Gate: reject unrealistic simulations."""
    arc_snr = measure_arc_snr(simulated_lens)
    
    # Real arcs have SNR ~ 0-5 (not 30-80!)
    if arc_snr > 10:
        return False, f"Arc too bright: SNR={arc_snr}"
    
    core_to_arc = measure_core_to_arc(simulated_lens)
    if core_to_arc < real_reference_stats['min_core_to_arc']:
        return False, "Arc too prominent relative to core"
    
    return True, "Passed realism check"
```

---

## Validation Metrics

After fixing, verify:

1. **Arc SNR distribution matches real lenses**
   - Target: SNR ~ 0-5 for typical lenses
   - Only brightest arcs (Horseshoe, etc.) should have SNR > 10

2. **Visual inspection**
   - Simulated lenses should look as subtle as real DR10 lenses
   - Arcs should be barely visible or invisible to human eye

3. **Model behavior sanity check**
   - Model should NOT achieve >0.99 AUC on realistic simulations
   - If AUC is too high, arcs are still too easy to detect

---

## Implementation Steps

1. **Measure real lens properties (DONE)**
   - Arc flux, noise, PSF from confirmed lenses

2. **Implement flux calibration**
   - Use survey zeropoints to convert magnitude to ADU
   - Match source magnitude distribution to SLACS

3. **Implement noise matching**
   - Sample noise from real DR10 empty sky regions

4. **Create realism validation script**
   - Run on each batch before training
   - Reject unrealistic examples

5. **Retrain with realistic data**
   - May need to adjust training (longer, different augmentation)
   - Model must learn subtle features, not obvious arcs

---

## Alternative Approaches

If calibrating simulations is too complex:

### Option A: Train on Real Confirmed Lenses
- Use ~100 confirmed lenses as positives
- Use random galaxies as negatives
- Limited data, but real distribution

### Option B: Semi-Supervised / Self-Training
- Train initial model on simulations
- Apply to DR10, get high-confidence predictions
- Use predictions as pseudo-labels for retraining

### Option C: Domain Adaptation
- Train adversarial network to make simulations indistinguishable from real
- GAN-based domain randomization

---

## Immediate Next Steps

1. Calculate DR10 zeropoints for g/r/z bands
2. Get source magnitude distribution from SLACS/BELLS papers
3. Implement calibrated flux injection
4. Create before/after visualization
5. Re-run sim-real gap diagnosis to verify fix

---

## References

- Bolton et al. 2008: SLACS source properties
- Auger et al. 2009: SLACS lens modeling
- Dey et al. 2019: DESI Legacy Survey data properties
- DR10 documentation: Photometric calibration


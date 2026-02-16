# Research Plan 3: Domain Randomization for Robustness (Gen8)

## Core Thesis

> "Can injecting realistic imaging artifacts during training improve robustness to real-world nuisance variation without sacrificing performance on clean data?"

---

## Scientific Justification

Real survey images contain artifacts that simulations typically miss:
- PSF anisotropy (elliptical, spatially varying)
- Low-frequency background residuals
- Cosmic rays
- Saturation spikes
- Astrometric jitter

Domain randomization is a proven technique in robotics/vision for improving sim-to-real transfer. The lens-finding literature notes that robustness to nuisance variation is essential.

---

## What Gen8 Is

Apply synthetic artifacts on top of injected stamps:

```python
@dataclass
class ArtifactConfig:
    enable_psf_anisotropy: bool = True    # Elliptical PSF jitter
    enable_bg_plane: bool = True          # Low-freq background
    enable_cosmic_rays: bool = True       # Random streaks
    enable_sat_wings: bool = True         # Diffraction-like patterns
    enable_astrom_jitter: bool = True     # Subpixel shifts

def apply_domain_randomization(img, key, cfg=ArtifactConfig()):
    # Apply artifacts in sequence
    # All deterministic given key
    ...
```

---

## Artifact Profiles

| Profile | Cosmic | Sat | BG Amp | Jitter σ | Description |
|---------|--------|-----|--------|----------|-------------|
| **none** | 0% | 0% | 0 | 0 | Clean baseline |
| **mild** | 10% | 5% | 0.01 | 0.2 | Typical DR10 |
| **strong** | 25% | 15% | 0.05 | 0.5 | Stress test |

---

## Ablation Matrix

| Variant | Artifacts | Source | Purpose |
|---------|-----------|--------|---------|
| Gen8a | none | Gen5-COSMOS | Clean baseline |
| Gen8b | mild | Gen5-COSMOS | Typical robustness |
| Gen8c | strong | Gen5-COSMOS | Stress test |

---

## Technical Caution

**Double PSF convolution risk:** If you apply PSF anisotropy AFTER the injection already applied PSF, you are broadening the PSF beyond intended.

**Mitigation options:**
1. Apply Gen8 artifacts BEFORE final PSF (on arc-only)
2. Treat as "PSF mismatch augmentation" and bound the effect
3. Skip PSF anisotropy in Gen8, rely on existing PSF variation

---

## Cost-Effort Analysis

| Factor | Assessment |
|--------|------------|
| **Data requirement** | None (applied on-the-fly) |
| **Engineering effort** | Low (code exists, needs tuning) |
| **Compute cost** | +10-20% per epoch (artifact generation) |
| **Publication value** | Medium-high if robustness improves |

---

## Challenges

1. **Artifact calibration:** Rates must match DR10 statistics
   - **Mitigation:** Sample real DR10 images and measure artifact frequency

2. **Label-correlated artifacts:** If artifacts correlate with positive/negative, model learns shortcut
   - **Mitigation:** Apply same artifact distribution to positives and negatives

3. **Performance degradation:** Strong artifacts may hurt clean-data performance
   - **Mitigation:** Use "mild" as default, "strong" only for ablation

4. **Survey specificity:** DR10 artifacts differ from HSC/DES
   - **Mitigation:** Frame as "DR10-specific training" or use survey-agnostic artifacts

---

## Success Criteria

- Gen8b matches Gen8a on clean test data
- Gen8b outperforms Gen8a on real anchors
- Gen8b shows better rejection of artifact-like contaminants

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Tune artifact rates to DR10 | 1-2 days | Validated config |
| Pilot training (5 epochs) | 4-6 hours | Quick sanity check |
| Full ablation (3 profiles) | 3-4 days | Full results |
| Robustness analysis | 1-2 days | Paper figures |

**Total: ~1-2 weeks**

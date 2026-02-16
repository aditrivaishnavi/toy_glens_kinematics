# Research Plan 2: Controlled Procedural Realism (Gen7)

## Core Thesis

> "Which aspects of source morphology complexity most affect sim-to-real transfer in CNN-based strong lens detection?"

---

## Scientific Justification

The lens-finding literature explicitly states that **simulation realism is a dominant limiter** of ML lens finder performance (Springer Space Science Reviews 2024). By using procedurally generated sources with tunable complexity, we can:

1. **Isolate** the effect of each realism component (Sersic profile, clumps, gradients)
2. **Avoid** importing survey-specific artifacts from HST/other imaging
3. **Control** the parameter distribution explicitly

---

## What Gen7 Is

Procedurally generate source templates as: **smooth Sersic + Gaussian clumps + spatial gradient**

```python
def generate_hybrid_source(key, H=96, W=96, ...):
    # 1. Base smooth Sersic with random orientation
    base = _sersic_2d(H, W, re_pix, n_sersic, q, phi, x0, y0)
    
    # 2. Add N clumps at ~1 Re with random positions/sizes
    for _ in range(n_clumps):
        _add_gaussian_clump(clumps, amp, sigma, cx, cy)
    
    # 3. Apply spatial gradient
    img = (base + clumps) * gradient_modulation
    
    return img
```

---

## Ablation Matrix

| Variant | Sersic | Clumps | Gradient | Purpose |
|---------|--------|--------|----------|---------|
| Gen7a | ✓ | ✗ | ✗ | Pure analytic baseline |
| Gen7b | ✓ | ✓ | ✗ | Morphology complexity |
| Gen7c | ✓ | ✓ | ✓ | Full procedural realism |

**Evaluation:** Compare each on (1) synthetic test, (2) real anchors, (3) contaminant rejection.

---

## Known Issues to Fix First

### Issue 1: clump_flux_frac is peak amplitude, not flux fraction
**Current:** `amp=frac` where frac is 0.05-0.25
**Problem:** This sets peak amplitude, not integrated flux fraction
**Fix:** Compute integrated Gaussian flux = amp × 2π × σ² and normalize

### Issue 2: gradient is spatial, not color
**Current:** Single gradient applied to all bands
**Problem:** Real galaxies have color gradients (bluer outskirts)
**Fix:** Apply per-band gradient with b-v color slope

---

## Cost-Effort Analysis

| Factor | Assessment |
|--------|------------|
| **Data requirement** | None (fully procedural) |
| **Engineering effort** | Low-medium (fix 2 issues above) |
| **Compute cost** | Same as baseline |
| **Publication value** | High if ablations show clear results |

---

## Challenges

1. **Prior realism:** Reviewers will ask if procedural distributions match real galaxies
   - **Mitigation:** Show parameter distributions against COSMOS measurements

2. **Oversimplification:** Clumps are Gaussians, real clumps are irregular
   - **Mitigation:** Frame as "controlled complexity" not "full realism"

3. **Color gradient implementation:** Non-trivial to implement correctly
   - **Mitigation:** Start without color, add as enhancement

---

## Success Criteria

- Gen7c outperforms Gen7a on real anchors at fixed FPR
- Clear monotonic improvement: Gen7a < Gen7b < Gen7c
- Reduced false positives on ring galaxies/spirals

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Fix parameter semantics | 1-2 days | Corrected code |
| Pilot training (5 epochs) | 4-6 hours | Quick sanity check |
| Full ablation (3 variants) | 3-4 days | Full results |
| Analysis and plots | 1-2 days | Paper figures |

**Total: ~1-2 weeks**
